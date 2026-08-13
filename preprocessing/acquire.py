"""Reproducible, bounded acquisition for HomeValue's public data sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

SOCRATA_DOMAIN = "https://datacatalog.cookcountyil.gov"
DATASETS = {
    "sales": {
        "id": "wvhk-k5uv",
        "columns": "pin,year,township_code,nbhd,class,sale_date,is_mydec_date,sale_price,doc_no,deed_type,mydec_deed_type,is_multisale,num_parcels_sale,sale_type,sale_filter_same_sale_within_365,sale_filter_less_than_10k,sale_filter_deed_type,row_id",
    },
    "characteristics": {
        "id": "x54s-btds",
        "columns": "pin,year,card,class,char_yrblt,char_bldg_sf,char_land_sf,char_beds,char_rooms,char_fbath,char_hbath,char_type_resd,char_cnst_qlty,char_gar1_size,char_gar1_att,char_bsmt,char_bsmt_fin,char_ext_wall,char_heat,char_air,char_renovation",
    },
    "parcels": {
        "id": "nj4t-kc8j",
        "columns": "pin,pin10,year,class,triad_name,township_code,township_name,nbhd_code,zip_code,lon,lat,census_tract_geoid,census_data_year,cook_municipality_name,chicago_community_area_num,chicago_community_area_name",
    },
}
ACS_VARIABLES = {
    "NAME": "name",
    "B19013_001E": "median_household_income",
    "B17001_001E": "poverty_universe",
    "B17001_002E": "poverty_population",
    "B25003_002E": "owner_occupied_units",
    "B25003_003E": "renter_occupied_units",
    "B25003_001E": "occupied_units",
    "B25002_003E": "vacant_units",
    "B25002_001E": "housing_units",
    "B01003_001E": "population",
    "B25010_001E": "average_household_size",
    "B25035_001E": "median_year_structure_built",
    "B15003_022E": "bachelors_degree",
    "B15003_023E": "masters_degree",
    "B15003_024E": "professional_degree",
    "B15003_025E": "doctorate_degree",
    "B15003_001E": "population_25_plus",
    "B08301_001E": "commuters_total",
    "B08301_003E": "commuters_drove_alone",
    "B08301_004E": "commuters_carpooled",
    "B08301_010E": "commuters_public_transit",
}


def normalize_pin(value: object) -> str | None:
    """Return a zero-padded 14-digit PIN, or None for an invalid value."""
    if value is None or pd.isna(value):
        return None
    digits = "".join(ch for ch in str(value).split(".", 1)[0] if ch.isdigit())
    return digits.zfill(14) if 1 <= len(digits) <= 14 else None


def _request_bytes(url: str, *, app_token: str | None = None, retries: int = 4) -> bytes:
    headers = {"User-Agent": "HomeValue/0.1 (public-data research)"}
    if app_token:
        headers["X-App-Token"] = app_token
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PullResult:
    files: list[Path]
    rows: int


def _read_pins(path: Path) -> list[str]:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    pins: set[str] = set()
    for file in files:
        if file.suffix == ".parquet":
            series = pd.read_parquet(file, columns=["pin"])["pin"]
        else:
            series = pd.read_csv(file, usecols=["pin"], dtype={"pin": "string"})["pin"]
        pins.update(filter(None, (normalize_pin(value) for value in series)))
    return sorted(pins)


def pull_cook_county(
    dataset: str,
    years: Iterable[int],
    output_root: Path,
    *,
    page_size: int = 50_000,
    row_limit: int | None = None,
    pins: list[str] | None = None,
    fetch: Callable[..., bytes] = _request_bytes,
) -> PullResult:
    """Pull selected columns, one year and bounded page at a time."""
    spec = DATASETS[dataset]
    files: list[Path] = []
    total_rows = 0
    token = os.getenv("SOCRATA_APP_TOKEN")
    for year in years:
        year_dir = output_root / "cook_county" / dataset / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        part = 0
        pin_batches: list[list[str] | None] = (
            [pins[index : index + 200] for index in range(0, len(pins), 200)] if pins else [None]
        )
        for pin_batch in pin_batches:
            offset = 0
            while row_limit is None or total_rows < row_limit:
                limit = min(page_size, row_limit - total_rows) if row_limit else page_size
                where = [f"year={int(year)}"]
                if pin_batch:
                    quoted = ",".join(f"'{pin}'" for pin in pin_batch)
                    where.append(f"pin in ({quoted})")
                query = {
                    "$select": spec["columns"],
                    "$where": " and ".join(where),
                    "$order": ":id",
                    "$limit": str(limit),
                    "$offset": str(offset),
                }
                url = f"{SOCRATA_DOMAIN}/resource/{spec['id']}.csv?{urllib.parse.urlencode(query)}"
                raw = fetch(url, app_token=token)
                frame = pd.read_csv(io.BytesIO(raw), dtype={"pin": "string"})
                if frame.empty:
                    break
                frame["pin"] = frame["pin"].map(normalize_pin).astype("string")
                out = year_dir / f"part-{part:05d}.parquet"
                frame.to_parquet(out, index=False)
                files.append(out)
                total_rows += len(frame)
                part += 1
                if len(frame) < limit:
                    break
                offset += len(frame)
            if row_limit is not None and total_rows >= row_limit:
                break
        manifest = {
            "dataset": dataset,
            "dataset_id": spec["id"],
            "year": year,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "columns": spec["columns"].split(","),
            "rows": sum(len(pd.read_parquet(p, columns=["pin"])) for p in files if p.parent == year_dir),
            "files": {p.name: _hash(p) for p in files if p.parent == year_dir},
        }
        _write_manifest(year_dir / "manifest.json", manifest)
    return PullResult(files, total_rows)


def pull_acs(year: int, output_root: Path, fetch: Callable[..., bytes] = _request_bytes) -> PullResult:
    params = {
        "get": ",".join(ACS_VARIABLES),
        "for": "tract:*",
        "in": "state:17 county:031",
    }
    if api_key := os.getenv("CENSUS_API_KEY"):
        params["key"] = api_key
    url = f"https://api.census.gov/data/{year}/acs/acs5?{urllib.parse.urlencode(params)}"
    body = fetch(url).decode("utf-8")
    try:
        rows = json.loads(body)
    except json.JSONDecodeError as error:
        message = "Census API returned a non-JSON response"
        if "Missing Key" in body:
            message += "; set CENSUS_API_KEY (the shared unauthenticated quota is exhausted)"
        raise RuntimeError(message) from error
    frame = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=ACS_VARIABLES)
    frame["geoid"] = frame["state"] + frame["county"] + frame["tract"]
    out_dir = output_root / "census_acs5" / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "cook_county_tracts.parquet"
    frame.to_parquet(out, index=False)
    _write_manifest(out_dir / "manifest.json", {"source": url, "rows": len(frame), "sha256": _hash(out), "retrieved_at": datetime.now(timezone.utc).isoformat()})
    return PullResult([out], len(frame))


def pull_cta(output_root: Path, fetch: Callable[..., bytes] = _request_bytes) -> PullResult:
    url = "https://www.transitchicago.com/downloads/sch_data/google_transit.zip"
    raw = fetch(url)
    out_dir = output_root / "cta_gtfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "google_transit.zip"
    zip_path.write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        safe = [name for name in archive.namelist() if not Path(name).is_absolute() and ".." not in Path(name).parts]
        for name in safe:
            archive.extract(name, out_dir)
    _write_manifest(out_dir / "manifest.json", {"source": url, "sha256": _hash(zip_path), "retrieved_at": datetime.now(timezone.utc).isoformat(), "members": safe})
    return PullResult([zip_path], len(safe))


def _years(value: str) -> list[int]:
    if ":" in value:
        start, end = map(int, value.split(":"))
        if start > end:
            raise argparse.ArgumentTypeError("year range must be ascending")
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    sub = parser.add_subparsers(dest="source", required=True)
    cook = sub.add_parser("cook-county")
    cook.add_argument("--dataset", choices=DATASETS, required=True)
    cook.add_argument("--years", type=_years, required=True)
    cook.add_argument("--page-size", type=int, default=50_000)
    cook.add_argument("--limit", type=int)
    cook.add_argument("--pins-from", type=Path)
    acs = sub.add_parser("acs")
    acs.add_argument("--year", type=int, required=True)
    sub.add_parser("cta")
    args = parser.parse_args(argv)
    if args.source == "cook-county":
        pins = _read_pins(args.pins_from) if args.pins_from else None
        result = pull_cook_county(args.dataset, args.years, args.output, page_size=args.page_size, row_limit=args.limit, pins=pins)
    elif args.source == "acs":
        result = pull_acs(args.year, args.output)
    else:
        result = pull_cta(args.output)
    print(f"Wrote {result.rows} rows/items to {len(result.files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
