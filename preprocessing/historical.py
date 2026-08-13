"""Leakage-safe historical alignment of sales and time-varying features."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class AlignmentPolicy:
    """Rules controlling which historical snapshot may describe a sale."""

    property_max_lag_years: int = 3
    parcel_max_lag_years: int = 3
    acs_max_lag_years: int = 5
    allow_future_snapshots: bool = False


def stable_sale_id(sales: pd.DataFrame) -> pd.Series:
    """Use the source row ID when present, otherwise hash stable transaction fields."""
    fallback_columns = ["pin", "sale_date", "sale_price", "doc_no"]
    missing = [column for column in fallback_columns if column not in sales]
    if missing:
        raise ValueError(f"cannot construct sale_id; missing columns: {', '.join(missing)}")
    values = sales[fallback_columns].astype("string").fillna("").agg("|".join, axis=1)
    fallback = values.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:24])
    if "row_id" not in sales:
        return fallback.astype("string")
    source = sales["row_id"].astype("string")
    return source.where(source.notna() & source.str.len().gt(0), fallback).astype("string")


def _validate_unique_sales(sales: pd.DataFrame) -> None:
    duplicates = sales["sale_id"].duplicated(keep=False)
    if duplicates.any():
        examples = sales.loc[duplicates, "sale_id"].head(3).tolist()
        raise ValueError(f"sale_id must be unique; duplicate examples: {examples}")


def _candidate_year(
    sale_year: int,
    available_years: list[int],
    max_lag: int,
    allow_future: bool,
) -> tuple[int | None, str]:
    historical = [year for year in available_years if year <= sale_year]
    if historical:
        matched = max(historical)
        if sale_year - matched <= max_lag:
            return matched, "exact" if matched == sale_year else "historical"
        return None, "unmatched_stale"
    if allow_future and available_years:
        return min(available_years), "current_state_future"
    return None, "unmatched_no_history"


def align_snapshots(
    sales: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    key: str,
    snapshot_year: str,
    max_lag_years: int,
    prefix: str,
    allow_future: bool = False,
    allow_multiple_rows: bool = False,
) -> pd.DataFrame:
    """Match snapshots by entity and latest non-future year.

    Multiple rows at a selected vintage are retained only when explicitly
    requested, which is needed for improvement-level property cards.
    """
    required_sales = {"sale_id", "sale_year", key}
    required_snapshots = {key, snapshot_year}
    if missing := sorted(required_sales.difference(sales.columns)):
        raise ValueError(f"sales missing alignment columns: {', '.join(missing)}")
    if missing := sorted(required_snapshots.difference(snapshots.columns)):
        raise ValueError(f"snapshots missing alignment columns: {', '.join(missing)}")

    source = snapshots.copy()
    source[snapshot_year] = pd.to_numeric(source[snapshot_year], errors="coerce").astype("Int64")
    duplicate_keys = source.duplicated([key, snapshot_year], keep=False)
    if duplicate_keys.any() and not allow_multiple_rows:
        raise ValueError(f"{prefix} snapshots are not unique by {key}/{snapshot_year}")

    year_lookup = {
        entity: sorted(group[snapshot_year].dropna().astype(int).unique().tolist())
        for entity, group in source.groupby(key, dropna=False)
    }
    selections = []
    for sale_id, sale_year, entity in sales[["sale_id", "sale_year", key]].itertuples(index=False):
        if pd.isna(sale_year) or pd.isna(entity):
            matched, status = None, "unmatched_missing_key"
        else:
            matched, status = _candidate_year(
                int(sale_year), year_lookup.get(entity, []), max_lag_years, allow_future
            )
        selections.append({
            "sale_id": sale_id,
            key: entity,
            f"{prefix}_match_year": matched,
            f"{prefix}_alignment_status": status,
        })
    selected = pd.DataFrame(selections)
    selected[f"{prefix}_match_year"] = pd.array(
        selected[f"{prefix}_match_year"], dtype="Int64"
    )
    renamed = source.rename(columns={snapshot_year: f"{prefix}_match_year"})
    aligned = selected.merge(
        renamed, on=[key, f"{prefix}_match_year"], how="left", validate="many_to_many"
    )
    sale_years = sales.set_index("sale_id")["sale_year"]
    aligned[f"{prefix}_lag_years"] = (
        aligned["sale_id"].map(sale_years) - aligned[f"{prefix}_match_year"]
    ).astype("Int64")
    return aligned


def align_acs(
    sale_geography: pd.DataFrame,
    acs: pd.DataFrame,
    policy: AlignmentPolicy,
) -> pd.DataFrame:
    """Align ACS vintage after historical parcel geography has been selected."""
    geography = sale_geography.copy()
    tract_column = next(
        (column for column in ("census_tract_geoid", "geoid") if column in geography), None
    )
    if tract_column is None:
        raise ValueError("parcel alignment has no census tract GEOID")
    geography["geoid"] = geography[tract_column].astype("string").str.zfill(11)
    acs_source = acs.copy()
    acs_source["geoid"] = acs_source["geoid"].astype("string").str.zfill(11)
    return align_snapshots(
        geography[["sale_id", "sale_year", "geoid"]],
        acs_source,
        key="geoid",
        snapshot_year="acs_year",
        max_lag_years=policy.acs_max_lag_years,
        prefix="acs",
        allow_future=policy.allow_future_snapshots,
    )


def _read_parquet(path: Path) -> pd.DataFrame:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {path}")
    frames = []
    for file in files:
        frame = pd.read_parquet(file)
        if "acs_year" not in frame and "census_acs5" in str(path):
            year_parts = [part for part in file.parts if part.startswith("year=")]
            if year_parts:
                frame["acs_year"] = int(year_parts[-1].split("=", 1)[1])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    # Socrata may infer a code column as numeric in one response page and text
    # in another (for example, parcel class includes both 234 and "EX").
    # Arrow cannot write an object column containing both Python integers and
    # strings, so preserve such identifier-like values uniformly as text.
    for column in combined.select_dtypes(include="object"):
        value_types = combined[column].dropna().map(type)
        if value_types.eq(str).any() and value_types.ne(str).any():
            combined[column] = combined[column].astype("string")
    return combined


def build_historical_alignment(
    sales_path: Path,
    characteristics_path: Path,
    parcels_path: Path,
    acs_path: Path,
    output_path: Path,
    policy: AlignmentPolicy | None = None,
) -> dict:
    """Build linked historical sales, property-card, parcel, and ACS tables."""
    policy = policy or AlignmentPolicy()
    sales = _read_parquet(sales_path)
    if "population_status" in sales:
        sales = sales.loc[sales["population_status"].eq("market")].copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    sales["sale_year"] = sales["sale_date"].dt.year.astype("Int64")
    sales["sale_id"] = stable_sale_id(sales)
    _validate_unique_sales(sales)

    characteristics = _read_parquet(characteristics_path)
    property_aligned = align_snapshots(
        sales, characteristics, key="pin", snapshot_year="year",
        max_lag_years=policy.property_max_lag_years, prefix="property",
        allow_future=policy.allow_future_snapshots, allow_multiple_rows=True,
    )
    parcels = _read_parquet(parcels_path)
    parcel_aligned = align_snapshots(
        sales, parcels, key="pin", snapshot_year="year",
        max_lag_years=policy.parcel_max_lag_years, prefix="parcel",
        allow_future=policy.allow_future_snapshots,
    )
    parcel_aligned = parcel_aligned.merge(
        sales[["sale_id", "sale_year"]], on="sale_id", how="left", validate="many_to_one"
    )
    acs = _read_parquet(acs_path)
    acs_aligned = align_acs(parcel_aligned, acs, policy)

    output_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sales": sales,
        "property_cards": property_aligned,
        "parcels": parcel_aligned,
        "acs": acs_aligned,
    }
    files = {}
    for name, frame in outputs.items():
        destination = output_path / f"{name}.parquet"
        frame.to_parquet(destination, index=False)
        files[name] = destination.name

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": asdict(policy),
        "market_sales": len(sales),
        "output_rows": {name: len(frame) for name, frame in outputs.items()},
        "alignment_status": {
            name: frame[f"{name}_alignment_status"].value_counts(dropna=False).to_dict()
            for name, frame in (("property", property_aligned), ("parcel", parcel_aligned),
                                ("acs", acs_aligned))
        },
        "files": files,
    }
    (output_path / "alignment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, default=Path("data/processed/residential_sales_population"))
    parser.add_argument("--characteristics", type=Path, default=Path("data/raw/cook_county/characteristics"))
    parser.add_argument("--parcels", type=Path, default=Path("data/raw/cook_county/parcels"))
    parser.add_argument("--acs", type=Path, default=Path("data/raw/census_acs5"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/historical_alignment"))
    parser.add_argument("--allow-future-snapshots", action="store_true")
    args = parser.parse_args()
    policy = AlignmentPolicy(allow_future_snapshots=args.allow_future_snapshots)
    report = build_historical_alignment(
        args.sales, args.characteristics, args.parcels, args.acs, args.output, policy
    )
    print(json.dumps(report["alignment_status"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
