"""Construct the canonical one-row-per-sale analytical table."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string")


def _sum_min_count(series: pd.Series) -> float:
    return series.sum(min_count=1)


def _first_non_null(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def _garage_spaces(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else 0.0 if str(value).strip().lower() == "no" else np.nan


def aggregate_property_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Aggregate improvement-level records without double-counting parcel land."""
    if "sale_id" not in cards:
        raise ValueError("property cards must contain sale_id")
    frame = cards.copy()
    frame["_building_sqft"] = _numeric(frame, "char_bldg_sf")
    frame["_land_sqft"] = _numeric(frame, "char_land_sf")
    frame["_bedrooms"] = _numeric(frame, "char_beds")
    frame["_rooms"] = _numeric(frame, "char_rooms")
    frame["_full_bathrooms"] = _numeric(frame, "char_fbath")
    frame["_half_bathrooms"] = _numeric(frame, "char_hbath")
    frame["_year_built"] = _numeric(frame, "char_yrblt")
    frame["_garage_spaces"] = _text(frame, "char_gar1_size").map(_garage_spaces)
    frame["_stories"] = pd.to_numeric(
        _text(frame, "char_type_resd").str.extract(r"(\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )
    basement = _text(frame, "char_bsmt").str.strip().str.lower()
    frame["_has_basement"] = basement.notna() & ~basement.isin(["", "none", "no", "0"])

    rows = []
    for sale_id, group in frame.groupby("sale_id", sort=False, dropna=False):
        matched = group["property_match_year"].notna() if "property_match_year" in group else pd.Series(True, index=group.index)
        usable = group.loc[matched]
        rows.append({
            "sale_id": sale_id,
            "property_match_year": _first_non_null(group.get("property_match_year", pd.Series(dtype="Int64"))),
            "property_alignment_status": _first_non_null(group.get("property_alignment_status", pd.Series(dtype="string"))),
            "property_lag_years": _first_non_null(group.get("property_lag_years", pd.Series(dtype="Int64"))),
            "property_card_count": int(len(usable)),
            "building_sqft": _sum_min_count(usable["_building_sqft"]),
            # Land area is repeated on improvement cards, so max avoids multiplying it.
            "land_sqft": usable["_land_sqft"].max(),
            "bedrooms": _sum_min_count(usable["_bedrooms"]),
            "rooms": _sum_min_count(usable["_rooms"]),
            "full_bathrooms": _sum_min_count(usable["_full_bathrooms"]),
            "half_bathrooms": _sum_min_count(usable["_half_bathrooms"]),
            "year_built": usable["_year_built"].min(),
            "stories": usable["_stories"].max(),
            "garage_spaces": _sum_min_count(usable["_garage_spaces"]),
            "has_basement": bool(usable["_has_basement"].any()) if len(usable) else pd.NA,
            "residence_type": _first_non_null(_text(usable, "char_type_resd")),
            "construction_quality": _first_non_null(_text(usable, "char_cnst_qlty")),
            "exterior_wall": _first_non_null(_text(usable, "char_ext_wall")),
            "heating_type": _first_non_null(_text(usable, "char_heat")),
            "air_conditioning": _first_non_null(_text(usable, "char_air")),
            "renovation": _first_non_null(_text(usable, "char_renovation")),
        })
    result = pd.DataFrame(rows)
    result["bathrooms"] = result["full_bathrooms"] + 0.5 * result["half_bathrooms"]
    return result


def _select_and_rename(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    available = {source: target for source, target in mapping.items() if source in frame}
    if "sale_id" not in frame:
        raise ValueError("aligned table must contain sale_id")
    return frame[["sale_id", *available]].rename(columns=available)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def construct_core_sales(
    sales: pd.DataFrame,
    property_cards: pd.DataFrame,
    parcels: pd.DataFrame,
    acs: pd.DataFrame,
) -> pd.DataFrame:
    """Join aligned sources and enforce exactly one row per sale."""
    if sales["sale_id"].duplicated().any():
        raise ValueError("sales must contain exactly one row per sale_id")
    for name, frame in (("parcels", parcels), ("acs", acs)):
        if frame["sale_id"].duplicated().any():
            raise ValueError(f"{name} must contain at most one row per sale_id")

    result = sales.copy()
    properties = aggregate_property_cards(property_cards)
    result = result.merge(properties, on="sale_id", how="left", validate="one_to_one")

    parcel_mapping = {
        "parcel_match_year": "parcel_match_year",
        "parcel_lag_years": "parcel_lag_years",
        "parcel_alignment_status": "parcel_alignment_status",
        "lon": "longitude",
        "lat": "latitude",
        "census_tract_geoid": "census_tract",
        "cook_municipality_name": "municipality",
        "chicago_community_area_num": "community_area_number",
        "chicago_community_area_name": "community_area",
        "zip_code": "zip_code",
    }
    result = result.merge(
        _select_and_rename(parcels, parcel_mapping), on="sale_id", how="left", validate="one_to_one"
    )
    acs_mapping = {
        "acs_match_year": "acs_vintage",
        "acs_lag_years": "acs_lag_years",
        "acs_alignment_status": "acs_alignment_status",
        "median_household_income": "median_household_income",
        "owner_occupied_units": "owner_occupied_units",
        "occupied_units": "occupied_units",
        "vacant_units": "vacant_units",
        "housing_units": "housing_units",
        "population": "tract_population",
        "population_25_plus": "population_25_plus",
        "bachelors_degree": "bachelors_degree",
        "masters_degree": "masters_degree",
        "professional_degree": "professional_degree",
        "doctorate_degree": "doctorate_degree",
    }
    result = result.merge(
        _select_and_rename(acs, acs_mapping), on="sale_id", how="left", validate="one_to_one"
    )

    result["sale_date"] = pd.to_datetime(result["sale_date"], errors="coerce")
    result["year"] = result["sale_date"].dt.year.astype("Int64")
    result["month"] = result["sale_date"].dt.month.astype("Int64")
    result["quarter"] = result["sale_date"].dt.quarter.astype("Int64")
    result["building_age"] = (result["year"] - result["year_built"]).clip(lower=0)
    if {"owner_occupied_units", "occupied_units"}.issubset(result):
        result["owner_occupancy_rate"] = _safe_ratio(
            result["owner_occupied_units"], result["occupied_units"]
        )
    if {"vacant_units", "housing_units"}.issubset(result):
        result["vacancy_rate"] = _safe_ratio(result["vacant_units"], result["housing_units"])
    education = [
        column for column in ("bachelors_degree", "masters_degree", "professional_degree", "doctorate_degree")
        if column in result
    ]
    if education and "population_25_plus" in result:
        attained = result[education].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        result["bachelors_or_higher_rate"] = _safe_ratio(attained, result["population_25_plus"])

    preferred = [
        "sale_id", "pin", "sale_date", "sale_price", "class", "nbhd", "year", "month", "quarter",
        "building_sqft", "land_sqft", "bedrooms", "bathrooms", "stories", "building_age",
        "garage_spaces", "has_basement", "residence_type", "construction_quality",
        "latitude", "longitude", "census_tract", "municipality", "community_area",
        "median_household_income", "owner_occupancy_rate", "vacancy_rate",
        "bachelors_or_higher_rate", "tract_population", "property_match_year",
        "parcel_match_year", "acs_vintage", "property_alignment_status",
        "parcel_alignment_status", "acs_alignment_status",
    ]
    ordered = [column for column in preferred if column in result]
    return result[ordered + [column for column in result if column not in ordered]]


def build_core_sales(input_path: Path, output_path: Path) -> dict:
    tables = {
        name: pd.read_parquet(input_path / f"{name}.parquet")
        for name in ("sales", "property_cards", "parcels", "acs")
    }
    core = construct_core_sales(**tables)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    core.to_parquet(output_path, index=False)
    missingness = {
        column: {"missing": int(core[column].isna().sum()), "rate": float(core[column].isna().mean())}
        for column in core.columns
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(core),
        "unique_sale_ids": int(core["sale_id"].nunique()),
        "columns": list(core.columns),
        "dtypes": {column: str(dtype) for column, dtype in core.dtypes.items()},
        "missingness": missingness,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    report_path = output_path.with_name(f"{output_path.stem}_schema_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/historical_alignment"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/core_sales.parquet"))
    args = parser.parse_args()
    report = build_core_sales(args.input, args.output)
    print(f"Wrote {report['rows']} canonical sales with {len(report['columns'])} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

