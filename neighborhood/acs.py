"""Engineer a limited, auditable ACS census-tract feature layer."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


COUNT_COLUMNS = (
    "poverty_universe", "poverty_population", "owner_occupied_units",
    "renter_occupied_units", "occupied_units", "vacant_units", "housing_units",
    "population", "tract_population", "population_25_plus", "bachelors_degree",
    "masters_degree", "professional_degree", "doctorate_degree", "commuters_total",
    "commuters_drove_alone", "commuters_carpooled", "commuters_public_transit",
)
FINAL_FEATURES = (
    "median_household_income", "poverty_rate", "bachelors_or_higher_rate",
    "graduate_degree_rate", "owner_occupancy_rate", "renter_occupancy_rate",
    "vacancy_rate", "median_housing_age", "average_household_size",
    "tract_population", "housing_units", "transit_commute_share",
    "automobile_commute_share", "population_density", "housing_unit_density",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    # Census estimate sentinels such as -666,666,666 mean unavailable.
    return values.mask(values <= -100_000_000)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def engineer_acs_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable rates while preserving ACS estimate vintages."""
    result = pd.DataFrame(index=frame.index)
    for identifier in ("sale_id", "geoid", "acs_match_year", "acs_vintage", "acs_lag_years", "acs_alignment_status"):
        if identifier in frame:
            result[identifier] = frame[identifier]
    result["median_household_income"] = _numeric(frame, "median_household_income")
    result["poverty_rate"] = _ratio(
        _numeric(frame, "poverty_population"), _numeric(frame, "poverty_universe")
    )
    bachelors = _numeric(frame, "bachelors_degree")
    masters = _numeric(frame, "masters_degree")
    professional = _numeric(frame, "professional_degree")
    doctorate = _numeric(frame, "doctorate_degree")
    population_25 = _numeric(frame, "population_25_plus")
    result["bachelors_or_higher_rate"] = _ratio(
        bachelors + masters + professional + doctorate, population_25
    )
    result["graduate_degree_rate"] = _ratio(masters + professional + doctorate, population_25)
    occupied = _numeric(frame, "occupied_units")
    result["owner_occupancy_rate"] = _ratio(_numeric(frame, "owner_occupied_units"), occupied)
    result["renter_occupancy_rate"] = _ratio(_numeric(frame, "renter_occupied_units"), occupied)
    housing = _numeric(frame, "housing_units")
    result["vacancy_rate"] = _ratio(_numeric(frame, "vacant_units"), housing)
    result["average_household_size"] = _numeric(frame, "average_household_size")
    vintage = _numeric(frame, "acs_match_year").fillna(_numeric(frame, "acs_vintage"))
    result["median_housing_age"] = (vintage - _numeric(frame, "median_year_structure_built")).clip(lower=0)
    population = _numeric(frame, "population").fillna(_numeric(frame, "tract_population"))
    result["tract_population"] = population
    result["housing_units"] = housing
    commuters = _numeric(frame, "commuters_total")
    result["transit_commute_share"] = _ratio(_numeric(frame, "commuters_public_transit"), commuters)
    automobile = _numeric(frame, "commuters_drove_alone") + _numeric(frame, "commuters_carpooled")
    result["automobile_commute_share"] = _ratio(automobile, commuters)
    area = _numeric(frame, "tract_land_sq_miles")
    if area.notna().any():
        result["population_density"] = _ratio(population, area)
        result["housing_unit_density"] = _ratio(housing, area)
    for column in result.columns:
        if column.endswith("_rate") or column.endswith("_share"):
            result[column] = result[column].where(result[column].between(0, 1))
    return result


def multicollinearity_diagnostics(features: pd.DataFrame, threshold: float = 0.8) -> dict:
    columns = [column for column in FINAL_FEATURES if column in features]
    numeric = features[columns].apply(pd.to_numeric, errors="coerce")
    usable = numeric.loc[:, numeric.notna().sum().ge(2)]
    usable = usable.loc[:, usable.nunique(dropna=True).gt(1)]
    if usable.empty:
        return {"features": [], "high_correlation_pairs": [], "vif": {}}
    correlation = usable.corr()
    pairs = []
    for left_index, left in enumerate(correlation.columns):
        for right in correlation.columns[left_index + 1:]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"feature_a": left, "feature_b": right, "correlation": float(value)})
    complete = usable.fillna(usable.median())
    standardized = (complete - complete.mean()) / complete.std(ddof=0)
    standardized = standardized.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    vif = {}
    if len(standardized) > len(standardized.columns) and len(standardized.columns) > 1:
        matrix = standardized.to_numpy(dtype=float)
        for index, column in enumerate(standardized.columns):
            value = variance_inflation_factor(matrix, index)
            vif[column] = None if not np.isfinite(value) else float(value)
    return {
        "features": usable.columns.tolist(),
        "correlation_threshold": threshold,
        "high_correlation_pairs": pairs,
        "vif": vif,
    }


def build_acs_layer(core_path: Path, aligned_acs_path: Path, output_dir: Path) -> dict:
    core = pd.read_parquet(core_path)
    aligned = pd.read_parquet(aligned_acs_path)
    if aligned["sale_id"].duplicated().any():
        raise ValueError("aligned ACS table must have at most one row per sale_id")
    features = engineer_acs_features(aligned)
    diagnostics = multicollinearity_diagnostics(features)
    engineered = [column for column in FINAL_FEATURES if column in features]
    drop_existing = [column for column in engineered if column in core]
    enriched = core.drop(columns=drop_existing).merge(
        features[["sale_id", *engineered]], on="sale_id", how="left", validate="one_to_one"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "acs_neighborhood_features.parquet", index=False)
    enriched.to_parquet(output_dir / "core_sales_with_acs.parquet", index=False)
    missingness = {
        column: float(features[column].isna().mean()) for column in engineered
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(features),
        "features": engineered,
        "missingness": missingness,
        "multicollinearity": diagnostics,
        "density_status": (
            "available" if "population_density" in features
            else "unavailable_without_tract_land_sq_miles"
        ),
    }
    (output_dir / "acs_feature_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--acs", type=Path, default=Path("data/processed/historical_alignment/acs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/acs_neighborhood"))
    args = parser.parse_args()
    report = build_acs_layer(args.core, args.acs, args.output)
    print(f"Built {len(report['features'])} ACS features for {report['rows']} sales")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

