import json

import pandas as pd

from neighborhood.acs import build_acs_layer, engineer_acs_features, multicollinearity_diagnostics


def acs_frame():
    return pd.DataFrame({
        "sale_id": ["a", "b", "c", "d"], "acs_match_year": [2020] * 4,
        "median_household_income": [50_000, 60_000, 70_000, 80_000],
        "poverty_universe": [100, 100, 100, 100], "poverty_population": [20, 15, 10, 5],
        "owner_occupied_units": [60, 65, 70, 75], "renter_occupied_units": [40, 35, 30, 25],
        "occupied_units": [100] * 4, "vacant_units": [10, 9, 8, 7], "housing_units": [110] * 4,
        "population": [300, 320, 340, 360], "population_25_plus": [200] * 4,
        "bachelors_degree": [40, 50, 60, 70], "masters_degree": [20, 25, 30, 35],
        "professional_degree": [5] * 4, "doctorate_degree": [5] * 4,
        "average_household_size": [2.5, 2.6, 2.7, 2.8],
        "median_year_structure_built": [1970, 1980, 1990, 2000],
        "commuters_total": [100] * 4, "commuters_drove_alone": [50, 55, 60, 65],
        "commuters_carpooled": [10] * 4, "commuters_public_transit": [30, 25, 20, 15],
    })


def test_engineers_interpretable_rates_and_housing_age():
    result = engineer_acs_features(acs_frame())
    assert result.loc[0, "poverty_rate"] == 0.2
    assert result.loc[0, "owner_occupancy_rate"] == 0.6
    assert result.loc[0, "renter_occupancy_rate"] == 0.4
    assert result.loc[0, "bachelors_or_higher_rate"] == 0.35
    assert result.loc[0, "graduate_degree_rate"] == 0.15
    assert result.loc[0, "transit_commute_share"] == 0.3
    assert result.loc[0, "automobile_commute_share"] == 0.6
    assert result.loc[0, "median_housing_age"] == 50
    assert "population_density" not in result


def test_density_requires_explicit_tract_area_and_sentinels_become_missing():
    frame = acs_frame()
    frame["tract_land_sq_miles"] = 2
    frame.loc[0, "median_household_income"] = -666_666_666
    result = engineer_acs_features(frame)
    assert result.loc[0, "population_density"] == 150
    assert pd.isna(result.loc[0, "median_household_income"])


def test_multicollinearity_reports_high_correlations():
    diagnostics = multicollinearity_diagnostics(engineer_acs_features(acs_frame()))
    assert diagnostics["features"]
    assert diagnostics["high_correlation_pairs"]


def test_build_writes_feature_layer_enriched_core_and_report(tmp_path):
    aligned_path = tmp_path / "acs.parquet"
    acs_frame().to_parquet(aligned_path, index=False)
    core_path = tmp_path / "core.parquet"
    pd.DataFrame({"sale_id": ["a", "b", "c", "d"], "sale_price": [1, 2, 3, 4]}).to_parquet(core_path, index=False)
    output = tmp_path / "features"
    report = build_acs_layer(core_path, aligned_path, output)
    assert report["density_status"] == "unavailable_without_tract_land_sq_miles"
    assert (output / "acs_neighborhood_features.parquet").exists()
    assert (output / "core_sales_with_acs.parquet").exists()
    parsed = json.loads((output / "acs_feature_report.json").read_text())
    assert "poverty_rate" in parsed["features"]

