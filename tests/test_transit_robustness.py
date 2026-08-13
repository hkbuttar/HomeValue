import json

import numpy as np

from tests.test_hedonic import hedonic_frame
from transit.robustness import TransitRobustnessConfig, analyze_transit_robustness


def test_runs_full_transit_robustness_ladder_and_geographic_subsets(tmp_path):
    frame = hedonic_frame()
    frame["cta_distance_miles"] = .1 + (frame.index % 25) / 10
    frame["sale_price"] *= np.exp(-.08 * frame["cta_distance_miles"])
    frame["median_household_income"] = 55000 + (frame.index % 10) * 3000
    frame["poverty_rate"] = .08 + (frame.index % 5) / 100
    frame["population_density"] = 8000 + (frame.index % 9) * 1000
    frame["municipality"] = np.where(frame.index % 2, "Chicago", "Evanston")
    frame["x_3435"] = np.arange(len(frame), dtype=float) * 100
    frame["y_3435"] = (frame.index % 15).astype(float) * 100
    source = tmp_path / "sales.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "robustness"
    report = analyze_transit_robustness(source, output, TransitRobustnessConfig(
        minimum_category_count=2, minimum_subset_rows=20, k_neighbors=4,
    ))
    names = {row["specification"] for row in report["specifications"]}
    assert {"property_controls", "plus_year_effects", "plus_neighborhood_controls", "plus_acs_controls", "spatial_dependence", "nonlinear_distance"}.issubset(names)
    assert {"subset_municipality=Chicago", "subset_municipality=Evanston"}.issubset(names)
    spatial = next(row for row in report["specifications"] if row["specification"] == "spatial_dependence")
    assert spatial["status"] == "fitted"
    assert -1 < spatial["spatial_rho"] < 1
    assert isinstance(report["conclusion"], str)
    assert (output / "transit_robustness_specifications.csv").exists()
    assert (output / "transit_robustness.png").exists()
    assert json.loads((output / "transit_robustness_results.json").read_text())["distance_column"] == "cta_distance_miles"
