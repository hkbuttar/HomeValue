import json

import numpy as np

from accessibility.gradients import analyze_gradients
from tests.test_hedonic import hedonic_frame


def test_lake_and_downtown_gradients_compare_nonlinear_forms(tmp_path):
    frame = hedonic_frame()
    frame["cta_distance_miles"] = 0.1 + (frame.index % 20) / 10
    frame["cta_stations_half_mile"] = (frame["cta_distance_miles"] <= 0.5).astype(int)
    frame["lake_distance_miles"] = 0.05 + (frame.index % 30) / 3
    frame["downtown_distance_miles"] = 0.5 + (frame.index % 30) / 2
    frame["sale_price"] *= np.exp(
        0.15 * np.exp(-frame["lake_distance_miles"])
        - 0.02 * frame["downtown_distance_miles"]
    )
    source = tmp_path / "core.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "gradients"
    report = analyze_gradients(source, output, minimum_category_count=2)
    assert set(report["models"]) == {"lake", "downtown"}
    assert set(report["models"]["lake"]) == {"linear", "cubic_spline", "gam_style"}
    assert report["best_out_of_sample_specifications"]["lake"] in report["models"]["lake"]
    assert "downtown_distance_joint_p_value_after_transit_and_neighborhood_controls" in report["evidence"]
    assert (output / "gradient_predictions.parquet").exists()
    assert (output / "gradient_curves.csv").exists()
    assert (output / "lake_downtown_gradients.png").exists()
    parsed = json.loads((output / "gradient_results.json").read_text())
    assert parsed["test_start_year"] == 2021

