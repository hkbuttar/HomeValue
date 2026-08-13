import json

import numpy as np
import pandas as pd

from validation.intervals import IntervalConfig, calibrate_valuation_intervals, conformal_quantile


def test_conformal_quantile_uses_finite_sample_correction():
    assert conformal_quantile(np.arange(1, 11), .90) == 10


def test_calibrates_positive_intervals_and_reports_segment_coverage(tmp_path):
    calibration = pd.DataFrame({
        "sale_id": [f"c{i}" for i in range(30)],
        "sale_price": np.linspace(100_000, 500_000, 30),
    })
    calibration["prediction_model"] = calibration["sale_price"] * np.tile([.9, 1.1, 1.0], 10)
    test = pd.DataFrame({
        "sale_id": [f"t{i}" for i in range(20)],
        "sale_price": np.linspace(120_000, 600_000, 20),
    })
    test["prediction_model"] = test["sale_price"] * np.tile([.95, 1.05], 10)
    features = pd.DataFrame({
        "sale_id": test["sale_id"], "nbhd": np.tile(["N1", "N2"], 10)
    })
    calibration_path, test_path, features_path = (
        tmp_path / "calibration.parquet", tmp_path / "test.parquet", tmp_path / "features.parquet"
    )
    calibration.to_parquet(calibration_path, index=False)
    test.to_parquet(test_path, index=False)
    features.to_parquet(features_path, index=False)
    output = tmp_path / "intervals"
    report = calibrate_valuation_intervals(
        calibration_path, test_path, output, features_path,
        IntervalConfig(nominal_coverage=.90, minimum_group_size=5, price_groups=4),
    )
    intervals = pd.read_parquet(output / "valuation_intervals.parquet")
    assert intervals["interval_lower"].gt(0).all()
    assert intervals["interval_lower"].lt(intervals["estimated_value"]).all()
    assert intervals["interval_upper"].gt(intervals["estimated_value"]).all()
    assert report["overall_coverage"]["model"] == 1.0
    coverage = pd.read_csv(output / "interval_coverage.csv")
    assert {"overall", "sale_price_group", "neighborhood"}.issubset(coverage["dimension"])
    curve = pd.read_csv(output / "calibration_curve.csv")
    assert set(curve["nominal_coverage"]) == {.5, .8, .9, .95}
    assert json.loads((output / "interval_results.json").read_text())["models"] == ["model"]
