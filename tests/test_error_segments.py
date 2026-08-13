import json

import numpy as np
import pandas as pd

from tests.test_ml_valuation import ml_frame
from validation.error_segments import ErrorSegmentConfig, analyze_error_segments


def test_reports_error_across_market_segments_and_flags_small_groups(tmp_path):
    features = ml_frame()
    features["municipality"] = np.where(features.index % 4, "Chicago", "Evanston")
    features["population_density"] = np.where(features["municipality"].eq("Chicago"), 14000, 5000)
    predictions = features[["sale_id", "sale_date", "year", "sale_price"]].copy()
    predictions["prediction_model_a"] = predictions["sale_price"] * np.where(
        features["nbhd"].eq("N1"), 1.05, 0.80
    )
    predictions["prediction_model_b"] = predictions["sale_price"] * 1.02
    feature_path, prediction_path = tmp_path / "features.parquet", tmp_path / "predictions.parquet"
    features.to_parquet(feature_path, index=False)
    predictions.to_parquet(prediction_path, index=False)
    output = tmp_path / "errors"
    report = analyze_error_segments(
        prediction_path, feature_path, output,
        config=ErrorSegmentConfig(minimum_group_size=10),
    )
    expected = {"sale_price_decile", "property_type", "neighborhood", "municipality", "building_age", "distance_to_transit", "time_period", "urban_suburban_context"}
    assert expected.issubset(report["dimensions"])
    metrics = pd.read_csv(output / "segment_error_metrics.csv")
    n2 = metrics.loc[(metrics["model"] == "model_a") & (metrics["dimension"] == "neighborhood") & (metrics["segment"] == "N2")]
    assert np.isclose(n2["median_ape"].iloc[0], .20)
    assert metrics["reliable_group"].isin([True, False]).all()
    assert (output / "worst_reliable_segments.csv").exists()
    assert (output / "predictions_with_segments.parquet").exists()
    parsed = json.loads((output / "error_segment_results.json").read_text())
    assert parsed["models"] == ["model_a", "model_b"]
