import json

import numpy as np

from ml.valuation import MLConfig, select_features, train_ml_models
from tests.test_hedonic import hedonic_frame


def ml_frame():
    frame = hedonic_frame()
    frame["median_household_income"] = 60_000 + (frame.index % 10) * 2_000
    frame["cta_distance_miles"] = 0.2 + (frame.index % 8) / 10
    frame["lake_distance_miles"] = 1 + (frame.index % 10) / 2
    frame["downtown_distance_miles"] = 2 + (frame.index % 12) / 2
    frame["latitude"] = 41.8
    frame["longitude"] = -87.7
    frame["suspicious_future_price"] = frame["sale_price"]
    return frame


def test_feature_allowlist_excludes_targets_ids_coordinates_and_unknown_columns():
    selected, groups = select_features(ml_frame())
    assert "building_sqft" in selected
    assert "cta_distance_miles" in groups["accessibility"]
    assert not {"sale_price", "sale_id", "pin", "latitude", "longitude", "suspicious_future_price"}.intersection(selected)


def test_training_fits_three_models_and_writes_artifacts(tmp_path):
    source = tmp_path / "core.parquet"
    ml_frame().to_parquet(source, index=False)
    output = tmp_path / "ml"
    config = MLConfig(random_forest_estimators=20, xgboost_estimators=30, maximum_category_levels=20)
    report = train_ml_models(source, output, config=config, benchmark_paths=[])
    assert set(report["ml_metrics"]) == {"random_forest", "hist_gradient_boosting", "xgboost"}
    assert report["test_start_year"] == 2021
    assert all(metrics["n"] == 30 for metrics in report["ml_metrics"].values())
    assert "latitude" not in report["numeric_features"]
    assert (output / "ml_models.joblib").exists()
    assert (output / "ml_predictions.parquet").exists()
    parsed = json.loads((output / "ml_results.json").read_text())
    assert parsed["transformed_feature_count"] > 0

