import json

import pandas as pd

from tests.test_ml_valuation import ml_frame
from validation.spatial_holdout import SpatialValidationConfig, run_spatial_holdout_validation


def test_compares_random_temporal_and_grouped_geographic_holdouts(tmp_path):
    frame = ml_frame()
    frame["census_tract"] = [f"T{index % 6}" for index in range(len(frame))]
    frame["municipality"] = ["Chicago" if index % 3 else "Evanston" for index in range(len(frame))]
    source = tmp_path / "sales.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "spatial"
    report = run_spatial_holdout_validation(source, output, SpatialValidationConfig(
        folds=2, random_forest_estimators=10, xgboost_estimators=15,
        maximum_category_levels=20,
    ))
    expected = {"random", "temporal", "spatial_census_tract", "spatial_nbhd", "spatial_municipality"}
    assert set(report["validation_schemes"]) == expected
    metrics = pd.read_csv(output / "holdout_metrics.csv")
    assert len(metrics) == len(expected) * 3
    assert metrics["n"].gt(0).all()
    comparisons = pd.read_csv(output / "random_comparison.csv")
    assert comparisons.loc[comparisons["validation_scheme"].eq("random"), "mae_increase_vs_random"].eq(0).all()
    assert (output / "holdout_predictions.parquet").exists()
    parsed = json.loads((output / "spatial_holdout_results.json").read_text())
    assert parsed["models"] == ["random_forest", "hist_gradient_boosting", "xgboost"]
