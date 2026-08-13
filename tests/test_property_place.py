import json

import numpy as np
import pandas as pd

from explainability.property_place import AttributionConfig, decompose_property_values
from tests.test_ml_valuation import ml_frame
from validation.out_of_time import OutOfTimeConfig, run_out_of_time_validation


def test_model_attributions_reconcile_property_place_and_time_to_estimate(tmp_path):
    data = ml_frame()
    data_path = tmp_path / "data.parquet"
    data.to_parquet(data_path, index=False)
    model_output = tmp_path / "models"
    run_out_of_time_validation(data_path, model_output, OutOfTimeConfig(
        random_forest_estimators=10, xgboost_estimators=15, maximum_category_levels=20
    ))
    sale_id = str(data.loc[data["year"].eq(2021), "sale_id"].iloc[0])
    output = tmp_path / "attributions"
    report = decompose_property_values(
        model_output / "final_models.joblib", data_path, data_path, output, [sale_id],
        AttributionConfig(permutations=8, maximum_properties=1),
    )
    assert report["properties_explained"] == 1
    assert report["maximum_absolute_reconciliation_error"] < 1e-6
    summary = pd.read_csv(output / "property_value_decompositions.csv").iloc[0]
    components = (
        summary["baseline_market_value"] + summary["property_contribution"]
        + summary["place_contribution"] + summary["time_market_contribution"]
        + summary["other_contribution"]
    )
    assert np.isclose(components, summary["estimated_value"])
    detail = pd.read_parquet(output / "property_value_attributions.parquet")
    assert {"property", "place", "time_market"}.issubset(detail["component"])
    assert json.loads((output / "property_place_results.json").read_text())["method"].startswith("Monte Carlo Shapley")
