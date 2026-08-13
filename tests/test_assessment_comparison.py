import json

import numpy as np
import pandas as pd

from assessment.comparison import AssessmentConfig, compare_assessments, prepare_assessments


def test_aligns_assessments_converts_values_and_compares_same_sales(tmp_path):
    sales = pd.DataFrame({
        "sale_id": [f"s{i}" for i in range(12)], "pin": [str(1000 + i) for i in range(12)],
        "year": [2022] * 12, "sale_price": np.linspace(200000, 420000, 12),
        "nbhd": np.tile(["N1", "N2"], 6), "municipality": "Chicago",
    })
    assessments = pd.DataFrame({
        "pin": sales["pin"], "year": 2022,
        "mailed_tot": sales["sale_price"] * .08,
        "certified_tot": sales["sale_price"] * .09,
        "board_tot": [np.nan, *(sales["sale_price"].iloc[1:] * .10)],
    })
    predictions = sales[["sale_id"]].copy()
    predictions["prediction_homevalue"] = sales["sale_price"] * 1.001
    sales_path, assessments_path, predictions_path = (
        tmp_path / "sales.parquet", tmp_path / "assessments.parquet", tmp_path / "predictions.parquet"
    )
    sales.to_parquet(sales_path, index=False)
    assessments.to_parquet(assessments_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    prepared = prepare_assessments(assessments, AssessmentConfig())
    assert prepared.loc[prepared["pin"].eq("00000000001000"), "assessment_stage"].iloc[0] == "certified"
    output = tmp_path / "comparison"
    report = compare_assessments(
        sales_path, assessments_path, output, predictions_path,
        AssessmentConfig(minimum_group_size=2, price_groups=4),
    )
    assert report["matched_rows"] == 12
    assert report["common_sample_rows"] == 12
    assert report["best_common_sample_mae_model"] == "homevalue"
    matched = pd.read_parquet(output / "matched_sales_assessments.parquet")
    assert np.isclose(matched.loc[matched["sale_id"].eq("s1"), "assessed_market_value"].iloc[0], sales.loc[1, "sale_price"])
    segments = pd.read_csv(output / "assessment_error_by_segment.csv")
    assert {"assessor", "homevalue"} == set(segments["model"])
    parsed = json.loads((output / "assessment_comparison_results.json").read_text())
    assert parsed["alignment_rule"].endswith("exact sale/tax year.")
