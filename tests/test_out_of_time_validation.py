import json

import pytest

from tests.test_ml_valuation import ml_frame
from validation.out_of_time import OutOfTimeConfig, chronological_split, run_out_of_time_validation


def test_three_way_split_rejects_overlapping_cutoffs():
    with pytest.raises(ValueError, match="must precede"):
        chronological_split(ml_frame(), validation_start_year=2021, test_start_year=2021)


def test_selects_on_validation_then_scores_untouched_final_period(tmp_path):
    source = tmp_path / "sales.parquet"
    ml_frame().to_parquet(source, index=False)
    output = tmp_path / "out_of_time"
    report = run_out_of_time_validation(source, output, OutOfTimeConfig(
        random_forest_estimators=15, xgboost_estimators=20, maximum_category_levels=20
    ))
    assert report["train_years"] == [2019]
    assert report["validation_years"] == [2020]
    assert report["final_test_years"] == [2021]
    assert report["final_test_was_used_for_selection"] is False
    assert report["selected_by_validation_mae"] in report["validation_metrics"]
    assert set(report["final_test_metrics"]) == {
        "random_forest", "hist_gradient_boosting", "xgboost"
    }
    assert all(metrics["n"] == 30 for metrics in report["final_test_metrics"].values())
    assert (output / "validation_predictions.parquet").exists()
    assert (output / "final_test_predictions.parquet").exists()
    assert (output / "final_models.joblib").exists()
    assert json.loads((output / "out_of_time_results.json").read_text())["test_start_year"] == 2021
