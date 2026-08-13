import json

import pandas as pd

from explainability.report import ExplainabilityConfig, build_explainability_report
from tests.test_ml_valuation import ml_frame
from validation.out_of_time import OutOfTimeConfig, run_out_of_time_validation


def test_builds_cross_method_explanations_and_agreement_table(tmp_path):
    data = ml_frame()
    data_path = tmp_path / "data.parquet"
    data.to_parquet(data_path, index=False)
    model_output = tmp_path / "models"
    run_out_of_time_validation(data_path, model_output, OutOfTimeConfig(
        random_forest_estimators=10, xgboost_estimators=15, maximum_category_levels=20
    ))
    hedonic = pd.DataFrame({
        "term": ["log_building_sqft", "bathrooms"], "coefficient": [.8, .04],
        "ci_lower": [.7, .01], "ci_upper": [.9, .07],
    })
    hedonic_path = tmp_path / "hedonic.csv"
    hedonic.to_csv(hedonic_path, index=False)
    spatial = pd.DataFrame({
        "model": ["sdm", "sdm"], "term": ["building_sqft", "rho"],
        "coefficient": [.6, .2], "p_value": [.001, .03],
    })
    spatial_path = tmp_path / "spatial.csv"
    spatial.to_csv(spatial_path, index=False)
    output = tmp_path / "explainability"
    report = build_explainability_report(
        output, hedonic_path, spatial_path, None,
        model_output / "final_models.joblib", data_path,
        ExplainabilityConfig(sample_size=20, permutation_repeats=2, partial_dependence_points=3),
    )
    assert report["ml"]["status"] == "calculated"
    assert report["hedonic_terms"] == 2
    effects = pd.read_csv(output / "hedonic_marginal_effects.csv")
    assert effects.loc[effects["term"].eq("log_building_sqft"), "marginal_effect"].iloc[0] == .8
    importance = pd.read_csv(output / "ml_permutation_importance.csv")
    assert "building_sqft" in set(importance["feature"])
    agreement = pd.read_csv(output / "method_agreement.csv")
    assert agreement.loc[agreement["concept"].eq("Living Area"), "methods_with_evidence"].iloc[0] == 3
    assert (output / "ml_partial_dependence.csv").exists()
    assert json.loads((output / "explainability_results.json").read_text())["ml"]["model"]
