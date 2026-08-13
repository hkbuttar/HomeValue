import json

import numpy as np
import pandas as pd

from tests.test_ml_valuation import ml_frame
from validation.rigor import RigorConfig, run_statistical_rigor_audit


def test_quantifies_uncertainty_seeds_weights_and_sample_sensitivity(tmp_path):
    data = ml_frame()
    data["x_3435"] = np.arange(len(data), dtype=float) * 100
    data["y_3435"] = (data.index % 10).astype(float) * 100
    data["is_strict_market_sale"] = data.index % 3 != 0
    data["is_moderate_market_sale"] = data.index % 5 != 0
    test = data.loc[data["year"].eq(2021), ["sale_id", "sale_price"]].copy()
    test["prediction_a"] = test["sale_price"] * 1.03
    test["prediction_b"] = test["sale_price"] * np.where(test.index % 2, .9, 1.1)
    data_path, prediction_path = tmp_path / "data.parquet", tmp_path / "predictions.parquet"
    data.to_parquet(data_path, index=False)
    test.to_parquet(prediction_path, index=False)
    temporal = tmp_path / "temporal.json"
    temporal.write_text(json.dumps({"final_test_was_used_for_selection": False}))
    spatial = tmp_path / "spatial.json"
    spatial.write_text(json.dumps({"validation_schemes": ["random", "spatial_nbhd"]}))
    hedonic = tmp_path / "hedonic.json"
    hedonic.write_text(json.dumps({"config": {"robust_covariance": "HC3"}}))
    durbin = tmp_path / "durbin.json"
    durbin.write_text(json.dumps({"full_sample_comparison": {"coefficient_stability_sar_to_sdm": {"x": {}}}}))
    output = tmp_path / "rigor"
    report = run_statistical_rigor_audit(
        prediction_path, data_path, output, temporal, spatial, hedonic, durbin,
        RigorConfig(
            bootstrap_iterations=20, repeated_seeds=(1, 2), repeated_model_estimators=10,
            spatial_neighbors=(2, 4), spatial_permutations=9,
        ),
    )
    assert report["checks"]["bootstrap_confidence_intervals"]
    assert report["checks"]["alternative_spatial_weights"]
    assert report["checks"]["transaction_filter_sensitivity"]
    assert len(pd.read_csv(output / "repeated_seed_metrics.csv")) == 2
    assert set(pd.read_csv(output / "sample_sensitivity.csv")["filter"]) == {
        "all_sales", "trimmed_price_tails", "is_strict_market_sale", "is_moderate_market_sale"
    }
    assert json.loads((output / "statistical_rigor_results.json").read_text())["checks_total"] == 11
