import json

import pandas as pd

from benchmark.models import build_model_benchmark


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def metrics(mae):
    return {"mae": mae, "rmse": mae * 1.3, "median_absolute_percentage_error": mae / 1_000_000}


def test_builds_main_table_without_assuming_ml_wins(tmp_path):
    baseline = _write(tmp_path / "baseline.json", {"metrics": {
        "global_median": metrics(80000), "segmented_ppsf": metrics(50000),
    }})
    comps = _write(tmp_path / "comps.json", {"weighted_price_metrics": metrics(45000)})
    hedonic = _write(tmp_path / "hedonic.json", {"metrics_dollars": metrics(42000)})
    lag = _write(tmp_path / "lag.json", {
        "full_sample_comparison": {
            "ols": {"residual_moran": {"moran_i": .12}},
            "spatial_lag": {"residual_moran": {"moran_i": .02}},
        },
        "spatial_block_validation": {"metrics": {"spatial_lag": metrics(39000)}},
    })
    error = _write(tmp_path / "error.json", {
        "full_sample_comparison": {"spatial_error": {"residual_moran": {"moran_i": .01}}},
        "spatial_block_validation": {"metrics": {"spatial_error": metrics(40000)}},
    })
    temporal = _write(tmp_path / "temporal.json", {
        "final_test_metrics": {"hist_gradient_boosting": metrics(41000)}
    })
    holdout = _write(tmp_path / "holdout.json", {"metrics": [{
        "validation_scheme": "spatial_nbhd", "model": "hist_gradient_boosting", **metrics(47000)
    }]})
    output = tmp_path / "benchmark"
    report = build_model_benchmark(output, baseline, comps, hedonic, lag, error, temporal, holdout)
    table = pd.read_csv(output / "model_benchmark.csv")
    assert table["model"].tolist() == [
        "Median", "PPSF Baseline", "Comparable Sales", "Hedonic OLS",
        "Spatial Lag", "Spatial Error", "Gradient Boosting",
    ]
    assert report["best_reported_primary_mae_model"] == "Spatial Lag"
    assert "not proof of universal superiority" in report["conclusion"]
    gradient = table.loc[table["model"].eq("Gradient Boosting")].iloc[0]
    assert gradient["temporal_test"] and gradient["spatial_test"]
    assert gradient["spatial_mae"] == 47000
    assert (output / "model_benchmark.md").exists()
    assert json.loads((output / "model_benchmark_results.json").read_text())["best_spatial_model"]["model"] == "Spatial Lag"
