import json

from spatial.error_model import SpatialErrorConfig, run_spatial_error
from tests.test_spatial_autocorrelation import spatial_frame


def test_spatial_error_compares_ols_sar_sem_and_block_predictions(tmp_path):
    source = tmp_path / "core.parquet"
    spatial_frame().to_parquet(source, index=False)
    output = tmp_path / "error"
    config = SpatialErrorConfig(
        k_neighbors=4, spatial_blocks=4, permutations=19,
        minimum_category_count=2, maximum_observations=100,
    )
    report = run_spatial_error(source, output, config=config)
    comparison = report["full_sample_comparison"]
    assert set(("ols", "spatial_lag", "spatial_error")).issubset(comparison)
    assert -1 < comparison["spatial_error"]["lambda"] < 1
    assert "lambda_p_value" in comparison["spatial_error"]
    assert report["lowest_aic_model"] in {"ols", "spatial_lag", "spatial_error"}
    metrics = report["spatial_block_validation"]["metrics"]
    assert set(metrics) == {"ols", "spatial_lag", "spatial_error"}
    assert all(value["n"] == report["spatial_block_validation"]["test_rows"] for value in metrics.values())
    assert (output / "spatial_model_coefficients.csv").exists()
    assert (output / "spatial_error_block_predictions.parquet").exists()
    parsed = json.loads((output / "spatial_error_results.json").read_text())
    assert "not proven causal mechanisms" in parsed["interpretation_caution"]

