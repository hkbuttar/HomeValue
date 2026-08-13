import json

from spatial.lag_model import SpatialLagConfig, run_spatial_lag
from tests.test_spatial_autocorrelation import spatial_frame


def test_spatial_lag_fits_compares_ols_and_validates_by_block(tmp_path):
    source = tmp_path / "core.parquet"
    spatial_frame().to_parquet(source, index=False)
    output = tmp_path / "lag"
    config = SpatialLagConfig(
        k_neighbors=4, spatial_blocks=4, permutations=19,
        minimum_category_count=2, maximum_observations=100,
    )
    report = run_spatial_lag(source, output, config=config)
    comparison = report["full_sample_comparison"]
    assert report["analysis_year"] == 2021
    assert {"ols", "spatial_lag"}.issubset(comparison)
    assert -1 < comparison["spatial_lag"]["rho"] < 1
    assert "residual_moran" in comparison["ols"]
    assert report["spatial_block_validation"]["test_rows"] > 0
    assert report["spatial_block_validation"]["ols"]["n"] == report["spatial_block_validation"]["test_rows"]
    assert (output / "spatial_lag_coefficients.csv").exists()
    assert (output / "spatial_lag_block_predictions.parquet").exists()
    parsed = json.loads((output / "spatial_lag_results.json").read_text())
    assert "not automatically a causal spillover" in parsed["rho_interpretation"]

