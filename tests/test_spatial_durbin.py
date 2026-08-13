import json

from spatial.durbin_model import DurbinConfig, run_spatial_durbin, spatial_justification
from tests.test_spatial_autocorrelation import spatial_frame


def diagnostics(significant):
    p = 0.01 if significant else 0.5
    return {
        "full_sample_comparison": {
            "ols": {"residual_moran": {"moran_i": 0.2, "p_permutation": p}},
            "spatial_lag": {"rho_p_value": p},
            "spatial_error": {"lambda_p_value": p},
        }
    }


def test_gate_requires_prior_spatial_evidence(tmp_path):
    source = tmp_path / "core.parquet"
    spatial_frame().to_parquet(source, index=False)
    diagnostic_path = tmp_path / "diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics(False)))
    report = run_spatial_durbin(source, diagnostic_path, tmp_path / "out")
    assert report["status"] == "skipped_not_justified"
    assert not report["justification"]["justified"]
    assert spatial_justification(diagnostics(True))["justified"]


def test_justified_durbin_fit_compares_sar_and_validates(tmp_path):
    source = tmp_path / "core.parquet"
    spatial_frame().to_parquet(source, index=False)
    diagnostic_path = tmp_path / "diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics(True)))
    output = tmp_path / "durbin"
    config = DurbinConfig(
        k_neighbors=4, spatial_blocks=4, permutations=19,
        minimum_category_count=2, maximum_observations=100,
    )
    report = run_spatial_durbin(source, diagnostic_path, output, config=config)
    assert report["status"] == "fitted_justified"
    comparison = report["full_sample_comparison"]
    assert {"ols", "spatial_lag", "spatial_durbin"}.issubset(comparison)
    assert comparison["spatial_durbin"]["wx_terms"]
    assert "sar_nested_likelihood_ratio_p_value" in comparison["spatial_durbin"]
    assert set(report["spatial_block_validation"]["metrics"]) == {
        "ols", "spatial_lag", "spatial_durbin"
    }
    assert (output / "spatial_durbin_coefficients.csv").exists()
    assert (output / "spatial_durbin_block_predictions.parquet").exists()

