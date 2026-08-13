import json

import numpy as np

from spatial.autocorrelation import SpatialAuditConfig, build_point_weights, prepare_spatial_sample, run_spatial_audit
from tests.test_hedonic import hedonic_frame


def spatial_frame():
    frame = hedonic_frame()
    # Three compact cross-sections with a smooth east-west price component.
    within_year = frame.groupby("year").cumcount()
    frame["latitude"] = 41.75 + (within_year // 6) * 0.01
    frame["longitude"] = -87.80 + (within_year % 6) * 0.01
    frame["sale_price"] *= np.exp(0.06 * (within_year % 6))
    frame["cta_distance_miles"] = 0.2 + (within_year % 6) / 10
    return frame


def test_spatial_sample_is_single_year_unique_pin_and_projected():
    config = SpatialAuditConfig(k_neighbors=4, permutations=19, minimum_category_count=2)
    sample, year = prepare_spatial_sample(spatial_frame(), config)
    assert year == 2021
    assert sample["pin"].is_unique
    assert {"x_3435", "y_3435", "price_per_sqft"}.issubset(sample)
    weights = build_point_weights(sample, config)
    assert set(weights) == {"knn_4", "distance_1_mile"}
    assert weights["knn_4"].n == len(sample)


def test_spatial_audit_calculates_all_morans_and_outputs(tmp_path):
    source = tmp_path / "core.parquet"
    spatial_frame().to_parquet(source, index=False)
    output = tmp_path / "spatial"
    config = SpatialAuditConfig(k_neighbors=4, distance_band_miles=5, permutations=19, minimum_category_count=2)
    report = run_spatial_audit(source, output, config=config)
    assert report["analysis_year"] == 2021
    assert report["sample_rows"] == 30
    assert set(report["weights"]) == {"knn_4", "distance_5_mile"}
    assert isinstance(report["residual_spatial_structure_detected"], bool)
    results = __import__("pandas").read_csv(output / "morans_i_results.csv")
    assert len(results) == 6
    assert set(results["variable"]) == {"sale_price", "price_per_sqft", "hedonic_log_residual"}
    assert (output / "moran_scatter_hedonic_log_residual.png").exists()
    parsed = json.loads((output / "spatial_autocorrelation_report.json").read_text())
    assert parsed["config"]["permutations"] == 19

