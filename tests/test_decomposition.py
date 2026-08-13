import json

from hedonic.decomposition import run_decomposition
from tests.test_hedonic import hedonic_frame


def test_nested_decomposition_and_location_increment(tmp_path):
    frame = hedonic_frame()
    frame["cta_distance_miles"] = 0.2 + (frame.index % 10) / 10
    source = tmp_path / "core.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "decomposition"
    report = run_decomposition(source, output, minimum_category_count=2)
    assert all(model["status"] == "fitted" for model in report["models"].values())
    assert "B_property_market_to_C_property_market_neighborhood" in report["incremental_comparisons"]
    assert report["available_accessibility_features"] == ["cta_distance_miles"]
    assert "out-of-sample MAE" in report["central_answer"]
    assert (output / "decomposition_predictions.parquet").exists()
    assert (output / "decomposition_coefficients.csv").exists()
    parsed = json.loads((output / "decomposition_results.json").read_text())
    assert parsed["models"]["D_property_market_neighborhood_accessibility"]["status"] == "fitted"


def test_accessibility_model_is_explicitly_unavailable_before_features_exist(tmp_path):
    frame = hedonic_frame()
    source = tmp_path / "core.parquet"
    frame.to_parquet(source, index=False)
    report = run_decomposition(source, tmp_path / "out", minimum_category_count=2)
    unavailable = report["models"]["D_property_market_neighborhood_accessibility"]
    assert unavailable["status"] == "unavailable"
    assert "accessibility processing" in unavailable["reason"]
