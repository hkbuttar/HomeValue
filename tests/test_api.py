import json

import pandas as pd
from fastapi.testclient import TestClient

from api.app import APISettings, create_app
from api.schemas import ValuationResponse


class FakeEngine:
    def predict(self, _request):
        return ValuationResponse(
            estimated_value=600000, lower_interval=540000, upper_interval=675000,
            baseline_market_value=300000, property_component=180000,
            location_component=90000, time_market_component=30000,
            confidence=.9, model_name="fixture",
        )


def _write_artifacts(root):
    def parquet(relative, frame):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def csv(relative, frame):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    def json_file(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    parquet("neighborhood_indices/neighborhood_price_indices.parquet", pd.DataFrame({
        "nbhd": ["N1", "N1", "N2"], "year": [2020, 2021, 2021],
        "sale_count": [10, 12, 8], "median_sale_price": [300000, 330000, 250000],
        "median_ppsf": [200, 220, 180],
    }))
    parquet("segmentation/neighborhood_segments.parquet", pd.DataFrame({
        "nbhd": ["N1", "N2"], "cluster": [0, 1], "archetype": ["Growth", "Affordable"],
    }))
    parquet("accessibility/core_sales_with_accessibility.parquet", pd.DataFrame({
        "nbhd": ["N1", "N1", "N2"],
        "community_area": ["HYDE PARK", "KENWOOD", None],
        "municipality": ["CHICAGO", "CHICAGO", "EVANSTON"],
    }))
    csv("benchmark/model_benchmark.csv", pd.DataFrame({"model": ["Hedonic OLS"], "mae": [42000]}))
    csv("validation/error_segments/segment_error_metrics.csv", pd.DataFrame({
        "model": ["xgboost"], "dimension": ["neighborhood"], "segment": ["N1"], "mae": [40000],
    }))
    for relative in (
        "spatial_autocorrelation/spatial_autocorrelation_report.json",
        "spatial_lag/spatial_lag_results.json", "spatial_error/spatial_error_results.json",
        "transit_robustness/transit_robustness_results.json",
        "amenity_gradients/gradient_results.json",
    ):
        json_file(relative, {"status": "fixture"})
    parquet("comparables/comparable_predictions.parquet", pd.DataFrame({
        "sale_id": ["target"], "pin": ["00000000000123"], "sale_date": ["2021-01-01"],
    }))
    parquet("comparables/comparable_links.parquet", pd.DataFrame({
        "target_sale_id": ["target"], "comparable_sale_id": ["comp"],
        "normalized_weight": [1.0], "distance_miles": [.2],
    }))


def test_api_exposes_typed_valuation_and_research_endpoints(tmp_path):
    _write_artifacts(tmp_path)
    client = TestClient(create_app(APISettings(data_root=tmp_path), FakeEngine()))
    valuation = client.post("/valuation/predict", json={"building_sqft": 1800})
    assert valuation.status_code == 200
    assert valuation.json()["estimated_value"] == 600000
    options = client.get("/valuation/neighborhoods").json()
    assert options["count"] == 2
    assert options["records"][0]["label"] == "Evanston — area N2"
    assert options["records"][1]["label"] == "Hyde Park — area N1"
    assert client.get("/market/summary").json() == {
        "latest_year": 2021, "geography_count": 2, "transaction_count": 20,
        "median_sale_price": 290000.0, "median_ppsf": 200.0,
    }
    neighborhood = client.get("/market/neighborhood/N1")
    assert neighborhood.status_code == 200
    assert neighborhood.json()["segment"]["archetype"] == "Growth"
    assert client.get("/market/comparables/123").json()["count"] == 1
    assert client.get("/models/performance").json()["records"][0]["model"] == "Hedonic OLS"
    assert client.get("/models/spatial").status_code == 200
    assert client.get("/models/errors").status_code == 200
    assert client.get("/accessibility/transit").status_code == 200
    assert client.get("/accessibility/lake").status_code == 200
    segments = client.get("/neighborhoods/segments").json()
    assert segments["count"] == 2
    assert segments["records"][0]["label"] == "Hyde Park — area N1"
    assert segments["records"][1]["label"] == "Evanston — area N2"


def test_api_documents_routes_and_handles_missing_engine_artifacts(tmp_path):
    client = TestClient(create_app(APISettings(data_root=tmp_path)))
    response = client.post("/valuation/predict", json={"building_sqft": 1800})
    assert response.status_code == 503
    schema = client.get("/openapi.json").json()
    expected = {
        "/valuation/predict", "/valuation/neighborhoods",
        "/market/summary", "/market/neighborhood/{neighborhood_id}",
        "/market/comparables/{pin}", "/models/performance", "/models/spatial",
        "/models/errors", "/accessibility/transit", "/accessibility/lake",
        "/neighborhoods/segments",
    }
    assert expected.issubset(schema["paths"])
