import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from api.schemas import ModelPerformanceResponse, ValuationRequest, ValuationResponse
from spatial.lag_model import _weights
from tests.test_ml_valuation import ml_frame
from validation.out_of_time import chronological_split
from validation.spatial_holdout import SpatialValidationConfig, _folds


def test_tiny_knn_geometry_has_known_neighbors_and_row_standardization():
    coordinates = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    weights = _weights(coordinates, k=1)
    matrix = weights.sparse.toarray()
    assert np.allclose(matrix.sum(axis=1), 1)
    assert matrix[0, 1] == 1
    assert matrix[1, 0] == 1
    assert matrix[2, 0] == 1
    assert np.diag(matrix).sum() == 0


def test_three_way_temporal_split_is_strictly_ordered():
    train, validation, test, _, _ = chronological_split(ml_frame())
    assert train["year"].max() < validation["year"].min()
    assert validation["year"].max() < test["year"].min()
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)


def test_geographic_fold_never_splits_a_neighborhood_across_train_and_test():
    frame = ml_frame()
    frame["nbhd"] = [f"N{index % 6}" for index in range(len(frame))]
    spatial_folds = [
        item for item in _folds(frame, SpatialValidationConfig(folds=3))
        if item[0] == "spatial_nbhd"
    ]
    assert len(spatial_folds) == 3
    for _, _, train, test in spatial_folds:
        assert set(frame.iloc[train]["nbhd"]).isdisjoint(frame.iloc[test]["nbhd"])


def test_api_valuation_contract_rejects_invalid_inputs_and_intervals():
    request = ValuationRequest(
        building_sqft=1800, bathrooms=2.5, pin="01234567890123",
        latitude=41.88, longitude=-87.63,
    )
    assert request.building_sqft == 1800
    with pytest.raises(ValidationError):
        ValuationRequest(building_sqft=-1)
    with pytest.raises(ValidationError, match="supplied together"):
        ValuationRequest(building_sqft=1800, latitude=41.88)
    with pytest.raises(ValidationError, match="must contain"):
        ValuationResponse(
            estimated_value=600000, lower_interval=620000, upper_interval=680000,
            property_component=300000, location_component=200000,
            time_market_component=100000, confidence=.9, model_name="test",
        )


def test_api_json_schemas_expose_required_contract_fields():
    valuation_schema = ValuationRequest.model_json_schema()
    assert "building_sqft" in valuation_schema["required"]
    response_schema = ValuationResponse.model_json_schema()
    assert {"estimated_value", "lower_interval", "upper_interval", "value_drivers"}.issubset(
        response_schema["properties"]
    )
    performance = ModelPerformanceResponse(
        model="hedonic", mae=42000, rmse=61000, mdape=.11,
        temporal_tested=True, spatial_tested=False,
    )
    assert performance.model_dump()["temporal_tested"] is True
