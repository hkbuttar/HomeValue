import json

import numpy as np

from api.schemas import ValuationRequest
from engine.homevalue import EngineConfig, HomeValueEngine
from tests.test_ml_valuation import ml_frame
from validation.out_of_time import OutOfTimeConfig, run_out_of_time_validation


def test_unified_engine_returns_range_drivers_and_prior_comparables(tmp_path):
    data = ml_frame()
    data_path = tmp_path / "reference.parquet"
    data.to_parquet(data_path, index=False)
    models = tmp_path / "models"
    training_report = run_out_of_time_validation(data_path, models, OutOfTimeConfig(
        random_forest_estimators=10, xgboost_estimators=15, maximum_category_levels=20
    ))
    selected = training_report["selected_by_validation_mae"]
    interval_path = tmp_path / "intervals.json"
    interval_path.write_text(json.dumps({
        "config": {"nominal_coverage": .9},
        "calibration": {selected: {"log_residual_radius": .12, "calibration_rows": 30}},
    }))
    engine = HomeValueEngine(
        models / "final_models.joblib", interval_path, data_path,
        config=EngineConfig(attribution_permutations=6, maximum_comparables=3),
    )
    row = data.iloc[-1]
    response = engine.predict(ValuationRequest(
        building_sqft=float(row["building_sqft"]), land_sqft=float(row["land_sqft"]),
        bedrooms=float(row["bedrooms"]), bathrooms=float(row["bathrooms"]),
        building_age=float(row["building_age"]), garage_spaces=float(row["garage_spaces"]),
        residence_type=str(row["residence_type"]), pin=str(row["pin"]).zfill(14),
        neighborhood=str(row["nbhd"]), latitude=float(row["latitude"]),
        longitude=float(row["longitude"]), valuation_date="2022-06-01",
    ))
    assert response.lower_interval < response.estimated_value < response.upper_interval
    reconciled = (
        response.baseline_market_value + response.property_component
        + response.location_component + response.time_market_component + response.other_component
    )
    assert np.isclose(reconciled, response.estimated_value)
    assert response.confidence == .9
    assert len(response.value_drivers) > 0
    assert 0 < len(response.comparables) <= 3
    assert all(item.sale_date < "2022-06-01" for item in response.comparables)
