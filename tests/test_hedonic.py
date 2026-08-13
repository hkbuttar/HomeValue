import json

import numpy as np
import pandas as pd

from hedonic.model import HedonicConfig, HedonicModel, train_hedonic


def hedonic_frame():
    rng = np.random.default_rng(7)
    rows = []
    counter = 0
    for year in (2019, 2020, 2021):
        for index in range(30):
            sqft = 900 + 40 * index + rng.normal(0, 20)
            bedrooms = 2 + index % 4
            bathrooms = 1 + (index % 3) * 0.5
            neighborhood = "N1" if index % 2 else "N2"
            log_price = 5.0 + 0.9 * np.log(sqft) + 0.04 * bathrooms + 0.03 * (year - 2019)
            price = np.exp(log_price + rng.normal(0, 0.03))
            rows.append({
                "sale_id": f"s{counter}", "pin": str(counter),
                "sale_date": f"{year}-{index % 12 + 1:02d}-01", "year": year,
                "sale_price": price, "building_sqft": sqft, "land_sqft": 4000,
                "bedrooms": bedrooms, "bathrooms": bathrooms, "building_age": 50,
                "stories": 2, "garage_spaces": 1, "has_basement": index % 2 == 0,
                "residence_type": "A" if index % 3 else "B", "nbhd": neighborhood,
            })
            counter += 1
    return pd.DataFrame(rows)


def test_model_predicts_positive_prices_and_handles_unseen_categories():
    frame = hedonic_frame()
    train = frame.query("year < 2021")
    test = frame.query("year == 2021").copy()
    test.loc[test.index[0], "nbhd"] = "UNSEEN"
    model = HedonicModel(HedonicConfig(minimum_category_count=2)).fit(train)
    predictions = model.predict(test)
    assert predictions.notna().all()
    assert predictions.gt(0).all()
    assert "log_building_sqft" in model.design_columns_
    assert model.smearing_factor_ > 0


def test_coefficients_include_robust_uncertainty():
    model = HedonicModel(HedonicConfig(minimum_category_count=2)).fit(hedonic_frame().query("year < 2021"))
    coefficients = model.coefficient_table()
    assert {"coefficient", "robust_std_error", "p_value", "ci_lower", "ci_upper"}.issubset(coefficients)
    assert coefficients["robust_std_error"].notna().all()


def test_training_writes_results_predictions_and_coefficients(tmp_path):
    source = tmp_path / "core.parquet"
    hedonic_frame().to_parquet(source, index=False)
    output = tmp_path / "hedonic"
    report = train_hedonic(source, output, config=HedonicConfig(minimum_category_count=2))
    assert report["test_start_year"] == 2021
    assert report["metrics_dollars"]["n"] == 30
    assert "building_sqft" in report["interpretations"]
    assert (output / "hedonic_predictions.parquet").exists()
    assert (output / "hedonic_coefficients.csv").exists()
    parsed = json.loads((output / "hedonic_results.json").read_text())
    assert parsed["interpretation_caution"].startswith("Coefficients are hedonic associations")

