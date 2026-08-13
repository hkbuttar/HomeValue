import json

import pandas as pd
import pytest

from ml.baselines import MedianBaseline, regression_metrics, temporal_split, train_baselines


def baseline_frame():
    return pd.DataFrame({
        "sale_id": ["a", "b", "c", "d", "e", "f"],
        "pin": ["1", "2", "3", "4", "5", "6"],
        "sale_date": pd.to_datetime(["2019-01-01", "2019-02-01", "2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01"]),
        "year": [2019, 2019, 2020, 2020, 2021, 2021],
        "sale_price": [100_000, 200_000, 120_000, 240_000, 150_000, 300_000],
        "building_sqft": [1000, 1000, 1000, 1000, 1000, 1000],
        "residence_type": ["A", "B", "A", "B", "A", "C"],
        "nbhd": ["N1", "N2", "N1", "N2", "N1", "N3"],
    })


def test_global_median_uses_training_values_only():
    train, test, cutoff = temporal_split(baseline_frame())
    model = MedianBaseline("global").fit(train)
    assert cutoff == 2021
    assert model.global_price_ == 160_000
    assert model.predict(test).tolist() == [160_000, 160_000]


def test_group_median_falls_back_for_unseen_group():
    train, test, _ = temporal_split(baseline_frame())
    model = MedianBaseline("type", ("residence_type",)).fit(train)
    assert model.predict(test).tolist() == [110_000, 160_000]


def test_segmented_ppsf_multiplies_subject_area_and_falls_back():
    train, test, _ = temporal_split(baseline_frame())
    test = test.copy()
    test.loc[test["residence_type"].eq("A"), "building_sqft"] = 2000
    model = MedianBaseline("ppsf", ("nbhd", "residence_type"), use_price_per_sqft=True).fit(train)
    assert model.predict(test).tolist() == [220_000, 160_000]


def test_segmented_ppsf_unseen_group_uses_global_rate_for_known_area():
    train, test, _ = temporal_split(baseline_frame())
    test = test.iloc[[1]].copy()
    test["building_sqft"] = 2000
    model = MedianBaseline("ppsf", ("nbhd", "residence_type"), use_price_per_sqft=True).fit(train)
    assert model.predict(test).iloc[0] == 320_000


def test_temporal_split_rejects_single_year():
    with pytest.raises(ValueError, match="at least two"):
        temporal_split(baseline_frame().query("year == 2021"))


def test_regression_metrics():
    metrics = regression_metrics([100, 200], [110, 180])
    assert metrics["mae"] == 15
    assert metrics["rmse"] == pytest.approx((250 ** 0.5))


def test_training_writes_predictions_metrics_and_artifacts(tmp_path):
    source = tmp_path / "core.parquet"
    baseline_frame().to_parquet(source, index=False)
    output = tmp_path / "baselines"
    report = train_baselines(source, output)
    assert set(report["metrics"]) == {
        "global_median", "property_type_median", "neighborhood_median", "segmented_ppsf"
    }
    assert report["train_years"] == [2019, 2020]
    assert report["test_years"] == [2021]
    assert (output / "baseline_predictions.parquet").exists()
    parsed = json.loads((output / "baseline_results.json").read_text())
    assert parsed["models"]["global_median"]["global_price"] == 160_000
