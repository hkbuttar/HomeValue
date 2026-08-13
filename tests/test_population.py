import json

import pandas as pd
import pytest

from preprocessing.population import PopulationRules, build_population, classify_sales


def sales_frame():
    return pd.DataFrame({
        "pin": ["123", "456", "789", "111", "222", "333"],
        "class": ["203", "299", "203", "211", "203", "203"],
        "sale_date": ["2023-01-02"] * 6,
        "sale_price": [250_000, 300_000, 1, 240_000, 220_000, 210_000],
        "sale_filter_same_sale_within_365": [False, False, False, False, None, False],
        "sale_filter_less_than_10k": [False, False, True, False, False, False],
        "sale_filter_deed_type": [False, False, False, False, False, False],
        "is_multisale": [False, False, False, False, False, True],
    })


def test_classifies_without_dropping_rows():
    result = classify_sales(sales_frame())
    assert result["population_status"].tolist() == [
        "market", "excluded", "excluded", "excluded", "ambiguous", "ambiguous"
    ]
    assert result["is_primary_population"].sum() == 1
    assert "non_target_property_class" in json.loads(result.loc[1, "exclusion_reasons"])
    assert set(json.loads(result.loc[2, "exclusion_reasons"])) == {
        "price_below_minimum", "sale_filter_less_than_10k"
    }


def test_small_multifamily_is_opt_in():
    result = classify_sales(sales_frame(), PopulationRules(include_small_multifamily=True))
    assert result.loc[3, "population_status"] == "market"


def test_missing_required_columns_fail_loudly():
    with pytest.raises(ValueError, match="sale_filter_deed_type"):
        classify_sales(sales_frame().drop(columns="sale_filter_deed_type"))


def test_invalid_pin_is_excluded():
    frame = sales_frame().iloc[[0]].copy()
    frame.loc[0, "pin"] = "not-a-pin"
    result = classify_sales(frame)
    assert result.loc[0, "population_status"] == "excluded"
    assert json.loads(result.loc[0, "exclusion_reasons"]) == ["invalid_pin"]


def test_build_population_writes_partition_and_manifest(tmp_path):
    source = tmp_path / "raw.parquet"
    sales_frame().to_parquet(source, index=False)
    output = tmp_path / "processed"
    manifest = build_population(source, output)
    assert manifest["rows"] == 6
    assert manifest["status_counts"] == {"market": 1, "ambiguous": 2, "excluded": 3}
    assert (output / "sale_year=2023/part-00000.parquet").exists()
    assert (output / "population_manifest.json").exists()
