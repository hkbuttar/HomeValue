import json

import pandas as pd

from market.exploratory import (
    build_exploration,
    market_cycles,
    prepare_sales,
    repeat_sale_analysis,
    summarize_group,
)


def market_frame():
    return pd.DataFrame({
        "sale_id": ["a", "b", "c", "d"],
        "pin": ["1", "1", "2", "3"],
        "sale_date": ["2020-01-01", "2020-06-01", "2021-01-01", "2021-03-01"],
        "sale_price": [200_000, 220_000, 300_000, 330_000],
        "building_sqft": [1000, 1000, 1500, 1500],
        "building_age": [50, 50, 20, 10],
        "municipality": ["Chicago", "Chicago", "Evanston", "Chicago"],
        "residence_type": ["1 Story", "1 Story", "2 Story", "2 Story"],
        "class": ["203", "203", "205", "205"],
        "nbhd": ["1", "1", "2", "3"],
        "census_tract": ["t1", "t1", "t2", "t3"],
        "longitude": [-87.7, -87.7, -87.68, -87.65],
        "latitude": [41.8, 41.8, 42.0, 41.9],
    })


def test_prepares_ppsf_and_group_summaries():
    frame = prepare_sales(market_frame())
    assert frame["price_per_sqft"].tolist() == [200, 220, 200, 220]
    yearly = summarize_group(frame, "year")
    assert yearly["transaction_count"].tolist() == [2, 2]
    assert yearly["median_sale_price"].tolist() == [210_000, 315_000]


def test_missing_optional_structure_produces_missing_ppsf():
    frame = prepare_sales(market_frame().drop(columns=["building_sqft", "building_age"]))
    assert frame["price_per_sqft"].isna().all()


def test_repeat_sales_and_market_cycles():
    frame = prepare_sales(market_frame())
    repeated, rapid = repeat_sale_analysis(frame)
    assert len(repeated) == len(rapid) == 1
    yearly = market_cycles(summarize_group(frame, "year"))
    assert yearly["market_phase"].tolist() == ["baseline", "boom"]
    assert yearly.loc[1, "median_price_growth"] == 0.5


def test_build_exploration_outputs_tables_figures_and_html(tmp_path):
    source = tmp_path / "core.parquet"
    market_frame().to_parquet(source, index=False)
    output = tmp_path / "exploration"
    metrics = build_exploration(source, output)
    assert metrics["rows"] == 4
    assert metrics["valid_ppsf_sales"] == 4
    assert (output / "market_exploration.html").exists()
    assert (output / "spatial_median_price.png").exists()
    assert (output / "summary_census_tract.csv").exists()
    parsed = json.loads((output / "exploration_metrics.json").read_text())
    assert parsed["repeat_sales"] == 1
