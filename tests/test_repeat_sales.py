import json

import numpy as np
import pandas as pd

from market.repeat_sales import build_repeat_sales_analysis, construct_repeat_pairs, repeat_sales_index


def repeat_frame():
    rows = []
    for pin_index in range(6):
        for year_index, year in enumerate((2019, 2020, 2021)):
            price = 150_000 * (1.1 ** year_index) * (1 + pin_index / 20)
            rows.append({
                "sale_id": f"p{pin_index}-{year}", "pin": f"p{pin_index}",
                "sale_date": f"{year}-06-01", "sale_price": price,
                "nbhd": "N1" if pin_index < 3 else "N2",
                "prediction_hedonic": price / np.exp(0.02 * pin_index),
            })
    return pd.DataFrame(rows)


def test_constructs_consecutive_pairs_and_appreciation():
    pairs = construct_repeat_pairs(repeat_frame())
    assert len(pairs) == 12
    assert pairs.groupby("pin").size().eq(2).all()
    assert pairs["holding_period_days"].gt(0).all()
    assert pairs["total_appreciation"].round(10).eq(0.1).all()
    assert {"previous_model_residual", "current_model_residual"}.issubset(pairs)


def test_repeat_sales_index_is_normalized_and_increases():
    index, diagnostics = repeat_sales_index(construct_repeat_pairs(repeat_frame()))
    assert index.iloc[0]["repeat_sales_index"] == 100
    assert index["repeat_sales_index"].is_monotonic_increasing
    assert diagnostics["pairs"] == 12


def test_builder_writes_pairs_index_summaries_chart_and_report(tmp_path):
    source = tmp_path / "core.parquet"
    repeat_frame().to_parquet(source, index=False)
    output = tmp_path / "repeat"
    report = build_repeat_sales_analysis(source, output)
    assert report["repeat_pairs"] == 12
    assert report["unique_repeat_properties"] == 6
    assert report["residual_persistence_pairs"] == 12
    assert (output / "repeat_sale_pairs.parquet").exists()
    assert (output / "repeat_sales_index.csv").exists()
    assert (output / "repeat_sales_by_neighborhood.csv").exists()
    assert (output / "repeat_sales_index.png").exists()
    parsed = json.loads((output / "repeat_sales_results.json").read_text())
    assert "renovations" in parsed["caution"]

