import json

import numpy as np
import pandas as pd

from spillovers.comps import ComparableConfig, ComparableTier, build_comparable_engine, generate_comparable_predictions


def comparable_frame():
    rows = []
    for index in range(18):
        rows.append({
            "sale_id": f"s{index}", "pin": f"p{index}",
            "sale_date": pd.Timestamp("2019-01-01") + pd.Timedelta(days=90 * index),
            "sale_price": 150_000 + 5_000 * index,
            "building_sqft": 1200 + 10 * (index % 3), "building_age": 50 + index % 4,
            "latitude": 41.88 + 0.001 * (index % 3),
            "longitude": -87.68 + 0.001 * (index % 3),
            "residence_type": "Single Family",
        })
    return pd.DataFrame(rows)


def config():
    return ComparableConfig(
        minimum_comparables=2, maximum_comparables=4,
        tiers=(ComparableTier("test", 2, 2000, 0.5, 25),),
    )


def test_comparable_links_are_strictly_historical_and_weighted():
    predictions, links = generate_comparable_predictions(comparable_frame(), config())
    assert len(links) > 0
    assert (links["comparable_sale_date"] < links["target_sale_date"]).all()
    assert links.groupby("target_sale_id")["normalized_weight"].sum().round(10).eq(1).all()
    assert predictions["comparable_count"].max() <= 4
    first = predictions.sort_values("sale_date").iloc[0]
    assert pd.isna(first["comparable_prediction"])


def test_same_pin_is_never_its_own_comparable():
    frame = comparable_frame()
    frame.loc[1, "pin"] = frame.loc[0, "pin"]
    _, links = generate_comparable_predictions(frame, config())
    lookup = frame.set_index("sale_id")["pin"]
    assert all(lookup[target] != lookup[comp] for target, comp in links[["target_sale_id", "comparable_sale_id"]].itertuples(index=False))


def test_missing_property_type_is_excluded_without_crashing():
    frame = comparable_frame()
    frame.loc[0, "residence_type"] = None
    predictions, _ = generate_comparable_predictions(frame, config())
    assert "s0" not in predictions["sale_id"].tolist()


def test_build_writes_predictions_links_and_latest_year_metrics(tmp_path):
    source = tmp_path / "core.parquet"
    comparable_frame().to_parquet(source, index=False)
    output = tmp_path / "comps"
    report = build_comparable_engine(source, output, config=config())
    assert report["sales"] == 18
    assert report["sales_with_comparables"] > 0
    assert report["evaluation_year"] == 2023
    assert report["weighted_price_metrics"]["n"] > 0
    assert (output / "comparable_predictions.parquet").exists()
    assert (output / "comparable_links.parquet").exists()
    parsed = json.loads((output / "comparable_results.json").read_text())
    assert parsed["leakage_rule"].startswith("Every comparable sale date")
