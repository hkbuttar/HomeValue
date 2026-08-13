import json

import pandas as pd

from ml.spatial_features import SpatialFeatureConfig, build_spatial_features, engineer_prior_spatial_features


def feature_frame():
    rows = []
    for index in range(16):
        rows.append({
            "sale_id": f"s{index}", "pin": f"p{index}",
            "sale_date": pd.Timestamp("2019-01-01") + pd.Timedelta(days=90 * index),
            "sale_price": 150_000 + 5_000 * index, "building_sqft": 1000 + 10 * index,
            "latitude": 41.88 + 0.0005 * (index % 3),
            "longitude": -87.68 + 0.0005 * (index % 3), "nbhd": "N1",
        })
    return pd.DataFrame(rows)


def test_features_use_only_strictly_prior_sales():
    config = SpatialFeatureConfig(
        radius_miles=2, lookback_days=2000,
        minimum_neighborhood_sales=2, appreciation_window_days=1000,
    )
    features, links = engineer_prior_spatial_features(feature_frame(), config)
    assert (links["prior_sale_date"] < links["target_sale_date"]).all()
    assert links.groupby("target_sale_id")["normalized_weight"].sum().round(10).eq(1).all()
    first = features.set_index("sale_id").loc["s0"]
    assert first["prior_nearby_sale_count"] == 0
    assert pd.isna(first["prior_nearby_sale_median"])
    later = features.set_index("sale_id").loc["s15"]
    assert later["prior_nearby_sale_count"] > 0
    assert later["prior_nearby_weighted_ppsf"] > 0


def test_same_pin_prior_sales_are_excluded():
    frame = feature_frame()
    frame.loc[1, "pin"] = frame.loc[0, "pin"]
    _, links = engineer_prior_spatial_features(frame)
    lookup = frame.set_index("sale_id")["pin"]
    assert all(lookup[target] != lookup[prior] for target, prior in links[["target_sale_id", "prior_sale_id"]].itertuples(index=False))


def test_builder_writes_features_links_enriched_data_and_report(tmp_path):
    source = tmp_path / "core.parquet"
    feature_frame().to_parquet(source, index=False)
    output = tmp_path / "features"
    report = build_spatial_features(source, output)
    assert report["strict_temporal_validation"]
    assert report["prior_links"] > 0
    assert (output / "prior_spatial_features.parquet").exists()
    assert (output / "prior_spatial_feature_links.parquet").exists()
    assert (output / "core_sales_with_spatial_features.parquet").exists()
    parsed = json.loads((output / "prior_spatial_feature_report.json").read_text())
    assert parsed["leakage_rule"].startswith("PriorSaleDate")

