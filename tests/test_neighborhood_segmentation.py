import json

import numpy as np
import pandas as pd

from segmentation.neighborhoods import SegmentationConfig, segment_neighborhoods


def segmentation_data():
    sales, indices = [], []
    counter = 0
    for neighborhood_index in range(8):
        neighborhood = f"N{neighborhood_index}"
        high = neighborhood_index >= 4
        for year in (2019, 2020, 2021):
            growth = 1 + (0.10 if high else 0.02) * (year - 2019)
            indices.append({
                "nbhd": neighborhood, "year": year,
                "hedonic_adjusted_index": 100 * growth,
                "median_ppsf_index": 100 * growth,
            })
            for sale_index in range(6):
                sqft = (1600 if high else 1000) + 10 * sale_index
                sales.append({
                    "sale_id": f"s{counter}", "pin": f"p{counter}",
                    "sale_date": f"{year}-06-{sale_index + 1:02d}",
                    "sale_price": sqft * (300 if high else 120) * growth,
                    "building_sqft": sqft, "residence_type": "Single Family",
                    "nbhd": neighborhood,
                    "median_household_income": 120_000 if high else 45_000,
                    "owner_occupancy_rate": 0.7 if high else 0.4,
                    "cta_distance_miles": 0.3 if high else 2.0,
                    "transit_commute_share": 0.35 if high else 0.08,
                })
                counter += 1
    return pd.DataFrame(sales), pd.DataFrame(indices)


def test_clusters_markets_validates_and_names_after_fit(tmp_path):
    sales, indices = segmentation_data()
    sales_path, indices_path = tmp_path / "sales.parquet", tmp_path / "indices.parquet"
    sales.to_parquet(sales_path, index=False)
    indices.to_parquet(indices_path, index=False)
    output = tmp_path / "segments"
    config = SegmentationConfig(
        minimum_neighborhood_sales=10, minimum_clusters=2, maximum_clusters=4,
        bootstrap_iterations=5, random_seed=7,
    )
    report = segment_neighborhoods(sales_path, indices_path, output, config)
    segments = pd.read_parquet(output / "neighborhood_segments.parquet")
    assert report["selected_clusters"] >= 2
    assert report["selected_silhouette"] > 0
    assert report["bootstrap_stability"]["iterations"] == 5
    assert segments["archetype"].notna().all()
    assert segments["nbhd"].is_unique
    assert (output / "archetype_profiles.csv").exists()
    assert (output / "neighborhood_archetypes.png").exists()
    parsed = json.loads((output / "segmentation_report.json").read_text())
    assert parsed["naming_rule"].startswith("Names were generated after fitting")

