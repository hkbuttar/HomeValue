import json

import numpy as np
import pandas as pd

from neighborhood.price_index import NeighborhoodIndexConfig, build_neighborhood_indices


def neighborhood_frame():
    rows = []
    counter = 0
    for neighborhood, growth in (("N1", 1.10), ("N2", 1.03)):
        for pin_index in range(8):
            for year_index, year in enumerate((2019, 2020, 2021, 2022)):
                sqft = 1000 + 50 * pin_index
                price = sqft * 150 * growth**year_index * (1 + pin_index / 50)
                rows.append({
                    "sale_id": f"s{counter}", "pin": f"{neighborhood}-{pin_index}",
                    "sale_date": f"{year}-06-01", "sale_price": price,
                    "building_sqft": sqft, "building_age": 40 + pin_index,
                    "bedrooms": 3, "bathrooms": 2, "stories": 2,
                    "garage_spaces": 1, "has_basement": True,
                    "residence_type": "A", "nbhd": neighborhood,
                })
                counter += 1
    return pd.DataFrame(rows)


def test_builds_three_indices_comovement_and_divergence(tmp_path):
    source = tmp_path / "core.parquet"
    neighborhood_frame().to_parquet(source, index=False)
    output = tmp_path / "indices"
    config = NeighborhoodIndexConfig(
        minimum_sales_per_year=3, minimum_repeat_pairs=3,
        minimum_overlap_years=2, maximum_plotted_neighborhoods=5,
    )
    report = build_neighborhood_indices(source, output, config)
    panel = pd.read_parquet(output / "neighborhood_price_indices.parquet")
    assert report["neighborhoods"] == 2
    assert report["repeat_index_neighborhoods"] == 2
    assert {"median_ppsf_index", "hedonic_adjusted_index", "neighborhood_repeat_sales_index"}.issubset(panel)
    bases = panel.sort_values("year").groupby("nbhd").first()
    assert np.allclose(bases["median_ppsf_index"], 100)
    assert (output / "neighborhood_growth_correlations.csv").exists()
    assert (output / "neighborhood_divergence.csv").exists()
    assert (output / "neighborhood_price_indices.png").exists()
    parsed = json.loads((output / "neighborhood_index_report.json").read_text())
    assert parsed["co_movement_pairs"] == 1

