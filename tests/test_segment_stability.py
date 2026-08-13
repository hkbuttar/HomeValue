import json

import pandas as pd

from segmentation.stability import StabilityConfig, analyze_segment_stability


def test_reestimates_segments_and_builds_transition_probabilities(tmp_path):
    rows = []
    for period, year in enumerate((2018, 2020, 2022)):
        for neighborhood in range(6):
            high = neighborhood >= 3
            # Neighborhood 2 moves into the higher-valued regime in the last period.
            if neighborhood == 2 and period == 2:
                high = True
            for sale in range(4):
                sqft = 1000 + 20 * sale + (200 if high else 0)
                rows.append({
                    "sale_id": f"{year}-{neighborhood}-{sale}", "nbhd": f"N{neighborhood}",
                    "sale_date": f"{year}-06-{sale + 1:02d}",
                    "sale_price": sqft * (310 if high else 145), "building_sqft": sqft,
                    "residence_type": "Single Family" if high else "Condo",
                    "median_household_income": 105000 if high else 52000,
                    "owner_occupancy_rate": .72 if high else .42,
                    "population_density": 7000 if high else 15000,
                    "cta_distance_miles": 1.8 if high else .4,
                    "transit_commute_share": .16 if high else .38,
                })
    sales_path = tmp_path / "sales.parquet"
    pd.DataFrame(rows).to_parquet(sales_path, index=False)
    output = tmp_path / "stability"
    report = analyze_segment_stability(
        sales_path, None, output,
        StabilityConfig(period_years=2, minimum_neighborhood_sales=4, clusters=2),
    )
    assert report["periods"] == ["2018-2019", "2020-2021", "2022-2022"]
    transitions = pd.read_csv(output / "transition_matrix.csv")
    totals = transitions.groupby(["from_period", "from_segment"])["transition_probability"].sum()
    assert totals.round(10).eq(1).all()
    persistence = pd.read_csv(output / "neighborhood_persistence.csv")
    assert persistence.loc[persistence["nbhd"].eq("N2"), "distinct_segments"].iloc[0] == 2
    assert (output / "segment_history.parquet").exists()
    assert json.loads((output / "stability_report.json").read_text())["overall_persistence_rate"] < 1
