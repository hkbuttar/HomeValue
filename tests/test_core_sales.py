import json

import pandas as pd
import pytest

from preprocessing.core_sales import aggregate_property_cards, build_core_sales, construct_core_sales


def aligned_tables():
    sales = pd.DataFrame({
        "sale_id": ["s1"], "pin": ["00000000000001"], "sale_date": ["2020-06-15"],
        "sale_price": [250_000], "class": ["203"], "nbhd": ["01001"],
    })
    cards = pd.DataFrame({
        "sale_id": ["s1", "s1"], "property_match_year": [2019, 2019],
        "property_alignment_status": ["historical", "historical"],
        "property_lag_years": [1, 1], "card": [1, 2],
        "char_bldg_sf": [1200, 400], "char_land_sf": [5000, 5000],
        "char_beds": [3, 1], "char_rooms": [6, 2], "char_fbath": [1, 1],
        "char_hbath": [1, 0], "char_yrblt": [1920, 1980],
        "char_type_resd": ["2 Story", "1 Story"], "char_gar1_size": ["2 cars", "No"],
        "char_bsmt": ["Full", "None"], "char_cnst_qlty": ["Average", None],
    })
    parcels = pd.DataFrame({
        "sale_id": ["s1"], "parcel_match_year": [2020], "parcel_lag_years": [0],
        "parcel_alignment_status": ["exact"], "lon": [-87.6], "lat": [41.8],
        "census_tract_geoid": ["17031010100"], "cook_municipality_name": ["Chicago"],
    })
    acs = pd.DataFrame({
        "sale_id": ["s1"], "acs_match_year": [2019], "acs_lag_years": [1],
        "acs_alignment_status": ["historical"], "median_household_income": [80_000],
        "owner_occupied_units": [60], "occupied_units": [100], "vacant_units": [10],
        "housing_units": [110], "population": [300], "population_25_plus": [200],
        "bachelors_degree": [40], "masters_degree": [20], "professional_degree": [5],
        "doctorate_degree": [5],
    })
    return sales, cards, parcels, acs


def test_aggregates_multiple_cards_without_duplicating_land():
    _, cards, _, _ = aligned_tables()
    result = aggregate_property_cards(cards).iloc[0]
    assert result["property_card_count"] == 2
    assert result["building_sqft"] == 1600
    assert result["land_sqft"] == 5000
    assert result["bedrooms"] == 4
    assert result["bathrooms"] == 2.5
    assert result["year_built"] == 1920
    assert result["garage_spaces"] == 2
    assert result["has_basement"] == True


def test_constructs_one_canonical_row_and_rates():
    result = construct_core_sales(*aligned_tables())
    assert len(result) == result["sale_id"].nunique() == 1
    row = result.iloc[0]
    assert row["building_age"] == 100
    assert row["owner_occupancy_rate"] == 0.6
    assert row["vacancy_rate"] == pytest.approx(10 / 110)
    assert row["bachelors_or_higher_rate"] == 0.35
    assert row["month"] == 6


def test_rejects_duplicate_parcel_rows():
    sales, cards, parcels, acs = aligned_tables()
    with pytest.raises(ValueError, match="parcels"):
        construct_core_sales(sales, cards, pd.concat([parcels, parcels]), acs)


def test_build_writes_data_and_schema_report(tmp_path):
    directory = tmp_path / "aligned"
    directory.mkdir()
    for name, frame in zip(("sales", "property_cards", "parcels", "acs"), aligned_tables()):
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    output = tmp_path / "core_sales.parquet"
    report = build_core_sales(directory, output)
    assert report["rows"] == report["unique_sale_ids"] == 1
    assert output.exists()
    schema = json.loads((tmp_path / "core_sales_schema_report.json").read_text())
    assert schema["missingness"]["building_sqft"]["missing"] == 0

