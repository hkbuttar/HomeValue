import pandas as pd
import pytest

from preprocessing.historical import (
    AlignmentPolicy,
    align_snapshots,
    build_historical_alignment,
    stable_sale_id,
)


def base_sales():
    return pd.DataFrame({
        "sale_id": ["a", "b", "c"],
        "pin": ["1", "1", "2"],
        "sale_year": pd.array([2020, 2023, 2020], dtype="Int64"),
    })


def test_alignment_never_uses_future_data_by_default():
    snapshots = pd.DataFrame({"pin": ["1", "1", "2"], "year": [2019, 2022, 2021]})
    result = align_snapshots(
        base_sales(), snapshots, key="pin", snapshot_year="year",
        max_lag_years=3, prefix="property",
    ).set_index("sale_id")
    assert result.loc["a", "property_match_year"] == 2019
    assert result.loc["b", "property_match_year"] == 2022
    assert pd.isna(result.loc["c", "property_match_year"])
    assert result.loc["c", "property_alignment_status"] == "unmatched_no_history"


def test_stale_and_explicit_future_statuses():
    snapshots = pd.DataFrame({"pin": ["1", "2"], "year": [2010, 2021]})
    stale = align_snapshots(
        base_sales().iloc[[0]], snapshots, key="pin", snapshot_year="year",
        max_lag_years=3, prefix="property",
    )
    assert stale.loc[0, "property_alignment_status"] == "unmatched_stale"
    future = align_snapshots(
        base_sales().iloc[[2]], snapshots, key="pin", snapshot_year="year",
        max_lag_years=3, prefix="property", allow_future=True,
    )
    assert future.loc[0, "property_alignment_status"] == "current_state_future"


def test_duplicate_snapshot_keys_fail_unless_cards_are_expected():
    snapshots = pd.DataFrame({"pin": ["1", "1"], "year": [2020, 2020], "card": [1, 2]})
    with pytest.raises(ValueError, match="not unique"):
        align_snapshots(
            base_sales().iloc[[0]], snapshots, key="pin", snapshot_year="year",
            max_lag_years=3, prefix="parcel",
        )
    result = align_snapshots(
        base_sales().iloc[[0]], snapshots, key="pin", snapshot_year="year",
        max_lag_years=3, prefix="property", allow_multiple_rows=True,
    )
    assert len(result) == 2


def test_stable_sale_id_prefers_source_id_and_has_fallback():
    sales = pd.DataFrame({
        "row_id": ["source-1", None], "pin": ["1", "2"],
        "sale_date": ["2020-01-01", "2020-02-01"], "sale_price": [1, 2],
        "doc_no": ["x", "y"],
    })
    ids = stable_sale_id(sales)
    assert ids.iloc[0] == "source-1"
    assert len(ids.iloc[1]) == 24


def test_end_to_end_alignment_writes_linked_tables(tmp_path):
    def write(name, frame):
        path = tmp_path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return path

    sales = pd.DataFrame({
        "pin": ["00000000000001"], "sale_date": ["2020-06-01"],
        "sale_price": [200_000], "doc_no": ["d"], "row_id": ["s1"],
        "population_status": ["market"],
    })
    characteristics = pd.DataFrame({
        "pin": ["00000000000001"], "year": [2019], "card": [1], "sqft": [1200]
    })
    parcels = pd.DataFrame({
        "pin": ["00000000000001"], "year": [2020],
        "census_tract_geoid": ["17031010100"],
    })
    acs = pd.DataFrame({"geoid": ["17031010100"], "acs_year": [2019], "income": [70_000]})
    report = build_historical_alignment(
        write("sales", sales), write("chars", characteristics), write("parcels", parcels),
        write("acs", acs), tmp_path / "out", AlignmentPolicy(),
    )
    assert report["market_sales"] == 1
    assert report["alignment_status"]["property"] == {"historical": 1}
    assert (tmp_path / "out/acs.parquet").exists()

