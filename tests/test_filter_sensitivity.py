import json

import numpy as np
import pandas as pd

from tests.test_ml_valuation import ml_frame
from validation.filter_sensitivity import FilterSensitivityConfig, analyze_filter_sensitivity, define_market_sale_samples


def test_compares_strict_and_moderate_sale_definitions(tmp_path):
    frame = ml_frame()
    frame["class"] = np.where(frame.index % 5 == 0, "211", "202")
    for column in ("sale_filter_same_sale_within_365", "sale_filter_less_than_10k", "sale_filter_deed_type"):
        frame[column] = pd.Series(False, index=frame.index, dtype="boolean")
    frame.loc[frame.index % 11 == 0, "sale_filter_deed_type"] = None
    frame.loc[frame.index % 17 == 0, "sale_filter_same_sale_within_365"] = True
    frame["is_multisale"] = frame.index % 19 == 0
    frame["x_3435"] = np.arange(len(frame), dtype=float) * 100
    frame["y_3435"] = (frame.index % 13).astype(float) * 120
    classified = define_market_sale_samples(frame, FilterSensitivityConfig())
    assert classified["is_moderate_market_sale"].sum() > classified["is_strict_market_sale"].sum()
    source = tmp_path / "sales.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "filters"
    report = analyze_filter_sensitivity(source, output, FilterSensitivityConfig(
        minimum_category_count=2, k_neighbors=4,
    ))
    assert report["definitions"]["moderate"]["sample_rows"] > report["definitions"]["strict"]["sample_rows"]
    assert report["definitions"]["strict"]["valuation"]["metrics"]["n"] > 0
    assert report["definitions"]["moderate"]["spatial"]["status"] == "fitted"
    summary = pd.read_csv(output / "filter_sensitivity_summary.csv")
    assert set(summary["sale_definition"]) == {"strict", "moderate"}
    assert (output / "filter_coefficient_stability.csv").exists()
    parsed = json.loads((output / "filter_sensitivity_results.json").read_text())
    assert "missing filter metadata is allowed" in parsed["moderate_definition"]
