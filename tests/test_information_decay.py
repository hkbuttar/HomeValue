import json

import pandas as pd

from spillovers.decay import InformationDecayConfig, analyze_information_decay
from tests.test_comparables import comparable_frame


def test_estimates_radius_information_decay_with_leakage_safe_comparables(tmp_path):
    frame = comparable_frame()
    # Spread observations far enough for the four radii to add candidates progressively.
    frame["longitude"] = -87.68 + (frame.index % 6) * .004
    frame["latitude"] = 41.88 + (frame.index // 6) * .003
    source = tmp_path / "sales.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "decay"
    report = analyze_information_decay(source, output, InformationDecayConfig(
        minimum_comparables=1, maximum_age_days=3000, evaluation_year=2023,
    ))
    curve = pd.read_csv(output / "information_decay_curve.csv")
    assert curve["radius_miles"].tolist() == [.25, .5, 1.0, 2.0]
    assert curve["coverage_rate"].is_monotonic_increasing
    assert report["leakage_rule"].startswith("Every comparable is strictly earlier")
    links = pd.read_parquet(output / "maximum_radius_comparable_links.parquet")
    assert (links["comparable_sale_date"] < links["target_sale_date"]).all()
    assert (output / "information_decay_curve.png").exists()
    parsed = json.loads((output / "information_decay_results.json").read_text())
    assert parsed["evaluation_year"] == 2023
