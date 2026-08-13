import json

import numpy as np

from tests.test_hedonic import hedonic_frame
from transit.premium import DistanceBasis, analyze_cta_premium


def test_distance_bases_fit_and_transform_consistently():
    distance = np.linspace(0.05, 4, 30)
    import pandas as pd
    series = pd.Series(distance)
    for kind in ("linear", "bands", "cubic_spline", "gam_style"):
        basis = DistanceBasis(kind).fit(series)
        transformed = basis.transform(series)
        assert len(transformed) == len(series)
        assert transformed.columns.tolist() == basis.columns_
        assert np.isfinite(transformed.to_numpy()).all()


def test_analysis_compares_nonlinear_forms_and_writes_outputs(tmp_path):
    frame = hedonic_frame()
    frame["cta_distance_miles"] = 0.05 + (frame.index % 30) / 8
    # Add a nonlinear premium to make the exercise identifiable in the fixture.
    frame["sale_price"] *= np.exp(0.12 * np.exp(-frame["cta_distance_miles"]))
    source = tmp_path / "core.parquet"
    frame.to_parquet(source, index=False)
    output = tmp_path / "premium"
    report = analyze_cta_premium(source, output, minimum_category_count=2)
    assert set(report["specifications"]) == {"linear", "bands", "cubic_spline", "gam_style"}
    assert report["best_out_of_sample_specification"] in report["specifications"]
    assert report["test_start_year"] == 2021
    assert (output / "cta_premium_predictions.parquet").exists()
    assert (output / "cta_premium_curves.csv").exists()
    assert (output / "cta_premium_curves.png").exists()
    parsed = json.loads((output / "cta_premium_results.json").read_text())
    assert parsed["reference_distance_miles"] == 3.0
    assert "transit_premium" in parsed["evidence"]

