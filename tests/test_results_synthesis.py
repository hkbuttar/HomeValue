import json

import pandas as pd

from reporting.results import CLASSIFICATIONS, synthesize_results


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_answers_all_questions_with_explicit_evidence_classifications(tmp_path):
    artifacts = {
        "decomposition": _write(tmp_path / "decomposition.json", {"models": {
            "A_property": {"status": "fitted", "in_sample_r_squared": .62, "out_of_sample": {"mae": 50000}},
        }, "incremental_comparisons": {"B_property_market_to_C_property_market_neighborhood": {
            "out_of_sample_mae_improvement": 8000, "delta_in_sample_r_squared": .08,
        }}}),
        "transit": _write(tmp_path / "transit.json", {"conclusion": "CTA association was not stable."}),
        "filters": _write(tmp_path / "filters.json", {"accessibility_findings_stable": False}),
        "gradients": _write(tmp_path / "gradients.json", {"evidence": {"lakefront_half_decay_distance_miles": 1.2}}),
        "autocorrelation": _write(tmp_path / "autocorrelation.json", {"conclusion": "Residual structure remained."}),
        "spatial": _write(tmp_path / "spatial.json", {"spatial_block_validation": {"metrics": {
            "ols": {"mae": 50000}, "spatial_error": {"mae": 46000},
        }}}),
        "benchmark": _write(tmp_path / "benchmark.json", {"best_reported_primary_mae_model": "Gradient Boosting"}),
        "temporal": _write(tmp_path / "temporal.json", {"selected_by_validation_mae": "xgboost", "final_test_metrics": {"xgboost": {"mae": 41000}}}),
        "spatial_holdout": _write(tmp_path / "holdout.json", {"validation_schemes": ["random", "spatial_nbhd"]}),
        "decay": _write(tmp_path / "decay.json", {"best_available_sample_mae_radius": 1.0, "common_sample_sales": 50}),
        "errors": _write(tmp_path / "errors.json", {"worst_reliable_segments": [{"dimension": "property_type", "segment": "Luxury", "median_ape": .18}]}),
        "stability": _write(tmp_path / "stability.json", {"overall_persistence_rate": .75}),
        "attribution": _write(tmp_path / "attribution/property_place_results.json", {"properties_explained": 2}),
    }
    pd.DataFrame({"property_contribution": [100000, 120000], "place_contribution": [70000, 90000]}).to_csv(
        artifacts["attribution"].parent / "property_value_decompositions.csv", index=False
    )
    output = tmp_path / "results"
    report = synthesize_results(artifacts, output)
    assert report["questions_answered"] == 13
    assert {item["number"] for item in report["results"]} == set(range(1, 14))
    assert {item["classification"] for item in report["results"]}.issubset(CLASSIFICATIONS)
    assert all(item["answer"] for item in report["results"])
    notebook = json.loads((output / "homevalue_results.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert (output / "homevalue_results.md").exists()
    assert json.loads((output / "homevalue_results.json").read_text())["questions_answered"] == 13
