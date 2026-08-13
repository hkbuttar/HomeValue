"""Synthesize HomeValue evidence into thirteen explicitly classified conclusions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


CLASSIFICATIONS = {"Robust", "Suggestive", "Exploratory", "Data-limited"}


def _read(path: Path | None) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}


def _money(value) -> str:
    return f"${value:,.0f}" if value is not None else "not estimated"


def _percent(value) -> str:
    return f"{100 * value:.1f}%" if value is not None else "not estimated"


def _result(number: int, question: str, classification: str, answer: str, evidence: list[str]):
    assert classification in CLASSIFICATIONS
    return {
        "number": number, "question": question, "classification": classification,
        "answer": answer, "evidence": evidence,
    }


def synthesize_results(artifacts: dict[str, Path], output_dir: Path) -> dict:
    reports = {name: _read(path) for name, path in artifacts.items()}
    decomposition = reports.get("decomposition", {})
    models = decomposition.get("models", {})
    property_model = models.get("A_property", {})
    property_market = models.get("B_property_market", {})
    neighborhood = models.get("C_property_market_neighborhood", {})
    incremental = decomposition.get("incremental_comparisons", {}).get(
        "B_property_market_to_C_property_market_neighborhood", {}
    )
    results = []
    if property_model.get("status") == "fitted":
        results.append(_result(1, "How much variation can property characteristics alone explain?", "Suggestive",
            f"The property-only hedonic model explained {property_model.get('in_sample_r_squared', 0):.3f} of in-sample log-price variation and had {_money(property_model.get('out_of_sample', {}).get('mae'))} future-period MAE.",
            ["hedonic decomposition", "temporal holdout"]))
    else:
        results.append(_result(1, "How much variation can property characteristics alone explain?", "Data-limited", "The property-only decomposition was unavailable.", []))
    if incremental:
        gain = incremental.get("out_of_sample_mae_improvement")
        results.append(_result(2, "How much additional information comes from neighborhood characteristics?", "Suggestive",
            f"Adding neighborhood controls changed future-period MAE by {_money(gain)} (positive means improvement) and changed in-sample R² by {incremental.get('delta_in_sample_r_squared', 0):.4f}.",
            ["nested hedonic decomposition", "temporal holdout"]))
    else:
        results.append(_result(2, "How much additional information comes from neighborhood characteristics?", "Data-limited", "Comparable property-plus-market and neighborhood models were unavailable.", []))
    transit, filters = reports.get("transit", {}), reports.get("filters", {})
    if transit:
        stable = filters.get("accessibility_findings_stable")
        core_consistent = (
            transit.get("survived_all_fitted_core_checks") is True
            or transit.get("attenuated_after_neighborhood_controls") is True
        )
        classification = "Robust" if stable is True and core_consistent else "Suggestive"
        results.append(_result(3, "Does CTA accessibility materially relate to value after controlling for location?", classification,
            transit.get("conclusion", "CTA robustness specifications were fitted."),
            ["progressive controls", "spatial dependence", "nonlinear distance", "filter sensitivity"]))
    else:
        results.append(_result(3, "Does CTA accessibility materially relate to value after controlling for location?", "Data-limited", "CTA robustness results were unavailable.", []))
    gradients = reports.get("gradients", {})
    half_decay = gradients.get("evidence", {}).get("lakefront_half_decay_distance_miles")
    results.append(_result(4, "How large and localized is the lakefront price gradient?",
        "Suggestive" if half_decay is not None else "Data-limited",
        f"The preferred conditional lake gradient reached half of its modeled decay by {half_decay:.2f} miles." if half_decay is not None else "A lakefront half-decay distance was not estimable.",
        ["nonlinear hedonic gradient", "future-period model selection"] if half_decay is not None else []))
    autocorrelation = reports.get("autocorrelation", {})
    if autocorrelation:
        results.append(_result(5, "Are hedonic-model residuals spatially autocorrelated?", "Robust",
            autocorrelation.get("conclusion", "Residual spatial diagnostics were completed."),
            ["Moran's I", "permutation inference", "multiple weight definitions"]))
    else:
        results.append(_result(5, "Are hedonic-model residuals spatially autocorrelated?", "Data-limited", "Residual spatial diagnostics were unavailable.", []))
    spatial = reports.get("spatial", {})
    spatial_metrics = spatial.get("spatial_block_validation", {}).get("metrics", {})
    ols_mae = spatial_metrics.get("ols", {}).get("mae")
    candidates = {name: value.get("mae") for name, value in spatial_metrics.items() if name != "ols" and isinstance(value, dict) and value.get("mae") is not None}
    if ols_mae is not None and candidates:
        best_name = min(candidates, key=candidates.get)
        gain = ols_mae - candidates[best_name]
        results.append(_result(6, "Do spatial econometric models materially improve on hedonic OLS?", "Suggestive",
            f"On the spatial block, {best_name} changed MAE by {_money(gain)} relative to OLS (positive means improvement).",
            ["spatial block holdout", "residual spatial diagnostics"]))
    else:
        results.append(_result(6, "Do spatial econometric models materially improve on hedonic OLS?", "Data-limited", "A same-block OLS-versus-spatial comparison was unavailable.", []))
    benchmark = reports.get("benchmark", {})
    if benchmark:
        best = benchmark.get("best_reported_primary_mae_model")
        results.append(_result(7, "Does ML outperform explicit spatial models?", "Exploratory",
            f"{best} had the lowest reported primary MAE, but the benchmark warns that holdout designs differ; this is not a uniform head-to-head victory.",
            ["main model benchmark", "validation-design comparability warning"]))
    else:
        results.append(_result(7, "Does ML outperform explicit spatial models?", "Data-limited", "The cross-model benchmark was unavailable.", []))
    temporal = reports.get("temporal", {})
    if temporal:
        selected = temporal.get("selected_by_validation_mae")
        metric = temporal.get("final_test_metrics", {}).get(selected, {})
        results.append(_result(8, "Does any ML advantage survive out-of-time testing?", "Data-limited",
            f"The validation-selected ML model ({selected}) recorded {_money(metric.get('mae'))} MAE on the untouched final period; explicit spatial models were not evaluated on this identical split.",
            ["train-validation-final-test sequence"]))
    else:
        results.append(_result(8, "Does any ML advantage survive out-of-time testing?", "Data-limited", "Final-period ML evidence was unavailable.", []))
    holdout = reports.get("spatial_holdout", {})
    spatial_schemes = [value for value in holdout.get("validation_schemes", []) if str(value).startswith("spatial_")]
    if spatial_schemes:
        results.append(_result(9, "Does any ML advantage survive geographic holdout testing?", "Data-limited",
            f"ML was evaluated under {len(spatial_schemes)} geographic holdout scheme(s); error penalties versus random folds are retained in the spatial report, but explicit spatial models lack identical folds.",
            ["geographically grouped cross-validation"]))
    else:
        results.append(_result(9, "Does any ML advantage survive geographic holdout testing?", "Data-limited", "Geographic holdout evidence was unavailable.", []))
    decay = reports.get("decay", {})
    radius = decay.get("best_available_sample_mae_radius")
    results.append(_result(10, "How far away can a comparable sale remain useful?",
        "Suggestive" if radius is not None and decay.get("common_sample_sales", 0) else "Data-limited",
        f"The lowest available-sample MAE occurred at {radius:g} miles; common-target marginal improvements should determine where added distance stops helping." if radius is not None else "The information-decay radius was not estimable.",
        ["strictly prior comparables", "common-target radius comparison"] if radius is not None else []))
    errors = reports.get("errors", {})
    worst = errors.get("worst_reliable_segments", [])
    if worst:
        leaders = ", ".join(f"{row.get('dimension')}={row.get('segment')} ({_percent(row.get('median_ape'))} MdAPE)" for row in worst[:3])
        results.append(_result(11, "Which property types or neighborhoods are hardest to value?", "Exploratory",
            f"The leading reliable error segments were {leaders}.", ["held-out segment error audit", "minimum group-size rule"]))
    else:
        results.append(_result(11, "Which property types or neighborhoods are hardest to value?", "Data-limited", "Reliable segment error rankings were unavailable.", []))
    stability = reports.get("stability", {})
    persistence = stability.get("overall_persistence_rate")
    results.append(_result(12, "How stable are neighborhood market archetypes over time?",
        "Exploratory" if persistence is not None else "Data-limited",
        f"The average adjacent-period regime persistence rate was {_percent(persistence)}; transitions describe relative regimes, not causal neighborhood change." if persistence is not None else "Longitudinal segment persistence was not estimable.",
        ["period re-estimation", "centroid label alignment", "adjusted Rand indices"] if persistence is not None else []))
    attribution_path = artifacts.get("attribution")
    attribution = reports.get("attribution", {})
    summary_path = attribution_path.parent / "property_value_decompositions.csv" if attribution_path else None
    if summary_path and summary_path.exists():
        import pandas as pd
        summary = pd.read_csv(summary_path)
        property_amount = float(summary["property_contribution"].median())
        place_amount = float(summary["place_contribution"].median())
        results.append(_result(13, "How much of an individual valuation is attributable to property versus place?", "Exploratory",
            f"Across {len(summary)} explained properties, median model attribution was {_money(property_amount)} for property features and {_money(place_amount)} for place features relative to the saved reference profile.",
            ["fitted-model Monte Carlo Shapley decomposition", "exact dollar reconciliation"]))
    else:
        results.append(_result(13, "How much of an individual valuation is attributable to property versus place?", "Data-limited", "Property-place attribution summaries were unavailable.", []))
    counts = {label: sum(result["classification"] == label for result in results) for label in CLASSIFICATIONS}
    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "created_at": generated, "questions_answered": len(results), "classification_counts": counts,
        "results": results,
        "classification_policy": {
            "Robust": "Supported by uncertainty-aware diagnostics and multiple relevant robustness checks.",
            "Suggestive": "Supported by held-out or controlled analysis, with material comparability or identification limits.",
            "Exploratory": "Descriptive model evidence useful for discovery but not a settled claim.",
            "Data-limited": "Required evidence is missing, non-comparable, or insufficient.",
        },
    }
    lines = ["# HomeValue: Results and Honest Comparison", "", f"Generated: {generated}", ""]
    for item in results:
        lines.extend([
            f"## {item['number']}. {item['question']}", "",
            f"**Classification: {item['classification']}**", "", item["answer"], "",
            "Evidence: " + (", ".join(item["evidence"]) if item["evidence"] else "not available"), "",
        ])
    markdown = "\n".join(lines)
    notebook = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in markdown.splitlines()]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "homevalue_results.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "homevalue_results.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "homevalue_results.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/results"))
    args = parser.parse_args()
    root = Path("data/processed")
    artifacts = {
        "decomposition": root / "decomposition/decomposition_results.json",
        "transit": root / "transit_robustness/transit_robustness_results.json",
        "filters": root / "validation/filter_sensitivity/filter_sensitivity_results.json",
        "gradients": root / "amenity_gradients/gradient_results.json",
        "autocorrelation": root / "spatial_autocorrelation/spatial_autocorrelation_report.json",
        "spatial": root / "spatial_error/spatial_error_results.json",
        "benchmark": root / "benchmark/model_benchmark_results.json",
        "temporal": root / "validation/out_of_time/out_of_time_results.json",
        "spatial_holdout": root / "validation/spatial/spatial_holdout_results.json",
        "decay": root / "information_decay/information_decay_results.json",
        "errors": root / "validation/error_segments/error_segment_results.json",
        "stability": root / "segmentation/stability/stability_report.json",
        "attribution": root / "explainability/property_place/property_place_results.json",
    }
    report = synthesize_results(artifacts, args.output)
    print(f"Answered {report['questions_answered']} questions: {report['classification_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
