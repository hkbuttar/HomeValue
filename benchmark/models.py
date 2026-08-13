"""Assemble the main cross-model valuation benchmark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODEL_ORDER = (
    "Median", "PPSF Baseline", "Comparable Sales", "Hedonic OLS",
    "Spatial Lag", "Spatial Error", "Gradient Boosting",
)


def _read(path: Path | None) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}


def _metric(metrics: dict | None, name: str):
    if not metrics:
        return None
    aliases = {
        "mdape": ("median_absolute_percentage_error", "mdape"),
        "mae": ("mae",), "rmse": ("rmse",),
    }
    for key in aliases[name]:
        if metrics.get(key) is not None:
            return metrics[key]
    return None


def _moran(report: dict, model: str):
    values = report.get("full_sample_comparison", {}).get(model, {})
    moran = values.get("residual_moran", {}) if isinstance(values, dict) else {}
    return moran.get("moran_i")


def _spatial_metrics(report: dict, model: str):
    return report.get("spatial_block_validation", {}).get("metrics", {}).get(model)


def _row(name: str, primary: dict | None, source: str, temporal: dict | None = None,
         spatial: dict | None = None, moran=None) -> dict:
    return {
        "model": name, "mae": _metric(primary, "mae"), "rmse": _metric(primary, "rmse"),
        "mdape": _metric(primary, "mdape"), "primary_metric_source": source,
        "spatial_residual_morans_i": moran,
        "temporal_test": temporal is not None, "temporal_mae": _metric(temporal, "mae"),
        "spatial_test": spatial is not None, "spatial_mae": _metric(spatial, "mae"),
    }


def _markdown(table: pd.DataFrame) -> str:
    def display(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    columns = table.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(display(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def build_model_benchmark(
    output_dir: Path,
    baseline_path: Path | None = None,
    comparables_path: Path | None = None,
    hedonic_path: Path | None = None,
    spatial_lag_path: Path | None = None,
    spatial_error_path: Path | None = None,
    temporal_path: Path | None = None,
    spatial_holdout_path: Path | None = None,
) -> dict:
    baseline, comparables, hedonic = _read(baseline_path), _read(comparables_path), _read(hedonic_path)
    lag, error = _read(spatial_lag_path), _read(spatial_error_path)
    temporal, holdout = _read(temporal_path), _read(spatial_holdout_path)
    baseline_metrics = baseline.get("metrics", {})
    temporal_metrics = temporal.get("final_test_metrics", {})
    spatial_holdout_metrics = holdout.get("metrics", [])

    def held_out(model: str):
        candidates = [
            row for row in spatial_holdout_metrics
            if row.get("model") == model and str(row.get("validation_scheme", "")).startswith("spatial_")
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda row: row.get("mae", float("inf")))

    rows = [
        _row("Median", baseline_metrics.get("global_median"), "temporal_holdout",
             temporal=baseline_metrics.get("global_median")),
        _row("PPSF Baseline", baseline_metrics.get("segmented_ppsf"), "temporal_holdout",
             temporal=baseline_metrics.get("segmented_ppsf")),
        _row("Comparable Sales", comparables.get("weighted_price_metrics"), "latest_year",
             temporal=comparables.get("weighted_price_metrics")),
        _row("Hedonic OLS", hedonic.get("metrics_dollars"), "temporal_holdout",
             temporal=hedonic.get("metrics_dollars"), moran=_moran(lag, "ols")),
    ]
    lag_spatial = _spatial_metrics(lag, "spatial_lag") or _spatial_metrics(error, "spatial_lag")
    error_spatial = _spatial_metrics(error, "spatial_error")
    rows.extend([
        _row("Spatial Lag", lag_spatial, "spatial_block_holdout", spatial=lag_spatial,
             moran=_moran(lag, "spatial_lag") or _moran(error, "spatial_lag")),
        _row("Spatial Error", error_spatial, "spatial_block_holdout", spatial=error_spatial,
             moran=_moran(error, "spatial_error")),
    ])
    gradient_name = "hist_gradient_boosting"
    gradient_temporal = temporal_metrics.get(gradient_name)
    gradient_spatial = held_out(gradient_name)
    rows.append(_row(
        "Gradient Boosting", gradient_temporal, "final_temporal_holdout",
        temporal=gradient_temporal, spatial=gradient_spatial,
    ))
    table = pd.DataFrame(rows)
    available = table.dropna(subset=["mae"])
    if available.empty:
        raise ValueError("no benchmark metrics were found in the supplied artifacts")
    table["primary_mae_rank"] = table["mae"].rank(method="min").astype("Int64")
    best = available.loc[available["mae"].idxmin(), "model"]
    spatial_models = table.loc[table["model"].isin(["Spatial Lag", "Spatial Error"])].dropna(subset=["mae"])
    spatial_note = None
    if len(spatial_models):
        best_spatial = spatial_models.loc[spatial_models["mae"].idxmin()]
        spatial_note = {
            "model": best_spatial["model"], "mae": float(best_spatial["mae"]),
            "residual_morans_i": (
                float(best_spatial["spatial_residual_morans_i"])
                if pd.notna(best_spatial["spatial_residual_morans_i"]) else None
            ),
        }
    conclusion = (
        f"{best} had the lowest reported primary MAE. Primary scores span different holdout designs, so this ranking is descriptive, not proof of universal superiority. "
        "Predictive accuracy and explanatory clarity are reported as separate objectives."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "model_benchmark.csv", index=False)
    markdown = _markdown(table)
    (output_dir / "model_benchmark.md").write_text(markdown + "\n", encoding="utf-8")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "models": MODEL_ORDER,
        "best_reported_primary_mae_model": best, "best_spatial_model": spatial_note,
        "rows": table.astype(object).where(pd.notna(table), None).to_dict(orient="records"),
        "conclusion": conclusion,
        "comparability_caution": "Compare temporal MAE with temporal MAE and spatial MAE with spatial MAE. The primary column preserves the strongest held-out result available per model and is not a perfectly uniform contest.",
    }
    (output_dir / "model_benchmark_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/benchmark"))
    parser.add_argument("--baselines", type=Path, default=Path("data/processed/baselines/baseline_results.json"))
    parser.add_argument("--comparables", type=Path, default=Path("data/processed/comparables/comparable_results.json"))
    parser.add_argument("--hedonic", type=Path, default=Path("data/processed/hedonic/hedonic_results.json"))
    parser.add_argument("--spatial-lag", type=Path, default=Path("data/processed/spatial_lag/spatial_lag_results.json"))
    parser.add_argument("--spatial-error", type=Path, default=Path("data/processed/spatial_error/spatial_error_results.json"))
    parser.add_argument("--temporal", type=Path, default=Path("data/processed/validation/out_of_time/out_of_time_results.json"))
    parser.add_argument("--spatial-holdout", type=Path, default=Path("data/processed/validation/spatial/spatial_holdout_results.json"))
    args = parser.parse_args()
    report = build_model_benchmark(
        args.output, args.baselines, args.comparables, args.hedonic, args.spatial_lag,
        args.spatial_error, args.temporal, args.spatial_holdout,
    )
    print(report["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
