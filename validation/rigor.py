"""Audit statistical rigor and quantify uncertainty in valuation claims."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.baselines import regression_metrics
from ml.valuation import MLConfig, _column_types, _estimators, select_features
from spatial.lag_model import _moran_residual, _weights
from validation.out_of_time import _fit_predict, chronological_split


@dataclass(frozen=True)
class RigorConfig:
    bootstrap_iterations: int = 500
    confidence_level: float = 0.95
    repeated_seeds: tuple[int, ...] = (7, 42, 99, 2024, 2025)
    repeated_model_estimators: int = 100
    spatial_neighbors: tuple[int, ...] = (4, 8, 12)
    spatial_permutations: int = 199
    outlier_tail_fraction: float = 0.01


def _metric_values(actual, predicted) -> tuple[float, float, float]:
    error = np.asarray(predicted, float) - np.asarray(actual, float)
    actual = np.asarray(actual, float)
    return (
        float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2))),
        float(np.median(np.abs(error / actual))),
    )


def _bootstrap(predictions: pd.DataFrame, columns: list[str], config: RigorConfig):
    rng = np.random.default_rng(42)
    actual = pd.to_numeric(predictions["sale_price"], errors="coerce").to_numpy(float)
    alpha = (1 - config.confidence_level) / 2
    rows, samples = [], {}
    for column in columns:
        predicted = pd.to_numeric(predictions[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
        y, estimate = actual[valid], predicted[valid]
        draws = np.empty((config.bootstrap_iterations, 3))
        for iteration in range(config.bootstrap_iterations):
            positions = rng.integers(0, len(y), len(y))
            draws[iteration] = _metric_values(y[positions], estimate[positions])
        samples[column] = (y, estimate)
        for metric_index, metric in enumerate(("mae", "rmse", "mdape")):
            point = _metric_values(y, estimate)[metric_index]
            rows.append({
                "model": column.removeprefix("prediction_"), "metric": metric,
                "estimate": point, "ci_lower": float(np.quantile(draws[:, metric_index], alpha)),
                "ci_upper": float(np.quantile(draws[:, metric_index], 1 - alpha)),
                "bootstrap_iterations": config.bootstrap_iterations, "n": len(y),
            })
    mae = pd.DataFrame(rows).query("metric == 'mae'").sort_values("estimate")
    comparison = None
    if len(mae) >= 2:
        best, runner = mae.iloc[0]["model"], mae.iloc[1]["model"]
        first_actual, first_prediction = samples[f"prediction_{best}"]
        second_actual, second_prediction = samples[f"prediction_{runner}"]
        # Inputs normally share rows; align by their common finite prefix if not.
        length = min(len(first_actual), len(second_actual))
        differences = []
        for _ in range(config.bootstrap_iterations):
            positions = rng.integers(0, length, length)
            best_mae = np.mean(np.abs(first_prediction[:length][positions] - first_actual[:length][positions]))
            runner_mae = np.mean(np.abs(second_prediction[:length][positions] - second_actual[:length][positions]))
            differences.append(runner_mae - best_mae)
        comparison = {
            "best_model": best, "runner_up_model": runner,
            "mae_advantage": float(np.mean(differences)),
            "ci_lower": float(np.quantile(differences, alpha)),
            "ci_upper": float(np.quantile(differences, 1 - alpha)),
            "clear_advantage": bool(np.quantile(differences, alpha) > 0),
        }
    return pd.DataFrame(rows), comparison


def _seed_stability(frame: pd.DataFrame, config: RigorConfig) -> pd.DataFrame:
    train, validation, test, _, _ = chronological_split(frame)
    development = pd.concat([train, validation], ignore_index=True)
    approved, _ = select_features(frame)
    numeric, categorical, _ = _column_types(development, approved, 100)
    features = [*numeric, *categorical]
    rows = []
    for seed in config.repeated_seeds:
        estimator = _estimators(MLConfig(
            random_seed=seed, random_forest_estimators=config.repeated_model_estimators,
            xgboost_estimators=config.repeated_model_estimators,
        ))["random_forest"]
        metrics, _, _, _ = _fit_predict(
            development, test, features, numeric, categorical, {"random_forest": estimator}
        )
        rows.append({"seed": seed, **metrics["random_forest"]})
    return pd.DataFrame(rows)


def _spatial_sensitivity(frame: pd.DataFrame, columns: list[str], config: RigorConfig) -> pd.DataFrame:
    if not {"x_3435", "y_3435"}.issubset(frame.columns) or len(frame) < 4:
        return pd.DataFrame()
    coordinates = frame[["x_3435", "y_3435"]].to_numpy(float)
    rows = []
    for neighbors in config.spatial_neighbors:
        k = min(neighbors, len(frame) - 1)
        weights = _weights(coordinates, k)
        for column in columns:
            valid = pd.to_numeric(frame[column], errors="coerce").notna()
            if valid.sum() != len(frame):
                continue
            residual = np.log(frame["sale_price"].to_numpy(float)) - np.log(frame[column].to_numpy(float))
            if np.std(residual) < 1e-12:
                continue
            result = _moran_residual(residual, weights, config.spatial_permutations, 42)
            rows.append({"model": column.removeprefix("prediction_"), "k_neighbors": k, **result})
    return pd.DataFrame(rows)


def run_statistical_rigor_audit(
    predictions_path: Path,
    data_path: Path,
    output_dir: Path,
    temporal_report_path: Path | None = None,
    spatial_report_path: Path | None = None,
    hedonic_report_path: Path | None = None,
    durbin_report_path: Path | None = None,
    config: RigorConfig | None = None,
) -> dict:
    config = config or RigorConfig()
    predictions, data = pd.read_parquet(predictions_path), pd.read_parquet(data_path)
    columns = [column for column in predictions if column.startswith("prediction_")]
    if not columns:
        raise ValueError("predictions contain no prediction_ columns")
    bootstrap, comparison = _bootstrap(predictions, columns, config)
    seed_results = _seed_stability(data, config)
    extras = [column for column in ("x_3435", "y_3435", "is_strict_market_sale", "is_moderate_market_sale") if column in data and column not in predictions]
    joined = predictions.merge(data[["sale_id", *extras]], on="sale_id", how="left", validate="one_to_one")
    spatial = _spatial_sensitivity(joined, columns, config)
    sensitivity_rows = []
    price = pd.to_numeric(joined["sale_price"], errors="coerce")
    lower, upper = price.quantile([config.outlier_tail_fraction, 1 - config.outlier_tail_fraction])
    filters = {"all_sales": pd.Series(True, index=joined.index), "trimmed_price_tails": price.between(lower, upper)}
    for flag in ("is_strict_market_sale", "is_moderate_market_sale"):
        if flag in joined:
            filters[flag] = joined[flag].fillna(False).astype(bool)
    for filter_name, mask in filters.items():
        for column in columns:
            valid = mask & pd.to_numeric(joined[column], errors="coerce").notna()
            metrics = regression_metrics(joined.loc[valid, "sale_price"], joined.loc[valid, column])
            sensitivity_rows.append({"filter": filter_name, "model": column.removeprefix("prediction_"), **metrics})
    sensitivity = pd.DataFrame(sensitivity_rows)
    temporal = json.loads(temporal_report_path.read_text()) if temporal_report_path and temporal_report_path.exists() else {}
    spatial_report = json.loads(spatial_report_path.read_text()) if spatial_report_path and spatial_report_path.exists() else {}
    hedonic = json.loads(hedonic_report_path.read_text()) if hedonic_report_path and hedonic_report_path.exists() else {}
    durbin = json.loads(durbin_report_path.read_text()) if durbin_report_path and durbin_report_path.exists() else {}
    checks = {
        "heteroskedasticity_robust_standard_errors": hedonic.get("config", {}).get("robust_covariance") in {"HC0", "HC1", "HC2", "HC3"},
        "spatial_diagnostics": bool(len(spatial)), "bootstrap_confidence_intervals": bool(len(bootstrap)),
        "repeated_model_seeds": len(seed_results) == len(config.repeated_seeds),
        "grouped_cross_validation": any(str(value).startswith("spatial_") for value in spatial_report.get("validation_schemes", [])),
        "temporal_holdout": temporal.get("final_test_was_used_for_selection") is False,
        "spatial_holdout": bool(spatial_report.get("validation_schemes")),
        "coefficient_stability_testing": "coefficient_stability_sar_to_sdm" in durbin.get("full_sample_comparison", {}),
        "alternative_spatial_weights": spatial["k_neighbors"].nunique() >= 2 if len(spatial) else False,
        "transaction_filter_sensitivity": any(name.startswith("is_") for name in filters),
        "outlier_sensitivity": "trimmed_price_tails" in filters,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(output_dir / "bootstrap_metric_intervals.csv", index=False)
    seed_results.to_csv(output_dir / "repeated_seed_metrics.csv", index=False)
    spatial.to_csv(output_dir / "spatial_weight_sensitivity.csv", index=False)
    sensitivity.to_csv(output_dir / "sample_sensitivity.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "checks": checks, "checks_passed": sum(checks.values()), "checks_total": len(checks),
        "paired_best_model_comparison": comparison,
        "seed_mae_range": [float(seed_results["mae"].min()), float(seed_results["mae"].max())],
        "claim_guidance": "Treat an effect as practically meaningful only when its magnitude matters and its uncertainty, validation, and sensitivity evidence are reported; statistical significance alone is insufficient.",
    }
    (output_dir / "statistical_rigor_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/validation/out_of_time/final_test_predictions.parquet"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation/rigor"))
    parser.add_argument("--temporal-report", type=Path, default=Path("data/processed/validation/out_of_time/out_of_time_results.json"))
    parser.add_argument("--spatial-report", type=Path, default=Path("data/processed/validation/spatial/spatial_holdout_results.json"))
    parser.add_argument("--hedonic-report", type=Path, default=Path("data/processed/hedonic/hedonic_results.json"))
    parser.add_argument("--durbin-report", type=Path, default=Path("data/processed/spatial_durbin/spatial_durbin_results.json"))
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()
    report = run_statistical_rigor_audit(
        args.predictions, args.data, args.output, args.temporal_report, args.spatial_report,
        args.hedonic_report, args.durbin_report,
        RigorConfig(bootstrap_iterations=args.bootstrap_iterations),
    )
    print(f"Statistical rigor checks passed: {report['checks_passed']}/{report['checks_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
