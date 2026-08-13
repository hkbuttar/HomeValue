"""Calibrate valuation intervals and evaluate their held-out coverage."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalConfig:
    nominal_coverage: float = 0.90
    minimum_group_size: int = 20
    price_groups: int = 10


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if not 0 < coverage < 1:
        raise ValueError("nominal coverage must be between zero and one")
    if not len(values):
        raise ValueError("no finite calibration residuals are available")
    probability = min(1.0, math.ceil((len(values) + 1) * coverage) / len(values))
    return float(np.quantile(values, probability, method="higher"))


def _prediction_columns(calibration: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    columns = sorted(
        column for column in calibration
        if column.startswith("prediction_") and column in test
    )
    if not columns:
        raise ValueError("calibration and test inputs share no prediction_ columns")
    return columns


def _log_scores(frame: pd.DataFrame, prediction_column: str) -> np.ndarray:
    actual = pd.to_numeric(frame["sale_price"], errors="coerce")
    predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = actual.gt(0) & predicted.gt(0)
    return np.abs(np.log(actual.loc[valid].to_numpy()) - np.log(predicted.loc[valid].to_numpy()))


def _coverage_summary(intervals: pd.DataFrame, dimensions: dict[str, str], minimum: int) -> pd.DataFrame:
    rows = []
    for model, model_data in intervals.groupby("model", observed=True):
        groups = {"overall": pd.Series("all", index=model_data.index)}
        groups.update({name: model_data[column] for name, column in dimensions.items()})
        for dimension, values in groups.items():
            working = model_data.assign(_segment=values)
            for segment, group in working.loc[working["_segment"].notna()].groupby("_segment", observed=True):
                count = len(group)
                rows.append({
                    "model": model, "dimension": dimension, "segment": str(segment), "n": count,
                    "reliable_group": count >= minimum,
                    "empirical_coverage": float(group["covered"].mean()),
                    "mean_interval_width": float(group["interval_width"].mean()),
                    "median_interval_width": float(group["interval_width"].median()),
                    "median_relative_width": float(group["relative_width"].median()),
                })
    return pd.DataFrame(rows)


def calibrate_valuation_intervals(
    calibration_path: Path,
    test_path: Path,
    output_dir: Path,
    features_path: Path | None = None,
    config: IntervalConfig | None = None,
) -> dict:
    config = config or IntervalConfig()
    calibration, test = pd.read_parquet(calibration_path), pd.read_parquet(test_path)
    prediction_columns = _prediction_columns(calibration, test)
    if features_path and features_path.exists():
        features = pd.read_parquet(features_path)
        if "sale_id" not in test or "sale_id" not in features:
            raise ValueError("test and feature data require sale_id")
        extras = [
            column for column in ("nbhd", "census_tract", "community_area", "municipality")
            if column in features and column not in test
        ]
        test = test.merge(
            features[["sale_id", *extras]], on="sale_id", how="left", validate="one_to_one"
        )
    actual = pd.to_numeric(test["sale_price"], errors="coerce")
    valid_actual = actual.gt(0)
    test = test.loc[valid_actual].copy()
    actual = actual.loc[valid_actual]
    try:
        test["_price_group"] = pd.qcut(
            actual, config.price_groups, labels=False, duplicates="drop"
        ).map(lambda value: f"D{int(value) + 1}" if pd.notna(value) else pd.NA)
    except ValueError:
        test["_price_group"] = pd.NA
    dimensions = {"sale_price_group": "_price_group"}
    neighborhood = next(
        (column for column in ("nbhd", "census_tract", "community_area") if column in test), None
    )
    if neighborhood:
        dimensions["neighborhood"] = neighborhood
    interval_parts, quantiles = [], {}
    for prediction_column in prediction_columns:
        model = prediction_column.removeprefix("prediction_")
        scores = _log_scores(calibration, prediction_column)
        radius = conformal_quantile(scores, config.nominal_coverage)
        quantiles[model] = {"log_residual_radius": radius, "calibration_rows": len(scores)}
        predicted = pd.to_numeric(test[prediction_column], errors="coerce")
        valid = predicted.gt(0) & predicted.notna()
        result = test.loc[valid, [
            column for column in ("sale_id", "sale_date", "year", "sale_price", *dimensions.values())
            if column in test
        ]].copy()
        result["model"], result["estimated_value"] = model, predicted.loc[valid]
        result["interval_lower"] = predicted.loc[valid] * np.exp(-radius)
        result["interval_upper"] = predicted.loc[valid] * np.exp(radius)
        result["interval_width"] = result["interval_upper"] - result["interval_lower"]
        result["relative_width"] = result["interval_width"] / result["estimated_value"]
        result["covered"] = actual.loc[valid].between(result["interval_lower"], result["interval_upper"])
        interval_parts.append(result)
    intervals = pd.concat(interval_parts, ignore_index=True)
    coverage = _coverage_summary(intervals, dimensions, config.minimum_group_size)
    curve_rows = []
    for prediction_column in prediction_columns:
        model = prediction_column.removeprefix("prediction_")
        scores = _log_scores(calibration, prediction_column)
        predicted = pd.to_numeric(test[prediction_column], errors="coerce")
        valid = predicted.gt(0) & predicted.notna()
        for nominal in (0.50, 0.80, 0.90, 0.95):
            radius = conformal_quantile(scores, nominal)
            covered = actual.loc[valid].between(
                predicted.loc[valid] * np.exp(-radius), predicted.loc[valid] * np.exp(radius)
            )
            curve_rows.append({
                "model": model, "nominal_coverage": nominal,
                "empirical_coverage": float(covered.mean()), "n": int(valid.sum()),
                "log_residual_radius": radius,
            })
    curve = pd.DataFrame(curve_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals.to_parquet(output_dir / "valuation_intervals.parquet", index=False)
    coverage.to_csv(output_dir / "interval_coverage.csv", index=False)
    curve.to_csv(output_dir / "calibration_curve.csv", index=False)
    overall = coverage.loc[coverage["dimension"].eq("overall")]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration_input": str(calibration_path), "test_input": str(test_path),
        "features_input": str(features_path) if features_path else None,
        "config": asdict(config), "models": sorted(quantiles), "calibration": quantiles,
        "overall_coverage": overall.set_index("model")["empirical_coverage"].to_dict(),
        "overall_median_width": overall.set_index("model")["median_interval_width"].to_dict(),
        "coverage_dimensions": list(dimensions),
        "validity_caution": "Split-conformal marginal coverage relies on calibration and future sales being sufficiently exchangeable; temporal or neighborhood drift can reduce coverage.",
    }
    (output_dir / "interval_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=Path("data/processed/validation/out_of_time/validation_predictions.parquet"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/validation/out_of_time/final_test_predictions.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation/intervals"))
    parser.add_argument("--nominal-coverage", type=float, default=.90)
    parser.add_argument("--minimum-group-size", type=int, default=20)
    args = parser.parse_args()
    report = calibrate_valuation_intervals(
        args.calibration, args.test, args.output, args.features,
        IntervalConfig(nominal_coverage=args.nominal_coverage, minimum_group_size=args.minimum_group_size),
    )
    print(f"Calibrated {len(report['models'])} models at {args.nominal_coverage:.0%} nominal coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
