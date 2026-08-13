"""Decompose predictive information from property, market, and location."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hedonic.model import ACCESSIBILITY_FEATURES, HedonicConfig, HedonicModel
from ml.baselines import regression_metrics, temporal_split


MODEL_SPECS = {
    "A_property": HedonicConfig(
        include_time=False, include_property_type=True, include_neighborhood=False
    ),
    "B_property_market": HedonicConfig(
        include_time=True, include_property_type=True, include_neighborhood=False
    ),
    "C_property_market_neighborhood": HedonicConfig(
        include_time=True, include_property_type=True, include_neighborhood=True
    ),
    "D_property_market_neighborhood_accessibility": HedonicConfig(
        include_time=True, include_property_type=True, include_neighborhood=True,
        include_accessibility=True,
    ),
}


def _accessibility_available(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in ACCESSIBILITY_FEATURES
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
    ]


def run_decomposition(
    input_path: Path,
    output_dir: Path,
    test_start_year: int | None = None,
    minimum_category_count: int = 20,
) -> dict:
    frame = pd.read_parquet(input_path)
    frame = frame.loc[pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)].copy()
    train, test, cutoff = temporal_split(frame, test_start_year)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = [
        column for column in ("sale_id", "pin", "sale_date", "year", "sale_price")
        if column in test
    ]
    predictions = test[identity].copy()
    models = {}
    coefficients = []
    accessibility = _accessibility_available(train)

    for name, base_config in MODEL_SPECS.items():
        if name.startswith("D_") and not accessibility:
            models[name] = {
                "status": "unavailable",
                "reason": "No retained accessibility features are available; rerun after accessibility processing.",
            }
            continue
        config = HedonicConfig(
            minimum_category_count=minimum_category_count,
            maximum_neighborhood_categories=base_config.maximum_neighborhood_categories,
            robust_covariance=base_config.robust_covariance,
            include_time=base_config.include_time,
            include_property_type=base_config.include_property_type,
            include_neighborhood=base_config.include_neighborhood,
            include_accessibility=base_config.include_accessibility,
        )
        model = HedonicModel(config).fit(train)
        predicted = model.predict(test)
        predictions[f"prediction_{name}"] = predicted
        table = model.coefficient_table()
        table.insert(0, "model", name)
        coefficients.append(table)
        models[name] = {
            "status": "fitted",
            "design_columns": model.design_columns_,
            "in_sample_r_squared": float(model.result_.rsquared),
            "in_sample_adjusted_r_squared": float(model.result_.rsquared_adj),
            "out_of_sample": regression_metrics(test["sale_price"], predicted),
        }

    fitted_names = [name for name in MODEL_SPECS if models[name]["status"] == "fitted"]
    incremental = {}
    for previous, current in zip(fitted_names, fitted_names[1:]):
        before, after = models[previous], models[current]
        incremental[f"{previous}_to_{current}"] = {
            "delta_in_sample_r_squared": after["in_sample_r_squared"] - before["in_sample_r_squared"],
            "delta_in_sample_adjusted_r_squared": after["in_sample_adjusted_r_squared"] - before["in_sample_adjusted_r_squared"],
            "out_of_sample_mae_improvement": before["out_of_sample"]["mae"] - after["out_of_sample"]["mae"],
            "out_of_sample_rmse_improvement": before["out_of_sample"]["rmse"] - after["out_of_sample"]["rmse"],
        }

    location_key = "B_property_market_to_C_property_market_neighborhood"
    location = incremental.get(location_key)
    if location:
        answer = (
            "Adding neighborhood controls changed out-of-sample MAE by "
            f"${location['out_of_sample_mae_improvement']:,.0f} and added "
            f"{location['delta_in_sample_r_squared']:.4f} in-sample R-squared. "
            "Positive MAE improvement means location improved future-period prediction."
        )
    else:
        answer = "The property-plus-market and neighborhood models could not both be fitted."

    predictions.to_parquet(output_dir / "decomposition_predictions.parquet", index=False)
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(
            output_dir / "decomposition_coefficients.csv", index=False
        )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "test_start_year": cutoff,
        "train_rows": len(train),
        "test_rows": len(test),
        "available_accessibility_features": accessibility,
        "models": models,
        "incremental_comparisons": incremental,
        "central_question": "How much predictive information does location add after property and market controls?",
        "central_answer": answer,
        "interpretation_caution": "The decomposition measures incremental association and prediction, not causal neighborhood effects.",
    }
    (output_dir / "decomposition_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/decomposition"))
    parser.add_argument("--test-start-year", type=int)
    parser.add_argument("--minimum-category-count", type=int, default=20)
    args = parser.parse_args()
    report = run_decomposition(
        args.input, args.output, args.test_start_year, args.minimum_category_count
    )
    print(report["central_answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
