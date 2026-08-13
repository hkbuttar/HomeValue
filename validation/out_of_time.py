"""Run train, later-validation, and final-test valuation benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.baselines import regression_metrics
from ml.valuation import MLConfig, _column_types, _estimators, build_preprocessor, select_features


@dataclass(frozen=True)
class OutOfTimeConfig:
    validation_start_year: int | None = None
    test_start_year: int | None = None
    random_seed: int = 42
    random_forest_estimators: int = 400
    xgboost_estimators: int = 600
    maximum_category_levels: int = 100


def chronological_split(frame: pd.DataFrame, validation_start_year: int | None = None,
                        test_start_year: int | None = None):
    data = frame.copy()
    if "year" not in data:
        if "sale_date" not in data:
            raise ValueError("data requires year or sale_date for out-of-time validation")
        data["year"] = pd.to_datetime(data["sale_date"], errors="coerce").dt.year
    year = pd.to_numeric(data["year"], errors="coerce")
    years = sorted(year.dropna().astype(int).unique())
    if len(years) < 3:
        raise ValueError("out-of-time validation requires at least three sale years")
    validation_cutoff = int(validation_start_year or years[-2])
    test_cutoff = int(test_start_year or years[-1])
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start_year must precede test_start_year")
    train = data.loc[year.lt(validation_cutoff)].copy()
    validation = data.loc[year.ge(validation_cutoff) & year.lt(test_cutoff)].copy()
    test = data.loc[year.ge(test_cutoff)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("cutoffs must leave nonempty train, validation, and final-test sets")
    return train, validation, test, validation_cutoff, test_cutoff


def _fit_predict(train, evaluation, features, numeric, categorical, estimators):
    preprocessor = build_preprocessor(numeric, categorical)
    transformed_train = preprocessor.fit_transform(train[features])
    transformed_evaluation = preprocessor.transform(evaluation[features])
    target = np.log(pd.to_numeric(train["sale_price"], errors="coerce"))
    metrics, predictions, fitted = {}, {}, {}
    for name, estimator in estimators.items():
        model = clone(estimator).fit(transformed_train, target)
        train_prediction = model.predict(transformed_train)
        smearing = float(np.exp(target.to_numpy() - train_prediction).mean())
        prediction = np.exp(np.clip(model.predict(transformed_evaluation), None, 50)) * smearing
        predictions[name] = prediction
        metrics[name] = {
            **regression_metrics(evaluation["sale_price"], prediction),
            "smearing_factor": smearing,
        }
        fitted[name] = model
    return metrics, predictions, preprocessor, fitted


def run_out_of_time_validation(input_path: Path, output_dir: Path,
                               config: OutOfTimeConfig | None = None) -> dict:
    config = config or OutOfTimeConfig()
    frame = pd.read_parquet(input_path)
    frame = frame.loc[pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)].copy()
    train, validation, test, validation_cutoff, test_cutoff = chronological_split(
        frame, config.validation_start_year, config.test_start_year
    )
    approved, groups = select_features(frame)
    numeric, categorical, dropped = _column_types(train, approved, config.maximum_category_levels)
    features = [*numeric, *categorical]
    if not features:
        raise ValueError("no usable approved valuation features remain")
    estimators = _estimators(MLConfig(
        random_seed=config.random_seed,
        random_forest_estimators=config.random_forest_estimators,
        xgboost_estimators=config.xgboost_estimators,
        maximum_category_levels=config.maximum_category_levels,
    ))
    validation_metrics, validation_predictions, _, _ = _fit_predict(
        train, validation, features, numeric, categorical, estimators
    )
    selected = min(validation_metrics, key=lambda name: validation_metrics[name]["mae"])
    development = pd.concat([train, validation], ignore_index=True)
    final_numeric, final_categorical, final_dropped = _column_types(
        development, approved, config.maximum_category_levels
    )
    final_features = [*final_numeric, *final_categorical]
    test_metrics, test_predictions, preprocessor, fitted = _fit_predict(
        development, test, final_features, final_numeric, final_categorical, estimators
    )
    identifiers = [
        column for column in ("sale_id", "pin", "sale_date", "year", "sale_price") if column in frame
    ]
    validation_output, test_output = validation[identifiers].copy(), test[identifiers].copy()
    for name, prediction in validation_predictions.items():
        validation_output[f"prediction_{name}"] = prediction
    for name, prediction in test_predictions.items():
        test_output[f"prediction_{name}"] = prediction
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_output.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    test_output.to_parquet(output_dir / "final_test_predictions.parquet", index=False)
    joblib.dump(
        {"preprocessor": preprocessor, "models": fitted, "selected_model": selected,
         "features": final_features},
        output_dir / "final_models.joblib",
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
        "config": asdict(config), "validation_start_year": validation_cutoff,
        "test_start_year": test_cutoff,
        "train_years": sorted(pd.to_numeric(train["year"]).astype(int).unique().tolist()),
        "validation_years": sorted(pd.to_numeric(validation["year"]).astype(int).unique().tolist()),
        "final_test_years": sorted(pd.to_numeric(test["year"]).astype(int).unique().tolist()),
        "train_rows": len(train), "validation_rows": len(validation), "final_test_rows": len(test),
        "feature_groups": groups, "features": final_features,
        "dropped_high_cardinality_features": sorted(set(dropped + final_dropped)),
        "validation_metrics": validation_metrics, "selected_by_validation_mae": selected,
        "final_test_metrics": test_metrics, "final_test_was_used_for_selection": False,
        "design": "Models were selected on later validation sales, refit on train plus validation, and evaluated once on the most recent held-out period.",
    }
    (output_dir / "out_of_time_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation/out_of_time"))
    parser.add_argument("--validation-start-year", type=int)
    parser.add_argument("--test-start-year", type=int)
    parser.add_argument("--random-forest-estimators", type=int, default=400)
    parser.add_argument("--xgboost-estimators", type=int, default=600)
    args = parser.parse_args()
    report = run_out_of_time_validation(args.input, args.output, OutOfTimeConfig(
        validation_start_year=args.validation_start_year, test_start_year=args.test_start_year,
        random_forest_estimators=args.random_forest_estimators,
        xgboost_estimators=args.xgboost_estimators,
    ))
    print(f"Selected {report['selected_by_validation_mae']}; scored {len(report['final_test_metrics'])} models on final test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
