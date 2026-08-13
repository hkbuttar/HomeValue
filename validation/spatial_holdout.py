"""Compare random, temporal, and geographically grouped valuation validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from ml.valuation import MLConfig, _column_types, _estimators, select_features
from validation.out_of_time import _fit_predict


GEOGRAPHY_CANDIDATES = ("census_tract", "nbhd", "municipality")


@dataclass(frozen=True)
class SpatialValidationConfig:
    folds: int = 5
    random_seed: int = 42
    random_forest_estimators: int = 400
    xgboost_estimators: int = 600
    maximum_category_levels: int = 100
    temporal_test_start_year: int | None = None


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)].copy()
    if "year" not in data:
        if "sale_date" not in data:
            raise ValueError("validation data requires year or sale_date")
        data["year"] = pd.to_datetime(data["sale_date"], errors="coerce").dt.year
    return data.reset_index(drop=True)


def _folds(frame: pd.DataFrame, config: SpatialValidationConfig):
    if config.folds < 2:
        raise ValueError("folds must be at least two")
    random_folds = min(config.folds, len(frame))
    for fold, (train, test) in enumerate(KFold(
        n_splits=random_folds, shuffle=True, random_state=config.random_seed
    ).split(frame), start=1):
        yield "random", fold, train, test
    years = sorted(pd.to_numeric(frame["year"], errors="coerce").dropna().astype(int).unique())
    if len(years) < 2:
        raise ValueError("temporal comparison requires at least two sale years")
    cutoff = int(config.temporal_test_start_year or years[-1])
    train = np.flatnonzero(pd.to_numeric(frame["year"], errors="coerce").lt(cutoff))
    test = np.flatnonzero(pd.to_numeric(frame["year"], errors="coerce").ge(cutoff))
    if not len(train) or not len(test):
        raise ValueError("temporal cutoff must leave nonempty train and test sets")
    yield "temporal", 1, train, test
    found = False
    for geography in GEOGRAPHY_CANDIDATES:
        if geography not in frame:
            continue
        groups = frame[geography].astype("string").fillna("<missing>")
        group_count = groups.nunique()
        if group_count < 2:
            continue
        found = True
        splitter = GroupKFold(n_splits=min(config.folds, group_count))
        for fold, (train, test) in enumerate(splitter.split(frame, groups=groups), start=1):
            yield f"spatial_{geography}", fold, train, test
    if not found:
        raise ValueError("spatial validation requires tract, neighborhood, or municipality groups")


def run_spatial_holdout_validation(input_path: Path, output_dir: Path,
                                   config: SpatialValidationConfig | None = None) -> dict:
    config = config or SpatialValidationConfig()
    frame = _prepare(pd.read_parquet(input_path))
    approved, groups = select_features(frame)
    estimators = _estimators(MLConfig(
        random_seed=config.random_seed,
        random_forest_estimators=config.random_forest_estimators,
        xgboost_estimators=config.xgboost_estimators,
        maximum_category_levels=config.maximum_category_levels,
    ))
    prediction_parts = []
    for scheme, fold, train_indices, test_indices in _folds(frame, config):
        train, test = frame.iloc[train_indices], frame.iloc[test_indices]
        numeric, categorical, _ = _column_types(train, approved, config.maximum_category_levels)
        features = [*numeric, *categorical]
        if not features:
            raise ValueError("no usable approved features remain in a validation fold")
        _, predictions, _, _ = _fit_predict(
            train, test, features, numeric, categorical, estimators
        )
        identifiers = [
            column for column in ("sale_id", "pin", "sale_date", "year", "sale_price")
            if column in test
        ]
        result = test[identifiers].copy()
        result["validation_scheme"], result["fold"] = scheme, fold
        for name, values in predictions.items():
            result[f"prediction_{name}"] = values
        prediction_parts.append(result)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows = []
    from ml.baselines import regression_metrics
    for scheme, scheme_data in predictions.groupby("validation_scheme", observed=True):
        for name in estimators:
            values = regression_metrics(scheme_data["sale_price"], scheme_data[f"prediction_{name}"])
            metric_rows.append({"validation_scheme": scheme, "model": name, **values})
    metrics = pd.DataFrame(metric_rows).sort_values(["validation_scheme", "model"])
    random_mae = metrics.loc[metrics["validation_scheme"].eq("random")].set_index("model")["mae"]
    comparisons = []
    for row in metrics.itertuples(index=False):
        baseline = random_mae.get(row.model)
        comparisons.append({
            "validation_scheme": row.validation_scheme, "model": row.model,
            "mae_increase_vs_random": float(row.mae - baseline) if baseline is not None else None,
            "mae_ratio_vs_random": float(row.mae / baseline) if baseline and baseline > 0 else None,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "holdout_predictions.parquet", index=False)
    metrics.to_csv(output_dir / "holdout_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output_dir / "random_comparison.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
        "config": asdict(config), "feature_groups": groups,
        "validation_schemes": sorted(metrics["validation_scheme"].unique().tolist()),
        "models": list(estimators), "metrics": metric_rows, "comparisons_to_random": comparisons,
        "interpretation": "A large spatial-versus-random error increase indicates limited geographic transfer and possible reliance on hyperlocal patterns.",
    }
    (output_dir / "spatial_holdout_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation/spatial"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--temporal-test-start-year", type=int)
    parser.add_argument("--random-forest-estimators", type=int, default=400)
    parser.add_argument("--xgboost-estimators", type=int, default=600)
    args = parser.parse_args()
    report = run_spatial_holdout_validation(args.input, args.output, SpatialValidationConfig(
        folds=args.folds, temporal_test_start_year=args.temporal_test_start_year,
        random_forest_estimators=args.random_forest_estimators,
        xgboost_estimators=args.xgboost_estimators,
    ))
    print(f"Compared {len(report['models'])} models across {len(report['validation_schemes'])} schemes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
