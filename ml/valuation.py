"""Train nonlinear valuation models with leakage-safe out-of-time validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from ml.baselines import regression_metrics, temporal_split


FEATURE_GROUPS = {
    "structural": (
        "building_sqft", "land_sqft", "bedrooms", "bathrooms", "stories",
        "building_age", "garage_spaces", "has_basement", "property_card_count",
        "residence_type", "construction_quality", "exterior_wall", "heating_type",
        "air_conditioning", "renovation", "class",
    ),
    "temporal": ("year", "month", "quarter"),
    "neighborhood": (
        "nbhd", "census_tract", "municipality", "community_area", "zip_code",
        "median_household_income", "poverty_rate", "bachelors_or_higher_rate",
        "graduate_degree_rate", "owner_occupancy_rate", "renter_occupancy_rate",
        "vacancy_rate", "median_housing_age", "average_household_size",
        "tract_population", "housing_units", "transit_commute_share",
        "automobile_commute_share", "population_density", "housing_unit_density",
    ),
    "accessibility": (
        "cta_distance_miles", "cta_stations_half_mile", "cta_stations_one_mile",
        "nearest_cta_line", "lake_distance_miles", "downtown_distance_miles",
        "park_distance_miles",
    ),
    "prior_spatial": (
        "prior_nearby_sale_median", "prior_nearby_sale_count",
        "prior_nearby_weighted_ppsf", "neighborhood_prior_appreciation",
    ),
}


@dataclass(frozen=True)
class MLConfig:
    random_seed: int = 42
    random_forest_estimators: int = 400
    xgboost_estimators: int = 600
    maximum_category_levels: int = 100


def select_features(frame: pd.DataFrame) -> tuple[list[str], dict[str, list[str]]]:
    """Select only approved pre-sale features; identifiers and raw coordinates never enter."""
    selected_by_group = {
        group: [column for column in columns if column in frame]
        for group, columns in FEATURE_GROUPS.items()
    }
    selected = [column for columns in selected_by_group.values() for column in columns]
    if not selected:
        raise ValueError("no approved valuation features are available")
    return selected, selected_by_group


def _column_types(train: pd.DataFrame, columns: list[str], maximum_levels: int) -> tuple[list[str], list[str], list[str]]:
    categorical, numeric, dropped = [], [], []
    for column in columns:
        series = train[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
        elif series.nunique(dropna=True) <= maximum_levels:
            categorical.append(column)
        else:
            dropped.append(column)
    return numeric, categorical, dropped


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop", verbose_feature_names_out=True)


def _estimators(config: MLConfig) -> dict:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=config.random_forest_estimators, min_samples_leaf=3,
            max_features=0.8, n_jobs=-1, random_state=config.random_seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=350, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=config.random_seed,
        ),
        "xgboost": XGBRegressor(
            n_estimators=config.xgboost_estimators, learning_rate=0.04,
            max_depth=6, min_child_weight=5, subsample=0.85,
            colsample_bytree=0.85, reg_lambda=2.0, objective="reg:squarederror",
            n_jobs=-1, random_state=config.random_seed,
        ),
    }


def _external_benchmarks(test: pd.DataFrame, paths: list[Path]) -> dict:
    results = {}
    for path in paths:
        if not path.exists():
            continue
        predictions = pd.read_parquet(path)
        if "sale_id" not in predictions:
            continue
        prediction_columns = [column for column in predictions if column.startswith("prediction_")]
        joined = test[["sale_id", "sale_price"]].merge(
            predictions[["sale_id", *prediction_columns]], on="sale_id", how="inner"
        )
        for column in prediction_columns:
            valid = pd.to_numeric(joined[column], errors="coerce").notna()
            if valid.any():
                name = column.removeprefix("prediction_")
                if name in results:
                    name = f"{path.parent.name}_{name}"
                results[name] = regression_metrics(
                    joined.loc[valid, "sale_price"], joined.loc[valid, column]
                )
    return results


def train_ml_models(
    input_path: Path,
    output_dir: Path,
    test_start_year: int | None = None,
    config: MLConfig | None = None,
    benchmark_paths: list[Path] | None = None,
) -> dict:
    config = config or MLConfig()
    frame = pd.read_parquet(input_path)
    frame = frame.loc[pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)].copy()
    train, test, cutoff = temporal_split(frame, test_start_year)
    approved, groups = select_features(frame)
    numeric, categorical, dropped_high_cardinality = _column_types(
        train, approved, config.maximum_category_levels
    )
    features = [*numeric, *categorical]
    if not features:
        raise ValueError("no usable approved features remain")
    preprocessor = build_preprocessor(numeric, categorical)
    transformed_train = preprocessor.fit_transform(train[features])
    transformed_test = preprocessor.transform(test[features])
    target_train = np.log(pd.to_numeric(train["sale_price"], errors="coerce"))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[[
        column for column in ("sale_id", "pin", "sale_date", "year", "sale_price")
        if column in test
    ]].copy()
    metrics, fitted = {}, {}
    for name, estimator in _estimators(config).items():
        estimator.fit(transformed_train, target_train)
        train_log_prediction = estimator.predict(transformed_train)
        smearing = float(np.exp(target_train.to_numpy() - train_log_prediction).mean())
        dollar_prediction = np.exp(np.clip(estimator.predict(transformed_test), None, 50)) * smearing
        predictions[f"prediction_{name}"] = dollar_prediction
        metrics[name] = {
            **regression_metrics(test["sale_price"], dollar_prediction),
            "smearing_factor": smearing,
        }
        fitted[name] = estimator
    joblib.dump(
        {"preprocessor": preprocessor, "models": fitted, "features": features},
        output_dir / "ml_models.joblib",
    )
    predictions.to_parquet(output_dir / "ml_predictions.parquet", index=False)
    default_benchmarks = [
        Path("data/processed/baselines/baseline_predictions.parquet"),
        Path("data/processed/hedonic/hedonic_predictions.parquet"),
        Path("data/processed/comparables/comparable_predictions.parquet"),
        Path("data/processed/spatial_error/spatial_error_block_predictions.parquet"),
    ]
    external = _external_benchmarks(
        test, default_benchmarks if benchmark_paths is None else benchmark_paths
    )
    combined = {**external, **{f"ml_{name}": value for name, value in metrics.items()}}
    best = min(combined, key=lambda name: combined[name]["mae"]) if combined else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path), "test_start_year": cutoff,
        "train_rows": len(train), "test_rows": len(test),
        "train_years": sorted(pd.to_numeric(train["year"]).astype(int).unique().tolist()),
        "test_years": sorted(pd.to_numeric(test["year"]).astype(int).unique().tolist()),
        "config": asdict(config), "feature_groups": groups,
        "numeric_features": numeric, "categorical_features": categorical,
        "dropped_high_cardinality_features": dropped_high_cardinality,
        "transformed_feature_count": int(transformed_train.shape[1]),
        "excluded_by_design": [
            "sale_price and all target-derived fields", "sale_id, PIN, and document identifiers",
            "raw latitude/longitude", "future or contemporaneous nearby sale outcomes",
            "post-sale quality flags and model predictions",
        ],
        "ml_metrics": metrics, "external_benchmark_metrics": external,
        "best_test_mae_model": best,
        "ml_materially_outperformed_available_benchmarks": (
            best is not None and best.startswith("ml_")
        ),
    }
    (output_dir / "ml_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/ml"))
    parser.add_argument("--test-start-year", type=int)
    parser.add_argument("--random-forest-estimators", type=int, default=400)
    parser.add_argument("--xgboost-estimators", type=int, default=600)
    args = parser.parse_args()
    config = MLConfig(
        random_forest_estimators=args.random_forest_estimators,
        xgboost_estimators=args.xgboost_estimators,
    )
    report = train_ml_models(args.input, args.output, args.test_start_year, config)
    print(f"Best available test-MAE model: {report['best_test_mae_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
