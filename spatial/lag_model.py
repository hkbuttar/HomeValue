"""Estimate a spatial autoregressive lag model and compare it with OLS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda import Moran
from libpysal.weights import KNN
from sklearn.cluster import KMeans
from spreg import ML_Lag

from hedonic.model import HedonicConfig, HedonicModel
from ml.baselines import regression_metrics
from spatial.autocorrelation import SpatialAuditConfig, prepare_spatial_sample


@dataclass(frozen=True)
class SpatialLagConfig:
    k_neighbors: int = 8
    spatial_blocks: int = 5
    permutations: int = 999
    random_seed: int = 42
    maximum_observations: int = 10_000
    minimum_category_count: int = 20


def _weights(coordinates: np.ndarray, k: int) -> object:
    weights = KNN.from_array(coordinates, k=min(k, len(coordinates) - 1))
    weights.transform = "r"
    return weights


def _independent_columns(design: pd.DataFrame) -> list[str]:
    """Deterministically remove constants and exact linear dependencies."""
    selected: list[str] = []
    # Account for the intercept that both OLS and spreg add later.
    matrix = np.ones((len(design), 1))
    rank = 1
    for column in design.columns:
        values = design[column].to_numpy(float)[:, None]
        candidate = np.column_stack([matrix, values])
        candidate_rank = np.linalg.matrix_rank(candidate)
        if candidate_rank > rank:
            selected.append(column)
            matrix = candidate
            rank = candidate_rank
    return selected


def _design_schema(frame: pd.DataFrame, minimum_category_count: int) -> tuple[HedonicModel, pd.DataFrame, list[str]]:
    model = HedonicModel(HedonicConfig(
        minimum_category_count=minimum_category_count,
        include_time=False,
        include_property_type=True,
        include_neighborhood=True,
        include_accessibility=True,
    )).fit(frame)
    design = model._design(frame).drop(columns="intercept")
    columns = _independent_columns(design)
    if not columns:
        raise ValueError("no independent hedonic predictors remain for the spatial lag model")
    return model, design[columns], columns


def _moran_residual(residual: np.ndarray, weights, permutations: int, seed: int) -> dict:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        result = Moran(np.asarray(residual).reshape(-1), weights, permutations=permutations)
    finally:
        np.random.set_state(state)
    return {
        "moran_i": float(result.I),
        "p_permutation": float(result.p_sim),
        "z_permutation": float(result.z_sim),
    }


def _fit_full(sample: pd.DataFrame, config: SpatialLagConfig) -> tuple[dict, pd.DataFrame]:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    weights = _weights(coordinates, config.k_neighbors)
    _, design, columns = _design_schema(sample, config.minimum_category_count)
    target = np.log(sample["sale_price"].to_numpy(float))
    lag = ML_Lag(
        target.reshape(-1, 1), design.to_numpy(float), weights,
        method="ord", spat_impacts="all", vm=True,
        name_y="log_sale_price", name_x=columns, name_w=f"knn_{min(config.k_neighbors, len(sample)-1)}",
    )
    retained_columns = lag.name_x[1:-1]
    ols = sm.OLS(
        target, sm.add_constant(design[retained_columns], has_constant="add")
    ).fit(cov_type="HC3")
    rho_z, rho_p = lag.z_stat[-1]
    comparison = {
        "ols": {
            "r_squared": float(ols.rsquared),
            "adjusted_r_squared": float(ols.rsquared_adj),
            "aic": float(ols.aic),
            "bic": float(ols.bic),
            "residual_moran": _moran_residual(ols.resid, weights, config.permutations, config.random_seed),
        },
        "spatial_lag": {
            "pseudo_r_squared": float(lag.pr2),
            "log_likelihood": float(lag.logll),
            "aic": float(lag.aic),
            "bic": float(lag.schwarz),
            "rho": float(lag.rho),
            "rho_standard_error": float(lag.std_err[-1]),
            "rho_z": float(rho_z),
            "rho_p_value": float(rho_p),
            "spatial_multipliers": {
                name: {"direct": float(values[0]), "indirect": float(values[1]), "total": float(values[2])}
                for name, values in lag.sp_multipliers.items()
            },
            "residual_moran": _moran_residual(lag.u, weights, config.permutations, config.random_seed),
        },
        "design_columns": retained_columns,
    }
    coefficient_names = [
        "constant" if name == "CONSTANT" else "rho" if name == "W_log_sale_price" or name == "W_dep_var" else name
        for name in lag.name_x
    ]
    coefficients = pd.DataFrame({
        "term": coefficient_names,
        "coefficient": lag.betas.reshape(-1),
        "standard_error": lag.std_err,
        "z_value": [item[0] for item in lag.z_stat],
        "p_value": [item[1] for item in lag.z_stat],
    })
    return comparison, coefficients


def _spatial_block_validation(sample: pd.DataFrame, config: SpatialLagConfig) -> tuple[dict, pd.DataFrame]:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    blocks = min(config.spatial_blocks, max(2, len(sample) // 5))
    labels = KMeans(n_clusters=blocks, random_state=config.random_seed, n_init=10).fit_predict(coordinates)
    # Hold out the easternmost spatial block, a deterministic geographic extrapolation.
    centers = pd.DataFrame(coordinates, columns=["x", "y"]).assign(block=labels).groupby("block")["x"].mean()
    holdout_block = int(centers.idxmax())
    test_mask = labels == holdout_block
    train_mask = ~test_mask
    if train_mask.sum() < 3 or test_mask.sum() < 1:
        raise ValueError("spatial block split produced an insufficient train or test sample")
    train, test = sample.loc[train_mask].copy(), sample.loc[test_mask].copy()
    schema, train_design, columns = _design_schema(train, config.minimum_category_count)
    test_design = schema._design(test).drop(columns="intercept").reindex(columns=columns, fill_value=0.0)
    train_coordinates = coordinates[train_mask]
    train_weights = _weights(train_coordinates, config.k_neighbors)
    train_target = np.log(train["sale_price"].to_numpy(float))
    lag = ML_Lag(
        train_target.reshape(-1, 1), train_design.to_numpy(float), train_weights,
        method="ord", spat_impacts="simple", vm=False,
        name_y="log_sale_price", name_x=columns,
    )
    retained_columns = lag.name_x[1:-1]
    train_design = train_design[retained_columns]
    test_design = test_design[retained_columns]
    ols = sm.OLS(train_target, sm.add_constant(train_design, has_constant="add")).fit()

    full_weights = _weights(coordinates, config.k_neighbors)
    matrix = full_weights.sparse.toarray()
    train_positions, test_positions = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    w_test_test = matrix[np.ix_(test_positions, test_positions)]
    w_test_train = matrix[np.ix_(test_positions, train_positions)]
    beta = lag.betas[:-1].reshape(-1)
    linear = sm.add_constant(test_design, has_constant="add").to_numpy(float) @ beta
    conditional = linear + float(lag.rho) * (w_test_train @ train_target)
    predicted_log_lag = np.linalg.solve(
        np.eye(len(test_positions)) - float(lag.rho) * w_test_test, conditional
    )
    predicted_log_ols = ols.predict(sm.add_constant(test_design, has_constant="add"))
    lag_smearing = float(np.exp(np.asarray(lag.u).reshape(-1)).mean())
    ols_smearing = float(np.exp(ols.resid).mean())
    predicted_lag = np.exp(np.clip(predicted_log_lag, None, 50)) * lag_smearing
    predicted_ols = np.exp(np.clip(predicted_log_ols, None, 50)) * ols_smearing
    validation = {
        "strategy": "easternmost_kmeans_spatial_block",
        "block_count": blocks,
        "held_out_block": holdout_block,
        "train_rows": len(train),
        "test_rows": len(test),
        "ols": regression_metrics(test["sale_price"], predicted_ols),
        "spatial_lag": regression_metrics(test["sale_price"], predicted_lag),
        "mae_improvement": regression_metrics(test["sale_price"], predicted_ols)["mae"]
        - regression_metrics(test["sale_price"], predicted_lag)["mae"],
        "prediction_mode": "conditional on observed training-neighbor prices; no held-out prices used",
    }
    predictions = test[[column for column in ("sale_id", "pin", "sale_date", "sale_price") if column in test]].copy()
    predictions["spatial_block"] = holdout_block
    predictions["prediction_ols"] = predicted_ols
    predictions["prediction_spatial_lag"] = predicted_lag
    return validation, predictions


def run_spatial_lag(
    input_path: Path,
    output_dir: Path,
    analysis_year: int | None = None,
    config: SpatialLagConfig | None = None,
) -> dict:
    config = config or SpatialLagConfig()
    audit_config = SpatialAuditConfig(
        k_neighbors=config.k_neighbors,
        permutations=config.permutations,
        random_seed=config.random_seed,
        maximum_observations=config.maximum_observations,
        minimum_category_count=config.minimum_category_count,
    )
    sample, year = prepare_spatial_sample(pd.read_parquet(input_path), audit_config, analysis_year)
    comparison, coefficients = _fit_full(sample, config)
    validation, predictions = _spatial_block_validation(sample, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_dir / "spatial_lag_coefficients.csv", index=False)
    predictions.to_parquet(output_dir / "spatial_lag_block_predictions.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "analysis_year": year,
        "sample_rows": len(sample),
        "config": asdict(config),
        "full_sample_comparison": comparison,
        "spatial_block_validation": validation,
        "rho_interpretation": (
            "Rho measures conditional spatial dependence in neighboring log sale prices. "
            "It is not automatically a causal spillover because omitted local factors can also induce dependence."
        ),
    }
    (output_dir / "spatial_lag_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/spatial_lag"))
    parser.add_argument("--analysis-year", type=int)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--spatial-blocks", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    config = SpatialLagConfig(
        k_neighbors=args.k_neighbors,
        spatial_blocks=args.spatial_blocks,
        permutations=args.permutations,
    )
    report = run_spatial_lag(args.input, args.output, args.analysis_year, config)
    rho = report["full_sample_comparison"]["spatial_lag"]["rho"]
    print(f"Estimated spatial lag rho: {rho:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
