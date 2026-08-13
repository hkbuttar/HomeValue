"""Estimate spatial error dependence and compare OLS, SAR, and SEM."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from spreg import ML_Error, ML_Lag

from ml.baselines import regression_metrics
from spatial.autocorrelation import SpatialAuditConfig, prepare_spatial_sample
from spatial.lag_model import _design_schema, _moran_residual, _weights


@dataclass(frozen=True)
class SpatialErrorConfig:
    k_neighbors: int = 8
    spatial_blocks: int = 5
    permutations: int = 999
    random_seed: int = 42
    maximum_observations: int = 10_000
    minimum_category_count: int = 20


def _fit_models(sample: pd.DataFrame, config: SpatialErrorConfig) -> tuple[dict, pd.DataFrame]:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    weights = _weights(coordinates, config.k_neighbors)
    _, design, columns = _design_schema(sample, config.minimum_category_count)
    target = np.log(sample["sale_price"].to_numpy(float))
    lag = ML_Lag(
        target.reshape(-1, 1), design.to_numpy(float), weights,
        method="ord", spat_impacts="all", vm=True,
        name_y="log_sale_price", name_x=columns,
    )
    retained = lag.name_x[1:-1]
    design = design[retained]
    ols = sm.OLS(target, sm.add_constant(design, has_constant="add")).fit(cov_type="HC3")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Method 'bounded' does not support relative tolerance")
        error = ML_Error(
            target.reshape(-1, 1), design.to_numpy(float), weights,
            method="ord", vm=True, name_y="log_sale_price", name_x=retained,
        )
    error_retained = error.name_x[1:-1]
    if error_retained != retained:
        raise ValueError("SAR and SEM retained different design columns")
    lambda_z, lambda_p = error.z_stat[-1]
    rho_z, rho_p = lag.z_stat[-1]
    comparison = {
        "ols": {
            "r_squared": float(ols.rsquared), "adjusted_r_squared": float(ols.rsquared_adj),
            "aic": float(ols.aic), "bic": float(ols.bic),
            "residual_moran": _moran_residual(ols.resid, weights, config.permutations, config.random_seed),
        },
        "spatial_lag": {
            "pseudo_r_squared": float(lag.pr2), "aic": float(lag.aic), "bic": float(lag.schwarz),
            "rho": float(lag.rho), "rho_standard_error": float(lag.std_err[-1]),
            "rho_z": float(rho_z), "rho_p_value": float(rho_p),
            "residual_moran": _moran_residual(lag.u, weights, config.permutations, config.random_seed),
        },
        "spatial_error": {
            "pseudo_r_squared": float(error.pr2), "aic": float(error.aic), "bic": float(error.schwarz),
            "lambda": float(error.lam), "lambda_standard_error": float(error.std_err[-1]),
            "lambda_z": float(lambda_z), "lambda_p_value": float(lambda_p),
            "residual_moran": _moran_residual(error.e_filtered, weights, config.permutations, config.random_seed),
        },
        "design_columns": retained,
    }
    coefficients = []
    for model_name, model, final_name in (
        ("spatial_lag", lag, "rho"), ("spatial_error", error, "lambda")
    ):
        names = ["constant", *retained, final_name]
        coefficients.append(pd.DataFrame({
            "model": model_name,
            "term": names,
            "coefficient": model.betas.reshape(-1),
            "standard_error": model.std_err,
            "z_value": [item[0] for item in model.z_stat],
            "p_value": [item[1] for item in model.z_stat],
        }))
    return comparison, pd.concat(coefficients, ignore_index=True)


def _block_validation(sample: pd.DataFrame, config: SpatialErrorConfig) -> tuple[dict, pd.DataFrame]:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    blocks = min(config.spatial_blocks, max(2, len(sample) // 5))
    labels = KMeans(n_clusters=blocks, random_state=config.random_seed, n_init=10).fit_predict(coordinates)
    centers = pd.DataFrame(coordinates, columns=["x", "y"]).assign(block=labels).groupby("block")["x"].mean()
    held_out = int(centers.idxmax())
    test_mask, train_mask = labels == held_out, labels != held_out
    train, test = sample.loc[train_mask].copy(), sample.loc[test_mask].copy()
    schema, train_design, columns = _design_schema(train, config.minimum_category_count)
    test_design = schema._design(test).drop(columns="intercept").reindex(columns=columns, fill_value=0.0)
    train_weights = _weights(coordinates[train_mask], config.k_neighbors)
    target = np.log(train["sale_price"].to_numpy(float))
    lag = ML_Lag(
        target.reshape(-1, 1), train_design.to_numpy(float), train_weights,
        method="ord", spat_impacts="simple", name_y="log_sale_price", name_x=columns,
    )
    retained = lag.name_x[1:-1]
    train_design, test_design = train_design[retained], test_design[retained]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Method 'bounded' does not support relative tolerance")
        error = ML_Error(
            target.reshape(-1, 1), train_design.to_numpy(float), train_weights,
            method="ord", name_y="log_sale_price", name_x=retained,
        )
    ols = sm.OLS(target, sm.add_constant(train_design, has_constant="add")).fit()

    full_weights = _weights(coordinates, config.k_neighbors)
    matrix = full_weights.sparse.toarray()
    train_positions, test_positions = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    w_tt = matrix[np.ix_(test_positions, test_positions)]
    w_tn = matrix[np.ix_(test_positions, train_positions)]
    x_test = sm.add_constant(test_design, has_constant="add").to_numpy(float)

    lag_linear = x_test @ lag.betas[:-1].reshape(-1)
    lag_conditional = lag_linear + float(lag.rho) * (w_tn @ target)
    lag_log = np.linalg.solve(np.eye(len(test)) - float(lag.rho) * w_tt, lag_conditional)
    error_linear = x_test @ error.betas[:-1].reshape(-1)
    conditional_error = np.linalg.solve(
        np.eye(len(test)) - float(error.lam) * w_tt,
        float(error.lam) * (w_tn @ np.asarray(error.u).reshape(-1)),
    )
    error_log = error_linear + conditional_error
    ols_log = ols.predict(sm.add_constant(test_design, has_constant="add"))
    predictions = {
        "ols": np.exp(np.clip(ols_log, None, 50)) * float(np.exp(ols.resid).mean()),
        "spatial_lag": np.exp(np.clip(lag_log, None, 50)) * float(np.exp(lag.u).mean()),
        "spatial_error": np.exp(np.clip(error_log, None, 50)) * float(np.exp(error.e_filtered).mean()),
    }
    metrics = {
        name: regression_metrics(test["sale_price"], values) for name, values in predictions.items()
    }
    validation = {
        "strategy": "easternmost_kmeans_spatial_block",
        "block_count": blocks, "held_out_block": held_out,
        "train_rows": len(train), "test_rows": len(test),
        "metrics": metrics,
        "sem_prediction_mode": "conditional on observed training-neighbor residuals; no held-out prices used",
        "sar_prediction_mode": "conditional on observed training-neighbor prices; no held-out prices used",
    }
    output = test[[column for column in ("sale_id", "pin", "sale_date", "sale_price") if column in test]].copy()
    output["spatial_block"] = held_out
    for name, values in predictions.items():
        output[f"prediction_{name}"] = values
    return validation, output


def run_spatial_error(
    input_path: Path,
    output_dir: Path,
    analysis_year: int | None = None,
    config: SpatialErrorConfig | None = None,
) -> dict:
    config = config or SpatialErrorConfig()
    audit_config = SpatialAuditConfig(
        k_neighbors=config.k_neighbors, permutations=config.permutations,
        random_seed=config.random_seed, maximum_observations=config.maximum_observations,
        minimum_category_count=config.minimum_category_count,
    )
    sample, year = prepare_spatial_sample(pd.read_parquet(input_path), audit_config, analysis_year)
    comparison, coefficients = _fit_models(sample, config)
    validation, predictions = _block_validation(sample, config)
    criteria = {name: comparison[name]["aic"] for name in ("ols", "spatial_lag", "spatial_error")}
    preferred = min(criteria, key=criteria.get)
    mechanism = {
        "spatial_lag": "The SAR fit is most consistent with direct conditional price dependence among neighbors.",
        "spatial_error": "The SEM fit is most consistent with spatially correlated omitted local conditions.",
        "ols": "Neither spatial specification improved AIC over OLS in this sample.",
    }[preferred]
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_dir / "spatial_model_coefficients.csv", index=False)
    predictions.to_parquet(output_dir / "spatial_error_block_predictions.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path), "analysis_year": year, "sample_rows": len(sample),
        "config": asdict(config), "full_sample_comparison": comparison,
        "spatial_block_validation": validation,
        "lowest_aic_model": preferred,
        "mechanism_assessment": mechanism,
        "interpretation_caution": (
            "AIC and dependence parameters distinguish statistical specifications, not proven causal mechanisms."
        ),
    }
    (output_dir / "spatial_error_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/spatial_error"))
    parser.add_argument("--analysis-year", type=int)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--spatial-blocks", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    config = SpatialErrorConfig(
        k_neighbors=args.k_neighbors, spatial_blocks=args.spatial_blocks,
        permutations=args.permutations,
    )
    report = run_spatial_error(args.input, args.output, args.analysis_year, config)
    spatial_error = report["full_sample_comparison"]["spatial_error"]
    print(f"Estimated spatial error lambda: {spatial_error['lambda']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

