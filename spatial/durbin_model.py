"""Conditionally estimate a Spatial Durbin robustness specification."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.cluster import KMeans
from spreg import ML_Lag

from ml.baselines import regression_metrics
from spatial.autocorrelation import SpatialAuditConfig, prepare_spatial_sample
from spatial.lag_model import _design_schema, _moran_residual, _weights


@dataclass(frozen=True)
class DurbinConfig:
    k_neighbors: int = 8
    spatial_blocks: int = 5
    permutations: int = 999
    random_seed: int = 42
    maximum_observations: int = 7_500
    minimum_category_count: int = 20
    significance_level: float = 0.05


def spatial_justification(diagnostics: dict, significance_level: float = 0.05) -> dict:
    comparison = diagnostics.get("full_sample_comparison", {})
    sar = comparison.get("spatial_lag", {})
    sem = comparison.get("spatial_error", {})
    ols_moran = comparison.get("ols", {}).get("residual_moran", {})
    triggers = {
        "significant_sar_rho": sar.get("rho_p_value") is not None and sar["rho_p_value"] < significance_level,
        "significant_sem_lambda": sem.get("lambda_p_value") is not None and sem["lambda_p_value"] < significance_level,
        "significant_ols_residual_moran": (
            ols_moran.get("p_permutation") is not None
            and ols_moran["p_permutation"] < significance_level
            and ols_moran.get("moran_i", 0) > 0
        ),
    }
    return {"justified": any(triggers.values()), "triggers": triggers}


def _fit_full(sample: pd.DataFrame, config: DurbinConfig) -> tuple[dict, pd.DataFrame]:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    weights = _weights(coordinates, config.k_neighbors)
    _, design, columns = _design_schema(sample, config.minimum_category_count)
    target = np.log(sample["sale_price"].to_numpy(float))
    sar = ML_Lag(
        target.reshape(-1, 1), design.to_numpy(float), weights,
        method="ord", spat_impacts="all", vm=True,
        name_y="log_sale_price", name_x=columns,
    )
    retained = sar.name_x[1:-1]
    design = design[retained]
    # Refit SAR on exactly the design that SDM expands with WX.
    sar = ML_Lag(
        target.reshape(-1, 1), design.to_numpy(float), weights,
        method="ord", spat_impacts="all", vm=True,
        name_y="log_sale_price", name_x=retained,
    )
    sdm = ML_Lag(
        target.reshape(-1, 1), design.to_numpy(float), weights,
        slx_lags=1, method="ord", spat_impacts="all", vm=True,
        name_y="log_sale_price", name_x=retained,
    )
    ols = sm.OLS(target, sm.add_constant(design, has_constant="add")).fit(cov_type="HC3")
    rho_z, rho_p = sdm.z_stat[-1]
    wx_terms = [name for name in sdm.name_x if name.startswith("W_") and name != "W_log_sale_price"]
    likelihood_ratio = max(0.0, 2 * (float(sdm.logll) - float(sar.logll)))
    lr_p_value = float(chi2.sf(likelihood_ratio, max(1, len(wx_terms))))

    sar_coefficients = dict(zip(sar.name_x, sar.betas.reshape(-1)))
    sdm_coefficients = dict(zip(sdm.name_x, sdm.betas.reshape(-1)))
    stability = {}
    for term in retained:
        before, after = float(sar_coefficients[term]), float(sdm_coefficients[term])
        stability[term] = {
            "sar": before,
            "sdm": after,
            "absolute_change": after - before,
            "relative_absolute_change": abs(after - before) / abs(before) if before != 0 else None,
            "sign_changed": bool(np.sign(before) != np.sign(after)),
        }
    comparison = {
        "ols": {
            "r_squared": float(ols.rsquared), "aic": float(ols.aic), "bic": float(ols.bic),
            "residual_moran": _moran_residual(ols.resid, weights, config.permutations, config.random_seed),
        },
        "spatial_lag": {
            "pseudo_r_squared": float(sar.pr2), "log_likelihood": float(sar.logll),
            "aic": float(sar.aic), "bic": float(sar.schwarz), "rho": float(sar.rho),
            "residual_moran": _moran_residual(sar.u, weights, config.permutations, config.random_seed),
        },
        "spatial_durbin": {
            "pseudo_r_squared": float(sdm.pr2), "log_likelihood": float(sdm.logll),
            "aic": float(sdm.aic), "bic": float(sdm.schwarz),
            "rho": float(sdm.rho), "rho_standard_error": float(sdm.std_err[-1]),
            "rho_z": float(rho_z), "rho_p_value": float(rho_p),
            "wx_terms": wx_terms,
            "sar_nested_likelihood_ratio": likelihood_ratio,
            "sar_nested_likelihood_ratio_df": len(wx_terms),
            "sar_nested_likelihood_ratio_p_value": lr_p_value,
            "spatial_multipliers": {
                name: {"direct": float(value[0]), "indirect": float(value[1]), "total": float(value[2])}
                for name, value in sdm.sp_multipliers.items()
            },
            "residual_moran": _moran_residual(sdm.u, weights, config.permutations, config.random_seed),
        },
        "coefficient_stability_sar_to_sdm": stability,
        "design_columns": retained,
    }
    coefficient_rows = []
    for model_name, model in (("sar", sar), ("sdm", sdm)):
        for name, coefficient, standard_error, z_stat in zip(
            model.name_x, model.betas.reshape(-1), model.std_err, model.z_stat
        ):
            coefficient_rows.append({
                "model": model_name, "term": name, "coefficient": coefficient,
                "standard_error": standard_error, "z_value": z_stat[0], "p_value": z_stat[1],
            })
    return comparison, pd.DataFrame(coefficient_rows)


def _model_test_matrix(model, test_design: pd.DataFrame, full_design: pd.DataFrame,
                       full_weights: np.ndarray, test_positions: np.ndarray) -> np.ndarray:
    columns = []
    for name in model.name_x[:-1]:
        if name == "CONSTANT":
            columns.append(np.ones(len(test_design)))
        elif name.startswith("W_"):
            source = name[2:]
            if source not in full_design:
                raise ValueError(f"cannot construct SDM lagged predictor {name}")
            columns.append(full_weights[test_positions] @ full_design[source].to_numpy(float))
        else:
            columns.append(test_design[name].to_numpy(float))
    return np.column_stack(columns)


def _block_validation(sample: pd.DataFrame, config: DurbinConfig) -> tuple[dict, pd.DataFrame]:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    blocks = min(config.spatial_blocks, max(2, len(sample) // 5))
    labels = KMeans(n_clusters=blocks, random_state=config.random_seed, n_init=10).fit_predict(coordinates)
    centers = pd.DataFrame(coordinates, columns=["x", "y"]).assign(block=labels).groupby("block")["x"].mean()
    held_out = int(centers.idxmax())
    train_mask, test_mask = labels != held_out, labels == held_out
    train, test = sample.loc[train_mask].copy(), sample.loc[test_mask].copy()
    schema, train_design, columns = _design_schema(train, config.minimum_category_count)
    full_design = schema._design(sample).drop(columns="intercept").reindex(columns=columns, fill_value=0.0)
    test_design = full_design.loc[test_mask]
    train_weights = _weights(coordinates[train_mask], config.k_neighbors)
    target = np.log(train["sale_price"].to_numpy(float))
    sar = ML_Lag(
        target.reshape(-1, 1), train_design.to_numpy(float), train_weights,
        method="ord", name_y="log_sale_price", name_x=columns,
    )
    retained = sar.name_x[1:-1]
    train_design = train_design[retained]
    full_design, test_design = full_design[retained], test_design[retained]
    sar = ML_Lag(
        target.reshape(-1, 1), train_design.to_numpy(float), train_weights,
        method="ord", name_y="log_sale_price", name_x=retained,
    )
    sdm = ML_Lag(
        target.reshape(-1, 1), train_design.to_numpy(float), train_weights,
        slx_lags=1, method="ord", name_y="log_sale_price", name_x=retained,
    )
    ols = sm.OLS(target, sm.add_constant(train_design, has_constant="add")).fit()
    full_weights_object = _weights(coordinates, config.k_neighbors)
    matrix = full_weights_object.sparse.toarray()
    train_positions, test_positions = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    w_tt = matrix[np.ix_(test_positions, test_positions)]
    w_tn = matrix[np.ix_(test_positions, train_positions)]

    def spatial_prediction(model) -> np.ndarray:
        predictors = _model_test_matrix(model, test_design, full_design, matrix, test_positions)
        linear = predictors @ model.betas[:-1].reshape(-1)
        conditional = linear + float(model.rho) * (w_tn @ target)
        log_prediction = np.linalg.solve(
            np.eye(len(test)) - float(model.rho) * w_tt, conditional
        )
        return np.exp(np.clip(log_prediction, None, 50)) * float(np.exp(model.u).mean())

    predictions = {
        "ols": np.exp(np.clip(ols.predict(sm.add_constant(test_design, has_constant="add")), None, 50))
        * float(np.exp(ols.resid).mean()),
        "spatial_lag": spatial_prediction(sar),
        "spatial_durbin": spatial_prediction(sdm),
    }
    validation = {
        "strategy": "easternmost_kmeans_spatial_block",
        "block_count": blocks, "held_out_block": held_out,
        "train_rows": len(train), "test_rows": len(test),
        "metrics": {
            name: regression_metrics(test["sale_price"], values)
            for name, values in predictions.items()
        },
        "prediction_mode": "conditional on observed training-neighbor prices and feature lags; no held-out prices used",
    }
    output = test[[column for column in ("sale_id", "pin", "sale_date", "sale_price") if column in test]].copy()
    output["spatial_block"] = held_out
    for name, values in predictions.items():
        output[f"prediction_{name}"] = values
    return validation, output


def run_spatial_durbin(
    input_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
    analysis_year: int | None = None,
    config: DurbinConfig | None = None,
    force: bool = False,
) -> dict:
    config = config or DurbinConfig()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    justification = spatial_justification(diagnostics, config.significance_level)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not justification["justified"] and not force:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped_not_justified",
            "justification": justification,
            "message": "Earlier spatial diagnostics do not justify SDM at the configured significance level.",
        }
        (output_dir / "spatial_durbin_results.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    audit_config = SpatialAuditConfig(
        k_neighbors=config.k_neighbors, permutations=config.permutations,
        random_seed=config.random_seed, maximum_observations=config.maximum_observations,
        minimum_category_count=config.minimum_category_count,
    )
    sample, year = prepare_spatial_sample(pd.read_parquet(input_path), audit_config, analysis_year)
    comparison, coefficients = _fit_full(sample, config)
    validation, predictions = _block_validation(sample, config)
    coefficients.to_csv(output_dir / "spatial_durbin_coefficients.csv", index=False)
    predictions.to_parquet(output_dir / "spatial_durbin_block_predictions.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "fitted_forced" if force and not justification["justified"] else "fitted_justified",
        "input": str(input_path), "diagnostics": str(diagnostics_path),
        "analysis_year": year, "sample_rows": len(sample), "config": asdict(config),
        "justification": justification,
        "full_sample_comparison": comparison,
        "spatial_block_validation": validation,
        "interpretation_caution": (
            "SDM neighboring-feature associations and impact multipliers are not causal spillovers without stronger identification."
        ),
    }
    (output_dir / "spatial_durbin_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument(
        "--diagnostics", type=Path,
        default=Path("data/processed/spatial_error/spatial_error_results.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/spatial_durbin"))
    parser.add_argument("--analysis-year", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--spatial-blocks", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    config = DurbinConfig(
        k_neighbors=args.k_neighbors, spatial_blocks=args.spatial_blocks,
        permutations=args.permutations,
    )
    report = run_spatial_durbin(
        args.input, args.diagnostics, args.output, args.analysis_year, config, args.force
    )
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

