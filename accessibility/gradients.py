"""Estimate nonlinear Lake Michigan and downtown price gradients."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from hedonic.model import HedonicConfig, HedonicModel
from ml.baselines import regression_metrics, temporal_split
from transit.premium import DistanceBasis


def _nuisance_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_result = pd.DataFrame(index=train.index)
    test_result = pd.DataFrame(index=test.index)
    medians = {}
    for column in columns:
        if column not in train:
            continue
        train_values = pd.to_numeric(train[column], errors="coerce")
        if train_values.notna().sum() < 2:
            continue
        median = float(train_values.median())
        medians[column] = median
        test_values = pd.to_numeric(test.get(column), errors="coerce")
        if "distance" in column:
            name = f"control_log1p_{column}"
            train_result[name] = np.log1p(train_values.clip(lower=0).fillna(median))
            test_result[name] = np.log1p(test_values.clip(lower=0).fillna(median))
        else:
            name = f"control_{column}"
            train_result[name] = train_values.fillna(median)
            test_result[name] = test_values.fillna(median)
    return train_result, test_result, medians


def _fit_gradient(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base: HedonicModel,
    distance_column: str,
    label: str,
    kind: str,
    nuisance_columns: list[str],
) -> tuple[dict, pd.Series, pd.DataFrame, pd.DataFrame]:
    basis = DistanceBasis(kind, prefix=label).fit(train[distance_column])
    nuisance_train, nuisance_test, medians = _nuisance_features(train, test, nuisance_columns)
    train_design = pd.concat(
        [base._design(train), nuisance_train, basis.transform(train[distance_column])], axis=1
    )
    test_design = pd.concat(
        [base._design(test), nuisance_test, basis.transform(test[distance_column])], axis=1
    )
    # Nullable pandas dtypes can promote a concatenated design to object even
    # when every value is numeric. Statsmodels requires a concrete numeric
    # matrix, so normalize the completed train/test designs together.
    train_design = train_design.apply(pd.to_numeric, errors="coerce").astype("float64")
    test_design = test_design.apply(pd.to_numeric, errors="coerce").astype("float64")
    if not np.isfinite(train_design.to_numpy()).all():
        raise ValueError(f"{label} {kind} training design contains non-finite values")
    if not np.isfinite(test_design.to_numpy()).all():
        raise ValueError(f"{label} {kind} test design contains non-finite values")
    target = np.log(pd.to_numeric(train["sale_price"], errors="coerce"))
    ordinary = sm.OLS(target, train_design, missing="raise").fit()
    result = ordinary.get_robustcov_results(cov_type="HC3")
    smearing = float(np.exp(target - ordinary.predict(train_design)).mean())
    prediction = np.exp(pd.Series(result.predict(test_design), index=test.index).clip(upper=50)) * smearing

    term_positions = [train_design.columns.get_loc(column) for column in basis.columns_]
    beta = np.asarray(result.params)[term_positions]
    covariance = np.asarray(result.cov_params())[np.ix_(term_positions, term_positions)]
    try:
        statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
        joint_p_value = float(chi2.sf(statistic, len(beta)))
    except (ValueError, np.linalg.LinAlgError):
        statistic, joint_p_value = None, None

    maximum = max(2.0, float(pd.to_numeric(train[distance_column]).quantile(0.95)))
    reference = float(pd.to_numeric(train[distance_column]).quantile(0.90))
    grid = pd.Series(np.linspace(0, maximum, 121))
    grid_basis = basis.transform(grid)
    reference_basis = basis.transform(pd.Series([reference])).iloc[0]
    contrast = grid_basis - reference_basis.to_numpy()
    log_effect = contrast.to_numpy() @ beta
    variance = np.einsum("ij,jk,ik->i", contrast.to_numpy(), covariance, contrast.to_numpy())
    standard_error = np.sqrt(np.maximum(variance, 0))
    curve = pd.DataFrame({
        "gradient": label,
        "specification": kind,
        "distance_miles": grid,
        "reference_distance_miles": reference,
        "price_difference_vs_reference": np.exp(log_effect) - 1,
        "ci_lower": np.exp(log_effect - 1.96 * standard_error) - 1,
        "ci_upper": np.exp(log_effect + 1.96 * standard_error) - 1,
    })
    intervals = result.conf_int()
    coefficients = pd.DataFrame({
        "gradient": label,
        "specification": kind,
        "term": train_design.columns,
        "coefficient": result.params,
        "robust_std_error": result.bse,
        "p_value": result.pvalues,
        "ci_lower": intervals[:, 0],
        "ci_upper": intervals[:, 1],
    })
    metrics = {
        "in_sample_r_squared": float(result.rsquared),
        "in_sample_adjusted_r_squared": float(result.rsquared_adj),
        "out_of_sample": regression_metrics(test["sale_price"], prediction),
        "distance_terms": basis.columns_,
        "distance_joint_wald_statistic": statistic,
        "distance_joint_p_value": joint_p_value,
        "reference_distance_miles": reference,
        "nuisance_imputations": medians,
    }
    return metrics, prediction, coefficients, curve


def _half_decay_distance(curve: pd.DataFrame) -> float | None:
    ordered = curve.sort_values("distance_miles")
    near = ordered.iloc[(ordered["distance_miles"] - 0.1).abs().argmin()][
        "price_difference_vs_reference"
    ]
    if not np.isfinite(near) or near <= 0:
        return None
    threshold = near / 2
    candidates = ordered.loc[
        ordered["distance_miles"].ge(0.1)
        & ordered["price_difference_vs_reference"].le(threshold)
    ]
    return float(candidates.iloc[0]["distance_miles"]) if len(candidates) else None


def _plot(curves: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, (gradient, group) in zip(axes, curves.groupby("gradient", sort=False)):
        for name, specification in group.groupby("specification"):
            axis.plot(
                specification["distance_miles"],
                100 * specification["price_difference_vs_reference"], label=name,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            title=f"{gradient.title()} price gradient",
            xlabel=f"Distance to {gradient} (miles)",
            ylabel="Estimated price difference vs reference (%)",
        )
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def analyze_gradients(
    input_path: Path,
    output_dir: Path,
    test_start_year: int | None = None,
    minimum_category_count: int = 20,
) -> dict:
    frame = pd.read_parquet(input_path)
    required = {"lake_distance_miles", "downtown_distance_miles", "cta_distance_miles"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(
            f"gradient analysis is missing {', '.join(missing)}; build accessibility features first"
        )
    valid = pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)
    for column in required:
        valid &= pd.to_numeric(frame[column], errors="coerce").ge(0)
    frame = frame.loc[valid].copy()
    train, test, cutoff = temporal_split(frame, test_start_year)
    base = HedonicModel(HedonicConfig(
        minimum_category_count=minimum_category_count,
        include_time=True, include_property_type=True, include_neighborhood=True,
        include_accessibility=False,
    )).fit(train)
    analyses = {
        "lake": {
            "distance": "lake_distance_miles",
            "controls": ["cta_distance_miles", "cta_stations_half_mile", "downtown_distance_miles"],
        },
        "downtown": {
            "distance": "downtown_distance_miles",
            "controls": ["cta_distance_miles", "cta_stations_half_mile", "lake_distance_miles"],
        },
    }
    identity = [column for column in ("sale_id", "pin", "sale_date", "year", "sale_price") if column in test]
    predictions = test[identity].copy()
    results, coefficient_tables, curve_tables = {}, [], []
    for label, definition in analyses.items():
        results[label] = {}
        for kind in ("linear", "cubic_spline", "gam_style"):
            metrics, prediction, coefficients, curve = _fit_gradient(
                train, test, base, definition["distance"], label, kind, definition["controls"]
            )
            results[label][kind] = metrics
            predictions[f"prediction_{label}_{kind}"] = prediction
            coefficient_tables.append(coefficients)
            curve_tables.append(curve)
    curves = pd.concat(curve_tables, ignore_index=True)
    best = {
        label: min(specifications, key=lambda name: specifications[name]["out_of_sample"]["mae"])
        for label, specifications in results.items()
    }
    lake_curve = curves.loc[
        curves["gradient"].eq("lake") & curves["specification"].eq(best["lake"])
    ]
    half_decay = _half_decay_distance(lake_curve)
    downtown_metrics = results["downtown"][best["downtown"]]
    evidence = {
        "lakefront_half_decay_distance_miles": half_decay,
        "downtown_distance_joint_p_value_after_transit_and_neighborhood_controls": downtown_metrics[
            "distance_joint_p_value"
        ],
        "downtown_distance_statistically_detectable_at_5_percent": (
            downtown_metrics["distance_joint_p_value"] is not None
            and downtown_metrics["distance_joint_p_value"] < 0.05
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "gradient_predictions.parquet", index=False)
    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        output_dir / "gradient_coefficients.csv", index=False
    )
    curves.to_csv(output_dir / "gradient_curves.csv", index=False)
    _plot(curves, output_dir / "lake_downtown_gradients.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "test_start_year": cutoff,
        "train_rows": len(train),
        "test_rows": len(test),
        "models": results,
        "best_out_of_sample_specifications": best,
        "evidence": evidence,
        "interpretation": "Gradients are conditional hedonic associations, not causal amenity effects.",
    }
    (output_dir / "gradient_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/amenity_gradients"))
    parser.add_argument("--test-start-year", type=int)
    parser.add_argument("--minimum-category-count", type=int, default=20)
    args = parser.parse_args()
    report = analyze_gradients(
        args.input, args.output, args.test_start_year, args.minimum_category_count
    )
    print(json.dumps(report["best_out_of_sample_specifications"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
