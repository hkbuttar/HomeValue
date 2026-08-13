"""Estimate linear and nonlinear CTA accessibility price associations."""

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
from sklearn.preprocessing import SplineTransformer

from hedonic.model import HedonicConfig, HedonicModel
from ml.baselines import regression_metrics, temporal_split

BAND_LABELS = ("0-0.25", "0.25-0.50", "0.50-1.00", "1.00-2.00", "over-2.00")


class DistanceBasis:
    """Train-fitted distance representation used by one premium specification."""

    def __init__(self, kind: str, prefix: str = "cta"):
        self.kind = kind
        self.prefix = prefix
        self.mean_: float | None = None
        self.scale_: float | None = None
        self.knots_: np.ndarray | None = None
        self.spline_: SplineTransformer | None = None
        self.columns_: list[str] = []

    def fit(self, distance_miles: pd.Series) -> "DistanceBasis":
        values = pd.to_numeric(distance_miles, errors="coerce").to_numpy(float).reshape(-1, 1)
        if not np.isfinite(values).all():
            raise ValueError("distance basis requires complete finite training distances")
        if self.kind == "linear":
            self.columns_ = [f"{self.prefix}_distance_linear"]
        elif self.kind == "bands":
            # The >2-mile band is the omitted reference.
            self.columns_ = [f"{self.prefix}_band={label}" for label in BAND_LABELS[:-1]]
        elif self.kind == "cubic_spline":
            self.mean_ = float(values.mean())
            self.scale_ = float(values.std()) or 1.0
            standardized = (values[:, 0] - self.mean_) / self.scale_
            self.knots_ = np.unique(np.quantile(standardized, [0.25, 0.5, 0.75]))
            self.columns_ = [f"{self.prefix}_spline=x", f"{self.prefix}_spline=x2", f"{self.prefix}_spline=x3"] + [
                f"{self.prefix}_spline=hinge_{index}" for index in range(len(self.knots_))
            ]
        elif self.kind == "gam_style":
            unique = np.unique(values)
            knots = min(7, max(3, len(unique)))
            self.spline_ = SplineTransformer(
                n_knots=knots, degree=3, knots="quantile", include_bias=False,
                extrapolation="linear",
            ).fit(values)
            self.columns_ = [f"{self.prefix}_gam_basis_{index}" for index in range(self.spline_.n_features_out_)]
        else:
            raise ValueError(f"unknown distance basis: {self.kind}")
        return self

    def transform(self, distance_miles: pd.Series) -> pd.DataFrame:
        values = pd.to_numeric(distance_miles, errors="coerce").to_numpy(float)
        if self.kind == "linear":
            matrix = values[:, None]
        elif self.kind == "bands":
            band = pd.Series(pd.cut(
                values, [-np.inf, 0.25, 0.5, 1.0, 2.0, np.inf],
                labels=BAND_LABELS, include_lowest=True,
            ), index=distance_miles.index, dtype="string")
            matrix = np.column_stack([band.eq(label).astype(float) for label in BAND_LABELS[:-1]])
        elif self.kind == "cubic_spline":
            standardized = (values - float(self.mean_)) / float(self.scale_)
            pieces = [standardized, standardized**2, standardized**3]
            pieces.extend(np.maximum(standardized - knot, 0) ** 3 for knot in self.knots_)
            matrix = np.column_stack(pieces)
        else:
            matrix = self.spline_.transform(values.reshape(-1, 1))
        return pd.DataFrame(matrix, index=distance_miles.index, columns=self.columns_, dtype=float)


def _distance_column(frame: pd.DataFrame) -> str:
    for column in ("cta_distance_miles", "cta_distance"):
        if column in frame:
            return column
    raise ValueError("input has no CTA distance feature; run Step 11 first")


def _fit_specification(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_model: HedonicModel,
    distance_column: str,
    kind: str,
) -> tuple[dict, pd.Series, pd.DataFrame, DistanceBasis, object]:
    basis = DistanceBasis(kind).fit(train[distance_column])
    train_design = pd.concat(
        [base_model._design(train), basis.transform(train[distance_column])], axis=1
    )
    test_design = pd.concat(
        [base_model._design(test), basis.transform(test[distance_column])], axis=1
    )
    target = np.log(pd.to_numeric(train["sale_price"], errors="coerce"))
    ordinary = sm.OLS(target, train_design, missing="raise").fit()
    result = ordinary.get_robustcov_results(cov_type="HC3")
    smearing = float(np.exp(target - ordinary.predict(train_design)).mean())
    log_prediction = pd.Series(result.predict(test_design), index=test.index)
    prediction = np.exp(log_prediction.clip(upper=50)) * smearing
    intervals = result.conf_int()
    coefficients = pd.DataFrame({
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
        "smearing_factor": smearing,
    }
    return metrics, prediction, coefficients, basis, result


def _premium_curve(kind: str, basis: DistanceBasis, result, all_columns: list[str]) -> pd.DataFrame:
    grid = pd.Series(np.linspace(0, 5, 101))
    reference = pd.Series([3.0])
    grid_basis = basis.transform(grid)
    reference_basis = basis.transform(reference).iloc[0]
    contrast = grid_basis - reference_basis.to_numpy()
    term_positions = [all_columns.index(column) for column in basis.columns_]
    coefficients = np.asarray(result.params)[term_positions]
    covariance = np.asarray(result.cov_params())[np.ix_(term_positions, term_positions)]
    log_effect = contrast.to_numpy() @ coefficients
    variance = np.einsum("ij,jk,ik->i", contrast.to_numpy(), covariance, contrast.to_numpy())
    standard_error = np.sqrt(np.maximum(variance, 0))
    return pd.DataFrame({
        "specification": kind,
        "distance_miles": grid,
        "premium_vs_3_miles": np.exp(log_effect) - 1,
        "ci_lower": np.exp(log_effect - 1.96 * standard_error) - 1,
        "ci_upper": np.exp(log_effect + 1.96 * standard_error) - 1,
    })


def _evidence(curve: pd.DataFrame) -> dict:
    indexed = curve.set_index("distance_miles")

    def nearest(value: float) -> pd.Series:
        return curve.iloc[(curve["distance_miles"] - value).abs().argmin()]

    near, walk, farther = nearest(0.1), nearest(0.4), nearest(2.0)
    return {
        "transit_premium": bool(curve.loc[curve["distance_miles"].le(1), "ci_lower"].max() > 0),
        "immediate_disamenity_pattern": bool(near["premium_vs_3_miles"] < walk["premium_vs_3_miles"]),
        "diminishing_benefit_pattern": bool(walk["premium_vs_3_miles"] > farther["premium_vs_3_miles"]),
        "near_station_premium_vs_3_miles": float(near["premium_vs_3_miles"]),
        "walkable_premium_vs_3_miles": float(walk["premium_vs_3_miles"]),
        "two_mile_premium_vs_3_miles": float(farther["premium_vs_3_miles"]),
        "caution": "Pattern flags are descriptive; transit placement and housing prices are jointly related to neighborhood conditions.",
    }


def _plot_curves(curves: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 6))
    for name, group in curves.groupby("specification"):
        axis.plot(group["distance_miles"], 100 * group["premium_vs_3_miles"], label=name)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Distance to nearest CTA rail station (miles)",
        ylabel="Estimated price difference versus 3 miles (%)",
        title="Conditional CTA accessibility premium curves",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def analyze_cta_premium(
    input_path: Path,
    output_dir: Path,
    test_start_year: int | None = None,
    minimum_category_count: int = 20,
) -> dict:
    frame = pd.read_parquet(input_path)
    distance_column = _distance_column(frame)
    distance_values = pd.to_numeric(frame[distance_column], errors="coerce")
    price = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame = frame.loc[distance_values.ge(0) & price.gt(0)].copy()
    train, test, cutoff = temporal_split(frame, test_start_year)
    base_model = HedonicModel(HedonicConfig(
        minimum_category_count=minimum_category_count,
        include_time=True, include_property_type=True, include_neighborhood=True,
        include_accessibility=False,
    )).fit(train)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = [column for column in ("sale_id", "pin", "sale_date", "year", "sale_price", distance_column) if column in test]
    predictions = test[identity].copy()
    metrics, coefficient_tables, curves = {}, [], []
    for kind in ("linear", "bands", "cubic_spline", "gam_style"):
        result_metrics, prediction, coefficients, basis, result = _fit_specification(
            train, test, base_model, distance_column, kind
        )
        metrics[kind] = result_metrics
        predictions[f"prediction_{kind}"] = prediction
        coefficient_tables.append(coefficients)
        all_columns = [*base_model.design_columns_, *basis.columns_]
        curves.append(_premium_curve(kind, basis, result, all_columns))
    curves_frame = pd.concat(curves, ignore_index=True)
    best = min(metrics, key=lambda name: metrics[name]["out_of_sample"]["mae"])
    evidence = _evidence(curves_frame.loc[curves_frame["specification"].eq(best)])
    predictions.to_parquet(output_dir / "cta_premium_predictions.parquet", index=False)
    pd.concat(coefficient_tables, ignore_index=True).to_csv(
        output_dir / "cta_premium_coefficients.csv", index=False
    )
    curves_frame.to_csv(output_dir / "cta_premium_curves.csv", index=False)
    _plot_curves(curves_frame, output_dir / "cta_premium_curves.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "distance_column": distance_column,
        "test_start_year": cutoff,
        "train_rows": len(train),
        "test_rows": len(test),
        "reference_distance_miles": 3.0,
        "specifications": metrics,
        "best_out_of_sample_specification": best,
        "evidence": evidence,
        "interpretation": "Estimated premiums are conditional hedonic associations, not causal transit effects.",
    }
    (output_dir / "cta_premium_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/cta_premium"))
    parser.add_argument("--test-start-year", type=int)
    parser.add_argument("--minimum-category-count", type=int, default=20)
    args = parser.parse_args()
    report = analyze_cta_premium(
        args.input, args.output, args.test_start_year, args.minimum_category_count
    )
    print(f"Best CTA distance specification: {report['best_out_of_sample_specification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
