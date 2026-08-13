"""Stress-test the CTA distance association across controls and geographies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from spreg import ML_Lag

from hedonic.model import HedonicConfig, HedonicModel
from spatial.lag_model import _weights
from transit.premium import DistanceBasis, _distance_column


ACS_CONTROLS = (
    "median_household_income", "poverty_rate", "bachelors_or_higher_rate",
    "owner_occupancy_rate", "vacancy_rate", "population_density",
)


@dataclass(frozen=True)
class TransitRobustnessConfig:
    minimum_category_count: int = 20
    minimum_subset_rows: int = 100
    k_neighbors: int = 8
    significance_level: float = 0.05
    reference_distance_miles: float = 3.0


def _design(frame: pd.DataFrame, distance: str, include_time: bool,
            include_neighborhood: bool, include_acs: bool, minimum_category_count: int):
    model = HedonicModel(HedonicConfig(
        minimum_category_count=minimum_category_count, include_time=include_time,
        include_property_type=True, include_neighborhood=include_neighborhood,
        include_accessibility=False,
    ))
    model._fit_schema(frame)
    design = model._design(frame)
    design["cta_distance_linear"] = pd.to_numeric(frame[distance], errors="coerce")
    if include_acs:
        for column in ACS_CONTROLS:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() >= 2:
                scale = float(values.std()) or 1.0
                design[f"acs_{column}"] = (values.fillna(values.median()) - values.mean()) / scale
    return design.astype(float)


def _ols_result(name: str, frame: pd.DataFrame, design: pd.DataFrame,
                distance_term: str = "cta_distance_linear") -> tuple[dict, pd.DataFrame]:
    target = np.log(pd.to_numeric(frame["sale_price"], errors="coerce"))
    result = sm.OLS(target, design, missing="raise").fit(cov_type="HC3")
    position = design.columns.get_loc(distance_term)
    parameters = np.asarray(result.params)
    standard_errors = np.asarray(result.bse)
    p_values = np.asarray(result.pvalues)
    interval = np.asarray(result.conf_int())[position]
    row = {
        "specification": name, "rows": len(frame), "status": "fitted",
        "distance_coefficient": float(parameters[position]),
        "distance_standard_error": float(standard_errors[position]),
        "distance_p_value": float(p_values[position]),
        "ci_lower": float(interval[0]), "ci_upper": float(interval[1]),
        "one_mile_closer_price_effect": float(np.exp(-parameters[position]) - 1),
        "adjusted_r_squared": float(result.rsquared_adj),
    }
    coefficients = pd.DataFrame({
        "specification": name, "term": design.columns, "coefficient": parameters,
        "standard_error": standard_errors, "p_value": p_values,
    })
    return row, coefficients


def _nonlinear_result(frame: pd.DataFrame, base: pd.DataFrame, distance: str,
                      reference_distance: float) -> tuple[dict, pd.DataFrame]:
    basis = DistanceBasis("cubic_spline").fit(frame[distance])
    design = pd.concat([base.drop(columns="cta_distance_linear"), basis.transform(frame[distance])], axis=1)
    target = np.log(pd.to_numeric(frame["sale_price"], errors="coerce"))
    result = sm.OLS(target, design).fit(cov_type="HC3")
    positions = [design.columns.get_loc(column) for column in basis.columns_]
    beta, covariance = np.asarray(result.params)[positions], np.asarray(result.cov_params())[np.ix_(positions, positions)]
    statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
    near = basis.transform(pd.Series([0.5])).iloc[0]
    reference = basis.transform(pd.Series([reference_distance])).iloc[0]
    contrast = near.to_numpy() - reference.to_numpy()
    effect = float(np.exp(contrast @ beta) - 1)
    rows = pd.DataFrame({
        "specification": "nonlinear_distance", "term": basis.columns_,
        "coefficient": beta, "standard_error": np.sqrt(np.diag(covariance)),
    })
    return {
        "specification": "nonlinear_distance", "rows": len(frame), "status": "fitted",
        "distance_coefficient": None, "distance_standard_error": None,
        "distance_p_value": float(chi2.sf(statistic, len(beta))),
        "ci_lower": None, "ci_upper": None,
        "one_mile_closer_price_effect": None,
        "near_vs_reference_price_effect": effect,
        "adjusted_r_squared": float(result.rsquared_adj),
    }, rows


def _spatial_result(frame: pd.DataFrame, design: pd.DataFrame, config: TransitRobustnessConfig):
    if not {"x_3435", "y_3435"}.issubset(frame.columns):
        return {"specification": "spatial_dependence", "rows": len(frame), "status": "unavailable_missing_projected_coordinates"}, pd.DataFrame()
    coordinates = frame[["x_3435", "y_3435"]].to_numpy(float)
    weights = _weights(coordinates, min(config.k_neighbors, len(frame) - 1))
    predictors = design.drop(columns="intercept", errors="ignore")
    target = np.log(pd.to_numeric(frame["sale_price"], errors="coerce")).to_numpy()
    result = ML_Lag(
        target.reshape(-1, 1), predictors.to_numpy(), weights, method="ord", vm=True,
        name_y="log_sale_price", name_x=predictors.columns.tolist(),
    )
    names = result.name_x
    position = names.index("cta_distance_linear")
    coefficient = float(result.betas.reshape(-1)[position])
    standard_error = float(result.std_err[position])
    z_value, p_value = result.z_stat[position]
    rows = pd.DataFrame({
        "specification": "spatial_dependence", "term": names,
        "coefficient": result.betas.reshape(-1), "standard_error": result.std_err,
        "p_value": [value[1] for value in result.z_stat],
    })
    return {
        "specification": "spatial_dependence", "rows": len(frame), "status": "fitted",
        "distance_coefficient": coefficient, "distance_standard_error": standard_error,
        "distance_p_value": float(p_value), "ci_lower": coefficient - 1.96 * standard_error,
        "ci_upper": coefficient + 1.96 * standard_error,
        "one_mile_closer_price_effect": float(np.exp(-coefficient) - 1),
        "spatial_rho": float(result.rho), "spatial_rho_p_value": float(result.z_stat[-1][1]),
        "pseudo_r_squared": float(result.pr2), "distance_z_value": float(z_value),
    }, rows


def _subsets(frame: pd.DataFrame, minimum: int):
    if "municipality" in frame:
        counts = frame["municipality"].value_counts()
        for value in counts[counts >= minimum].index[:8]:
            yield f"municipality={value}", frame.loc[frame["municipality"].eq(value)]
    elif "longitude" in frame:
        split = pd.to_numeric(frame["longitude"], errors="coerce").median()
        for label, mask in (("west", frame["longitude"].le(split)), ("east", frame["longitude"].gt(split))):
            subset = frame.loc[mask]
            if len(subset) >= minimum:
                yield f"longitude={label}", subset


def analyze_transit_robustness(input_path: Path, output_dir: Path,
                               config: TransitRobustnessConfig | None = None) -> dict:
    config = config or TransitRobustnessConfig()
    frame = pd.read_parquet(input_path)
    distance = _distance_column(frame)
    valid = pd.to_numeric(frame[distance], errors="coerce").ge(0) & pd.to_numeric(frame["sale_price"], errors="coerce").gt(0)
    frame = frame.loc[valid].dropna(subset=[distance]).copy()
    specifications, coefficient_tables = [], []
    stages = [
        ("property_controls", False, False, False),
        ("plus_year_effects", True, False, False),
        ("plus_neighborhood_controls", True, True, False),
        ("plus_acs_controls", True, True, True),
    ]
    designs = {}
    for name, time, neighborhood, acs in stages:
        designs[name] = _design(
            frame, distance, time, neighborhood, acs, config.minimum_category_count
        )
        row, coefficients = _ols_result(name, frame, designs[name])
        specifications.append(row)
        coefficient_tables.append(coefficients)
    spatial_row, spatial_coefficients = _spatial_result(frame, designs["plus_acs_controls"], config)
    specifications.append(spatial_row)
    if len(spatial_coefficients):
        coefficient_tables.append(spatial_coefficients)
    nonlinear_row, nonlinear_coefficients = _nonlinear_result(
        frame, designs["plus_acs_controls"], distance, config.reference_distance_miles
    )
    specifications.append(nonlinear_row)
    coefficient_tables.append(nonlinear_coefficients)
    subset_names = []
    for label, subset in _subsets(frame, config.minimum_subset_rows):
        name = f"subset_{label}"
        row, coefficients = _ols_result(name, subset, _design(
            subset, distance, True, True, True, config.minimum_category_count
        ))
        specifications.append(row)
        coefficient_tables.append(coefficients)
        subset_names.append(name)
    results = pd.DataFrame(specifications)
    results["distance_significant"] = pd.to_numeric(results["distance_p_value"], errors="coerce").lt(config.significance_level)
    initial = results.loc[results["specification"].eq("plus_year_effects")].iloc[0]
    neighborhood = results.loc[results["specification"].eq("plus_neighborhood_controls")].iloc[0]
    attenuated = bool(initial["distance_significant"] and not neighborhood["distance_significant"])
    fitted_core = results.loc[results["specification"].isin([
        "property_controls", "plus_year_effects", "plus_neighborhood_controls",
        "plus_acs_controls", "spatial_dependence", "nonlinear_distance",
    ]) & results["status"].eq("fitted")]
    survives = bool(fitted_core["distance_significant"].all())
    conclusion = (
        "The apparent CTA distance association disappeared after neighborhood controls were introduced."
        if attenuated else
        "The CTA distance association remained statistically detectable across every fitted core robustness specification."
        if survives else
        "The CTA distance association was not stable across the fitted robustness specifications."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "transit_robustness_specifications.csv", index=False)
    pd.concat(coefficient_tables, ignore_index=True).to_csv(output_dir / "transit_robustness_coefficients.csv", index=False)
    plottable = results.dropna(subset=["distance_coefficient", "ci_lower", "ci_upper"])
    figure, axis = plt.subplots(figsize=(9, max(4, .45 * len(plottable))))
    axis.errorbar(
        plottable["distance_coefficient"], np.arange(len(plottable)),
        xerr=[plottable["distance_coefficient"] - plottable["ci_lower"], plottable["ci_upper"] - plottable["distance_coefficient"]],
        fmt="o",
    )
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_yticks(np.arange(len(plottable)), plottable["specification"])
    axis.set(xlabel="Log-price coefficient per additional CTA mile", title="CTA premium robustness")
    figure.tight_layout()
    figure.savefig(output_dir / "transit_robustness.png", dpi=170)
    plt.close(figure)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
        "config": asdict(config), "distance_column": distance, "sample_rows": len(frame),
        "specifications": specifications, "alternative_geographic_subsets": subset_names,
        "attenuated_after_neighborhood_controls": attenuated,
        "survived_all_fitted_core_checks": survives, "conclusion": conclusion,
        "interpretation_caution": "These are conditional associations, not causal transit effects.",
    }
    (output_dir / "transit_robustness_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/transit_robustness"))
    parser.add_argument("--minimum-category-count", type=int, default=20)
    parser.add_argument("--minimum-subset-rows", type=int, default=100)
    parser.add_argument("--k-neighbors", type=int, default=8)
    args = parser.parse_args()
    report = analyze_transit_robustness(args.input, args.output, TransitRobustnessConfig(
        minimum_category_count=args.minimum_category_count,
        minimum_subset_rows=args.minimum_subset_rows, k_neighbors=args.k_neighbors,
    ))
    print(report["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
