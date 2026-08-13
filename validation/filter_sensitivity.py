"""Compare major findings under strict and moderate market-sale definitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from spreg import ML_Lag

from hedonic.model import HedonicConfig, HedonicModel
from ml.baselines import regression_metrics, temporal_split
from preprocessing.population import PopulationRules, QUALITY_FILTERS, _boolean
from spatial.lag_model import _weights


@dataclass(frozen=True)
class FilterSensitivityConfig:
    strict_minimum_price: float = 25_000
    moderate_minimum_price: float = 10_000
    minimum_category_count: int = 20
    k_neighbors: int = 8


def define_market_sale_samples(frame: pd.DataFrame, config: FilterSensitivityConfig | None = None) -> pd.DataFrame:
    config = config or FilterSensitivityConfig()
    required = {*QUALITY_FILTERS, "class", "sale_price", "sale_date", "building_sqft", "latitude", "longitude"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"filter sensitivity requires Cook County quality indicators: {', '.join(missing)}")
    result = frame.copy()
    quality = pd.DataFrame({column: _boolean(result[column]) for column in QUALITY_FILTERS})
    explicit_failure = quality.eq(True).any(axis=1)
    explicit_clean = quality.eq(False).all(axis=1)
    multisale = _boolean(result["is_multisale"]) if "is_multisale" in result else pd.Series(False, index=result.index, dtype="boolean")
    classes = result["class"].astype("string").str.replace(r"\.0$", "", regex=True)
    rules = PopulationRules()
    price = pd.to_numeric(result["sale_price"], errors="coerce")
    core_complete = pd.Series(True, index=result.index)
    for column in ("sale_date", "building_sqft", "latitude", "longitude"):
        if column in result:
            core_complete &= result[column].notna()
    result["is_strict_market_sale"] = (
        explicit_clean & multisale.ne(True).fillna(False)
        & classes.isin(rules.single_family_classes)
        & price.ge(config.strict_minimum_price) & core_complete
    )
    result["is_moderate_market_sale"] = (
        ~explicit_failure & multisale.ne(True).fillna(False)
        & classes.isin((*rules.single_family_classes, *rules.small_multifamily_classes))
        & price.ge(config.moderate_minimum_price) & core_complete
    )
    return result


def _hedonic_analysis(sample: pd.DataFrame, config: FilterSensitivityConfig):
    train, test, cutoff = temporal_split(sample)
    model_config = HedonicConfig(
        minimum_category_count=config.minimum_category_count,
        include_time=True, include_property_type=True, include_neighborhood=True,
        include_accessibility=True,
    )
    model = HedonicModel(model_config).fit(train)
    prediction = model.predict(test)
    full_model = HedonicModel(model_config).fit(sample)
    coefficients = full_model.coefficient_table()
    coefficients["implied_one_unit_price_change"] = np.expm1(coefficients["coefficient"])
    return {
        "test_start_year": cutoff, "train_rows": len(train), "test_rows": len(test),
        "metrics": regression_metrics(test["sale_price"], prediction),
    }, coefficients


def _spatial_parameter(sample: pd.DataFrame, config: FilterSensitivityConfig) -> dict:
    if not {"x_3435", "y_3435"}.issubset(sample.columns) or len(sample) < 5:
        return {"status": "unavailable_missing_projected_coordinates"}
    columns = [
        column for column in (
            "building_sqft", "bathrooms", "building_age", "cta_distance_miles",
            "lake_distance_miles", "median_household_income",
        ) if column in sample
    ]
    design = sample[columns].apply(pd.to_numeric, errors="coerce")
    design = design.fillna(design.median())
    design = design.loc[:, design.nunique().gt(1)]
    design = (design - design.mean()) / design.std().replace(0, 1)
    _, triangular, pivots = qr(design.to_numpy(), mode="economic", pivoting=True)
    tolerance = np.finfo(float).eps * max(design.shape) * abs(triangular).max()
    rank = int(np.sum(np.abs(np.diag(triangular)) > tolerance))
    design = design.iloc[:, sorted(pivots[:rank])]
    columns = design.columns.tolist()
    target = np.log(pd.to_numeric(sample["sale_price"], errors="coerce")).to_numpy()
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    weights = _weights(coordinates, min(config.k_neighbors, len(sample) - 1))
    model = ML_Lag(
        target.reshape(-1, 1), design.to_numpy(), weights, method="ord", vm=True,
        name_y="log_sale_price", name_x=columns,
    )
    return {
        "status": "fitted", "rho": float(model.rho),
        "rho_standard_error": float(model.std_err[-1]),
        "rho_p_value": float(model.z_stat[-1][1]), "pseudo_r_squared": float(model.pr2),
    }


def _accessibility_summary(coefficients: pd.DataFrame, prefix: str) -> list[dict]:
    selected = coefficients.loc[coefficients["term"].str.contains(prefix, case=False, regex=False)].copy()
    return selected[[
        "term", "coefficient", "robust_std_error", "p_value", "ci_lower", "ci_upper",
        "implied_one_unit_price_change",
    ]].to_dict(orient="records")


def analyze_filter_sensitivity(input_path: Path, output_dir: Path,
                               config: FilterSensitivityConfig | None = None) -> dict:
    config = config or FilterSensitivityConfig()
    classified = define_market_sale_samples(pd.read_parquet(input_path), config)
    definitions = {
        "strict": classified.loc[classified["is_strict_market_sale"]].copy(),
        "moderate": classified.loc[classified["is_moderate_market_sale"]].copy(),
    }
    results, coefficient_tables = {}, []
    for name, sample in definitions.items():
        if sample.empty:
            raise ValueError(f"{name} market-sale definition produced no observations")
        valuation, coefficients = _hedonic_analysis(sample, config)
        coefficients.insert(0, "sale_definition", name)
        coefficient_tables.append(coefficients)
        results[name] = {
            "sample_rows": len(sample), "sample_share": float(len(sample) / len(classified)),
            "valuation": valuation, "spatial": _spatial_parameter(sample, config),
            "cta_results": _accessibility_summary(coefficients, "cta"),
            "lake_results": _accessibility_summary(coefficients, "lake"),
        }
    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    comparison_rows = []
    shared_terms = set(
        coefficients.loc[coefficients["sale_definition"].eq("strict"), "term"]
    ).intersection(coefficients.loc[coefficients["sale_definition"].eq("moderate"), "term"])
    for term in sorted(shared_terms):
        values = coefficients.loc[coefficients["term"].eq(term)].set_index("sale_definition")
        strict, moderate = values.loc["strict"], values.loc["moderate"]
        comparison_rows.append({
            "term": term, "strict_coefficient": strict["coefficient"],
            "moderate_coefficient": moderate["coefficient"],
            "absolute_change": moderate["coefficient"] - strict["coefficient"],
            "relative_absolute_change": (
                abs(moderate["coefficient"] - strict["coefficient"]) / abs(strict["coefficient"])
                if strict["coefficient"] != 0 else None
            ),
            "sign_changed": bool(np.sign(strict["coefficient"]) != np.sign(moderate["coefficient"])),
            "strict_significant": bool(strict["p_value"] < .05),
            "moderate_significant": bool(moderate["p_value"] < .05),
        })
    comparison = pd.DataFrame(comparison_rows)
    focus = comparison.loc[comparison["term"].str.contains("cta|lake", case=False, regex=True)]
    stable_focus = bool(
        len(focus) and not focus["sign_changed"].any()
        and (focus["strict_significant"] == focus["moderate_significant"]).all()
    )
    conclusion = (
        "CTA and lake coefficient directions and significance classifications were stable across sale definitions."
        if stable_focus else
        "At least one CTA or lake finding changed direction or significance classification across sale definitions; interpret it as filter-sensitive."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    classified.to_parquet(output_dir / "sales_with_filter_definitions.parquet", index=False)
    coefficients.to_csv(output_dir / "filter_coefficient_estimates.csv", index=False)
    comparison.to_csv(output_dir / "filter_coefficient_stability.csv", index=False)
    summary = pd.DataFrame([
        {
            "sale_definition": name, "sample_rows": values["sample_rows"],
            "test_mae": values["valuation"]["metrics"]["mae"],
            "test_rmse": values["valuation"]["metrics"]["rmse"],
            "test_mdape": values["valuation"]["metrics"]["median_absolute_percentage_error"],
            "spatial_rho": values["spatial"].get("rho"),
            "spatial_rho_p_value": values["spatial"].get("rho_p_value"),
        } for name, values in results.items()
    ])
    summary.to_csv(output_dir / "filter_sensitivity_summary.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
        "config": asdict(config), "input_rows": len(classified), "definitions": results,
        "accessibility_findings_stable": stable_focus, "conclusion": conclusion,
        "strict_definition": "Explicitly clean Cook quality flags, single-family class, non-multisale, complete core fields, and strict minimum price.",
        "moderate_definition": "No explicit Cook quality failure, single-family or small-multifamily class, non-multisale, complete core fields, and moderate minimum price; missing filter metadata is allowed.",
    }
    (output_dir / "filter_sensitivity_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation/filter_sensitivity"))
    parser.add_argument("--minimum-category-count", type=int, default=20)
    parser.add_argument("--k-neighbors", type=int, default=8)
    args = parser.parse_args()
    report = analyze_filter_sensitivity(args.input, args.output, FilterSensitivityConfig(
        minimum_category_count=args.minimum_category_count, k_neighbors=args.k_neighbors,
    ))
    print(report["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
