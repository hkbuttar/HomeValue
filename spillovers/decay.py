"""Estimate how comparable-sale predictive information changes with distance."""

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

from ml.baselines import regression_metrics
from spillovers.comps import ComparableConfig, ComparableTier, generate_comparable_predictions


@dataclass(frozen=True)
class InformationDecayConfig:
    radii_miles: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    minimum_comparables: int = 3
    maximum_age_days: int = 1825
    maximum_sqft_log_difference: float = 0.55
    maximum_building_age_difference: float = 50.0
    distance_decay_miles: float = 0.75
    recency_half_life_days: float = 365.0
    evaluation_year: int | None = None


def _radius_predictions(base: pd.DataFrame, links: pd.DataFrame, radius: float,
                        minimum: int) -> pd.DataFrame:
    eligible = links.loc[links["distance_miles"].le(radius)].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["sale_id", "prediction", "comparable_count", "effective_count"])
    eligible["weighted_price"] = eligible["raw_weight"] * eligible["comparable_sale_price"]
    grouped = eligible.groupby("target_sale_id", observed=True).agg(
        comparable_count=("comparable_sale_id", "size"),
        weight_sum=("raw_weight", "sum"), weighted_price=("weighted_price", "sum"),
        squared_weight_sum=("raw_weight", lambda values: float(np.square(values).sum())),
    ).reset_index().rename(columns={"target_sale_id": "sale_id"})
    grouped = grouped.loc[grouped["comparable_count"].ge(minimum)].copy()
    grouped["prediction"] = grouped["weighted_price"] / grouped["weight_sum"]
    grouped["effective_count"] = grouped["weight_sum"] ** 2 / grouped["squared_weight_sum"]
    return grouped[["sale_id", "prediction", "comparable_count", "effective_count"]]


def _plot(curve: pd.DataFrame, output: Path) -> None:
    figure, error_axis = plt.subplots(figsize=(9, 6))
    coverage_axis = error_axis.twinx()
    error_axis.plot(curve["radius_miles"], curve["mae"], marker="o", color="#1f77b4")
    coverage_axis.plot(curve["radius_miles"], 100 * curve["coverage_rate"], marker="s", color="#d62728")
    error_axis.set(xlabel="Comparable radius (miles)", ylabel="Available-sample MAE ($)", title="Valuation information decay by radius")
    coverage_axis.set_ylabel("Coverage (%)")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def analyze_information_decay(input_path: Path, output_dir: Path,
                              config: InformationDecayConfig | None = None) -> dict:
    config = config or InformationDecayConfig()
    radii = tuple(sorted(set(float(radius) for radius in config.radii_miles)))
    if not radii or radii[0] <= 0:
        raise ValueError("radii_miles must contain positive values")
    source = pd.read_parquet(input_path)
    broad_config = ComparableConfig(
        minimum_comparables=1, maximum_comparables=max(100, len(source)),
        distance_decay_miles=config.distance_decay_miles,
        recency_half_life_days=config.recency_half_life_days,
        tiers=(ComparableTier(
            "decay_maximum", max(radii), config.maximum_age_days,
            config.maximum_sqft_log_difference, config.maximum_building_age_difference,
        ),),
    )
    base, links = generate_comparable_predictions(source, broad_config)
    year = int(config.evaluation_year or base["sale_date"].dt.year.max())
    evaluation = base.loc[base["sale_date"].dt.year.eq(year), ["sale_id", "sale_price"]].copy()
    wide = evaluation.copy()
    radius_frames = []
    for radius in radii:
        estimates = _radius_predictions(base, links, radius, config.minimum_comparables)
        estimates["radius_miles"] = radius
        radius_frames.append(estimates)
        wide = wide.merge(
            estimates[["sale_id", "prediction", "comparable_count", "effective_count"]].rename(columns={
                "prediction": f"prediction_{radius:g}",
                "comparable_count": f"comparable_count_{radius:g}",
                "effective_count": f"effective_count_{radius:g}",
            }), on="sale_id", how="left", validate="one_to_one",
        )
    curve_rows = []
    for radius in radii:
        column = f"prediction_{radius:g}"
        valid = wide[column].notna()
        metrics = regression_metrics(wide.loc[valid, "sale_price"], wide.loc[valid, column]) if valid.any() else None
        curve_rows.append({
            "radius_miles": radius, "eligible_sales": int(valid.sum()),
            "coverage_rate": float(valid.mean()),
            "mae": metrics["mae"] if metrics else None,
            "rmse": metrics["rmse"] if metrics else None,
            "median_absolute_error": metrics["median_absolute_error"] if metrics else None,
            "mape": metrics["mape"] if metrics else None,
        })
    curve = pd.DataFrame(curve_rows)
    common_mask = wide[[f"prediction_{radius:g}" for radius in radii]].notna().all(axis=1)
    common_metrics = {}
    for radius in radii:
        column = f"prediction_{radius:g}"
        common_metrics[str(radius)] = (
            regression_metrics(wide.loc[common_mask, "sale_price"], wide.loc[common_mask, column])
            if common_mask.any() else None
        )
    previous_mae = None
    for index, row in curve.iterrows():
        current = common_metrics[str(row["radius_miles"])]
        current_mae = current["mae"] if current else None
        curve.loc[index, "common_sample_mae"] = current_mae
        curve.loc[index, "marginal_mae_improvement"] = (
            previous_mae - current_mae if previous_mae is not None and current_mae is not None else np.nan
        )
        previous_mae = current_mae
    curve["incremental_coverage"] = curve["coverage_rate"].diff()
    best_available = curve.dropna(subset=["mae"])
    best_radius = float(best_available.loc[best_available["mae"].idxmin(), "radius_miles"]) if len(best_available) else None
    long_predictions = pd.concat(radius_frames, ignore_index=True).merge(
        evaluation, on="sale_id", how="inner", validate="many_to_one"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_dir / "information_decay_curve.csv", index=False)
    wide.to_parquet(output_dir / "radius_predictions.parquet", index=False)
    long_predictions.to_parquet(output_dir / "radius_comparable_details.parquet", index=False)
    links.to_parquet(output_dir / "maximum_radius_comparable_links.parquet", index=False)
    _plot(curve, output_dir / "information_decay_curve.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
        "config": asdict(config), "evaluation_year": year, "evaluation_sales": len(evaluation),
        "common_sample_sales": int(common_mask.sum()), "curve": curve_rows,
        "common_sample_metrics": common_metrics, "best_available_sample_mae_radius": best_radius,
        "leakage_rule": "Every comparable is strictly earlier than its target sale.",
        "interpretation": "Marginal MAE improvement measures the added predictive value of expanding to the next radius on identical targets; incremental coverage is reported separately.",
    }
    (output_dir / "information_decay_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/information_decay"))
    parser.add_argument("--evaluation-year", type=int)
    parser.add_argument("--minimum-comparables", type=int, default=3)
    args = parser.parse_args()
    report = analyze_information_decay(args.input, args.output, InformationDecayConfig(
        minimum_comparables=args.minimum_comparables, evaluation_year=args.evaluation_year,
    ))
    print(f"Best available-sample MAE radius: {report['best_available_sample_mae_radius']} miles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
