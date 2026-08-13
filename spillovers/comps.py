"""Build a leakage-safe, weighted local comparable-sales valuation engine."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import BallTree

from accessibility.cta import FEET_PER_MILE, PROJECTED_CRS
from ml.baselines import regression_metrics


@dataclass(frozen=True)
class ComparableTier:
    name: str
    radius_miles: float
    maximum_age_days: int
    maximum_sqft_log_difference: float
    maximum_building_age_difference: float


@dataclass(frozen=True)
class ComparableConfig:
    minimum_comparables: int = 3
    maximum_comparables: int = 10
    distance_decay_miles: float = 0.75
    recency_half_life_days: float = 365.0
    size_decay: float = 0.20
    age_decay_years: float = 15.0
    tiers: tuple[ComparableTier, ...] = (
        ComparableTier("strict", 1.0, 365, 0.25, 15),
        ComparableTier("relaxed", 3.0, 1095, 0.40, 30),
        ComparableTier("broad", 5.0, 1825, 0.55, 50),
    )


def _prepare(sales: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {
        "sale_id", "pin", "sale_date", "sale_price", "building_sqft",
        "building_age", "latitude", "longitude",
    }
    if missing := sorted(required.difference(sales.columns)):
        raise ValueError(f"comparable-sales input is missing: {', '.join(missing)}")
    frame = sales.copy().reset_index(drop=True)
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    for column in ("sale_price", "building_sqft", "building_age", "latitude", "longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        frame["sale_date"].notna() & frame["sale_price"].gt(0)
        & frame["building_sqft"].gt(0) & frame["building_age"].ge(0)
        & frame["latitude"].between(-90, 90) & frame["longitude"].between(-180, 180)
    )
    frame = frame.loc[valid].copy().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("comparable-sales engine requires at least two valid sales")
    property_type = next(
        (column for column in ("residence_type", "class") if column in frame), None
    )
    if property_type is None:
        raise ValueError("comparable-sales input requires residence_type or class")
    frame = frame.loc[frame[property_type].notna()].copy().reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("comparable-sales engine requires at least two sales with property type")
    frame[property_type] = frame[property_type].astype("string")
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
    frame["x_3435"], frame["y_3435"] = x, y
    return frame, property_type


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, quantile, side="left")])


def _weights(distance: np.ndarray, days: np.ndarray, sqft_difference: np.ndarray,
             age_difference: np.ndarray, config: ComparableConfig) -> np.ndarray:
    return (
        np.exp(-distance / config.distance_decay_miles)
        * np.exp(-np.log(2) * days / config.recency_half_life_days)
        * np.exp(-sqft_difference / config.size_decay)
        * np.exp(-age_difference / config.age_decay_years)
    )


def generate_comparable_predictions(
    sales: pd.DataFrame,
    config: ComparableConfig | None = None,
    batch_size: int = 5_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each sale using only earlier nearby comparable transactions."""
    config = config or ComparableConfig()
    frame, property_type = _prepare(sales)
    coordinates = frame[["x_3435", "y_3435"]].to_numpy(float)
    tree = BallTree(coordinates, metric="euclidean")
    maximum_radius = max(tier.radius_miles for tier in config.tiers) * FEET_PER_MILE
    outputs, links = [], []
    dates = frame["sale_date"].to_numpy(dtype="datetime64[ns]")
    prices = frame["sale_price"].to_numpy(float)
    sqft = frame["building_sqft"].to_numpy(float)
    ages = frame["building_age"].to_numpy(float)
    pins = frame["pin"].astype("string").to_numpy()
    types = frame[property_type].to_numpy()

    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        candidate_arrays, distance_arrays = tree.query_radius(
            coordinates[start:stop], r=maximum_radius, return_distance=True, sort_results=True
        )
        for local_position, (candidates, distances_feet) in enumerate(
            zip(candidate_arrays, distance_arrays)
        ):
            target = start + local_position
            distance_miles = distances_feet / FEET_PER_MILE
            days = (dates[target] - dates[candidates]).astype("timedelta64[D]").astype(float)
            sqft_difference = np.abs(np.log(sqft[candidates] / sqft[target]))
            age_difference = np.abs(ages[candidates] - ages[target])
            base = (
                (days > 0) & (pins[candidates] != pins[target])
                & (types[candidates] == types[target])
            )
            selected = np.array([], dtype=int)
            selected_metrics = None
            selected_tier = None
            for tier in config.tiers:
                mask = (
                    base & (distance_miles <= tier.radius_miles)
                    & (days <= tier.maximum_age_days)
                    & (sqft_difference <= tier.maximum_sqft_log_difference)
                    & (age_difference <= tier.maximum_building_age_difference)
                )
                positions = np.flatnonzero(mask)
                if len(positions) >= config.minimum_comparables:
                    raw_weight = _weights(
                        distance_miles[positions], days[positions], sqft_difference[positions],
                        age_difference[positions], config,
                    )
                    order = np.argsort(raw_weight)[::-1][: config.maximum_comparables]
                    selected = candidates[positions[order]]
                    selected_metrics = (
                        distance_miles[positions[order]], days[positions[order]],
                        sqft_difference[positions[order]], age_difference[positions[order]],
                        raw_weight[order],
                    )
                    selected_tier = tier.name
                    break
            record = {
                "sale_id": frame.loc[target, "sale_id"],
                "comparable_prediction": np.nan,
                "comparable_ppsf_prediction": np.nan,
                "comparable_interval_low": np.nan,
                "comparable_interval_high": np.nan,
                "comparable_count": 0,
                "comparable_effective_count": np.nan,
                "comparable_tier": pd.NA,
                "mean_comparable_distance_miles": np.nan,
                "most_recent_comparable_days": np.nan,
            }
            if len(selected):
                distances, recency, size_delta, age_delta, raw_weight = selected_metrics
                normalized = raw_weight / raw_weight.sum()
                selected_prices = prices[selected]
                selected_ppsf = selected_prices / sqft[selected]
                record.update({
                    "comparable_prediction": float(normalized @ selected_prices),
                    "comparable_ppsf_prediction": float((normalized @ selected_ppsf) * sqft[target]),
                    "comparable_interval_low": _weighted_quantile(selected_prices, raw_weight, 0.10),
                    "comparable_interval_high": _weighted_quantile(selected_prices, raw_weight, 0.90),
                    "comparable_count": len(selected),
                    "comparable_effective_count": float(1 / np.sum(normalized**2)),
                    "comparable_tier": selected_tier,
                    "mean_comparable_distance_miles": float(normalized @ distances),
                    "most_recent_comparable_days": float(recency.min()),
                })
                for index, candidate in enumerate(selected):
                    links.append({
                        "target_sale_id": frame.loc[target, "sale_id"],
                        "target_sale_date": frame.loc[target, "sale_date"],
                        "comparable_sale_id": frame.loc[candidate, "sale_id"],
                        "comparable_sale_date": frame.loc[candidate, "sale_date"],
                        "comparable_sale_price": prices[candidate],
                        "distance_miles": distances[index],
                        "recency_days": recency[index],
                        "sqft_log_difference": size_delta[index],
                        "building_age_difference": age_delta[index],
                        "raw_weight": raw_weight[index],
                        "normalized_weight": normalized[index],
                        "tier": selected_tier,
                    })
            outputs.append(record)
    predictions = frame.merge(pd.DataFrame(outputs), on="sale_id", how="left", validate="one_to_one")
    return predictions, pd.DataFrame(links)


def build_comparable_engine(
    input_path: Path,
    output_dir: Path,
    evaluation_year: int | None = None,
    config: ComparableConfig | None = None,
) -> dict:
    config = config or ComparableConfig()
    predictions, links = generate_comparable_predictions(pd.read_parquet(input_path), config)
    years = predictions["sale_date"].dt.year
    year = int(evaluation_year if evaluation_year is not None else years.max())
    evaluation = predictions.loc[
        years.eq(year) & predictions["comparable_prediction"].notna()
    ]
    metrics = (
        regression_metrics(evaluation["sale_price"], evaluation["comparable_prediction"])
        if len(evaluation) else None
    )
    ppsf_metrics = (
        regression_metrics(evaluation["sale_price"], evaluation["comparable_ppsf_prediction"])
        if len(evaluation) else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "comparable_predictions.parquet", index=False)
    links.to_parquet(output_dir / "comparable_links.parquet", index=False)
    predicted = predictions["comparable_prediction"].notna()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "config": asdict(config),
        "sales": len(predictions),
        "sales_with_comparables": int(predicted.sum()),
        "coverage_rate": float(predicted.mean()),
        "tier_counts": {
            str(key): int(value)
            for key, value in predictions["comparable_tier"].value_counts(dropna=False).items()
        },
        "evaluation_year": year,
        "evaluation_rows": len(evaluation),
        "weighted_price_metrics": metrics,
        "weighted_ppsf_metrics": ppsf_metrics,
        "leakage_rule": "Every comparable sale date is strictly earlier than its target sale date.",
    }
    (output_dir / "comparable_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/comparables"))
    parser.add_argument("--evaluation-year", type=int)
    parser.add_argument("--minimum-comparables", type=int, default=3)
    parser.add_argument("--maximum-comparables", type=int, default=10)
    args = parser.parse_args()
    config = ComparableConfig(
        minimum_comparables=args.minimum_comparables,
        maximum_comparables=args.maximum_comparables,
    )
    report = build_comparable_engine(args.input, args.output, args.evaluation_year, config)
    print(f"Comparable coverage: {report['coverage_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
