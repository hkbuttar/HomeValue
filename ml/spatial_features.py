"""Construct leakage-safe prior-sale spatial features for valuation models."""

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


@dataclass(frozen=True)
class SpatialFeatureConfig:
    radius_miles: float = 1.0
    lookback_days: int = 1095
    distance_decay_miles: float = 0.5
    recency_half_life_days: float = 365.0
    minimum_neighborhood_sales: int = 3
    appreciation_window_days: int = 730


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sale_id", "pin", "sale_date", "sale_price", "building_sqft",
        "latitude", "longitude",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"spatial feature input is missing: {', '.join(missing)}")
    result = frame.copy().reset_index(drop=True)
    result["sale_date"] = pd.to_datetime(result["sale_date"], errors="coerce")
    for column in ("sale_price", "building_sqft", "latitude", "longitude"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid = (
        result["sale_date"].notna() & result["sale_price"].gt(0)
        & result["building_sqft"].gt(0)
        & result["latitude"].between(-90, 90)
        & result["longitude"].between(-180, 180)
    )
    result = result.loc[valid].copy().reset_index(drop=True)
    if result["sale_id"].duplicated().any():
        raise ValueError("sale_id must be unique")
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(result["longitude"].to_numpy(), result["latitude"].to_numpy())
    result["x_3435"], result["y_3435"] = x, y
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")])


def _neighborhood_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in ("nbhd", "census_tract") if column in frame), None)


def _neighborhood_appreciation(
    frame: pd.DataFrame,
    target: int,
    config: SpatialFeatureConfig,
    neighborhood_column: str | None,
    neighborhood_lookup: dict,
) -> tuple[float, int]:
    if neighborhood_column is None or pd.isna(frame.loc[target, neighborhood_column]):
        return np.nan, 0
    target_date = frame.loc[target, "sale_date"]
    neighborhood = frame.loc[target, neighborhood_column]
    group = neighborhood_lookup.get(neighborhood)
    if group is None:
        return np.nan, 0
    dates = group["sale_date"].to_numpy(dtype="datetime64[ns]")
    end = np.searchsorted(dates, np.datetime64(target_date), side="left")
    start_date = np.datetime64(target_date - pd.Timedelta(days=config.appreciation_window_days))
    start = np.searchsorted(dates, start_date, side="left")
    recent = group.iloc[start:end].copy()
    if len(recent) < 2 * config.minimum_neighborhood_sales:
        return np.nan, len(recent)
    recent["period"] = np.where(
        (target_date - recent["sale_date"]).dt.days
        <= config.appreciation_window_days / 2,
        "newer", "older",
    )
    medians = recent.groupby("period")["sale_price"].median()
    counts = recent.groupby("period")["sale_price"].size()
    if not {"older", "newer"}.issubset(medians.index):
        return np.nan, len(recent)
    if counts["older"] < config.minimum_neighborhood_sales or counts["newer"] < config.minimum_neighborhood_sales:
        return np.nan, len(recent)
    return float(medians["newer"] / medians["older"] - 1), len(recent)


def engineer_prior_spatial_features(
    sales: pd.DataFrame,
    config: SpatialFeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute local summaries using only sales strictly before each target."""
    config = config or SpatialFeatureConfig()
    frame = _prepare(sales)
    coordinates = frame[["x_3435", "y_3435"]].to_numpy(float)
    tree = BallTree(coordinates)
    candidates_all, distances_all = tree.query_radius(
        coordinates, r=config.radius_miles * FEET_PER_MILE,
        return_distance=True, sort_results=True,
    )
    dates = frame["sale_date"].to_numpy(dtype="datetime64[ns]")
    prices = frame["sale_price"].to_numpy(float)
    sqft = frame["building_sqft"].to_numpy(float)
    pins = frame["pin"].astype("string").to_numpy()
    neighborhood_column = _neighborhood_column(frame)
    neighborhood_lookup = (
        {
            key: group.sort_values("sale_date").reset_index(drop=True)
            for key, group in frame.loc[frame[neighborhood_column].notna()].groupby(
                neighborhood_column, observed=True
            )
        }
        if neighborhood_column else {}
    )
    rows, audit_links = [], []
    for target, (candidates, distance_feet) in enumerate(zip(candidates_all, distances_all)):
        days = (dates[target] - dates[candidates]).astype("timedelta64[D]").astype(float)
        distance_miles = distance_feet / FEET_PER_MILE
        valid = (
            (days > 0) & (days <= config.lookback_days)
            & (pins[candidates] != pins[target])
        )
        selected = candidates[valid]
        selected_days = days[valid]
        selected_distances = distance_miles[valid]
        appreciation, neighborhood_count = _neighborhood_appreciation(
            frame, target, config, neighborhood_column, neighborhood_lookup
        )
        record = {
            "sale_id": frame.loc[target, "sale_id"],
            "prior_nearby_sale_median": np.nan,
            "prior_nearby_sale_count": len(selected),
            "prior_nearby_weighted_ppsf": np.nan,
            "prior_nearby_most_recent_days": np.nan,
            "prior_nearby_mean_distance_miles": np.nan,
            "neighborhood_prior_appreciation": appreciation,
            "neighborhood_prior_sale_count": neighborhood_count,
        }
        if len(selected):
            weights = (
                np.exp(-selected_distances / config.distance_decay_miles)
                * np.exp(-np.log(2) * selected_days / config.recency_half_life_days)
            )
            normalized = weights / weights.sum()
            record.update({
                "prior_nearby_sale_median": _weighted_median(prices[selected], weights),
                "prior_nearby_weighted_ppsf": float(
                    normalized @ (prices[selected] / sqft[selected])
                ),
                "prior_nearby_most_recent_days": float(selected_days.min()),
                "prior_nearby_mean_distance_miles": float(normalized @ selected_distances),
            })
            for candidate, distance_value, day_value, weight in zip(
                selected, selected_distances, selected_days, normalized
            ):
                audit_links.append({
                    "target_sale_id": frame.loc[target, "sale_id"],
                    "target_sale_date": frame.loc[target, "sale_date"],
                    "prior_sale_id": frame.loc[candidate, "sale_id"],
                    "prior_sale_date": frame.loc[candidate, "sale_date"],
                    "distance_miles": distance_value,
                    "recency_days": day_value,
                    "normalized_weight": weight,
                })
        rows.append(record)
    return pd.DataFrame(rows), pd.DataFrame(audit_links)


def build_spatial_features(
    input_path: Path,
    output_dir: Path,
    config: SpatialFeatureConfig | None = None,
) -> dict:
    config = config or SpatialFeatureConfig()
    source = pd.read_parquet(input_path)
    features, links = engineer_prior_spatial_features(source, config)
    feature_columns = [column for column in features if column != "sale_id"]
    enriched = source.drop(columns=[column for column in feature_columns if column in source]).merge(
        features, on="sale_id", how="left", validate="one_to_one"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "prior_spatial_features.parquet", index=False)
    links.to_parquet(output_dir / "prior_spatial_feature_links.parquet", index=False)
    enriched.to_parquet(output_dir / "core_sales_with_spatial_features.parquet", index=False)
    coverage = {
        column: float(features[column].notna().mean())
        for column in feature_columns
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path), "sales": len(features),
        "config": asdict(config), "coverage": coverage,
        "prior_links": len(links),
        "strict_temporal_validation": bool(
            links.empty or (links["prior_sale_date"] < links["target_sale_date"]).all()
        ),
        "leakage_rule": "PriorSaleDate is strictly less than TargetSaleDate for every linked sale.",
    }
    (output_dir / "prior_spatial_feature_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/spatial_features"))
    parser.add_argument("--radius-miles", type=float, default=1.0)
    parser.add_argument("--lookback-days", type=int, default=1095)
    args = parser.parse_args()
    config = SpatialFeatureConfig(radius_miles=args.radius_miles, lookback_days=args.lookback_days)
    report = build_spatial_features(args.input, args.output, config)
    print(f"Built prior-only spatial features for {report['sales']} sales")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
