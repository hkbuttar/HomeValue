"""Re-estimate neighborhood market segments over time and measure transitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import RobustScaler

from segmentation.neighborhoods import PROFILE_FEATURES, build_market_profiles


@dataclass(frozen=True)
class StabilityConfig:
    period_years: int = 3
    minimum_periods: int = 2
    minimum_neighborhood_sales: int = 10
    clusters: int = 4
    random_seed: int = 42


def _periods(years: pd.Series, width: int) -> list[tuple[int, int]]:
    if width < 1:
        raise ValueError("period_years must be positive")
    first, last = int(years.min()), int(years.max())
    return [(start, min(start + width - 1, last)) for start in range(first, last + 1, width)]


def _shared_features(period_profiles: list[pd.DataFrame]) -> list[str]:
    return [
        feature for feature in PROFILE_FEATURES
        if all(feature in frame and frame[feature].notna().sum() >= 2 for frame in period_profiles)
    ]


def _fit_periods(period_profiles: list[pd.DataFrame], features: list[str], geography: str,
                 config: StabilityConfig) -> tuple[pd.DataFrame, list[np.ndarray]]:
    pooled = pd.concat([frame[features] for frame in period_profiles], ignore_index=True)
    imputer = SimpleImputer(strategy="median").fit(pooled)
    scaler = RobustScaler().fit(imputer.transform(pooled))
    assignments, centroids = [], []
    for number, profiles in enumerate(period_profiles):
        matrix = scaler.transform(imputer.transform(profiles[features]))
        distinct = len(np.unique(matrix, axis=0))
        clusters = min(config.clusters, distinct, len(matrix))
        if clusters < 2:
            raise ValueError("each period requires at least two distinct neighborhood profiles")
        model = KMeans(
            n_clusters=clusters, random_state=config.random_seed + number, n_init=30
        ).fit(matrix)
        current = profiles[[geography, "period", "period_start", "period_end"]].copy()
        current["raw_cluster"] = model.labels_
        assignments.append(current)
        centroids.append(model.cluster_centers_)
    return pd.concat(assignments, ignore_index=True), centroids


def _align_labels(assignments: pd.DataFrame, centroids: list[np.ndarray]) -> pd.DataFrame:
    aligned = assignments.copy()
    periods = list(aligned["period"].drop_duplicates())
    mappings: list[dict[int, int]] = [{index: index for index in range(len(centroids[0]))}]
    next_label = len(centroids[0])
    reference = centroids[0].copy()
    reference_labels = list(range(len(reference)))
    for current_centroids in centroids[1:]:
        distances = np.linalg.norm(reference[:, None, :] - current_centroids[None, :, :], axis=2)
        previous_rows, current_columns = linear_sum_assignment(distances)
        mapping = {int(column): reference_labels[int(row)] for row, column in zip(previous_rows, current_columns)}
        for cluster in range(len(current_centroids)):
            if cluster not in mapping:
                mapping[cluster] = next_label
                next_label += 1
        mappings.append(mapping)
        reference = current_centroids
        reference_labels = [mapping[index] for index in range(len(current_centroids))]
    aligned["segment"] = -1
    for period, mapping in zip(periods, mappings):
        mask = aligned["period"].eq(period)
        aligned.loc[mask, "segment"] = aligned.loc[mask, "raw_cluster"].map(mapping)
    return aligned


def _transitions(assignments: pd.DataFrame, geography: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = assignments.sort_values([geography, "period_start"]).copy()
    period_order = {
        period: order for order, period in enumerate(
            assignments.sort_values("period_start")["period"].drop_duplicates()
        )
    }
    ordered["next_segment"] = ordered.groupby(geography)["segment"].shift(-1)
    ordered["next_period"] = ordered.groupby(geography)["period"].shift(-1)
    adjacent = ordered.loc[
        ordered["next_segment"].notna()
        & ordered["next_period"].notna()
        & ordered.apply(
            lambda row: period_order.get(row["next_period"], -1)
            == period_order.get(row["period"], -1) + 1,
            axis=1,
        )
    ].copy()
    adjacent["to_segment"] = adjacent["next_segment"].astype(int)
    adjacent = adjacent.rename(columns={"segment": "from_segment", "period": "from_period"})
    counts = adjacent.groupby(
        ["from_period", "next_period", "from_segment", "to_segment"], observed=True
    ).size().rename("transition_count").reset_index().rename(columns={"next_period": "to_period"})
    if counts.empty:
        counts["transition_probability"] = pd.Series(dtype=float)
    else:
        totals = counts.groupby(["from_period", "to_period", "from_segment"])["transition_count"].transform("sum")
        counts["transition_probability"] = counts["transition_count"] / totals
    history = assignments.sort_values([geography, "period_start"])
    persistence = history.groupby(geography).agg(
        periods_observed=("period", "size"), dominant_segment=("segment", lambda values: values.mode().iloc[0]),
        distinct_segments=("segment", "nunique"),
    ).reset_index()
    same = adjacent.assign(stayed=adjacent["from_segment"].eq(adjacent["to_segment"]))
    rates = same.groupby(geography)["stayed"].mean().rename("persistence_rate").reset_index()
    persistence = persistence.merge(rates, on=geography, how="left")
    return counts, persistence


def analyze_segment_stability(
    sales_path: Path,
    indices_path: Path | None,
    output_dir: Path,
    config: StabilityConfig | None = None,
) -> dict:
    config = config or StabilityConfig()
    sales = pd.read_parquet(sales_path)
    sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
    valid_years = sales["sale_date"].dropna().dt.year
    if valid_years.empty:
        raise ValueError("sales input contains no valid sale dates")
    indices = pd.read_parquet(indices_path) if indices_path and indices_path.exists() else None
    period_profiles, labels, geography = [], [], None
    for start, end in _periods(valid_years, config.period_years):
        period_sales = sales.loc[sales["sale_date"].dt.year.between(start, end)]
        period_indices = None
        if indices is not None:
            period_indices = indices.loc[pd.to_numeric(indices["year"], errors="coerce").between(start, end)]
        profiles, current_geography = build_market_profiles(period_sales, period_indices)
        geography = geography or current_geography
        profiles = profiles.loc[profiles["sale_count"].ge(config.minimum_neighborhood_sales)].copy()
        if len(profiles) >= config.clusters:
            label = f"{start}-{end}"
            profiles["period"], profiles["period_start"], profiles["period_end"] = label, start, end
            period_profiles.append(profiles)
            labels.append(label)
    if len(period_profiles) < config.minimum_periods:
        raise ValueError(f"stability analysis requires at least {config.minimum_periods} eligible periods")
    features = _shared_features(period_profiles)
    if len(features) < 2:
        raise ValueError("stability analysis requires at least two features available in every period")
    assignments, centroids = _fit_periods(period_profiles, features, geography, config)
    assignments = _align_labels(assignments, centroids)
    transitions, persistence = _transitions(assignments, geography)
    consecutive_ari = []
    for previous, current in zip(labels, labels[1:]):
        joined = assignments.loc[assignments["period"].eq(previous), [geography, "segment"]].merge(
            assignments.loc[assignments["period"].eq(current), [geography, "segment"]],
            on=geography, suffixes=("_previous", "_current"), validate="one_to_one",
        )
        consecutive_ari.append({
            "from_period": previous, "to_period": current, "shared_neighborhoods": len(joined),
            "adjusted_rand_index": float(adjusted_rand_score(
                joined["segment_previous"], joined["segment_current"]
            )) if len(joined) >= 2 else None,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output_dir / "segment_history.parquet", index=False)
    transitions.to_csv(output_dir / "transition_matrix.csv", index=False)
    persistence.to_csv(output_dir / "neighborhood_persistence.csv", index=False)
    overall_persistence = persistence["persistence_rate"].mean()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "geography": geography, "periods": labels, "features": features,
        "neighborhood_period_observations": len(assignments),
        "overall_persistence_rate": float(overall_persistence) if pd.notna(overall_persistence) else None,
        "consecutive_adjusted_rand": consecutive_ari,
        "interpretation": "Persistence measures broadly similar regimes after centroid-based label alignment; transitions indicate relative market-regime movement, not causal change.",
    }
    (output_dir / "stability_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, default=Path("data/processed/spatial_features/core_sales_with_spatial_features.parquet"))
    parser.add_argument("--indices", type=Path, default=Path("data/processed/neighborhood_indices/neighborhood_price_indices.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/segmentation/stability"))
    parser.add_argument("--period-years", type=int, default=3)
    parser.add_argument("--minimum-neighborhood-sales", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()
    report = analyze_segment_stability(
        args.sales, args.indices, args.output,
        StabilityConfig(period_years=args.period_years, minimum_neighborhood_sales=args.minimum_neighborhood_sales, clusters=args.clusters),
    )
    persistence = report["overall_persistence_rate"]
    persistence_text = f"{persistence:.3f}" if persistence is not None else "not estimable"
    print(f"Analyzed {len(report['periods'])} periods; persistence={persistence_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
