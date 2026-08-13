"""Cluster neighborhood housing markets and validate the resulting archetypes."""

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
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class SegmentationConfig:
    minimum_neighborhood_sales: int = 20
    minimum_clusters: int = 2
    maximum_clusters: int = 8
    bootstrap_iterations: int = 50
    bootstrap_fraction: float = 0.8
    random_seed: int = 42


PROFILE_FEATURES = (
    "median_sale_price", "median_ppsf", "annual_appreciation", "price_volatility",
    "annual_sale_velocity", "single_family_share", "median_building_sqft",
    "median_household_income", "owner_occupancy_rate", "population_density",
    "cta_distance_miles", "transit_commute_share",
)


def _geography(frame: pd.DataFrame) -> str:
    for column in ("nbhd", "census_tract", "community_area", "municipality"):
        if column in frame:
            return column
    raise ValueError("segmentation input has no neighborhood geography")


def build_market_profiles(sales: pd.DataFrame, indices: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str]:
    geography = _geography(sales)
    frame = sales.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    frame["sale_price"] = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame["building_sqft"] = pd.to_numeric(frame["building_sqft"], errors="coerce")
    frame = frame.loc[
        frame[geography].notna() & frame["sale_date"].notna()
        & frame["sale_price"].gt(0) & frame["building_sqft"].gt(0)
    ].copy()
    frame[geography] = frame[geography].astype("string")
    frame["year"] = frame["sale_date"].dt.year
    frame["price_per_sqft"] = frame["sale_price"] / frame["building_sqft"]
    property_column = next((column for column in ("residence_type", "class") if column in frame), None)
    if property_column:
        values = frame[property_column].astype("string").str.lower()
        frame["_single_family"] = (
            values.str.contains("single|story|town", regex=True, na=False)
            | frame.get("class", pd.Series("", index=frame.index)).astype("string").isin(
                ["202", "203", "204", "205", "206", "207", "208", "209", "210", "234", "278", "295"]
            )
        )
    else:
        frame["_single_family"] = np.nan
    contextual_features = (
        "median_household_income", "owner_occupancy_rate", "population_density",
        "cta_distance_miles", "transit_commute_share",
    )
    for column in contextual_features:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = frame.groupby(geography, observed=True)
    profiles = grouped.agg(
        sale_count=("sale_id", "size"),
        first_sale_year=("year", "min"), latest_sale_year=("year", "max"),
        median_sale_price=("sale_price", "median"), median_ppsf=("price_per_sqft", "median"),
        price_volatility=("price_per_sqft", lambda values: np.log(values).std()),
        median_building_sqft=("building_sqft", "median"),
        single_family_share=("_single_family", "mean"),
    ).reset_index()
    years_observed = (profiles["latest_sale_year"] - profiles["first_sale_year"] + 1).clip(lower=1)
    profiles["annual_sale_velocity"] = profiles["sale_count"] / years_observed
    for source in contextual_features:
        if source in frame:
            values = grouped[source].median().rename(source).reset_index()
            profiles = profiles.merge(values, on=geography, how="left", validate="one_to_one")
    if indices is not None and len(indices):
        index_geography = geography if geography in indices else _geography(indices)
        index = indices.copy()
        index[index_geography] = index[index_geography].astype("string")
        index = index.sort_values("year")
        level_column = next(
            (column for column in ("hedonic_adjusted_index", "median_ppsf_index") if column in index), None
        )
        if level_column:
            def growth(group):
                valid = group.dropna(subset=[level_column])
                if len(valid) < 2:
                    return np.nan
                elapsed = max(1, int(valid["year"].iloc[-1] - valid["year"].iloc[0]))
                return (valid[level_column].iloc[-1] / valid[level_column].iloc[0]) ** (1 / elapsed) - 1
            appreciation = index.groupby(index_geography, observed=True).apply(
                growth, include_groups=False
            ).rename("annual_appreciation").reset_index()
            profiles = profiles.merge(
                appreciation, left_on=geography, right_on=index_geography,
                how="left", validate="one_to_one",
            ).drop(columns=[index_geography] if index_geography != geography else [])
    return profiles, geography


def _matrix(profiles: pd.DataFrame) -> tuple[np.ndarray, list[str], SimpleImputer, RobustScaler]:
    features = [
        column for column in PROFILE_FEATURES
        if column in profiles and pd.to_numeric(profiles[column], errors="coerce").notna().sum() >= 2
    ]
    if len(features) < 2:
        raise ValueError("segmentation requires at least two usable market-profile features")
    numeric = profiles[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(numeric))
    return matrix, features, imputer, scaler


def _choose_clusters(matrix: np.ndarray, config: SegmentationConfig) -> tuple[int, dict[int, float]]:
    distinct_profiles = len(np.unique(matrix, axis=0))
    maximum = min(config.maximum_clusters, len(matrix) - 1, distinct_profiles)
    minimum = min(config.minimum_clusters, maximum)
    if maximum < 2:
        raise ValueError("segmentation requires at least three eligible neighborhoods")
    scores = {}
    for clusters in range(minimum, maximum + 1):
        labels = KMeans(
            n_clusters=clusters, random_state=config.random_seed, n_init=20
        ).fit_predict(matrix)
        scores[clusters] = float(silhouette_score(matrix, labels))
    selected = max(scores, key=lambda value: (scores[value], -value))
    return selected, scores


def _bootstrap_stability(matrix: np.ndarray, labels: np.ndarray, clusters: int,
                         config: SegmentationConfig) -> dict:
    rng = np.random.default_rng(config.random_seed)
    scores = []
    sample_size = max(clusters + 1, int(len(matrix) * config.bootstrap_fraction))
    for iteration in range(config.bootstrap_iterations):
        sampled = np.sort(rng.choice(len(matrix), size=sample_size, replace=False))
        boot_labels = KMeans(
            n_clusters=clusters, random_state=config.random_seed + iteration + 1, n_init=10
        ).fit_predict(matrix[sampled])
        scores.append(float(adjusted_rand_score(labels[sampled], boot_labels)))
    return {
        "iterations": len(scores), "mean_adjusted_rand": float(np.mean(scores)),
        "median_adjusted_rand": float(np.median(scores)),
        "minimum_adjusted_rand": float(np.min(scores)),
    }


def _archetype_names(profiles: pd.DataFrame, features: list[str]) -> dict[int, str]:
    numeric = profiles[features].apply(pd.to_numeric, errors="coerce")
    standardized = (numeric - numeric.mean()) / numeric.std().replace(0, np.nan)
    centroids = standardized.assign(cluster=profiles["cluster"]).groupby("cluster").mean()
    names = {}
    used = set()
    for cluster, row in centroids.iterrows():
        high = row.dropna().sort_values(ascending=False)
        low = row.dropna().sort_values()
        descriptors = []
        if "median_sale_price" in row and row["median_sale_price"] > 0.5:
            descriptors.append("High-Value")
        elif "median_sale_price" in row and row["median_sale_price"] < -0.5:
            descriptors.append("Affordable")
        if "annual_appreciation" in row and row["annual_appreciation"] > 0.4:
            descriptors.append("Growth")
        if "transit_commute_share" in row and row["transit_commute_share"] > 0.4:
            descriptors.append("Transit-Oriented")
        elif "cta_distance_miles" in row and row["cta_distance_miles"] > 0.5:
            descriptors.append("Transit-Distant")
        if "annual_sale_velocity" in row and row["annual_sale_velocity"] < -0.5:
            descriptors.append("Low-Turnover")
        if "owner_occupancy_rate" in row and row["owner_occupancy_rate"] > 0.5:
            descriptors.append("Owner-Occupied")
        if not descriptors:
            lead = high.index[0] if len(high) else low.index[0]
            descriptors.append(lead.replace("_", " ").title())
        base = " ".join(descriptors[:3])
        name = base
        suffix = 2
        while name in used:
            name = f"{base} {suffix}"
            suffix += 1
        names[int(cluster)] = name
        used.add(name)
    return names


def _plot(profiles: pd.DataFrame, output: Path) -> None:
    x = "median_sale_price"
    y = "annual_appreciation" if "annual_appreciation" in profiles else "median_ppsf"
    figure, axis = plt.subplots(figsize=(9, 6))
    for name, group in profiles.groupby("archetype", observed=True):
        axis.scatter(group[x], group[y], label=name, alpha=0.75)
    axis.set(xlabel=x.replace("_", " ").title(), ylabel=y.replace("_", " ").title(),
             title="Neighborhood market archetypes")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def segment_neighborhoods(
    sales_path: Path,
    indices_path: Path | None,
    output_dir: Path,
    config: SegmentationConfig | None = None,
) -> dict:
    config = config or SegmentationConfig()
    sales = pd.read_parquet(sales_path)
    indices = pd.read_parquet(indices_path) if indices_path and indices_path.exists() else None
    profiles, geography = build_market_profiles(sales, indices)
    profiles = profiles.loc[profiles["sale_count"].ge(config.minimum_neighborhood_sales)].copy()
    matrix, features, imputer, scaler = _matrix(profiles)
    clusters, silhouette_scores = _choose_clusters(matrix, config)
    model = KMeans(n_clusters=clusters, random_state=config.random_seed, n_init=50).fit(matrix)
    profiles["cluster"] = model.labels_
    names = _archetype_names(profiles, features)
    profiles["archetype"] = profiles["cluster"].map(names)
    stability = _bootstrap_stability(matrix, model.labels_, clusters, config)
    cluster_profiles = profiles.groupby(["cluster", "archetype"], observed=True)[features].median().reset_index()
    cluster_profiles["neighborhood_count"] = cluster_profiles["cluster"].map(
        profiles["cluster"].value_counts()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_parquet(output_dir / "neighborhood_segments.parquet", index=False)
    cluster_profiles.to_csv(output_dir / "archetype_profiles.csv", index=False)
    _plot(profiles, output_dir / "neighborhood_archetypes.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sales_input": str(sales_path), "indices_input": str(indices_path) if indices_path else None,
        "geography": geography, "config": asdict(config),
        "eligible_neighborhoods": len(profiles), "features": features,
        "selected_clusters": clusters,
        "silhouette_scores": {str(key): value for key, value in silhouette_scores.items()},
        "selected_silhouette": silhouette_scores[clusters],
        "bootstrap_stability": stability, "archetype_names": names,
        "naming_rule": "Names were generated after fitting from relative cluster profiles.",
        "caution": "Archetypes summarize this feature set and period; they are not intrinsic neighborhood identities.",
    }
    (output_dir / "segmentation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sales", type=Path,
        default=Path("data/processed/spatial_features/core_sales_with_spatial_features.parquet"),
    )
    parser.add_argument(
        "--indices", type=Path,
        default=Path("data/processed/neighborhood_indices/neighborhood_price_indices.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/segmentation"))
    parser.add_argument("--minimum-neighborhood-sales", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=50)
    args = parser.parse_args()
    config = SegmentationConfig(
        minimum_neighborhood_sales=args.minimum_neighborhood_sales,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    report = segment_neighborhoods(args.sales, args.indices, args.output, config)
    print(f"Selected {report['selected_clusters']} neighborhood archetypes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
