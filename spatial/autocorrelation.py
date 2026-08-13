"""Test spatial autocorrelation in prices, PPSF, and hedonic residuals."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda import Moran
from libpysal.weights import DistanceBand, KNN, Queen
from pyproj import Transformer

from accessibility.cta import FEET_PER_MILE, PROJECTED_CRS
from hedonic.model import HedonicConfig, HedonicModel


@dataclass(frozen=True)
class SpatialAuditConfig:
    k_neighbors: int = 8
    distance_band_miles: float = 1.0
    permutations: int = 999
    random_seed: int = 42
    maximum_observations: int = 25_000
    minimum_category_count: int = 20


def prepare_spatial_sample(
    sales: pd.DataFrame,
    config: SpatialAuditConfig,
    analysis_year: int | None = None,
) -> tuple[pd.DataFrame, int]:
    """Create a single-year, one-sale-per-PIN spatial cross-section."""
    required = {"sale_id", "pin", "sale_date", "sale_price", "building_sqft", "latitude", "longitude"}
    if missing := sorted(required.difference(sales.columns)):
        raise ValueError(f"spatial audit input is missing: {', '.join(missing)}")
    frame = sales.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    frame["year"] = frame["sale_date"].dt.year.astype("Int64")
    year = int(analysis_year if analysis_year is not None else frame["year"].dropna().max())
    frame = frame.loc[frame["year"].eq(year)].copy()
    for column in ("sale_price", "building_sqft", "latitude", "longitude"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame["sale_price"].gt(0) & frame["building_sqft"].gt(0)
        & frame["latitude"].between(-90, 90) & frame["longitude"].between(-180, 180)
    ].sort_values("sale_date").drop_duplicates("pin", keep="last")
    if len(frame) < 3:
        raise ValueError(f"analysis year {year} has fewer than three valid spatial sales")
    if len(frame) > config.maximum_observations:
        frame = frame.sample(config.maximum_observations, random_state=config.random_seed).sort_index()
    frame["price_per_sqft"] = frame["sale_price"] / frame["building_sqft"]
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    x, y = transformer.transform(frame["longitude"].to_numpy(), frame["latitude"].to_numpy())
    frame["x_3435"] = x
    frame["y_3435"] = y
    return frame.reset_index(drop=True), year


def build_point_weights(sample: pd.DataFrame, config: SpatialAuditConfig) -> dict:
    coordinates = sample[["x_3435", "y_3435"]].to_numpy(float)
    k = min(config.k_neighbors, len(sample) - 1)
    if k < 1:
        raise ValueError("at least two observations are required for spatial weights")
    weights = {
        f"knn_{k}": KNN.from_array(coordinates, k=k),
        f"distance_{config.distance_band_miles:g}_mile": DistanceBand.from_array(
            coordinates,
            threshold=config.distance_band_miles * FEET_PER_MILE,
            binary=True,
            silence_warnings=True,
        ),
    }
    for item in weights.values():
        item.transform = "r"
    return weights


def _moran(values: pd.Series, weights, config: SpatialAuditConfig) -> dict:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(array).all():
        raise ValueError("Moran's I requires complete finite values")
    state = np.random.get_state()
    np.random.seed(config.random_seed)
    try:
        statistic = Moran(array, weights, permutations=config.permutations)
    finally:
        np.random.set_state(state)
    return {
        "moran_i": float(statistic.I),
        "expected_i": float(statistic.EI),
        "p_normal": float(statistic.p_norm),
        "p_permutation": float(statistic.p_sim),
        "z_permutation": float(statistic.z_sim),
        "permutations": config.permutations,
    }


def _hedonic_residuals(sample: pd.DataFrame, minimum_category_count: int) -> tuple[pd.Series, dict]:
    model = HedonicModel(HedonicConfig(
        minimum_category_count=minimum_category_count,
        include_time=False,
        include_property_type=True,
        include_neighborhood=True,
        include_accessibility=True,
    )).fit(sample)
    actual = np.log(pd.to_numeric(sample["sale_price"], errors="coerce"))
    residual = actual - model.predict_log(sample)
    return residual.rename("hedonic_log_residual"), {
        "r_squared": float(model.result_.rsquared),
        "adjusted_r_squared": float(model.result_.rsquared_adj),
        "design_columns": model.design_columns_,
    }


def _tract_weights(
    sample: pd.DataFrame,
    polygons_path: Path,
    variables: list[str],
) -> tuple[pd.DataFrame, object] | None:
    tract_column = next((column for column in ("census_tract", "geoid") if column in sample), None)
    if tract_column is None:
        return None
    polygons = gpd.read_file(polygons_path)
    polygon_id = next(
        (column for column in ("geoid", "GEOID", "geoid10", "tractce") if column in polygons), None
    )
    if polygon_id is None:
        raise ValueError("tract polygons require a GEOID-like column")
    aggregated = sample.groupby(tract_column)[variables].median().reset_index()
    aggregated[tract_column] = aggregated[tract_column].astype("string").str.zfill(11)
    polygons[polygon_id] = polygons[polygon_id].astype("string").str.zfill(11)
    joined = polygons.merge(aggregated, left_on=polygon_id, right_on=tract_column, how="inner")
    if len(joined) < 3:
        return None
    weights = Queen.from_dataframe(joined, ids=joined[polygon_id].tolist(), silence_warnings=True)
    weights.transform = "r"
    return pd.DataFrame(joined.drop(columns="geometry")), weights


def _scatter(sample: pd.DataFrame, weights, variable: str, output: Path) -> None:
    values = sample[variable].to_numpy(float)
    standardized = (values - values.mean()) / values.std()
    lag = weights.sparse @ standardized
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(standardized, lag, alpha=0.45, s=18)
    slope = np.polyfit(standardized, lag, 1)
    x = np.linspace(standardized.min(), standardized.max(), 100)
    axis.plot(x, slope[0] * x + slope[1], color="#b91c1c")
    axis.axhline(0, color="black", linewidth=0.6)
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set(xlabel="Standardized value", ylabel="Spatial lag", title=f"Moran scatter: {variable}")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run_spatial_audit(
    input_path: Path,
    output_dir: Path,
    analysis_year: int | None = None,
    tract_polygons: Path | None = None,
    config: SpatialAuditConfig | None = None,
) -> dict:
    config = config or SpatialAuditConfig()
    sample, year = prepare_spatial_sample(pd.read_parquet(input_path), config, analysis_year)
    residuals, hedonic_fit = _hedonic_residuals(sample, config.minimum_category_count)
    sample["hedonic_log_residual"] = residuals
    variables = ["sale_price", "price_per_sqft", "hedonic_log_residual"]
    weights = build_point_weights(sample, config)
    records = []
    weight_metadata = {}
    for name, spatial_weights in weights.items():
        islands = list(spatial_weights.islands)
        weight_metadata[name] = {
            "observations": spatial_weights.n,
            "island_count": len(islands),
            "islands": [int(value) if isinstance(value, (int, np.integer)) else str(value) for value in islands[:100]],
        }
        for variable in variables:
            records.append({"weights": name, "unit": "sale", "variable": variable, **_moran(sample[variable], spatial_weights, config)})

    tract_status = "not_requested"
    if tract_polygons is not None:
        tract = _tract_weights(sample, tract_polygons, variables)
        if tract is None:
            tract_status = "insufficient_matched_tracts"
        else:
            tract_frame, tract_weights = tract
            tract_status = "fitted"
            weight_metadata["tract_queen"] = {
                "observations": tract_weights.n,
                "island_count": len(tract_weights.islands),
            }
            for variable in variables:
                records.append({"weights": "tract_queen", "unit": "tract_median", "variable": variable, **_moran(tract_frame[variable], tract_weights, config)})

    results = pd.DataFrame(records)
    residual_tests = results.loc[results["variable"].eq("hedonic_log_residual")]
    residual_structure = bool(
        (residual_tests["moran_i"].gt(0) & residual_tests["p_permutation"].lt(0.05)).any()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_dir / "spatial_audit_sample.parquet", index=False)
    results.to_csv(output_dir / "morans_i_results.csv", index=False)
    first_weights = next(iter(weights.values()))
    for variable in variables:
        _scatter(sample, first_weights, variable, output_dir / f"moran_scatter_{variable}.png")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "analysis_year": year,
        "sample_rows": len(sample),
        "config": asdict(config),
        "projected_crs": PROJECTED_CRS,
        "weights": weight_metadata,
        "tract_adjacency_status": tract_status,
        "hedonic_fit": hedonic_fit,
        "residual_spatial_structure_detected": residual_structure,
        "conclusion": (
            "Positive residual spatial autocorrelation remains after observed controls; spatial econometric models are justified."
            if residual_structure else
            "No positive residual spatial autocorrelation was detected at the 5% permutation threshold for the tested weights."
        ),
    }
    (output_dir / "spatial_autocorrelation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/spatial_autocorrelation"))
    parser.add_argument("--analysis-year", type=int)
    parser.add_argument("--tract-polygons", type=Path)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--distance-band-miles", type=float, default=1.0)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    config = SpatialAuditConfig(
        k_neighbors=args.k_neighbors,
        distance_band_miles=args.distance_band_miles,
        permutations=args.permutations,
    )
    report = run_spatial_audit(
        args.input, args.output, args.analysis_year, args.tract_polygons, config
    )
    print(report["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

