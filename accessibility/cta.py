"""Build property-level CTA rail accessibility features from official GTFS."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors

PROJECTED_CRS = "EPSG:3435"  # NAD83 / Illinois East (US survey feet)
FEET_PER_MILE = 5280.0


def _read_gtfs(path: Path, filename: str, columns: list[str]) -> pd.DataFrame:
    file = path / filename
    if not file.exists():
        raise FileNotFoundError(f"CTA GTFS is missing {filename}")
    return pd.read_csv(file, usecols=columns, dtype="string")


def extract_rail_stations(gtfs_path: Path) -> pd.DataFrame:
    """Return one parent station per CTA rail complex with served lines."""
    stops = _read_gtfs(
        gtfs_path, "stops.txt",
        ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
    )
    routes = _read_gtfs(
        gtfs_path, "routes.txt", ["route_id", "route_short_name", "route_long_name", "route_type"]
    )
    trips = _read_gtfs(gtfs_path, "trips.txt", ["route_id", "trip_id"])
    stop_times = _read_gtfs(gtfs_path, "stop_times.txt", ["trip_id", "stop_id"])
    rail_routes = routes.loc[pd.to_numeric(routes["route_type"], errors="coerce").eq(1)].copy()
    if rail_routes.empty:
        raise ValueError("GTFS contains no route_type=1 rail routes")
    rail_trips = trips.merge(rail_routes[["route_id"]], on="route_id", how="inner")
    served = stop_times.merge(rail_trips, on="trip_id", how="inner")[["stop_id", "route_id"]].drop_duplicates()
    stop_lookup = stops.set_index("stop_id")
    parent = stops.set_index("stop_id")["parent_station"].replace("", pd.NA)
    served["station_id"] = served["stop_id"].map(parent).fillna(served["stop_id"])
    route_names = rail_routes.set_index("route_id")["route_short_name"].fillna(
        rail_routes.set_index("route_id")["route_long_name"]
    )
    served["line"] = served["route_id"].map(route_names).fillna(served["route_id"])
    line_lookup = served.groupby("station_id")["line"].agg(
        lambda values: "|".join(sorted(set(values.dropna().astype(str))))
    )

    records = []
    for station_id, platforms in served.groupby("station_id", sort=True):
        if station_id in stop_lookup.index:
            station = stop_lookup.loc[station_id]
            if isinstance(station, pd.DataFrame):
                station = station.iloc[0]
            latitude = pd.to_numeric(station["stop_lat"], errors="coerce")
            longitude = pd.to_numeric(station["stop_lon"], errors="coerce")
            name = station["stop_name"]
        else:
            platform_rows = stops.loc[stops["stop_id"].isin(platforms["stop_id"])]
            latitude = pd.to_numeric(platform_rows["stop_lat"], errors="coerce").mean()
            longitude = pd.to_numeric(platform_rows["stop_lon"], errors="coerce").mean()
            name = platform_rows["stop_name"].dropna().iloc[0] if platform_rows["stop_name"].notna().any() else station_id
        records.append({
            "station_id": station_id,
            "station_name": name,
            "latitude": latitude,
            "longitude": longitude,
            "lines": line_lookup.get(station_id, ""),
        })
    stations = pd.DataFrame(records).dropna(subset=["latitude", "longitude"])
    if stations.empty:
        raise ValueError("no CTA rail stations have valid coordinates")
    return stations.reset_index(drop=True)


def compute_cta_features(properties: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Compute projected nearest-station and radius-count features."""
    required = {"latitude", "longitude"}
    if missing := sorted(required.difference(properties.columns)):
        raise ValueError(f"properties are missing coordinates: {', '.join(missing)}")
    station_required = {"station_id", "station_name", "latitude", "longitude", "lines"}
    if missing := sorted(station_required.difference(stations.columns)):
        raise ValueError(f"stations are missing: {', '.join(missing)}")
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    station_x, station_y = transformer.transform(
        stations["longitude"].to_numpy(float), stations["latitude"].to_numpy(float)
    )
    station_points = np.column_stack([station_x, station_y])
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(station_points)

    result = pd.DataFrame(index=properties.index)
    for identifier in ("sale_id", "pin"):
        if identifier in properties:
            result[identifier] = properties[identifier]
    latitude = pd.to_numeric(properties["latitude"], errors="coerce")
    longitude = pd.to_numeric(properties["longitude"], errors="coerce")
    valid = latitude.notna() & longitude.notna() & latitude.between(-90, 90) & longitude.between(-180, 180)
    result["cta_distance_miles"] = np.nan
    result["cta_stations_half_mile"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["cta_stations_one_mile"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["nearest_cta_station_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["nearest_cta_station_name"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["nearest_cta_line"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if valid.any():
        property_x, property_y = transformer.transform(
            longitude.loc[valid].to_numpy(float), latitude.loc[valid].to_numpy(float)
        )
        points = np.column_stack([property_x, property_y])
        distances, indices = neighbors.kneighbors(points)
        nearest = indices[:, 0]
        result.loc[valid, "cta_distance_miles"] = distances[:, 0] / FEET_PER_MILE
        result.loc[valid, "cta_stations_half_mile"] = [
            len(items) for items in neighbors.radius_neighbors(points, radius=0.5 * FEET_PER_MILE, return_distance=False)
        ]
        result.loc[valid, "cta_stations_one_mile"] = [
            len(items) for items in neighbors.radius_neighbors(points, radius=FEET_PER_MILE, return_distance=False)
        ]
        result.loc[valid, "nearest_cta_station_id"] = stations.iloc[nearest]["station_id"].to_numpy()
        result.loc[valid, "nearest_cta_station_name"] = stations.iloc[nearest]["station_name"].to_numpy()
        result.loc[valid, "nearest_cta_line"] = stations.iloc[nearest]["lines"].to_numpy()
    result["cta_feature_temporal_status"] = "current_network_snapshot"
    return result


def build_cta_layer(input_path: Path, gtfs_path: Path, output_dir: Path) -> dict:
    properties = pd.read_parquet(input_path)
    if "sale_id" in properties and properties["sale_id"].duplicated().any():
        raise ValueError("input must contain at most one row per sale_id")
    stations = extract_rail_stations(gtfs_path)
    features = compute_cta_features(properties, stations)
    feature_columns = [column for column in features if column not in {"sale_id", "pin"}]
    enriched = properties.drop(columns=[column for column in feature_columns if column in properties]).merge(
        features.drop(columns=["pin"], errors="ignore"), on="sale_id", how="left", validate="one_to_one"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(output_dir / "cta_rail_stations.parquet", index=False)
    features.to_parquet(output_dir / "cta_accessibility_features.parquet", index=False)
    enriched.to_parquet(output_dir / "core_sales_with_cta.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "projected_crs": PROJECTED_CRS,
        "distance_unit": "miles",
        "rail_station_count": len(stations),
        "sales": len(features),
        "matched_sales": int(features["cta_distance_miles"].notna().sum()),
        "match_rate": float(features["cta_distance_miles"].notna().mean()),
        "temporal_status": "current_network_snapshot",
        "temporal_caution": "Current CTA infrastructure is not a historical station snapshot for older sales.",
    }
    (output_dir / "cta_accessibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--gtfs", type=Path, default=Path("data/raw/cta_gtfs"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cta_accessibility"))
    args = parser.parse_args()
    report = build_cta_layer(args.input, args.gtfs, args.output)
    print(f"Matched {report['matched_sales']} sales to {report['rail_station_count']} CTA stations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

