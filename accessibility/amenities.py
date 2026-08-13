"""Acquire official amenity geometries and build projected distance features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import distance, points
from shapely.geometry import shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from accessibility.cta import FEET_PER_MILE, PROJECTED_CRS

CHICAGO_DOMAIN = "https://data.cityofchicago.org"
SOURCES = {
    "parks": {"id": "ejsh-fztr", "select": "park_no,park,acres,the_geom"},
    "hydro": {"id": "knfe-65pw", "select": "display,the_geom"},
}
DOWNTOWN_REFERENCE = {
    "name": "Chicago City Hall downtown reference",
    "longitude": -87.6325,
    "latitude": 41.8837,
}


def _fetch(url: str) -> bytes:
    headers = {"User-Agent": "HomeValue/0.1 (public-data research)"}
    if token := os.getenv("CHICAGO_DATA_APP_TOKEN"):
        headers["X-App-Token"] = token
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        return response.read()


def acquire_amenities(
    output_dir: Path,
    fetch: Callable[[str], bytes] = _fetch,
) -> dict:
    """Download bounded GeoJSON extracts from official Chicago datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    urls = {}
    for name, source in SOURCES.items():
        query = urllib.parse.urlencode({"$select": source["select"], "$limit": 50_000})
        url = f"{CHICAGO_DOMAIN}/resource/{source['id']}.geojson?{query}"
        payload = fetch(url)
        # Validate before persisting an HTML error page as spatial data.
        parsed = json.loads(payload)
        if parsed.get("type") != "FeatureCollection":
            raise ValueError(f"{name} endpoint did not return a GeoJSON FeatureCollection")
        destination = output_dir / f"{name}.geojson"
        destination.write_bytes(payload)
        files[name] = {
            "path": destination.name,
            "features": len(parsed.get("features", [])),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        urls[name] = url
    downtown_payload = json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": DOWNTOWN_REFERENCE["name"]},
            "geometry": {
                "type": "Point",
                "coordinates": [DOWNTOWN_REFERENCE["longitude"], DOWNTOWN_REFERENCE["latitude"]],
            },
        }],
    }).encode()
    downtown_path = output_dir / "downtown.geojson"
    downtown_path.write_bytes(downtown_payload)
    files["downtown"] = {
        "path": downtown_path.name,
        "features": 1,
        "sha256": hashlib.sha256(downtown_payload).hexdigest(),
    }
    urls["downtown"] = "configured Chicago City Hall reference point"
    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sources": urls,
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _features(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError(f"invalid GeoJSON FeatureCollection: {path}")
    return payload.get("features", [])


def _project_geometry(geometry, transformer: Transformer):
    return transform(transformer.transform, geometry)


def load_amenity_geometries(raw_dir: Path, minimum_park_acres: float = 10.0) -> dict:
    """Load and project major parks, Lake Michigan, and the official CBD."""
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    parks, park_names = [], []
    for feature in _features(raw_dir / "parks.geojson"):
        properties = feature.get("properties", {})
        acres = pd.to_numeric(properties.get("acres"), errors="coerce")
        if pd.isna(acres) or acres < minimum_park_acres or not feature.get("geometry"):
            continue
        parks.append(_project_geometry(shape(feature["geometry"]), transformer))
        park_names.append(str(properties.get("park") or properties.get("park_no") or "Unknown park"))
    if not parks:
        raise ValueError("no park geometries meet the configured acreage threshold")

    hydro_candidates = []
    for feature in _features(raw_dir / "hydro.geojson"):
        if not feature.get("geometry"):
            continue
        hydro_candidates.append(_project_geometry(shape(feature["geometry"]), transformer))
    if not hydro_candidates:
        raise ValueError("hydro data contains no waterbody geometries")
    # In the official Hydro layer, Lake Michigan is the largest displayed polygon.
    lake = max(hydro_candidates, key=lambda geometry: geometry.area)

    downtown_parts = [
        _project_geometry(shape(feature["geometry"]), transformer)
        for feature in _features(raw_dir / "downtown.geojson") if feature.get("geometry")
    ]
    if not downtown_parts:
        raise ValueError("central business district data contains no geometry")
    downtown = unary_union(downtown_parts)
    return {
        "parks": parks,
        "park_names": park_names,
        "lake": lake,
        "downtown": downtown,
        "minimum_park_acres": minimum_park_acres,
    }


def compute_amenity_features(properties: pd.DataFrame, amenities: dict) -> pd.DataFrame:
    """Compute vectorized projected distances in miles."""
    if missing := sorted({"latitude", "longitude"}.difference(properties.columns)):
        raise ValueError(f"properties are missing coordinates: {', '.join(missing)}")
    transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
    latitude = pd.to_numeric(properties["latitude"], errors="coerce")
    longitude = pd.to_numeric(properties["longitude"], errors="coerce")
    valid = latitude.notna() & longitude.notna() & latitude.between(-90, 90) & longitude.between(-180, 180)
    result = pd.DataFrame(index=properties.index)
    for identifier in ("sale_id", "pin"):
        if identifier in properties:
            result[identifier] = properties[identifier]
    for column in ("lake_distance_miles", "downtown_distance_miles", "park_distance_miles"):
        result[column] = np.nan
    result["nearest_major_park"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if valid.any():
        x, y = transformer.transform(
            longitude.loc[valid].to_numpy(float), latitude.loc[valid].to_numpy(float)
        )
        property_points = points(x, y)
        result.loc[valid, "lake_distance_miles"] = distance(property_points, amenities["lake"]) / FEET_PER_MILE
        result.loc[valid, "downtown_distance_miles"] = distance(
            property_points, amenities["downtown"].centroid
        ) / FEET_PER_MILE
        tree = STRtree(amenities["parks"])
        indices, distances = tree.query_nearest(property_points, return_distance=True)
        input_positions, park_positions = indices
        valid_index = result.index[valid].to_numpy()
        result.loc[valid_index[input_positions], "park_distance_miles"] = distances / FEET_PER_MILE
        names = np.asarray(amenities["park_names"], dtype=object)[park_positions]
        result.loc[valid_index[input_positions], "nearest_major_park"] = names
    result["amenity_feature_temporal_status"] = "current_geometry_snapshot"
    return result


def build_amenity_layer(
    input_path: Path,
    raw_dir: Path,
    output_dir: Path,
    minimum_park_acres: float = 10.0,
) -> dict:
    properties = pd.read_parquet(input_path)
    if "sale_id" in properties and properties["sale_id"].duplicated().any():
        raise ValueError("input must contain at most one row per sale_id")
    amenities = load_amenity_geometries(raw_dir, minimum_park_acres)
    features = compute_amenity_features(properties, amenities)
    new_columns = [column for column in features if column not in {"sale_id", "pin"}]
    enriched = properties.drop(columns=[column for column in new_columns if column in properties]).merge(
        features.drop(columns=["pin"], errors="ignore"), on="sale_id", how="left", validate="one_to_one"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_dir / "amenity_accessibility_features.parquet", index=False)
    enriched.to_parquet(output_dir / "core_sales_with_accessibility.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "projected_crs": PROJECTED_CRS,
        "distance_unit": "miles",
        "major_park_minimum_acres": minimum_park_acres,
        "major_park_count": len(amenities["parks"]),
        "sales": len(features),
        "matched_sales": int(features["park_distance_miles"].notna().sum()),
        "match_rate": float(features["park_distance_miles"].notna().mean()),
        "lake_selection": "largest polygon in City of Chicago Hydro layer",
        "downtown_reference": DOWNTOWN_REFERENCE["name"],
        "temporal_status": "current_geometry_snapshot",
    }
    (output_dir / "amenity_accessibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--output", type=Path, default=Path("data/raw/chicago_amenities"))
    build = subparsers.add_parser("build")
    build.add_argument(
        "--input", type=Path,
        default=Path("data/processed/cta_accessibility/core_sales_with_cta.parquet"),
    )
    build.add_argument("--raw", type=Path, default=Path("data/raw/chicago_amenities"))
    build.add_argument("--output", type=Path, default=Path("data/processed/accessibility"))
    build.add_argument("--minimum-park-acres", type=float, default=10.0)
    args = parser.parse_args()
    if args.command == "acquire":
        manifest = acquire_amenities(args.output)
        print(f"Downloaded {len(manifest['files'])} amenity layers")
    else:
        report = build_amenity_layer(args.input, args.raw, args.output, args.minimum_park_acres)
        print(f"Matched {report['matched_sales']} sales to {report['major_park_count']} major parks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
