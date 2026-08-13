import json

import pandas as pd

from accessibility.amenities import (
    acquire_amenities,
    build_amenity_layer,
    compute_amenity_features,
    load_amenity_geometries,
)


def feature(properties, coordinates):
    return {"type": "Feature", "properties": properties, "geometry": {"type": "Polygon", "coordinates": [coordinates]}}


def write_geojson(path, name, features):
    (path / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def amenity_files(path):
    path.mkdir()
    park = [[-87.64, 41.87], [-87.62, 41.87], [-87.62, 41.89], [-87.64, 41.89], [-87.64, 41.87]]
    small = [[-87.70, 41.80], [-87.699, 41.80], [-87.699, 41.801], [-87.70, 41.801], [-87.70, 41.80]]
    lake = [[-87.61, 41.70], [-87.50, 41.70], [-87.50, 42.05], [-87.61, 42.05], [-87.61, 41.70]]
    pond = [[-87.7, 41.8], [-87.69, 41.8], [-87.69, 41.81], [-87.7, 41.81], [-87.7, 41.8]]
    cbd = [[-87.64, 41.87], [-87.62, 41.87], [-87.62, 41.89], [-87.64, 41.89], [-87.64, 41.87]]
    write_geojson(path, "parks", [feature({"park": "Major Park", "acres": 20}, park), feature({"park": "Tiny Park", "acres": 1}, small)])
    write_geojson(path, "hydro", [feature({"display": 1}, lake), feature({"display": 1}, pond)])
    write_geojson(path, "downtown", [feature({"name": "CBD"}, cbd)])


def test_acquisition_validates_and_manifests_geojson(tmp_path):
    payload = json.dumps({"type": "FeatureCollection", "features": []}).encode()
    calls = []
    manifest = acquire_amenities(tmp_path, fetch=lambda url: calls.append(url) or payload)
    assert len(calls) == 2
    assert set(manifest["files"]) == {"parks", "hydro", "downtown"}
    assert (tmp_path / "manifest.json").exists()


def test_projected_distances_major_park_filter_and_missing_coordinates(tmp_path):
    raw = tmp_path / "raw"
    amenity_files(raw)
    amenities = load_amenity_geometries(raw, minimum_park_acres=10)
    assert amenities["park_names"] == ["Major Park"]
    properties = pd.DataFrame({
        "sale_id": ["a", "b"], "pin": ["1", "2"],
        "latitude": [41.88, None], "longitude": [-87.63, None],
    })
    result = compute_amenity_features(properties, amenities)
    assert result.loc[0, "park_distance_miles"] == 0
    assert result.loc[0, "nearest_major_park"] == "Major Park"
    assert result.loc[0, "downtown_distance_miles"] < 0.1
    assert result.loc[0, "lake_distance_miles"] > 0
    assert pd.isna(result.loc[1, "lake_distance_miles"])


def test_build_writes_features_enriched_table_and_report(tmp_path):
    raw = tmp_path / "raw"
    amenity_files(raw)
    source = tmp_path / "core.parquet"
    pd.DataFrame({
        "sale_id": ["a"], "pin": ["1"], "latitude": [41.88], "longitude": [-87.63]
    }).to_parquet(source, index=False)
    output = tmp_path / "out"
    report = build_amenity_layer(source, raw, output)
    assert report["major_park_count"] == 1
    assert report["matched_sales"] == 1
    assert (output / "core_sales_with_accessibility.parquet").exists()
    parsed = json.loads((output / "amenity_accessibility_report.json").read_text())
    assert parsed["temporal_status"] == "current_geometry_snapshot"
