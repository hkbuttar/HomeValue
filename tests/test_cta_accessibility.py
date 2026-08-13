import json

import pandas as pd

from accessibility.cta import build_cta_layer, compute_cta_features, extract_rail_stations


def write_gtfs(path):
    path.mkdir()
    pd.DataFrame([
        {"stop_id": "S1", "stop_name": "Station One", "stop_lat": 41.88, "stop_lon": -87.63, "location_type": "1", "parent_station": ""},
        {"stop_id": "P1", "stop_name": "Platform One", "stop_lat": 41.8801, "stop_lon": -87.6301, "location_type": "0", "parent_station": "S1"},
        {"stop_id": "S2", "stop_name": "Station Two", "stop_lat": 41.90, "stop_lon": -87.65, "location_type": "1", "parent_station": ""},
        {"stop_id": "P2", "stop_name": "Platform Two", "stop_lat": 41.9001, "stop_lon": -87.6501, "location_type": "0", "parent_station": "S2"},
        {"stop_id": "BUS", "stop_name": "Bus Stop", "stop_lat": 41.88, "stop_lon": -87.64, "location_type": "0", "parent_station": ""},
    ]).to_csv(path / "stops.txt", index=False)
    pd.DataFrame([
        {"route_id": "Red", "route_short_name": "Red", "route_long_name": "Red Line", "route_type": "1"},
        {"route_id": "Bus", "route_short_name": "1", "route_long_name": "Bus", "route_type": "3"},
    ]).to_csv(path / "routes.txt", index=False)
    pd.DataFrame([{"route_id": "Red", "trip_id": "T1"}, {"route_id": "Bus", "trip_id": "T2"}]).to_csv(path / "trips.txt", index=False)
    pd.DataFrame([
        {"trip_id": "T1", "stop_id": "P1"}, {"trip_id": "T1", "stop_id": "P2"},
        {"trip_id": "T2", "stop_id": "BUS"},
    ]).to_csv(path / "stop_times.txt", index=False)


def test_extracts_only_parent_rail_stations(tmp_path):
    gtfs = tmp_path / "gtfs"
    write_gtfs(gtfs)
    stations = extract_rail_stations(gtfs)
    assert stations["station_id"].tolist() == ["S1", "S2"]
    assert stations["lines"].tolist() == ["Red", "Red"]


def test_projected_nearest_distance_counts_and_missing_coordinates(tmp_path):
    gtfs = tmp_path / "gtfs"
    write_gtfs(gtfs)
    stations = extract_rail_stations(gtfs)
    properties = pd.DataFrame({
        "sale_id": ["a", "b"], "pin": ["1", "2"],
        "latitude": [41.88, None], "longitude": [-87.63, None],
    })
    result = compute_cta_features(properties, stations)
    assert result.loc[0, "cta_distance_miles"] < 0.01
    assert result.loc[0, "cta_stations_half_mile"] == 1
    assert result.loc[0, "nearest_cta_station_name"] == "Station One"
    assert result.loc[0, "nearest_cta_line"] == "Red"
    assert pd.isna(result.loc[1, "cta_distance_miles"])


def test_build_writes_stations_features_enriched_data_and_report(tmp_path):
    gtfs = tmp_path / "gtfs"
    write_gtfs(gtfs)
    source = tmp_path / "core.parquet"
    pd.DataFrame({
        "sale_id": ["a"], "pin": ["1"], "latitude": [41.88], "longitude": [-87.63]
    }).to_parquet(source, index=False)
    output = tmp_path / "cta"
    report = build_cta_layer(source, gtfs, output)
    assert report["rail_station_count"] == 2
    assert report["matched_sales"] == 1
    assert report["projected_crs"] == "EPSG:3435"
    assert (output / "core_sales_with_cta.parquet").exists()
    parsed = json.loads((output / "cta_accessibility_report.json").read_text())
    assert parsed["temporal_status"] == "current_network_snapshot"

