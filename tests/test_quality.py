import json

import pandas as pd

from preprocessing.quality import QualityRules, audit_sales, build_quality_report


def quality_frame():
    return pd.DataFrame({
        "sale_id": ["a", "b", "c", "d"],
        "pin": ["1", "1", "2", "3"],
        "sale_date": ["2020-01-01", "2020-06-01", "2021-01-01", "2021-02-01"],
        "sale_price": [200_000, 210_000, None, 200_000_000],
        "doc_no": ["a", "b", "c", "d"],
        "year": [2020, 2020, 2021, 2021],
        "class": ["203", "204", "203", "203"],
        "building_sqft": [1200, 0, 1500, 100_000],
        "land_sqft": [4000, 4000, None, 5000],
        "bedrooms": [3, 3, None, 4],
        "bathrooms": [2, 2, None, 3],
        "building_age": [50, -1, None, 300],
        "latitude": [41.8, 41.8, None, 50],
        "longitude": [-87.7, -87.7, None, -90],
        "property_alignment_status": ["exact", "exact", "unmatched_no_history", "exact"],
        "parcel_alignment_status": ["exact", "exact", "unmatched_no_history", "exact"],
        "acs_alignment_status": ["historical", "historical", "unmatched_no_history", "historical"],
    })


def finding(report, name):
    return next(item for item in report["findings"] if item["check"] == name)


def test_audit_flags_quality_failures_without_dropping_rows():
    flagged, report = audit_sales(quality_frame(), QualityRules())
    assert len(flagged) == 4
    assert finding(report, "missing_sale_price")["count"] == 1
    assert finding(report, "implausible_sale_price")["count"] == 1
    assert finding(report, "nonpositive_building_area")["count"] == 1
    assert finding(report, "impossible_building_age")["count"] == 2
    assert finding(report, "rapid_resale")["count"] == 1
    assert finding(report, "property_class_inconsistency")["count"] == 2
    assert flagged["dq_issue_count"].gt(0).all()


def test_profiles_include_distribution_history_and_join_rates():
    _, report = audit_sales(quality_frame())
    price = report["feature_profiles"]["sale_price"]
    assert "distribution" in price
    assert price["historical_coverage"] == {"2020": 1.0, "2021": 0.5}
    assert report["join_success_rate"]["acs"] == 0.75


def test_profiles_boolean_features_as_counts_not_numeric_quantiles():
    frame = quality_frame().assign(
        source_flag=pd.array([True, False, None, True], dtype="boolean")
    )

    _, report = audit_sales(frame)

    profile = report["feature_profiles"]["source_flag"]
    assert profile["top_values"] == {"True": 2, "False": 1}
    assert "distribution" not in profile


def test_report_builder_writes_html_json_and_flags(tmp_path):
    source = tmp_path / "core.parquet"
    quality_frame().to_parquet(source, index=False)
    output = tmp_path / "quality"
    report = build_quality_report(source, output)
    assert report["rows"] == 4
    assert (output / "quality_flags.parquet").exists()
    assert "HomeValue Data Quality Report" in (output / "data_quality_report.html").read_text()
    parsed = json.loads((output / "data_quality_report.json").read_text())
    assert parsed["input_sha256"]


def test_missing_optional_feature_columns_are_reported_not_crashed():
    minimal = quality_frame().drop(columns=["building_sqft", "building_age", "latitude", "longitude"])
    _, report = audit_sales(minimal)
    assert finding(report, "missing_coordinates")["count"] == len(minimal)
    assert finding(report, "missing_structural_characteristics")["count"] == len(minimal)
