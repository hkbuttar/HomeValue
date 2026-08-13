"""Generate a formal quality audit for the canonical HomeValue sales table."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityRules:
    minimum_price: float = 10_000
    maximum_price: float = 100_000_000
    minimum_building_sqft: float = 200
    maximum_building_sqft: float = 20_000
    maximum_building_age: float = 250
    rapid_resale_days: int = 365
    cook_latitude_min: float = 41.45
    cook_latitude_max: float = 42.16
    cook_longitude_min: float = -88.27
    cook_longitude_max: float = -87.50


STRUCTURAL_FEATURES = (
    "building_sqft", "land_sqft", "bedrooms", "bathrooms", "building_age"
)


def _flag(frame: pd.DataFrame, mask: pd.Series, name: str) -> int:
    frame[f"dq_{name}"] = mask.fillna(False).astype(bool)
    return int(frame[f"dq_{name}"].sum())


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _finding(name: str, count: int, rows: int, severity: str, detail: str) -> dict:
    return {
        "check": name,
        "severity": severity,
        "count": count,
        "rate": count / rows if rows else 0.0,
        "detail": detail,
    }


def _feature_profile(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    series = frame[column]
    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "missing": int(series.isna().sum()),
        "missing_rate": float(series.isna().mean()),
        "unique": int(series.nunique(dropna=True)),
    }
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_bool_dtype(series):
        profile["top_values"] = {
            str(key): int(value)
            for key, value in series.astype("string").value_counts().items()
        }
    elif pd.api.types.is_numeric_dtype(series) and numeric.notna().any():
        quantiles = numeric.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
        profile["distribution"] = {
            "count": int(numeric.count()),
            "mean": float(numeric.mean()),
            "std": float(numeric.std()) if numeric.count() > 1 else None,
            **{f"p{int(q * 100):02d}": float(value) for q, value in quantiles.items()},
        }
    else:
        profile["top_values"] = {
            str(key): int(value) for key, value in series.astype("string").value_counts().head(10).items()
        }
    if "year" in frame:
        coverage = frame.assign(_present=series.notna()).groupby("year", dropna=False)["_present"].mean()
        profile["historical_coverage"] = {
            "unknown" if pd.isna(year) else str(int(year)): float(rate)
            for year, rate in coverage.items()
        }
    return profile


def audit_sales(sales: pd.DataFrame, rules: QualityRules | None = None) -> tuple[pd.DataFrame, dict]:
    """Run non-destructive quality checks and return flagged rows plus metrics."""
    rules = rules or QualityRules()
    required = {"sale_id", "pin", "sale_date", "sale_price"}
    if missing := sorted(required.difference(sales.columns)):
        raise ValueError(f"core sales table is missing: {', '.join(missing)}")
    frame = sales.copy()
    frame["sale_date"] = pd.to_datetime(frame["sale_date"], errors="coerce")
    price = pd.to_numeric(frame["sale_price"], errors="coerce")
    sqft = _numeric_column(frame, "building_sqft")
    age = _numeric_column(frame, "building_age")
    rows = len(frame)
    findings = []

    def add(name: str, mask: pd.Series, severity: str, detail: str) -> None:
        findings.append(_finding(name, _flag(frame, mask, name), rows, severity, detail))

    add("duplicate_sale_id", frame["sale_id"].duplicated(keep=False), "error", "Repeated canonical sale identifier.")
    transaction_columns = [column for column in ("pin", "sale_date", "sale_price", "doc_no") if column in frame]
    add("duplicate_transaction", frame.duplicated(transaction_columns, keep=False), "error", "Repeated transaction identity fields.")
    year = frame["sale_date"].dt.year
    add("duplicate_pin_year", frame.assign(_year=year).duplicated(["pin", "_year"], keep=False), "info", "Multiple sales for a PIN in one calendar year; may be valid.")
    add("missing_sale_price", price.isna(), "error", "Sale price is missing or nonnumeric.")
    add("implausible_sale_price", price.lt(rules.minimum_price) | price.gt(rules.maximum_price), "error", "Price is outside configured plausible bounds.")
    add("nonpositive_building_area", sqft.le(0), "error", "Building square footage is zero or negative.")

    positive_log = np.log(sqft.where(sqft.gt(0)))
    q1, q3 = positive_log.quantile([0.25, 0.75])
    iqr = q3 - q1
    statistical_sqft_outlier = (positive_log.lt(q1 - 1.5 * iqr) | positive_log.gt(q3 + 1.5 * iqr)) if pd.notna(iqr) else pd.Series(False, index=frame.index)
    hard_sqft_outlier = sqft.lt(rules.minimum_building_sqft) | sqft.gt(rules.maximum_building_sqft)
    add("building_sqft_outlier", statistical_sqft_outlier | hard_sqft_outlier, "warning", "Outside hard bounds or 1.5 IQR on log area.")
    add("impossible_building_age", age.lt(0) | age.gt(rules.maximum_building_age), "error", "Building age is negative or above configured maximum.")

    latitude = _numeric_column(frame, "latitude")
    longitude = _numeric_column(frame, "longitude")
    add("missing_coordinates", latitude.isna() | longitude.isna(), "warning", "Latitude or longitude is missing.")
    add("coordinates_outside_cook_county", latitude.notna() & longitude.notna() & (
        ~latitude.between(rules.cook_latitude_min, rules.cook_latitude_max)
        | ~longitude.between(rules.cook_longitude_min, rules.cook_longitude_max)
    ), "error", "Coordinates fall outside the configured Cook County bounding box.")

    property_failed = frame.get("property_alignment_status", pd.Series("", index=frame.index)).astype("string").str.startswith("unmatched")
    parcel_failed = frame.get("parcel_alignment_status", pd.Series("", index=frame.index)).astype("string").str.startswith("unmatched")
    acs_failed = frame.get("acs_alignment_status", pd.Series("", index=frame.index)).astype("string").str.startswith("unmatched")
    add("failed_property_join", property_failed, "warning", "No acceptable historical property snapshot.")
    add("failed_pin_geography_join", parcel_failed, "warning", "No acceptable historical parcel snapshot.")
    add("failed_acs_join", acs_failed, "warning", "No acceptable ACS tract snapshot.")

    ordered = frame[["pin", "sale_date"]].sort_values(["pin", "sale_date"])
    gaps = ordered.groupby("pin")["sale_date"].diff().dt.days
    rapid_index = ordered.index[gaps.between(0, rules.rapid_resale_days, inclusive="both")]
    rapid = pd.Series(frame.index.isin(rapid_index), index=frame.index)
    repeated = frame["pin"].duplicated(keep=False)
    add("repeated_sale_property", repeated, "info", "PIN appears in more than one retained sale.")
    add("rapid_resale", rapid, "warning", f"Previous retained sale occurred within {rules.rapid_resale_days} days.")

    if "class" in frame:
        class_counts = frame.assign(_year=year).groupby(["pin", "_year"], dropna=False)["class"].transform("nunique")
        add("property_class_inconsistency", class_counts.gt(1), "warning", "PIN has multiple property classes in one year.")
    else:
        add("property_class_inconsistency", pd.Series(False, index=frame.index), "warning", "Property class column is absent.")
    available_structural = [column for column in STRUCTURAL_FEATURES if column in frame]
    structural_missing = (
        pd.Series(True, index=frame.index)
        if len(available_structural) < len(STRUCTURAL_FEATURES)
        else frame[available_structural].isna().any(axis=1)
    )
    add("missing_structural_characteristics", structural_missing, "warning", "At least one retained structural feature is missing.")

    flag_columns = [column for column in frame if column.startswith("dq_")]
    frame["dq_issue_count"] = frame[flag_columns].sum(axis=1)
    profiles = {column: _feature_profile(frame, column) for column in sales.columns}
    join_success = {
        "property": float((~property_failed).mean()),
        "parcel": float((~parcel_failed).mean()),
        "acs": float((~acs_failed).mean()),
    }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rules": asdict(rules),
        "rows": rows,
        "findings": findings,
        "feature_profiles": profiles,
        "join_success_rate": join_success,
    }
    return frame, report


def _render_html(report: dict) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(finding[key]))}</td>" for key in ("check", "severity", "count", "rate", "detail")) + "</tr>"
        for finding in report["findings"]
    )
    features = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{profile['dtype']}</td><td>{profile['missing']}</td>"
        f"<td>{profile['missing_rate']:.2%}</td><td>{profile['unique']}</td></tr>"
        for name, profile in report["feature_profiles"].items()
    )
    joins = "".join(f"<li>{html.escape(name)}: {rate:.2%}</li>" for name, rate in report["join_success_rate"].items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>HomeValue Data Quality Report</title>
<style>body{{font:14px system-ui;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}th,td{{border:1px solid #ddd;padding:.45rem;text-align:left}}th{{background:#f3f4f6}}code{{background:#f3f4f6;padding:.1rem .3rem}}</style></head>
<body><h1>HomeValue Data Quality Report</h1><p>Generated {html.escape(report['created_at'])} for <strong>{report['rows']:,}</strong> canonical sales.</p>
<h2>Audit findings</h2><table><thead><tr><th>Check</th><th>Severity</th><th>Count</th><th>Rate</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Join success</h2><ul>{joins}</ul><h2>Feature coverage</h2><table><thead><tr><th>Feature</th><th>Type</th><th>Missing</th><th>Missing rate</th><th>Unique</th></tr></thead><tbody>{features}</tbody></table>
<p>Full distributions and historical coverage are available in <code>data_quality_report.json</code>.</p></body></html>"""


def build_quality_report(input_path: Path, output_dir: Path, rules: QualityRules | None = None) -> dict:
    sales = pd.read_parquet(input_path)
    flagged, report = audit_sales(sales, rules)
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_path = output_dir / "quality_flags.parquet"
    flagged.to_parquet(flagged_path, index=False)
    report["input_sha256"] = hashlib.sha256(input_path.read_bytes()).hexdigest()
    report["flags_sha256"] = hashlib.sha256(flagged_path.read_bytes()).hexdigest()
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "data_quality_report.html").write_text(_render_html(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/quality"))
    args = parser.parse_args()
    report = build_quality_report(args.input, args.output)
    total = sum(finding["count"] for finding in report["findings"] if finding["severity"] != "info")
    print(f"Audited {report['rows']} sales; found {total} non-informational flags")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
