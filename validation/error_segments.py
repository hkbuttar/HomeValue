"""Audit held-out valuation errors across market and property segments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ErrorSegmentConfig:
    minimum_group_size: int = 20
    price_deciles: int = 10
    urban_density_threshold: float = 10_000.0


def _merge_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if "sale_id" not in predictions or "sale_id" not in features:
        raise ValueError("predictions and feature data require sale_id")
    prediction_columns = [column for column in predictions if column.startswith("prediction_")]
    if not prediction_columns:
        raise ValueError("predictions contain no prediction_ columns")
    extra = [column for column in features if column not in predictions and column != "sale_price"]
    return predictions.merge(
        features[["sale_id", *extra]], on="sale_id", how="left", validate="one_to_one"
    )


def _add_segments(frame: pd.DataFrame, config: ErrorSegmentConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    data = frame.copy()
    price = pd.to_numeric(data["sale_price"], errors="coerce")
    try:
        data["_price_decile"] = pd.qcut(
            price, q=config.price_deciles, labels=False, duplicates="drop"
        ).map(lambda value: f"D{int(value) + 1}" if pd.notna(value) else pd.NA)
    except ValueError:
        data["_price_decile"] = pd.Series(pd.NA, index=data.index, dtype="string")
    dimensions = {"sale_price_decile": "_price_decile"}
    property_type = next((column for column in ("residence_type", "class") if column in data), None)
    neighborhood = next((column for column in ("nbhd", "census_tract", "community_area") if column in data), None)
    if property_type:
        dimensions["property_type"] = property_type
    if neighborhood:
        dimensions["neighborhood"] = neighborhood
    if "municipality" in data:
        dimensions["municipality"] = "municipality"
    if "building_age" in data:
        age = pd.to_numeric(data["building_age"], errors="coerce")
        data["_building_age_group"] = pd.cut(
            age, [-np.inf, 10, 30, 60, 100, np.inf],
            labels=["0-10", "11-30", "31-60", "61-100", "100+"],
        )
        dimensions["building_age"] = "_building_age_group"
    if "cta_distance_miles" in data:
        distance = pd.to_numeric(data["cta_distance_miles"], errors="coerce")
        data["_transit_distance_group"] = pd.cut(
            distance, [-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=["within_0.5_miles", "0.5-1_mile", "1-2_miles", "over_2_miles"],
        )
        dimensions["distance_to_transit"] = "_transit_distance_group"
    if "year" not in data and "sale_date" in data:
        data["year"] = pd.to_datetime(data["sale_date"], errors="coerce").dt.year
    if "year" in data:
        dimensions["time_period"] = "year"
    if "archetype" in data:
        dimensions["market_archetype"] = "archetype"
    if "population_density" in data:
        density = pd.to_numeric(data["population_density"], errors="coerce")
        data["_urban_context"] = np.where(
            density.isna(), pd.NA,
            np.where(density.ge(config.urban_density_threshold), "urban", "suburban"),
        )
        dimensions["urban_suburban_context"] = "_urban_context"
    elif "municipality" in data:
        municipality = data["municipality"].astype("string").str.lower()
        data["_urban_context"] = np.where(municipality.eq("chicago"), "urban", "suburban")
        dimensions["urban_suburban_context"] = "_urban_context"
    return data, dimensions


def _summaries(frame: pd.DataFrame, dimensions: dict[str, str], minimum: int) -> pd.DataFrame:
    rows = []
    actual = pd.to_numeric(frame["sale_price"], errors="coerce")
    for prediction_column in [column for column in frame if column.startswith("prediction_")]:
        predicted = pd.to_numeric(frame[prediction_column], errors="coerce")
        working = frame.copy()
        working["_error"] = predicted - actual
        working["_absolute_error"] = working["_error"].abs()
        working["_ape"] = working["_absolute_error"] / actual.where(actual.gt(0))
        for dimension, column in dimensions.items():
            valid = working.loc[working[column].notna() & working["_ape"].notna()]
            for segment, group in valid.groupby(column, observed=True):
                count = len(group)
                rows.append({
                    "model": prediction_column.removeprefix("prediction_"),
                    "dimension": dimension, "segment": str(segment), "n": count,
                    "reliable_group": count >= minimum,
                    "mae": float(group["_absolute_error"].mean()),
                    "rmse": float(np.sqrt(np.mean(group["_error"] ** 2))),
                    "median_ape": float(group["_ape"].median()),
                    "mean_ape": float(group["_ape"].mean()),
                    "median_signed_percentage_error": float((group["_error"] / actual.loc[group.index]).median()),
                })
    return pd.DataFrame(rows)


def analyze_error_segments(
    predictions_path: Path,
    features_path: Path,
    output_dir: Path,
    segments_path: Path | None = None,
    config: ErrorSegmentConfig | None = None,
) -> dict:
    config = config or ErrorSegmentConfig()
    if config.minimum_group_size < 1:
        raise ValueError("minimum_group_size must be positive")
    predictions, features = pd.read_parquet(predictions_path), pd.read_parquet(features_path)
    frame = _merge_features(predictions, features)
    if segments_path and segments_path.exists():
        segments = pd.read_parquet(segments_path)
        geography = next((column for column in ("nbhd", "census_tract", "community_area") if column in frame and column in segments), None)
        if geography and "archetype" in segments:
            frame = frame.merge(
                segments[[geography, "archetype"]].drop_duplicates(geography),
                on=geography, how="left", validate="many_to_one",
            )
    frame, dimensions = _add_segments(frame, config)
    metrics = _summaries(frame, dimensions, config.minimum_group_size)
    if metrics.empty:
        raise ValueError("no valid segment error summaries could be calculated")
    reliable = metrics.loc[metrics["reliable_group"]].copy()
    worst = (
        reliable.sort_values(["model", "median_ape"], ascending=[True, False])
        .groupby("model", observed=True).head(10)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "segment_error_metrics.csv", index=False)
    worst.to_csv(output_dir / "worst_reliable_segments.csv", index=False)
    frame.to_parquet(output_dir / "predictions_with_segments.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "predictions_input": str(predictions_path), "features_input": str(features_path),
        "segments_input": str(segments_path) if segments_path else None,
        "config": asdict(config), "models": sorted(metrics["model"].unique().tolist()),
        "dimensions": list(dimensions), "summary_rows": len(metrics),
        "reliable_summary_rows": len(reliable),
        "worst_reliable_segments": worst.to_dict(orient="records"),
        "percentage_error_note": "APE uses positive observed sale price as its denominator. Median APE is emphasized; groups below the minimum size are retained but excluded from worst-segment rankings.",
    }
    (output_dir / "error_segment_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/validation/out_of_time/final_test_predictions.parquet"))
    parser.add_argument("--features", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--segments", type=Path, default=Path("data/processed/segmentation/neighborhood_segments.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation/error_segments"))
    parser.add_argument("--minimum-group-size", type=int, default=20)
    args = parser.parse_args()
    report = analyze_error_segments(
        args.predictions, args.features, args.output, args.segments,
        ErrorSegmentConfig(minimum_group_size=args.minimum_group_size),
    )
    print(f"Audited {len(report['models'])} models across {len(report['dimensions'])} dimensions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
