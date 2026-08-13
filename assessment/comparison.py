"""Compare aligned Cook County assessed values with sales and HomeValue estimates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.baselines import regression_metrics
from preprocessing.acquire import normalize_pin


@dataclass(frozen=True)
class AssessmentConfig:
    residential_assessment_ratio: float = 0.10
    minimum_group_size: int = 20
    price_groups: int = 10


STAGES = (("board_tot", "board_of_review"), ("certified_tot", "certified"), ("mailed_tot", "mailed"))


def prepare_assessments(frame: pd.DataFrame, config: AssessmentConfig | None = None) -> pd.DataFrame:
    config = config or AssessmentConfig()
    year_column = "year" if "year" in frame else "tax_year" if "tax_year" in frame else None
    if "pin" not in frame or year_column is None:
        raise ValueError("assessment data requires pin and year or tax_year")
    available = [(column, label) for column, label in STAGES if column in frame]
    if not available:
        raise ValueError("assessment data requires board_tot, certified_tot, or mailed_tot")
    result = frame.copy()
    result["pin"] = result["pin"].map(normalize_pin).astype("string")
    result["assessment_year"] = pd.to_numeric(result[year_column], errors="coerce").astype("Int64")
    result["assessed_total"] = np.nan
    result["assessment_stage"] = pd.Series(pd.NA, index=result.index, dtype="string")
    for column, label in available:
        value = pd.to_numeric(result[column], errors="coerce")
        use = result["assessed_total"].isna() & value.gt(0)
        result.loc[use, "assessed_total"] = value.loc[use]
        result.loc[use, "assessment_stage"] = label
    if "level_of_assessment" in result:
        ratio = pd.to_numeric(result["level_of_assessment"], errors="coerce")
        ratio = ratio.where(ratio.le(1), ratio / 100)
    else:
        ratio = pd.Series(config.residential_assessment_ratio, index=result.index)
    result["assessment_ratio"] = ratio
    result["assessed_market_value"] = result["assessed_total"] / ratio.where(ratio.gt(0))
    result = result.dropna(subset=["pin", "assessment_year", "assessed_market_value"])
    return result.sort_values("assessment_stage").drop_duplicates(["pin", "assessment_year"], keep="first")


def _segment_metrics(frame: pd.DataFrame, prediction_columns: list[str], config: AssessmentConfig) -> pd.DataFrame:
    data = frame.copy()
    data["price_group"] = pd.qcut(
        data["sale_price"], config.price_groups, labels=False, duplicates="drop"
    ).map(lambda value: f"D{int(value) + 1}" if pd.notna(value) else pd.NA)
    dimensions = {"sale_price_group": "price_group"}
    for column in ("nbhd", "census_tract", "municipality"):
        if column in data:
            dimensions[column] = column
    rows = []
    for prediction_column in prediction_columns:
        for dimension, column in dimensions.items():
            for segment, group in data.loc[data[column].notna()].groupby(column, observed=True):
                valid = pd.to_numeric(group[prediction_column], errors="coerce").notna()
                if not valid.any():
                    continue
                metrics = regression_metrics(group.loc[valid, "sale_price"], group.loc[valid, prediction_column])
                error = group.loc[valid, prediction_column] - group.loc[valid, "sale_price"]
                rows.append({
                    "model": prediction_column.removeprefix("prediction_"),
                    "dimension": dimension, "segment": str(segment), "n": int(valid.sum()),
                    "reliable_group": int(valid.sum()) >= config.minimum_group_size,
                    **metrics, "median_signed_percentage_error": float(
                        (error / group.loc[valid, "sale_price"]).median()
                    ),
                })
    return pd.DataFrame(rows)


def compare_assessments(
    sales_path: Path,
    assessments_path: Path,
    output_dir: Path,
    predictions_path: Path | None = None,
    config: AssessmentConfig | None = None,
) -> dict:
    config = config or AssessmentConfig()
    if not 0 < config.residential_assessment_ratio <= 1:
        raise ValueError("residential_assessment_ratio must be in (0, 1]")
    sales = pd.read_parquet(sales_path).copy()
    sales["pin"] = sales["pin"].map(normalize_pin).astype("string")
    if "year" not in sales:
        sales["year"] = pd.to_datetime(sales["sale_date"], errors="coerce").dt.year
    sales["year"] = pd.to_numeric(sales["year"], errors="coerce").astype("Int64")
    assessments = prepare_assessments(pd.read_parquet(assessments_path), config)
    matched = sales.merge(
        assessments[["pin", "assessment_year", "assessed_total", "assessment_stage", "assessment_ratio", "assessed_market_value"]],
        left_on=["pin", "year"], right_on=["pin", "assessment_year"],
        how="inner", validate="many_to_one",
    )
    matched["prediction_assessor"] = matched["assessed_market_value"]
    prediction_columns = ["prediction_assessor"]
    if predictions_path and predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        homevalue_columns = [column for column in predictions if column.startswith("prediction_")]
        if "sale_id" not in matched or "sale_id" not in predictions:
            raise ValueError("HomeValue comparison requires sale_id in sales and predictions")
        matched = matched.merge(
            predictions[["sale_id", *homevalue_columns]], on="sale_id", how="left",
            validate="one_to_one",
        )
        prediction_columns.extend(homevalue_columns)
    overall = {}
    for column in prediction_columns:
        valid = pd.to_numeric(matched[column], errors="coerce").notna()
        overall[column.removeprefix("prediction_")] = (
            regression_metrics(matched.loc[valid, "sale_price"], matched.loc[valid, column])
            if valid.any() else None
        )
    common = matched[prediction_columns].notna().all(axis=1)
    common_metrics = {
        column.removeprefix("prediction_"): regression_metrics(
            matched.loc[common, "sale_price"], matched.loc[common, column]
        ) for column in prediction_columns
    } if common.any() else {}
    segment_metrics = _segment_metrics(matched, prediction_columns, config)
    assessment_ratio = matched["assessed_market_value"] / matched["sale_price"]
    best_common = min(common_metrics, key=lambda name: common_metrics[name]["mae"]) if common_metrics else None
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output_dir / "matched_sales_assessments.parquet", index=False)
    segment_metrics.to_csv(output_dir / "assessment_error_by_segment.csv", index=False)
    pd.DataFrame([
        {"model": model, **metrics} for model, metrics in common_metrics.items()
    ]).to_csv(output_dir / "matched_model_comparison.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "sales_input": str(sales_path),
        "assessments_input": str(assessments_path),
        "predictions_input": str(predictions_path) if predictions_path else None,
        "config": asdict(config), "sales_rows": len(sales), "matched_rows": len(matched),
        "match_rate": float(len(matched) / len(sales)) if len(sales) else 0,
        "assessment_stage_counts": matched["assessment_stage"].value_counts().to_dict(),
        "overall_metrics": overall, "common_sample_rows": int(common.sum()),
        "common_sample_metrics": common_metrics, "best_common_sample_mae_model": best_common,
        "median_assessed_market_to_sale_ratio": float(assessment_ratio.median()),
        "conversion_note": "Source totals are assessed values, not market values. Market value equals assessed total divided by the row's level_of_assessment when supplied, otherwise the configured residential ratio.",
        "alignment_rule": "Sales match assessments on normalized 14-digit PIN and exact sale/tax year.",
    }
    (output_dir / "assessment_comparison_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, default=Path("data/processed/accessibility/core_sales_with_accessibility.parquet"))
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("data/processed/validation/out_of_time/final_test_predictions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/assessment_comparison"))
    parser.add_argument("--assessment-ratio", type=float, default=.10)
    args = parser.parse_args()
    report = compare_assessments(
        args.sales, args.assessments, args.output, args.predictions,
        AssessmentConfig(residential_assessment_ratio=args.assessment_ratio),
    )
    print(f"Matched {report['matched_rows']} sales; best common-sample MAE: {report['best_common_sample_mae_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
