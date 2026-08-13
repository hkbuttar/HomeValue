"""Train and evaluate simple, leakage-safe valuation benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class MedianBaseline:
    """Training-only median estimator with a global fallback."""

    name: str
    group_columns: tuple[str, ...] = ()
    use_price_per_sqft: bool = False
    target: str = "sale_price"
    sqft_column: str = "building_sqft"
    global_price_: float | None = None
    global_rate_: float | None = None
    medians_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "MedianBaseline":
        if self.target not in frame:
            raise ValueError(f"training data is missing {self.target}")
        target = pd.to_numeric(frame[self.target], errors="coerce")
        valid_price = target.gt(0) & target.notna()
        if not valid_price.any():
            raise ValueError("training data contains no positive sale prices")
        self.global_price_ = float(target.loc[valid_price].median())
        values = target.copy()
        if self.use_price_per_sqft:
            if self.sqft_column not in frame:
                raise ValueError(f"training data is missing {self.sqft_column}")
            sqft = pd.to_numeric(frame[self.sqft_column], errors="coerce")
            valid_rate = valid_price & sqft.gt(0)
            rates = target.loc[valid_rate] / sqft.loc[valid_rate]
            if rates.empty:
                raise ValueError("training data contains no valid price-per-square-foot rows")
            self.global_rate_ = float(rates.median())
            values = target / sqft.where(sqft.gt(0))
        if self.group_columns:
            missing = sorted(set(self.group_columns).difference(frame.columns))
            if missing:
                raise ValueError(f"training data is missing grouping columns: {', '.join(missing)}")
            working = frame.loc[valid_price, list(self.group_columns)].copy()
            working["_value"] = values.loc[valid_price]
            working = working.dropna(subset=[*self.group_columns, "_value"])
            self.medians_ = working.groupby(list(self.group_columns), observed=True)["_value"].median()
        else:
            self.medians_ = None
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.global_price_ is None:
            raise RuntimeError("baseline must be fitted before prediction")
        prediction = pd.Series(self.global_price_, index=frame.index, dtype="float64")
        if self.group_columns and self.medians_ is not None and len(self.medians_):
            if len(self.group_columns) == 1:
                grouped = frame[self.group_columns[0]].map(self.medians_)
            else:
                keys = pd.MultiIndex.from_frame(frame[list(self.group_columns)])
                grouped = pd.Series(self.medians_.reindex(keys).to_numpy(), index=frame.index)
            if self.use_price_per_sqft:
                sqft = (
                    pd.to_numeric(frame[self.sqft_column], errors="coerce")
                    if self.sqft_column in frame
                    else pd.Series(np.nan, index=frame.index)
                )
                grouped = grouped.fillna(float(self.global_rate_)) * sqft.where(sqft.gt(0))
            prediction = grouped.fillna(prediction)
        elif self.use_price_per_sqft:
            sqft = (
                pd.to_numeric(frame[self.sqft_column], errors="coerce")
                if self.sqft_column in frame
                else pd.Series(np.nan, index=frame.index)
            )
            rate_prediction = sqft.where(sqft.gt(0)) * float(self.global_rate_)
            prediction = rate_prediction.fillna(prediction)
        return prediction.clip(lower=0)

    def artifact(self) -> dict:
        groups = []
        if self.medians_ is not None:
            for key, value in self.medians_.items():
                key_values = key if isinstance(key, tuple) else (key,)
                groups.append({
                    **{column: str(item) for column, item in zip(self.group_columns, key_values)},
                    "median": float(value),
                })
        return {
            "name": self.name,
            "group_columns": list(self.group_columns),
            "use_price_per_sqft": self.use_price_per_sqft,
            "global_price": self.global_price_,
            "global_price_per_sqft": self.global_rate_,
            "group_medians": groups,
        }


def temporal_split(frame: pd.DataFrame, test_start_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    data = frame.copy()
    if "year" not in data:
        if "sale_date" not in data:
            raise ValueError("data requires year or sale_date for temporal validation")
        data["year"] = pd.to_datetime(data["sale_date"], errors="coerce").dt.year
    years = sorted(pd.to_numeric(data["year"], errors="coerce").dropna().astype(int).unique())
    if len(years) < 2:
        raise ValueError("temporal validation requires at least two sale years")
    cutoff = int(test_start_year if test_start_year is not None else years[-1])
    train = data.loc[pd.to_numeric(data["year"], errors="coerce").lt(cutoff)].copy()
    test = data.loc[pd.to_numeric(data["year"], errors="coerce").ge(cutoff)].copy()
    if train.empty or test.empty:
        raise ValueError(f"cutoff {cutoff} must leave nonempty train and test sets")
    return train, test, cutoff


def regression_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int | None]:
    actual_values = np.asarray(list(actual), dtype=float)
    predicted_values = np.asarray(list(predicted), dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values, predicted_values = actual_values[valid], predicted_values[valid]
    if not len(actual_values):
        raise ValueError("no valid prediction pairs")
    errors = predicted_values - actual_values
    nonzero = actual_values != 0
    denominator = np.sum((actual_values - actual_values.mean()) ** 2)
    return {
        "n": int(len(actual_values)),
        "mae": float(np.mean(np.abs(errors))),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "median_absolute_percentage_error": (
            float(np.median(np.abs(errors[nonzero] / actual_values[nonzero])))
            if nonzero.any() else None
        ),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "mape": float(np.mean(np.abs(errors[nonzero] / actual_values[nonzero]))) if nonzero.any() else None,
        "r2": float(1 - np.sum(errors**2) / denominator) if denominator > 0 else None,
    }


def make_baselines(frame: pd.DataFrame) -> list[MedianBaseline]:
    property_column = "residence_type" if "residence_type" in frame else "class"
    neighborhood_column = "nbhd" if "nbhd" in frame else "census_tract"
    missing = [name for name, column in (("property type", property_column), ("neighborhood", neighborhood_column)) if column not in frame]
    if missing:
        raise ValueError(f"cannot build baseline grouping for: {', '.join(missing)}")
    return [
        MedianBaseline("global_median"),
        MedianBaseline("property_type_median", (property_column,)),
        MedianBaseline("neighborhood_median", (neighborhood_column,)),
        MedianBaseline("segmented_ppsf", (neighborhood_column, property_column), use_price_per_sqft=True),
    ]


def train_baselines(
    input_path: Path,
    output_dir: Path,
    test_start_year: int | None = None,
) -> dict:
    frame = pd.read_parquet(input_path)
    price = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame = frame.loc[price.gt(0)].copy()
    train, test, cutoff = temporal_split(frame, test_start_year)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[[column for column in ("sale_id", "pin", "sale_date", "year", "sale_price") if column in test]].copy()
    results = {}
    artifacts = {}
    for model in make_baselines(frame):
        model.fit(train)
        prediction = model.predict(test)
        predictions[f"prediction_{model.name}"] = prediction
        results[model.name] = regression_metrics(test["sale_price"], prediction)
        artifacts[model.name] = model.artifact()
    predictions.to_parquet(output_dir / "baseline_predictions.parquet", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "test_start_year": cutoff,
        "train_years": sorted(pd.to_numeric(train["year"]).astype(int).unique().tolist()),
        "test_years": sorted(pd.to_numeric(test["year"]).astype(int).unique().tolist()),
        "train_rows": len(train),
        "test_rows": len(test),
        "metrics": results,
        "models": artifacts,
    }
    (output_dir / "baseline_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/baselines"))
    parser.add_argument("--test-start-year", type=int)
    args = parser.parse_args()
    report = train_baselines(args.input, args.output, args.test_start_year)
    best = min(report["metrics"], key=lambda name: report["metrics"][name]["mae"])
    print(f"Evaluated {len(report['metrics'])} baselines; lowest test MAE: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
