"""Fit and evaluate an interpretable log-price hedonic regression."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ml.baselines import regression_metrics, temporal_split


@dataclass(frozen=True)
class HedonicConfig:
    minimum_category_count: int = 20
    maximum_neighborhood_categories: int = 150
    robust_covariance: str = "HC3"
    include_time: bool = True
    include_property_type: bool = True
    include_neighborhood: bool = True
    include_accessibility: bool = False


ACCESSIBILITY_FEATURES = (
    "cta_distance", "cta_distance_miles", "lake_distance", "lake_distance_miles",
    "downtown_distance", "downtown_distance_miles", "park_distance",
    "park_distance_miles", "cta_stations_half_mile", "cta_stations_one_mile",
)


class HedonicModel:
    """OLS log-price model with train-fitted preprocessing and robust errors."""

    def __init__(self, config: HedonicConfig | None = None):
        self.config = config or HedonicConfig()
        self.numeric_sources_: dict[str, str] = {}
        self.imputations_: dict[str, float] = {}
        self.category_sources_: dict[str, str] = {}
        self.category_levels_: dict[str, list[str]] = {}
        self.design_columns_: list[str] = []
        self.year_origin_: int | None = None
        self.result_ = None
        self.smearing_factor_: float | None = None

    def _derived_numeric(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        def numeric(column: str) -> pd.Series:
            if column not in frame:
                return pd.Series(np.nan, index=frame.index, dtype="float64")
            return pd.to_numeric(frame[column], errors="coerce")

        sqft = numeric("building_sqft")
        land = numeric("land_sqft")
        derived = {
            "log_building_sqft": np.log(sqft.where(sqft.gt(0))),
            "log_land_sqft": np.log(land.where(land.gt(0))),
            "bedrooms": numeric("bedrooms"),
            "bathrooms": numeric("bathrooms"),
            "building_age": numeric("building_age"),
            "stories": numeric("stories"),
            "garage_spaces": numeric("garage_spaces"),
            "has_basement": frame.get(
                "has_basement", pd.Series(pd.NA, index=frame.index)
            ).astype("boolean").astype("Float64").astype(float),
        }
        if self.config.include_accessibility:
            for column in ACCESSIBILITY_FEATURES:
                if column not in frame:
                    continue
                values = numeric(column)
                if "distance" in column:
                    derived[f"log1p_{column}"] = np.log1p(values.where(values.ge(0)))
                else:
                    derived[column] = values
        return derived

    def _fit_schema(self, frame: pd.DataFrame) -> None:
        derived = self._derived_numeric(frame)
        self.numeric_sources_ = {
            name: name for name, values in derived.items() if values.notna().sum() >= 2
        }
        if "log_building_sqft" not in self.numeric_sources_:
            raise ValueError("hedonic model requires at least two valid building-square-foot values")
        self.imputations_ = {
            name: float(derived[name].median()) for name in self.numeric_sources_
        }
        if "year" in frame:
            years = pd.to_numeric(frame["year"], errors="coerce")
        else:
            years = pd.to_datetime(frame["sale_date"], errors="coerce").dt.year
        self.year_origin_ = int(years.dropna().min()) if self.config.include_time else 0

        property_column = next(
            (column for column in ("residence_type", "class") if column in frame), None
        )
        neighborhood_column = next(
            (column for column in ("nbhd", "census_tract") if column in frame), None
        )
        candidates = {}
        if self.config.include_property_type:
            candidates["property_type"] = property_column
        if self.config.include_neighborhood:
            candidates["neighborhood"] = neighborhood_column
        category_values = {}
        if self.config.include_time:
            sale_month = pd.to_datetime(frame["sale_date"], errors="coerce").dt.month.astype("Int64")
            category_values["sale_month"] = sale_month.astype("string")
        for feature, source in candidates.items():
            if source:
                category_values[feature] = frame[source].astype("string")
                self.category_sources_[feature] = source
        if self.config.include_time:
            self.category_sources_["sale_month"] = "__sale_month__"

        for feature, values in category_values.items():
            counts = values.dropna().value_counts()
            eligible = counts[counts >= self.config.minimum_category_count]
            if feature == "neighborhood":
                eligible = eligible.head(self.config.maximum_neighborhood_categories)
            if eligible.empty and not counts.empty:
                eligible = counts.head(1)
            # Most frequent level is the omitted reference category.
            self.category_levels_[feature] = [*eligible.index.astype(str).tolist(), "__OTHER__"]

    def _design(self, frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if fit:
            self._fit_schema(frame)
        if self.year_origin_ is None:
            raise RuntimeError("model schema is not fitted")
        derived = self._derived_numeric(frame)
        design = pd.DataFrame({"intercept": 1.0}, index=frame.index)
        for name in self.numeric_sources_:
            design[name] = derived[name].fillna(self.imputations_[name]).astype(float)
        if self.config.include_time:
            years = (
                pd.to_numeric(frame["year"], errors="coerce")
                if "year" in frame
                else pd.to_datetime(frame["sale_date"], errors="coerce").dt.year
            )
            design["year_trend"] = years.fillna(self.year_origin_) - self.year_origin_

        for feature, levels in self.category_levels_.items():
            if feature == "sale_month":
                values = pd.to_datetime(frame["sale_date"], errors="coerce").dt.month.astype("Int64").astype("string")
            else:
                source = self.category_sources_[feature]
                values = frame.get(source, pd.Series(pd.NA, index=frame.index)).astype("string")
            known = set(levels[:-1])
            values = values.where(values.isin(known), "__OTHER__")
            for level in levels[1:]:
                design[f"{feature}={level}"] = values.eq(level).astype(float)
        if fit:
            self.design_columns_ = design.columns.tolist()
        return design.reindex(columns=self.design_columns_, fill_value=0.0).astype(float)

    def fit(self, frame: pd.DataFrame) -> "HedonicModel":
        price = pd.to_numeric(frame["sale_price"], errors="coerce")
        valid = price.gt(0)
        training = frame.loc[valid].copy()
        target = np.log(price.loc[valid])
        design = self._design(training, fit=True)
        ordinary = sm.OLS(target, design, missing="raise").fit()
        self.result_ = ordinary.get_robustcov_results(cov_type=self.config.robust_covariance)
        residuals = target - ordinary.predict(design)
        self.smearing_factor_ = float(np.exp(residuals).mean())
        return self

    def predict_log(self, frame: pd.DataFrame) -> pd.Series:
        if self.result_ is None:
            raise RuntimeError("hedonic model must be fitted before prediction")
        values = self.result_.predict(self._design(frame))
        return pd.Series(values, index=frame.index, name="prediction_log_price")

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        log_prediction = self.predict_log(frame).clip(lower=0, upper=50)
        return (np.exp(log_prediction) * float(self.smearing_factor_)).rename("prediction_hedonic")

    def coefficient_table(self) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("hedonic model is not fitted")
        intervals = self.result_.conf_int()
        return pd.DataFrame({
            "term": self.design_columns_,
            "coefficient": self.result_.params,
            "robust_std_error": self.result_.bse,
            "t_value": self.result_.tvalues,
            "p_value": self.result_.pvalues,
            "ci_lower": intervals[:, 0],
            "ci_upper": intervals[:, 1],
        })


def _interpretations(coefficients: pd.DataFrame) -> dict[str, str]:
    values = coefficients.set_index("term")["coefficient"]
    interpretations = {}
    if "log_building_sqft" in values:
        interpretations["building_sqft"] = (
            f"A 1% increase in building area is associated with approximately "
            f"{values['log_building_sqft']:.3f}% change in price, holding modeled controls fixed."
        )
    for term in ("bedrooms", "bathrooms", "building_age", "garage_spaces"):
        if term in values:
            percent = 100 * (np.exp(values[term]) - 1)
            interpretations[term] = (
                f"One additional unit is associated with {percent:.2f}% change in price, "
                "holding modeled controls fixed."
            )
    return interpretations


def train_hedonic(
    input_path: Path,
    output_dir: Path,
    test_start_year: int | None = None,
    config: HedonicConfig | None = None,
) -> dict:
    frame = pd.read_parquet(input_path)
    price = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame = frame.loc[price.gt(0)].copy()
    train, test, cutoff = temporal_split(frame, test_start_year)
    model = HedonicModel(config).fit(train)
    dollar_prediction = model.predict(test)
    log_prediction = model.predict_log(test)
    coefficients = model.coefficient_table()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test[[column for column in ("sale_id", "pin", "sale_date", "year", "sale_price") if column in test]].copy()
    predictions["prediction_hedonic"] = dollar_prediction
    predictions["prediction_log_price"] = log_prediction
    predictions.to_parquet(output_dir / "hedonic_predictions.parquet", index=False)
    coefficients.to_csv(output_dir / "hedonic_coefficients.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "test_start_year": cutoff,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_years": sorted(pd.to_numeric(train["year"]).astype(int).unique().tolist()),
        "test_years": sorted(pd.to_numeric(test["year"]).astype(int).unique().tolist()),
        "config": asdict(model.config),
        "design_columns": model.design_columns_,
        "numeric_imputations": model.imputations_,
        "category_levels": model.category_levels_,
        "smearing_factor": model.smearing_factor_,
        "metrics_dollars": regression_metrics(test["sale_price"], dollar_prediction),
        "metrics_log_price": regression_metrics(np.log(test["sale_price"]), log_prediction),
        "fit_statistics": {
            "r_squared": float(model.result_.rsquared),
            "adjusted_r_squared": float(model.result_.rsquared_adj),
            "aic": float(model.result_.aic),
            "bic": float(model.result_.bic),
        },
        "interpretations": _interpretations(coefficients),
        "interpretation_caution": "Coefficients are hedonic associations, not automatically causal effects.",
    }
    (output_dir / "hedonic_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/core_sales.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/hedonic"))
    parser.add_argument("--test-start-year", type=int)
    parser.add_argument("--minimum-category-count", type=int, default=20)
    args = parser.parse_args()
    config = HedonicConfig(minimum_category_count=args.minimum_category_count)
    report = train_hedonic(args.input, args.output, args.test_start_year, config)
    print(f"Hedonic test MAE: ${report['metrics_dollars']['mae']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
