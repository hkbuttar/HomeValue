"""Serve one calibrated valuation with context, comparables, and value drivers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

from accessibility.cta import FEET_PER_MILE, PROJECTED_CRS, compute_cta_features
from api.schemas import ComparableSummary, ValuationRequest, ValuationResponse, ValueDriver
from explainability.property_place import _attributions, _feature_groups, _reference_row


@dataclass(frozen=True)
class EngineConfig:
    attribution_permutations: int = 32
    maximum_comparables: int = 5
    comparable_radius_miles: float = 2.0
    random_seed: int = 42


class HomeValueEngine:
    def __init__(
        self,
        model_artifact_path: Path,
        interval_report_path: Path,
        reference_sales_path: Path,
        cta_stations_path: Path | None = None,
        neighborhood_profiles_path: Path | None = None,
        config: EngineConfig | None = None,
    ):
        self.config = config or EngineConfig()
        self.artifact = joblib.load(model_artifact_path)
        self.interval_report = json.loads(interval_report_path.read_text(encoding="utf-8"))
        self.reference_sales = pd.read_parquet(reference_sales_path)
        self.cta_stations = (
            pd.read_parquet(cta_stations_path)
            if cta_stations_path and cta_stations_path.exists() else None
        )
        self.neighborhood_profiles = (
            pd.read_parquet(neighborhood_profiles_path)
            if neighborhood_profiles_path and neighborhood_profiles_path.exists() else None
        )
        self.features = self.artifact["features"]
        self.preprocessor = self.artifact["preprocessor"]
        self.model_name = self.artifact.get("selected_model") or next(iter(self.artifact["models"]))
        self.model = self.artifact["models"][self.model_name]
        self.smearing_factor = self.artifact.get("smearing_factors", {}).get(self.model_name, 1.0)
        calibration = self.interval_report.get("calibration", {}).get(self.model_name)
        if not calibration or calibration.get("log_residual_radius") is None:
            raise ValueError(f"interval report has no calibration for model {self.model_name}")
        self.interval_radius = float(calibration["log_residual_radius"])
        self.nominal_coverage = float(
            self.interval_report.get("config", {}).get("nominal_coverage", .90)
        )
        self.reference_profile = _reference_row(self.reference_sales, self.features)

    def _enrich(self, request: ValuationRequest) -> pd.DataFrame:
        values = request.model_dump()
        valuation_date = values.pop("valuation_date") or date.today()
        neighborhood = values.pop("neighborhood", None)
        values["year"], values["month"] = valuation_date.year, valuation_date.month
        values["quarter"] = (valuation_date.month - 1) // 3 + 1
        if neighborhood:
            for column in ("nbhd", "census_tract", "community_area"):
                if column in self.features:
                    values[column] = neighborhood
                    break
        row = pd.DataFrame([values])
        if self.cta_stations is not None and request.latitude is not None:
            cta = compute_cta_features(row, self.cta_stations).iloc[0]
            for column in cta.index:
                if column in self.features:
                    row[column] = cta[column]
        if self.neighborhood_profiles is not None and neighborhood:
            geography = next(
                (column for column in ("nbhd", "census_tract", "community_area")
                 if column in self.neighborhood_profiles), None
            )
            if geography:
                match = self.neighborhood_profiles.loc[
                    self.neighborhood_profiles[geography].astype(str).eq(str(neighborhood))
                ]
                if len(match):
                    for column in self.features:
                        if column in match and (column not in row or pd.isna(row.loc[0, column])):
                            row.loc[0, column] = match.iloc[0][column]
        return row.reindex(columns=self.features)

    def _comparables(self, request: ValuationRequest) -> list[ComparableSummary]:
        required = {"sale_id", "sale_date", "sale_price", "latitude", "longitude"}
        if request.latitude is None or not required.issubset(self.reference_sales.columns):
            return []
        sales = self.reference_sales.copy()
        sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
        cutoff = pd.Timestamp(request.valuation_date or date.today())
        valid = (
            sales["sale_date"].lt(cutoff)
            & pd.to_numeric(sales["sale_price"], errors="coerce").gt(0)
            & sales["latitude"].notna() & sales["longitude"].notna()
        )
        if request.pin and "pin" in sales:
            valid &= sales["pin"].astype(str).ne(request.pin)
        if request.residence_type and "residence_type" in sales:
            valid &= sales["residence_type"].astype(str).eq(request.residence_type)
        candidates = sales.loc[valid].copy()
        if candidates.empty:
            return []
        transformer = Transformer.from_crs("EPSG:4326", PROJECTED_CRS, always_xy=True)
        target_x, target_y = transformer.transform(request.longitude, request.latitude)
        x, y = transformer.transform(
            candidates["longitude"].to_numpy(float), candidates["latitude"].to_numpy(float)
        )
        candidates["distance_miles"] = np.hypot(x - target_x, y - target_y) / FEET_PER_MILE
        candidates = candidates.loc[
            candidates["distance_miles"].le(self.config.comparable_radius_miles)
        ].sort_values(["distance_miles", "sale_date"], ascending=[True, False]).head(
            self.config.maximum_comparables
        )
        return [ComparableSummary(
            sale_id=str(row.sale_id), sale_price=float(row.sale_price),
            distance_miles=float(row.distance_miles), sale_date=row.sale_date.date().isoformat(),
        ) for row in candidates.itertuples()]

    def predict(self, request: ValuationRequest | dict) -> ValuationResponse:
        request = request if isinstance(request, ValuationRequest) else ValuationRequest.model_validate(request)
        target = self._enrich(request)
        baseline, estimated, contributions = _attributions(
            target, self.reference_profile, self.features, self.preprocessor, self.model,
            self.config.attribution_permutations, np.random.default_rng(self.config.random_seed),
            self.smearing_factor,
        )
        groups = _feature_groups(self.features)
        grouped = pd.Series(contributions).groupby(pd.Series(groups)).sum().to_dict()
        drivers = [ValueDriver(
            component=groups[feature], feature=feature, dollar_contribution=float(value)
        ) for feature, value in sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)]
        return ValuationResponse(
            estimated_value=estimated,
            lower_interval=estimated * np.exp(-self.interval_radius),
            upper_interval=estimated * np.exp(self.interval_radius),
            baseline_market_value=baseline,
            property_component=float(grouped.get("property", 0)),
            location_component=float(grouped.get("place", 0)),
            time_market_component=float(grouped.get("time_market", 0)),
            other_component=float(grouped.get("other", 0)),
            confidence=self.nominal_coverage, model_name=self.model_name,
            value_drivers=drivers, comparables=self._comparables(request),
        )
