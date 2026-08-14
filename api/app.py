"""FastAPI application exposing HomeValue valuation and research artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    MarketOverviewResponse,
    NeighborhoodDetailResponse,
    RecordCollection,
    ResearchResponse,
    ValuationRequest,
    ValuationResponse,
)
from engine import HomeValueEngine
from preprocessing.acquire import normalize_pin


@dataclass(frozen=True)
class APISettings:
    data_root: Path = Path("data/processed")
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    @classmethod
    def from_environment(cls):
        root = Path(os.getenv("HOMEVALUE_DATA_ROOT", "data/processed"))
        origins = tuple(
            value.strip() for value in os.getenv(
                "HOMEVALUE_CORS_ORIGINS", "http://localhost:3000"
            ).split(",") if value.strip()
        )
        return cls(data_root=root, cors_origins=origins)

    def path(self, relative: str) -> Path:
        return self.data_root / relative


class ArtifactRepository:
    def __init__(self, settings: APISettings):
        self.settings = settings

    def parquet(self, relative: str) -> pd.DataFrame:
        path = self.settings.path(relative)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact unavailable: {relative}")
        return pd.read_parquet(path)

    def csv(self, relative: str) -> pd.DataFrame:
        path = self.settings.path(relative)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact unavailable: {relative}")
        return pd.read_csv(path)

    def json(self, relative: str) -> dict:
        path = self.settings.path(relative)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact unavailable: {relative}")
        return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    return [_clean(record) for record in frame.to_dict(orient="records")]


def _geography(frame: pd.DataFrame) -> str:
    for column in ("nbhd", "census_tract", "community_area", "municipality"):
        if column in frame:
            return column
    raise HTTPException(status_code=500, detail="Neighborhood artifact has no geography field")


def _neighborhood_name_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "nbhd" not in frame:
        raise HTTPException(status_code=500, detail="Valuation artifact has no neighborhood field")
    working = frame.loc[frame["nbhd"].notna()].copy()
    working["neighborhood_id"] = working["nbhd"].astype("string")
    place = pd.Series(pd.NA, index=working.index, dtype="string")
    for column in ("community_area", "municipality"):
        if column in working:
            candidate = working[column].astype("string").str.strip()
            place = place.fillna(candidate.where(candidate.str.len().gt(0)))
    working["place_name"] = place.fillna("Cook County")
    counts = (
        working.groupby(["neighborhood_id", "place_name"], observed=True)
        .size().rename("sale_count").reset_index()
        .sort_values(["neighborhood_id", "sale_count", "place_name"], ascending=[True, False, True])
    )
    primary = counts.drop_duplicates("neighborhood_id").copy()
    primary["name"] = primary["place_name"].str.title()
    primary["label"] = primary["name"] + " — area " + primary["neighborhood_id"]
    return primary.sort_values(["name", "neighborhood_id"])[
        ["neighborhood_id", "name", "label", "sale_count"]
    ]


def _neighborhood_options(frame: pd.DataFrame) -> list[dict]:
    return _records(_neighborhood_name_frame(frame))


def _build_engine(settings: APISettings) -> HomeValueEngine:
    optional_stations = settings.path("cta_accessibility/cta_rail_stations.parquet")
    optional_profiles = settings.path("segmentation/neighborhood_segments.parquet")
    return HomeValueEngine(
        settings.path("validation/out_of_time/final_models.joblib"),
        settings.path("validation/intervals/interval_results.json"),
        settings.path("accessibility/core_sales_with_accessibility.parquet"),
        optional_stations, optional_profiles,
    )


def create_app(settings: APISettings | None = None, engine: HomeValueEngine | None = None) -> FastAPI:
    settings = settings or APISettings.from_environment()
    repository = ArtifactRepository(settings)
    app = FastAPI(
        title="HomeValue API", version="0.1.0",
        description="Chicago housing valuation and spatial market intelligence.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins),
        allow_credentials=True, allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    app.state.engine = engine

    def valuation_engine():
        if app.state.engine is None:
            try:
                app.state.engine = _build_engine(settings)
            except (FileNotFoundError, ValueError, KeyError) as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"Valuation engine artifacts are not ready: {type(error).__name__}",
                ) from error
        return app.state.engine

    @app.get("/health")
    def health():
        return {"status": "ok", "valuation_engine_loaded": app.state.engine is not None}

    @app.post("/valuation/predict", response_model=ValuationResponse)
    def predict(request: ValuationRequest):
        return valuation_engine().predict(request)

    @app.get("/valuation/neighborhoods", response_model=RecordCollection)
    def valuation_neighborhoods():
        frame = repository.parquet("accessibility/core_sales_with_accessibility.parquet")
        records = _neighborhood_options(frame)
        return RecordCollection(
            count=len(records), offset=0, limit=min(500, max(1, len(records))), records=records
        )

    @app.get("/market/summary", response_model=MarketOverviewResponse)
    def market_summary():
        frame = repository.parquet("neighborhood_indices/neighborhood_price_indices.parquet")
        geography = _geography(frame)
        year = pd.to_numeric(frame.get("year"), errors="coerce")
        latest_year = int(year.max()) if year.notna().any() else None
        current = frame.loc[year.eq(latest_year)] if latest_year is not None else frame
        sale_count = pd.to_numeric(current.get("sale_count"), errors="coerce")
        return MarketOverviewResponse(
            latest_year=latest_year, geography_count=int(current[geography].nunique()),
            transaction_count=int(sale_count.fillna(0).sum()),
            median_sale_price=(
                float(pd.to_numeric(current["median_sale_price"], errors="coerce").median())
                if "median_sale_price" in current else None
            ),
            median_ppsf=(
                float(pd.to_numeric(current["median_ppsf"], errors="coerce").median())
                if "median_ppsf" in current else None
            ),
        )

    @app.get("/market/neighborhood/{neighborhood_id}", response_model=NeighborhoodDetailResponse)
    def neighborhood(neighborhood_id: str):
        history = repository.parquet("neighborhood_indices/neighborhood_price_indices.parquet")
        geography = _geography(history)
        selected = history.loc[history[geography].astype(str).eq(neighborhood_id)].sort_values("year")
        if selected.empty:
            raise HTTPException(status_code=404, detail="Neighborhood not found")
        segment = None
        try:
            segments = repository.parquet("segmentation/neighborhood_segments.parquet")
            segment_geography = _geography(segments)
            match = segments.loc[segments[segment_geography].astype(str).eq(neighborhood_id)]
            segment = _records(match.head(1))[0] if len(match) else None
        except HTTPException:
            pass
        return NeighborhoodDetailResponse(
            neighborhood_id=neighborhood_id, price_history=_records(selected), segment=segment
        )

    @app.get("/market/comparables/{pin}", response_model=RecordCollection)
    def comparables(pin: str, limit: int = Query(25, ge=1, le=500), offset: int = Query(0, ge=0)):
        normalized = normalize_pin(pin)
        if normalized is None:
            raise HTTPException(status_code=422, detail="PIN must normalize to 14 digits")
        predictions = repository.parquet("comparables/comparable_predictions.parquet")
        matches = predictions.loc[predictions["pin"].map(normalize_pin).eq(normalized)].sort_values("sale_date")
        if matches.empty:
            raise HTTPException(status_code=404, detail="No comparable analysis found for PIN")
        target = matches.iloc[-1]["sale_id"]
        links = repository.parquet("comparables/comparable_links.parquet")
        selected = links.loc[links["target_sale_id"].eq(target)].sort_values("normalized_weight", ascending=False)
        page = selected.iloc[offset:offset + limit]
        return RecordCollection(count=len(selected), offset=offset, limit=limit, records=_records(page))

    @app.get("/models/performance", response_model=RecordCollection)
    def performance(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        frame = repository.csv("benchmark/model_benchmark.csv")
        return RecordCollection(count=len(frame), offset=offset, limit=limit, records=_records(frame.iloc[offset:offset + limit]))

    @app.get("/models/spatial", response_model=ResearchResponse)
    def spatial_models():
        return ResearchResponse(available=True, results={
            "autocorrelation": repository.json("spatial_autocorrelation/spatial_autocorrelation_report.json"),
            "lag": repository.json("spatial_lag/spatial_lag_results.json"),
            "error": repository.json("spatial_error/spatial_error_results.json"),
        })

    @app.get("/models/errors", response_model=RecordCollection)
    def model_errors(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        frame = repository.csv("validation/error_segments/segment_error_metrics.csv")
        return RecordCollection(count=len(frame), offset=offset, limit=limit, records=_records(frame.iloc[offset:offset + limit]))

    @app.get("/accessibility/transit", response_model=ResearchResponse)
    def transit():
        return ResearchResponse(available=True, results=repository.json("transit_robustness/transit_robustness_results.json"))

    @app.get("/accessibility/lake", response_model=ResearchResponse)
    def lake():
        return ResearchResponse(available=True, results=repository.json("amenity_gradients/gradient_results.json"))

    @app.get("/neighborhoods/segments", response_model=RecordCollection)
    def segments(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        frame = repository.parquet("segmentation/neighborhood_segments.parquet")
        geography = _geography(frame)
        names = _neighborhood_name_frame(
            repository.parquet("accessibility/core_sales_with_accessibility.parquet")
        ).drop(columns="sale_count")
        names["neighborhood_id"] = names["neighborhood_id"].astype("string")
        frame = frame.copy()
        frame[geography] = frame[geography].astype("string")
        frame = frame.merge(
            names, left_on=geography, right_on="neighborhood_id",
            how="left", validate="many_to_one",
        )
        return RecordCollection(count=len(frame), offset=offset, limit=limit, records=_records(frame.iloc[offset:offset + limit]))

    return app


app = create_app()
