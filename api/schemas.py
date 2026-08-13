"""Typed request and response contracts for the HomeValue API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ValuationRequest(StrictSchema):
    building_sqft: float = Field(gt=0, le=100_000)
    land_sqft: float | None = Field(default=None, gt=0)
    bedrooms: float | None = Field(default=None, ge=0, le=30)
    bathrooms: float | None = Field(default=None, ge=0, le=30)
    building_age: float | None = Field(default=None, ge=0, le=300)
    garage_spaces: float | None = Field(default=None, ge=0, le=20)
    residence_type: str | None = None
    pin: str | None = Field(default=None, pattern=r"^\d{14}$")
    neighborhood: str | None = None
    municipality: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    valuation_date: date | None = None

    @model_validator(mode="after")
    def coordinates_are_paired(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        return self


class ValueDriver(StrictSchema):
    component: str
    feature: str
    dollar_contribution: float


class ComparableSummary(StrictSchema):
    sale_id: str
    sale_price: float = Field(gt=0)
    distance_miles: float = Field(ge=0)
    sale_date: str


class ValuationResponse(StrictSchema):
    estimated_value: float = Field(gt=0)
    lower_interval: float = Field(gt=0)
    upper_interval: float = Field(gt=0)
    baseline_market_value: float = Field(gt=0)
    property_component: float
    location_component: float
    time_market_component: float
    other_component: float = 0
    confidence: float = Field(ge=0, le=1)
    model_name: str
    value_drivers: list[ValueDriver] = Field(default_factory=list)
    comparables: list[ComparableSummary] = Field(default_factory=list)

    @model_validator(mode="after")
    def estimate_is_inside_interval(self):
        if not self.lower_interval <= self.estimated_value <= self.upper_interval:
            raise ValueError("valuation interval must contain estimated_value")
        reconciled = (
            self.baseline_market_value + self.property_component + self.location_component
            + self.time_market_component + self.other_component
        )
        if abs(reconciled - self.estimated_value) > max(1e-6 * self.estimated_value, .01):
            raise ValueError("valuation components must reconcile to estimated_value")
        return self


class MarketSummaryResponse(StrictSchema):
    geography: str
    median_sale_price: float | None = Field(default=None, ge=0)
    annual_appreciation: float | None = None
    transaction_count: int = Field(ge=0)
    market_archetype: str | None = None


class ModelPerformanceResponse(StrictSchema):
    model: str
    mae: float | None = Field(default=None, ge=0)
    rmse: float | None = Field(default=None, ge=0)
    mdape: float | None = Field(default=None, ge=0)
    temporal_tested: bool
    spatial_tested: bool
