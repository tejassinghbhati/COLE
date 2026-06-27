"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Custom cost profile to predict + explain. All indices are NYC=100 scale."""
    housing_index: float = Field(100, ge=0, le=300)
    groceries_index: float = Field(100, ge=0, le=300)
    transport_index: float = Field(100, ge=0, le=300)
    utilities_index: float = Field(100, ge=0, le=300)
    restaurant_index: float = Field(100, ge=0, le=300)
    healthcare_index: float = Field(100, ge=0, le=300)
    childcare_index: float = Field(100, ge=0, le=300)
    median_income_usd: float = Field(3000, ge=100, le=20000)
    population_density: float = Field(5000, ge=0, le=50000)
    tourism_intensity: float = Field(0.5, ge=0, le=1)
    region: str = "North America"


class Driver(BaseModel):
    feature: str
    label: str
    value: float | str | None = None
    contribution: float


class Explanation(BaseModel):
    predicted_index: float
    baseline_index: float
    drivers: list[Driver]
