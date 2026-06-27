"""Pydantic request/response models for the API. All indices are New Delhi = 100."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Custom cost profile to predict + explain. Indices are re-based: New Delhi = 100."""
    rent_index: float = Field(100, ge=0, le=2000)
    groceries_index: float = Field(100, ge=0, le=1000)
    restaurant_index: float = Field(100, ge=0, le=1000)
    purchasing_power_index: float = Field(100, ge=0, le=500)
    region: str = "South Asia"


class Driver(BaseModel):
    feature: str
    label: str
    value: float | str | None = None
    contribution: float


class Explanation(BaseModel):
    predicted_index: float
    baseline_index: float
    drivers: list[Driver]
