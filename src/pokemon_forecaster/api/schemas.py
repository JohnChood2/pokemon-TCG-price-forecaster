"""Request/response schemas for the FastAPI service."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    ds: date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastRequest(BaseModel):
    card_id: str = Field(..., examples=["swsh1-1"])
    variant: str = Field(default="holofoil")
    horizon_days: int = Field(default=14, ge=1, le=90)


class ForecastResponse(BaseModel):
    card_id: str
    variant: str
    model: str
    horizon_days: int
    forecast: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    version: str
