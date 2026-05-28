"""Pydantic request/response schemas for the FastAPI service.

These models serve three purposes:

1. **Input validation** — FastAPI validates incoming JSON against ``ForecastRequest``
   and returns HTTP 422 automatically if a field is the wrong type or out of range.
2. **Output serialisation** — FastAPI uses ``ForecastResponse`` to coerce and
   serialise the return value to JSON.
3. **OpenAPI documentation** — Field descriptions and examples appear in the
   auto-generated ``/docs`` UI.

Schema details
--------------
``ForecastRequest``
    The user sends this in the POST body to ``/predict``.

``ForecastResponse``
    What the API returns — a list of ``ForecastPoint`` objects, one per day.

``ForecastPoint``
    One forecast observation: a date, a point estimate, and an 80% confidence
    interval.  The lower/upper bounds come from Prophet's built-in uncertainty
    quantification.

``HealthResponse``
    Returned by ``GET /health`` — minimal liveness payload.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    """A single forecasted price observation for one future day.

    Attributes
    ----------
    ds:
        The date of this forecast point.
    yhat:
        Point estimate — the model's expected market price (USD) on ``ds``.
    yhat_lower:
        Lower bound of the 80% credible interval.  There is approximately a
        10% chance the actual price will fall *below* this value.
    yhat_upper:
        Upper bound of the 80% credible interval.  There is approximately a
        10% chance the actual price will rise *above* this value.
    """

    ds: date
    yhat: float
    yhat_lower: float
    yhat_upper: float


class ForecastRequest(BaseModel):
    """Body schema for ``POST /predict``.

    Attributes
    ----------
    card_id:
        pokemontcg.io canonical card identifier.  Format is ``{set_id}-{number}``,
        e.g. ``"swsh1-1"`` (Sword & Shield base set, card #1).
        Find IDs at https://pokemontcg.io or by running
        ``uv run pokemon-ingest --query 'set.id:swsh1' --limit 5``.
    variant:
        TCGPlayer price tier.  Common values:
        - ``"holofoil"``            — standard holo rare (most common)
        - ``"normal"``              — non-holo version
        - ``"reverseHolofoil"``     — reverse holo print
        - ``"1stEditionHolofoil"``  — first edition (vintage sets only)
    horizon_days:
        How many days into the future to forecast.  Capped at 90 — Prophet's
        uncertainty intervals widen rapidly beyond that and become unhelpful.
    """

    card_id: str = Field(
        ...,
        description="pokemontcg.io card ID, e.g. 'swsh1-1'.",
        examples=["swsh1-1"],
    )
    variant: str = Field(
        default="holofoil",
        description="TCGPlayer price variant: holofoil, normal, reverseHolofoil, etc.",
    )
    horizon_days: int = Field(
        default=14,
        ge=1,
        le=90,
        description="Days to forecast ahead (1–90).",
    )


class ForecastResponse(BaseModel):
    """Response schema for ``POST /predict``.

    Attributes
    ----------
    card_id:
        Echoed from the request — confirms which card was forecast.
    variant:
        Echoed from the request.
    model:
        Name of the forecasting backend used (e.g. ``"prophet"``).
        Useful for debugging and A/B comparisons if multiple backends exist.
    horizon_days:
        Echoed from the request.
    forecast:
        Ordered list of ``ForecastPoint`` objects, one per day, starting
        from tomorrow through ``horizon_days`` in the future.
    """

    card_id: str
    variant: str
    model: str
    horizon_days: int
    forecast: list[ForecastPoint]


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``.

    Attributes
    ----------
    status:
        Always ``"ok"`` while the process is alive.
    version:
        Current package version (from ``pokemon_forecaster.__version__``).
        Useful for verifying which image is running in a Kubernetes pod.
    """

    status: str
    version: str
