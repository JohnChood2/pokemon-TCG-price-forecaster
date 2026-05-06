"""FastAPI service exposing forecasts."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from pokemon_forecaster import __version__
from pokemon_forecaster.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
)
from pokemon_forecaster.config import settings
from pokemon_forecaster.data.storage import PriceStore
from pokemon_forecaster.models import Forecaster, ProphetForecaster

logger = logging.getLogger(__name__)


def _model_path(card_id: str, variant: str) -> Path:
    return settings.model_dir / f"{card_id}__{variant}.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure model dir exists, init store
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = PriceStore()
    logger.info("API ready (version %s)", __version__)
    yield
    # Shutdown: nothing to clean up yet


app = FastAPI(
    title="Pokémon TCG Price Forecaster",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest) -> ForecastResponse:
    """Forecast prices for the next N days.

    Loads a per-card model from disk if available; otherwise trains one on the
    fly using whatever history is in the DB. In production you'd retrain in a
    nightly batch job and serve the cached artifact only.
    """
    store: PriceStore = app.state.store
    history_rows = store.get_history(req.card_id, req.variant)
    if len(history_rows) < 30:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough history for {req.card_id}/{req.variant} "
                f"({len(history_rows)} snapshots, need >=30). Run more ingests."
            ),
        )

    history = pd.DataFrame(
        [{"ds": r.snapshot_date, "y": r.market} for r in history_rows if r.market is not None]
    )

    path = _model_path(req.card_id, req.variant)
    if path.exists():
        model: Forecaster = ProphetForecaster.load(path)
    else:
        model = ProphetForecaster()
        model.fit(history)
        model.save(path)

    forecast_df = model.predict(req.horizon_days)
    return ForecastResponse(
        card_id=req.card_id,
        variant=req.variant,
        model=model.name,
        horizon_days=req.horizon_days,
        forecast=forecast_df.to_dict(orient="records"),
    )


def run() -> None:
    """Entry point exposed as `pokemon-api` script."""
    uvicorn.run(
        "pokemon_forecaster.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    run()
