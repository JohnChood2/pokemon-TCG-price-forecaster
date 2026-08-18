"""FastAPI application — the public-facing REST interface for price forecasts.

Endpoints
---------
GET  /health   — liveness/readiness probe; returns version string.
POST /predict  — run a price forecast for a given (card_id, variant, horizon).

Request/response shapes are defined in ``api/schemas.py`` as Pydantic models.
FastAPI auto-generates OpenAPI docs at ``/docs`` and ``/redoc``.

Predict endpoint logic
----------------------
1. Load the card's price history from the database (``PriceStore.get_history``).
2. Reject the request with HTTP 400 if fewer than 30 rows exist — Prophet
   needs at least that much data for a meaningful forecast.
3. If a cached ``.joblib`` model file exists for this (card, variant), load it.
   Otherwise, train a new ``ProphetForecaster`` on the fly and save it for
   the next request (lazy training).
4. Call ``model.predict(horizon_days)`` and return the forecast as JSON.

On-demand vs batch training
----------------------------
The current implementation trains lazily on the first request for a card.
In production, ``cli/train.py`` should be run nightly (after the ingest job)
to pre-train all models and cache them to disk.  The API then *only* loads
pre-trained artefacts and never blocks a request on training time.

Model caching
-------------
Model files are stored at ``{MODEL_DIR}/{card_id}__{variant}.joblib``.
Once a file exists, the API loads it on every request.  To force a retrain,
delete the file — the next request will train and save a fresh model.

App lifespan
------------
The ``lifespan`` context manager runs startup/shutdown logic:
- Startup: ensures the model directory exists; initialises the ``PriceStore``.
- Shutdown: currently a no-op (SQLite and joblib don't need explicit teardown).
"""

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

# Minimum history rows required before we'll attempt a forecast.
# Below 30, Prophet's uncertainty intervals are unreliable.
MIN_HISTORY_ROWS = 30


def _model_path(card_id: str, variant: str) -> Path:
    """Return the expected path for a trained model artefact.

    Naming convention: ``{MODEL_DIR}/{card_id}__{variant}.joblib``
    The double underscore separates card ID from variant to avoid ambiguity
    since card IDs can contain hyphens (e.g. ``swsh1-1``).
    """
    return settings.model_dir / f"{card_id}__{variant}.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ASGI lifespan handler — runs startup logic, yields, then shutdown logic.

    Startup
    -------
    - Ensure the model artefact directory exists.
    - Attach a ``PriceStore`` instance to ``app.state`` so request handlers
      can access it without creating a new engine per request.

    Shutdown
    --------
    Currently nothing to clean up.  If you add a connection pool (e.g. async
    Postgres), close it here.
    """
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = PriceStore()
    logger.info("API ready (version %s)", __version__)
    yield
    # Future: close DB pool, flush metrics, etc.


app = FastAPI(
    title="Pokémon TCG Price Forecaster",
    description=(
        "Forecast Pokémon TCG card prices 1-90 days into the future using "
        "historical TCGPlayer market data.  Powered by Prophet."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness / readiness probe.

    Used by the Kubernetes ``livenessProbe`` and ``readinessProbe`` in
    ``k8s/deployment.yaml``.  Returns HTTP 200 with ``{"status": "ok"}``
    as long as the process is alive — the probe does not check DB connectivity.
    Add a DB ping here if you need a true readiness check.
    """
    return HealthResponse(status="ok", version=__version__)


@app.post("/predict", response_model=ForecastResponse, tags=["forecast"])
def predict(req: ForecastRequest) -> ForecastResponse:
    """Generate a price forecast for a Pokémon TCG card.

    The endpoint follows a lazy-training pattern:

    1. Fetch the card's full price history from the database.
    2. Validate that there are enough rows to produce a reliable forecast.
    3. If a serialised model file exists for this (card, variant), load it.
       Otherwise train a new model on the fly, save it, and use it.
    4. Generate and return the forecast.

    Parameters (request body — see ``ForecastRequest`` schema)
    -----------------------------------------------------------
    card_id:
        pokemontcg.io card identifier, e.g. ``"swsh1-1"``.
    variant:
        TCGPlayer price tier, e.g. ``"holofoil"`` (default).
    horizon_days:
        How many days ahead to forecast (1-90, default 14).

    Response (see ``ForecastResponse`` schema)
    ------------------------------------------
    Returns a list of ``ForecastPoint`` objects, one per forecast day, each
    with ``ds`` (date), ``yhat`` (expected price), ``yhat_lower`` and
    ``yhat_upper`` (80% confidence interval bounds).

    Errors
    ------
    400 — fewer than 30 price snapshots in the database for this card/variant.
          Run ``pokemon-ingest`` more days to accumulate history.
    """
    store: PriceStore = app.state.store

    # Step 1: load price history from the database.
    history_rows = store.get_history(req.card_id, req.variant)

    # Step 2: require minimum history for a meaningful forecast.
    if len(history_rows) < MIN_HISTORY_ROWS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough history for {req.card_id}/{req.variant} "
                f"({len(history_rows)} snapshots, need >={MIN_HISTORY_ROWS}). "
                "Run the ingest job more days to accumulate price history."
            ),
        )

    # Step 3: build the Prophet-compatible (ds, y) DataFrame.
    # Filter out rows where market price is None (card had no TCGPlayer listing
    # on that day — can happen for freshly released cards with no sales yet).
    history = pd.DataFrame(
        [{"ds": r.snapshot_date, "y": r.market} for r in history_rows if r.market is not None]
    )

    # Step 4: load or train the model.
    path = _model_path(req.card_id, req.variant)
    if path.exists():
        # Fast path — load the pre-trained artefact from disk.
        # In production, the nightly `pokemon-train` job refreshes these files.
        model: Forecaster = ProphetForecaster.load(path)
        logger.info("Loaded cached model for %s/%s", req.card_id, req.variant)
    else:
        # Slow path — train on demand, save for next time.
        # This adds ~2-10 seconds of latency on the first request for a card.
        logger.info("No cached model for %s/%s — training on demand", req.card_id, req.variant)
        model = ProphetForecaster()
        model.fit(history)
        model.save(path)

    # Step 5: generate the forecast and return it.
    forecast_df = model.predict(req.horizon_days)
    return ForecastResponse(
        card_id=req.card_id,
        variant=req.variant,
        model=model.name,
        horizon_days=req.horizon_days,
        forecast=forecast_df.to_dict(orient="records"),
    )


def run() -> None:
    """Uvicorn entry point — exposed as the ``pokemon-api`` console script.

    Invoked by ``uv run pokemon-api`` (or ``python -m pokemon_forecaster.api.main``).
    In Docker, the CMD in the Dockerfile calls this module directly.

    ``reload=False`` in production — hot-reload only makes sense in dev and
    would cause k8s pods to restart unnecessarily.
    """
    uvicorn.run(
        "pokemon_forecaster.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    run()
