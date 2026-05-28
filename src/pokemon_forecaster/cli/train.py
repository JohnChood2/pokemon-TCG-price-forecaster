"""Batch model trainer — fits a ProphetForecaster for every (card, variant) with enough history.

When to run
-----------
Run this *after* the daily ingest job has completed:

    uv run pokemon-train --min-history 60 --variant holofoil

Or as a Kubernetes Job triggered after the CronJob (see k8s/ for examples).

What it does
------------
1. Queries the database for all (card_id, variant, snapshot_date, market) rows
   for the chosen variant.
2. Groups by (card_id, variant) and skips any card with fewer than
   ``--min-history`` non-null market rows.
3. Fits a ``ProphetForecaster`` on the full history for each qualifying card.
4. Saves the trained model to ``{MODEL_DIR}/{card_id}__{variant}.joblib``.

The API's ``/predict`` endpoint will load these artefacts from disk on the
next request, rather than training on demand (which adds latency).

Why pre-train?
--------------
On-demand training in the API (the current fallback) adds 2–10 seconds per
first request.  Pre-training during the nightly batch window means all requests
serve from cached artefacts and respond in milliseconds.

Error handling
--------------
Prophet occasionally fails on pathological data (e.g. all-zero prices, too
many NaN gaps).  Failures are logged as WARNINGs and skipped so one bad card
doesn't abort the entire training run.

Minimum history recommendations
--------------------------------
- ``--min-history 30``  — acceptable for a demo; uncertainty intervals are wide.
- ``--min-history 60``  — recommended; covers ~2 months of daily ingests.
- ``--min-history 90``  — conservative; better seasonality estimation.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import select

from pokemon_forecaster.config import settings
from pokemon_forecaster.data.storage import PriceSnapshot, PriceStore
from pokemon_forecaster.models import ProphetForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the ``pokemon-train`` CLI script.

    Trains a separate model for each (card_id, variant) pair that has enough
    history, then saves the serialised model to the configured model directory.
    """
    parser = argparse.ArgumentParser(
        description="Batch-train price forecasters for all cards with enough history.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=60,
        metavar="N",
        help=(
            "Minimum number of non-null market price rows required before "
            "training a model for a card (default: 60)."
        ),
    )
    parser.add_argument(
        "--variant",
        default="holofoil",
        help=(
            "Only train models for this TCGPlayer price variant "
            "(default: holofoil).  Pass 'normal' or 'reverseHolofoil' to "
            "train other variants instead."
        ),
    )
    args = parser.parse_args()

    store = PriceStore()
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    trained = 0   # models successfully saved
    skipped = 0   # cards skipped (insufficient data or training failure)

    # Step 1: pull all (card_id, variant, date, price) rows for the chosen variant.
    # We load everything into a DataFrame first so we can group and filter in
    # pandas rather than running one SQL query per card.
    with store.session() as session:
        rows = session.execute(
            select(
                PriceSnapshot.card_id,
                PriceSnapshot.variant,
                PriceSnapshot.snapshot_date,
                PriceSnapshot.market,
            ).where(PriceSnapshot.variant == args.variant)
        ).all()

    df = pd.DataFrame(rows, columns=["card_id", "variant", "ds", "y"])

    # Drop rows where market price is NULL — Prophet can't train on missing targets.
    df = df.dropna(subset=["y"])

    logger.info(
        "Loaded %d rows for variant=%r — will train with min_history=%d",
        len(df),
        args.variant,
        args.min_history,
    )

    # Step 2: train one model per (card_id, variant) group.
    for (card_id, variant), group in df.groupby(["card_id", "variant"]):
        if len(group) < args.min_history:
            # Not enough history — skip silently (counted in the final summary).
            skipped += 1
            continue

        model = ProphetForecaster()
        try:
            # Prophet expects columns named exactly "ds" and "y".
            model.fit(group[["ds", "y"]])
        except Exception as e:  # noqa: BLE001 — Prophet surfaces many internal errors
            logger.warning("Failed to train %s/%s: %s", card_id, variant, e)
            skipped += 1
            continue

        # Save the fitted model to disk for the API to load on the next request.
        path = settings.model_dir / f"{card_id}__{variant}.joblib"
        model.save(path)
        trained += 1

        if trained % 25 == 0:
            logger.info("Progress: %d models trained so far", trained)

    logger.info(
        "Training complete | trained=%d  skipped=%d (total_groups=%d)",
        trained,
        skipped,
        trained + skipped,
    )


if __name__ == "__main__":
    main()
