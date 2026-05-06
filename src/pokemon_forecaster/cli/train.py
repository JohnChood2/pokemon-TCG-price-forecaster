"""Batch-train models for cards with enough history.

Run via:
    uv run pokemon-train --min-history 60
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
    parser = argparse.ArgumentParser(description="Train forecasters per (card, variant).")
    parser.add_argument("--min-history", type=int, default=60)
    parser.add_argument("--variant", default="holofoil")
    args = parser.parse_args()

    store = PriceStore()
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    trained = 0
    skipped = 0
    with store.session() as session:
        # Find all (card_id, variant) pairs with enough rows.
        rows = session.execute(
            select(
                PriceSnapshot.card_id,
                PriceSnapshot.variant,
                PriceSnapshot.snapshot_date,
                PriceSnapshot.market,
            ).where(PriceSnapshot.variant == args.variant)
        ).all()

    df = pd.DataFrame(rows, columns=["card_id", "variant", "ds", "y"])
    df = df.dropna(subset=["y"])

    for (card_id, variant), group in df.groupby(["card_id", "variant"]):
        if len(group) < args.min_history:
            skipped += 1
            continue
        model = ProphetForecaster()
        try:
            model.fit(group[["ds", "y"]])
        except Exception as e:  # noqa: BLE001 — Prophet is noisy
            logger.warning("Failed %s/%s: %s", card_id, variant, e)
            skipped += 1
            continue
        path = settings.model_dir / f"{card_id}__{variant}.joblib"
        model.save(path)
        trained += 1
        if trained % 25 == 0:
            logger.info("Trained %d models so far", trained)

    logger.info("Done. Trained: %d, skipped: %d", trained, skipped)


if __name__ == "__main__":
    main()
