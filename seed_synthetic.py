"""Seed synthetic price history so you can exercise the full pipeline today
without waiting 30+ days of real ingests.

Generates N days of daily prices for a single (card, variant) using:
    base * (1 + linear_trend) * (1 + weekly_seasonality) * hype_bump * noise

Plus a one-off "hype event" partway through that decays over ~2 weeks — gives
Prophet something interesting to fit (and the confidence bands fan out
realistically when there's a recent shock).

Usage (from the project root):
    uv run python seed_synthetic.py --card-id swsh1-1 --variant holofoil

Idempotent: skips dates that already have a snapshot for this (card, variant),
so you can safely re-run it after a real ingest without unique-constraint errors.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from datetime import date, timedelta

from sqlalchemy import select

from pokemon_forecaster.data.storage import Card, PriceSnapshot, PriceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed")


def synthetic_price(t: int, total_days: int, base: float, seed: int) -> float:
    """Plausible market price for day `t` of `total_days`.

    Components:
      - Linear trend: +20% over the window
      - Weekly seasonality: ±5% sinusoid (Friday peaks)
      - Hype bump: +40% spike at ~2/3 through, decays exponentially over ~2 weeks
      - Gaussian noise: σ = 3%
    """
    rng = random.Random(seed + t)  # deterministic per-day
    trend = 1.0 + 0.20 * (t / max(total_days, 1))
    weekly = 1.0 + 0.05 * math.sin(2 * math.pi * t / 7)

    hype_center = int(total_days * 0.66)
    hype_dist = abs(t - hype_center)
    hype = 1.0 + 0.40 * math.exp(-hype_dist / 5.0) if hype_dist < 21 else 1.0

    noise = 1.0 + rng.gauss(0, 0.03)
    return round(base * trend * weekly * hype * noise, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic price history.")
    parser.add_argument("--card-id", default="swsh1-1", help="Card ID (must exist in DB)")
    parser.add_argument("--variant", default="holofoil")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--base-price", type=float, default=5.00)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    store = PriceStore()
    today = date.today()
    inserted = 0
    skipped = 0

    with store.session() as session:
        card = session.get(Card, args.card_id)
        if card is None:
            raise SystemExit(
                f"Card {args.card_id!r} not found. Run `make ingest` first, "
                f"or pick a card_id that exists in your DB."
            )
        logger.info(
            "Seeding %d days of %s prices for %s (%s)",
            args.days, args.variant, args.card_id, card.name,
        )

        # Don't violate the (card_id, variant, snapshot_date) unique constraint.
        existing_dates = set(
            session.execute(
                select(PriceSnapshot.snapshot_date).where(
                    PriceSnapshot.card_id == args.card_id,
                    PriceSnapshot.variant == args.variant,
                )
            ).scalars().all()
        )

        for offset in range(args.days, 0, -1):
            d = today - timedelta(days=offset)
            if d in existing_dates:
                skipped += 1
                continue
            t = args.days - offset
            market = synthetic_price(t, args.days, args.base_price, args.seed)
            session.add(
                PriceSnapshot(
                    card_id=args.card_id,
                    variant=args.variant,
                    snapshot_date=d,
                    market=market,
                    low=round(market * 0.7, 2),
                    mid=round(market, 2),
                    high=round(market * 1.5, 2),
                    direct_low=round(market * 0.65, 2),
                )
            )
            inserted += 1

        session.commit()

    logger.info("Done. Inserted %d snapshots, skipped %d existing.", inserted, skipped)


if __name__ == "__main__":
    main()
