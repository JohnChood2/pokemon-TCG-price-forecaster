"""Daily ingest pipeline — pull fresh card data from pokemontcg.io and persist prices.

Overview
--------
This script is the heart of the data pipeline. It calls the Pokémon TCG API,
iterates over every matching card, upserts card metadata into the ``cards``
table, and appends today's price snapshot into ``price_snapshots``.

Run it manually during development:

    uv run pokemon-ingest --query 'set.id:swsh1' --batch-size 50

Or let the Kubernetes CronJob invoke it automatically every night at 06:00 UTC
(see ``k8s/cronjob-ingest.yaml``). The CronJob passes ``--batch-size 50`` so
each database commit covers exactly 50 cards, keeping memory usage flat and
giving you fine-grained progress visibility in the pod logs.

Why daily?
----------
The pokemontcg.io API only exposes *current* prices — there is no historical
endpoint. The only way to build a time series is to run this ingest on a
schedule and accumulate one row per (card_id, variant, date). After ~30 days
you have enough data to train Prophet; after ~60 days XGBoost becomes viable.

Batch commits
-------------
SQLAlchemy accumulates ORM objects in its identity map. Without periodic
commits that memory grows linearly with the number of cards. ``--batch-size``
(default 50) controls how many cards are flushed per ``session.commit()``.
Smaller values = lower peak RAM; larger = fewer round-trips to the DB.
"""

from __future__ import annotations

import argparse
import logging

from pokemon_forecaster.data.client import PokemonTCGClient
from pokemon_forecaster.data.storage import PriceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the ``pokemon-ingest`` CLI script.

    Argument reference
    ------------------
    --query       Lucene-style filter forwarded to the pokemontcg.io ``/cards``
                  endpoint.  Omit to fetch *all* cards (slow — ~20 000+).
                  Examples:
                    'set.id:swsh1'            → Sword & Shield Base Set only
                    'rarity:"Rare Holo"'      → any holo rare, all sets
                    'set.id:swsh1 rarity:Rare' → intersection

    --batch-size  How many cards to accumulate before committing to the DB.
                  Defaults to 50.  Lower values reduce peak memory; higher
                  values reduce DB round-trips.  50 is a good balance for
                  SQLite; bump to 200-500 if you switch to Postgres.

    --limit       Hard cap on total cards processed.  Useful during development
                  so you don't accidentally pull 20 000 cards on a laptop.
    """
    parser = argparse.ArgumentParser(
        description="Ingest Pokémon TCG card metadata and price snapshots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Lucene-style filter, e.g. 'set.id:swsh1' or 'rarity:\"Rare Holo\"'. "
            "Omit to fetch all cards."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Commit to the DB every N cards (default: 50). Keeps memory usage bounded.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Stop after processing N cards total. Useful for smoke-testing.",
    )
    args = parser.parse_args()

    store = PriceStore()
    inserted = 0  # total price snapshot rows written this run
    seen = 0      # total cards processed this run

    logger.info(
        "Starting ingest | query=%r  batch_size=%d  limit=%s",
        args.query,
        args.batch_size,
        args.limit or "none",
    )

    with PokemonTCGClient() as client, store.session() as session:
        for card in client.iter_cards(query=args.query):
            seen += 1

            # Upsert the card's metadata (name, set, rarity, etc.)
            # This is idempotent — re-running the ingest will update any
            # fields that changed (e.g. corrections to rarity text).
            store.upsert_card(session, card)

            # Append today's price snapshot for every price variant the API
            # returns (holofoil, normal, reverseHolofoil, …).
            # The unique constraint on (card_id, variant, snapshot_date)
            # prevents duplicate rows if the ingest runs more than once per day.
            inserted += store.insert_price_snapshots(
                session, card_id=card["id"], tcgplayer_payload=card.get("tcgplayer")
            )

            # Commit in batches to bound memory use. SQLAlchemy's identity map
            # holds a reference to every pending ORM object; flushing regularly
            # keeps the working set small.
            if seen % args.batch_size == 0:
                session.commit()
                logger.info(
                    "Batch committed | cards=%d  snapshots_so_far=%d", seen, inserted
                )

            if args.limit and seen >= args.limit:
                logger.info("--limit %d reached, stopping early.", args.limit)
                break

        # Final commit for the last partial batch (< batch_size cards).
        session.commit()

    logger.info(
        "Ingest complete | cards_processed=%d  price_snapshots_inserted=%d",
        seen,
        inserted,
    )


if __name__ == "__main__":
    main()
