"""Pull a fresh batch of cards + prices from pokemontcg.io and persist them.

Run manually:
    uv run pokemon-ingest --query 'set.id:swsh1'
Or in production via a Kubernetes CronJob — once a day is plenty.
"""

from __future__ import annotations

import argparse
import logging

from pokemon_forecaster.data.client import PokemonTCGClient
from pokemon_forecaster.data.storage import PriceStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Pokémon TCG prices.")
    parser.add_argument(
        "--query",
        default=None,
        help="Lucene-style filter, e.g. 'set.id:swsh1' or 'rarity:\"Rare Holo\"'.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap cards processed (debug).")
    args = parser.parse_args()

    store = PriceStore()
    inserted = 0
    seen = 0

    with PokemonTCGClient() as client, store.session() as session:
        for card in client.iter_cards(query=args.query):
            seen += 1
            store.upsert_card(session, card)
            inserted += store.insert_price_snapshots(
                session, card_id=card["id"], tcgplayer_payload=card.get("tcgplayer")
            )
            if args.limit and seen >= args.limit:
                break
            if seen % 100 == 0:
                session.commit()
                logger.info("Processed %d cards, %d snapshots inserted", seen, inserted)
        session.commit()

    logger.info("Done. %d cards processed, %d price snapshots inserted.", seen, inserted)


if __name__ == "__main__":
    main()
