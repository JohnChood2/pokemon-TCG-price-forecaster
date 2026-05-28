"""HTTP client for the Pokémon TCG API (pokemontcg.io v2).

This module is the only place in the codebase that talks to the external API.
Keeping all HTTP logic here means everything else is testable without network
access — just inject a fake list of card dicts.

API overview
------------
pokemontcg.io exposes a free REST API for Pokémon TCG card metadata and
*current* TCGPlayer market prices.  Key facts:

- Base URL: https://api.pokemontcg.io/v2
- Auth: optional API key header ``X-Api-Key`` (free registration at
  https://dev.pokemontcg.io/).  Without it you get ~1 000 req/day;
  with a free key you get ~20 000.
- Pagination: ``?page=N&pageSize=250`` (250 is the max per page).
- Filtering: Lucene-style ``?q=`` parameter, e.g. ``set.id:swsh1``.
- Price data lives inside each card's ``tcgplayer.prices`` block:
    {
      "holofoil":        {"market": 12.34, "low": 8.0, "mid": 10.0, "high": 15.0},
      "reverseHolofoil": {"market": 3.10, ...},
      "normal":          {"market": 1.50, ...}
    }

Snapshot model
--------------
The API only gives you *current* prices — there's no historical endpoint.
To build a time series you must call the ingest script on a schedule and
accumulate one ``price_snapshots`` row per (card_id, variant, date).

Retry strategy
--------------
Transient 5xx errors and rate-limit responses are retried automatically via
``tenacity`` with exponential back-off (1s → 2s → 4s → 8s, max 4 attempts).
This means a brief API blip won't break a nightly ingest run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pokemon_forecaster.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pokemontcg.io/v2"

# The API allows up to 250 cards per page.  Using the maximum reduces the
# number of HTTP round-trips and makes pagination faster.
PAGE_SIZE = 250


class PokemonTCGClient:
    """Thin wrapper around the Pokémon TCG REST API.

    Intended to be used as a context manager so the underlying ``httpx.Client``
    connection pool is always closed cleanly:

        with PokemonTCGClient() as client:
            for card in client.iter_cards(query="set.id:swsh1"):
                process(card)

    The client reads ``settings.pokemontcg_api_key`` automatically.  You can
    override it by passing ``api_key=`` explicitly — handy in tests.

    Cards returned include a ``tcgplayer`` field with current market prices
    across variants (normal, holofoil, reverseHolofoil, 1stEditionHolofoil, …).
    Each call captures a *snapshot* — to build a time series, run ingest on a
    schedule (e.g. daily via the Kubernetes CronJob) and append rows.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        """Initialise the client.

        Parameters
        ----------
        api_key:
            pokemontcg.io API key.  Falls back to ``settings.pokemontcg_api_key``
            (read from the ``POKEMONTCG_API_KEY`` env var / .env file).
        timeout:
            HTTP request timeout in seconds.  The API can be slow on large
            queries, so 30s is a reasonable default.
        """
        headers: dict[str, str] = {}
        key = api_key or settings.pokemontcg_api_key
        if key:
            headers["X-Api-Key"] = key
        self._client = httpx.Client(base_url=BASE_URL, headers=headers, timeout=timeout)

    def __enter__(self) -> PokemonTCGClient:
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the underlying connection pool on exit."""
        self._client.close()

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=1, max=20),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a GET request and return the parsed JSON body.

        Decorated with ``@retry`` — automatically retries up to 4 times with
        exponential back-off on any exception (network error, 5xx, timeout).
        ``reraise=True`` means the *original* exception propagates if all
        retries are exhausted.
        """
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def iter_cards(self, query: str | None = None) -> Iterator[dict[str, Any]]:
        """Paginate through all cards matching *query* and yield one card at a time.

        Parameters
        ----------
        query:
            Lucene-style filter string forwarded to the ``?q=`` parameter.
            Examples:
              ``"set.id:swsh1"``             → Sword & Shield Base Set only
              ``'rarity:"Rare Holo"'``       → all holo rares across every set
              ``"set.id:swsh1 rarity:Rare"`` → intersection of both filters
            Pass ``None`` to iterate over all ~20 000+ cards in the database.

        Yields
        ------
        dict
            A single card object from the API, including ``id``, ``name``,
            ``set``, ``rarity``, ``tcgplayer``, and many other fields.

        Notes
        -----
        Memory stays constant regardless of result size because we ``yield``
        one card at a time rather than building a list.  The API is paginated
        at ``PAGE_SIZE`` cards per request; we fetch the next page only when
        the current one is exhausted.
        """
        page = 1
        while True:
            params: dict[str, Any] = {"page": page, "pageSize": PAGE_SIZE}
            if query:
                params["q"] = query

            payload = self._get("/cards", params=params)
            cards = payload.get("data", [])

            if not cards:
                # Empty page means we've consumed all results.
                return

            yield from cards

            if len(cards) < PAGE_SIZE:
                # Partial page → this was the last one.
                return

            page += 1
            logger.info(
                "Fetched page %d (%d cards fetched so far)", page - 1, (page - 1) * PAGE_SIZE
            )

    def get_card(self, card_id: str) -> dict[str, Any]:
        """Fetch a single card by its unique ID (e.g. ``"swsh1-1"``).

        Useful for targeted refreshes or one-off lookups.
        """
        return self._get(f"/cards/{card_id}")["data"]  # type: ignore[no-any-return]

    def list_sets(self) -> list[dict[str, Any]]:
        """Return metadata for every TCG set (name, series, release date, etc.)."""
        return self._get("/sets")["data"]  # type: ignore[no-any-return]
