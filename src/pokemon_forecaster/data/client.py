"""Client for pokemontcg.io — returns card metadata with TCGPlayer prices."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from pokemon_forecaster.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pokemontcg.io/v2"
PAGE_SIZE = 250  # API max


class PokemonTCGClient:
    """Thin wrapper around the public Pokémon TCG API.

    Cards returned include a ``tcgplayer`` field with current market prices
    across variants (normal, holofoil, reverseHolofoil, 1stEditionHolofoil, ...).
    Each call captures a *snapshot* — to build a time series, run ingest on a
    schedule (e.g. daily via a cron CronJob in k8s) and append rows.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        headers = {}
        key = api_key or settings.pokemontcg_api_key
        if key:
            headers["X-Api-Key"] = key
        self._client = httpx.Client(base_url=BASE_URL, headers=headers, timeout=timeout)

    def __enter__(self) -> PokemonTCGClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def iter_cards(self, query: str | None = None) -> Iterator[dict[str, Any]]:
        """Paginate through all cards matching `query` (Lucene-style, e.g. 'set.id:swsh1').

        Yields one card dict at a time so memory stays flat regardless of result size.
        """
        page = 1
        while True:
            params: dict[str, Any] = {"page": page, "pageSize": PAGE_SIZE}
            if query:
                params["q"] = query
            payload = self._get("/cards", params=params)
            cards = payload.get("data", [])
            if not cards:
                return
            yield from cards
            if len(cards) < PAGE_SIZE:
                return
            page += 1
            logger.info("Fetched page %d (%d cards so far)", page - 1, (page - 1) * PAGE_SIZE)

    def get_card(self, card_id: str) -> dict[str, Any]:
        return self._get(f"/cards/{card_id}")["data"]

    def list_sets(self) -> list[dict[str, Any]]:
        return self._get("/sets")["data"]
