"""Database schema and data-access layer.

Schema design
-------------
Two tables:

``cards``
    One row per unique Pokémon TCG card.  Stores static metadata: name, set,
    rarity, collector number, release date.  Updated (upserted) on every ingest
    run so corrections to the source API propagate automatically.

``price_snapshots``
    The time-series table — one row per (card_id, variant, snapshot_date).
    ``variant`` is the TCGPlayer price tier: 'holofoil', 'normal', 'reverseHolofoil', etc.
    ``market`` is the TCGPlayer market price in USD at that snapshot date.

    A ``UniqueConstraint`` on (card_id, variant, snapshot_date) means the ingest
    can safely be re-run on the same day without creating duplicate rows — the
    second run will raise an ``IntegrityError`` for any row that already exists
    (SQLAlchemy silently rolls back that row if you add ``INSERT OR IGNORE``).

Long-format time series
-----------------------
The schema stores prices in "long" (tidy) format:

    card_id  | variant   | snapshot_date | market
    ---------+-----------+---------------+-------
    swsh1-1  | holofoil  | 2024-01-01    | 12.50
    swsh1-1  | holofoil  | 2024-01-02    | 13.00
    swsh1-1  | normal    | 2024-01-01    |  1.20
    ...

This is the shape that Prophet, pandas, and scikit-learn expect — you just
filter by (card_id, variant) and you have a ready-to-use time series.

SQLite vs Postgres
------------------
The default ``DATABASE_URL`` uses SQLite, which is fine for development and
low-traffic deployments (the DB is a single file, no server required).
Switch to Postgres for production (``DATABASE_URL=postgresql://…``) —
SQLAlchemy's ORM is database-agnostic so no code changes are needed.

Data access
-----------
Callers should use ``PriceStore`` exclusively and never import SQLAlchemy
directly.  This keeps the rest of the codebase decoupled from the ORM and
makes it easy to swap the storage backend later.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)

from pokemon_forecaster.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared declarative base — all ORM models inherit from this."""


class Card(Base):
    """Static metadata for a single Pokémon TCG card.

    The ``id`` field (e.g. ``"swsh1-1"``) is the pokemontcg.io canonical
    identifier and serves as the primary key.  It is also the key you pass to
    the ``/predict`` API endpoint.
    """

    __tablename__ = "cards"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    set_id: Mapped[str] = mapped_column(String, index=True)  # e.g. "swsh1"
    set_name: Mapped[str] = mapped_column(String)  # e.g. "Sword & Shield"
    rarity: Mapped[str | None] = mapped_column(String, nullable=True)  # "Rare Holo", etc.
    number: Mapped[str | None] = mapped_column(String, nullable=True)  # collector number
    release_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Bidirectional relationship — access all price history via card.snapshots.
    snapshots: Mapped[list[PriceSnapshot]] = relationship(back_populates="card")


class PriceSnapshot(Base):
    """One price observation for a (card, variant) pair on a given date.

    ``market`` is the most important column — TCGPlayer's market price
    is a volume-weighted average of recent sales, which is a better
    predictor of fair value than the listed low/mid/high prices.

    The unique constraint prevents running the ingest twice on the same day
    from creating duplicate rows.
    """

    __tablename__ = "price_snapshots"
    __table_args__ = (
        # Prevents duplicate rows for the same card+variant on the same day.
        UniqueConstraint("card_id", "variant", "snapshot_date", name="uq_card_variant_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    variant: Mapped[str] = mapped_column(String, index=True)
    # snapshot_date is today's date at ingest time — this becomes the ``ds``
    # (datestamp) column that Prophet and other models expect.
    snapshot_date: Mapped[date] = mapped_column(Date, index=True)

    # TCGPlayer price tiers, all in USD.  ``market`` is the most reliable;
    # the others are kept for future feature engineering.
    market: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    direct_low: Mapped[float | None] = mapped_column(Float, nullable=True)  # TCGPlayer Direct

    # Wall-clock time of insertion — useful for debugging late/duplicate runs.
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    card: Mapped[Card] = relationship(back_populates="snapshots")


# ---------------------------------------------------------------------------
# Data-access object
# ---------------------------------------------------------------------------


class PriceStore:
    """DAO (Data Access Object) for the card + price-snapshot tables.

    All database interactions in the rest of the application go through this
    class.  It owns the SQLAlchemy engine and exposes a small, purpose-built
    interface rather than leaking ORM details to callers.

    Usage pattern
    -------------
    The ingest CLI uses the context-manager ``session()`` to batch commits::

        store = PriceStore()
        with store.session() as session:
            for card in client.iter_cards():
                store.upsert_card(session, card)
                store.insert_price_snapshots(session, card["id"], card.get("tcgplayer"))
            session.commit()

    The API uses ``get_history()`` which opens and closes its own session
    internally so the caller doesn't need to think about transactions.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Create the engine and ensure all tables exist.

        Parameters
        ----------
        database_url:
            SQLAlchemy connection string.  Defaults to ``settings.database_url``
            (read from ``DATABASE_URL`` env var or .env file).  Passing an
            explicit URL is useful in tests (e.g. ``"sqlite:///:memory:"``).
        """
        from pathlib import Path

        # Prefer an explicit argument; fall back to the configured settings.
        url = database_url if database_url is not None else settings.database_url

        # Defensive: treat empty/whitespace string as unset. This prevents an
        # empty env var from overriding the intended default in pydantic.
        if not isinstance(url, str) or not url.strip():
            raise RuntimeError(
                "Invalid DATABASE_URL: value is empty. Set DATABASE_URL to a valid "
                "SQLAlchemy URL (e.g. 'sqlite:///data/prices.db' or 'postgresql://user:pass@host/db'), "
                "or remove the DATABASE_URL env var to use the built-in default."
            )

        # SQLite's ``ATTACH`` will silently fail if the parent directory doesn't
        # exist, so we create it here.  Postgres doesn't need this.
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(url, future=True)
        # ``create_all`` is a no-op if the tables already exist, so it's safe
        # to call on every startup.
        Base.metadata.create_all(self.engine)

    def upsert_card(self, session: Session, card_payload: dict[str, Any]) -> None:
        """Insert a new card or update all fields if it already exists.

        Parameters
        ----------
        session:
            An open SQLAlchemy Session (caller is responsible for commit).
        card_payload:
            Raw card dict from the pokemontcg.io API.  Must contain at least
            ``id`` and ``name``; all other fields are optional with sensible
            fallbacks.

        Notes
        -----
        We always overwrite all fields so that corrections upstream (e.g. a
        rarity string being fixed in the API) propagate to the local DB on the
        next ingest run.
        """
        existing = session.get(Card, card_payload["id"])

        # Parse the release date from the API's "YYYY/MM/DD" string format.
        release_str = card_payload.get("set", {}).get("releaseDate")
        release_date = datetime.strptime(release_str, "%Y/%m/%d").date() if release_str else None

        fields = {
            "id": card_payload["id"],
            "name": card_payload["name"],
            "set_id": card_payload.get("set", {}).get("id", ""),
            "set_name": card_payload.get("set", {}).get("name", ""),
            "rarity": card_payload.get("rarity"),
            "number": card_payload.get("number"),
            "release_date": release_date,
        }

        if existing is None:
            session.add(Card(**fields))
        else:
            # Update in-place — SQLAlchemy will emit an UPDATE on commit.
            for k, v in fields.items():
                setattr(existing, k, v)

    def insert_price_snapshots(
        self,
        session: Session,
        card_id: str,
        tcgplayer_payload: dict[str, Any] | None,
        snapshot_date: date | None = None,
    ) -> int:
        """Append today's price rows for every variant in a tcgplayer block.

        Parameters
        ----------
        session:
            An open SQLAlchemy Session (caller is responsible for commit).
        card_id:
            The pokemontcg.io card ID (e.g. ``"swsh1-1"``).
        tcgplayer_payload:
            The ``tcgplayer`` sub-dict from the API card object.  If ``None``
            or missing a ``prices`` block, this method is a no-op.
        snapshot_date:
            The date to stamp on these rows.  Defaults to today (``date.today()``).
            Override in tests or backfill scripts to insert historical data.

        Returns
        -------
        int
            Number of snapshot rows added to the session.

        Notes
        -----
        The API's ``tcgplayer.prices`` dict looks like::

            {
              "holofoil":        {"market": 12.34, "low": 8.0, "mid": 10.0, "high": 15.0},
              "reverseHolofoil": {"market": 3.10, ...},
              "normal":          {"market": 1.50, ...}
            }

        We create one ``PriceSnapshot`` row per variant.  The
        ``UniqueConstraint`` on (card_id, variant, snapshot_date) means that
        running this twice in a day will raise an ``IntegrityError`` for the
        duplicate rows — handle that at the session level if needed.
        """
        if not tcgplayer_payload:
            return 0

        prices = tcgplayer_payload.get("prices", {})
        if not prices:
            return 0

        snap = snapshot_date or date.today()
        inserted = 0

        for variant, p in prices.items():
            if not isinstance(p, dict):
                continue  # skip malformed entries
            session.add(
                PriceSnapshot(
                    card_id=card_id,
                    variant=variant,
                    snapshot_date=snap,
                    market=p.get("market"),
                    low=p.get("low"),
                    mid=p.get("mid"),
                    high=p.get("high"),
                    direct_low=p.get("directLow"),
                )
            )
            inserted += 1

        return inserted

    def get_history(self, card_id: str, variant: str = "holofoil") -> list[PriceSnapshot]:
        """Return all price snapshots for a card+variant, ordered by date ascending.

        Parameters
        ----------
        card_id:
            pokemontcg.io card identifier (e.g. ``"swsh1-1"``).
        variant:
            TCGPlayer price tier.  Defaults to ``"holofoil"`` since that's the
            most-tracked variant for collectible rares.

        Returns
        -------
        list[PriceSnapshot]
            Rows ordered oldest-first.  The API endpoint requires at least 30
            rows to make a forecast; 60+ is recommended for reliable results.
        """
        with Session(self.engine) as session:
            stmt = (
                select(PriceSnapshot)
                .where(
                    PriceSnapshot.card_id == card_id,
                    PriceSnapshot.variant == variant,
                )
                .order_by(PriceSnapshot.snapshot_date)
            )
            return list(session.scalars(stmt))

    def session(self) -> Session:
        """Return a new SQLAlchemy Session for use in a ``with`` block.

        The caller is responsible for calling ``session.commit()`` and the
        context manager handles ``session.close()``.

        Example::

            with store.session() as session:
                store.upsert_card(session, card)
                session.commit()
        """
        return Session(self.engine)
