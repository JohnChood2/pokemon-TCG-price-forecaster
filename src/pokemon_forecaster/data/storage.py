"""Persistent storage for cards, price snapshots, and set metadata.

Schema is intentionally simple — one row per (card_id, variant, snapshot_date).
This shape is what most forecasting libraries expect: a long-format time series.
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


class Base(DeclarativeBase):
    pass


class Card(Base):
    __tablename__ = "cards"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    set_id: Mapped[str] = mapped_column(String, index=True)
    set_name: Mapped[str] = mapped_column(String)
    rarity: Mapped[str | None] = mapped_column(String, nullable=True)
    number: Mapped[str | None] = mapped_column(String, nullable=True)
    release_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    snapshots: Mapped[list[PriceSnapshot]] = relationship(back_populates="card")


class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"
    __table_args__ = (
        UniqueConstraint("card_id", "variant", "snapshot_date", name="uq_card_variant_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    variant: Mapped[str] = mapped_column(String, index=True)  # 'normal', 'holofoil', etc
    snapshot_date: Mapped[date] = mapped_column(Date, index=True)
    market: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    direct_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    card: Mapped[Card] = relationship(back_populates="snapshots")


class PriceStore:
    """Thin DAO — the rest of the app should not import SQLAlchemy directly."""

    def __init__(self, database_url: str | None = None) -> None:
        from pathlib import Path
        url = database_url or settings.database_url
        # SQLite won't create parent directories; do it ourselves.
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)

    def upsert_card(self, session: Session, card_payload: dict[str, Any]) -> None:
        """Insert or update a card row from an API payload."""
        existing = session.get(Card, card_payload["id"])
        release_str = card_payload.get("set", {}).get("releaseDate")
        release_date = (
            datetime.strptime(release_str, "%Y/%m/%d").date() if release_str else None
        )
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
            for k, v in fields.items():
                setattr(existing, k, v)

    def insert_price_snapshots(
        self,
        session: Session,
        card_id: str,
        tcgplayer_payload: dict[str, Any] | None,
        snapshot_date: date | None = None,
    ) -> int:
        """Pull every variant's prices out of a tcgplayer block and insert rows.

        The API returns a `tcgplayer.prices` dict like:
            {"holofoil": {"market": 12.34, "low": 8.0, ...}, "normal": {...}}
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
                continue
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
        with Session(self.engine) as session:
            stmt = (
                select(PriceSnapshot)
                .where(PriceSnapshot.card_id == card_id, PriceSnapshot.variant == variant)
                .order_by(PriceSnapshot.snapshot_date)
            )
            return list(session.scalars(stmt))

    def session(self) -> Session:
        return Session(self.engine)
