import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any  # Added Any, List, TYPE_CHECKING

from sqlalchemy import JSON, DateTime, Float, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship  # Added Mapped, mapped_column
from sqlalchemy.sql import func

from gal_friday.core.events import TradeSignalProposedEvent

from .base import Base

if TYPE_CHECKING:
    from .order import Order  # For relationship hint
    from .trade import Trade  # For relationship hint


class Signal(Base):
    __tablename__ = "signals"

    signal_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trading_pair: Mapped[str] = mapped_column(String(16), nullable=False)
    exchange: Mapped[str] = mapped_column(String(32), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    entry_type: Mapped[str] = mapped_column(String(10), nullable=False)
    proposed_entry_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 8), nullable=True)
    proposed_sl_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    proposed_tp_price: Mapped[Decimal] = mapped_column(Numeric(18, 8), nullable=False)
    prediction_event_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    prediction_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(10), nullable=False)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_check_details: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Relationships
    orders: Mapped[list["Order"]] = relationship("Order", back_populates="signal", cascade="all, delete-orphan")
    trade: Mapped["Trade | None"] = relationship("Trade", back_populates="signal", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str: # Added -> str
        return (
            f"<Signal(signal_id={self.signal_id}, trading_pair='{self.trading_pair}', "
            f"strategy_id='{self.strategy_id}', status='{self.status}')>"
        )

    def to_event(self) -> "TradeSignalProposedEvent":
        """Convert this Signal into a TradeSignalProposedEvent."""
        event_data = {
            "source_module": self.__class__.__name__,
            "event_id": uuid.uuid4(),
            "timestamp": datetime.utcnow(),
            "signal_id": self.signal_id,
            "trading_pair": self.trading_pair,
            "exchange": self.exchange,
            "side": self.side,
            "entry_type": self.entry_type,
            "proposed_entry_price": self.proposed_entry_price,
            "proposed_sl_price": self.proposed_sl_price,
            "proposed_tp_price": self.proposed_tp_price,
            "strategy_id": self.strategy_id,
            "triggering_prediction_event_id": self.prediction_event_id,
            "triggering_prediction": (
                {"value": self.prediction_value} if self.prediction_value is not None else None
            ),
        }

        return TradeSignalProposedEvent(**event_data)
