import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, Numeric, Float, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class Signal(Base):
    __tablename__ = "signals"

    signal_id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trading_pair = Column(String(16), nullable=False)
    exchange = Column(String(32), nullable=False)
    strategy_id = Column(String(64), nullable=False)
    side = Column(String(4), nullable=False)  # CHECK constraint handled by application/DB
    entry_type = Column(String(10), nullable=False)
    proposed_entry_price = Column(Numeric(18, 8), nullable=True)
    proposed_sl_price = Column(Numeric(18, 8), nullable=False)
    proposed_tp_price = Column(Numeric(18, 8), nullable=False)
    prediction_event_id = Column(PG_UUID(as_uuid=True), nullable=True)
    prediction_value = Column(Float, nullable=True)
    status = Column(String(10), nullable=False)  # CHECK constraint handled by application/DB
    rejection_reason = Column(Text, nullable=True)
    risk_check_details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Relationships
    orders = relationship("Order", back_populates="signal", cascade="all, delete-orphan")
    trade = relationship("Trade", back_populates="signal", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str: # Added -> str
        return (
            f"<Signal(signal_id={self.signal_id}, trading_pair='{self.trading_pair}', "
            f"strategy_id='{self.strategy_id}', status='{self.status}')>"
        )

    def to_event(self) -> 'TradeSignalProposedEvent': # Added to_event with type hints
        """Converts the Signal object to a TradeSignalProposedEvent."""
        # Assuming TradeSignalProposedEvent is importable from gal_friday.core.events
        # import uuid # Already imported
        # from datetime import datetime # Already imported
        # from decimal import Decimal # For type conversion if necessary
        # from gal_friday.core.events import TradeSignalProposedEvent

        event_data = {
            "source_module": self.__class__.__name__,
            "event_id": uuid.uuid4(), # New event ID
            "timestamp": datetime.utcnow(), # Event creation time
            "signal_id": self.signal_id, # Use the Signal's own ID
            "trading_pair": self.trading_pair,
            "exchange": self.exchange,
            "side": self.side,
            "entry_type": self.entry_type,
            "proposed_entry_price": self.proposed_entry_price,
            "proposed_sl_price": self.proposed_sl_price,
            "proposed_tp_price": self.proposed_tp_price,
            "strategy_id": self.strategy_id,
            "triggering_prediction_event_id": self.prediction_event_id,
            # 'triggering_prediction' field in event might need more data if available
            "triggering_prediction": {"value": self.prediction_value} if self.prediction_value is not None else None,
        }
        # In a real implementation:
        # from gal_friday.core.events import TradeSignalProposedEvent
        # return TradeSignalProposedEvent(**event_data)

        # Returning dict for now
        return event_data # Should be TradeSignalProposedEvent(**event_data)
