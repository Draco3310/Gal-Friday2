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

    def __repr__(self):
        return (
            f"<Signal(signal_id={self.signal_id}, trading_pair='{self.trading_pair}', "
            f"strategy_id='{self.strategy_id}', status='{self.status}')>"
        )
