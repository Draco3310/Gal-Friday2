import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, Numeric, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class Order(Base):
    __tablename__ = "orders"

    order_pk = Column(Integer, primary_key=True, autoincrement=True)
    client_order_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4, index=True)
    exchange_order_id = Column(String(64), unique=True, nullable=True, index=True)
    signal_id = Column(PG_UUID(as_uuid=True), ForeignKey("signals.signal_id"), index=True)
    
    trading_pair = Column(String(16), nullable=False)
    exchange = Column(String(32), nullable=False)
    side = Column(String(4), nullable=False)  # CHECK constraint handled by application/DB
    order_type = Column(String(16), nullable=False)
    quantity_ordered = Column(Numeric(18, 8), nullable=False)
    limit_price = Column(Numeric(18, 8), nullable=True)
    stop_price = Column(Numeric(18, 8), nullable=True)
    status = Column(String(20), nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    submitted_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationships
    signal = relationship("Signal", back_populates="orders")
    fills = relationship("Fill", back_populates="order", cascade="all, delete-orphan")
    
    # Relationships to Trade table (will be fully defined in Trade model via back_populates)
    # trade_entry: Mapped[Optional["Trade"]] = relationship(foreign_keys="Trade.entry_order_pk", back_populates="entry_order")
    # trade_exit: Mapped[Optional["Trade"]] = relationship(foreign_keys="Trade.exit_order_pk", back_populates="exit_order")
    # The above lines are commented out as they are defined by the 'Trade' model's back_populates

    def __repr__(self):
        return (
            f"<Order(order_pk={self.order_pk}, client_order_id={self.client_order_id}, "
            f"exchange_order_id='{self.exchange_order_id}', status='{self.status}')>"
        )
