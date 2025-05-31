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

    def __repr__(self) -> str: # Added -> str
        return (
            f"<Order(order_pk={self.order_pk}, client_order_id={self.client_order_id}, "
            f"exchange_order_id='{self.exchange_order_id}', status='{self.status}')>"
        )

    def to_event(self) -> 'ExecutionReportEvent': # Added to_event with type hints
        """Converts the Order object to an ExecutionReportEvent."""
        # Assuming ExecutionReportEvent is importable from gal_friday.core.events
        # import uuid # Already imported
        # from datetime import datetime # Already imported
        # from decimal import Decimal
        # from gal_friday.core.events import ExecutionReportEvent

        # Calculate quantity_filled and average_fill_price from associated fills
        quantity_filled_val = Decimal("0")
        total_fill_value = Decimal("0")
        commission_val = Decimal("0")
        commission_asset_val = None

        if self.fills:
            for fill in self.fills:
                quantity_filled_val += fill.quantity_filled
                total_fill_value += fill.quantity_filled * fill.fill_price
                commission_val += fill.commission
                if commission_asset_val is None and fill.commission_asset: # Take first commission asset
                    commission_asset_val = fill.commission_asset

        average_fill_price_val = total_fill_value / quantity_filled_val if quantity_filled_val > 0 else None

        event_data = {
            "source_module": self.__class__.__name__,
            "event_id": uuid.uuid4(), # New event ID
            "timestamp": datetime.utcnow(), # Event creation time
            "exchange_order_id": self.exchange_order_id,
            "trading_pair": self.trading_pair,
            "exchange": self.exchange,
            "order_status": self.status.upper(), # Ensure uppercase
            "order_type": self.order_type.upper(), # Ensure uppercase
            "side": self.side.upper(), # Ensure uppercase
            "quantity_ordered": self.quantity_ordered,
            "signal_id": self.signal_id,
            "client_order_id": str(self.client_order_id), # Ensure string
            "quantity_filled": quantity_filled_val,
            "average_fill_price": average_fill_price_val,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "commission": commission_val,
            "commission_asset": commission_asset_val,
            "timestamp_exchange": self.last_updated_at or self.submitted_at or self.created_at, # Best available exchange-related timestamp
            "error_message": self.error_message,
        }
        # In a real implementation:
        # from gal_friday.core.events import ExecutionReportEvent
        # from decimal import Decimal # at top
        # return ExecutionReportEvent(**event_data)

        # Returning dict for now
        return event_data # Should be ExecutionReportEvent(**event_data)
