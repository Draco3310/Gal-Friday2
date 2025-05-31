import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, Numeric, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base
from gal_friday.core.events import MarketDataTradeEvent


class Trade(Base):
    __tablename__ = "trades"

    trade_pk = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(PG_UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4, index=True)
    signal_id = Column(PG_UUID(as_uuid=True), ForeignKey("signals.signal_id"), index=True)
    
    trading_pair = Column(String(16), nullable=False, index=True)
    exchange = Column(String(32), nullable=False)
    strategy_id = Column(String(64), nullable=False)
    side = Column(String(4), nullable=False)  # Side of the entry
    
    entry_order_pk = Column(Integer, ForeignKey("orders.order_pk"), nullable=True)
    exit_order_pk = Column(Integer, ForeignKey("orders.order_pk"), nullable=True)
    
    entry_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    exit_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    quantity = Column(Numeric(18, 8), nullable=False)
    average_entry_price = Column(Numeric(18, 8), nullable=False)
    average_exit_price = Column(Numeric(18, 8), nullable=False)
    total_commission = Column(Numeric(18, 8), nullable=False) # Should be in quote currency
    realized_pnl = Column(Numeric(18, 8), nullable=False) # Quote currency
    realized_pnl_pct = Column(Float, nullable=False)
    exit_reason = Column(String(32), nullable=False)

    # Relationships
    signal = relationship("Signal", back_populates="trade")
    
    entry_order = relationship(
        "Order", 
        foreign_keys=[entry_order_pk], 
        backref="trade_as_entry", # Use backref to avoid conflict if Order directly relates to Trade
        # Or ensure Order.trade_entry uses a specific foreign_keys if defined there
    )
    exit_order = relationship(
        "Order", 
        foreign_keys=[exit_order_pk],
        backref="trade_as_exit",
    )

    def __repr__(self) -> str: # Added -> str
        return (
            f"<Trade(trade_id={self.trade_id}, trading_pair='{self.trading_pair}', "
            f"strategy_id='{self.strategy_id}', realized_pnl={self.realized_pnl})>"
        )

    def to_event(self) -> 'MarketDataTradeEvent': # Added to_event with type hints
        """Converts the Trade object to a MarketDataTradeEvent.
        This represents the entry part of the trade as a market event.
        Exit would be a separate event if needed.
        """
        # Assuming MarketDataTradeEvent is importable from gal_friday.core.events
        # import uuid # Already imported
        # from datetime import datetime # Already imported
        # from decimal import Decimal # For type conversion if necessary, sqlalchemy handles it
        # from gal_friday.core.events import MarketDataTradeEvent

        # The Trade model represents a completed round-trip trade (entry and exit).
        # A MarketDataTradeEvent typically represents a single execution (fill).
        # Here, we can represent the entry fill as an example.
        # A more complete system might generate two MarketDataTradeEvents (one for entry, one for exit)
        # or use a different event type for completed round-trip trades.

        event_data = {
            "source_module": self.__class__.__name__,
            "event_id": uuid.uuid4(), # New event ID for this specific event
            "timestamp": datetime.utcnow(), # Event creation time
            "trading_pair": self.trading_pair,
            "exchange": self.exchange,
            "timestamp_exchange": self.entry_timestamp, # Timestamp of the entry
            "price": self.average_entry_price,
            "volume": self.quantity,
            "side": self.side, # Side of the entry trade
            "trade_id": str(self.trade_id), # Use the Trade's own ID or a fill ID if available
        }
        # In a real implementation:
        # from gal_friday.core.events import MarketDataTradeEvent
        # return MarketDataTradeEvent(**event_data)

        # Returning dict for now
        return event_data # Should be MarketDataTradeEvent(**event_data)
