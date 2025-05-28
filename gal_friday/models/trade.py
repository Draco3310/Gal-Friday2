import uuid
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, Numeric, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


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

    def __repr__(self):
        return (
            f"<Trade(trade_id={self.trade_id}, trading_pair='{self.trading_pair}', "
            f"strategy_id='{self.strategy_id}', realized_pnl={self.realized_pnl})>"
        )
