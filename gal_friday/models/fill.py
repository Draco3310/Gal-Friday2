from datetime import datetime

from sqlalchemy import Column, Integer, String, Numeric, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .base import Base


class Fill(Base):
    __tablename__ = "fills"

    fill_pk = Column(Integer, primary_key=True, autoincrement=True)
    fill_id = Column(String(64), nullable=True)  # Exchange fill ID
    order_pk = Column(Integer, ForeignKey("orders.order_pk"), nullable=False, index=True)
    exchange_order_id = Column(String(64), index=True) # From Order, denormalized for easier query
    
    trading_pair = Column(String(16), nullable=False)
    exchange = Column(String(32), nullable=False)
    side = Column(String(4), nullable=False)
    quantity_filled = Column(Numeric(18, 8), nullable=False)
    fill_price = Column(Numeric(18, 8), nullable=False)
    commission = Column(Numeric(18, 8), nullable=False)
    commission_asset = Column(String(16), nullable=False)
    liquidity_type = Column(String(10), nullable=True)  # 'MAKER' or 'TAKER'
    filled_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Relationships
    order = relationship("Order", back_populates="fills")

    # Constraints
    __table_args__ = (
        UniqueConstraint('exchange', 'fill_id', name='uq_exchange_fill_id'),
    )

    def __repr__(self):
        return (
            f"<Fill(fill_pk={self.fill_pk}, fill_id='{self.fill_id}', order_pk={self.order_pk}, "
            f"quantity_filled={self.quantity_filled}, fill_price={self.fill_price})>"
        )
