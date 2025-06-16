"""SQLAlchemy model for the 'orders' table."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class Order(Base):
    """Represents an order in the system."""

    __tablename__ = "orders"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    # Assuming trade_signals table will have a TradeSignal model and 'id' as primary key
    signal_id: Mapped[UUID] = mapped_column(
        ForeignKey("trade_signals.id"), nullable=False, index=True)
    # Foreign key to positions table - nullable since orders may not immediately affect positions
    position_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("positions.id"), nullable=True, index=True)
    trading_pair: Mapped[str] = mapped_column(String(20), nullable=False)
    exchange: Mapped[str] = mapped_column(String(50), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filled_quantity: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), server_default="0")
    average_fill_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), nullable=True)
    commission: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), index=True)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, onupdate=func.current_timestamp())

    # Relationship to TradeSignal (assuming TradeSignal model will be defined)
    signal = relationship("TradeSignal", back_populates="orders")
    
    # Relationship to Position - many orders can contribute to one position
    position = relationship("Position", back_populates="orders")

    __table_args__ = (
        Index("idx_orders_signal_id", "signal_id"),
        Index("idx_orders_position_id", "position_id"),
        Index("idx_orders_status", "status"),
        Index("idx_orders_created_at", "created_at"))

    def __repr__(self) -> str:
        return (
            f"<Order(id={self.id}, trading_pair='{self.trading_pair}', "
            f"status='{self.status}')>"
        )