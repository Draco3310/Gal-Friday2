"""SQLAlchemy model for the 'positions' table."""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Index, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class Position(Base):
    """Represents a trading position."""

    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    trading_pair: Mapped[str] = mapped_column(String(20), nullable=False, index=True) # Added index based on schema
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    current_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    realized_pnl: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), server_default="0")
    unrealized_pnl: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), server_default="0")
    opened_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), index=True, # Added index based on schema
    )
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool | None] = mapped_column(
        Boolean, server_default="true", index=True, # Added index based on schema
    )

    # Relationship to Order - one position can have multiple contributing orders
    orders = relationship("Order", back_populates="position", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_positions_pair", "trading_pair"),
        Index("idx_positions_active", "is_active"),
        Index("idx_positions_opened_at", "opened_at"))

    def __repr__(self) -> str:
        return (
            f"<Position(id={self.id}, trading_pair='{self.trading_pair}', "
            f"side='{self.side}', is_active={self.is_active})>"
        )