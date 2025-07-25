"""SQLAlchemy model for the 'position_adjustments' table."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID as PythonUUID

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class PositionAdjustment(Base):
    """Represents an adjustment made to a position during reconciliation."""

    __tablename__ = "position_adjustments"

    adjustment_id: Mapped[PythonUUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.uuid_generate_v4())
    # reconciliation_id is a ForeignKey to reconciliation_events.reconciliation_id
    reconciliation_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("reconciliation_events.reconciliation_id"),
        nullable=False,
        index=True)
    trading_pair: Mapped[str] = mapped_column(String(20), nullable=False)  # From 003, added index
    adjustment_type: Mapped[str] = mapped_column(String(50), nullable=False)  # From 003
    old_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    new_value: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    adjusted_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        index=True,  # Added index based on 003
    )

    # Relationship to ReconciliationEvent
    reconciliation_event = relationship("ReconciliationEvent", back_populates="adjustments")

    __table_args__ = (
        Index("idx_adjustments_reconciliation", "reconciliation_id"),
        Index("idx_adjustments_pair", "trading_pair"),  # From 003
        Index("idx_adjustments_timestamp", "adjusted_at"),  # From 003
    )

    def __repr__(self) -> str:
        return (
            f"<PositionAdjustment(adjustment_id={self.adjustment_id}, "
            f"reconciliation_id={self.reconciliation_id}, type='{self.adjustment_type}')>"
        )
