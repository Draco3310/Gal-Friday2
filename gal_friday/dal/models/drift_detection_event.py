"""SQLAlchemy model for the 'drift_detection_events' table."""

from datetime import datetime
from decimal import Decimal  # Though schema uses DECIMAL(10,6) which fits float, Decimal is safer
from uuid import UUID as PythonUUID

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,  # Not used in this specific table schema, but common
    Index,
    Numeric,
    String)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base
from typing import Any


class DriftDetectionEvent(Base):
    """Represents a drift detection event."""

    __tablename__ = "drift_detection_events"

    event_id: Mapped[PythonUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    # Assuming model_versions.model_id is UUID and ModelVersion model exists
    model_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("model_versions.model_id"), nullable=False, index=True, # Added index based on schema
    )
    drift_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True) # Added index
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    drift_score: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    is_significant: Mapped[bool | None] = mapped_column(Boolean, server_default="false", index=True) # Added index
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True) # Added index

    # Relationship to ModelVersion
    model_version = relationship("ModelVersion", backref="drift_events")

    __table_args__ = (
        Index("idx_drift_model", "model_id"), # Already covered by FK index
        Index("idx_drift_type", "drift_type"),
        Index("idx_drift_detected", "detected_at"),
        Index("idx_drift_significant", "is_significant"))

    def __repr__(self) -> str:
        return (
            f"<DriftDetectionEvent(event_id={self.event_id}, model_id={self.model_id}, "
            f"drift_type='{self.drift_type}', is_significant={self.is_significant})>"
        )