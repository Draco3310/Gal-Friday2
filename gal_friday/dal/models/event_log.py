"""Event log storage for event sourcing."""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID  # noqa: N811
from sqlalchemy.orm import Mapped, mapped_column

from gal_friday.dal.models.models_base import Base


class EventLog(Base):
    """Event log storage for event sourcing."""

    __tablename__ = "event_logs"

    event_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_module: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_event_logs_timestamp", "timestamp"),
        Index("idx_event_logs_event_type", "event_type"),
        Index("idx_event_logs_correlation_id", "data", postgresql_using="gin"),
    )
