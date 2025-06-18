"""Event log storage for event sourcing."""


from sqlalchemy import JSON, Column, DateTime, Index, String
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID

from gal_friday.dal.models.models_base import Base


class EventLog(Base):
    """Event log storage for event sourcing."""

    __tablename__ = "event_logs"

    event_id = Column(PostgresUUID(as_uuid=True), primary_key=True)
    event_type = Column(String(100), nullable=False)
    source_module = Column(String(100), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    data = Column(JSON, nullable=False)

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_event_logs_timestamp", "timestamp"),
        Index("idx_event_logs_event_type", "event_type"),
        Index("idx_event_logs_correlation_id", "data", postgresql_using="gin"))
