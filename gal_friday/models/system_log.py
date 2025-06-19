from datetime import UTC, datetime
from typing import Any
import uuid

from sqlalchemy import JSON, BigInteger, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import (
    UUID as PG_UUID,  # Added JSONB for potential use
)
from sqlalchemy.orm import Mapped, mapped_column  # Added Mapped, mapped_column
from sqlalchemy.sql import func

from gal_friday.core.events import LogEvent

from .base import Base


class SystemLog(Base):
    __tablename__ = "system_logs"

    log_pk: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    log_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        server_default=func.now(), index=True,
    )
    source_module: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    log_level: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    # Contextual fields
    trading_pair: Mapped[str | None] = mapped_column(String(16), nullable=True)
    signal_id: Mapped[uuid.UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    order_pk: Mapped[int | None] = mapped_column(Integer, nullable=True)

    exception_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    stack_trace: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Using generic JSON for broader compatibility, can be switched to JSONB if PG-specific features are needed.
    context: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str: # Added -> str
        return (
            f"<SystemLog(log_pk={self.log_pk}, source_module='{self.source_module}', "
            f"log_level='{self.log_level}', message='{self.message[:50]}...')>"
        )

    def to_event(self) -> "LogEvent": # Added to_event with type hints
        """Converts the SystemLog object to a LogEvent."""
        # Assuming LogEvent is importable from gal_friday.core.events
        # import uuid # Already imported
        # from datetime import datetime # Already imported
        # from gal_friday.core.events import LogEvent

        # Construct context, including optional fields if they exist
        event_context: dict[str, Any] = self.context or {}
        if self.trading_pair:
            event_context["trading_pair"] = self.trading_pair
        if self.signal_id:
            event_context["signal_id"] = str(self.signal_id)
        if self.order_pk:
            event_context["order_pk"] = self.order_pk
        if self.exception_type:
            event_context["exception_type"] = self.exception_type
        if self.stack_trace: # Be cautious about event size with full stack traces
            event_context["stack_trace_preview"] = self.stack_trace[:200]

        # Call LogEvent constructor with explicit keyword arguments
        # This helps mypy with type checking against the LogEvent constructor signature.
        return LogEvent(
            source_module=self.source_module,
            event_id=uuid.uuid4(),  # Generate a new UUID for this event instance
            timestamp=self.log_timestamp or datetime.now(UTC), # Use the log's timestamp if available
            level=self.log_level.upper(),
            message=self.message,
            context=event_context)
