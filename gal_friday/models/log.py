import datetime
import uuid  # For LogEvent
from typing import Any  # For LogEvent context

from sqlalchemy import (  # DateTime removed as TIMESTAMPTZ is used
    TIMESTAMP,
    BigInteger,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column  # For Mapped type hints
from sqlalchemy.types import JSON  # Generic JSON type

from gal_friday.core.events import LogEvent

from .base import Base


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True) # Assuming BigInteger maps to int
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.utcnow)
    logger_name: Mapped[str] = mapped_column(String(255), nullable=False)
    level_name: Mapped[str] = mapped_column(String(50), nullable=False)
    level_no: Mapped[int] = mapped_column(Integer, nullable=False) # Added Mapped for consistency
    message: Mapped[str] = mapped_column(Text, nullable=False) # Changed to Mapped[str]
    pathname: Mapped[str | None] = mapped_column(Text, nullable=True) # Added Mapped for consistency
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True) # Added Mapped for consistency
    lineno: Mapped[int | None] = mapped_column(Integer, nullable=True)
    func_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True) # Assuming JSON maps to dict
    exception_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str: # Added -> str
        return f"<Log(id={self.id}, name='{self.logger_name}', level='{self.level_name}')>"

    def to_event(self) -> "LogEvent": # Added to_event with type hints
        """Converts the Log object to a LogEvent."""
        # Assuming LogEvent is importable from gal_friday.core.events
        # from gal_friday.core.events import LogEvent

        # Prepare context, merging context_json with other relevant fields if desired
        event_context: dict[str, Any] = self.context_json or {}
        if self.pathname:
            event_context["pathname"] = self.pathname
        if self.filename:
            event_context["filename"] = self.filename
        if self.lineno is not None:
            event_context["lineno"] = self.lineno
        if self.func_name:
            event_context["func_name"] = self.func_name
        if self.exception_text: # Be cautious with potentially large exception texts in events
            event_context["exception_preview"] = self.exception_text[:200]

        # Call LogEvent constructor with explicit arguments
        return LogEvent(
            source_module=self.logger_name,
            event_id=uuid.uuid4(),  # Generate a new UUID for the event
            timestamp=self.timestamp if self.timestamp is not None else datetime.datetime.utcnow(),
            level=self.level_name.upper(),
            message=self.message,
            context=event_context,
        )
