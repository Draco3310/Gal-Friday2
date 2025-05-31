import datetime

import uuid # For LogEvent
from typing import Any # For LogEvent context
from sqlalchemy import Column, BigInteger, String, Integer, Text, TIMESTAMP # DateTime removed as TIMESTAMPTZ is used
from sqlalchemy.types import JSON  # Generic JSON type
from sqlalchemy.orm import Mapped, mapped_column # For Mapped type hints
from sqlalchemy.sql import func # For server_default func.now() if needed

from .base import Base
from gal_friday.core.events import LogEvent


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True) # Assuming BigInteger maps to int
    timestamp: Mapped[datetime.datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, default=datetime.datetime.utcnow)
    logger_name: Mapped[str] = mapped_column(String(255), nullable=False)
    level_name: Mapped[str] = mapped_column(String(50), nullable=False)
    level_no = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    pathname = Column(Text, nullable=True)
    filename = Column(String(255), nullable=True)
    lineno: Mapped[int | None] = mapped_column(Integer, nullable=True)
    func_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True) # Assuming JSON maps to dict
    exception_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str: # Added -> str
        return f"<Log(id={self.id}, name='{self.logger_name}', level='{self.level_name}')>"

    def to_event(self) -> 'LogEvent': # Added to_event with type hints
        """Converts the Log object to a LogEvent."""
        # Assuming LogEvent is importable from gal_friday.core.events
        # from gal_friday.core.events import LogEvent

        # Prepare context, merging context_json with other relevant fields if desired
        event_context = self.context_json or {}
        if self.pathname:
            event_context['pathname'] = self.pathname
        if self.filename:
            event_context['filename'] = self.filename
        if self.lineno is not None:
            event_context['lineno'] = self.lineno
        if self.func_name:
            event_context['func_name'] = self.func_name
        if self.exception_text: # Be cautious with potentially large exception texts in events
            event_context['exception_preview'] = self.exception_text[:200]

        event_data = {
            "source_module": self.logger_name, # logger_name seems more appropriate for source_module
            "event_id": uuid.uuid4(),
            "timestamp": self.timestamp or datetime.datetime.utcnow(), # Use log's timestamp
            "level": self.level_name.upper(),
            "message": self.message,
            "context": event_context,
        }
        # In a real implementation:
        # from gal_friday.core.events import LogEvent
        # return LogEvent(**event_data)

        # Returning dict for now
        return event_data # Should be LogEvent(**event_data)
