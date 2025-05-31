import uuid
from datetime import datetime

from sqlalchemy import Column, BigInteger, String, Text, DateTime, JSON, Integer # Added Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID # For signal_id
from sqlalchemy.sql import func

from .base import Base
from gal_friday.core.events import LogEvent


class SystemLog(Base):
    __tablename__ = "system_logs"

    log_pk = Column(BigInteger, primary_key=True, autoincrement=True)
    log_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    source_module = Column(String(64), nullable=False, index=True)
    log_level = Column(String(10), nullable=False, index=True)  # CHECK constraint handled by application/DB
    message = Column(Text, nullable=False)
    
    # Contextual fields
    trading_pair = Column(String(16), nullable=True)
    signal_id = Column(PG_UUID(as_uuid=True), nullable=True) # Assuming this might link to signals.signal_id
    order_pk = Column(Integer, nullable=True) # Assuming this might link to orders.order_pk
    
    exception_type = Column(Text, nullable=True)
    stack_trace = Column(Text, nullable=True)
    context = Column(JSON, nullable=True) # For arbitrary additional context

    def __repr__(self) -> str: # Added -> str
        return (
            f"<SystemLog(log_pk={self.log_pk}, source_module='{self.source_module}', "
            f"log_level='{self.log_level}', message='{self.message[:50]}...')>"
        )

    def to_event(self) -> 'LogEvent': # Added to_event with type hints
        """Converts the SystemLog object to a LogEvent."""
        # Assuming LogEvent is importable from gal_friday.core.events
        # import uuid # Already imported
        # from datetime import datetime # Already imported
        # from gal_friday.core.events import LogEvent

        # Construct context, including optional fields if they exist
        event_context = self.context or {}
        if self.trading_pair:
            event_context['trading_pair'] = self.trading_pair
        if self.signal_id:
            event_context['signal_id'] = str(self.signal_id)
        if self.order_pk:
            event_context['order_pk'] = self.order_pk
        if self.exception_type:
            event_context['exception_type'] = self.exception_type
        if self.stack_trace: # Be cautious about event size with full stack traces
            event_context['stack_trace_preview'] = self.stack_trace[:200]


        event_data = {
            "source_module": self.source_module,
            # LogEvent usually generates its own event_id and timestamp upon creation.
            # If we want to preserve the original log's timestamp, LogEvent needs to allow it.
            "event_id": uuid.uuid4(), # New UUID for the event itself
            "timestamp": self.log_timestamp or datetime.utcnow(), # Use log_timestamp if available
            "level": self.log_level.upper(), # Ensure level is uppercase e.g. INFO, ERROR
            "message": self.message,
            "context": event_context,
        }
        # In a real implementation:
        # from gal_friday.core.events import LogEvent
        # return LogEvent(**event_data)

        # Returning dict for now
        return LogEvent(**event_data) # Should be LogEvent(**event_data)
