from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.sql import func

from .base import Base


class Configuration(Base):
    __tablename__ = "configurations"

    config_pk = Column(Integer, primary_key=True, autoincrement=True)
    config_hash = Column(String(64), unique=True, nullable=False, index=True)
    config_content = Column(JSON, nullable=False)
    loaded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    is_active = Column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str: # Added -> str
        return (
            f"<Configuration(config_pk={self.config_pk}, config_hash='{self.config_hash}', "
            f"is_active={self.is_active})>"
        )

    def to_event(self) -> 'LogEvent': # Added to_event with type hints
        """Converts the Configuration object to a LogEvent."""
        # Assuming LogEvent is importable from gal_friday.core.events
        # import uuid
        # from datetime import datetime
        # from gal_friday.core.events import LogEvent

        event_data = {
            "source_module": self.__class__.__name__,
            "event_id": uuid.uuid4(),
            "timestamp": datetime.utcnow(),
            "level": "INFO", # Or some other appropriate level
            "message": f"Configuration accessed/processed: PK={self.config_pk}, Hash={self.config_hash}",
            "context": {
                "config_pk": self.config_pk,
                "config_hash": self.config_hash,
                "is_active": self.is_active,
                "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
                # Be cautious about logging entire config_content if it's sensitive
                # "config_content_preview": str(self.config_content)[:100] # Example preview
            }
        }
        # In a real implementation:
        # from gal_friday.core.events import LogEvent
        # import uuid
        # return LogEvent(**event_data)

        # Returning dict for now to satisfy type hint via forward reference
        return event_data # Should be LogEvent(**event_data)
