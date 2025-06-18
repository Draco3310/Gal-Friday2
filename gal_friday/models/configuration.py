from datetime import UTC, datetime
from typing import TYPE_CHECKING
import uuid

from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from .base import Base

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from gal_friday.core.events import LogEvent


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

    def to_event(self) -> "LogEvent":
        """Convert this configuration to a ``LogEvent``."""
        from gal_friday.core.events import LogEvent

        context = {
            "config_pk": self.config_pk,
            "config_hash": self.config_hash,
            "is_active": self.is_active,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "source_module": self.__class__.__name__,
        }

        message = f"Configuration accessed/processed: PK={self.config_pk}, Hash={self.config_hash}"

        return LogEvent(
            source_module=self.__class__.__name__,
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            level="INFO",
            message=message,
            context=context,
        )
