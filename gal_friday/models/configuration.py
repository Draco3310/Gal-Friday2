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

    def __repr__(self):
        return (
            f"<Configuration(config_pk={self.config_pk}, config_hash='{self.config_hash}', "
            f"is_active={self.is_active})>"
        )
