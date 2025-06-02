"""SQLAlchemy model for the 'logs' table."""

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func  # For server_default=func.now()

from .models_base import Base


class Log(Base):
    """Represents a log entry in the database."""

    __tablename__ = "logs" # Matches table_name in AsyncPostgresHandler and schema

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True) # BIGSERIAL
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    ) # TIMESTAMPTZ, default NOW()
    logger_name: Mapped[str] = mapped_column(String(255), nullable=False)
    level_name: Mapped[str] = mapped_column(String(50), nullable=False)
    level_no: Mapped[int] = mapped_column(Integer, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    pathname: Mapped[str | None] = mapped_column(Text, nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lineno: Mapped[int | None] = mapped_column(Integer, nullable=True)
    func_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    context_json: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True
    )  # Stored as dict, maps to JSONB
    exception_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        """Return a string representation of the Log entry."""
        return (
            f"<Log(id={self.id}, timestamp='{self.timestamp}', "
            f"logger_name='{self.logger_name}', level='{self.level_name}')>"
        )
