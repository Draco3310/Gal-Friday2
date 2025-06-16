from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models_base import Base
from typing import Any


class DataQualityIssue(Base):
    """Persistent record of critical data quality issues."""

    __tablename__ = "data_quality_issues"

    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    alert_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    trading_pair: Mapped[str] = mapped_column(String(20), nullable=False)
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    context_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    suggested_action: Mapped[str | None] = mapped_column(String(100), nullable=True)
    publication_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    issue_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    persisted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_data_quality_issue_timestamp", "issue_timestamp"),
        Index("idx_data_quality_issue_trading_pair", "trading_pair"))

    def __repr__(self) -> str:  # pragma: no cover - simple repr
        return (
            f"<DataQualityIssue(id={self.id}, trading_pair='{self.trading_pair}', "
            f"alert_type='{self.alert_type}', severity='{self.severity}')>"
        )