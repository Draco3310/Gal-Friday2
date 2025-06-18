"""SQLAlchemy model for the 'retraining_jobs' table."""

from datetime import datetime
from typing import Any
from uuid import UUID as PythonUUID

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class RetrainingJob(Base):
    """Represents a model retraining job."""

    __tablename__ = "retraining_jobs"

    job_id: Mapped[PythonUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    # Assuming model_versions.model_id is UUID and ModelVersion model exists
    model_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("model_versions.model_id"), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    trigger: Mapped[str] = mapped_column(String(50), nullable=False)
    drift_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="pending", index=True)
    start_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Assuming new_model_id also refers to model_versions.model_id
    new_model_id: Mapped[PythonUUID | None] = mapped_column(
        ForeignKey("model_versions.model_id"), nullable=True)
    performance_comparison: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.current_timestamp(), index=True)
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

    # Relationships
    original_model_version = relationship(
        "ModelVersion", foreign_keys=[model_id], backref="retraining_jobs_triggered")
    new_model_version = relationship(
        "ModelVersion", foreign_keys=[new_model_id], backref="retraining_jobs_created")

    __table_args__ = (
        Index("idx_retraining_model", "model_id"), # Already covered by ForeignKey index for model_id
        Index("idx_retraining_status", "status"),
        Index("idx_retraining_created", "created_at"))

    def __repr__(self) -> str:
        return f"<RetrainingJob(job_id={self.job_id}, model_name='{self.model_name}', status='{self.status}')>"
