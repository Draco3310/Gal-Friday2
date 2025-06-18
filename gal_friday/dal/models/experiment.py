"""SQLAlchemy model for the 'experiments' table."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID as PythonUUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class Experiment(Base):
    """Represents an A/B testing experiment."""

    __tablename__ = "experiments"

    experiment_id: Mapped[PythonUUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Assuming model_versions.model_id is UUID and ModelVersion model exists
    control_model_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("model_versions.model_id"), nullable=False)
    treatment_model_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("model_versions.model_id"), nullable=False)
    allocation_strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    traffic_split: Mapped[Decimal] = mapped_column(Numeric(3, 2), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    min_samples_per_variant: Mapped[int | None] = mapped_column(
        Integer, server_default="1000")
    primary_metric: Mapped[str] = mapped_column(String(100), nullable=False)
    secondary_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    confidence_level: Mapped[Decimal | None] = mapped_column(
        Numeric(3, 2), server_default="0.95")
    minimum_detectable_effect: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4), server_default="0.01")
    max_loss_threshold: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="created", index=True)
    completion_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    results: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    config_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, server_default=func.current_timestamp())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships (assuming ModelVersion model is defined)
    control_model = relationship(
        "ModelVersion", foreign_keys=[control_model_id], backref="controlled_experiments")
    treatment_model = relationship(
        "ModelVersion", foreign_keys=[treatment_model_id], backref="treatment_experiments")

    # Consider relationships to experiment_assignments and experiment_outcomes if those models are also created.
    # assignments = relationship("ExperimentAssignment", back_populates="experiment")
    # outcomes = relationship("ExperimentOutcome", back_populates="experiment")


    __table_args__ = (
        CheckConstraint("traffic_split > 0 AND traffic_split < 1", name="chk_traffic_split_range"),
        Index("idx_experiments_status", "status"),
        Index("idx_experiments_dates", "start_time", "end_time"),
        Index("idx_experiments_models", "control_model_id", "treatment_model_id"))

    def __repr__(self) -> str:
        return f"<Experiment(experiment_id={self.experiment_id}, name='{self.name}', status='{self.status}')>"
