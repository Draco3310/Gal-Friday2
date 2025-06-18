"""SQLAlchemy model for the 'experiment_outcomes' table."""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID as PythonUUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Numeric, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class ExperimentOutcome(Base):
    """Represents an outcome recorded for an experiment event."""

    __tablename__ = "experiment_outcomes"

    outcome_id: Mapped[PythonUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    experiment_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("experiments.experiment_id"), index=True)
    # Assuming event_id is a generic UUID, not necessarily FK to experiment_assignments.event_id
    # If it should be, then add ForeignKey("experiment_assignments.event_id")
    event_id: Mapped[PythonUUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    variant: Mapped[str] = mapped_column(String(20), nullable=False, index=True) # Added index based on schema
    outcome_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    correct_prediction: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    signal_generated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    trade_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True) # Added index

    # Relationship to Experiment
    experiment = relationship("Experiment") # Add back_populates="outcomes" to Experiment if needed

    __table_args__ = (
        Index("idx_outcomes_experiment", "experiment_id"), # Already indexed by FK
        Index("idx_outcomes_variant", "experiment_id", "variant"), # Explicitly from schema
        Index("idx_outcomes_timestamp", "recorded_at"), # Explicitly from schema
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentOutcome(outcome_id={self.outcome_id}, "
            f"experiment_id={self.experiment_id}, event_id={self.event_id}, variant='{self.variant}')>"
        )
