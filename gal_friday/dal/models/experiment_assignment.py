"""SQLAlchemy model for the 'experiment_assignments' table."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, PrimaryKeyConstraint, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .models_base import Base


class ExperimentAssignment(Base):
    """Represents an assignment of an event to an experiment variant."""

    __tablename__ = "experiment_assignments"

    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiments.experiment_id"), primary_key=True,
    )
    event_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True) # Assuming this is a generic UUID for an event
    variant: Mapped[str] = mapped_column(String(20), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True) # Added index

    # Relationship to Experiment
    experiment = relationship("Experiment") # Add back_populates="assignments" to Experiment if needed

    __table_args__ = (
        PrimaryKeyConstraint("experiment_id", "event_id"),
        Index("idx_assignments_experiment", "experiment_id"), # Already indexed by PK component
        Index("idx_assignments_timestamp", "assigned_at"), # Explicitly created from schema
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentAssignment(experiment_id={self.experiment_id}, "
            f"event_id={self.event_id}, variant='{self.variant}')>"
        )
