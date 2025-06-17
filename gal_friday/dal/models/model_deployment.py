"""SQLAlchemy model for the 'model_deployments' table."""

from datetime import datetime
from uuid import UUID as PythonUUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base
from typing import Any


class ModelDeployment(Base):
    """Represents a deployment of a specific model version."""

    __tablename__ = "model_deployments"

    deployment_id: Mapped[PythonUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    # model_id is a ForeignKey to model_versions.model_id
    model_id: Mapped[PythonUUID] = mapped_column(
        ForeignKey("model_versions.model_id"), nullable=True, index=True, # Schema allows NULL, added index
    )
    deployed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False) # No server_default in schema
    deployed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deployment_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_active: Mapped[bool | None] = mapped_column(Boolean, server_default="true", index=True) # Added index

    # Relationship to ModelVersion
    model_version = relationship("ModelVersion", back_populates="deployments")

    __table_args__ = (
        Index("idx_deployments_model", "model_id"),
        Index("idx_deployments_active", "is_active"))

    def __repr__(self) -> str:
        return (
            f"<ModelDeployment(deployment_id={self.deployment_id}, model_id={self.model_id}, "
            f"deployed_at='{self.deployed_at}', is_active={self.is_active})>"
        )