"""Model repository implementation using SQLAlchemy."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from sqlalchemy import select, update as sqlalchemy_update
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.model_version import ModelVersion
from gal_friday.dal.models.model_deployment import ModelDeployment
# Assuming ModelStage might be an enum or string constants used by a service layer now
# from gal_friday.model_lifecycle.registry import ModelStage


if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ModelRepository(BaseRepository[ModelVersion]):
    """Repository for ModelVersion data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService"
    ) -> None:
        """Initialize the model repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, ModelVersion, logger)

    async def add_model_version(self, model_version_data: dict[str, Any]) -> ModelVersion:
        """Adds a new model version.
        Expects model_version_data to have all necessary fields for ModelVersion.
        'model_id' should be a UUID object if provided, or will be generated if not present and model expects it.
        """
        if "model_id" not in model_version_data: # Ensure model_id is present if not auto-generated by DB
             model_version_data["model_id"] = model_version_data.get("model_id", uuid.uuid4())
        # Ensure datetime objects are timezone-aware if needed by DB schema or comparisons
        if "created_at" in model_version_data and isinstance(model_version_data["created_at"], datetime):
            if model_version_data["created_at"].tzinfo is None:
                 model_version_data["created_at"] = model_version_data["created_at"].replace(tzinfo=timezone.utc)
        if "training_completed_at" in model_version_data and isinstance(model_version_data["training_completed_at"], datetime):
            if model_version_data["training_completed_at"].tzinfo is None:
                 model_version_data["training_completed_at"] = model_version_data["training_completed_at"].replace(tzinfo=timezone.utc)

        return await self.create(model_version_data)

    async def get_model_version(self, model_id: uuid.UUID) -> ModelVersion | None:
        """Get model version by its UUID."""
        return await self.get_by_id(model_id)

    async def get_model_versions_by_name(
        self, model_name: str, version: str | None = None
    ) -> Sequence[ModelVersion]:
        """Get model versions by name, optionally filtered by version."""
        filters = {"model_name": model_name}
        if version:
            filters["version"] = version
        return await self.find_all(filters=filters, order_by="created_at DESC")

    async def get_latest_model_version_by_name(self, model_name: str) -> ModelVersion | None:
        """Get the latest model version for a given model name."""
        versions = await self.find_all(
            filters={"model_name": model_name}, order_by="created_at DESC", limit=1
        )
        return versions[0] if versions else None

    async def get_model_versions_by_stage(
        self, model_name: str, stage: str # Assuming stage is a string now
    ) -> Sequence[ModelVersion]:
        """Get model versions by name and stage."""
        return await self.find_all(
            filters={"model_name": model_name, "stage": stage},
            order_by="created_at DESC",
        )
    
    async def list_all_model_versions(
        self, model_name: str | None = None, stage: str | None = None
    ) -> Sequence[ModelVersion]:
        """List model versions with optional filters for name and stage."""
        filters = {}
        if model_name:
            filters["model_name"] = model_name
        if stage:
            filters["stage"] = stage
        return await self.find_all(filters=filters if filters else None, order_by="created_at DESC")


    async def update_model_version_stage(
        self, model_id: uuid.UUID, new_stage: str, deployed_by: str | None = None # Assuming stage is string
    ) -> ModelVersion | None:
        """Update model version's stage. If promoting to 'production', creates a deployment record."""
        # deployed_by is optional, only used if new_stage is production-like
        updated_model = await self.update(model_id, {"stage": new_stage})

        # Example: "PRODUCTION" string, adjust if using an enum or different values
        if updated_model and new_stage.upper() == "PRODUCTION":
            await self._create_deployment_record(
                model_version=updated_model,
                deployed_by=deployed_by or "system", # Default to system if not specified
                deployment_config={"auto_promoted_stage": new_stage}
            )
        return updated_model

    async def _create_deployment_record(
        self, model_version: ModelVersion, deployed_by: str, deployment_config: dict | None = None
    ) -> ModelDeployment:
        """Internal helper to create a deployment record and deactivate old ones for the same model name."""
        async with self.session_maker() as session:
            # Deactivate previous active deployments for this model_name
            # This assumes model_id on ModelDeployment is a FK to ModelVersion.model_id
            # And ModelVersion has model_name.
            update_stmt = (
                sqlalchemy_update(ModelDeployment)
                .where(
                    ModelDeployment.model_id.in_(
                        select(ModelVersion.model_id).where(ModelVersion.model_name == model_version.model_name)
                    ),
                    ModelDeployment.is_active == True,
                    ModelDeployment.model_id != model_version.model_id # Don't deactivate if re-deploying same version
                )
                .values(is_active=False)
            )
            await session.execute(update_stmt)

            # Create new deployment record
            new_deployment = ModelDeployment(
                model_id=model_version.model_id, # Must be the PK of ModelVersion
                deployed_at=datetime.now(timezone.utc),
                deployed_by=deployed_by,
                deployment_config=deployment_config or {},
                is_active=True,
            )
            session.add(new_deployment)
            await session.commit()
            await session.refresh(new_deployment)
            self.logger.info(
                f"Created new deployment record {new_deployment.deployment_id} "
                f"for model_id {model_version.model_id}",
                source_module=self._source_module
            )
            return new_deployment

    async def get_active_deployment(self, model_name: str) -> ModelDeployment | None:
        """Get the currently active deployment for a given model name."""
        async with self.session_maker() as session:
            stmt = (
                select(ModelDeployment)
                .join(ModelVersion, ModelDeployment.model_id == ModelVersion.model_id)
                .where(ModelVersion.model_name == model_name, ModelDeployment.is_active == True)
                .order_by(ModelDeployment.deployed_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            deployment = result.scalar_one_or_none()
            if deployment:
                 self.logger.debug(f"Found active deployment for model {model_name}", source_module=self._source_module)
            else:
                 self.logger.debug(f"No active deployment found for model {model_name}", source_module=self._source_module)
            return deployment

    async def get_deployments_for_model_version(self, model_id: uuid.UUID) -> Sequence[ModelDeployment]:
        """Get all deployment records for a specific model version ID."""
        # This uses a direct query on ModelDeployment, could also be BaseRepository[ModelDeployment].find_all
        async with self.session_maker() as session:
            stmt = select(ModelDeployment).where(ModelDeployment.model_id == model_id).order_by(ModelDeployment.deployed_at.desc())
            result = await session.execute(stmt)
            deployments = result.scalars().all()
            self.logger.debug(f"Found {len(deployments)} deployments for model_id {model_id}", source_module=self._source_module)
            return deployments
