"""Model repository implementation."""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import asyncpg

from gal_friday.dal.base import BaseRepository
from gal_friday.model_lifecycle.registry import ModelMetadata, ModelStage, ModelStatus

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ModelRepository(BaseRepository[ModelMetadata]):
    """Repository for model metadata persistence."""

    def __init__(self, db_pool: asyncpg.Pool, logger: "LoggerService") -> None:
        """Initialize the model repository.

        Args:
            db_pool: Async database connection pool
            logger: Logger service instance
        """
        super().__init__(db_pool, logger, "model_versions")

    def _row_to_entity(self, row: dict[str, Any]) -> ModelMetadata:
        """Convert database row to model metadata."""
        # Parse dates
        for date_field in ["created_at", "training_completed_at"]:
            if row.get(date_field):
                row[date_field] = row[date_field]

        # Parse JSON fields
        for json_field in ["metrics", "hyperparameters", "feature_importance"]:
            if row.get(json_field):
                row[json_field] = (
                    json.loads(row[json_field])
                    if isinstance(row[json_field], str)
                    else row[json_field]
                )

        # Parse enums
        row["stage"] = ModelStage(row["stage"])
        row["status"] = ModelStatus(row.get("status", "ready"))

        # Additional fields not in DB
        row.setdefault("validation_metrics", {})
        row.setdefault("test_metrics", {})
        row.setdefault("preprocessing_params", {})
        row.setdefault("deployment_history", [])
        row.setdefault("tags", {})
        row.setdefault("description", "")
        row.setdefault("training_data_start", None)
        row.setdefault("training_data_end", None)
        row.setdefault("training_samples", 0)
        row.setdefault("features", [])
        row.setdefault("target_variable", "")
        row.setdefault("training_duration_seconds", None)
        row.setdefault("trained_by", "system")
        row.setdefault("artifact_size_bytes", 0)
        row.setdefault("artifact_hash", None)

        return ModelMetadata(
            model_id=str(row["model_id"]),
            model_name=row["model_name"],
            version=row["version"],
            model_type=row.get("model_type", ""),
            created_at=row["created_at"],
            training_completed_at=row.get("training_completed_at"),
            stage=row["stage"],
            status=row["status"],
            metrics=row["metrics"],
            hyperparameters=row["hyperparameters"],
            feature_importance=row["feature_importance"],
            artifact_path=row.get("artifact_path"),
            validation_metrics=row["validation_metrics"],
            test_metrics=row["test_metrics"],
            preprocessing_params=row["preprocessing_params"],
            deployment_history=row["deployment_history"],
            tags=row["tags"],
            description=row["description"],
            training_data_start=row["training_data_start"],
            training_data_end=row["training_data_end"],
            training_samples=row["training_samples"],
            features=row["features"],
            target_variable=row["target_variable"],
            training_duration_seconds=row["training_duration_seconds"],
            trained_by=row["trained_by"],
            artifact_size_bytes=row["artifact_size_bytes"],
            artifact_hash=row["artifact_hash"],
        )

    async def save_model(self, metadata: ModelMetadata) -> str:
        """Save model metadata to database."""
        query = """
            INSERT INTO model_versions (
                model_id, model_name, version, model_type, created_at,
                training_completed_at, stage, metrics, hyperparameters,
                feature_importance, artifact_path
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (model_name, version)
            DO UPDATE SET
                stage = EXCLUDED.stage,
                metrics = EXCLUDED.metrics,
                hyperparameters = EXCLUDED.hyperparameters,
                feature_importance = EXCLUDED.feature_importance,
                artifact_path = EXCLUDED.artifact_path
            RETURNING model_id
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval(
                query,
                metadata.model_id,
                metadata.model_name,
                metadata.version,
                metadata.model_type,
                metadata.created_at.isoformat(),
                (metadata.training_completed_at.isoformat()
                 if metadata.training_completed_at else None),
                metadata.stage.value,
                json.dumps(metadata.metrics),
                json.dumps(metadata.hyperparameters),
                json.dumps(metadata.feature_importance),
                metadata.artifact_path,
            )

            # Save deployment record if in production
            if metadata.stage == ModelStage.PRODUCTION:
                await self._create_deployment_record(str(result), metadata.trained_by)

            return str(result)

    async def get_model_by_id(self, model_id: str) -> ModelMetadata | None:
        """Get model by ID."""
        query = "SELECT * FROM model_versions WHERE model_id = $1"

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, model_id)
            if row:
                return self._row_to_entity(dict(row))
            return None

    async def get_model_by_version(self, model_name: str, version: str) -> ModelMetadata | None:
        """Get model by name and version."""
        query = """
            SELECT * FROM model_versions
            WHERE model_name = $1 AND version = $2
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, model_name, version)
            if row:
                return self._row_to_entity(dict(row))
            return None

    async def get_model_by_stage(self, model_name: str, stage: ModelStage) -> ModelMetadata | None:
        """Get model by name and stage."""
        query = """
            SELECT * FROM model_versions
            WHERE model_name = $1 AND stage = $2
            ORDER BY created_at DESC
            LIMIT 1
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, model_name, stage.value)
            if row:
                return self._row_to_entity(dict(row))
            return None

    async def get_latest_model(self, model_name: str) -> ModelMetadata | None:
        """Get latest version of a model."""
        query = """
            SELECT * FROM model_versions
            WHERE model_name = $1
            ORDER BY created_at DESC
            LIMIT 1
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, model_name)
            if row:
                return self._row_to_entity(dict(row))
            return None

    async def list_models(self,
                         model_name: str | None = None,
                         stage: ModelStage | None = None) -> list[ModelMetadata]:
        """List models with optional filters."""
        query_parts = ["SELECT * FROM model_versions WHERE 1=1"]
        params = []
        param_count = 0

        if model_name:
            param_count += 1
            query_parts.append(f"AND model_name = ${param_count}")
            params.append(model_name)

        if stage:
            param_count += 1
            query_parts.append(f"AND stage = ${param_count}")
            params.append(stage.value)

        query_parts.append("ORDER BY created_at DESC")
        query = " ".join(query_parts)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def update_model_stage(self,
                               model_id: str,
                               new_stage: ModelStage,
                               updated_by: str) -> bool:
        """Update model stage."""
        query = """
            UPDATE model_versions
            SET stage = $1
            WHERE model_id = $2
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.execute(query, new_stage.value, model_id)

            # Create deployment record if promoting to production
            rows_affected = int(result.split()[-1])
            if new_stage == ModelStage.PRODUCTION and rows_affected > 0:
                await self._create_deployment_record(model_id, updated_by)

            return rows_affected > 0

    async def _create_deployment_record(self, model_id: str, deployed_by: str) -> None:
        """Create a deployment record."""
        query = """
            INSERT INTO model_deployments (
                model_id, deployed_at, deployed_by, deployment_config, is_active
            ) VALUES ($1, $2, $3, $4, $5)
        """

        # Deactivate previous deployments
        deactivate_query = """
            UPDATE model_deployments
            SET is_active = FALSE
            WHERE model_id IN (
                SELECT model_id FROM model_versions
                WHERE model_name = (
                    SELECT model_name FROM model_versions WHERE model_id = $1
                )
            )
        """
        async with self.db_pool.acquire() as conn, conn.transaction():
            # Deactivate old deployments
            await conn.execute(deactivate_query, model_id)

            # Create new deployment
            await conn.execute(
                query,
                model_id,
                datetime.now(UTC),
                deployed_by,
                json.dumps({"auto_deployed": True}),
                True,
            )
