"""Centralized model registry for version control and metadata management."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker  # Added

# from gal_friday.dal.base import BaseEntity # BaseEntity is removed
from gal_friday.dal.models.model_version import ModelVersion as ModelVersionModel
if TYPE_CHECKING:
    from gal_friday.config_manager import ConfigManager
    from gal_friday.dal.repositories.model_repository import ModelRepository
    from gal_friday.logger_service import LoggerService
    from gal_friday.utils.secrets_manager import SecretsManager

    from .cloud_storage import GCSBackend, S3Backend  # Added for type hinting

# Type variables for generic typing
T = TypeVar("T", bound="Predictor")
ArrayLike = npt.ArrayLike | Sequence[float] | Sequence[Sequence[float]]
Params = Mapping[str, Any] | None

@runtime_checkable
class Predictor(Protocol):
    """Protocol for model objects that can make predictions.

    This protocol defines the interface that all predictor models must implement
    to be compatible with the model registry.
    """

    def predict(self, x: ArrayLike) -> npt.NDArray[np.float64]:
        """Make predictions on input data.

        Args:
            x: Input data of shape (n_samples, n_features)

        Returns:
            Array of predictions, shape (n_samples,)
        """
        ...

    def fit(self, x: ArrayLike, y: ArrayLike | None = None) -> Predictor:
        """Fit the model to the training data.

        Args:
            x: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            self: Returns the instance itself
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values
        """
        ...


    def set_params(self: T, **params: object) -> T:  # type: ignore[no-untyped-def]
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters to set as keyword arguments.

        Returns:
            self: Returns the instance itself
        """
        ...

    coef_: npt.NDArray[np.float64] | None = None  # Coefficients for linear models


class ModelValidationError(Exception):
    """Raised when a loaded model fails validation."""


# Import enums from separate module to avoid circular dependencies
from .enums import ModelStage, ModelStatus


@dataclass
class ModelMetadata:
    """Complete model metadata."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    version: str = ""
    model_type: str = ""  # xgboost, random_forest, lstm

    # Training info
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None  # Added
    training_completed_at: datetime | None = None
    training_duration_seconds: float | None = None
    trained_by: str = "system"

    # Data info
    training_data_path: str | None = None  # Added
    training_data_start: datetime | None = None
    training_data_end: datetime | None = None
    training_samples: int = 0
    features: list[str] = field(default_factory=list)
    target_variable: str = ""

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Model parameters
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    preprocessing_params: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Lifecycle info
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: ModelStatus = ModelStatus.TRAINING
    deployment_history: list[dict[str, Any]] = field(default_factory=list)

    # Storage info
    artifact_path: str | None = None
    artifact_size_bytes: int = 0
    artifact_hash: str | None = None

    # Additional metadata
    tags: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,  # Added
            "training_completed_at": (
                self.training_completed_at.isoformat()
                if self.training_completed_at
                else None
            ),
            "training_duration_seconds": self.training_duration_seconds,
            "trained_by": self.trained_by,
            "training_data_path": self.training_data_path,  # Added
            "training_data_start": (
                self.training_data_start.isoformat()
                if self.training_data_start
                else None
            ),
            "training_data_end": (
                self.training_data_end.isoformat()
                if self.training_data_end
                else None
            ),
            "training_samples": self.training_samples,
            "features": self.features,
            "target_variable": self.target_variable,
            "metrics": self.metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "hyperparameters": self.hyperparameters,
            "preprocessing_params": self.preprocessing_params,
            "feature_importance": self.feature_importance,
            "stage": self.stage.value,
            "status": self.status.value,
            "deployment_history": self.deployment_history,
            "artifact_path": self.artifact_path,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_hash": self.artifact_hash,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create entity from database record."""
        # Parse dates
        date_fields = [
            "created_at",
            "updated_at", # Added
            "training_completed_at",
            "training_data_start",
            "training_data_end",
        ]
        for date_field in date_fields:
            if data.get(date_field) and isinstance(data[date_field], str):
                dt_obj = datetime.fromisoformat(data[date_field])
                # Ensure timezone awareness, assuming UTC if naive
                if dt_obj.tzinfo is None:
                    dt_obj = dt_obj.replace(tzinfo=UTC)
                data[date_field] = dt_obj

        # Parse enums
        if "stage" in data and isinstance(data["stage"], str):
            data["stage"] = ModelStage(data["stage"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ModelStatus(data["status"])

        # Set defaults for missing fields
        data.setdefault("validation_metrics", {})
        data.setdefault("test_metrics", {})
        data.setdefault("preprocessing_params", {})
        data.setdefault("deployment_history", [])
        data.setdefault("tags", {})
        data.setdefault("description", "")
        data.setdefault("training_data_start", None)
        data.setdefault("training_data_end", None)
        data.setdefault("training_samples", 0)
        data.setdefault("features", [])
        data.setdefault("target_variable", "")
        data.setdefault("training_duration_seconds", None)
        data.setdefault("trained_by", "system")
        data.setdefault("artifact_size_bytes", 0)
        data.setdefault("artifact_hash", None)
        data.setdefault("updated_at", None)  # Added
        data.setdefault("training_data_path", None)  # Added

        return cls(**data)


@dataclass
class ModelArtifact:
    """Container for model artifacts."""
    model: Any  # The actual model object
    preprocessor: Any | None = None  # Scaler, encoder, etc.
    feature_names: list[str] = field(default_factory=list)
    metadata: ModelMetadata | None = None

    def save(self, path: Path) -> None:
        """Save model artifact to disk."""
        path.mkdir(parents=True, exist_ok=True)

        # Save model with joblib which is safer for numpy arrays
        model_path = path / "model.joblib"
        joblib.dump(self.model, model_path)

        # Save preprocessor if exists
        if self.preprocessor:
            preprocessor_path = path / "preprocessor.joblib"
            joblib.dump(self.preprocessor, preprocessor_path)

        # Save metadata
        if self.metadata:
            metadata_path = path / "metadata.json"
            with metadata_path.open("w") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        # Save feature names
        features_path = path / "features.json"
        with features_path.open("w") as f:
            json.dump(self.feature_names, f)

    @classmethod
    def _validate_model(cls, model: Predictor) -> None:
        """Validate that the loaded model meets our requirements."""
        if not hasattr(model, "predict"):
            raise ModelValidationError("Model must have a 'predict' method")
        if not hasattr(model, "fit"):
            raise ModelValidationError("Model must have a 'fit' method")
        if not isinstance(model, BaseEstimator):
            raise ModelValidationError("Model must be a scikit-learn BaseEstimator")

    @classmethod
    def load(cls, path: Path) -> ModelArtifact:
        """Load model artifact from disk with validation.

        Args:
            path: Path to the directory containing the model artifacts
        Returns:
            Loaded ModelArtifact instance
        Raises:
            FileNotFoundError: If model file is not found
            ModelValidationError: If the model fails validation
            Exception: For any other errors during loading
        """
        try:
            # Load model with joblib
            model_path = path / "model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = joblib.load(model_path)
            cls._validate_model(model)

            # Load preprocessor if exists
            preprocessor = None
            preprocessor_path = path / "preprocessor.joblib"
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                cls._validate_model(preprocessor)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load model: {e}") from e
        except ModelValidationError as e:
            raise ModelValidationError(f"Model validation failed: {e}") from e
        except Exception as e:
            raise Exception(f"Error loading model: {e}") from e

        # Load feature names
        features_path = path / "features.json"
        with features_path.open() as f:
            feature_names = json.load(f)

        # Load metadata if exists
        metadata = None
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata_dict = json.load(f)
                # Parse dates
                for date_field in [
                    "created_at", "updated_at", "training_completed_at", # Added updated_at
                    "training_data_start", "training_data_end",
                ]:
                    if metadata_dict.get(date_field) and isinstance(metadata_dict[date_field], str):
                        dt_obj = datetime.fromisoformat(str(metadata_dict[date_field]))
                        if dt_obj.tzinfo is None: # Make timezone aware
                            dt_obj = dt_obj.replace(tzinfo=UTC)
                        metadata_dict[date_field] = dt_obj

                # Parse enums
                metadata_dict["stage"] = ModelStage(metadata_dict["stage"])
                metadata_dict["status"] = ModelStatus(metadata_dict["status"])
                # Add new fields if they might be in the JSON
                metadata_dict.setdefault("training_data_path", None)

                metadata = ModelMetadata(**metadata_dict)

        return cls(
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            metadata=metadata,
        )


class Registry: # Renamed from ModelRegistry for clarity as per plan
    """Centralized model registry with versioning and lifecycle management."""
    cloud_storage: GCSBackend | S3Backend | None = None # Added type hint

    def __init__(
        self,
        config_manager: ConfigManager, # Renamed
        session_maker: async_sessionmaker[AsyncSession], # Changed from model_repo
        logger_service: LoggerService, # Renamed
        secrets_manager: SecretsManager, # Renamed
    ) -> None:
        """Initialize Registry.

        Args:
            config_manager: Configuration manager instance
            session_maker: SQLAlchemy async_sessionmaker for database sessions.
            logger_service: Logger instance
            secrets_manager: Secrets manager instance
        """
        self.config_manager = config_manager
        self.session_maker = session_maker # Store session_maker
        
        # Import ModelRepository at runtime to avoid circular dependency
        from gal_friday.dal.repositories.model_repository import ModelRepository
        self.model_repo = ModelRepository(session_maker, logger_service) # Instantiate repo
        
        self.logger = logger_service
        self.secrets = secrets_manager
        self._source_module = self.__class__.__name__

        # Storage configuration
        self.storage_path = Path(self.config_manager.get("model_registry.storage_path", "./models"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Cloud storage backend (optional)
        self.use_cloud_storage = self.config_manager.get_bool("model_registry.use_cloud_storage", default=False)
        self.cloud_storage = None
        if self.use_cloud_storage:
            self._init_cloud_storage()

    def _init_cloud_storage(self) -> None:
        """Initialize cloud storage backend."""
        provider = self.config_manager.get("model_registry.cloud_provider", "").lower()

        if provider == "gcs":
            from .cloud_storage import GCSBackend
            self.cloud_storage = GCSBackend(self.config_manager, self.logger)
        elif provider == "s3":
            from .cloud_storage import S3Backend
            self.cloud_storage = S3Backend(self.config_manager, self.logger)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")

        self.logger.info(
            f"Initialized {provider.upper()} cloud storage",
            source_module=self._source_module,
        )

    async def register_model(
        self,
        artifact: ModelArtifact,
        model_name: str,
        version: str | None = None,
    ) -> str:
        """Register a new model version.

        Args:
            artifact: Model artifact to register
            model_name: Name of the model
            version: Optional version string (auto-generated if not provided)

        Returns:
            model_id of registered model
        """
        try:
            # Auto-generate version if not provided
            if not version:
                version = await self._generate_version(model_name)

            # Create metadata if not provided
            if not artifact.metadata:
                artifact.metadata = ModelMetadata()

            # Update metadata
            artifact.metadata.model_name = model_name
            artifact.metadata.version = version
            # artifact.metadata.created_at = datetime.now(UTC) # Already defaulted
            artifact.metadata.updated_at = datetime.now(UTC) # Set updated_at on registration
            artifact.metadata.status = ModelStatus.READY

            # Calculate artifact hash
            artifact_hash = self._calculate_artifact_hash(artifact)
            artifact.metadata.artifact_hash = artifact_hash

            # Store artifact
            artifact_path = self._get_artifact_path(model_name, version)
            artifact.save(artifact_path)

            # Update metadata with storage info
            artifact.metadata.artifact_path = str(artifact_path)
            artifact.metadata.artifact_size_bytes = self._get_directory_size(artifact_path)

            # Save to database
            model_version_data = {
                "model_id": uuid.UUID(artifact.metadata.model_id),
                "model_name": artifact.metadata.model_name,
                "version": artifact.metadata.version,
                "created_at": artifact.metadata.created_at,
                # ModelVersionModel does not have updated_at, training_data_path
                "training_completed_at": artifact.metadata.training_completed_at,
                "stage": artifact.metadata.stage.value,
                "metrics": artifact.metadata.metrics,
                "hyperparameters": artifact.metadata.hyperparameters,
                "feature_importance": artifact.metadata.feature_importance,
                "artifact_path": artifact.metadata.artifact_path,
            }
            created_model_version = await self.model_repo.add_model_version(model_version_data)

            if artifact.metadata.stage == ModelStage.PRODUCTION:
                 await self.model_repo.update_model_version_stage(
                     uuid.UUID(str(created_model_version.model_id)),
                     ModelStage.PRODUCTION.value,
                     deployed_by=artifact.metadata.trained_by,
                 )

            if self.use_cloud_storage:
                await self._upload_to_cloud(artifact_path, model_name, version)

            model_id_uuid = created_model_version.model_id
            if not isinstance(model_id_uuid, uuid.UUID):
                error_msg = f"Returned model_id is not a UUID instance: {type(model_id_uuid)}"
                self.logger.error(error_msg, source_module=self._source_module)
                raise TypeError(error_msg)

            self.logger.info(
                f"Model registered: {model_name} v{version}",
                source_module=self._source_module,
                context={"model_id": model_id_uuid.hex},
            )
            return model_id_uuid.hex

        except Exception:
            self.logger.exception(
                "Failed to register model",
                source_module=self._source_module,
            )
            raise

    async def get_model(
        self,
        model_name: str,
        version: str | None = None,
        stage: ModelStage | None = None,
    ) -> ModelArtifact:
        try:
            model_version_model: ModelVersionModel | None = None
            if version:
                versions = await self.model_repo.get_model_versions_by_name(model_name, version)
                if versions: model_version_model = versions[0]
            elif stage:
                stages = await self.model_repo.get_model_versions_by_stage(model_name, stage.value)
                if stages: model_version_model = stages[0]
            else:
                model_version_model = await self.model_repo.get_latest_model_version_by_name(model_name)

            if not model_version_model:
                raise ValueError(f"Model not found: {model_name} (version={version}, stage={stage})")

            metadata_dto = self._model_version_to_metadata_dto(model_version_model)

            deployments = await self.model_repo.get_deployments_for_model_version(uuid.UUID(str(model_version_model.model_id)))
            metadata_dto.deployment_history = [
                {
                    "deployed_at": dep.deployed_at.isoformat(),
                    "deployed_by": dep.deployed_by,
                    "is_active": dep.is_active,
                    "config": dep.deployment_config,
                } for dep in deployments
            ]

            if metadata_dto.artifact_path is None:
                raise ValueError(f"Model artifact path not found for: {model_name}")

            artifact_path = Path(metadata_dto.artifact_path)

            if self.use_cloud_storage and not artifact_path.exists():
                await self._download_from_cloud(
                    artifact_path, model_name, metadata_dto.version,
                )

            artifact = ModelArtifact.load(artifact_path)
            artifact.metadata = metadata_dto

            return artifact

        except Exception:
            self.logger.exception(
                f"Failed to get model: {model_name}",
                source_module=self._source_module,
            )
            raise

    async def list_models(
        self,
        model_name: str | None = None,
        stage: ModelStage | None = None,
    ) -> list[ModelMetadata]:
        model_version_models = await self.model_repo.list_all_model_versions(
            model_name=model_name,
            stage=stage.value if stage else None,
        )
        return [self._model_version_to_metadata_dto(mvm) for mvm in model_version_models]

    async def get_all_models(self) -> list[ModelMetadata]:
        model_version_models = await self.model_repo.list_all_model_versions()
        return [self._model_version_to_metadata_dto(mvm) for mvm in model_version_models]

    async def get_model_count(
        self,
        stage: ModelStage | None = None,
    ) -> int:
        models_list = await self.model_repo.list_all_model_versions(stage=stage.value if stage else None)
        return len(models_list)

    async def promote_model(
        self,
        model_id_str: str,
        to_stage: ModelStage,
        promoted_by: str = "system",
    ) -> bool:
        model_uuid = uuid.UUID(model_id_str)
        try:
            model_version = await self.model_repo.get_model_version(model_uuid)
            if not model_version:
                raise ValueError(f"Model not found: {model_id_str}")

            from_stage = ModelStage(model_version.stage) if model_version.stage else ModelStage.DEVELOPMENT

            if not self._is_valid_promotion(from_stage, to_stage):
                raise ValueError(
                    f"Invalid promotion: {from_stage.value} -> {to_stage.value}",
                )

            if to_stage == ModelStage.PRODUCTION:
                active_prod_deployments = await self.model_repo.get_active_deployment(model_version.model_name)
                if active_prod_deployments and active_prod_deployments.model_id != model_uuid:
                    await self.model_repo.update_model_version_stage(
                        uuid.UUID(str(active_prod_deployments.model_id)),
                        ModelStage.STAGING.value,
                        deployed_by="system_demotion",
                    )

            updated_model = await self.model_repo.update_model_version_stage(
                model_uuid, to_stage.value, promoted_by,
            )

            if updated_model:
                 # Update ModelMetadata's updated_at field if we were to fetch and return it here
                self.logger.info(
                    f"Model promoted: {model_version.model_name} v{model_version.version} "
                    f"from {from_stage.value} to {to_stage.value}",
                    source_module=self._source_module,
                )
                return True
            return False
        except Exception:
            self.logger.exception(
                "Failed to promote model",
                source_module=self._source_module,
            )
            raise

    async def delete_model(
        self,
        model_id: str,
        force: bool = False,
    ) -> bool:
        try:
            model_uuid = uuid.UUID(model_id)
            model_version = await self.model_repo.get_model_version(model_uuid)
            if not model_version:
                self.logger.warning(f"Model not found for deletion: {model_id}", source_module=self._source_module)
                return False

            current_stage = ModelStage(model_version.stage) if model_version.stage else ModelStage.DEVELOPMENT

            if current_stage == ModelStage.PRODUCTION and not force:
                raise ValueError("Cannot delete/archive active production model without force=True. Demote first.")

            updated_model = await self.model_repo.update_model_version_stage(
                model_uuid, ModelStage.ARCHIVED.value, "system_archive",
            )

            if not updated_model:
                 self.logger.error(f"Failed to archive model {model_id}", source_module=self._source_module)
                 return False

            if self.config_manager.get_bool("model_registry.delete_archived_artifacts", default=False) and model_version.artifact_path:
                artifact_path = Path(model_version.artifact_path)
                if artifact_path.exists() and artifact_path.is_dir():
                    try:
                        shutil.rmtree(artifact_path)
                        self.logger.info(f"Deleted local artifacts for archived model {model_id} at {artifact_path}", source_module=self._source_module)
                    except OSError as e:
                        self.logger.error(f"Error deleting artifacts for model {model_id} at {artifact_path}: {e}", source_module=self._source_module)
            return True
        except Exception:
            self.logger.exception(
                "Failed to delete model",
                source_module=self._source_module,
            )
            raise

    async def _generate_version(self, model_name: str) -> str:
        latest_model_version = await self.model_repo.get_latest_model_version_by_name(model_name)
        if not latest_model_version:
            return "1.0.0"
        current_version_str = latest_model_version.version
        try:
            parts = list(map(int, current_version_str.split(".")))
            if len(parts) == 3:
                parts[2] += 1
                return ".".join(map(str, parts))
            return f"{current_version_str}.{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        except ValueError:
            return f"{current_version_str}.{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    def _get_artifact_path(self, model_name: str, version: str) -> Path:
        return self.storage_path / model_name / version

    def _calculate_artifact_hash(self, artifact: ModelArtifact) -> str:
        hasher = hashlib.sha256()
        hasher.update(str(artifact.model.get_params()).encode())
        if hasattr(artifact.model, "coef_"):
            hasher.update(artifact.model.coef_.tobytes())
        if artifact.preprocessor:
            hasher.update(str(artifact.preprocessor.get_params()).encode())
            if hasattr(artifact.preprocessor, "scale_"):
                hasher.update(artifact.preprocessor.scale_.tobytes())
        hasher.update(json.dumps(sorted(artifact.feature_names)).encode())
        return hasher.hexdigest()

    def _get_directory_size(self, path: Path) -> int:
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def _is_valid_promotion(self, from_stage: ModelStage, to_stage: ModelStage) -> bool:
        valid_paths = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [
                ModelStage.PRODUCTION,
                ModelStage.DEVELOPMENT,
                ModelStage.ARCHIVED,
            ],
            ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],
        }
        return to_stage in valid_paths.get(from_stage, [])

    async def _demote_model(self, model_id: str) -> None:
        await self.model_repo.update_model_version_stage(
            uuid.UUID(model_id), ModelStage.STAGING.value, "system_demotion",
        )

    def _model_version_to_metadata_dto(self, model_version: ModelVersionModel) -> ModelMetadata:
        model_id_uuid = model_version.model_id
        if not isinstance(model_id_uuid, uuid.UUID):
            error_msg = f"ModelVersionModel.model_id is not a UUID instance: {type(model_id_uuid)} for model_name {model_version.model_name}"
            self.logger.error(error_msg, source_module=self._source_module)
            raise TypeError(error_msg)

        # Ensure created_at and training_completed_at are timezone-aware (UTC)
        created_at_utc = model_version.created_at.replace(tzinfo=UTC if model_version.created_at.tzinfo is None else None)
        training_completed_at_utc = None
        if model_version.training_completed_at:
            training_completed_at_utc = model_version.training_completed_at.replace(
                tzinfo=UTC if model_version.training_completed_at.tzinfo is None else None,
            )

        # updated_at does not exist on ModelVersionModel, so it defaults to None in ModelMetadata
        # training_data_path also does not exist on ModelVersionModel

        return ModelMetadata(
            model_id=model_id_uuid.hex,
            model_name=model_version.model_name,
            version=model_version.version,
            created_at=created_at_utc,
            training_completed_at=training_completed_at_utc,
            stage=ModelStage(model_version.stage) if model_version.stage else ModelStage.DEVELOPMENT,
            status=ModelStatus.READY,
            metrics=model_version.metrics or {},
            hyperparameters=model_version.hyperparameters or {},
            feature_importance=model_version.feature_importance or {},
            artifact_path=model_version.artifact_path,
            updated_at=None,  # Explicitly set to None as it's not in ModelVersionModel
            training_data_path=None,  # Explicitly set to None as it's not in ModelVersionModel
            # model_type, training_duration_seconds etc. will use defaults from ModelMetadata
        )

    async def _upload_to_cloud(self, local_path: Path, model_name: str, version: str) -> None:
        if not self.cloud_storage:
            return
        remote_path = f"models/{model_name}/{version}"
        success = await self.cloud_storage.upload(local_path, remote_path)
        if success:
            self.logger.info(
                f"Uploaded model to cloud: {remote_path}",
                source_module=self._source_module,
            )
        else:
            self.logger.error(
                f"Failed to upload model to cloud: {remote_path}",
                source_module=self._source_module,
            )

    async def _download_from_cloud(self, local_path: Path, model_name: str, version: str) -> None:
        if not self.cloud_storage:
            return
        remote_path = f"models/{model_name}/{version}"
        success = await self.cloud_storage.download(remote_path, local_path)
        if success:
            self.logger.info(
                f"Downloaded model from cloud: {remote_path}",
                source_module=self._source_module,
            )
        else:
            raise RuntimeError(f"Failed to download model from cloud: {remote_path}")
REPLACE_BLOCK_LINE_START=1
REPLACE_BLOCK_LINE_END=867
