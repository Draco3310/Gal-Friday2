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

from gal_friday.dal.base import BaseEntity

if TYPE_CHECKING:
    from gal_friday.config_manager import ConfigManager
    from gal_friday.dal.repositories.model_repository import ModelRepository
    from gal_friday.logger_service import LoggerService
    from gal_friday.utils.secrets_manager import SecretsManager

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

    def fit(self, x: ArrayLike, y: ArrayLike | None = None) -> T:
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


class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Model training/deployment status."""
    TRAINING = "training"
    EVALUATING = "evaluating"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata(BaseEntity):
    """Complete model metadata."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    version: str = ""
    model_type: str = ""  # xgboost, random_forest, lstm

    # Training info
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    training_completed_at: datetime | None = None
    training_duration_seconds: float | None = None
    trained_by: str = "system"

    # Data info
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
            "training_completed_at": (
                self.training_completed_at.isoformat()
                if self.training_completed_at
                else None
            ),
            "training_duration_seconds": self.training_duration_seconds,
            "trained_by": self.trained_by,
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
            "training_completed_at",
            "training_data_start",
            "training_data_end",
        ]
        for date_field in date_fields:
            if data.get(date_field) and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])

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
                    "created_at", "training_completed_at",
                    "training_data_start", "training_data_end",
                ]:
                    if metadata_dict.get(date_field):
                        metadata_dict[date_field] = datetime.fromisoformat(
                            str(metadata_dict[date_field]),
                        )
                # Parse enums
                metadata_dict["stage"] = ModelStage(metadata_dict["stage"])
                metadata_dict["status"] = ModelStatus(metadata_dict["status"])
                metadata = ModelMetadata(**metadata_dict)

        return cls(
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            metadata=metadata,
        )


class ModelRegistry:
    """Centralized model registry with versioning and lifecycle management."""

    def __init__(
        self,
        config: ConfigManager,
        model_repo: ModelRepository,
        logger: LoggerService,
        secrets: SecretsManager,
    ) -> None:
        """Initialize ModelRegistry.

        Args:
            config: Configuration manager instance
            model_repo: Model repository for data access
            logger: Logger instance
            secrets: Secrets manager instance
        """
        self.config = config
        self.model_repo = model_repo
        self.logger = logger
        self.secrets = secrets
        self._source_module = self.__class__.__name__

        # Storage configuration
        self.storage_path = Path(config.get("model_registry.storage_path", "./models"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Cloud storage backend (optional)
        self.use_cloud_storage = config.get_bool("model_registry.use_cloud_storage", False)
        if self.use_cloud_storage:
            self._init_cloud_storage()

    def _init_cloud_storage(self) -> None:
        """Initialize cloud storage backend (GCS, S3, etc.)."""
        # Implementation depends on cloud provider

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
            artifact.metadata.created_at = datetime.now(UTC)
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
            model_id = await self.model_repo.save_model(artifact.metadata)
            artifact.metadata.model_id = model_id

            # Upload to cloud storage if enabled
            if self.use_cloud_storage:
                await self._upload_to_cloud(artifact_path, model_name, version)

            self.logger.info(
                f"Model registered: {model_name} v{version}",
                source_module=self._source_module,
                context={"model_id": model_id},
            )

            return model_id

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
        """Retrieve a model by name and version or stage.

        Args:
            model_name: Name of the model
            version: Specific version (if not provided, gets latest)
            stage: Get model in specific stage (production, staging)

        Returns:
            ModelArtifact
        """
        try:
            # Get metadata from database
            if version:
                metadata = await self.model_repo.get_model_by_version(model_name, version)
            elif stage:
                metadata = await self.model_repo.get_model_by_stage(model_name, stage)
            else:
                # Get latest version
                metadata = await self.model_repo.get_latest_model(model_name)

            if not metadata:
                raise ValueError(f"Model not found: {model_name}")

            # Load artifact from storage
            if metadata.artifact_path is None:
                raise ValueError(f"Model artifact path not found for: {model_name}")

            artifact_path = Path(metadata.artifact_path)

            # Download from cloud if needed
            if self.use_cloud_storage and not artifact_path.exists():
                await self._download_from_cloud(
                    artifact_path, model_name, metadata.version,
                )

            # Load artifact
            artifact = ModelArtifact.load(artifact_path)
            artifact.metadata = metadata

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
        """List registered models.

        Args:
            model_name: Filter by model name
            stage: Filter by stage

        Returns:
            List of model metadata
        """
        return await self.model_repo.list_models(model_name, stage)

    async def get_all_models(self) -> list[ModelMetadata]:
        """Get all registered models.

        Returns:
            List of all model metadata
        """
        return await self.model_repo.list_models()

    async def get_model_count(
        self,
        stage: ModelStage | None = None,
    ) -> int:
        """Get count of registered models.

        Args:
            stage: Optional stage filter

        Returns:
            Number of models
        """
        models = await self.model_repo.list_models(stage=stage)
        return len(models)

    async def promote_model(
        self,
        model_id: str,
        to_stage: ModelStage,
        promoted_by: str = "system",
    ) -> bool:
        """Promote model to a different stage.

        Args:
            model_id: Model to promote
            to_stage: Target stage
            promoted_by: User/system promoting the model

        Returns:
            Success status
        """
        try:
            # Get current metadata
            metadata = await self.model_repo.get_model_by_id(model_id)
            if not metadata:
                raise ValueError(f"Model not found: {model_id}")

            from_stage = metadata.stage

            # Validate promotion path
            if not self._is_valid_promotion(from_stage, to_stage):
                raise ValueError(
                    f"Invalid promotion: {from_stage.value} -> {to_stage.value}",
                )

            # If promoting to production, demote current production model
            if to_stage == ModelStage.PRODUCTION:
                current_prod = await self.model_repo.get_model_by_stage(
                    metadata.model_name, ModelStage.PRODUCTION,
                )
                if current_prod and current_prod.model_id != model_id:
                    await self._demote_model(current_prod.model_id)

            # Update model stage
            metadata.stage = to_stage
            metadata.deployment_history.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "from_stage": from_stage.value,
                "to_stage": to_stage.value,
                "promoted_by": promoted_by,
            })

            # Save to database
            success = await self.model_repo.update_model_stage(
                model_id, to_stage, promoted_by,
            )

            if success:
                self.logger.info(
                    f"Model promoted: {metadata.model_name} v{metadata.version} "
                    f"from {from_stage.value} to {to_stage.value}",
                    source_module=self._source_module,
                )

            return success

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
        """Delete a model (or archive it).

        Args:
            model_id: Model to delete
            force: Force deletion even if in production

        Returns:
            Success status
        """
        try:
            # Get metadata
            metadata = await self.model_repo.get_model_by_id(model_id)
            if not metadata:
                return False

            # Don't delete production models unless forced
            if metadata.stage == ModelStage.PRODUCTION and not force:
                raise ValueError("Cannot delete production model without force=True")

            # Archive instead of delete
            metadata.stage = ModelStage.ARCHIVED
            metadata.status = ModelStatus.DEPRECATED

            # Update database
            success = await self.model_repo.update_model_stage(
                model_id, ModelStage.ARCHIVED, "system",
            )

            # Optionally delete artifacts
            if (
                self.config.get_bool("model_registry.delete_archived_artifacts", False)
                and metadata.artifact_path is not None
            ):
                artifact_path = Path(metadata.artifact_path)
                if artifact_path.exists():
                    shutil.rmtree(artifact_path)

            return success

        except Exception:
            self.logger.exception(
                "Failed to delete model",
                source_module=self._source_module,
            )
            raise

    async def _generate_version(self, model_name: str) -> str:
        """Generate next version number for model."""
        latest = await self.model_repo.get_latest_model(model_name)

        if not latest:
            return "1.0.0"

        # Parse current version and increment
        parts = latest.version.split(".")
        version_parts_count = 3
        if len(parts) == version_parts_count:
            major, minor, patch = parts
            return f"{major}.{minor}.{int(patch) + 1}"
        # Fallback to timestamp-based version
        return datetime.now(UTC).strftime("%Y%m%d.%H%M%S")

    def _get_artifact_path(self, model_name: str, version: str) -> Path:
        """Get storage path for model artifacts."""
        return self.storage_path / model_name / version

    def _calculate_artifact_hash(self, artifact: ModelArtifact) -> str:
        """Calculate hash of model artifact for integrity checking."""
        hasher = hashlib.sha256()

        # Hash model representation
        hasher.update(str(artifact.model.get_params()).encode())
        if hasattr(artifact.model, "coef_"):
            hasher.update(artifact.model.coef_.tobytes())

        # Hash preprocessor if exists
        if artifact.preprocessor:
            hasher.update(str(artifact.preprocessor.get_params()).encode())
            if hasattr(artifact.preprocessor, "scale_"):
                hasher.update(artifact.preprocessor.scale_.tobytes())

        # Hash feature names
        hasher.update(json.dumps(sorted(artifact.feature_names)).encode())

        return hasher.hexdigest()

    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory."""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size

    def _is_valid_promotion(self, from_stage: ModelStage, to_stage: ModelStage) -> bool:
        """Check if promotion path is valid."""
        valid_paths = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [
                ModelStage.PRODUCTION,
                ModelStage.DEVELOPMENT,
                ModelStage.ARCHIVED,
            ],
            ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],  # Can restore archived models
        }

        return to_stage in valid_paths.get(from_stage, [])

    async def _demote_model(self, model_id: str) -> None:
        """Demote model from production to staging."""
        await self.model_repo.update_model_stage(
            model_id, ModelStage.STAGING, "system",
        )

    async def _upload_to_cloud(self, local_path: Path, model_name: str, version: str) -> None:
        """Upload model artifacts to cloud storage."""
        # Implementation depends on cloud provider
        # Example for GCS:
        # bucket_name = self.config.get("model_registry.gcs_bucket")
        # blob_name = f"models/{model_name}/{version}"
        # ... upload logic ...

    async def _download_from_cloud(self, local_path: Path, model_name: str, version: str) -> None:
        """Download model artifacts from cloud storage."""
        # Implementation depends on cloud provider
