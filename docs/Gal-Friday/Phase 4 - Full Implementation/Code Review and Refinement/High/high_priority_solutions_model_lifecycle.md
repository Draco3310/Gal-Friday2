# High Priority Solution: Model Lifecycle Management Implementation

## Overview
This document provides the complete implementation plan for ML model lifecycle management in the Gal-Friday system. Proper model management is critical for maintaining prediction accuracy, enabling A/B testing, and ensuring reliable trading decisions.

## Current State Problems

1. **No Model Versioning**
   - Models stored as files without version tracking
   - No metadata about training parameters
   - Cannot rollback to previous versions
   - No audit trail for model changes

2. **Missing Model Registry**
   - No centralized model repository
   - No model lineage tracking
   - Manual deployment process
   - No performance history

3. **No A/B Testing**
   - Cannot compare model performance
   - No gradual rollout capability
   - All-or-nothing deployments
   - No statistical validation

4. **Manual Retraining**
   - No automated retraining pipeline
   - No drift detection
   - No performance monitoring triggers
   - Manual hyperparameter tuning

## Solution Architecture

### 1. Model Registry Implementation

#### 1.1 Core Model Registry
```python
# gal_friday/model_lifecycle/registry.py
"""Centralized model registry for version control and metadata management."""

import os
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.dal.repositories.model_repository import ModelRepository
from gal_friday.utils.secrets_manager import SecretsManager


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
class ModelMetadata:
    """Complete model metadata."""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    version: str = ""
    model_type: str = ""  # xgboost, random_forest, lstm
    
    # Training info
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    training_completed_at: Optional[datetime] = None
    training_duration_seconds: Optional[float] = None
    trained_by: str = "system"
    
    # Data info
    training_data_start: Optional[datetime] = None
    training_data_end: Optional[datetime] = None
    training_samples: int = 0
    features: List[str] = field(default_factory=list)
    target_variable: str = ""
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model parameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Lifecycle info
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: ModelStatus = ModelStatus.TRAINING
    deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Storage info
    artifact_path: Optional[str] = None
    artifact_size_bytes: int = 0
    artifact_hash: Optional[str] = None
    
    # Additional metadata
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "training_completed_at": self.training_completed_at.isoformat() if self.training_completed_at else None,
            "training_duration_seconds": self.training_duration_seconds,
            "trained_by": self.trained_by,
            "training_data_start": self.training_data_start.isoformat() if self.training_data_start else None,
            "training_data_end": self.training_data_end.isoformat() if self.training_data_end else None,
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
            "description": self.description
        }


@dataclass
class ModelArtifact:
    """Container for model artifacts."""
    model: Any  # The actual model object
    preprocessor: Optional[Any] = None  # Scaler, encoder, etc.
    feature_names: List[str] = field(default_factory=list)
    metadata: Optional[ModelMetadata] = None
    
    def save(self, path: Path):
        """Save model artifact to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Save preprocessor if exists
        if self.preprocessor:
            preprocessor_path = path / "preprocessor.pkl"
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
                
        # Save metadata
        if self.metadata:
            metadata_path = path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
                
        # Save feature names
        features_path = path / "features.json"
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
            
    @classmethod
    def load(cls, path: Path) -> 'ModelArtifact':
        """Load model artifact from disk."""
        # Load model
        model_path = path / "model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Load preprocessor if exists
        preprocessor = None
        preprocessor_path = path / "preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
                
        # Load feature names
        features_path = path / "features.json"
        with open(features_path, 'r') as f:
            feature_names = json.load(f)
            
        # Load metadata if exists
        metadata = None
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                metadata = ModelMetadata(**metadata_dict)
                
        return cls(
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            metadata=metadata
        )


class ModelRegistry:
    """Centralized model registry with versioning and lifecycle management."""
    
    def __init__(self,
                 config: ConfigManager,
                 model_repo: ModelRepository,
                 logger: LoggerService,
                 secrets: SecretsManager):
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
            
    def _init_cloud_storage(self):
        """Initialize cloud storage backend (GCS, S3, etc.)."""
        # Implementation depends on cloud provider
        pass
        
    async def register_model(self, 
                           artifact: ModelArtifact,
                           model_name: str,
                           version: Optional[str] = None) -> str:
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
                context={"model_id": model_id}
            )
            
            return model_id
            
        except Exception:
            self.logger.exception(
                "Failed to register model",
                source_module=self._source_module
            )
            raise
            
    async def get_model(self, 
                       model_name: str,
                       version: Optional[str] = None,
                       stage: Optional[ModelStage] = None) -> ModelArtifact:
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
            artifact_path = Path(metadata.artifact_path)
            
            # Download from cloud if needed
            if self.use_cloud_storage and not artifact_path.exists():
                await self._download_from_cloud(
                    artifact_path, model_name, metadata.version
                )
                
            # Load artifact
            artifact = ModelArtifact.load(artifact_path)
            artifact.metadata = metadata
            
            return artifact
            
        except Exception:
            self.logger.exception(
                f"Failed to get model: {model_name}",
                source_module=self._source_module
            )
            raise
            
    async def list_models(self, 
                         model_name: Optional[str] = None,
                         stage: Optional[ModelStage] = None) -> List[ModelMetadata]:
        """List registered models.
        
        Args:
            model_name: Filter by model name
            stage: Filter by stage
            
        Returns:
            List of model metadata
        """
        return await self.model_repo.list_models(model_name, stage)
        
    async def promote_model(self, 
                          model_id: str,
                          to_stage: ModelStage,
                          promoted_by: str = "system") -> bool:
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
                    f"Invalid promotion: {from_stage.value} -> {to_stage.value}"
                )
                
            # If promoting to production, demote current production model
            if to_stage == ModelStage.PRODUCTION:
                current_prod = await self.model_repo.get_model_by_stage(
                    metadata.model_name, ModelStage.PRODUCTION
                )
                if current_prod and current_prod.model_id != model_id:
                    await self._demote_model(current_prod.model_id)
                    
            # Update model stage
            metadata.stage = to_stage
            metadata.deployment_history.append({
                "timestamp": datetime.now(UTC).isoformat(),
                "from_stage": from_stage.value,
                "to_stage": to_stage.value,
                "promoted_by": promoted_by
            })
            
            # Save to database
            success = await self.model_repo.update_model_stage(
                model_id, to_stage, promoted_by
            )
            
            if success:
                self.logger.info(
                    f"Model promoted: {metadata.model_name} v{metadata.version} "
                    f"from {from_stage.value} to {to_stage.value}",
                    source_module=self._source_module
                )
                
            return success
            
        except Exception:
            self.logger.exception(
                "Failed to promote model",
                source_module=self._source_module
            )
            raise
            
    async def delete_model(self, model_id: str, force: bool = False) -> bool:
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
                model_id, ModelStage.ARCHIVED, "system"
            )
            
            # Optionally delete artifacts
            if self.config.get_bool("model_registry.delete_archived_artifacts", False):
                artifact_path = Path(metadata.artifact_path)
                if artifact_path.exists():
                    shutil.rmtree(artifact_path)
                    
            return success
            
        except Exception:
            self.logger.exception(
                "Failed to delete model",
                source_module=self._source_module
            )
            raise
            
    async def _generate_version(self, model_name: str) -> str:
        """Generate next version number for model."""
        latest = await self.model_repo.get_latest_model(model_name)
        
        if not latest:
            return "1.0.0"
            
        # Parse current version and increment
        parts = latest.version.split(".")
        if len(parts) == 3:
            major, minor, patch = parts
            return f"{major}.{minor}.{int(patch) + 1}"
        else:
            # Fallback to timestamp-based version
            return datetime.now(UTC).strftime("%Y%m%d.%H%M%S")
            
    def _get_artifact_path(self, model_name: str, version: str) -> Path:
        """Get storage path for model artifacts."""
        return self.storage_path / model_name / version
        
    def _calculate_artifact_hash(self, artifact: ModelArtifact) -> str:
        """Calculate hash of model artifact for integrity checking."""
        hasher = hashlib.sha256()
        
        # Hash model pickle
        hasher.update(pickle.dumps(artifact.model))
        
        # Hash preprocessor if exists
        if artifact.preprocessor:
            hasher.update(pickle.dumps(artifact.preprocessor))
            
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
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT]  # Can restore archived models
        }
        
        return to_stage in valid_paths.get(from_stage, [])
        
    async def _demote_model(self, model_id: str):
        """Demote model from production to staging."""
        await self.model_repo.update_model_stage(
            model_id, ModelStage.STAGING, "system"
        )
```

### 2. A/B Testing Framework

#### 2.1 A/B Testing Manager
```python
# gal_friday/model_lifecycle/ab_testing.py
"""A/B testing framework for model comparison."""

import asyncio
import random
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
from scipy import stats

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.model_lifecycle.registry import ModelRegistry, ModelArtifact
from gal_friday.dal.repositories.experiment_repository import ExperimentRepository


class ExperimentStatus(Enum):
    """A/B test experiment status."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AllocationStrategy(Enum):
    """Traffic allocation strategies."""
    RANDOM = "random"
    WEIGHTED = "weighted"
    EPSILON_GREEDY = "epsilon_greedy"
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Models
    control_model_id: str = ""
    treatment_model_id: str = ""
    
    # Traffic allocation
    allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM
    traffic_split: float = 0.5  # Proportion going to treatment
    
    # Duration
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: Optional[datetime] = None
    min_samples_per_variant: int = 1000
    
    # Metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=list)
    
    # Statistical parameters
    confidence_level: float = 0.95
    minimum_detectable_effect: float = 0.01
    
    # Constraints
    max_loss_threshold: Optional[float] = None  # Stop if treatment performs this much worse
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "control_model_id": self.control_model_id,
            "treatment_model_id": self.treatment_model_id,
            "allocation_strategy": self.allocation_strategy.value,
            "traffic_split": self.traffic_split,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "min_samples_per_variant": self.min_samples_per_variant,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "confidence_level": self.confidence_level,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "max_loss_threshold": self.max_loss_threshold
        }


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""
    experiment_id: str
    status: ExperimentStatus
    
    # Sample counts
    control_samples: int = 0
    treatment_samples: int = 0
    
    # Metrics
    control_metrics: Dict[str, float] = field(default_factory=dict)
    treatment_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical results
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None
    
    # Recommendation
    winner: Optional[str] = None  # "control", "treatment", or None
    recommendation: str = ""
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if results are statistically significant."""
        return self.p_value is not None and self.p_value < alpha


@dataclass
class PredictionOutcome:
    """Outcome of a single prediction for tracking."""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str = ""
    model_variant: str = ""  # "control" or "treatment"
    
    # Prediction details
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    features: Dict[str, float] = field(default_factory=dict)
    prediction: float = 0.0
    confidence: float = 0.0
    
    # Actual outcome (when available)
    actual_outcome: Optional[float] = None
    outcome_timestamp: Optional[datetime] = None
    
    # Performance metrics
    error: Optional[float] = None
    squared_error: Optional[float] = None
    
    def calculate_error(self):
        """Calculate error metrics when outcome is available."""
        if self.actual_outcome is not None:
            self.error = self.prediction - self.actual_outcome
            self.squared_error = self.error ** 2


class ABTestingManager:
    """Manages A/B testing experiments for models."""
    
    def __init__(self,
                 config: ConfigManager,
                 model_registry: ModelRegistry,
                 experiment_repo: ExperimentRepository,
                 logger: LoggerService):
        self.config = config
        self.model_registry = model_registry
        self.experiment_repo = experiment_repo
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Active experiments
        self.active_experiments: Dict[str, ExperimentConfig] = {}
        self.experiment_models: Dict[str, Tuple[ModelArtifact, ModelArtifact]] = {}
        
        # Outcome tracking
        self.pending_outcomes: Dict[str, PredictionOutcome] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start A/B testing manager."""
        # Load active experiments from database
        active = await self.experiment_repo.get_active_experiments()
        for exp in active:
            await self._load_experiment(exp.experiment_id)
            
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitor_experiments())
        
    async def stop(self):
        """Stop A/B testing manager."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            experiment_id
        """
        try:
            # Validate models exist
            control_model = await self.model_registry.get_model(
                model_name=config.control_model_id.split("/")[0],
                version=config.control_model_id.split("/")[1] if "/" in config.control_model_id else None
            )
            treatment_model = await self.model_registry.get_model(
                model_name=config.treatment_model_id.split("/")[0],
                version=config.treatment_model_id.split("/")[1] if "/" in config.treatment_model_id else None
            )
            
            # Store models
            self.experiment_models[config.experiment_id] = (control_model, treatment_model)
            
            # Calculate required sample size
            required_samples = self._calculate_sample_size(
                config.minimum_detectable_effect,
                config.confidence_level
            )
            config.min_samples_per_variant = max(
                config.min_samples_per_variant,
                required_samples
            )
            
            # Save to database
            await self.experiment_repo.create_experiment(config)
            
            # Activate experiment
            self.active_experiments[config.experiment_id] = config
            
            self.logger.info(
                f"Created A/B test experiment: {config.name}",
                source_module=self._source_module,
                context={
                    "experiment_id": config.experiment_id,
                    "required_samples": required_samples
                }
            )
            
            return config.experiment_id
            
        except Exception:
            self.logger.exception(
                "Failed to create experiment",
                source_module=self._source_module
            )
            raise
            
    async def route_request(self, 
                          experiment_id: str,
                          context: Optional[Dict[str, Any]] = None) -> Tuple[str, ModelArtifact]:
        """Route prediction request to appropriate model variant.
        
        Args:
            experiment_id: Active experiment
            context: Optional context for routing decision
            
        Returns:
            Tuple of (variant_name, model_artifact)
        """
        config = self.active_experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment not found: {experiment_id}")
            
        control_model, treatment_model = self.experiment_models[experiment_id]
        
        # Determine allocation based on strategy
        if config.allocation_strategy == AllocationStrategy.RANDOM:
            use_treatment = random.random() < config.traffic_split
            
        elif config.allocation_strategy == AllocationStrategy.EPSILON_GREEDY:
            # Get current performance
            results = await self._get_current_results(experiment_id)
            if results and results.control_samples > 100 and results.treatment_samples > 100:
                # Exploit: choose better performing
                control_perf = results.control_metrics.get(config.primary_metric, 0)
                treatment_perf = results.treatment_metrics.get(config.primary_metric, 0)
                use_treatment = treatment_perf > control_perf
            else:
                # Explore: random allocation
                use_treatment = random.random() < config.traffic_split
                
        elif config.allocation_strategy == AllocationStrategy.THOMPSON_SAMPLING:
            # Implement Thompson sampling based on beta distribution
            use_treatment = await self._thompson_sampling_allocation(experiment_id)
            
        else:
            # Default to configured split
            use_treatment = random.random() < config.traffic_split
            
        if use_treatment:
            return ("treatment", treatment_model)
        else:
            return ("control", control_model)
            
    async def record_prediction(self,
                              experiment_id: str,
                              variant: str,
                              features: Dict[str, float],
                              prediction: float,
                              confidence: float) -> str:
        """Record a prediction for later outcome tracking.
        
        Args:
            experiment_id: Experiment ID
            variant: Model variant used ("control" or "treatment")
            features: Input features
            prediction: Model prediction
            confidence: Prediction confidence
            
        Returns:
            prediction_id for tracking
        """
        outcome = PredictionOutcome(
            experiment_id=experiment_id,
            model_variant=variant,
            features=features,
            prediction=prediction,
            confidence=confidence
        )
        
        # Store for outcome tracking
        self.pending_outcomes[outcome.prediction_id] = outcome
        
        # Save to database
        await self.experiment_repo.save_prediction_outcome(outcome)
        
        return outcome.prediction_id
        
    async def record_outcome(self,
                           prediction_id: str,
                           actual_outcome: float):
        """Record actual outcome for a prediction.
        
        Args:
            prediction_id: ID from record_prediction
            actual_outcome: Actual observed outcome
        """
        outcome = self.pending_outcomes.get(prediction_id)
        if not outcome:
            # Try loading from database
            outcome = await self.experiment_repo.get_prediction_outcome(prediction_id)
            
        if outcome:
            outcome.actual_outcome = actual_outcome
            outcome.outcome_timestamp = datetime.now(UTC)
            outcome.calculate_error()
            
            # Update in database
            await self.experiment_repo.update_prediction_outcome(outcome)
            
            # Remove from pending
            self.pending_outcomes.pop(prediction_id, None)
            
    async def get_experiment_results(self, experiment_id: str) -> ExperimentResult:
        """Get current results of an experiment.
        
        Args:
            experiment_id: Experiment to analyze
            
        Returns:
            ExperimentResult with statistical analysis
        """
        config = self.active_experiments.get(experiment_id)
        if not config:
            # Try loading from database
            config = await self.experiment_repo.get_experiment(experiment_id)
            
        # Get all outcomes
        outcomes = await self.experiment_repo.get_experiment_outcomes(experiment_id)
        
        # Separate by variant
        control_outcomes = [o for o in outcomes if o.model_variant == "control" and o.actual_outcome is not None]
        treatment_outcomes = [o for o in outcomes if o.model_variant == "treatment" and o.actual_outcome is not None]
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.RUNNING if experiment_id in self.active_experiments else ExperimentStatus.COMPLETED,
            control_samples=len(control_outcomes),
            treatment_samples=len(treatment_outcomes)
        )
        
        if control_outcomes and treatment_outcomes:
            # Calculate metrics
            result.control_metrics = self._calculate_metrics(control_outcomes)
            result.treatment_metrics = self._calculate_metrics(treatment_outcomes)
            
            # Perform statistical test
            control_values = [o.actual_outcome for o in control_outcomes]
            treatment_values = [o.actual_outcome for o in treatment_outcomes]
            
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
            result.p_value = p_value
            
            # Effect size (Cohen's d)
            effect_size = (np.mean(treatment_values) - np.mean(control_values)) / np.sqrt(
                (np.var(control_values) + np.var(treatment_values)) / 2
            )
            result.effect_size = effect_size
            
            # Confidence interval
            diff = np.mean(treatment_values) - np.mean(control_values)
            se = np.sqrt(np.var(control_values) / len(control_values) + 
                        np.var(treatment_values) / len(treatment_values))
            ci_margin = stats.t.ppf((1 + config.confidence_level) / 2, 
                                   len(control_values) + len(treatment_values) - 2) * se
            result.confidence_interval = (diff - ci_margin, diff + ci_margin)
            
            # Determine winner
            if result.is_significant(1 - config.confidence_level):
                if diff > 0:
                    result.winner = "treatment"
                    result.recommendation = f"Treatment model performs {abs(effect_size):.2f} standard deviations better"
                else:
                    result.winner = "control"
                    result.recommendation = f"Control model performs {abs(effect_size):.2f} standard deviations better"
            else:
                result.recommendation = "No significant difference detected yet"
                
        return result
        
    async def complete_experiment(self, experiment_id: str, force: bool = False) -> ExperimentResult:
        """Complete an experiment and provide final recommendation.
        
        Args:
            experiment_id: Experiment to complete
            force: Force completion even if minimum samples not reached
            
        Returns:
            Final ExperimentResult
        """
        config = self.active_experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment not active: {experiment_id}")
            
        # Get final results
        results = await self.get_experiment_results(experiment_id)
        
        # Check if we have enough samples
        if not force:
            if (results.control_samples < config.min_samples_per_variant or
                results.treatment_samples < config.min_samples_per_variant):
                raise ValueError(
                    f"Insufficient samples: control={results.control_samples}, "
                    f"treatment={results.treatment_samples}, "
                    f"required={config.min_samples_per_variant}"
                )
                
        # Remove from active experiments
        self.active_experiments.pop(experiment_id)
        self.experiment_models.pop(experiment_id)
        
        # Update status in database
        await self.experiment_repo.complete_experiment(experiment_id, results)
        
        self.logger.info(
            f"Completed experiment: {config.name}",
            source_module=self._source_module,
            context={
                "winner": results.winner,
                "p_value": results.p_value,
                "effect_size": results.effect_size
            }
        )
        
        return results
        
    def _calculate_sample_size(self, 
                             minimum_detectable_effect: float,
                             confidence_level: float) -> int:
        """Calculate required sample size for experiment."""
        # Using standard formula for two-sample t-test
        alpha = 1 - confidence_level
        beta = 0.2  # 80% power
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(1 - beta)
        
        # Assume standard deviation of 1 for simplicity
        # In practice, would estimate from historical data
        n = 2 * ((z_alpha + z_beta) ** 2) / (minimum_detectable_effect ** 2)
        
        return int(np.ceil(n))
        
    def _calculate_metrics(self, outcomes: List[PredictionOutcome]) -> Dict[str, float]:
        """Calculate performance metrics from outcomes."""
        if not outcomes:
            return {}
            
        errors = [o.error for o in outcomes if o.error is not None]
        squared_errors = [o.squared_error for o in outcomes if o.squared_error is not None]
        
        metrics = {
            "mean_absolute_error": np.mean(np.abs(errors)) if errors else 0,
            "mean_squared_error": np.mean(squared_errors) if squared_errors else 0,
            "root_mean_squared_error": np.sqrt(np.mean(squared_errors)) if squared_errors else 0,
            "sample_count": len(outcomes)
        }
        
        # Add custom metrics based on prediction type
        # This would be extended based on specific use case
        
        return metrics
        
    async def _thompson_sampling_allocation(self, experiment_id: str) -> bool:
        """Thompson sampling for adaptive allocation."""
        # Get current results
        results = await self._get_current_results(experiment_id)
        
        if not results or results.control_samples < 10 or results.treatment_samples < 10:
            # Not enough data, use random allocation
            return random.random() < 0.5
            
        # Model success/failure as beta distribution
        # For simplicity, using accuracy as success metric
        control_successes = results.control_metrics.get("accuracy", 0.5) * results.control_samples
        control_failures = results.control_samples - control_successes
        
        treatment_successes = results.treatment_metrics.get("accuracy", 0.5) * results.treatment_samples  
        treatment_failures = results.treatment_samples - treatment_successes
        
        # Sample from beta distributions
        control_sample = np.random.beta(control_successes + 1, control_failures + 1)
        treatment_sample = np.random.beta(treatment_successes + 1, treatment_failures + 1)
        
        return treatment_sample > control_sample
        
    async def _get_current_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get cached current results for an experiment."""
        try:
            return await self.get_experiment_results(experiment_id)
        except:
            return None
            
    async def _monitor_experiments(self):
        """Monitor active experiments for completion or issues."""
        while True:
            try:
                for exp_id, config in list(self.active_experiments.items()):
                    # Check if experiment should end
                    if config.end_time and datetime.now(UTC) > config.end_time:
                        await self.complete_experiment(exp_id, force=True)
                        continue
                        
                    # Check for early stopping conditions
                    results = await self._get_current_results(exp_id)
                    if results and config.max_loss_threshold:
                        # Check if treatment is performing too poorly
                        control_perf = results.control_metrics.get(config.primary_metric, 0)
                        treatment_perf = results.treatment_metrics.get(config.primary_metric, 0)
                        
                        if control_perf - treatment_perf > config.max_loss_threshold:
                            self.logger.warning(
                                f"Early stopping experiment {exp_id}: "
                                f"treatment underperforming by {control_perf - treatment_perf}",
                                source_module=self._source_module
                            )
                            await self.complete_experiment(exp_id, force=True)
                            
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in experiment monitoring",
                    source_module=self._source_module
                )
                
    async def _load_experiment(self, experiment_id: str):
        """Load experiment from database."""
        config = await self.experiment_repo.get_experiment(experiment_id)
        if config:
            self.active_experiments[experiment_id] = config
            
            # Load models
            control_model = await self.model_registry.get_model(
                model_name=config.control_model_id.split("/")[0],
                version=config.control_model_id.split("/")[1] if "/" in config.control_model_id else None
            )
            treatment_model = await self.model_registry.get_model(
                model_name=config.treatment_model_id.split("/")[0],
                version=config.treatment_model_id.split("/")[1] if "/" in config.treatment_model_id else None
            )
            
            self.experiment_models[experiment_id] = (control_model, treatment_model)
```

### 3. Automated Retraining Pipeline

#### 3.1 Model Retraining Service
```python
# gal_friday/model_lifecycle/retraining_service.py
"""Automated model retraining with drift detection."""

import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.model_lifecycle.registry import ModelRegistry, ModelArtifact, ModelMetadata
from gal_friday.model_training.trainer import ModelTrainer
from gal_friday.monitoring.alerting_system import AlertingSystem, Alert, AlertSeverity


class DriftType(Enum):
    """Types of drift detection."""
    CONCEPT_DRIFT = "concept_drift"  # Change in P(y|X)
    DATA_DRIFT = "data_drift"  # Change in P(X)
    PERFORMANCE_DRIFT = "performance_drift"  # Model performance degradation


@dataclass  
class DriftDetectionResult:
    """Result of drift detection analysis."""
    drift_detected: bool
    drift_type: Optional[DriftType] = None
    drift_score: float = 0.0
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    

class ModelRetrainingService:
    """Service for automated model retraining and drift detection."""
    
    def __init__(self,
                 config: ConfigManager,
                 model_registry: ModelRegistry,
                 model_trainer: ModelTrainer,
                 alerting: AlertingSystem,
                 logger: LoggerService):
        self.config = config
        self.model_registry = model_registry
        self.model_trainer = model_trainer
        self.alerting = alerting
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Configuration
        self.check_interval_hours = config.get_int("retraining.check_interval_hours", 24)
        self.performance_threshold = config.get_float("retraining.performance_threshold", 0.05)
        self.drift_threshold = config.get_float("retraining.drift_threshold", 0.01)
        self.min_samples_for_retraining = config.get_int("retraining.min_samples", 10000)
        
        # State
        self._monitoring_task: Optional[asyncio.Task] = None
        self._model_baselines: Dict[str, Dict[str, Any]] = {}
        
    async def start(self):
        """Start retraining service."""
        self.logger.info(
            "Starting model retraining service",
            source_module=self._source_module
        )
        
        # Initialize baselines for production models
        await self._initialize_baselines()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_models())
        
    async def stop(self):
        """Stop retraining service."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
    async def _initialize_baselines(self):
        """Initialize performance baselines for production models."""
        production_models = await self.model_registry.list_models(
            stage=ModelStage.PRODUCTION
        )
        
        for model_meta in production_models:
            baseline = {
                "model_id": model_meta.model_id,
                "baseline_metrics": model_meta.test_metrics,
                "feature_distributions": {},  # Would be populated from training data
                "last_check": datetime.now(UTC)
            }
            self._model_baselines[model_meta.model_name] = baseline
            
    async def check_model_drift(self, 
                              model_name: str,
                              recent_data: Any,
                              recent_predictions: Any) -> DriftDetectionResult:
        """Check for various types of drift.
        
        Args:
            model_name: Name of model to check
            recent_data: Recent feature data
            recent_predictions: Recent predictions and outcomes
            
        Returns:
            DriftDetectionResult
        """
        baseline = self._model_baselines.get(model_name)
        if not baseline:
            return DriftDetectionResult(drift_detected=False)
            
        # Check data drift
        data_drift_result = await self._check_data_drift(
            baseline.get("feature_distributions", {}),
            recent_data
        )
        
        # Check performance drift
        perf_drift_result = await self._check_performance_drift(
            baseline.get("baseline_metrics", {}),
            recent_predictions
        )
        
        # Check concept drift
        concept_drift_result = await self._check_concept_drift(
            model_name,
            recent_data,
            recent_predictions
        )
        
        # Combine results
        if data_drift_result.drift_detected:
            return data_drift_result
        elif perf_drift_result.drift_detected:
            return perf_drift_result
        elif concept_drift_result.drift_detected:
            return concept_drift_result
        else:
            return DriftDetectionResult(drift_detected=False)
            
    async def _check_data_drift(self,
                               baseline_distributions: Dict,
                               recent_data: Any) -> DriftDetectionResult:
        """Check for data distribution drift using statistical tests."""
        # Implement Kolmogorov-Smirnov test or similar
        # This is a simplified example
        
        drift_scores = []
        
        for feature_name, baseline_dist in baseline_distributions.items():
            if feature_name in recent_data:
                # KS test
                statistic, p_value = stats.ks_2samp(
                    baseline_dist,
                    recent_data[feature_name]
                )
                
                drift_scores.append(statistic)
                
                if p_value < self.drift_threshold:
                    return DriftDetectionResult(
                        drift_detected=True,
                        drift_type=DriftType.DATA_DRIFT,
                        drift_score=statistic,
                        p_value=p_value,
                        details={"feature": feature_name}
                    )
                    
        return DriftDetectionResult(
            drift_detected=False,
            drift_score=np.mean(drift_scores) if drift_scores else 0
        )
        
    async def _check_performance_drift(self,
                                     baseline_metrics: Dict,
                                     recent_predictions: Any) -> DriftDetectionResult:
        """Check for performance degradation."""
        # Calculate recent metrics
        recent_metrics = self._calculate_recent_metrics(recent_predictions)
        
        # Compare with baseline
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in recent_metrics:
                recent_value = recent_metrics[metric_name]
                
                # Calculate relative change
                if baseline_value > 0:
                    relative_change = abs(recent_value - baseline_value) / baseline_value
                    
                    if relative_change > self.performance_threshold:
                        return DriftDetectionResult(
                            drift_detected=True,
                            drift_type=DriftType.PERFORMANCE_DRIFT,
                            drift_score=relative_change,
                            details={
                                "metric": metric_name,
                                "baseline": baseline_value,
                                "recent": recent_value
                            }
                        )
                        
        return DriftDetectionResult(drift_detected=False)
        
    async def _check_concept_drift(self,
                                 model_name: str,
                                 recent_data: Any,
                                 recent_predictions: Any) -> DriftDetectionResult:
        """Check for concept drift using error distribution analysis."""
        # Implement ADWIN, DDM, or similar algorithms
        # This is a simplified example
        
        # Get prediction errors over time
        errors = self._calculate_prediction_errors(recent_predictions)
        
        if len(errors) < 100:
            return DriftDetectionResult(drift_detected=False)
            
        # Split into windows and compare
        window_size = len(errors) // 4
        early_window = errors[:window_size]
        recent_window = errors[-window_size:]
        
        # Test if error distributions are different
        statistic, p_value = stats.ttest_ind(early_window, recent_window)
        
        if p_value < self.drift_threshold:
            return DriftDetectionResult(
                drift_detected=True,
                drift_type=DriftType.CONCEPT_DRIFT,
                drift_score=abs(statistic),
                p_value=p_value,
                details={
                    "early_mean_error": np.mean(early_window),
                    "recent_mean_error": np.mean(recent_window)
                }
            )
            
        return DriftDetectionResult(drift_detected=False)
        
    async def trigger_retraining(self, 
                               model_name: str,
                               reason: str,
                               drift_result: Optional[DriftDetectionResult] = None):
        """Trigger model retraining.
        
        Args:
            model_name: Model to retrain
            reason: Reason for retraining
            drift_result: Optional drift detection result
        """
        self.logger.info(
            f"Triggering retraining for {model_name}: {reason}",
            source_module=self._source_module
        )
        
        try:
            # Get current production model
            current_model = await self.model_registry.get_model(
                model_name, stage=ModelStage.PRODUCTION
            )
            
            # Prepare training configuration
            training_config = {
                "model_name": model_name,
                "model_type": current_model.metadata.model_type,
                "features": current_model.metadata.features,
                "hyperparameters": current_model.metadata.hyperparameters,
                "reason": reason,
                "triggered_by": "drift_detection" if drift_result else "scheduled"
            }
            
            # Add drift information if available
            if drift_result:
                training_config["drift_info"] = {
                    "type": drift_result.drift_type.value,
                    "score": drift_result.drift_score,
                    "details": drift_result.details
                }
                
            # Start retraining
            new_model = await self.model_trainer.train_model(training_config)
            
            # Register new model
            new_model_id = await self.model_registry.register_model(
                new_model,
                model_name,
                tags={"retraining_reason": reason}
            )
            
            # Create A/B test to validate
            await self._create_validation_experiment(
                current_model.metadata.model_id,
                new_model_id,
                reason
            )
            
            # Send alert
            await self.alerting.send_alert(Alert(
                alert_id=f"retrain_{model_name}_{new_model_id}",
                title=f"Model Retraining Completed: {model_name}",
                message=f"New model trained due to: {reason}. A/B test created for validation.",
                severity=AlertSeverity.INFO,
                source=self._source_module,
                tags={"model_name": model_name, "new_model_id": new_model_id}
            ))
            
        except Exception:
            self.logger.exception(
                f"Failed to retrain model: {model_name}",
                source_module=self._source_module
            )
            
            await self.alerting.send_alert(Alert(
                alert_id=f"retrain_failed_{model_name}",
                title=f"Model Retraining Failed: {model_name}",
                message=f"Failed to retrain model due to: {reason}",
                severity=AlertSeverity.ERROR,
                source=self._source_module,
                tags={"model_name": model_name}
            ))
            
    async def _monitor_models(self):
        """Monitor models for drift and performance issues."""
        while True:
            try:
                await asyncio.sleep(self.check_interval_hours * 3600)
                
                # Check each production model
                for model_name, baseline in self._model_baselines.items():
                    # Get recent data and predictions
                    recent_data = await self._get_recent_feature_data(model_name)
                    recent_predictions = await self._get_recent_predictions(model_name)
                    
                    if not recent_data or not recent_predictions:
                        continue
                        
                    # Check for drift
                    drift_result = await self.check_model_drift(
                        model_name,
                        recent_data,
                        recent_predictions
                    )
                    
                    if drift_result.drift_detected:
                        await self.trigger_retraining(
                            model_name,
                            f"{drift_result.drift_type.value} detected",
                            drift_result
                        )
                        
                    # Update baseline check time
                    baseline["last_check"] = datetime.now(UTC)
                    
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in model monitoring",
                    source_module=self._source_module
                )
```

## Implementation Steps

### Phase 1: Model Registry (4 days)
1. Implement core registry with versioning
2. Create model metadata schema
3. Build artifact storage system
4. Add database integration
5. Implement cloud storage backend

### Phase 2: A/B Testing Framework (3 days)
1. Build experiment configuration system
2. Implement traffic routing logic
3. Create outcome tracking
4. Add statistical analysis
5. Build experiment monitoring

### Phase 3: Automated Retraining (3 days)
1. Implement drift detection algorithms
2. Create retraining triggers
3. Build training pipeline integration
4. Add performance monitoring
5. Create validation framework

### Phase 4: Integration & UI (2 days)
1. Add dashboard components
2. Create model comparison views
3. Build experiment management UI
4. Add deployment controls
5. Implement monitoring dashboards

## Success Criteria

1. **Model Management**
   - 100% of models versioned and tracked
   - < 1 minute to deploy new model version
   - Complete audit trail for all changes

2. **A/B Testing**
   - Statistical significance detection with 95% confidence
   - Automatic traffic allocation optimization
   - Real-time experiment monitoring

3. **Retraining**
   - Drift detection within 24 hours
   - Automated retraining completion < 2 hours
   - Performance improvement validation

4. **Operational**
   - Zero-downtime model updates
   - Automatic rollback on performance degradation
   - Complete experiment history retention

## Monitoring and Maintenance

1. **Model Performance Metrics**
   - Prediction accuracy trends
   - Latency by model version
   - Error rates over time
   - Feature importance changes

2. **Experiment Metrics**
   - Active experiment count
   - Average experiment duration
   - Win rate by model type
   - Statistical power achieved

3. **Retraining Metrics**
   - Drift detection frequency
   - Retraining success rate
   - Performance improvement rate
   - Training duration trends

4. **Alerts**
   - Model performance degradation
   - Drift detection
   - Experiment anomalies
   - Retraining failures 