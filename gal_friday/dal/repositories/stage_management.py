"""Enterprise-grade model stage management system."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import uuid

# Import ModelStage from enums module to avoid circular dependencies
from gal_friday.model_lifecycle.enums import ModelStage


class StageTransition(str, Enum):
    """Valid stage transitions for model lifecycle management."""
    DEV_TO_STAGING = "development_to_staging"
    STAGING_TO_PROD = "staging_to_production"
    PROD_TO_RETIRED = "production_to_retired"
    RETIRED_TO_ARCHIVED = "retired_to_archived"
    ANY_TO_DEV = "any_to_development"
    STAGING_TO_DEV = "staging_to_development"
    PROD_TO_STAGING = "production_to_staging"


class StageValidationLevel(str, Enum):
    """Validation levels for stage transitions."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    ENTERPRISE = "enterprise"


class StageValidationError(Exception):
    """Exception raised for invalid stage operations."""

    def __init__(self, message: str, stage: ModelStage | None = None,
                 model_id: str | None = None, details: dict[str, Any] | None = None) -> None:
        """Initialize stage validation error.

        Args:
            message: Error message
            stage: Stage that failed validation
            model_id: Model ID if applicable
            details: Additional error details
        """
        super().__init__(message)
        self.stage = stage
        self.model_id = model_id
        self.details = details or {}


@dataclass
class StageConfig:
    """Configuration for a specific model stage."""
    stage: ModelStage
    requires_approval: bool = False
    max_models: int | None = None
    auto_monitoring: bool = True
    backup_required: bool = False
    performance_thresholds: dict[str, float] = field(default_factory=dict[str, Any])
    allowed_transitions: set[ModelStage] = field(default_factory=set)
    metadata_requirements: list[str] = field(default_factory=list[Any])
    validation_level: StageValidationLevel = StageValidationLevel.BASIC


@dataclass
class StageTransitionRecord:
    """Record of a stage transition for audit trail."""
    transition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    from_stage: ModelStage | None = None
    to_stage: ModelStage = ModelStage.DEVELOPMENT
    transition_type: StageTransition = StageTransition.ANY_TO_DEV
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_id: str | None = None
    reason: str | None = None
    approval_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    success: bool = True
    error_message: str | None = None


@dataclass
class StageMetrics:
    """Metrics for stage monitoring."""
    stage: ModelStage
    model_count: int = 0
    successful_transitions: int = 0
    failed_transitions: int = 0
    last_transition: datetime | None = None
    average_transition_time: float | None = None
    performance_violations: int = 0


class ModelStageManager:
    """Enterprise-grade model stage management system."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the stage manager.

        Args:
            config: Configuration dictionary for stage management
        """
        self.config = config

        # Initialize stage configurations
        self.stage_configs = self._initialize_stage_configs(config)

        # Initialize transition rules
        self.transition_rules = self._initialize_transition_rules()

        # Audit trail storage (in production, this would be persisted to database)
        self.transition_history: list[StageTransitionRecord] = []

        # Stage metrics tracking
        self.stage_metrics = {
            stage: StageMetrics(stage=stage)
            for stage in ModelStage
        }

    def _initialize_stage_configs(self, config: dict[str, Any]) -> dict[ModelStage, StageConfig]:
        """Initialize stage configurations with enterprise-grade defaults."""
        configs = {}

        # Development stage - permissive for rapid iteration
        configs[ModelStage.DEVELOPMENT] = StageConfig(
            stage=ModelStage.DEVELOPMENT,
            requires_approval=False,
            max_models=None,  # Unlimited models in development
            auto_monitoring=True,
            backup_required=False,
            performance_thresholds={},
            allowed_transitions={ModelStage.STAGING},
            metadata_requirements=["model_type", "created_by"],
            validation_level=StageValidationLevel.BASIC,
        )

        # Staging stage - validation and testing
        configs[ModelStage.STAGING] = StageConfig(
            stage=ModelStage.STAGING,
            requires_approval=config.get("staging_requires_approval", True),
            max_models=config.get("max_staging_models", 10),
            auto_monitoring=True,
            backup_required=True,
            performance_thresholds={
                "accuracy": config.get("staging_min_accuracy", 0.75),
                "precision": config.get("staging_min_precision", 0.70),
                "recall": config.get("staging_min_recall", 0.70),
            },
            allowed_transitions={ModelStage.PRODUCTION, ModelStage.DEVELOPMENT},
            metadata_requirements=["model_type", "created_by", "validation_results"],
            validation_level=StageValidationLevel.STRICT,
        )

        # Production stage - strict controls and monitoring
        configs[ModelStage.PRODUCTION] = StageConfig(
            stage=ModelStage.PRODUCTION,
            requires_approval=config.get("production_requires_approval", True),
            max_models=config.get("max_production_models", 3),
            auto_monitoring=True,
            backup_required=True,
            performance_thresholds={
                "accuracy": config.get("production_min_accuracy", 0.85),
                "precision": config.get("production_min_precision", 0.80),
                "recall": config.get("production_min_recall", 0.80),
                "latency_ms": config.get("production_max_latency", 100),
                "uptime_percent": config.get("production_min_uptime", 99.9),
            },
            allowed_transitions={ModelStage.STAGING, ModelStage.ARCHIVED},
            metadata_requirements=[
                "model_type", "created_by", "validation_results",
                "approval_id", "performance_test_results",
            ],
            validation_level=StageValidationLevel.ENTERPRISE,
        )

        # Archived stage - long-term storage
        configs[ModelStage.ARCHIVED] = StageConfig(
            stage=ModelStage.ARCHIVED,
            requires_approval=False,
            max_models=None,
            auto_monitoring=False,
            backup_required=True,
            performance_thresholds={},
            allowed_transitions={ModelStage.DEVELOPMENT},  # Can resurrect archived models
            metadata_requirements=["retirement_reason", "archived_by"],
            validation_level=StageValidationLevel.NONE,
        )

        return configs

    def _initialize_transition_rules(self) -> dict[tuple[ModelStage, ModelStage], StageTransition]:
        """Initialize valid transition rules."""
        return {
            (ModelStage.DEVELOPMENT, ModelStage.STAGING): StageTransition.DEV_TO_STAGING,
            (ModelStage.STAGING, ModelStage.PRODUCTION): StageTransition.STAGING_TO_PROD,
            (ModelStage.PRODUCTION, ModelStage.STAGING): StageTransition.PROD_TO_STAGING,
            (ModelStage.PRODUCTION, ModelStage.ARCHIVED): StageTransition.PROD_TO_RETIRED,
            (ModelStage.STAGING, ModelStage.DEVELOPMENT): StageTransition.STAGING_TO_DEV,
            (ModelStage.ARCHIVED, ModelStage.DEVELOPMENT): StageTransition.ANY_TO_DEV,
        }

    def is_production_stage(self, stage: str | ModelStage) -> bool:
        """Check if a stage is production.

        Args:
            stage: Stage to check (string or enum)

        Returns:
            True if stage is production
        """
        if isinstance(stage, str):
            return stage.upper() == ModelStage.PRODUCTION.value.upper()
        return stage == ModelStage.PRODUCTION

    def validate_stage_transition(
        self,
        model_id: str,
        from_stage: ModelStage | None,
        to_stage: ModelStage,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None]:
        """Validate if a stage transition is allowed.

        Args:
            model_id: Model identifier
            from_stage: Current stage (None for new models)
            to_stage: Target stage
            metadata: Model metadata for validation

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            metadata = metadata or {}

            # Check if transition is explicitly allowed
            if from_stage and to_stage not in self.stage_configs[from_stage].allowed_transitions:
                return False, f"Transition from {from_stage.value} to {to_stage.value} not allowed"

            # Validate stage requirements
            stage_config = self.stage_configs[to_stage]

            # Check required metadata fields
            for required_field in stage_config.metadata_requirements:
                if required_field not in metadata:
                    return False, f"Missing required metadata field '{required_field}' for stage {to_stage.value}"

            # Check performance thresholds
            for metric, threshold in stage_config.performance_thresholds.items():
                if metric in metadata:
                    actual_value = metadata.get(metric, 0)
                    if actual_value < threshold:
                        return False, (
                            f"Performance metric '{metric}' ({actual_value}) below threshold "
                            f"({threshold}) for stage {to_stage.value}"
                        )

            # Check stage capacity
            if not self._check_stage_capacity(to_stage):
                return False, f"Stage {to_stage.value} has reached maximum capacity"

            # Check approval requirements
            if stage_config.requires_approval and not metadata.get("approval_id"):
                return False, f"Stage {to_stage.value} requires approval"

            # return True, None moved to else block

        except Exception as e:
            return False, f"Validation error: {e!s}"
        else:
            return True, None

    def _check_stage_capacity(self, stage: ModelStage) -> bool:
        """Check if stage has capacity for new models."""
        stage_config = self.stage_configs[stage]

        if stage_config.max_models is None:
            return True

        current_count = self.stage_metrics[stage].model_count
        return current_count < stage_config.max_models

    def record_stage_transition(
        self,
        model_id: str,
        from_stage: ModelStage | None,
        to_stage: ModelStage,
        user_id: str | None = None,
        reason: str | None = None,
        approval_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> StageTransitionRecord:
        """Record a stage transition for audit trail.

        Args:
            model_id: Model identifier
            from_stage: Previous stage
            to_stage: New stage
            user_id: User who initiated the transition
            reason: Reason for transition
            approval_id: Approval ID if required
            metadata: Additional metadata
            success: Whether transition was successful
            error_message: Error message if failed

        Returns:
            Created transition record
        """
        # Determine transition type
        transition_type = StageTransition.ANY_TO_DEV
        if from_stage:
            transition_key = (from_stage, to_stage)
            transition_type = self.transition_rules.get(transition_key, StageTransition.ANY_TO_DEV)

        # Create transition record
        record = StageTransitionRecord(
            model_id=model_id,
            from_stage=from_stage,
            to_stage=to_stage,
            transition_type=transition_type,
            user_id=user_id,
            reason=reason,
            approval_id=approval_id,
            metadata=metadata or {},
            success=success,
            error_message=error_message,
        )

        # Store in audit trail
        self.transition_history.append(record)

        # Update metrics
        self._update_stage_metrics(to_stage, success)

        return record

    def _update_stage_metrics(self, stage: ModelStage, success: bool) -> None:
        """Update stage metrics after a transition attempt."""
        metrics = self.stage_metrics[stage]

        if success:
            metrics.successful_transitions += 1
        else:
            metrics.failed_transitions += 1

        metrics.last_transition = datetime.now(UTC)

    def get_stage_config(self, stage: ModelStage) -> StageConfig:
        """Get configuration for a specific stage."""
        return self.stage_configs[stage]

    def get_stage_metrics(self, stage: ModelStage) -> StageMetrics:
        """Get metrics for a specific stage."""
        return self.stage_metrics[stage]

    def get_transition_history(
        self,
        model_id: str | None = None,
        limit: int | None = None,
    ) -> list[StageTransitionRecord]:
        """Get transition history records.

        Args:
            model_id: Filter by model ID (optional)
            limit: Maximum number of records to return

        Returns:
            List of transition records
        """
        records = self.transition_history

        if model_id:
            records = [r for r in records if r.model_id == model_id]

        # Sort by timestamp descending (most recent first)
        records = sorted(records, key=lambda r: r.timestamp, reverse=True)

        if limit:
            records = records[:limit]

        return records

    def get_stage_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of all stages."""
        summary = {}

        for stage in ModelStage:
            config = self.stage_configs[stage]
            metrics = self.stage_metrics[stage]

            summary[stage.value] = {
                "config": {
                    "requires_approval": config.requires_approval,
                    "max_models": config.max_models,
                    "auto_monitoring": config.auto_monitoring,
                    "backup_required": config.backup_required,
                    "performance_thresholds": config.performance_thresholds,
                    "validation_level": config.validation_level.value,
                },
                "metrics": {
                    "model_count": metrics.model_count,
                    "successful_transitions": metrics.successful_transitions,
                    "failed_transitions": metrics.failed_transitions,
                    "last_transition": metrics.last_transition.isoformat() if metrics.last_transition else None,
                    "performance_violations": metrics.performance_violations,
                },
                "allowed_transitions": [s.value for s in config.allowed_transitions],
            }

        return summary


# Default configuration for the stage manager
DEFAULT_STAGE_CONFIG = {
    "staging_requires_approval": True,
    "production_requires_approval": True,
    "max_staging_models": 10,
    "max_production_models": 3,
    "staging_min_accuracy": 0.75,
    "staging_min_precision": 0.70,
    "staging_min_recall": 0.70,
    "production_min_accuracy": 0.85,
    "production_min_precision": 0.80,
    "production_min_recall": 0.80,
    "production_max_latency": 100,
    "production_min_uptime": 99.9,
}
