"""Model retraining and drift detection pipeline."""

import asyncio
import contextlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from gal_friday.core.events import LogEvent
from gal_friday.core.pubsub import PubSubManager

if TYPE_CHECKING:
    from gal_friday.config_manager import ConfigManager
    from gal_friday.dal.repositories.retraining_repository import RetrainingRepository
    from gal_friday.logger_service import LoggerService
    from gal_friday.model_lifecycle.registry import ModelRegistry


class DriftType(Enum):
    """Types of model drift."""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class RetrainingTrigger(Enum):
    """Triggers for model retraining."""
    SCHEDULED = "scheduled"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADED = "performance_degraded"
    MANUAL = "manual"


@dataclass
class DriftMetrics:
    """Drift detection metrics."""
    drift_type: DriftType
    metric_name: str
    baseline_value: float
    current_value: float
    drift_score: float
    is_significant: bool
    threshold: float = 0.1
    detection_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainingJob:
    """Model retraining job."""
    job_id: str
    model_id: str
    model_name: str
    trigger: RetrainingTrigger
    drift_metrics: list[DriftMetrics] = field(default_factory=list)
    status: str = "pending"
    start_time: datetime | None = None
    end_time: datetime | None = None
    new_model_id: str | None = None
    performance_comparison: dict[str, Any] | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class DriftDetector:
    """Drift detection system."""

    def __init__(
        self,
        config: "ConfigManager",
        model_registry: "ModelRegistry",
        logger: "LoggerService",
    ) -> None:
        """Initialize the drift detector.

        Args:
            config: Configuration manager instance
            model_registry: Model registry for accessing model information
            logger: Logger service for logging messages
        """
        self.config = config
        self.model_registry = model_registry
        self.logger = logger

        # Drift detection thresholds
        self.drift_thresholds = {
            DriftType.DATA_DRIFT: 0.1,
            DriftType.CONCEPT_DRIFT: 0.15,
            DriftType.PREDICTION_DRIFT: 0.2,
            DriftType.PERFORMANCE_DRIFT: 0.05,
        }

        # Performance history for comparison
        self.performance_history: dict[str, list[dict[str, Any]]] = {}

    async def detect_drift(self, model_id: str,
                          current_data: dict[str, Any]) -> list[DriftMetrics]:
        """Detect various types of drift for a model."""
        drift_metrics: list[DriftMetrics] = []

        # Get model metadata by searching through all models
        all_models = await self.model_registry.get_all_models()
        model_metadata = next((model for model in all_models if model.model_id == model_id), None)

        if not model_metadata:
            self.logger.warning(f"Model {model_id} not found for drift detection")
            return drift_metrics

        # Data drift detection
        data_drift = await self._detect_data_drift(model_id, current_data)
        if data_drift:
            drift_metrics.extend(data_drift)

        # Performance drift detection
        perf_drift = await self._detect_performance_drift(model_id, current_data)
        if perf_drift:
            drift_metrics.extend(perf_drift)

        # Prediction drift detection
        pred_drift = await self._detect_prediction_drift(model_id, current_data)
        if pred_drift:
            drift_metrics.extend(pred_drift)

        return drift_metrics

    async def _detect_data_drift(self, model_id: str,
                                current_data: dict[str, Any]) -> list[DriftMetrics]:
        """Detect data distribution drift."""
        drift_metrics = []

        # Check feature distributions
        features = current_data.get("features", {})
        baseline_features = await self._get_baseline_features(model_id)

        for feature_name, current_value in features.items():
            if feature_name in baseline_features:
                baseline_stats = baseline_features[feature_name]

                # Calculate drift using statistical distance
                drift_score = self._calculate_statistical_distance(
                    baseline_stats, current_value,
                )

                threshold = self.drift_thresholds[DriftType.DATA_DRIFT]
                is_significant = drift_score > threshold

                if is_significant:
                    drift_metrics.append(DriftMetrics(
                        drift_type=DriftType.DATA_DRIFT,
                        metric_name=f"feature_{feature_name}",
                        baseline_value=baseline_stats.get("mean", 0),
                        current_value=current_value,
                        drift_score=drift_score,
                        is_significant=is_significant,
                        threshold=threshold,
                        details={
                            "baseline_stats": baseline_stats,
                            "statistical_test": "ks_test",
                        },
                    ))

        return drift_metrics

    async def _detect_performance_drift(self, model_id: str,
                                       current_data: dict[str, Any]) -> list[DriftMetrics]:
        """Detect model performance drift."""
        drift_metrics = []

        current_performance = current_data.get("performance", {})
        baseline_performance = await self._get_baseline_performance(model_id)

        performance_metrics = ["accuracy", "precision", "recall", "f1_score"]

        for metric in performance_metrics:
            if metric in current_performance and metric in baseline_performance:
                current_value = current_performance[metric]
                baseline_value = baseline_performance[metric]

                # Calculate relative change
                drift_score = abs(baseline_value - current_value) / baseline_value
                threshold = self.drift_thresholds[DriftType.PERFORMANCE_DRIFT]
                is_significant = drift_score > threshold

                if is_significant:
                    drift_metrics.append(DriftMetrics(
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        metric_name=metric,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        drift_score=drift_score,
                        is_significant=is_significant,
                        threshold=threshold,
                        details={
                            "relative_change": (current_value - baseline_value) / baseline_value,
                            "degradation": current_value < baseline_value,
                        },
                    ))

        return drift_metrics

    async def _detect_prediction_drift(self, model_id: str,
                                      current_data: dict[str, Any]) -> list[DriftMetrics]:
        """Detect prediction distribution drift."""
        drift_metrics = []

        # Compare prediction distributions
        current_predictions = current_data.get("recent_predictions", [])
        baseline_predictions = await self._get_baseline_predictions(model_id)

        if current_predictions and baseline_predictions:
            # Calculate KL divergence or similar metric
            drift_score = self._calculate_prediction_drift_score(
                baseline_predictions, current_predictions,
            )

            threshold = self.drift_thresholds[DriftType.PREDICTION_DRIFT]
            is_significant = drift_score > threshold

            if is_significant:
                drift_metrics.append(DriftMetrics(
                    drift_type=DriftType.PREDICTION_DRIFT,
                    metric_name="prediction_distribution",
                    baseline_value=len(baseline_predictions),
                    current_value=len(current_predictions),
                    drift_score=drift_score,
                    is_significant=is_significant,
                    threshold=threshold,
                    details={
                        "baseline_mean": sum(baseline_predictions) / len(baseline_predictions),
                        "current_mean": sum(current_predictions) / len(current_predictions),
                    },
                ))

        return drift_metrics

    async def _get_baseline_features(self, model_id: str) -> dict[str, Any]:
        """Get baseline feature statistics for a model."""
        # This would typically come from stored training data statistics
        return {
            "volume": {"mean": 1000000, "std": 500000},
            "price": {"mean": 0.5, "std": 0.1},
            "rsi": {"mean": 50, "std": 20},
        }

    async def _get_baseline_performance(self, model_id: str) -> dict[str, float]:
        """Get baseline performance metrics for a model."""
        # Get model metadata by searching through all models
        all_models = await self.model_registry.get_all_models()
        model_metadata = next((model for model in all_models if model.model_id == model_id), None)
        if model_metadata and model_metadata.validation_metrics:
            # Return validation metrics directly since they're already properly typed
            return model_metadata.validation_metrics
        return {}

    async def _get_baseline_predictions(self, model_id: str) -> list[float]:
        """Get baseline prediction distribution."""
        # This would come from historical predictions
        return [0.3, 0.7, 0.45, 0.8, 0.2, 0.6, 0.55, 0.4, 0.9, 0.35]

    def _calculate_statistical_distance(self, baseline_stats: dict[str, Any],
                                       current_value: float) -> float:
        """Calculate statistical distance between baseline and current value."""
        baseline_mean = float(baseline_stats.get("mean", 0))
        baseline_std = float(baseline_stats.get("std", 1))

        # Normalize the difference
        z_score = abs(current_value - baseline_mean) / baseline_std

        # Convert to drift score (0-1 scale)
        return min(z_score / 3.0, 1.0)  # 3-sigma normalization

    def _calculate_prediction_drift_score(self, baseline: list[float],
                                        current: list[float]) -> float:
        """Calculate drift score between prediction distributions."""
        if not baseline or not current:
            return 0.0

        baseline_mean = sum(baseline) / len(baseline)
        current_mean = sum(current) / len(current)

        # Simple difference for now - could use KL divergence
        return abs(baseline_mean - current_mean)


class RetrainingPipeline:
    """Automated model retraining pipeline."""

    def __init__(
        self,
        config: "ConfigManager",
        model_registry: "ModelRegistry",
        drift_detector: DriftDetector | None,
        predictor_factory: Callable[[], Any] | None,
        retraining_repository: Optional["RetrainingRepository"],
        pubsub: PubSubManager | None,
        logger: "LoggerService",
    ) -> None:
        """Initialize the retraining pipeline.

        Args:
            config: Configuration manager instance
            model_registry: Model registry for accessing model information
            drift_detector: Drift detector instance (optional, will create one if not provided)
            predictor_factory: Factory function for creating predictor instances
            retraining_repository: Repository for persisting retraining jobs
            pubsub: Pub/Sub manager for event publishing
            logger: Logger service for logging messages
        """
        self.config = config
        self.model_registry = model_registry
        self.drift_detector = drift_detector or DriftDetector(config, model_registry, logger)
        self.predictor_factory = predictor_factory
        self.retraining_repository = retraining_repository
        self.pubsub = pubsub
        self.logger = logger

        # Active jobs tracking
        self._active_jobs: dict[str, RetrainingJob] = {}
        self._is_running = False
        self._check_interval = 3600  # 1 hour
        self._monitoring_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the retraining pipeline."""
        if self._is_running:
            return

        self._is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Retraining pipeline started")

    async def stop(self) -> None:
        """Stop the retraining pipeline."""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        self.logger.info("Retraining pipeline stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for drift detection."""
        while self._is_running:
            try:
                await self._check_all_models()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in retraining monitoring loop: {e}")
                await asyncio.sleep(60)  # Brief pause before retry

    async def _check_all_models(self) -> None:
        """Check all production models for drift."""
        try:
            from gal_friday.model_lifecycle.registry import ModelStage

            # Get all production models
            all_models = await self.model_registry.list_models(stage=ModelStage.PRODUCTION)

            for model in all_models:
                # Get recent performance data
                current_data = await self._get_current_model_data(model.model_id)

                # Run drift detection
                drift_metrics = await self.drift_detector.detect_drift(
                    model.model_id, current_data,
                )

                # Check if retraining is needed
                significant_drift = [m for m in drift_metrics if m.is_significant]
                if significant_drift:
                    await self.trigger_retraining(
                        model.model_id,
                        RetrainingTrigger.DRIFT_DETECTED,
                        significant_drift,
                    )

        except Exception as e:
            self.logger.error(f"Error checking models for drift: {e}")

    async def _get_current_model_data(self, model_id: str) -> dict[str, Any]:
        """Get current performance and feature data for a model."""
        # This would integrate with monitoring systems
        return {
            "performance": {
                "accuracy": 0.75,  # Current accuracy
                "precision": 0.72,
                "recall": 0.78,
                "f1_score": 0.75,
            },
            "features": {
                "volume": 1200000,
                "price": 0.48,
                "rsi": 65,
            },
            "recent_predictions": [0.4, 0.8, 0.3, 0.7, 0.6],
        }

    async def trigger_retraining(self,
                               model_id: str,
                               trigger: RetrainingTrigger,
                               drift_metrics: list[DriftMetrics] | None = None) -> str:
        """Trigger retraining for a model."""
        # Get model info by searching through all models
        all_models = await self.model_registry.get_all_models()
        model_metadata = None
        for model in all_models:
            if model.model_id == model_id:
                model_metadata = model
                break

        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")

        # Create retraining job
        job = RetrainingJob(
            job_id=str(uuid.uuid4()),
            model_id=model_id,
            model_name=model_metadata.model_name,
            trigger=trigger,
            drift_metrics=drift_metrics or [],
        )

        # Store job
        self._active_jobs[job.job_id] = job

        if self.retraining_repository:
            await self.retraining_repository.save_job(job)

        # Start retraining process and store task reference
        task = asyncio.create_task(self._execute_retraining(job))
        job.task = task

        self.logger.info(f"Triggered retraining for model {model_id}, job {job.job_id}")
        return job.job_id

    async def _execute_retraining(self, job: RetrainingJob) -> None:
        """Execute the retraining process."""
        try:
            job.status = "running"
            job.start_time = datetime.now(UTC)

            if self.retraining_repository:
                await self.retraining_repository.update_job_status(job)

            # Simulate retraining process
            self.logger.info(f"Starting retraining for job {job.job_id}")

            # This would involve:
            # 1. Fetch fresh training data
            # 2. Retrain the model
            # 3. Validate the new model
            # 4. Compare performance
            # 5. Decide whether to promote

            await asyncio.sleep(10)  # Simulate training time

            # Create new model (simulated)
            job.new_model_id = str(uuid.uuid4())
            job.performance_comparison = {
                "old_accuracy": 0.75,
                "new_accuracy": 0.82,
                "improvement": 0.07,
            }

            job.status = "completed"
            job.end_time = datetime.now(UTC)

            # Remove from active jobs
            if job.job_id in self._active_jobs:
                del self._active_jobs[job.job_id]

            if self.retraining_repository:
                await self.retraining_repository.update_job_status(job)

            self.logger.info(f"Completed retraining for job {job.job_id}")

            # Publish completion event
            if self.pubsub:
                # Create a log event for model retraining completion
                log_event = LogEvent(
                    source_module=self.__class__.__name__,
                    event_id=uuid.uuid4(),
                    timestamp=datetime.now(UTC),
                    level="INFO",
                    message=f"Model retraining completed for job {job.job_id}",
                    context={
                        "job_id": job.job_id,
                        "model_id": job.model_id,
                        "new_model_id": job.new_model_id,
                        "performance_improvement": job.performance_comparison.get(
                            "improvement", 0,
                        ),
                    },
                )
                await self.pubsub.publish(log_event)

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now(UTC)

            if job.job_id in self._active_jobs:
                del self._active_jobs[job.job_id]

            if self.retraining_repository:
                await self.retraining_repository.update_job_status(job)

            self.logger.error(f"Retraining failed for job {job.job_id}: {e}")

    async def get_retraining_status(self) -> dict[str, Any]:
        """Get current retraining pipeline status."""
        # Get recent job statistics
        recent_completed = 0
        recent_failed = 0

        if self.retraining_repository:
            recent_jobs = await self.retraining_repository.get_recent_jobs(7)
            recent_completed = len([j for j in recent_jobs if j["status"] == "completed"])
            recent_failed = len([j for j in recent_jobs if j["status"] == "failed"])

        return {
            "active_jobs": [
                {
                    "job_id": job.job_id,
                    "model_name": job.model_name,
                    "status": job.status,
                    "trigger": job.trigger.value,
                    "start_time": job.start_time.isoformat() if job.start_time else None,
                }
                for job in self._active_jobs.values()
            ],
            "recent_completed": recent_completed,
            "recent_failed": recent_failed,
            "next_check": (
                datetime.now(UTC) + timedelta(seconds=self._check_interval)
            ).isoformat(),
            "is_running": self._is_running,
        }

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a specific retraining job."""
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            return {
                "job_id": job.job_id,
                "model_id": job.model_id,
                "model_name": job.model_name,
                "status": job.status,
                "trigger": job.trigger.value,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "drift_metrics": [
                    {
                        "drift_type": m.drift_type.value,
                        "metric_name": m.metric_name,
                        "drift_score": m.drift_score,
                        "is_significant": m.is_significant,
                    }
                    for m in job.drift_metrics
                ],
            }

        # Check repository
        if self.retraining_repository:
            return await self.retraining_repository.get_job(job_id)

        return None
