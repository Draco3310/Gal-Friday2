"""A/B Testing Experiment Manager for model comparison."""

import asyncio
import random
import uuid
from collections.abc import Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy import stats

from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.dal.repositories.experiment_repository import ExperimentRepository
from gal_friday.logger_service import LoggerService

if TYPE_CHECKING:
    from gal_friday.model_lifecycle.registry import ModelRegistry

# Type variable for generic event type
T = TypeVar("T")
# Event handler type
event_handler = Callable[[Any], Coroutine[Any, Any, None]]

# Constants
MIN_SAMPLES_FOR_SIGNIFICANCE = 30
CONFIDENCE_LEVEL_95 = 0.05
EXPLORATION_RATE = 0.1
MIN_IMPROVEMENT_PERCENT = 5.0
DEFAULT_TRAFFIC_SPLIT = 0.5
DEFAULT_CONFIDENCE_LEVEL = Decimal("0.95")
DEFAULT_MIN_DETECTABLE_EFFECT = Decimal("0.01")


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AllocationStrategy(Enum):
    """Traffic allocation strategies."""
    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    WEIGHTED = "weighted"
    EPSILON_GREEDY = "epsilon_greedy"


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    control_model_id: str = ""
    treatment_model_id: str = ""
    allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM
    traffic_split: Decimal = Decimal("0.5")  # Percentage to treatment
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    min_samples_per_variant: int = 1000
    primary_metric: str = "prediction_accuracy"
    secondary_metrics: list[str] = field(default_factory=list)
    confidence_level: Decimal = Decimal("0.95")
    minimum_detectable_effect: Decimal = Decimal("0.01")
    max_loss_threshold: Decimal | None = None  # For early stopping

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "control_model_id": self.control_model_id,
            "treatment_model_id": self.treatment_model_id,
            "allocation_strategy": self.allocation_strategy.value,
            "traffic_split": str(self.traffic_split),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "min_samples_per_variant": self.min_samples_per_variant,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "confidence_level": str(self.confidence_level),
            "minimum_detectable_effect": str(self.minimum_detectable_effect),
            "max_loss_threshold": (
                str(self.max_loss_threshold)
                if self.max_loss_threshold
                else None
            ),
        }


@dataclass
class VariantPerformance:
    """Performance metrics for a model variant."""
    model_id: str
    variant_name: str  # 'control' or 'treatment'
    sample_count: int = 0

    # Prediction performance
    predictions_made: int = 0
    correct_predictions: int = 0

    # Trading performance
    signals_generated: int = 0
    profitable_trades: int = 0
    total_return: Decimal = Decimal("0")
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal | None = None

    # Statistical metrics
    mean_accuracy: float | None = None
    std_accuracy: float | None = None
    confidence_interval: tuple[float, float] | None = None

    def update_metrics(self, outcome: dict[str, Any]) -> None:
        """Update performance metrics with new outcome."""
        self.sample_count += 1
        self.predictions_made += 1

        if outcome.get("correct_prediction"):
            self.correct_predictions += 1

        if outcome.get("signal_generated"):
            self.signals_generated += 1

        if outcome.get("trade_profitable"):
            self.profitable_trades += 1

        if "return" in outcome:
            self.total_return += Decimal(str(outcome["return"]))

        # Update accuracy
        if self.predictions_made > 0:
            self.mean_accuracy = self.correct_predictions / self.predictions_made


class ExperimentManager:
    """Manages A/B testing experiments for model comparison."""

    def __init__(
        self,
        config: ConfigManager,
        model_registry: "ModelRegistry",
        experiment_repo: ExperimentRepository,
        pubsub: PubSubManager,
        logger: LoggerService,
    ) -> None:
        """Initialize the ExperimentManager.

        Args:
            config: Configuration manager instance.
            model_registry: Model registry for accessing models.
            experiment_repo: Repository for experiment data.
            pubsub: PubSub manager for event handling.
            logger: Logger instance for logging.
        """
        self.config = config
        self.model_registry = model_registry
        self.experiment_repo = experiment_repo
        self.pubsub = pubsub
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Active experiments
        self.active_experiments: dict[str, ExperimentConfig] = {}
        self.experiment_performance: dict[str, dict[str, VariantPerformance]] = {}

        # Configuration
        self.max_concurrent_experiments = config.get_int("experiments.max_concurrent", 3)
        self.auto_stop_on_significance = config.get_bool(
            "experiments.auto_stop_on_significance", True,
        )
        self.check_interval_minutes = config.get_int("experiments.check_interval_minutes", 60)

        # State
        self._monitor_task: asyncio.Task[None] | None = None
        self._prediction_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None

    async def start(self) -> None:
        """Start the experiment manager."""
        self.logger.info(
            "Starting experiment manager",
            source_module=self._source_module,
        )

        # Load active experiments from database
        await self._load_active_experiments()

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_experiments())

        # Subscribe to prediction events for routing
        self._prediction_handler = self._route_prediction
        # Note: EventType.PREDICTION_REQUESTED doesn't exist, using PREDICTION_GENERATED instead
        # self.pubsub.subscribe(
        #     EventType.PREDICTION_GENERATED,
        #     self._prediction_handler,
        # )  # type: ignore[attr-defined]

    async def stop(self) -> None:
        """Stop the experiment manager."""
        if self._monitor_task:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task

        if self._prediction_handler:
            # self.pubsub.unsubscribe(EventType.PREDICTION_GENERATED, self._prediction_handler)
            pass

    async def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B testing experiment."""
        try:
            # Validate configuration
            await self._validate_experiment_config(config)

            # Check concurrent experiment limit
            max_concurrent = self.max_concurrent_experiments
            if len(self.active_experiments) >= max_concurrent:
                msg = f"Maximum concurrent experiments ({max_concurrent}) reached"
                raise ValueError(msg)

            # Initialize performance tracking
            self.experiment_performance[config.experiment_id] = {
                "control": VariantPerformance(
                    model_id=config.control_model_id,
                    variant_name="control",
                ),
                "treatment": VariantPerformance(
                    model_id=config.treatment_model_id,
                    variant_name="treatment",
                ),
            }

            # Save to database
            await self.experiment_repo.save_experiment(config)

            # Add to active experiments
            self.active_experiments[config.experiment_id] = config

            self.logger.info(
                f"Created experiment: {config.name}",
                source_module=self._source_module,
                context={
                    "experiment_id": config.experiment_id,
                    "control": config.control_model_id,
                    "treatment": config.treatment_model_id,
                },
            )

            return config.experiment_id

        except Exception:
            self.logger.exception(
                "Failed to create experiment",
                source_module=self._source_module,
            )
            raise

    async def _validate_experiment_config(self, config: ExperimentConfig) -> None:
        """Validate experiment configuration."""
        # Check models exist and are in appropriate stages
        # Note: get_model_by_id doesn't exist, using get_model instead
        control_model = await self.model_registry.get_model(config.control_model_id)  # type: ignore[attr-defined]
        treatment_model = await self.model_registry.get_model(config.treatment_model_id)  # type: ignore[attr-defined]

        if not control_model:
            raise ValueError(f"Control model not found: {config.control_model_id}")

        if not treatment_model:
            raise ValueError(f"Treatment model not found: {config.treatment_model_id}")

        # Validate traffic split
        if not (0 < config.traffic_split < 1):
            raise ValueError(f"Traffic split must be between 0 and 1: {config.traffic_split}")

        # Validate dates
        if config.end_time and config.end_time <= config.start_time:
            raise ValueError("End time must be after start time")

    async def _route_prediction(self, event: T) -> None:
        """Route prediction requests to appropriate model based on experiments."""
        # This would intercept prediction requests and route to the appropriate model
        # based on active experiments and allocation strategy

        for exp_id, config in self.active_experiments.items():
            if self._should_participate_in_experiment(event, config):
                variant = self._select_variant(event, config)

                # Route to appropriate model
                if variant == "control":
                    model_id = config.control_model_id
                else:
                    model_id = config.treatment_model_id

                # Record assignment
                await self._record_assignment(exp_id, variant, event)

                # Update event with selected model
                event.experiment_info = {
                    "experiment_id": exp_id,
                    "variant": variant,
                    "model_id": model_id,
                }

                break

    def _should_participate_in_experiment(
        self,
        event: T,
        config: ExperimentConfig,
    ) -> bool:
        """Determine if this prediction should participate in the experiment.

        Args:
            event: The prediction event to check.
            config: Experiment configuration.

        Returns:
            bool: True if the event should participate, False otherwise.
        """
        now = datetime.now(UTC)
        return not (now < config.start_time or (config.end_time and now > config.end_time))

    def _select_variant(
        self,
        event: T,
        config: ExperimentConfig,
    ) -> str:
        """Select variant based on allocation strategy.

        Args:
            event: The prediction event.
            config: Experiment configuration.

        Returns:
            str: Selected variant ('treatment' or 'control').
        """
        traffic_split = float(config.traffic_split)
        # Not used for security, just for random assignment
        rand_val = random.random()  # noqa: S311

        if config.allocation_strategy == AllocationStrategy.RANDOM:
            return "treatment" if rand_val < traffic_split else "control"

        if config.allocation_strategy == AllocationStrategy.DETERMINISTIC:
            # Hash-based assignment for consistency
            event_id = getattr(event, "event_id", str(id(event)))
            hash_value = hash(f"{event_id}{config.experiment_id}")
            return "treatment" if (hash_value % 100) < (traffic_split * 100) else "control"

        if config.allocation_strategy == AllocationStrategy.EPSILON_GREEDY:
            # Exploit best performer with probability (1-epsilon)
            if rand_val < EXPLORATION_RATE:
                # Explore with equal probability, using same random value for consistency
                explore_threshold = EXPLORATION_RATE / 2
                return "treatment" if rand_val < explore_threshold else "control"
            # Exploit
            return self._get_best_performer(config.experiment_id)

        # Default to random
        return "treatment" if rand_val < traffic_split else "control"

    def _get_best_performer(
        self,
        experiment_id: str,
    ) -> str:
        """Get the best performing variant so far."""
        performance = self.experiment_performance.get(experiment_id, {})
        control_perf = performance.get("control", VariantPerformance("", ""))
        treatment_perf = performance.get("treatment", VariantPerformance("", ""))
        control_accuracy = control_perf.mean_accuracy or 0.0
        treatment_accuracy = treatment_perf.mean_accuracy or 0.0

        return "treatment" if treatment_accuracy > control_accuracy else "control"

    async def _record_assignment(
        self,
        experiment_id: str,
        variant: str,
        event: T,
    ) -> None:
        """Record variant assignment."""
        await self.experiment_repo.record_assignment(
            experiment_id=experiment_id,
            variant=variant,
            event_id=str(event.event_id),
            timestamp=datetime.now(UTC),
        )

    async def record_outcome(
        self,
        experiment_id: str,
        event_id: str,
        outcome: dict[str, Any],
    ) -> None:
        """Record the outcome of a prediction for analysis."""
        try:
            # Get the variant assignment
            assignment = await self.experiment_repo.get_assignment(experiment_id, event_id)
            if not assignment:
                return

            variant = assignment["variant"]

            # Update performance metrics
            if experiment_id in self.experiment_performance:
                performance = self.experiment_performance[experiment_id][variant]
                performance.update_metrics(outcome)

            # Save outcome to database
            await self.experiment_repo.save_outcome(
                experiment_id=experiment_id,
                event_id=event_id,
                variant=variant,
                outcome=outcome,
                timestamp=datetime.now(UTC),
            )

        except Exception:
            self.logger.exception(
                "Failed to record experiment outcome",
                source_module=self._source_module,
            )

    async def _monitor_experiments(self) -> None:
        """Monitor experiments and check for statistical significance."""
        while True:
            try:
                await asyncio.sleep(self.check_interval_minutes * 60)

                for exp_id, config in list(self.active_experiments.items()):
                    # Check if experiment should end
                    should_stop, reason = await self._check_stopping_criteria(exp_id, config)

                    if should_stop:
                        await self._complete_experiment(exp_id, reason)

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in experiment monitoring",
                    source_module=self._source_module,
                )

    async def _check_stopping_criteria(self,
                                     experiment_id: str,
                                     config: ExperimentConfig) -> tuple[bool, str]:
        """Check if experiment should be stopped."""
        performance = self.experiment_performance.get(experiment_id, {})
        control_perf = performance.get("control", VariantPerformance("", ""))
        treatment_perf = performance.get("treatment", VariantPerformance("", ""))

        # Check time limit
        if config.end_time and datetime.now(UTC) > config.end_time:
            return True, "Time limit reached"

        # Check sample size before significance test
        if all([
            control_perf.sample_count >= config.min_samples_per_variant,
            treatment_perf.sample_count >= config.min_samples_per_variant,
            self.auto_stop_on_significance,
        ]):
            is_significant, p_value = self._calculate_significance(
                control_perf,
                treatment_perf,
            )
            if is_significant:
                return True, f"Statistical significance reached (p={p_value:.4f})"

        # Check for early stopping due to poor performance
        if all([
            config.max_loss_threshold is not None,
            treatment_perf.total_return < -config.max_loss_threshold,
        ]):
            return True, "Maximum loss threshold exceeded"

        return False, ""

    def _calculate_significance(
        self,
        control: VariantPerformance,
        treatment: VariantPerformance,
    ) -> tuple[bool, float]:
        """Calculate statistical significance between variants.

        Args:
            control: Performance metrics for control variant.
            treatment: Performance metrics for treatment variant.

        Returns:
            tuple[bool, float]: (is_significant, p_value)
        """
        if (control.predictions_made < MIN_SAMPLES_FOR_SIGNIFICANCE or
                treatment.predictions_made < MIN_SAMPLES_FOR_SIGNIFICANCE):
            return False, 1.0

        # Simple two-proportion z-test
        p1 = control.correct_predictions / control.predictions_made
        p2 = treatment.correct_predictions / treatment.predictions_made
        n1 = control.predictions_made
        n2 = treatment.predictions_made

        # Pooled proportion
        p_pool = (control.correct_predictions + treatment.correct_predictions) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se == 0:
            return False, 1.0

        # Z-score
        z = (p1 - p2) / se

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Check significance at specified confidence level
        is_significant = p_value < CONFIDENCE_LEVEL_95

        return is_significant, p_value

    async def _complete_experiment(self, experiment_id: str, reason: str) -> None:
        """Complete an experiment and analyze results."""
        try:
            config = self.active_experiments[experiment_id]
            performance = self.experiment_performance[experiment_id]

            # Calculate final statistics
            control_perf = performance["control"]
            treatment_perf = performance["treatment"]

            # Determine winner
            winner = None
            if control_perf.mean_accuracy and treatment_perf.mean_accuracy:
                if treatment_perf.mean_accuracy > control_perf.mean_accuracy:
                    winner = "treatment"
                    lift = ((treatment_perf.mean_accuracy - control_perf.mean_accuracy) /
                           control_perf.mean_accuracy * 100)
                else:
                    winner = "control"
                    lift = 0
            else:
                lift = 0

            # Save experiment results
            results = {
                "experiment_id": experiment_id,
                "status": ExperimentStatus.COMPLETED.value,
                "completion_reason": reason,
                "winner": winner,
                "lift_percentage": float(lift),
                "control_performance": {
                    "accuracy": control_perf.mean_accuracy,
                    "sample_size": control_perf.sample_count,
                    "signals_generated": control_perf.signals_generated,
                    "total_return": str(control_perf.total_return),
                },
                "treatment_performance": {
                    "accuracy": treatment_perf.mean_accuracy,
                    "sample_size": treatment_perf.sample_count,
                    "signals_generated": treatment_perf.signals_generated,
                    "total_return": str(treatment_perf.total_return),
                },
            }

            await self.experiment_repo.save_results(experiment_id, results)

            # Remove from active experiments
            del self.active_experiments[experiment_id]
            del self.experiment_performance[experiment_id]

            self.logger.info(
                f"Experiment completed: {config.name}",
                source_module=self._source_module,
                context={
                    "experiment_id": experiment_id,
                    "reason": reason,
                    "winner": winner,
                    "lift": f"{lift:.2f}%",
                },
            )

            # If treatment wins significantly, consider promoting it
            if winner == "treatment" and lift > MIN_IMPROVEMENT_PERCENT:
                self.logger.info(
                    f"Treatment model shows {lift:.2f}% improvement. "
                    "Consider promoting to production.",
                    source_module=self._source_module,
                )

        except Exception:
            self.logger.exception(
                "Failed to complete experiment",
                source_module=self._source_module,
            )

    async def _load_active_experiments(self) -> None:
        """Load active experiments from database."""
        try:
            experiments = await self.experiment_repo.get_active_experiments()

            for exp_data in experiments:
                config = ExperimentConfig(**exp_data)
                self.active_experiments[config.experiment_id] = config

                # Load performance data
                await self.experiment_repo.get_experiment_performance(
                    config.experiment_id,
                )

                # Reconstruct performance objects
                # This would need proper deserialization in production

            self.logger.info(
                f"Loaded {len(self.active_experiments)} active experiments",
                source_module=self._source_module,
            )

        except Exception:
            self.logger.exception(
                "Failed to load active experiments",
                source_module=self._source_module,
            )

    async def get_experiment_status(self, experiment_id: str) -> dict[str, Any]:
        """Get current status of an experiment."""
        if experiment_id not in self.active_experiments:
            # Try to load from database
            exp_data = await self.experiment_repo.get_experiment(experiment_id)
            if not exp_data:
                return {"error": "Experiment not found"}

            return exp_data

        config = self.active_experiments[experiment_id]
        performance = self.experiment_performance.get(experiment_id, {})

        control_perf = performance.get("control", VariantPerformance("", ""))
        treatment_perf = performance.get("treatment", VariantPerformance("", ""))

        # Calculate current statistics
        is_significant, p_value = self._calculate_significance(control_perf, treatment_perf)

        return {
            "experiment_id": experiment_id,
            "name": config.name,
            "status": "active",
            "start_time": config.start_time.isoformat(),
            "control": {
                "model_id": config.control_model_id,
                "samples": control_perf.sample_count,
                "accuracy": control_perf.mean_accuracy,
                "signals": control_perf.signals_generated,
            },
            "treatment": {
                "model_id": config.treatment_model_id,
                "samples": treatment_perf.sample_count,
                "accuracy": treatment_perf.mean_accuracy,
                "signals": treatment_perf.signals_generated,
            },
            "statistical_significance": {
                "is_significant": is_significant,
                "p_value": p_value,
                "confidence_level": float(config.confidence_level),
            },
        }
