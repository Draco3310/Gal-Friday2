# Prediction Service Module
"""Machine learning prediction service for market data analysis.

This module provides infrastructure for running ML model inference against
market features, handling the prediction lifecycle from feature consumption
to prediction publishing.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional, TypeVar
import uuid

import asyncio
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# Removed: import xgboost as xgb - specific models handled by predictors
# Import necessary components
from gal_friday.core.events import (
    EventType,
    FeatureEvent,
    PredictionConfigUpdatedEvent,
    PredictionEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.interfaces.predictor_interface import PredictorInterface  # Added
from gal_friday.logger_service import LoggerService
from gal_friday.ml_prediction_pipeline import (
    MLPredictionPipeline,
    ModelTrainingConfig,
    ModelType,
    PredictionRequest as MLPredictionRequest,
)

# Forward reference for ConfigurationManager
if TYPE_CHECKING:
    from collections.abc import Coroutine

    # Use string literal to avoid circular imports
    ConfigurationManager = Any


# Set up logging for this module specifically
# For PredictionService specific logging, self.logger is preferred.

# Define a type for futures to track
T = TypeVar("T")
InferenceTaskType = asyncio.Future[dict[str, Any]]  # Return type from runners


# Removed old _run_inference_task, as this logic is now in predictor classes


# --- Custom Exceptions ---
class PredictionServiceError(Exception):
    """Base exception for PredictionService errors."""


class PredictorImportError(PredictionServiceError):
    """Raised when there's an error importing predictor classes."""

    message = "Failed to import predictor classes"


class ModelConfigError(PredictionServiceError):
    """Raised when there's an error in model configuration."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the ModelConfigError with an error message and optional context.

        Args:
            message: The error message
            context: Optional dictionary with additional context about the error
        """
        self.message = message
        self.context = context
        super().__init__(self.message)

    invalid_config = "Invalid model configuration"
    unsupported_predictor = "Unsupported predictor type"


class PredictorInitError(PredictionServiceError):
    """Raised when there's an error initializing a predictor."""

    message = "Failed to initialize predictor"


class CriticalModelError(PredictionServiceError):
    """Raised when a critical model fails to load or initialize."""

    load_failed = "Failed to load critical models"
    init_failed = "Failed to initialize critical models"


# --- PredictionService Class ---
class PredictionService:
    """Consume feature events and run ML model inference.

    This class handles feature events (which contain feature data structured as
    `PublishedFeaturesV1.model_dump()`), runs ML model inference for multiple
    configured models, and publishes prediction events.
    Predictor implementations (e.g., XGBoostPredictor, SKLearnPredictor) now expect
    pre-scaled, numerical features directly from the FeatureEngine, as scaling is
    centralized upstream.
    """

    def __init__(
        self,
        config: dict[str, Any],  # Overall application config
        pubsub_manager: PubSubManager,
        process_pool_executor: ProcessPoolExecutor,
        logger_service: LoggerService,
        configuration_manager: Optional["ConfigurationManager"] = None,  # Added
    ) -> None:
        """Initialize the PredictionService.

        Args:
        ----
            config (dict[str, Any]): Overall application configuration.
                           Expected to contain a 'prediction_service' key with a 'models' list[Any].
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            process_pool_executor (ProcessPoolExecutor): Executor for running inference.
            logger_service (LoggerService): The shared logger instance.
            configuration_manager (Optional[ConfigurationManager]):
                Optional configuration manager instance.
        """
        self._service_config = config.get("prediction_service", {})
        self.pubsub = pubsub_manager
        self._process_pool_executor: ProcessPoolExecutor | None = process_pool_executor
        self.logger = logger_service  # LoggerService is already initialized
        self._is_running = False
        self._source_module = self.__class__.__name__
        self.configuration_manager = configuration_manager  # Store it

        # Enhanced shutdown handling
        self._shutting_down = asyncio.Event()
        self._shutdown_timeout = self._service_config.get("shutdown_timeout_seconds", 30.0)
        self._in_flight_tasks: dict[str, asyncio.Task[Any]] = {}  # task_id -> Task[Any]

        self._feature_event_handler: Callable[
            [FeatureEvent], Coroutine[Any, Any, None],
        ] = self._handle_feature_event
        # Added handler for config updates
        self._config_update_handler: Callable[
            [PredictionConfigUpdatedEvent],
            Coroutine[Any, Any, None],
        ] = self._handle_prediction_config_updated_event

        # Model configurations and initialized predictors/runners
        self._model_configs: list[dict[str, Any]] = self._service_config.get("models", [])
        self._predictors: dict[str, PredictorInterface] = {}
        self._predictor_runners: dict[str, Callable[..., Any]] = {}
        self._lstm_feature_buffers: dict[str, deque[Any]] = {}  # Added for LSTM sequence buffering

        # Initialize ML pipeline for training and advanced predictions
        self._ml_pipeline = MLPredictionPipeline(
            config=self._service_config.get("ml_pipeline", {}),
            logger=logger_service,
        )

        if not self._model_configs:
            self.logger.warning(
                "PredictionService: No models configured under 'prediction_service.models'.")
            # Depending on strictness, could raise ValueError here

        self._initialize_predictors()

        # Track tasks submitted to the executor
        self._active_inference_tasks: set[InferenceTaskType] = set()

    def _get_predictor_map(self) -> dict[str, Any]:
        """Import and return a mapping of predictor types to their classes and runners.

        Returns:
        -------
            dict[str, Any]: Mapping[Any, Any] of predictor types to (predictor_class, runner_method) tuples.

        Raises:
        ------
            PredictorImportError: If any required predictor class cannot be imported.
        """
        try:
            from gal_friday.predictors.lstm_predictor import LSTMPredictor
            from gal_friday.predictors.sklearn_predictor import SKLearnPredictor
            from gal_friday.predictors.xgboost_predictor import XGBoostPredictor
        except ImportError as e:
            self.logger.critical(
                "CRITICAL: Failed to import predictor classes: %s. "
                "PredictionService cannot function.",
                str(e),
                exc_info=True)
            raise PredictorImportError(PredictorImportError.message) from e
        else:
            return {
                "xgboost": (XGBoostPredictor, XGBoostPredictor.run_inference_in_process),
                "sklearn": (SKLearnPredictor, SKLearnPredictor.run_inference_in_process),
                "lstm": (LSTMPredictor, LSTMPredictor.run_inference_in_process),
            }

    def _validate_model_config(
        self,
        model_conf: dict[str, Any],
    ) -> tuple[str, str, str, type[PredictorInterface], Callable[..., Any], dict[str, Any], bool]:
        """Validate model configuration and extract necessary fields."""
        model_id: str = str(model_conf.get("model_id", ""))
        predictor_type: str = str(model_conf.get("predictor_type", ""))
        model_path: str = str(model_conf.get("model_path", ""))
        is_critical: bool = bool(model_conf.get("is_critical", False))

        if not all([model_id, predictor_type, model_path]):
            log_msg = (
                "Missing required model configuration fields. Required: 'model_id', "
                "'predictor_type', 'model_path'"
            )
            error_msg = f"{ModelConfigError.invalid_config}: {log_msg}"
            context = {"config": model_conf}
            self.logger.exception(
                error_msg,
                source_module=self._source_module,
                context=context)
            raise ModelConfigError(error_msg, context=context)

        predictor_map = self._get_predictor_map()
        predictor_info = predictor_map.get(predictor_type.lower())
        if not predictor_info:
            log_msg = f"Unsupported predictor type: {predictor_type}"
            error_context = {
                "predictor_type": predictor_type,
                "supported_types": list[Any](predictor_map.keys()),
            }
            error_msg = f"{ModelConfigError.unsupported_predictor}: {log_msg}"
            self.logger.error(
                error_msg,
                source_module=self._source_module,
                context=error_context)
            raise ModelConfigError(error_msg, context=context)

        predictor_class, runner_method = predictor_info
        return (
            model_id,
            model_path,
            predictor_type,
            predictor_class,
            runner_method,
            model_conf,
            is_critical)

    def _initialize_lstm_buffer(self, model_id: str, model_conf: dict[str, Any]) -> None:
        """Initialize LSTM feature buffer if needed."""
        sequence_length: int = int(model_conf.get("sequence_length", 1))
        if sequence_length < 1:
            self.logger.warning(
                "Invalid sequence_length for LSTM model %(model_id)s. Defaulting to 1.",
                source_module=self._source_module,
                context={"model_id": model_id, "sequence_length": sequence_length})
            sequence_length = 1  # Treat as non-sequential if invalid

        if sequence_length > 1:  # Only create buffer if sequence_length > 1
            self._lstm_feature_buffers[model_id] = deque(maxlen=sequence_length)
            self.logger.info(
                "Initialized feature buffer for LSTM model %(model_id)s with sequence "
                "length %(seq_length)s.",
                source_module=self._source_module,
                context={"model_id": model_id, "seq_length": sequence_length})

    def _initialize_predictors(self) -> None:
        """Initialize predictor instances, runners, and LSTM buffers from configuration."""
        # Call to populate internal state, but we don't need to store the return value
        self._get_predictor_map()

        # Clear existing predictors and buffers before re-initializing (for dynamic reload)
        self._predictors.clear()
        self._predictor_runners.clear()
        self._lstm_feature_buffers.clear()

        loaded_critical_models = 0
        total_critical_models = sum(1 for mc in self._model_configs if mc.get("critical"))

        for orig_model_conf in self._model_configs:
            validation_result = self._validate_model_config(orig_model_conf)
            (
                model_id,
                model_path,
                predictor_type,
                predictor_class,
                runner_method,
                model_conf,
                is_critical) = validation_result

            try:
                predictor_config = self._PredictorConfig(
                    model_id=model_id,
                    model_path=model_path,
                    predictor_type=predictor_type,
                    predictor_class=predictor_class,
                    runner_method=runner_method,
                    model_conf=model_conf,
                    is_critical=is_critical)
                self._initialize_single_predictor(predictor_config)
                if is_critical:
                    loaded_critical_models += 1

            except Exception as e:
                if is_critical:
                    self.logger.critical(
                        "Failed to initialize predictor for model_id %(model_id)s "
                        "(Critical: %(is_critical)s)",
                        source_module=self._source_module,
                        context={"model_id": model_id, "is_critical": is_critical})
                    raise PredictorInitError(PredictorInitError.message) from e
                self.logger.exception(
                    "Failed to initialize predictor for model_id %(model_id)s "
                    "(Critical: %(is_critical)s)",
                    source_module=self._source_module,
                    context={"model_id": model_id, "is_critical": is_critical})

        self._validate_critical_models_loaded(loaded_critical_models, total_critical_models)
        self.logger.info(
            "Predictor initialization complete. %(predictor_count)s predictors loaded.",
            source_module=self._source_module,
            context={
                "predictors_loaded": len(self._predictors),
                "predictor_count": len(self._predictors),
            })

    @dataclass
    class _PredictorConfig:
        """Configuration for initializing a predictor instance."""

        model_id: str
        model_path: str
        predictor_type: str
        predictor_class: type[PredictorInterface]
        runner_method: Callable[..., Any]
        model_conf: dict[str, Any]
        is_critical: bool

    def _initialize_single_predictor(
        self,
        config: _PredictorConfig) -> None:
        """Initialize a single predictor instance.

        Args:
            config: Configuration object containing all necessary parameters
                for initializing the predictor.
        """
        predictor_instance = config.predictor_class(
            model_path=config.model_path,
            model_id=config.model_id,
            config=config.model_conf)
        self._predictors[config.model_id] = predictor_instance
        self._predictor_runners[config.model_id] = config.runner_method

        # Initialize LSTM feature buffer if this is an LSTM model
        if config.predictor_type == "lstm":
            self._initialize_lstm_buffer(config.model_id, config.model_conf)

        self.logger.info(
            "Initialized predictor - ID: %(model_id)s (Critical: %(is_critical)s), "
            "Type: %(predictor_type)s, Model: %(model_path)s, "
            "Scaler: %(scaler_path)s",
            source_module=self._source_module,
            context={
                "model_id": config.model_id,
                "predictor_type": config.predictor_type,
                "model_path": config.model_path,
                "is_critical": config.is_critical,
                "scaler_path": config.model_conf.get("scaler_path", "None"),
            })

    def _validate_critical_models_loaded(
        self,
        loaded_critical_models: int,
        total_critical_models: int) -> None:
        """Validate that all critical models were loaded successfully."""
        if total_critical_models > 0 and loaded_critical_models < total_critical_models:
            # Calculate how many models failed to load, but don't need to store in variable
            self.logger.critical(
                "Critical model initialization failed. Loaded %(loaded)s out of %(total)s "
                "critical models.",
                source_module=self._source_module,
                context={
                    "loaded_critical_models": loaded_critical_models,
                    "total_critical_models": total_critical_models,
                    "loaded": loaded_critical_models,
                    "total": total_critical_models,
                    "error": "Critical model initialization failed",
                })
            raise CriticalModelError(CriticalModelError.load_failed)

    async def start(self) -> None:
        """Start listening for feature events and configuration updates.

        Predictor assets are loaded during __init__.
        """
        if self._is_running:
            self.logger.warning(
                "PredictionService already running.",
                source_module=self._source_module)
            return

        # Critical model check: _initialize_predictors raises RuntimeError if critical models fail
        # This check is implicitly handled because if _initialize_predictors (called in __init__)
        # fails with a RuntimeError for critical models, the service instance won't be successfully
        # created, or if it is, this start method might not be reached or should also check a
        # post-init flag. For robustness, let's ensure predictors are loaded if init didn't raise.
        if not self._predictors and sum(1 for mc in self._model_configs if mc.get("critical")) > 0:
            # This case implies _initialize_predictors might have failed to load critical models
            # but didn't raise an exception that stopped __init__
            # (e.g. if it caught it and only logged).
            # The current _initialize_predictors *does* raise, so this is more of a safeguard.
            self.logger.critical(
                "PredictionService cannot start: Critical models failed initialization. "
                "Check logs.",
                source_module=self._source_module)
            raise CriticalModelError(CriticalModelError.init_failed)
        if not self._predictors:
            self.logger.info("PredictionService starting with no valid predictors initialized.")
            # Allow starting if no models are critical, or no models configured.

        self._is_running = True
        try:
            self.pubsub.subscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info("Subscribed to %s.", EventType.FEATURES_CALCULATED.name)

            # Subscribe to config updates
            if (
                hasattr(EventType, "PREDICTION_CONFIG_UPDATED")
                and self.configuration_manager is not None
            ):
                # Check if configuration_manager is available because it's optional in __init__
                self.pubsub.subscribe(
                    EventType.PREDICTION_CONFIG_UPDATED,
                    self._config_update_handler)
                self.logger.info(
                    "Subscribed to %(event_type)s for dynamic model reloading.",
                    source_module=self._source_module,
                    context={"event_type": EventType.PREDICTION_CONFIG_UPDATED.name})
            elif self.configuration_manager is None:
                self.logger.info(
                    "ConfigurationManager not provided; dynamic model reloading via events "
                    "will be disabled.",
                    source_module=self._source_module)

        except Exception:
            self.logger.exception(
                "Error during pubsub subscription in start()",
                source_module=self._source_module)
            self._is_running = False  # Don't consider service started if subscriptions fail
            raise  # Re-raise to signal failure to start properly

        loaded_count = len(
            [p for p in self._predictors.values() if getattr(p, "is_critical", False)])
        total_critical = sum(
            1 for p in self._predictors.values() if getattr(p, "is_critical", False)
        )
        self.logger.info(
            "Initialized %(loaded_count)s of %(total_critical)s critical models",
            source_module=self._source_module,
            context={
                "models_loaded": len(self._predictors),
                "loaded_count": loaded_count,
                "total_critical": total_critical,
            })
        for model_id, predictor in self._predictors.items():
            predictor_type = type(predictor).__name__
            expected_features = predictor.expected_feature_names
            context = {
                "model_id": str(model_id),
                "predictor_type": str(predictor_type),
                "expected_features": str(expected_features),
            }
            self.logger.info(
                "Initialized predictor for model_id: %(model_id)s, type: %(predictor_type)s",
                source_module=self._source_module,
                context=context)

    async def stop(self) -> None:
        """Stop event processing and cancel pending inferences with graceful shutdown."""
        if not self._is_running:
            self.logger.info(
                "PredictionService already stopped or not started.",
                source_module=self._source_module)
            return

        self.logger.info(
            "Initiating graceful shutdown of PredictionService...",
            source_module=self._source_module)

        # Set shutdown flag to prevent new tasks
        self._shutting_down.set()
        self._is_running = False

        # Unsubscribe from FeatureEvent first
        try:
            self.pubsub.unsubscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info(
                "Unsubscribed from %(event_type)s.",
                source_module=self._source_module,
                context={"event_type": EventType.FEATURES_CALCULATED.name})
        except Exception:
            self.logger.exception(
                "Error unsubscribing from %(event_type)s",
                source_module=self._source_module,
                context={"event_type": EventType.FEATURES_CALCULATED.name})

        # Unsubscribe from PREDICTION_CONFIG_UPDATED
        if (
            hasattr(EventType, "PREDICTION_CONFIG_UPDATED")
            and self.configuration_manager is not None
        ):
            try:
                self.pubsub.unsubscribe(
                    EventType.PREDICTION_CONFIG_UPDATED,
                    self._config_update_handler)
                self.logger.info(
                    "Unsubscribed from %(event_type)s.",
                    source_module=self._source_module,
                    context={"event_type": EventType.PREDICTION_CONFIG_UPDATED.name})
            except Exception:
                self.logger.exception(
                    "Error unsubscribing from %(event_type)s",
                    source_module=self._source_module,
                    context={"event_type": EventType.PREDICTION_CONFIG_UPDATED.name})

        # Handle in-flight tasks gracefully
        await self._graceful_shutdown_tasks()

        self.logger.info(
            "PredictionService stopped.",
            source_module=self._source_module)

    async def _graceful_shutdown_tasks(self) -> None:
        """Gracefully shutdown in-flight prediction tasks."""
        # Get all active tasks
        all_tasks = list[Any](self._active_inference_tasks) + list[Any](self._in_flight_tasks.values())

        if not all_tasks:
            self.logger.info(
                "No in-flight prediction tasks to wait for.",
                source_module=self._source_module)
            return

        self.logger.info(
            "Waiting for %(task_count)s in-flight prediction tasks to complete...",
            source_module=self._source_module,
            context={"task_count": len(all_tasks)})

        # Wait for tasks to complete with timeout
        try:
            done, pending = await asyncio.wait(
                all_tasks,
                timeout=self._shutdown_timeout,
                return_when=asyncio.ALL_COMPLETED)

            # Handle completed tasks
            for task in done:
                try:
                    await task  # Re-raise any exceptions for logging
                except asyncio.CancelledError:
                    self.logger.debug(
                        "Prediction task was cancelled during shutdown",
                        source_module=self._source_module)
                except Exception as e:
                    self.logger.exception(
                        "Prediction task failed during shutdown: %(error)s",
                        source_module=self._source_module,
                        context={"error": str(e)})

            # Cancel any remaining tasks
            if pending:
                self.logger.warning(
                    "Cancelling %(pending_count)s prediction tasks that didn't complete in time",
                    source_module=self._source_module,
                    context={"pending_count": len(pending)})
                for task in pending:
                    task.cancel()

                # Wait a short time for cancellation to propagate
                await asyncio.sleep(0.1)

        except Exception:
            self.logger.exception(
                "Error during graceful shutdown of prediction tasks",
                source_module=self._source_module)

        # Clear tracking sets
        self._active_inference_tasks.clear()
        self._in_flight_tasks.clear()

        self.logger.info(
            "Prediction task shutdown complete",
            source_module=self._source_module)

    def _apply_confidence_floor(
        self,
        predictions: list[dict[str, Any]],
        confidence_floor: float,
        target: str | None = None) -> list[dict[str, Any]]:
        """Apply confidence floor filtering to predictions.

        Args:
            predictions: List of prediction dictionaries
            confidence_floor: Minimum confidence threshold (0.0 to 1.0)
            target: Optional target name for logging context

        Returns:
            Filtered list[Any] of predictions that meet the confidence floor
        """
        if confidence_floor <= 0.0:
            return predictions  # No filtering needed

        passed_predictions = []
        target_context = f" for target {target}" if target else ""

        for prediction in predictions:
            model_id = prediction.get("model_id", "unknown")
            model_confidence = prediction.get("confidence")

            if model_confidence is None:
                # Policy: Assume predictions without confidence pass the floor
                self.logger.debug(
                    "Prediction from %(model_id)s%(target_context)s has no confidence score. "
                    "Assuming it passes the floor.",
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "target_context": target_context,
                    })
                passed_predictions.append(prediction)
                continue

            if model_confidence >= confidence_floor:
                passed_predictions.append(prediction)
            else:
                self.logger.info(
                    "Prediction from %(model_id)s%(target_context)s (confidence: %(confidence).3f) "
                    "dropped. Below floor of %(floor).3f.",
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "target_context": target_context,
                        "confidence": model_confidence,
                        "floor": confidence_floor,
                    })

        return passed_predictions

    def _get_confidence_floor_for_target(self, target: str) -> float:
        """Get the confidence floor for a specific prediction target.

        Args:
            target: The prediction target name

        Returns:
            Confidence floor value (0.0 to 1.0)
        """
        # Check for per-target configuration first
        confidence_floors = self._service_config.get("confidence_floors", {})
        if isinstance(confidence_floors, dict) and target in confidence_floors:
            return float(confidence_floors[target])

        # Fall back to global confidence floor
        return float(self._service_config.get("confidence_floor", 0.0))

    async def _handle_feature_event(self, event: FeatureEvent) -> None:
        """Handles incoming `FeatureEvent` objects.

        The method performs the following actions:
        1.  Validates the event type.
        2.  Checks if the service is running and if the `ProcessPoolExecutor` is available.
        3.  If the event contains features from a Pydantic model (i.e., `event.features`
            is a dictionary of feature names to float values):
            a.  For LSTM models, it prepares the 1D feature vector for the current timestep
                using `_prepare_features_for_model` and appends it to the respective
                model's sequence buffer (`self._lstm_feature_buffers`).
            b.  It then calls `_trigger_predictions_if_ready` to check which models
                (LSTM or non-LSTM) have sufficient data/are ready for inference.
        4.  If `event.features` is not in the expected format, logs a warning.

        Args:
            event: The `FeatureEvent` containing the calculated features. The
                   `event.features` attribute is expected to be a dictionary
                   mapping feature names (strings) to their float values, typically
                   derived from `PublishedFeaturesV1.model_dump()`.
        """
        # Type system guarantees event is a FeatureEvent
        if self._process_pool_executor is None:
            self.logger.exception(
                "ProcessPoolExecutor not available. Cannot run predictions.",
                source_module=self._source_module,
                context={"executor_status": "unavailable"})
            return

        if not self._is_running:
            self.logger.warning(
                "PredictionService is not running. Ignoring feature event.",
                source_module=self._source_module,
                context={"service_status": "not_running"})
            return

        # Update LSTM buffers first
        for model_id, buffer in self._lstm_feature_buffers.items():
            predictor_instance = self._predictors.get(model_id)
            if not predictor_instance:  # Should not happen if initialized correctly
                continue
            expected_features = predictor_instance.expected_feature_names
            if not expected_features:
                log_msg = (
                    f"LSTM Model {model_id} has no expected_feature_names. "
                    "Cannot buffer features."
                )
                self.logger.warning(
                    log_msg,
                    source_module=self._source_module,
                    context={"model_id": model_id})
                continue

            # Prepare the 1D feature vector for this specific LSTM model
            feature_vector_1d = self._prepare_features_for_model(event.features, expected_features)
            if feature_vector_1d is not None:
                buffer.append(feature_vector_1d)
                log_msg = (
                    f"Appended features to buffer for LSTM model {model_id}. "
                    f"Buffer size: {len(buffer)}/{buffer.maxlen}"
                )
                self.logger.debug(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "buffer_size": len(buffer),
                        "maxlen": buffer.maxlen,
                    })
            else:
                log_msg = (
                    f"Could not prepare feature vector for LSTM {model_id} "
                    f"from event {event.event_id}. Not buffering."
                )
                self.logger.warning(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "event_id": event.event_id,
                    })

        # Now check which models are ready for prediction and trigger pipeline
        await self._trigger_predictions_if_ready(event)

    async def _trigger_predictions_if_ready(self, event: FeatureEvent) -> None:
        """Check which models are ready for prediction and trigger pipeline.

        Determines which models are ready for prediction (non-LSTMs or LSTMs with full buffers)
        and launches the prediction pipeline for them.
        """
        # List of dicts containing model info and feature inputs
        ready_models_info: list[dict[str, Any]] = []

        for model_id, predictor_instance in self._predictors.items():
            model_config = next(
                (mc for mc in self._model_configs if mc.get("model_id") == model_id),
                None)
            if not model_config:
                continue

            predictor_type = model_config.get("predictor_type")
            feature_input_for_model: np.ndarray[Any, Any] | None = None

            if predictor_type == "lstm" and model_id in self._lstm_feature_buffers:
                buffer = self._lstm_feature_buffers[model_id]
                if len(buffer) == buffer.maxlen:  # Buffer is full, ready for sequence prediction
                    feature_input_for_model = np.array(
                        list[Any](buffer))  # Shape: (sequence_length, n_features_per_timestep)
                    log_msg = f"LSTM model {model_id} buffer full. Preparing sequence."
                    self.logger.debug(
                        log_msg,
                        source_module=self._source_module,
                        context={"model_id": model_id})
                else:
                    log_msg = (
                        f"LSTM model {model_id} buffer not full ({len(buffer)}/{buffer.maxlen}). "
                        "Skipping."
                    )
                    self.logger.debug(
                        log_msg,
                        source_module=self._source_module,
                        context={
                            "model_id": model_id,
                            "current_size": len(buffer),
                            "max_size": buffer.maxlen,
                        })
                    continue  # Not ready yet
            else:  # Non-LSTM model or LSTM with sequence_length=1 (handled by _prepare_features)
                expected_features = predictor_instance.expected_feature_names
                if not expected_features:
                    log_msg = f"Model {model_id} has no expected_features. Cannot prepare input."
                    self.logger.warning(
                        log_msg,
                        source_module=self._source_module,
                        context={"model_id": model_id})
                    continue
                feature_input_for_model = self._prepare_features_for_model(
                    event.features,
                    expected_features)

            if feature_input_for_model is not None:
                ready_models_info.append(
                    {
                        "model_id": model_id,
                        "model_config": model_config,  # Pass full config for the runner
                        "feature_input": feature_input_for_model,
                        "runner_func": self._predictor_runners[model_id],
                    })
            # Log if feature prep failed for a non-LSTM that should have run
            elif not (predictor_type == "lstm" and model_id in self._lstm_feature_buffers):
                log_msg = (
                    f"Feature prep failed for model {model_id}, event {event.event_id}. "
                    "Not adding."
                )
                self.logger.warning(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "event_id": event.event_id,
                    })

        if ready_models_info:
            log_msg = (
                f"Triggering prediction pipeline for {len(ready_models_info)} models, "
                f"event {event.event_id}."
            )
            self.logger.info(
                log_msg,
                source_module=self._source_module,
                context={
                    "models_count": len(ready_models_info),
                    "event_id": event.event_id,
                })
            # Store task reference to prevent it from being garbage collected
            task = asyncio.create_task(
                self._run_multi_model_prediction_pipeline(event, ready_models_info))
            # Add a callback to log any exceptions that might occur in the task
            task.add_done_callback(
                lambda t: t.exception() or True,  # Logs the exception if one occurred
            )
        else:
            log_msg = f"No models ready for prediction for event {event.event_id}."
            self.logger.debug(
                log_msg,
                source_module=self._source_module,
                context={"event_id": event.event_id})

    async def _run_multi_model_prediction_pipeline(
        self,
        event: FeatureEvent,
        ready_models_data: list[dict[str, Any]]) -> None:
        """Orchestrate inference for models that are ready and publish prediction events.

        Applies ensembling strategy if configured.

        Args:
            event: The original FeatureEvent.
            ready_models_data: List of dicts, each containing model_id, model_config,
                               feature_input, and runner_func.
        """
        # Submit all inference tasks
        inference_results = await self._submit_inference_tasks(event, ready_models_data)
        if not inference_results:
            return

        # Process results and get successful predictions
        successful_predictions = self._process_inference_results(
            event,
            inference_results["model_ids"],
            inference_results["results"])

        # Apply ensembling if needed and publish results
        await self._apply_ensembling_and_publish(event, successful_predictions)

    async def _submit_inference_tasks(
        self,
        event: FeatureEvent,
        ready_models_data: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Submit inference tasks for all ready models."""
        loop = asyncio.get_running_loop()
        inference_futures_dict: dict[str, asyncio.Future[Any]] = {}

        for model_data in ready_models_data:
            model_id = model_data["model_id"]
            model_config = model_data["model_config"]
            feature_input = model_data["feature_input"]
            runner_func = model_data["runner_func"]
            expected_features = model_config.get("model_feature_names", [])

            log_msg = f"Submitting inference for model {model_id}, event {event.event_id}."
            self.logger.debug(
                log_msg,
                source_module=self._source_module,
                context={
                    "model_id": model_id,
                    "event_id": str(event.event_id),
                })

            future: asyncio.Future[Any] = loop.run_in_executor(
                self._process_pool_executor,
                runner_func,
                model_id,
                model_config["model_path"],
                model_config.get("scaler_path"),
                feature_input,
                expected_features,
                model_config)
            self._active_inference_tasks.add(future)
            future.add_done_callback(lambda f: self._active_inference_tasks.discard(f))
            inference_futures_dict[model_id] = future

        if not inference_futures_dict:
            log_msg = f"No inference tasks were submitted for event {event.event_id}."
            self.logger.info(
                log_msg,
                source_module=self._source_module,
                context={"event_id": str(event.event_id)})
            return None

        futures_to_await = list[Any](inference_futures_dict.values())
        log_msg = f"Awaiting {len(futures_to_await)} inference results for event {event.event_id}."
        self.logger.debug(
            log_msg,
            source_module=self._source_module,
            context={
                "event_id": str(event.event_id),
                "num_futures": len(futures_to_await),
            })

        results = await asyncio.gather(*futures_to_await, return_exceptions=True)
        log_msg = (
            f"Received {len(results)} inference results/exceptions for event {event.event_id}."
        )
        self.logger.debug(
            log_msg,
            source_module=self._source_module,
            context={
                "event_id": str(event.event_id),
                "result_count": len(results),
            })

        return {
            "model_ids": list[Any](inference_futures_dict.keys()),
            "results": results,
        }

    def _process_inference_results(
        self,
        event: FeatureEvent,
        model_ids: list[str],
        results: list[Any]) -> list[dict[str, Any]]:
        """Process inference results and return successful predictions."""
        successful_predictions: list[dict[str, Any]] = []

        for i, result_or_exc in enumerate(results):
            model_id = model_ids[i]

            if isinstance(result_or_exc, Exception):
                log_msg = (
                    f"Inference task for model {model_id} raised an exception: "
                    f"{result_or_exc!s}"
                )
                self.logger.error(
                    log_msg,
                    source_module=self._source_module,
                    context={"model_id": model_id},
                    exc_info=result_or_exc)
                continue

            result: dict[str, Any] = result_or_exc
            if result.get("model_id") != model_id:
                log_msg = (
                    f"Mismatched model_id in result for event {event.event_id}. "
                    f"Expected {model_id}, got {result.get('model_id')}. Using expected."
                )
                self.logger.warning(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "expected_model_id": model_id,
                        "actual_model_id": result.get("model_id"),
                        "event_id": str(event.event_id),
                    })

            if "error" in result:
                log_msg = (
                    f"Inference failed for model {model_id} (event {event.event_id}): "
                    f"{result['error']}"
                )
                self.logger.error(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "event_id": str(event.event_id),
                        "error": str(result["error"]),
                    })
            elif "prediction" in result:
                model_config = next(
                    (mc for mc in self._model_configs if mc.get("model_id") == model_id),
                    None)
                if model_config:
                    successful_predictions.append(
                        {
                            "model_id": model_id,
                            "prediction_value": result["prediction"],
                            "prediction_target": model_config.get(
                                "prediction_target",
                                "unknown_target"),
                            "confidence": result.get("confidence"),
                            "config": model_config,
                        })
                else:
                    log_msg = (
                        f"Could not find config for successfully predicted model_id {model_id}. "
                        "This is unexpected."
                    )
                    self.logger.error(
                        log_msg,
                        source_module=self._source_module,
                        context={"model_id": model_id})
            else:
                log_msg = (
                    f"Invalid inference result format for model {model_id} "
                    f"(event {event.event_id}): {result!s}"
                )
                self.logger.error(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "event_id": str(event.event_id),
                        "result": str(result),
                    })

        return successful_predictions

    async def _apply_ensembling_and_publish(
        self,
        event: FeatureEvent,
        successful_predictions: list[dict[str, Any]]) -> None:
        """Apply ensembling strategy and publish final predictions."""
        ensemble_strategy = self._service_config.get("ensemble_strategy", "none").lower()
        ensemble_weights = self._service_config.get("ensemble_weights", {})

        # Group predictions by target for ensembling
        predictions_by_target: dict[str, list[dict[str, Any]]] = {}
        for pred_data in successful_predictions:
            target = pred_data["prediction_target"]
            if target not in predictions_by_target:
                predictions_by_target[target] = []
            predictions_by_target[target].append(pred_data)

        final_predictions_to_publish: list[dict[str, Any]] = []

        if ensemble_strategy == "none" or not predictions_by_target:
            # Apply confidence floor to individual predictions
            global_confidence_floor = self._get_confidence_floor_for_target("global")
            final_predictions_to_publish = self._apply_confidence_floor(
                successful_predictions,
                global_confidence_floor)
        else:
            # Apply ensembling for each target group
            for target, preds_for_target in predictions_by_target.items():
                if not preds_for_target:
                    continue

                # Apply confidence floor before ensembling
                target_confidence_floor = self._get_confidence_floor_for_target(target)
                filtered_preds = self._apply_confidence_floor(
                    preds_for_target,
                    target_confidence_floor,
                    target)

                if not filtered_preds:
                    self.logger.info(
                        "No predictions for target %(target)s met confidence floor for ensembling. "
                        "No ensemble prediction will be generated.",
                        source_module=self._source_module,
                        context={"target": target})
                    continue

                target_cleaned = target.replace("_", "")[:15]
                ensemble_id = f"ensemble_{target_cleaned}"

                if ensemble_strategy == "average":
                    self._apply_average_ensembling(
                        filtered_preds,
                        target,
                        ensemble_id,
                        final_predictions_to_publish)
                elif ensemble_strategy == "weighted_average":
                    self._apply_weighted_average_ensembling(
                        filtered_preds,
                        target,
                        ensemble_id,
                        ensemble_weights,
                        final_predictions_to_publish)
                # Add other ensembling strategies here as needed

        # Publish all final predictions
        for pred in final_predictions_to_publish:
            prediction_event = PredictionEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair=event.trading_pair,
                exchange=event.exchange,
                timestamp_prediction_for=datetime.now(UTC),  # Adjust as needed
                model_id=pred["model_id"],
                prediction_target=pred["prediction_target"],
                prediction_value=float(pred["prediction_value"]),
                confidence=(
                    float(pred.get("confidence", 0.0))
                    if pred.get("confidence") is not None
                    else None
                ),
                associated_features={
                    "triggering_features": event.features, # dict[str, float] from PublishedFeaturesV1.model_dump()
                    "metadata": {
                        "source_feature_event_id": str(event.event_id),
                        "ensemble_strategy_used": ensemble_strategy,
                    },
                })
            # Await the async _publish_prediction method
            await self._publish_prediction(prediction_event)

    def _apply_average_ensembling(
        self,
        predictions: list[dict[str, Any]],
        target: str,
        ensemble_id: str,
        output: list[dict[str, Any]]) -> None:
        """Apply average ensembling to predictions."""
        if not predictions:
            return

        avg_value = sum(p["prediction_value"] for p in predictions) / len(predictions)

        # Calculate average confidence if available
        confidences = [
            p["confidence"]
            for p in predictions
            if "confidence" in p and p["confidence"] is not None
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        output.append(
            {
                "model_id": ensemble_id,
                "prediction_value": avg_value,
                "prediction_target": target,
                "confidence": avg_confidence,
                "config": {"model_id": ensemble_id, "prediction_target": target},
            })

    def _apply_weighted_average_ensembling(
        self,
        predictions: list[dict[str, Any]],
        target: str,
        ensemble_id: str,
        weights: dict[str, float],
        output: list[dict[str, Any]]) -> None:
        """Apply weighted average ensembling to predictions.

        Args:
            predictions: List of prediction dictionaries
            target: The prediction target being ensembled
            ensemble_id: ID for the ensemble model
            weights: Dictionary mapping model IDs to their weights
            output: List to store the ensembled prediction
        """
        valid_weights = {
            k: float(v)
            for k, v in weights.items()
            if v is not None and not np.isnan(float(v or 0))
        }

        if not valid_weights:
            context = {"target": str(target), "weights": str(weights)}
            log_msg = (
                f"No valid weights for ensembling target {target}. " "Skipping weighted average."
            )
            self.logger.warning(
                log_msg,
                source_module=self._source_module,
                context=context)
            return

        weighted_sum = 0.0
        weighted_confidence_sum = 0.0
        total_weight = 0.0

        for pred in predictions:
            model_id = pred.get("model_id", "")
            weight = valid_weights.get(model_id, 0.0)

            if weight > 0:
                weighted_sum += pred["prediction_value"] * weight
                if pred.get("confidence") is not None:
                    weighted_confidence_sum += pred["confidence"] * weight
                total_weight += weight

        if total_weight > 0:
            avg_value = weighted_sum / total_weight
            avg_confidence = (
                weighted_confidence_sum / total_weight if weighted_confidence_sum > 0 else None
            )

            output.append(
                {
                    "model_id": ensemble_id,
                    "prediction_value": avg_value,
                    "prediction_target": target,
                    "confidence": avg_confidence,
                    "config": {"model_id": ensemble_id, "prediction_target": target},
                })

    def _prepare_features_for_model(
        self,
        event_features: dict[str, float],
        expected_model_features: list[str]) -> np.ndarray[Any, Any] | None:
        """Converts a feature dictionary from a `FeatureEvent` into a 1D numpy array.

        The input `event_features` is expected to be a dictionary mapping feature names
        to their float values, as produced by `PublishedFeaturesV1.model_dump()` in the
        `FeatureEngine`. This method aligns these features with the
        `expected_model_features` list[Any] provided by a specific predictor.

        Features are already scaled by `FeatureEngine`; this method focuses on ordering
        and handling any missing features (which should be rare if configurations
        are aligned).

        Args:
            event_features: A dictionary where keys are feature names (strings) and
                            values are their corresponding float values. This comes from
                            the `features` payload of a `FeatureEvent`.
            expected_model_features: A list[Any] of strings defining the order of feature
                                     names that the target model expects.

        Returns:
            A 1D `np.ndarray[Any, Any]` of float32 type containing the feature values in the
            order specified by `expected_model_features`. Missing or NaN values are
            replaced with imputed defaults using `_impute_feature_value`. Returns
            `None` if no features could be processed or if all processed features
            result in NaN.
        """
        ordered_feature_values: list[float] = []
        missing_features_log: list[str] = []
        # Type[Any] errors are less likely now as Pydantic handles validation in FeatureEngine
        # but keeping the log list[Any] in case of unexpected issues or if event_features bypasses Pydantic.
        type_errors_log: list[str] = []

        for feature_name in expected_model_features:
            value = event_features.get(feature_name)  # Value is already float or None

            if value is None:
                # Feature missing from the validated Pydantic model's output.
                missing_features_log.append(str(feature_name))
                imputed_value = self._impute_feature_value(feature_name)
                ordered_feature_values.append(imputed_value)
                self.logger.debug(
                    "Imputed missing feature '%s' with value %.4f",
                    feature_name,
                    imputed_value)
                continue

            # Check for NaN explicitly, as Pydantic allows NaN for float if not otherwise restricted
            if np.isnan(value):
                # This case means the feature was present but its value was NaN.
                imputed_value = self._impute_feature_value(feature_name)
                self.logger.debug(
                    "Feature '%s' had NaN value. Imputed with %.4f",
                    feature_name,
                    imputed_value)
                ordered_feature_values.append(imputed_value)
                continue

            # Value should be a float if it's not None and not NaN
            # No explicit float() conversion needed here as it comes from Pydantic model dump.
            ordered_feature_values.append(value)


        # Log any issues
        if missing_features_log:
            context = {"missing_features": missing_features_log}
            log_msg = "Missing features: {}".format(", ".join(missing_features_log))
            self.logger.warning(log_msg, source_module=self._source_module, context=context)
        if type_errors_log:
            context = {
                "type_errors": type_errors_log,
                "expected_features": expected_model_features,
            }
            log_msg = "Type[Any] conversion errors: {}".format(", ".join(type_errors_log))
            self.logger.warning(log_msg, source_module=self._source_module, context=context)

        if not ordered_feature_values:  # Should not happen if expected_model_features is not empty
            self.logger.error(
                "No feature values were processed. This is unexpected.",
                source_module=self._source_module)
            return None

        feature_array = np.array(ordered_feature_values, dtype=np.float32)

        # Check if all values ended up being NaN
        if np.all(np.isnan(feature_array)):
            self.logger.error(
                "All features resulted in NaN for model. Cannot make prediction.",
                source_module=self._source_module,
                context={"expected_features": expected_model_features})
            return None

        return feature_array  # Returns a 1D array

    def _impute_feature_value(self, feature_name: str) -> float:
        """Return a context-aware default value for a missing feature."""
        feature_lower = feature_name.lower()

        if "rsi" in feature_lower:
            return 50.0
        if "macd" in feature_lower:
            return 0.0
        if any(term in feature_lower for term in ["volume", "vol", "vwap"]):
            return 0.0
        if any(term in feature_lower for term in ["price", "spread", "wap"]):
            return 0.0
        if any(term in feature_lower for term in ["atr", "volatility", "stdev"]):
            return 0.0
        if "pct" in feature_lower or "percent" in feature_lower:
            return 0.0
        if "imbalance" in feature_lower:
            return 0.0
        return 0.0

    async def _publish_prediction(self, event: PredictionEvent) -> None:
        """Publish the prediction event to subscribers."""
        try:
            await self.pubsub.publish(event)
            context = {
                "event_id": str(event.event_id),
                "model_id": str(event.model_id),
                "trading_pair": event.trading_pair,
                "timestamp": str(event.timestamp_prediction_for),
                "prediction_value": event.prediction_value,
            }
            log_msg = f"Published prediction event {event.event_id} for model {event.model_id}"
            self.logger.debug(log_msg, source_module=self._source_module, context=context)
        except Exception as e:
            context = {
                "event_id": str(event.event_id),
                "model_id": str(event.model_id),
                "error": str(e),
            }
            log_msg = f"Failed to publish prediction event {event.event_id}"
            self.logger.exception(log_msg, source_module=self._source_module, context=context)

    # --- Dynamic Model Reloading ---
    async def _handle_prediction_config_updated_event(
        self,
        event: PredictionConfigUpdatedEvent) -> None:
        """Handle prediction service configuration update event.

        Re-initializes predictors based on the new configuration from the event payload.
        """
        # Type system guarantees event is a PredictionConfigUpdatedEvent
        # Get new config from well-typed event
        new_service_config = event.new_prediction_service_config

        self.logger.info(
            "Prediction_service configuration update event received. Re-initializing predictors.")

        # Store the new service-specific config part
        self._service_config = new_service_config
        self._model_configs = self._service_config.get("models", [])

        if not self._model_configs:
            self.logger.warning(
                "No models found in new configuration during re-initialization. "
                "Service may become inactive if all models are removed.")

        # Enhanced handling of in-flight tasks during reconfiguration
        await self._handle_reconfiguration_with_in_flight_tasks()

        try:
            # This will clear and repopulate predictors & buffers
            # It also raises RuntimeError if critical models fail.
            self._initialize_predictors()
            self.logger.info(
                "Predictors re-initialized successfully based on new configuration.",
                source_module=self._source_module)
            # Log new state
            log_msg = f"PredictionService reconfigured. {len(self._predictors)} models now active."
            self.logger.info(
                log_msg,
                source_module=self._source_module,
                context={"active_models": len(self._predictors)})
            for model_id, predictor in self._predictors.items():
                log_msg = f"Active Model ID: {model_id}, Type: {type(predictor).__name__}"
                self.logger.info(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "predictor_type": type(predictor).__name__,
                        "expected_features": predictor.expected_feature_names,
                    })
        except RuntimeError as e_crit:
            error_msg_template = (
                "CRITICAL: Failed to re-initialize predictors with new config due to: {}. "
                "Some critical models may have failed to load. Service might be unstable or stop."
            )
            log_msg = error_msg_template.format(e_crit)
            self.logger.critical(
                log_msg,
                source_module=self._source_module,
                context={"error": str(e_crit)})
            # If this happens, the service is likely in a bad state regarding its predictors.
            # Depending on overall application design, might need to signal a broader system issue.
        except Exception as e:
            log_msg = f"Fatal error in prediction service: {e}"
            self.logger.critical(
                log_msg,
                source_module=self._source_module,
                context={"error": str(e)})

    async def _handle_reconfiguration_with_in_flight_tasks(self) -> None:
        """Handle in-flight tasks during reconfiguration more gracefully."""
        current_tasks = list[Any](self._active_inference_tasks) + list[Any](self._in_flight_tasks.values())

        if not current_tasks:
            return

        self.logger.info(
            "Waiting for %(task_count)s in-flight tasks to complete before reconfiguration...",
            source_module=self._source_module,
            context={"task_count": len(current_tasks)})

        # Give current tasks a shorter timeout during reconfiguration
        reconfig_timeout = min(self._shutdown_timeout, 10.0)  # Max 10 seconds

        try:
            done, pending = await asyncio.wait(
                current_tasks,
                timeout=reconfig_timeout,
                return_when=asyncio.ALL_COMPLETED)

            if pending:
                self.logger.warning(
                    "Cancelling %(pending_count)s tasks that didn't complete before reconfiguration",
                    source_module=self._source_module,
                    context={"pending_count": len(pending)})
                for task in pending:
                    task.cancel()

                # Wait briefly for cancellation
                await asyncio.sleep(0.1)

        except Exception:
            self.logger.exception(
                "Error handling in-flight tasks during reconfiguration",
                source_module=self._source_module)

    async def _create_prediction_task(self, task_coro: Any, task_name: str = "") -> str:
        """Create a new prediction task with proper tracking.

        Args:
            task_coro: The coroutine to run as a task
            task_name: Optional name for the task

        Returns:
            Task[Any] ID for tracking

        Raises:
            RuntimeError: If service is shutting down
        """
        if self._shutting_down.is_set():
            raise RuntimeError("Cannot start new prediction: service is shutting down")

        task_id = str(uuid.uuid4())
        task_name_final = task_name or f"prediction-{task_id[:8]}"

        task = asyncio.create_task(task_coro, name=task_name_final)

        # Add callback to clean up the task when done
        task.add_done_callback(lambda t: self._in_flight_tasks.pop(task_id, None))

        # Store the task
        self._in_flight_tasks[task_id] = task
        return task_id

    # --- ML Pipeline Integration Methods ---

    async def train_model(self,
                         symbol: str,
                         training_data: pd.DataFrame,
                         model_type: ModelType = ModelType.RANDOM_FOREST,
                         hyperparameters: dict[str, Any] | None = None) -> str:
        """Train a new ML model for the specified symbol.

        Args:
            symbol: Trading symbol to train the model for
            training_data: DataFrame with OHLCV market data
            model_type: Type[Any] of ML model to train
            hyperparameters: Optional hyperparameters for the model

        Returns:
            Model version identifier

        Raises:
            MLPipelineError: If training fails
        """
        try:
            self.logger.info(
                "Starting model training for %(symbol)s using ML pipeline",
                source_module=self._source_module,
                context={
                    "symbol": symbol,
                    "model_type": model_type.value,
                    "data_points": len(training_data),
                },
            )

            # Create training configuration
            training_config = ModelTrainingConfig(
                model_type=model_type,
                target_column="close",
                hyperparameters=hyperparameters or {},
                cv_folds=5,
                performance_threshold=0.0,
            )

            # Use the ML pipeline for comprehensive training and deployment
            model_version = await self._ml_pipeline.train_and_deploy_model(
                symbol, training_data, training_config,
            )

            self.logger.info(
                "Model training completed for %(symbol)s. Version: %(version)s",
                source_module=self._source_module,
                context={"symbol": symbol, "version": model_version},
            )

        except Exception as e:
            self.logger.error(
                "Model training failed for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": symbol, "error": str(e)},
                exc_info=True,
            )
            raise
        else:
            return model_version

    async def predict_with_ml_pipeline(self,
                                     symbol: str,
                                     features: dict[str, float],
                                     prediction_horizon: int = 60,
                                     confidence_level: float = 0.95) -> dict[str, Any]:
        """Generate advanced prediction using the ML pipeline.

        This method provides enhanced predictions with confidence intervals,
        feature importance, and comprehensive metrics.

        Args:
            symbol: Trading symbol to predict for
            features: Dictionary of feature values
            prediction_horizon: Prediction horizon in minutes
            confidence_level: Confidence level for intervals (0.0 to 1.0)

        Returns:
            Dictionary with prediction results and metadata

        Raises:
            MLPipelineError: If prediction fails
        """
        try:
            # Create prediction request
            request = MLPredictionRequest(
                symbol=symbol,
                features=features,
                prediction_horizon=prediction_horizon,
                confidence_level=confidence_level,
            )

            # Get prediction from ML pipeline
            result = await self._ml_pipeline.predict_price(request)

            self.logger.debug(
                "ML pipeline prediction for %(symbol)s: $%(price).2f",
                source_module=self._source_module,
                context={
                    "symbol": symbol,
                    "price": result.predicted_price,
                },
            )

        except Exception as e:
            self.logger.error(
                "ML pipeline prediction failed for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": symbol, "error": str(e)},
                exc_info=True,
            )
            raise
        else:
            return {
                "symbol": result.symbol,
                "predicted_price": result.predicted_price,
                "confidence_interval": result.confidence_interval,
                "prediction_timestamp": result.prediction_timestamp,
                "model_version": result.model_version,
                "feature_importance": result.feature_importance,
                "request_id": result.request_id,
                "model_id": result.model_id,
                "accuracy_metrics": result.accuracy_metrics,
                "confidence_level": confidence_level,
            }

    def get_ml_pipeline_status(self) -> dict[str, Any]:
        """Get comprehensive ML pipeline status and metrics.

        Returns:
            Dictionary with pipeline status and performance metrics
        """
        try:
            return self._ml_pipeline.get_pipeline_status()
        except Exception as e:
            self.logger.error(
                "Failed to get ML pipeline status: %(error)s",
                source_module=self._source_module,
                context={"error": str(e)},
                exc_info=True,
            )
            return {"error": str(e)}

    def get_model_performance_metrics(self, symbol: str) -> dict[str, Any] | None:
        """Get detailed performance metrics for a specific model.

        Args:
            symbol: Trading symbol to get metrics for

        Returns:
            Dictionary with performance metrics or None if not available
        """
        try:
            return self._ml_pipeline.get_model_performance(symbol)
        except Exception as e:
            self.logger.error(
                "Failed to get model performance for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": symbol, "error": str(e)},
                exc_info=True,
            )
            return None

    async def retrain_model_if_needed(self,
                                    symbol: str,
                                    current_data: pd.DataFrame,
                                    performance_threshold: float = 0.7) -> bool:
        """Check if model needs retraining and retrain if necessary.

        Args:
            symbol: Trading symbol to check
            current_data: Current market data for retraining
            performance_threshold: Minimum performance score to avoid retraining

        Returns:
            True if model was retrained, False otherwise
        """
        try:
            # Get current model performance
            performance = self.get_model_performance_metrics(symbol)

            if not performance:
                self.logger.info(
                    "No performance data available for %(symbol)s, triggering training",
                    source_module=self._source_module,
                    context={"symbol": symbol},
                )
                await self.train_model(symbol, current_data)
                return True

            # Check if retraining is needed
            current_score = performance.get("performance_metrics", {}).get("validation_score", 0.0)

            if current_score < performance_threshold:
                self.logger.info(
                    "Model performance for %(symbol)s (%(score).4f) below threshold (%(threshold).4f), retraining",
                    source_module=self._source_module,
                    context={
                        "symbol": symbol,
                        "score": current_score,
                        "threshold": performance_threshold,
                    },
                )
                await self.train_model(symbol, current_data)
                return True

        except Exception as e:
            self.logger.error(
                "Failed to check/retrain model for %(symbol)s: %(error)s",
                source_module=self._source_module,
                context={"symbol": symbol, "error": str(e)},
                exc_info=True,
            )
            return False
        else:
            return False
