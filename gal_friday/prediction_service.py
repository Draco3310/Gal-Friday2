# Prediction Service Module
"""
Machine learning prediction service for market data analysis.

This module provides infrastructure for running ML model inference against
market features, handling the prediction lifecycle from feature consumption
to prediction publishing.
"""

import asyncio
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, TypeVar
from collections.abc import Callable
import uuid

import numpy as np

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

# Forward reference for ConfigurationManager
if TYPE_CHECKING:
    from collections.abc import Coroutine
    from typing import Any, Optional

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

    def __init__(self, message: str, context: dict | None = None) -> None:
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

    This class handles feature events, runs ML model inference for multiple
    configured models, and publishes prediction events. Preprocessing
    (e.g., scaling) is handled by individual predictor implementations.
    """

    def __init__(
        self,
        config: dict[str, Any],  # Overall application config
        pubsub_manager: PubSubManager,
        process_pool_executor: ProcessPoolExecutor,
        logger_service: LoggerService,
        configuration_manager: Optional["ConfigurationManager"] = None,  # Added
    ) -> None:
        """
        Initialize the PredictionService.

        Args
        ----
            config (dict): Overall application configuration.
                           Expected to contain a 'prediction_service' key with a 'models' list.
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            process_pool_executor (ProcessPoolExecutor): Executor for running inference.
            logger_service (LoggerService): The shared logger instance.
        """
        self._service_config = config.get("prediction_service", {})
        self.pubsub = pubsub_manager
        self._process_pool_executor = process_pool_executor
        self.logger = logger_service  # LoggerService is already initialized
        self._is_running = False
        self._source_module = self.__class__.__name__
        self.configuration_manager = configuration_manager  # Store it

        self._feature_event_handler: Callable[[FeatureEvent], Coroutine[Any, Any, None]] = (
            self._handle_feature_event
        )
        # Added handler for config updates
        self._config_update_handler: Callable[
            [PredictionConfigUpdatedEvent], Coroutine[Any, Any, None]
        ] = self._handle_prediction_config_updated_event

        # Model configurations and initialized predictors/runners
        self._model_configs: list[dict[str, Any]] = self._service_config.get("models", [])
        self._predictors: dict[str, PredictorInterface] = {}
        self._predictor_runners: dict[str, Callable] = {}
        self._lstm_feature_buffers: dict[str, deque] = {}  # Added for LSTM sequence buffering

        if not self._model_configs:
            self.logger.warning(
                "PredictionService: No models configured under 'prediction_service.models'."
            )
            # Depending on strictness, could raise ValueError here

        self._initialize_predictors()

        # Track tasks submitted to the executor
        self._active_inference_tasks: set[InferenceTaskType] = set()

    def _get_predictor_map(self) -> dict:
        """Import and return a mapping of predictor types to their classes and runners.

        Returns
        -------
            dict: Mapping of predictor types to (predictor_class, runner_method) tuples.

        Raises
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
                exc_info=True,
            )
            raise PredictorImportError(PredictorImportError.message) from e
        else:
            return {
                "xgboost": (XGBoostPredictor, XGBoostPredictor.run_inference_in_process),
                "sklearn": (SKLearnPredictor, SKLearnPredictor.run_inference_in_process),
                "lstm": (LSTMPredictor, LSTMPredictor.run_inference_in_process),
            }

    def _validate_model_config(
        self, model_conf: dict
    ) -> tuple[str, str, str, type[PredictorInterface], Callable[..., Any], dict[str, Any], bool]:
        """Validate model configuration and extract necessary fields."""
        model_id: str = str(model_conf.get("model_id", ""))
        predictor_type: str = str(model_conf.get("predictor_type", ""))
        model_path: str = str(model_conf.get("model_path", ""))
        is_critical: bool = bool(model_conf.get("is_critical", False))

        if not all([model_id, predictor_type, model_path]):
            log_msg = ("Missing required model configuration fields. Required: 'model_id', "
                     "'predictor_type', 'model_path'")
            error_msg = f"{ModelConfigError.invalid_config}: {log_msg}"
            context = {"config": model_conf}
            self.logger.error(
                error_msg,
                source_module=self._source_module,
                context=context,
            )
            raise ModelConfigError(error_msg, context=context)

        predictor_map = self._get_predictor_map()
        predictor_info = predictor_map.get(predictor_type.lower())
        if not predictor_info:
            log_msg = f"Unsupported predictor type: {predictor_type}"
            error_context = {
                "predictor_type": predictor_type,
                "supported_types": list(predictor_map.keys())
            }
            error_msg = f"{ModelConfigError.unsupported_predictor}: {log_msg}"
            self.logger.error(
                error_msg,
                source_module=self._source_module,
                context=error_context
            )
            raise ModelConfigError(error_msg, context=context)

        predictor_class, runner_method = predictor_info
        return (model_id, model_path, predictor_type, predictor_class,
                runner_method, model_conf, is_critical)

    def _initialize_lstm_buffer(self, model_id: str, model_conf: dict[str, Any]) -> None:
        """Initialize LSTM feature buffer if needed."""
        sequence_length: int = int(model_conf.get("sequence_length", 1))
        if sequence_length < 1:
            self.logger.warning(
                "Invalid sequence_length for LSTM model %(model_id)s. Defaulting to 1.",
                source_module=self._source_module,
                context={"model_id": model_id, "sequence_length": sequence_length},
            )
            sequence_length = 1  # Treat as non-sequential if invalid

        if sequence_length > 1:  # Only create buffer if sequence_length > 1
            self._lstm_feature_buffers[model_id] = deque(maxlen=sequence_length)
            self.logger.info(
                "Initialized feature buffer for LSTM model %(model_id)s with sequence "
                "length %(seq_length)s.",
                source_module=self._source_module,
                context={"model_id": model_id, "seq_length": sequence_length}
            )

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
            (model_id, model_path, predictor_type, predictor_class,
             runner_method, model_conf, is_critical) = validation_result

            try:
                predictor_config = self._PredictorConfig(
                    model_id=model_id,
                    model_path=model_path,
                    predictor_type=predictor_type,
                    predictor_class=predictor_class,
                    runner_method=runner_method,
                    model_conf=model_conf,
                    is_critical=is_critical
                )
                self._initialize_single_predictor(predictor_config)
                if is_critical:
                    loaded_critical_models += 1

            except Exception as e:
                if is_critical:
                    self.logger.critical(
                        "Failed to initialize predictor for model_id %(model_id)s "
                        "(Critical: %(is_critical)s)",
                        source_module=self._source_module,
                        context={"model_id": model_id, "is_critical": is_critical},
                        exc_info=True,
                    )
                    raise PredictorInitError(PredictorInitError.message) from e
                self.logger.exception(
                    "Failed to initialize predictor for model_id %(model_id)s "
                    "(Critical: %(is_critical)s)",
                    source_module=self._source_module,
                    context={"model_id": model_id, "is_critical": is_critical}
                )

        self._validate_critical_models_loaded(loaded_critical_models, total_critical_models)
        self.logger.info(
            "Predictor initialization complete. %(predictor_count)s predictors loaded.",
            source_module=self._source_module,
            context={
                "predictors_loaded": len(self._predictors),
                "predictor_count": len(self._predictors)
            }
        )

    @dataclass
    class _PredictorConfig:
        """Configuration for initializing a predictor instance."""

        model_id: str
        model_path: str
        predictor_type: str
        predictor_class: type[PredictorInterface]
        runner_method: Callable
        model_conf: dict
        is_critical: bool

    def _initialize_single_predictor(
        self,
        config: _PredictorConfig,
    ) -> None:
        """Initialize a single predictor instance.

        Args:
            config: Configuration object containing all necessary parameters
                for initializing the predictor.
        """
        predictor_instance = config.predictor_class(
            model_path=config.model_path,
            model_id=config.model_id,
            config=config.model_conf,
        )
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
            },
        )

    def _validate_critical_models_loaded(
        self, loaded_critical_models: int, total_critical_models: int
    ) -> None:
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
                    "error": "Critical model initialization failed"
                }
            )
            raise CriticalModelError(CriticalModelError.load_failed)

    async def start(self) -> None:
        """Start listening for feature events and configuration updates.

        Predictor assets are loaded during __init__.
        """
        if self._is_running:
            self.logger.warning(
                "PredictionService already running.",
                source_module=self._source_module
            )
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
                source_module=self._source_module
            )
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
                    EventType.PREDICTION_CONFIG_UPDATED, self._config_update_handler
                )
                self.logger.info(
                    "Subscribed to %(event_type)s for dynamic model reloading.",
                    source_module=self._source_module,
                    context={"event_type": EventType.PREDICTION_CONFIG_UPDATED.name}
                )
            elif self.configuration_manager is None:
                self.logger.info(
                    "ConfigurationManager not provided; dynamic model reloading via events "
                    "will be disabled.",
                    source_module=self._source_module
                )

        except Exception:
            self.logger.exception(
                "Error during pubsub subscription in start()",
                source_module=self._source_module
            )
            self._is_running = False  # Don't consider service started if subscriptions fail
            raise  # Re-raise to signal failure to start properly

        loaded_count = len([p for p in self._predictors.values()
                      if getattr(p, "is_critical", False)])
        total_critical = sum(1 for p in self._predictors.values()
                         if getattr(p, "is_critical", False))
        self.logger.info(
            "Initialized %(loaded_count)s of %(total_critical)s critical models",
            source_module=self._source_module,
            context={
                "models_loaded": len(self._predictors),
                "loaded_count": loaded_count,
                "total_critical": total_critical
            }
        )
        for model_id, predictor in self._predictors.items():
            predictor_type = type(predictor).__name__
            expected_features = predictor.expected_feature_names
            context = {
                "model_id": str(model_id),
                "predictor_type": str(predictor_type),
                "expected_features": str(expected_features)
            }
            self.logger.info(
                "Initialized predictor for model_id: %(model_id)s, type: %(predictor_type)s",
                source_module=self._source_module,
                context=context
            )

    async def stop(self) -> None:
        """Stop event processing and cancel pending inferences."""
        if not self._is_running:
            self.logger.info(
                "PredictionService already stopped or not started.",
                source_module=self._source_module
            )
            return
        self._is_running = False

        # Unsubscribe from FeatureEvent first
        try:
            self.pubsub.unsubscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info(
                "Unsubscribed from %(event_type)s.",
                source_module=self._source_module,
                context={"event_type": EventType.FEATURES_CALCULATED.name}
            )
        except Exception:
            self.logger.exception(
                "Error unsubscribing from %(event_type)s",
                source_module=self._source_module,
                context={"event_type": EventType.FEATURES_CALCULATED.name}
            )

        # Unsubscribe from PREDICTION_CONFIG_UPDATED
        if (
            hasattr(EventType, "PREDICTION_CONFIG_UPDATED")
            and self.configuration_manager is not None
        ):
            try:
                self.pubsub.unsubscribe(
                    EventType.PREDICTION_CONFIG_UPDATED, self._config_update_handler
                )
                self.logger.info(
                    "Unsubscribed from %(event_type)s.",
                    source_module=self._source_module,
                    context={"event_type": EventType.PREDICTION_CONFIG_UPDATED.name}
                )
            except Exception:
                self.logger.exception(
                    "Error unsubscribing from %(event_type)s",
                    source_module=self._source_module,
                    context={"event_type": EventType.PREDICTION_CONFIG_UPDATED.name}
                )

        if self._active_inference_tasks:
            pending_count = len(self._active_inference_tasks)
            log_msg = ("Stopping PredictionService. Cancelling %(pending_count)s "
                      "pending inferences...")
            self.logger.info(
                log_msg,
                source_module=self._source_module,
                context={"pending_inferences": pending_count, "pending_count": pending_count}
            )
            for task in list(self._active_inference_tasks):  # Iterate over a copy
                if not task.done():
                    task.cancel()
            # Wait for cancellations
            await asyncio.gather(*self._active_inference_tasks, return_exceptions=True)
            self._active_inference_tasks.clear()
            self.logger.info(
                "Pending inferences cancelled.",
                source_module=self._source_module
            )
        self.logger.info(
            "PredictionService stopped.",
            source_module=self._source_module
        )

    async def _handle_feature_event(self, event: FeatureEvent) -> None:
        """
        Handle incoming feature event, update LSTM buffers, and trigger pipeline.

        Updates LSTM buffers if needed and triggers the prediction pipeline for any
        models that are ready for prediction.
        """
        if not isinstance(event, FeatureEvent):
            event_type = type(event).__name__
            log_msg = f"Received non-FeatureEvent: {event_type}"
            self.logger.warning(
                log_msg,
                source_module=self._source_module,
                context={"event_type": event_type}
            )
            return

        if self._process_pool_executor is None:
            self.logger.error(
                "ProcessPoolExecutor not available. Cannot run predictions.",
                source_module=self._source_module,
                context={"executor_status": "unavailable"}
            )
            return

        if not self._is_running:
            self.logger.warning(
                "PredictionService is not running. Ignoring feature event.",
                source_module=self._source_module,
                context={"service_status": "not_running"}
            )
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
                    context={"model_id": model_id}
                )
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
                        "maxlen": buffer.maxlen
                    }
                )
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
                        "event_id": event.event_id
                    }
                )

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
                (
                    mc
                    for mc in self._model_configs
                    if mc.get("model_id") == model_id
                ),
                None,
            )
            if not model_config:
                continue

            predictor_type = model_config.get("predictor_type")
            feature_input_for_model: np.ndarray | None = None

            if predictor_type == "lstm" and model_id in self._lstm_feature_buffers:
                buffer = self._lstm_feature_buffers[model_id]
                if len(buffer) == buffer.maxlen:  # Buffer is full, ready for sequence prediction
                    feature_input_for_model = np.array(
                        list(buffer)
                    )  # Shape: (sequence_length, n_features_per_timestep)
                    log_msg = f"LSTM model {model_id} buffer full. Preparing sequence."
                    self.logger.debug(
                        log_msg,
                        source_module=self._source_module,
                        context={"model_id": model_id}
                    )
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
                            "max_size": buffer.maxlen
                        }
                    )
                    continue  # Not ready yet
            else:  # Non-LSTM model or LSTM with sequence_length=1 (handled by _prepare_features)
                expected_features = predictor_instance.expected_feature_names
                if not expected_features:
                    log_msg = f"Model {model_id} has no expected_features. Cannot prepare input."
                    self.logger.warning(
                        log_msg,
                        source_module=self._source_module,
                        context={"model_id": model_id}
                    )
                    continue
                feature_input_for_model = self._prepare_features_for_model(
                    event.features, expected_features
                )

            if feature_input_for_model is not None:
                ready_models_info.append(
                    {
                        "model_id": model_id,
                        "model_config": model_config,  # Pass full config for the runner
                        "feature_input": feature_input_for_model,
                        "runner_func": self._predictor_runners[model_id],
                    }
                )
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
                        "event_id": event.event_id
                    }
                )

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
                    "event_id": event.event_id
                }
            )
            # Store task reference to prevent it from being garbage collected
            task = asyncio.create_task(
                self._run_multi_model_prediction_pipeline(event, ready_models_info)
            )
            # Add a callback to log any exceptions that might occur in the task
            task.add_done_callback(
                lambda t: t.exception() or True  # Logs the exception if one occurred
            )
        else:
            log_msg = f"No models ready for prediction for event {event.event_id}."
            self.logger.debug(
                log_msg,
                source_module=self._source_module,
                context={"event_id": event.event_id}
            )

    async def _run_multi_model_prediction_pipeline(
        self, event: FeatureEvent, ready_models_data: list[dict[str, Any]]
    ) -> None:
        """
        Orchestrate inference for models that are ready and publish prediction events.

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
            event, inference_results["model_ids"], inference_results["results"]
        )

        # Apply ensembling if needed and publish results
        await self._apply_ensembling_and_publish(event, successful_predictions)

    async def _submit_inference_tasks(
        self, event: FeatureEvent, ready_models_data: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Submit inference tasks for all ready models."""
        loop = asyncio.get_running_loop()
        inference_futures_dict: dict[str, asyncio.Future] = {}

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
                    "event_id": str(event.event_id)
                }
            )

            future: asyncio.Future = loop.run_in_executor(
                self._process_pool_executor,
                runner_func,
                model_id,
                model_config["model_path"],
                model_config.get("scaler_path"),
                feature_input,
                expected_features,
                model_config,
            )
            self._active_inference_tasks.add(future)
            future.add_done_callback(lambda f: self._active_inference_tasks.discard(f))
            inference_futures_dict[model_id] = future

        if not inference_futures_dict:
            log_msg = f"No inference tasks were submitted for event {event.event_id}."
            self.logger.info(
                log_msg,
                source_module=self._source_module,
                context={"event_id": str(event.event_id)}
            )
            return None

        futures_to_await = list(inference_futures_dict.values())
        log_msg = f"Awaiting {len(futures_to_await)} inference results for event {event.event_id}."
        self.logger.debug(
            log_msg,
            source_module=self._source_module,
            context={
                "event_id": str(event.event_id),
                "num_futures": len(futures_to_await),
            },
        )

        results = await asyncio.gather(*futures_to_await, return_exceptions=True)
        log_msg = (
            f"Received {len(results)} inference results/exceptions for event "
            f"{event.event_id}."
        )
        self.logger.debug(
            log_msg,
            source_module=self._source_module,
            context={
                "event_id": str(event.event_id),
                "result_count": len(results),
            },
        )

        return {
            "model_ids": list(inference_futures_dict.keys()),
            "results": results
        }

    def _process_inference_results(
        self, event: FeatureEvent, model_ids: list[str], results: list[Any]
    ) -> list[dict[str, Any]]:
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
                    exc_info=result_or_exc
                )
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
                    },
                )

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
                        "error": str(result["error"])
                    },
                )
            elif "prediction" in result:
                model_config = next(
                    (mc for mc in self._model_configs if mc.get("model_id") == model_id),
                    None
                )
                if model_config:
                    successful_predictions.append({
                        "model_id": model_id,
                        "prediction_value": result["prediction"],
                        "prediction_target": model_config.get(
                            "prediction_target",
                            "unknown_target"
                        ),
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
                        context={"model_id": model_id}
                    )
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
                        "result": str(result)
                    },
                )

        return successful_predictions

    async def _apply_ensembling_and_publish(
        self, event: FeatureEvent, successful_predictions: list[dict[str, Any]]
    ) -> None:
        """Apply ensembling strategy and publish final predictions."""
        ensemble_strategy = self._service_config.get("ensemble_strategy", "none").lower()
        ensemble_weights = self._service_config.get("ensemble_weights", {})
        # TODO: Implement confidence floor in future versions
        _ = float(self._service_config.get("confidence_floor", 0.1))  # Will be used later

        # Group predictions by target for ensembling
        predictions_by_target: dict[str, list[dict[str, Any]]] = {}
        for pred_data in successful_predictions:
            target = pred_data["prediction_target"]
            if target not in predictions_by_target:
                predictions_by_target[target] = []
            predictions_by_target[target].append(pred_data)

        final_predictions_to_publish: list[dict[str, Any]] = []

        if ensemble_strategy == "none" or not predictions_by_target:
            # Publish individual predictions
            final_predictions_to_publish = successful_predictions
        else:
            # Apply ensembling for each target group
            for target, preds_for_target in predictions_by_target.items():
                if not preds_for_target:
                    continue

                target_cleaned = target.replace("_", "")[:15]
                ensemble_id = f"ensemble_{target_cleaned}"

                if ensemble_strategy == "average":
                    self._apply_average_ensembling(
                        preds_for_target, target, ensemble_id, final_predictions_to_publish
                    )
                elif ensemble_strategy == "weighted_average":
                    self._apply_weighted_average_ensembling(
                        preds_for_target, target, ensemble_id, ensemble_weights,
                        final_predictions_to_publish
                    )
                # Add other ensembling strategies here as needed

        # Publish all final predictions
        for pred in final_predictions_to_publish:
            prediction_event = PredictionEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                trading_pair=event.trading_pair,
                exchange=event.exchange,
                timestamp_prediction_for=datetime.utcnow(),  # Adjust as needed
                model_id=pred["model_id"],
                prediction_target=pred["prediction_target"],
                prediction_value=float(pred["prediction_value"]),
                confidence=(
                    float(pred.get("confidence", 0.0))
                    if pred.get("confidence") is not None
                    else None
                ),
                associated_features={
                    "source_event_id": str(event.event_id),
                    "ensemble_strategy": ensemble_strategy,
                },
            )
            # Await the async _publish_prediction method
            await self._publish_prediction(prediction_event)

    def _apply_average_ensembling(
        self,
        predictions: list[dict[str, Any]],
        target: str,
        ensemble_id: str,
        output: list[dict[str, Any]]
    ) -> None:
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

        output.append({
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
        output: list[dict[str, Any]]
    ) -> None:
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
                f"No valid weights for ensembling target {target}. "
                "Skipping weighted average."
            )
            self.logger.warning(
                log_msg,
                source_module=self._source_module,
                context=context
            )
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
                weighted_confidence_sum / total_weight
                if weighted_confidence_sum > 0
                else None
            )

            output.append({
                "model_id": ensemble_id,
                "prediction_value": avg_value,
                "prediction_target": target,
                "confidence": avg_confidence,
                "config": {"model_id": ensemble_id, "prediction_target": target},
            })

    def _prepare_features_for_model(
        self, event_features: dict[str, str], expected_model_features: list[str]
    ) -> np.ndarray | None:
        """Convert feature dictionary to a 1D numpy array."""
        ordered_feature_values: list[float] = []
        missing_features_log: list[str] = []
        type_errors_log: list[str] = []

        for feature_name in expected_model_features:
            if feature_name not in event_features:
                missing_features_log.append(str(feature_name))
                ordered_feature_values.append(float("nan"))
                continue

            try:
                # Explicitly convert to Python float first to catch any conversion issues
                value = float(str(event_features[feature_name]))
                ordered_feature_values.append(value)
            except (ValueError, TypeError) as e:
                type_errors_log.append(f"{feature_name}: {e!s}")
                ordered_feature_values.append(float("nan"))

        # Log any issues
        if missing_features_log:
            context = {"missing_features": missing_features_log}
            log_msg = "Missing features: {}".format(", ".join(missing_features_log))
            self.logger.warning(log_msg, source_module=self._source_module, context=context)
        if type_errors_log:
            context = {
                "type_errors": type_errors_log,
                "expected_features": expected_model_features
            }
            log_msg = "Type conversion errors: {}".format(", ".join(type_errors_log))
            self.logger.warning(log_msg, source_module=self._source_module, context=context)

        if not ordered_feature_values:  # Should not happen if expected_model_features is not empty
            self.logger.error(
                "No feature values were processed. This is unexpected.",
                source_module=self._source_module
            )
            return None

        feature_array = np.array(ordered_feature_values, dtype=np.float32)

        # Check if all values ended up being NaN
        if np.all(np.isnan(feature_array)):
            self.logger.error(
                "All features resulted in NaN for model. Cannot make prediction.",
                source_module=self._source_module,
                context={"expected_features": expected_model_features}
            )
            return None

        return feature_array  # Returns a 1D array

    async def _publish_prediction(self, event: PredictionEvent) -> None:
        """Publish the prediction event to subscribers."""
        try:
            await self.pubsub.publish(event)
            context = {
                "event_id": str(event.event_id),
                "model_id": str(event.model_id),
                "trading_pair": event.trading_pair,
                "timestamp": str(event.timestamp_prediction_for),
                "prediction_value": event.prediction_value
            }
            log_msg = f"Published prediction event {event.event_id} for model {event.model_id}"
            self.logger.debug(log_msg, source_module=self._source_module, context=context)
        except Exception as e:
            context = {
                "event_id": str(event.event_id),
                "model_id": str(event.model_id),
                "error": str(e)
            }
            log_msg = f"Failed to publish prediction event {event.event_id}"
            self.logger.exception(log_msg, source_module=self._source_module, context=context)

    # --- Dynamic Model Reloading ---
    async def _handle_prediction_config_updated_event(
        self, event: PredictionConfigUpdatedEvent
    ) -> None:
        """Handle prediction service configuration update event.

        Re-initializes predictors based on the new configuration from the event payload.
        """
        if not isinstance(event, PredictionConfigUpdatedEvent):
            self.logger.warning(
                "Received incorrect event type for config update: %s. "
                "Expected PredictionConfigUpdatedEvent.",
                type(event)
            )
            return

        # Get new config from well-typed event
        new_service_config = event.new_prediction_service_config
        if not isinstance(new_service_config, dict):
            self.logger.error(
                "Invalid payload in PredictionConfigUpdatedEvent: "
                "new_prediction_service_config is not a dict. Type: %s",
                type(new_service_config)
            )
            return

        self.logger.info(
            "Prediction_service configuration update event received. Re-initializing predictors."
        )

        # Store the new service-specific config part
        self._service_config = new_service_config
        self._model_configs = self._service_config.get("models", [])

        if not self._model_configs:
            self.logger.warning(
                "No models found in new configuration during re-initialization. "
                "Service may become inactive if all models are removed."
            )

        # TODO: Consider more graceful handling of in-flight tasks
        # for models being removed/changed.
        # For now, active tasks for old/removed models will complete
        # or be cancelled on service stop/timeout. New FeatureEvents
        # arriving *during* _initialize_predictors might be slightly
        # delayed or use new predictors.

        try:
            # This will clear and repopulate predictors & buffers
            # It also raises RuntimeError if critical models fail.
            self._initialize_predictors()
            self.logger.info(
                "Predictors re-initialized successfully based on new configuration.",
                source_module=self._source_module
            )
            # Log new state
            log_msg = f"PredictionService reconfigured. {len(self._predictors)} models now active."
            self.logger.info(
                log_msg,
                source_module=self._source_module,
                context={"active_models": len(self._predictors)}
            )
            for model_id, predictor in self._predictors.items():
                log_msg = f"Active Model ID: {model_id}, Type: {type(predictor).__name__}"
                self.logger.info(
                    log_msg,
                    source_module=self._source_module,
                    context={
                        "model_id": model_id,
                        "predictor_type": type(predictor).__name__,
                        "expected_features": predictor.expected_feature_names
                    }
                )
        except RuntimeError as e_crit:
            error_msg_template = (
                "CRITICAL: Failed to re-initialize predictors with new config due to: {}. "
                "Some critical models may have failed to load. Service might be unstable or stop."
            )
            log_msg = error_msg_template.format(e_crit)
            self.logger.critical(
                log_msg,
                source_module=self._source_module,
                context={"error": str(e_crit)}
            )
            # If this happens, the service is likely in a bad state regarding its predictors.
            # Depending on overall application design, might need to signal a broader system issue.
        except Exception as e:
            log_msg = f"Fatal error in prediction service: {e}"
            self.logger.critical(
                log_msg,
                source_module=self._source_module,
                context={"error": str(e)},
                exc_info=True
            )
