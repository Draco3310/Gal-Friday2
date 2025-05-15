# Prediction Service Module
"""
Machine learning prediction service for market data analysis.

This module provides infrastructure for running ML model inference against
market features, handling the prediction lifecycle from feature consumption
to prediction publishing.
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, Set
import uuid

import numpy as np
# Removed: import xgboost as xgb - specific models handled by predictors

# Import necessary components
from gal_friday.core.events import (
    EventType,
    FeatureEvent,
    PredictionEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.interfaces.predictor_interface import PredictorInterface # Added

if TYPE_CHECKING:
    from collections.abc import Coroutine


# Set up logging for this module specifically
# log = logging.getLogger(__name__) # Keep this if used elsewhere, or use self.logger
# For PredictionService specific logging, self.logger is preferred.

# Define a type for futures to track
T = TypeVar("T")
InferenceTaskType = asyncio.Future[Dict[str, Any]] # Return type from runners


# Removed old _run_inference_task, as this logic is now in predictor classes


# --- PredictionService Class ---
class PredictionService:
    """
    Consume feature events, run ML model inference for multiple configured models,
    and publish prediction events. Preprocessing (e.g., scaling) is handled by
    individual predictor implementations.
    """

    def __init__(
        self,
        config: Dict[str, Any], # Overall application config
        pubsub_manager: PubSubManager,
        process_pool_executor: ProcessPoolExecutor,
        logger_service: LoggerService,
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
        self.logger = logger_service.get_logger(self.__class__.__name__) # Module-specific logger
        self._is_running = False
        self._source_module = self.__class__.__name__

        self._feature_event_handler: Callable[[FeatureEvent], Coroutine[Any, Any, None]] = (
            self._handle_feature_event
        )

        # Model configurations and initialized predictors/runners
        self._model_configs: List[Dict[str, Any]] = self._service_config.get("models", [])
        self._predictors: Dict[str, PredictorInterface] = {}
        self._predictor_runners: Dict[str, Callable] = {}

        if not self._model_configs:
            self.logger.warning("PredictionService: No models configured under 'prediction_service.models'.")
            # Depending on strictness, could raise ValueError here

        self._initialize_predictors()

        # Track tasks submitted to the executor
        self._active_inference_tasks: Set[InferenceTaskType] = set()

    def _initialize_predictors(self) -> None:
        """Initializes predictor instances and their runners from configuration."""
        # Explicitly import concrete predictor classes and their runners here
        # This could be made more dynamic (e.g., plugin system) if many predictors are expected
        try:
            from gal_friday.predictors.xgboost_predictor import (
                XGBoostPredictor,
                # run_inference_in_process as xgboost_runner, # Assuming direct import
            )
            from gal_friday.predictors.sklearn_predictor import (
                SKLearnPredictor,
                # run_inference_in_process as sklearn_runner, # Assuming direct import
            )
        except ImportError as e:
            self.logger.error(f"Failed to import predictor classes: {e}", exc_info=True)
            raise ValueError("Predictor class import failed. Check predictor modules.") from e


        predictor_map = {
            "xgboost": (XGBoostPredictor, XGBoostPredictor.run_inference_in_process),
            "sklearn": (SKLearnPredictor, SKLearnPredictor.run_inference_in_process),
            # Add other predictor types here
        }

        for model_conf in self._model_configs:
            model_id = model_conf.get("model_id")
            predictor_type = model_conf.get("predictor_type")
            model_path = model_conf.get("model_path")

            if not all([model_id, predictor_type, model_path]):
                self.logger.error(
                    f"Skipping model config due to missing model_id, predictor_type, or model_path: {model_conf}"
                )
                continue

            PredictorClass, predictor_runner_static_method = predictor_map.get(predictor_type, (None, None))

            if not PredictorClass or not predictor_runner_static_method:
                self.logger.error(
                    f"Unknown predictor_type: {predictor_type} for model_id: {model_id}. Skipping."
                )
                continue

            try:
                # Pass the specific model_conf dict to the predictor's __init__
                # The predictor's __init__ should use this config to find its specific paths etc.
                predictor_instance = PredictorClass(
                    model_path=model_path,
                    model_id=model_id,
                    config=model_conf # Pass the full model-specific config dictionary
                )
                self._predictors[model_id] = predictor_instance
                self._predictor_runners[model_id] = predictor_runner_static_method
                self.logger.info(
                    f"Initialized predictor for model_id: {model_id} of type {predictor_type} "
                    f"using model: {model_path} and scaler: {model_conf.get('scaler_path')}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize predictor for model_id {model_id}: {e}", exc_info=True
                )
                # Decide whether to raise an error and stop service, or just skip this model
                # For now, skipping the faulty model.

    async def start(self) -> None:
        """
        Start listening for feature events. Predictor assets are loaded during __init__.
        """
        if self._is_running:
            self.logger.warning("PredictionService already running.")
            return
        
        if not self._predictors: # No valid predictors were initialized
            self.logger.error("PredictionService cannot start: No valid predictors initialized. Check configuration.")
            # Optionally raise an error to prevent service from running in a useless state
            return

        self._is_running = True
        self.pubsub.subscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
        self.logger.info(
             f"PredictionService started. {len(self._predictors)} models configured and initialized."
        )
        for model_id, predictor in self._predictors.items():
            self.logger.info(
                f"  Model ID: {model_id}, Type: {type(predictor).__name__}, "
                f"Expected features: {predictor.expected_feature_names}"
            )


    async def stop(self) -> None:
        """Stop event processing and cancel pending inferences."""
        if not self._is_running:
            self.logger.info("PredictionService already stopped or not started.")
            return # Ensure it's truly not running or never started successfully
        self._is_running = False

        try:
            self.pubsub.unsubscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info("Unsubscribed from FEATURES_CALCULATED.")
        except Exception as e:
            self.logger.error(f"Error unsubscribing: {e}", exc_info=True)

        if self._active_inference_tasks:
            self.logger.info(f"Cancelling {len(self._active_inference_tasks)} pending inference tasks...")
            for task in list(self._active_inference_tasks): # Iterate over a copy
                if not task.done():
                    task.cancel()
            # Wait for cancellations
            await asyncio.gather(*self._active_inference_tasks, return_exceptions=True)
            self._active_inference_tasks.clear()
            self.logger.info("Pending inference tasks cancelled.")
        self.logger.info("PredictionService stopped.")

    async def _handle_feature_event(self, event: FeatureEvent) -> None:
        """Handle incoming feature events by scheduling the multi-model prediction pipeline."""
        if not isinstance(event, FeatureEvent):
            self.logger.warning(f"Received non-FeatureEvent: {type(event)}.")
            return

        if self._process_pool_executor is None:
            self.logger.error("ProcessPoolExecutor not available. Cannot run predictions.")
            return

        if not self._is_running:
            self.logger.warning("PredictionService is not running. Ignoring feature event.")
            return

        asyncio.create_task(self._run_multi_model_prediction_pipeline(event))

    async def _run_multi_model_prediction_pipeline(self, event: FeatureEvent) -> None:
        """
        Orchestrate feature preparation, inference, and publishing for all configured models.
        """
        loop = asyncio.get_running_loop()
        inference_futures_dict: Dict[str, InferenceTaskType] = {}

        for model_id, predictor_instance in self._predictors.items():
            model_config = next((mc for mc in self._model_configs if mc.get("model_id") == model_id), None)
            if not model_config: # Should not happen if _initialize_predictors worked
                self.logger.error(f"Config not found for model_id {model_id} during pipeline. Skipping.")
                continue

            expected_features = predictor_instance.expected_feature_names
            if not expected_features:
                self.logger.warning(
                    f"Model {model_id} has no expected_feature_names defined. Skipping inference."
                )
                continue

            feature_vector_1d = self._prepare_features_for_model(
                event.features, expected_features
            )

            if feature_vector_1d is None:
                self.logger.warning(
                    f"Feature preparation failed for model {model_id} for event {event.event_id}. Skipping."
                )
                continue # Error logged in _prepare_features_for_model

            runner_func = self._predictor_runners.get(model_id)
            if not runner_func: # Should not happen
                self.logger.error(f"Runner function not found for model {model_id}. Skipping.")
                continue

            self.logger.debug(f"Submitting inference for model {model_id}, event {event.event_id}.")
            future: InferenceTaskType = loop.run_in_executor(
                self._process_pool_executor,
                runner_func,  # The static method from the predictor class
                model_id,
                model_config['model_path'],
                model_config.get('scaler_path'), # Optional
                feature_vector_1d,
                expected_features, # model_feature_names for the DMatrix in XGBoost, etc.
                model_config # Pass the full model-specific config dict
            )
            self._active_inference_tasks.add(future)
            # Assign model_id to future for easier tracking if needed, though result dict has it
            # setattr(future, 'model_id', model_id) # Optional helper for debugging
            future.add_done_callback(lambda f: self._active_inference_tasks.discard(f))
            inference_futures_dict[model_id] = future
        
        if not inference_futures_dict:
            self.logger.info(f"No inference tasks were submitted for event {event.event_id}.")
            return

        # Await all scheduled inference tasks for this event
        model_ids_submitted = list(inference_futures_dict.keys())
        futures_to_await = [inference_futures_dict[mid] for mid in model_ids_submitted]
        
        self.logger.debug(f"Awaiting {len(futures_to_await)} inference results for event {event.event_id}.")
        # results_list will contain the return values of the runner functions or exceptions
        results_list = await asyncio.gather(*futures_to_await, return_exceptions=True)
        self.logger.debug(f"Received {len(results_list)} inference results/exceptions for event {event.event_id}.")

        # --- Process results and potentially ensemble ---
        # For now, publish individual predictions. Ensembling is a future step for FR-307.
        for i, result_or_exc in enumerate(results_list):
            model_id_for_result = model_ids_submitted[i] # Get model_id based on order

            if isinstance(result_or_exc, Exception):
                self.logger.error(
                    f"Inference task for model {model_id_for_result} raised an exception: {result_or_exc}",
                    exc_info=result_or_exc # Include stack trace if it's an exception object
                )
                continue # Skip this result

            # result_or_exc is a Dict[str, Any] here
            result: Dict[str, Any] = result_or_exc
            
            # The result dict should contain 'model_id' from the runner for confirmation
            if result.get("model_id") != model_id_for_result:
                self.logger.warning(
                    f"Mismatched model_id in result for event {event.event_id}. "
                    f"Expected {model_id_for_result}, got {result.get('model_id')}. Using expected."
                )
            
            current_model_config = next(
                (mc for mc in self._model_configs if mc.get("model_id") == model_id_for_result), {}
            )

            if "error" in result:
                self.logger.error(
                    f"Inference failed for model {model_id_for_result} (event {event.event_id}): {result['error']}"
                )
            elif "prediction" in result:
                prediction_value = result["prediction"]
                prediction_target = current_model_config.get("prediction_target", "unknown_target")
                confidence = result.get("confidence") # If predictors start returning it

                prediction_event = PredictionEvent(
                    source_module=self._source_module,
                    event_id=uuid.uuid4(), # New UUID for each prediction event
                    timestamp=datetime.utcnow(),
                    trading_pair=event.trading_pair,
                    exchange=event.exchange,
                    timestamp_prediction_for=event.timestamp_features_for,
                    model_id=model_id_for_result, # Specific model that made this prediction
                    prediction_target=prediction_target,
                    prediction_value=prediction_value,
                    confidence=confidence,
                    associated_features=event.features,
                )
                try:
                    await self._publish_prediction(prediction_event)
                except Exception as e_pub:
                    self.logger.error(
                        f"Failed to publish prediction for model {model_id_for_result}: {e_pub}",
                        exc_info=True
                    )
            else:
                self.logger.error(
                    f"Invalid inference result format for model {model_id_for_result} (event {event.event_id}): {result}"
                )


    def _prepare_features_for_model(
        self, event_features: Dict[str, str], expected_model_features: List[str]
    ) -> Optional[np.ndarray]:
        """
        Convert feature dictionary to a 1D numpy array of floats for model input,
        ordered according to expected_model_features.
        Handles missing features and type conversion errors.
        """
        ordered_feature_values: List[float] = []
        missing_features_log: List[str] = []
        type_errors_log: List[str] = []

        for feature_name in expected_model_features:
            if feature_name not in event_features:
                missing_features_log.append(feature_name)
                ordered_feature_values.append(np.nan)
                continue

            value_str = event_features[feature_name]
            try:
                value_float = float(value_str)
                ordered_feature_values.append(value_float)
            except (ValueError, TypeError):
                type_errors_log.append(f"{feature_name}='{value_str}'")
                ordered_feature_values.append(np.nan)

        if missing_features_log:
            self.logger.warning(
                f"Missing features for model (using {expected_model_features}): "
                f"{missing_features_log}. Filled with NaN. Input features: {list(event_features.keys())}"
            )
        if type_errors_log:
            self.logger.error(
                f"Type conversion errors for features (using {expected_model_features}): "
                f"{type_errors_log}. Filled with NaN."
            )

        if not ordered_feature_values: # Should not happen if expected_model_features is not empty
             self.logger.error("No feature values were processed. This is unexpected.")
             return None

        feature_array = np.array(ordered_feature_values, dtype=np.float32)
        
        # Check if all values ended up being NaN
        if np.all(np.isnan(feature_array)):
            self.logger.error(
                f"All features resulted in NaN for model (using {expected_model_features}). "
                "Cannot make prediction."
            )
            return None

        return feature_array # Returns a 1D array

    async def _publish_prediction(self, event: PredictionEvent) -> None:
        """Publish the prediction event to subscribers."""
        try:
            await self.pubsub.publish(event)
            self.logger.debug(
                f"Published PredictionEvent (ID: {event.event_id}) for model {event.model_id} "
                f"for {event.trading_pair} at {event.timestamp_prediction_for} "
                f"with value: {event.prediction_value:.4f}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to publish PredictionEvent (ID: {event.event_id}): {e}", exc_info=True
            )

# Example Usage Placeholder:
# Configuration would be more complex, loaded from YAML/JSON.
# Example of what `self._service_config` might look like after loading:
# {
#     "models": [
#         {
#             "model_id": "xgb_v1",
#             "predictor_type": "xgboost",
#             "model_path": "path/to/model.xgb",
#             "scaler_path": "path/to/scaler.pkl",
#             "model_feature_names": ["feat1", "feat2"],
#             "prediction_target": "price_up_prob"
#         },
#         {
#             "model_id": "sklearn_rf_v1",
#             "predictor_type": "sklearn",
#             "model_path": "path/to/model.joblib",
#             "scaler_path": "path/to/scaler.joblib",
#             "model_feature_names": ["feat1", "feat3", "feat4"],
#             "prediction_target": "direction"
#         }
#     ],
#     "ensemble_strategy": "none" # Or "average", "weighted_average" etc.
# }
