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
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, TypeVar
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
from gal_friday.interfaces.predictor_interface import PredictorInterface  # Added
from gal_friday.logger_service import LoggerService

if TYPE_CHECKING:
    from collections.abc import Coroutine


# Set up logging for this module specifically
# log = logging.getLogger(__name__) # Keep this if used elsewhere, or use self.logger
# For PredictionService specific logging, self.logger is preferred.

# Define a type for futures to track
T = TypeVar("T")
InferenceTaskType = asyncio.Future[Dict[str, Any]]  # Return type from runners


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
        config: Dict[str, Any],  # Overall application config
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
        self.logger = logger_service.get_logger(self.__class__.__name__)  # Module-specific logger
        self._is_running = False
        self._source_module = self.__class__.__name__

        self._feature_event_handler: Callable[[FeatureEvent], Coroutine[Any, Any, None]] = (
            self._handle_feature_event
        )

        # Model configurations and initialized predictors/runners
        self._model_configs: List[Dict[str, Any]] = self._service_config.get("models", [])
        self._predictors: Dict[str, PredictorInterface] = {}
        self._predictor_runners: Dict[str, Callable] = {}
        self._lstm_feature_buffers: Dict[str, deque] = {}  # Added for LSTM sequence buffering

        if not self._model_configs:
            self.logger.warning(
                "PredictionService: No models configured under 'prediction_service.models'."
            )
            # Depending on strictness, could raise ValueError here

        self._initialize_predictors()

        # Track tasks submitted to the executor
        self._active_inference_tasks: Set[InferenceTaskType] = set()

    def _initialize_predictors(self) -> None:
        """Initializes predictor instances, runners, and LSTM buffers from configuration."""
        try:
            from gal_friday.predictors.lstm_predictor import LSTMPredictor
            from gal_friday.predictors.sklearn_predictor import SKLearnPredictor
            from gal_friday.predictors.xgboost_predictor import XGBoostPredictor
        except ImportError as e:
            self.logger.critical(
                f"CRITICAL: Failed to import predictor classes: {e}. PredictionService cannot function.",
                exc_info=True,
            )
            raise ValueError("Predictor class import failed. Check predictor modules.") from e

        predictor_map = {
            "xgboost": (XGBoostPredictor, XGBoostPredictor.run_inference_in_process),
            "sklearn": (SKLearnPredictor, SKLearnPredictor.run_inference_in_process),
            "lstm": (LSTMPredictor, LSTMPredictor.run_inference_in_process),
        }

        # Clear existing predictors and buffers before re-initializing (for dynamic reload)
        self._predictors.clear()
        self._predictor_runners.clear()
        self._lstm_feature_buffers.clear()  # Clear LSTM buffers

        loaded_critical_models = 0
        total_critical_models = sum(1 for mc in self._model_configs if mc.get("critical"))

        for model_conf in self._model_configs:
            model_id = model_conf.get("model_id")
            predictor_type = model_conf.get("predictor_type")
            model_path = model_conf.get("model_path")
            is_critical = model_conf.get("critical", False)

            if not all([model_id, predictor_type, model_path]):
                log_msg = f"Skipping model config due to missing model_id, predictor_type, or model_path: {model_conf}"
                if is_critical:
                    self.logger.critical(f"CRITICAL: {log_msg}")
                    raise ValueError(f"Critical model {model_id or 'Unknown'} misconfigured.")
                self.logger.error(log_msg)
                continue

            PredictorClass, predictor_runner_static_method = predictor_map.get(
                predictor_type, (None, None)
            )

            if not PredictorClass or not predictor_runner_static_method:
                log_msg = f"Unknown predictor_type: {predictor_type} for model_id: {model_id}."
                if is_critical:
                    self.logger.critical(f"CRITICAL: {log_msg}")
                    raise ValueError(
                        f"Critical model {model_id} has unknown predictor_type: {predictor_type}."
                    )
                self.logger.error(f"{log_msg} Skipping.")
                continue

            try:
                predictor_instance = PredictorClass(
                    model_path=model_path, model_id=model_id, config=model_conf
                )
                self._predictors[model_id] = predictor_instance
                self._predictor_runners[model_id] = predictor_runner_static_method

                # Initialize buffer for LSTM models requiring sequences
                if predictor_type == "lstm":
                    sequence_length = model_conf.get("sequence_length", 1)
                    if not isinstance(sequence_length, int) or sequence_length < 1:
                        self.logger.warning(
                            f"Invalid sequence_length for LSTM model {model_id}. Defaulting to 1."
                        )
                        sequence_length = 1  # Treat as non-sequential if invalid

                    if sequence_length > 1:
                        self._lstm_feature_buffers[model_id] = deque(maxlen=sequence_length)
                        self.logger.info(
                            f"Initialized feature buffer for LSTM model {model_id} with sequence length {sequence_length}."
                        )

                self.logger.info(
                    f"Successfully initialized predictor for model_id: {model_id} (Critical: {is_critical}) "
                    f"of type {predictor_type} using model: {model_path} and scaler: {model_conf.get('scaler_path')}"
                )
                if is_critical:
                    loaded_critical_models += 1
            except Exception as e:
                log_msg = f"Failed to initialize predictor for model_id {model_id} (Critical: {is_critical}): {e}"
                if is_critical:
                    self.logger.critical(f"CRITICAL: {log_msg}", exc_info=True)
                    raise ValueError(f"Critical model {model_id} failed to initialize.") from e
                self.logger.error(log_msg, exc_info=True)

        if total_critical_models > 0 and loaded_critical_models < total_critical_models:
            self.logger.critical(
                f"CRITICAL: Not all critical models were loaded. Loaded {loaded_critical_models}/{total_critical_models}. "
                "PredictionService will not function correctly."
            )
            # This state should prevent the service from starting or functioning if start() checks this.
            raise RuntimeError("Failed to load one or more critical prediction models.")

        self.logger.info(
            f"Predictor initialization complete. {len(self._predictors)} predictors loaded."
        )

    async def start(self) -> None:
        """
        Start listening for feature events. Predictor assets are loaded during __init__.
        """
        if self._is_running:
            self.logger.warning("PredictionService already running.")
            return

        if not self._predictors:  # No valid predictors were initialized
            self.logger.error(
                "PredictionService cannot start: No valid predictors initialized. Check configuration."
            )
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
            return  # Ensure it's truly not running or never started successfully
        self._is_running = False

        try:
            self.pubsub.unsubscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info("Unsubscribed from FEATURES_CALCULATED.")
        except Exception as e:
            self.logger.error(f"Error unsubscribing: {e}", exc_info=True)

        if self._active_inference_tasks:
            self.logger.info(
                f"Cancelling {len(self._active_inference_tasks)} pending inference tasks..."
            )
            for task in list(self._active_inference_tasks):  # Iterate over a copy
                if not task.done():
                    task.cancel()
            # Wait for cancellations
            await asyncio.gather(*self._active_inference_tasks, return_exceptions=True)
            self._active_inference_tasks.clear()
            self.logger.info("Pending inference tasks cancelled.")
        self.logger.info("PredictionService stopped.")

    async def _handle_feature_event(self, event: FeatureEvent) -> None:
        """Handles incoming feature event, updates LSTM buffers, and triggers pipeline if models are ready."""
        if not isinstance(event, FeatureEvent):
            self.logger.warning(f"Received non-FeatureEvent: {type(event)}.")
            return

        if self._process_pool_executor is None:
            self.logger.error("ProcessPoolExecutor not available. Cannot run predictions.")
            return

        if not self._is_running:
            self.logger.warning("PredictionService is not running. Ignoring feature event.")
            return

        # Update LSTM buffers first
        for model_id, buffer in self._lstm_feature_buffers.items():
            predictor_instance = self._predictors.get(model_id)
            if not predictor_instance:  # Should not happen if initialized correctly
                continue
            expected_features = predictor_instance.expected_feature_names
            if not expected_features:
                self.logger.warning(
                    f"LSTM Model {model_id} has no expected_feature_names. Cannot buffer features."
                )
                continue

            # Prepare the 1D feature vector for this specific LSTM model
            feature_vector_1d = self._prepare_features_for_model(event.features, expected_features)
            if feature_vector_1d is not None:
                buffer.append(feature_vector_1d)
                self.logger.debug(
                    f"Appended features to buffer for LSTM model {model_id}. Buffer size: {len(buffer)}/{buffer.maxlen}"
                )
            else:
                self.logger.warning(
                    f"Could not prepare feature vector for LSTM {model_id} from event {event.event_id}. Not buffering."
                )

        # Now check which models are ready for prediction and trigger pipeline
        await self._trigger_predictions_if_ready(event)

    async def _trigger_predictions_if_ready(self, event: FeatureEvent) -> None:
        """
        Checks which models are ready for prediction (non-LSTMs or LSTMs with full buffers)
        and launches the prediction pipeline for them.
        """
        ready_models_info: List[
            Dict[str, Any]
        ] = []  # List of dicts: {"model_id": str, "feature_input": np.ndarray}

        for model_id, predictor_instance in self._predictors.items():
            model_config = next(
                (mc for mc in self._model_configs if mc.get("model_id") == model_id), None
            )
            if not model_config:
                continue

            predictor_type = model_config.get("predictor_type")
            feature_input_for_model: Optional[np.ndarray] = None

            if predictor_type == "lstm" and model_id in self._lstm_feature_buffers:
                buffer = self._lstm_feature_buffers[model_id]
                if len(buffer) == buffer.maxlen:  # Buffer is full, ready for sequence prediction
                    feature_input_for_model = np.array(
                        list(buffer)
                    )  # Shape: (sequence_length, n_features_per_timestep)
                    self.logger.debug(
                        f"LSTM model {model_id} buffer full. Preparing sequence for prediction."
                    )
                else:
                    self.logger.debug(
                        f"LSTM model {model_id} buffer not full ({len(buffer)}/{buffer.maxlen}). Skipping prediction."
                    )
                    continue  # Not ready yet
            else:  # Non-LSTM model or LSTM with sequence_length=1 (handled by _prepare_features_for_model)
                expected_features = predictor_instance.expected_feature_names
                if not expected_features:
                    self.logger.warning(
                        f"Model {model_id} has no expected_features. Cannot prepare input."
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
                self.logger.warning(
                    f"Feature preparation failed for model {model_id} for event {event.event_id}. Not adding to ready list."
                )

        if ready_models_info:
            self.logger.info(
                f"Triggering prediction pipeline for {len(ready_models_info)} models for event {event.event_id}."
            )
            asyncio.create_task(
                self._run_multi_model_prediction_pipeline(event, ready_models_info)
            )
        else:
            self.logger.debug(f"No models ready for prediction for event {event.event_id}.")

    async def _run_multi_model_prediction_pipeline(
        self, event: FeatureEvent, ready_models_data: List[Dict[str, Any]]
    ) -> None:
        """
        Orchestrate inference for models that are ready and publish prediction events.
        Applies ensembling strategy if configured.
        Args:
            event: The original FeatureEvent.
            ready_models_data: List of dicts, each containing model_id, model_config, feature_input, and runner_func.
        """
        loop = asyncio.get_running_loop()
        inference_futures_dict: Dict[str, InferenceTaskType] = {}

        for model_data in ready_models_data:
            model_id = model_data["model_id"]
            model_config = model_data["model_config"]
            feature_input = model_data[
                "feature_input"
            ]  # This is 1D for non-LSTM, 2D for LSTM sequence
            runner_func = model_data["runner_func"]

            # Expected features list is still needed by some runners (e.g. XGBoost for DMatrix)
            # For LSTMs, model_feature_names in config refers to features *per timestep*
            expected_features_for_runner = model_config.get("model_feature_names", [])

            self.logger.debug(
                f"Submitting inference for model {model_id}, event {event.event_id}."
            )
            future: InferenceTaskType = loop.run_in_executor(
                self._process_pool_executor,
                runner_func,
                model_id,
                model_config["model_path"],
                model_config.get("scaler_path"),
                feature_input,  # This is now the prepared 1D vector or 2D sequence
                expected_features_for_runner,
                model_config,
            )
            self._active_inference_tasks.add(future)
            future.add_done_callback(lambda f: self._active_inference_tasks.discard(f))
            inference_futures_dict[model_id] = future

        if not inference_futures_dict:
            self.logger.info(f"No inference tasks were submitted for event {event.event_id}.")
            return

        # Await all scheduled inference tasks for this event
        model_ids_submitted = list(inference_futures_dict.keys())
        futures_to_await = [inference_futures_dict[mid] for mid in model_ids_submitted]

        self.logger.debug(
            f"Awaiting {len(futures_to_await)} inference results for event {event.event_id}."
        )
        # results_list will contain the return values of the runner functions or exceptions
        results_list = await asyncio.gather(*futures_to_await, return_exceptions=True)
        self.logger.debug(
            f"Received {len(results_list)} inference results/exceptions for event {event.event_id}."
        )

        # Store successful individual model predictions before ensembling
        successful_raw_predictions: List[Dict[str, Any]] = []

        for i, result_or_exc in enumerate(results_list):
            model_id_for_result = model_ids_submitted[i]

            if isinstance(result_or_exc, Exception):
                self.logger.error(
                    f"Inference task for model {model_id_for_result} raised an exception: {result_or_exc}",
                    exc_info=result_or_exc,
                )
                continue

            result: Dict[str, Any] = result_or_exc
            if result.get("model_id") != model_id_for_result:
                self.logger.warning(
                    f"Mismatched model_id in result for event {event.event_id}. Expected {model_id_for_result}, got {result.get('model_id')}. Using expected."
                )

            if "error" in result:
                self.logger.error(
                    f"Inference failed for model {model_id_for_result} (event {event.event_id}): {result['error']}"
                )
            elif "prediction" in result:
                current_model_config = next(
                    (
                        mc
                        for mc in self._model_configs
                        if mc.get("model_id") == model_id_for_result
                    ),
                    None,
                )
                if current_model_config:
                    successful_raw_predictions.append(
                        {
                            "model_id": model_id_for_result,
                            "prediction_value": result["prediction"],
                            "prediction_target": current_model_config.get(
                                "prediction_target", "unknown_target"
                            ),
                            "confidence": result.get("confidence"),  # Optional
                            "config": current_model_config,  # Pass full config for potential use in ensembling
                        }
                    )
                else:
                    self.logger.error(
                        f"Could not find config for successfully predicted model_id {model_id_for_result}. This is unexpected."
                    )
            else:
                self.logger.error(
                    f"Invalid inference result format for model {model_id_for_result} (event {event.event_id}): {result}"
                )

        # --- Apply Ensembling Strategy ---
        ensemble_strategy = self._service_config.get("ensemble_strategy", "none").lower()
        ensemble_weights = self._service_config.get("ensemble_weights", {})
        # New config for confidence weighted average
        confidence_floor = float(self._service_config.get("confidence_floor", 0.1))

        # Group predictions by target for ensembling
        predictions_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for pred_data in successful_raw_predictions:
            target = pred_data["prediction_target"]
            if target not in predictions_by_target:
                predictions_by_target[target] = []
            predictions_by_target[target].append(pred_data)

        final_predictions_to_publish: List[Dict[str, Any]] = []

        if ensemble_strategy == "none" or not predictions_by_target:
            # Publish individual predictions
            for pred_data in successful_raw_predictions:
                final_predictions_to_publish.append(pred_data)
        else:
            # Apply ensembling for each target group
            for target, preds_for_target in predictions_by_target.items():
                if not preds_for_target:
                    continue

                ensembled_value: Optional[float] = None
                ensembled_confidence: Optional[float] = None
                ensembled_model_id = (
                    f"ensemble_{target.replace('_', '')[:15]}"  # Create a generic ensemble ID
                )

                if ensemble_strategy == "average":
                    if preds_for_target:
                        ensembled_value = sum(
                            p["prediction_value"] for p in preds_for_target
                        ) / len(preds_for_target)
                        ensembled_confidence = (
                            np.mean(
                                [
                                    p.get("confidence")
                                    for p in preds_for_target
                                    if p.get("confidence") is not None
                                ]
                            )
                            if any(p.get("confidence") is not None for p in preds_for_target)
                            else None
                        )

                elif ensemble_strategy == "weighted_average":
                    weighted_sum = 0.0
                    total_weight = 0.0
                    weighted_confidence_sum = 0.0
                    for p_data in preds_for_target:
                        weight = ensemble_weights.get(p_data["model_id"], 0.0)
                        if weight > 0:
                            weighted_sum += p_data["prediction_value"] * weight
                            if p_data.get("confidence") is not None:
                                weighted_confidence_sum += p_data["confidence"] * weight
                            total_weight += weight
                    if total_weight > 0:
                        ensembled_value = weighted_sum / total_weight
                        ensembled_confidence = (
                            weighted_confidence_sum / total_weight
                            if weighted_confidence_sum > 0
                            else None
                        )
                    else:
                        self.logger.warning(
                            f"No valid weights for ensembling target {target}. Cannot compute weighted_average."
                        )

                elif ensemble_strategy == "confidence_weighted_average":
                    weighted_sum = 0.0
                    total_confidence_for_weighting = 0.0
                    valid_preds_count = 0
                    for p_data in preds_for_target:
                        confidence = p_data.get("confidence")
                        if isinstance(confidence, (float, int)) and confidence >= confidence_floor:
                            weighted_sum += p_data["prediction_value"] * confidence
                            total_confidence_for_weighting += confidence
                            valid_preds_count += 1
                    if total_confidence_for_weighting > 0:
                        ensembled_value = weighted_sum / total_confidence_for_weighting
                        ensembled_confidence = (
                            total_confidence_for_weighting / valid_preds_count
                        )  # Average of confidences used
                    else:
                        self.logger.warning(
                            f"No valid confidence scores >= {confidence_floor} for target {target} with confidence_weighted_average. Falling back to simple average if possible."
                        )
                        if preds_for_target:  # Fallback to simple average
                            ensembled_value = sum(
                                p["prediction_value"] for p in preds_for_target
                            ) / len(preds_for_target)
                            ensembled_confidence = (
                                np.mean(
                                    [
                                        p.get("confidence")
                                        for p in preds_for_target
                                        if p.get("confidence") is not None
                                    ]
                                )
                                if any(p.get("confidence") is not None for p in preds_for_target)
                                else None
                            )

                elif ensemble_strategy == "simple_voting":
                    votes: Dict[Any, int] = {}
                    # For voting, prediction_value is the class label
                    for p_data in preds_for_target:
                        class_label = p_data["prediction_value"]
                        votes[class_label] = votes.get(class_label, 0) + 1

                    if votes:
                        max_votes = 0
                        winning_classes = []
                        for label, count in votes.items():
                            if count > max_votes:
                                max_votes = count
                                winning_classes = [label]
                            elif count == max_votes:
                                winning_classes.append(label)

                        if len(winning_classes) == 1:
                            ensembled_value = winning_classes[
                                0
                            ]  # This is the class label, not a float probability
                            # Confidence for simple voting could be #votes_for_winner / #total_votes
                            total_votes_cast = sum(votes.values())
                            ensembled_confidence = (
                                float(max_votes / total_votes_cast)
                                if total_votes_cast > 0
                                else None
                            )
                        else:
                            self.logger.warning(
                                f"Tie in simple voting for target {target}: {winning_classes} with {max_votes} votes each. No consensus."
                            )
                            ensembled_value = None  # Or a specific "NO_CONSENSUS" label
                            ensembled_confidence = None
                    else:
                        ensembled_value = None  # No predictions to vote on
                        ensembled_confidence = None

                # Add other strategies from Advanced Ensemble Strategies.md here later

                if ensembled_value is not None:
                    final_predictions_to_publish.append(
                        {
                            "model_id": ensembled_model_id,
                            "prediction_value": ensembled_value,  # Can be float or class label
                            "prediction_target": target,
                            "confidence": ensembled_confidence,
                        }
                    )
                elif ensemble_strategy not in [
                    "average",
                    "weighted_average",
                    "confidence_weighted_average",
                    "simple_voting",
                ]:
                    self.logger.warning(
                        f"Unknown or unhandled ensemble strategy '{ensemble_strategy}' for target {target}. Publishing individual predictions for this target if any."
                    )
                    for pred_data in (
                        preds_for_target
                    ):  # Fallback to individual if strategy unknown/failed for this target
                        final_predictions_to_publish.append(pred_data)
                else:  # Strategy was average/weighted but resulted in None (e.g. no weights)
                    self.logger.warning(
                        f"Ensemble strategy '{ensemble_strategy}' for target {target} resulted in no value. Publishing individual if any."
                    )
                    for pred_data in preds_for_target:
                        final_predictions_to_publish.append(pred_data)

        # Publish final predictions (either individual or ensembled)
        for final_pred_data in final_predictions_to_publish:
            prediction_event = PredictionEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                trading_pair=event.trading_pair,
                exchange=event.exchange,
                timestamp_prediction_for=event.timestamp_features_for,
                model_id=final_pred_data["model_id"],
                prediction_target=final_pred_data["prediction_target"],
                prediction_value=final_pred_data["prediction_value"],
                confidence=final_pred_data.get("confidence"),
                associated_features=event.features,
            )
            try:
                await self._publish_prediction(prediction_event)
            except Exception as e_pub:
                self.logger.error(
                    f"Failed to publish final prediction for model/ensemble {final_pred_data['model_id']}: {e_pub}",
                    exc_info=True,
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

        if not ordered_feature_values:  # Should not happen if expected_model_features is not empty
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

        return feature_array  # Returns a 1D array

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

    # --- Dynamic Model Reloading ---
    async def handle_config_update(self, event: Any) -> None:  # Define actual event type later
        """
        Handles an event indicating that the configuration may have changed.
        Re-initializes predictors based on the (potentially new) configuration.
        This is a basic step towards dynamic model reloading.
        """
        self.logger.info("Configuration update event received. Re-initializing predictors.")
        try:
            # Assuming self.config is updated by an external ConfigurationManager before this event is sent
            # For simplicity, re-fetch the service-specific config part
            # In a real scenario, the new config object might be passed in the event itself.
            # For now, we assume the main `config` dict passed to __init__ is mutable and updated elsewhere, or
            # that a new one is available via some shared mechanism (not shown here)

            # Re-fetch model configurations from the global config object
            # This is a simplification. A more robust way would be to get new config from event or a config manager.
            # For this example, we'll assume self._service_config refers to the prediction_service part of an
            # already updated global config.

            # If your main config object (passed in __init__) can be replaced, you'd need a mechanism
            # for that. If it's mutable and updated in place by a ConfigManager, then this is okay.
            self.logger.info("Re-fetching model configurations for re-initialization.")
            self._model_configs = self._service_config.get("models", [])

            if not self._model_configs:
                self.logger.warning(
                    "No models found in configuration during re-initialization. Service may become inactive."
                )

            # Before re-initializing, you might want to handle active tasks for models being removed/changed.
            # This is complex. A simpler approach for now is just to re-initialize.
            # Active tasks for old models will complete or be cancelled on service stop.

            self._initialize_predictors()  # This will clear and repopulate self._predictors and self._predictor_runners
            self.logger.info("Predictors re-initialized successfully.")
            # Log new state
            self.logger.info(
                f"PredictionService reconfigured. {len(self._predictors)} models now active."
            )
            for model_id, predictor in self._predictors.items():
                self.logger.info(
                    f"  Active Model ID: {model_id}, Type: {type(predictor).__name__}, "
                    f"Expected features: {predictor.expected_feature_names}"
                )

        except ValueError as ve:  # Catch critical init errors
            self.logger.critical(
                f"CRITICAL: Failed to re-initialize predictors due to: {ve}. Service may be unstable.",
                exc_info=True,
            )
            # Depending on policy, could try to revert to old predictors or stop service.
        except Exception:
            self.logger.error("Error during predictor re-initialization.", exc_info=True)

    async def _subscribe_to_config_updates(self) -> None:
        """Subscribes to configuration update events if the EventType exists."""
        if hasattr(EventType, "CONFIG_UPDATED"):
            try:
                await self.pubsub.subscribe(EventType.CONFIG_UPDATED, self.handle_config_update)
                self.logger.info(
                    "Subscribed to CONFIG_UPDATED events for dynamic model reloading."
                )
            except Exception:
                self.logger.error("Failed to subscribe to CONFIG_UPDATED events.", exc_info=True)

    # Modify start() to include subscription to config updates
    # Original start() method needs to be located and this new one merged/replaced.
    # For this edit, I will assume the original start() is being replaced by this new one.
    async def start_with_config_subscription(self) -> None:  # Renamed to avoid conflict if merging
        """
        Start listening for feature events and configuration updates.
        Predictor assets are loaded during __init__.
        """
        if self._is_running:
            self.logger.warning("PredictionService already running.")
            return

        if not self._predictors and sum(1 for mc in self._model_configs if mc.get("critical")) > 0:
            self.logger.critical(
                "PredictionService cannot start: Critical models failed to initialize. Check configuration."
            )
            raise RuntimeError("Critical models failed to initialize for PredictionService.")
        if not self._predictors:
            self.logger.warning("PredictionService starting with no valid predictors initialized.")

        self._is_running = True
        self.pubsub.subscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
        await self._subscribe_to_config_updates()  # Subscribe to config updates

        self.logger.info(
            f"PredictionService started. {len(self._predictors)} models configured and initialized."
        )
        for model_id, predictor in self._predictors.items():
            self.logger.info(
                f"  Model ID: {model_id}, Type: {type(predictor).__name__}, "
                f"Expected features: {predictor.expected_feature_names}"
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
