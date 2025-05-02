# Prediction Service Module

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import logging
import numpy as np
import xgboost as xgb
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union, Coroutine, TypeVar, cast, Callable

# Import necessary components
from .core.events import (
    Event, 
    EventType, 
    FeatureEvent, 
    PredictionEvent  # Import PredictionEvent from core.events
)
from .logger_service import LoggerService
from .core.pubsub import PubSubManager

# Set up logging
log = logging.getLogger(__name__)

# Define a type for futures to track
T = TypeVar('T')
InferenceTaskType = asyncio.Future[Dict[str, Any]]


# --- Model Inference Function (runs in separate process) ---
def _run_inference_task(
    model_path: str, feature_vector: np.ndarray, model_feature_names: List[str]
) -> Dict[str, Any]:
    """
    Loads a model (native XGBoost format) and runs inference in a separate process.

    Args:
        model_path: Path to the saved model file (e.g., .xgb, .ubj, .json).
        feature_vector: Numpy array of features, ordered correctly.
        model_feature_names: Expected feature names (required for DMatrix).

    Returns:
        Dictionary containing prediction results (e.g., {'prediction': 0.72})
        or an error: {'error': 'Error message'}
    """
    try:
        # --- Load the native XGBoost model ---
        model = xgb.Booster()
        model.load_model(model_path)

        # --- Prepare data and predict ---
        dmatrix = xgb.DMatrix(
            feature_vector.reshape(1, -1),
            feature_names=model_feature_names
        )
        prediction = model.predict(dmatrix)

        # Prediction is usually a numpy array, get the single value
        if isinstance(prediction, np.ndarray) and prediction.size == 1:
            prediction_float = float(prediction.item())
        else:
            # Handle unexpected prediction format
            return {
                "error": (
                    f"Unexpected prediction output format: {type(prediction)}, "
                    f"value: {prediction}"
                )
            }

        return {"prediction": prediction_float}

    except FileNotFoundError:
        return {"error": f"Model file not found: {model_path}"}
    except xgb.core.XGBoostError as xgb_err:
        return {"error": f"XGBoost error: {xgb_err}"}
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}


# --- PredictionService Class ---
class PredictionService:
    """
    Consumes feature events, runs ML model inference (XGBoost native format)
    in a separate process pool, and publishes prediction events.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pubsub_manager: PubSubManager,
        process_pool_executor: ProcessPoolExecutor,
        logger_service: LoggerService,
    ):
        """
        Initializes the PredictionService.

        Args:
            config (dict): Configuration settings. Expected structure:
                 prediction_service:
                    model_path: "/path/to/model/mvp_xgboost_v1.ubj" # Recommend .ubj format
                    model_id: "mvp_xgboost_v1" # Identifier for the model
                    # List of feature names IN THE ORDER the model expects them
                    model_feature_names: ["mid_price", "spread_pct", ...]
                    prediction_target: "prob_price_up_1min" # Description of what's predicted
            pubsub_manager (PubSubManager): For subscribing/publishing events.
            process_pool_executor (ProcessPoolExecutor): Executor for running inference.
            logger_service (LoggerService): The shared logger instance.
        """
        self._config = config.get("prediction_service", {})
        self.pubsub = pubsub_manager  # Store PubSubManager
        self._process_pool_executor = process_pool_executor
        self.logger = logger_service
        self._is_running = False
        self._main_task: Optional[asyncio.Task[Any]] = None
        self._source_module = self.__class__.__name__
        
        # Store handler for unsubscribing
        self._feature_event_handler: Callable[[FeatureEvent], Coroutine[Any, Any, None]] = self._handle_feature_event

        # Model configuration
        self._model_path = self._config.get("model_path")
        self._model_id = self._config.get("model_id", "unknown_model")
        self._model_feature_names = self._config.get("model_feature_names", [])  # CRITICAL
        self._prediction_target = self._config.get("prediction_target", "unknown_target")

        if not self._model_path:
            raise ValueError("PredictionService: 'model_path' configuration is required.")
        if not self._model_feature_names:
            raise ValueError("PredictionService: 'model_feature_names' configuration is required.")

        # Track tasks submitted to the executor to allow cancellation on stop
        self._active_inference_tasks: Set[InferenceTaskType] = set()

    async def start(self) -> None:
        """Loads model info and starts listening for feature events."""
        if self._is_running:
            self.logger.warning(
                "PredictionService already running.",
                source_module=self.__class__.__name__
            )
            return
        self._is_running = True
        
        # Subscribe to FeatureEvent
        self.pubsub.subscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)

        # self._main_task = asyncio.create_task(self._run_event_loop()) # Remove loop
        self.logger.info(
            f"PredictionService started. Using model: {self._model_id} "
            f"from {self._model_path}",
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            f"Expecting features ({len(self._model_feature_names)}): "
            f"{self._model_feature_names}",
            source_module=self.__class__.__name__,
        )

    async def stop(self) -> None:
        """Stops the event processing loop and cancels pending inference futures."""
        if not self._is_running:
            return
        self._is_running = False
        # if self._main_task:
        #    ...
        # self._main_task = None

        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.FEATURES_CALCULATED, self._feature_event_handler)
            self.logger.info("Unsubscribed from FEATURES_CALCULATED.", source_module=self._source_module)
        except Exception as e:
            self.logger.error(f"Error unsubscribing PredictionService: {e}", exc_info=True, source_module=self._source_module)
            
        # Cancel running inference tasks
        if self._active_inference_tasks:
            self.logger.info(f"Cancelling {len(self._active_inference_tasks)} pending inference tasks...")
            for task in self._active_inference_tasks:
                task.cancel()
            # Wait for cancellation to complete
            await asyncio.gather(*self._active_inference_tasks, return_exceptions=True)
            self._active_inference_tasks.clear()
            self.logger.info("Pending inference tasks cancelled.")
            
        # Cancel main task if it existed
        # if self._main_task:
        #    ...
        # self._main_task = None
        self.logger.info("PredictionService stopped.", source_module=self.__class__.__name__)

    async def _handle_feature_event(self, event: FeatureEvent) -> None:
        """Handles incoming FeatureEvent directly."""
        # Check type
        if not isinstance(event, FeatureEvent):
            self.logger.warning(f"Received non-FeatureEvent: {type(event)}", source_module=self._source_module)
            return

        # Check if executor is available
        if self._process_pool_executor is None:
            self.logger.error(
                "ProcessPoolExecutor not available, cannot run prediction.",
                source_module=self.__class__.__name__,
            )
            return
            
        # Schedule the prediction pipeline
        asyncio.create_task(self._run_prediction_pipeline(event))

    async def _run_prediction_pipeline(self, event: FeatureEvent) -> None:
        """Orchestrates the feature preparation, inference, and publishing for an event."""
        try:
            # 1. Prepare features
            # Use event.features directly
            feature_vector = self._prepare_features_for_model(event.features)
            if feature_vector is None:
                # Error already logged in prepare_features_for_model
                return

            # 2. Run inference in executor
            loop = asyncio.get_running_loop()
            inference_future: InferenceTaskType = loop.run_in_executor(
                self._process_pool_executor,
                _run_inference_task, # The function to run
                self._model_path,     # Args for the function
                feature_vector,
                self._model_feature_names
            )
            # Track the future
            self._active_inference_tasks.add(inference_future)
            inference_future.add_done_callback(lambda _: self._active_inference_tasks.discard(inference_future))

            # 3. Await result
            self.logger.debug(f"Waiting for inference result for event {event.event_id}")
            result = await inference_future
            self.logger.debug(f"Received inference result for event {event.event_id}")

            # 4. Process result and publish
            if "error" in result:
                self.logger.error(
                    f"Inference failed for event {event.event_id}: {result['error']}",
                    source_module=self.__class__.__name__
                )
            elif "prediction" in result:
                 # Create PredictionEvent using the one from core.events
                 event_id = uuid.uuid4()
                 timestamp = datetime.utcnow()
                 
                 # Create the event directly without using a factory method
                 prediction_event = PredictionEvent(
                     source_module=self._source_module,
                     event_id=event_id,
                     timestamp=timestamp,
                     trading_pair=event.trading_pair,
                     exchange=event.exchange,
                     timestamp_prediction_for=event.timestamp_features_for,
                     model_id=self._model_id,
                     prediction_target=self._prediction_target,
                     prediction_value=result["prediction"],
                     confidence=None,  # Add confidence if model provides it
                     associated_features=event.features  # Pass original features
                 )
                 await self._publish_prediction(prediction_event)
            else:
                self.logger.error(
                    f"Invalid inference result format for event {event.event_id}: {result}",
                    source_module=self.__class__.__name__
                )

        except asyncio.CancelledError:
             self.logger.info(f"Prediction pipeline cancelled for event {event.event_id}")
             # No need to reraise typically, task was cancelled externally
        except Exception as e:
            self.logger.error(
                f"Error in prediction pipeline for event {event.event_id}: {e}",
                source_module=self.__class__.__name__,
                exc_info=True
            )
            # Optionally publish an error event or take other action

    def _prepare_features_for_model(self, features: Dict[str, str]) -> Optional[np.ndarray]:
        """
        Converts the feature dictionary from FeatureEvent into a numpy array
        ordered according to self._model_feature_names. Handles missing features
        and type conversion errors.
        """
        ordered_feature_values = []
        missing_features = []
        type_errors = []

        for feature_name in self._model_feature_names:
            if feature_name not in features:
                missing_features.append(feature_name)
                ordered_feature_values.append(np.nan)  # Append NaN for missing features
                continue  # Process next feature

            value_str = features[feature_name]
            try:
                # Convert feature value string to float
                value_float = float(value_str)
                ordered_feature_values.append(value_float)
            except (ValueError, TypeError):
                type_errors.append(f"{feature_name}='{value_str}'")
                ordered_feature_values.append(np.nan)  # Append NaN on conversion error

        if missing_features:
            msg = (
                f"Missing expected features for model {self._model_id}: "
                f"{missing_features}. Filling with NaN. Input features: "
                f"{list(features.keys())}"
            )
            self.logger.warning(msg, source_module=self.__class__.__name__)

        if type_errors:
            msg = (
                f"Type conversion error for features: {type_errors}. "
                "Filling with NaN."
            )
            self.logger.error(msg, source_module=self.__class__.__name__)

        # Check if all values ended up being NaN (e.g., all features missing/invalid)
        if all(np.isnan(v) for v in ordered_feature_values):
            msg = (
                f"All features resulted in NaN for model {self._model_id}. "
                "Cannot make prediction."
            )
            self.logger.error(
                msg,
                source_module=self.__class__.__name__
            )
            return None

        return np.array(ordered_feature_values, dtype=np.float32)

    async def _publish_prediction(self, event: PredictionEvent) -> None:
        """Publishes the PredictionEvent."""
        try:
            await self.pubsub.publish(event)
            self.logger.debug(
                f"Published PredictionEvent for {event.trading_pair} at {event.timestamp_prediction_for}",
                source_module=self.__class__.__name__
            )
        except Exception as e:
            self.logger.error(
                f"Failed to publish PredictionEvent: {e}",
                source_module=self.__class__.__name__,
                exc_info=True
            )


# Example Usage Placeholder (Requires ProcessPoolExecutor from main app)
# if __name__ == "__main__": ...
