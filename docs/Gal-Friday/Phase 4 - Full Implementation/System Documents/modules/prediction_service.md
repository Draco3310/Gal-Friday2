# PredictionService Module Documentation

## Module Overview

The `gal_friday.prediction_service.py` module is a core component of the Gal-Friday trading system, responsible for generating predictions based on market features. It consumes `FeatureEvent`s, orchestrates inference across one or more configured machine learning models (supporting various types like XGBoost, Scikit-learn, and LSTM), applies ensembling strategies to combine predictions, and finally publishes the results as `PredictionEvent`s via the `PubSubManager`. The service is designed for asynchronous operation and uses a `ProcessPoolExecutor` to run potentially CPU-bound model inference tasks without blocking the main application event loop.

## Key Features

-   **Multi-Model Management:** Manages and utilizes multiple machine learning models concurrently. Supports different model architectures through a common interface.
-   **Standardized Model Interaction:** Employs a `PredictorInterface` that defines a contract for all predictor types, ensuring consistent interaction patterns (e.g., for XGBoost, Scikit-learn, LSTM models).
-   **Non-Blocking Inference:** Leverages `concurrent.futures.ProcessPoolExecutor` to run model inference in separate processes, preventing CPU-bound tasks from blocking the asynchronous event loop.
-   **LSTM Sequence Handling:** For LSTM models, it manages input sequences by buffering features from multiple `FeatureEvent`s until the required sequence length is met.
-   **Configuration Validation:** Performs validation of model configurations at startup to ensure all necessary parameters are present and correctly specified.
-   **Dynamic Configuration Reloading:** Supports dynamic updates to its model configurations by subscribing to `PredictionConfigUpdatedEvent`. This allows for adding, removing, or modifying models without a full service restart.
-   **Ensembling Strategies:** Implements various strategies to combine predictions from multiple models targeting the same outcome, such as "average", "weighted_average", or "none" (no ensembling).
-   **Event-Driven Architecture:**
    -   Subscribes to `FeatureEvent` to receive input features for prediction.
    -   Subscribes to `PredictionConfigUpdatedEvent` to handle dynamic configuration changes.
    -   Publishes `PredictionEvent` containing the output from individual models or ensembled predictions.

## Custom Exceptions

The module defines several custom exceptions to handle specific error conditions:

-   **`PredictionServiceError(Exception)`**: Base class for all errors originating from the PredictionService.
-   **`PredictorImportError(PredictionServiceError)`**: Raised when there's an error importing a predictor class (e.g., due to missing dependencies or incorrect class names).
-   **`ModelConfigError(PredictionServiceError)`**: Raised when a model's configuration is invalid (e.g., missing required fields like `model_path` or `predictor_type`).
-   **`PredictorInitError(PredictionServiceError)`**: Raised when a predictor instance fails to initialize (e.g., model file not found, issues loading the model).
-   **`CriticalModelError(PredictionServiceError)`**: Raised if a model marked as 'critical' in the configuration fails to load or initialize, preventing the service from starting correctly.

## Class `PredictionService`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config (dict)`: A dictionary containing the configuration specific to the prediction service, typically extracted from the global application configuration (e.g., `app_config["prediction_service"]`).
    -   `pubsub_manager (PubSubManager)`: An instance of `PubSubManager` for event subscription and publication.
    -   `process_pool_executor (concurrent.futures.ProcessPoolExecutor)`: An instance of `ProcessPoolExecutor` for running inference tasks.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
    -   `configuration_manager (Optional[ConfigurationManager])`: An optional `ConfigurationManager` instance. If provided, allows the service to fetch the latest configurations, potentially supporting more dynamic updates beyond `PredictionConfigUpdatedEvent`.
-   **Actions:**
    -   Stores references to the provided `pubsub_manager`, `process_pool_executor`, `logger_service`, and `configuration_manager`.
    -   Loads its service-specific configuration from the `config` dictionary.
    -   Initializes internal data structures:
        -   `_predictors (dict)`: Stores initialized predictor instances, keyed by `model_id`.
        -   `_model_configs (dict)`: Stores validated model configurations, keyed by `model_id`.
        -   `_lstm_buffers (defaultdict(deque))`: Buffers for LSTM models, storing sequences of features.
        -   `_pending_inference_tasks (dict)`: Tracks active inference tasks submitted to the process pool.
    -   Calls `_initialize_predictors()` to load and set up all configured models.
    -   Sets `_is_ready` flag based on successful initialization, particularly checking critical models.

### Predictor Initialization and Management

-   **`_get_predictor_map() -> dict`**:
    -   Returns a dictionary mapping predictor type strings (e.g., "xgboost", "sklearn", "lstm") to their corresponding predictor classes (e.g., `XGBoostPredictor`) and the static method used to run inference in a separate process (e.g., `XGBoostPredictor.run_inference_in_process`).
    -   This map is used to dynamically instantiate and execute predictors.

-   **`_validate_model_config(model_conf: dict) -> None`**:
    -   Validates an individual model's configuration dictionary (`model_conf`).
    -   Checks for mandatory fields like `model_id`, `predictor_type`, `model_path`, and `prediction_target`.
    -   For LSTM models, also validates `sequence_length`.
    -   Raises `ModelConfigError` if validation fails.

-   **`_initialize_lstm_buffer(model_id: str, model_conf: dict) -> None`**:
    -   If the model is an LSTM (`predictor_type == "lstm"`), this method initializes a `collections.deque` with a `maxlen` equal to the model's `sequence_length`.
    -   This deque is stored in `_lstm_buffers[model_id]` and will be used to accumulate feature sets for sequential input.

-   **`_initialize_predictors() -> None`**:
    -   The main method responsible for iterating through all model configurations defined in `self._service_config["models"]`.
    -   For each model configuration:
        -   Calls `_validate_model_config()`.
        -   Calls `_initialize_single_predictor()` to create and store the predictor instance.
        -   Calls `_initialize_lstm_buffer()` if it's an LSTM model.
    -   After attempting to initialize all predictors, it calls `_validate_critical_models_loaded()` to ensure all models marked as `is_critical` have been successfully loaded.
    -   Sets the service's `_is_ready` flag.

-   **`_initialize_single_predictor(config: dict) -> Optional[PredictorInterface]`**:
    -   Takes a single model `config` dictionary.
    -   Uses `_get_predictor_map()` to find the appropriate predictor class based on `config["predictor_type"]`.
    -   Instantiates the predictor class with the model config, logger, and model-specific parameters.
    -   Stores the initialized predictor in `self._predictors[config["model_id"]]`.
    -   Handles potential `PredictorImportError` or `PredictorInitError` during this process, logging errors. Returns the predictor instance or `None` on failure.

-   **`_validate_critical_models_loaded(all_model_configs: list, successfully_loaded_ids: set) -> None`**:
    -   Checks if all models marked with `is_critical: true` in `all_model_configs` are present in the `successfully_loaded_ids` set.
    -   If any critical model failed to load, it raises a `CriticalModelError`, which typically prevents the service from starting or marks it as not ready.

### Service Lifecycle

-   **`async start() -> None`**:
    -   Subscribes the service to:
        -   `FeatureEvent` via `_handle_feature_event`.
        -   `PredictionConfigUpdatedEvent` via `_handle_prediction_config_updated_event`.
    -   Logs that the PredictionService has started.
    -   Checks if all critical models were initialized successfully. If not, logs a critical error and the service may not process predictions.

-   **`async stop() -> None`**:
    -   Unsubscribes from all events previously subscribed to.
    -   Cancels any pending inference tasks in `_pending_inference_tasks`.
    -   Logs that the PredictionService is stopping.

### Event Handling

-   **`async _handle_feature_event(event: FeatureEvent) -> None`**:
    -   The primary handler for incoming `FeatureEvent`s.
    -   If the service is not ready (e.g., critical models failed), it logs a warning and ignores the event.
    -   Iterates through all configured LSTM models:
        -   Prepares features specifically for that LSTM model using `_prepare_features_for_model()`.
        -   Appends the prepared features to the corresponding buffer in `_lstm_buffers`.
    -   Calls `_trigger_predictions_if_ready(event)` to initiate the prediction pipeline for models that have sufficient data.

-   **`async _handle_prediction_config_updated_event(event: PredictionConfigUpdatedEvent) -> None`**:
    -   Handles `PredictionConfigUpdatedEvent`, which signals that the prediction service's configuration (e.g., models, ensembling strategy) has changed.
    -   It typically reloads its service-specific configuration (e.g., from `ConfigurationManager` or the event payload).
    -   Clears existing predictors and LSTM buffers.
    -   Calls `_initialize_predictors()` again to set up the service with the new configuration.
    -   Logs the configuration update and re-initialization process.

### Prediction Pipeline

-   **`async _trigger_predictions_if_ready(event: FeatureEvent) -> None`**:
    -   Determines which models are ready to make a prediction based on the current `FeatureEvent`.
    -   For LSTM models: checks if their respective buffers in `_lstm_buffers` are full (i.e., contain `sequence_length` feature sets). If full, it extracts the sequence.
    -   For non-LSTM models: they are generally always ready upon receiving a new `FeatureEvent`. Features are prepared using `_prepare_features_for_model()`.
    -   Collects data for all ready models into a `ready_models_data` structure (mapping model_id to prepared features).
    -   If any models are ready, calls `_run_multi_model_prediction_pipeline()`.

-   **`async _run_multi_model_prediction_pipeline(event: FeatureEvent, ready_models_data: dict) -> None`**:
    -   Orchestrates the inference process for all models identified in `ready_models_data`.
    -   Calls `_submit_inference_tasks()` to dispatch inference jobs to the `ProcessPoolExecutor`.
    -   Awaits the results and then calls `_process_inference_results()` to handle them.
    -   Finally, calls `_apply_ensembling_and_publish()` with the successful predictions.

-   **`async _submit_inference_tasks(event: FeatureEvent, ready_models_data: dict) -> tuple[list, list]`**:
    -   Takes the `FeatureEvent` and `ready_models_data`.
    -   For each `model_id` and its `features` in `ready_models_data`:
        -   Retrieves the predictor instance and its process runner method from `_get_predictor_map()`.
        -   Submits the `run_inference_in_process` method to the `_process_pool_executor` with the necessary arguments (model path, features, scaler path, etc.).
        -   Stores the returned `Future` object in `_pending_inference_tasks` and a list of futures to await.
    -   Returns a list of `model_ids` for which tasks were submitted and the corresponding list of `futures`.

-   **`_process_inference_results(event: FeatureEvent, model_ids: list, results: list) -> dict`**:
    -   Receives the original `FeatureEvent`, the list of `model_ids` for which inference was run, and the `results` (outputs from the `Future` objects, which could be successful predictions or exceptions).
    -   Iterates through the results:
        -   If a result is an exception, logs the error.
        -   If successful, it's typically a dictionary containing prediction values (e.g., `{"prediction": 0.75, "confidence": 0.9}`).
        -   Stores successful predictions in a dictionary keyed by `model_id`.
    -   Removes completed tasks from `_pending_inference_tasks`.
    -   Returns the dictionary of successful predictions.

-   **`_prepare_features_for_model(event_features: dict, expected_model_features: list) -> Optional[numpy.ndarray]`**:
    -   Takes the feature dictionary from a `FeatureEvent` (`event_features`) and a list of feature names that the specific model expects (`expected_model_features`, usually part of the model's configuration).
    -   Creates a 1D NumPy array containing the values of the `expected_model_features` in the correct order.
    -   Handles potential missing features (e.g., by logging an error or using a default value, though the latter should be used cautiously).
    -   Performs necessary type conversions to ensure features are in a format suitable for the model (e.g., float).
    -   Returns the prepared NumPy array or `None` if critical features are missing.

### Ensembling and Publishing

-   **`async _apply_ensembling_and_publish(event: FeatureEvent, successful_predictions: dict) -> None`**:
    -   Takes the original `FeatureEvent` and the `successful_predictions` dictionary (model_id -> prediction_output).
    -   Groups predictions by their `prediction_target` (as defined in model configurations).
    -   For each target:
        -   If `ensemble_strategy` is "none", publishes individual `PredictionEvent` for each model's output for that target.
        -   If `ensemble_strategy` is "average", calls `_apply_average_ensembling()`.
        -   If `ensemble_strategy` is "weighted_average", calls `_apply_weighted_average_ensembling()`.
        -   The ensembled result is then used to create and publish a single `PredictionEvent` for that target.

-   **`_apply_average_ensembling(predictions_for_target: list) -> dict`**:
    -   Calculates a simple average of the `prediction` values from the list of predictions for a specific target.
    -   May also average confidence scores if available.
    -   Returns a dictionary representing the ensembled prediction.

-   **`_apply_weighted_average_ensembling(predictions_for_target: list, ensemble_weights: dict) -> dict`**:
    -   Calculates a weighted average of `prediction` values.
    -   `ensemble_weights` (from service config) maps `model_id` to its weight.
    -   Normalizes weights if they don't sum to 1.
    -   Returns a dictionary representing the weighted ensembled prediction.

-   **`async _publish_prediction(prediction_event_data: dict)`**:
    -   Constructs a `PredictionEvent` object using `prediction_event_data`.
    -   The `prediction_event_data` would include `event_id`, `timestamp`, `source_event_id` (from `FeatureEvent`), `model_id` (or "ensemble_model_id"), `prediction_target`, `prediction_value`, `confidence`, and other relevant metadata.
    -   Publishes the `PredictionEvent` using `_pubsub_manager.publish()`.

### Configuration (Key options from `prediction_service` section of app config)

The service is configured through a dictionary, typically found under `prediction_service` in the main application configuration.

-   **`models (List[dict])`**: A list where each dictionary defines a model:
    -   `model_id (str)`: A unique identifier for this model instance (e.g., "lstm_btc_price_1h").
    -   `predictor_type (str)`: The type of predictor, corresponding to keys in `_get_predictor_map()` (e.g., "xgboost", "sklearn", "lstm").
    -   `model_path (str)`: Filesystem path to the serialized/trained model file (e.g., "models/xgboost_model.json", "models/lstm_model.h5").
    -   `scaler_path (Optional[str])`: Path to a corresponding feature scaler object (e.g., a pickled Scikit-learn scaler) if the model requires it.
    -   `is_critical (Optional[bool])`: Defaults to `False`. If `True`, the PredictionService will fail to start or become ready if this model cannot be loaded.
    -   `sequence_length (Optional[int])`: Required for `predictor_type: "lstm"`. Specifies the number of timesteps the LSTM model expects as input.
    -   `prediction_target (str)`: A descriptive name for what this model predicts (e.g., "BTC/USD_price_direction_1h", "ETH/USD_volatility_cluster_15m"). This is used for grouping predictions for ensembling.
    -   `expected_features (List[str])`: A list of feature names that this model expects as input, in the order the model was trained on them.
    -   *Other model-specific parameters*: Additional parameters can be included that are specific to the predictor type (e.g., hyperparameters, layer configurations if the predictor handles model building).
-   **`ensemble_strategy (str)`**: Defines how predictions from multiple models for the same `prediction_target` are combined.
    -   `"none"`: No ensembling. Each model's prediction is published individually.
    -   `"average"`: Simple averaging of prediction values.
    -   `"weighted_average"`: Weighted averaging based on `ensemble_weights`.
-   **`ensemble_weights (Optional[dict])`**: Required if `ensemble_strategy` is "weighted_average". A dictionary mapping `model_id` to a numerical weight (e.g., `{"lstm_model_1": 0.6, "xgboost_model_1": 0.4}`).
-   **`confidence_floor (Optional[float])`**: (Currently noted as a TODO/future enhancement). A minimum confidence score a prediction must have to be considered valid or used in ensembling.

## Interfaces Used

-   **`PredictorInterface` (from `gal_friday.interfaces.predictor_interface`)**:
    -   This is a crucial abstract base class or protocol that all concrete predictor implementations (e.g., `LSTMPredictor`, `SKLearnPredictor`, `XGBoostPredictor`) must adhere to.
    -   It defines a standard contract for model interaction, typically including:
        -   `__init__(model_config: dict, logger: LoggerService, **kwargs)`: Constructor.
        -   `predict(features: numpy.ndarray) -> dict`: Performs inference on the provided features and returns a dictionary of results (e.g., prediction, confidence). This method is called within the separate process.
        -   `@staticmethod run_inference_in_process(model_path: str, features: numpy.ndarray, scaler_path: Optional[str] = None, **kwargs) -> dict`: A static method designed to be the target for `ProcessPoolExecutor`. It loads the model and scaler (if any) using the provided paths, calls the instance's `predict` method (or equivalent logic), and returns the prediction result. This ensures that model loading happens in the child process, avoiding issues with non-picklable models or large objects in the main process.

## Dependencies

-   **`asyncio`**: For asynchronous programming.
-   **`uuid`**: For generating unique event IDs.
-   **`concurrent.futures.ProcessPoolExecutor`**: For running inference tasks in separate processes.
-   **`numpy`**: For numerical operations, especially preparing feature arrays for models.
-   **`collections.deque`**: For LSTM feature sequence buffers.
-   **`gal_friday.core.events`**: Definitions for `FeatureEvent`, `PredictionEvent`, `PredictionConfigUpdatedEvent`.
-   **`gal_friday.core.pubsub.PubSubManager`**: For event-driven communication.
-   **`gal_friday.interfaces.predictor_interface.PredictorInterface`**: The interface for predictor models.
-   **`gal_friday.logger_service.LoggerService`**: For structured logging.
-   **`gal_friday.configuration_manager.ConfigurationManager`** (Optional): For dynamic configuration updates.
-   Specific predictor modules that implement `PredictorInterface` (e.g., `gal_friday.predictors.LSTMPredictor`, `gal_friday.predictors.SKLearnPredictor`, `gal_friday.predictors.XGBoostPredictor`). These would contain the actual model loading and inference logic for each type.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `PredictionService` module.
