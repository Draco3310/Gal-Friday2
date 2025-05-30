# Predictors Folder (`gal_friday/predictors`) Documentation

## Folder Overview

The `gal_friday/predictors` folder is dedicated to housing concrete implementations of various machine learning models that generate predictive signals for the Gal-Friday trading system. Each predictor class within this folder is designed to adhere to a common `PredictorInterface` (defined in `gal_friday.interfaces` or `gal_friday.core.types`). This standardized approach allows the `PredictionService` to manage and utilize different types of ML models (e.g., LSTMs, Scikit-learn models, XGBoost models) in a uniform manner. A key architectural pattern employed by these predictors is the offloading of CPU-bound inference tasks to separate processes.

## Key Predictor Implementations

This folder contains specific implementations for different ML model types:

### `lstm_predictor.py` (`LSTMPredictor` class)

-   **Purpose:** Implements the `PredictorInterface` for Long Short-Term Memory (LSTM) neural network models, which are well-suited for time-series sequence prediction.
-   **Functionality:**
    -   **Model Loading:**
        -   Supports loading pre-trained LSTM models saved in TensorFlow/Keras formats (e.g., `.h5`, `.keras`).
        -   Can also be adapted to load PyTorch LSTM models (e.g., from a state dictionary via a `.pth` file, typically requiring the model's class definition to be available or specified in configuration for instantiation).
    -   **Scaler Handling:** Loads and applies associated feature scalers (e.g., `StandardScaler` or `MinMaxScaler` from Scikit-learn) that were saved during the model training process, typically using `joblib.load()`.
    -   **Sequence Input:** Designed to accept input features formatted as sequences (e.g., a 3D NumPy array of shape `(n_samples, timesteps, features_per_timestep)`), which is characteristic of LSTM model requirements.
    -   **Prediction:** Performs inference using the loaded LSTM model and returns the prediction output (e.g., a probability, a predicted value, or multiple output heads).

### `sklearn_predictor.py` (`SKLearnPredictor` class)

-   **Purpose:** Implements the `PredictorInterface` for a wide range of machine learning models that are compatible with the Scikit-learn library's API (e.g., Logistic Regression, Support Vector Machines, Random Forests, Gradient Boosting Machines if not using the specialized XGBoost predictor).
-   **Functionality:**
    -   **Model Loading:** Loads serialized Scikit-learn model objects (pipelines or individual estimators) using `joblib.load()` from a `.joblib` or `.pkl` file.
    -   **Scaler Handling:** Similar to `LSTMPredictor`, loads and applies any associated feature scalers saved with `joblib`.
    -   **Prediction Output:**
        -   For classification models, it can be configured to return class probabilities (using the model's `predict_proba()` method) or direct class label predictions (`predict()`).
        -   For regression models, it returns the predicted continuous value.

### `xgboost_predictor.py` (`XGBoostPredictor` class)

-   **Purpose:** Implements the `PredictorInterface` specifically for models trained using the XGBoost (Extreme Gradient Boosting) library.
-   **Functionality:**
    -   **Model Loading:** Loads XGBoost models using the library's native loading mechanism (e.g., `xgb.Booster().load_model()` for binary model files or JSON/UBJ formats).
    -   **Scaler Handling:** Loads and applies associated feature scalers using `joblib.load()`.
    -   **Feature Preparation:** Converts input features (typically a NumPy array or Pandas DataFrame) into XGBoost's internal `xgb.DMatrix` data structure, which is optimized for performance.
    -   **Prediction:** Performs inference using the loaded XGBoost booster and returns the prediction output.

## Common Pattern: Offloaded Inference via `run_inference_in_process`

A critical design pattern shared by all concrete predictor implementations (`LSTMPredictor`, `SKLearnPredictor`, `XGBoostPredictor`) is the inclusion of a class method named `run_inference_in_process`.

-   **Purpose:** This static or class method is specifically designed to be executed in a separate, dedicated process, typically managed by a `concurrent.futures.ProcessPoolExecutor` (see `utils/background_process.py`).
-   **Lifecycle within the Separate Process:**
    1.  **Model Loading:** The method receives file paths to the serialized model and any associated scaler artifacts. Its first task within the new process is to load these artifacts from disk. This ensures that potentially large model objects are loaded into the memory space of the child process, not the main application process.
    2.  **Feature Preprocessing:** It takes the raw input features (usually a NumPy array passed from the main process) and applies any necessary preprocessing steps, primarily using the loaded scaler.
    3.  **Inference:** It then calls the underlying model's prediction method with the preprocessed features.
    4.  **Return Result:** Finally, it returns the prediction result (or error information if inference fails) back to the main process, which receives it via the `Future` object associated with the submitted task.
-   **Importance:** Machine learning model inference, especially for complex models or large input batches, can be CPU-bound and computationally intensive. Performing this inference directly in the main asynchronous event loop of the Gal-Friday application would block the loop, severely degrading the system's responsiveness and its ability to handle other concurrent tasks (like processing real-time market data or responding to API requests). By offloading inference to separate processes, the main event loop remains unblocked, ensuring high throughput and low latency for other critical operations.

## Utilities (`utils/` subfolder)

This subfolder contains utility classes that support the offloaded inference pattern and background task management.

### `utils/background_process.py` (`BackgroundProcessManager`, `BackgroundProcess` classes)

-   **`BackgroundProcessManager` class:**
    -   **Purpose:** A singleton class that manages a shared `concurrent.futures.ProcessPoolExecutor` instance for the entire application.
    -   **Functionality:**
        -   Initializes and holds a reference to the `ProcessPoolExecutor`.
        -   Provides a method (e.g., `submit_task(target_function, *args, **kwargs)`) for services like `PredictionService` to submit functions (such as a predictor's `run_inference_in_process` method) to be executed in the background process pool.
        -   Manages the lifecycle of the executor (e.g., shutdown).
-   **`BackgroundProcess` class:**
    -   **Purpose:** Acts as a wrapper around the `concurrent.futures.Future` object that is returned when a task is submitted to the `ProcessPoolExecutor`.
    -   **Functionality:**
        -   Provides methods to check the status of the background task (e.g., `is_running()`, `is_done()`, `exception()`).
        -   Offers convenient ways to retrieve the result of the task, either synchronously (blocking until completion) or asynchronously (e.g., by providing a callback or allowing polling).
-   **Overall Purpose:** These utilities provide the core infrastructure that enables services, particularly the `PredictionService`, to easily run functions—most notably model inference—in background processes without needing to manage the `ProcessPoolExecutor` or `Future` objects directly.

### `__init__.py` (in `predictors/` and `predictors/utils/`)

-   **Purpose:** These files mark their respective directories (`predictors` and `predictors/utils`) as Python packages.
-   **Key Aspects:**
    -   Enable modules to be imported using package notation.
    -   The `predictors/__init__.py` typically exports the concrete predictor classes (`LSTMPredictor`, `SKLearnPredictor`, `XGBoostPredictor`) and potentially the `BackgroundProcessManager` from `utils` if it's intended to be accessed directly via `from gal_friday.predictors import ...`.

## Interaction with `PredictionService`

-   The `PredictionService` (located in `gal_friday/prediction_service.py`) is the primary consumer of the predictor classes defined in this folder.
-   Based on its configuration (which specifies which models to use, their types, and paths to their artifacts), the `PredictionService` instantiates the appropriate concrete predictor classes (e.g., `LSTMPredictor` for an LSTM model configuration).
-   When the `PredictionService` receives a `FeatureEvent` and needs to generate a prediction, it uses the `BackgroundProcessManager` to invoke the `run_inference_in_process` class method of the relevant predictor.
-   It passes the required feature data and file paths for the model and scaler artifacts to this method. The actual loading of the model and the inference computation then happens in a separate process managed by the `ProcessPoolExecutor`.
-   The `PredictionService` then asynchronously awaits the result from the background process via the `BackgroundProcess` wrapper or a `Future` object.

## Adherence to Standards

The design of the `predictors` folder, with its emphasis on:
-   **Interface Adherence:** All predictors implementing a common `PredictorInterface`.
-   **Offloaded Inference:** Decoupling CPU-bound inference from the main application loop.
-   **Modularity:** Separating different model types into their own predictor classes.

promotes a robust, maintainable, and performant architecture for integrating machine learning models into the trading system. This structure allows for easy addition of new model types and ensures that the core application remains responsive.
