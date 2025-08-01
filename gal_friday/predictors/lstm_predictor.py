"""Implementation of the PredictorInterface for LSTM models (TensorFlow/Keras or PyTorch)."""

# Standard library imports
import importlib
import logging
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypeAlias, TypeVar, runtime_checkable

# Third-party imports
import joblib
import numpy as np
from scipy.special import softmax

# PyTorch import (conditional)
try:
    import torch
except ImportError:
    torch = None

# Local application imports
from gal_friday.interfaces.predictor_interface import PredictorInterface

# Type variable for exception classes
E = TypeVar("E", bound=Exception)


class LSTMPredictorError(Exception):
    """Base exception for LSTM predictor errors."""


class InvalidDimensionsError(LSTMPredictorError):
    """Raised when input or output dimensions are invalid."""


class UnsupportedFrameworkError(LSTMPredictorError):
    """Raised when an unsupported framework is specified."""


class PredictionError(LSTMPredictorError):
    """Raised when a prediction fails."""


@runtime_checkable
class PredictorModel(Protocol):
    """Protocol defining the interface for prediction models."""

    def predict(self, x: np.ndarray[Any, Any], verbose: int = 0) -> np.ndarray[Any, Any]:
        """Make predictions on input data.

        Args:
            x: Input data for prediction
            verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)

        Returns:
            Numpy array of predictions.
        """
        ...


@runtime_checkable
class TorchModel(Protocol):
    """Protocol for PyTorch model inference."""

    def __call__(self, x: Any) -> Any:  # Using Any to avoid torch import issues
        """Forward pass for PyTorch model.

        Args:
            x: Input tensor
        Returns:
            Model output tensor
        """
        ...


# Type[Any] alias for model assets
ModelAsset: TypeAlias = PredictorModel | TorchModel


class LSTMPredictor(PredictorInterface):
    """Implementation of PredictorInterface for LSTM models."""

    def _raise_error(
        self,
        error_class: type[E],
        message: str,
        log_level: str = "error",
        **kwargs: object) -> NoReturn:
        """Log and raise an error with the specified class and message.

        Args:
            error_class: The exception class to raise
            message: Error message to log and raise
            log_level: Log level to use (default: "error")
            **kwargs: Additional keyword arguments to pass to the logger
        """
        log_method = getattr(self.logger, log_level, self.logger.error)
        # Extract source_module and context from kwargs if present but don't pass them to logger
        # as standard logging.Logger doesn't support these parameters
        _ = kwargs.pop("source_module", None)
        _ = kwargs.pop("context", None)
        # Call log method with standard parameters
        log_method(message)
        raise error_class(message)

    # Class constants
    BINARY_OUTPUTS = 2  # Number of outputs for binary classification
    # Expected number of dimensions for LSTM input (timesteps, features)
    EXPECTED_INPUT_DIMENSIONS = 2

    def __init__(
        self,
        model_path: str,
        model_id: str,
        config: dict[str, Any] | None = None) -> None:
        """Initialize the LSTMPredictor.

        Args:
            model_path: Path to the LSTM model file (e.g., .h5, .keras, .pth for state_dict).
            model_id: Unique identifier for this model.
            config: Additional configuration parameters. Expected to contain:
                    'framework': 'tensorflow' or 'pytorch'.
                    'model_feature_names': List[str] (features per timestep).
                    'sequence_length': int (number of timesteps).
                    'scaler_path' is no longer used as features are expected pre-scaled.
                    If PyTorch: 'model_class_module', 'model_class_name', 'model_init_args' (dict[str, Any]).
        """
        super().__init__(model_path, model_id, config)
        self.framework = self.config.get("framework", "tensorflow").lower()
        self.logger.info(
            "LSTMPredictor initialized for model %s using %s framework.",
            self.model_id,
            self.framework)

    def _load_tensorflow_model(self) -> None:
        """Load a TensorFlow model from the specified path."""
        import tensorflow as tf

        self.model = tf.keras.models.load_model(self.model_path)
        self.logger.info(
            "TensorFlow/Keras LSTM model loaded successfully from %s",
            self.model_path)

    def _load_pytorch_model_instance(self) -> None:
        """Load a PyTorch model instance from the specified path and config."""
        import torch

        model_class_module_str = self.config.get("model_class_module")
        model_class_name_str = self.config.get("model_class_name")
        model_init_args = self.config.get("model_init_args", {})

        if not model_class_module_str or not model_class_name_str:
            error_msg = (
                f"PyTorch model {self.model_id} requires "
                "'model_class_module' and 'model_class_name' in config."
            )
            self._raise_error(ValueError, error_msg)

        try:
            module = importlib.import_module(model_class_module_str)
            model_class = getattr(module, model_class_name_str)
            self.model = model_class(**model_init_args)

            # Load model weights
            device = torch.device("cpu")
            self.model.load_state_dict(torch.load(self.model_path, map_location=device))
            self.model.to(device).eval()

            self.logger.info(
                "PyTorch LSTM model (%s) loaded and state_dict applied from %s",
                model_class_name_str,
                self.model_path)
        except Exception:
            self.logger.exception(
                "Failed to import PyTorch model class %s.%s for model %s",
                model_class_module_str,
                model_class_name_str,
                self.model_id)
            self._raise_error(ImportError, "Failed to import PyTorch model class")

    def _load_scaler_asset(self) -> None:
        """Load and configure the scaler asset if path is provided."""
        if not self.scaler_path:
            self.logger.info(
                "No scaler_path provided for LSTM model %s. Proceeding without a scaler.",
                self.model_id)
            self.scaler = None
            return

        try:
            if not Path(self.scaler_path).exists():
                error_msg = f"Scaler file not found: {self.scaler_path}"
                self._raise_error(FileNotFoundError, error_msg)

            self.scaler = joblib.load(self.scaler_path)
            self.logger.info("Scaler loaded successfully from %s", self.scaler_path)

        except FileNotFoundError:
            raise
        except Exception:
            self.logger.exception(
                "Failed to load scaler from %s",
                self.scaler_path)
            raise

    def _validate_model_file(self) -> None:
        """Validate that the model file exists and is accessible."""
        model_file_path = Path(self.model_path)
        if not model_file_path.exists():
            error_msg = (
                f"LSTM Model file not found: {self.model_path}. "
                "Please ensure the file exists and the path is correct."
            )
            self._raise_error(FileNotFoundError, error_msg)

    def load_assets(self) -> None:
        """Load an LSTM model and its associated scaler.

        For PyTorch, loads a state_dict and requires model class info in config.
        Sets self.model and self.scaler.
        """
        self.logger.info(
            "Loading assets for LSTM model: %s using framework: %s",
            self.model_id,
            self.framework)

        try:
            self._validate_model_file()

            if self.framework == "tensorflow":
                self._load_tensorflow_model()
            elif self.framework == "pytorch":
                self._load_pytorch_model_instance()
            else:
                self._raise_error(
                    UnsupportedFrameworkError,
                    f"Unsupported LSTM framework: {self.framework}")

            self._load_scaler_asset()

        except ImportError as e:
            self.logger.exception(
                "Import error for %s. Is it installed/correctly specified?",
                self.framework)
            self._raise_error(
                ImportError,
                f"Required ML framework ({self.framework}) or model class not found: {e!s}")
        except Exception as e:
            self.logger.exception("Failed to load LSTM model from %s", self.model_path)
            self._raise_error(
                Exception,
                f"Failed to load LSTM model from {self.model_path}: {e!s}")

    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate predictions using the loaded LSTM model (instance method).

        This method now assumes `features` is the 2D sequence (timesteps, features_per_step).
        This method is primarily for consistency with PredictorInterface; actual inference
        for process pool is handled by `run_inference_in_process`.
        """
        # Check if model is loaded
        if self.model is None:
            self.logger.error(f"LSTM model {self.model_id} is not loaded. Cannot predict.")
            raise TypeError(f"LSTM model {self.model_id} is not loaded")

        # Validate input dimensions - expecting (timesteps, features_per_step)
        if features.ndim != self.EXPECTED_INPUT_DIMENSIONS:
            error_msg = (
                f"Input features must be a {self.EXPECTED_INPUT_DIMENSIONS}D array for "
                f"LSTMPredictor.predict, got {features.ndim}D."
            )
            self.logger.error(error_msg)
            raise InvalidDimensionsError("Invalid input dimensions")

        # Apply scaling if available - REMOVED as features are pre-scaled
        scaled_sequence = features # Use features directly
        self.logger.debug("Using pre-scaled features directly for LSTM prediction.")


        # Reshape for model input: (batch_size, timesteps, features_per_timestep)
        model_input = scaled_sequence.reshape(1, *scaled_sequence.shape)

        # Validate framework
        if self.framework not in ["tensorflow", "pytorch"]:
            self.logger.error(f"Unsupported framework {self.framework} for prediction")
            raise UnsupportedFrameworkError(f"Framework '{self.framework}' is not supported")

        # Generate predictions based on framework
        try:
            if self.framework == "tensorflow":
                raw_predictions = self.model.predict(model_input, verbose=0)
                return np.asarray(raw_predictions, dtype=np.float32).flatten()
            # Must be pytorch
            import torch

            with torch.no_grad():
                device = torch.device("cpu")
                torch_input = torch.from_numpy(model_input).float().to(device)
                raw_predictions = self.model(torch_input)

            # Explicitly type the output to help mypy understand it's an ndarray[Any, Any]
            model_output: np.ndarray[Any, Any] = raw_predictions.cpu().numpy().flatten().astype(np.float32)
            if model_output.shape != (1,):
                self.logger.error(
                    f"Model output shape {model_output.shape} is not compatible with "
                    f"expected dimensions for model {self.model_id}")
                raise InvalidDimensionsError(
                    "Model output dimensions do not match expected format")
        except Exception as e:
            if isinstance(e, InvalidDimensionsError | UnsupportedFrameworkError):
                raise
            self.logger.exception("Error during prediction for model")
            raise PredictionError(f"Failed to generate prediction: {e}") from e
        else:
            return model_output

    @property
    def expected_feature_names(self) -> list[str] | None:
        """Return the list[Any] of feature names (per timestep) the model expects from config."""
        return self.config.get("model_feature_names")

    @classmethod
    def _load_tf_model(
        cls,
        model_path: str,
        model_id: str,
        logger: logging.Logger) -> tuple[Any, str] | dict[str, str]:
        """Load a TensorFlow model from the given path."""
        import tensorflow as tf

        try:
            tf.get_logger().setLevel("ERROR")
            model_asset = tf.keras.models.load_model(model_path)
            logger.debug("TF Model %s loaded from %s", model_id, model_path)
        except Exception as e_load:
            error_msg = f"Failed to load TF model {model_path}: {e_load!s}"
            logger.exception("Error loading TF model:")
            return {"error": error_msg, "model_id": model_id}
        return model_asset, ""

    @classmethod
    def _load_pytorch_model(
        cls,
        model_path: str,
        model_id: str,
        predictor_config: dict[str, Any],
        logger: logging.Logger) -> tuple[Any, str] | dict[str, str]:
        """Load a PyTorch model from the given path."""
        import importlib

        import torch

        try:
            model_class_module = predictor_config.get("model_class_module")
            model_class_name = predictor_config.get("model_class_name")
            model_init_args = predictor_config.get("model_init_args", {})

            if not model_class_module or not model_class_name:
                error_msg = (
                    f"PyTorch model {model_id} requires 'model_class_module' and "
                    "'model_class_name' in config."
                )
                return {"error": error_msg, "model_id": model_id}

            module = importlib.import_module(model_class_module)
            model_class = getattr(module, model_class_name)

            model_asset = model_class(**model_init_args)
            model_asset.load_state_dict(torch.load(model_path))
            model_asset.eval()
            logger.debug(
                "PyTorch Model %s (%s) loaded from %s",
                model_id,
                model_class_name,
                model_path)
        except Exception as e_load:
            error_msg = f"Failed to load PyTorch model {model_path}: {e_load!s}"
            logger.exception("Error loading PyTorch model:")
            return {"error": error_msg, "model_id": model_id}
        return model_asset, ""

    @classmethod
    def _load_scaler(
        cls,
        scaler_path: str,
        model_id: str,
        logger: logging.Logger) -> tuple[Any, str] | dict[str, str]:
        """Load a scaler from the given path."""
        if not Path(scaler_path).exists():
            error_msg = f"Scaler file not found: {scaler_path}"
            logger.error(error_msg)
            return {"error": error_msg, "model_id": model_id}

        try:
            scaler_asset = joblib.load(scaler_path)
            logger.debug("Scaler for %s loaded from %s", model_id, scaler_path)
        except Exception as e_scale_load:
            error_msg = f"Failed to load scaler {scaler_path}: {e_scale_load!s}"
            logger.exception("Error loading scaler:")
            return None, error_msg
        else:
            return scaler_asset, ""

    @classmethod
    def _process_features(
        cls,
        feature_sequence: np.ndarray[Any, Any], # Expected to be pre-scaled
        scaler_asset: Any, # Kept for compatibility, but should be None
        model_id: str,
        logger: logging.Logger) -> tuple[np.ndarray[Any, Any], str] | dict[str, str]:
        """Process the input features (reshaping only, scaling is removed)."""
        if feature_sequence.ndim != cls.EXPECTED_INPUT_DIMENSIONS:
            error_msg = (
                f"Feature sequence must be {cls.EXPECTED_INPUT_DIMENSIONS}D "
                f"for LSTM, got {feature_sequence.ndim}D."
            )
            logger.error(error_msg)
            return {"error": error_msg, "model_id": model_id}

        # Scaling logic is removed as features are expected to be pre-scaled.
        processed_sequence = feature_sequence
        if scaler_asset is not None:
            logger.warning(
                "scaler_asset provided to _process_features for model %s, "
                "but it will be ignored as features are expected pre-scaled.",
                model_id)

        logger.debug(
            "Using pre-scaled feature sequence (shape: %s) directly.",
            processed_sequence.shape)
        return processed_sequence, ""

    @classmethod
    def _make_prediction(
        cls,
        model_asset: ModelAsset,
        model_input: np.ndarray[Any, Any],
        model_framework: str,
        predictor_config: dict[str, Any]) -> tuple[float, float | None]:
        """Make prediction using the loaded model."""
        is_binary_sigmoid = predictor_config.get("binary_sigmoid_output", False)

        if model_framework == "tf":
            if isinstance(model_asset, PredictorModel):
                prediction_output = model_asset.predict(model_input, verbose=0)
            else:
                raise AttributeError("Model does not implement the PredictorModel protocol")
        else:  # PyTorch
            import torch

            with torch.no_grad():
                input_tensor = torch.FloatTensor(model_input)
                # Check if model_asset is callable
                if callable(model_asset):
                    prediction = model_asset(input_tensor)
                else:
                    raise TypeError("PyTorch model is not callable")
                prediction_output = (
                    prediction[0].numpy() if isinstance(prediction, tuple) else prediction.numpy()
                )

        prediction_output_flat = prediction_output.flatten()

        if len(prediction_output_flat) == cls.BINARY_OUTPUTS and not is_binary_sigmoid:
            # Multi-class, take softmax
            exp_scores = np.exp(prediction_output_flat - np.max(prediction_output_flat))
            probs = exp_scores / exp_scores.sum()
            final_value = float(np.argmax(probs))
            confidence = float(np.max(probs))
        elif len(prediction_output_flat) == 1 or is_binary_sigmoid:
            # Binary classification or regression
            final_value = float(prediction_output_flat[0])
            if predictor_config.get("task_type", "classification") == "classification":
                confidence = max(final_value, 1 - final_value)
            else:
                confidence = None
        else:
            # Multi-class with unexpected format
            final_value = float(np.argmax(prediction_output_flat))
            confidence = float(np.max(softmax(prediction_output_flat, axis=0)))

        return final_value, confidence

    @classmethod
    def run_inference_in_process(
        cls,
        model_id: str,
        model_path: str,
        scaler_path: str | None, # Kept for interface compatibility, but not used for scaling.
        feature_sequence: np.ndarray[Any, Any], # Expected to be pre-scaled.
        predictor_specific_config: dict[str, Any]) -> dict[str, Any]:
        """Load LSTM model, (optionally log scaler_path), use pre-scaled features, and predict.

        Executed in a separate process. Expects `feature_sequence` to be pre-scaled and 2D.
        """
        logger = logging.getLogger(f"{cls.__name__}:{model_id}.run_inference_in_process")
        logger.debug(
            "Starting LSTM inference for model %s in separate process. Input seq shape: %s",
            model_id,
            feature_sequence.shape if feature_sequence is not None else "None")

        try:
            # --- Load model based on framework ---
            model_framework = predictor_specific_config.get("framework", "tf").lower()

            if model_framework == "tf":
                result = cls._load_tf_model(model_path, model_id, logger)
            elif model_framework == "pt": # Corrected from 'pytorch' to 'pt' for consistency if used elsewhere
                result = cls._load_pytorch_model(
                    model_path,
                    model_id,
                    predictor_specific_config,
                    logger)
            else:
                error_msg = f"Unsupported model framework: {model_framework}"
                logger.error(error_msg)
                return {"error": error_msg, "model_id": model_id}

            if isinstance(result, dict): # Error dictionary returned
                return result
            model_asset = result[0] # model_asset, "" was returned

            # --- Scaler handling (logging only, no loading for transform) ---
            if scaler_path:
                logger.info(
                    "scaler_path '%s' provided for model %s but will be ignored "
                    "as features are expected pre-scaled by FeatureEngine.",
                    scaler_path, model_id)
            # scaler_asset remains None as it's not loaded for transformation purposes.

            # --- Process features (reshape only, features are pre-scaled) ---
            # The scaler_asset argument to _process_features will be None.
            result = cls._process_features(
                feature_sequence,
                None, # Pass None for scaler_asset
                model_id,
                logger)
            if isinstance(result, dict): # Error dictionary returned
                return result
            processed_sequence = result[0] # processed_sequence, "" was returned

            # --- Prepare model input ---
            model_input = processed_sequence.reshape(
                1,
                processed_sequence.shape[0],
                processed_sequence.shape[1])
            logger.debug("LSTM model input shape: %s", model_input.shape)

            # --- Make prediction ---
            final_value, confidence = cls._make_prediction(
                model_asset=model_asset,
                model_input=processed_sequence,
                model_framework=model_framework,
                predictor_config=predictor_specific_config)
        except Exception as e:
            error_msg = f"LSTM Inference failed: {e!s}"
            logger.exception("Error during inference:")
            return {"error": error_msg, "model_id": model_id}
        else:
            logger.debug(
                "LSTM Prediction for %s successful: %s, Confidence: %s",
                model_id,
                final_value,
                confidence)
            return {
                "prediction": final_value,
                "confidence": confidence,
                "model_id": model_id,
            }
