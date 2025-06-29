"""Implementation of the PredictorInterface for scikit-learn models.

This module provides a sklearn-compatible predictor that can be used
with the prediction service to generate trade signals based on
scikit-learn models.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np

from gal_friday.interfaces.predictor_interface import PredictorInterface


@dataclass
class InferenceRequest:
    """Container for inference request parameters."""

    model_id: str
    model_path: str
    scaler_path: str | None
    feature_vector: np.ndarray[Any, Any]
    model_feature_names: list[str]
    predictor_specific_config: dict[str, Any]


# Define protocol classes for type checking
class ModelWithProba(Protocol):
    """Protocol for models that implement predict_proba method."""

    def predict_proba(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict class probabilities for samples.

        Args:
            x: Input samples.

        Returns:
            Array of class probabilities.
        """
        ...


class ModelWithPredict(Protocol):
    """Protocol for models that implement the predict method.

    This protocol defines the interface for models that provide prediction
    capabilities through a `predict` method, which is common in scikit-learn
    compatible models.
    """
    def predict(self, x: np.ndarray[Any, Any]) -> Any:
        """Predict target values for the input data.

        Args:
            x: Input samples as a numpy array of shape (n_samples, n_features).

        Returns:
            Predicted target values, typically as a numpy array but may be scalar.
        """
        ...


class Model(ModelWithPredict, Protocol):
    """Protocol combining both predict and predict_proba methods.

    This protocol extends ModelWithPredict to include the predict_proba method,
    which is used by classifiers that can predict class probabilities.
    """
    def predict_proba(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict class probabilities for the input data.

        Args:
            x: Input samples as a numpy array of shape (n_samples, n_features).

        Returns:
            Array of class probabilities with shape (n_samples, n_classes).
        """
        ...


class Transformer(Protocol):
    """Protocol for data transformers.

    This protocol defines the interface for transformers that can transform
    input data, typically used for feature scaling or other preprocessing steps.
    """
    def transform(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Transform input data.

        Args:
            x: Input data to transform as a numpy array.

        Returns:
            Transformed data as a numpy array.
        """
        ...


class SKLearnPredictor(PredictorInterface):
    """Scikit-learn model predictor implementation."""

    def __init__(
        self,
        model_path: str,
        model_id: str,
        config: dict[str, Any] | None = None) -> None:
        """Initialize the SKLearn predictor.

        Args:
            model_path: Path to the scikit-learn model file (joblib/pickle).
            model_id: Unique identifier for this model.
            config: Additional configuration parameters for this model.
                    Expected to contain 'model_feature_names': List[str] (if not in model.feature_names_in_).
                    'scaler_path' is no longer used as features are expected pre-scaled.
        """
        super().__init__(model_path, model_id, config)
        self.model: Any = None
        self.scaler: Any = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{model_id}")
        self.logger.debug("SKLearnPredictor initialized for model: %s", model_id)

    def _raise_error(self, error_type: type[Exception], message: str) -> None:
        """Abstracted error raising helper function.

        Args:
            error_type: The exception class to raise
            message: Error message to log and include in the exception

        Raises:
        ------
            Exception: The specified error type with the given message
        """
        self.logger.error(message)

        def _raise() -> None:
            def __raise() -> None:
                raise error_type(message)

            return __raise()

        return _raise()

    def load_assets(self) -> None:
        """Load a scikit-learn model and its associated scaler from specified paths.

        Sets self.model and self.scaler.

        Raises:
        ------
            FileNotFoundError: If the model or scaler file does not exist.
            Exception: If the model or scaler cannot be loaded.
        """
        self.logger.info("Loading assets for model: %s", self.model_id)
        # Load Model
        try:

            def _raise_model_not_found() -> None:
                error_msg = f"Model file not found: {self.model_path}"
                return self._raise_error(FileNotFoundError, error_msg)

            if not Path(self.model_path).exists():
                _raise_model_not_found()

            self.model = joblib.load(self.model_path)
            self.logger.info("Scikit-learn model loaded successfully from %s", self.model_path)
        except FileNotFoundError:
            raise
        except Exception:
            self.logger.exception("Failed to load scikit-learn model from %s", self.model_path)
            raise

        # Load Scaler
        self.scaler = None # Scaler is no longer loaded or used by this predictor.
        self.logger.info(
            "Scaler attribute is set to None. Features are expected to be pre-scaled.")

    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate predictions using the scikit-learn model.
        Features are expected to be pre-scaled by the FeatureEngine.

        Args:
        ----
            features: A 1D numpy array of pre-scaled, ordered feature values.

        Returns:
        -------
            Prediction results as a numpy array.
            For classifiers with predict_proba, returns class probabilities
            (typically of the positive class). For regressors or
            classifiers without predict_proba, returns predicted values.

        Raises:
        ------
            ValueError: If features have wrong shape or contain invalid values.
            TypeError: If model is not properly loaded or is of an unexpected type.
            Exception: If prediction fails for any other reason.
        """
        if self.model is None:
            error_msg = "Model not loaded"
            self.logger.error(error_msg)
            raise TypeError(error_msg)

        if features.ndim != 1:
            error_msg = f"Input features must be a 1D array, got {features.ndim}D."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Features are assumed to be pre-scaled.
        processed_features = features.reshape(1, -1)
        self.logger.debug("Using pre-scaled features directly for prediction.")

        try:
            # For classifiers, prefer predict_proba and return probability of
            # positive class
            def _raise_invalid_model() -> None:
                error_msg = "Model lacks predict/predict_proba methods"
                return self._raise_error(TypeError, error_msg)

            if not (hasattr(self.model, "predict_proba") or hasattr(self.model, "predict")):
                _raise_invalid_model()

            if hasattr(self.model, "predict_proba"):
                # predict_proba returns shape (n_samples, n_classes)
                # We want the probability of the positive class (index 1)
                all_class_probabilities = self.model.predict_proba(processed_features)
                prediction_output = all_class_probabilities[:, 1]  # prob of class 1
            else:
                # For regressors or classifiers without predict_proba
                prediction_output = self.model.predict(processed_features)

            # Ensure output is a 1D numpy array of float32 for consistency
            return np.asarray(prediction_output, dtype=np.float32).flatten()

        except Exception as e:
            self.logger.exception("Scikit-learn prediction failed.")
            error_msg = f"Prediction error: {e!s}"
            raise ValueError(error_msg) from e

    @property
    def expected_feature_names(self) -> list[str] | None:
        """Return the list[Any] of feature names the model expects.

        Tries to get it from model.feature_names_in_ first, then from config.
        """
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            return list[Any](self.model.feature_names_in_)
        return self.config.get("model_feature_names")

    @classmethod
    def _load_model(
        cls,
        model_path: str,
        model_id: str) -> tuple[Any | None, dict[str, Any] | None]:
        """Load the model from disk.

        Returns:
        -------
            tuple: (model, error_dict) where error_dict is None if successful
        """
        if not Path(model_path).exists():
            return None, {
                "error": f"Model file not found: {model_path}",
                "model_id": model_id,
            }
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return None, {
                "error": f"Error loading model: {e!s}",
                "model_id": model_id,
            }
        else:
            logging.getLogger(
                f"{cls.__name__}:{model_id}").debug("Model loaded from %s", model_path)
            return model, None

    @classmethod
    def _load_scaler(
        cls,
        scaler_path: str | None,
        model_id: str) -> tuple[Any | None, dict[str, Any] | None]:
        """Load the scaler from disk.

        Returns:
        -------
            tuple: (scaler, error_dict) where error_dict is None if successful
        """
        if not scaler_path:
            return None, None

        if not Path(scaler_path).exists():
            return None, {
                "error": f"Scaler file not found: {scaler_path}",
                "model_id": model_id,
            }

        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            return None, {
                "error": f"Error loading scaler: {e!s}",
                "model_id": model_id,
            }
        else:
            logging.getLogger(
                f"{cls.__name__}:{model_id}").debug("Scaler loaded from %s", scaler_path)
            return scaler, None

    @classmethod
    def _prepare_features(
        cls,
        feature_vector: np.ndarray[Any, Any],
        scaler: Transformer | None,
        model_id: str) -> tuple[np.ndarray[Any, Any] | None, dict[str, Any] | None]:
        """Prepare and scale features.

        Returns:
        -------
            tuple: (processed_features, error_dict) where error_dict is None if successful
        """
        if feature_vector.ndim != 1:
            return None, {
                "error": "Feature vector must be 1D for processing.",
                "model_id": model_id,
            }

        features_2d = feature_vector.reshape(1, -1)

        if scaler is not None:
            try:
                processed_features = scaler.transform(features_2d)
            except Exception as e:
                return None, {
                    "error": f"Scaler error: {e!s}",
                    "model_id": model_id,
                }
            else:
                logging.getLogger(
                    f"{cls.__name__}:{model_id}").debug("Features scaled.")
                return processed_features, None

        logging.getLogger(
            f"{cls.__name__}:{model_id}").debug("No scaler used.")
        return features_2d, None

    @classmethod
    def _make_prediction(
        cls,
        model: Model,
        processed_features: np.ndarray[Any, Any],
        model_id: str) -> tuple[float | None, float | None, dict[str, Any] | None]:
        """Make prediction using the model.

        Returns:
        -------
            tuple: (prediction, confidence, error_dict) where error_dict is None if successful
        """
        try:
            if hasattr(model, "predict_proba"):
                return cls._predict_with_proba(model, processed_features, model_id)
            if not hasattr(model, "predict"):
                return (
                    None,
                    None,
                    {
                        "error": "Model has no predict or predict_proba method.",
                        "model_id": model_id,
                    })
            return cls._predict_without_proba(model, processed_features, model_id)
        except Exception as e:
            return (
                None,
                None,
                {
                    "error": f"Prediction failed: {e!s}",
                    "model_id": model_id,
                })

    @classmethod
    def _predict_with_proba(
        cls,
        model: ModelWithProba,
        processed_features: np.ndarray[Any, Any],
        model_id: str) -> tuple[float | None, float | None, dict[str, Any] | None]:
        """Make prediction for models with predict_proba method."""
        all_class_probabilities = model.predict_proba(processed_features)
        min_classes_for_binary = 2

        if all_class_probabilities.shape[1] >= min_classes_for_binary:
            prediction = float(all_class_probabilities[0, 1])
            return prediction, prediction, None
        if all_class_probabilities.shape[1] == 1:
            prediction = float(all_class_probabilities[0, 0])
            return prediction, prediction, None

        return (
            None,
            None,
            {
                "error": "predict_proba returned no class probabilities.",
                "model_id": model_id,
            })

    @classmethod
    def _predict_without_proba(
        cls,
        model: ModelWithPredict,
        processed_features: np.ndarray[Any, Any],
        model_id: str) -> tuple[float | None, float | None, dict[str, Any] | None]:
        """Make prediction for models without predict_proba method."""
        prediction_output = model.predict(processed_features)

        # Handle scalar output first (including native Python types)
        if isinstance(prediction_output, float | int):
            return float(prediction_output), None, None

        # Handle numpy array output
        if isinstance(prediction_output, np.ndarray):
            if prediction_output.size == 1:
                return float(prediction_output.item()), None, None
            return (
                None,
                None,
                {
                    "error": f"Unsupported array shape: {prediction_output.shape}",
                    "model_id": model_id,
                })

        return (
            None,
            None,
            {
                "error": f"Unsupported prediction output type: {type(prediction_output)}",
                "model_id": model_id,
            })

    # Error messages as module-level constants
    _ERROR_MSG_MODEL_LOADING = "Model loading failed"
    _ERROR_MSG_SCALER_LOADING = "Scaler loading failed"
    _ERROR_MSG_FEATURE_PREP = "Feature preparation failed"
    _ERROR_MSG_PREDICTION = "Prediction failed"

    @classmethod
    def _raise_with_result(cls, result: dict[str, Any], error_type: str) -> None:
        """Raise an error with the given result.

        Args:
            result: The result dictionary to update with error information.
            error_type: The type of error that occurred.

        Raises:
        ------
            ValueError: Always raises with the given error type.
        """
        result["error"] = error_type
        raise ValueError(error_type) from None

    @classmethod
    def run_inference_in_process(
        cls,
        request: InferenceRequest) -> dict[str, Any]:
        """Load model and scaler, preprocess, predict.

        Executed in a separate process.

        Args:
            request: An InferenceRequest instance containing all necessary parameters.
                     `request.scaler_path` is ignored as features are pre-scaled.
                     `request.feature_vector` is expected to be pre-scaled.

        Returns:
        -------
            Dictionary containing prediction results or error information.
        """
        # Ignore unused arguments that are part of the interface
        _ = request.model_feature_names, request.predictor_specific_config
        logger = logging.getLogger(f"{cls.__name__}:{request.model_id}")
        logger.debug("Starting inference in separate process")

        result = {"model_id": request.model_id}

        try:
            # 1. Load Model
            model, error = cls._load_model(request.model_path, request.model_id)
            if error:
                result.update(error)
                result["error"] = cls._ERROR_MSG_MODEL_LOADING
                logger.error(result["error"])
                return result

            # 2. Log if scaler path provided (but ignored)
            if request.scaler_path:
                logger.info("scaler_path provided in InferenceRequest but will be ignored.")

            # 3. Prepare features - Scaling part of _prepare_features is effectively bypassed.
            # Now, features are pre-scaled, so just reshape.
            if request.feature_vector.ndim != 1:
                result["error"] = f"{cls._ERROR_MSG_FEATURE_PREP}: Feature vector must be 1D for processing."
                logger.error(result["error"])
                return result

            processed_features = request.feature_vector.reshape(1, -1)
            logger.debug("Using pre-scaled feature vector directly (after reshape).")

            # 4. Make prediction
            if model is None:
                result["error"] = "Model is None after loading"
                logger.error(result["error"])
                return result

            prediction, confidence, error = cls._make_prediction(
                cast("Model", model),
                processed_features,
                request.model_id)

            if error:
                result.update(error)
                result["error"] = cls._ERROR_MSG_PREDICTION
                logger.error(result["error"])
                return result

            # Update result with successful prediction
            result.update(
                {
                    "prediction": str(prediction) if prediction is not None else "",
                    "confidence": str(confidence) if confidence is not None else "",
                })
            logger.debug(
                "Prediction successful: %s, Confidence: %s",
                prediction,
                confidence)

        except FileNotFoundError as e:
            result["error"] = f"File not found: {e!s}"
            logger.exception(result["error"])
            return result
        except Exception as e:
            result["error"] = f"Inference failed: {e!s}"
            logger.exception(result["error"])
            return result
        else:
            return result
