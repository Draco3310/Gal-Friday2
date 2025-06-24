"""Implementation of the PredictorInterface for XGBoost models.

This module provides an XGBoost-compatible predictor that can be used
with the prediction service to generate trade signals based on
XGBoost models.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib  # For loading scalers
import numpy as np

if TYPE_CHECKING:
    import xgboost as xgb
else:
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None  # type: ignore[assignment]

from gal_friday.interfaces.predictor_interface import PredictorInterface


class ModelLoadError(Exception):
    """Exception raised when a model fails to load."""


class XGBoostPredictor(PredictorInterface):
    """Implementation of PredictorInterface for XGBoost models."""

    def __init__(
        self,
        model_path: str,
        model_id: str,
        config: dict[str, Any] | None = None) -> None:
        """Initialize the XGBoost predictor.

        Args:
        ----
            model_path: Path to the XGBoost model file
            model_id: Unique identifier for this model
            config: Additional configuration parameters for this model.
                    Expected to contain 'model_feature_names': List[str].
                    'scaler_path' is no longer used as features are expected pre-scaled.
        """
        # _expected_features must be set before super().__init__ if load_assets uses it.
        # However, we'll fetch it from self.config inside load_assets now.
        super().__init__(model_path, model_id, config)

    def load_assets(self) -> None:
        """Load an XGBoost model and its associated scaler from the specified paths.

        Sets self.model and self.scaler.

        Raises:
        ------
            xgb.core.XGBoostError: If the model cannot be loaded.
            FileNotFoundError: If the model or scaler file does not exist.
            Exception: If the scaler cannot be loaded.
        """
        self.logger.info("Loading assets for model: %s", self.model_id)
        # Load Model
        try:

            def _raise_model_not_found() -> None:
                error_msg = f"Model file not found: {self.model_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg) from None

            if not Path(self.model_path).exists():
                _raise_model_not_found()

            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            self.logger.info(
                "XGBoost model loaded successfully from %s",
                self.model_path)

        except xgb.core.XGBoostError as e:
            error_msg = f"Failed to load XGBoost model from {self.model_path}"
            self.logger.exception(error_msg)
            raise ModelLoadError(error_msg) from e

        # Load Scaler
        self.scaler = None # Scaler is no longer loaded or used by this predictor.
        self.logger.info(
            "Scaler attribute is set to None. Features are expected to be pre-scaled.")

    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate predictions using the XGBoost model.
        Features are expected to be pre-scaled by the FeatureEngine.

        Args:
        ----
            features: A 1D numpy array of pre-scaled, ordered feature values.

        Returns:
        -------
            Prediction results as a numpy array (usually probabilities for binary:logistic)

        Raises:
        ------
            ValueError: If features have wrong shape or contain invalid values.
            TypeError: If model is not properly loaded.
            xgb.core.XGBoostError: If prediction fails.
        """
        if self.model is None:
            msg = "XGBoost model is not loaded."
            self.logger.error(msg)
            raise TypeError(msg)

        if features.ndim != 1:
            msg = f"Input features must be a 1D array, got {features.ndim}D."
            self.logger.error(msg)
            raise ValueError(msg)

        # Features are assumed to be pre-scaled.
        processed_features = features.reshape(1, -1)
        self.logger.debug("Using pre-scaled features directly for prediction.")

        # Ensure expected_feature_names are available for DMatrix
        model_feature_names = self.expected_feature_names
        if not model_feature_names:
            self.logger.warning(
                "expected_feature_names not available. "
                "DMatrix will be created without feature names.")

        try:
            dmatrix = xgb.DMatrix(processed_features, feature_names=model_feature_names)
            raw_predictions = self.model.predict(dmatrix)
            # For binary:logistic, predict returns probabilities of the positive class.
            # This is typically a 1D array of shape (1) or (n_samples)
            # Ensure it's a simple float or 1D array.
            if isinstance(raw_predictions, np.ndarray):
                # If it's like array([0.7]), get 0.7. If it's already a scalar float, it's fine.
                return raw_predictions.astype(np.float32)  # Ensure correct type
            return np.array(
                [raw_predictions],
                dtype=np.float32)  # Convert scalar to 1-element array

        except xgb.core.XGBoostError:
            self.logger.exception("XGBoost prediction failed.")
            raise
        except Exception:
            self.logger.exception("An unexpected error occurred during prediction.")
            raise

    @property
    def expected_feature_names(self) -> list[str] | None:
        """Return the list[Any] of feature names the model expects from config."""
        return self.config.get("model_feature_names")

    @classmethod
    def _load_model(
        cls,
        model_path: str,
        model_id: str,
        logger: logging.Logger) -> tuple[Any | None, dict[str, Any]]:
        """Load XGBoost model from file."""
        if not Path(model_path).exists():
            return None, {"error": f"Model file not found: {model_path}", "model_id": model_id}

        if xgb is None:
            return None, {"error": "XGBoost library not available", "model_id": model_id}

        try:
            model = xgb.Booster()
            model.load_model(model_path)
            logger.debug("Model %s loaded from %s", model_id, model_path)
        except Exception as e:
            return None, {"error": f"Failed to load model: {e!s}", "model_id": model_id}
        else:
            return model, {}

    @classmethod
    def _load_scaler(
        cls,
        scaler_path: str,
        model_id: str,
        logger: logging.Logger) -> tuple[Any, dict[str, Any]]:
        """Load and return a scaler from file."""
        if not scaler_path or not Path(scaler_path).is_file():
            return None, {}
        try:
            scaler = joblib.load(scaler_path)
            logger.debug("Scaler for %s loaded from %s", model_id, scaler_path)
        except Exception as e:
            return None, {"error": f"Failed to load scaler: {e!s}", "model_id": model_id}
        else:
            return scaler, {}

    @classmethod
    def _raise_scaler_not_found(cls, scaler_path: str, logger: logging.Logger) -> None:
        """Raise FileNotFoundError for missing scaler file."""
        error_msg = f"Scaler file not found: {scaler_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from None

    @staticmethod
    def _prepare_features(
        feature_vector: np.ndarray[Any, Any],
        scaler: object | None,  # Could be StandardScaler, MinMaxScaler, etc.
        model_id: str,
        logger: logging.Logger) -> tuple[np.ndarray[Any, Any] | None, dict[str, Any]]:
        """Prepare and scale features for prediction."""
        if feature_vector.ndim != 1:
            error_msg = "Feature vector must be 1D for processing."
            return None, {"error": error_msg, "model_id": model_id}

        features_2d = feature_vector.reshape(1, -1)

        if scaler is None:
            logger.debug("No scaler used.")
            return features_2d, {}

        try:
            # Try to get the transform method safely with getattr
            transform_method = getattr(scaler, "transform", None)
            if transform_method is None:
                logger.warning(f"Scaler for model {model_id} doesn't have transform method")
                return features_2d, {}
            # Use the transform method we safely retrieved
            processed_features = transform_method(features_2d)
            logger.debug("Features scaled.")
        except Exception as e:
            return None, {"error": f"Error applying scaler: {e!s}", "model_id": model_id}
        else:
            return processed_features, {}

    @classmethod
    def _make_prediction(
        cls,
        model: Any,
        features: np.ndarray[Any, Any],
        feature_names: list[str],  # Renamed to match call site
        model_id: str,
        logger: logging.Logger) -> dict[str, Any]:
        """Make prediction using the loaded model."""
        if xgb is None:
            return {"error": "XGBoost library not available", "model_id": model_id}

        try:
            dmatrix = xgb.DMatrix(features, feature_names=feature_names)
            prediction_val = model.predict(dmatrix)

            if isinstance(prediction_val, np.ndarray) and prediction_val.size == 1:
                prediction_float = float(prediction_val.item())
            elif isinstance(prediction_val, float | np.floating):
                prediction_float = float(prediction_val)
            else:
                error_message = (
                    f"Unexpected prediction output format: {type(prediction_val)}, "
                    f"value: {prediction_val}"
                )
                logger.error(error_message)
                return {"error": error_message, "model_id": model_id}

            # For binary:logistic, the prediction_float is the probability of the positive class.
            confidence_float = prediction_float

            logger.debug(
                "Prediction for %s successful: %s, Confidence: %s",
                model_id,
                prediction_float,
                confidence_float)
            result = {
                "prediction": prediction_float,
                "confidence": confidence_float,
                "model_id": model_id,
            }
        except Exception as e:
            logger.exception("Prediction failed")
            return {"error": f"Prediction failed: {e!s}", "model_id": model_id}
        else:
            return result

    @classmethod
    def run_inference_in_process(
        cls,
        model_id: str,
        model_path: str,
        scaler_path: str | None, # Kept for interface compatibility, but not used.
        feature_vector: np.ndarray[Any, Any],  # Expects 1D pre-scaled feature vector
        model_feature_names: list[str],  # From model config
        _predictor_specific_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run inference in a separate process. Expects pre-scaled features.

        Args:
            model_id (str): Unique identifier for the model
            model_path (str): Path to the XGBoost model file
            scaler_path (Optional[str]): No longer used; features should be pre-scaled.
            feature_vector (np.ndarray[Any, Any]): 1D numpy array of pre-scaled feature values.
            model_feature_names (list[str]): List of feature names expected by the model.
            _predictor_specific_config (dict[str, Any] | None): Optional additional configuration.

        Returns:
        -------
            dict[str, Any]: Dictionary containing prediction results or error information
        """
        logger = logging.getLogger(__name__)
        logger.debug("Starting inference for model: %s", model_id)

        # Initialize result with model_id and default error state
        result: dict[str, Any] = {"model_id": model_id, "error": None}
        model = None
        # Scaler is no longer loaded or used in this static method.
        processed_features = None

        try:
            # 1. Load Model
            model, error = cls._load_model(model_path, model_id, logger)
            if error:
                result["error"] = error.get("error", "Failed to load model")
                return result

            # 2. Scaler loading is removed.
            if scaler_path:
                logger.info("scaler_path provided but will be ignored as features are expected pre-scaled.")


            # 3. Process features: Features are expected pre-scaled. Reshape only.
            # The _prepare_features method (which included scaling) is no longer used here.
            if feature_vector.ndim != 1:
                result["error"] = "Feature vector must be 1D."
                logger.error(result["error"])
                return result
            processed_features = feature_vector.reshape(1, -1)
            logger.debug("Using pre-scaled feature vector directly (after reshape).")

            # 4. Make prediction if no errors so far
            if processed_features is not None and model is not None: # Error condition for ndim already handled
                prediction_result = cls._make_prediction(
                    model=model,
                    features=processed_features,
                    feature_names=model_feature_names,
                    model_id=model_id,
                    logger=logger)
                # If _make_prediction returned an error, use it
                if "error" in prediction_result:
                    result["error"] = prediction_result["error"]
                else:
                    # If no error, return the prediction result directly
                    return prediction_result
            elif not result["error"]:
                if model is None:
                    result["error"] = "Model failed to load"
                # else block removed as it was unreachable

        except Exception as e:
            error_msg = f"Unexpected error during inference: {e!s}"
            logger.exception(error_msg)
            result["error"] = error_msg

        # If we reach here, there was an error or we need to return the error result
        return result
