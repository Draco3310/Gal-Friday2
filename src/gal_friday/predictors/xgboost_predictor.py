"""Implementation of the PredictorInterface for XGBoost models.

This module provides an XGBoost-compatible predictor that can be used
with the prediction service to generate trade signals based on
XGBoost models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib # For loading scalers
import numpy as np
import xgboost as xgb

from ..interfaces.predictor_interface import PredictorInterface


class XGBoostPredictor(PredictorInterface):
    """Implementation of PredictorInterface for XGBoost models."""

    def __init__(self, model_path: str, model_id: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the XGBoost predictor.

        Args
        ----
            model_path: Path to the XGBoost model file
            model_id: Unique identifier for this model
            config: Additional configuration parameters for this model.
                    Expected to contain 'scaler_path': Optional[str] and
                    'model_feature_names': List[str].
        """
        # _expected_features must be set before super().__init__ if load_assets uses it.
        # However, we'll fetch it from self.config inside load_assets now.
        super().__init__(model_path, model_id, config)

    def load_assets(self) -> None:
        """Load an XGBoost model and its associated scaler from the specified paths.

        Sets self.model and self.scaler.

        Raises
        ------
            xgb.core.XGBoostError: If the model cannot be loaded.
            FileNotFoundError: If the model or scaler file does not exist.
            Exception: If the scaler cannot be loaded.
        """
        self.logger.info(f"Loading assets for model: {self.model_id}")
        # Load Model
        try:
            if not Path(self.model_path).exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            self.logger.info(f"XGBoost model loaded successfully from {self.model_path}")

        except xgb.core.XGBoostError as e:
            self.logger.exception(f"Failed to load XGBoost model from {self.model_path}")
            raise e # Re-raise the specific exception
        except FileNotFoundError as e:
            raise e # Re-raise

        # Load Scaler
        if self.scaler_path:
            try:
                if not Path(self.scaler_path).exists():
                    self.logger.error(f"Scaler file not found: {self.scaler_path}")
                    raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info(f"Scaler loaded successfully from {self.scaler_path}")
            except FileNotFoundError as e:
                raise e # Re-raise
            except Exception as e:
                self.logger.exception(f"Failed to load scaler from {self.scaler_path}")
                raise e # Re-raise a generic exception or a custom one
        else:
            self.logger.info("No scaler_path provided. Proceeding without a scaler.")
            self.scaler = None


    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the XGBoost model.

        Args
        ----
            features: A 1D numpy array of raw, ordered feature values.

        Returns
        -------
            Prediction results as a numpy array (usually probabilities for binary:logistic)

        Raises
        ------
            ValueError: If features have wrong shape or contain invalid values.
            TypeError: If model is not properly loaded.
            xgb.core.XGBoostError: If prediction fails.
        """
        if self.model is None:
            self.logger.error("Model not loaded. Cannot predict.")
            raise TypeError("XGBoost model is not loaded.")

        if features.ndim != 1:
            self.logger.error(f"Input features must be a 1D array, got {features.ndim}D.")
            raise ValueError(f"Input features must be a 1D array, got {features.ndim}D.")

        features_2d = features.reshape(1, -1)

        if self.scaler:
            try:
                processed_features = self.scaler.transform(features_2d)
                self.logger.debug("Features scaled successfully.")
            except Exception as e:
                self.logger.exception("Error applying scaler transform.")
                raise ValueError(f"Error during scaling: {e!s}") from e
        else:
            processed_features = features_2d
            self.logger.debug("No scaler found or used. Using raw features.")

        # Ensure expected_feature_names are available for DMatrix
        model_feature_names = self.expected_feature_names
        if not model_feature_names:
            self.logger.warning(
                "expected_feature_names not available. DMatrix will be created without feature names."
            )

        try:
            dmatrix = xgb.DMatrix(processed_features, feature_names=model_feature_names)
            raw_predictions = self.model.predict(dmatrix)
            # For binary:logistic, predict returns probabilities of the positive class.
            # This is typically a 1D array of shape (1,) or (n_samples,)
            # Ensure it's a simple float or 1D array.
            if isinstance(raw_predictions, np.ndarray):
                # If it's like array([0.7]), get 0.7. If it's already a scalar float, it's fine.
                return raw_predictions.astype(np.float32) # Ensure correct type
            return np.array([raw_predictions], dtype=np.float32) # Convert scalar to 1-element array

        except xgb.core.XGBoostError as e:
            self.logger.exception("XGBoost prediction failed.")
            raise e
        except Exception as e:
            self.logger.exception("An unexpected error occurred during prediction.")
            raise ValueError(f"Unexpected error during prediction: {e!s}") from e

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names the model expects from config."""
        return self.config.get("model_feature_names")

    @classmethod
    def run_inference_in_process(
        cls,
        model_id: str,
        model_path: str,
        scaler_path: Optional[str],
        feature_vector: np.ndarray, # Expects 1D raw feature vector
        model_feature_names: List[str], # From model config
        predictor_specific_config: Dict[str, Any] # Full model config for this predictor
    ) -> Dict[str, Any]:
        """
        Load model and scaler, preprocess, predict. Executed in a separate process.
        """
        logger = logging.getLogger(f"{cls.__name__}:{model_id}.run_inference_in_process")
        logger.debug(f"Starting inference for model {model_id} in separate process.")

        try:
            # 1. Load Model
            if not Path(model_path).exists():
                return {"error": f"Model file not found: {model_path}", "model_id": model_id}
            model = xgb.Booster()
            model.load_model(model_path)
            logger.debug(f"Model {model_id} loaded from {model_path}")

            # 2. Load Scaler
            scaler = None
            if scaler_path:
                if not Path(scaler_path).exists():
                    return {"error": f"Scaler file not found: {scaler_path}", "model_id": model_id}
                try:
                    scaler = joblib.load(scaler_path)
                    logger.debug(f"Scaler for {model_id} loaded from {scaler_path}")
                except Exception as e:
                    return {"error": f"Failed to load scaler: {e!s}", "model_id": model_id}

            # 3. Prepare features (reshape and scale)
            if feature_vector.ndim != 1:
                 return {"error": "Feature vector must be 1D for processing.", "model_id": model_id}
            features_2d = feature_vector.reshape(1, -1)

            if scaler:
                try:
                    processed_features = scaler.transform(features_2d)
                    logger.debug("Features scaled.")
                except Exception as e:
                    return {"error": f"Error applying scaler: {e!s}", "model_id": model_id}
            else:
                processed_features = features_2d
                logger.debug("No scaler used.")

            # 4. Predict
            # model_feature_names is passed directly, taken from this model's config
            dmatrix = xgb.DMatrix(processed_features, feature_names=model_feature_names)
            prediction = model.predict(dmatrix)

            prediction_float: float
            if isinstance(prediction, np.ndarray) and prediction.size == 1:
                prediction_float = float(prediction.item())
            elif isinstance(prediction, (float, np.floating)): # Handles scalar output
                prediction_float = float(prediction)
            else:
                return {
                    "error": f"Unexpected prediction output format: {type(prediction)}, value: {prediction}",
                    "model_id": model_id
                }
            logger.debug(f"Prediction for {model_id} successful: {prediction_float}")
            return {"prediction": prediction_float, "model_id": model_id}

        except FileNotFoundError as e: # Should be caught by Path.exists earlier
            logger.error(f"File not found error during inference for {model_id}: {e!s}")
            return {"error": f"File not found during inference: {e!s}", "model_id": model_id}
        except xgb.core.XGBoostError as xgb_err:
            logger.error(f"XGBoost error during inference for {model_id}: {xgb_err}")
            return {"error": f"XGBoost error: {xgb_err}", "model_id": model_id}
        except Exception as e:
            logger.error(f"Generic error during inference for {model_id}: {e!s}", exc_info=True)
            return {"error": f"Inference failed: {e!s}", "model_id": model_id}

# Example of how it might be used (for illustration, not part of the class)
# if __name__ == '__main__':
#     # This is a placeholder for where actual configuration would come from
#     # For testing, you would need a dummy model and scaler.
#     pass
