"""Implementation of the PredictorInterface for scikit-learn models.

This module provides a sklearn-compatible predictor that can be used
with the prediction service to generate trade signals based on
scikit-learn models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ..interfaces.predictor_interface import PredictorInterface


class SKLearnPredictor(PredictorInterface):
    """Implementation of PredictorInterface for scikit-learn models."""

    def __init__(
        self, model_path: str, model_id: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the SKLearn predictor.

        Args
        ----
            model_path: Path to the scikit-learn model file (joblib/pickle).
            model_id: Unique identifier for this model.
            config: Additional configuration parameters for this model.
                    Expected to contain 'scaler_path': Optional[str] and
                    'model_feature_names': List[str] (if not in model.feature_names_in_).
        """
        super().__init__(model_path, model_id, config)

    def load_assets(self) -> None:
        """Load a scikit-learn model and its associated scaler from specified paths.

        Sets self.model and self.scaler.

        Raises
        ------
            FileNotFoundError: If the model or scaler file does not exist.
            Exception: If the model or scaler cannot be loaded.
        """
        self.logger.info(f"Loading assets for model: {self.model_id}")
        # Load Model
        try:
            if not Path(self.model_path).exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Scikit-learn model loaded successfully from {self.model_path}")
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            self.logger.exception(f"Failed to load scikit-learn model from {self.model_path}")
            raise e

        # Load Scaler
        if self.scaler_path:
            try:
                if not Path(self.scaler_path).exists():
                    self.logger.error(f"Scaler file not found: {self.scaler_path}")
                    raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info(f"Scaler loaded successfully from {self.scaler_path}")
            except FileNotFoundError as e:
                raise e
            except Exception as e:
                self.logger.exception(f"Failed to load scaler from {self.scaler_path}")
                raise e
        else:
            self.logger.info("No scaler_path provided. Proceeding without a scaler.")
            self.scaler = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the scikit-learn model.

        Args
        ----
            features: A 1D numpy array of raw, ordered feature values.

        Returns
        -------
            Prediction results as a numpy array.
            For classifiers with predict_proba, returns class probabilities (typically of the positive class).
            For regressors or classifiers without predict_proba, returns predicted values.

        Raises
        ------
            ValueError: If features have wrong shape or contain invalid values.
            TypeError: If model is not properly loaded or is of an unexpected type.
            Exception: If prediction fails for any other reason.
        """
        if self.model is None:
            self.logger.error("Model not loaded. Cannot predict.")
            raise TypeError("Scikit-learn model is not loaded.")

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

        try:
            # For classifiers, prefer predict_proba and return probability of the positive class (class 1)
            if hasattr(self.model, "predict_proba"):
                # predict_proba returns shape (n_samples, n_classes)
                # We typically want the probability of the positive class (index 1)
                all_class_probabilities = self.model.predict_proba(processed_features)
                # Assuming binary classification, take prob of class 1
                prediction_output = all_class_probabilities[:, 1]
            elif hasattr(self.model, "predict"):
                # For regressors or classifiers without predict_proba
                prediction_output = self.model.predict(processed_features)
            else:
                self.logger.error(
                    "Loaded model object does not have a 'predict' or 'predict_proba' method."
                )
                raise TypeError("Model does not support predict or predict_proba methods.")

            # Ensure output is a 1D numpy array of float32 for consistency
            return np.asarray(prediction_output, dtype=np.float32).flatten()

        except Exception as e:
            self.logger.exception("Scikit-learn prediction failed.")
            raise ValueError(f"Prediction error: {e!s}") from e

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names the model expects.
        Tries to get it from model.feature_names_in_ first, then from config.
        """
        if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            return list(self.model.feature_names_in_)
        return self.config.get("model_feature_names")

    @classmethod
    def run_inference_in_process(
        cls,
        model_id: str,
        model_path: str,
        scaler_path: Optional[str],
        feature_vector: np.ndarray,  # Expects 1D raw feature vector
        model_feature_names: List[
            str
        ],  # From model config, might be used if model doesn't store them
        predictor_specific_config: Dict[str, Any],  # Full model config
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
            model = joblib.load(model_path)
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
            prediction_float: float
            confidence_float: Optional[float] = None

            if hasattr(model, "predict_proba"):
                # For classifiers that support probability
                all_class_probabilities = model.predict_proba(
                    processed_features
                )  # Shape: (n_samples, n_classes)
                if all_class_probabilities.shape[1] >= 2:
                    # Assuming binary classification, prediction is prob of positive class (index 1)
                    prediction_float = float(all_class_probabilities[0, 1])
                    confidence_float = prediction_float  # Use this probability as confidence
                elif (
                    all_class_probabilities.shape[1] == 1
                ):  # Single class output (unusual for proba)
                    prediction_float = float(all_class_probabilities[0, 0])
                    confidence_float = prediction_float  # Or abs(prediction_float - 0.5) * 2 for nearness to 0 or 1
                else:  # No classes
                    return {
                        "error": "predict_proba returned no class probabilities.",
                        "model_id": model_id,
                    }
            elif hasattr(model, "predict"):
                # For regressors or classifiers without predict_proba
                prediction_output = model.predict(processed_features)
                if isinstance(prediction_output, np.ndarray) and prediction_output.size == 1:
                    prediction_float = float(prediction_output.item())
                elif isinstance(prediction_output, (float, int, np.floating, np.integer)):
                    prediction_float = float(prediction_output)
                else:
                    return {
                        "error": f"Unsupported prediction output type from model.predict: {type(prediction_output)}",
                        "model_id": model_id,
                    }
                # Confidence is None for models without predict_proba, unless a specific method is defined
                confidence_float = None
            else:
                return {
                    "error": "Model has no predict or predict_proba method.",
                    "model_id": model_id,
                }

            logger.debug(
                f"Prediction for {model_id} successful: {prediction_float}, Confidence: {confidence_float}"
            )
            return {
                "prediction": prediction_float,
                "confidence": confidence_float,
                "model_id": model_id,
            }

        except FileNotFoundError as e:
            logger.error(f"File not found error during inference for {model_id}: {e!s}")
            return {"error": f"File not found: {e!s}", "model_id": model_id}
        except Exception as e:
            logger.error(f"Generic error during inference for {model_id}: {e!s}", exc_info=True)
            return {"error": f"Inference failed: {e!s}", "model_id": model_id}
