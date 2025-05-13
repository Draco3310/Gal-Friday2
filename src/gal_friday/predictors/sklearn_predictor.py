"""Implementation of the PredictorInterface for scikit-learn models.

This module provides a sklearn-compatible predictor that can be used
with the prediction service to generate trade signals based on
scikit-learn models.
"""

from typing import Any, List, Optional

import joblib
import numpy as np

from ..interfaces.predictor_interface import PredictorInterface


class SklearnPredictor(PredictorInterface):
    """Implementation of PredictorInterface for scikit-learn models."""

    def load_model(self) -> Any:
        """Load a scikit-learn model from the specified path using joblib.

        Returns
        -------
            The loaded scikit-learn model

        Raises
        ------
            FileNotFoundError: If the model file does not exist
            Exception: If the model cannot be loaded for any other reason
        """
        try:
            model = joblib.load(self.model_path)

            # Try to get feature names from model if available
            if hasattr(model, "feature_names_in_"):
                self._expected_features = list(model.feature_names_in_)
            else:
                # Fallback to config
                self._expected_features = self.config.get("model_feature_names", [])

            return model
        except Exception as exc:
            # Let the exception propagate to be handled by the caller
            self.logger.error(f"Failed to load sklearn model: {exc}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the scikit-learn model.

        Args
        ----
            features: A numpy array of preprocessed features

        Returns
        -------
            Prediction results as a numpy array.
            For classifiers with predict_proba, returns class probabilities.
            For regressors or classifiers without predict_proba, returns predicted values.

        Raises
        ------
            ValueError: If features have wrong shape or contain invalid values
            TypeError: If model is not properly loaded
            Exception: If prediction fails for any other reason
        """
        try:
            # Verify model is loaded
            if self.model is None:
                raise TypeError(f"Model {self.model_id} is not loaded")

            # Check if this is a classifier with predict_proba
            if hasattr(self.model, "predict_proba"):
                # Return probabilities for all classes
                raw_predictions = self.model.predict_proba(features)
                predictions = np.asarray(raw_predictions, dtype=np.float32)
            elif hasattr(self.model, "predict"):
                # Otherwise use regular predict (regression or classifier without proba)
                raw_predictions = self.model.predict(features)
                predictions = np.asarray(raw_predictions, dtype=np.float32)

                # Ensure consistent output format for single-output regression
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
            else:
                raise TypeError(f"Model {self.model_id} has no predict or predict_proba method")

            return predictions

        except Exception as exc:
            # Propagate exceptions to be handled by caller
            self.logger.error(f"Prediction failed: {exc}")
            raise

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names the model expects.

        Returns
        -------
            List of feature names or None if not available
        """
        if not hasattr(self, "_expected_features") or self._expected_features is None:
            return None
        return self._expected_features
