"""Implementation of the PredictorInterface for XGBoost models.

This module provides an XGBoost-compatible predictor that can be used
with the prediction service to generate trade signals based on
XGBoost models.
"""

import os
from typing import Any, List, Optional

import numpy as np
import xgboost as xgb

from ..interfaces.predictor_interface import PredictorInterface


class XGBoostPredictor(PredictorInterface):
    """Implementation of PredictorInterface for XGBoost models."""

    def __init__(self, model_path: str, model_id: str, config: Optional[dict] = None):
        """
        Initialize the XGBoost predictor.

        Args:
            model_path: Path to the XGBoost model file
            model_id: Unique identifier for this model
            config: Additional configuration parameters for this model
        """
        # Initialize _expected_features before loading the model
        self._expected_features: List[str] = []
        # Call parent initializer which will load the model
        super().__init__(model_path, model_id, config)

    def load_model(self) -> Any:
        """Load an XGBoost model from the specified path.

        Returns:
            The loaded XGBoost Booster model

        Raises:
            xgb.core.XGBoostError: If the model cannot be loaded
            FileNotFoundError: If the model file does not exist
        """
        try:
            # First check if file exists to ensure FileNotFoundError is raised
            if not os.path.exists(self.model_path):
                self.logger.error(f"Model file not found: {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            bst = xgb.Booster()
            bst.load_model(self.model_path)

            # Store expected features from config or model attributes
            self._expected_features = self.config.get("feature_names", [])
            # If model_feature_names exists in config,
            # use that instead (for consistency with sklearn)
            if "model_feature_names" in self.config:
                self._expected_features = self.config["model_feature_names"]

            return bst
        except (xgb.core.XGBoostError, FileNotFoundError) as exc:
            # Let the exception propagate to be handled by the caller
            self.logger.error(f"Failed to load XGBoost model: {exc}")
            raise

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions using the XGBoost model.

        Args:
            features: A numpy array of preprocessed features

        Returns:
            Prediction results as a numpy array

        Raises:
            ValueError: If features have wrong shape or contains invalid values
            TypeError: If model is not properly loaded
            xgb.core.XGBoostError: If prediction fails
        """
        try:
            # Verify model is loaded
            if self.model is None:
                raise TypeError(f"Model {self.model_id} is not loaded")

            # Create DMatrix with feature names if available
            if self._expected_features and len(self._expected_features) == features.shape[1]:
                dmatrix = xgb.DMatrix(features, feature_names=self._expected_features)
            else:
                # If shape doesn't match or no feature names, use raw data
                dmatrix = xgb.DMatrix(features)

            # Run prediction
            raw_predictions = self.model.predict(dmatrix)

            # Ensure predictions are a numpy ndarray
            predictions = np.asarray(raw_predictions, dtype=np.float32)

            # Handle different prediction outputs
            if (
                hasattr(self.model, "objective")
                and self.model.objective
                and isinstance(self.model.objective, str)
            ):
                if self.model.objective.startswith("binary:"):
                    # Binary classification: Ensure consistent output shape
                    if predictions.ndim == 1:
                        # For binary classification with single probability output,
                        # reshape to [class0_prob, class1_prob] format
                        return np.vstack([1 - predictions, predictions]).T
                    else:
                        # If already in multi-column format, return as is
                        return predictions

            # For regression or other outputs, ensure it's a numpy array
            return predictions

        except Exception as exc:
            # Propagate exceptions to be handled by caller
            self.logger.error(f"Prediction failed: {exc}")
            raise

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names the model expects.

        Returns:
            List of feature names or None if not available
        """
        if not hasattr(self, "_expected_features"):
            return None
        if self._expected_features is None or len(self._expected_features) == 0:
            return None
        # Explicitly cast to List[str] to satisfy mypy
        return list(map(str, self._expected_features))
