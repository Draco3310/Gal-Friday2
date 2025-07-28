"""Base class implementation for all predictor models.

This module provides a concrete base class that implements the PredictorInterface
protocol and provides common functionality for all predictor implementations.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any

import numpy as np


class BasePredictor(ABC):
    """Abstract base class for all predictor implementations.

    This class implements the PredictorInterface protocol and provides
    common functionality that all predictor implementations need.
    """

    def __init__(self, model_path: str, model_id: str, config: dict[str, Any] | None = None) -> None:
        """Initialize the base predictor.

        Args:
            model_path: Path to the model file
            model_id: Unique identifier for this model
            config: Optional configuration parameters
        """
        self.model_path = model_path
        self.model_id = model_id
        self.config = config or {}

        # Initialize attributes required by PredictorInterface protocol
        self.scaler_path: str | None = self.config.get("scaler_path")
        self.model: Any = None
        self.scaler: Any = None

        # Initialize logger
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        # Load assets after initialization
        self.load_assets()

    @abstractmethod
    def load_assets(self) -> None:
        """Load the prediction model and any associated assets.

        This method must be implemented by subclasses to load their
        specific model and scaler assets.
        """
        ...

    @abstractmethod
    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate predictions from the raw, ordered feature vector.

        This method must be implemented by subclasses to provide
        their specific prediction logic.

        Args:
            features: Raw, ordered 1D numpy array of feature values

        Returns:
            Prediction results as a numpy array
        """
        ...

    @property
    def expected_feature_names(self) -> list[str] | None:
        """Return the list of feature names the model expects, if applicable.

        Default implementation returns feature names from config.
        Subclasses can override this method for custom behavior.
        """
        return self.config.get("model_feature_names")

    def _validate_model_loaded(self) -> None:
        """Validate that the model has been properly loaded.

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError(f"Model not loaded for {self.model_id}")

    def _validate_features(self, features: np.ndarray[Any, Any]) -> None:
        """Validate input features.

        Args:
            features: Input features to validate

        Raises:
            ValueError: If features are invalid
        """
        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a numpy array")

        if features.size == 0:
            raise ValueError("Features array cannot be empty")
