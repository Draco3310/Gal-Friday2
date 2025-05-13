"""Interface definition for prediction model implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class PredictorInterface(ABC):
    """Abstract Base Class for all prediction models."""

    def __init__(self, model_path: str, model_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the model file
            model_id: Unique identifier for this model
            config: Additional configuration parameters for this model
        """
        self.model_path = model_path
        self.model_id = model_id
        self.config = config or {}
        self.model: Any = None  # Explicitly typed as Any
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{self.model_id}")
        # Load the model during initialization
        self.model = self.load_model()

    @abstractmethod
    def load_model(self) -> Any:
        """Load the prediction model from the specified path."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions from the preprocessed features.

        Args:
            features: Preprocessed feature vector or matrix

        Returns:
            Prediction results as a numpy array

        Raises:
            ValueError: If features have wrong shape or contain invalid values
            TypeError: If model is not properly loaded
            Exception: If prediction fails for any other reason
        """

    @property
    @abstractmethod
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names the model expects, if applicable."""
