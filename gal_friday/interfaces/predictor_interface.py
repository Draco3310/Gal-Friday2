"""Interface definition for prediction model implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PredictorInterface(ABC):
    """Abstract Base Class for all prediction models."""

    def __init__(
        self,
        model_path: str,
        model_id: str,
        config: dict[str, Any] | None = None) -> None:
        """Initialize the predictor.

        Args:
        ----
            model_path: Path to the model file
            model_id: Unique identifier for this model
            config: Additional configuration parameters for this model.
                    Expected to contain 'scaler_path': Optional[str] if a scaler is used.
        """
        self.model_path = model_path
        self.model_id = model_id
        self.config = config or {}
        self.scaler_path: str | None = self.config.get("scaler_path")
        self.model: Any = None
        self.scaler: Any = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}:{self.model_id}")
        # Load the model and scaler during initialization
        self.load_assets()

    @abstractmethod
    def load_assets(self) -> None:
        """Load the prediction model and any associated assets.

        Load the prediction model and any associated assets (e.g., scaler)
        from the specified paths.
        Sets self.model and self.scaler.
        """

    @abstractmethod
    def predict(self, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Generate predictions from the raw, ordered feature vector.

        Implementations should handle any necessary internal preprocessing
        (e.g., scaling with self.scaler) before actual model inference.

        Args:
        ----
            features: Raw, ordered 1D numpy array of feature values.

        Returns:
        -------
            Prediction results as a numpy array

        Raises:
        ------
            ValueError: If features have wrong shape or contain invalid values
            TypeError: If model is not properly loaded
            Exception: If prediction fails for any other reason
        """

    @property
    @abstractmethod
    def expected_feature_names(self) -> list[str] | None:
        """Return the list[Any] of feature names the model expects, if applicable."""