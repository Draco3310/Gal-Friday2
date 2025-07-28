"""Interface definition for prediction model implementations."""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PredictorProtocol(Protocol):
    """Protocol for all prediction models."""

    model_path: str
    model_id: str
    config: dict[str, Any]
    scaler_path: str | None
    model: Any
    scaler: Any

    def load_assets(self) -> None:
        """Load the prediction model and any associated assets.

        Load the prediction model and any associated assets (e.g., scaler)
        from the specified paths.
        Sets self.model and self.scaler.
        """
        ...

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
        ...

    @property
    def expected_feature_names(self) -> list[str] | None:
        """Return the list of feature names the model expects, if applicable."""
        ...


# Backward compatibility alias
PredictorInterface = PredictorProtocol
