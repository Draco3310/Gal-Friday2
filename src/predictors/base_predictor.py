from abc import ABC, abstractmethod
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BasePredictor(ABC):
    """
    Abstract base class for ML predictors.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get('model_path')
        self.model = None
        if self.model_path:
            try:
                self.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load model from {self.model_path}: {e}")
                # Don't raise here to allow partial initialization if needed,
                # but predict should handle model being None

    @abstractmethod
    def load_model(self, model_path: str):
        """
        Loads the model from disk.
        """
        pass

    @abstractmethod
    def predict(self, enriched_df: pd.DataFrame) -> float:
        """
        Returns a confidence score (0.0 to 1.0) for the last row of data.

        Args:
            enriched_df: DataFrame with computed features.

        Returns:
            float: Probability/Confidence of a BUY signal (0.0 - 1.0).
        """
        pass
