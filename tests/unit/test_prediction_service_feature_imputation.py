"""Unit tests for PredictionService feature imputation logic."""

from unittest.mock import MagicMock

import numpy as np

from gal_friday.prediction_service import PredictionService

NEUTRAL_RSI = 50.0


class TestPrepareFeaturesForModelImputation:
    """Tests for the `_prepare_features_for_model` helper."""

    def setup_method(self) -> None:
        """Create a minimal PredictionService instance for testing."""
        self.service = PredictionService(
            config={"prediction_service": {"models": []}},
            pubsub_manager=MagicMock(),
            process_pool_executor=MagicMock(),
            logger_service=MagicMock(),
        )

    def test_nan_value_imputed(self) -> None:
        """Ensure NaN inputs are replaced with contextual defaults."""
        features = {"rsi_14_default": np.nan}
        expected = ["rsi_14_default"]
        result = self.service._prepare_features_for_model(features, expected)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result[0] == NEUTRAL_RSI
