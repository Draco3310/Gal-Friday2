"""
Tests for ensemble prediction methods in the PredictionService.

These tests specifically focus on the combination strategies for ensemble predictions
and various edge cases in ensemble processing.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from gal_friday.prediction_service import PredictionService


class TestEnsembleCombinationStrategies:
    """Test suite for ensemble prediction combination strategies."""

    @pytest.fixture
    def prediction_service(self):
        """Create a minimal PredictionService instance for testing combination strategies."""
        # Create mocks for dependencies
        pubsub = MagicMock()
        pubsub.publish = AsyncMock()
        process_pool = MagicMock()
        logger = MagicMock()
        config = {
            "prediction_service": {
                "models": [{"model_id": "model1"}, {"model_id": "model2"}, {"model_id": "model3"}]
            }
        }

        # Create service instance
        service = PredictionService(
            config=config,
            pubsub_manager=pubsub,
            process_pool_executor=process_pool,
            logger_service=logger,
        )

        return service

    def test_combine_predictions_average_strategy(self, prediction_service):
        """Test the _combine_predictions method with average strategy."""
        # Setup
        results = [
            {"prediction": 0.8, "model_id": "model1"},
            {"prediction": 0.6, "model_id": "model2"},
            {"prediction": 0.4, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 1.0),
            ({"model_id": "model3"}, 1.0),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "average")

        # Verify
        assert combined == 0.6  # (0.8 + 0.6 + 0.4) / 3

    def test_combine_predictions_weighted_strategy(self, prediction_service):
        """Test the _combine_predictions method with weighted strategy."""
        # Setup
        results = [
            {"prediction": 0.9, "model_id": "model1"},
            {"prediction": 0.5, "model_id": "model2"},
            {"prediction": 0.1, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 0.5),
            ({"model_id": "model2"}, 0.3),
            ({"model_id": "model3"}, 0.2),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "weighted")

        # Verify
        expected = (0.9 * 0.5 + 0.5 * 0.3 + 0.1 * 0.2) / (0.5 + 0.3 + 0.2)
        assert abs(combined - expected) < 0.0001

    def test_combine_predictions_with_decimal_precision(self, prediction_service):
        """Test precision of weighted calculations using Decimal."""
        # Setup - Use values that would cause floating point errors
        results = [
            {"prediction": 0.1, "model_id": "model1"},
            {"prediction": 0.2, "model_id": "model2"},
            {"prediction": 0.3, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 0.7),
            ({"model_id": "model2"}, 0.2),
            ({"model_id": "model3"}, 0.1),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "weighted")

        # Calculate with Decimal for comparison (highest precision)
        p1, p2, p3 = Decimal("0.1"), Decimal("0.2"), Decimal("0.3")
        w1, w2, w3 = Decimal("0.7"), Decimal("0.2"), Decimal("0.1")
        expected = float((p1 * w1 + p2 * w2 + p3 * w3) / (w1 + w2 + w3))

        # Verify
        assert abs(combined - expected) < 0.0000001

    def test_combine_predictions_with_errors(self, prediction_service):
        """Test _combine_predictions when some models fail."""
        # Setup
        results = [
            {"prediction": 0.7, "model_id": "model1"},
            {"error": "Failed to load model"},
            {"prediction": 0.3, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 1.0),
            ({"model_id": "model3"}, 1.0),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "average")

        # Verify
        assert combined == 0.5  # (0.7 + 0.3) / 2

    def test_combine_predictions_all_errors(self, prediction_service):
        """Test _combine_predictions when all models fail."""
        # Setup
        results = [
            {"error": "Model failed to load"},
            {"error": "Feature preparation failed"},
            {"error": "Prediction failed"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 1.0),
            ({"model_id": "model3"}, 1.0),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "average")

        # Verify
        assert combined is None

    def test_combine_predictions_exception_handling(self, prediction_service):
        """Test _combine_predictions with exceptions in results."""
        # Setup
        results = [
            {"prediction": 0.8, "model_id": "model1"},
            RuntimeError("Model error"),
            {"prediction": 0.2, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 1.0),
            ({"model_id": "model3"}, 1.0),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "average")

        # Verify
        assert combined == 0.5  # (0.8 + 0.2) / 2

    def test_combine_predictions_with_outliers(self, prediction_service):
        """Test _combine_predictions with outlier values."""
        # Setup
        results = [
            {"prediction": 0.1, "model_id": "model1"},
            {"prediction": 0.9, "model_id": "model2"},
            {"prediction": 0.1, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 1.0),
            ({"model_id": "model3"}, 1.0),
        ]

        # Test average (should include outlier)
        combined_avg = prediction_service._combine_predictions(results, model_configs, "average")
        assert combined_avg == pytest.approx(0.3667, abs=0.001)  # (0.1 + 0.9 + 0.1) / 3

        # Test weighted with higher weight on outlier
        model_configs_weighted = [
            ({"model_id": "model1"}, 0.2),
            ({"model_id": "model2"}, 0.6),  # Higher weight on outlier
            ({"model_id": "model3"}, 0.2),
        ]

        combined_weighted = prediction_service._combine_predictions(
            results, model_configs_weighted, "weighted"
        )
        (0.1 * 0.2 + 0.9 * 0.6 + 0.1 * 0.2) / (0.2 + 0.6 + 0.2)
        assert combined_weighted == pytest.approx(0.58, abs=0.01)

    def test_combine_predictions_with_zero_weights(self, prediction_service):
        """Test _combine_predictions with some zero weights."""
        # Setup
        results = [
            {"prediction": 0.8, "model_id": "model1"},
            {"prediction": 0.4, "model_id": "model2"},
            {"prediction": 0.2, "model_id": "model3"},
        ]

        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 0.0),  # Zero weight
            ({"model_id": "model3"}, 1.0),
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "weighted")

        # Verify - model2 should be ignored due to zero weight
        (0.8 * 1.0 + 0.2 * 1.0) / (1.0 + 1.0)
        assert combined == pytest.approx(0.5, abs=0.0001)

    def test_combine_predictions_with_missing_results(self, prediction_service):
        """Test _combine_predictions with fewer results than configs."""
        # Setup
        results = [
            {"prediction": 0.7, "model_id": "model1"},
            {"prediction": 0.3, "model_id": "model2"},
        ]

        # More configs than results
        model_configs = [
            ({"model_id": "model1"}, 1.0),
            ({"model_id": "model2"}, 1.0),
            ({"model_id": "model3"}, 1.0),  # No matching result
        ]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "average")

        # Verify
        assert combined == 0.5  # (0.7 + 0.3) / 2


class TestEnsembleExtremeCases:
    """Test suite for extreme cases in ensemble predictions."""

    @pytest.fixture
    def prediction_service(self):
        """Create a PredictionService instance for testing extreme cases."""
        pubsub = MagicMock()
        pubsub.publish = AsyncMock()
        process_pool = MagicMock()
        logger = MagicMock()
        config = {
            "prediction_service": {"models": [{"model_id": "model1"}, {"model_id": "model2"}]}
        }

        service = PredictionService(
            config=config,
            pubsub_manager=pubsub,
            process_pool_executor=process_pool,
            logger_service=logger,
        )

        return service

    def test_single_model_ensemble(self, prediction_service):
        """Test ensemble with just one model."""
        # Setup
        results = [{"prediction": 0.75, "model_id": "model1"}]

        model_configs = [({"model_id": "model1"}, 1.0)]

        # Test average strategy
        avg_combined = prediction_service._combine_predictions(results, model_configs, "average")
        assert avg_combined == 0.75

        # Test weighted strategy
        weighted_combined = prediction_service._combine_predictions(
            results, model_configs, "weighted"
        )
        assert weighted_combined == 0.75

    def test_extreme_weight_disparity(self, prediction_service):
        """Test ensemble with extreme weight differences (e.g. 0.99 vs 0.01)."""
        # Setup
        results = [
            {"prediction": 1.0, "model_id": "model1"},
            {"prediction": 0.0, "model_id": "model2"},
        ]

        model_configs = [({"model_id": "model1"}, 0.99), ({"model_id": "model2"}, 0.01)]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "weighted")

        # Verify
        (1.0 * 0.99 + 0.0 * 0.01) / (0.99 + 0.01)
        assert combined == pytest.approx(0.99, abs=0.0001)

    def test_edge_case_predictions(self, prediction_service):
        """Test ensemble with edge case prediction values (0 and 1)."""
        # Setup
        results = [
            {"prediction": 0.0, "model_id": "model1"},
            {"prediction": 1.0, "model_id": "model2"},
        ]

        model_configs = [({"model_id": "model1"}, 1.0), ({"model_id": "model2"}, 1.0)]

        # Test
        combined = prediction_service._combine_predictions(results, model_configs, "average")

        # Verify
        assert combined == 0.5

    def test_identical_predictions(self, prediction_service):
        """Test ensemble when all models predict the same value."""
        # Setup
        results = [
            {"prediction": 0.42, "model_id": "model1"},
            {"prediction": 0.42, "model_id": "model2"},
        ]

        model_configs = [({"model_id": "model1"}, 0.7), ({"model_id": "model2"}, 0.3)]

        # Test average
        avg_combined = prediction_service._combine_predictions(results, model_configs, "average")
        assert avg_combined == 0.42

        # Test weighted - should still be 0.42 regardless of weights
        weighted_combined = prediction_service._combine_predictions(
            results, model_configs, "weighted"
        )
        assert weighted_combined == 0.42
