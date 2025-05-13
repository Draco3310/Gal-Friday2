"""
Tests for the PredictionService ensemble functionality.

These tests verify that the PredictionService correctly handles ensemble prediction methods
by combining outputs from multiple models.
"""

import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gal_friday.core.events import FeatureEvent, PredictionEvent
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.prediction_service import PredictionService


class TestPredictionServiceEnsemble:
    """Test suite for PredictionService ensemble functionality."""

    @pytest.fixture
    def mock_pubsub(self):
        """Create a mock PubSubManager."""
        pubsub = MagicMock(spec=PubSubManager)
        pubsub.publish = AsyncMock()
        pubsub.subscribe = MagicMock()
        pubsub.unsubscribe = MagicMock()
        return pubsub

    @pytest.fixture
    def mock_logger(self):
        """Create a mock LoggerService."""
        logger = MagicMock(spec=LoggerService)
        return logger

    @pytest.fixture
    def mock_process_pool(self):
        """Create a mock ProcessPoolExecutor."""
        process_pool = MagicMock(spec=ProcessPoolExecutor)
        return process_pool

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            "prediction_service": {
                "models": [
                    {
                        "model_id": "xgb_model_v1",
                        "trading_pair": "BTC/USD",
                        "model_path": "models/xgb_model_v1.xgb",
                        "model_type": "xgboost",
                        "model_feature_names": ["feature1", "feature2", "feature3"],
                        "prediction_target": "price_movement_5min",
                    },
                    {
                        "model_id": "sklearn_model_v1",
                        "trading_pair": "BTC/USD",
                        "model_path": "models/sklearn_model_v1.joblib",
                        "model_type": "sklearn",
                        "model_feature_names": ["feature1", "feature2", "feature3"],
                        "prediction_target": "price_movement_5min",
                    },
                ],
                "ensembles": [
                    {
                        "ensemble_id": "avg_ensemble_v1",
                        "trading_pair": "BTC/USD",
                        "model_ids": ["xgb_model_v1", "sklearn_model_v1"],
                        "strategy": "average",
                        "output_model_id": "avg_ensemble_v1",
                        "prediction_target": "price_movement_5min",
                    },
                    {
                        "ensemble_id": "weighted_ensemble_v1",
                        "trading_pair": "BTC/USD",
                        "model_ids": ["xgb_model_v1", "sklearn_model_v1"],
                        "strategy": "weighted",
                        "weights": {"xgb_model_v1": 0.7, "sklearn_model_v1": 0.3},
                        "output_model_id": "weighted_ensemble_v1",
                        "prediction_target": "price_movement_5min",
                    },
                ],
            }
        }

    @pytest.fixture
    def sample_feature_event(self):
        """Create a sample FeatureEvent for testing."""
        return FeatureEvent(
            source_module="test_module",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            trading_pair="BTC/USD",
            exchange="kraken",
            timestamp_features_for=datetime.now(timezone.utc),
            features={"feature1": "0.5", "feature2": "1.2", "feature3": "-0.3"},
        )

    @pytest.fixture
    def prediction_service(self, sample_config, mock_pubsub, mock_process_pool, mock_logger):
        """Create a PredictionService instance for testing."""
        service = PredictionService(
            config=sample_config,
            pubsub_manager=mock_pubsub,
            process_pool_executor=mock_process_pool,
            logger_service=mock_logger,
        )
        return service

    @pytest.mark.asyncio
    async def test_find_matching_ensembles(self, prediction_service, sample_feature_event):
        """Test the _find_matching_ensembles method correctly identifies ensembles for an event."""
        matching_ensembles = prediction_service._find_matching_ensembles(sample_feature_event)

        assert len(matching_ensembles) == 2
        assert matching_ensembles[0]["ensemble_id"] == "avg_ensemble_v1"
        assert matching_ensembles[1]["ensemble_id"] == "weighted_ensemble_v1"

        # Test with non-matching trading pair
        non_matching_event = FeatureEvent(
            source_module="test_module",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            trading_pair="ETH/USD",  # Different trading pair
            exchange="kraken",
            timestamp_features_for=datetime.now(timezone.utc),
            features={"feature1": "0.5", "feature2": "1.2", "feature3": "-0.3"},
        )

        non_matching_ensembles = prediction_service._find_matching_ensembles(non_matching_event)
        assert len(non_matching_ensembles) == 0

    @pytest.mark.asyncio
    async def test_combine_predictions_average_strategy(self, prediction_service):
        """Test the _combine_predictions method with average strategy."""
        # Mock results from individual models
        results = [
            {"prediction": 0.7, "model_id": "xgb_model_v1"},
            {"prediction": 0.3, "model_id": "sklearn_model_v1"},
        ]

        # Model configs with weights
        model_configs = [
            ({"model_id": "xgb_model_v1"}, 1.0),
            ({"model_id": "sklearn_model_v1"}, 1.0),
        ]

        # Test average strategy
        combined = prediction_service._combine_predictions(results, model_configs, "average")
        assert combined == 0.5  # (0.7 + 0.3) / 2

    @pytest.mark.asyncio
    async def test_combine_predictions_weighted_strategy(self, prediction_service):
        """Test the _combine_predictions method with weighted strategy."""
        # Mock results from individual models
        results = [
            {"prediction": 0.8, "model_id": "xgb_model_v1"},
            {"prediction": 0.2, "model_id": "sklearn_model_v1"},
        ]

        # Model configs with weights
        model_configs = [
            ({"model_id": "xgb_model_v1"}, 0.7),
            ({"model_id": "sklearn_model_v1"}, 0.3),
        ]

        # Test weighted strategy
        combined = prediction_service._combine_predictions(results, model_configs, "weighted")

        # Expected: (0.8 * 0.7 + 0.2 * 0.3) / (0.7 + 0.3) = (0.56 + 0.06) / 1.0 = 0.62
        expected = (0.8 * 0.7 + 0.2 * 0.3) / (0.7 + 0.3)
        assert abs(combined - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_combine_predictions_with_errors(self, prediction_service):
        """Test the _combine_predictions method with some errors in results."""
        # Mock results where one model failed
        results = [
            {"prediction": 0.6, "model_id": "xgb_model_v1"},
            {"error": "Model failed to load"},
        ]

        # Model configs with weights
        model_configs = [
            ({"model_id": "xgb_model_v1"}, 1.0),
            ({"model_id": "sklearn_model_v1"}, 1.0),
        ]

        # Test average strategy with one error
        combined = prediction_service._combine_predictions(results, model_configs, "average")
        assert combined == 0.6  # Only one valid prediction

        # Test with all errors
        all_error_results = [{"error": "First model failed"}, {"error": "Second model failed"}]

        combined = prediction_service._combine_predictions(
            all_error_results, model_configs, "average"
        )
        assert combined is None

    @pytest.mark.asyncio
    async def test_run_ensemble_pipeline(
        self, prediction_service, sample_feature_event, mock_pubsub
    ):
        """Test the _run_ensemble_pipeline method end-to-end."""
        ensemble_config = {
            "ensemble_id": "test_ensemble",
            "trading_pair": "BTC/USD",
            "model_ids": ["xgb_model_v1", "sklearn_model_v1"],
            "strategy": "weighted",
            "weights": {"xgb_model_v1": 0.7, "sklearn_model_v1": 0.3},
            "output_model_id": "test_ensemble",
            "prediction_target": "price_movement_5min",
        }

        # Mock the run_in_executor to return successful model predictions
        async def mock_run_in_executor(executor, func, *args, **kwargs):
            if args[0].get("model_id") == "xgb_model_v1":
                return {"prediction": 0.8, "model_id": "xgb_model_v1"}
            else:
                return {"prediction": 0.2, "model_id": "sklearn_model_v1"}

        loop = MagicMock()
        loop.run_in_executor = AsyncMock(side_effect=mock_run_in_executor)

        with patch("asyncio.get_running_loop", return_value=loop):
            # Run the ensemble pipeline
            await prediction_service._run_ensemble_pipeline(sample_feature_event, ensemble_config)

            # Verify a prediction event was published
            assert mock_pubsub.publish.called

            # Check the published event
            published_event = mock_pubsub.publish.call_args[0][0]
            assert isinstance(published_event, PredictionEvent)
            assert published_event.model_id == "test_ensemble"
            assert published_event.trading_pair == "BTC/USD"
            assert published_event.prediction_target == "price_movement_5min"

            # Expected weighted result: (0.8 * 0.7 + 0.2 * 0.3) / (0.7 + 0.3) = 0.62
            expected = (0.8 * 0.7 + 0.2 * 0.3) / (0.7 + 0.3)
            assert abs(published_event.prediction_value - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_handle_feature_event_with_ensemble(
        self, prediction_service, sample_feature_event, mock_pubsub
    ):
        """Test that _handle_feature_event correctly processes ensembles."""
        # Mock _run_ensemble_pipeline
        prediction_service._run_ensemble_pipeline = AsyncMock()

        # Mock _run_single_model_pipeline
        prediction_service._run_single_model_pipeline = AsyncMock()

        # Call _handle_feature_event
        await prediction_service._handle_feature_event(sample_feature_event)

        # Verify _run_ensemble_pipeline was called twice (for both configured ensembles)
        assert prediction_service._run_ensemble_pipeline.call_count == 2

        # Verify _run_single_model_pipeline was not called (ensembles take priority)
        assert not prediction_service._run_single_model_pipeline.called

    @pytest.mark.asyncio
    async def test_ensemble_error_handling(
        self, prediction_service, sample_feature_event, mock_logger
    ):
        """Test error handling in ensemble processing."""
        ensemble_config = {
            "ensemble_id": "test_ensemble",
            "trading_pair": "BTC/USD",
            "model_ids": ["xgb_model_v1", "nonexistent_model"],
            "strategy": "average",
            "output_model_id": "test_ensemble",
            "prediction_target": "price_movement_5min",
        }

        # Mock to simulate a model not found in configuration
        prediction_service._find_matching_models = MagicMock(return_value=[])

        # Run the ensemble pipeline
        await prediction_service._run_ensemble_pipeline(sample_feature_event, ensemble_config)

        # Verify error was logged
        assert mock_logger.error.called
        error_message = mock_logger.error.call_args[0][0]
        assert "No valid models found for ensemble" in error_message
