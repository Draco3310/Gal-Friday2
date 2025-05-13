"""
Integration tests for the prediction pipeline.

These tests verify that the prediction service works correctly with actual
predictor implementations and real configuration.
"""

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
import xgboost as xgb

from gal_friday.core.events import EventType, FeatureEvent
from gal_friday.logger_service import LoggerService
from gal_friday.prediction_service import PredictionService


@pytest.fixture
def temp_xgboost_model():
    """Create a temporary XGBoost model file for testing."""
    # Create a simple XGBoost model
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)
    dtrain = xgb.DMatrix(X, label=y, feature_names=["f1", "f2", "f3", "f4", "f5"])
    params = {"max_depth": 2, "eta": 1, "objective": "binary:logistic"}
    model = xgb.train(params, dtrain, num_boost_round=2)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".xgb", delete=False) as temp:
        model.save_model(temp.name)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_sklearn_model():
    """Create a temporary scikit-learn model file for testing."""
    from sklearn.ensemble import RandomForestClassifier

    # Create a simple random forest classifier
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)
    model = RandomForestClassifier(n_estimators=5, max_depth=2)
    model.fit(X, y)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp:
        joblib.dump(model, temp.name)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_scaler():
    """Create a temporary scaler file for testing."""
    from sklearn.preprocessing import StandardScaler

    # Create and fit a scaler
    X = np.random.rand(20, 5)
    scaler = StandardScaler()
    scaler.fit(X)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp:
        joblib.dump(scaler, temp.name)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def config_with_real_models(temp_xgboost_model, temp_sklearn_model, temp_scaler):
    """Create a configuration with paths to actual model files."""
    return {
        "prediction_service": {
            "process_pool_workers": 2,
            "models": [
                {
                    "model_id": "xgb_test",
                    "trading_pair": "XRP/USD",
                    "model_path": temp_xgboost_model,
                    "model_type": "xgboost",
                    "model_feature_names": ["f1", "f2", "f3", "f4", "f5"],
                    "prediction_target": "prob_price_up_0.1pct_5min",
                    "preprocessing": {"scaler_path": temp_scaler, "max_nan_percentage": 15.0},
                },
                {
                    "model_id": "rf_test",
                    "trading_pair": "XRP/USD",
                    "model_path": temp_sklearn_model,
                    "model_type": "sklearn",
                    "model_feature_names": ["f1", "f2", "f3", "f4", "f5"],
                    "prediction_target": "prob_price_up_0.1pct_5min",
                    "preprocessing": {"max_nan_percentage": 10.0},
                },
            ],
            "ensembles": [
                {
                    "ensemble_id": "test_ensemble",
                    "trading_pair": "XRP/USD",
                    "model_ids": ["xgb_test", "rf_test"],
                    "strategy": "weighted_average",
                    "weights": {"xgb_test": 0.6, "rf_test": 0.4},
                    "prediction_target": "prob_price_up_0.1pct_5min",
                }
            ],
        }
    }


@pytest.fixture
def mock_pubsub():
    """Create a mocked PubSubManager that captures published events."""

    class MockPubSub:
        def __init__(self):
            self.subscriptions = {}
            self.published_events = []

        def subscribe(self, event_type, callback):
            if event_type not in self.subscriptions:
                self.subscriptions[event_type] = []
            self.subscriptions[event_type].append(callback)

        def publish(self, event):
            self.published_events.append(event)

            # Call subscribers for this event type
            if event.event_type in self.subscriptions:
                for callback in self.subscriptions[event.event_type]:
                    callback(event)

    return MockPubSub()


@pytest.fixture
def sample_feature_event():
    """Create a sample feature event with all required features."""
    return FeatureEvent(
        event_id="test_integration_event",
        timestamp=datetime.now(),
        trading_pair="XRP/USD",
        exchange="kraken",
        features={
            "f1": 0.5,
            "f2": 0.6,
            "f3": 0.7,
            "f4": 0.8,
            "f5": 0.9,
        },
    )


def test_end_to_end_prediction_pipeline(
    config_with_real_models, mock_pubsub, sample_feature_event
):
    """Test the end-to-end prediction pipeline with real models."""
    # Create a mock config manager
    mock_config = MagicMock()
    mock_config.get.return_value = config_with_real_models["prediction_service"]

    # Create a mock logger
    mock_logger = MagicMock(spec=LoggerService)

    # Create a process pool
    process_pool = ProcessPoolExecutor(max_workers=2)

    try:
        # Create the prediction service
        prediction_service = PredictionService(
            config=mock_config, pubsub=mock_pubsub, logger=mock_logger, process_pool=process_pool
        )

        # Process a feature event
        prediction_service.handle_feature_event(sample_feature_event)

        # Check for prediction events (may need to wait briefly)
        import time

        time.sleep(1)  # Give time for async processing

        # Filter prediction events from published events
        prediction_events = [
            event
            for event in mock_pubsub.published_events
            if event.event_type == EventType.PREDICTION_GENERATED
        ]

        # Verify we got predictions
        assert len(prediction_events) >= 1

        # Check individual model predictions
        individual_predictions = [
            event
            for event in prediction_events
            if hasattr(event, "model_id") and event.model_id in ["xgb_test", "rf_test"]
        ]
        assert len(individual_predictions) >= 2

        # Check for ensemble prediction
        ensemble_predictions = [
            event
            for event in prediction_events
            if hasattr(event, "model_id") and event.model_id == "test_ensemble"
        ]
        assert len(ensemble_predictions) >= 1

        # Verify prediction format
        for event in prediction_events:
            assert hasattr(event, "prediction_value")
            assert isinstance(event.prediction_value, float)
            assert 0 <= event.prediction_value <= 1  # Binary classification probability

    finally:
        # Clean up
        process_pool.shutdown()


def test_feature_quality_checks_integration(config_with_real_models, mock_pubsub):
    """Test that feature quality checks work in integration."""
    # Create a mock config manager
    mock_config = MagicMock()
    mock_config.get.return_value = config_with_real_models["prediction_service"]

    # Create a mock logger
    mock_logger = MagicMock(spec=LoggerService)

    # Create a process pool
    process_pool = ProcessPoolExecutor(max_workers=2)

    try:
        # Create the prediction service
        prediction_service = PredictionService(
            config=mock_config, pubsub=mock_pubsub, logger=mock_logger, process_pool=process_pool
        )

        # Create a feature event with NaN values (above threshold)
        bad_feature_event = FeatureEvent(
            event_id="test_bad_features",
            timestamp=datetime.now(),
            trading_pair="XRP/USD",
            exchange="kraken",
            features={
                "f1": 0.5,
                "f2": None,  # NaN
                "f3": None,  # NaN
                "f4": 0.8,
                "f5": 0.9,
            },
        )

        # Process the feature event
        prediction_service.handle_feature_event(bad_feature_event)

        # Wait briefly for processing
        import time

        time.sleep(1)

        # Check logs for quality warnings
        assert any(
            "feature quality" in str(args) for args, kwargs in mock_logger.warning.call_args_list
        )

    finally:
        # Clean up
        process_pool.shutdown()
