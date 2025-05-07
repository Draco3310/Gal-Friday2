"""
Tests for the prediction_service module.
"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from datetime import datetime

from gal_friday.prediction_service import PredictionService
from gal_friday.event_bus import EventBus
from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import MarketDataEvent, PredictionEvent


@pytest.fixture
def prediction_config():
    """Fixture providing prediction service configuration."""
    return {
        "prediction": {
            "models": {
                "price_direction": {
                    "enabled": True,
                    "type": "ensemble",
                    "model_path": "models/price_direction_model.pkl",
                    "prediction_horizon": "1h",
                    "features": [
                        "rsi_14",
                        "macd",
                        "bbands_width",
                        "volume_change",
                        "price_momentum"],
                    "retrain_interval_days": 7},
                "volatility": {
                    "enabled": True,
                    "type": "garch",
                    "model_path": "models/volatility_model.pkl",
                    "prediction_horizon": "1d",
                    "features": [
                            "historical_volatility",
                            "volume",
                            "returns"],
                    "retrain_interval_days": 14}},
            "feature_generation": {
                "lookback_periods": [
                    14,
                    30,
                    50],
                "indicators": [
                    "rsi",
                    "macd",
                    "bbands",
                    "atr",
                    "obv"]}}}


def test_prediction_service_initialization(prediction_config, event_bus):
    """Test that the PredictionService initializes correctly."""
    config = ConfigManager(config_dict=prediction_config)

    with patch("gal_friday.prediction_service.load_model") as mock_load_model:
        # Mock the model loading
        mock_load_model.return_value = MagicMock()

        # Initialize service
        prediction_service = PredictionService(config, event_bus)

        assert prediction_service is not None
        assert len(prediction_service.models) == 2
        assert "price_direction" in prediction_service.models
        assert "volatility" in prediction_service.models
        assert prediction_service.model_configs["price_direction"]["prediction_horizon"] == "1h"
        assert prediction_service.model_configs["volatility"]["prediction_horizon"] == "1d"


@patch("gal_friday.prediction_service.load_model")
@patch("gal_friday.feature_engine.FeatureEngine")
def test_prediction_service_feature_calculation(
        mock_feature_engine,
        mock_load_model,
        prediction_config,
        event_bus,
        mock_ohlcv_data):
    """Test feature calculation for prediction."""
    config = ConfigManager(config_dict=prediction_config)

    # Mock the feature engine
    mock_feature_engine_instance = MagicMock()
    mock_feature_engine.return_value = mock_feature_engine_instance

    # Create sample features dataframe
    features_df = pd.DataFrame({
        'rsi_14': np.random.random(100),
        'macd': np.random.random(100) - 0.5,
        'bbands_width': np.random.random(100) * 0.1,
        'volume_change': np.random.random(100) * 0.2 - 0.1,
        'price_momentum': np.random.random(100) * 0.15 - 0.05,
        'historical_volatility': np.random.random(100) * 0.2,
        'volume': np.random.random(100) * 1000 + 500,
        'returns': np.random.random(100) * 0.1 - 0.05
    }, index=mock_ohlcv_data["BTC/USD"].index)

    mock_feature_engine_instance.calculate_features.return_value = features_df

    # Mock the model loading
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(
        [1, 0, 1, 1, 0])  # Sample predictions
    mock_load_model.return_value = mock_model

    # Initialize service
    prediction_service = PredictionService(config, event_bus)
    prediction_service.feature_engine = mock_feature_engine_instance

    # Add market data
    prediction_service.market_data["BTC/USD"] = mock_ohlcv_data["BTC/USD"].copy()

    # Calculate features for BTC/USD
    features = prediction_service._calculate_features("BTC/USD")

    # Verify feature calculation
    mock_feature_engine_instance.calculate_features.assert_called_once()
    assert features is not None
    assert set(
        features.columns) >= set(
        prediction_config["prediction"]["models"]["price_direction"]["features"])


@patch("gal_friday.prediction_service.load_model")
def test_prediction_service_make_predictions(
        mock_load_model, prediction_config, event_bus):
    """Test making predictions with models."""
    config = ConfigManager(config_dict=prediction_config)

    # Mock the price direction model
    price_model = MagicMock()
    price_model.predict.return_value = np.array([1])  # Predict price up
    price_model.predict_proba.return_value = np.array(
        [[0.3, 0.7]])  # 70% confidence

    # Mock the volatility model
    vol_model = MagicMock()
    vol_model.predict.return_value = np.array(
        [0.15])  # 15% predicted volatility

    # Set up model loading to return our mocks
    def mock_load_model_func(path):
        if "price_direction" in path:
            return price_model
        elif "volatility" in path:
            return vol_model
        else:
            return MagicMock()

    mock_load_model.side_effect = mock_load_model_func

    # Initialize service with mock event bus
    mock_event_bus = MagicMock()
    prediction_service = PredictionService(config, mock_event_bus)

    # Create sample features
    features = pd.DataFrame({
        'rsi_14': [70],
        'macd': [0.5],
        'bbands_width': [0.05],
        'volume_change': [0.1],
        'price_momentum': [0.08],
        'historical_volatility': [0.12],
        'volume': [1000],
        'returns': [0.02]
    })

    # Make predictions
    prediction_service._make_predictions("BTC/USD", features)

    # Verify models were called
    price_model.predict.assert_called_once()
    price_model.predict_proba.assert_called_once()
    vol_model.predict.assert_called_once()

    # Verify prediction events were published
    assert mock_event_bus.publish.call_count == 2

    # Extract published events
    published_events = [call.args[0]
                        for call in mock_event_bus.publish.call_args_list]

    # Verify price direction prediction event
    price_events = [
        e for e in published_events if e.prediction_type == "price_direction"]
    assert len(price_events) == 1
    assert price_events[0].symbol == "BTC/USD"
    assert price_events[0].value == 1  # Up
    assert price_events[0].confidence == 0.7

    # Verify volatility prediction event
    vol_events = [
        e for e in published_events if e.prediction_type == "volatility"]
    assert len(vol_events) == 1
    assert vol_events[0].symbol == "BTC/USD"
    assert vol_events[0].value == 0.15  # 15% volatility


@patch("gal_friday.prediction_service.load_model")
def test_prediction_service_handle_market_data(
        mock_load_model, prediction_config, event_bus):
    """Test handling market data events."""
    config = ConfigManager(config_dict=prediction_config)

    # Mock the model loading
    mock_load_model.return_value = MagicMock()

    # Mock methods in the prediction service
    with patch.object(PredictionService, '_calculate_features') as mock_calc_features:
        with patch.object(PredictionService, '_make_predictions') as mock_make_predictions:
            # Set up mock return values
            mock_calc_features.return_value = pd.DataFrame({
                'feature1': [1.0],
                'feature2': [2.0]
            })

            # Initialize service
            prediction_service = PredictionService(config, event_bus)

            # Create a market data event
            market_data = MarketDataEvent(
                timestamp=datetime.now(),
                symbol="BTC/USD",
                price=50000.0
            )

            # Process the market data event
            prediction_service.handle_market_data(market_data)

            # Verify data was stored
            assert "BTC/USD" in prediction_service.latest_prices
            assert prediction_service.latest_prices["BTC/USD"] == 50000.0

            # Wait a bit to allow for potential async processing
            # Verify feature calculation and prediction were called if appropriate
            # This depends on your implementation details


@patch("gal_friday.prediction_service.load_model")
@patch("gal_friday.prediction_service.save_model")
def test_prediction_service_retrain_models(
        mock_save_model,
        mock_load_model,
        prediction_config,
        event_bus,
        mock_ohlcv_data):
    """Test model retraining."""
    config = ConfigManager(config_dict=prediction_config)

    # Mock the model loading and sklearn
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    with patch("sklearn.ensemble.RandomForestClassifier") as mock_rf:
        with patch("sklearn.model_selection.train_test_split") as mock_split:
            # Set up mock sklearn
            mock_rf_instance = MagicMock()
            mock_rf.return_value = mock_rf_instance
            mock_split.return_value = (
                np.array([[1, 2], [3, 4]]),  # X_train
                np.array([[5, 6], [7, 8]]),  # X_test
                np.array([0, 1]),            # y_train
                np.array([1, 0])             # y_test
            )

            # Mock feature engine
            mock_feature_engine = MagicMock()
            mock_feature_engine.calculate_features.return_value = pd.DataFrame({
                'rsi_14': [30, 40, 50, 60, 70],
                'macd': [-0.2, -0.1, 0, 0.1, 0.2],
                'bbands_width': [0.01, 0.02, 0.03, 0.04, 0.05],
                'volume_change': [-0.1, -0.05, 0, 0.05, 0.1],
                'price_momentum': [-0.05, -0.02, 0, 0.02, 0.05]
            })

            # Initialize service
            prediction_service = PredictionService(config, event_bus)
            prediction_service.feature_engine = mock_feature_engine

            # Add market data
            prediction_service.market_data["BTC/USD"] = mock_ohlcv_data["BTC/USD"].copy()

            # Call retrain method
            prediction_service.retrain_models()

            # Verify model was trained
            assert mock_rf_instance.fit.call_count >= 1

            # Verify model was saved
            assert mock_save_model.call_count >= 1
