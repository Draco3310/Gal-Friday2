"""
Tests for the feature_engine module.
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from gal_friday.feature_engine import FeatureEngine
from gal_friday.config_manager import ConfigManager


@pytest.fixture
def feature_config():
    """Fixture providing feature engine configuration."""
    return {
        "feature_generation": {
            "lookback_periods": [14, 30, 50],
            "indicators": [
                "rsi", "macd", "bbands", "atr", "obv", 
                "ema", "sma", "stoch", "adx", "williams_r"
            ],
            "derived_features": {
                "price_momentum": True,
                "volatility_features": True,
                "trend_features": True,
                "volume_features": True,
                "pattern_recognition": True
            },
            "normalize": True,
            "target_generation": {
                "horizons": [1, 5, 10, 20],  # Looking ahead periods
                "methods": ["returns", "direction", "volatility"]
            }
        }
    }


def test_feature_engine_initialization(feature_config):
    """Test that the FeatureEngine initializes correctly."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    assert feature_engine is not None
    assert feature_engine.lookback_periods == feature_config["feature_generation"]["lookback_periods"]
    assert feature_engine.indicators == feature_config["feature_generation"]["indicators"]
    assert feature_engine.normalize == feature_config["feature_generation"]["normalize"]


def test_feature_engine_calculate_rsi(feature_config, mock_ohlcv_data):
    """Test calculating RSI indicator."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate RSI
    result = feature_engine._calculate_rsi(df, period=14)
    
    # Verify result
    assert "rsi_14" in result.columns
    assert len(result) == len(df)
    assert all(0 <= val <= 100 for val in result["rsi_14"].dropna())
    # First few values should be NaN due to lookback window
    assert result["rsi_14"].isna().sum() > 0


def test_feature_engine_calculate_macd(feature_config, mock_ohlcv_data):
    """Test calculating MACD indicator."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate MACD
    result = feature_engine._calculate_macd(df)
    
    # Verify result
    assert "macd" in result.columns
    assert "macd_signal" in result.columns
    assert "macd_hist" in result.columns
    assert len(result) == len(df)
    # First few values should be NaN due to lookback window
    assert result["macd"].isna().sum() > 0


def test_feature_engine_calculate_bollinger_bands(feature_config, mock_ohlcv_data):
    """Test calculating Bollinger Bands."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate Bollinger Bands
    result = feature_engine._calculate_bollinger_bands(df, period=20)
    
    # Verify result
    assert "bb_upper_20" in result.columns
    assert "bb_middle_20" in result.columns
    assert "bb_lower_20" in result.columns
    assert "bbands_width_20" in result.columns
    assert len(result) == len(df)
    # Check that upper band is always above middle band
    valid_indices = ~result["bb_upper_20"].isna()
    assert all(result.loc[valid_indices, "bb_upper_20"] >= result.loc[valid_indices, "bb_middle_20"])
    # Check that lower band is always below middle band
    assert all(result.loc[valid_indices, "bb_lower_20"] <= result.loc[valid_indices, "bb_middle_20"])


def test_feature_engine_calculate_sma(feature_config, mock_ohlcv_data):
    """Test calculating Simple Moving Average."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate SMA
    result = feature_engine._calculate_sma(df, period=10)
    
    # Verify result
    assert "sma_10" in result.columns
    assert len(result) == len(df)
    # First 9 values should be NaN
    assert result["sma_10"].iloc[:9].isna().all()
    # Rest should have values
    assert not result["sma_10"].iloc[9:].isna().any()


def test_feature_engine_calculate_ema(feature_config, mock_ohlcv_data):
    """Test calculating Exponential Moving Average."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate EMA
    result = feature_engine._calculate_ema(df, period=10)
    
    # Verify result
    assert "ema_10" in result.columns
    assert len(result) == len(df)
    # First few values should be NaN due to lookback window
    assert result["ema_10"].isna().sum() > 0


def test_feature_engine_calculate_atr(feature_config, mock_ohlcv_data):
    """Test calculating Average True Range."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate ATR
    result = feature_engine._calculate_atr(df, period=14)
    
    # Verify result
    assert "atr_14" in result.columns
    assert len(result) == len(df)
    # First few values should be NaN due to lookback window
    assert result["atr_14"].isna().sum() > 0
    # ATR should be positive
    assert all(val >= 0 for val in result["atr_14"].dropna())


def test_feature_engine_calculate_momentum_features(feature_config, mock_ohlcv_data):
    """Test calculating momentum features."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate momentum features
    result = feature_engine._calculate_momentum_features(df)
    
    # Verify result
    momentum_columns = [col for col in result.columns if "momentum" in col or "roc" in col]
    assert len(momentum_columns) > 0
    assert len(result) == len(df)


def test_feature_engine_calculate_volatility_features(feature_config, mock_ohlcv_data):
    """Test calculating volatility features."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate volatility features
    result = feature_engine._calculate_volatility_features(df)
    
    # Verify result
    volatility_columns = [
        col for col in result.columns 
        if "volatility" in col or "std" in col or "range" in col
    ]
    assert len(volatility_columns) > 0
    assert len(result) == len(df)


def test_feature_engine_calculate_volume_features(feature_config, mock_ohlcv_data):
    """Test calculating volume features."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate volume features
    result = feature_engine._calculate_volume_features(df)
    
    # Verify result
    volume_columns = [
        col for col in result.columns 
        if "volume" in col or "obv" in col
    ]
    assert len(volume_columns) > 0
    assert len(result) == len(df)


def test_feature_engine_normalize_features(feature_config, mock_ohlcv_data):
    """Test feature normalization."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Create sample features DataFrame
    features_df = pd.DataFrame({
        'rsi_14': np.random.uniform(0, 100, 100),
        'macd': np.random.uniform(-10, 10, 100),
        'price': np.random.uniform(40000, 60000, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })
    
    # Normalize features
    result = feature_engine._normalize_features(features_df)
    
    # Verify result
    assert len(result) == len(features_df)
    assert all(result.columns == features_df.columns)
    # Check that values are normalized
    for col in result.columns:
        assert -3 <= result[col].min() <= 3
        assert -3 <= result[col].max() <= 3


def test_feature_engine_calculate_all_features(feature_config, mock_ohlcv_data):
    """Test calculating all features."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Calculate all features
    result = feature_engine.calculate_features(df)
    
    # Verify result
    assert len(result) == len(df)
    # Check that we have more columns than the original dataframe
    assert len(result.columns) > len(df.columns)
    
    # Check that we have various types of features
    feature_types = {
        'momentum': ['rsi', 'momentum', 'roc'],
        'trend': ['macd', 'ema', 'sma', 'trend'],
        'volatility': ['bbands', 'atr', 'volatility', 'std'],
        'volume': ['obv', 'volume']
    }
    
    for feature_type, keywords in feature_types.items():
        found = False
        for keyword in keywords:
            matching_columns = [col for col in result.columns if keyword in col.lower()]
            if matching_columns:
                found = True
                break
        assert found, f"No {feature_type} features found in the result"


def test_feature_engine_generate_target_variables(feature_config, mock_ohlcv_data):
    """Test generating target variables for supervised learning."""
    config = ConfigManager(config_dict=feature_config)
    feature_engine = FeatureEngine(config)
    
    # Get BTC/USD data
    df = mock_ohlcv_data["BTC/USD"].copy()
    
    # Generate target variables
    result = feature_engine.generate_target_variables(df)
    
    # Verify result
    assert len(result) == len(df)
    
    # Check for target variables with different horizons
    horizons = feature_config["feature_generation"]["target_generation"]["horizons"]
    methods = feature_config["feature_generation"]["target_generation"]["methods"]
    
    for horizon in horizons:
        for method in methods:
            target_col = f"target_{method}_{horizon}"
            assert target_col in result.columns
    
    # Last N rows should have NaN values for targets (where N is the max horizon)
    max_horizon = max(horizons)
    assert result[f"target_returns_{max_horizon}"].iloc[-max_horizon:].isna().all()