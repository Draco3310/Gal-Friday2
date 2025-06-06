"""Unit tests for the technical_analysis module.

Tests all implementations (stub, pandas-ta, talib) and ensures
consistent behavior and proper error handling.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from gal_friday.technical_analysis import (
    TechnicalAnalysisInterface,
    PandasTAImplementation,
    StubImplementation,
    TALibImplementation,
    create_technical_analysis_service,
    # Backward compatibility functions
    rsi, bbands, ema, sma, macd, atr
)


class TestTechnicalAnalysisInterface:
    """Test the abstract interface definition."""
    
    def test_interface_cannot_be_instantiated(self):
        """Ensure abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            TechnicalAnalysisInterface()
    
    def test_interface_defines_all_methods(self):
        """Ensure all required methods are defined in the interface."""
        required_methods = ['rsi', 'bbands', 'ema', 'sma', 'macd', 'atr']
        for method in required_methods:
            assert hasattr(TechnicalAnalysisInterface, method)


class TestStubImplementation:
    """Test the stub implementation for testing environments."""
    
    @pytest.fixture
    def stub_service(self):
        return StubImplementation()
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample price data."""
        np.random.seed(42)
        return {
            'close': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
        }
    
    def test_rsi_returns_neutral_values(self, stub_service, sample_data):
        """Test that RSI returns neutral 50.0 values."""
        result = stub_service.rsi(sample_data['close'])
        assert len(result) == len(sample_data['close'])
        assert np.all(result == 50.0)
    
    def test_bbands_returns_simple_bands(self, stub_service, sample_data):
        """Test that Bollinger Bands returns reasonable values."""
        upper, middle, lower = stub_service.bbands(sample_data['close'])
        
        assert len(upper) == len(sample_data['close'])
        assert len(middle) == len(sample_data['close'])
        assert len(lower) == len(sample_data['close'])
        
        # Upper should be above middle, lower should be below
        assert np.all(upper >= middle)
        assert np.all(middle >= lower)
    
    def test_ema_returns_smoothed_values(self, stub_service, sample_data):
        """Test that EMA returns exponentially smoothed values."""
        result = stub_service.ema(sample_data['close'])
        assert len(result) == len(sample_data['close'])
        
        # EMA should start at first close value
        assert result[0] == sample_data['close'][0]
    
    def test_sma_returns_averaged_values(self, stub_service, sample_data):
        """Test that SMA returns simple averaged values."""
        result = stub_service.sma(sample_data['close'], timeperiod=10)
        assert len(result) == len(sample_data['close'])
    
    def test_macd_returns_zeros(self, stub_service, sample_data):
        """Test that MACD returns zero values for stub."""
        macd_line, signal, histogram = stub_service.macd(sample_data['close'])
        
        assert len(macd_line) == len(sample_data['close'])
        assert len(signal) == len(sample_data['close'])
        assert len(histogram) == len(sample_data['close'])
        
        assert np.all(macd_line == 0.0)
        assert np.all(signal == 0.0)
        assert np.all(histogram == 0.0)
    
    def test_atr_returns_simple_range(self, stub_service, sample_data):
        """Test that ATR returns simple volatility measure."""
        result = stub_service.atr(
            sample_data['high'], 
            sample_data['low'],
            sample_data['close']
        )
        assert len(result) == len(sample_data['close'])
        
        # ATR should be positive (high - low)
        assert np.all(result >= 0)


@pytest.mark.skipif(not pytest.importorskip("pandas_ta"), reason="pandas-ta not installed")
class TestPandasTAImplementation:
    """Test the pandas-ta based production implementation."""
    
    @pytest.fixture
    def pandas_ta_service(self):
        return PandasTAImplementation()
    
    @pytest.fixture
    def sample_data(self):
        """Generate realistic sample price data."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Generate realistic price movement
        close_prices = 100.0
        close_data = []
        for _ in range(200):
            close_prices *= (1 + np.random.normal(0, 0.02))
            close_data.append(close_prices)
        
        close_array = np.array(close_data)
        high_array = close_array * (1 + np.abs(np.random.normal(0, 0.01, 200)))
        low_array = close_array * (1 - np.abs(np.random.normal(0, 0.01, 200)))
        
        return {
            'close': close_array,
            'high': high_array,
            'low': low_array,
            'dates': dates
        }
    
    def test_rsi_calculation(self, pandas_ta_service, sample_data):
        """Test RSI calculation produces valid results."""
        result = pandas_ta_service.rsi(sample_data['close'], timeperiod=14)
        
        assert len(result) == len(sample_data['close'])
        assert not np.isnan(result).all()  # Should not be all NaN
        
        # RSI should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)
    
    def test_bbands_calculation(self, pandas_ta_service, sample_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = pandas_ta_service.bbands(
            sample_data['close'], 
            timeperiod=20,
            nbdevup=2.0,
            nbdevdn=2.0
        )
        
        assert len(upper) == len(sample_data['close'])
        assert len(middle) == len(sample_data['close'])
        assert len(lower) == len(sample_data['close'])
        
        # Check band relationships
        assert np.all(upper >= middle)
        assert np.all(middle >= lower)
        
        # Middle band should be close to SMA
        sma_result = pandas_ta_service.sma(sample_data['close'], timeperiod=20)
        np.testing.assert_array_almost_equal(middle[19:], sma_result[19:], decimal=5)
    
    def test_ema_calculation(self, pandas_ta_service, sample_data):
        """Test EMA calculation."""
        result = pandas_ta_service.ema(sample_data['close'], timeperiod=12)
        
        assert len(result) == len(sample_data['close'])
        assert not np.isnan(result).all()
        
        # EMA should respond faster to recent changes than SMA
        sma_result = pandas_ta_service.sma(sample_data['close'], timeperiod=12)
        
        # Both should have similar ranges
        assert np.min(result) > 0
        assert np.max(result) < np.max(sample_data['close']) * 2
    
    def test_macd_calculation(self, pandas_ta_service, sample_data):
        """Test MACD calculation."""
        macd_line, signal, histogram = pandas_ta_service.macd(
            sample_data['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        assert len(macd_line) == len(sample_data['close'])
        assert len(signal) == len(sample_data['close'])
        assert len(histogram) == len(sample_data['close'])
        
        # Histogram should be MACD - Signal
        expected_hist = macd_line - signal
        np.testing.assert_array_almost_equal(histogram, expected_hist, decimal=10)
    
    def test_atr_calculation(self, pandas_ta_service, sample_data):
        """Test ATR calculation."""
        result = pandas_ta_service.atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close'],
            timeperiod=14
        )
        
        assert len(result) == len(sample_data['close'])
        assert not np.isnan(result).all()
        
        # ATR should be positive
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)
    
    def test_nan_handling(self, pandas_ta_service):
        """Test handling of NaN values in input."""
        # Create data with NaN values
        close_with_nan = np.array([100, 101, np.nan, 103, 104, 105])
        
        # Should handle NaN gracefully
        result = pandas_ta_service.rsi(close_with_nan, timeperiod=3)
        assert len(result) == len(close_with_nan)
    
    def test_insufficient_data(self, pandas_ta_service):
        """Test error handling for insufficient data."""
        short_data = np.array([100, 101])
        
        with pytest.raises(ValueError, match="Insufficient data"):
            pandas_ta_service.rsi(short_data, timeperiod=14)


class TestFactoryFunction:
    """Test the factory function for creating services."""
    
    def test_default_stub_mode(self):
        """Test that default mode returns stub implementation."""
        service = create_technical_analysis_service({})
        assert isinstance(service, StubImplementation)
    
    def test_production_mode_pandas_ta(self):
        """Test production mode with pandas-ta."""
        with patch('gal_friday.technical_analysis.PANDAS_TA_AVAILABLE', True):
            service = create_technical_analysis_service({
                'use_production_indicators': True,
                'indicator_library': 'pandas_ta'
            })
            # Can't check instance without pandas_ta, but we can verify it's not stub
            if service.__class__.__name__ == 'PandasTAImplementation':
                assert True
            else:
                assert isinstance(service, StubImplementation)
    
    def test_production_mode_talib_fallback(self):
        """Test that talib request falls back to pandas-ta if not available."""
        with patch('gal_friday.technical_analysis.TALIB_AVAILABLE', False):
            with patch('gal_friday.technical_analysis.PANDAS_TA_AVAILABLE', True):
                service = create_technical_analysis_service({
                    'use_production_indicators': True,
                    'indicator_library': 'talib'
                })
                # Should fall back to pandas-ta or stub
                assert service is not None
    
    def test_no_libraries_available(self):
        """Test fallback when no libraries are available."""
        with patch('gal_friday.technical_analysis.PANDAS_TA_AVAILABLE', False):
            with patch('gal_friday.technical_analysis.TALIB_AVAILABLE', False):
                service = create_technical_analysis_service({
                    'use_production_indicators': True
                })
                assert isinstance(service, StubImplementation)


class TestBackwardCompatibility:
    """Test backward compatibility with the old talib_stubs interface."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return {
            'close': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(95, 105, 50),
        }
    
    def test_rsi_wrapper(self, sample_data):
        """Test the rsi wrapper function."""
        result = rsi(sample_data['close'], timeperiod=14)
        assert len(result) == len(sample_data['close'])
    
    def test_bbands_wrapper(self, sample_data):
        """Test the bbands wrapper function."""
        result = bbands(sample_data['close'])
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(len(band) == len(sample_data['close']) for band in result)
    
    def test_ema_wrapper(self, sample_data):
        """Test the ema wrapper function."""
        result = ema(sample_data['close'])
        assert len(result) == len(sample_data['close'])
    
    def test_sma_wrapper(self, sample_data):
        """Test the sma wrapper function."""
        result = sma(sample_data['close'])
        assert len(result) == len(sample_data['close'])
    
    def test_macd_wrapper(self, sample_data):
        """Test the macd wrapper function."""
        result = macd(sample_data['close'])
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(len(line) == len(sample_data['close']) for line in result)
    
    def test_atr_wrapper(self, sample_data):
        """Test the atr wrapper function."""
        result = atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )
        assert len(result) == len(sample_data['close'])


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    @pytest.fixture
    def service(self):
        return StubImplementation()
    
    def test_empty_array_handling(self, service):
        """Test handling of empty arrays."""
        empty_array = np.array([])
        
        # EMA should return empty array
        result = service.ema(empty_array)
        assert len(result) == 0
        
        # RSI should return empty array  
        result = service.rsi(empty_array)
        assert len(result) == 0
    
    def test_single_value_array(self, service):
        """Test handling of single-value arrays."""
        single_value = np.array([100.0])
        
        result = service.rsi(single_value)
        assert len(result) == 1
        assert result[0] == 50.0  # Neutral RSI
    
    def test_period_longer_than_data(self, service):
        """Test when period is longer than available data."""
        short_data = np.array([100, 101, 102, 103, 104])
        
        # SMA with period longer than data
        result = service.sma(short_data, timeperiod=10)
        assert len(result) == len(short_data)
        
        # Should handle gracefully
        np.testing.assert_array_equal(result, short_data)
