"""Technical Analysis module with dependency injection for Gal-Friday.

This module provides an abstract interface for technical analysis calculations
with multiple implementations (production, stub, talib) that can be switched
via configuration. It replaces the temporary talib_stubs.py file.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import logging
from decimal import Decimal

# Optional imports with graceful fallback
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None


class TechnicalAnalysisInterface(ABC):
    """Abstract interface for technical analysis calculations."""
    
    @abstractmethod
    def rsi(self, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        pass
    
    @abstractmethod
    def bbands(
        self, 
        close: np.ndarray, 
        timeperiod: int = 20,  # Changed from 5 to match standard
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands.
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        pass
    
    @abstractmethod
    def ema(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        pass
    
    @abstractmethod
    def sma(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Calculate Simple Moving Average."""
        pass
    
    @abstractmethod
    def macd(
        self,
        close: np.ndarray,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Moving Average Convergence/Divergence.
        
        Returns:
            Tuple of (macd, signal, histogram)
        """
        pass
    
    @abstractmethod
    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray, 
        close: np.ndarray,
        timeperiod: int = 14
    ) -> np.ndarray:
        """Calculate Average True Range."""
        pass


class PandasTAImplementation(TechnicalAnalysisInterface):
    """Production implementation using pandas-ta library."""
    
    def __init__(self):
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta library not installed. Run: pip install pandas-ta")
        self.logger = logging.getLogger(__name__)
    
    def _validate_and_convert(self, data: np.ndarray, min_length: int = 1) -> pd.Series:
        """Validate and convert input data to pandas Series."""
        if not isinstance(data, (np.ndarray, pd.Series)):
            data = np.array(data, dtype=np.float64)
        
        if len(data) < min_length:
            raise ValueError(f"Insufficient data: need at least {min_length} points, got {len(data)}")
        
        # Convert to pandas Series for pandas-ta
        if isinstance(data, np.ndarray):
            series = pd.Series(data, dtype=np.float64)
        else:
            series = data.astype(np.float64)
        
        # Handle NaN values - forward fill then backward fill
        if series.isna().any():
            self.logger.warning("NaN values detected in input data, applying forward/backward fill")
            series = series.ffill().bfill()
        
        return series
    
    def rsi(self, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index using pandas-ta."""
        close_series = self._validate_and_convert(close, timeperiod + 1)
        
        result = ta.rsi(close_series, length=timeperiod)
        
        # Fill initial NaN values with neutral RSI value
        result = result.fillna(50.0)
        
        return result.values
    
    def bbands(
        self, 
        close: np.ndarray, 
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using pandas-ta."""
        close_series = self._validate_and_convert(close, timeperiod)
        
        # pandas-ta uses 'std' parameter
        bbands_df = ta.bbands(close_series, length=timeperiod, std=nbdevup)
        
        if bbands_df is None or bbands_df.empty:
            # Fallback calculation
            middle = close_series.rolling(window=timeperiod).mean()
            std = close_series.rolling(window=timeperiod).std()
            upper = middle + (std * nbdevup)
            lower = middle - (std * nbdevdn)
        else:
            # Extract columns - pandas-ta uses specific naming
            upper_col = [col for col in bbands_df.columns if col.startswith('BBU_')][0]
            middle_col = [col for col in bbands_df.columns if col.startswith('BBM_')][0]
            lower_col = [col for col in bbands_df.columns if col.startswith('BBL_')][0]
            
            upper = bbands_df[upper_col]
            middle = bbands_df[middle_col]
            lower = bbands_df[lower_col]
        
        # Fill NaN values
        upper = upper.fillna(close_series)
        middle = middle.fillna(close_series)
        lower = lower.fillna(close_series)
        
        return (upper.values, middle.values, lower.values)
    
    def ema(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Calculate Exponential Moving Average using pandas-ta."""
        close_series = self._validate_and_convert(close, 1)
        
        result = ta.ema(close_series, length=timeperiod)
        
        # Fill initial NaN values with the first available close price
        result = result.bfill()
        
        return result.values
    
    def sma(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Calculate Simple Moving Average using pandas-ta."""
        close_series = self._validate_and_convert(close, timeperiod)
        
        result = ta.sma(close_series, length=timeperiod)
        
        # Fill initial NaN values
        result = result.bfill()
        
        return result.values
    
    def macd(
        self,
        close: np.ndarray,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using pandas-ta."""
        close_series = self._validate_and_convert(close, slowperiod)
        
        macd_df = ta.macd(close_series, fast=fastperiod, slow=slowperiod, signal=signalperiod)
        
        if macd_df is None or macd_df.empty:
            # Fallback calculation
            ema_fast = ta.ema(close_series, length=fastperiod)
            ema_slow = ta.ema(close_series, length=slowperiod)
            macd_line = ema_fast - ema_slow
            signal_line = ta.ema(macd_line, length=signalperiod)
            histogram = macd_line - signal_line
        else:
            # Extract columns - pandas-ta returns MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            macd_col = [col for col in macd_df.columns if col.startswith('MACD_') and not col.endswith('h') and not col.endswith('s')][0]
            signal_col = [col for col in macd_df.columns if col.endswith('s')][0]
            hist_col = [col for col in macd_df.columns if col.endswith('h')][0]
            
            macd_line = macd_df[macd_col]
            signal_line = macd_df[signal_col]
            histogram = macd_df[hist_col]
        
        # Fill NaN values with 0
        macd_line = macd_line.fillna(0.0)
        signal_line = signal_line.fillna(0.0)
        histogram = histogram.fillna(0.0)
        
        return (macd_line.values, signal_line.values, histogram.values)
    
    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeperiod: int = 14
    ) -> np.ndarray:
        """Calculate Average True Range using pandas-ta."""
        high_series = self._validate_and_convert(high, timeperiod)
        low_series = self._validate_and_convert(low, timeperiod)
        close_series = self._validate_and_convert(close, timeperiod)
        
        result = ta.atr(high=high_series, low=low_series, close=close_series, length=timeperiod)
        
        # Fill initial NaN values with 0
        result = result.fillna(0.0)
        
        return result.values


class TALibImplementation(TechnicalAnalysisInterface):
    """Implementation using the actual TA-Lib library."""
    
    def __init__(self):
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib library not installed. See https://github.com/mrjbq7/ta-lib for installation instructions.")
        self.logger = logging.getLogger(__name__)
    
    def _validate_input(self, data: np.ndarray, min_length: int = 1) -> np.ndarray:
        """Validate and clean input data."""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        
        if len(data) < min_length:
            raise ValueError(f"Insufficient data: need at least {min_length} points")
        
        # TA-Lib handles NaN differently, so we need to be careful
        if np.isnan(data).any():
            self.logger.warning("NaN values detected in input data")
        
        return data.astype(np.float64)
    
    def rsi(self, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Calculate RSI using TA-Lib."""
        close = self._validate_input(close, timeperiod + 1)
        return talib.RSI(close, timeperiod=timeperiod)
    
    def bbands(
        self, 
        close: np.ndarray, 
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using TA-Lib."""
        close = self._validate_input(close, timeperiod)
        return talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    
    def ema(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Calculate EMA using TA-Lib."""
        close = self._validate_input(close)
        return talib.EMA(close, timeperiod=timeperiod)
    
    def sma(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Calculate SMA using TA-Lib."""
        close = self._validate_input(close, timeperiod)
        return talib.SMA(close, timeperiod=timeperiod)
    
    def macd(
        self,
        close: np.ndarray,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using TA-Lib."""
        close = self._validate_input(close, slowperiod)
        return talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    
    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeperiod: int = 14
    ) -> np.ndarray:
        """Calculate ATR using TA-Lib."""
        high = self._validate_input(high, timeperiod)
        low = self._validate_input(low, timeperiod)
        close = self._validate_input(close, timeperiod)
        return talib.ATR(high, low, close, timeperiod=timeperiod)


class StubImplementation(TechnicalAnalysisInterface):
    """Stub implementation for testing with more realistic behavior."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def rsi(self, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Return neutral RSI values for testing."""
        return np.full(len(close), 50.0)
    
    def bbands(
        self, 
        close: np.ndarray, 
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return simple bands based on close price."""
        if len(close) < timeperiod:
            middle = close
        else:
            # Simple moving average for middle band
            middle = np.convolve(close, np.ones(timeperiod)/timeperiod, mode='same')
        
        # Create simple bands
        offset = 0.02 * middle  # 2% bands
        upper = middle + (offset * nbdevup)
        lower = middle - (offset * nbdevdn)
        
        return (upper, middle, lower)
    
    def ema(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Return simple exponential smoothing for testing."""
        if len(close) == 0:
            return np.array([])
        
        result = np.zeros_like(close)
        alpha = 2.0 / (timeperiod + 1.0)
        result[0] = close[0]
        
        for i in range(1, len(close)):
            result[i] = alpha * close[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def sma(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Return simple moving average for testing."""
        if len(close) < timeperiod:
            return close
        return np.convolve(close, np.ones(timeperiod)/timeperiod, mode='same')
    
    def macd(
        self,
        close: np.ndarray,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return zero MACD values for testing."""
        zeros = np.zeros(len(close))
        return (zeros, zeros, zeros)
    
    def atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timeperiod: int = 14
    ) -> np.ndarray:
        """Return simple volatility measure for testing."""
        # Simple range calculation
        true_range = high - low
        if len(true_range) < timeperiod:
            return true_range
        
        # Simple moving average of range
        return np.convolve(true_range, np.ones(timeperiod)/timeperiod, mode='same')


def create_technical_analysis_service(config: Dict[str, Any]) -> TechnicalAnalysisInterface:
    """Factory function to create the appropriate technical analysis implementation.
    
    Args:
        config: Configuration dictionary with optional keys:
            - use_production_indicators (bool): Use production implementation
            - indicator_library (str): 'pandas_ta' or 'talib'
    
    Returns:
        Technical analysis service instance
    """
    use_production = config.get('use_production_indicators', False)
    
    if not use_production:
        return StubImplementation()
    
    library = config.get('indicator_library', 'pandas_ta').lower()
    
    if library == 'talib':
        if TALIB_AVAILABLE:
            return TALibImplementation()
        else:
            logging.warning("TA-Lib requested but not available. Falling back to pandas-ta.")
    
    if PANDAS_TA_AVAILABLE:
        return PandasTAImplementation()
    else:
        logging.error("No technical analysis library available. Using stub implementation.")
        return StubImplementation()


# Backward compatibility layer - mimics the old talib_stubs interface
_default_service = None

def _get_service() -> TechnicalAnalysisInterface:
    """Get the default service instance."""
    global _default_service
    if _default_service is None:
        # Default to production mode with pandas-ta
        _default_service = create_technical_analysis_service({
            'use_production_indicators': True,
            'indicator_library': 'pandas_ta'
        })
    return _default_service


# Wrapper functions for backward compatibility
def rsi(close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index."""
    return _get_service().rsi(close, timeperiod)


def bbands(
    close: np.ndarray,
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands."""
    return _get_service().bbands(close, timeperiod, nbdevup, nbdevdn, matype)


def ema(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    return _get_service().ema(close, timeperiod)


def sma(close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
    """Calculate Simple Moving Average."""
    return _get_service().sma(close, timeperiod)


def macd(
    close: np.ndarray,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Moving Average Convergence/Divergence."""
    return _get_service().macd(close, fastperiod, slowperiod, signalperiod)


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timeperiod: int = 14,
) -> np.ndarray:
    """Calculate Average True Range."""
    return _get_service().atr(high, low, close, timeperiod)
