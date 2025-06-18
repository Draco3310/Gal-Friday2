"""Technical Analysis module with dependency injection for Gal-Friday.

This module provides an abstract interface for technical analysis calculations
with multiple implementations (production, stub, talib) that can be switched
via configuration. It replaces the temporary talib_stubs.py file.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, cast

import numpy as np
import pandas as pd

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
    def rsi(self, close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
        """Calculate Relative Strength Index."""

    @abstractmethod
    def bbands(
        self,
        close: np.ndarray[Any, Any],
        timeperiod: int = 20,  # Changed from 5 to match standard
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """

    @abstractmethod
    def ema(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Exponential Moving Average."""

    @abstractmethod
    def sma(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Simple Moving Average."""

    @abstractmethod
    def macd(
        self,
        close: np.ndarray[Any, Any],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Moving Average Convergence/Divergence.

        Returns:
            Tuple of (macd, signal, histogram)
        """

    @abstractmethod
    def atr(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Average True Range."""


class PandasTAImplementation(TechnicalAnalysisInterface):
    """Production implementation using pandas-ta library."""

    def __init__(self) -> None:
        """Initialize the instance."""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta library not installed. Run: pip install pandas-ta")
        self.logger = logging.getLogger(__name__)

    def _validate_and_convert(self, data: np.ndarray[Any, Any] | pd.Series[Any] | list[float], min_length: int = 1) -> pd.Series[Any]:
        """Validate and convert input data to pandas Series.

        Args:
            data: Input data as numpy array, pandas Series, or list
            min_length: Minimum required data length

        Returns:
            Pandas Series with float64 dtype
        """
        # Convert to pandas Series based on input type
        if isinstance(data, pd.Series):
            series = data.astype(np.float64)
        elif isinstance(data, np.ndarray):
            series = pd.Series(data, dtype=np.float64)
        else:
            # Handle lists or other array-like objects
            series = pd.Series(data, dtype=np.float64)

        # Validate length
        if len(series) < min_length:
            raise ValueError(f"Insufficient data: need at least {min_length} points, got {len(series)}")

        # Handle NaN values - forward fill then backward fill
        if series.isna().any():
            self.logger.warning("NaN values detected in input data, applying forward/backward fill")
            series = series.ffill().bfill()

        return series

    def rsi(self, close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
        """Calculate Relative Strength Index using pandas-ta."""
        close_series = self._validate_and_convert(close, timeperiod + 1)

        result = ta.rsi(close_series, length=timeperiod)

        # Fill initial NaN values with neutral RSI value
        result = result.fillna(50.0)

        return np.asarray(result.values, dtype=np.float64)

    def bbands(
        self,
        close: np.ndarray[Any, Any],
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
            upper_col = next(col for col in bbands_df.columns if col.startswith("BBU_"))
            middle_col = next(col for col in bbands_df.columns if col.startswith("BBM_"))
            lower_col = next(col for col in bbands_df.columns if col.startswith("BBL_"))

            upper = bbands_df[upper_col]
            middle = bbands_df[middle_col]
            lower = bbands_df[lower_col]

        # Fill NaN values
        upper = upper.fillna(close_series)
        middle = middle.fillna(close_series)
        lower = lower.fillna(close_series)

        return (
            cast("np.ndarray[Any, Any]", upper.values),
            cast("np.ndarray[Any, Any]", middle.values),
            cast("np.ndarray[Any, Any]", lower.values),
        )

    def ema(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Exponential Moving Average using pandas-ta."""
        close_series = self._validate_and_convert(close, 1)

        result = ta.ema(close_series, length=timeperiod)

        # Fill initial NaN values with the first available close price
        result = result.bfill()

        return np.asarray(result.values, dtype=np.float64)

    def sma(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Simple Moving Average using pandas-ta."""
        close_series = self._validate_and_convert(close, timeperiod)

        result = ta.sma(close_series, length=timeperiod)

        # Fill initial NaN values
        result = result.bfill()

        return np.asarray(result.values, dtype=np.float64)

    def macd(
        self,
        close: np.ndarray[Any, Any],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
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
            macd_col = next(col for col in macd_df.columns if col.startswith("MACD_") and not col.endswith("h") and not col.endswith("s"))
            signal_col = next(col for col in macd_df.columns if col.endswith("s"))
            hist_col = next(col for col in macd_df.columns if col.endswith("h"))

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
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Average True Range using pandas-ta."""
        high_series = self._validate_and_convert(high, timeperiod)
        low_series = self._validate_and_convert(low, timeperiod)
        close_series = self._validate_and_convert(close, timeperiod)

        result = ta.atr(high=high_series, low=low_series, close=close_series, length=timeperiod)

        # Fill initial NaN values with 0
        result = result.fillna(0.0)

        return np.asarray(result.values, dtype=np.float64)


class TALibImplementation(TechnicalAnalysisInterface):
    """Implementation using the actual TA-Lib library."""

    def __init__(self) -> None:
        """Initialize the instance."""
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib library not installed. See https://github.com/mrjbq7/ta-lib for installation instructions.")
        self.logger = logging.getLogger(__name__)

    def _validate_input(self, data: np.ndarray[Any, Any] | list[float], min_length: int = 1) -> np.ndarray[Any, Any]:
        """Validate and clean input data.

        Args:
            data: Input data as numpy array or list
            min_length: Minimum required data length

        Returns:
            Numpy array with float64 dtype
        """
        # Ensure we have a numpy array
        arr = data.astype(np.float64) if isinstance(data, np.ndarray) else np.array(data, dtype=np.float64)

        if len(arr) < min_length:
            raise ValueError(f"Insufficient data: need at least {min_length} points")

        # TA-Lib handles NaN differently, so we need to be careful
        if np.isnan(arr).any():
            self.logger.warning("NaN values detected in input data")

        return arr

    def rsi(self, close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
        """Calculate RSI using TA-Lib."""
        close = self._validate_input(close, timeperiod + 1)
        result = talib.RSI(close, timeperiod=timeperiod)
        return np.asarray(result, dtype=np.float64)

    def bbands(
        self,
        close: np.ndarray[Any, Any],
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Bollinger Bands using TA-Lib."""
        close = self._validate_input(close, timeperiod)
        upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        return (np.asarray(upper, dtype=np.float64), np.asarray(middle, dtype=np.float64), np.asarray(lower, dtype=np.float64))

    def ema(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate EMA using TA-Lib."""
        close = self._validate_input(close)
        result = talib.EMA(close, timeperiod=timeperiod)
        return np.asarray(result, dtype=np.float64)

    def sma(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate SMA using TA-Lib."""
        close = self._validate_input(close, timeperiod)
        result = talib.SMA(close, timeperiod=timeperiod)
        return np.asarray(result, dtype=np.float64)

    def macd(
        self,
        close: np.ndarray[Any, Any],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate MACD using TA-Lib."""
        close = self._validate_input(close, slowperiod)
        macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return (np.asarray(macd, dtype=np.float64), np.asarray(signal, dtype=np.float64), np.asarray(hist, dtype=np.float64))

    def atr(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate ATR using TA-Lib."""
        high = self._validate_input(high, timeperiod)
        low = self._validate_input(low, timeperiod)
        close = self._validate_input(close, timeperiod)
        result = talib.ATR(high, low, close, timeperiod=timeperiod)
        return np.asarray(result, dtype=np.float64)


class StubImplementation(TechnicalAnalysisInterface):
    """Stub implementation for testing with more realistic behavior."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def rsi(self, close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
        """Return neutral RSI values for testing."""
        return np.full(len(close), 50.0)

    def bbands(
        self,
        close: np.ndarray[Any, Any],
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return simple bands based on close price."""
        if len(close) < timeperiod:
            middle = close
        else:
            # Simple moving average for middle band
            middle = np.convolve(close, np.ones(timeperiod)/timeperiod, mode="same")

        # Create simple bands
        offset = 0.02 * middle  # 2% bands
        upper = middle + (offset * nbdevup)
        lower = middle - (offset * nbdevdn)

        return (upper, middle, lower)

    def ema(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Return simple exponential smoothing for testing."""
        if len(close) == 0:
            return np.array([])

        result = np.zeros_like(close)
        alpha = 2.0 / (timeperiod + 1.0)
        result[0] = close[0]

        for i in range(1, len(close)):
            result[i] = alpha * close[i] + (1 - alpha) * result[i-1]

        return result

    def sma(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Return simple moving average for testing."""
        if len(close) < timeperiod:
            return close
        return np.convolve(close, np.ones(timeperiod)/timeperiod, mode="same")

    def macd(
        self,
        close: np.ndarray[Any, Any],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return zero MACD values for testing."""
        zeros = np.zeros(len(close))
        return (zeros, zeros, zeros)

    def atr(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Return simple volatility measure for testing."""
        # Simple range calculation
        true_range = high - low
        if len(true_range) < timeperiod:
            return np.asarray(true_range, dtype=np.float64)

        # Simple moving average of range
        return np.convolve(true_range, np.ones(timeperiod)/timeperiod, mode="same")


def create_technical_analysis_service(config: dict[str, Any]) -> TechnicalAnalysisInterface:
    """Factory function to create the appropriate technical analysis implementation.

    Args:
        config: Configuration dictionary with optional keys:
            - use_production_indicators (bool): Use production implementation
            - indicator_library (str): 'pandas_ta' or 'talib'

    Returns:
        Technical analysis service instance
    """
    use_production = config.get("use_production_indicators", False)

    if not use_production:
        return StubImplementation()

    library = config.get("indicator_library", "pandas_ta").lower()

    if library == "talib":
        if TALIB_AVAILABLE:
            return TALibImplementation()
        logging.warning("TA-Lib requested but not available. Falling back to pandas-ta.")

    if PANDAS_TA_AVAILABLE:
        return PandasTAImplementation()
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
            "use_production_indicators": True,
            "indicator_library": "pandas_ta",
        })
    return _default_service


# Wrapper functions for backward compatibility
def rsi(close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
    """Calculate Relative Strength Index."""
    return _get_service().rsi(close, timeperiod)


def bbands(
    close: np.ndarray[Any, Any],
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Calculate Bollinger Bands."""
    return _get_service().bbands(close, timeperiod, nbdevup, nbdevdn, matype)


def ema(close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
    """Calculate Exponential Moving Average."""
    return _get_service().ema(close, timeperiod)


def sma(close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
    """Calculate Simple Moving Average."""
    return _get_service().sma(close, timeperiod)


def macd(
    close: np.ndarray[Any, Any],
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Calculate Moving Average Convergence/Divergence."""
    return _get_service().macd(close, fastperiod, slowperiod, signalperiod)


def atr(
    high: np.ndarray[Any, Any],
    low: np.ndarray[Any, Any],
    close: np.ndarray[Any, Any],
    timeperiod: int = 14) -> np.ndarray[Any, Any]:
    """Calculate Average True Range."""
    return _get_service().atr(high, low, close, timeperiod)
