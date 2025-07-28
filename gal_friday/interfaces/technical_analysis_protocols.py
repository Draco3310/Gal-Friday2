"""Technical analysis protocols for type-safe, protocol-based architecture.

This module provides protocol definitions for technical analysis components, enabling
duck-typing compatibility and better architectural separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class TechnicalAnalysisProtocol(Protocol):
    """Protocol for technical analysis calculation components.

    This protocol defines the interface for components that provide technical
    analysis calculations like indicators, oscillators, and statistical measures.
    """

    def rsi(self, close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
        """Calculate Relative Strength Index.

        Args:
            close: Array of closing prices
            timeperiod: Period for RSI calculation (default: 14)

        Returns:
            Array of RSI values (0-100 range)
        """
        ...

    def bbands(
        self,
        close: np.ndarray[Any, Any],
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Bollinger Bands.

        Args:
            close: Array of closing prices
            timeperiod: Period for moving average calculation (default: 20)
            nbdevup: Number of standard deviations for upper band (default: 2.0)
            nbdevdn: Number of standard deviations for lower band (default: 2.0)
            matype: Moving average type (default: 0 = SMA)

        Returns:
            Tuple of (upper_band, middle_band, lower_band) arrays
        """
        ...

    def ema(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Exponential Moving Average.

        Args:
            close: Array of closing prices
            timeperiod: Period for EMA calculation (default: 30)

        Returns:
            Array of EMA values
        """
        ...

    def sma(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Simple Moving Average.

        Args:
            close: Array of closing prices
            timeperiod: Period for SMA calculation (default: 30)

        Returns:
            Array of SMA values
        """
        ...

    def macd(
        self,
        close: np.ndarray[Any, Any],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Moving Average Convergence/Divergence.

        Args:
            close: Array of closing prices
            fastperiod: Fast EMA period (default: 12)
            slowperiod: Slow EMA period (default: 26)
            signalperiod: Signal line EMA period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram) arrays
        """
        ...

    def atr(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Average True Range.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            timeperiod: Period for ATR calculation (default: 14)

        Returns:
            Array of ATR values
        """
        ...


@runtime_checkable
class TechnicalAnalysisExtendedProtocol(Protocol):
    """Extended protocol for advanced technical analysis components.

    This protocol extends the basic technical analysis protocol with additional
    indicators and advanced calculation methods.
    """

    # Include all basic methods
    def rsi(self, close: np.ndarray[Any, Any], timeperiod: int = 14) -> np.ndarray[Any, Any]:
        """Calculate Relative Strength Index."""
        ...

    def bbands(
        self,
        close: np.ndarray[Any, Any],
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0,
        matype: int = 0,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Bollinger Bands."""
        ...

    def ema(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Exponential Moving Average."""
        ...

    def sma(self, close: np.ndarray[Any, Any], timeperiod: int = 30) -> np.ndarray[Any, Any]:
        """Calculate Simple Moving Average."""
        ...

    def macd(
        self,
        close: np.ndarray[Any, Any],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Moving Average Convergence/Divergence."""
        ...

    def atr(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Average True Range."""
        ...

    # Extended methods
    def stoch(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowd_period: int = 3,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Stochastic Oscillator.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            fastk_period: Fast %K period (default: 5)
            slowk_period: Slow %K period (default: 3)
            slowd_period: Slow %D period (default: 3)

        Returns:
            Tuple of (slowk, slowd) arrays
        """
        ...

    def adx(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Average Directional Movement Index.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            timeperiod: Period for ADX calculation (default: 14)

        Returns:
            Array of ADX values
        """
        ...

    def cci(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Commodity Channel Index.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            timeperiod: Period for CCI calculation (default: 14)

        Returns:
            Array of CCI values
        """
        ...

    def williams_r(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        timeperiod: int = 14,
    ) -> np.ndarray[Any, Any]:
        """Calculate Williams %R.

        Args:
            high: Array of high prices
            low: Array of low prices
            close: Array of closing prices
            timeperiod: Period for Williams %R calculation (default: 14)

        Returns:
            Array of Williams %R values
        """
        ...


@runtime_checkable
class TechnicalAnalysisValidatorProtocol(Protocol):
    """Protocol for technical analysis data validation components.

    This protocol defines the interface for components that validate input data
    and parameters for technical analysis calculations.
    """

    def validate_price_data(
        self,
        data: np.ndarray[Any, Any],
        min_length: int = 1,
        allow_nan: bool = False,
    ) -> bool:
        """Validate price data for technical analysis calculations.

        Args:
            data: Price data array to validate
            min_length: Minimum required data length
            allow_nan: Whether to allow NaN values in data

        Returns:
            True if data is valid, False otherwise
        """
        ...

    def validate_ohlc_data(
        self,
        high: np.ndarray[Any, Any],
        low: np.ndarray[Any, Any],
        close: np.ndarray[Any, Any],
        open_price: np.ndarray[Any, Any] | None = None,
    ) -> bool:
        """Validate OHLC data consistency.

        Args:
            high: High price array
            low: Low price array
            close: Close price array
            open_price: Optional open price array

        Returns:
            True if OHLC data is consistent, False otherwise
        """
        ...

    def get_validation_errors(self) -> list[str]:
        """Get list of validation errors from last validation.

        Returns:
            List of validation error messages
        """
        ...


@runtime_checkable
class TechnicalAnalysisFactoryProtocol(Protocol):
    """Protocol for technical analysis component factory.

    This protocol defines the interface for factories that create technical
    analysis components based on configuration and available libraries.
    """

    def create_calculator(self, implementation: str = "auto") -> TechnicalAnalysisProtocol:
        """Create a technical analysis calculator instance.

        Args:
            implementation: Implementation type ("talib", "pandas_ta", "numpy", "auto")

        Returns:
            Technical analysis calculator instance

        Raises:
            ValueError: If implementation is not supported
            ImportError: If required library is not available
        """
        ...

    def get_available_implementations(self) -> list[str]:
        """Get list of available implementation types.

        Returns:
            List of available implementation names
        """
        ...

    def is_implementation_available(self, implementation: str) -> bool:
        """Check if an implementation is available.

        Args:
            implementation: Implementation name to check

        Returns:
            True if implementation is available, False otherwise
        """
        ...
