"""Provide a backtesting environment for algorithmic trading strategies.

This module contains the BacktestingEngine which orchestrates backtesting simulations
using historical data. It handles loading data, initializing simulation services,
executing the simulation, and calculating performance metrics.
"""

# Standard library imports
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
)

# Third-party imports
import numpy as np
import pandas as pd

# Type variables for generic types - defined at module level
_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
ConfigValue = str | int | float | bool | dict | list | None  # Type alias for config values


def decimal_to_float(obj: Any) -> float:  # noqa: ANN401
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(msg)


# Initialize logger
log = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    # Import core types

    # Import implementation types
    from gal_friday.feature_engine import FeatureEngine as _FeatureEngine
    from gal_friday.logger_service import LoggerService as _LoggerService
    from gal_friday.portfolio_manager import PortfolioManager as _PortfolioManager
    from gal_friday.prediction_service import PredictionService as _PredictionService
    from gal_friday.risk_manager import RiskManager as _RiskManager
    from gal_friday.simulated_market_price_service import (
        SimulatedMarketPriceService as _SimulatedMarketPriceService,
    )
    from gal_friday.strategy_arbitrator import StrategyArbitrator as _StrategyArbitrator

    # Define type aliases that use the implementation types
    LoggerServiceType = _LoggerService[Any]
    SimulatedMarketPriceServiceType = _SimulatedMarketPriceService
    PortfolioManagerType = _PortfolioManager
    FeatureEngineType = _FeatureEngine
    PredictionServiceType = _PredictionService
    RiskManagerType = _RiskManager
    StrategyArbitratorType = _StrategyArbitrator

    # Use Protocol from typing_extensions for better compatibility
    from typing_extensions import Protocol as ProtocolType

    Protocol = ProtocolType  # type: ignore
else:
    try:
        from typing import Protocol as TypingProtocol  # type: ignore

        ProtocolType = TypingProtocol
    except ImportError:
        from typing_extensions import Protocol as TypingProtocol  # type: ignore

        ProtocolType = TypingProtocol

    from collections.abc import Awaitable, Callable, Coroutine
    from typing import (
        TYPE_CHECKING,
        Any,
        Generic,
        TypeVar,
    )

# Third-party imports

# Define a type for async functions
AsyncFunction = TypeVar("AsyncFunction", bound=Callable[..., Coroutine[Any, Any, Any]])

# Type aliases
if TYPE_CHECKING:
    from .core.events import Event, EventType
    from .core.events import MarketDataOHLCVEvent as MarketDataEvent

    # Type aliases for backtesting
    BacktestHistoricalDataProvider = "BacktestHistoricalDataProviderImpl"
    PubSubManager = "PubSubManager"
    ExchangeInfoService = "ExchangeInfoService"
    LoggerService = "LoggerService[Any]"
    MarketPriceService = "MarketPriceService"
    PortfolioManager = "PortfolioManager"
    ExecutionHandler = "ExecutionHandler"
    FeatureEngine = "FeatureEngine"
    PredictionService = "PredictionService"
    RiskManager = "RiskManager"
    StrategyArbitrator = "StrategyArbitrator"
    SimulatedMarketPriceService = "SimulatedMarketPriceService"

# Runtime imports
# Import service implementations with type checking only
if TYPE_CHECKING:
    from gal_friday.feature_engine import FeatureEngine as FeatureEngineImpl
    from gal_friday.logger_service import LoggerService as LoggerServiceImpl
    from gal_friday.portfolio_manager import PortfolioManager as PortfolioManagerImpl
    from gal_friday.prediction_service import PredictionService as PredictionServiceImpl
    from gal_friday.risk_manager import RiskManager as RiskManagerImpl
    from gal_friday.simulated_execution_handler import (
        SimulatedExecutionHandler as SimulatedExecutionHandlerImpl,
    )
    from gal_friday.simulated_market_price_service import (
        SimulatedMarketPriceService as SimulatedMarketPriceServiceImpl,
    )
    from gal_friday.strategy_arbitrator import StrategyArbitrator as StrategyArbitratorImpl

    from .config_manager import ConfigManager as ConfigManagerImpl
    from .core.events import Event, EventType
    from .core.events import MarketDataOHLCVEvent as MarketDataEvent
else:
    # Define placeholder classes for runtime
    class BacktestHistoricalDataProviderImpl:
        """Placeholder for BacktestHistoricalDataProvider implementation."""

    class ConfigManagerImpl:
        """Placeholder for ConfigManager implementation."""

    class ExchangeInfoServiceImpl:
        """Placeholder for ExchangeInfoService implementation."""

    class FeatureEngineImpl:
        """Placeholder for FeatureEngine implementation."""

    class LoggerServiceImpl:
        """Placeholder for LoggerService implementation."""

    class PortfolioManagerImpl:
        """Placeholder for PortfolioManager implementation."""

    class PredictionServiceImpl:
        """Placeholder for PredictionService implementation."""

    class RiskManagerImpl:
        """Placeholder for RiskManager implementation."""

    class SimulatedExecutionHandlerImpl:
        """Placeholder for SimulatedExecutionHandler implementation."""

    class SimulatedMarketPriceServiceImpl:
        """Placeholder for SimulatedMarketPriceService implementation."""

    class StrategyArbitratorImpl:
        """Placeholder for StrategyArbitrator implementation."""


# Define SignalEvent if not available
if not TYPE_CHECKING:

    class SignalEvent(Event):
        """Signal event for trading signals."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            """Initialize SignalEvent with any arguments.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.
            """
            super().__init__(*args, **kwargs)

    # Define missing event classes
    class Event:
        """Base class for all events in the backtesting engine."""

    class MarketDataEvent:
        """Event representing market data updates in the backtesting engine."""

    class BacktestHistoricalDataProvider:
        """Placeholder for BacktestHistoricalDataProvider if not defined."""

    # SignalEvent is defined later in the file

    class EventType(Enum):
        """Event types for the backtesting engine."""

        MARKET_DATA = "market_data"
        SIGNAL = "signal"


if TYPE_CHECKING:
    from typing import Protocol as TypingProtocol
else:
    try:
        from typing_extensions import Protocol as TypingProtocol  # type: ignore
    except ImportError:
        from typing import Protocol as TypingProtocol  # type: ignore

# Alias for Protocol to avoid conflicts
Protocol = TypingProtocol

# Type variable for generic types
T = TypeVar("T")
KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type
T_co = TypeVar("T_co", covariant=True)  # Covariant type variable
V_co = TypeVar("V_co", covariant=True)  # Covariant type variable

# Type aliases
ConfigType = dict[str, Any]
Timestamp = datetime | float | int | str
Number = int | float | Decimal
Price = Decimal | float | int
Quantity = Decimal | float | int
Symbol = str
ExchangeID = str
OrderID = str
TradeID = str
PositionID = str
AccountID = str
StrategyID = str
ModelID = str
FeatureName = str
FeatureValue = Number | str | bool | None
FeatureVector = dict[FeatureName, FeatureValue]
FeatureMatrix = list[FeatureVector]
Timeframe = str
OrderSide = str  # 'buy' or 'sell'
OrderType = str  # 'market', 'limit', 'stop', etc.
OrderStatus = str  # 'new', 'filled', 'canceled', etc.
TradeDirection = str  # 'long' or 'short'
ExchangeName = str

# Re-export commonly used types from typing module
DictStrAny = dict[str, Any]
ListStr = list[str]
DictStrNum = dict[str, Number]
DictStrStr = dict[str, str]
ListDictStrAny = list[dict[str, Any]]
CallableT = TypeVar("CallableT", bound=Callable[..., Any])

# Generic function type for event handlers
EventHandler = Callable[[Event], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]

# Type for progress callback
ProgressCallback = Callable[[float, str], None]


class LoggerAdapterType(logging.LoggerAdapter, Generic[T]):
    """A generic logger adapter that adds type information to log messages.

    This adapter extends logging.LoggerAdapter to work with generic types,
    allowing for type-safe logging with additional context.
    """


# Import TA-Lib for technical indicators
try:
    import talib as ta
except ImportError:
    log = logging.getLogger(__name__)
    log.warning("TA-Lib not installed. ATR calculation will not work.")
    # Create a minimal placeholder for ta module

    class TaLib:
        """Provide minimal placeholder for TA-Lib functionality when library is not available."""

        @staticmethod
        def atr(high: pd.Series, _low: pd.Series, _close: pd.Series, _length: int) -> pd.Series:
            """Return a series of ATR values or None when TA-Lib is not installed."""
            log.exception("TA-Lib not installed. Cannot calculate ATR.")
            # Return Series with same index to avoid potential issues later
            return pd.Series([None] * len(high), index=high.index)

    ta = TaLib()


# Define stubs for optional dependencies
class PubSubManagerBase:  # type: ignore
    """Base stub for PubSubManager when not available."""


class RiskManagerBase:  # type: ignore
    """Base stub for RiskManager when not available."""


# Import optional dependencies if available
if "PubSubManager" not in globals():
    PubSubManager: type[PubSubManagerBase] = PubSubManagerBase  # type: ignore

if "RiskManager" not in globals():
    RiskManager: type[RiskManagerBase] = RiskManagerBase  # type: ignore

try:
    from gal_friday.core.pubsub import PubSubManager as _PubSubManager  # type: ignore

    if "PubSubManager" in globals() and globals()["PubSubManager"] is not Any:  # type: ignore
        PubSubManager = _PubSubManager  # type: ignore
except ImportError:
    pass

try:
    from gal_friday.risk_manager import RiskManager as _RiskManager  # type: ignore

    if "RiskManager" in globals() and globals()["RiskManager"] is not Any:  # type: ignore
        RiskManager = _RiskManager  # type: ignore
except ImportError:
    pass


# Configure logging
log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .core.events import Event

    class ExchangeInfoServiceImpl:  # type: ignore
        """Stub implementation of ExchangeInfoService for type checking."""


# Define missing event classes if they don't exist in core.events
class SignalEvent(Event):
    """Signal event for trading signals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize SignalEvent with any additional keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments to pass to parent class.
        """
        super().__init__(*args, **kwargs)


class BacktestHistoricalDataProviderImpl:
    """Provides historical data access for backtesting components."""

    def __init__(
        self,
        all_historical_data: dict[str, pd.DataFrame],
        logger: logging.Logger,
    ) -> None:
        self._data: dict[str, pd.DataFrame] = all_historical_data
        self.logger = logger
        self._validate_data()

    def _validate_data(self) -> None:
        """Validate the loaded historical data."""
        if not self._data:
            msg = "No historical data provided"
            raise ValueError(msg)
        for pair, df in self._data.items():
            if not isinstance(df, pd.DataFrame):
                msg = f"Expected DataFrame for pair {pair}, got {type(df)}"
                raise TypeError(msg)
            required_columns = {"open", "high", "low", "close", "volume"}
            missing = required_columns - set(df.columns)
            if missing:
                msg = f"Missing required columns {missing} for pair {pair}"
                raise ValueError(msg)

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> pd.Series | None:
        """Get the next bar after the given timestamp for the trading pair.

        Args:
            trading_pair: The trading pair to get the next bar for
            timestamp: The reference timestamp

        Returns:
        -------
            The next bar as a pandas Series, or None if no more bars
        """
        try:
            if trading_pair not in self._data:
                self.logger.warning("No data for trading pair: %s", trading_pair)
                return None

            df = self._data[trading_pair]
            next_bars = df[df.index > timestamp].head(1)
            return next_bars.iloc[0] if not next_bars.empty else None

        except Exception:
            self.logger.exception(
                "Error getting next bar for %s at %s",
                trading_pair,
                timestamp,
            )
            return None

    def get_atr(self, trading_pair: str, timestamp: datetime, period: int = 14) -> float | None:
        """Get the ATR (Average True Range) value for the given timestamp.

        Args:
            trading_pair: The trading pair to get ATR for
            timestamp: The timestamp to get ATR at
            period: The ATR period (default: 14)

        Returns:
        -------
            The ATR value as a float, or None if not available
        """
        MIN_BARS_FOR_ATR = 2  # noqa: N806
        try:
            if trading_pair not in self._data:
                self.logger.warning("No data for trading pair: %s", trading_pair)
                return None

            df = self._data[trading_pair]

            # If ATR column exists, use it directly
            if "atr" in df.columns:
                atr_series = df.loc[df.index <= timestamp, "atr"]
                if not atr_series.empty:
                    return float(atr_series.iloc[-1])

            # Otherwise calculate simple ATR if possible
            required_cols = {"high", "low", "close"}
            if not required_cols.issubset(df.columns):
                self.logger.warning("Missing required columns for ATR calculation")
                return None

            # Simple ATR calculation (not as accurate as TA-Lib)
            df_slice = df[df.index <= timestamp].tail(period + 1)
            if len(df_slice) < MIN_BARS_FOR_ATR:
                return None

            tr = pd.DataFrame()
            tr["h-l"] = df_slice["high"] - df_slice["low"]
            tr["h-pc"] = abs(df_slice["high"] - df_slice["close"].shift(1))
            tr["l-pc"] = abs(df_slice["low"] - df_slice["close"].shift(1))
            tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)

            atr = tr["tr"].rolling(window=period).mean().iloc[-1]
            return float(atr) if pd.notnull(atr) else None

        except Exception:
            self.logger.exception(
                "Error calculating ATR for %s at %s",
                trading_pair,
                timestamp,
            )
            return None

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Retrieve historical OHLCV data for a given pair and time range.

        Args:
            trading_pair: The trading pair to get data for
            start_time: Start of the time range
            end_time: End of the time range
            interval: Data interval (e.g., '1d', '1h', '1m')

        Returns:
        -------
            DataFrame with OHLCV data or None if no data available.
        """
        try:
            self.logger.debug(
                "BacktestHistoricalDataProvider.get_historical_ohlcv called for %s.",
                trading_pair,
            )

            if trading_pair not in self._data:
                self.logger.warning("No data for trading pair: %s", trading_pair)
                return None

            pair_df = self._data[trading_pair]

            # Ensure start_time and end_time are timezone-aware if df.index is
            if pair_df.index.tz is not None:
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=pair_df.index.tz)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=pair_df.index.tz)

            # Ensure we're using proper datetime indexing
            if not isinstance(pair_df.index, pd.DatetimeIndex):
                self.logger.warning("DataFrame index is not a DatetimeIndex, cannot slice by time")
                return None

            mask = (pair_df.index >= start_time) & (pair_df.index <= end_time)
            filtered = pair_df.loc[mask].copy()

            # Resample to requested interval if needed
            result = None
            if interval != "1d":  # Assuming 1d is the base interval
                try:
                    resampled = (
                        filtered.resample(interval)
                        .agg(
                            {
                                "open": "first",
                                "high": "max",
                                "low": "min",
                                "close": "last",
                                "volume": "sum",
                            },
                        )
                        .dropna()
                    )
                    if not resampled.empty:
                        result = resampled
                except Exception as resample_error:
                    self.logger.warning(
                        "Error resampling data to %s interval: %s",
                        interval,
                        str(resample_error),
                    )
                    # If resampling fails, fall through to return filtered data if not empty

        except Exception:
            self.logger.exception(
                "Error getting historical OHLCV for %s",
                trading_pair,
            )
            return None

        # Return the appropriate result based on the processing
        if interval != "1d" and result is not None:
            return result
        return None if filtered.empty else filtered

    async def get_historical_trades(
        self,
        _trading_pair: str,  # Unused, kept for interface compatibility
        _start_time: datetime,  # Unused, kept for interface compatibility
        _end_time: datetime,  # Unused, kept for interface compatibility
    ) -> pd.DataFrame | None:
        """Retrieve historical trade data for a given pair and time range.

        Note:
            This is a placeholder method as the backtest provider doesn't have trade data.

        Returns:
        -------
            Always returns None as trade data is not available in backtest mode.
        """
        self.logger.warning("Historical trade data not available in backtest mode")
        return None


# --- Helper Function for Reporting --- #

# Helper functions for calculate_performance_metrics


def _calculate_basic_returns_and_equity(
    equity_curve: pd.Series,
    initial_capital: Decimal,
    results: dict,
) -> float | None:
    """Calculate basic returns and equity metrics."""
    final_equity_value = equity_curve.iloc[-1]
    final_equity = Decimal(str(final_equity_value))
    results["initial_capital"] = float(initial_capital)
    results["final_equity"] = float(final_equity)

    initial_capital_float = float(initial_capital)
    final_equity_float = float(final_equity)
    total_return_pct: float | None = None
    if initial_capital_float > 0:
        total_return_pct = ((final_equity_float / initial_capital_float) - 1.0) * 100.0
        results["total_return_pct"] = total_return_pct
    else:
        results["total_return_pct"] = 0.0
        total_return_pct = 0.0  # Ensure it's assigned for return type
    return total_return_pct


def _calculate_annualized_return(
    equity_curve: pd.Series,
    total_return_pct: float | None,
    results: dict,
) -> None:
    """Calculate annualized return."""
    if total_return_pct is None:
        results["annualized_return_pct"] = None
        return
    try:
        min_points_for_duration = 2
        if len(equity_curve) >= min_points_for_duration:
            first_date = equity_curve.index[0]
            last_date = equity_curve.index[-1]
            if isinstance(first_date, (pd.Timestamp, datetime)) and isinstance(
                last_date, (pd.Timestamp, datetime),
            ):
                duration_days = (last_date - first_date).total_seconds() / (60 * 60 * 24)
                if duration_days > 0:
                    total_return_factor = 1.0 + (total_return_pct / 100.0)
                    annualized_return = (
                        (total_return_factor ** (365.0 / duration_days)) - 1.0
                    ) * 100.0
                    results["annualized_return_pct"] = annualized_return
                    log.debug(
                        "Calculated annualized return over %.2f days: %.2f%%",
                        duration_days,
                        annualized_return,
                    )
                else:
                    log.warning("Backtest duration <= 0 days, cannot calculate annualized return.")
                    results["annualized_return_pct"] = total_return_pct
            else:
                log.warning("Equity curve index is not timestamp type for annualized return.")
                results["annualized_return_pct"] = None
        else:
            log.warning("Insufficient data for annualized return.")
            results["annualized_return_pct"] = None
    except Exception:
        log.exception("Error calculating annualized return")
        results["annualized_return_pct"] = None


def _calculate_drawdown_metrics(equity_curve: pd.Series, results: dict) -> None:
    """Calculate drawdown metrics."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    results["max_drawdown_pct"] = abs(max_drawdown * 100.0)


def _calculate_risk_adjusted_metrics(returns: pd.Series, results: dict) -> None:
    """Calculate Sharpe and Sortino ratios."""
    returns_std = returns.std()
    if not returns.empty and returns_std != 0:
        sharpe_ratio = (returns.mean() / returns_std) * np.sqrt(252)
        results["sharpe_ratio_annualized_approx"] = float(sharpe_ratio)
    else:
        results["sharpe_ratio_annualized_approx"] = 0.0

    downside_returns = returns[returns < 0]
    if not downside_returns.empty:
        downside_std = downside_returns.std()
        if downside_std != 0:
            sortino_ratio = returns.mean() / downside_std * np.sqrt(252)
            results["sortino_ratio_annualized_approx"] = float(sortino_ratio)
        else:
            results["sortino_ratio_annualized_approx"] = np.inf
    else:
        results["sortino_ratio_annualized_approx"] = np.inf


def _calculate_trade_statistics(trade_log: list[dict[str, Any]], results: dict) -> None:
    """Calculate various trade statistics."""
    num_trades = len(trade_log)
    results["total_trades"] = num_trades
    if num_trades > 0:
        pnl_list = [Decimal(str(trade.get("pnl", 0))) for trade in trade_log]
        results["total_pnl"] = float(sum(pnl_list))
        winning_trades = [pnl for pnl in pnl_list if pnl > 0]
        losing_trades = [pnl for pnl in pnl_list if pnl < 0]
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        results["winning_trades"] = num_wins
        results["losing_trades"] = num_losses
        results["win_rate_pct"] = float(num_wins / num_trades * 100) if num_trades > 0 else 0.0
        gross_profit = sum(winning_trades)
        gross_loss = abs(sum(losing_trades))
        results["gross_profit"] = float(str(gross_profit))
        results["gross_loss"] = float(str(gross_loss))
        gross_profit_float = float(str(gross_profit)) if gross_profit else 0.0
        gross_loss_float = float(str(gross_loss)) if gross_loss else 0.0
        if gross_loss_float > 0:
            results["profit_factor"] = gross_profit_float / gross_loss_float
        else:
            results["profit_factor"] = float("inf")
        results["average_trade_pnl"] = float(str(sum(pnl_list) / num_trades))
        results["average_win"] = (
            float(str(sum(winning_trades) / num_wins)) if num_wins > 0 else 0.0
        )
        results["average_loss"] = (
            float(str(sum(losing_trades) / num_losses)) if num_losses > 0 else 0.0
        )
        avg_win = results["average_win"]
        avg_loss = results["average_loss"]
        if avg_loss != 0:
            results["avg_win_loss_ratio"] = float(str(abs(avg_win / avg_loss)))
        else:
            results["avg_win_loss_ratio"] = float("inf")
    else:
        # Default values if no trades occurred
        default_trade_stats = {
            "total_pnl": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "average_trade_pnl": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "avg_win_loss_ratio": 0.0,
        }
        results.update(default_trade_stats)


def _calculate_average_holding_period(trade_log: list[dict[str, Any]], results: dict) -> None:
    """Calculate average holding period of trades."""
    results["average_holding_period_hours"] = None
    results["average_holding_period_days"] = None
    if not trade_log:  # No trades, no holding period
        return
    try:
        holding_periods = []
        for trade in trade_log:
            if trade.get("entry_time") and trade.get("exit_time"):
                try:
                    entry_time = pd.to_datetime(trade["entry_time"])
                    exit_time = pd.to_datetime(trade["exit_time"])
                    duration_hours = (exit_time - entry_time).total_seconds() / 3600
                    holding_periods.append(duration_hours)
                except Exception as e:
                    log.warning("Error parsing trade times for holding period: %s", e)

        if holding_periods:
            avg_holding_period = sum(holding_periods) / len(holding_periods)
            results["average_holding_period_hours"] = avg_holding_period
            results["average_holding_period_days"] = avg_holding_period / 24
            log.debug("Average holding period: %.2f hours", avg_holding_period)
        else:
            log.warning("No valid trade durations for avg holding period calculation.")
    except Exception:
        log.exception("Error calculating average holding period")


def calculate_performance_metrics(
    equity_curve: pd.Series,
    trade_log: list[dict[str, Any]],
    initial_capital: Decimal,
) -> dict[str, Any]:
    """Calculate standard backtesting performance metrics by orchestrating helper functions."""
    if equity_curve.empty:
        log.warning("Equity curve is empty, cannot calculate metrics.")
        return {"error": "Equity curve is empty, cannot calculate metrics."}

    results: dict[str, Any] = {}
    equity_curve = pd.to_numeric(equity_curve, errors="coerce").astype(float)
    equity_curve = equity_curve.dropna()
    if equity_curve.empty:
        log.warning("Equity curve became empty after numeric conversion.")
        return {"error": "Equity curve has no valid numeric data."}

    returns = equity_curve.pct_change().dropna()

    total_return_pct = _calculate_basic_returns_and_equity(equity_curve, initial_capital, results)
    _calculate_annualized_return(equity_curve, total_return_pct, results)
    _calculate_drawdown_metrics(equity_curve, results)
    _calculate_risk_adjusted_metrics(returns, results)
    _calculate_trade_statistics(trade_log, results)  # This helper handles the num_trades > 0 logic
    _calculate_average_holding_period(trade_log, results)

    return results


class ConfigManagerProtocol(Protocol):
    """Protocol defining the required interface for config manager."""

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        """Get a configuration value by key.

        Args:
            key: The configuration key to retrieve.
            default: Default value to return if key is not found.

        Returns:
        -------
            The configuration value or default if key not found.
        """

    def __getitem__(self, key: str) -> ConfigValue:
        """Get a configuration value by key using dictionary-style access.

        Args:
            key: The configuration key to retrieve.

        Returns:
        -------
            The configuration value.
        """

    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists.

        Args:
            key: The configuration key to check.

        Returns:
        -------
            True if the key exists, False otherwise.
        """

    def get_all(self) -> dict[str, ConfigValue]:
        """Get all configuration values.

        Returns:
        -------
            A dictionary containing all configuration values.
        """

    def set(self, key: str, value: ConfigValue) -> None:
        """Set a configuration value.

        Args:
            key: The configuration key to set.
            value: The value to set.
        """


# Type aliases for service classes
if TYPE_CHECKING:
    from gal_friday.backtest_historical_data_provider import (
        BacktestHistoricalDataProvider as _BacktestHistoricalDataProvider,
    )
    from gal_friday.core.pubsub import PubSubManager as _PubSubManager
    from gal_friday.exchange_info_service import ExchangeInfoService as _ExchangeInfoService
    from gal_friday.execution_handler import ExecutionHandler as _ExecutionHandler
    from gal_friday.feature_engine import FeatureEngine as _FeatureEngine
    from gal_friday.logger_service import LoggerService as _LoggerService
    from gal_friday.market_price_service import MarketPriceService as _MarketPriceService
    from gal_friday.portfolio_manager import PortfolioManager as _PortfolioManager
    from gal_friday.prediction_service import PredictionService as _PredictionService
    from gal_friday.risk_manager import RiskManager as _RiskManager
    from gal_friday.simulated_market_price_service import (
        SimulatedMarketPriceService as _SimulatedMarketPriceService,
    )
    from gal_friday.strategy_arbitrator import StrategyArbitrator as _StrategyArbitrator
else:
    # Define dummy types for runtime
    class _LoggerService:
        ...

    class _MarketPriceService:
        ...

    class _SimulatedMarketPriceService:
        ...

    class _PortfolioManager:
        ...

    class _FeatureEngine:
        ...

    class _PredictionService:
        ...

    class _RiskManager:
        ...

    class _StrategyArbitrator:
        ...

    class _BacktestHistoricalDataProvider:
        ...

    class _ExchangeInfoService:
        ...

    class _PubSubManager:
        ...

    class _ExecutionHandler:
        ...


class BacktestingEngine:
    """Orchestrates backtesting simulations using historical data."""

    def __init__(
        self,
        config: ConfigManagerProtocol,
        data_dir: str = "data",
        results_dir: str = "results",
        max_workers: int = 4,
    ) -> None:
        """Initialize the BacktestingEngine.

        Args:
            config: Configuration manager instance
            data_dir: Directory to load historical data from
            results_dir: Directory to save backtest results to
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.max_workers = max_workers

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize data storage
        self._data: dict[str, pd.DataFrame] = {}
        self._current_step = 0
        self._current_time: datetime | None = None
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

        # Initialize services with proper type annotations
        self.pubsub_manager: _PubSubManager | None = None
        self.logger_service: _LoggerService | None = None
        self.historical_data_provider: _BacktestHistoricalDataProvider | None = None
        self.market_price_service: _MarketPriceService | None = None
        self.portfolio_manager: _PortfolioManager | None = None
        self.execution_handler: _ExecutionHandler | None = None
        self.feature_engine: _FeatureEngine | None = None
        self.prediction_service: _PredictionService | None = None
        self.risk_manager: _RiskManager | None = None
        self.strategy_arbitrator: _StrategyArbitrator | None = None
        self.exchange_info_service: _ExchangeInfoService | None = None

        # Attribute to store the execution report handler for unsubscribing
        self._backtest_exec_report_handler: None | (
            Callable[..., Coroutine[Any, Any, None]]
        ) = None

        log.info("BacktestingEngine initialized.")

    def _get_backtest_config(self) -> dict[str, Any]:
        """Get backtest configuration from the config manager."""
        config = self.config.get("backtest", {})
        if not isinstance(config, dict):
            log.warning("Backtest config is not a dictionary, returning empty dict")
            return {}
        return config

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Validate the backtest configuration."""
        required = ["data_path", "start_date", "end_date", "trading_pairs"]
        missing = [field for field in required if field not in config]
        if missing:
            log.error("Missing required config fields: %s", ", ".join(missing))
            return False
        return True

    def _load_raw_data(self, data_path: str) -> dict[str, pd.DataFrame] | None:
        """Load raw data from the specified path."""
        try:
            path = Path(data_path)
            if not path.exists():
                log.error("Data file not found: %s", data_path)
                return None

            # Assuming CSV format with 'pair' column for multiple pairs
            df = pd.read_csv(path, parse_dates=["timestamp"])
            if "pair" not in df.columns:
                log.error("Data file must contain 'pair' column")
                return None

            return dict(df.groupby("pair"))

        except Exception:
            log.exception("Error loading data from %s", data_path)
            return None

    def _clean_and_validate_data(
        self,
        data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame] | None:
        """Clean and validate the loaded data."""
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            cleaned = {}
            for pair, df in data.items():
                # Ensure required columns exist
                required_cols = {"open", "high", "low", "close", "volume"}
                if not required_cols.issubset(df.columns):
                    log.warning("Missing required columns in data for %s", pair)
                    continue

                # Filter by date range
                filtered_df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
                if filtered_df.empty:
                    log.warning("No data in date range for %s", pair)
                else:
                    cleaned[pair] = filtered_df

            return self._get_result_or_none(cleaned)

        except Exception:
            log.exception("Error cleaning and validating data")
            return None

    def _get_result_or_none(
        self,
        value: dict[str, pd.DataFrame] | None,
    ) -> dict[str, pd.DataFrame] | None:
        """Return the value if truthy, otherwise None.

        This helper function is used to satisfy the TRY300 rule by moving the
        return logic to a separate function.

        Args:
            value: The value to check and return if truthy.

        Returns:
        -------
            The original value if truthy, otherwise None.
        """
        return value if value else None

    def _process_pairs_data(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame] | None:
        """Process data for each trading pair.

        Args:
            data: Dictionary mapping trading pairs to their DataFrames
            config: Configuration dictionary for processing

        Returns:
        -------
            Dictionary of processed DataFrames or None if processing fails
        """
        try:
            processed: dict[str, pd.DataFrame] = {}
            for pair, df in data.items():
                # Ensure we have the required columns
                required_cols = {"open", "high", "low", "close", "volume"}
                if not required_cols.issubset(df.columns):
                    log.warning("Skipping pair %s: missing required columns", pair)
                    continue

                # Make a copy to avoid modifying the original data
                processed_df = df.copy()

                # Add any additional processing here
                # For example, technical indicators, feature engineering, etc.

                # Only add to processed if the DataFrame is not empty
                if not processed_df.empty:
                    processed[pair] = processed_df
                else:
                    log.warning("Processed DataFrame is empty for pair %s", pair)

            return self._get_result_or_none(processed)

        except Exception:
            log.exception("Error processing pairs data")
            return None

    def _process_and_save_results(
        self,
        services: dict[str, Any],
        run_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Process and save backtest results.

        Args:
            services: Dictionary of initialized services
            run_config: Configuration for the backtest run

        Returns:
        -------
            Dictionary containing the results of the backtest
        """
        results: dict[str, Any] = {}

        try:
            # Get portfolio manager to access final portfolio state
            portfolio_manager = services.get("portfolio_manager")
            if portfolio_manager and hasattr(portfolio_manager, "get_portfolio_summary"):
                portfolio_summary = portfolio_manager.get_portfolio_summary()
                results["portfolio_summary"] = portfolio_summary

            # Get trade history if available
            execution_handler = services.get("execution_handler")
            if execution_handler and hasattr(execution_handler, "get_trade_history"):
                trade_history = execution_handler.get_trade_history()
                results["trade_history"] = trade_history

            # Calculate performance metrics
            if "portfolio_summary" in results and "equity_curve" in services:
                equity_curve = services["equity_curve"]
                initial_capital = Decimal(str(run_config.get("initial_capital", 10000)))
                metrics = calculate_performance_metrics(
                    equity_curve=equity_curve,
                    trade_log=results.get("trade_history", []),
                    initial_capital=initial_capital,
                )
                results["metrics"] = metrics

            # Save results to file
            output_dir = Path(run_config.get("output_dir", "results"))
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"backtest_results_{timestamp}.json"

            # Save results to JSON file with decimal conversion
            with results_file.open("w") as f:
                json.dump(results, f, default=decimal_to_float, indent=2)

            log.info("Backtest results saved to %s", results_file)

        except Exception as e:
            log.exception("Error processing and saving results")
            # Include error in results instead of raising to ensure we always return a dict
            results["error"] = str(e)

        return results

    async def _execute_simulation(  # noqa: PLR0912
        self,
        services: dict[str, Any],
        run_config: dict[str, Any],
    ) -> None:
        """Execute the backtest simulation with proper time-series iteration.

        Args:
            services: Dictionary of initialized services
            run_config: Configuration for the backtest run
        """
        log.info("Starting backtest simulation")

        # Get configuration
        trading_pairs = run_config.get("trading_pairs", [])
        start_date = pd.to_datetime(run_config["start_date"])
        end_date = pd.to_datetime(run_config["end_date"])

        # Get services
        market_price_service = services.get("market_price_service")
        portfolio_manager = services.get("portfolio_manager")
        execution_handler = services.get("execution_handler")
        feature_engine = services.get("feature_engine")
        prediction_service = services.get("prediction_service")
        strategy_arbitrator = services.get("strategy_arbitrator")
        risk_manager = services.get("risk_manager")

        if not all([market_price_service, portfolio_manager, execution_handler]):
            raise ValueError("Required services not available for simulation")

        # Start all services
        for service_name, service in services.items():
            if hasattr(service, "start") and callable(service.start):
                try:
                    log.info("Starting service: %s", service_name)
                    if asyncio.iscoroutinefunction(service.start):
                        await service.start()
                    else:
                        result = service.start()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception:
                    log.exception("Error starting service %s", service_name)
                    raise

        try:
            # Get unified timeline from all trading pairs
            all_timestamps = set()
            for pair in trading_pairs:
                if pair in self._data:
                    pair_data = self._data[pair]
                    pair_timestamps = pair_data[
                        (pair_data["timestamp"] >= start_date) &
                        (pair_data["timestamp"] <= end_date)
                    ]["timestamp"]
                    all_timestamps.update(pair_timestamps)

            if not all_timestamps:
                log.error("No data available for simulation period")
                return

            # Sort timestamps for proper time progression
            sorted_timestamps = sorted(all_timestamps)
            log.info(f"Simulation will process {len(sorted_timestamps)} time steps")

            # Initialize equity curve tracking
            equity_curve_data = []

            # Main simulation loop - iterate through time
            for i, current_timestamp in enumerate(sorted_timestamps):
                self._current_step = i
                self._current_time = current_timestamp

                log.debug(f"Processing timestamp {current_timestamp} (step {i})")

                try:
                    # 1. Update market price service with current timestamp
                    if market_price_service is not None and hasattr(market_price_service, "update_time"):
                        market_price_service.update_time(current_timestamp)

                    # 2. Generate market data events for all pairs at this timestamp
                    for pair in trading_pairs:
                        await self._process_market_data_for_timestamp(
                            pair, current_timestamp, services,
                        )

                    # 3. Process any pending limit orders and stop-loss/take-profit
                    if execution_handler is not None and hasattr(execution_handler, "check_active_limit_orders"):
                        # Get current bar data for limit order processing
                        for pair in trading_pairs:
                            current_bar = self._get_bar_at_timestamp(pair, current_timestamp)
                            if current_bar is not None:
                                await execution_handler.check_active_limit_orders(
                                    current_bar, current_timestamp,
                                )
                                if hasattr(execution_handler, "check_active_sl_tp"):
                                    await execution_handler.check_active_sl_tp(
                                        current_bar, current_timestamp,
                                    )

                    # 4. Update portfolio value and record equity curve
                    if portfolio_manager is not None and hasattr(portfolio_manager, "get_current_state"):
                        portfolio_state = portfolio_manager.get_current_state()
                        if portfolio_state and "total_value" in portfolio_state:
                            equity_curve_data.append({
                                "timestamp": current_timestamp,
                                "equity": portfolio_state["total_value"],
                            })

                    # 5. Small delay to prevent overwhelming the event loop
                    if i % 1000 == 0:  # Every 1000 steps
                        await asyncio.sleep(0.001)  # Minimal delay

                except Exception as e:
                    log.error(f"Error processing timestamp {current_timestamp}: {e}")
                    # Continue processing unless it's a critical error
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    continue

            # Store final equity curve in services for metrics calculation
            if equity_curve_data:
                equity_df = pd.DataFrame(equity_curve_data)
                equity_df.set_index("timestamp", inplace=True)
                services["equity_curve"] = equity_df["equity"]
                log.info(f"Simulation completed. Final equity: {equity_df['equity'].iloc[-1]}")
            else:
                log.warning("No equity curve data collected during simulation")

        except asyncio.CancelledError:
            log.info("Simulation cancelled")
            raise
        except Exception:
            log.exception("Error during simulation")
            raise
        finally:
            # Stop all services in reverse order
            service_names = list(services)
            for service_name in reversed(service_names):
                service = services[service_name]
                if hasattr(service, "stop") and callable(service.stop):
                    try:
                        log.info("Stopping %s...", service_name)
                        if asyncio.iscoroutinefunction(service.stop):
                            await service.stop()
                        else:
                            result = service.stop()
                            if asyncio.iscoroutine(result):
                                await result
                    except Exception:
                        log.exception("Error stopping %s", service_name)

            # Clean up process pool if it exists
            if "prediction_service" in services:
                prediction_service = services["prediction_service"]
                has_pool = hasattr(prediction_service, "process_pool_executor")
                if prediction_service is not None and has_pool:
                    executor = getattr(prediction_service, "process_pool_executor", None)
                    if executor is not None and hasattr(executor, "shutdown"):
                        try:
                            executor.shutdown(wait=True)
                        except Exception:
                            log.exception("Error shutting down process pool")

            log.info("All services shut down")

    async def _process_market_data_for_timestamp(
        self,
        trading_pair: str,
        timestamp: datetime,
        services: dict[str, Any],
    ) -> None:
        """Process market data for a specific trading pair and timestamp.
        
        Args:
            trading_pair: The trading pair to process
            timestamp: Current simulation timestamp
            services: Dictionary of initialized services
        """
        try:
            # Get bar data for this timestamp
            bar_data = self._get_bar_at_timestamp(trading_pair, timestamp)
            if bar_data is None:
                return  # No data for this pair at this timestamp

            # Create and publish market data event
            from .core.events import MarketDataOHLCVEvent

            market_event = MarketDataOHLCVEvent(
                source_module=self.__class__.__name__,
                event_id=uuid.uuid4(),
                timestamp=timestamp,
                trading_pair=trading_pair,
                exchange="simulated",  # Add required exchange field
                interval="1d",  # Add required interval field
                timestamp_bar_start=timestamp,  # Add required timestamp_bar_start field
                open=str(bar_data.get("open", 0)),
                high=str(bar_data.get("high", 0)),
                low=str(bar_data.get("low", 0)),
                close=str(bar_data.get("close", 0)),
                volume=str(bar_data.get("volume", 0)),
            )

            # Send to feature engine if available
            feature_engine = services.get("feature_engine")
            if feature_engine and hasattr(feature_engine, "handle_market_data_event"):
                await feature_engine.handle_market_data_event(market_event)

        except Exception as e:
            log.error(f"Error processing market data for {trading_pair} at {timestamp}: {e}")

    def _get_bar_at_timestamp(self, trading_pair: str, timestamp: datetime) -> pd.Series | None:
        """Get OHLCV bar data for a specific trading pair and timestamp.
        
        Args:
            trading_pair: Trading pair to get data for
            timestamp: Timestamp to look up
            
        Returns:
            Series containing OHLCV data or None if not found
        """
        try:
            if trading_pair not in self._data:
                return None

            pair_data = self._data[trading_pair]

            # Find exact timestamp match
            matching_rows = pair_data[pair_data["timestamp"] == timestamp]

            if matching_rows.empty:
                return None

            return matching_rows.iloc[0]

        except Exception as e:
            log.error(f"Error getting bar data for {trading_pair} at {timestamp}: {e}")
            return None
