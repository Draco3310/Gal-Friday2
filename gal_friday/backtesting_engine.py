"""Provide a backtesting environment for algorithmic trading strategies.

This module contains the BacktestingEngine, which orchestrates backtesting simulations
using historical data. It handles loading data, initializing core simulation services
(like FeatureRegistryClient, LoggerService, PubSubManager, and FeatureEngine),
executing the simulation by processing historical data through the FeatureEngine,
and managing the overall backtest lifecycle.

The engine is designed to allow other components (e.g., PredictionService,
StrategyArbitrator, PortfolioManager, RiskManager, ExecutionHandler) to be
injected (typically pre-initialized with shared core services) and participate
in the event-driven simulation.
"""

# Standard library imports
from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
import datetime as dt  # For dt.datetime and dt.timezone.utc
from decimal import Decimal
from enum import Enum
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
import uuid

import asyncio

# Third-party imports
import numpy as np
import pandas as pd

# Local application imports
from .core.events import Event, EventType  # E402: Moved up

# Type[Any] variables for generic types - defined at module level
ConfigValue = str | int | float | bool | dict[str, Any] | list[Any] | None  # Type[Any] alias for config values
_T = TypeVar("_T")  # Generic type variable for protocols


def decimal_to_float(obj: Any) -> float:  # noqa: ANN401
    """Convert Decimal to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(msg)


# Initialize logger
log = logging.getLogger(__name__)

# Type[Any] checking imports
if TYPE_CHECKING:
    # Import core types

    # Import implementation types
    from gal_friday.core.feature_registry_client import (
        FeatureRegistryClient,  # Added import
    )
    from gal_friday.feature_engine import FeatureEngine  # Actual FeatureEngine
    from gal_friday.logger_service import LoggerService  # Actual LoggerService
    from gal_friday.portfolio_manager import PortfolioManager as _PortfolioManager
    from gal_friday.prediction_service import PredictionService as _PredictionService
    from gal_friday.risk_manager import RiskManager as _RiskManager
    from gal_friday.simulated_market_price_service import (
        SimulatedMarketPriceService as _SimulatedMarketPriceService,
    )
    from gal_friday.strategy_arbitrator import StrategyArbitrator as _StrategyArbitrator

    # Define type aliases that use the implementation types
    LoggerServiceType = LoggerService
    SimulatedMarketPriceServiceType = _SimulatedMarketPriceService
    PortfolioManagerType = _PortfolioManager
    FeatureEngineType = FeatureEngine
    PredictionServiceType = _PredictionService
    RiskManagerType = _RiskManager
    StrategyArbitratorType = _StrategyArbitrator

    # Import Protocol for type checking
    from typing import (
        Protocol as ProtocolClass,  # Use different name to avoid confusion
    )
else:
    # Runtime Protocol handling - use typing_extensions for compatibility
    from collections.abc import Callable, Coroutine
    from typing import TYPE_CHECKING, Any, TypeVar

    from typing_extensions import Protocol as ProtocolClass

# Third-party imports

# Define a type for async functions
AsyncFunction = TypeVar("AsyncFunction", bound=Callable[..., Any])

# Type[Any] aliases

if TYPE_CHECKING:
    from .core.feature_registry_client import FeatureRegistryClient

# Runtime imports - Import service implementations
try:
    from gal_friday.feature_engine import FeatureEngine
except ImportError:
    logging.getLogger(__name__).warning("Failed to import FeatureEngine")
    FeatureEngine = None  # type: ignore[assignment,misc]

try:
    from gal_friday.logger_service import LoggerService
except ImportError:
    logging.getLogger(__name__).warning("Failed to import LoggerService")
    LoggerService = None  # type: ignore[assignment,misc]

try:
    from gal_friday.portfolio_manager import PortfolioManager
except ImportError:
    logging.getLogger(__name__).warning("Failed to import PortfolioManager")
    PortfolioManager = None  # type: ignore[assignment,misc]

try:
    from gal_friday.prediction_service import PredictionService
except ImportError:
    logging.getLogger(__name__).warning("Failed to import PredictionService")
    PredictionService = None  # type: ignore[assignment,misc]

try:
    from gal_friday.risk_manager import RiskManager
except ImportError:
    logging.getLogger(__name__).warning("Failed to import RiskManager")
    RiskManager = None  # type: ignore[assignment,misc]

try:
    from gal_friday.simulated_execution_handler import (
        SimulatedExecutionHandler as ExecutionHandler,
    )
except ImportError:
    logging.getLogger(__name__).warning("Failed to import SimulatedExecutionHandler")
    ExecutionHandler = None  # type: ignore[assignment,misc]

try:
    from gal_friday.simulated_market_price_service import SimulatedMarketPriceService
except ImportError:
    logging.getLogger(__name__).warning("Failed to import SimulatedMarketPriceService")
    SimulatedMarketPriceService = None  # type: ignore[assignment,misc]

try:
    from gal_friday.strategy_arbitrator import StrategyArbitrator
except ImportError:
    logging.getLogger(__name__).warning("Failed to import StrategyArbitrator")
    StrategyArbitrator = None  # type: ignore[assignment,misc]

try:
    from .config_manager import ConfigManager
except ImportError:
    logging.getLogger(__name__).warning("Failed to import ConfigManager")
    ConfigManager = None  # type: ignore[assignment,misc]

try:
    from .core.pubsub import PubSubManager
except ImportError:
    logging.getLogger(__name__).warning("Failed to import PubSubManager")
    PubSubManager = None  # type: ignore[assignment,misc]

# Import enhanced technical analysis
from .technical_analysis_enhanced import (
    IndicatorConfig,
    IndicatorType,
    TechnicalAnalysisManager,
)

# Global technical analysis manager (will be initialized when needed)
_ta_manager: TechnicalAnalysisManager | None = None

def get_ta_manager(config: dict[str, Any], logger: logging.Logger) -> TechnicalAnalysisManager:
    """Get or create the global technical analysis manager."""
    global _ta_manager
    if _ta_manager is None:
        _ta_manager = TechnicalAnalysisManager(config, logger)
    return _ta_manager

# Import TA-Lib for backward compatibility
try:
    import talib as ta
except ImportError:
    log = logging.getLogger(__name__)
    log.warning("TA-Lib not installed. Using enhanced technical analysis with fallbacks.")
    # Create a wrapper that uses the enhanced technical analysis

    class TaLib:
        """Enhanced TA-Lib wrapper using the production technical analysis system."""

        @staticmethod
        def atr(high: pd.Series[Any], low: pd.Series[Any], close: pd.Series[Any], length: int) -> pd.Series[Any]:
            """Calculate ATR using enhanced technical analysis.

            Returns:
            -------
                Pandas Series[Any] with ATR values
            """
            # Get config and logger from somewhere (this is a limitation of static method)
            # In production, this would be passed properly
            import logging
            logger = logging.getLogger(__name__)
            config: dict[str, Any] = {}

            try:
                ta_manager = get_ta_manager(config, logger)

                # Cast to numpy arrays to satisfy mypy
                data: dict[str, np.ndarray[Any, Any]] = {
                    "high": np.asarray(high.values),
                    "low": np.asarray(low.values),
                    "close": np.asarray(close.values),
                }

                indicator_config = IndicatorConfig(
                    indicator_type=IndicatorType.ATR,
                    parameters={"timeperiod": length},
                    fallback_value=20.0,
                )

                result = ta_manager.calculate_indicator(indicator_config, data)
                return pd.Series(result.values, index=high.index)

            except Exception:
                logger.exception("ATR calculation failed: ")
                # Return fallback values
                atr_value = Decimal("20.0")
                return pd.Series([atr_value] * len(high), index=high.index)

    ta = TaLib()


# Import production components from backtesting_components
from .backtesting_components import (
    BacktestExchangeInfoService,
    BacktestPubSubManager,
    BacktestRiskManager,
)

# Import optional dependencies if available
PubSubManagerClass: type[Any] = BacktestPubSubManager  # Use production backtest implementation
if PubSubManager is not None:
    # Allow using the real PubSubManager if explicitly configured
    PubSubManagerClass = PubSubManager

RiskManagerClass: type[Any] = BacktestRiskManager  # Use production backtest implementation
if RiskManager is not None:
    # Allow using the real RiskManager if explicitly configured
    RiskManagerClass = RiskManager


# Configure logging
log = logging.getLogger(__name__)


# Use production BacktestExchangeInfoService
ExchangeInfoServiceImpl = BacktestExchangeInfoService


# Define missing event classes if they don't exist in core.events
class SignalEvent(Event):
    """Signal event for trading signals."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SignalEvent with any additional keyword arguments.

        Args:
            *args: Variable length argument list[Any].
            **kwargs: Arbitrary keyword arguments to pass to parent class.
        """
        super().__init__(*args, **kwargs)


class BacktestHistoricalDataProviderImpl:
    """Provides historical data access for backtesting components."""

    def __init__(
        self,
        all_historical_data: dict[str, pd.DataFrame],
        logger: logging.Logger) -> None:
        """Initialize the BacktestHistoricalDataProviderImpl.

        Args:
            all_historical_data: A dictionary where keys are trading pairs (str)
                                 and values are pandas DataFrames containing
                                 OHLCV data for that pair.
            logger: A logging.Logger instance for logging messages.
        """
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
                msg = f"Expected DataFrame for pair {pair}, got {type(df)}" # type: ignore[unreachable]
                raise TypeError(msg)
            required_columns = {"open", "high", "low", "close", "volume"}
            missing = required_columns - set(df.columns)
            if missing:
                msg = f"Missing required columns {missing} for pair {pair}"
                raise ValueError(msg)

    def get_next_bar(self, trading_pair: str, timestamp: dt.datetime) -> pd.Series[Any] | None: # F821
        """Get the next bar after the given timestamp for the trading pair.

        Args:
            trading_pair: The trading pair to get the next bar for
            timestamp: The reference timestamp

        Returns:
        -------
            The next bar as a pandas Series[Any], or None if no more bars
        """
        try:
            if trading_pair not in self._data:
                self.logger.warning("No data for trading pair: %s", trading_pair)
                return None

            pair_data_df = self._data[trading_pair] # PD901
            next_bars = pair_data_df[pair_data_df.index > timestamp].head(1)
            return next_bars.iloc[0] if not next_bars.empty else None

        except Exception:
            self.logger.exception(
                "Error getting next bar for %s at %s",
                trading_pair,
                timestamp)
            return None

    def get_atr( # E501
        self, _trading_pair: str, _timestamp: dt.datetime, _period: int = 14) -> float | None:
        """Get the ATR (Average True Range) value for the given timestamp.

        Args:
            _trading_pair: The trading pair to get ATR for
            _timestamp: The timestamp to get ATR at
            _period: The ATR period (default: 14)

        Returns:
        -------
            The ATR value as a float, or None if not available (always None as it's deprecated).
        """
        # This method is now deprecated as ATR should be sourced from FeatureEngine.
        self.logger.warning(
            "BacktestHistoricalDataProviderImpl.get_atr() is deprecated. "
            "ATR, along with other features, should be generated by FeatureEngine "
            "and accessed from its published events. Returning None.")
        return None

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: dt.datetime, # F821
        end_time: dt.datetime, # F821
        interval: str = "1d") -> pd.DataFrame | None:
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
                trading_pair)

            if trading_pair not in self._data:
                self.logger.warning("No data for trading pair: %s", trading_pair)
                return None

            pair_df = self._data[trading_pair]

            # Ensure start_time and end_time are timezone-aware if df.index is
            if hasattr(pair_df.index, "tz") and pair_df.index.tz is not None:
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=pair_df.index.tz)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=pair_df.index.tz)

            # Ensure we're using proper datetime indexing
            if not isinstance(pair_df.index, pd.DatetimeIndex):
                self.logger.warning("DataFrame index is not a DatetimeIndex, cannot slice by time")
                return None

            # Type cast for comparison
            start_time_cast = pd.Timestamp(start_time)
            end_time_cast = pd.Timestamp(end_time)
            mask = (pair_df.index >= start_time_cast) & (pair_df.index <= end_time_cast)
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
                            })
                        .dropna()
                    )
                    if not resampled.empty:
                        result = resampled
                except Exception as resample_error:  # Resampling can raise various errors
                    self.logger.warning(
                        "Error resampling data to %s interval: %s",
                        interval,
                        str(resample_error))
                    # If resampling fails, fall through to return filtered data if not empty

        except Exception:  # Broad exception for OHLCV retrieval
            self.logger.exception(
                "Error getting historical OHLCV for %s",
                trading_pair)
            return None

        # Return the appropriate result based on the processing
        if interval != "1d" and result is not None:
            return result
        return None if filtered.empty else filtered

    async def get_historical_trades(
        self,
        trading_pair: str,
        start_time: dt.datetime,
        end_time: dt.datetime) -> pd.DataFrame | None:
        """Retrieve historical trade data for a given pair and time range.

        Note:
            In production, this would connect to the database or API to fetch real trade data.
            For backtesting, we can synthesize trade data from OHLCV bars if needed.

        Returns:
        -------
            DataFrame with trade data or None if not available.
        """
        # Check if we have OHLCV data for this pair
        if trading_pair not in self._data:
            self.logger.warning(f"No data available for pair {trading_pair}")
            return None

        # Get OHLCV data for the time range
        ohlcv_data = self._data[trading_pair]
        mask = (ohlcv_data.index >= start_time) & (ohlcv_data.index <= end_time)
        filtered_data = ohlcv_data[mask]

        if filtered_data.empty:
            return None

        # Synthesize trade data from OHLCV
        # In production, this would be real tick data
        trade_data = []
        for timestamp, row in filtered_data.iterrows():
            # Create synthetic trades based on OHLCV
            # High/Low trades
            trade_data.append({
                "timestamp": timestamp,
                "price": row["high"],
                "volume": row["volume"] * 0.25,  # Distribute volume
                "side": "buy",
            })
            trade_data.append({
                "timestamp": timestamp,
                "price": row["low"],
                "volume": row["volume"] * 0.25,
                "side": "sell",
            })
            # Close price trade
            trade_data.append({
                "timestamp": timestamp,
                "price": row["close"],
                "volume": row["volume"] * 0.5,
                "side": "buy" if row["close"] > row["open"] else "sell",
            })

        return pd.DataFrame(trade_data).set_index("timestamp")


# --- Helper Function for Reporting --- #

# Helper functions for calculate_performance_metrics


def _calculate_basic_returns_and_equity(
    equity_curve: pd.Series[Any],
    initial_capital: Decimal,
    results: dict[str, Any]) -> float | None:
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
    equity_curve: pd.Series[Any],
    total_return_pct: float | None,
    results: dict[str, Any]) -> None:
    """Calculate annualized return."""
    if total_return_pct is None:
        results["annualized_return_pct"] = None
        return
    try:
        min_points_for_duration = 2
        if len(equity_curve) >= min_points_for_duration:
            first_date = equity_curve.index[0]
            last_date = equity_curve.index[-1]
            if isinstance(first_date, pd.Timestamp | dt.datetime) and isinstance( # F821
                last_date, pd.Timestamp | dt.datetime, # F821
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
                        annualized_return)
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


def _calculate_drawdown_metrics(equity_curve: pd.Series[Any], results: dict[str, Any]) -> None:
    """Calculate drawdown metrics."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    results["max_drawdown_pct"] = abs(max_drawdown * 100.0)


def _calculate_risk_adjusted_metrics(returns: pd.Series[Any], results: dict[str, Any]) -> None:
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


def _calculate_trade_statistics(trade_log: list[dict[str, Any]], results: dict[str, Any]) -> None:
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
        else: # Line 853
            results["avg_win_loss_ratio"] = float("inf")
    else: # This else corresponds to `if num_trades > 0:`
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


def _calculate_average_holding_period(trade_log: list[dict[str, Any]], results: dict[str, Any]) -> None:
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
                except Exception as e:  # Date parsing or arithmetic errors
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
    equity_curve: pd.Series[Any],
    trade_log: list[dict[str, Any]],
    initial_capital: Decimal) -> dict[str, Any]:
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


class ConfigManagerProtocol(ProtocolClass):
    """Protocol for configuration management in backtesting."""

    def get(self, key: str, default: _T | None = None) -> _T:
        """Get a configuration value by key."""

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value by key."""

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


# Type[Any] aliases for service classes
if TYPE_CHECKING:
    from gal_friday.backtest_historical_data_provider import (
        BacktestHistoricalDataProvider as _BacktestHistoricalDataProvider,
    )
    from gal_friday.core.pubsub import PubSubManager as _PubSubManager
    from gal_friday.exchange_info_service import (
        ExchangeInfoService as _ExchangeInfoService,
    )
    from gal_friday.execution_handler import ExecutionHandler as _ExecutionHandler
    from gal_friday.feature_engine import FeatureEngine as _FeatureEngine
    from gal_friday.logger_service import LoggerService as _LoggerService
    from gal_friday.market_price_service import (
        MarketPriceService as _MarketPriceService,
    )
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


class BacktestMode(str, Enum):
    """Backtesting execution modes."""
    VECTORIZED = "vectorized"      # Fast vectorized backtesting
    EVENT_DRIVEN = "event_driven"  # Realistic event-driven simulation


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""
    start_date: dt.datetime
    end_date: dt.datetime
    initial_capital: float
    symbols: list[str]
    mode: BacktestMode = BacktestMode.EVENT_DRIVEN
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    benchmark_symbol: str | None = None
    output_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary format expected by existing code."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "trading_pairs": self.symbols,  # Map to existing field name
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
            "benchmark_symbol": self.benchmark_symbol,
            "output_dir": self.output_dir,
        }


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics results."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    benchmark_return: float | None = None
    alpha: float | None = None
    beta: float | None = None

    @classmethod
    def from_existing_metrics(cls, metrics: dict[str, Any]) -> PerformanceMetrics:
        """Create PerformanceMetrics from existing metrics dictionary."""
        return cls(
            total_return=metrics.get("total_return_pct", 0.0) / 100.0,
            annualized_return=metrics.get("annualized_return_pct", 0.0) / 100.0,
            volatility=0.0,  # Will be calculated separately
            sharpe_ratio=metrics.get("sharpe_ratio_annualized_approx", 0.0),
            max_drawdown=metrics.get("max_drawdown_pct", 0.0) / 100.0,
            total_trades=metrics.get("total_trades", 0),
            win_rate=metrics.get("win_rate_pct", 0.0) / 100.0,
            profit_factor=metrics.get("profit_factor", 0.0),
            sortino_ratio=metrics.get("sortino_ratio_annualized_approx", 0.0))


class BacktestError(Exception):
    """Exception raised for backtesting errors."""


class BacktestingEngine:
    """Orchestrates backtesting simulations using historical data."""

    def __init__(
        self,
        config: ConfigManagerProtocol,
        data_dir: str = "data",
        results_dir: str = "results",
        max_workers: int = 4) -> None:
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
        self._current_time: dt.datetime | None = None # F821
        self._start_time: dt.datetime | None = None # F821
        self._end_time: dt.datetime | None = None # F821

        # Initialize services with proper type annotations
        self.pubsub_manager: Any = None  # Will be PubSubManager or PubSubManagerStub
        self.logger_service: LoggerService | None = None # Use actual LoggerService type
        self.historical_data_provider: _BacktestHistoricalDataProvider | None = None
        self.market_price_service: _MarketPriceService | None = None
        self.portfolio_manager: _PortfolioManager | None = None
        self.execution_handler: _ExecutionHandler | None = None
        self.feature_engine: FeatureEngine | None = None # Use actual FeatureEngine type
        self.prediction_service: _PredictionService | None = None
        self.risk_manager: _RiskManager | None = None
        self.strategy_arbitrator: _StrategyArbitrator | None = None
        self.exchange_info_service: _ExchangeInfoService | None = None
        self.feature_registry_client: FeatureRegistryClient | None = None # Added attribute

        # Attribute to store the execution report handler for unsubscribing
        self._backtest_exec_report_handler: None | (
            Callable[..., Coroutine[Any, Any, None]]
        ) = None

        # For capturing features from FeatureEngine
        self.current_features: dict[str, float] | None = None
        self.last_features_timestamp: str | None = None

        log.info("BacktestingEngine initialized.")

    async def _initialize_services(self) -> None:
        """Initializes minimal services needed by BacktestingEngine for proper backtesting.

        - PubSubManager: For event-based communication between services.
        - LoggerService: A shared logging service if not already available.
        - FeatureEngine: Calculates features from market data and publishes them to PubSub.

        These services are made available as instance attributes (e.g., self.pubsub_manager,
        self.logger_service, self.feature_engine) for the BacktestingEngine to coordinate
        the backtest's event flow.
        """
        self.logger.info("Initializing BacktestingEngine services...")

        # --- 1. PubSubManager ---
        # Create a minimal logger for PubSubManager if self.logger isn't suitable
        pubsub_logger = logging.getLogger("gal_friday.backtesting.pubsub")
        # Use the correct constructor signature: logger, config_manager
        self.pubsub_manager = PubSubManagerClass(
            logger=pubsub_logger,
            config_manager=self.config,
        )
        # If PubSubManager has async start, call it
        if hasattr(self.pubsub_manager, "start") and \
           asyncio.iscoroutinefunction(self.pubsub_manager.start):
            await self.pubsub_manager.start()
        self.logger.info("PubSubManager initialized for backtesting.")

        # --- 2. LoggerService ---
        if LoggerService is not None and self.logger_service is None:
            # Create LoggerService without database support for backtesting
            self.logger_service = LoggerService(
                config_manager=self.config,
                pubsub_manager=self.pubsub_manager,
                db_session_maker=None, # No database support for backtesting
            )
            self.logger.info("LoggerService initialized for backtesting (no DB support).")
        elif self.logger_service:
            self.logger.info("Using existing LoggerService instance.")

        # --- 3. FeatureEngine ---
        # Create feature_engine if not provided externally
        # Note: FeatureEngine doesn't use FeatureRegistryClient - it loads
        # features from YAML directly
        if FeatureEngine is not None:
            # Pass the pubsub_manager regardless of whether it's real or stub
            # The stub implements the same interface
            self.feature_engine = FeatureEngine(
                config=self.config.get_all() if hasattr(self.config, "get_all") else {},
                pubsub_manager=self.pubsub_manager,
                logger_service=self.logger_service,
                historical_data_service=None)
            await self.feature_engine.start()
            self.logger.info(
                "FeatureEngine initialized and "
                "_backtest_feature_event_handler subscribed.")
        else:
            self.logger.warning("FeatureEngine not available. Feature processing disabled.")  # type: ignore[unreachable]

    async def _backtest_feature_event_handler(self, event_dict: dict[str, Any]) -> None:
        """Handles FEATURES_CALCULATED events specifically for BacktestingEngine.

        This handler captures the latest features and their timestamp, making them
        available on `self.current_features` and `self.last_features_timestamp`.
        This can be useful
        for debugging the backtest itself or for internal assertions after a run.

        This handler is NOT essential for the primary flow of features to downstream services
        like PredictionService or StrategyArbitrator, as those services should subscribe
        to feature events themselves via the shared PubSubManager.
        """
        # The event_dict is what PubSubManager delivers, which is the raw dict[str, Any] form of an Event.
        # Assuming EventType.FEATURES_CALCULATED.name is the string representation.
        if event_dict.get("event_type") == EventType.FEATURES_CALCULATED.name:
            payload = event_dict.get("payload")
            if payload and isinstance(payload, dict):
                self.current_features = payload.get("features") # This is dict[str, float]
                self.last_features_timestamp = payload.get("timestamp_features_for")
            else:
                self.logger.warning(
                    "FEATURES_CALCULATED event received with missing or invalid payload.")
        else:
            self.logger.debug(
                "Backtest handler received non-feature event: %s",
                event_dict.get("event_type"))


    async def _stop_services(self) -> None:
        """Stops any running services initiated by _initialize_services."""
        self.logger.info("Stopping backtesting services (FeatureEngine, PubSub)...")
        if self.feature_engine:
            await self.feature_engine.stop()
            self.logger.info("FeatureEngine stopped.")
        if self.pubsub_manager and \
           hasattr(self.pubsub_manager, "stop_consuming"): # PubSubManager has stop_consuming
            await self.pubsub_manager.stop_consuming()
            self.logger.info("PubSubManager stopped.")
        # Add other services to stop here if necessary

    def _get_backtest_config(self) -> dict[str, Any]:
        """Get backtest configuration from the config manager."""
        config: Any = self.config.get("backtest", {})
        if not isinstance(config, dict):
            log.warning("Backtest config is not a dictionary, returning empty dict[str, Any]")
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
            local_df = pd.read_csv(path, parse_dates=["timestamp"])

            if not isinstance(local_df, pd.DataFrame):
                log.error(  # type: ignore[unreachable]
                    "pd.read_csv did not return a DataFrame for path %s. Got type: %s",
                    data_path,
                    type(local_df))
                return None

            if "pair" not in local_df.columns:
                log.error("Data file %s must contain 'pair' column", data_path)
                return None

            grouped_data = local_df.groupby("pair")

            # Convert grouped data to dict of DataFrames
            # Cast pair to str to ensure type compatibility
            return {str(pair): group for pair, group in grouped_data}

        except Exception: # File I/O or parsing errors
            log.exception(
                "Error during data loading/processing in _load_raw_data (path: %s)",
                 data_path) # TRY401
            return None

    def _clean_and_validate_data(
        self,
        data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str) -> dict[str, pd.DataFrame] | None:
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
        value: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame] | None:
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
        data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame] | None:
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
        run_config: dict[str, Any]) -> dict[str, Any]:
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
                    initial_capital=initial_capital)
                results["metrics"] = metrics

            # Save results to file
            output_dir = Path(run_config.get("output_dir", "results"))
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"backtest_results_{timestamp}.json"

            # Save results to JSON file with decimal conversion
            with results_file.open("w") as f:
                json.dump(results, f, default=decimal_to_float, indent=2)

            log.info("Backtest results saved to %s", results_file)

        except Exception as e:
            log.exception("Error processing and saving results")
            # Include error in results instead of raising to ensure we always return a dict[str, Any]
            results["error"] = str(e)

        return results

    async def _execute_simulation(  # noqa: C901, PLR0915, PLR0912
        self,
        services: dict[str, Any],
        run_config: dict[str, Any]) -> None:
        """Execute the backtest simulation with proper time-series iteration.

        Args:
            services: Dictionary of pre-initialized services (e.g., `PortfolioManager`,
                      `ExecutionHandler`, `PredictionService`, `StrategyArbitrator`). These
                      services should be instantiated with the shared `PubSubManager`,
                      `LoggerService`, and `FeatureRegistryClient` (where applicable)
                      created in `_initialize_services` to ensure they participate
                      correctly in the backtesting event flow.
            run_config: Configuration specific to this backtest run, typically from
                      the 'backtest' section of the main application configuration.
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

        _emsg_required_services = "Required services not available for simulation." # N806
        if not all([market_price_service, portfolio_manager, execution_handler]):
            raise ValueError(_emsg_required_services) # EM101, TRY003

        # Initialize services (PubSub, FeatureEngine, etc.)
        await self._initialize_services()

        # Original service startup loop (ensure it doesn't re-init FE/PubSub
        # if they are among `services`)
        for service_name, service in services.items():
            # Avoid re-initializing services that _initialize_services is now responsible for
            if service_name in ["feature_engine", "pubsub_manager", "logger_service"] and \
               getattr(self, service_name, None) is not None:
                log.info(
                    "Service %s already initialized by BacktestingEngine. "
                    "Skipping general start.",
                    service_name)
                continue

            if hasattr(service, "start") and callable(service.start):
                try:
                    log.info("Starting service: %s", service_name)
                    if asyncio.iscoroutinefunction(service.start):
                        await service.start()
                    else:
                        service_start_result = service.start()
                        if asyncio.iscoroutine(service_start_result):
                            await service_start_result
                except Exception:
                    log.exception("Error starting service %s", service_name)
                    raise

        try:
            # Get unified timeline from all trading pairs
            all_timestamps: set[pd.Timestamp] = set()
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
            log.info("Simulation will process %s time steps", len(sorted_timestamps)) # G004

            # Initialize equity curve tracking
            equity_curve_data = []

            # Main simulation loop - iterate through time
            for i, current_timestamp in enumerate(sorted_timestamps):
                self._current_step = i
                self._current_time = current_timestamp

                log.debug("Processing timestamp %s (step %s)", current_timestamp, i) # G004

                try:
                    # 1. Update market price service with current timestamp
                    if market_price_service is not None and \
                       hasattr(market_price_service, "update_time"):
                        market_price_service.update_time(current_timestamp)

                    # 2. Generate market data events for all pairs at this timestamp
                    for pair in trading_pairs:
                        await self._process_market_data_for_timestamp(
                            pair, current_timestamp, services)

                    # 3. Process any pending limit orders and stop-loss/take-profit
                    if execution_handler is not None and \
                       hasattr(execution_handler, "check_active_limit_orders"):
                        # Get current bar data for limit order processing
                        for pair in trading_pairs:
                            current_bar = self._get_bar_at_timestamp(pair, current_timestamp)
                            if current_bar is not None:
                                await execution_handler.check_active_limit_orders(
                                    current_bar, current_timestamp)
                                if hasattr(execution_handler, "check_active_sl_tp"):
                                    await execution_handler.check_active_sl_tp(
                                        current_bar, current_timestamp)

                    # 4. Update portfolio value and record equity curve
                    if portfolio_manager is not None and \
                       hasattr(portfolio_manager, "get_current_state"):
                        portfolio_state = portfolio_manager.get_current_state()
                        if portfolio_state and "total_value" in portfolio_state:
                            equity_curve_data.append({
                                "timestamp": current_timestamp,
                                "equity": portfolio_state["total_value"],
                            })

                    # 5. Small delay to prevent overwhelming the event loop
                    if i % 1000 == 0:  # Every 1000 steps
                        await asyncio.sleep(0.001)  # Minimal delay

                except Exception as e:  # Catch errors within loop step
                    log.exception("Error processing timestamp %s", current_timestamp) # TRY400
                    # Continue processing unless it's a critical error
                    if isinstance(e, KeyboardInterrupt | SystemExit):
                        raise
                    continue

            # Store final equity curve in services for metrics calculation
            if equity_curve_data:
                equity_df = pd.DataFrame(equity_curve_data)
                equity_df = equity_df.set_index("timestamp") # PD002
                services["equity_curve"] = equity_df["equity"]
                log.info(
                    "Simulation completed. Final equity: %s",
                    equity_df["equity"].iloc[-1])
            else:
                log.warning("No equity curve data collected during simulation")

        except asyncio.CancelledError:
            log.info("Simulation cancelled")
            raise
        except Exception:
            log.exception("Error during simulation")
            raise
        finally:
            # Stop services initialized by BacktestingEngine first
            await self._stop_services()

            # Stop other services that were passed in
            original_services_to_stop = {
                name: svc for name, svc in services.items()
                if name not in ["feature_engine", "pubsub_manager", "logger_service"]
            }  # Exclude already stopped
            for service_name in reversed(list[Any](original_services_to_stop.keys())):
                service = original_services_to_stop[service_name]
                if hasattr(service, "stop") and callable(service.stop):
                    try:
                        log.info("Stopping original service: %s...", service_name)
                        if asyncio.iscoroutinefunction(service.stop):
                            await service.stop()
                        else:
                            stop_result = service.stop()
                            if asyncio.iscoroutine(stop_result):
                                await stop_result
                    except Exception:
                        log.exception("Error stopping original service %s", service_name)

            log.info("All services shut down.")


    async def _process_market_data_for_timestamp(
        self,
        trading_pair: str,
        timestamp: dt.datetime, # F821
        _services: dict[str, Any], # services dict[str, Any] is passed for context if needed
    ) -> None:
        """Process market data for a specific trading pair and timestamp.

        This will involve creating a MarketDataOHLCVEvent and passing it to the FeatureEngine.

        Args:
            trading_pair: The trading pair to process.
            timestamp: Current simulation timestamp.
            _services: Dictionary of initialized services (used to access FeatureEngine).
        """
        try:
            bar_data = self._get_bar_at_timestamp(trading_pair, timestamp)
            if bar_data is None:
                return

            # Create MarketDataOHLCVEvent dictionary payload
            # Ensure bar_data["timestamp"] is used for timestamp_bar_start
            # Values for OHLCV are expected as strings by MarketDataOHLCVEvent
            # according to its definition
            market_event_payload = {
                "trading_pair": trading_pair,
                "exchange": self.config.get("exchange_name", "simulated_exchange"),
                "interval": self.config.get(
                    "ohlcv_interval", "1d"),  # Make interval configurable or use a default
                "timestamp_bar_start": (
                    bar_data["timestamp"].isoformat() + "Z"
                ),  # Timestamp of the bar itself
                "open": str(bar_data["open"]),
                "high": str(bar_data["high"]),
                "low": str(bar_data["low"]),
                "close": str(bar_data["close"]),
                "volume": str(bar_data["volume"]),
            }

            market_event_dict = {
                "event_id": str(uuid.uuid4()),
                "event_type": EventType.MARKET_DATA_OHLCV.name,
                "timestamp": dt.datetime.now(dt.UTC).isoformat() + "Z", # DTZ003
                "source_module": self.__class__.__name__,
                "payload": market_event_payload,
            }

            if self.feature_engine:
                await self.feature_engine.process_market_data(market_event_dict)
            else:
                self.logger.warning(
                    "FeatureEngine not initialized. "
                    "Cannot process market data for feature generation.")

        except Exception:  # Broad catch for market data processing step
            self.logger.exception(
                "Error processing market data for %s at %s",
                trading_pair,
                timestamp)

    def _get_bar_at_timestamp(  # E501
        self, trading_pair: str, timestamp: dt.datetime) -> pd.Series[Any] | None:
        """Get OHLCV bar data for a specific trading pair and timestamp.

        Args:
            trading_pair: Trading pair to get data for
            timestamp: Timestamp to look up

        Returns:
            Series[Any] containing OHLCV data or None if not found
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

        except Exception:  # Data lookup can fail in various ways
            log.exception("Error getting bar data for %s at %s", trading_pair, timestamp) # TRY400
            return None

    async def run_backtest(
        self,
        config: BacktestConfig,
        strategy: Any = None,
        services: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive backtest with performance analytics.

        This is the main public API for running backtests with the comprehensive framework.

        Args:
            config: BacktestConfig instance with backtest parameters
            strategy: Optional strategy instance to test
            services: Optional pre-initialized services dictionary

        Returns:
            Comprehensive backtest results including performance metrics and analytics
        """
        try:
            start_time = dt.datetime.now()
            self.logger.info("Starting comprehensive backtest")
            self.logger.info(f"Mode: {config.mode.value}")
            self.logger.info(f"Period: {config.start_date} to {config.end_date}")
            self.logger.info(f"Initial capital: ${config.initial_capital:,.2f}")
            self.logger.info(f"Symbols: {config.symbols}")

            # Convert config to format expected by existing methods
            run_config = config.to_dict()

            # Load historical data
            await self._load_historical_data_for_symbols(config.symbols, config.start_date, config.end_date)

            # Initialize or use provided services
            if services is None:
                services = await self._create_default_services(run_config)

            # Execute backtesting based on mode
            if config.mode == BacktestMode.VECTORIZED:
                await self._run_vectorized_backtest(services, run_config, strategy)
            elif config.mode == BacktestMode.EVENT_DRIVEN:
                await self._execute_simulation(services, run_config)

            # Process results and calculate enhanced metrics
            results = self._process_and_save_results(services, run_config)

            # Add benchmarking if specified
            if config.benchmark_symbol:
                await self._add_benchmark_analysis(results, config)

            # Calculate enhanced performance metrics
            enhanced_metrics = self._calculate_enhanced_metrics(results, config)
            results["enhanced_metrics"] = enhanced_metrics

            # Add execution metadata
            execution_time = (dt.datetime.now() - start_time).total_seconds()
            results.update({
                "config": config.__dict__,
                "execution_time_seconds": execution_time,
                "backtest_mode": config.mode.value,
                "framework_version": "comprehensive_v1.0",
            })

            self.logger.info(f"Comprehensive backtest completed in {execution_time:.2f} seconds")
            if enhanced_metrics:
                self.logger.info(f"Total return: {enhanced_metrics.total_return:.2%}")
                self.logger.info(f"Sharpe ratio: {enhanced_metrics.sharpe_ratio:.2f}")
                self.logger.info(f"Max drawdown: {enhanced_metrics.max_drawdown:.2%}")

            return results

        except Exception as e:
            self.logger.exception("Comprehensive backtest failed: ")
            raise BacktestError(f"Backtesting failed: {e}")

    async def _create_default_services(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Create default services for backtesting with enterprise-grade initialization.

        This method creates and initializes all required services for backtesting including
        portfolio management, risk management, prediction services, and strategy arbitration.

        Args:
            run_config: Configuration dictionary containing backtest parameters

        Returns:
            Dictionary of initialized services ready for backtesting
        """
        services: dict[str, Any] = {}

        try:
            # Initialize core services if not already initialized
            if not hasattr(self, "pubsub_manager") or self.pubsub_manager is None:
                await self._initialize_services()

            # 1. Initialize Market Price Service (Simulated) first
            if SimulatedMarketPriceServiceType is not None:
                # Pass the loaded historical data to SimulatedMarketPriceService
                services["market_price_service"] = SimulatedMarketPriceServiceType(
                    historical_data=self._data,  # Use the loaded data
                    config_manager=None,  # SimulatedMarketPriceService expects ConfigManager, not Protocol
                    logger=self.logger,
                )
                self.logger.info("Simulated Market Price Service initialized")

            # 2. Initialize Portfolio Manager (needs market price service)
            if PortfolioManagerType is not None:
                # Create a mock session maker for backtesting
                from unittest.mock import Mock
                mock_session_maker = Mock()

                # Create a mock ConfigManager that wraps our ConfigManagerProtocol
                mock_config_manager = Mock()
                mock_config_manager.get = self.config.get
                mock_config_manager.get_int = self.config.get_int
                mock_config_manager.__getitem__ = self.config.__getitem__
                mock_config_manager.__contains__ = self.config.__contains__
                mock_config_manager.get_all = self.config.get_all
                mock_config_manager.set = self.config.set

                # Ensure logger_service is not None
                if self.logger_service is None:
                    raise BacktestError("LoggerService must be initialized before creating services")

                services["portfolio_manager"] = PortfolioManagerType(
                    config_manager=mock_config_manager,
                    pubsub_manager=self.pubsub_manager,
                    market_price_service=services.get("market_price_service", Mock()),
                    logger_service=self.logger_service,
                    session_maker=mock_session_maker,
                )
                self.logger.info("Portfolio Manager initialized for backtesting")

            # 3. Initialize Risk Manager
            if RiskManagerType is not None and "portfolio_manager" in services:
                # Create exchange info service for risk manager
                from unittest.mock import Mock
                mock_exchange_info = Mock()

                # Ensure logger_service is not None
                if self.logger_service is None:
                    raise BacktestError("LoggerService must be initialized before creating services")

                services["risk_manager"] = RiskManagerType(
                    config=self.config.get_all() if hasattr(self.config, "get_all") else {},
                    pubsub_manager=self.pubsub_manager,
                    portfolio_manager=services["portfolio_manager"],
                    logger_service=self.logger_service,
                    market_price_service=services.get("market_price_service", Mock()),
                    exchange_info_service=mock_exchange_info,
                )
                self.logger.info("Risk Manager initialized for backtesting")

            # 4. Initialize Prediction Service
            if PredictionServiceType is not None:
                from concurrent.futures import ProcessPoolExecutor
                # Create a process pool executor for prediction service
                process_pool = ProcessPoolExecutor(max_workers=2)

                # Ensure logger_service is not None
                if self.logger_service is None:
                    raise BacktestError("LoggerService must be initialized before creating services")

                services["prediction_service"] = PredictionServiceType(
                    config=self.config.get_all() if hasattr(self.config, "get_all") else {},
                    pubsub_manager=self.pubsub_manager,
                    process_pool_executor=process_pool,
                    logger_service=self.logger_service,
                    configuration_manager=None,  # Optional parameter
                )
                self.logger.info("Prediction Service initialized for backtesting")

            # 5. Initialize Strategy Arbitrator
            if StrategyArbitratorType is not None:
                # Create a mock feature registry client
                from unittest.mock import Mock
                mock_feature_registry = Mock()

                # Ensure logger_service is not None
                if self.logger_service is None:
                    raise BacktestError("LoggerService must be initialized before creating services")

                services["strategy_arbitrator"] = StrategyArbitratorType(
                    config=self.config.get_all() if hasattr(self.config, "get_all") else {},
                    pubsub_manager=self.pubsub_manager,
                    logger_service=self.logger_service,
                    market_price_service=services.get("market_price_service", Mock()),
                    feature_registry_client=mock_feature_registry,
                    risk_manager=services.get("risk_manager"),
                    portfolio_manager=services.get("portfolio_manager"),
                )
                self.logger.info("Strategy Arbitrator initialized for backtesting")

            # 6. Initialize Execution Handler (Simulated)
            if ExecutionHandler is not None:
                # Create a mock historical data service
                from unittest.mock import Mock
                mock_data_service = Mock()

                # Create a mock ConfigManager
                mock_exec_config_manager = Mock()
                mock_exec_config_manager.get = self.config.get
                mock_exec_config_manager.get_int = self.config.get_int
                mock_exec_config_manager.__getitem__ = self.config.__getitem__
                mock_exec_config_manager.__contains__ = self.config.__contains__
                mock_exec_config_manager.get_all = self.config.get_all

                # Ensure logger_service is not None
                if self.logger_service is None:
                    raise BacktestError("LoggerService must be initialized before creating services")

                services["execution_handler"] = ExecutionHandler(
                    config_manager=mock_exec_config_manager,
                    pubsub_manager=self.pubsub_manager,
                    data_service=mock_data_service,  # Expects HistoricalDataService
                    logger_service=self.logger_service,
                )
                self.logger.info("Backtest Execution Handler initialized")

            # 7. Start all services that require async initialization
            for service_name, service in services.items():
                if hasattr(service, "start") and asyncio.iscoroutinefunction(service.start):
                    await service.start()
                    self.logger.debug(f"Started service: {service_name}")

            self.logger.info(f"Successfully initialized {len(services)} services for backtesting")
            return services

        except Exception as e:
            self.logger.exception("Failed to create default services: ")
            # Clean up any partially initialized services
            for service_name, service in services.items():
                if hasattr(service, "stop") and asyncio.iscoroutinefunction(service.stop):
                    try:
                        await service.stop()
                    except Exception:
                        self.logger.exception(f"Error stopping service {service_name} during cleanup")
            raise BacktestError(f"Failed to initialize services: {e}")

    async def _load_historical_data_for_symbols(
        self,
        symbols: list[str],
        start_date: dt.datetime,
        end_date: dt.datetime,
    ) -> None:
        """Load historical data for specified symbols and date range.

        Enterprise-grade data loading with multiple sources, caching, validation,
        and comprehensive error handling.

        Args:
            symbols: List of trading symbols to load data for
            start_date: Start date for data loading
            end_date: End date for data loading

        Raises:
            BacktestError: If data loading fails or no data is available
        """
        self.logger.info(
            f"Loading historical data for {len(symbols)} symbols: {symbols}",
        )

        # Get data loading configuration
        data_config = self._get_data_loading_config()

        # Initialize data loading statistics
        loading_stats: dict[str, Any] = {
            "requested_symbols": len(symbols),
            "successful_loads": 0,
            "failed_loads": 0,
            "cache_hits": 0,
            "data_sources_used": set[Any](),
            "total_data_points": 0,
        }

        try:
            # Initialize enterprise data loader if not already done
            if not hasattr(self, "_enterprise_data_loader"):
                await self._initialize_enterprise_data_loader(data_config)

            # Load data for each symbol with parallel processing for performance
            loading_tasks = []
            for symbol in symbols:
                task = self._load_symbol_data_with_retry(
                    symbol, start_date, end_date, data_config, loading_stats,
                )
                loading_tasks.append(task)

            # Execute loading tasks with controlled concurrency
            max_concurrent = data_config.get("max_concurrent_loads", 5)
            semaphore = asyncio.Semaphore(max_concurrent)

            async def load_with_semaphore(coro: Coroutine[Any, Any, Any]) -> Any:
                async with semaphore:
                    return await coro

            # Wait for all loading tasks to complete
            results = await asyncio.gather(
                *[load_with_semaphore(task) for task in loading_tasks],
                return_exceptions=True,
            )

            # Process results and handle any exceptions
            successful_symbols = []
            failed_symbols = []

            for i, result in enumerate(results):
                symbol = symbols[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to load data for {symbol}: {result}")
                    failed_symbols.append(symbol)
                    loading_stats["failed_loads"] = loading_stats.get("failed_loads", 0) + 1
                elif result:
                    successful_symbols.append(symbol)
                    loading_stats["successful_loads"] = loading_stats.get("successful_loads", 0) + 1
                else:
                    self.logger.warning(f"No data available for {symbol}")
                    failed_symbols.append(symbol)
                    loading_stats["failed_loads"] = loading_stats.get("failed_loads", 0) + 1

            # Validate loading success
            self._validate_data_loading_results(
                successful_symbols, failed_symbols, data_config, loading_stats,
            )

            # Log comprehensive loading statistics
            self._log_data_loading_statistics(loading_stats)

            self.logger.info(
                f"Successfully loaded data for {len(successful_symbols)}/{len(symbols)} symbols",
            )

        except Exception as e:
            self.logger.exception("Critical error in data loading: ")
            raise BacktestError(f"Failed to load historical data: {e}")

    def _get_data_loading_config(self) -> dict[str, Any]:
        """Get comprehensive data loading configuration."""
        base_config = {
            "max_concurrent_loads": 5,
            "retry_attempts": 3,
            "retry_delay_base": 1.0,
            "retry_delay_multiplier": 2.0,
            "cache_enabled": True,
            "data_validation_enabled": True,
            "quality_threshold": 0.8,
            "allow_partial_failures": False,
            "timeout_seconds": 30.0,
            "default_data_source": "auto",
            "fallback_data_sources": ["local_files", "database"],
        }

        # Override with user configuration
        user_config: Any = self.config.get("backtest.data_loading", {})
        if isinstance(user_config, dict):
            base_config.update(user_config)

        # Add data source specific configuration
        base_config.update({
            "file_sources": {
                "data_path": self.config.get("backtest.data_path"),
                "supported_formats": ["csv", "parquet", "json"],
                "default_format": "csv",
            },
            "database_sources": {
                "primary_db": self.config.get("database.url"),
                "timeseries_db": self.config.get("influxdb.url"),
                "query_timeout": 60.0,
            },
            "api_sources": {
                "rate_limit_delay": 1.0,
                "api_timeout": 30.0,
                "max_requests_per_minute": 60,
            },
        })

        return base_config

    async def _initialize_enterprise_data_loader(self, data_config: dict[str, Any]) -> None:
        """Initialize enterprise-grade data loading infrastructure."""
        try:
            # Initialize the comprehensive data loader with caching
            loader_config = {
                "memory_cache_size": data_config.get("memory_cache_size", 1000),
                "disk_cache_path": data_config.get("disk_cache_path", "./cache/backtest_data"),
                "cache_enabled": data_config["cache_enabled"],
                "providers": data_config.get("enabled_providers", ["local_files", "database"]),
                "timeout_seconds": data_config["timeout_seconds"],
            }

            self._enterprise_data_loader = EnterpriseHistoricalDataLoader(
                config=loader_config,
                config_manager=self.config,
                logger=self.logger,
            )

            await self._enterprise_data_loader.initialize()

            self.logger.info("Enterprise data loader initialized successfully")

        except Exception as e:
            self.logger.exception("Failed to initialize enterprise data loader: ")
            raise BacktestError(f"Data loader initialization failed: {e}")

    async def _load_symbol_data_with_retry(
        self,
        symbol: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        data_config: dict[str, Any],
        loading_stats: dict[str, Any],
    ) -> bool:
        """Load data for a single symbol with retry logic and comprehensive error handling."""
        retry_attempts = data_config["retry_attempts"]
        retry_delay = data_config["retry_delay_base"]
        retry_multiplier = data_config["retry_delay_multiplier"]

        for attempt in range(retry_attempts + 1):
            try:
                # Create data request with comprehensive parameters
                data_request = {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data_source": data_config["default_data_source"],
                    "fallback_sources": data_config["fallback_data_sources"],
                    "include_volume": True,
                    "validate_data": data_config["data_validation_enabled"],
                    "cache_result": data_config["cache_enabled"],
                    "quality_threshold": data_config["quality_threshold"],
                    "timeout_seconds": data_config["timeout_seconds"],
                }

                # Load data using enterprise loader
                data_result = await self._enterprise_data_loader.load_historical_data(data_request)

                if data_result and data_result.get("data") is not None:
                    # Store the loaded data
                    df = data_result["data"]
                    self._data[symbol] = df

                    # Update statistics
                    loading_stats["total_data_points"] += len(df)
                    if data_result.get("cache_hit"):
                        loading_stats["cache_hits"] += 1
                    if data_result.get("data_source"):
                        loading_stats["data_sources_used"].add(data_result["data_source"])

                    self.logger.debug(
                        f"Successfully loaded {len(df)} data points for {symbol} "
                        f"from {data_result.get('data_source', 'unknown')}",
                    )
                    return True
                if attempt < retry_attempts:
                    self.logger.warning(
                        f"No data returned for {symbol}, attempt {attempt + 1}/{retry_attempts + 1}",
                    )
                else:
                    self.logger.error(f"No data available for {symbol} after all retry attempts")
                    return False

            except Exception as e:
                if attempt < retry_attempts:
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {symbol}: {e}. "
                        f"Retrying in {retry_delay:.1f}s...",
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= retry_multiplier
                else:
                    self.logger.exception(f"All retry attempts failed for {symbol}: ")
                    raise BacktestError(f"Failed to load data for {symbol}: {e}")

        return False

    def _validate_data_loading_results(
        self,
        successful_symbols: list[str],
        failed_symbols: list[str],
        data_config: dict[str, Any],
        loading_stats: dict[str, Any],
    ) -> None:
        """Validate data loading results and determine if backtest can proceed."""
        total_symbols = len(successful_symbols) + len(failed_symbols)
        success_rate = len(successful_symbols) / total_symbols if total_symbols > 0 else 0

        # Check if we have any data at all
        if not successful_symbols:
            raise BacktestError(
                f"No historical data could be loaded for any symbols. "
                f"Failed symbols: {failed_symbols}",
            )

        # Check if partial failures are acceptable
        if failed_symbols and not data_config.get("allow_partial_failures", False):
            raise BacktestError(
                f"Data loading failed for some symbols and partial failures are not allowed. "
                f"Failed symbols: {failed_symbols}",
            )

        # Check minimum success rate if configured
        min_success_rate = data_config.get("min_success_rate", 0.8)
        if success_rate < min_success_rate:
            raise BacktestError(
                f"Data loading success rate ({success_rate:.1%}) is below minimum "
                f"required rate ({min_success_rate:.1%})",
            )

        # Validate data quality
        self._validate_loaded_data_quality(successful_symbols, data_config)

    def _validate_loaded_data_quality(
        self,
        symbols: list[str],
        data_config: dict[str, Any],
    ) -> None:
        """Validate the quality of loaded data."""
        if not data_config.get("data_validation_enabled", True):
            return

        quality_threshold = data_config.get("quality_threshold", 0.8)
        quality_issues = []

        for symbol in symbols:
            if symbol not in self._data:
                continue

            df = self._data[symbol]
            quality_score = self._calculate_data_quality_score(df, symbol)

            if quality_score < quality_threshold:
                quality_issues.append(f"{symbol}: {quality_score:.2f}")
                self.logger.warning(
                    f"Data quality for {symbol} ({quality_score:.2f}) "
                    f"is below threshold ({quality_threshold:.2f})",
                )

        if quality_issues and not data_config.get("allow_low_quality_data", False):
            raise BacktestError(
                f"Data quality issues detected: {quality_issues}. "
                f"Set 'allow_low_quality_data: true' to proceed anyway.",
            )

    def calculate_technical_indicators(
        self,
        symbol: str,
        indicators: list[str],
        custom_params: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, pd.Series[Any]]:
        """Calculate multiple technical indicators for a symbol using enhanced system.

        Args:
            symbol: Trading pair symbol
            indicators: List of indicator names (e.g., ['atr_14', 'rsi_14', 'sma_20'])
            custom_params: Optional custom parameters for indicators

        Returns:
            Dictionary mapping indicator names to pandas Series[Any]
        """
        if symbol not in self._data:
            raise ValueError(f"No data available for symbol {symbol}")

        df = self._data[symbol]

        # Initialize technical analysis manager if needed
        # Convert config to dict format for ta_manager
        config_dict = {}
        if hasattr(self.config, "get_all"):
            config_dict = self.config.get_all()
        elif hasattr(self.config, "__dict__"):
            config_dict = self.config.__dict__
        ta_manager = get_ta_manager(config_dict, self.logger)

        # Default indicator configurations
        default_configs = {
            "atr_14": IndicatorConfig(
                indicator_type=IndicatorType.ATR,
                parameters={"timeperiod": 14},
                fallback_value=20.0,
            ),
            "atr_20": IndicatorConfig(
                indicator_type=IndicatorType.ATR,
                parameters={"timeperiod": 20},
                fallback_value=20.0,
            ),
            "rsi_14": IndicatorConfig(
                indicator_type=IndicatorType.RSI,
                parameters={"timeperiod": 14},
                fallback_value=50.0,
            ),
            "sma_20": IndicatorConfig(
                indicator_type=IndicatorType.SMA,
                parameters={"timeperiod": 20},
            ),
            "sma_50": IndicatorConfig(
                indicator_type=IndicatorType.SMA,
                parameters={"timeperiod": 50},
            ),
            "ema_12": IndicatorConfig(
                indicator_type=IndicatorType.EMA,
                parameters={"timeperiod": 12},
            ),
            "ema_26": IndicatorConfig(
                indicator_type=IndicatorType.EMA,
                parameters={"timeperiod": 26},
            ),
            "bbands_20": IndicatorConfig(
                indicator_type=IndicatorType.BBANDS,
                parameters={"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
            ),
            "macd": IndicatorConfig(
                indicator_type=IndicatorType.MACD,
                parameters={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            ),
            "vwap": IndicatorConfig(
                indicator_type=IndicatorType.VWAP,
                parameters={},
            ),
        }

        # Convert DataFrame to numpy arrays
        data: dict[str, Any] = {
            "open": df["open"].values if "open" in df.columns else None,
            "high": df["high"].values if "high" in df.columns else None,
            "low": df["low"].values if "low" in df.columns else None,
            "close": df["close"].values if "close" in df.columns else None,
            "volume": df["volume"].values if "volume" in df.columns else None,
        }

        # Remove None values and ensure correct type
        data_clean: dict[str, np.ndarray[Any, Any]] = {k: v for k, v in data.items() if v is not None}

        results = {}

        for indicator_name in indicators:
            try:
                # Get configuration
                if indicator_name in default_configs:
                    config = default_configs[indicator_name]
                else:
                    # Try to parse custom indicator name (e.g., "rsi_20")
                    parts = indicator_name.split("_")
                    if len(parts) >= 2 and parts[0].upper() in [t.value.upper() for t in IndicatorType]:
                        indicator_type = IndicatorType(parts[0].lower())
                        period = int(parts[1]) if parts[1].isdigit() else 14
                        config = IndicatorConfig(
                            indicator_type=indicator_type,
                            parameters={"timeperiod": period},
                        )
                    else:
                        self.logger.warning(f"Unknown indicator: {indicator_name}")
                        continue

                # Apply custom parameters if provided
                if custom_params and indicator_name in custom_params:
                    config.parameters.update(custom_params[indicator_name])

                # Calculate indicator
                result = ta_manager.calculate_indicator(config, data_clean)

                # Handle multiple outputs
                if isinstance(result.values, tuple):
                    # For indicators like MACD, Bollinger Bands
                    for i, values in enumerate(result.values):
                        suffix = ["", "_signal", "_hist"] if config.indicator_type == IndicatorType.MACD else [f"_{i}" for _ in result.values]
                        key = f"{indicator_name}{suffix[i] if i > 0 else ''}"
                        results[key] = pd.Series(values, index=df.index)
                else:
                    # Single output
                    results[indicator_name] = pd.Series(result.values, index=df.index)

            except Exception:
                self.logger.exception(f"Failed to calculate {indicator_name}: ")
                continue

        return results

    def _calculate_data_quality_score(self, df: pd.DataFrame, symbol: str) -> float:
        """Calculate a data quality score for the loaded data."""
        try:
            # Basic quality checks
            total_rows = len(df)
            if total_rows == 0:
                return 0.0

            quality_factors = []

            # Check for missing values
            missing_ratio = df.isnull().sum().sum() / (total_rows * len(df.columns))
            quality_factors.append(1.0 - missing_ratio)

            # Check for reasonable OHLCV relationships
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                # High should be >= Open, Low, Close
                high_valid = ((df["high"] >= df["open"]) &
                            (df["high"] >= df["low"]) &
                            (df["high"] >= df["close"])).mean()

                # Low should be <= Open, High, Close
                low_valid = ((df["low"] <= df["open"]) &
                           (df["low"] <= df["high"]) &
                           (df["low"] <= df["close"])).mean()

                quality_factors.extend([high_valid, low_valid])

            # Check for reasonable price movements (no extreme outliers)
            if "close" in df.columns and len(df) > 1:
                price_changes = df["close"].pct_change().abs()
                extreme_moves = (price_changes > 0.5).sum()  # >50% moves
                extreme_ratio = extreme_moves / len(price_changes)
                quality_factors.append(1.0 - min(extreme_ratio, 1.0))

            # Check data continuity (no huge gaps)
            if "timestamp" in df.columns:
                df_sorted = df.sort_values("timestamp")
                time_diffs = df_sorted["timestamp"].diff()
                if len(time_diffs) > 1:
                    median_diff = time_diffs.median()
                    # Convert to numeric for comparison
                    median_seconds = median_diff.total_seconds() if hasattr(median_diff, "total_seconds") else float(median_diff)
                    time_diffs_seconds = time_diffs.dt.total_seconds() if hasattr(time_diffs, "dt") else time_diffs
                    large_gaps = (time_diffs_seconds > median_seconds * 10).sum()  # type: ignore[operator]
                    gap_ratio = large_gaps / len(time_diffs)
                    quality_factors.append(1.0 - min(gap_ratio, 1.0))

            # Calculate overall quality score
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating quality score for {symbol}: {e}")
            return 0.5  # Default middle score if calculation fails

    def _log_data_loading_statistics(self, loading_stats: dict[str, Any]) -> None:
        """Log comprehensive data loading statistics."""
        self.logger.info("Data Loading Statistics:")
        self.logger.info(f"  Requested Symbols: {loading_stats['requested_symbols']}")
        self.logger.info(f"  Successful Loads: {loading_stats['successful_loads']}")
        self.logger.info(f"  Failed Loads: {loading_stats['failed_loads']}")
        self.logger.info(f"  Cache Hits: {loading_stats['cache_hits']}")
        self.logger.info(f"  Total Data Points: {loading_stats['total_data_points']:,}")

        if loading_stats["data_sources_used"]:
            sources = ", ".join(loading_stats["data_sources_used"])
            self.logger.info(f"  Data Sources Used: {sources}")

        success_rate = (loading_stats["successful_loads"] /
                       loading_stats["requested_symbols"] * 100
                       if loading_stats["requested_symbols"] > 0 else 0)
        self.logger.info(f"  Success Rate: {success_rate:.1f}%")

    async def _run_vectorized_backtest(
        self,
        services: dict[str, Any],
        run_config: dict[str, Any],
        strategy: Any = None,
    ) -> None:
        """Run vectorized backtesting for faster execution."""
        self.logger.info("Running vectorized backtest")

        # Get configuration
        trading_pairs = run_config.get("trading_pairs", [])
        start_date = pd.to_datetime(run_config["start_date"])
        end_date = pd.to_datetime(run_config["end_date"])
        initial_capital = run_config.get("initial_capital", 10000)

        # Initialize portfolio tracking
        cash = initial_capital
        positions = dict.fromkeys(trading_pairs, 0.0)
        portfolio_history = []

        # Get all data for vectorized processing
        all_data = {}
        for symbol in trading_pairs:
            if symbol in self._data:
                symbol_data = self._data[symbol]
                mask = (symbol_data["timestamp"] >= start_date) & (symbol_data["timestamp"] <= end_date)
                all_data[symbol] = symbol_data[mask].sort_values("timestamp")

        if not all_data:
            raise BacktestError("No data available for vectorized backtesting")

        # Get unified timeline
        all_timestamps = set()
        for data in all_data.values():
            all_timestamps.update(data["timestamp"])

        sorted_timestamps = sorted(all_timestamps)

        # Vectorized processing
        for timestamp in sorted_timestamps:
            # Calculate portfolio value at this timestamp
            position_value = 0
            current_prices = {}

            for symbol in trading_pairs:
                if symbol in all_data:
                    symbol_data = all_data[symbol]
                    timestamp_data = symbol_data[symbol_data["timestamp"] == timestamp]
                    if not timestamp_data.empty:
                        price = timestamp_data.iloc[0]["close"]
                        current_prices[symbol] = price
                        position_value += positions[symbol] * price

            total_value = cash + position_value

            # Record portfolio state
            portfolio_history.append({
                "timestamp": timestamp,
                "total_value": total_value,
                "cash": cash,
                "position_value": position_value,
                "positions": positions.copy(),
            })

            # Apply strategy signals if provided
            if strategy and hasattr(strategy, "generate_signals"):
                signals = strategy.generate_signals(timestamp, current_prices, positions, cash)
                # Process signals (simplified implementation)
                for signal in signals:
                    symbol = signal.get("symbol")
                    action = signal.get("action")
                    quantity = signal.get("quantity", 0)

                    if symbol in current_prices and action in ["buy", "sell"]:
                        price = current_prices[symbol]
                        trade_value = quantity * price
                        commission = trade_value * run_config.get("commission_rate", 0.001)

                        if action == "buy" and cash >= trade_value + commission:
                            cash -= trade_value + commission
                            positions[symbol] += quantity
                        elif action == "sell" and positions[symbol] >= quantity:
                            cash += trade_value - commission
                            positions[symbol] -= quantity

        # Store results for metrics calculation
        if portfolio_history:
            portfolio_df = pd.DataFrame(portfolio_history)
            portfolio_df.set_index("timestamp", inplace=True)
            services["equity_curve"] = portfolio_df["total_value"]

        self.logger.info(f"Vectorized backtest completed with {len(portfolio_history)} data points")

    async def _add_benchmark_analysis(self, results: dict[str, Any], config: BacktestConfig) -> None:
        """Add benchmark comparison analysis to results."""
        if not config.benchmark_symbol:
            return

        self.logger.info(f"Adding benchmark analysis against {config.benchmark_symbol}")

        try:
            # Load benchmark data (simplified - would need actual data loading)
            benchmark_data = self._data.get(config.benchmark_symbol)
            if benchmark_data is None:
                self.logger.warning(f"No benchmark data available for {config.benchmark_symbol}")
                return

            # Filter benchmark data for the same period
            start_date = pd.to_datetime(config.start_date)
            end_date = pd.to_datetime(config.end_date)
            benchmark_mask = (benchmark_data["timestamp"] >= start_date) & (benchmark_data["timestamp"] <= end_date)
            benchmark_period_data = benchmark_data[benchmark_mask].sort_values("timestamp")

            if benchmark_period_data.empty:
                self.logger.warning("No benchmark data for the specified period")
                return

            # Calculate benchmark return
            initial_benchmark_price = benchmark_period_data.iloc[0]["close"]
            final_benchmark_price = benchmark_period_data.iloc[-1]["close"]
            benchmark_return = (final_benchmark_price - initial_benchmark_price) / initial_benchmark_price

            # Add benchmark metrics to results
            results["benchmark_analysis"] = {
                "benchmark_symbol": config.benchmark_symbol,
                "benchmark_return": float(benchmark_return),
                "benchmark_return_pct": float(benchmark_return * 100),
                "initial_price": float(initial_benchmark_price),
                "final_price": float(final_benchmark_price),
            }

            # Calculate alpha and beta if we have portfolio returns
            if "metrics" in results:
                portfolio_return = results["metrics"].get("total_return_pct", 0) / 100
                alpha = portfolio_return - benchmark_return
                results["benchmark_analysis"]["alpha"] = float(alpha)
                results["benchmark_analysis"]["alpha_pct"] = float(alpha * 100)

                self.logger.info(f"Benchmark return: {benchmark_return:.2%}")
                self.logger.info(f"Alpha: {alpha:.2%}")

        except Exception as e:
            self.logger.exception("Error in benchmark analysis: ")
            results["benchmark_analysis"] = {"error": str(e)}

    def _calculate_enhanced_metrics(self, results: dict[str, Any], config: BacktestConfig) -> PerformanceMetrics | None:
        """Calculate enhanced performance metrics from results."""
        try:
            if "metrics" not in results:
                self.logger.warning("No base metrics available for enhancement")
                return None

            base_metrics = results["metrics"]
            enhanced = PerformanceMetrics.from_existing_metrics(base_metrics)

            # Add benchmark comparison if available
            if "benchmark_analysis" in results:
                benchmark_analysis = results["benchmark_analysis"]
                enhanced.benchmark_return = benchmark_analysis.get("benchmark_return")
                enhanced.alpha = benchmark_analysis.get("alpha")
                # Beta calculation would require more sophisticated analysis
                enhanced.beta = 1.0  # Placeholder

            # Calculate Calmar ratio (return/max_drawdown)
            if enhanced.max_drawdown > 0:
                enhanced.calmar_ratio = enhanced.annualized_return / enhanced.max_drawdown
            else:
                enhanced.calmar_ratio = float("inf") if enhanced.annualized_return > 0 else 0.0

            # Calculate volatility if we have equity curve
            if "equity_curve" in results:
                equity_curve = results["equity_curve"]
                if hasattr(equity_curve, "pct_change"):
                    returns = equity_curve.pct_change().dropna()
                    enhanced.volatility = float(returns.std() * np.sqrt(252))  # Annualized

            return enhanced

        except Exception:
            self.logger.exception("Error calculating enhanced metrics: ")
            return None

    def run_strategy_comparison(
        self,
        strategies: list[Any],
        config: BacktestConfig,
    ) -> dict[str, Any]:
        """Run backtests on multiple strategies for comparison.

        Args:
            strategies: List of strategy instances to compare
            config: BacktestConfig for all strategies

        Returns:
            Dictionary with results for each strategy
        """
        results = {}

        for i, strategy in enumerate(strategies):
            strategy_name = getattr(strategy, "name", f"Strategy_{i+1}")
            self.logger.info(f"Running backtest for {strategy_name}")

            try:
                # Run individual backtest - this would be async in practice
                strategy_results = asyncio.run(self.run_backtest(config, strategy))
                results[strategy_name] = strategy_results

            except Exception as e:
                self.logger.exception(f"Error testing {strategy_name}: ")
                results[strategy_name] = {"error": str(e)}

        # Add comparison summary
        results["comparison_summary"] = self._create_strategy_comparison_summary(results)

        return results

    def _create_strategy_comparison_summary(self, strategy_results: dict[str, Any]) -> dict[str, Any]:
        """Create a summary comparing strategy performance."""
        summary = {
            "best_total_return": {"strategy": None, "return": float("-inf")},
            "best_sharpe_ratio": {"strategy": None, "sharpe": float("-inf")},
            "lowest_drawdown": {"strategy": None, "drawdown": float("inf")},
            "most_trades": {"strategy": None, "trades": 0},
        }

        for strategy_name, results in strategy_results.items():
            if strategy_name == "comparison_summary" or "error" in results:
                continue

            enhanced_metrics = results.get("enhanced_metrics")
            if not enhanced_metrics:
                continue

            # Check for best total return
            if enhanced_metrics.total_return > summary["best_total_return"]["return"]:  # type: ignore[index]
                summary["best_total_return"] = {
                    "strategy": strategy_name,
                    "return": enhanced_metrics.total_return,
                }

            # Check for best Sharpe ratio
            if enhanced_metrics.sharpe_ratio > summary["best_sharpe_ratio"]["sharpe"]:  # type: ignore[index]
                summary["best_sharpe_ratio"] = {
                    "strategy": strategy_name,
                    "sharpe": enhanced_metrics.sharpe_ratio,
                }

            # Check for lowest drawdown
            if enhanced_metrics.max_drawdown < summary["lowest_drawdown"]["drawdown"]:  # type: ignore[index]
                summary["lowest_drawdown"] = {
                    "strategy": strategy_name,
                    "drawdown": enhanced_metrics.max_drawdown,
                }

            # Check for most trades
            if enhanced_metrics.total_trades > summary["most_trades"]["trades"]:  # type: ignore[index]
                summary["most_trades"] = {
                    "strategy": strategy_name,
                    "trades": enhanced_metrics.total_trades,
                }

        return summary

    def generate_report(self, results: dict[str, Any], output_path: str | None = None) -> str:
        """Generate a comprehensive backtest report.

        Args:
            results: Backtest results dictionary
            output_path: Optional path to save the report

        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE BACKTESTING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Configuration section
        if "config" in results:
            config = results["config"]
            report_lines.append("CONFIGURATION:")
            report_lines.append(f"  Period: {config.get('start_date')} to {config.get('end_date')}")
            report_lines.append(f"  Initial Capital: ${config.get('initial_capital', 0):,.2f}")
            report_lines.append(f"  Symbols: {config.get('symbols', [])}")
            report_lines.append(f"  Mode: {config.get('mode', 'N/A')}")
            report_lines.append(f"  Commission Rate: {config.get('commission_rate', 0):.4f}")
            report_lines.append("")

        # Performance metrics section
        enhanced_metrics = results.get("enhanced_metrics")
        if enhanced_metrics:
            report_lines.append("PERFORMANCE METRICS:")
            report_lines.append(f"  Total Return: {enhanced_metrics.total_return:.2%}")
            report_lines.append(f"  Annualized Return: {enhanced_metrics.annualized_return:.2%}")
            report_lines.append(f"  Volatility: {enhanced_metrics.volatility:.2%}")
            report_lines.append(f"  Sharpe Ratio: {enhanced_metrics.sharpe_ratio:.2f}")
            report_lines.append(f"  Sortino Ratio: {enhanced_metrics.sortino_ratio:.2f}")
            report_lines.append(f"  Calmar Ratio: {enhanced_metrics.calmar_ratio:.2f}")
            report_lines.append(f"  Max Drawdown: {enhanced_metrics.max_drawdown:.2%}")
            report_lines.append(f"  Win Rate: {enhanced_metrics.win_rate:.1%}")
            report_lines.append(f"  Total Trades: {enhanced_metrics.total_trades}")
            report_lines.append(f"  Profit Factor: {enhanced_metrics.profit_factor:.2f}")
            report_lines.append("")

        # Benchmark comparison section
        if "benchmark_analysis" in results:
            benchmark = results["benchmark_analysis"]
            if "error" not in benchmark:
                report_lines.append("BENCHMARK COMPARISON:")
                report_lines.append(f"  Benchmark Symbol: {benchmark.get('benchmark_symbol', 'N/A')}")
                report_lines.append(f"  Benchmark Return: {benchmark.get('benchmark_return_pct', 0):.2f}%")
                report_lines.append(f"  Alpha: {benchmark.get('alpha_pct', 0):.2f}%")
                report_lines.append("")

        # Execution details
        if "execution_time_seconds" in results:
            report_lines.append("EXECUTION DETAILS:")
            report_lines.append(f"  Execution Time: {results['execution_time_seconds']:.2f} seconds")
            report_lines.append(f"  Framework Version: {results.get('framework_version', 'N/A')}")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(report_content)
                self.logger.info(f"Report saved to {output_path}")
            except Exception:
                self.logger.exception("Error saving report: ")

        return report_content

    def optimize_parameters(
        self,
        strategy_class: Any,
        parameter_grid: dict[str, list[Any]],
        config: BacktestConfig,
        optimization_metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """Optimize strategy parameters using grid search.

        Args:
            strategy_class: Strategy class to instantiate with different parameters
            parameter_grid: Dictionary of parameter names to lists of values to test
            config: Base BacktestConfig for optimization
            optimization_metric: Metric to optimize ("sharpe_ratio", "total_return", etc.)

        Returns:
            Dictionary with optimization results
        """
        import itertools

        self.logger.info(f"Starting parameter optimization with {optimization_metric}")

        # Generate all parameter combinations
        param_names = list[Any](parameter_grid.keys())
        param_values = list[Any](parameter_grid.values())
        param_combinations = list[Any](itertools.product(*param_values))

        self.logger.info(f"Testing {len(param_combinations)} parameter combinations")

        optimization_results = []
        best_result = None
        best_metric_value = float("-inf")

        for i, param_combo in enumerate(param_combinations):
            # Create parameter dictionary for this combination
            params = dict[str, Any](zip(param_names, param_combo, strict=False))

            try:
                # Instantiate strategy with these parameters
                strategy = strategy_class(**params)

                # Run backtest
                results = asyncio.run(self.run_backtest(config, strategy))

                # Extract optimization metric
                enhanced_metrics = results.get("enhanced_metrics")
                if enhanced_metrics:
                    metric_value = getattr(enhanced_metrics, optimization_metric, 0)

                    optimization_results.append({
                        "parameters": params,
                        "metric_value": metric_value,
                        "enhanced_metrics": enhanced_metrics,
                        "results": results,
                    })

                    # Check if this is the best result so far
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = optimization_results[-1]

                    self.logger.info(f"Combination {i+1}/{len(param_combinations)}: "
                                   f"{optimization_metric}={metric_value:.4f}, params={params}")

            except Exception as e:
                self.logger.exception(f"Error testing parameter combination {params}: ")
                optimization_results.append({
                    "parameters": params,
                    "error": str(e),
                })

        return {
            "optimization_metric": optimization_metric,
            "best_result": best_result,
            "all_results": optimization_results,
            "parameter_grid": parameter_grid,
            "total_combinations_tested": len(param_combinations),
        }

    def load_data_from_csv(self, file_path: str, symbol_column: str = "symbol") -> None:
        """Load historical data from CSV file for backtesting.

        Args:
            file_path: Path to CSV file
            symbol_column: Column name containing symbol/trading pair names
        """
        try:
            self.logger.info(f"Loading data from CSV: {file_path}")

            df = pd.read_csv(file_path)

            # Ensure timestamp column exists and is properly formatted
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise BacktestError("CSV file must contain a 'timestamp' column")

            # Required OHLCV columns
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise BacktestError(f"Missing required columns: {missing_columns}")

            # Split data by symbol if symbol column exists
            if symbol_column in df.columns:
                for symbol in df[symbol_column].unique():
                    symbol_data = df[df[symbol_column] == symbol].copy()
                    symbol_data = symbol_data.sort_values("timestamp")
                    self._data[symbol] = symbol_data
                    self.logger.info(f"Loaded {len(symbol_data)} data points for {symbol}")
            else:
                # Single symbol data
                df = df.sort_values("timestamp")
                symbol = "DEFAULT_SYMBOL"
                self._data[symbol] = df
                self.logger.info(f"Loaded {len(df)} data points for {symbol}")

            self.logger.info(f"Successfully loaded data for {len(self._data)} symbols")

        except Exception as e:
            self.logger.exception("Error loading data from CSV: ")
            raise BacktestError(f"Failed to load data from CSV: {e}")

    def get_data_summary(self) -> dict[str, Any]:
        """Get summary information about loaded data.

        Returns:
            Dictionary with data summary statistics
        """
        if not self._data:
            return {"error": "No data loaded"}

        summary = {
            "symbols": list[Any](self._data.keys()),
            "symbol_count": len(self._data),
            "symbol_details": {},
        }

        for symbol, data in self._data.items():
            if not data.empty:
                summary["symbol_details"][symbol] = {  # type: ignore[index]
                    "data_points": len(data),
                    "start_date": data["timestamp"].min().isoformat(),
                    "end_date": data["timestamp"].max().isoformat(),
                    "columns": list[Any](data.columns),
                }

        return summary

    def clear_data(self) -> None:
        """Clear all loaded historical data."""
        self._data.clear()
        self.logger.info("Cleared all historical data")


class EnterpriseHistoricalDataLoader:
    """Enterprise-grade historical data loader with multiple sources, caching, and validation."""

    def __init__(
        self,
        config: dict[str, Any],
        config_manager: ConfigManagerProtocol,
        logger: logging.Logger,
    ) -> None:
        """Initialize the enterprise data loader.

        Args:
            config: Loader-specific configuration
            config_manager: Global configuration manager
            logger: Logger instance for comprehensive logging
        """
        self.config = config
        self.config_manager = config_manager
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Data providers registry
        self._providers: dict[str, Any] = {}

        # Cache infrastructure
        self._memory_cache: dict[str, Any] = {}
        self._disk_cache_enabled = config.get("cache_enabled", True)
        self._disk_cache_path = Path(config.get("disk_cache_path", "./cache/backtest_data"))

        # Performance tracking
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "provider_requests": 0,
            "failed_requests": 0,
            "total_data_points_loaded": 0,
        }

        # Request tracking for rate limiting
        self._request_times: dict[str, list[dt.datetime]] = {}

    async def initialize(self) -> None:
        """Initialize the enterprise data loader and its providers."""
        try:
            self.logger.info("Initializing Enterprise Historical Data Loader")

            # Create cache directory
            if self._disk_cache_enabled:
                self._disk_cache_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Disk cache enabled at: {self._disk_cache_path}")

            # Initialize data providers
            await self._initialize_providers()

            self.logger.info(
                f"Enterprise data loader initialized with {len(self._providers)} providers",
            )

        except Exception:
            self.logger.exception("Failed to initialize enterprise data loader: ")
            raise

    def _get_config_dict(self) -> dict[str, Any]:
        """Convert ConfigManagerProtocol to dictionary for provider initialization.

        This method provides a robust way to extract configuration as a dictionary
        from various ConfigManager implementations, ensuring compatibility with
        providers that expect dict[str, Any].

        Returns:
            Dictionary containing all configuration values
        """
        config_dict: dict[str, Any] = {}

        try:
            # Try the standard get_all method first
            if hasattr(self.config_manager, "get_all") and callable(self.config_manager.get_all):
                config_dict = self.config_manager.get_all()
                if isinstance(config_dict, dict):
                    return config_dict

            # Fallback to internal _config attribute if available
            if hasattr(self.config_manager, "_config"):
                internal_config = getattr(self.config_manager, "_config", None)
                if isinstance(internal_config, dict):
                    return internal_config

            # Last resort: build config dict from known keys
            # This ensures we always return a valid config dict
            self.logger.warning(
                "ConfigManager doesn't expose get_all() or _config, "
                "building minimal config dict",
            )

            # Define critical configuration sections
            config_sections = [
                "database", "api", "backtest", "risk_manager",
                "portfolio_manager", "execution_handler",
            ]

            for section in config_sections:
                section_config: Any = self.config_manager.get(section, {})
                if section_config:
                    config_dict[section] = section_config

        except Exception:
            self.logger.exception("Error extracting config dict: ")
            # Return empty dict rather than failing

        return config_dict

    async def _initialize_providers(self) -> None:
        """Initialize available data providers based on configuration."""
        providers_config = self.config.get("providers", ["local_files"])

        for provider_name in providers_config:
            try:
                if provider_name == "local_files":
                    self._providers["local_files"] = LocalFileDataProvider(
                        config=self.config_manager,
                        logger=self.logger,
                    )
                elif provider_name == "database":
                    # Import provider dynamically to avoid circular imports
                    from gal_friday.providers.database_provider import (
                        DatabaseDataProvider,
                    )

                    # Get configuration as dictionary using our robust helper method
                    config_dict = self._get_config_dict()

                    self._providers["database"] = DatabaseDataProvider(
                        config=config_dict,
                        logger=self.logger,
                    )
                elif provider_name == "api":
                    # Import provider dynamically to avoid circular imports
                    from gal_friday.providers.api_provider import APIDataProvider

                    # Get configuration as dictionary using our robust helper method
                    config_dict = self._get_config_dict()

                    self._providers["api"] = APIDataProvider(
                        config=config_dict,
                        logger=self.logger,
                    )

                self.logger.info(f"Initialized data provider: {provider_name}")

            except Exception as e:
                self.logger.warning(f"Failed to initialize provider {provider_name}: {e}")

    async def load_historical_data(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Load historical data with comprehensive error handling and caching.

        Args:
            request: Data request parameters including symbol, dates, sources, etc.

        Returns:
            Dictionary containing loaded data and metadata, or None if failed
        """
        self._stats["total_requests"] += 1

        try:
            # Generate cache key for this request
            cache_key = self._generate_cache_key(request)

            # Try cache first if enabled
            if request.get("cache_result", True):
                cached_result = await self._get_from_cache(cache_key)
                if cached_result:
                    self._stats["cache_hits"] += 1
                    self.logger.debug(f"Cache hit for {request['symbol']}")
                    return cached_result

            # Load from providers
            result = await self._load_from_providers(request)

            # Cache the result if successful
            if result and request.get("cache_result", True):
                await self._store_in_cache(cache_key, result)

            if result:
                self._stats["total_data_points_loaded"] += len(result.get("data", []))

            return result

        except Exception:
            self._stats["failed_requests"] += 1
            self.logger.exception(f"Failed to load data for {request.get('symbol', 'unknown')}: ")
            return None

    async def _load_from_providers(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Load data from available providers with fallback logic."""
        symbol = request["symbol"]
        data_source = request.get("data_source", "auto")
        fallback_sources = request.get("fallback_sources", ["local_files"])

        # Determine providers to try
        providers_to_try = []

        if data_source == "auto":
            # Use intelligent provider selection
            providers_to_try = self._select_optimal_providers(request)
        elif data_source in self._providers:
            providers_to_try = [data_source]
        else:
            # Use fallback sources
            providers_to_try = [src for src in fallback_sources if src in self._providers]

        if not providers_to_try:
            self.logger.error(f"No available providers for {symbol}")
            return None

        # Try each provider until successful
        for provider_name in providers_to_try:
            try:
                self.logger.debug(f"Trying provider {provider_name} for {symbol}")

                provider = self._providers[provider_name]
                result = await provider.load_data(request)

                if result and result.get("data") is not None:
                    # Validate data quality if requested
                    if request.get("validate_data", True):
                        quality_score = self._validate_data_quality(result["data"], symbol)
                        quality_threshold = request.get("quality_threshold", 0.8)

                        if quality_score < quality_threshold:
                            self.logger.warning(
                                f"Data quality for {symbol} from {provider_name} "
                                f"({quality_score:.2f}) below threshold ({quality_threshold:.2f})",
                            )
                            continue  # Try next provider

                    # Success - add metadata
                    result.update({
                        "data_source": provider_name,
                        "cache_hit": False,
                        "quality_score": quality_score if "quality_score" in locals() else None,
                    })

                    self._stats["provider_requests"] += 1
                    self.logger.debug(f"Successfully loaded {symbol} from {provider_name}")
                    return result  # type: ignore[no-any-return]

            except Exception as e:
                self.logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                continue

        # All providers failed
        self.logger.error(f"All providers failed for {symbol}")
        return None

    def _select_optimal_providers(self, request: dict[str, Any]) -> list[str]:
        """Select optimal data providers based on request characteristics."""
        providers = []

        # Prioritize based on data availability and performance
        if "local_files" in self._providers:
            providers.append("local_files")  # Fastest for backtesting

        if "database" in self._providers:
            providers.append("database")  # Good for large datasets

        if "api" in self._providers:
            providers.append("api")  # Most comprehensive but slowest

        return providers

    def _generate_cache_key(self, request: dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        key_parts = [
            request["symbol"],
            request["start_date"].isoformat(),
            request["end_date"].isoformat(),
            str(request.get("include_volume", True)),
            str(request.get("data_source", "auto")),
        ]
        return "_".join(key_parts)

    async def _get_from_cache(self, cache_key: str) -> dict[str, Any] | None:
        """Retrieve data from cache (memory first, then disk)."""
        # Try memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]  # type: ignore[no-any-return]

        # Try disk cache
        if self._disk_cache_enabled:
            cache_file = self._disk_cache_path / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    import pickle
                    with cache_file.open("rb") as f:
                        cached_data = pickle.load(f)

                    # Store in memory cache for next time
                    self._memory_cache[cache_key] = cached_data
                    return cached_data  # type: ignore[no-any-return]

                except Exception as e:
                    self.logger.warning(f"Failed to load from disk cache: {e}")

        return None

    async def _store_in_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        """Store data in cache (both memory and disk)."""
        # Store in memory cache
        self._memory_cache[cache_key] = data

        # Store in disk cache
        if self._disk_cache_enabled:
            try:
                import pickle
                cache_file = self._disk_cache_path / f"{cache_key}.pkl"
                with cache_file.open("wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                self.logger.warning(f"Failed to store in disk cache: {e}")

    def _validate_data_quality(self, df: pd.DataFrame, symbol: str) -> float:
        """Validate data quality and return a score between 0 and 1."""
        if df.empty:
            return 0.0

        quality_factors = []

        # Check for missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        quality_factors.append(1.0 - missing_ratio)

        # Check OHLCV data consistency
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # High >= max(open, low, close)
            high_valid = ((df["high"] >= df["open"]) & (df["high"] >= df["low"]) &
                         (df["high"] >= df["close"])).mean()

            # Low <= min(open, high, close)
            low_valid = ((df["low"] <= df["open"]) & (df["low"] <= df["high"]) &
                        (df["low"] <= df["close"])).mean()

            quality_factors.extend([high_valid, low_valid])

        # Check for reasonable data distribution
        if "close" in df.columns and len(df) > 1:
            price_changes = df["close"].pct_change().abs()
            extreme_moves = (price_changes > 1.0).sum()  # >100% moves
            extreme_ratio = extreme_moves / len(price_changes)
            quality_factors.append(1.0 - min(extreme_ratio, 1.0))

        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0

    def get_statistics(self) -> dict[str, Any]:
        """Get loader performance statistics."""
        return self._stats.copy()


class LocalFileDataProvider:
    """Data provider for loading from local files (CSV, Parquet, etc.)."""

    def __init__(self, config: ConfigManagerProtocol, logger: logging.Logger) -> None:
        """Initialize the local file data provider."""
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def load_data(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Load data from local files."""
        try:
            symbol = request["symbol"]
            start_date = request["start_date"]
            end_date = request["end_date"]

            # Get data path from configuration
            data_path: Any = self.config.get("backtest.data_path")
            if not data_path:
                self.logger.error("No data path configured for local file provider")
                return None

            # Load data based on file format
            # Ensure data_path is a string or PathLike
            if isinstance(data_path, str | Path):
                path_obj = Path(data_path)
            else:
                self.logger.error(f"Invalid data path type: {type(data_path)}")
                return None

            if path_obj.suffix.lower() == ".csv":
                df = pd.read_csv(str(path_obj), parse_dates=["timestamp"])
            elif path_obj.suffix.lower() == ".parquet":
                df = pd.read_parquet(str(path_obj))
            else:
                self.logger.error(f"Unsupported file format: {path_obj.suffix}")
                return None

            # Filter by symbol if multi-symbol file
            if "symbol" in df.columns:
                df = df[df["symbol"] == symbol]
            elif "pair" in df.columns:
                df = df[df["pair"] == symbol]

            # Filter by date range
            if "timestamp" in df.columns:
                df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

            # Validate required columns
            required_cols = {"open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                self.logger.error(f"Missing required columns: {missing}")
                return None

            if df.empty:
                self.logger.warning(f"No data found for {symbol} in specified date range")
                return None

            return {"data": df, "source": "local_file"}

        except Exception:
            self.logger.exception("Failed to load data from local file: ")
            return None


class DatabaseDataProviderAdapter:
    """Adapter to use the production DatabaseDataProvider in backtesting."""

    def __init__(self, config: ConfigManagerProtocol, logger: logging.Logger) -> None:
        """Initialize the database data provider adapter."""
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Import and initialize the production DatabaseDataProvider
        from .providers.database_provider import DatabaseDataProvider
        config_dict = {}
        if hasattr(config, "get_all"):
            config_dict = config.get_all()
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        self._provider = DatabaseDataProvider(config_dict, logger)
        self._initialized = False

    async def load_data(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Load data from database sources using production provider."""
        try:
            # Initialize provider if needed
            if not self._initialized:
                await self._provider.initialize()
                self._initialized = True

            # Convert request to DataRequest format
            from .simulated_market_price_service import DataRequest

            data_request = DataRequest(
                symbol=request.get("symbol", "XRP/USD"),
                start_date=request.get("start_date", dt.datetime.now() - dt.timedelta(days=30)),
                end_date=request.get("end_date", dt.datetime.now()),
                frequency=request.get("frequency", "1m"),
            )

            # Fetch data
            historical_points = await self._provider.fetch_data(data_request)

            if not historical_points:
                return None

            # Convert to DataFrame format expected by backtesting
            data = []
            for point in historical_points:
                data.append({
                    "timestamp": point.timestamp,
                    "open": point.open,
                    "high": point.high,
                    "low": point.low,
                    "close": point.close,
                    "volume": point.volume,
                })

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)

            return {"data": df, "source": "database"}

        except Exception:
            self.logger.exception("Failed to load data from database: ")
            return None

    async def cleanup(self) -> None:
        """Clean up database connections."""
        if self._initialized:
            await self._provider.cleanup()


class APIDataProviderAdapter:
    """Adapter to use the production APIDataProvider in backtesting."""

    def __init__(self, config: ConfigManagerProtocol, logger: logging.Logger) -> None:
        """Initialize the API data provider adapter."""
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Import and initialize the production APIDataProvider
        from .providers.api_provider import APIDataProvider
        config_dict = {}
        if hasattr(config, "get_all"):
            config_dict = config.get_all()
        elif hasattr(config, "__dict__"):
            config_dict = config.__dict__
        self._provider = APIDataProvider(config_dict, logger)
        self._initialized = False

    async def load_data(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Load data from external APIs using production provider."""
        try:
            # Initialize provider if needed
            if not self._initialized:
                await self._provider.initialize()
                self._initialized = True

            # Convert request to DataRequest format
            from .simulated_market_price_service import DataRequest

            data_request = DataRequest(
                symbol=request.get("symbol", "XRP/USD"),
                start_date=request.get("start_date", dt.datetime.now() - dt.timedelta(days=30)),
                end_date=request.get("end_date", dt.datetime.now()),
                frequency=request.get("frequency", "1m"),
            )

            # Fetch data
            historical_points = await self._provider.fetch_data(data_request)

            if not historical_points:
                return None

            # Convert to DataFrame format expected by backtesting
            data = []
            for point in historical_points:
                data.append({
                    "timestamp": point.timestamp,
                    "open": point.open,
                    "high": point.high,
                    "low": point.low,
                    "close": point.close,
                    "volume": point.volume,
                })

            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index("timestamp", inplace=True)

            return {"data": df, "source": "api"}

        except Exception:
            self.logger.exception("Failed to load data from API: ")
            return None

    async def cleanup(self) -> None:
        """Clean up API connections."""
        if self._initialized:
            await self._provider.cleanup()
