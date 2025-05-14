"""Provide a backtesting environment for algorithmic trading strategies.

This module contains the BacktestingEngine which orchestrates backtesting simulations
using historical data. It handles loading data, initializing simulation services,
executing the simulation, and calculating performance metrics.
"""

# Backtesting Engine Module
# Adjusted imports to fix F401 and E501
import asyncio
from datetime import datetime
from decimal import Decimal, InvalidOperation
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np  # Add numpy import for np references
import pandas as pd

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
        def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
            """Return a series of ATR values or None when TA-Lib is not installed."""
            log.error("TA-Lib not installed. Cannot calculate ATR.")
            # Return Series with same index to avoid potential issues later
            return pd.Series([None] * len(high), index=high.index)

    ta = TaLib()

from .config_manager import ConfigManager

# Import necessary components for instantiation
from .core.events import ExecutionReportEvent

# LoggerService is imported locally where needed or via TYPE_CHECKING
# from .logger_service import LoggerService # Removed F401

# PubSubManager is imported and aliased locally where needed or via TYPE_CHECKING
# from .core.pubsub import PubSubManager as CorePubSubManager # Removed F401

# Set up logging
log = logging.getLogger(__name__)


def create_placeholder_class(name: str, **methods: Any) -> Type[Any]:
    """Create a placeholder class with specified async methods."""
    # Creates a simple class with an __init__ that does nothing
    # and async methods (passed in `methods`) that just `asyncio.sleep(0)`.
    return type(
        name,
        (),
        {
            "__init__": lambda *args, **kwargs: None,
            **{method_name: lambda *args, **kwargs: asyncio.sleep(0) for method_name in methods},
        },
    )


# Type hints and imports for static analysis (e.g., mypy)
if TYPE_CHECKING:
    from .core.pubsub import PubSubManager
    from .feature_engine import FeatureEngine
    from .portfolio_manager import PortfolioManager
    from .prediction_service import PredictionService
    from .risk_manager import RiskManager
    from .simulated_execution_handler import SimulatedExecutionHandler
    from .simulated_market_price_service import SimulatedMarketPriceService
    from .strategy_arbitrator import StrategyArbitrator
else:
    # Import implementations with fallbacks for runtime if modules are missing
    def import_with_fallback(module_path: str, class_name: str, methods: List[str]) -> Any:
        """Import a class with fallback to placeholder if not available."""
        try:
            # Try to import from the primary location (relative import)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except ImportError:
            # Create placeholder class if import fails
            log.warning(f"{class_name} not found at {module_path}, using placeholder.")
            return create_placeholder_class(class_name, **dict.fromkeys(methods))

    # Define imports with fallbacks
    # Ensure PubSubManager only attempts to import from .core.pubsub
    PubSubManager = import_with_fallback(
        ".core.pubsub", "PubSubManager", ["publish", "subscribe", "start", "stop"]
    )

    SimulatedMarketPriceService = import_with_fallback(
        ".simulated_market_price_service",
        "SimulatedMarketPriceService",
        ["update_time", "get_latest_price", "start", "stop"],
    )

    PortfolioManager = import_with_fallback(
        ".portfolio_manager", "PortfolioManager", ["start", "stop"]
    )

    RiskManager = import_with_fallback(".risk_manager", "RiskManager", ["start", "stop"])

    SimulatedExecutionHandler = import_with_fallback(
        ".simulated_execution_handler", "SimulatedExecutionHandler", ["start", "stop"]
    )

    StrategyArbitrator = import_with_fallback(
        ".strategy_arbitrator", "StrategyArbitrator", ["start", "stop"]
    )

    PredictionService = import_with_fallback(
        ".prediction_service", "PredictionService", ["start", "stop"]
    )

    FeatureEngine = import_with_fallback(".feature_engine", "FeatureEngine", ["start", "stop"])


# Helper class to adapt standard logger to LoggerService interface for SimExecHandler
class StandardLoggerAdapter:
    """Adapts a standard Python logger to the LoggerService interface."""

    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance
        self._source_module: Optional[str] = None # Can be set externally if needed

    def _log_with_source(
        self,
        level: int,
        msg: str,
        source_module: Optional[str] = None,
        exc_info: Optional[bool] = None,
        extra_context: Optional[Dict[str, Any]] = None, # Renamed context to avoid conflict
    ) -> None:
        effective_source = source_module or self._source_module or "UnknownSource"
        # Standard logger doesn't directly take 'context', usually passed via 'extra'
        # For simplicity, we'll just prepend source_module to the message.
        self.logger.log(level, f"[{effective_source}] {msg}", exc_info=exc_info)

    def log(
        self,
        level: int,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
        exc_info: Optional[bool] = None,
    ) -> None:
        self._log_with_source(level, msg, source_module, exc_info, context)

    def info(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
    ) -> None:
        self._log_with_source(logging.INFO, msg, source_module, context=context)

    def debug(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
    ) -> None:
        self._log_with_source(logging.DEBUG, msg, source_module, context=context)

    def warning(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
    ) -> None:
        self._log_with_source(logging.WARNING, msg, source_module, context=context)

    def error(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
        exc_info: Optional[bool] = None,
    ) -> None:
        self._log_with_source(logging.ERROR, msg, source_module, exc_info, context)

    def critical(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
        exc_info: Optional[bool] = None,
    ) -> None:
        self._log_with_source(logging.CRITICAL, msg, source_module, exc_info, context)


# Helper class for providing historical data to SimulatedExecutionHandler
class BacktestHistoricalDataProvider:  # Implements HistoricalDataService protocol (partially)
    """Provides historical data access for backtesting components."""

    def __init__(self, all_historical_data: Dict[str, pd.DataFrame], logger: logging.Logger):
        self._data: Dict[str, pd.DataFrame] = all_historical_data
        self.logger = logger
        self._validate_data()

    def _validate_data(self) -> None:
        for pair, df in self._data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.error(
                    f"Data for pair {pair} does not have a DatetimeIndex. "
                    "SimulatedExecutionHandler may fail."
                )
            if "atr" not in df.columns:
                self.logger.warning(
                    f"DataFrame for pair {pair} is missing 'atr' column. "
                    "Volatility-based slippage might not work correctly."
                )

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> Optional[pd.Series]:
        """Get the bar immediately following the given timestamp for the trading pair."""
        if trading_pair not in self._data:
            self.logger.warning(f"No data for {trading_pair} in BacktestHistoricalDataProvider.")
            return None
        pair_df = self._data[trading_pair]
        try:
            # Find all bars strictly after the given timestamp
            later_bars = pair_df[pair_df.index > timestamp]
            if not later_bars.empty:
                return later_bars.iloc[0]
            self.logger.warning(f"No 'next' bar available for {trading_pair} after {timestamp}.")
            return None
        except Exception as e:
            self.logger.error(
                f"Error in get_next_bar for {trading_pair} at {timestamp}: {e}", exc_info=True
            )
            return None

    def get_atr(
        self, trading_pair: str, timestamp: datetime, period: int = 14 # period unused here
    ) -> Optional[Decimal]:
        """Get the ATR value for the given timestamp for the trading pair.
        Assumes 'atr' column is pre-calculated and represents ATR valid at 'timestamp'.
        """
        if trading_pair not in self._data:
            self.logger.warning(
                f"No data for {trading_pair} in BacktestHistoricalDataProvider for ATR."
            )
            return None
        pair_df = self._data[trading_pair]
        try:
            # Try to get data at the exact timestamp
            if timestamp in pair_df.index:
                atr_val = pair_df.loc[timestamp, "atr"]
            else:
                # If exact timestamp not found, get the latest ATR at or before the timestamp
                prior_data = pair_df[pair_df.index <= timestamp]
                if not prior_data.empty:
                    atr_val = prior_data.iloc[-1]["atr"]
                    self.logger.debug(
                        f"ATR for {trading_pair} at {timestamp} (not exact match): "
                        f"using ATR from {prior_data.index[-1]}"
                    )
                else:
                    self.logger.warning(
                        f"No ATR data found at or before {timestamp} for {trading_pair} (get_atr)."
                    )
                    return None

            if pd.isna(atr_val):
                self.logger.warning(f"ATR is NaN for {trading_pair} at {timestamp}.")
                return None
            return Decimal(str(atr_val))
        except KeyError: # Should be caught by the check above, but as a safeguard
            self.logger.warning(
                 f"Timestamp {timestamp} or 'atr' column not found for ATR lookup for {trading_pair}."
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Error in get_atr for {trading_pair} at {timestamp}: {e}", exc_info=True
            )
            return None

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str # Unused in this simplified provider
    ) -> Optional[pd.DataFrame]:
        """Retrieve historical OHLCV data for a given pair and time range."""
        self.logger.debug(
            f"BacktestHistoricalDataProvider.get_historical_ohlcv called for {trading_pair}."
        )
        if trading_pair in self._data:
            # Ensure start_time and end_time are timezone-aware if df.index is
            pair_df = self._data[trading_pair]
            if pair_df.index.tz is not None:
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=pair_df.index.tz)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=pair_df.index.tz)
            return pair_df.loc[start_time:end_time].copy()
        self.logger.warning(f"No data for {trading_pair} to serve get_historical_ohlcv.")
        return None

    async def get_historical_trades(
        self, trading_pair: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Not implemented for this backtesting provider."""
        self.logger.warning(
            "BacktestHistoricalDataProvider.get_historical_trades called (not implemented)."
        )
        return None

# --- Helper Function for Reporting --- #
def calculate_performance_metrics(
    equity_curve: pd.Series, trade_log: List[Dict], initial_capital: Decimal
) -> Dict[str, Any]:
    """Calculate standard backtesting performance metrics."""
    if equity_curve.empty:
        log.warning("Equity curve is empty, cannot calculate metrics.")
        return {"error": "Equity curve is empty, cannot calculate metrics."}

    results: dict[str, Any] = {}
    # Ensure equity curve is numeric and float for calculations
    equity_curve = pd.to_numeric(equity_curve, errors="coerce").astype(float)
    equity_curve = equity_curve.dropna()  # Drop NaNs resulting from coercion
    if equity_curve.empty:
        log.warning("Equity curve became empty after numeric conversion.")
        return {"error": "Equity curve has no valid numeric data."}

    returns = equity_curve.pct_change().dropna()

    # Basic Returns
    final_equity_value = equity_curve.iloc[-1]
    final_equity = Decimal(str(final_equity_value))
    results["initial_capital"] = float(initial_capital)
    results["final_equity"] = float(final_equity)

    # Ensure all calculations use float types
    initial_capital_float = float(initial_capital)
    final_equity_float = float(final_equity)
    if initial_capital_float > 0:
        total_return_pct = ((final_equity_float / initial_capital_float) - 1.0) * 100.0
        results["total_return_pct"] = total_return_pct
    else:
        results["total_return_pct"] = 0.0

    # Calculate Annualized Return
    try:
        # Get the time span of the backtest in days
        if len(equity_curve) >= 2:
            first_date = equity_curve.index[0]
            last_date = equity_curve.index[-1]

            # Ensure timestamps are datetime objects
            if isinstance(first_date, (pd.Timestamp, datetime)):
                # Calculate duration in days
                duration_days = (last_date - first_date).total_seconds() / (60 * 60 * 24)

                if duration_days > 0:
                    # Calculate annualized return: (1 + total_return)^(365/duration_days) - 1
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
                    log.warning(
                        "Backtest duration is zero or negative days, "
                        "cannot calculate annualized return."
                    )
                    results["annualized_return_pct"] = (
                        total_return_pct  # Same as total return for very short periods
                    )
            else:
                log.warning("Equity curve index is not timestamp type: %s", type(first_date))
                results["annualized_return_pct"] = None
        else:
            log.warning("Insufficient data points in equity curve to calculate duration.")
            results["annualized_return_pct"] = None
    except Exception as e:
        log.error("Error calculating annualized return: %s", e)
        results["annualized_return_pct"] = None

    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    results["max_drawdown_pct"] = abs(max_drawdown * 100.0)

    # Sharpe Ratio (Simplified - Assumes daily returns, 0% risk-free)
    # More accurate version would need risk-free rate and adjust for period
    returns_std = returns.std()
    if not returns.empty and returns_std != 0:
        # Annualized for daily (approx 252 trading days)
        sharpe_ratio = (returns.mean() / returns_std) * np.sqrt(252)
        results["sharpe_ratio_annualized_approx"] = float(sharpe_ratio)
    else:
        results["sharpe_ratio_annualized_approx"] = 0.0

    # Sortino Ratio (Simplified - vs 0% target, annualised approx)
    downside_returns = returns[returns < 0]
    if not downside_returns.empty:
        downside_std = downside_returns.std()
        if downside_std != 0:
            sortino_ratio = returns.mean() / downside_std * np.sqrt(252)
            results["sortino_ratio_annualized_approx"] = float(sortino_ratio)
        else:
            # Avoid division by zero if no downside deviation
            results["sortino_ratio_annualized_approx"] = np.inf
    else:
        # No downside returns, Sortino is infinite
        results["sortino_ratio_annualized_approx"] = np.inf

    # Trade Stats
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

        # Use explicit conversion to handle division properly
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

        # Calculate average win/loss ratio
        avg_win = results["average_win"]
        avg_loss = results["average_loss"]
        if avg_loss != 0:
            results["avg_win_loss_ratio"] = float(str(abs(avg_win / avg_loss)))
        else:
            # If average loss is 0, ratio is infinite (or undefined)
            results["avg_win_loss_ratio"] = float("inf")

        # Calculate Average Holding Period
        try:
            holding_periods = []
            for trade in trade_log:
                if trade.get("entry_time") and trade.get("exit_time"):
                    try:
                        entry_time = pd.to_datetime(trade["entry_time"])
                        exit_time = pd.to_datetime(trade["exit_time"])

                        # Calculate duration in hours
                        duration_hours = (exit_time - entry_time).total_seconds() / 3600
                        holding_periods.append(duration_hours)
                    except Exception as e:
                        log.warning("Error parsing trade times: %s", e)

            if holding_periods:
                avg_holding_period = sum(holding_periods) / len(holding_periods)
                results["average_holding_period_hours"] = avg_holding_period

                # Also provide in days for convenience
                results["average_holding_period_days"] = avg_holding_period / 24

                log.debug("Average holding period: %.2f hours", avg_holding_period)
            else:
                log.warning(
                    "No valid trade durations found for average holding period calculation"
                )
                results["average_holding_period_hours"] = None
                results["average_holding_period_days"] = None
        except Exception as e:
            log.error("Error calculating average holding period: %s", e)
            results["average_holding_period_hours"] = None
            results["average_holding_period_days"] = None
    else:
        # Default values if no trades occurred
        results["total_pnl"] = 0.0
        results["winning_trades"] = 0
        results["losing_trades"] = 0
        results["win_rate_pct"] = 0.0
        results["gross_profit"] = 0.0
        results["gross_loss"] = 0.0
        results["profit_factor"] = 0.0
        results["average_trade_pnl"] = 0.0
        results["average_win"] = 0.0
        results["average_loss"] = 0.0
        results["avg_win_loss_ratio"] = 0.0
        results["average_holding_period_hours"] = None
        results["average_holding_period_days"] = None

    return results


class BacktestingEngine:
    """Orchestrates backtesting simulations using historical data."""

    def __init__(self, config_manager: "ConfigManager"):
        """
        Initialize the BacktestingEngine.

        Args
        ----
            config_manager: The application's configuration manager.
        """
        self.config = config_manager
        # Ensure config is an actual ConfigManager instance
        if not isinstance(self.config, ConfigManager):
            log.error(
                "BacktestingEngine received an invalid ConfigManager object: "
                f"{type(config_manager)}"
            )
            # Consider raising an error or handling this case appropriately
            # For now, let's try to proceed assuming it might work duck-type wise
            # but log severely.
            # raise TypeError("config_manager must be an instance of ConfigManager")

        # Attribute to store the execution report handler for unsubscribing
        from collections.abc import Coroutine
        from typing import Callable

        self._backtest_exec_report_handler: Optional[
            Callable[[ExecutionReportEvent], Coroutine[Any, Any, bool]]
        ] = None

        log.info("BacktestingEngine initialized.")

    def _load_historical_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load, clean, validate, and preprocess historical OHLCV data.

        Returns
        -------
            A dictionary mapping trading pairs to their historical data DataFrames,
            or None if loading fails.
        """
        try:
            config = self._get_backtest_config()
            if not self._validate_config(config):
                return None

            # Step 1: Load raw data from file
            all_data = self._load_raw_data(config["data_path"])
            if all_data is None:
                return None

            # Step 2: Clean and validate the loaded data
            all_data = self._clean_and_validate_data(
                all_data, config["start_date"], config["end_date"]
            )
            if all_data is None:
                return None

            # Step 3: Process data for each trading pair
            processed_data = self._process_pairs_data(all_data, config)
            if not processed_data:
                log.error("Failed to load/process data for any configured pairs.")
                return None

            return processed_data

        except Exception as e:
            log.exception("Unexpected error during historical data loading: %s", e)
            return None

    def _filter_pair_data(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Filter the main DataFrame for one specific trading pair.

        Args
        ----
            data: The full historical data DataFrame with multiple pairs.
            pair: The trading pair symbol to filter for.

        Returns
        -------
            A DataFrame containing only data for the specified pair.
        """
        pair_df = data[data["pair"] == pair].copy()

        if pair_df.empty:
            log.warning(
                "No data found for configured pair: %s in the loaded file/date range.",
                pair,
            )

        return pair_df

    def _get_backtest_config(self) -> Dict[str, Any]:
        """Get and prepare backtest configuration parameters."""
        return {
            "data_path": self.config.get("backtest.data_path"),
            "start_date": self.config.get("backtest.start_date"),
            "end_date": self.config.get("backtest.end_date"),
            "pairs": self.config.get_list("trading.pairs"),
            "needs_atr": (
                self.config.get("backtest.slippage_model", "fixed") == "volatility"
            ),  # Check if ATR is needed
            "atr_period": self.config.get_int("backtest.atr_period", 14),  # Default to 14 periods
            "initial_capital": self.config.get_decimal(
                "backtest.initial_capital", Decimal("100000")
            ),
            "output_path": self.config.get("backtest.output_path", "backtests/results"),
        }

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate all necessary backtesting configuration parameters.

        Performs comprehensive validation of the backtest configuration:
        - Required paths (data_path, output_path)
        - Date ranges (start_date, end_date)
        - Trading pairs
        - Initial capital and money management settings
        - Slippage settings and ATR period if using volatility-based slippage

        Args
        ----
            config: Dictionary containing configuration parameters.

        Returns
        -------
            Boolean indicating whether the configuration is valid.
        """
        validation_errors = []

        # Validate data path
        data_path = config.get("data_path")
        if not data_path:
            validation_errors.append("Historical data path not configured ('backtest.data_path').")
        elif not os.path.exists(data_path):
            validation_errors.append(f"Historical data path not found: '{data_path}'")

        # Validate date range
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        if not start_date:
            validation_errors.append("Backtest start_date not configured ('backtest.start_date').")
        if not end_date:
            validation_errors.append("Backtest end_date not configured ('backtest.end_date').")

        # Check date format validity if both dates are present
        if start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                if start_dt >= end_dt:
                    validation_errors.append(
                        "Invalid date range: start_date (%s) must be earlier "
                        "than end_date (%s).",
                        start_date,
                        end_date,
                    )
            except Exception as e:
                validation_errors.append(f"Invalid date format: {e!s}")

        # Validate trading pairs
        pairs = config.get("pairs", [])
        if not pairs:
            validation_errors.append("No trading pairs configured ('trading.pairs').")

        # Validate initial capital
        initial_capital = config.get("initial_capital")
        if not initial_capital:
            validation_errors.append(
                "Initial capital not configured ('backtest.initial_capital')."
            )
        else:
            try:
                # Ensure it can be converted to Decimal and is positive
                capital_decimal = Decimal(str(initial_capital))
                if capital_decimal <= 0:
                    validation_errors.append(
                        f"Initial capital must be positive: {initial_capital}"
                    )
            except (ValueError, TypeError, InvalidOperation) as e:
                validation_errors.append(f"Invalid initial capital value: {e!s}")

        # Validate output path
        output_path = config.get("output_path")
        if not output_path:
            validation_errors.append("Output path not configured ('backtest.output_path').")
        else:
            # Check if parent directory exists or can be created
            parent_dir = os.path.dirname(output_path)
            if parent_dir and not os.path.exists(parent_dir):
                try:
                    # Test if we can create it
                    os.makedirs(parent_dir, exist_ok=True)
                except OSError as e:
                    validation_errors.append(f"Cannot create output directory: {e!s}")

        # Validate slippage settings
        slippage_model = self.config.get("backtest.slippage_model", "fixed")
        if slippage_model not in ["fixed", "volatility", "volume", "none"]:
            validation_errors.append(
                f"Invalid slippage model: '{slippage_model}'. "
                "Valid options: 'fixed', 'volatility', 'volume', 'none'."
            )

        # If using volatility-based slippage, validate ATR period
        if slippage_model == "volatility":
            atr_period = config.get("atr_period")
            if not atr_period:
                validation_errors.append(
                    "ATR period not configured ('backtest.atr_period') "
                    "but required for volatility-based slippage."
                )
            elif not isinstance(atr_period, int) or atr_period <= 0:
                validation_errors.append(
                    f"Invalid ATR period: {atr_period}. Must be a positive integer."
                )

        # Log all validation errors
        if validation_errors:
            for error in validation_errors:
                log.error(f"Configuration error: {error}")
            return False

        log.info("Backtest configuration validated successfully.")
        return True

    def _load_raw_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Load raw data from the specified path."""
        try:
            log.info(f"Loading historical data from: {data_path}")
            return pd.read_parquet(data_path)
        except FileNotFoundError:
            log.error(f"Historical data file not found at path: {data_path}")
            return None
        except Exception as e:
            # Catch more general exceptions during file reading
            log.error(f"Error loading data from {data_path}: {e}")
            return None

    def _clean_and_validate_data(
        self, data: pd.DataFrame, start_date_str: str, end_date_str: str
    ) -> Optional[pd.DataFrame]:
        """Clean and validate the loaded data."""
        try:
            # 1. Ensure datetime index
            processed_data = self._ensure_datetime_index(data)  # Convert to proper datetime index
            if processed_data is None:
                return None

            # 2. Validate required columns exist
            if not self._validate_required_columns(processed_data):
                return None

            # 3. Convert date strings to timezone-aware datetime objects
            try:
                # Assume UTC if no timezone info present in strings
                start_date = pd.to_datetime(start_date_str).tz_localize(None).tz_localize("UTC")
                # Convert to timezone-aware
                end_date = pd.to_datetime(end_date_str).tz_localize(None).tz_localize("UTC")
                # Convert to timezone-aware
            except Exception as date_err:
                log.error(f"Invalid date format in config: {date_err}")
                return None

            log.info(f"Filtering data for range: {start_date} to {end_date}")

            # 4. Filter data by date range (inclusive)
            # Ensure index is also UTC for comparison
            if isinstance(processed_data.index, pd.DatetimeIndex):
                processed_data.index = processed_data.index.tz_convert("UTC")
            else:
                log.warning(
                    "Processed data index is not a DatetimeIndex, cannot convert timezone."
                )
                return None

            filtered_data = processed_data[
                (processed_data.index >= start_date) & (processed_data.index <= end_date)
            ]
            log.info(f"{len(filtered_data)} rows remaining after date filtering.")

            if filtered_data.empty:
                log.error("No data available for the specified date range.")
                return None

            return filtered_data

        except Exception as e:
            log.error(f"Error cleaning and validating data: {e}", exc_info=True)
            return None

    def _ensure_datetime_index(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure the DataFrame has a proper UTC datetime index."""
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is None:
                log.warning("Data index is timezone naive. Assuming UTC.")
                return data.tz_localize("UTC")
            if data.index.tz.zone != "UTC":  # type: ignore
                log.warning(f"Data index has timezone {data.index.tz}. Converting to UTC.")
                return data.tz_convert("UTC")  # Convert to UTC timezone
            return data  # Already UTC

        log.warning(
            "Loaded data does not have a DatetimeIndex. "
            "Attempting to set index from common timestamp columns..."
        )
        # Common timestamp column names
        ts_cols = ["timestamp", "time", "date", "datetime"]
        found_col = None
        for col in ts_cols:
            if col in data.columns:
                found_col = col
                break

        if not found_col:
            log.error(
                f"Cannot find a suitable timestamp column (tried: {ts_cols}) to set as index."
            )
            return None

        try:
            # Attempt conversion to datetime
            data[found_col] = pd.to_datetime(data[found_col], errors="coerce")
            # Drop rows where conversion failed
            data = data.dropna(subset=[found_col])
            if data.empty:
                log.error(f"No valid timestamps found in column '{found_col}'.")
                return None

            # Make timezone-aware (assume UTC if naive)
            if data[found_col].dt.tz is None:
                log.warning(
                    f"Timestamp column '{found_col}' is timezone naive. Localizing to UTC."
                )
                data[found_col] = data[found_col].dt.tz_localize("UTC")
            else:
                data[found_col] = data[found_col].dt.tz_convert("UTC")

            # Set and sort index
            return data.set_index(found_col).sort_index()

        except Exception as e:
            log.exception("Error converting or setting index using column '%s': %s", found_col, e)
            return None

    def _validate_required_columns(self, data: pd.DataFrame) -> bool:
        """Validate that all required columns are present."""
        required_cols = ["open", "high", "low", "close", "volume", "pair"]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            log.error("Loaded data missing required columns: %s", ", ".join(missing))
            return False
        return True

    def _process_pairs_data(
        self, data: pd.DataFrame, config: dict[str, Any]
    ) -> dict[str, pd.DataFrame]:
        """Process data for each trading pair specified in the config."""
        processed_data: dict[str, pd.DataFrame] = {}
        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        for pair in config["pairs"]:
            log.debug(f"Processing data for pair: {pair}")

            # Step 1: Filter data for this specific pair
            pair_df = self._filter_pair_data(data, pair)

            if pair_df.empty:
                continue  # Skip to next pair

            # Step 2: Convert OHLCV columns to Decimal type
            if not self._convert_ohlcv_to_decimal(pair_df, pair, ohlcv_cols):
                log.error("Skipping pair %s due to data conversion error.", pair)
                continue  # Skip this pair if conversion fails

            # Step 3: Handle potential NaN values
            self._handle_nan_values(pair_df, pair)

            # Step 4: Calculate ATR if needed by the slippage model
            if config["needs_atr"]:
                atr_period = config["atr_period"]
                pair_df_with_atr = self._calculate_atr(pair_df, pair, atr_period)
                if pair_df_with_atr is None:
                    log.error("Skipping pair %s due to ATR calculation error.", pair)
                    continue  # Skip if ATR calculation fails
                pair_df = pair_df_with_atr  # Update df if ATR was added

            processed_data[pair] = pair_df
            log.info("Successfully processed data for %s (%d rows).", pair, len(pair_df))

        return processed_data

    def _convert_ohlcv_to_decimal(self, df: pd.DataFrame, pair: str, columns: list[str]) -> bool:
        """Convert specified OHLCV columns to Decimal type in place."""
        try:
            for col in columns:
                # Ensure column exists before trying conversion
                if col not in df.columns:
                    log.error("Column '%s' not found for pair %s.", col, pair)
                    return False
                # Convert via float->string to handle various numeric types
                # Coerce errors to NaT/None which will be handled later
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                df[col] = numeric_col.apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
        except Exception:
            # Log specific column where error occurred if possible
            col_name = col if "col" in locals() else "unknown"
            log.exception(
                "Error converting column '%s' to Decimal for pair %s. Check data source format.",
                col_name,
                pair,
            )
            return False
        else:
            return True
    def _handle_nan_values(self, df: pd.DataFrame, pair: str) -> None:
        """Log warnings about NaN values found in the DataFrame."""
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            nan_summary = nan_counts[nan_counts > 0]  # Filter only cols with NaNs
            log.warning("NaN values found in data for %s:\n%s", pair, nan_summary.to_string())
            log.warning(
                "Proceeding with NaN values for %s. "
                "Downstream components (features, strategies) must handle them.",
                pair,
            )
            # Consider adding imputation logic here if desired (e.g., ffill)

    def _calculate_atr(
        self, df: pd.DataFrame, pair: str, atr_period: int
    ) -> Optional[pd.DataFrame]:
        """Calculate ATR using TA-Lib if available, add 'atr' column."""
        # Check if ATR already exists and is suitable (e.g., not all NaN)
        if "atr" in df.columns and df["atr"].notna().any():
            # Ensure it's Decimal type
            if not all(isinstance(x, Decimal) for x in df["atr"].dropna()):
                log.warning("Existing 'atr' column for %s is not Decimal. Converting.", pair)
                try:
                    df["atr"] = df["atr"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
                except Exception:
                    log.exception("Error converting ATR for %s to Decimal", pair)
                    return None  # Indicate failure
            log.info("Using existing 'atr' column for %s.", pair)
            return df

        log.info("Calculating ATR(%s) for %s...", atr_period, pair)
        try:
            # TA-Lib requires float inputs
            # Use .copy() to avoid SettingWithCopyWarning if df is a slice
            high_f = df["high"].astype(float).copy()
            low_f = df["low"].astype(float).copy()
            close_f = df["close"].astype(float).copy()

            # Fill NaNs temporarily for TA-Lib calculation if necessary
            # (TA-Lib might handle NaNs, but explicit handling is safer)
            # Consider a suitable fill strategy (e.g., ffill, mean)
            high_f.ffill(inplace=True)
            low_f.ffill(inplace=True)
            close_f.ffill(inplace=True)

            # Check for sufficient non-NaN data after filling
            if (
                high_f.notna().sum() < atr_period
                or low_f.notna().sum() < atr_period
                or close_f.notna().sum() < atr_period
            ):
                log.warning(
                    "Insufficient non-NaN data to calculate ATR(%s) for %s.", atr_period, pair
                )
                df["atr"] = pd.Series([None] * len(df), index=df.index)  # Add None column
                return df  # Return with None column

            # Use getattr to safely access the atr function from ta object
            # (which could be the real TA-Lib or our placeholder)
            atr_func = getattr(ta, "atr", None)
            if atr_func:
                atr_values = atr_func(high=high_f, low=low_f, close=close_f, timeperiod=atr_period)
                # Convert result back to Decimal
                df["atr"] = pd.Series(atr_values, index=df.index).apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) else None
                )
                log.info("ATR calculation complete for %s.", pair)
            else:
                # This case should ideally be hit only if TA-Lib failed import AND
                # the placeholder class somehow lost its atr method.
                log.error(
                    "ATR function not available (TA-Lib missing?). "
                    "Cannot calculate ATR for %s.", pair
                )
                df["atr"] = pd.Series([None] * len(df), index=df.index)
                # Ensure column exists even if calculation failed
                if "atr" not in df.columns:
                    df["atr"] = pd.Series([None] * len(df), index=df.index)

        except Exception:
            log.exception("Failed during ATR calculation for %s", pair)
            log.warning(
                "Proceeding without ATR for %s. "
                "Volatility-based slippage/risk rules might not work.",
                pair,
            )
            # Ensure 'atr' column exists even if calculation failed
            if "atr" not in df.columns:
                df["atr"] = pd.Series([None] * len(df), index=df.index)
        return df

    async def run_backtest(self) -> Optional[dict[str, Any]]:
        """Orchestrate and run the backtest simulation."""
        log.info("Starting backtest run...")
        start_run_time = datetime.now(tz=datetime.now().astimezone().tzinfo)
        services: Optional[dict[str, Any]] = None
        equity_curve: dict[datetime, Decimal] = {}
        trade_log: list[dict[str, Any]] = []

        try:
            # Step 1: Set up the backtest environment (config, output dir)
            setup_result = self._setup_backtest_environment(start_run_time)
            if not setup_result:
                log.error("Backtest environment setup failed.")
                return None
            run_config, run_output_dir = setup_result # Renamed config to run_config

            # Step 2: Load and prepare historical data
            # _load_historical_data uses self.config, not the run_config directly.
            # Ensure self.config is correctly reflecting what's needed for data loading.
            # The run_config from _setup_backtest_environment is mainly for run parameters.
            historical_data = self._load_historical_data() # Uses self.config
            if historical_data is None or not historical_data:
                log.error("Backtest failed: No historical data loaded or data is empty.")
                return None

            # Step 3: Initialize simulation services
            services = await self._initialize_backtest_services(historical_data, run_config)
            if not services:
                log.error("Failed to initialize backtesting services.")
                return None

            # Step 4: Execute simulation
            simulation_result = await self._execute_simulation(
                services, historical_data, run_config
            )
            if not simulation_result:
                log.error("Simulation execution failed or returned no results.")
                # Keep equity_curve and trade_log as empty if simulation fails
            else:
                equity_curve, trade_log = simulation_result

            # Step 5: Process and save results
            results = self._process_and_save_results(
                run_config, equity_curve, trade_log, run_output_dir, start_run_time
            )
            log.info("Backtest run finished successfully.")

        except Exception:
            log.exception("Unhandled exception during backtest run.")
            return None # Or return partial results if meaningful
        else:
            return results
        finally:
            if services:
                await self._shutdown_services(services)

    def _setup_backtest_environment(
        self, start_run_time: datetime
    ) -> Optional[tuple[dict[str, Any], str]]:
        """Set up the backtest environment including config and output directory.

        Args
        ----
            start_run_time: The start time of the backtest run.

        Returns
        -------
            A tuple of (config, output_directory) if successful, None otherwise.
        """
        try:
            # Get and validate configuration
            config = self._get_backtest_config()
            if not self._validate_config(config):
                log.error("Backtest configuration validation failed.")
                return None

            # Set up output directory
            run_output_dir = self._setup_output_directory(config["output_path"], start_run_time)
            if not run_output_dir:
                log.error("Failed to set up output directory.")
                return None

        except Exception:
            log.exception("Error during backtest environment setup")
            return None
        else:
            return config, run_output_dir

    def _update_equity_curve(
        self, services: dict[str, Any], timestamp: datetime, equity_curve: dict[datetime, Decimal]
    ) -> None:
        """Update the equity curve with the current portfolio value."""
        portfolio_manager = services.get("portfolio_manager")
        if portfolio_manager and hasattr(portfolio_manager, "get_current_state"):
            try:
                # Ensure get_current_state is not async, or await it if it is.
                # Assuming it's synchronous as per typical portfolio state access.
                current_state = portfolio_manager.get_current_state()
                if "total_equity" in current_state:
                    try:
                        current_equity = Decimal(str(current_state["total_equity"]))
                        equity_curve[timestamp] = current_equity
                    except (InvalidOperation, ValueError):
                        log.exception(
                            "Timestamp %s: Error converting total_equity '%s' to Decimal",
                            timestamp,
                            current_state["total_equity"],
                        )
                else:
                    log.warning(
                        "Timestamp %s: 'total_equity' not found in portfolio state.", timestamp
                    )
            except Exception: # Catch broader exceptions from get_current_state
                log.exception(
                    "Error getting/recording equity state at %s from portfolio_manager",
                    timestamp,
                )
        else:
            log.warning(
                "Portfolio manager or get_current_state not found at ts %s.", timestamp
            )

    def _update_market_price_service_time(
        self, services: dict[str, Any], timestamp: datetime
    ) -> None:
        """Update the simulated market price service with the current time."""
        market_price_service = services.get("market_price_service")
        if market_price_service:
            # Ensure service has the update_time method
            if hasattr(market_price_service, "update_time"):
                market_price_service.update_time(timestamp)
            else:
                log.warning("Market price service missing 'update_time' method.")
        else:
            log.warning("Market price service not found in services.")

    async def _execute_simulation(
        self,
        services: dict[str, Any],
        historical_data: dict[str, pd.DataFrame],
        run_config: dict[str, Any] # Renamed from config
    ) -> Optional[tuple[dict[datetime, Decimal], list[dict[str, Any]]]]:
        """Run the main simulation loop chronologically through historical data."""
        log.info("Starting simulation execution...")
        equity_curve: dict[datetime, Decimal] = {}
        # Trade log will be retrieved from PortfolioManager at the end

        market_price_service = services["market_price_service"]
        sim_exec_handler = services["sim_exec_handler"] # type: SimulatedExecutionHandler
        portfolio_manager = services["portfolio_manager"]

        # 1. Generate master timeline
        all_timestamps = set()
        for pair_df in historical_data.values():
            all_timestamps.update(pair_df.index.tolist())

        if not all_timestamps:
            log.error("No timestamps found in historical data. Cannot run simulation.")
            return equity_curve, [] # Return empty results

        sorted_timestamps = sorted(all_timestamps)
        log.info(
            "Simulation will run from %s to %s over %d unique timestamps.",
            sorted_timestamps[0],
            sorted_timestamps[-1],
            len(sorted_timestamps),
        )

        # 2. Main simulation loop
        for current_timestamp in sorted_timestamps:
            log.debug("Processing simulation step for timestamp: %s", current_timestamp)

            # a. Update Market Price Service & Publish MarketDataEvents
            # This service should internally fetch data for current_timestamp and publish
            if hasattr(market_price_service, "update_time") and \
               callable(market_price_service.update_time):
                await market_price_service.update_time(current_timestamp)
            else:
                log.error("MarketPriceService does not have an async update_time method.")
                # This is a critical failure, might need to stop simulation
                return equity_curve, []


            # b. Allow event propagation (async processing of market data, features, signals)
            await asyncio.sleep(0) # Yield control to the event loop

            # c. Check and process SL/TP orders
            # The SimulatedExecutionHandler's check_active_sl_tp expects current bar data.
            # We need to provide it for each relevant pair.
            for pair_symbol in run_config.get("pairs", []): # Iterate over configured pairs
                if pair_symbol in historical_data:
                    pair_df = historical_data[pair_symbol]
                    if current_timestamp in pair_df.index:
                        current_bar_for_pair = pair_df.loc[current_timestamp]
                        # Ensure sim_exec_handler is correct type if needed
                        await sim_exec_handler.check_active_sl_tp(
                            current_bar_for_pair, current_timestamp
                        )
                    # else: No data for this pair at this ts, skip SL/TP check.

            # d. Update equity curve
            # _update_equity_curve uses services dict, timestamp, and equity_curve dict
            self._update_equity_curve(services, current_timestamp, equity_curve)

        log.info("Simulation loop completed.")

        # 3. Retrieve trade log from PortfolioManager
        trade_log: list[dict[str, Any]] = []
        if hasattr(portfolio_manager, "get_trade_log") and \
           callable(portfolio_manager.get_trade_log):
            trade_log = portfolio_manager.get_trade_log()
            log.info("Retrieved %d trades from PortfolioManager.", len(trade_log))
        else:
            log.warning("PortfolioManager lacks get_trade_log; trade log will be empty.")

        return equity_curve, trade_log

    async def _initialize_backtest_services(
        self,
        historical_data: dict[str, pd.DataFrame],
        run_config: dict[str, Any] # Renamed from config
    ) -> Optional[dict[str, Any]]:
        """Initialize and configure all services required for the backtest simulation."""
        services: dict[str, Any] = {}
        try:
            log.info("Initializing backtesting services...")

            logger_adapter = StandardLoggerAdapter(log) # Using engine's main logger
            historical_data_provider = BacktestHistoricalDataProvider(historical_data, log)

            # 1. PubSubManager
            # Assuming PubSubManager takes config_manager and logger
            # It's imported via import_with_fallback
            services["pubsub_manager"] = PubSubManager(
                config_manager=self.config,
                logger=log # or logger_adapter if PubSub wants that interface
            )
            log.debug("PubSubManager initialized.")

            # 2. SimulatedMarketPriceService
            # Assuming it takes config, pubsub, historical_data_provider (or raw data)
            services["market_price_service"] = SimulatedMarketPriceService(
                config_manager=self.config,
                pubsub_manager=services["pubsub_manager"],
                historical_data_provider=historical_data_provider, # Or pass raw historical_data
                logger=log # or logger_adapter
            )
            log.debug("SimulatedMarketPriceService initialized.")

            # 3. PortfolioManager
            initial_capital = run_config.get("initial_capital", Decimal("100000"))
            if not isinstance(initial_capital, Decimal):
                initial_capital = Decimal(str(initial_capital))
            services["portfolio_manager"] = PortfolioManager(
                config_manager=self.config,
                pubsub_manager=services["pubsub_manager"],
                initial_capital=initial_capital,
                logger=log # or logger_adapter
            )
            log.debug("PortfolioManager initialized with capital: %s", initial_capital)

            # 4. SimulatedExecutionHandler (from simulated_execution_handler.py)
            # This one definitely needs the specific LoggerService interface via adapter
            # And HistoricalDataService via our provider
            from .simulated_execution_handler import SimulatedExecutionHandler
            services["sim_exec_handler"] = SimulatedExecutionHandler(
                config_manager=self.config, # Uses self.config for its own detailed settings
                pubsub_manager=services["pubsub_manager"],
                data_service=historical_data_provider,
                logger_service=logger_adapter
            )
            log.debug("SimulatedExecutionHandler initialized.")

            # 5. FeatureEngine
            services["feature_engine"] = FeatureEngine(
                config_manager=self.config,
                pubsub_manager=services["pubsub_manager"],
                market_price_service=services["market_price_service"], # Needs prices
                logger=log # or logger_adapter
            )
            log.debug("FeatureEngine initialized.")

            # 6. PredictionService
            services["prediction_service"] = PredictionService(
                config_manager=self.config,
                pubsub_manager=services["pubsub_manager"],
                feature_engine=services["feature_engine"], # Needs features
                logger=log # or logger_adapter
            )
            log.debug("PredictionService initialized.")

            # 7. RiskManager
            services["risk_manager"] = RiskManager(
                config_manager=self.config,
                pubsub_manager=services["pubsub_manager"],
                portfolio_manager=services["portfolio_manager"], # Needs portfolio state
                logger=log # or logger_adapter
            )
            log.debug("RiskManager initialized.")

            # 8. StrategyArbitrator
            services["strategy_arbitrator"] = StrategyArbitrator(
                config_manager=self.config,
                pubsub_manager=services["pubsub_manager"],
                prediction_service=services["prediction_service"], # Needs predictions
                risk_manager=services["risk_manager"], # To submit signals for approval
                logger=log # or logger_adapter
            )
            log.debug("StrategyArbitrator initialized.")

            # Start services
            for service_name, service_obj in services.items():
                if hasattr(service_obj, "start") and callable(service_obj.start):
                    log.info("Starting service: %s...", service_name)
                    await service_obj.start()
            log.info("All services started.")

        except Exception:
            log.exception("Failed to initialize one or more backtesting services.")
            # Attempt to stop any services that might have started
            await self._shutdown_services(services)
            return None
        else:
            return services

    async def _shutdown_services(self, services: dict[str, Any]) -> None:
        """Attempt to stop all services that have a stop() method."""
        log.info("Shutting down backtesting services...")
        for service_name, service_obj in reversed(list(services.items())): # Stop in reverse
            if hasattr(service_obj, "stop") and callable(service_obj.stop):
                try:
                    log.info("Stopping service: %s...", service_name)
                    await service_obj.stop()
                except Exception:
                    log.exception("Error stopping service %s", service_name)
        log.info("All services shut down.")

