"""Provide a backtesting environment for algorithmic trading strategies.

This module contains the BacktestingEngine which orchestrates backtesting simulations
using historical data. It handles loading data, initializing simulation services,
executing the simulation, and calculating performance metrics.
"""

# Backtesting Engine Module
# Adjusted imports to fix F401 and E501
import asyncio
import logging
import os
import uuid  # Add uuid import for generating UUIDs
from concurrent.futures import ProcessPoolExecutor  # Added missing import
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type, cast

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
from .core.events import EventType, ExecutionReportEvent, MarketDataOHLCVEvent

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
            **{method_name: (lambda *args, **kwargs: asyncio.sleep(0)) for method_name in methods},
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
            return create_placeholder_class(class_name, **{m: None for m in methods})

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


# --- Helper Function for Reporting --- #
def calculate_performance_metrics(  # noqa: C901 too complex
    equity_curve: pd.Series, trade_log: List[Dict], initial_capital: Decimal
) -> Dict[str, Any]:
    """Calculate standard backtesting performance metrics."""
    if equity_curve.empty:
        log.warning("Equity curve is empty, cannot calculate metrics.")
        return {"error": "Equity curve is empty, cannot calculate metrics."}

    results: Dict[str, Any] = {}
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
                        f"Calculated annualized return over {duration_days:.2f} days: "
                        f"{annualized_return:.2f}%"
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
                log.warning(f"Equity curve index is not timestamp type: {type(first_date)}")
                results["annualized_return_pct"] = None
        else:
            log.warning("Insufficient data points in equity curve to calculate duration.")
            results["annualized_return_pct"] = None
    except Exception as e:
        log.error(f"Error calculating annualized return: {e}")
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
                        log.warning(f"Error parsing trade times: {e}")

            if holding_periods:
                avg_holding_period = sum(holding_periods) / len(holding_periods)
                results["average_holding_period_hours"] = avg_holding_period

                # Also provide in days for convenience
                results["average_holding_period_days"] = avg_holding_period / 24

                log.debug(f"Average holding period: {avg_holding_period:.2f} hours")
            else:
                log.warning(
                    "No valid trade durations found for average holding period calculation"
                )
                results["average_holding_period_hours"] = None
                results["average_holding_period_days"] = None
        except Exception as e:
            log.error(f"Error calculating average holding period: {e}")
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

        Args:
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
        from typing import Callable, Coroutine

        self._backtest_exec_report_handler: Optional[
            Callable[[ExecutionReportEvent], Coroutine[Any, Any, bool]]
        ] = None

        log.info("BacktestingEngine initialized.")

    def _load_historical_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load, clean, validate, and preprocess historical OHLCV data.

        Returns:
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
            log.exception(f"Unexpected error during historical data loading: {e}", exc_info=True)
            return None

    def _filter_pair_data(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Filter the main DataFrame for one specific trading pair.

        Args:
            data: The full historical data DataFrame with multiple pairs.
            pair: The trading pair symbol to filter for.

        Returns:
            A DataFrame containing only data for the specified pair.
        """
        pair_df = data[data["pair"] == pair].copy()

        if pair_df.empty:
            log.warning(
                f"No data found for configured pair: {pair} " "in the loaded file/date range."
            )

        return pair_df

    def _get_backtest_config(self) -> Dict[str, Any]:
        """Get and prepare backtest configuration parameters."""
        return {
            "data_path": self.config.get("backtest.data_path"),
            "start_date": self.config.get("backtest.start_date"),
            "end_date": self.config.get("backtest.end_date"),
            "pairs": self.config.get_list("trading.pairs"),
            "needs_atr": self.config.get("backtest.slippage_model", "fixed")
            == "volatility",  # Check if ATR is needed
            "atr_period": self.config.get_int("backtest.atr_period", 14),  # Default to 14 periods
            "initial_capital": self.config.get_decimal(
                "backtest.initial_capital", Decimal("100000")
            ),
            "output_path": self.config.get("backtest.output_path", "backtests/results"),
        }

    def _validate_config(self, config: Dict[str, Any]) -> bool:  # noqa: C901 too complex
        """Validate all necessary backtesting configuration parameters.

        Performs comprehensive validation of the backtest configuration:
        - Required paths (data_path, output_path)
        - Date ranges (start_date, end_date)
        - Trading pairs
        - Initial capital and money management settings
        - Slippage settings and ATR period if using volatility-based slippage

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
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
                        f"Invalid date range: start_date ({start_date}) must be earlier "
                        f"than end_date ({end_date})."
                    )
            except Exception as e:
                validation_errors.append(f"Invalid date format: {str(e)}")

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
                validation_errors.append(f"Invalid initial capital value: {str(e)}")

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
                    validation_errors.append(f"Cannot create output directory: {str(e)}")

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
            log.error("Error cleaning and validating data: " f"{e}", exc_info=True)
            return None

    def _ensure_datetime_index(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure the DataFrame has a proper UTC datetime index."""
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is None:
                log.warning("Data index is timezone naive. Assuming UTC.")
                return data.tz_localize("UTC")
            elif data.index.tz.zone != "UTC":  # type: ignore
                log.warning(f"Data index has timezone {data.index.tz}. " "Converting to UTC.")
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
                "Cannot find a suitable timestamp column " f"(tried: {ts_cols}) to set as index."
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
                    f"Timestamp column '{found_col}' is timezone naive. " "Localizing to UTC."
                )
                data[found_col] = data[found_col].dt.tz_localize("UTC")
            else:
                data[found_col] = data[found_col].dt.tz_convert("UTC")

            # Set and sort index
            return data.set_index(found_col).sort_index()

        except Exception as e:
            log.error("Error converting or setting index using column " f"'{found_col}': {e}")
            return None

    def _validate_required_columns(self, data: pd.DataFrame) -> bool:
        """Validate that all required columns are present."""
        required_cols = ["open", "high", "low", "close", "volume", "pair"]
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            log.error("Loaded data missing required columns: " f"{', '.join(missing)}")
            return False
        return True

    def _process_pairs_data(
        self, data: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Process data for each trading pair specified in the config."""
        processed_data: Dict[str, pd.DataFrame] = {}
        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        for pair in config["pairs"]:
            log.debug(f"Processing data for pair: {pair}")

            # Step 1: Filter data for this specific pair
            pair_df = self._filter_pair_data(data, pair)

            if pair_df.empty:
                continue  # Skip to next pair

            # Step 2: Convert OHLCV columns to Decimal type
            if not self._convert_ohlcv_to_decimal(pair_df, pair, ohlcv_cols):
                log.error(f"Skipping pair {pair} due to data conversion error.")
                continue  # Skip this pair if conversion fails

            # Step 3: Handle potential NaN values
            self._handle_nan_values(pair_df, pair)

            # Step 4: Calculate ATR if needed by the slippage model
            if config["needs_atr"]:
                atr_period = config["atr_period"]
                pair_df_with_atr = self._calculate_atr(pair_df, pair, atr_period)
                if pair_df_with_atr is None:
                    log.error(f"Skipping pair {pair} due to ATR calculation error.")
                    continue  # Skip if ATR calculation fails
                pair_df = pair_df_with_atr  # Update df if ATR was added

            processed_data[pair] = pair_df
            log.info(f"Successfully processed data for {pair} ({len(pair_df)} rows).")

        return processed_data

    def _convert_ohlcv_to_decimal(self, df: pd.DataFrame, pair: str, columns: List[str]) -> bool:
        """Convert specified OHLCV columns to Decimal type in place."""
        try:
            for col in columns:
                # Ensure column exists before trying conversion
                if col not in df.columns:
                    log.error(f"Column '{col}' not found for pair {pair}.")
                    return False
                # Convert via float->string to handle various numeric types
                # Coerce errors to NaT/None which will be handled later
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                df[col] = numeric_col.apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
            return True
        except Exception as e:
            # Log specific column where error occurred if possible
            col_name = col if "col" in locals() else "unknown"
            log.error(
                f"Error converting column '{col_name}' to Decimal for "
                f"pair {pair}: {e}. Check data source format."
            )
            return False

    def _handle_nan_values(self, df: pd.DataFrame, pair: str) -> None:
        """Log warnings about NaN values found in the DataFrame."""
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            nan_summary = nan_counts[nan_counts > 0]  # Filter only cols with NaNs
            log.warning(f"NaN values found in data for {pair}:\n{nan_summary.to_string()}")
            log.warning(
                f"Proceeding with NaN values for {pair}. "
                "Downstream components (features, strategies) must handle them."
            )
            # Consider adding imputation logic here if desired (e.g., ffill)
            # df.ffill(inplace=True) # Example: Forward fill

    def _calculate_atr(
        self, df: pd.DataFrame, pair: str, atr_period: int
    ) -> Optional[pd.DataFrame]:
        """Calculate ATR using TA-Lib if available, add 'atr' column."""
        # Check if ATR already exists and is suitable (e.g., not all NaN)
        if "atr" in df.columns and df["atr"].notna().any():
            # Ensure it's Decimal type
            if not all(isinstance(x, Decimal) for x in df["atr"].dropna()):
                log.warning(f"Existing 'atr' column for {pair} is not Decimal. " "Converting.")
                try:
                    df["atr"] = df["atr"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
                except Exception as e:
                    log.error(f"Failed to convert existing ATR column to Decimal for {pair}: {e}")
                    return None  # Indicate failure
            log.info(f"Using existing 'atr' column for {pair}.")
            return df

        log.info(f"Calculating ATR({atr_period}) for {pair}...")
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
                    f"Insufficient non-NaN data to calculate ATR({atr_period}) for {pair}."
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
                log.info(f"ATR calculation complete for {pair}.")
            else:
                # This case should ideally be hit only if TA-Lib failed import AND
                # the placeholder class somehow lost its atr method.
                log.error(
                    f"ATR function not available (TA-Lib missing?). "
                    f"Cannot calculate ATR for {pair}."
                )
                df["atr"] = pd.Series([None] * len(df), index=df.index)
                # Ensure column exists

        except Exception as e:
            log.error(f"Failed during ATR calculation for {pair}: {e}", exc_info=True)
            log.warning(
                f"Proceeding without ATR for {pair}. "
                "Volatility-based slippage/risk rules might not work."
            )
            # Ensure 'atr' column exists even if calculation failed
            if "atr" not in df.columns:
                df["atr"] = pd.Series([None] * len(df), index=df.index)
            # Depending on requirements, might return None to signal failure
            # return None
        return df

    async def run_backtest(self) -> Optional[Dict[str, Any]]:
        """Orchestrate and run the backtest simulation."""
        log.info("Starting backtest run...")
        start_run_time = datetime.now(tz=datetime.now().astimezone().tzinfo)

        try:
            # Step 1: Set up the backtest environment (config, output dir)
            setup_result = self._setup_backtest_environment(start_run_time)
            if not setup_result:
                log.error("Backtest environment setup failed.")
                return None

            config, run_output_dir = setup_result

            # Step 2: Load and prepare historical data
            historical_data = self._load_and_prepare_data()
            if historical_data is None:
                log.error("Backtest cannot proceed: Failed to load historical data.")
                return None

            # Step 3: Initialize simulation services
            services = self._initialize_backtest_services(historical_data, config)
            if not services:
                log.error("Failed to initialize backtesting services.")
                return None

            # Step 4: Execute simulation
            simulation_result = await self._execute_simulation(services, historical_data, config)
            if not simulation_result:
                log.error("Simulation execution failed.")
                # Create empty data structures for results processing
                equity_curve: Dict[datetime, Decimal] = {}
                trade_log: List[Dict[str, Any]] = []
            else:
                equity_curve, trade_log = simulation_result

            # Step 5: Process and save results
            results = self._process_and_save_results(
                config, equity_curve, trade_log, run_output_dir, start_run_time
            )

            log.info("Backtest run finished successfully.")
            return results

        except Exception as e:
            log.exception("Unhandled exception during backtest run.", exc_info=e)
            return None

    def _setup_backtest_environment(
        self, start_run_time: datetime
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """Set up the backtest environment including config and output directory.

        Args:
            start_run_time: The start time of the backtest run.

        Returns:
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

            return config, run_output_dir

        except Exception as e:
            log.exception(f"Error during backtest environment setup: {e}", exc_info=True)
            return None

    def _load_and_prepare_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Load and prepare historical data for the backtest.

        Returns:
            A dictionary mapping trading pairs to their historical data DataFrames,
            or None if loading fails.
        """
        # This is a simple wrapper around _load_historical_data for now
        # Can be expanded with additional data preparation steps in the future
        return self._load_historical_data()

    def _initialize_backtest_services(
        self, historical_data: Dict[str, pd.DataFrame], config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Initialize all required backtesting services.

        This method already exists, so we're keeping it as is.
        """
        # Implementation already exists
        return self._initialize_services(historical_data, config)

    async def _execute_simulation(
        self,
        services: Dict[str, Any],
        historical_data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],
    ) -> Optional[Tuple[Dict[datetime, Decimal], List[Dict[str, Any]]]]:
        """Execute the backtest simulation.

        Args:
            services: Dictionary of initialized services.
            historical_data: Dictionary of historical price data by pair.
            config: Backtest configuration parameters.

        Returns:
            A tuple of (equity_curve, trade_log) if successful, None otherwise.
        """
        # Initialize results data structures
        trade_log: List[Dict[str, Any]] = []
        equity_curve: Dict[datetime, Decimal] = {}
        open_positions_sim: Dict[str, Dict] = {}

        try:
            # Start all services and subscribe handlers
            if not await self._start_services(services, trade_log, open_positions_sim):
                log.error("Failed to start one or more services.")
                await self._stop_services(services)
                return None

            # Prepare timestamps for simulation steps
            timestamps = self._prepare_simulation_timestamps(historical_data)
            if not timestamps:
                log.error("Failed to prepare simulation timestamps.")
                await self._stop_services(services)
                return None

            # Record initial equity
            self._record_initial_equity(timestamps[0], config["initial_capital"], equity_curve)

            # Run the main simulation loop
            success = await self._run_simulation_loop(
                services, timestamps, historical_data, equity_curve
            )
            if not success:
                log.warning("Simulation loop encountered an error. Results may be partial.")

            return equity_curve, trade_log

        except Exception as e:
            log.exception("Error during simulation execution:", exc_info=e)
            return None
        finally:
            # Ensure all services are stopped regardless of success/failure
            log.info("Stopping services...")
            await self._stop_services(services)
            log.info("Services stopped.")

    def _record_initial_equity(
        self,
        first_timestamp: datetime,
        initial_capital: Decimal,
        equity_curve: Dict[datetime, Decimal],
    ) -> None:
        """Record the initial equity value before the simulation starts."""
        # Ensure initial capital is Decimal
        initial_capital_decimal = Decimal(str(initial_capital))

        # Create initial timestamp 1 second before the first timestamp
        initial_ts = first_timestamp - pd.Timedelta(seconds=1)

        # Record initial equity
        equity_curve[initial_ts] = initial_capital_decimal
        log.debug(f"Recorded initial equity of {initial_capital_decimal} at {initial_ts}")

    def _process_and_save_results(
        self,
        config: Dict[str, Any],
        equity_curve: Dict[datetime, Decimal],
        trade_log: List[Dict],
        run_output_dir: str,
        start_run_time: datetime,
    ) -> Dict[str, Any]:
        """Process the backtest results, calculate metrics, and save to files.

        Args:
            config: Backtest configuration parameters.
            equity_curve: Dictionary mapping timestamps to equity values.
            trade_log: List of trade dictionaries with entry/exit information.
            run_output_dir: Output directory path.
            start_run_time: Start time of the backtest run.

        Returns:
            Dictionary with summary results and metrics.
        """
        log.info("Processing backtest results...")

        # Convert equity curve dictionary to a Series for calculations
        if equity_curve:
            # Sort by timestamp to ensure chronological order
            sorted_timestamps = sorted(equity_curve.keys())
            equity_series = pd.Series(
                [equity_curve[ts] for ts in sorted_timestamps], index=sorted_timestamps
            )

            # Calculate performance metrics
            initial_capital = Decimal(str(config["initial_capital"]))
            metrics = calculate_performance_metrics(equity_series, trade_log, initial_capital)

            # Save equity curve to CSV
            try:
                equity_df = pd.DataFrame({"equity": equity_series})
                equity_csv_path = os.path.join(run_output_dir, "equity_curve.csv")
                equity_df.to_csv(equity_csv_path)
                log.info(f"Saved equity curve to {equity_csv_path}")
            except Exception as e:
                log.error(f"Failed to save equity curve: {e}")

            # Save trade log to CSV
            try:
                if trade_log:
                    trades_df = pd.DataFrame(trade_log)
                    trades_csv_path = os.path.join(run_output_dir, "trades.csv")
                    trades_df.to_csv(trades_csv_path, index=False)
                    log.info(f"Saved trade log to {trades_csv_path}")
            except Exception as e:
                log.error(f"Failed to save trade log: {e}")

            # Save metrics summary
            try:
                summary_path = os.path.join(run_output_dir, "summary.json")
                with open(summary_path, "w") as f:
                    import json

                    # Convert Decimal objects to float for JSON serialization
                    metrics_str = json.dumps(
                        metrics,
                        indent=2,
                        default=lambda x: float(x) if isinstance(x, Decimal) else x,
                    )
                    f.write(metrics_str)
                log.info(f"Saved performance metrics to {summary_path}")
            except Exception as e:
                log.error(f"Failed to save metrics summary: {e}")
        else:
            log.warning("Empty equity curve. Cannot calculate performance metrics.")
            metrics = {"error": "No equity data collected during backtest."}

        # Prepare results dictionary
        results = {
            "run_id": os.path.basename(run_output_dir),
            "start_time": start_run_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_run_time).total_seconds(),
            "metrics": metrics,
            "output_dir": run_output_dir,
            "trade_count": len(trade_log),
            "equity_points": len(equity_curve),
        }

        return results

    def _prepare_simulation_timestamps(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[List[datetime]]:
        """Prepare a unified, sorted list of UTC timestamps for simulation."""
        try:
            # Collect all unique timestamps from all pairs' DataFrames
            all_timestamps: Set[datetime] = set()
            for pair, df in historical_data.items():
                if not isinstance(df.index, pd.DatetimeIndex):
                    log.error(f"Data for pair {pair} does not have DatetimeIndex.")
                    return None
                # Ensure timestamps are UTC before adding
                utc_timestamps = (
                    df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
                )
                all_timestamps.update(utc_timestamps)

            if not all_timestamps:
                log.error("No timestamps found in historical data after setup.")
                return None

            # Sort timestamps chronologically
            sorted_timestamps = sorted(list(all_timestamps))
            total_steps = len(sorted_timestamps)
            log.info(
                f"Prepared {total_steps} unique simulation timestamps "
                f"from {sorted_timestamps[0]} to {sorted_timestamps[-1]}."
            )
            return sorted_timestamps

        except Exception as e:
            log.error(f"Error preparing simulation timestamps: {e}", exc_info=True)
            return None

    async def _run_simulation_loop(
        self,
        services: Dict[str, Any],
        timestamps: List[datetime],
        historical_data: Dict[str, pd.DataFrame],
        equity_curve: Dict[datetime, Decimal],
    ) -> bool:
        """Run the main simulation loop over all timestamps."""
        total_steps = len(timestamps)
        if total_steps == 0:
            log.warning("No timestamps to simulate.")
            return True  # No steps to run, technically successful

        log.info(f"Starting simulation loop for {total_steps} steps...")
        # Log progress roughly every 5% or at least every 100 steps
        log_progress_step = max(1, min(total_steps // 20, 100))

        try:
            for i, timestamp in enumerate(timestamps):
                # Log progress periodically
                if i % log_progress_step == 0 or i == total_steps - 1:
                    progress = f"{i+1}/{total_steps}({(i+1)/total_steps*100:.1f}%)"
                    log.info(f"Sim Step: {progress} | Timestamp: {timestamp}")

                # --- Run single simulation step ---
                step_success = await self._run_simulation_step(
                    services, timestamp, historical_data, equity_curve
                )
                if not step_success:
                    log.error(
                        f"Simulation step failed at index {i}, "
                        f"timestamp {timestamp}. Stopping loop."
                    )
                    return False  # Stop loop on first failure

            log.info(f"Simulation loop finished successfully after {total_steps} steps.")
            return True

        except Exception as e:
            # Catch unexpected errors during the loop
            log.exception(f"Error during simulation loop at step {i}: {e}", exc_info=True)
            return False  # Indicate failure

    async def _run_simulation_step(
        self,
        services: Dict[str, Any],
        timestamp: datetime,
        historical_data: Dict[str, pd.DataFrame],
        equity_curve: Dict[datetime, Decimal],
    ) -> bool:
        """Run a single step of the simulation for a given timestamp."""
        try:
            # 1. Update the simulated market price service with the current time
            #    This makes the latest prices available via get_latest_price
            market_price_service = services.get("market_price_service")
            if market_price_service:
                # Ensure service has the update_time method
                if hasattr(market_price_service, "update_time"):
                    market_price_service.update_time(timestamp)
                else:
                    log.warning("Market price service missing 'update_time' method.")
            else:
                log.warning("Market price service not found in services.")
                # Decide if this is critical - maybe return False?

            # 2. Publish market data events for this timestamp
            #    This triggers downstream processing (features, strategies)
            pubsub_manager = services.get("pubsub_manager")
            if pubsub_manager:
                publish_success = await self._publish_market_data(
                    pubsub_manager, timestamp, historical_data
                )
                if not publish_success:
                    log.error(f"Failed to publish market data for {timestamp}")
                    return False  # Treat as critical error for the step
            else:
                log.error("PubSubManager not found in services. Cannot publish data.")
                return False  # Critical

            # 3. Allow event processing by other services (yield control)
            #    Gives time for subscribers (FeatureEngine, StrategyArbitrator, etc.)
            #    to react to the market data events published above.
            await asyncio.sleep(0)

            # 4. Update equity curve by getting current portfolio value
            #    This should happen *after* potential trades triggered by this
            #    timestamp's data have been processed (or attempted).
            portfolio_manager = services.get("portfolio_manager")
            if portfolio_manager and hasattr(portfolio_manager, "get_current_state"):
                try:
                    current_state = portfolio_manager.get_current_state()
                    # Ensure 'total_equity' exists and is convertible to Decimal
                    if "total_equity" in current_state:
                        current_equity = Decimal(str(current_state["total_equity"]))
                        equity_curve[timestamp] = current_equity
                        # Optional: More detailed logging
                        # log.debug(f"Timestamp {timestamp}, Equity: {current_equity:.2f}")
                    else:
                        log.warning(
                            f"Timestamp {timestamp}: 'total_equity' not found in portfolio state."
                        )
                        # Decide how to handle: use previous value? record NaN?
                        # equity_curve[timestamp] = equity_curve.get(list(equity_curve.keys())[-1])
                        # Example: use last known
                except Exception as equity_err:
                    log.error(f"Error getting/recording equity state at {timestamp}: {equity_err}")
                    # Decide if this is critical. Maybe continue but log error.
                    # return False
            else:
                log.warning("Portfolio manager or 'get_current_state' method not found.")
                # Cannot update equity curve for this step

            return True  # Step completed successfully

        except Exception as e:
            log.exception(f"Error in simulation step at {timestamp}: {e}", exc_info=True)
            return False  # Indicate step failure

    async def _publish_market_data(
        self,
        pubsub_manager: Any,  # Should ideally be PubSubManager type
        timestamp: datetime,
        historical_data: Dict[str, pd.DataFrame],
    ) -> bool:
        """Publish MarketDataOHLCVEvent for each pair with data at this timestamp."""
        publish_tasks = []
        try:
            for pair, df in historical_data.items():
                # Check if the current timestamp exists in this pair's DataFrame index
                if timestamp in df.index:
                    bar_data = df.loc[timestamp]

                    # Validate that essential bar data is not NaN
                    required_fields = ["open", "high", "low", "close", "volume"]
                    if bar_data[required_fields].isnull().any():
                        log.warning(
                            f"Skipping market data event creation for {pair} at {timestamp}: "
                            "Required data contains NaN values."
                        )
                        continue

                    # Create the event object
                    try:
                        event = MarketDataOHLCVEvent(
                            event_id=uuid.uuid4(),
                            timestamp=timestamp,  # Event creation time
                            source_module="BacktestingEngine",
                            trading_pair=pair,
                            exchange="SIMULATED",  # Indicate simulated data
                            interval="?",  # TODO: Determine interval from data or config
                            timestamp_bar_start=timestamp,  # Time the bar represents
                            # Convert Decimal/numeric types to string for event
                            open=str(bar_data["open"]),
                            high=str(bar_data["high"]),
                            low=str(bar_data["low"]),
                            close=str(bar_data["close"]),
                            volume=str(bar_data["volume"]),
                        )
                        # If ATR is available, add it as an additional info to log
                        if "atr" in bar_data and pd.notna(bar_data["atr"]):
                            log.debug(f"ATR for {pair} at {timestamp}: {bar_data['atr']}")
                        # Add the publish task to a list
                        publish_tasks.append(asyncio.create_task(pubsub_manager.publish(event)))
                    except Exception as event_err:
                        log.error(
                            f"Error creating MarketDataOHLCVEvent for {pair} "
                            f"at {timestamp}: {event_err}"
                        )
                        # Continue to next pair, but log the error

            # Wait for all publish tasks for this timestamp to complete
            if publish_tasks:
                await asyncio.gather(*publish_tasks)
                # Optional: Add logging for successful publishes per timestamp
                # log.debug(f"Published {len(publish_tasks)} market data events for {timestamp}")

            return True  # Indicate success even if some pairs had no data

        except Exception as e:
            log.exception(
                f"Error during market data publishing for {timestamp}: {e}", exc_info=True
            )
            # Cancel any tasks that might have been created before the error
            for task in publish_tasks:
                if not task.done():
                    task.cancel()
            return False  # Indicate failure

    def _setup_output_directory(self, base_path: str, start_time: datetime) -> Optional[str]:
        """Create and return the path to the output directory for this run."""
        try:
            # Generate a unique run ID using the start time
            run_id = f"backtest_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"
            run_output_dir = os.path.join(base_path, run_id)

            # Create the directory, including intermediate directories
            os.makedirs(run_output_dir, exist_ok=True)
            log.info(f"Results will be saved to: {run_output_dir}")
            return run_output_dir
        except OSError as e:
            log.error(f"Could not create output directory {run_output_dir}: {e}")
            return None
        except Exception as e:
            log.error(f"Unexpected error setting up output directory: {e}")

            return None

    def _initialize_services(
        self,
        historical_data: Dict[str, pd.DataFrame],
        config: Dict[str, Any],  # Pass combined config
    ) -> Optional[Dict[str, Any]]:
        """Initialize all required backtesting services."""
        log.info("Initializing simulation components...")
        try:
            # --- Import necessary classes ---
            # Use concrete implementation for PubSub in backtesting
            import logging

            # Potentially needed for ProcessPoolExecutor typing
            from concurrent.futures import ProcessPoolExecutor

            from .core.pubsub import PubSubManager as EventBusPubSubManager
            from .historical_data_service import HistoricalDataService
            from .logger_service import LoggerService
            from .market_price_service import MarketPriceService
            from .simulated_market_price_service import SimulatedMarketPriceService

            # Removed local Logger import, use global log or pass logger_service
            # --- Instantiate Core Services ---
            # Create PubSub specifically for this backtest instance
            pubsub_logger = logging.getLogger("gal_friday.backtesting_engine.pubsub")
            pubsub_manager = EventBusPubSubManager(
                logger=pubsub_logger, config_manager=self.config
            )

            # LoggerService (if used by other services)
            # Ensure config_manager is passed if LoggerService expects it
            # (original code did not for this instance)
            logger_service: LoggerService = LoggerService(
                config_manager=self.config, pubsub_manager=pubsub_manager
            )

            # Market Price Service (using simulated implementation)
            sim_mp_logger = logging.getLogger(
                "gal_friday.backtesting_engine.SimulatedMarketPriceService"
            )
            market_price_service = SimulatedMarketPriceService(
                historical_data=historical_data, config_manager=self.config, logger=sim_mp_logger
            )

            # Portfolio Manager
            price_service_cast = cast(MarketPriceService, market_price_service)
            portfolio_manager = PortfolioManager(
                config_manager=self.config,
                pubsub_manager=pubsub_manager,  # Use pubsub_manager directly
                market_price_service=price_service_cast,
                logger_service=logger_service,
            )

            # Risk Manager
            risk_section_config = self.config.get("risk_manager") or {}
            risk_manager = RiskManager(
                config=risk_section_config,
                pubsub_manager=pubsub_manager,
                portfolio_manager=portfolio_manager,
                logger_service=logger_service,
                market_price_service=market_price_service,
            )

            # Execution Handler (using simulated implementation)
            historical_service_cast = cast(HistoricalDataService, market_price_service)
            sim_execution_handler = SimulatedExecutionHandler(
                config_manager=self.config,
                pubsub_manager=pubsub_manager,
                data_service=historical_service_cast,
                logger_service=logger_service,
            )

            # --- Instantiate Strategy/Analysis Services ---
            # Prediction Service (optional, depends on strategy)
            # May require a process pool for heavy computation
            # Ensure max_workers is configured appropriately or defaults sensibly
            max_workers = self.config.get_int("prediction.max_workers", 1)
            process_pool_executor = ProcessPoolExecutor(max_workers=max_workers)

            # Assuming PredictionService config is under "prediction_service" key
            prediction_service_config = self.config.get("prediction_service") or {}
            prediction_service = PredictionService(
                config=prediction_service_config,
                pubsub_manager=pubsub_manager,
                process_pool_executor=process_pool_executor,
                logger_service=logger_service,
            )

            # Feature Engine
            feature_engine_config = self.config.get("feature_engine") or {}
            feature_engine = FeatureEngine(
                config=feature_engine_config,
                pubsub_manager=pubsub_manager,
                logger_service=logger_service,
                historical_data_service=cast(HistoricalDataService, market_price_service),
            )

            # Strategy Arbitrator
            strategy_arbitrator_config = self.config.get("strategy_arbitrator") or {}
            strategy_arbitrator = StrategyArbitrator(
                config=strategy_arbitrator_config,
                pubsub_manager=pubsub_manager,
                logger_service=logger_service,
                market_price_service=market_price_service,
            )

            # --- Assemble Services Dictionary ---
            services = {
                "pubsub_manager": pubsub_manager,
                "logger_service": logger_service,
                "market_price_service": market_price_service,
                "portfolio_manager": portfolio_manager,
                "risk_manager": risk_manager,
                "execution_handler": sim_execution_handler,
                "prediction_service": prediction_service,
                "feature_engine": feature_engine,
                "strategy_arbitrator": strategy_arbitrator,
                # Store executor if needed for shutdown later
                "process_pool_executor": process_pool_executor,
            }

            log.info("Simulation components initialized successfully.")
            return services

        except ImportError as e:
            log.error(f"Failed to import a required module: {e}. Check dependencies.")
            return None
        except Exception as e:
            log.exception("Failed to initialize one or more services.", exc_info=e)
            # Cleanup partially created resources if necessary (e.g., executor)
            if "process_pool_executor" in locals() and process_pool_executor:
                process_pool_executor.shutdown(wait=False)
            return None

    async def _start_services(
        self,
        services: Dict[str, Any],
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> bool:
        """Start all services and subscribe the results collector."""
        log.info("Starting simulation services...")
        try:
            pubsub_manager = services.get("pubsub_manager")
            if not pubsub_manager:
                log.error("PubSubManager not found in services. Cannot subscribe.")
                return False

            # --- Subscribe Results Collector ---
            # Define the handler function locally
            async def handle_sim_execution_report(event: "ExecutionReportEvent") -> bool:
                # These variables are used by the handler but never assigned
                # Remove nonlocal declarations since they're passed as parameters
                return await self._handle_execution_report(event, trade_log, open_positions_sim)

            # Store the handler reference on the instance for unsubscribing later
            self._backtest_exec_report_handler = handle_sim_execution_report

            # Subscribe the handler to execution reports
            # Ensure EventType.EXECUTION_REPORT is correctly defined/imported
            pubsub_manager.subscribe(
                EventType.EXECUTION_REPORT, self._backtest_exec_report_handler
            )
            log.info("Subscribed results collector to execution reports.")

            # --- Start Services in Order ---
            # Define the order based on dependencies (e.g., portfolio before risk)
            start_order = [
                "logger_service",  # Start first if others depend on it
                "portfolio_manager",
                "risk_manager",
                "execution_handler",  # Simulated handler might need prices/portfolio
                "feature_engine",
                "prediction_service",  # Often depends on features
                "strategy_arbitrator",  # Depends on signals/predictions
                # PubSubManager itself might have a start method (e.g., for background tasks)
                "pubsub_manager",
            ]

            start_tasks = []
            for service_name in start_order:
                service = services.get(service_name)
                if service and hasattr(service, "start"):
                    log.debug(f"Creating start task for {service_name}...")
                    start_tasks.append(
                        # Create task to run start() concurrently
                        asyncio.create_task(service.start(), name=f"start_{service_name}")
                    )
                elif service_name not in services:
                    log.warning(f"Service '{service_name}' not found in initialized services.")
                # No warning if service exists but has no start() method

            # Wait for all start tasks to complete
            if start_tasks:
                log.info(f"Waiting for {len(start_tasks)} services to start...")
                # Use gather to run concurrently and wait for completion
                # return_exceptions=True allows us to see errors from individual starts
                results = await asyncio.gather(*start_tasks, return_exceptions=True)

                # Check results for exceptions
                all_started = True
                for i, result in enumerate(results):
                    task_name = start_tasks[i].get_name()  # Get name set during creation
                    if isinstance(result, Exception):
                        log.error(f"Error starting service task '{task_name}': {result}")
                        all_started = False
                if not all_started:
                    log.error("One or more services failed to start.")
                    return False  # Indicate overall start failure

            log.info("All specified simulation services started successfully.")
            return True

        except Exception as e:
            # Catch errors during the subscription or task creation/gathering phase
            log.exception("Failed to start services.", exc_info=e)
            return False

    async def _stop_services(self, services: Dict[str, Any]) -> None:
        """Stop all running services, typically in reverse order of start."""
        log.info("Stopping simulation services...")

        # Unsubscribe the execution report handler
        await self._unsubscribe_handlers(services)
        # Stop all service tasks
        await self._stop_service_tasks(services)
        # Shutdown the process pool executor
        self._shutdown_process_pool(services)

        log.info("All specified services stopped.")

    async def _unsubscribe_handlers(self, services: Dict[str, Any]) -> None:
        """Unsubscribe event handlers to prevent memory leaks."""
        pubsub_manager = services.get("pubsub_manager")
        # Check if the handler was stored and pubsub exists
        if (
            pubsub_manager
            and hasattr(self, "_backtest_exec_report_handler")
            and self._backtest_exec_report_handler
        ):
            try:
                log.debug("Unsubscribing backtest execution report handler...")
                pubsub_manager.unsubscribe(
                    EventType.EXECUTION_REPORT, self._backtest_exec_report_handler
                )
                log.info("Unsubscribed backtest execution report handler.")
                # Clear the stored handler after unsubscribing
                self._backtest_exec_report_handler = None
            except Exception as e:
                log.error(f"Error unsubscribing backtest execution report handler: {e}")
        elif hasattr(self, "_backtest_exec_report_handler") and self._backtest_exec_report_handler:
            # Handler exists but pubsub doesn't (shouldn't normally happen if init was ok)
            log.warning("PubSubManager not found, cannot unsubscribe handler.")

    async def _stop_service_tasks(self, services: Dict[str, Any]) -> None:
        """Stop service tasks in the appropriate order."""
        # Reverse of typical start order
        stop_order = [
            "strategy_arbitrator",
            "prediction_service",
            "feature_engine",
            "execution_handler",
            "risk_manager",
            "portfolio_manager",
            "market_price_service",  # Stop simulated price updates
            "logger_service",  # Stop logging service if needed
            "pubsub_manager",  # Stop pubsub last
        ]

        stop_tasks = []
        for service_name in stop_order:
            service = services.get(service_name)
            if service and hasattr(service, "stop"):
                log.debug(f"Creating stop task for {service_name}...")
                stop_tasks.append(asyncio.create_task(service.stop(), name=f"stop_{service_name}"))

        # Wait for all stop tasks to complete
        if stop_tasks:
            log.info(f"Waiting for {len(stop_tasks)} services to stop...")
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            # Log any errors encountered during stopping
            for i, result in enumerate(results):
                task_name = stop_tasks[i].get_name()
                if isinstance(result, Exception):
                    log.error(f"Error stopping service task '{task_name}': {result}")

    def _shutdown_process_pool(self, services: Dict[str, Any]) -> None:
        """Shutdown the process pool executor if present."""
        executor = services.get("process_pool_executor")
        if executor and isinstance(executor, ProcessPoolExecutor):
            log.info("Shutting down process pool executor...")
            try:
                # Use wait=True to ensure processes are cleaned up before exiting
                executor.shutdown(wait=True)
                log.info("Process pool executor shut down.")
            except Exception as e:
                log.error(f"Error shutting down process pool executor: {e}")

    async def _handle_execution_report(
        self,
        event: "ExecutionReportEvent",
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> bool:
        """Handle FILLED execution reports to log trades for performance calc."""
        # Only process filled orders for trade logging
        if event.order_status != "FILLED":
            log.debug(
                f"Ignoring non-FILLED execution report: {event.exchange_order_id} "
                f"status {event.order_status}"
            )
            return False  # Indicate not processed for logging

        log.debug(f"Processing FILLED execution report: {event.exchange_order_id}")

        # Validate necessary fields exist in the event
        required_fields = [
            "signal_id",
            "trading_pair",
            "side",
            "quantity_filled",
            "average_fill_price",
            "timestamp_exchange",  # Assuming this is fill time
        ]
        missing_fields = [
            f for f in required_fields if not hasattr(event, f) or getattr(event, f) is None
        ]
        if missing_fields:
            log.error(
                f"Execution report {event.exchange_order_id} missing required fields "
                f"for logging: {missing_fields}"
            )
            return False

        try:
            # Process the fill based on event details
            self._process_execution_fill(event, trade_log, open_positions_sim)
            return True  # Indicate successful processing
        except Exception as e:
            log.exception(
                f"Error processing execution report {event.exchange_order_id}: {e}", exc_info=True
            )
            return False  # Indicate processing failure

    def _process_execution_fill(
        self,
        event: "ExecutionReportEvent",
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> None:
        """Process a FILLED execution report to update trade log or open positions."""
        # Extract and convert necessary data, ensuring Decimal type
        try:
            fill_price = Decimal(str(event.average_fill_price))
            fill_qty = Decimal(str(event.quantity_filled))
            # Commission might be optional or zero
            commission = (
                Decimal(str(event.commission)) if event.commission is not None else Decimal("0")
            )
            # Use signal_id as the key to match entry/exit for a strategy signal
            # Assuming one signal leads to one entry and one exit for simplicity here
            trade_key = str(event.signal_id)
            trading_pair = event.trading_pair
            fill_time = event.timestamp_exchange  # Should be datetime

        except (TypeError, ValueError, AttributeError) as conversion_err:
            log.error(
                f"Error converting execution report data for {event.exchange_order_id}: "
                f"{conversion_err}"
            )
            return  # Cannot process without valid data

        log.debug(
            f"Processing fill for trade key {trade_key}: "
            f"{event.side} {fill_qty} {trading_pair} @ {fill_price}"
        )

        # --- Simple Long-Only PnL Calculation Logic ---
        # Assumes:
        # - A BUY fill opens a position for a given trade_key.
        # - A SELL fill closes the corresponding open BUY position.
        # - Does not handle partial fills closing positions or multiple entries/exits per key.
        # - Does not handle short positions.

        if event.side.upper() == "BUY":
            # Opening a new position or adding to existing (simple model assumes new)
            if trade_key in open_positions_sim:
                log.warning(
                    f"Received BUY fill for already open trade key {trade_key}. "
                    "Overwriting previous entry data (simple model)."
                )
            open_positions_sim[trade_key] = {
                "pair": trading_pair,
                "entry_time": fill_time,
                "entry_price": fill_price,
                "quantity": fill_qty,  # Store the entry quantity
                "commission_entry": commission,
                "side": "BUY",  # Mark as open long position
            }
            log.debug(f"Opened/Updated long position for trade key {trade_key}")

        elif event.side.upper() == "SELL":
            # Closing an existing long position
            if trade_key in open_positions_sim and open_positions_sim[trade_key]["side"] == "BUY":
                entry_data = open_positions_sim.pop(trade_key)  # Remove from open positions

                # --- Calculate PnL ---
                # Ensure quantities match for simple model (or handle partial closes)
                if fill_qty != entry_data["quantity"]:
                    log.warning(
                        f"Sell quantity ({fill_qty}) differs from entry quantity "
                        f"({entry_data['quantity']}) for trade key {trade_key}. "
                        "Calculating PnL based on sell quantity (simple model)."
                    )
                    # Adjust PnL calc or use min(fill_qty, entry_data['quantity'])?
                    # Using fill_qty for this simple example.

                pnl = (
                    (fill_price - entry_data["entry_price"]) * fill_qty  # Profit/Loss
                    - entry_data["commission_entry"]  # Subtract entry commission
                    - commission  # Subtract exit commission
                )

                # --- Log the completed trade ---
                # Safely format timestamps
                entry_time_str = (
                    entry_data["entry_time"].isoformat() if entry_data.get("entry_time") else ""
                )
                exit_time_str = fill_time.isoformat() if fill_time else ""

                trade_log.append(
                    {
                        "signal_id": trade_key,
                        "pair": trading_pair,
                        "entry_time": entry_time_str,
                        "exit_time": exit_time_str,
                        "side": "LONG",  # Indicate it was a long trade
                        "quantity": str(fill_qty),  # Log the quantity traded (exit qty)
                        "entry_price": str(entry_data["entry_price"]),
                        "exit_price": str(fill_price),
                        "commission": str(entry_data["commission_entry"] + commission),
                        "pnl": str(pnl),
                    }
                )
                log.info(f"Closed LONG trade {trade_key} ({trading_pair}). PnL: {pnl:.4f}")
            else:
                log.warning(
                    f"Received SELL fill for trade key {trade_key} "
                    "but no matching open BUY position found in log. "
                    "Ignoring for PnL calculation."
                )
        else:
            log.warning(
                f"Received fill with unhandled side: {event.side} " f"for trade key {trade_key}"
            )

    def _process_backtest_results(
        self,
        config: Dict[str, Any],
        equity_curve: Dict[datetime, Decimal],
        trade_log: List[Dict],
        run_output_dir: str,
        start_run_time: datetime,
    ) -> Dict[str, Any]:
        """Process final equity curve, calculate metrics, and save results."""
        log.info("Processing final backtest results...")
        results: Dict[str, Any] = {}

        # Basic implementation to satisfy type checking
        results = {
            "run_id": run_output_dir.split(os.sep)[-1],
            "start_time": start_run_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_run_time).total_seconds(),
            "metrics": {},
            "output_dir": run_output_dir,
            "trade_count": len(trade_log),
            "equity_points": len(equity_curve),
        }

        return results
