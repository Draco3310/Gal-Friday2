# Backtesting Engine Module
from typing import Dict, Optional, List, TYPE_CHECKING, Any, Type, Callable, Coroutine, Set, Union, Sequence
from datetime import datetime
from decimal import Decimal
import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np  # Add numpy import for np references

# Import TA-Lib for technical indicators
try:
    import talib as ta
except ImportError:
    log = logging.getLogger(__name__)
    log.warning("TA-Lib not installed. ATR calculation will not work.")
    # Create a minimal placeholder for ta module
    class TaLib:
        @staticmethod
        def atr(*args, **kwargs):
            log.error("TA-Lib not installed. Cannot calculate ATR.")
            return pd.Series([None] * len(args[0]))
    ta = TaLib()

# Import necessary components for instantiation
from .core.events import MarketDataOHLCVEvent, EventType, ExecutionReportEvent
from .config_manager import ConfigManager
from .logger_service import LoggerService

# Set up logging
log = logging.getLogger(__name__)


def create_placeholder_class(name: str, **methods: Any) -> Type[Any]:
    """Create a placeholder class with specified methods."""
    return type(
        name,
        (),
        {
            "__init__": lambda *args, **kwargs: None,
            **{name: (lambda *args, **kwargs: asyncio.sleep(0)) for name in methods},
        },
    )


# Type hints and imports
if TYPE_CHECKING:
    from .core.pubsub import PubSubManager
    from .simulated_market_price_service import SimulatedMarketPriceService
    from .portfolio_manager import PortfolioManager
    from .risk_manager import RiskManager
    from .simulated_execution_handler import SimulatedExecutionHandler
    from .strategy_arbitrator import StrategyArbitrator
    from .prediction_service import PredictionService
    from .feature_engine import FeatureEngine
else:
    # Import implementations with fallbacks
    def import_with_fallback(module_path: str, class_name: str, methods: List[str]) -> Any:
        """Import a class with fallback to placeholder if not available."""
        try:
            # Try to import from the primary location
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
def calculate_performance_metrics(
    equity_curve: pd.Series, trade_log: List[Dict], initial_capital: Decimal
) -> Dict[str, Any]:
    """Calculates standard backtesting performance metrics."""
    if equity_curve.empty:
        return {"error": "Equity curve is empty, cannot calculate metrics."}

    results = {}
    returns = equity_curve.pct_change().dropna()

    # Basic Returns
    final_equity = Decimal(str(equity_curve.iloc[-1]))
    results["initial_capital"] = float(initial_capital)
    results["final_equity"] = float(final_equity)
    results["total_return_pct"] = (
        float((final_equity / initial_capital - 1) * 100) if initial_capital > 0 else 0.0
    )
    # TODO: Annualized Return (requires duration calculation)

    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    results["max_drawdown_pct"] = float(abs(drawdown.min() * 100)) if not drawdown.empty else 0.0

    # Sharpe Ratio (Simplified - Assumes daily returns, 0% risk-free)
    # More accurate version would need risk-free rate and adjust for period (daily, hourly etc.)
    if not returns.empty and returns.std() != 0:
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252)
        )  # Annualized for daily (approx)
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
            results["sortino_ratio_annualized_approx"] = (
                np.inf
            )  # Avoid division by zero if no downside deviation
    else:
        results["sortino_ratio_annualized_approx"] = np.inf  # No downside returns

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
        results["gross_profit"] = float(gross_profit)
        results["gross_loss"] = float(gross_loss)

        results["profit_factor"] = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf
        results["average_trade_pnl"] = float(sum(pnl_list) / num_trades)
        results["average_win"] = float(sum(winning_trades) / num_wins) if num_wins > 0 else 0.0
        results["average_loss"] = float(sum(losing_trades) / num_losses) if num_losses > 0 else 0.0
        results["avg_win_loss_ratio"] = (
            float(abs(results["average_win"] / results["average_loss"]))
            if results["average_loss"] != 0
            else np.inf
        )
    else:
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

    return results


class BacktestingEngine:
    """Orchestrates backtesting simulations using historical data."""

    def __init__(self, config_manager: "ConfigManager"):
        """
        Initializes the BacktestingEngine.

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

        # Other initializations (e.g., for simulation components) will happen in run_backtest
        log.info("BacktestingEngine initialized.")

    def _load_historical_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Loads, cleans, validates, and preprocesses historical OHLCV data.

        Returns:
            A dictionary mapping trading pairs to their historical data DataFrames,
            or None if loading fails.
        """
        try:
            config = self._get_backtest_config()
            if not self._validate_config(config):
                return None

            all_data = self._load_raw_data(config["data_path"])
            if all_data is None:
                return None

            all_data = self._clean_and_validate_data(
                all_data, config["start_date"], config["end_date"]
            )
            if all_data is None:
                return None

            processed_data = self._process_pairs_data(all_data, config)
            if not processed_data:
                log.error("Failed to load or process data for any configured trading pairs.")
                return None

            return processed_data

        except Exception as e:
            log.exception(
                f"An unexpected error occurred during historical data loading: {e}", exc_info=True
            )
            return None

    def _get_backtest_config(self) -> Dict[str, Any]:
        """Get and prepare backtest configuration parameters."""
        return {
            "data_path": self.config.get("backtest.data_path"),
            "start_date": self.config.get("backtest.start_date"),
            "end_date": self.config.get("backtest.end_date"),
            "pairs": self.config.get_list("trading.pairs"),
            "needs_atr": self.config.get("backtest.slippage_model", "fixed") == "volatility",
            "atr_period": self.config.get_int("backtest.atr_period", 14),
        }

    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate backtest configuration parameters."""
        data_path = config["data_path"]
        if not data_path or not os.path.exists(data_path):
            log.error("Historical data path not found or not configured: " f"{data_path}")
            return False

        if not config["start_date"] or not config["end_date"]:
            log.error("Backtest start_date or end_date not configured.")
            return False

        if not config["pairs"]:
            log.error("No trading pairs configured for backtest ('trading.pairs').")
            return False

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
            log.error(f"Error loading data: {e}")
            return None

    def _clean_and_validate_data(
        self, data: pd.DataFrame, start_date_str: str, end_date_str: str
    ) -> Optional[pd.DataFrame]:
        """Clean and validate the loaded data."""
        try:
            data = self._ensure_datetime_index(data)
            if data is None:
                return None

            if not self._validate_required_columns(data):
                return None

            start_date = pd.to_datetime(start_date_str).tz_convert("UTC")
            end_date = pd.to_datetime(end_date_str).tz_convert("UTC")
            log.info(f"Date range: {start_date} to {end_date}")

            data = data[(data.index >= start_date) & (data.index <= end_date)]
            log.info(f"{len(data)} rows remaining after date filtering.")

            if data.empty:
                log.error("No data available for the specified date range.")
                return None

            return data

        except Exception as e:
            log.error(f"Error cleaning and validating data: {e}")
            return None

    def _ensure_datetime_index(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Ensure the DataFrame has a proper datetime index."""
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is None:
                log.warning("Data index is timezone naive. Assuming UTC.")
                return data.tz_localize("UTC")
            return data.tz_convert("UTC")

        log.warning("Loaded data does not have a DatetimeIndex. " "Attempting to set index...")
        ts_cols = ["timestamp", "time", "date"]
        found_col = None
        for col in ts_cols:
            if col in data.columns:
                found_col = col
                break

        if not found_col:
            log.error("Cannot find a suitable timestamp column to set as index.")
            return None

        data[found_col] = pd.to_datetime(data[found_col])
        if data[found_col].dt.tz is None:
            log.warning(f"Timestamp column '{found_col}' is timezone naive. " "Assuming UTC.")
            data[found_col] = data[found_col].dt.tz_localize("UTC")
        else:
            data[found_col] = data[found_col].dt.tz_convert("UTC")

        return data.set_index(found_col).sort_index()

    def _validate_required_columns(self, data: pd.DataFrame) -> bool:
        """Validate that all required columns are present."""
        required_cols = ["open", "high", "low", "close", "volume"]
        if "pair" not in data.columns:
            log.error("Loaded data missing required 'pair' column " "for multi-pair file.")
            return False

        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            log.error("Loaded data missing required OHLCV columns: " f"{missing}")
            return False

        return True

    def _process_pairs_data(
        self, data: pd.DataFrame, config: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        """Process data for each trading pair."""
        processed_data: Dict[str, pd.DataFrame] = {}
        required_cols = ["open", "high", "low", "close", "volume"]

        for pair in config["pairs"]:
            pair_df = data[data["pair"] == pair].copy()
            if pair_df.empty:
                log.warning(
                    "No data found for configured pair: " f"{pair} in the loaded file/date range."
                )
                continue

            if not self._convert_ohlcv_to_decimal(pair_df, pair, required_cols):
                continue

            self._handle_nan_values(pair_df, pair)

            if config["needs_atr"]:
                pair_df = self._calculate_atr(pair_df, pair, config["atr_period"])
                if pair_df is None:
                    continue

            processed_data[pair] = pair_df
            log.info(f"Successfully processed data for {pair} " f"({len(pair_df)} rows).")

        return processed_data

    def _convert_ohlcv_to_decimal(self, df: pd.DataFrame, pair: str, columns: List[str]) -> bool:
        """Convert OHLCV columns to Decimal type."""
        try:
            for col in columns:
                # Convert values to string first to ensure Decimal can parse them
                df[col] = df[col].apply(lambda x: Decimal(str(float(x))) if pd.notna(x) else None)
            return True
        except Exception as e:
            log.error(
                f"Error converting column '{col}' to Decimal for "
                f"pair {pair}: {e}. Check data source format."
            )
            return False

    def _handle_nan_values(self, df: pd.DataFrame, pair: str) -> None:
        """Handle NaN values in the DataFrame."""
        if df.isnull().values.any():
            nan_counts = df.isnull().sum()
            log.warning(f"NaN values found in data for {pair}:\n" f"{nan_counts[nan_counts > 0]}")
            log.warning(
                f"Proceeding with NaN values for {pair}. "
                "Feature/Strategy logic must handle them."
            )

    def _calculate_atr(
        self, df: pd.DataFrame, pair: str, atr_period: int
    ) -> Optional[pd.DataFrame]:
        """Calculate ATR if needed and not present."""
        if "atr" not in df.columns or df["atr"].isnull().all():
            log.info(f"Calculating ATR({atr_period}) for {pair}...")
            try:
                temp_high = df["high"].astype(float)
                temp_low = df["low"].astype(float)
                temp_close = df["close"].astype(float)
                df["atr"] = ta.atr(
                    high=temp_high, low=temp_low, close=temp_close, length=atr_period
                )
                df["atr"] = df["atr"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
                log.info(f"ATR calculation complete for {pair}.")
            except Exception as e:
                log.error(f"Failed to calculate ATR for {pair}: {e}", exc_info=True)
                log.warning(
                    f"Proceeding without ATR for {pair}. " "Volatility slippage will be zero."
                )
        elif not hasattr(pd.api.types, "is_decimal_dtype") or not pd.api.types.is_decimal_dtype(df["atr"]):
            # Check if is_decimal_dtype exists first, otherwise handle differently
            df["atr"] = df["atr"].apply(lambda x: Decimal(str(x)) if pd.notna(x) else None)
        return df

    async def run_backtest(self) -> Optional[Dict[str, Any]]:
        log.info("Starting backtest run...")
        start_run_time = datetime.now()

        # Setup configuration and output directory
        config = self._setup_backtest_config(start_run_time)
        if not config:
            return None

        run_output_dir = self._setup_output_directory(config["output_path"], start_run_time)
        if not run_output_dir:
            return None

        # Load historical data
        historical_data = self._load_historical_data()
        if historical_data is None:
            log.error("Backtest cannot proceed: Failed to load historical data.")
            return None

        # Initialize and start services
        services = self._initialize_services(historical_data)
        if not services:
            return None

        trade_log: List[Dict[str, Any]] = []
        equity_curve: Dict[datetime, Decimal] = {}
        open_positions_sim: Dict[str, Dict] = {}

        try:
            # Start all services
            if not await self._start_services(services, trade_log, open_positions_sim):
                return None

            # Run simulation loop
            timestamps = self._prepare_simulation_timestamps(historical_data)
            if not timestamps:
                return None

            # Record initial equity
            equity_curve[timestamps[0] - pd.Timedelta(seconds=1)] = config["initial_capital"]

            # Run simulation steps
            if not await self._run_simulation_loop(
                services, timestamps, historical_data, equity_curve
            ):
                return None

        except Exception as e:
            log.exception("Error during backtest run.", exc_info=e)
            return None
        finally:
            # Ensure services are stopped
            await self._stop_services(services)

        # Process and save results
        results = await self._process_backtest_results(
            config, equity_curve, trade_log, run_output_dir, start_run_time
        )
        return results

    def _prepare_simulation_timestamps(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[List[datetime]]:
        """Prepare a unified, sorted list of timestamps for simulation.

        Args:
            historical_data: Dictionary of historical data frames per trading pair

        Returns:
            Sorted list of unique timestamps or None if preparation fails
        """
        try:
            # Get a unified, sorted list of all timestamps across all pairs
            all_timestamps: Set[datetime] = set()  # Add type annotation for all_timestamps
            for df in historical_data.values():
                all_timestamps.update(df.index)

            if not all_timestamps:
                log.error("No timestamps found in historical data after setup.")
                return None

            sorted_timestamps = sorted(list(all_timestamps))
            total_steps = len(sorted_timestamps)
            log.info(f"Prepared {total_steps} simulation timestamps.")
            return sorted_timestamps

        except Exception as e:
            log.error(f"Error preparing simulation timestamps: {e}")
            return None

    async def _run_simulation_loop(
        self,
        services: Dict[str, Any],
        timestamps: List[datetime],
        historical_data: Dict[str, pd.DataFrame],
        equity_curve: Dict[datetime, Decimal],
    ) -> bool:
        """Run the main simulation loop over all timestamps."""
        try:
            total_steps = len(timestamps)
            log_progress_step = max(1, total_steps // 20)  # Log every 5%

            for i, timestamp in enumerate(timestamps):
                if i % log_progress_step == 0:
                    progress = f"{i}/{total_steps} ({i/total_steps*100:.1f}%)"
                    log.info(f"Simulation Progress: {progress} " f"Timestamp: {timestamp}")

                # Run single simulation step
                if not await self._run_simulation_step(
                    services, timestamp, historical_data, equity_curve
                ):
                    return False

            log.info(f"Simulation loop finished after {total_steps} steps.")
            return True

        except Exception as e:
            log.error(f"Error in simulation loop: {e}")
            return False

    async def _run_simulation_step(
        self,
        services: Dict[str, Any],
        timestamp: datetime,
        historical_data: Dict[str, pd.DataFrame],
        equity_curve: Dict[datetime, Decimal],
    ) -> bool:
        """Run a single step of the simulation."""
        try:
            # 1. Update the simulated market price service
            services["market_price_service"].update_time(timestamp)

            # 2. Publish market data events
            if not await self._publish_market_data(
                services["pubsub_manager"], timestamp, historical_data
            ):
                return False

            # 3. Allow event processing
            await asyncio.sleep(0)

            # 4. Update equity curve
            try:
                current_state = services["portfolio_manager"].get_current_state()
                current_equity = Decimal(current_state["total_equity"])
                equity_curve[timestamp] = current_equity
                log.debug(f"Timestamp {timestamp}, " f"Equity: {current_equity:.2f}")
            except Exception as e:
                log.error(f"Error getting equity state at timestamp {timestamp}: {e}")
                # Continue despite equity tracking error
                pass

            return True

        except Exception as e:
            log.error(f"Error in simulation step at {timestamp}: {e}")
            return False

    async def _publish_market_data(
        self,
        pubsub_manager: Any,
        timestamp: datetime,
        historical_data: Dict[str, pd.DataFrame],
    ) -> bool:
        """Publish market data events for the current timestamp."""
        try:
            publish_tasks = []
            for pair, df in historical_data.items():
                if timestamp in df.index:
                    bar_data = df.loc[timestamp]
                    # Create the specific MarketDataOHLCVEvent object with required parameters
                    event = MarketDataOHLCVEvent(
                        event_id=f"backtest-{timestamp.isoformat()}-{pair}",  # Add required event_id
                        timestamp=timestamp,  # Add required timestamp
                        source_module="BacktestingEngine",
                        trading_pair=pair,
                        exchange="SIMULATED",
                        interval="1m",  # Assuming 1m interval for now
                        timestamp_bar_start=timestamp,
                        open=str(bar_data["open"]),
                        high=str(bar_data["high"]),
                        low=str(bar_data["low"]),
                        close=str(bar_data["close"]),
                        volume=str(bar_data["volume"]),
                    )
                    publish_tasks.append(asyncio.create_task(pubsub_manager.publish(event)))

            if publish_tasks:
                await asyncio.gather(*publish_tasks)
            return True

        except Exception as e:
            log.error(f"Error publishing market data: {e}")
            return False

    def _setup_backtest_config(self, start_time: datetime) -> Optional[Dict[str, Any]]:
        """Set up and validate initial backtest configuration."""
        try:
            initial_capital = self.config.get_decimal(
                "backtest.initial_capital", Decimal("100000")
            )
            output_path = self.config.get("backtest.output_path", "backtests/results")
            run_id = f"backtest_{start_time.strftime('%Y%m%d_%H%M%S')}"

            return {
                "initial_capital": initial_capital,
                "output_path": output_path,
                "run_id": run_id,
            }
        except Exception as e:
            log.error(f"Failed to setup backtest configuration: {e}")
            return None

    def _setup_output_directory(self, base_path: str, start_time: datetime) -> Optional[str]:
        """Create and validate the output directory for backtest results.

        Args:
            base_path: Base path for output directory
            start_time: The backtest start time for run ID generation

        Returns:
            Path to the output directory or None if setup fails
        """
        try:
            run_id = f"backtest_{start_time.strftime('%Y%m%d_%H%M%S')}"
            run_output_dir = os.path.join(base_path, run_id)

            os.makedirs(run_output_dir, exist_ok=True)
            log.info(f"Results will be saved to: {run_output_dir}")
            return run_output_dir
        except OSError as e:
            log.error(f"Could not create output directory {run_output_dir}: {e}")
            return None

    def _initialize_services(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> Optional[Dict[str, Any]]:
        """Initialize all required backtesting services.

        Args:
            historical_data: Dictionary of historical data frames per trading pair

        Returns:
            Dictionary containing service instances or None if initialization fails
        """
        try:
            log.info("Setting up simulation components...")
            
            # Create a logger service instance
            logger_service = LoggerService()

            # Core services for simulation
            pubsub_manager = PubSubManager(logger=logger_service)
            market_price_service = SimulatedMarketPriceService(historical_data)
            portfolio_manager = PortfolioManager(
                self.config, 
                pubsub_manager, 
                market_price_service,
                logger_service=logger_service
            )
            risk_manager = RiskManager(
                self.config.get_dict(), 
                pubsub_manager, 
                portfolio_manager,
                logger_service=logger_service
            )
            sim_execution_handler = SimulatedExecutionHandler(
                self.config, 
                pubsub_manager, 
                market_price_service,
                logger_service=logger_service
            )
            prediction_service = PredictionService(
                self.config.get_dict(), 
                pubsub_manager
            )
            strategy_arbitrator = StrategyArbitrator(
                self.config.get_dict(), 
                pubsub_manager,
                logger_service=logger_service
            )
            feature_engine = FeatureEngine(
                self.config.get_dict(), 
                pubsub_manager,
                logger_service=logger_service
            )

            services = {
                "pubsub_manager": pubsub_manager,
                "market_price_service": market_price_service,
                "portfolio_manager": portfolio_manager,
                "risk_manager": risk_manager,
                "execution_handler": sim_execution_handler,
                "strategy_arbitrator": strategy_arbitrator,
                "prediction_service": prediction_service,
                "feature_engine": feature_engine,
            }

            log.info("Simulation components instantiated.")

            return services

        except Exception as e:
            log.error(f"Failed to initialize services: {e}")
            return None

    async def _start_services(
        self,
        services: Dict[str, Any],
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> bool:
        """Start all services and set up event handlers.

        Args:
            services: Dictionary of service instances
            trade_log: List to store trade information
            open_positions_sim: Dictionary to track open positions

        Returns:
            True if all services started successfully, False otherwise
        """
        try:
            # Subscribe the results collector
            if services["pubsub_manager"]:

                async def handle_sim_execution_report(event: "ExecutionReportEvent") -> bool:
                    return await self._handle_execution_report(event, trade_log, open_positions_sim)

                # Store the handler for unsubscribing later
                self._backtest_exec_report_handler = handle_sim_execution_report
                
                # Subscribe using EventType enum
                services["pubsub_manager"].subscribe(
                    EventType.EXECUTION_REPORT, self._backtest_exec_report_handler
                )

            # Start all services
            start_order = [
                "portfolio_manager",
                "risk_manager",
                "execution_handler",
                "strategy_arbitrator",
                "prediction_service",
                "feature_engine",
            ]

            log.info(f"Starting {len(start_order)} simulation services...")
            start_tasks = [
                asyncio.create_task(services[service].start())
                for service in start_order
                if service in services and hasattr(services[service], "start")
            ]
            await asyncio.gather(*start_tasks)
            log.info("Simulation services started and subscribed.")
            return True

        except Exception as e:
            log.error(f"Failed to start services: {e}")
            return False

    async def _stop_services(self, services: Dict[str, Any]) -> None:
        """Stop all running services in reverse order of start."""
        stop_order = [
            "feature_engine",
            "prediction_service",
            "strategy_arbitrator",
            "execution_handler",
            "risk_manager",
            "portfolio_manager",
            "pubsub_manager", # Stop pubsub last after its subscribers
        ]
        
        # Unsubscribe handlers before stopping pubsub
        if "pubsub_manager" in services:
            # Use the stored handler reference to unsubscribe
            if hasattr(self, '_backtest_exec_report_handler') and self._backtest_exec_report_handler:
                try:
                    services["pubsub_manager"].unsubscribe(EventType.EXECUTION_REPORT, self._backtest_exec_report_handler)
                    log.info("Unsubscribed backtest execution report handler.")
                    self._backtest_exec_report_handler = None # Clear stored handler
                except Exception as e:
                    log.error(f"Error unsubscribing backtest execution report handler: {e}")
            else:
                 log.warning("Backtest execution report handler not found for unsubscribing.")

        log.info(f"Stopping {len(stop_order)} simulation services...")

    async def _handle_execution_report(
        self,
        event: "ExecutionReportEvent",
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> bool:  # Changed return type from None to bool
        """Handle execution reports for trade logging.

        Args:
            event: The execution report event
            trade_log: List to store trade information
            open_positions_sim: Dictionary to track open positions
            
        Returns:
            True if processed successfully, False otherwise
        """
        if event.order_status != "FILLED":
            return False  # Return False when status is not FILLED

        try:
            self._process_execution_fill(event, trade_log, open_positions_sim)
            return True  # Return True when processed successfully
        except Exception as e:
            log.error(f"Error processing execution report: {e}", exc_info=True)
            return False  # Return False on error

    def _process_execution_fill(
        self,
        event: "ExecutionReportEvent",
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> None:
        """Process a filled execution report.

        Args:
            event: The execution report event
            trade_log: List to store trade information
            open_positions_sim: Dictionary to track open positions
        """
        log.debug(
            "ResultsCollector received FILL: "
            f"{event.signal_id} {event.side} {event.quantity_filled} "
            f"{event.trading_pair}@{event.average_fill_price}"
        )

        fill_price = Decimal(event.average_fill_price)
        fill_qty = Decimal(event.quantity_filled)
        commission = Decimal(event.commission if event.commission else "0")
        trade_key = f"{event.trading_pair}_{event.signal_id}"

        if event.side.upper() == "BUY":
            self._handle_buy_fill(
                event, fill_price, fill_qty, commission, trade_key, open_positions_sim
            )
        elif event.side.upper() == "SELL":
            self._handle_sell_fill(
                event, fill_price, fill_qty, commission, trade_key, trade_log, open_positions_sim
            )

    def _handle_buy_fill(
        self,
        event: "ExecutionReportEvent",
        fill_price: Decimal,
        fill_qty: Decimal,
        commission: Decimal,
        trade_key: str,
        open_positions_sim: Dict[str, Dict],
    ) -> None:
        """Handle a buy fill event.

        Args:
            event: The execution report event
            fill_price: The fill price
            fill_qty: The fill quantity
            commission: The commission paid
            trade_key: The trade key
            open_positions_sim: Dictionary to track open positions
        """
        open_positions_sim[trade_key] = {
            "entry_time": event.timestamp_exchange,
            "entry_price": fill_price,
            "quantity": fill_qty,
            "commission_entry": commission,
            "side": "BUY",
        }

    def _handle_sell_fill(
        self,
        event: "ExecutionReportEvent",
        fill_price: Decimal,
        fill_qty: Decimal,
        commission: Decimal,
        trade_key: str,
        trade_log: List[Dict],
        open_positions_sim: Dict[str, Dict],
    ) -> None:
        """Handle a sell fill event.

        Args:
            event: The execution report event
            fill_price: The fill price
            fill_qty: The fill quantity
            commission: The commission paid
            trade_key: The trade key
            trade_log: List to store trade information
            open_positions_sim: Dictionary to track open positions
        """
        if trade_key in open_positions_sim and open_positions_sim[trade_key]["side"] == "BUY":
            entry_data = open_positions_sim.pop(trade_key)
            pnl = (
                (fill_price - entry_data["entry_price"]) * fill_qty
                - entry_data["commission_entry"]
                - commission
            )
            # Safely handle potentially None timestamp values
            entry_time_str = entry_data["entry_time"].isoformat() if entry_data["entry_time"] else ""
            exit_time_str = event.timestamp_exchange.isoformat() if event.timestamp_exchange else ""
            
            trade_log.append(
                {
                    "signal_id": event.signal_id,
                    "pair": event.trading_pair,
                    "entry_time": entry_time_str,
                    "exit_time": exit_time_str,
                    "side": "LONG",
                    "quantity": str(fill_qty),
                    "entry_price": str(entry_data["entry_price"]),
                    "exit_price": str(fill_price),
                    "commission": str(entry_data["commission_entry"] + commission),
                    "pnl": str(pnl),
                }
            )
            log.debug(f"Logged LONG trade PnL: {pnl:.4f}")
        else:
            log.warning(
                f"Received SELL fill for {trade_key} without matching BUY entry "
                "in simple log. Ignoring for PnL calc."
            )

    async def _process_backtest_results(
        self,
        config: Dict[str, Any],
        equity_curve: Dict[datetime, Decimal],
        trade_log: List[Dict],
        run_output_dir: str,
        start_run_time: datetime,
    ) -> Dict[str, Any]:
        """Process and save backtest results.

        Args:
            config: Backtest configuration
            equity_curve: Dictionary mapping timestamps to equity values
            trade_log: List of trade information
            run_output_dir: Directory to save results
            start_run_time: Time when backtest started

        Returns:
            Dictionary containing backtest results and metrics
        """
        try:
            # Convert equity curve to pandas Series
            equity_series = pd.Series(
                [float(v) for v in equity_curve.values()], index=list(equity_curve.keys())
            )

            # Calculate performance metrics
            metrics = calculate_performance_metrics(
                equity_series, trade_log, config["initial_capital"]
            )

            # Save results
            results: Dict[str, Any] = {
                "run_id": config["run_id"],
                "start_time": start_run_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "metrics": metrics,
                "output_dir": run_output_dir,
            }

            # Save detailed results to files
            self._save_detailed_results(results, equity_series, trade_log, run_output_dir)

            log.info("Backtest results processed and saved successfully.")
            return results

        except Exception as e:
            log.error(f"Error processing backtest results: {e}")
            # Return empty dict instead of None to match return type
            return {}

    def _save_detailed_results(
        self,
        results: Dict[str, Any],
        equity_series: pd.Series,
        trade_log: List[Dict],
        run_output_dir: str,
    ) -> None:
        """Save detailed backtest results to files.

        Args:
            results: Dictionary containing backtest results
            equity_series: Series containing equity curve data
            trade_log: List of trade information
            run_output_dir: Directory to save results
        """
        try:
            # Save equity curve
            equity_df = equity_series.reset_index()
            equity_df.columns = ["Timestamp", "Equity"]
            equity_path = os.path.join(run_output_dir, "equity_curve.csv")
            equity_df.to_csv(equity_path, index=False)
            log.info(f"Equity curve saved to {equity_path}")

            # Save trade log if not empty
            trade_df = pd.DataFrame(trade_log)
            if not trade_df.empty:
                trade_path = os.path.join(run_output_dir, "trade_log.csv")
                trade_df.to_csv(trade_path, index=False)
                log.info(f"Trade log saved to {trade_path}")
            else:
                log.info("No trades executed during backtest. " "Trade log not saved.")

            # Save summary metrics
            summary_path = os.path.join(run_output_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=4)
            log.info(f"Summary metrics saved to {summary_path}")

        except Exception as e:
            log.error(f"Error saving detailed results: {e}")


def _create_dummy_data(start_date: str, end_date: str, pairs: List[str]) -> pd.DataFrame:
    """Create dummy data for testing.

    Args:
        start_date: Start date for dummy data
        end_date: End date for dummy data
        pairs: List of trading pairs

    Returns:
        DataFrame containing dummy data
    """
    try:
        timestamps = pd.date_range(start_date, end_date, freq="1min", tz="UTC")
    except Exception as e:
        print(f"Error creating date range: {e}. Check config date formats.")
        return pd.DataFrame()

    data_list = []
    for ts in timestamps:
        # XRP/USD Data
        if "XRP/USD" in pairs:
            data_list.append(
                {
                    "timestamp": ts,
                    "pair": "XRP/USD",
                    "open": 0.50 + (ts.minute + ts.hour * 60) * 0.0001,
                    "high": 0.51 + (ts.minute + ts.hour * 60) * 0.0001,
                    "low": 0.49 + (ts.minute + ts.hour * 60) * 0.0001,
                    "close": 0.505 + (ts.minute + ts.hour * 60) * 0.0001,
                    "volume": 1000 + (ts.minute + ts.hour * 60) * 10,
                }
            )
        # DOGE/USD Data (introduce some NaNs)
        if "DOGE/USD" in pairs:
            doge_close = 0.15 + (ts.minute + ts.hour * 60) * 0.00005
            data_list.append(
                {
                    "timestamp": ts,
                    "pair": "DOGE/USD",
                    "open": (doge_close - 0.001 if ts.minute % 10 != 0 else None),
                    "high": doge_close + 0.005,
                    "low": doge_close - 0.005,
                    "close": doge_close,
                    "volume": 5000 + (ts.minute + ts.hour * 60) * 50,
                }
            )

    if not data_list:
        print("Warning: No data generated for the pairs specified in the config.")
        return pd.DataFrame(
            columns=["timestamp", "pair", "open", "high", "low", "close", "volume"]
        ).set_index("timestamp")

    return pd.DataFrame(data_list).set_index("timestamp")


def _setup_example_environment() -> Optional[tuple]:
    """Set up the example environment.

    Returns:
        Tuple of (config_manager, data_path) or None if setup fails
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a ConfigManager instance using the main config file
    main_config_path = "config/config.yaml"
    if not os.path.exists(main_config_path):
        print(
            f"ERROR: Main configuration file not found at {main_config_path}. "
            "Cannot run example."
        )
        return None

    config = ConfigManager(config_path=main_config_path)

    # --- Prepare dummy data based on config --- #
    dummy_data_dir = os.path.dirname(config.get("backtest.data_path", "data/dummy_data.parquet"))
    dummy_data_path = config.get("backtest.data_path", "data/dummy_data.parquet")
    if not os.path.exists(dummy_data_dir):
        os.makedirs(dummy_data_dir)

    return config, dummy_data_path


def _run_example(config: "ConfigManager", dummy_data_path: str) -> None:
    """Run the example backtest.

    Args:
        config: The configuration manager
        dummy_data_path: Path to the dummy data file
    """
    start_date_test = config.get("backtest.start_date", "2023-01-01T00:00:00Z")
    end_date_test = config.get("backtest.end_date", "2023-01-01T05:00:00Z")
    pairs_test = config.get_list("trading.pairs", ["XRP/USD", "DOGE/USD"])

    # Create and save dummy data
    df = _create_dummy_data(start_date_test, end_date_test, pairs_test)
    df.to_parquet(dummy_data_path)
    print(f"Created/Updated dummy data file based on config: {dummy_data_path}")

    # Test the loading function using the real ConfigManager
    engine = BacktestingEngine(config)
    loaded_data = engine._load_historical_data()

    if loaded_data:
        print("\nHistorical data loaded successfully!")
        for pair, data in loaded_data.items():
            print(f"--- {pair} ({len(data)} rows) ---")
            print(data.head())
            print("...")
            print(data.tail())
            print(f"ATR Present: {'atr' in data.columns}")
            if "atr" in data.columns:
                print(f"ATR dtype: {data['atr'].dtype}")
                print(f"ATR non-NaN count: {data['atr'].notna().sum()}")
    else:
        print("\nFailed to load historical data.")


if __name__ == "__main__":
    setup_result = _setup_example_environment()
    if setup_result:
        config, dummy_data_path = setup_result
        _run_example(config, dummy_data_path)
