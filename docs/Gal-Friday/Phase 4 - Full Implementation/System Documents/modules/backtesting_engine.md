# BacktestingEngine Module Documentation

## Module Overview

The `gal_friday.backtesting_engine.py` module provides a comprehensive framework for simulating algorithmic trading strategies against historical market data. Its primary purpose is to allow developers and researchers to evaluate the performance of their strategies in a controlled, simulated environment before deploying them in live trading. The engine orchestrates various simulated core trading services, processes historical data chronologically, and generates detailed performance metrics and reports.

## Key Features

-   **Full Simulation Orchestration:** Manages the entire backtesting lifecycle, from data loading and service initialization to event processing and results generation.
-   **Historical Data Handling:**
    -   Loads historical market data for multiple trading pairs.
    -   Expects data in CSV format with 'pair', 'timestamp', and OHLCV (Open, High, Low, Close, Volume) columns. Other relevant columns like 'vwap' or 'trades' might also be used.
-   **Simulated Service Environment:**
    -   Initializes and manages simulated versions of essential trading services, including:
        -   `SimulatedMarketPriceService`: Provides market prices based on historical data.
        -   `SimulatedExecutionHandler`: Simulates order execution (market, limit, SL/TP) against historical bars.
        -   `PortfolioManager`: Tracks simulated funds, positions, and portfolio value.
        -   `FeatureEngine`: Generates features from simulated market data events.
        -   `PredictionService`: Runs predictive models on generated features.
        -   `StrategyArbitrator`: Applies strategy logic to predictions.
        -   `RiskManager`: Assesses proposed trades against simulated risk parameters.
-   **Chronological Event-Driven Simulation:**
    -   Iterates through historical data, advancing timestamp by timestamp across all loaded pairs.
    -   At each timestamp, it generates `MarketDataOHLCVEvent`s from the historical data to drive the feature engineering and prediction pipeline.
-   **Portfolio Performance Tracking:** Monitors and records the portfolio's equity throughout the simulation, generating an equity curve.
-   **Comprehensive Performance Metrics:** Upon completion of the backtest, calculates a wide range of performance metrics, including returns, annualized returns, drawdown metrics (max, average), risk-adjusted metrics (Sharpe, Sortino), and various trade statistics (win rate, profit factor, average holding period).
-   **Results Output:** Saves backtesting results, including a summary, a detailed trade log, and all calculated performance metrics, to a JSON file for later analysis.
-   **Configuration Driven:** The backtesting process (data paths, date ranges, strategy parameters) is configurable via a `ConfigManagerProtocol`-compliant configuration object.

## Helper Functions (for metrics calculation)

The module includes several helper functions, primarily used internally by `calculate_performance_metrics`, to compute various aspects of strategy performance. These functions operate on the equity curve and trade log data.

-   **`decimal_to_float(obj)`**: A utility function to recursively convert `Decimal` objects within nested data structures (lists, dicts) to floats, which is necessary for JSON serialization of results.
-   **`_calculate_basic_returns_and_equity(...)`**: Calculates total return, equity curve, and other basic return statistics.
-   **`_calculate_annualized_return(...)`**: Computes the annualized return based on the total return and the backtest period.
-   **`_calculate_drawdown_metrics(...)`**: Analyzes the equity curve to find maximum drawdown, average drawdown, and longest drawdown period.
-   **`_calculate_risk_adjusted_metrics(...)`**: Calculates metrics like Sharpe Ratio and Sortino Ratio, typically requiring risk-free rate and return series.
-   **`_calculate_trade_statistics(...)`**: Computes statistics related to individual trades, such as win rate, loss rate, average win/loss, profit factor, total number of trades, etc.
-   **`_calculate_average_holding_period(...)`**: Determines the average duration for which trades were held.
-   **`calculate_performance_metrics(equity_curve: pd.Series, trade_log: List[Dict], initial_capital: Decimal, risk_free_rate: float = 0.0) -> Dict[str, Any]`**:
    -   The main orchestrator for performance metric calculation.
    -   Takes the equity curve (Pandas Series), trade log (list of trade dictionaries), initial capital, and an optional risk-free rate.
    -   Calls the various private helper functions above to compute a comprehensive dictionary of performance metrics.

## Protocols & Type Aliases

-   **`ConfigManagerProtocol`**: An informal protocol (often a `typing.Protocol`) that defines the expected interface for the configuration manager object passed to the `BacktestingEngine`. It should provide methods like `get(key, default)` to access configuration values.
-   **Type Aliases**: The module may define various type aliases for clarity and conciseness, such as `Timestamp` (e.g., `pd.Timestamp`), `Price` (`Decimal`), `Symbol` (`str`), `EventHandler` (`Callable[[Event], Coroutine[Any, Any, None]]`).

## Internal Helper Classes

-   **`BacktestHistoricalDataProviderImpl`**:
    -   A class responsible for loading, storing, and providing access to historical market data bars during the backtest.
    -   It typically loads data from CSV files into a Pandas DataFrame.
    -   Provides methods to retrieve a specific bar for a given trading pair and timestamp.
    -   May include functionality to calculate technical indicators like ATR (Average True Range), potentially using TA-Lib if available, or a fallback implementation.
-   **Placeholder Service Implementations (e.g., `FeatureEngineImpl`, `PortfolioManagerImpl`)**:
    -   The code might include placeholder or stub implementations for core services when `typing.TYPE_CHECKING` is false. These are primarily for type hinting purposes or to allow the script to be executed in a limited way without the full application context.
    -   In a full backtest run, these are expected to be replaced by fully functional (simulated) versions of the respective services.

## Class `BacktestingEngine`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config (ConfigManagerProtocol)`: An instance of a configuration manager that provides access to backtest and application configurations.
    -   `data_dir (str)`: The root directory where historical market data CSV files are stored.
    -   `results_dir (str)`: The directory where backtest results (JSON files, logs) will be saved.
    -   `max_workers (Optional[int])`: The maximum number of worker processes for parallel tasks (e.g., data loading, some computations). *Note: The provided snippet might not show active use of `max_workers` for the core simulation loop, which is typically single-threaded for chronological event processing.*
-   **Actions:**
    -   Stores the `config` object.
    -   Sets up `Path` objects for `data_dir` and `results_dir`, creating the results directory if it doesn't exist.
    -   Initializes a `LoggerService` instance for logging backtest progress and errors.
    -   Initializes internal data storage attributes (e.g., `_raw_data_cache`, `_pair_data_map`).
    -   Initializes attributes for core simulated services to `None` (e.g., `_market_price_service`, `_execution_handler`, `_portfolio_manager`, etc.). These are expected to be instantiated and configured by a main orchestrating method like `run_backtest()`.

### Configuration & Data Loading

-   **`_get_backtest_config() -> dict`**:
    -   Retrieves the backtest-specific configuration section (e.g., under a key like "backtest_settings") from the main `config` object.
-   **`_validate_config(config_dict: dict) -> None`**:
    -   Validates that required fields are present in the `config_dict` (e.g., `start_date`, `end_date`, `trading_pairs`, `initial_capital`).
    -   Raises `ValueError` if critical configurations are missing or invalid.
-   **`_load_raw_data(data_path: Path) -> pd.DataFrame`**:
    -   Loads historical data from a CSV file specified by `data_path` into a Pandas DataFrame.
    -   Handles potential file reading errors.
-   **`_clean_and_validate_data(data: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame`**:
    -   Filters the loaded `data` to include only records within the specified `start_date` and `end_date`.
    -   Performs data validation: checks for essential columns (e.g., 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'pair').
    -   Converts 'timestamp' column to `pd.Timestamp` objects if not already.
    -   Handles missing data (e.g., by logging, forward-filling, or erroring based on strategy).
-   **`_process_pairs_data(data: pd.DataFrame) -> Dict[str, pd.DataFrame]`**:
    -   Takes the combined, cleaned DataFrame for all pairs.
    -   Splits the data by 'pair', creating a dictionary where keys are trading pairs (e.g., "BTC/USD") and values are DataFrames containing data for that specific pair.
    -   May perform additional pair-specific processing or feature engineering setup here (though major feature engineering is usually delegated to `FeatureEngine`).

### Simulation Execution

-   **`async _execute_simulation(services: Dict[str, Any], run_config: Dict[str, Any]) -> None`**:
    -   The core loop that drives the backtesting simulation.
    -   `services`: A dictionary containing initialized instances of all simulated services.
    -   `run_config`: The specific configuration for this backtest run.
    -   **1. Start Services:** Calls the `start()` method on all registered/simulated services (e.g., `PubSubManager`, `PortfolioManager`, `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, `RiskManager`).
    -   **2. Prepare Timestamps:** Gathers all unique timestamps from the loaded historical data for all trading pairs and sorts them chronologically. This forms the main timeline for the simulation.
    -   **3. Iterate Through Time:** Loops through each unique `timestamp`:
        -   Updates the current time in `SimulatedMarketPriceService` to this `timestamp`.
        -   For each `trading_pair` that has data at this `timestamp`:
            -   Calls `_process_market_data_for_timestamp(trading_pair, timestamp, services)` to fetch the historical bar and publish a `MarketDataOHLCVEvent`. This event will flow through `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, and `RiskManager`.
        -   After processing market data events for the current timestamp (allowing strategy logic to potentially generate orders), it calls methods on `SimulatedExecutionHandler` to process any pending limit orders or SL/TP orders against the current bar's OHLC prices.
        -   Records the current portfolio equity from `PortfolioManager` to build the equity curve.
        -   Includes a small `asyncio.sleep(0)` to yield control, allowing other asyncio tasks (like those within services) to run.
    -   **4. Stop Services:** Upon completion of all timestamps or if an unhandled error occurs, calls the `stop()` method on all services to allow for graceful shutdown and resource cleanup.
    -   Handles exceptions during the loop, logs them, and ensures services are stopped.

-   **`async _process_market_data_for_timestamp(trading_pair: str, timestamp: pd.Timestamp, services: Dict[str, Any]) -> None`**:
    -   Retrieves the historical OHLCV bar for the given `trading_pair` and `timestamp` using `_get_bar_at_timestamp()`.
    -   If data is found, constructs a `MarketDataOHLCVEvent` with the bar data.
    -   Publishes this event using the `PubSubManager` instance (obtained from `services`). This event is then consumed by the `FeatureEngine`.

-   **`_get_bar_at_timestamp(trading_pair: str, timestamp: pd.Timestamp) -> Optional[pd.Series]`**:
    -   Accesses the pre-loaded and processed historical data (stored in `_pair_data_map`).
    -   Returns the Pandas Series representing the OHLCV bar for the `trading_pair` at the specified `timestamp`.
    -   Returns `None` if no data is available for that exact timestamp and pair.

### Results Processing

-   **`async _process_and_save_results(services: Dict[str, Any], run_config: Dict[str, Any]) -> None`**:
    -   Called after the simulation loop (`_execute_simulation`) completes.
    -   Retrieves final portfolio summary (equity, balances, positions) and the detailed trade history from the simulated `PortfolioManager`.
    -   Retrieves the recorded equity curve.
    -   Calls `calculate_performance_metrics()` with the equity curve, trade log, and initial capital to get a comprehensive dictionary of performance metrics.
    -   Combines the run configuration, portfolio summary, trade log, equity curve (as a list of [timestamp, value]), and performance metrics into a single results dictionary.
    -   Uses `decimal_to_float()` to prepare the results for JSON serialization.
    -   Saves this results dictionary to a JSON file in the `results_dir`. The filename typically includes the strategy name and run timestamp.
    -   Logs the location of the saved results file.

### Note on Service Initialization and Overall Orchestration

The provided documentation describes the internal workings of the `BacktestingEngine`. A complete backtesting script would typically have a public method (e.g., `async run_backtest(self, run_name: str)`) within the `BacktestingEngine` class. This method would:
1.  Load and validate the specific backtest run configuration.
2.  Load and process historical data using methods like `_load_raw_data`, `_clean_and_validate_data`.
3.  **Initialize all simulated services**: This is a crucial step where instances of `PubSubManager`, `SimulatedMarketPriceService` (using the loaded historical data), `SimulatedExecutionHandler`, `PortfolioManager`, `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, and `RiskManager` are created and configured for the backtest environment. These services would use the backtest's `ConfigManagerProtocol` for their own configurations.
4.  Pass these initialized services to `_execute_simulation()`.
5.  Call `_process_and_save_results()` upon completion.

The `__init__` method initializes service attributes to `None`, indicating that their actual instantiation and setup are handled by such an orchestrating method.

## Dependencies

-   **`asyncio`**: For asynchronous operations, especially if services use async features.
-   **`json`**: For saving results.
-   **`logging`**: For logging (via `LoggerService`).
-   **`uuid`**: For generating unique IDs (e.g., for events or trades).
-   **`datetime`**: For date and time manipulations.
-   **`decimal.Decimal`**: For precise financial calculations.
-   **`enum.Enum`**: If custom enums are used (e.g., for event types, order sides).
-   **`pathlib.Path`**: For filesystem path manipulations.
-   **`numpy`**: Used by Pandas and potentially for numerical calculations in metrics.
-   **`pandas`**: For historical data manipulation (DataFrames, Series) and equity curve.
-   **`talib` (Technical Analysis Library)**: Optional, potentially used by `BacktestHistoricalDataProviderImpl` for ATR calculation or other indicators. The engine should have a fallback if TA-Lib is not installed.
-   **Core Application Modules:**
    -   `gal_friday.config.ConfigManagerProtocol` (or similar for configuration).
    -   `gal_friday.core.pubsub.PubSubManager`.
    -   Simulated versions of services:
        -   `gal_friday.market_price_service.SimulatedMarketPriceService`
        -   `gal_friday.execution_handler.SimulatedExecutionHandler`
        -   `gal_friday.portfolio_manager.PortfolioManager` (configured for simulation)
        -   `gal_friday.feature_engine.FeatureEngine`
        -   `gal_friday.prediction_service.PredictionService`
        -   `gal_friday.strategy_arbitrator.StrategyArbitrator`
        -   `gal_friday.risk_manager.RiskManager`
    -   `gal_friday.core.events` (for `MarketDataOHLCVEvent` and other system events).
    -   `gal_friday.logger_service.LoggerService`.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `BacktestingEngine` module.
