# Main Application Module (`gal_friday/main.py`) Documentation

## Module Overview

The `gal_friday/main.py` module serves as the primary entry point and central orchestrator for the Gal-Friday algorithmic trading system. It is responsible for initializing all application components, managing their lifecycles (startup, running, shutdown), and ensuring graceful operation. This includes loading configurations, setting up logging, initializing database connections, running schema migrations, and instantiating and coordinating all core services required for live trading or simulation.

## Key Features

-   **Command-Line Argument Handling:** Parses command-line arguments, notably `--config` for specifying the configuration file path and `--log-level` for overriding the default logging level.
-   **Configuration Management:** Initializes `ConfigManager` to load and provide access to application-wide settings from the specified YAML configuration file.
-   **Centralized Logging Setup:** Establishes global application logging using `LoggerService`, allowing for configurable log levels and multiple output handlers (console, file, database, etc.) after an initial basic logging setup.
-   **Process Pool Management:** Manages a `concurrent.futures.ProcessPoolExecutor` for offloading CPU-bound tasks, such as machine learning model inference in the `PredictionService`, to separate processes.
-   **Core Service Orchestration:** Instantiates, manages, and coordinates the lifecycle of all critical application services:
    -   **`PubSubManager`**: For asynchronous event-driven communication between services.
    -   **Database Infrastructure**: Initializes an asynchronous database connection pool (e.g., `DatabaseConnectionPool`) and an SQLAlchemy `async_sessionmaker` for database interactions.
    -   **`MigrationManager`**: Ensures the database schema is up-to-date by running migrations at startup.
    -   **`LoggerService`**: The fully configured logging service.
    -   **Mode-Dependent Services**: Dynamically instantiates services based on the application's run mode (e.g., "live", "backtest", "simulation"):
        -   `MarketPriceService` (e.g., `KrakenMarketPriceService` for live data, `SimulatedMarketPriceService` for backtests).
        -   `HistoricalDataService` (e.g., `KrakenHistoricalDataService`, or a simulated version).
        -   `ExecutionHandler` (e.g., `KrakenExecutionHandler` for live trading, `SimulatedExecutionHandler` for backtests).
    -   **Core Trading Logic Services**: `DataIngestor`, `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, `PortfolioManager`, `RiskManager`.
    -   **`MonitoringService`**: Monitors system health, manages trading halt/resume state, and handles critical alerts.
    -   **`CLIService`**: Provides a command-line interface for runtime interaction and control.
-   **Sequential Service Startup:** Orchestrates the startup sequence of all services in a dependency-aware order.
-   **Graceful Shutdown:** Implements robust shutdown procedures initiated by SIGINT (Ctrl+C), SIGTERM signals, or critical application errors. Services are stopped in reverse order of startup to ensure clean resource release.
-   **Custom Exception Handling:** Uses custom exceptions for critical failures during initialization or runtime to provide specific error information and facilitate controlled shutdown.

## Global Variables & State

-   **`__version__ (str)`**: A module-level variable storing the current version of the Gal-Friday application.
-   **`shutdown_event (asyncio.Event)`**: An asyncio event that is set when a shutdown is requested. Various parts of the application can listen for this event to terminate gracefully.
-   **`global_state (GlobalState)`**: A simple class or namespace intended to hold globally accessible application state, such as the main asyncio event loop (`global_state.main_event_loop`).

## Helper Functions

-   **`setup_logging(config: ConfigManagerProtocol, log_level_override: Optional[str] = None) -> LoggerService`**:
    -   Configures the application's primary logging system.
    -   Initially sets up basic console logging.
    -   Once `ConfigManager` is available, it fully initializes `LoggerService` based on configurations found in `config` (e.g., file logging, database logging, InfluxDB logging) and applies any `log_level_override` from CLI arguments.
    -   Returns the initialized `LoggerService` instance.
-   **`handle_shutdown(sig: signal.Signals, frame: Optional[types.FrameType]) -> None`**:
    -   A signal handler function registered for SIGINT and SIGTERM.
    -   When a caught signal is received, it logs the event and sets the global `shutdown_event`, triggering the application's graceful shutdown procedure.

## Class `GalFridayApp`

This class encapsulates the entire application, managing its state and lifecycle.

### Initialization (`__init__`)

-   **Actions:**
    -   Initializes attributes to store instances of all services, configuration objects, the process pool executor, and the PubSubManager, typically setting them to `None` initially.
    -   Sets up an internal list (`_services_to_manage`) to track instantiated services for easier lifecycle management.

### Core Lifecycle Methods

-   **`async initialize(args: argparse.Namespace) -> None`**:
    -   The main initialization orchestrator.
    -   **1. Load Configuration:** Calls `_load_configuration(args.config_file)` to load settings using `ConfigManager`.
    -   **2. Setup Initial Logging:** Calls `setup_logging()` with the loaded config and CLI log level override to establish basic logging.
    -   **3. Setup Process Pool Executor:** Calls `_setup_executor()` to initialize `ProcessPoolExecutor`.
    -   **4. Instantiate PubSubManager:** Calls `_instantiate_pubsub()`.
    -   **5. Initialize Database:** Initializes the database connection pool (`DatabaseConnectionPool`) and SQLAlchemy `async_sessionmaker`.
    -   **6. Instantiate LoggerService (Full):** Creates the full `LoggerService` instance, now with database capabilities using the sessionmaker. This replaces the initial basic logger.
    -   **7. Run Database Migrations:** Instantiates `MigrationManager` and calls its `run_migrations()` method.
    -   **8. Instantiate Core Services:** Sequentially instantiates all other application services in a dependency-aware order. This involves calling various private helper methods like:
        -   `_instantiate_market_data_services(run_mode)`
        -   `_instantiate_execution_handler(run_mode)`
        -   `_instantiate_portfolio_manager()`
        -   `_instantiate_data_ingestor()`
        -   `_instantiate_feature_engine()`
        -   `_instantiate_prediction_service()`
        -   `_init_strategy_arbitrator()`
        -   `_instantiate_risk_manager()`
        -   `_instantiate_monitoring_service()`
        -   `_instantiate_cli_service()`
        Each instantiation checks for necessary prerequisite services using helpers like `_ensure_class_available()`.
    -   Adds each successfully instantiated service to `_services_to_manage`.

-   **`async start() -> None`**:
    -   Starts the `PubSubManager` to enable event processing.
    -   If the `ConfigManager` supports file watching for dynamic configuration updates, starts that feature.
    -   Calls `_create_and_run_service_start_tasks()` to concurrently start all services in `_services_to_manage` that have an `async start()` method.
    -   Uses `_handle_service_startup_results()` to check for any failures during service startup and logs them. If a critical service fails to start, it might trigger a shutdown.

-   **`async stop() -> None`**:
    -   Stops `ConfigManager` file watching if applicable.
    -   Calls `_initiate_service_shutdown()` to gracefully stop all managed services in reverse order of their startup.
    -   Calls `_cancel_active_tasks()` to cancel any remaining asyncio tasks that haven't completed.
    -   Closes the database connection pool (`DatabaseConnectionPool.close()`).
    -   Calls `_shutdown_process_executor()` to shut down the `ProcessPoolExecutor`.
    -   Logs the completion of the shutdown process.

-   **`async run() -> None`**:
    -   The main execution method for the application.
    -   Calls `await self.initialize(args)` (where `args` are parsed command-line arguments).
    -   Calls `await self.start()`.
    -   Logs that the application is now running and ready.
    -   Enters a loop, waiting for `shutdown_event.wait()` to be set.
    -   Once `shutdown_event` is set (e.g., by a signal or an internal error), it proceeds to call `await self.stop()`.
    -   Handles potential exceptions during the run lifecycle, ensuring `stop()` is called.

### Internal Helper Methods

-   **`_load_configuration(config_path: str) -> None`**: Initializes `ConfigManager` with the given `config_path`.
-   **`_setup_executor() -> None`**: Creates and stores a `concurrent.futures.ProcessPoolExecutor` instance.
-   **`_instantiate_pubsub() -> None`**: Creates and stores a `PubSubManager` instance.
-   **`_instantiate_execution_handler(run_mode: str) -> None`**:
    -   Selects the appropriate `ExecutionHandler` class (e.g., `KrakenExecutionHandler` for "live" mode, `SimulatedExecutionHandler` for "backtest" or "simulation" mode) based on `run_mode` from the configuration.
    -   Instantiates and stores it.
-   **`_instantiate_market_data_services(run_mode: str) -> None`**: Similarly instantiates appropriate `MarketPriceService` and `HistoricalDataService` based on `run_mode`.
-   **`_ensure_class_available(instance: Optional[Any], class_name: str, required_for: str) -> None`**: Utility method to check if a required service instance (a prerequisite) has been successfully initialized. Raises a custom `DependencyNotInstantiatedError` if not.
-   **`_raise_dependency_not_instantiated(dependency_name: str, service_name: str) -> None`**: Helper to raise the `DependencyNotInstantiatedError`.
-   **`_create_kraken_exchange_spec() -> Optional[ExchangeSpecification]`**: (If Kraken is a primary exchange) A helper to create an `ExchangeSpecification` object for Kraken, containing details like fees, rate limits, etc. This might be passed to Kraken-specific services.
-   **`_create_and_run_service_start_tasks()`**: Creates a list of asyncio tasks for calling the `start()` method of each managed service.
-   **`_handle_service_startup_results(results: List[Tuple[str, bool, Optional[Exception]]])`**: Processes the results from starting services, logging successes or failures.
-   **`_initiate_service_shutdown()`**: Iterates through `_services_to_manage` in reverse and calls their `stop()` method.
-   **`_cancel_active_tasks()`**: Cancels any remaining asyncio tasks.
-   **`_shutdown_process_executor()`**: Gracefully shuts down the `ProcessPoolExecutor`.

## Main Execution Block (`if __name__ == "__main__":`)

-   Uses `argparse.ArgumentParser` to define and parse command-line arguments:
    -   `--config-file` (or `-c`): Path to the `config.yaml` file. Defaults to a path relative to the script.
    -   `--log-level` (or `-l`): Overrides the log level defined in the config file (e.g., "DEBUG", "INFO").
-   Calls `asyncio.run(main_async(args))` to start the asynchronous application.

## Function `main_async(args: argparse.Namespace)`

-   **Purpose:** The primary asynchronous function that sets up and runs the `GalFridayApp`.
-   **Actions:**
    -   Sets `global_state.main_event_loop = asyncio.get_running_loop()`.
    -   Registers `handle_shutdown` for `signal.SIGINT` and `signal.SIGTERM` to ensure graceful shutdown on these signals.
    -   Creates an instance of `GalFridayApp`.
    -   Calls `await app.run(args)` to start and manage the application lifecycle.
    -   Includes error handling to log critical failures during `app.run()`.

## Dependencies

-   **Standard Libraries:**
    -   `argparse`: For parsing command-line arguments.
    -   `asyncio`: The core library for asynchronous programming.
    -   `concurrent.futures.ProcessPoolExecutor`: For managing a pool of worker processes.
    -   `functools.partial`: Used for signal handling.
    -   `logging`: For basic logging setup before `LoggerService` is fully initialized.
    -   `signal`: For handling OS signals (SIGINT, SIGTERM).
    -   `sys`: For system-specific parameters and functions.
    -   `pathlib.Path`: For object-oriented filesystem path manipulation.
    -   `types.FrameType`: For type hinting in signal handlers.
-   **SQLAlchemy:**
    -   `sqlalchemy.ext.asyncio.async_sessionmaker`: For creating asynchronous database sessions.
-   **Core Application Modules:**
    -   All service modules: `ConfigManager`, `LoggerService`, `PubSubManager`, `DatabaseConnectionPool` (or similar DB interface), `MigrationManager`, `MarketPriceService` (and its implementations), `HistoricalDataService` (and implementations), `ExecutionHandler` (and implementations), `PortfolioManager`, `DataIngestor`, `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, `RiskManager`, `MonitoringService`, `CLIService`.
    -   Custom exception definitions.
    -   `ExchangeSpecification` (if used).

## Configuration

The `main.py` module itself doesn't define configurations but relies heavily on the `config.yaml` file (path provided via CLI) for all operational parameters. This includes:
-   Run mode ("live", "paper", "simulation", "backtest").
-   Database connection URLs.
-   API keys for exchanges (though these should be securely managed, potentially via environment variables or a secrets manager, accessed through `ConfigManager`).
-   Parameters for every service initialized by `GalFridayApp`.
-   Logging levels and output configurations.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the understanding, development, and maintenance of the Gal-Friday application's main entry point and orchestration logic.
