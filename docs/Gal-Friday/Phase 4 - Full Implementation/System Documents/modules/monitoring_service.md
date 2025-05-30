# MonitoringService Module Documentation

## Module Overview

The `gal_friday.monitoring_service.py` module is a vital component of the Gal-Friday trading system, tasked with overseeing overall system health, monitoring key performance and risk metrics, and managing the global HALT state of trading operations. It continuously checks various conditions and can automatically trigger a system-wide halt if predefined thresholds are breached. It also processes manual halt/resume requests (e.g., from the CLI) and handles `PotentialHaltTriggerEvent`s from other services. Upon a system halt, it can be configured to manage open positions according to predefined rules.

## Key Features

-   **Global HALT State Management:** Manages the system's operational state, broadcasting "RUNNING" or "HALTED" statuses via `SystemStateEvent`s to inform other components.
-   **Halt Coordination:** Integrates with a `HaltCoordinator` (or an internal instance) to manage multiple potential halt conditions. A halt is triggered if any registered condition in the coordinator becomes true.
-   **Periodic Health and Risk Checks:** Regularly performs a suite of checks:
    -   **Portfolio Drawdown:** Monitors total and daily portfolio drawdown percentages against configured limits.
    -   **Consecutive Trading Losses:** Tracks the number of consecutive losing trades.
    -   **Market Volatility:** (Placeholder for specific calculation logic) Intended to monitor market volatility and potentially halt if it exceeds safe thresholds.
    -   **System Resource Usage:** Monitors system CPU and memory utilization using `psutil`.
    -   **API Connectivity & Error Rates:** Tracks API error frequency to detect issues with exchange communication.
    -   **Market Data Freshness:** Ensures that market data for actively traded pairs is not stale.
-   **Event-Driven Monitoring:**
    -   Subscribes to `PotentialHaltTriggerEvent` to act on halt requests from other critical services.
    -   Listens for `SYSTEM_ERROR` events as a potential indicator for issues.
    -   Monitors `MARKET_DATA_L2` and `MARKET_DATA_OHLCV` events to track data freshness.
    -   Processes `EXECUTION_REPORT` events to update consecutive loss counters.
-   **Manual Halt/Resume Control:** Provides methods (`trigger_halt`, `trigger_resume`) that can be invoked externally (e.g., by `CLIService`) to manually halt or resume trading operations.
-   **Configurable Position Handling on HALT:** Behavior regarding open positions when a system HALT is triggered (e.g., "close_all", "maintain") is configurable.
-   **Configuration Driven:** Thresholds for various checks, monitoring intervals, and halt behaviors are loaded from the `ConfigManager`.

## Class `MonitoringService`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config_manager (ConfigManager)`: An instance of `ConfigManager` for accessing application configuration.
    -   `pubsub_manager (PubSubManager)`: An instance of `PubSubManager` for event subscription and publication.
    -   `portfolio_manager (PortfolioManager)`: An instance of `PortfolioManager` to fetch current portfolio state (equity, drawdown).
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
    -   `execution_handler (Optional[ExecutionHandler])`: An optional `ExecutionHandler` instance, used if positions need to be closed on halt.
    -   `halt_coordinator (Optional[HaltCoordinator])`: An optional `HaltCoordinator`. If not provided, the service instantiates its own.
-   **Actions:**
    -   Stores references to all provided services.
    -   Calls `_load_configuration()` to load monitoring-specific settings.
    -   Initializes the system halt state (`_is_system_halted = False`).
    -   Sets up internal trackers:
        -   `_last_market_data_timestamps (defaultdict(datetime))`: Tracks last seen timestamps for market data per pair.
        -   `_api_error_timestamps (deque)`: Stores timestamps of recent API errors.
        -   `_consecutive_losses (int)`: Counter for consecutive losing trades.
    -   Instantiates `HaltCoordinator` if one is not passed in, providing it with necessary services like `PortfolioManager` and `LoggerService`.
    -   Initializes `_monitoring_task` to `None`.

### Service Lifecycle

-   **`async start() -> None`**:
    -   Starts the periodic monitoring loop by creating an asyncio task for `_run_periodic_checks()`.
    -   Subscribes to relevant events:
        -   `EventType.POTENTIAL_HALT_TRIGGER` (handler: `_handle_potential_halt_trigger`)
        -   `EventType.SYSTEM_ERROR` (handler: can be generic or specific for logging/assessment)
        -   `EventType.MARKET_DATA_L2` (handler: `_update_market_data_timestamp`)
        -   `EventType.MARKET_DATA_OHLCV` (handler: `_update_market_data_timestamp`)
        -   `EventType.EXECUTION_REPORT` (handler: `_handle_execution_report`)
        -   Potentially `APIErrorEvent` if defined (handler: `_handle_api_error`)
    -   Publishes an initial `SystemStateEvent` indicating the system is "RUNNING".
    -   Logs that the MonitoringService has started.

-   **`async stop() -> None`**:
    -   Stops the periodic monitoring task by cancelling `_monitoring_task` and awaiting its completion.
    -   Unsubscribes from all registered events.
    -   Logs that the MonitoringService is stopping.

### State Management

-   **`is_halted() -> bool`**:
    -   Returns the current system HALT state (`self._is_system_halted`). This method is often called by other services (like `RiskManager` or `ExecutionHandler`) to check if new actions are permitted.

-   **`async trigger_halt(reason: str, source: str) -> None`**:
    -   Initiates a system HALT if not already halted.
    -   Sets `self._is_system_halted = True`.
    -   Notifies the `_halt_coordinator` by calling `_halt_coordinator.set_manual_halt(True, reason, source)` or a similar method to record the halt condition.
    -   Logs the halt event with the reason and source.
    -   Calls `_publish_state_change("HALTED", reason, source)`.
    -   Calls `await self._handle_positions_on_halt()` to manage open positions according to configuration.

-   **`async trigger_resume(source: str) -> None`**:
    -   Resumes system operations if currently HALTED.
    -   Sets `self._is_system_halted = False`.
    -   Clears relevant halt conditions in `_halt_coordinator` (e.g., `_halt_coordinator.set_manual_halt(False)`).
    -   Logs the resume event.
    -   Calls `_publish_state_change("RUNNING", "System resumed", source)`.

-   **`_publish_state_change(new_state: str, reason: str, source: str) -> None`**:
    -   A helper method to construct and publish a `SystemStateEvent` via `PubSubManager`.
    -   The event includes the `new_state` ("RUNNING" or "HALTED"), `reason`, and `source` of the state change.

-   **`async _handle_positions_on_halt() -> None`**:
    -   Implements the logic defined by `_halt_position_behavior` configuration.
    -   If configured to "close_all_positions":
        -   Retrieves all open positions from `_portfolio_manager.get_open_positions()`.
        -   For each open position, creates and publishes a `ClosePositionCommand` (or directly interacts with `ExecutionHandler` to place market orders to close them). This requires `ExecutionHandler` to be available.
        -   Logs actions taken for each position.
    -   If configured to "maintain_positions", it logs that positions are being maintained.
    -   Other strategies (e.g., "reduce_risk") could also be implemented.

### Event Handlers

-   **`async _handle_potential_halt_trigger(event: PotentialHaltTriggerEvent) -> None`**:
    -   Processes `PotentialHaltTriggerEvent`s received from other services.
    -   Extracts the `reason` and `source_module` from the event.
    -   Calls `await self.trigger_halt(reason, source=source_module)`.

-   **`_update_market_data_timestamp(event: Union[MarketDataL2Event, MarketDataOHLCVEvent]) -> None`**:
    -   Updates `_last_market_data_timestamps[event.trading_pair]` with `event.timestamp`. Used by `_check_market_data_freshness`.

-   **`async _handle_execution_report(event: ExecutionReportEvent) -> None`**:
    -   Processes filled orders from `ExecutionReportEvent` to track consecutive losses.
    -   If the trade resulted in a loss (requires P&L calculation, often from `PortfolioManager` or by comparing fill price to an average entry price of a closed portion), increments `_consecutive_losses`.
    -   If a profit, resets `_consecutive_losses = 0`.
    -   If `_consecutive_losses` exceeds `_consecutive_loss_limit`, updates the `HaltCoordinator` (e.g., `_halt_coordinator.set_consecutive_losses_exceeded(True)`).

-   **`async _handle_api_error(event: APIErrorEvent) -> None`**: (Assuming `APIErrorEvent` is defined and published)
    -   Appends the current timestamp to `_api_error_timestamps`.
    -   Prunes `_api_error_timestamps` to keep only errors within `_api_error_threshold_period_s`.
    -   If the count of recent errors exceeds `_api_error_threshold_count`, updates `HaltCoordinator`.

### Periodic Checks (`_run_periodic_checks` loop calls `_check_all_halt_conditions`)

-   **`async _run_periodic_checks() -> None`**:
    -   The main loop for periodic monitoring. Runs every `_check_interval_seconds`.
    -   Calls `await _check_all_halt_conditions()`.
    -   If `_halt_coordinator.is_any_condition_met()` returns true and the system is not already halted, it calls `await self.trigger_halt(reason="Automatic halt triggered by coordinator", source="MonitoringService")`.

-   **`async _check_all_halt_conditions() -> None`**:
    -   Orchestrates calls to all individual condition-checking methods:
        -   `await _check_drawdown_conditions()`
        -   `await _check_market_volatility()`
        -   `await _check_system_health()`
        -   `await _check_api_connectivity()`
        -   `await _check_market_data_freshness()`
        -   `await _check_position_risk()` (if implemented)
        -   The consecutive loss check is primarily event-driven but could also be verified here from `_consecutive_losses`.

-   **`async _check_drawdown_conditions() -> None`**:
    -   Fetches current portfolio state from `_portfolio_manager.get_current_state()`.
    -   Compares `portfolio_state['drawdown_metrics']['total_drawdown_pct']` and `portfolio_state['drawdown_metrics']['daily_drawdown_pct']` against configured limits (e.g., `_halt_coordinator.config.max_total_drawdown_pct`).
    -   Updates `_halt_coordinator` with the status of these checks (e.g., `_halt_coordinator.set_total_drawdown_exceeded(True/False)`).

-   **`async _check_market_volatility() -> None`**:
    -   **Placeholder:** This method would contain logic to calculate or fetch current market volatility for key trading pairs.
    -   Compare it against pre-defined thresholds.
    -   Update `_halt_coordinator` if volatility is extreme.

-   **`async _check_system_health() -> None`**:
    -   Calls `_check_system_resources()`.

-   **`async _check_api_connectivity() -> None`**:
    -   **Placeholder:** This might involve sending a test ping to the exchange API or checking the rate of API errors from `_api_error_timestamps`.
    -   Update `_halt_coordinator` if connectivity issues are detected.

-   **`async _check_market_data_freshness() -> None`**:
    -   Iterates through `_active_trading_pairs` (from config).
    -   For each pair, checks `_last_market_data_timestamps[pair]` against the current time.
    -   If `current_time - last_timestamp > _data_staleness_threshold_s`, considers data stale for that pair.
    -   Updates `_halt_coordinator` if critical data streams are stale.

-   **`async _check_position_risk() -> None`**:
    -   **Placeholder:** Could involve more granular checks on individual positions, e.g., if a single position's unrealized P&L exceeds a certain negative threshold not covered by overall drawdown.

-   **`_check_system_resources() -> None`**:
    -   Uses `psutil.cpu_percent()` and `psutil.virtual_memory().percent` to get current CPU and memory usage.
    -   If usage exceeds `_cpu_threshold_pct` or `_memory_threshold_pct`, updates `_halt_coordinator`.

### Internal Helpers

-   **`_load_configuration() -> None`**:
    -   Loads various settings from the `monitoring` section of the configuration via `_config_manager`.
    -   Examples: `check_interval_seconds`, `api_failure_threshold_count`, `api_error_threshold_period_s`, `data_staleness_threshold_s`, `cpu_threshold_pct`, `memory_threshold_pct`, `consecutive_loss_limit`, `halt_position_behavior`.
    -   Also loads relevant trading pairs (`trading.pairs`) for data freshness checks.

-   **`async _calculate_volatility(pair: str) -> Optional[Decimal]`**:
    -   **Placeholder:** This method would implement the actual logic to calculate market volatility for a given `pair` (e.g., using ATR from historical data, or from real-time tick data if available).

## Mock Implementations (for testing)

The `monitoring_service.py` file may include mock implementations for its dependencies (e.g., `TestLoggerService`, `MockConfigManager`, `MockPortfolioManager`, `MockPubSubManager`, `MockHaltCoordinator`) typically within an `if __name__ == "__main__":` block or imported from a dedicated mocks file.
An `example_main()` asynchronous function is also likely present to demonstrate instantiation and basic operation of the `MonitoringService` with these mocks for standalone testing.

## Dependencies

-   **Standard Libraries:**
    -   `asyncio`: For asynchronous operations and background tasks.
    -   `logging`: For standard logging (though usually wrapped by `LoggerService`).
    -   `time`: For time-related operations if needed (less common in async).
    -   `uuid`: For generating unique IDs.
    -   `datetime`, `collections.defaultdict`, `collections.deque`: For data storage and timestamping.
    -   `decimal.Decimal`: For precise numerical comparisons.
-   **Third-Party Libraries:**
    -   `psutil`: For fetching system resource usage (CPU, memory).
-   **Core Application Modules:**
    -   `gal_friday.config_manager.ConfigManager`
    -   `gal_friday.core.pubsub.PubSubManager`
    -   `gal_friday.portfolio_manager.PortfolioManager`
    -   `gal_friday.logger_service.LoggerService`
    -   `gal_friday.execution_handler.ExecutionHandler` (Optional)
    -   `gal_friday.halt_coordinator.HaltCoordinator` (or its interface)
    -   Various `Event` types from `gal_friday.core.events` (e.g., `SystemStateEvent`, `PotentialHaltTriggerEvent`, `MarketDataL2Event`, `MarketDataOHLCVEvent`, `ExecutionReportEvent`, `APIErrorEvent`).

## Configuration (Key options from `monitoring` and other relevant sections of app config)

-   **`monitoring.check_interval_seconds (int)`**: Frequency for running periodic health and risk checks.
-   **`monitoring.api_error_threshold_count (int)`**: Number of API errors within `api_error_threshold_period_s` to trigger an API connectivity concern.
-   **`monitoring.api_error_threshold_period_s (int)`**: Time window (in seconds) for tracking API error frequency.
-   **`monitoring.data_staleness_threshold_s (int)`**: Maximum allowed delay (in seconds) for market data before it's considered stale.
-   **`monitoring.cpu_threshold_pct (float)`**: CPU usage percentage threshold.
-   **`monitoring.memory_threshold_pct (float)`**: Memory usage percentage threshold.
-   **`monitoring.consecutive_loss_limit (int)`**: Maximum number of consecutive losing trades allowed.
-   **`monitoring.halt.position_behavior (str)`**: Defines action on open positions upon system HALT (e.g., "close_all_positions", "maintain_positions").
-   **`risk_manager.limits.max_total_drawdown_pct (Decimal)`**: (Accessed via `PortfolioManager` or directly from config/`HaltCoordinator`) Max total portfolio drawdown.
-   **`risk_manager.limits.max_daily_drawdown_pct (Decimal)`**: Max daily portfolio drawdown.
-   **`trading.pairs (List[str])`**: List of active trading pairs, used for market data freshness checks.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `MonitoringService` module.
