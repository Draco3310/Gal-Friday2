# PortfolioManager Module Documentation

## Module Overview

The `gal_friday.portfolio_manager.py` module serves as the central orchestrator for managing the application's trading portfolio. It integrates three key sub-components: `FundsManager` (for tracking currency balances and available capital), `PositionManager` (for tracking open and closed positions), and `ValuationService` (for calculating real-time portfolio value, equity, profit/loss, and drawdown metrics).

The PortfolioManager subscribes to `ExecutionReportEvent`s to update its internal state based on trade fills and order cancellations. It initializes the portfolio with starting capital and positions from the configuration. A key responsibility is to maintain a comprehensive and real-time view of the portfolio, which can be queried synchronously. It also supports periodic reconciliation of its internal state with an external exchange, provided a compatible `ExecutionHandler` is available.

## Key Features

-   **Integrated Portfolio View:** Combines data from `FundsManager`, `PositionManager`, and `ValuationService` to provide a holistic view of the portfolio.
-   **Event-Driven Updates:** Subscribes to `ExecutionReportEvent` to react to trade executions (fills) and order cancellations, updating funds, positions, and triggering re-valuation.
-   **Initial State Configuration:** Loads initial funds (capital) and open positions from the application configuration at startup.
-   **Funds and Position Management:**
    -   Updates available funds in `FundsManager` when trades are executed.
    -   Updates open/closed positions in `PositionManager` based on trade details.
    -   Accounts for trade commissions by adjusting fund balances.
-   **Real-time Valuation:** Triggers the `ValuationService` to update portfolio equity, Net Asset Value (NAV), Profit & Loss (P&L), and drawdown metrics after any state change.
-   **Synchronous State Retrieval:** Provides a `get_current_state()` method that returns a snapshot of the current portfolio, including balances, positions with market values, overall equity, drawdown metrics, and market exposure. This uses cached data for performance.
-   **Exchange Reconciliation:**
    -   Supports periodic reconciliation of internal fund balances and open positions with those reported by an external exchange.
    -   This feature requires the provided `ExecutionHandler` to implement the `ReconcilableExecutionHandler` protocol.
    -   Reconciliation interval, discrepancy thresholds, and auto-reconciliation behavior are configurable.
-   **Configurable Drawdown Resets:** Allows configuration of daily and weekly reset times for drawdown calculations managed by the `ValuationService`.
-   **State Caching and Concurrency Control:** Uses an `asyncio.Lock` to manage concurrent access and updates to its cached portfolio state, ensuring data consistency.

## Protocols

### `ReconcilableExecutionHandler`

A protocol (informal interface) that an `ExecutionHandler` instance must conform to if it is to be used for portfolio reconciliation. It must implement the following asynchronous methods:

-   **`async get_account_balances() -> Dict[str, Dict[str, Decimal]]`**:
    -   Returns a dictionary where keys are currency symbols (e.g., "USD", "BTC").
    -   Each value is another dictionary with keys like `"total_balance"` and `"available_balance"`, holding `Decimal` values.
-   **`async get_open_positions() -> List[Dict[str, Any]]`**:
    -   Returns a list of dictionaries, each representing an open position on the exchange.
    -   Each dictionary should contain keys such as `"symbol"` (e.g., "BTC/USD"), `"quantity"` (`Decimal`), `"average_entry_price"` (`Decimal`), `"side"` (str, e.g., "LONG", "SHORT"), and potentially `"unrealized_pnl"`.

## Class `PortfolioManager`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config_manager (ConfigManager)`: An instance of `ConfigManager` for accessing application configuration.
    -   `pubsub_manager (PubSubManager)`: An instance of `PubSubManager` for event subscription.
    -   `market_price_service (MarketPriceService)`: An instance of `MarketPriceService` used by `ValuationService` to get current market prices for assets.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
    -   `execution_handler (Optional[ExecutionHandler])`: An optional instance of an `ExecutionHandler`. If provided and it conforms to `ReconcilableExecutionHandler`, reconciliation features can be enabled.
-   **Actions:**
    -   Stores references to all provided services.
    -   Initializes its core components:
        -   `_funds_manager = FundsManager(...)`
        -   `_position_manager = PositionManager(...)`
        -   `_valuation_service = ValuationService(...)`
    -   Calls `_initialize_state()` to load initial capital and positions from `config_manager`.
    -   Calls `_configure_reconciliation()` to set up reconciliation parameters.
    -   Calls `_configure_drawdown_resets()` to pass drawdown reset configurations to `ValuationService`.
    -   Initializes an `asyncio.Lock` (`_state_lock`) for managing access to cached portfolio data.
    -   Initializes `_cached_portfolio_state` dictionary.
    -   Sets up internal flags for managing the reconciliation task.

### Internal Initialization & Configuration

-   **`_initialize_state() -> None`**:
    -   Retrieves `initial_capital` (dict of currency to amount) and `initial_positions` (dict of pair to position details) from the `portfolio` section of the configuration via `config_manager`.
    -   Populates `_funds_manager` with initial capital.
    -   Populates `_position_manager` with initial positions.
    -   Triggers an initial portfolio valuation by calling `_update_portfolio_value_and_cache()`.

-   **`_configure_reconciliation() -> None`**:
    -   Loads reconciliation settings from the `portfolio.reconciliation` section of the configuration:
        -   `interval_seconds`: How often to run reconciliation.
        -   `threshold`: Permissible percentage difference before a discrepancy is flagged.
        -   `auto_update`: Boolean indicating whether to automatically adjust internal state to match exchange data.
    -   Stores these settings as internal attributes (e.g., `_reconciliation_interval_s`).

-   **`_configure_drawdown_resets() -> None`**:
    -   Loads drawdown reset configurations from `portfolio.drawdown` section:
        -   `daily_reset_hour_utc`
        -   `weekly_reset_day` (0 for Monday, 6 for Sunday)
    -   Calls methods on `_valuation_service` (e.g., `set_daily_drawdown_reset_time`, `set_weekly_drawdown_reset_time`) to configure these resets.

### Service Lifecycle

-   **`async start() -> None`**:
    -   Subscribes `_handle_execution_report` to `ExecutionReportEvent` via `PubSubManager`.
    -   If reconciliation is enabled (`_reconciliation_interval_s > 0`) and the `_execution_handler` is suitable (checked by `_execution_handler_available_for_reconciliation()`), it starts the `_run_periodic_reconciliation()` background task.
    -   Logs that the PortfolioManager service has started.

-   **`async stop() -> None`**:
    -   Unsubscribes from `ExecutionReportEvent`.
    -   If the reconciliation task is running, it signals it to stop and awaits its completion.
    -   Logs that the PortfolioManager service is stopping.

### Event Handling

-   **`async _handle_execution_report(event: ExecutionReportEvent) -> None`**:
    -   The core handler for `ExecutionReportEvent`s.
    -   Acquires `_state_lock` to ensure atomic updates.
    -   If `event.status` indicates a fill (e.g., "FILLED", "PARTIALLY_FILLED"):
        -   Calls `_parse_execution_values(event)` to extract and validate trade details (price, quantity, fees, etc.).
        -   Updates `_funds_manager` to reflect changes in currency balances due to the trade cost/proceeds and fees.
        -   Updates `_position_manager` to reflect changes in the position (new position, increased/decreased size, closed position).
        -   Calls `_update_portfolio_value_and_cache()` to re-calculate and cache the latest portfolio valuation.
        -   Calls `_log_updated_state()` to log a summary.
    -   If `event.status` indicates a cancellation (e.g., "CANCELLED", "REJECTED"):
        -   Calls `_handle_order_cancellation(event)`.
    -   Releases `_state_lock`.

-   **`_handle_order_cancellation(event: ExecutionReportEvent) -> None`**:
    -   Currently, this method primarily logs the details of the cancelled or rejected order.
    -   Future enhancements might include logic to manage reserved funds for open orders.

### State Management & Valuation

-   **`_parse_execution_values(event: ExecutionReportEvent) -> Optional[dict]`**:
    -   Extracts key numerical data from an `ExecutionReportEvent` such as `fill_price`, `fill_quantity`, `commission_amount`, `commission_currency`.
    -   Converts these values to `Decimal` for precision.
    -   Performs basic validation (e.g., ensuring values are positive).
    -   Returns a dictionary of these parsed values or `None` if validation fails.

-   **`async _update_portfolio_value_and_cache() -> None`**:
    -   This method is called after any state change (e.g., trade fill, initial load, reconciliation adjustment).
    -   It's assumed to be called while `_state_lock` is held.
    -   Calls `_valuation_service.update_portfolio_value(self._funds_manager, self._position_manager)` to get the latest portfolio valuation metrics (equity, P&L, drawdown, exposure).
    -   Updates `_cached_portfolio_state` with these new metrics, current fund balances from `_funds_manager`, open positions from `_position_manager`, and the current timestamp.
    -   This cached state is then used by `get_current_state()`.

-   **`_log_updated_state() -> None`**:
    -   Logs a summary of the current portfolio state after an update, including total equity, available funds per currency, and key details of open positions.

### State Retrieval (Public Methods)

These methods provide synchronous access to the portfolio's state, typically using the data cached in `_cached_portfolio_state`.

-   **`get_current_state() -> dict`**:
    -   Synchronously returns a comprehensive snapshot of the current portfolio.
    -   Data is sourced from `_cached_portfolio_state` (protected by `_state_lock` during updates).
    -   The returned dictionary includes:
        -   `timestamp`: Time of the last cache update.
        -   `valuation_currency`: The currency in which overall portfolio values are denominated.
        -   `total_equity`: Current total equity of the portfolio.
        -   `funds`: Dictionary of currency balances.
        -   `positions`: List of dictionaries, each representing an open position with details like `pair`, `quantity`, `average_entry_price`, `current_market_value`, `unrealized_pnl`.
        -   `market_exposure`: Total market exposure.
        -   `drawdown_metrics`: Current daily, weekly, and max drawdown percentages.
        -   Other relevant metrics from `ValuationService`.

-   **`get_available_funds(currency: str) -> Decimal`**:
    -   Returns the available funds for a specific `currency` by querying `_funds_manager.get_available_balance(currency)`.

-   **`get_current_equity() -> Decimal`**:
    -   Returns the current total portfolio equity, usually from `_cached_portfolio_state["total_equity"]`.

-   **`get_position_history(pair: str) -> List[dict]`**:
    -   Returns the history of trades for a given `pair` by querying `_position_manager.get_trade_history(pair)`.

-   **`get_open_positions() -> List[PositionInfo]`**:
    -   Returns a list of `PositionInfo` objects (or similar data structures) representing all currently open positions, by querying `_position_manager.get_all_open_positions()`.

### Reconciliation

-   **`async _run_periodic_reconciliation() -> None`**:
    -   A background task that runs periodically if reconciliation is enabled.
    -   Sleeps for `_reconciliation_interval_s`.
    -   Calls `_reconcile_with_exchange()` to perform the reconciliation logic.
    -   Handles graceful shutdown via an internal stop event.

-   **`_execution_handler_available_for_reconciliation() -> bool`**:
    -   Checks if `self._execution_handler` is not `None` and if it implements the methods defined in the `ReconcilableExecutionHandler` protocol (i.e., `get_account_balances` and `get_open_positions`).
    -   Returns `True` if reconciliation can be performed, `False` otherwise.

-   **`async _reconcile_with_exchange() -> None`**:
    -   Acquires `_state_lock`.
    -   Fetches current account balances and open positions from the exchange using `self._execution_handler.get_account_balances()` and `self._execution_handler.get_open_positions()`.
    -   Retrieves internal balances from `_funds_manager` and internal positions from `_position_manager`.
    -   Calls `_compare_balances()` and `_compare_positions()` to find discrepancies.
    -   If `_auto_reconcile_enabled` is `True` and discrepancies are found:
        -   Calls `_auto_reconcile_balances()` and `_auto_reconcile_positions()`.
    -   Logs a summary of the reconciliation process and any discrepancies or adjustments made.
    -   Releases `_state_lock`.

-   **`_compare_balances(internal_balances: dict, exchange_balances: dict) -> List[str]`**:
    -   Compares fund balances between the internal state and the exchange data.
    -   Returns a list of strings describing any discrepancies found that exceed `_reconciliation_threshold`.

-   **`_compare_positions(internal_positions: list, exchange_positions: list) -> List[str]`**:
    -   Compares open positions (quantity, and potentially average entry price) between the internal state and exchange data.
    -   Returns a list of strings describing any discrepancies.

-   **`_auto_reconcile_balances(exchange_balances: dict) -> None`**:
    -   Delegates to `_funds_manager.reconcile_with_exchange_balances(exchange_balances)` to update internal fund balances to match the exchange.

-   **`_auto_reconcile_positions(exchange_positions: list) -> None`**:
    -   Calls `_reconcile_positions_with_exchange(exchange_positions)` to adjust internal positions.

-   **`async _reconcile_positions_with_exchange(exchange_positions: List[dict]) -> None`**:
    -   The core logic for adjusting internal positions to match the exchange's state.
    -   Iterates through exchange positions:
        -   If an exchange position doesn't exist internally or its quantity differs significantly, a "reconciliation trade" is created.
        -   This might involve closing the internal position if it's not on the exchange, or adjusting its quantity and average entry price.
        -   Uses `_create_reconciliation_trade()` or `_create_position_from_exchange()` which internally call methods on `_position_manager` and `_funds_manager` to record these adjustments as if they were trades (often with zero price or special commission handling).
    -   Iterates through internal positions:
        -   If an internal position is not found on the exchange, it's typically closed internally via a reconciliation trade.
    -   After adjustments, calls `_update_portfolio_value_and_cache()`.

    -   **`_create_reconciliation_trade(...)`**: Helper to create a synthetic trade record in `PositionManager` and adjust funds in `FundsManager` to align an existing internal position with an exchange position.
    -   **`_create_position_from_exchange(...)`**: Helper to create a new internal position based entirely on exchange data, effectively opening a position via reconciliation.

### Utilities

-   **`_split_symbol(symbol: str) -> Optional[Tuple[str, str]]`**:
    -   A utility function to split a trading pair symbol string (e.g., "BTC/USD") into its base and quote currency components (e.g., `("BTC", "USD")`).
    -   Returns a tuple or `None` if the symbol format is incorrect.

## Dependencies

-   **`asyncio`**: For asynchronous operations and concurrency control (`asyncio.Lock`).
-   **`datetime`**: For timestamping.
-   **`decimal.Decimal`**: For precise financial calculations.
-   **`gal_friday.config_manager.ConfigManager`**: For accessing application configuration.
-   **`gal_friday.core.events.ExecutionReportEvent`**: The event type consumed to update portfolio state.
-   **`gal_friday.core.pubsub.PubSubManager`**: For event subscription.
-   **`gal_friday.exceptions`**: For custom application exceptions (though not explicitly listed as defined *within* this module, it might use them).
-   **`gal_friday.execution_handler.ExecutionHandler`** (Optional): The interface/concrete class for interacting with exchanges, needed for reconciliation.
-   **`gal_friday.interfaces.market_price_service_interface.MarketPriceService`**: Interface for the market price service.
-   **`gal_friday.logger_service.LoggerService`**: For structured logging.
-   **`gal_friday.portfolio.funds_manager.FundsManager`**: Sub-component for managing funds.
-   **`gal_friday.portfolio.position_manager.PositionManager`**: Sub-component for managing positions.
-   **`gal_friday.portfolio.valuation_service.ValuationService`**: Sub-component for portfolio valuation.

## Configuration (Key options from `portfolio` section of app config)

The `PortfolioManager` and its sub-components are configured via the `portfolio` section of the application's main configuration file.

-   **`valuation_currency (str)`**: The currency in which the overall portfolio value and equity are primarily reported (e.g., "USD"). Defaults to "USD".
-   **`initial_capital (Dict[str, float/Decimal])`**: A dictionary mapping currency codes to their initial amounts. Example: `{"USD": 10000.00, "BTC": 0.5}`.
-   **`initial_positions (List[Dict])`**: A list of dictionaries, each defining an initial open position. Example:
    ```json
    [
      {
        "pair": "BTC/USD",
        "quantity": 0.1,
        "average_entry_price": 50000.00,
        "side": "LONG" // or "SHORT"
      }
    ]
    ```
-   **`reconciliation` (dict, optional)**: Settings for exchange reconciliation.
    -   `interval_seconds (int)`: Frequency of reconciliation in seconds. If 0 or not present, periodic reconciliation is disabled.
    -   `threshold (float/Decimal)`: The percentage difference threshold beyond which a discrepancy in balances or positions is flagged (e.g., `0.01` for 0.01%).
    -   `auto_update (bool)`: If `True`, the PortfolioManager will attempt to automatically adjust its internal state to match the exchange's data when discrepancies are found.
-   **`drawdown` (dict, optional)**: Settings for drawdown calculation resets, passed to `ValuationService`.
    -   `daily_reset_hour_utc (int)`: The UTC hour (0-23) at which daily drawdown figures are reset.
    -   `weekly_reset_day (int)`: The day of the week (0 for Monday, 6 for Sunday) on which weekly drawdown figures are reset.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `PortfolioManager` module.
