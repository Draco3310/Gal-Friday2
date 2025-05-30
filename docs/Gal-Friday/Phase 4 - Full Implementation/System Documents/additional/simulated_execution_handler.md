# SimulatedExecutionHandler Module (`gal_friday/simulated_execution_handler.py`) Documentation

## Module Overview

The `gal_friday.simulated_execution_handler.py` module provides a simulated environment for trade execution, designed primarily for backtesting and paper trading scenarios within the Gal-Friday trading system. It implements the `ExecutionHandlerInterface` (or a similar contract) to process approved trade signals. Instead of interacting with a live exchange, it mimics order fills, slippage, and commission based on historical OHLCV data provided by a `HistoricalDataService`. This allows for the evaluation of trading strategies under simulated market conditions.

## Key Features

-   **Simulated Order Execution:**
    -   Processes `TradeSignalApprovedEvent`s.
    -   Simulates the execution of **Market Orders** against the open price of the next available historical bar, incorporating slippage.
    -   Simulates **Limit Orders**, checking if the limit price is met within the high-low range of subsequent historical bars.
-   **Slippage Modeling:** Implements configurable slippage models to simulate price differences between the expected fill price and the actual simulated fill price. Supported models can include:
    -   *Fixed Percentage:* A constant percentage of the price.
    -   *Volatility-Based:* Slippage proportional to market volatility (e.g., a fraction of Average True Range - ATR).
    -   *Market Impact (Order Size vs. Bar Volume):* Slippage increases with the order size relative to the volume of the historical bar against which it's filled.
-   **Commission Modeling:** Applies configurable taker and maker commission fees to simulated fills, accurately reflecting trading costs.
-   **Partial Fill Simulation:** Can simulate partial fills for orders based on a configurable "liquidity ratio" relative to the historical bar's volume. This mimics scenarios where an order might not be fully executable at a single price point due to available liquidity.
-   **Active Limit Order Management:**
    -   Tracks active (unfilled) limit orders.
    -   On each new historical bar, checks if these limit orders would have been filled.
    -   Handles limit order timeouts: if a limit order is not filled within a configurable duration or number of bars, it can be automatically cancelled or, optionally, converted into a market order.
-   **Stop-Loss (SL) and Take-Profit (TP) Simulation:**
    -   When an entry order is simulated as filled, associated SL and TP prices (from the original signal) are registered.
    -   **Stop-Loss:** Typically simulated as a market order that executes on the open of the bar *after* the bar in which the stop price was breached (to simulate the delay in stop order activation and execution).
    -   **Take-Profit:** Typically simulated as a limit order that executes if the take-profit price is touched or crossed within the high-low range of a bar.
-   **Event Publication:** Publishes `ExecutionReportEvent`s to the `PubSubManager` for all simulated order lifecycle events (e.g., NEW, SUBMITTED, FILLED, PARTIALLY_FILLED, CANCELED, REJECTED), mimicking the behavior of a live execution handler. This allows `PortfolioManager` and other services to update their state based on simulated trades.
-   **Historical Data Dependency:** Relies on a `HistoricalDataService` to provide historical OHLCV bars, which form the basis for all simulation logic (determining fill prices, checking if limit/SL/TP prices are met).
-   **Configurable Delays:** Can simulate order processing delays to mimic network latency or exchange processing times before an order is considered "live" or fillable.

## Internal Data Structures (Dataclasses)

The module likely uses several internal dataclasses to manage state and parameters for simulation:

-   **`PortfolioState`, `Position`**: (Note: These specific dataclasses might be placeholders or simplified internal representations if the `SimulatedExecutionHandler` does not fully replicate the main `PortfolioManager`'s state. In a typical backtesting setup, the main `PortfolioManager` would still be used and updated by execution reports from this simulator.)
-   **`CustomReportOverrides(Dataclass)`**: Allows for overriding specific fields when generating an `ExecutionReportEvent`, useful for injecting specific details during simulation.
-   **`SimulatedReportParams(Dataclass)`**: A container for parameters needed to create a simulated `ExecutionReportEvent` (e.g., client order ID, exchange order ID, status, fill details).
-   **`FillDetails(Dataclass)`**: Holds details of a simulated fill, such as `fill_price`, `fill_quantity`, `commission_paid`, `fill_timestamp`, `liquidity_type` (MAKER/TAKER).
-   **`MarketExitParams(Dataclass)`**: Used to pass parameters for simulating market order exits, particularly for stop-loss orders, including the trigger bar and exit bar details.

## Class `SimulatedExecutionHandler`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config_manager (ConfigManager)`: For accessing simulation-specific configurations (fees, slippage models, liquidity ratios, timeouts).
    -   `pubsub_manager (PubSubManager)`: For publishing `ExecutionReportEvent`s.
    -   `data_service (HistoricalDataService)`: Crucial for providing the historical OHLCV bars used to simulate fills.
    -   `logger_service (LoggerService)`: For logging simulation activities and errors.
-   **Actions:**
    -   Loads simulation configurations from `config_manager`:
        -   Taker and maker commission percentages.
        -   Slippage model type and its parameters.
        -   Assumed liquidity ratio for partial fill simulation.
        -   Limit order timeout duration and behavior (cancel or convert to market).
        -   Simulated order processing delay.
    -   Initializes internal state:
        -   `_active_limit_orders (dict)`: Stores details of limit orders that are currently active and waiting to be filled or timed out. Keyed by client order ID.
        -   `_active_sl_tp_orders (dict)`: Stores stop-loss and take-profit details for open positions. Keyed by a position identifier or client order ID of the entry trade.
        -   `_consecutive_errors (int)`: Counter for simulation errors.

### Service Lifecycle (`start`, `stop`)

-   **`async start() -> None`**:
    -   Subscribes `handle_trade_signal_approved` to `EventType.TRADE_SIGNAL_APPROVED`.
    -   Logs that the SimulatedExecutionHandler service has started.
    -   Unlike a live execution handler, it typically does not manage external connections, so `start` is often simpler.
-   **`async stop() -> None`**:
    -   Unsubscribes from events.
    -   Logs that the SimulatedExecutionHandler service is stopping.
    -   May perform cleanup of any internal state if necessary.

### Main Signal Handling (`handle_trade_signal_approved`)

-   **`async handle_trade_signal_approved(event: TradeSignalApprovedEvent) -> None`**:
    -   Receives an `TradeSignalApprovedEvent`.
    -   Validates basic order parameters from the event.
    -   Introduces a configured processing delay (`await asyncio.sleep(self._processing_delay_s)`).
    -   Fetches the "current" historical OHLCV bar (which would be the bar active at `event.timestamp + processing_delay`) from the `_data_service` (e.g., `await self._data_service.get_bar_at_timestamp(...)` or assumes the backtesting engine provides the current bar).
    -   Calls `await self._simulate_order_fill(event, current_bar)` to determine the fill outcome.
    -   If the order is filled (fully or partially):
        -   Publishes one or more `ExecutionReportEvent`s (e.g., SUBMITTED, then FILLED/PARTIALLY_FILLED).
        -   If fully filled and the original signal included SL/TP prices, calls `_register_or_update_sl_tp()` to arm these contingent orders.
    -   If it's a limit order that is not immediately filled (IOC/FOK might be handled here, or it's GTC):
        -   Adds it to `_active_limit_orders` for checking against subsequent bars.
        -   Publishes an `ExecutionReportEvent` with status "NEW" or "OPEN".
    -   Handles errors by calling `_handle_execution_error()`.

### Order Simulation Logic

-   **`async _simulate_order_fill(event: TradeSignalApprovedEvent, current_bar: pd.Series) -> Optional[FillDetails]`**:
    -   Dispatches to `_simulate_market_order()` or `_handle_limit_order_placement()` based on `event.order_type`.
    -   Returns `FillDetails` if a fill occurs, otherwise `None` (e.g., for a GTC limit order not yet filled).

-   **`async _simulate_market_order(event: TradeSignalApprovedEvent, bar: pd.Series, commission_pct: Decimal, fill_timestamp: datetime) -> FillDetails`**:
    -   **Fill Price:** Calculated as `bar['open']` (simulating execution at the start of the next bar after signal) plus calculated `slippage`.
    -   **Slippage:** Calls `_calculate_slippage(event.side, bar, event.quantity_ordered)`.
    -   **Partial Fill:**
        -   Calculates `simulated_executable_volume = bar['volume'] * self._fill_liquidity_ratio`.
        -   `filled_quantity = min(event.quantity_ordered, simulated_executable_volume)`.
    -   **Commission:** Calculates commission based on `filled_quantity`, `fill_price`, and `commission_pct`.
    -   Returns `FillDetails` object. Publishes appropriate execution reports (SUBMITTED, (PARTIALLY_)FILLED).

-   **`async _handle_limit_order_placement(event: TradeSignalApprovedEvent, bar: pd.Series, fill_timestamp: datetime) -> Optional[FillDetails]`**:
    -   Checks if the limit order would fill immediately on the `bar` using `_check_limit_order_fill_on_bar()`.
    -   If it fills immediately (e.g., marketable limit order or price touched):
        -   Fill price is typically the limit price itself or slightly better if the bar moved favorably.
        -   Determines MAKER/TAKER status using `_determine_limit_order_liquidity()`.
        -   Calculates commission based on MAKER/TAKER fee.
        -   Simulates partial fill based on volume.
        -   Returns `FillDetails`.
    -   If not immediately filled (and not IOC/FOK that would be cancelled):
        -   Adds the order details (client_order_id, limit_price, quantity, side, timeout_at) to `_active_limit_orders`.
        -   Publishes an `ExecutionReportEvent` with status "NEW" or "OPEN".
        -   Returns `None`.

-   **`_check_limit_order_fill_on_bar(side: str, limit_price: Decimal, bar: pd.Series) -> bool`**:
    -   For a BUY limit: returns `True` if `bar['low'] <= limit_price`.
    -   For a SELL limit: returns `True` if `bar['high'] >= limit_price`.
    -   Returns `False` otherwise.

-   **`_determine_limit_order_liquidity(side: str, fill_price: Decimal, bar: pd.Series) -> str`**:
    -   A heuristic to determine if a limit order fill was likely a MAKER or TAKER.
    -   Example: If a BUY limit order fills at `fill_price`, and `fill_price < bar['open']` (or some other reference like previous close), it might be considered MAKER. If `fill_price >= bar['open']`, it might be TAKER. This is a simplification.

-   **`_calculate_slippage(side: str, bar: pd.Series, order_quantity: Decimal) -> Decimal`**:
    -   Implements the configured slippage model:
        -   **Fixed Percentage:** `slippage_amount = bar['open'] * self._slippage_fixed_pct`.
        -   **Volatility-Based (ATR):** Needs ATR for the bar. `slippage_amount = atr_value * self._slippage_atr_fraction`. (Requires `HistoricalDataService` to provide ATR or data to calculate it).
        -   **Market Impact:** `slippage_factor = order_quantity / bar['volume']`. `slippage_amount = bar['open'] * slippage_factor * self._slippage_market_impact_factor`.
    -   Returns the slippage amount (positive for buys, negative for sells if it means a worse price).

### Active Order Management (typically called by the Backtesting Engine's main loop)

-   **`async check_active_limit_orders(current_bar: pd.Series, bar_timestamp: datetime) -> None`**:
    -   Iterates through a copy of `_active_limit_orders`.
    -   For each active limit order:
        -   If `_check_limit_order_fill_on_bar(order.side, order.limit_price, current_bar)` is `True`:
            -   Simulate fill (determine fill price, quantity, commission similar to `_handle_limit_order_placement`).
            -   Publish `ExecutionReportEvent`(s) for FILL/PARTIAL_FILL.
            -   If fully filled, `_register_or_update_sl_tp()` and remove from `_active_limit_orders`.
            -   If partially filled, update remaining quantity.
        -   Else if `bar_timestamp >= order.timeout_at`:
            -   Handle timeout:
                -   If configured to cancel: Publish CANCELED report, remove from `_active_limit_orders`.
                -   If configured to convert to market: Create a new market order signal/event (using details from the timed-out limit order) and simulate its fill against the *next* bar's open (or current bar's close/open). Remove from `_active_limit_orders`.
            -   Publish relevant `ExecutionReportEvent`.

-   **`async check_active_sl_tp(current_bar: pd.Series, bar_timestamp: datetime) -> None`**:
    -   Iterates through `_active_sl_tp_orders`.
    -   For each open position with SL/TP armed:
        -   **Check Stop-Loss:** If `(side == "LONG" and current_bar['low'] <= sl_price)` or `(side == "SHORT" and current_bar['high'] >= sl_price)`:
            -   Trigger SL: Call `await self._process_sl_exit(position_details, current_bar, bar_timestamp)`.
            -   Remove/update SL/TP status for the position.
        -   **Check Take-Profit (if SL not triggered):** If `(side == "LONG" and current_bar['high'] >= tp_price)` or `(side == "SHORT" and current_bar['low'] <= tp_price)`:
            -   Trigger TP: Call `await self._process_tp_exit(position_details, current_bar, bar_timestamp)`.
            -   Remove/update SL/TP status for the position.

### SL/TP Processing

-   **`_register_or_update_sl_tp(client_order_id_entry: str, signal: TradeSignalApprovedEvent, filled_quantity: Decimal, position_side: str) -> None`**:
    -   Called when an entry order is filled.
    -   Stores the `signal.stop_loss_price`, `signal.take_profit_price`, `filled_quantity`, and `position_side` in `_active_sl_tp_orders`, associated with `client_order_id_entry` or a derived position ID.

-   **`async _process_sl_exit(position_details: dict, trigger_bar: pd.Series, trigger_timestamp: datetime) -> None`**:
    -   Simulates a stop-loss being hit. SL orders are typically market orders.
    -   **Execution Bar:** Fetches the OHLCV bar *after* the `trigger_bar` from `_data_service` (as SL execution happens on the next available price).
    -   **Fill Price:** `next_bar['open']` + slippage (calculated for market order).
    -   Simulates fill quantity (can be partial based on volume of `next_bar`).
    -   Calculates commission.
    -   Publishes `ExecutionReportEvent` for the SL fill.
    -   Updates `_active_sl_tp_orders` to reflect closure or partial closure.

-   **`async _process_tp_exit(position_details: dict, trigger_bar: pd.Series, trigger_timestamp: datetime) -> None`**:
    -   Simulates a take-profit being hit. TP orders are typically limit orders.
    -   **Fill Price:** The `take_profit_price` itself (or potentially better if `trigger_bar` moved significantly past it, though simple simulation often uses TP price).
    -   **Execution Bar:** The `trigger_bar` (fill occurs on the bar TP was hit).
    -   Simulates fill quantity (can be partial based on volume of `trigger_bar` at TP price).
    -   Determines MAKER/TAKER status.
    -   Calculates commission.
    -   Publishes `ExecutionReportEvent` for the TP fill.
    -   Updates `_active_sl_tp_orders`.

### Reporting

-   **`async _publish_simulated_report(originating_event_details: dict, params: SimulatedReportParams, overrides: Optional[CustomReportOverrides] = None) -> None`**:
    -   Constructs an `ExecutionReportEvent` dictionary using details from `originating_event_details` (like original signal ID, pair), `params` (status, fill info), and any `overrides`.
    -   Assigns a new unique `event_id` and `timestamp`.
    -   Publishes the event dictionary via `self._pubsub_manager.publish(EventType.EXECUTION_REPORT, report_dict)`.

### Error Handling

-   **`async _handle_execution_error(event_data: Union[TradeSignalApprovedEvent, dict], error: Exception, context_message: str) -> None`**:
    -   Logs the error details using `_logger_service`.
    -   Publishes an `ExecutionReportEvent` with status "ERROR" or "REJECTED", including the error message.
    -   Increments `_consecutive_errors`. If it exceeds a configured threshold, publishes a `PotentialHaltTriggerEvent` to signal a critical problem in the simulation execution logic.

## Dependencies

-   **Standard Libraries:** `uuid`, `datetime`, `decimal`.
-   **Third-Party Libraries:** `pandas` (for handling OHLCV bar data).
-   **Core Application Modules:**
    -   `gal_friday.config_manager.ConfigManager`
    -   `gal_friday.core.pubsub.PubSubManager`
    -   `gal_friday.interfaces.historical_data_service_interface.HistoricalDataService` (or a concrete implementation like `KrakenHistoricalDataService` if used directly).
    -   `gal_friday.logger_service.LoggerService`
    -   `gal_friday.core.events.EventType` and specific event classes (`TradeSignalApprovedEvent`, `ExecutionReportEvent`, `PotentialHaltTriggerEvent`, `ClosePositionCommand`).

## Configuration

The `SimulatedExecutionHandler` relies on configurations from `config.yaml` (accessed via `ConfigManager`), typically under a "backtest" or "simulation" section:

-   **`commission_taker_pct (Decimal)`**: Taker commission rate.
-   **`commission_maker_pct (Decimal)`**: Maker commission rate.
-   **`slippage.model (str)`**: Type of slippage model ("fixed_percentage", "volatility_based", "market_impact").
-   **`slippage.parameters (dict)`**: Parameters specific to the chosen slippage model (e.g., `fixed_pct`, `atr_fraction`, `market_impact_factor`).
-   **`fill_liquidity_ratio (Decimal)`**: Percentage of bar volume assumed to be available for filling an order (e.g., 0.1 for 10%).
-   **`limit_order.timeout_seconds (int)`**: Duration after which an unfilled GTC limit order might be cancelled or converted.
-   **`limit_order.timeout_action (str)`**: Action on limit order timeout ("cancel" or "convert_to_market").
-   **`order_processing_delay_seconds (float)`**: Simulated delay before an order is processed.
-   Consecutive error threshold for triggering a halt.

## Adherence to Standards

The `SimulatedExecutionHandler` is a key component for **enabling realistic strategy evaluation and debugging in a controlled, offline environment**. By mimicking aspects of a live trading environment like slippage, commissions, and order fill probabilities, it allows for more robust testing of trading strategies before they are deployed to live markets. Its adherence to the same event publishing patterns (`ExecutionReportEvent`) as a live handler ensures that other system components like `PortfolioManager` can operate consistently in both simulated and live modes.
