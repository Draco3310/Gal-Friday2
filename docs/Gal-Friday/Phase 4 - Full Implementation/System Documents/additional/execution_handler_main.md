# ExecutionHandler Module (`gal_friday/execution_handler.py`) Documentation

## Module Overview

The `gal_friday/execution_handler.py` module implements the primary logic for interacting with an external trading exchange, specifically Kraken in this context, to manage the complete lifecycle of trading orders. This includes translating approved trade signals into exchange-specific order parameters, placing these orders (market, limit), monitoring their status, handling cancellations, and importantly, managing the placement of contingent orders like Stop-Loss (SL) and Take-Profit (TP) once an entry order is filled. It also handles emergency position closures and ensures robust communication with the exchange by implementing retry logic and rate limiting.

This `ExecutionHandler` is a concrete implementation, likely of an `ExecutionHandlerInterface` defined elsewhere, and focuses on the specifics of the Kraken REST API.

## Key Features

-   **Exchange Communication:** Manages all authenticated and public REST API communication with the Kraken exchange using `aiohttp` for asynchronous requests.
-   **Order Placement from Signals:** Consumes `TradeSignalApprovedEvent`s from the `PubSubManager` and translates them into parameters suitable for Kraken's `AddOrder` API endpoint.
-   **Order Type Support:**
    -   Places **market orders**.
    -   Places **limit orders** and can monitor them for a configurable timeout, cancelling if not filled.
-   **Contingent Order Management:**
    -   Automatically places **Stop-Loss** and/or **Take-Profit** orders once an initial entry order (market or limit) is confirmed as filled.
    -   These contingent orders are typically placed as stop-loss-limit or take-profit-limit orders based on the parameters derived from the original signal.
-   **Order Status Monitoring:**
    -   Actively monitors the status of open orders by periodically polling Kraken's `QueryOrders` endpoint.
    -   Detects changes in order status (e.g., NEW, OPEN, FILLED, CANCELED, EXPIRED, REJECTED).
-   **Event Publishing:** Publishes `ExecutionReportEvent`s to the `PubSubManager` for various stages of an order's lifecycle (e.g., when a new order is acknowledged by the exchange, when it's partially or fully filled, when it's cancelled, or if it's rejected). This keeps the rest of the system (especially `PortfolioManager`) informed.
-   **Resilient API Interaction:**
    -   Implements retry logic for API requests that might fail due to transient network issues or temporary exchange errors.
    -   Utilizes a `RateLimitTracker` to manage API call frequency, respecting Kraken's rate limits to avoid being blocked.
-   **Emergency Operations:** Handles `ClosePositionCommand` events (e.g., triggered by `MonitoringService` during a system HALT or by manual CLI command) to close specified positions immediately using market orders, potentially bypassing normal rate limits for critical actions.
-   **Exchange Information Aware:** Loads and uses exchange-specific trading pair information at startup (e.g., price precision, volume precision, minimum order sizes) to correctly format order parameters.

## Internal Helper Classes

-   **`InvalidAPICredentialFormatError(ValueError)`**: Custom exception raised if the format of API credentials (key/secret) is incorrect.
-   **`ContingentOrderParamsRequest(Dataclass)`**: A data structure used to pass parameters needed for preparing contingent (SL/TP) orders. Contains fields like `trading_pair`, `side` (opposite of entry), `order_type` (e.g., "stop-loss-limit"), `quantity`, `price` (stop price), `limit_price` (limit component of SL/TP-limit), and `original_signal_id`.
-   **`OrderStatusReportParameters(Dataclass)`**: A data structure for packaging information needed when creating and publishing an `ExecutionReportEvent`, such as `client_order_id`, `exchange_order_id`, `status`, `filled_quantity`, `average_fill_price`, etc.
-   **`RateLimitTracker`**:
    -   Manages API call rates for different endpoints or categories of endpoints.
    -   Tracks the number of calls made within specific time windows.
    -   Provides a method (e.g., `wait_for_slot_async()`) that an API calling function can `await` to ensure it doesn't violate exchange rate limits.

## Class `ExecutionHandler`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config_manager (ConfigManager)`: For accessing API URLs, rate limit settings, order timeouts, and other configurations.
    -   `pubsub_manager (PubSubManager)`: For publishing `ExecutionReportEvent`s and subscribing to commands.
    -   `monitoring_service (MonitoringService)`: To check system HALT status before placing new orders.
    -   `logger_service (LoggerService)`: For logging all activities and errors.
    -   `event_store (Optional[EventStore])`: Optional service for persisting events, including execution reports.
-   **Actions:**
    -   Loads Kraken API credentials (key and secret) securely, typically via `ConfigManager` which might use `SecretsManager`. Validates credential format.
    -   Retrieves Kraken API base URL and other relevant paths from configuration.
    -   Initializes an `aiohttp.ClientSession` (though actual creation might be deferred to `start()`).
    -   Initializes `RateLimitTracker` with configured limits.
    -   Initializes internal data structures:
        -   `_order_status_monitoring_tasks (dict)`: To keep track of asyncio tasks monitoring individual orders.
        -   `_client_to_exchange_order_id_map (dict)`: Maps internal client order IDs to exchange order IDs.
        -   `_exchange_pair_info (dict)`: To store loaded trading pair details (precision, min sizes).
        -   `_pending_contingent_orders (dict)`: Potentially to track if SL/TP orders are pending for a filled entry.

### Service Lifecycle (`start`, `stop`)

-   **`async start() -> None`**:
    -   Creates the `aiohttp.ClientSession` for making HTTP requests.
    -   Calls `await self._load_exchange_info()` to fetch and store trading pair details from Kraken.
    -   Subscribes `handle_trade_signal_approved` to `EventType.TRADE_SIGNAL_APPROVED`.
    -   Subscribes `handle_close_position_command` to `EventType.CLOSE_POSITION_COMMAND`.
    -   Logs that the ExecutionHandler service has started.

-   **`async stop() -> None`**:
    -   Unsubscribes from all events.
    -   Cancels all ongoing order monitoring tasks in `_order_status_monitoring_tasks`. This involves iterating through the tasks, calling `task.cancel()`, and awaiting their completion with error handling.
    -   Closes the `aiohttp.ClientSession` (`await self._http_session.close()`).
    -   Logs that the ExecutionHandler service is stopping.

### Core Order Processing Workflow (`handle_trade_signal_approved`)

-   **`async handle_trade_signal_approved(event: TradeSignalApprovedEvent) -> None`**:
    -   **1. Check HALT Status:** Verifies `not self._monitoring_service.is_halted()`. If halted, logs and rejects the signal by publishing an error execution report.
    -   **2. Translate Signal:** Calls `_translate_signal_to_kraken_params(event)` to convert the generic `TradeSignalApprovedEvent` into a dictionary of parameters suitable for Kraken's `AddOrder` API endpoint. This includes formatting prices and quantities to Kraken's required precision.
    -   **3. Generate Client Order ID:** Creates a unique client order ID (`cl_ord_id`, typically a UUID) for internal tracking.
    -   **4. Place Order:** Calls `await self._make_private_request_with_retry("AddOrder", kraken_params)` to submit the order to Kraken.
    -   **5. Handle Response:** Passes the API response to `_handle_add_order_response(response, cl_ord_id, event)`.
        -   If successful (order accepted by Kraken):
            -   Extracts the `exchange_order_id` from the response.
            -   Maps `cl_ord_id` to `exchange_order_id`.
            -   Publishes an initial `ExecutionReportEvent` with status "NEW" or "PENDING_NEW".
            -   Calls `_start_order_monitoring(cl_ord_id, exchange_order_id, event)` to begin polling for status updates.
        -   If failed (order rejected by Kraken or API error):
            -   Publishes an `ExecutionReportEvent` with status "REJECTED" and the error reason.

### Order Monitoring & Management

-   **`_start_order_monitoring(cl_ord_id: str, kraken_order_id: str, originating_event: TradeSignalApprovedEvent) -> None`**:
    -   Spawns an asyncio task for `_monitor_order_status(kraken_order_id, cl_ord_id, originating_event.signal_id)`.
    -   Stores this task in `_order_status_monitoring_tasks` associated with `kraken_order_id`.
    -   If the originating order is a limit order, it may also spawn a task for `_monitor_limit_order_timeout` based on configured timeouts.

-   **`async _monitor_order_status(exchange_order_id: str, client_order_id: str, signal_id: str) -> None`**:
    -   Enters a loop that periodically (e.g., every few seconds, configurable) calls Kraken's `QueryOrders` endpoint (via `_make_private_request_with_retry`) with the `exchange_order_id`.
    -   Parses the response to get the current status, filled quantity, average fill price, etc.
    -   If the status changes (e.g., from OPEN to PARTIALLY_FILLED or FILLED) or if there are new fills:
        -   Publishes an updated `ExecutionReportEvent`.
        -   If the order is fully FILLED:
            -   Calls `await self._handle_sl_tp_orders(originating_event_details_associated_with_this_order, exchange_order_id, total_filled_quantity)` to place SL/TP orders.
            -   The monitoring loop for this order terminates.
        -   If the order is CANCELED, REJECTED, or EXPIRED, publishes the final status and terminates the loop.
    -   Includes error handling for API call failures during polling.
    -   Removes its task from `_order_status_monitoring_tasks` upon completion.

-   **`async _monitor_limit_order_timeout(exchange_order_id: str, client_order_id: str, timeout_seconds: int) -> None`**:
    -   Waits for `timeout_seconds`.
    -   After the timeout, checks if the order (identified by `exchange_order_id`) is still open (not fully filled or cancelled). This might involve querying its status one last time or checking internal state.
    -   If it's still open, calls `await self.cancel_order(exchange_order_id, client_order_id)` to attempt cancellation due to timeout.
    -   Logs the timeout cancellation attempt.

-   **`async cancel_order(exchange_order_id: str, client_order_id: str) -> bool`**:
    -   Constructs parameters for Kraken's `CancelOrder` endpoint using `txid=exchange_order_id`.
    -   Calls `await self._make_private_request_with_retry("CancelOrder", params)`.
    -   Handles the response:
        -   If cancellation is successful or pending, publishes an `ExecutionReportEvent` with status "PENDING_CANCEL" or "CANCELED".
        -   Returns `True` on success, `False` on failure, logging errors.

### Contingent Order Handling

-   **`async _handle_sl_tp_orders(originating_event: TradeSignalApprovedEvent, filled_entry_order_id: str, filled_quantity: Decimal) -> None`**:
    -   Called after an entry order (from `originating_event`) is confirmed as fully filled.
    -   Extracts SL and TP prices from `originating_event.stop_loss_price` and `originating_event.take_profit_price`.
    -   For each contingent order (SL and/or TP) that has a valid price:
        -   Constructs a `ContingentOrderParamsRequest` dataclass instance. The side will be opposite to the entry order. Quantity will match `filled_quantity`. Order type will be appropriate (e.g., "stop-loss-limit", "take-profit-limit").
        -   Calls `_prepare_contingent_order_params()` to get Kraken-specific parameters.
        -   Places the contingent order using `await self._make_private_request_with_retry("AddOrder", ...)`.
        -   Handles the response similarly to the entry order (`_handle_add_order_response`), including publishing "NEW" execution reports and starting monitoring for these new SL/TP orders.

-   **`_prepare_contingent_order_params(request: ContingentOrderParamsRequest) -> dict`**:
    -   Takes the generic `ContingentOrderParamsRequest`.
    -   Translates it into Kraken-specific API parameters for placing stop-loss or take-profit orders. This involves setting fields like `ordertype`, `price` (stop price), `price2` (limit price for limit-based contingent orders), `pair`, `type` (buy/sell), `volume`.
    -   Uses `_exchange_pair_info` for correct price/volume formatting.

### Emergency Operations

-   **`async handle_close_position_command(event: ClosePositionCommand) -> None`**:
    -   Receives a `ClosePositionCommand` (e.g., from `MonitoringService` during a global HALT or from `CLIService`).
    -   Determines the `trading_pair`, current `position_quantity`, and `side` (long/short) from the event or by querying `PortfolioManager`.
    -   Constructs parameters for a market order to close the position (opposite side, full quantity).
    -   Places this market order using `await self._make_private_request_with_retry("AddOrder", ...)`, potentially with a flag to bypass normal rate limits if the situation is critical and the rate limiter supports it.
    -   Publishes execution reports for this closure order.

### API Interaction & Utilities

-   **`async _load_exchange_info() -> None`**:
    -   Makes a public API request to Kraken's `AssetPairs` endpoint (or similar) at startup.
    -   Parses the response to get details for all tradable pairs, including:
        -   `pair_decimals` (price precision).
        -   `lot_decimals` (quantity precision).
        -   `ordermin` (minimum order size).
        -   Other relevant limits or properties.
    -   Stores this information in `self._exchange_pair_info` for use in formatting order parameters.

-   **`async _make_public_request_with_retry(endpoint_path: str, params: Optional[dict] = None, max_retries: int = 3) -> dict`**:
    -   Handles making GET requests to public Kraken API endpoints.
    -   Includes retry logic (e.g., exponential backoff) for `max_retries` attempts if the request fails due to network issues or temporary API errors.
    -   Uses `RateLimitTracker` before making the call.

-   **`async _make_private_request(uri_path: str, data: dict, timeout_seconds: Optional[int] = None) -> dict`**:
    -   The core method for making authenticated POST requests to private Kraken API endpoints.
    -   Adds a nonce to `data` using a helper from `gal_friday.utils.kraken_api`.
    -   Generates the API signature using `generate_kraken_signature` (from `gal_friday.utils.kraken_api`) with the URI path, nonce-augmented data, and the API secret.
    -   Constructs headers including `API-Key` and `API-Sign`.
    -   Makes the POST request using `self._http_session`.
    -   Parses the JSON response, checks for Kraken API errors (in the `error` field of the response), and raises specific exceptions (e.g., `KrakenAPIError`) if errors are present.
    -   Returns the `result` part of the JSON response if successful.

-   **`async _make_private_request_with_retry(uri_path: str, data: dict, max_retries: int = 3) -> dict`**:
    -   Wraps `_make_private_request` with retry logic and rate limiting via `RateLimitTracker`.
    -   Handles specific API errors that might be retryable.

-   **`_translate_signal_to_kraken_params(event: TradeSignalApprovedEvent) -> dict`**:
    -   Takes a `TradeSignalApprovedEvent` from the system's internal format.
    -   Maps its fields (trading pair, side, order type, quantity, limit price, stop price) to the corresponding parameter names and formats required by Kraken's `AddOrder` API endpoint.
    -   Uses `_format_decimal` and information from `_exchange_pair_info` to ensure prices and quantities are formatted to the correct decimal precision for the specific trading pair on Kraken.

-   **`_format_decimal(value: Optional[Decimal], precision: int) -> Optional[str]`**:
    -   Formats a `Decimal` value to a string with a specified number of `precision` decimal places. Handles `None` inputs.

-   **`_publish_error_execution_report(client_order_id: str, signal_id: str, trading_pair: str, error_message: str, exchange_order_id: Optional[str] = None) -> None`**: Helper to create and publish an `ExecutionReportEvent` with status "ERROR" or "REJECTED".
-   **`_publish_status_execution_report(params: OrderStatusReportParameters)`**: Helper to create and publish an `ExecutionReportEvent` for various statuses like "NEW", "FILLED", "CANCELED".

## Dependencies

-   **Standard Libraries:** `asyncio`, `secrets` (for `token_urlsafe`), `time`, `uuid`, `decimal`, `hashlib`, `hmac`, `urllib.parse`.
-   **Third-Party Libraries:** `aiohttp` (for asynchronous HTTP requests).
-   **Core Application Modules:**
    -   `gal_friday.config_manager.ConfigManager`
    -   `gal_friday.core.pubsub.PubSubManager`
    -   `gal_friday.monitoring_service.MonitoringService`
    -   `gal_friday.logger_service.LoggerService`
    -   `gal_friday.event_store.EventStore` (Optional)
    -   `gal_friday.core.events.EventType` and specific event classes (`TradeSignalApprovedEvent`, `ExecutionReportEvent`, `ClosePositionCommand`).
    -   Utilities from `gal_friday.utils.kraken_api` (e.g., `generate_kraken_signature`).
    -   Potentially `gal_friday.interfaces.ExecutionHandlerInterface` (if explicitly implemented).

## Configuration

The `ExecutionHandler` relies on configurations from `config.yaml` (accessed via `ConfigManager`) for:
-   Kraken API base URL (e.g., `kraken.api_url`).
-   Kraken API key and secret (expected to be managed securely, e.g., by `SecretsManager` and accessed via `ConfigManager`).
-   Rate limiting parameters (calls per second/minute for various endpoint categories).
-   Order monitoring intervals and timeouts for limit orders.
-   Default leverage, user reference IDs for orders.

## Adherence to Standards

This `ExecutionHandler` module is crucial for reliable trade execution. By encapsulating all Kraken-specific API interaction logic, it:
-   Provides a clear separation between the core trading logic of Gal-Friday and the specifics of a particular exchange.
-   Enhances robustness through retry mechanisms, rate limiting, and detailed error handling.
-   Ensures that all trading actions and their outcomes are properly reported back to the system via standardized `ExecutionReportEvent`s.
This modular and resilient design is key to building a dependable automated trading system.
