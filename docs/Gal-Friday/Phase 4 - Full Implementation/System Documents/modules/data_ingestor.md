# DataIngestor Module Documentation

## Module Overview

The `gal_friday.data_ingestor.py` module is responsible for ingesting real-time market data from the Kraken WebSocket API v2. It connects to the WebSocket, subscribes to various data streams (L2 Order Book, OHLCV, Trades), processes the incoming data, maintains a local L2 order book state, performs validations including checksums, and publishes standardized market data events to the `PubSubManager` for consumption by other application modules. It also handles connection management, including robust reconnection logic and liveness monitoring.

## Key Features

-   **Kraken WebSocket API v2 Integration:** Connects to and interacts with Kraken's WebSocket API v2.
-   **Multiple Data Stream Subscription:** Capable of subscribing to and processing:
    -   Level 2 Order Book (channel `book`) data for specified trading pairs.
    -   OHLCV (Open, High, Low, Close, Volume) data for specified trading pairs and intervals (channel `ohlc`).
    -   Trade data for specified trading pairs (channel `trade`).
-   **Message Parsing and Validation:** Parses incoming JSON messages and validates their structure and content against expected formats.
-   **Local L2 Order Book Management:**
    -   Maintains an in-memory representation of the L2 order book for each subscribed trading pair using `sortedcontainers.SortedDict` for efficient sorted access.
    -   Processes initial snapshots and subsequent updates to keep the local book synchronized.
    -   Performs checksum validation (CRC32) against checksums provided by Kraken to ensure data integrity.
-   **Standardized Event Publishing:** Publishes market data and system status as standardized events via `PubSubManager`:
    -   `MarketDataL2Event`: For L2 order book updates.
    -   `MarketDataOHLCVEvent`: For OHLCV data updates. (Note: Current implementation primarily validates incoming OHLCV data; full processing and event publishing logic may be partial or require further development, as detailed in the relevant sections).
    -   `SystemStateEvent`: To communicate connection status changes (e.g., connected, disconnected).
    -   `PotentialHaltTriggerEvent`: Published when critical errors or excessive checksum mismatches occur, signaling potential issues requiring attention or system halt.
-   **Robust Reconnection Logic:** Implements an exponential backoff strategy for reconnecting to the WebSocket API in case of disconnections or connection failures.
-   **Connection Liveness Monitoring:** Actively monitors the connection through:
    -   A general message timeout (expecting any message within a configurable interval).
    -   A specific heartbeat timeout (expecting Kraken's heartbeat messages).
-   **Configurable Parameters:** Many operational parameters are configurable via `ConfigManager`, including WebSocket URL, connection timeouts, reconnection parameters, L2 book depth, and OHLC intervals.
-   **Detailed Logging:** Utilizes `LoggerService` for comprehensive logging of operations, errors, and state changes.

## Data Classes (Internal Payloads)

The module defines internal data structures for parsing messages, but event creation often happens directly. These are primarily for internal representation:
-   `MarketDataL2Payload`: Potentially for initial L2 data parsing (currently `MarketDataL2Event` is created more directly).
-   `MarketDataOHLCVPayload`: Potentially for initial OHLCV data parsing (currently `MarketDataOHLCVEvent` is created more directly).
-   `SystemStatusPayload`: Potentially for initial status message parsing (currently `SystemStateEvent` is created more directly).
*These internal payload classes are not deeply documented here as their usage is abstracted by the event publishing mechanism.*

## Class `DataIngestor`

### Constants

-   **`INTERVAL_MAP: Dict[int, str]`**: Maps Kraken's integer representation of OHLC intervals to human-readable strings (e.g., `1: "1m"`, `60: "1h"`).
-   **`INTERVAL_INT_MAP: Dict[str, int]`**: The reverse of `INTERVAL_MAP`, mapping readable strings back to Kraken's integer intervals (e.g., `"1m": 1`).

### Initialization (`__init__`)

-   **Parameters:**
    -   `config (ConfigManager)`: An instance of `ConfigManager` for accessing application configuration.
    -   `pubsub_manager (PubSubManager)`: An instance of `PubSubManager` for publishing events.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
-   **Actions:**
    -   Stores the provided `config`, `pubsub_manager`, and `logger_service` instances.
    -   Initializes internal state variables:
        -   WebSocket connection object (`_ws_connection`).
        -   Connection status flags (`_is_connected`, `_is_connecting`, `_stop_requested`).
        -   L2 order books: `_order_books` (a dictionary mapping trading pairs to `SortedDict` for bids and asks).
        -   Last message and heartbeat timestamps for liveness monitoring.
        -   Subscription management dictionaries (`_active_subscriptions`, `_pending_subscriptions`).
        -   Error counters for reconnection and checksum mismatches.
        -   Asyncio tasks for background operations (`_listen_task`, `_liveness_monitor_task`).
    -   Calls `_load_configuration()` to fetch settings.
    -   Calls `_validate_initial_config()` to check critical configurations.

### Configuration Methods

-   **`_load_configuration()`**:
    -   Loads various parameters from `ConfigManager`, such as:
        -   `data_ingestor.kraken_ws_url`
        -   `data_ingestor.connection_timeout_s`
        -   `data_ingestor.max_heartbeat_interval_s`
        -   Reconnection parameters (`reconnect_delay_s`, `max_reconnect_attempts`, `max_reconnect_delay_s`)
        -   `data_ingestor.book_depth`
        -   `data_ingestor.ohlc_intervals` (list of strings like "1m", "1h")
        -   `trading.pairs` (list of trading pairs like "BTC/USD")
        -   Checksum failure thresholds.
    -   Sets default values if configurations are not found.

-   **`_validate_initial_config()`**:
    -   Validates that essential configurations are present and correctly formatted.
    -   Checks `_trading_pairs`, `_book_depth`, and `_ohlc_intervals_numeric`.
    -   Raises `ValueError` if critical configurations are missing or invalid, preventing startup.

### Connection Management

-   **`async start()`**:
    -   The main public method to initiate and manage the data ingestion process.
    -   Enters a loop that attempts to establish a connection, listen for messages, and handle reconnections.
    -   Continues looping until `stop()` is called.

-   **`async stop()`**:
    -   Public method to gracefully stop the data ingestor.
    -   Sets `_stop_requested` flag.
    -   Cancels background tasks (`_listen_task`, `_liveness_monitor_task`).
    -   Calls `_cleanup_connection()` to close the WebSocket.

-   **`async _establish_connection() -> bool`**:
    -   Attempts to connect to the Kraken WebSocket URL (`_kraken_ws_url`).
    -   Uses `websockets.connect()` with configured timeouts.
    -   If successful, sets `_is_connected` to `True`, resets reconnection attempts, and returns `True`.
    -   Handles connection exceptions, logs errors, and returns `False` on failure.

-   **`async _setup_connection(subscription_msg: str)`**:
    -   Called after a WebSocket connection is established.
    -   Sends the initial `subscription_msg` to Kraken.
    -   Starts the `_monitor_connection_liveness_loop()` as a background task.
    -   Publishes a `SystemStateEvent` indicating a "CONNECTED" status.

-   **`async _cleanup_connection()`**:
    -   Closes the WebSocket connection (`_ws_connection.close()`).
    -   Sets `_is_connected` to `False`.
    -   Cancels and awaits the liveness monitor task if it's running.
    -   Publishes a `SystemStateEvent` indicating a "DISCONNECTED" status.

-   **`async _reconnect_with_backoff()`**:
    -   Implements the reconnection strategy.
    -   Increments reconnection attempt counter.
    -   If max attempts are exceeded, logs a critical error, publishes `PotentialHaltTriggerEvent`, and stops further attempts for a longer period.
    -   Calculates a delay using exponential backoff (capped at `_max_reconnect_delay_s`).
    -   Waits for the calculated delay before the next connection attempt.

-   **`async _monitor_connection_liveness_loop()`**:
    -   A background task that runs while connected.
    -   Periodically checks:
        -   If any message has been received within `_max_message_interval_s`.
        -   If a Kraken heartbeat message has been received within `_max_heartbeat_interval_s`.
    -   If timeouts occur, logs an error, triggers `_cleanup_connection()`, and breaks the loop, which will lead to a reconnection attempt by the main `start()` loop.

### Subscription Management

-   **`_build_subscription_message() -> str`**:
    -   Constructs the JSON message required by Kraken to subscribe to data streams.
    -   Includes `method: "subscribe"`, `params` detailing channels (`book`, `ohlc`, `trade`), trading pairs, book depth, and OHLC intervals.
    -   Uses `req_id` for tracking subscription responses.

### Message Handling

-   **`async _message_listen_loop()`**:
    -   The primary loop for receiving messages from the WebSocket.
    -   Continuously awaits messages from `_ws_connection`.
    -   Updates `_last_message_received_ts`.
    -   Calls `_process_message()` for each received message.
    -   Handles `websockets.exceptions.ConnectionClosed` and other exceptions, triggering cleanup and reconnection logic.

-   **`async _process_message(message_json: str)`**:
    -   Parses the incoming `message_json` string into a Python dictionary.
    -   Identifies message type based on keys:
        -   `"method"`: Indicates a method response message (e.g., subscription acknowledgement).
        -   `"channel"`: Indicates a data message from a subscribed channel.
    -   Routes to `_handle_method_message()` or `_handle_channel_message()` accordingly.
    -   Handles JSON parsing errors.

-   **`async _handle_method_message(data: dict)`**:
    -   Processes messages that are responses to client requests (e.g., subscribe, ping).
    -   If `method == "subscribe"`, calls `_handle_subscribe_ack()`.
    -   Handles other method types like `pong` or error responses from Kraken.

-   **`async _handle_channel_message(data: dict)`**:
    -   Processes data messages from subscribed channels.
    -   Routes based on `data["channel"]`:
        -   `"status"`: Calls `_handle_status_update()`.
        -   `"heartbeat"`: Updates `_last_heartbeat_received_ts`.
        -   `"book"`: Calls `_handle_book_data()`.
        -   `"ohlc"`: Calls `_handle_ohlc_data()`.
        -   `"trade"`: **Currently, this also calls `_handle_book_data(data)`. This is likely a placeholder or an error. The documentation should reflect this accurately.** It implies trade messages are being processed as if they are book data, which would be incorrect.
    -   Logs unknown channel messages.

-   **`async _handle_subscribe_ack(data: dict)`**:
    -   Processes the acknowledgement from Kraken for a subscription request.
    -   Updates `_active_subscriptions` based on success or failure.
    -   Logs subscription status for each pair and channel.
    -   If a subscription fails, it might trigger a halt or specific error handling.

-   **`async _handle_status_update(data: dict)`**:
    -   Processes connection status messages sent by Kraken (e.g., "online", "maintenance").
    -   Publishes a `SystemStateEvent` with the reported status.
    -   May trigger reconnection or halt if a critical status like "error" or "maintenance" is received.

### L2 Order Book Processing

-   **`async _handle_book_data(data: dict)`**:
    -   The main handler for L2 order book messages (`channel == "book"`).
    -   Validates the message structure using `_validate_book_message()`.
    -   Iterates through items in `data["data"]` (each corresponding to a trading pair).
    -   Calls `_process_book_item()` for each book data item.

-   **`_validate_book_message(data: dict) -> bool`**:
    -   Validates the overall structure of a book message (presence of "channel", "data", "type").
    -   Returns `True` if valid, `False` otherwise.

-   **`_validate_book_item(book_item: dict) -> bool`**:
    -   Validates an individual book data entry within `data["data"]`.
    -   Checks for "symbol", "checksum", and either "bids"/"asks" (for snapshots) or "updates" (for incremental updates).
    -   Returns `True` if valid, `False` otherwise.

-   **`async _process_book_item(book_item: dict, message_type: str)`**:
    -   Orchestrates the processing of a single book item for a trading pair.
    -   Determines if it's a snapshot (`message_type == "snapshot"`) or an update.
    -   Calls `_apply_book_snapshot()` or `_apply_book_update()`.
    -   Calls `_truncate_book_to_depth()`.
    -   Calls `_validate_and_update_checksum()`.
    -   If checksum is valid and data has changed, calls `_publish_book_event()`.

-   **`_apply_book_snapshot(pair: str, bids_data: list, asks_data: list)`**:
    -   Clears the existing order book for the `pair`.
    -   Populates bids and asks from the snapshot data into `_order_books[pair]["bids"]` and `_order_books[pair]["asks"]`.
    -   Prices are stored as `Decimal`, amounts as `Decimal`.

-   **`_apply_book_update(pair: str, updates_data: list)`**:
    -   Applies incremental updates to the order book for the `pair`.
    -   Iterates through `updates_data`, each entry containing "price", "qty", and "side".
    -   Calls `_update_price_levels()` to modify the appropriate side (bids or asks).

-   **`_update_price_levels(levels: SortedDict, price: Decimal, quantity: Decimal)`**:
    -   Helper function to update price levels in a `SortedDict` (either bids or asks).
    -   If `quantity` is zero, the price level is removed.
    -   Otherwise, the price level is updated with the new quantity.

-   **`_truncate_book_to_depth(pair: str)`**:
    -   Ensures the local order book for the `pair` does not exceed the configured `_book_depth`.
    -   For bids, removes lowest prices if depth is exceeded.
    -   For asks, removes highest prices if depth is exceeded.

-   **`_calculate_book_checksum(book_state: dict) -> str`**:
    -   Calculates a CRC32 checksum for the current state of an order book (`book_state`).
    -   The specific algorithm should match Kraken's requirements (typically involves concatenating price and quantity strings for top N levels and then CRC32).
    -   **Note:** The actual implementation details of Kraken's checksum algorithm are critical here and must be correctly implemented. This documentation assumes a `crc32` function is used appropriately.

-   **`async _validate_and_update_checksum(pair: str, received_checksum: str) -> bool`**:
    -   Calculates the checksum of the local order book state for `pair` using `_calculate_book_checksum()`.
    -   Compares it with the `received_checksum` from Kraken.
    -   If they match, returns `True`.
    -   If they don't match, calls `_handle_checksum_mismatch()` and returns `False`.

-   **`async _handle_checksum_mismatch(pair: str, local_checksum: str, remote_checksum: str)`**:
    -   Logs a warning or error about the checksum mismatch.
    -   Increments a checksum error counter for the `pair`.
    -   If the mismatch count exceeds a threshold (`_max_checksum_mismatches_before_resub`), it may trigger a resubscription or system halt by calling `_trigger_halt_if_needed()`.
    -   May request a new snapshot by re-subscribing.

-   **`_handle_no_updates_case(pair: str, original_checksum: str, new_checksum: str)`**:
    -   Addresses a specific scenario where Kraken sends a new checksum indicating a change, but the update message itself contains no data that alters the local book state (or the updates result in the same state).
    -   Logs this specific condition.

-   **`async _publish_book_event(pair: str, timestamp: str)`**:
    -   Creates a `MarketDataL2Event` using the current state of the order book for `pair` from `_order_books`.
    -   Populates the event with bids, asks, pair, timestamp, and source.
    -   Publishes the event using `_pubsub_manager.publish()`.

### OHLCV Data Processing

-   **`async _handle_ohlc_data(data: dict)`**:
    -   Handler for incoming OHLC messages (`channel == "ohlc"`).
    -   Iterates through `data["data"]`, where each item is an OHLC payload for a specific pair and interval.
    -   Calls `_validate_ohlc_item()` for each OHLC data entry.
    -   **Note:** The current implementation in typical snippets focuses on validation. If full processing (e.g., creating `MarketDataOHLCVEvent` and publishing it) is intended, this method would need to extract OHLC fields (open, high, low, close, volume, vwap, count, interval, pair, timestamp) and publish the event. **This documentation reflects that the primary action shown is validation, with event publishing being an implicit next step if data is valid.**

-   **`_validate_ohlc_item(ohlc_item: list, pair: str) -> bool`**:
    -   Validates the structure and types of data within an OHLC data list.
    -   Checks for correct number of elements and types (e.g., timestamp, open, high, low, close, vwap, volume, count).
    -   Returns `True` if valid, `False` otherwise.
    -   If valid, and if event publishing is implemented, a `MarketDataOHLCVEvent` would be created and published here or in `_handle_ohlc_data`.

### Trade Data Processing

-   **Note on Trade Handling:** The provided code structure indicates that `_handle_channel_message` routes messages with `channel == "trade"` to `await self._handle_book_data(data)`.
    -   **This is highly likely an error or a temporary placeholder.** Trade data has a different structure and purpose than L2 book data.
    -   **Documentation must state this clearly:** The system currently attempts to process trade messages using L2 order book logic, which is incorrect. A dedicated `_handle_trade_data(data)` method and `MarketDataTradeEvent` would be expected for proper trade processing. Users of this documentation should be aware that trade data is not correctly processed or published in its distinct form under the current described routing.

### Error Handling & Halt System

-   **`async _handle_connection_error(error: Exception, context: str)`**:
    -   Logs connection-related errors with context (e.g., "establishing connection", "listening for messages").
    -   May call `_trigger_halt_if_needed()` if the error is severe or repeated.

-   **`async _trigger_halt_if_needed(error_message: str, context: str)`**:
    -   Increments a general critical error counter.
    -   If the error counter exceeds a configured threshold (`_critical_error_threshold_before_halt`), it publishes a `PotentialHaltTriggerEvent` to the `PubSubManager`.
    -   This event signals other parts of the system that a persistent critical issue exists in data ingestion, potentially requiring a system-wide operational halt.
    -   Logs the triggering event.

## Event Publishing

The `DataIngestor` publishes the following events via `PubSubManager`:

-   **`MarketDataL2Event`**:
    -   Published after a valid L2 order book update (snapshot or incremental) is processed and its checksum is validated.
    -   Contains: `pair`, `timestamp`, `source` ("kraken_ws_v2"), `bids` (list of `[price, quantity]`), `asks` (list of `[price, quantity]`).
-   **`SystemStateEvent`**:
    -   Published when the connection status to Kraken WebSocket changes (e.g., "CONNECTED", "DISCONNECTED", "ERROR", or statuses reported by Kraken like "online", "maintenance").
    -   Contains: `timestamp`, `service_name` ("DataIngestor"), `status_code`, `status_message`.
-   **`PotentialHaltTriggerEvent`**:
    -   Published when critical, unrecoverable, or persistent errors occur, such as:
        -   Exceeding maximum reconnection attempts.
        -   Exceeding maximum checksum mismatches for a pair.
        -   Other critical internal errors exceeding a threshold.
    -   Contains: `timestamp`, `source_module` ("DataIngestor"), `error_code`, `reason`, `details`.
-   **`MarketDataOHLCVEvent`**:
    -   This event is intended for publishing processed OHLCV data.
    -   **Current Implementation Note:** While `_handle_ohlc_data` and `_validate_ohlc_item` exist, the direct creation and publishing of `MarketDataOHLCVEvent` might be conditional on validation success and may require further explicit implementation details in the codebase beyond validation. The documentation assumes that if validation passes, an event with fields like `pair`, `interval`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `vwap`, `count`, `source` would be published.

## Dependencies

-   **`asyncio`**: For asynchronous operations.
-   **`json`**: For parsing JSON messages.
-   **`logging`**: For application logging (via `LoggerService`).
-   **`uuid`**: For generating unique identifiers (e.g., for events).
-   **`datetime`**: For timestamping.
-   **`websockets`**: The library used for WebSocket client connections.
-   **`sortedcontainers.SortedDict`**: For maintaining sorted L2 order books.
-   **`zlib.crc32`** (implicitly, for checksum calculation).
-   **`gal_friday.config_manager.ConfigManager`**: For configuration management.
-   **`gal_friday.core.pubsub.PubSubManager`**: For publishing events.
-   **`gal_friday.logger_service.LoggerService`**: For structured logging.
-   **`gal_friday.core.events`**: Contains definitions for `MarketDataL2Event`, `MarketDataOHLCVEvent`, `SystemStateEvent`, `PotentialHaltTriggerEvent`.

## Configuration (Key options from `ConfigManager`)

The following configuration keys (typically under a `data_ingestor` section) are important for the `DataIngestor`:

-   `data_ingestor.kraken_ws_url` (str): The URL of the Kraken WebSocket API v2.
-   `data_ingestor.connection_timeout_s` (float): Timeout for establishing the WebSocket connection.
-   `data_ingestor.max_message_interval_s` (float): Max time without any message before considering connection stale.
-   `data_ingestor.max_heartbeat_interval_s` (float): Max time without a Kraken heartbeat before considering connection stale.
-   `data_ingestor.reconnect_delay_s` (float): Initial delay before attempting reconnection.
-   `data_ingestor.max_reconnect_attempts` (int): Maximum number of consecutive reconnection attempts.
-   `data_ingestor.max_reconnect_delay_s` (float): Maximum delay for exponential backoff.
-   `data_ingestor.book_depth` (int): The depth of the L2 order book to maintain and subscribe to.
-   `data_ingestor.ohlc_intervals` (List[str]): List of OHLC intervals to subscribe to (e.g., `["1m", "5m", "1h"]`).
-   `trading.pairs` (List[str]): List of trading pairs to subscribe to (e.g., `["BTC/USD", "ETH/EUR"]`).
-   `data_ingestor.max_checksum_mismatches_before_resub` (int): Threshold for checksum errors before attempting resubscription.
-   `data_ingestor.critical_error_threshold_before_halt` (int): Threshold for critical errors before publishing `PotentialHaltTriggerEvent`.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `DataIngestor` module.
