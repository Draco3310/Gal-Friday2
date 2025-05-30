# KrakenHistoricalDataService Module (`gal_friday/kraken_historical_data_service.py`) Documentation

## Module Overview

The `gal_friday.kraken_historical_data_service.py` module provides a concrete implementation of the `HistoricalDataService` interface (defined in `gal_friday.interfaces`). Its primary purpose is to supply historical market data, such as Open-High-Low-Close-Volume (OHLCV) bars and individual trades, specifically from the Kraken cryptocurrency exchange.

A key architectural feature of this service is its use of **InfluxDB as a caching layer**. It prioritizes fetching data from the local InfluxDB instance. If data is missing in InfluxDB or if there are gaps, the service then falls back to querying the Kraken public REST API. Any newly fetched data from Kraken is subsequently stored back into InfluxDB to serve future requests more efficiently. The service also incorporates resilience patterns like rate limiting and circuit breaking for Kraken API calls.

## Key Features

-   **Implements `HistoricalDataService` Interface:** Adheres to the contract defined for historical data providers in the Gal-Friday system.
-   **OHLCV Data Retrieval:** Fetches historical OHLCV bars for specified trading pairs, time ranges, and intervals.
-   **Trade Data Retrieval (Planned/Partial):** Includes methods for fetching historical trade data, though the direct API fetching part for trades might be a placeholder or under development.
-   **InfluxDB Caching Strategy:**
    -   First attempts to retrieve requested historical data from a configured InfluxDB instance.
    -   If data is not found in InfluxDB or is incomplete for the requested range, it then queries the Kraken API.
    -   Stores data newly fetched from Kraken back into InfluxDB to optimize subsequent requests for the same data.
-   **Average True Range (ATR) Calculation:** Provides a method to calculate ATR for a given trading pair and period, using historical OHLCV data (typically fetched from InfluxDB) and the `pandas-ta` library.
-   **Resilient API Interaction:**
    -   **`RateLimitTracker`:** Manages the frequency of calls to the Kraken API to avoid exceeding rate limits, based on configurable API tiers.
    -   **`CircuitBreaker`:** Implements the circuit breaker pattern for calls to the Kraken API. This helps prevent cascading failures by temporarily stopping requests to a failing service (Kraken API) and giving it time to recover.
-   **Asynchronous Operations:** Uses `aiohttp` for making asynchronous HTTP GET requests to Kraken's public API endpoints. InfluxDB client interactions are also asynchronous.
-   **Data Validation:** Includes basic validation for OHLCV data fetched from InfluxDB or API to check for common issues like NaNs, negative prices/volumes, or incorrect OHLC relationships (e.g., low > high).

## Helper Classes

The module defines or uses several helper classes to manage API interactions and resilience:

-   **`CircuitBreakerError(Exception)`**: Custom exception raised by the `CircuitBreaker` when it is in an "OPEN" state (meaning further calls are blocked).
-   **`RateLimitTracker`**:
    -   Manages API call frequency against configured limits for different tiers (e.g., Kraken has different rate limits for various endpoints based on a cost system).
    -   Provides an asynchronous method (e.g., `wait_for_slot_async()`) that delays execution if necessary to stay within rate limits.
-   **`CircuitBreaker`**:
    -   Implements the circuit breaker pattern.
    -   Tracks failures for API calls. If the failure rate exceeds a threshold, the circuit "opens," and subsequent calls are failed immediately for a configured timeout period.
    -   After the timeout, it enters a "HALF-OPEN" state, allowing a limited number of test calls. If successful, the circuit "closes"; otherwise, it re-opens.

## Class `KrakenHistoricalDataService`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config (dict)`: The application's global configuration dictionary, expected to provide access to InfluxDB settings (URL, token, org, bucket), Kraken API base URL, API rate limit tiers, and circuit breaker parameters. This is often accessed via a `ConfigManager`-like interface.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
-   **Actions:**
    -   Stores `logger_service`.
    -   Initializes the `InfluxDBClient` (asynchronous version) and its `QueryApi` and `WriteApi` using connection details from `config`.
    -   Initializes the `RateLimitTracker` with rate limit configurations from `config`.
    -   Initializes the `CircuitBreaker` with parameters (failure threshold, recovery timeout) from `config`.
    -   Stores the Kraken API base URL from `config`.
    -   Initializes an `aiohttp.ClientSession` for making HTTP requests.

### Core Methods (from `HistoricalDataService` interface)

-   **`async get_historical_ohlcv(trading_pair: str, start_time: datetime, end_time: datetime, interval: str) -> Optional[pd.DataFrame]`**:
    -   The primary method for fetching OHLCV data.
    -   **1. Query InfluxDB:** First, attempts to retrieve the data for the given `trading_pair`, `interval`, and `start_time`/`end_time` range from InfluxDB using `_query_ohlcv_data_from_influxdb()`.
    -   **2. Identify Missing Ranges:** If data is found in InfluxDB but is incomplete, or if no data is found, it calls `_get_missing_ranges()` to determine the specific time periods for which data needs to be fetched from the Kraken API.
    -   **3. Fetch from Kraken API:** For each missing range:
        -   Calls `await self._fetch_ohlcv_data(trading_pair, range_start, range_end, interval)` which wraps the API call with rate limiting and circuit breaking.
        -   **Important Note:** The underlying `_fetch_ohlcv_data_from_api()` method that makes the actual Kraken API call for OHLCV data is currently a **placeholder in the described codebase and returns dummy data.** For real functionality, this needs to be implemented to call Kraken's `/0/public/OHLC` endpoint.
    -   **4. Store in InfluxDB:** If new data is successfully fetched from Kraken, it's processed (validated, formatted) and then stored back into InfluxDB using `_store_ohlcv_data_in_influxdb()`.
    -   **5. Combine and Return:** Combines data from InfluxDB and newly fetched data (if any), sorts it, and returns a Pandas DataFrame. Returns `None` if no data can be obtained.

-   **`async get_historical_trades(trading_pair: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]`**:
    -   Similar logic to `get_historical_ohlcv`, but for individual trade data.
    -   Prioritizes fetching from InfluxDB via `_query_trades_data_from_influxdb()`.
    -   If data is missing, it's intended to call `fetch_trades()` to get data from Kraken's `/0/public/Trades` endpoint.
    -   **Important Note:** While `fetch_trades()` might make an API call, the full pipeline of identifying missing ranges and integrating InfluxDB caching for trades might be less developed or a TODO compared to OHLCV in the described codebase.
    -   Newly fetched trade data would also be stored in InfluxDB.

-   **`async get_next_bar(trading_pair: str, timestamp: datetime, interval: str) -> Optional[pd.Series]`**:
    -   Retrieves the single OHLCV bar immediately following the given `timestamp` for the specified `trading_pair` and `interval` from InfluxDB.
    -   Useful for bar-by-bar simulations in a backtester if it's not pre-loading all data.

-   **`async get_atr(trading_pair: str, timestamp: datetime, interval: str, period: int = 14) -> Optional[Decimal]`**:
    -   Fetches a window of historical OHLCV data (e.g., `period + X` bars) up to the given `timestamp` from InfluxDB (or via `get_historical_ohlcv`).
    -   Calculates the Average True Range (ATR) using the `pandas-ta` library (`df.ta.atr(length=period)`).
    -   Returns the latest ATR value as a `Decimal`, or `None` if data is insufficient.

### Internal Data Fetching & Storage

-   **`async _fetch_ohlcv_data(trading_pair: str, start_time: datetime, end_time: datetime, interval: str) -> Optional[pd.DataFrame]`**:
    -   A wrapper method that applies the `CircuitBreaker` and `RateLimitTracker` before calling `_fetch_ohlcv_data_from_api()`.
    -   Handles `CircuitBreakerError` and rate limit delays.

-   **`async _fetch_ohlcv_data_from_api(trading_pair_kraken: str, interval_seconds: int, since_timestamp: int) -> Optional[pd.DataFrame]`**:
    -   **Placeholder Note:** This method is described as currently returning **dummy data**.
    -   **Intended Logic:**
        -   Construct parameters for Kraken's `/0/public/OHLC` endpoint (using `pair=trading_pair_kraken`, `interval` mapped from `interval_seconds`, `since=since_timestamp`).
        -   Make the GET request using `_make_public_request()`.
        -   Parse the JSON response, which contains OHLCV data in a specific array format.
        -   Convert this data into a Pandas DataFrame with columns like 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'count'.
        -   Handle Kraken API errors and pagination if necessary.

-   **`async fetch_trades(trading_pair: str, since_timestamp: Optional[int] = None) -> Optional[pd.DataFrame]`**:
    -   Constructs parameters for Kraken's `/0/public/Trades` endpoint.
    -   Calls `_make_public_request()` to fetch raw trade data.
    -   **Partially Implemented Note:** The transformation of this raw trade data into a structured DataFrame and subsequent caching/gap-filling logic might be incomplete or a TODO.

-   **`async _store_ohlcv_data_in_influxdb(df: pd.DataFrame, trading_pair: str, interval: str) -> None`**:
    -   Takes a Pandas DataFrame of OHLCV data.
    -   Constructs InfluxDB `Point` objects from each row, setting `trading_pair` and `interval` as tags, and OHLCV values as fields.
    -   Uses the `InfluxDBClient.WriteApi` to write these points to the configured InfluxDB bucket.

-   **`async _query_ohlcv_data_from_influxdb(trading_pair: str, start_time: datetime, end_time: datetime, interval: str) -> Optional[pd.DataFrame]`**:
    -   Builds a Flux query to select OHLCV data for the given `trading_pair`, `interval`, and time range from InfluxDB.
    -   Executes the query using `InfluxDBClient.QueryApi`.
    -   Converts the query result (Flux tables) into a Pandas DataFrame.

-   **`async _query_trades_data_from_influxdb(trading_pair: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]`**:
    -   Similar to `_query_ohlcv_data_from_influxdb` but for querying trade data.

### Utilities & Helpers

-   **`_interval_to_seconds(interval: str) -> int`**: Converts interval strings like "1m", "5m", "1h", "1d" into their corresponding duration in seconds. Used for calculations and potentially for some API parameters if they expect seconds. Kraken API uses minutes for OHLC intervals.
-   **`_validate_ohlcv_data(df: pd.DataFrame) -> bool`**:
    -   Performs basic sanity checks on an OHLCV DataFrame:
        -   No NaN values in critical columns.
        -   Prices (O, H, L, C) and Volume are non-negative.
        -   High is the maximum of O, H, L, C; Low is the minimum.
    -   Returns `True` if data is valid, `False` otherwise.
-   **`_get_missing_ranges(df_from_db: Optional[pd.DataFrame], required_start_time: datetime, required_end_time: datetime, interval_seconds: int) -> List[Tuple[datetime, datetime]]`**:
    -   Compares the data available in `df_from_db` (from InfluxDB) against the `required_start_time` and `required_end_time`.
    -   Identifies contiguous time ranges within the required window for which no data exists in `df_from_db`.
    -   Returns a list of `(start_missing_range, end_missing_range)` tuples.
-   **`async _get_latest_timestamp_from_influxdb(trading_pair: str, interval: str) -> Optional[datetime]`**:
    -   Queries InfluxDB for the timestamp of the most recent OHLCV data point stored for the given `trading_pair` and `interval`.
-   **`async _make_public_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]`**:
    -   A generic helper to make asynchronous GET requests to public Kraken API `endpoint` with optional `params`.
    -   Handles JSON response parsing and basic error checking (e.g., HTTP status codes). This method itself does not include rate limiting or circuit breaking; those are applied by calling wrappers.
-   **`_get_kraken_pair_name(trading_pair: str) -> str`**:
    -   Maps Gal-Friday's internal, standardized trading pair names (e.g., "BTC/USD") to the specific format required by the Kraken API (e.g., "XBTUSD", "XXBTZUSD"). This often involves a lookup table or defined mapping rules.

## Dependencies

-   **Standard Libraries:** `asyncio`, `logging`, `datetime`, `decimal`.
-   **Third-Party Libraries:**
    -   `pandas`: For DataFrame manipulation of historical data.
    -   `pandas-ta`: For calculating technical indicators like ATR.
    -   `influxdb_client.aio` (for asynchronous InfluxDB v2+ interaction).
    -   `aiohttp`: For making asynchronous HTTP requests to the Kraken API.
-   **Core Application Modules:**
    -   `gal_friday.interfaces.historical_data_service_interface.HistoricalDataService` (the interface it implements).
    -   `gal_friday.logger_service.LoggerService`.
    -   `gal_friday.config_manager.ConfigManager` (or a protocol providing similar access to configuration).
    -   Potentially custom error types from `gal_friday.exceptions`.

## Configuration

The `KrakenHistoricalDataService` relies on configurations accessed via `ConfigManager` for:
-   **InfluxDB Settings:**
    -   `influxdb.url`
    -   `influxdb.token`
    -   `influxdb.org`
    -   `influxdb.bucket` (for OHLCV data, and potentially a separate one for trades)
-   **Kraken API:**
    -   `kraken.api_url` (base URL for public API, e.g., "https://api.kraken.com")
    -   `kraken.api_rate_limit_tier` (e.g., "starter", "intermediate", "pro" to select appropriate rate limits for `RateLimitTracker`).
-   **Circuit Breaker:**
    -   `circuit_breaker.failure_threshold` (e.g., 5 failures).
    -   `circuit_breaker.recovery_timeout_seconds` (e.g., 60 seconds).
-   Mappings for trading pair names if not hardcoded.

## Important Note on API Fetching Implementation

It is crucial to highlight that, as per the typical description of such a service in early to mid-development stages, the direct API fetching logic, particularly in `_fetch_ohlcv_data_from_api()` and parts of `fetch_trades()`, is often initially a **placeholder returning dummy data or is not fully implemented with robust pagination and error handling for all edge cases.**

For the service to function fully in all scenarios (especially when InfluxDB is empty or has significant gaps), these API interaction methods must be completely implemented to:
1.  Correctly call the respective Kraken API endpoints (`/0/public/OHLC`, `/0/public/Trades`).
2.  Handle Kraken's specific request parameters (like `since` for pagination of OHLCV data).
3.  Parse the complex JSON responses from Kraken accurately.
4.  Implement proper pagination if Kraken returns data in chunks.
5.  Perform thorough error handling for API-specific errors returned in the JSON response.

Without these full implementations, the service will heavily rely on data being pre-populated or subsequently cached in InfluxDB for most of its functionality.

## Adherence to Standards

The `KrakenHistoricalDataService` demonstrates good software engineering practices by:
-   Implementing a defined interface (`HistoricalDataService`), promoting modularity.
-   Employing a caching strategy (InfluxDB) to improve performance and reduce load on external APIs.
-   Incorporating resilience patterns like rate limiting and circuit breaking to handle external service interactions more robustly.
-   Using asynchronous programming (`asyncio`, `aiohttp`) for efficient I/O operations.
These features contribute to a more reliable and performant data provisioning layer for the Gal-Friday system.
