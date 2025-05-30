# FeatureEngine Module Documentation

## Module Overview

The `gal_friday.feature_engine.py` module is responsible for processing incoming market data from various sources (OHLCV, L2 order book, and live trades) to calculate a wide array of technical indicators and market-derived features. Once calculated, these features are aggregated and published as a single `FeatureEvent` for each relevant timestamp and trading pair. This event then serves as the primary input for downstream modules like the `PredictionService`.

## Key Features

-   **Multi-Source Data Consumption:** Subscribes to and processes events for:
    -   `MARKET_DATA_OHLCV`: Historical and real-time Open, High, Low, Close, Volume data.
    -   `MARKET_DATA_L2`: Real-time Level 2 order book snapshots or updates.
    -   `MARKET_DATA_TRADE`: Real-time individual trade executions.
-   **Stateful Data Management:**
    -   Maintains historical OHLCV data for each trading pair in Pandas DataFrames, allowing for lookback calculations.
    -   Keeps the latest L2 order book snapshot for each pair.
    -   Stores a recent history of trades (e.g., within the current OHLCV bar) in `collections.deque` for each pair.
-   **Calculation Trigger:** Feature calculation is primarily triggered upon the arrival and processing of a new `MARKET_DATA_OHLCV` event, typically signifying the close of a new bar.
-   **Leverages `pandas-ta`:** Utilizes the `pandas-ta` library for efficient calculation of many standard technical indicators (e.g., RSI, MACD, Bollinger Bands).
-   **Configurable Feature Set:** Supports a flexible and configurable set of features, allowing users to define which features to calculate and their specific parameters.
    -   **OHLCV-based:** RSI, MACD, Bollinger Bands, VWAP (Volume Weighted Average Price, calculated from OHLCV data), Rate of Change (ROC), Average True Range (ATR), Standard Deviation of returns (StDev).
    -   **L2 Order Book-based:** Bid-Ask Spread (absolute and percentage), Order Book Imbalance, Weighted Average Price (WAP from book depth), Cumulative Depth at various levels.
    -   **Trade-based:** True Volume Delta (analyzing aggressor side of trades), VWAP (calculated from recent trades within a bar).
-   **Unified Feature Event Publishing:** Aggregates all calculated features for a specific trading pair and timestamp into a single `FeatureEvent` (published as a dictionary payload) via `PubSubManager`.
-   **Parameter Configurability:** Parameters for each feature (e.g., periods for moving averages, levels for book analysis, source for VWAP) are configurable through the application's main configuration file.

## Class `FeatureEngine`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config (dict)`: The application's global configuration dictionary, from which feature-specific configurations are extracted.
    -   `pubsub_manager (PubSubManager)`: An instance of `PubSubManager` for event subscription and publication.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for structured logging.
    -   `historical_data_service (Optional[HistoricalDataService])`: An optional service that could be used to bootstrap historical OHLCV data when the engine starts. *Note: The provided code snippets primarily show dynamic history building from live/streamed OHLCV events rather than explicit bootstrapping via this service in `start()`.*
-   **Actions:**
    -   Stores references to `pubsub_manager` and `logger_service`.
    -   Calls `_extract_feature_configs()` to load and store parameters for each configured feature from the `config`.
    -   Initializes data storage structures:
        -   `_ohlcv_history (defaultdict(pd.DataFrame))`: Stores OHLCV DataFrames, keyed by trading pair.
        -   `_l2_books (defaultdict(dict))`: Stores the latest L2 order book snapshot (bids/asks), keyed by trading pair.
        -   `_trade_history (defaultdict(deque))`: Stores recent trades, keyed by trading pair, with a configured maximum length.
    -   Calculates `_min_history_required` based on the maximum period length among all configured OHLCV-based features.

### Event Handling & Data Storage

-   **`start() -> None`**:
    -   Subscribes the `process_market_data` method to relevant market data events:
        -   `EventType.MARKET_DATA_OHLCV`
        -   `EventType.MARKET_DATA_L2`
        -   `EventType.MARKET_DATA_TRADE`
    -   Logs that the FeatureEngine service has started.

-   **`stop() -> None`**:
    -   Unsubscribes from all market data events.
    -   Logs that the FeatureEngine service is stopping.

-   **`async process_market_data(market_data_event_dict: dict) -> None`**:
    -   The main asynchronous entry point for all subscribed market data events (received as dictionaries).
    -   Extracts `event_type`, `trading_pair`, `timestamp`, and `payload` from the event dictionary.
    -   Routes processing based on `event_type`:
        -   If `MARKET_DATA_OHLCV`: Calls `_handle_ohlcv_update()`. After updating history, if enough data is available, it triggers `_calculate_and_publish_features()` for the current `trading_pair` and `timestamp`.
        -   If `MARKET_DATA_L2`: Calls `_handle_l2_update()`.
        -   If `MARKET_DATA_TRADE`: Calls `_handle_trade_event()`.
    -   Logs errors if event processing fails.

-   **`_handle_ohlcv_update(trading_pair: str, ohlcv_payload: dict) -> None`**:
    -   Converts the incoming `ohlcv_payload` (open, high, low, close, volume, timestamp) into a Pandas DataFrame row.
    -   Appends this new row to the `_ohlcv_history[trading_pair]` DataFrame.
    -   Ensures the DataFrame is sorted by timestamp.
    -   Removes duplicate timestamps, keeping the last entry.
    -   Prunes the history to keep a manageable size (e.g., `_min_history_required + buffer` rows) to prevent excessive memory usage.

-   **`_handle_l2_update(trading_pair: str, l2_payload: dict) -> None`**:
    -   Updates `_l2_books[trading_pair]` with the new L2 order book data (typically bids, asks, timestamp) from `l2_payload`.

-   **`_handle_trade_event(event_dict: dict) -> None`**:
    -   Extracts trade details (price, quantity, side, timestamp) from `event_dict["payload"]`.
    -   Appends the new trade to `_trade_history[event_dict["trading_pair"]]`. This deque has a maximum length to keep only recent trades relevant for intra-bar calculations (like trade-based VWAP or volume delta).

### Feature Calculation Orchestration

-   **`_extract_feature_configs() -> None`**:
    -   Retrieves the `features` dictionary from the main application configuration.
    -   Stores this dictionary in `self._feature_configs`. This dictionary maps user-defined feature names (e.g., "rsi_14", "vwap_trades") to their parameter dictionaries.

-   **`_get_min_history_required() -> int`**:
    -   Iterates through all configured OHLCV-based features (like RSI, MACD, Bollinger Bands, ATR, StDev, ROC, OHLCV-VWAP).
    -   Determines the maximum period length specified across all these features (e.g., if RSI period is 14 and MACD slow is 26, it would consider 26).
    -   Returns this maximum period, which dictates the minimum number of OHLCV bars needed before these features can be reliably calculated.

-   **`async _calculate_and_publish_features(trading_pair: str, timestamp_features_for: datetime) -> None`**:
    -   This is the core method that orchestrates the calculation of all configured features for a given `trading_pair` at a specific `timestamp_features_for` (usually the timestamp of the latest closed OHLCV bar).
    -   **1. Data Retrieval:**
        -   Gets the relevant OHLCV history DataFrame from `_ohlcv_history[trading_pair]`.
        -   Gets the latest L2 order book snapshot from `_l2_books[trading_pair]`.
        -   Gets the recent trade history from `_trade_history[trading_pair]`.
    -   **2. Feature Iteration:**
        -   Iterates through each `feature_name` and its `params` in `self._feature_configs`.
        -   Determines the type of feature (e.g., "rsi", "macd", "spread") based on `params.get("type", feature_name.split('_')[0])` or a similar mapping logic.
        -   Calls the appropriate `_process_..._feature` handler method for that feature type, passing the `params` and the collected `data_sources` (OHLCV, L2, trades).
    -   **3. Aggregation:** Collects all successfully calculated features into a `calculated_features` dictionary.
    -   **4. Publishing:**
        -   If features were calculated, constructs a `FeatureEvent` payload (a dictionary). This payload includes `trading_pair`, `timestamp`, and the `calculated_features` dictionary.
        -   Publishes this payload using `self._pubsub_manager.publish(EventType.FEATURE_EVENT, feature_event_payload)`.
    -   Handles cases where data might be insufficient for certain features, logging warnings.

-   **`_format_feature_value(value: Any) -> Optional[str]`**:
    -   Formats numeric feature values (Decimal, float, int) into a string representation with a fixed precision (e.g., 8 decimal places).
    -   Returns `None` or an empty string if the value is NaN, None, or cannot be formatted. This ensures consistent data types in the `FeatureEvent`.

### Individual Feature Calculation Handlers (`_process_..._feature` methods)

These private methods are responsible for invoking the core calculation logic for each type of feature. They act as intermediaries between the orchestration logic and the actual computation helpers.

-   Example: `_process_rsi_feature(params: dict, data_sources: dict) -> Optional[Dict[str, str]]`
    -   Extracts `period` from `params`.
    -   Retrieves `close_series_decimal` from `data_sources["ohlcv"]`.
    -   Calls `_calculate_rsi(close_series_decimal, period)`.
    -   Formats the result using `_format_feature_value`.
    -   Returns a dictionary like `{"rsi_value": formatted_rsi_value}` or `None` if calculation failed.
-   Similar methods exist for MACD, Bollinger Bands, VWAP (dispatching to OHLCV or trade-based), ROC, ATR, StDev, Bid-Ask Spread, Order Book Imbalance, Weighted Average Price (WAP), Cumulative Depth, and True Volume Delta.

### Core Calculation Helpers (`_calculate_...` methods)

These private methods contain the actual mathematical logic for calculating each feature.

-   **OHLCV-based:**
    -   `_calculate_rsi(close_series_decimal: pd.Series, period: int) -> Optional[Decimal]`
    -   `_calculate_macd(close_series_decimal: pd.Series, fast: int, slow: int, signal: int) -> Optional[Dict[str, Decimal]]` (returns MACD line, signal line, histogram)
    -   `_calculate_bollinger_bands(close_series_decimal: pd.Series, length: int, std_dev: float) -> Optional[Dict[str, Decimal]]` (returns upper, middle, lower bands)
    -   `_calculate_vwap(ohlcv_df_decimal: pd.DataFrame, length: Optional[int] = None) -> Optional[Decimal]` (OHLCV-based, optionally windowed)
    -   `_calculate_roc(close_series_decimal: pd.Series, period: int) -> Optional[Decimal]`
    -   `_calculate_atr(ohlcv_df_decimal: pd.DataFrame, length: int) -> Optional[Decimal]`
    -   `_calculate_stdev(close_series_decimal: pd.Series, length: int) -> Optional[Decimal]` (standard deviation of close price returns)
    -   *These methods typically use `pandas-ta` library functions on Pandas Series/DataFrames of Decimals, converting results back to Decimal.*
-   **L2 Order Book-based:**
    -   `_calculate_bid_ask_spread(l2_book: dict) -> Optional[Dict[str, Decimal]]` (returns absolute and percentage spread)
    -   `_calculate_order_book_imbalance(l2_book: dict, levels: int = 5) -> Optional[Decimal]`
    -   `_calculate_wap(l2_book: dict, levels: int = 5) -> Optional[Decimal]` (Weighted Average Price from top N book levels)
    -   `_calculate_depth(l2_book: dict, levels: int = 5) -> Optional[Dict[str, Decimal]]` (cumulative bid depth, ask depth)
-   **Trade-based:**
    -   `_calculate_true_volume_delta_from_trades(trades: deque, current_bar_start_time: datetime, bar_interval_seconds: int) -> Optional[Decimal]` (sum of signed volumes based on aggressor side within the current bar)
    -   `_calculate_vwap_from_trades(trades: deque, current_bar_start_time: datetime, bar_interval_seconds: int) -> Optional[Decimal]` (VWAP calculated from trades within the current bar)

## Dependencies

-   **Standard Libraries:**
    -   `uuid`: For generating unique event IDs.
    -   `datetime`: For timestamp operations.
    -   `decimal.Decimal`: For precise financial calculations.
    -   `collections.defaultdict`, `collections.deque`: For internal data storage.
-   **Third-Party Libraries:**
    -   `pandas`: For handling and manipulating OHLCV time series data.
    -   `pandas-ta`: A library providing a wide range of technical indicators that can be applied to Pandas DataFrames.
-   **Core Application Modules:**
    -   `gal_friday.core.pubsub.PubSubManager`: For event publishing.
    -   `gal_friday.logger_service.LoggerService`: For structured logging.
    -   `gal_friday.core.events.EventType`: For defining the type of event being published.
    -   `gal_friday.services.historical_data_service.HistoricalDataService` (Interface, optional for bootstrapping): For fetching initial historical data.

## Configuration (Key options from `features` section of app config)

The feature calculations are driven by the `features` section in the main application configuration file (e.g., `config.yaml`).

-   The top-level key is `features`.
-   Under `features`, each key represents a user-defined name for a specific feature instance (e.g., `rsi_14`, `macd_custom`, `vwap_from_trades`).
-   The value for each feature name is a dictionary specifying its `type` (which maps to a calculation handler, e.g., "rsi", "macd", "vwap") and its specific `parameters`.

**Example Configuration Structure:**

```yaml
application_config:
  # ... other configurations ...
  features:
    rsi_14:
      type: "rsi"
      period: 14
      enabled: true
    macd_default:
      type: "macd"
      fast: 12
      slow: 26
      signal: 9
      enabled: true
    b_bands_20_2:
      type: "bollinger_bands" # Or just "bbands" if mapped
      length: 20
      std_dev: 2.0
      enabled: true
    vwap_ohlcv_period_14: # VWAP from OHLCV data over a 14-bar window
      type: "vwap"
      source: "ohlcv" # Differentiates from trade-based VWAP
      length: 14 # Window length for OHLCV-based VWAP
      enabled: true
    vwap_trades_bar: # VWAP from trades within the current OHLCV bar
      type: "vwap"
      source: "trades"
      # bar_interval_seconds: 60 # This would be implicitly the OHLCV bar interval
      enabled: true
    order_book_imbalance_5_levels:
      type: "order_book_imbalance"
      levels: 5
      enabled: true
    # ... other features like atr, roc, stdev, spread, wap, depth, volume_delta
```

-   The `_extract_feature_configs` method loads this structure.
-   The `_calculate_and_publish_features` method iterates through these configured features, using the `type` to dispatch to the correct `_process_..._feature` handler and passing the associated parameters.
-   An `enabled: true` flag is good practice for easily toggling features.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `FeatureEngine` module.
