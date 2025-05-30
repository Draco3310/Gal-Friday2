# SimulatedMarketPriceService Module (`gal_friday/simulated_market_price_service.py`) Documentation

## Module Overview

The `gal_friday.simulated_market_price_service.py` module provides a simulated implementation of the `MarketPriceService` interface (defined in `gal_friday.interfaces`). Its primary purpose is to supply market price data (latest prices, bid/ask spreads, synthetic order book snapshots, historical OHLCV) within a backtesting or paper trading environment. Instead of connecting to a live exchange, this service uses pre-loaded historical OHLCV data (typically as Pandas DataFrames) as its source of truth, allowing the simulation of market conditions based on this historical record.

## Key Features

-   **Interface Adherence:** Implements the `MarketPriceService` interface, ensuring it can be seamlessly swapped with a live market price service (like `KrakenMarketPriceService`) without changing the consuming code.
-   **Historical Data Driven:** Operates on a provided dataset of historical OHLCV data (usually a dictionary of Pandas DataFrames, keyed by trading pair).
-   **Time Simulation:** The current point in simulated time is controlled externally by calling the `update_time(timestamp)` method. All price and derived data requests are then relative to this "current" simulated timestamp.
-   **Latest Price Simulation (`get_latest_price`)**: Returns the price (typically the 'close' price, but configurable) from the historical OHLCV bar that corresponds to the current simulation timestamp for a given trading pair.
-   **Simulated Bid/Ask Spread (`get_bid_ask_spread`)**:
    -   Calculates a synthetic bid/ask spread around the latest simulated price.
    -   Supports a default spread percentage, which can be configured globally or overridden per trading pair.
    -   Optionally adjusts the spread based on historical volatility (ATR calculated using `pandas-ta` from the historical data), making the simulated spread wider during more volatile periods.
-   **Synthetic Order Book Snapshots (`get_order_book_snapshot`)**:
    -   Generates a synthetic Level 2 order book with configurable depth (number of levels), price steps between levels (as a percentage), and a volume decay model (simulating decreasing volume further from the mid-price).
-   **Currency Conversion Simulation (`convert_amount`)**:
    -   Simulates currency conversions using the latest available simulated prices.
    -   Can perform conversions using direct pairs (e.g., EUR/USD), reverse pairs (using 1/price of USD/EUR), or via a configured intermediary currency (e.g., converting TokenA -> USD -> TokenB).
-   **Historical Data Access (`get_historical_ohlcv`)**: Provides access to segments of the historical OHLCV data it was initialized with, allowing other components (like `FeatureEngine` during backtesting) to fetch historical context.
-   **Configuration Driven:** Simulation parameters such as the price column to use from OHLCV data, spread settings, volatility adjustment parameters, and depth simulation characteristics are configurable, ideally via a `ConfigManager`.

## Internal Data Structures (Dataclasses)

-   **`BookLevelConstructionContext(Dataclass)`**: A helper dataclass likely used internally by `_create_book_level_entries` to pass around parameters and state during the construction of synthetic order book levels (e.g., current price level, volume, side).

## Class `SimulatedMarketPriceService`

### Initialization (`__init__`)

-   **Parameters:**
    -   `historical_data (Dict[str, pd.DataFrame])`: A dictionary where keys are trading pair strings (e.g., "BTC/USD") and values are Pandas DataFrames containing the OHLCV historical data for that pair. Each DataFrame must have at least 'timestamp', 'open', 'high', 'low', 'close', 'volume' columns, with 'timestamp' as a datetime index or column.
    -   `config_manager (Optional[ConfigManager])`: An optional `ConfigManager` instance to load simulation-specific configurations. If not provided, the service will use hardcoded default parameters.
    -   `logger (Optional[logging.Logger])`: An optional logger instance. If not provided, it will set up a basic logger.
-   **Actions:**
    -   Stores the `historical_data`.
    -   Initializes `_current_timestamp` to `None`.
    -   Initializes `_config_manager` and `_logger`.
    -   Calls `_load_simulation_config()` to load parameters for spread calculation, volatility adjustments, synthetic depth generation, and currency conversion from `_config_manager` or set defaults.

### Core Interface Methods

-   **`async start() -> None` / `async stop() -> None`**:
    -   Lifecycle methods required by the `MarketPriceService` interface.
    -   For this simulator, these are typically no-op (no external connections or background tasks are managed directly by this service that require explicit start/stop).

-   **`async update_time(timestamp: datetime) -> None`**:
    -   Sets the internal `_current_timestamp` to the provided `timestamp`. This advances the simulation's "current time" for all subsequent price lookups.

-   **`async get_latest_price(trading_pair: str) -> Optional[Dict[str, Decimal]]`**:
    -   Retrieves the historical OHLCV bar for the `trading_pair` at or before the `_current_timestamp` using `_get_price_from_dataframe_asof()`.
    -   Uses the configured `_price_column_to_use` (e.g., 'close') from that bar as the latest price.
    -   Returns a dictionary like `{"price": latest_price, "timestamp": bar_timestamp}` or `None` if no data is available.

-   **`async get_bid_ask_spread(trading_pair: str) -> Optional[Dict[str, Decimal]]`**:
    -   Fetches the latest price for `trading_pair` using `get_latest_price()`.
    -   Calculates a base spread amount using `_spread_default_pct` (or a pair-specific override from config).
    -   If volatility-adjusted spread (`_spread_volatility_enabled`) is enabled:
        -   Calculates a normalized ATR using `_calculate_normalized_atr()`.
        -   Adjusts the spread based on this ATR and `_spread_volatility_multiplier`, capped by `_spread_max_adjustment_factor`.
    -   Calculates `bid = latest_price - (spread_amount / 2)` and `ask = latest_price + (spread_amount / 2)`.
    -   Returns `{"bid": bid_price, "ask": ask_price, "mid": latest_price, "timestamp": ...}` or `None`.

-   **`async get_order_book_snapshot(trading_pair: str, depth: int = 5) -> Optional[Dict[str, Any]]`**:
    -   If `_depth_simulation_enabled` is true:
        -   Fetches the latest price (mid-price) using `get_latest_price()`.
        -   Generates synthetic bid and ask levels:
            -   Starts from the mid-price and steps outwards using `_depth_price_step_pct`.
            -   Assigns volumes to each level, starting with `_depth_base_volume` and decaying with `_depth_volume_decay_factor`.
            -   Uses `_create_book_level_entries()` for this.
        -   Returns a dictionary like `{"bids": [[price, volume], ...], "asks": [[price, volume], ...], "timestamp": ...}`.
    -   Returns `None` if disabled or no base price is available.

-   **`async get_price_timestamp(trading_pair: str) -> Optional[datetime]`**:
    -   Returns the `_current_timestamp` if there's corresponding data available for the `trading_pair` at that time (effectively the timestamp of the bar used for `get_latest_price`).

-   **`async is_price_fresh(trading_pair: str, max_age_seconds: int) -> bool`**:
    -   In a simulated context driven by `update_time`, this check is slightly different from a live scenario. It primarily verifies if data *exists* for the `trading_pair` at the `_current_timestamp`.
    -   It might not strictly check `max_age_seconds` against a wall clock but rather confirms data availability for the simulation's current point in time.

-   **`async convert_amount(from_amount: Decimal, from_currency: str, to_currency: str) -> Optional[Decimal]`**:
    -   Simulates currency conversion.
    -   Uses `_get_direct_or_reverse_price()` to find a direct (e.g., FROM/TO) or reverse (TO/FROM) exchange rate.
    -   If not found, attempts conversion via an `_conversion_intermediary_currency` (e.g., USD) using `_get_cross_conversion_price()`.
    -   Returns the converted amount or `None` if a conversion path cannot be found using available simulated prices.

-   **`async get_historical_ohlcv(trading_pair: str, timeframe: str, since: Optional[datetime] = None, limit: Optional[int] = None, until: Optional[datetime] = None) -> Optional[pd.DataFrame]`**:
    -   Provides access to a segment of the historical OHLCV data that was initially loaded.
    -   Filters `self._historical_data[trading_pair]` based on `since`, `until`, and applies `limit`.
    -   The `timeframe` parameter might be used for validation if the loaded data is expected to match a certain interval, but primarily the loaded data's intrinsic interval is used.

-   **`async get_raw_atr(trading_pair: str, period: int = 14) -> Optional[Decimal]`**:
    -   Calls `_get_atr_dataframe_slice()` to get the necessary historical data up to `_current_timestamp`.
    -   Calls `_calculate_atr_from_slice()` to compute the ATR.

### Internal Simulation Logic

-   **`_load_simulation_config() -> None`**:
    -   Loads parameters from the `simulation` section of the configuration (if `ConfigManager` provided) or sets defaults.
    -   Keys include: `price_column_to_use`, `spread_default_pct`, `spread_pairs_override`, `spread_volatility_enabled`, `spread_volatility_lookback_period`, `spread_volatility_min_data_points`, `spread_volatility_atr_high_col`, `spread_volatility_atr_low_col`, `spread_volatility_atr_close_col`, `spread_volatility_multiplier`, `spread_max_adjustment_factor`, `depth_simulation_enabled`, `depth_num_levels`, `depth_price_step_pct`, `depth_base_volume`, `depth_volume_decay_factor`, `depth_price_precision`, `depth_volume_precision`, `conversion_intermediary_currency`.

-   **`_get_atr_dataframe_slice(trading_pair: str, lookback_period: int, required_data_points: int) -> Optional[pd.DataFrame]`**:
    -   Slices the historical DataFrame for `trading_pair` to get data up to `_current_timestamp`, ensuring enough data points (`lookback_period + required_data_points`) are available for ATR calculation.

-   **`_calculate_atr_from_slice(df_slice: pd.DataFrame, lookback_period: int, high_col: str, low_col: str, close_col: str) -> Optional[Decimal]`**:
    -   Calculates ATR on the provided `df_slice` using `pandas-ta` (`df_slice.ta.atr(length=lookback_period, high=high_col, low=low_col, close=close_col)`).
    -   Returns the latest ATR value.

-   **`_calculate_normalized_atr(trading_pair: str, current_price: Decimal) -> Optional[Decimal]`**:
    -   Calculates ATR using `_get_raw_atr()`.
    -   Normalizes it by dividing by `current_price` if both are available.

-   **`_get_price_from_dataframe_asof(trading_pair: str, desired_timestamp: datetime) -> Optional[pd.Series]`**:
    -   Helper to get the OHLCV bar (as a Pandas Series) from `self._historical_data[trading_pair]` that is active at the `desired_timestamp`.
    -   Uses `pandas.DataFrame.asof()` if the DataFrame's index is the timestamp, or manual filtering if timestamps are in a column. This ensures it gets the latest bar at or before the `desired_timestamp`.

-   **`_create_book_level_entries(start_price: Decimal, num_levels: int, price_step_pct: Decimal, base_volume: Decimal, volume_decay_factor: Decimal, side: str, price_precision: int, volume_precision: int) -> List[List[Decimal]]`**:
    -   Internal helper to generate levels for the synthetic order book snapshot.
    -   For bids, it iterates downwards from `start_price - step_size`.
    -   For asks, it iterates upwards from `start_price + step_size`.
    -   Volume for each level is calculated by applying `volume_decay_factor` to the previous level's volume.
    -   Prices and volumes are rounded to specified precisions.

-   **`async _get_direct_or_reverse_price(from_curr: str, to_curr: str) -> Optional[Decimal]`**:
    -   Attempts to find a price for the pair `FROM/TO` or `TO/FROM` using `get_latest_price()`.
    -   If `TO/FROM` is found, returns `1 / price`.

-   **`async _get_cross_conversion_price(from_curr: str, to_curr: str, intermediary_curr: str) -> Optional[Decimal]`**:
    -   Attempts to find prices for `FROM/INTERMEDIARY` and `TO/INTERMEDIARY`.
    -   If both found, calculates the cross rate: `(price_from_inter / price_to_inter)`.

## Dependencies

-   **Standard Libraries:** `logging`, `datetime`, `decimal`.
-   **Third-Party Libraries:**
    -   `pandas`: Essential for storing and accessing historical OHLCV data.
    -   `pandas-ta` (Technical Analysis Library): Optional, but used if volatility-adjusted spread or ATR calculations are enabled. The service should gracefully handle its absence if features relying on it are disabled.
-   **Core Application Modules:**
    -   `gal_friday.interfaces.market_price_service_interface.MarketPriceService` (the interface it implements).
    -   `gal_friday.config_manager.ConfigManager` (Optional, for loading simulation parameters).
    -   `gal_friday.logger_service.LoggerService` (Optional, for logging).

## Configuration

If a `ConfigManager` is provided, `SimulatedMarketPriceService` loads its parameters from a `simulation.market_price_service` (or similar) section in `config.yaml`. Key parameters include:

-   **`price_column_to_use (str)`**: The column in the OHLCV DataFrame to use as the "latest price" (e.g., "close", "open"). Default: "close".
-   **`spread` (dict)**:
    -   `default_pct (Decimal)`: Default bid-ask spread as a percentage of the mid-price.
    -   `pairs (Dict[str, Decimal])`: Dictionary to override `default_pct` for specific trading pairs.
    -   `volatility_adjustment_enabled (bool)`: Whether to adjust spread based on ATR.
    -   `volatility_lookback_period (int)`: Period for ATR calculation.
    -   `volatility_min_data_points (int)`: Minimum data points needed before ATR adjustment is active.
    -   `volatility_atr_high_col (str)`, `volatility_atr_low_col (str)`, `volatility_atr_close_col (str)`: Column names for ATR calculation.
    -   `volatility_multiplier (Decimal)`: Factor by which normalized ATR influences the spread.
    -   `volatility_max_adjustment_factor (Decimal)`: Maximum factor by which the base spread can be widened due to volatility.
-   **`depth` (dict)**:
    -   `simulation_enabled (bool)`: Whether to generate synthetic order book depth.
    -   `num_levels (int)`: Number of bid/ask levels to generate.
    -   `price_step_pct (Decimal)`: Percentage step between price levels in the synthetic book.
    -   `base_volume (Decimal)`: Initial volume for the levels closest to the mid-price.
    -   `volume_decay_factor (Decimal)`: Factor by which volume decreases for levels further from the mid-price.
    -   `price_precision (int)`, `volume_precision (int)`: Decimal precision for formatting book prices/volumes.
-   **`conversion` (dict)**:
    -   `intermediary_currency (str)`: The default currency to use for cross-conversions (e.g., "USD").

If `ConfigManager` is not provided, the service uses hardcoded internal defaults for these parameters.

## Adherence to Standards

The `SimulatedMarketPriceService` is crucial for creating a **controlled, reproducible, and realistic market environment for backtesting and paper trading**. By adhering to the `MarketPriceService` interface, it ensures that strategies and other services can operate consistently across simulated and live environments. The configurability of its simulation parameters (spread, slippage indirectly via price usage, depth) allows for testing strategy robustness under various hypothetical market conditions.
