# Market Price Folder (`gal_friday/market_price`) Documentation

## Folder Overview

The `gal_friday/market_price` folder is dedicated to housing concrete implementations of market price services. These services are responsible for fetching real-time and, in some cases, historical market price data from various external exchanges or data vendors. Each implementation within this folder is designed to adhere to the `MarketPriceServiceInterface` (defined in `gal_friday.interfaces` or `gal_friday.core.types`), ensuring a consistent way for other parts of the Gal-Friday trading system to access market price information, regardless of the source.

## Key Modules and Their Roles

Currently, the primary focus of this folder is often on specific exchange implementations:

### `kraken_service.py` (`KrakenMarketPriceService` class)

-   **Purpose:** Implements the `MarketPriceServiceInterface` to provide market price data specifically from the Kraken cryptocurrency exchange.
-   **Functionality:**
    -   **Asynchronous API Communication:** Utilizes the `aiohttp` library to make asynchronous HTTP requests to Kraken's public API endpoints, ensuring non-blocking I/O operations.
    -   **Real-time Price Retrieval:**
        -   Fetches the latest trade price, best bid price, and best ask price for specified trading pairs using Kraken's `/0/public/Ticker` endpoint.
        -   The service parses the response from this endpoint to extract the relevant price information.
    -   **Historical Data Retrieval:**
        -   Retrieves historical Open, High, Low, Close, and Volume (OHLCV) data from Kraken's `/0/public/OHLC` endpoint. This typically requires parameters like trading pair, interval (e.g., 1 minute, 1 hour), and optional start/end timestamps.
    -   **Currency Conversion:**
        -   Provides capabilities to convert amounts from one currency to another. This is often achieved by:
            -   Looking up direct ticker prices (e.g., EUR/USD if converting EUR to USD).
            -   Using reverse ticker prices (e.g., using USD/EUR if EUR/USD is not directly available).
            -   Chaining conversions through a common intermediary currency (e.g., converting ALTCOIN to BTC, then BTC to USD, if a direct ALTCOIN/USD pair is illiquid or unavailable). Common intermediaries often include USD, USDT, or EUR.
    -   **Pair Formatting:** Handles the mapping between Gal-Friday's internal standardized trading pair formats (e.g., "BTC/USD") and Kraken's specific API representations for asset pairs (which can vary, e.g., "XBTUSD", "XXBTZUSD", "DOTUSD"). This includes both request formatting and response parsing.
    -   **Data Freshness Tracking:** Stores timestamps associated with fetched prices to allow consumers of the service (or the service itself) to perform data freshness checks, ensuring that decisions are not based on stale data.
-   **Dependencies:**
    -   `aiohttp`: For making asynchronous HTTP requests.
    -   `gal_friday.config_manager.ConfigManager` (or a protocol): For accessing API URLs and other configuration settings.
    -   `gal_friday.logger_service.LoggerService`: For logging service activities and errors.
    -   `gal_friday.interfaces.market_price_service_interface.MarketPriceService` (or similar path): The interface it implements.
    -   Standard libraries like `asyncio`, `datetime`, `decimal`.

### `__init__.py`

-   **Purpose:** Marks the `market_price` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules within the `market_price` directory to be imported using package notation (e.g., `from gal_friday.market_price.kraken_service import KrakenMarketPriceService`).
    -   It explicitly exports `KrakenMarketPriceService`, making it directly available when importing from `gal_friday.market_price`. This simplifies access for other parts of the application that need to instantiate or type-hint this specific implementation.

## Design and Extensibility

The structure of the `market_price` folder, in conjunction with the `MarketPriceServiceInterface`, promotes a modular and extensible design for market data acquisition:

-   **Interface-Driven:** By having concrete implementations like `KrakenMarketPriceService` adhere to a common `MarketPriceServiceInterface`, the rest of the Gal-Friday application can request market price data in a standardized way.
-   **Source Agnostic Consumption:** High-level services (e.g., `PortfolioManager`, `RiskManager`) depend on the `MarketPriceServiceInterface` abstraction, not on any specific implementation. This means they don't need to know whether prices are coming from Kraken, Binance, a file, or a simulated source during backtesting.
-   **Ease of Adding New Sources:** To integrate a new exchange or data vendor, a developer would:
    1.  Create a new service class within the `market_price` folder (e.g., `BinanceMarketPriceService`).
    2.  Implement the methods defined in `MarketPriceServiceInterface` using the new source's API or data access mechanism.
    3.  The application's main orchestrator (e.g., in `main.py`) can then be configured to instantiate and inject this new service based on the application's run mode or configuration settings.
-   **Simplified Testing:** When testing services that consume market price data, mock implementations of `MarketPriceServiceInterface` can be easily injected, allowing for controlled and predictable test scenarios without actual external API calls.

## Interactions

-   **Consumers of Market Price Data:**
    -   **`PortfolioManager`**: Uses a `MarketPriceService` to get current market prices for valuing open positions, calculating unrealized P&L, and determining overall portfolio equity.
    -   **`StrategyArbitrator`**: May use it to fetch current market prices (e.g., best bid/ask) when determining entry prices for limit orders or for context in strategy evaluation.
    -   **`RiskManager`**: Relies on current market prices for various checks, such as "fat-finger" validation (comparing proposed order prices against current market levels) and potentially for calculating exposure in real-time.
    -   **`FeatureEngine`**: While primarily driven by `DataIngestor` events, it might use a market price service for supplemental price information or currency conversions if needed for certain feature calculations.
-   **Backtesting Environment:**
    -   The `BacktestingEngine` would typically use a `SimulatedMarketPriceService` (which would also implement `MarketPriceServiceInterface`). This simulated service would feed historical prices from a loaded dataset chronologically, mimicking a live market environment.
-   **Configuration:**
    -   The specific implementation of `MarketPriceService` to be used (e.g., `KrakenMarketPriceService` in live mode) is determined at application startup by the main orchestrator, based on the application's configuration (`config.yaml`).

## Adherence to Standards

The design of the `market_price` folder and its components supports robust software architecture by:
-   **Adhering to a Common Interface:** Ensuring that all market price service implementations are interchangeable from the perspective of a consuming service.
-   **Encapsulating External Dependencies:** Isolating the specifics of interacting with external exchange APIs within dedicated service classes.
-   **Promoting Modularity:** Allowing different data sources to be developed, tested, and maintained independently.

This approach contributes to a more flexible, maintainable, and testable Gal-Friday trading system.
