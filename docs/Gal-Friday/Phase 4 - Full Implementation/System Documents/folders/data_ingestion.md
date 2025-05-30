# Data Ingestion Folder (`gal_friday/data_ingestion`) Documentation

## Folder Overview

The `gal_friday/data_ingestion` folder houses specialized tools and services dedicated to acquiring, managing, and ensuring the quality of various market data streams and time-series datasets utilized by the Gal-Friday trading system. These components play a critical role in both historical data analysis (e.g., for backtesting and model training) and real-time data processing for live trading operations. The focus is on robust data handling, quality assurance, and efficient real-time data acquisition.

## Key Modules and Their Roles

The `data_ingestion` folder contains the following key Python modules:

### `gap_detector.py` (`GapDetector` class)

-   **Purpose:** This module provides the `GapDetector` class, which is designed to detect, analyze, and offer strategies for filling gaps in time-series data, particularly historical market data (e.g., OHLCV bars).
-   **Key Features:**
    -   **Interval Auto-Detection:** Capable of automatically detecting the expected frequency or interval of the time-series data (e.g., 1-minute, 5-minute, 1-hour bars).
    -   **Gap Classification:** Identifies missing data points (gaps) and can classify them by severity (e.g., minor, major, critical) based on the length of the gap relative to the data interval or other configurable criteria.
    -   **Gap Filling Methods:** Offers various strategies to fill detected gaps, including:
        -   *Interpolation:* Linear or other methods to estimate missing values based on surrounding data points.
        -   *Forward Fill (ffill):* Propagating the last known value forward.
        -   *Zero Fill/Constant Fill:* Filling missing values with zero or another specified constant (use with caution, context-dependent).
    -   **Gap Pattern Analysis:** Can provide insights into patterns of data gaps, such as frequent occurrences at specific times of day or days of the week, which might indicate issues with the data source or collection process.
-   **Importance:** Data quality is paramount in algorithmic trading. The `GapDetector` is crucial for maintaining the integrity and completeness of historical datasets. Reliable historical data directly impacts the accuracy of feature engineering, the performance of machine learning models trained on that data, and the validity of backtest results.

### `websocket_market_data.py` (`WebSocketMarketDataService` class)

-   **Purpose:** This module implements the `WebSocketMarketDataService`, a service focused on managing WebSocket connections to exchanges for receiving real-time market data streams.
-   **Implementation Details:**
    -   It is specifically shown to use an underlying client, `KrakenWebSocketClient` (expected to be found in `gal_friday.execution.websocket_client` or a similar location), for interacting with the Kraken exchange's WebSocket API.
    -   It is designed to handle the complexities of WebSocket communication, including connection establishment, authentication (if required by the client), and message parsing.
-   **Functionality:**
    -   **Connection Management:** Establishes and maintains a persistent WebSocket connection to the specified exchange.
    -   **Multi-Stream Subscription:** Manages subscriptions to various real-time data channels for multiple trading pairs. Supported channels typically include:
        -   `book`: Level 2 order book updates.
        -   `ticker`: Real-time price ticker information.
        -   `trade`: Live trade executions.
        -   `ohlc`: Real-time OHLCV bar updates.
    -   **Dynamic Subscriptions (Potential):** The service might support adding or removing subscriptions dynamically during runtime, although this can be subject to limitations of the specific exchange's WebSocket API (e.g., Kraken's v1 WebSocket API had limitations regarding unsubscribing from specific pairs without reconnecting).
-   **Relation to `DataIngestor`:**
    -   The main `DataIngestor` module (located at the parent `gal_friday/` level) is the primary orchestrator for data ingestion, processing, and event publishing.
    -   `WebSocketMarketDataService` can be seen as a specialized component that `DataIngestor` might utilize or delegate to for managing the WebSocket communication aspect, particularly if supporting multiple exchanges or complex WebSocket feed types. Alternatively, it could serve as a standalone service for specific, focused WebSocket data needs if the main `DataIngestor` handles other types of data or has a different WebSocket client implementation. The exact interaction depends on the architectural design of the `DataIngestor`.

### `__init__.py`

-   **Purpose:** Marks the `data_ingestion` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules within the `data_ingestion` directory to be imported using package notation (e.g., `from gal_friday.data_ingestion.gap_detector import GapDetector`).
    -   It also explicitly exports `WebSocketMarketDataService` (i.e., `from .websocket_market_data import WebSocketMarketDataService`), making it directly available when importing from `gal_friday.data_ingestion`.

## Interactions and Importance

The components within the `gal_friday/data_ingestion` folder play a supportive yet critical role in the overall data pipeline of the Gal-Friday system:

-   **Ensuring Data Quality:** The `GapDetector` is a key tool for pre-processing and validating historical data. By identifying and addressing gaps, it ensures that the data fed into backtesting engines, feature calculators, and machine learning model training processes is as reliable and complete as possible. This directly contributes to the trustworthiness of any analysis or decisions based on that data.
-   **Real-Time Data Acquisition:** The `WebSocketMarketDataService` provides a focused and potentially exchange-specific mechanism for acquiring live market data feeds. Real-time data is the lifeblood of any live trading system, and this service ensures that data such as L2 order books, trades, and tickers are efficiently streamed into the system for processing by modules like `DataIngestor` and subsequently `FeatureEngine`.
-   **Supporting Robust Ingestion:** These tools help in building a more robust and resilient data ingestion pipeline. `GapDetector` helps in dealing with imperfections in historical data sources, while `WebSocketMarketDataService` encapsulates the complexities of real-time WebSocket communication, potentially making the main `DataIngestor` cleaner or more focused on data transformation and event publishing.
-   **Foundation for Trading Decisions:** Ultimately, the quality and timeliness of data acquisition, managed by components in this folder, directly influence the quality of features generated, the accuracy of predictions made, and the effectiveness of trading decisions executed by the system.

## Adherence to Standards

The components within the `data_ingestion` folder are designed with robust data handling practices in mind. Emphasis is placed on data quality assurance (`GapDetector`) and reliable real-time data stream management (`WebSocketMarketDataService`). These practices are aligned with the general principles of building dependable and maintainable software systems, contributing to the overall stability and reliability of the Gal-Friday trading application.
