# Execution Folder (`gal_friday/execution`) Documentation

## Folder Overview

The `gal_friday/execution` folder is responsible for managing all direct communication and interaction with external trading exchanges. This critical layer handles two primary functions:
1.  **Trade Execution:** Managing the lifecycle of trading orders (placement, cancellation, modification, status checks) via the exchange's REST APIs.
2.  **Real-Time Data Streaming:** Establishing and maintaining WebSocket connections for receiving live market data (e.g., order books, trades, tickers) and private account updates (e.g., own trades, open orders).

Components within this folder are designed to be exchange-specific where necessary (e.g., `KrakenExecutionHandler`, `KrakenWebSocketClient`) but adhere to internal system interfaces for consistent integration with the rest of the Gal-Friday application.

## Key Modules and Their Roles

The `execution` folder contains several key Python modules, each tailored to a specific aspect of exchange interaction:

### `kraken.py` (`KrakenExecutionHandler` class)

-   **Purpose:** Implements the `ExecutionHandlerInterface` (defined in `gal_friday.core.types` or `gal_friday.interfaces`) specifically for the Kraken cryptocurrency exchange.
-   **Functionality:**
    -   **Authenticated REST API Interaction:** Handles all authenticated communication with Kraken's REST API endpoints.
    -   **Order Management:** Provides methods to:
        -   Submit new orders (limit, market, stop-loss, take-profit).
        -   Cancel existing open orders.
        -   Modify orders (if supported by the API and order type).
        -   Query the status of specific orders or all open orders.
    -   **Account Information Retrieval:**
        -   Fetches current account balances for various assets.
        -   Retrieves lists of currently open positions.
    -   **Kraken-Specific Logic:**
        -   Implements Kraken's authentication mechanism (API key, private key, nonce, signature generation for private API calls).
        -   Handles Kraken-specific API error codes and rate limits, with appropriate retry logic or error reporting.
        -   Converts data from Gal-Friday's internal formats to Kraken API request formats and vice-versa for responses.
-   **Importance:** This class is the direct interface for executing trading decisions and managing account state on the Kraken exchange.

### `websocket_client.py` (`KrakenWebSocketClient` class)

-   **Purpose:** Manages WebSocket connections to the Kraken exchange for receiving real-time data streams.
-   **Functionality:**
    -   **Multi-Feed Connectivity:** Establishes and maintains connections to both:
        -   **Public WebSocket Feeds:** For market data like tickers, L2 order books, live trades, and OHLCV updates for various trading pairs.
        -   **Private WebSocket Feeds:** For authenticated data related to the user's account, such as notifications for their own trades (`ownTrades`), updates to their open orders (`openOrders`), and other account-specific messages.
    -   **Subscription Management:** Handles subscribing and potentially unsubscribing (within API limits) to various channels and trading pairs.
    -   **Message Parsing and Event Publishing:**
        -   Receives raw JSON messages from the WebSocket.
        -   Parses these messages into standardized internal system event objects (e.g., `MarketDataL2Event`, `ExecutionReportEvent` for own trades, `MarketDataTradeEvent` for public trades).
        -   Publishes these standardized events to the `PubSubManager` for consumption by other services like `DataIngestor` or `PortfolioManager`.
    -   **Authentication for Private Feeds:** Implements Kraken's token-based authentication mechanism for accessing private WebSocket channels. This usually involves obtaining a temporary WebSocket token via a REST API call.
    -   **Connection Resilience:** Includes logic for:
        -   Handling heartbeats to keep the connection alive.
        -   Detecting disconnections and implementing automatic reconnection strategies.
-   **Importance:** Provides the live data streams that are essential for real-time decision-making, portfolio tracking, and market analysis within Gal-Friday.

### `websocket_connection_manager.py` (`WebSocketConnectionManager` class)

-   **Purpose:** Provides tools and logic for monitoring, managing the health, and overseeing the lifecycle of WebSocket connections, applicable to clients like `KrakenWebSocketClient`.
-   **Key Features:**
    -   **Connection Metrics Tracking:** Monitors key metrics for each WebSocket connection, such as:
        -   Number of messages received and sent.
        -   Connection uptime.
        -   Number of errors or disconnections.
        -   Latency (if measurable through ping/pong or message acknowledgements).
    -   **Health State Definition:** Defines and manages different health states for a WebSocket connection (e.g., `CONNECTING`, `CONNECTED`, `HEALTHY`, `DEGRADED`, `UNHEALTHY`, `DISCONNECTED`).
    -   **Reconnect Logic Orchestration:** Includes logic to determine if a reconnect is necessary based on configurable thresholds (e.g., consecutive errors, prolonged silence, failed heartbeats) and manages reconnection attempts, possibly with backoff strategies.
    -   **Proactive Monitoring:** May actively ping the WebSocket server or expect regular heartbeats to assess connection health.
-   **Importance:** Enhances the robustness of real-time data feeds by systematically managing WebSocket connections, attempting to recover from transient issues, and providing clear status information about connection health.

### `websocket_processor.py` (`WebSocketMessageProcessor`, `SequenceTracker`, `MessageCache` classes)

This module contains components designed to ensure the reliability and integrity of data received from WebSocket streams before it's fully parsed into system events.

-   **`WebSocketMessageProcessor` class:**
    -   **Purpose:** Acts as an intermediary processing stage for raw WebSocket messages.
    -   **Functionality:**
        -   Performs initial validation of message structure or common headers.
        -   May route messages to specific parsers based on channel or type.
        -   Integrates with `SequenceTracker` and `MessageCache`.

-   **`SequenceTracker` class:**
    -   **Purpose:** Monitors message sequence numbers for each subscribed WebSocket channel (if the exchange provides them, like Kraken does for book updates).
    -   **Functionality:**
        -   Tracks the last received sequence number for a channel.
        -   Detects gaps in sequence numbers, indicating potentially missed messages.
        -   Logs detected gaps and may trigger alerts or resynchronization requests (e.g., requesting a new order book snapshot if book update messages are missed).
-   **`MessageCache` class:**
    -   **Purpose:** Provides a mechanism for temporarily caching incoming WebSocket messages.
    -   **Functionality:**
        -   **Deduplication:** Can help in deduplicating messages if the source might send them more than once.
        -   **Short-Term Storage:** Caches messages with a configurable Time-To-Live (TTL) and cache size limit.
        -   **Replay/Gap Filling (Potential):** In advanced scenarios, a cache could be used to fill small, detected gaps if missing messages arrive out of order shortly after, or to provide a very short-term history for context.
-   **Importance:** These components collectively enhance the reliability of the data stream. `SequenceTracker` is vital for data integrity on sequenced channels. `WebSocketMessageProcessor` standardizes early-stage handling, and `MessageCache` can help with minor network glitches or message ordering issues.

### `__init__.py`

-   **Purpose:** Marks the `execution` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules within the `execution` directory to be imported using package notation (e.g., `from gal_friday.execution.kraken import KrakenExecutionHandler`).
    -   It often exports key classes from its modules (e.g., `KrakenExecutionHandler`, `KrakenWebSocketClient`, `WebSocketConnectionManager`) to make them directly accessible at the `gal_friday.execution` package level, simplifying imports for other parts of the application.

## Interactions and Importance

The `execution` folder serves as the critical bridge connecting Gal-Friday's internal trading logic and data processing pipelines with external trading exchanges. Its components are fundamental for both information gathering and action-taking:

-   **Action Arm (Trade Execution):** The `KrakenExecutionHandler` (or similar handlers for other exchanges) translates abstract commands generated by higher-level services (like `StrategyArbitrator` via `OrderExecutionManager`) into concrete, exchange-specific API calls. This is how the system places orders, cancels them, and queries account details.
-   **Sensory Arm (Data Ingestion):** The `KrakenWebSocketClient` (or similar) is responsible for tapping into the live pulse of the market. It feeds real-time market data (order books, trades, etc.) and private account updates into the system, where they are processed by `DataIngestor`, `FeatureEngine`, and `PortfolioManager`.
-   **Reliability and Robustness:** The `WebSocketConnectionManager` and the components within `websocket_processor.py` (`WebSocketMessageProcessor`, `SequenceTracker`, `MessageCache`) are crucial for ensuring that the WebSocket data streams are as reliable and complete as possible. This involves managing connection health, detecting data gaps, and handling message sequencing.
-   **Decoupling Exchange Specifics:** While components like `KrakenExecutionHandler` and `KrakenWebSocketClient` are exchange-specific, they are designed to implement generic interfaces (e.g., `ExecutionHandlerInterface`). This allows the rest of the Gal-Friday system to interact with them in a standardized way, making it easier to add support for other exchanges in the future by creating new implementations of these interfaces within the `execution` folder.

In summary, the `execution` folder encapsulates all the low-level details of interacting with specific exchanges, providing a clean and abstracted interface to the rest of the application. Without these components, Gal-Friday would be unable to receive live data or execute trades.

## Adherence to Standards

The design of the `execution` layer prioritizes reliable, resilient, and standardized communication with external exchanges. It aims to encapsulate exchange-specific protocols and authentication mechanisms, presenting a more uniform interface to the application's core logic. Error handling, rate limit considerations, and data integrity checks (like sequence tracking) are key aspects that contribute to a professional and robust execution management system.
