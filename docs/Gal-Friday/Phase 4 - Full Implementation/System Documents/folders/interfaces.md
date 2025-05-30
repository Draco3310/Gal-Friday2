# Interfaces Folder (`gal_friday/interfaces`) Documentation

## Folder Overview

The `gal_friday/interfaces` folder plays a pivotal role in the architectural design of the Gal-Friday trading system. Its primary purpose is to define the abstract contracts for all major services and components. These contracts are established using Python's Abstract Base Classes (ABCs) from the `abc` module and `typing.Protocol` (for structural subtyping).

By defining clear and stable interfaces, this folder promotes:
-   **Modularity:** Components are developed against these contracts, reducing interdependencies.
-   **Loose Coupling:** Services interact with each other through these abstractions rather than concrete implementations, making the system more flexible.
-   **Testability:** Interfaces allow for the easy creation of mock objects or test doubles, facilitating unit and integration testing.
-   **Extensibility:** Adding new functionalities, such as support for a new exchange or a new type of prediction model, becomes a matter of implementing the relevant interface without necessarily overhauling existing components.

This approach is central to achieving a clean, maintainable, and scalable system architecture.

## Key Interface Files and Their Purpose

The `interfaces` folder contains Python files, each defining one or more abstract contracts:

### `execution_handler_interface.py` (`ExecutionHandlerInterface`)

-   **Purpose:** Defines the standard set of operations for interacting with any trading exchange. This interface abstracts the exchange-specific details of order management and account information retrieval.
-   **Key Methods & Components:**
    -   Order Management: `submit_order(OrderRequest)`, `cancel_order(order_id, client_order_id)`, `modify_order(order_id, new_params)`, `get_order_status(order_id)`.
    -   Account Information: `get_account_balances()`, `get_open_positions()`, `get_trade_history()`.
    -   Market Information: `get_supported_trading_pairs()`, `get_asset_details(asset_symbol)`.
    -   **Universal Enums:** Defines standard enumerations for `OrderType` (e.g., LIMIT, MARKET, STOP_LOSS), `OrderStatus` (e.g., OPEN, FILLED, CANCELED), `OrderSide` (BUY, SELL), and `TimeInForce` (e.g., GTC, IOC, FOK).
    -   **Standard Dataclasses:** Defines common data structures like `OrderRequest` (for placing orders), `OrderResponse` (for order status and execution reports), and `PositionInfo` (for describing open positions).
    -   **`ExecutionHandlerFactory` Protocol:** Specifies an interface for a factory that can create instances of exchange-specific `ExecutionHandlerInterface` implementations.
-   **Importance:** Allows the core trading logic (e.g., `OrderExecutionManager`) to operate independently of the specific exchange being used.

### `feature_engine_interface.py` (`FeatureEngineInterface`)

-   **Purpose:** Outlines the contract for components responsible for feature engineering from various market data sources.
-   **Key Methods & Components:**
    -   Data Input: Methods to process different types of market data (e.g., `process_ohlcv_event(event)`, `process_l2_event(event)`).
    -   Feature Calculation: Abstract methods or properties related to the registration and calculation of features. This might include `register_feature_specification(FeatureSpec)` and a method to trigger feature calculation for a given timestamp, returning a `FeatureVector` or similar structure.
    -   **`FeatureSpec` Dataclass:** Defines how a feature should be calculated (e.g., name, type, parameters, input data sources).
    -   **`FeatureVector` Dataclass:** Represents the collection of calculated feature values for a specific timestamp and asset.
    -   Support for Multimodal Inputs: The interface is designed to accommodate features derived from various data types (OHLCV, L2 book, trades, sentiment, etc.) for advanced models, potentially including Multi-Agent Reinforcement Learning (MARL) applications.
    -   **`FeatureEngineFactory` Protocol:** Defines an interface for a factory responsible for creating `FeatureEngineInterface` instances.
-   **Importance:** Standardizes how features are generated, allowing different feature engineering strategies or libraries to be plugged into the system.

### `historical_data_service_interface.py` (`HistoricalDataService`)

-   **Purpose:** Defines the standard methods for retrieving historical market data. This is essential for backtesting trading strategies, training machine learning models, and providing historical context for live feature calculation.
-   **Key Methods:**
    -   `get_historical_ohlcv(pair: str, interval: str, start_date: datetime, end_date: datetime) -> pd.DataFrame`.
    -   `get_historical_trades(pair: str, start_date: datetime, end_date: datetime) -> List[TradeData]`.
    -   `get_next_bar(pair: str, current_timestamp: datetime) -> Optional[OhlcvBar]`: Used in backtesting to simulate bar-by-bar progression.
    -   `get_atr(pair: str, period: int, end_date: datetime) -> Optional[Decimal]`: Example of a method to get a specific pre-calculated indicator or raw data for it.
-   **Importance:** Provides a consistent way for different parts of the application (e.g., `BacktestingEngine`, `FeatureEngine` for bootstrapping) to access historical market data, irrespective of the underlying data source (e.g., local CSV files, a database, a third-party API).

### `kraken_api_interface.py` (`KrakenAPIInterface`)

-   **Purpose:** Provides a comprehensive, low-level abstraction specifically for *all* interactions with the Kraken exchange's REST API, covering both public and private (authenticated) endpoints.
-   **Key Methods & Components:**
    -   **Public Endpoints:** Methods for fetching system status, asset information (`GetAssetInfo`), tradable asset pairs (`GetTradableAssetPairs`), order book data (`GetOrderBook`), recent trades (`GetRecentTrades`), OHLC data (`GetOHLCData`), ticker information (`GetTickerInformation`).
    -   **Private Endpoints:** Methods for managing user account data, including balance (`GetAccountBalance`), trade balance (`GetTradeBalance`), open orders (`GetOpenOrders`), closed orders (`GetClosedOrders`), trade history (`GetTradesHistory`), open positions (`GetOpenPositions`), ledger info (`GetLedgersInfo`), and placing/canceling orders (`AddOrder`, `CancelOrder`).
    -   **Kraken-Specific Enums and Dataclasses:** Defines enums and dataclasses that precisely match Kraken API request and response structures (e.g., for order types, ledger types, specific error codes).
    -   **`KrakenAPIRateLimit` Protocol:** An interface to define how rate limiting for the Kraken API should be handled, potentially implemented by a rate limiter utility.
-   **Importance:** Encapsulates all direct Kraken REST API calls, providing a clear and testable interface. Higher-level components like `KrakenExecutionHandler` would use this interface rather than making raw HTTP requests directly. This isolates Kraken-specific communication logic.

### `market_price_service_interface.py` (`MarketPriceService`)

-   **Purpose:** Specifies the contract for services that provide real-time (or simulated real-time) market price information for various trading pairs.
-   **Key Methods:**
    -   `get_latest_price(pair: str) -> Optional[PriceInfo]`: Returns the most recent price information (e.g., last trade price, mid-price).
    -   `get_bid_ask_spread(pair: str) -> Optional[SpreadInfo]`: Returns current best bid, best ask, and spread.
    -   `is_data_fresh(pair: str, threshold_seconds: int) -> bool`: Checks if the market data for a pair is recent.
    -   `get_usd_conversion_rate(currency: str) -> Optional[Decimal]`: Provides conversion rates to a common currency (e.g., USD) for valuation purposes.
    -   **`PriceInfo` and `SpreadInfo` Dataclasses:** Define structures for returning price and spread data.
-   **Importance:** Decouples components needing price information (e.g., `PortfolioManager`, `RiskManager`) from the specific source of that information (e.g., live WebSocket feed, simulated data).

### `predictor_interface.py` (`PredictorInterface`)

-   **Purpose:** Defines the standard interface for all prediction models used within the `PredictionService`.
-   **Key Methods & Properties:**
    -   `load_model(model_path: str, scaler_path: Optional[str] = None, **kwargs) -> None`: Method to load the trained model artifacts (e.g., from a file) and any associated preprocessors like scalers.
    -   `predict(features: np.ndarray) -> Dict[str, Any]`: Takes a NumPy array of input features (ordered according to `expected_feature_names`) and returns a dictionary containing prediction outputs (e.g., `{"prediction_value": 0.75, "confidence": 0.9}`).
    -   `expected_feature_names: List[str]` (Property): A property that returns a list of feature names the model expects, in the correct order.
-   **Importance:** Allows the `PredictionService` to manage and use different types of models (e.g., XGBoost, Scikit-learn, LSTM/TensorFlow/PyTorch) in a uniform way, as long as they adhere to this interface.

### `strategy_interface.py` (`StrategyInterface`, `MARLStrategyInterface`, `EnsembleStrategyInterface`)

-   **Purpose:** Defines the contract for various types of trading strategies, enabling the `StrategyArbitrator` to manage and evaluate them consistently.
-   **Key Methods & Components:**
    -   **`StrategyInterface` (Base):**
        -   `analyze_market(prediction_event: PredictionEvent, current_portfolio_state: dict) -> Optional[StrategyAction]`: Core method that takes prediction and portfolio context to generate a trading action.
        -   `update_parameters(params: dict)`: For dynamic strategy parameter updates.
        -   `get_state() -> StrategyState`: Returns the current internal state of the strategy.
    -   **`MARLStrategyInterface` (for Multi-Agent Reinforcement Learning):** May extend `StrategyInterface` with methods specific to MARL, like handling observations, actions, and rewards for multiple agents.
    -   **`EnsembleStrategyInterface` (for ensembling multiple models/strategies):** May define methods for aggregating signals or predictions from constituent strategies.
    -   **`StrategyAction` Dataclass:** Defines the output of a strategy's analysis (e.g., BUY, SELL, HOLD, proposed SL/TP).
    -   **`StrategyState` Dataclass:** Represents the internal state of a strategy that might need to be persisted or monitored.
    -   **`StrategyFactory` Protocol:** Defines an interface for a factory that can create instances of different strategy implementations.
-   **Importance:** Provides a pluggable framework for trading strategies, allowing new strategies to be developed and integrated by implementing the defined interface(s).

### `__init__.py`

-   **Purpose:** Marks the `interfaces` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules within the `interfaces` directory to be imported using package notation (e.g., `from gal_friday.interfaces.execution_handler_interface import ExecutionHandlerInterface`).
    -   Crucially, it exports the primary interface classes (e.g., `ExecutionHandlerInterface`, `MarketPriceService`, `PredictorInterface`, etc.) to make them directly available at the `gal_friday.interfaces` package level. This simplifies imports for modules that depend on these interfaces.

## Design Philosophy

The design philosophy behind the `interfaces` folder is rooted in the **Dependency Inversion Principle (DIP)**, one of the SOLID principles of object-oriented design.
-   **Abstraction over Concretion:** High-level modules (e.g., `OrderExecutionManager`, `PredictionService`) depend on the abstractions (interfaces/protocols) defined in this folder, rather than on concrete low-level implementations (e.g., `KrakenExecutionHandler`, a specific XGBoost predictor class).
-   **Decoupling:** This decouples the high-level policy logic from the low-level implementation details. For instance, the `PredictionService` doesn't need to know how an XGBoost model works internally, only that it conforms to `PredictorInterface`.
-   **Enhanced Testability:** When testing a service that depends on an interface, mock implementations of that interface can be easily created and injected. This allows for isolated unit testing without needing to set up real external dependencies (like an exchange connection or a complex ML model).
-   **System Evolution and Extensibility:** The use of interfaces makes the system more adaptable to change.
    -   Adding support for a new exchange involves creating a new class that implements `ExecutionHandlerInterface` and `MarketPriceService` (and potentially others like `HistoricalDataService`). The core trading logic remains unchanged.
    -   Integrating a new type of machine learning model is simplified to creating a wrapper that implements `PredictorInterface`.
    -   New strategies can be introduced by implementing `StrategyInterface`.

## Adherence to Standards

The extensive use of well-defined interfaces (ABCs and Protocols) is a cornerstone of robust, maintainable, and scalable software architecture. This practice aligns with established software engineering principles that promote modular design, reduce coupling, and improve the overall quality and flexibility of the Gal-Friday trading system. It reflects a commitment to creating a professional-grade application.
