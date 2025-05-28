# High-Level Architecture Concept

**Project: Gal-Friday**

**Version: 0.1**

**Date: 2025-04-27**

**Status: Draft**

---

**Table of Contents:**

1.  Introduction
2.  Selected Architecture: Modular Monolith
3.  Core Modules
4.  Interaction Patterns & Data Flow
5.  Key Considerations & Trade-offs
6.  Future Evolution

---

## 1. Introduction

This document outlines the proposed high-level software architecture concept for the Gal-Friday automated trading system. This concept is based on the project requirements defined in the SRS (`srs_gal_friday_v0.1` [R1]) and informed by the findings of the architectural design research performed via Deep Research. The primary goal is to establish a robust, performant, and maintainable structure for the system, particularly focusing on the needs of the full system.

## 2. Selected Architecture: Modular Monolith

Based on the comparative analysis provided by Deep Research, the selected architectural pattern for Gal-Friday is a **Modular Monolith**.

**Rationale:**

* **Low Latency:** This pattern facilitates minimal latency for the critical trading path (Market Data -> Prediction -> Risk Check -> Order Execution) by enabling fast, in-process communication between core modules. This is crucial for the effectiveness of the planned scalping and day trading strategies.
* **Simplified Data Consistency:** Managing the strongly consistent, real-time state of the portfolio (positions, equity, risk limits) is significantly simpler within a single process, avoiding the complexities of distributed transactions or eventual consistency inherent in microservices.
* **Reduced Complexity:** The development, deployment, testing, and operational overhead are considerably lower compared to a distributed microservices architecture, allowing the initial focus to be on core trading logic and ML integration.
* **Manageable Scalability (Initial):** While monolithic scaling is less granular, the primary anticipated bottleneck (CPU-bound ML inference) can likely be addressed within the monolith using techniques like multiprocessing pools, mitigating the most critical scaling concern for the full system scope.
* **Evolution Path:** A well-designed modular monolith provides a foundation that can potentially evolve towards microservices in the future if significant scaling needs arise, by extracting well-defined modules.

This approach prioritizes performance for the core trading loop and development simplicity for the initial build, accepting the trade-off of less granular scalability and fault isolation compared to microservices.

## 3. Core Modules

The Gal-Friday system will be internally structured into distinct logical modules, aligned with the features defined in the SRS [R1]. These modules will reside within the single monolithic application process but maintain clear boundaries and responsibilities:

1.  **`DataIngestor`:** Connects to Kraken WebSocket, receives L2/OHLCV data, parses, standardizes, and publishes market data events internally. Handles connection management and reconnections.
2.  **`FeatureEngine`:** Subscribes to market data events, calculates technical indicators and order book features, and publishes feature events/vectors.
3.  **`PredictionService`:** Subscribes to feature events, loads/manages ML models, performs predictions (offloading CPU-intensive inference to separate processes), and publishes prediction events.
4.  **`StrategyArbitrator`:** Subscribes to prediction and potentially feature/market data events, applies trading strategy logic, generates proposed trade signals (including preliminary SL/TP), and publishes these signals.
5.  **`PortfolioManager`:** Maintains the real-time state of the trading account (cash, positions, equity, P&L) by consuming execution report events. Provides consistent state information to other modules (especially RiskManager). Handles reconciliation with the exchange.
6.  **`RiskManager`:** Subscribes to proposed trade signals, performs all pre-trade checks (drawdown limits, risk per trade, position sizing, exposure limits, etc.) against the consistent state provided by `PortfolioManager`, and publishes approved/rejected trade events. Also monitors overall portfolio risk metrics for HALT conditions.
7.  **`ExecutionHandler`:** Subscribes to approved trade events, interacts securely with the Kraken API (REST/WebSocket) to place/manage orders, handles API responses/errors (using Circuit Breakers/Retries), and publishes execution report events (fills, errors).
8.  **`LoggerService`:** Subscribes to events from various modules or receives direct logging calls, formats logs, and writes them to configured outputs (files, PostgreSQL, InfluxDB).
9.  **`MonitoringService`:** Monitors system health (API connectivity, data freshness, resource usage), checks for HALT conditions (based on risk metrics from `PortfolioManager` or critical errors), and triggers HALT procedures.
10. **`BacktestingEngine`:** Orchestrates the simulation of trading logic using historical data, leveraging the core `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, `RiskManager`, `PortfolioManager`, and a simulated `ExecutionHandler`.
11. **`ConfigurationManager`:** Loads and provides access to system configuration parameters (risk settings, strategy thresholds, API keys, etc.).
12. **`CLIService`:** Provides the command-line interface for user interaction (start, stop, status, HALT).

## 4. Interaction Patterns & Data Flow

Within the Modular Monolith, communication will primarily leverage efficient in-process mechanisms, prioritizing low latency for critical paths while maintaining logical decoupling:

* **Internal Event Bus / Queues (asyncio-based):** An internal, lightweight event bus or system of `asyncio.Queue` instances will be the primary mechanism for communication between most modules. This promotes loose coupling and asynchronous processing without network overhead.
    * *Example Flow:* `DataIngestor` publishes `MarketDataEvent` -> `FeatureEngine` subscribes, processes, publishes `FeatureEvent` -> `PredictionService` subscribes, processes, publishes `PredictionEvent` -> `StrategyArbitrator` subscribes, processes, publishes `TradeSignalProposedEvent`.
* **Synchronous Access for Critical State:** For time-sensitive, consistency-critical operations, direct but controlled access or synchronous calls might be used.
    * *Example:* The `RiskManager`, upon receiving a `TradeSignalProposedEvent`, needs the *absolute latest* portfolio state. It will likely query the `PortfolioManager` directly (via a method call within the same process) to get the required equity and position data synchronously before performing its checks. This ensures the risk assessment uses up-to-the-millisecond data.
* **Offloading CPU-Bound Tasks:** The `PredictionService` will use `loop.run_in_executor()` with a `ProcessPoolExecutor` to run computationally intensive ML model inference tasks in separate processes. This prevents blocking the main asyncio event loop. Input features are passed to the process pool, and prediction results are returned asynchronously (e.g., via a future or callback) to be published as events.
* **API Interaction:** The `ExecutionHandler` will use asynchronous libraries (`aiohttp`, potentially async `ccxt` wrappers) to interact with the external Kraken APIs non-blockingly.

**Simplified Core Data Flow:**

1.  `DataIngestor` receives WebSocket data -> Publishes `MarketDataEvent`.
2.  `FeatureEngine` consumes `MarketDataEvent` -> Calculates features -> Publishes `FeatureEvent`.
3.  `PredictionService` consumes `FeatureEvent` -> Runs ML model (offloaded) -> Publishes `PredictionEvent`.
4.  `StrategyArbitrator` consumes `PredictionEvent` -> Applies rules -> Publishes `TradeSignalProposedEvent`.
5.  `RiskManager` consumes `TradeSignalProposedEvent` -> Synchronously gets state from `PortfolioManager` -> Performs checks -> Publishes `TradeSignalApprovedEvent` (or Rejected).
6.  `ExecutionHandler` consumes `TradeSignalApprovedEvent` -> Performs final HALT check -> Asynchronously places order via Kraken API -> Receives fill/error -> Publishes `ExecutionReportEvent`.
7.  `PortfolioManager` consumes `ExecutionReportEvent` -> Updates internal state.
8.  `LoggerService` consumes various events -> Writes to logs/DBs.
9.  `MonitoringService` consumes events/checks state -> Triggers HALT if needed.

## 5. Key Considerations & Trade-offs

* **Modularity Discipline:** Strict adherence to module boundaries and defined interfaces (events, function signatures) is crucial to prevent the monolith from degrading into a tightly coupled "big ball of mud."
* **Error Handling:** Robust error handling within each module and asynchronous task is vital to prevent failures from crashing the entire process. Unhandled exceptions in background tasks need careful management.
* **Testing:** The monolithic structure simplifies end-to-end and integration testing compared to microservices, but thorough unit testing for each module remains essential. Backtesting consistency is easier to achieve.
* **Deployment:** The entire application is deployed as a single unit. While simpler initially, any change requires redeploying the whole application.
* **Technology Stack:** The entire application shares the same Python runtime and dependencies.

## 6. Future Evolution

This Modular Monolith architecture provides a solid foundation. If future requirements demand more granular scaling or technology diversification, well-defined modules can be identified and potentially extracted into separate microservices using patterns like the Strangler Fig pattern. The initial focus on clear modular boundaries within the monolith facilitates this potential future transition.

---
**End of Document**
