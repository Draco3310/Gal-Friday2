# Core Module Interface Definitions

**Project: Gal-Friday**

**Version: 0.1**

**Date: 2025-04-27**

**Status: Draft**

---

**Table of Contents:**

1.  Introduction
2.  Interface Definitions (Python Class/Method Signatures)
    2.1  `DataIngestor`
    2.2  `FeatureEngine`
    2.3  `PredictionService`
    2.4  `StrategyArbitrator`
    2.5  `PortfolioManager`
    2.6  `RiskManager`
    2.7  `ExecutionHandler`
    2.8  `LoggerService`
    2.9  `MonitoringService`
    2.10 `BacktestingEngine`
    2.11 `ConfigurationManager`
    2.12 `CLIService`
    2.13 Internal Event Bus/Queue Interface (Conceptual)

---

## 1. Introduction

This document specifies the primary interfaces for the core modules of the Gal-Friday system. It defines the key classes and their public methods, including function signatures with type hints, following Python conventions. These interfaces represent the contracts between modules within the Modular Monolith architecture, guiding implementation and ensuring consistent interaction. Asynchronous methods (`async def`) are indicated where I/O operations or interaction with the asyncio event loop are expected.

*(Note: Type hints use standard Python types or reference the structures defined in the `inter_module_comm_gal_friday_v0.1` document where applicable. Dependencies like event buses or configuration objects are assumed to be injected during initialization.)*

## 2. Interface Definitions (Python Class/Method Signatures)

### 2.1 `DataIngestor`

* **Purpose:** Connects to Kraken WebSocket, subscribes to market data feeds, parses messages, and publishes standardized `MarketDataEvent`s.
* **Class:** `DataIngestor`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus)`
    * `async start(self) -> None`: Initiates connections and starts listening loop.
    * `async stop(self) -> None`: Closes connections gracefully.
    * `async _handle_message(self, message: Any) -> None`: (Internal) Parses raw message.
    * `async _publish_market_data(self, event_payload: dict) -> None`: (Internal) Puts event onto the event bus.
    * `async _reconnect(self) -> None`: (Internal) Handles reconnection logic.

### 2.2 `FeatureEngine`

* **Purpose:** Consumes `MarketDataEvent`s, calculates features, and publishes `FeatureEvent`s.
* **Class:** `FeatureEngine`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus)`
    * `async start(self) -> None`: Starts listening for market data events.
    * `async stop(self) -> None`: Stops processing.
    * `async handle_market_data_event(self, event: dict) -> None`: Consumes market data, updates internal state (e.g., price series, order book), triggers feature calculation.
    * `_calculate_features(self, trading_pair: str, timestamp: datetime) -> dict`: (Internal) Performs the actual feature calculations based on current internal state.
    * `async _publish_features(self, feature_payload: dict) -> None`: (Internal) Puts `FeatureEvent` onto the event bus.

### 2.3 `PredictionService`

* **Purpose:** Consumes `FeatureEvent`s, runs ML models (offloading CPU work), and publishes `PredictionEvent`s.
* **Class:** `PredictionService`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus, process_pool_executor: ProcessPoolExecutor)`
    * `async start(self) -> None`: Loads models, starts listening for feature events.
    * `async stop(self) -> None`: Stops processing.
    * `async handle_feature_event(self, event: dict) -> None`: Consumes features, triggers prediction generation.
    * `async _generate_prediction(self, features: dict, trading_pair: str, timestamp: datetime) -> dict`: (Internal) Prepares data and offloads inference task to the process pool.
    * `_run_model_inference(self, model_id: str, processed_features: Any) -> dict`: (Internal, potentially static/module-level function run in process pool) Executes the actual ML model inference.
    * `async _publish_prediction(self, prediction_payload: dict) -> None`: (Internal) Puts `PredictionEvent` onto the event bus.

### 2.4 `StrategyArbitrator`

* **Purpose:** Consumes `PredictionEvent`s (and potentially others), applies strategy logic, and publishes `TradeSignalProposedEvent`s.
* **Class:** `StrategyArbitrator`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus)`
    * `async start(self) -> None`: Starts listening for prediction events.
    * `async stop(self) -> None`: Stops processing.
    * `async handle_prediction_event(self, event: dict) -> None`: Consumes predictions, evaluates strategy rules.
    * `_evaluate_strategy(self, prediction_data: dict) -> Optional[dict]`: (Internal) Applies specific strategy logic (e.g., threshold checks) and determines preliminary SL/TP. Returns proposed signal payload or None.
    * `async _publish_trade_signal_proposed(self, signal_payload: dict) -> None`: (Internal) Puts `TradeSignalProposedEvent` onto the event bus.

### 2.5 `PortfolioManager`

* **Purpose:** Maintains real-time, consistent portfolio state (cash, positions, equity, P&L, drawdown). Consumes `ExecutionReportEvent`s. Provides state to `RiskManager`.
* **Class:** `PortfolioManager`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus)`
    * `async start(self) -> None`: Initializes state (e.g., fetch initial balance from exchange), starts listening for execution reports.
    * `async stop(self) -> None`: Stops processing.
    * `async handle_execution_report(self, event: dict) -> None`: Updates internal portfolio state based on fills/order updates.
    * `get_current_state(self, trading_pair: Optional[str] = None) -> dict`: **(Synchronous Method)** Returns the latest portfolio state snapshot (structure defined in Inter-Module Comm doc). Called directly by `RiskManager`.
    * `async reconcile_with_exchange(self) -> None`: Periodically fetches actual balance/positions from Kraken API and compares/adjusts internal state.
    * `_update_pnl_and_drawdown(self) -> None`: (Internal) Recalculates P&L and drawdown metrics based on current state and market prices (may need market price feed).
    * `async _publish_portfolio_metrics(self) -> None`: (Internal) Periodically publishes `portfolio_metrics` to InfluxDB via `LoggerService` or directly.

### 2.6 `RiskManager`

* **Purpose:** Consumes `TradeSignalProposedEvent`s, performs pre-trade checks using state from `PortfolioManager`, and publishes `TradeSignalApprovedEvent` or `TradeSignalRejectedEvent`. Monitors overall risk for HALT.
* **Class:** `RiskManager`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus, portfolio_manager: PortfolioManager)`
    * `async start(self) -> None`: Starts listening for proposed trade signals.
    * `async stop(self) -> None`: Stops processing.
    * `async handle_trade_signal_proposed(self, event: dict) -> None`: Consumes proposed signal, performs checks.
    * `_perform_pre_trade_checks(self, proposed_signal: dict) -> Tuple[bool, Optional[str], Optional[dict]]`: (Internal) Calls `portfolio_manager.get_current_state()`, calculates position size, checks all limits (drawdown, exposure, balance, etc.). Returns (is_approved, rejection_reason, approved_payload).
    * `_calculate_position_size(self, current_equity: Decimal, risk_per_trade_pct: Decimal, sl_distance: Decimal) -> Decimal`: (Internal) Calculates size based on fixed fractional method.
    * `async _publish_trade_signal_approved(self, approved_payload: dict) -> None`: (Internal) Puts `TradeSignalApprovedEvent` onto the event bus.
    * `async _publish_trade_signal_rejected(self, rejected_payload: dict) -> None`: (Internal) Puts `TradeSignalRejectedEvent` onto the event bus.
    * `async check_portfolio_risk_limits(self) -> None`: Periodically checks overall drawdown limits from `PortfolioManager` state and triggers HALT via `MonitoringService` if breached.

### 2.7 `ExecutionHandler`

* **Purpose:** Consumes `TradeSignalApprovedEvent`s, interacts with Kraken API to place/manage orders, handles API errors/retries/circuit breaking, and publishes `ExecutionReportEvent`s.
* **Class:** `ExecutionHandler`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus)`
    * `async start(self) -> None`: Initializes API client (e.g., `ccxt`), starts listening for approved signals.
    * `async stop(self) -> None`: Cancels open orders (optional, configurable), stops processing.
    * `async handle_trade_signal_approved(self, event: dict) -> None`: Consumes approved signal, prepares and places order(s) via API.
    * `async _place_order(self, order_details: dict) -> dict`: (Internal) Interacts with Kraken API (using retries/circuit breaker). Returns initial API response or error details.
    * `async _handle_api_response(self, response: Any, original_order_details: dict) -> None`: (Internal) Parses API response, publishes initial `ExecutionReportEvent`.
    * `async _monitor_order_status(self, exchange_order_id: str) -> None`: (Internal, if not using WebSocket for private updates) Periodically poll order status.
    * `async _handle_exchange_update(self, update_data: dict) -> None`: (Internal, if using WebSocket for private updates) Processes real-time fill/status updates from exchange feed.
    * `async _publish_execution_report(self, report_payload: dict) -> None`: (Internal) Puts `ExecutionReportEvent` onto the event bus.

### 2.8 `LoggerService`

* **Purpose:** Consumes `LogEvent`s or receives direct calls, writes logs to configured destinations (files, console, PostgreSQL, InfluxDB).
* **Class:** `LoggerService`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: Optional[EventBus] = None)`: Can listen to events or be called directly.
    * `async start(self) -> None`: Initializes log handlers (file, DB connections).
    * `async stop(self) -> None`: Flushes buffers, closes handlers.
    * `async handle_log_event(self, event: dict) -> None`: (If event-driven) Processes log event.
    * `log(self, level: str, message: str, source_module: str, context: Optional[dict] = None, exc_info: Optional[Any] = None) -> None`: **(Synchronous or Async)** Direct logging method. Writes to appropriate handlers based on level and config.
    * `log_timeseries(self, measurement: str, tags: dict, fields: dict, timestamp: Optional[datetime] = None) -> None`: **(Synchronous or Async)** Method to write data to InfluxDB.

### 2.9 `MonitoringService`

* **Purpose:** Monitors overall system health, checks for HALT conditions, and triggers HALT state.
* **Class:** `MonitoringService`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, event_bus: EventBus)`
    * `async start(self) -> None`: Starts periodic health checks (API connectivity, data freshness).
    * `async stop(self) -> None`: Stops monitoring loops.
    * `async check_health(self) -> None`: Performs periodic health checks.
    * `async handle_potential_halt_trigger(self, reason: str, source: str) -> None`: Receives signals (e.g., from `RiskManager`, `DataIngestor`, `CLI`) about potential HALT conditions.
    * `async trigger_halt(self, reason: str, halt_action: str) -> None`: Sets the system HALT state, publishes `SystemStateEvent`.
    * `async trigger_resume(self) -> None`: Clears the HALT state (likely triggered by `CLIService`), publishes `SystemStateEvent`.
    * `is_halted(self) -> bool`: **(Synchronous)** Returns the current HALT status flag. (Checked by `ExecutionHandler`).

### 2.10 `BacktestingEngine`

* **Purpose:** Orchestrates strategy simulation using historical data.
* **Class:** `BacktestingEngine`
* **Methods:**
    * `__init__(self, config: ConfigurationManager)`: Initializes with specific backtest configuration.
    * `run_backtest(self, start_date: datetime, end_date: datetime, initial_capital: Decimal) -> dict`: Loads historical data, simulates the event loop using core modules (instantiated in backtest mode), generates performance report.
    * `_load_historical_data(self, ...) -> Any`: (Internal) Fetches data (e.g., from DB or files).
    * `_simulate_event_loop(self, ...) -> None`: (Internal) Drives the simulation, feeding data to modules.
    * `_generate_report(self, ...) -> dict`: (Internal) Calculates performance metrics.

### 2.11 `ConfigurationManager`

* **Purpose:** Loads, validates, and provides access to system configuration.
* **Class:** `ConfigurationManager`
* **Methods:**
    * `__init__(self, config_path: str)`
    * `load_config(self) -> None`: Loads configuration from file(s).
    * `get(self, key: str, default: Optional[Any] = None) -> Any`: Retrieves a configuration value.
    * `get_trading_pairs(self) -> List[str]`
    * `get_risk_parameters(self) -> dict`
    * `get_strategy_parameters(self, strategy_id: str) -> dict`
    * `get_api_keys(self) -> dict`: Accesses securely stored API keys.

### 2.12 `CLIService`

* **Purpose:** Handles command-line arguments and user interaction.
* **Class:** `CLIService`
* **Methods:**
    * `__init__(self, config: ConfigurationManager, monitoring_service: MonitoringService, main_app_controller: Any)`: Takes dependencies needed to control the application.
    * `parse_args(self) -> dict`: Parses command-line arguments (e.g., run mode: live/backtest/paper, config file path).
    * `handle_command(self, command: str) -> None`: Processes commands received while running (e.g., 'status', 'stop', 'halt', 'resume').

### 2.13 Internal Event Bus/Queue Interface (Conceptual)

* **Purpose:** Abstract interface for publishing and subscribing to internal events. Could be implemented using `asyncio.Queue` or a dedicated pub/sub library.
* **Interface:** `EventBus` (Abstract Base Class or Protocol)
* **Methods:**
    * `async publish(self, event_type: str, payload: dict) -> None`: Publishes an event.
    * `subscribe(self, event_type: str, handler: Callable[[dict], Coroutine]) -> None`: Registers a handler coroutine for a specific event type.
    * `unsubscribe(self, event_type: str, handler: Callable[[dict], Coroutine]) -> None`: Removes a handler.

---
**End of Document**
