# Inter-Module Communication Definitions

**Project: Gal-Friday**

**Version: 1.0**

**Date: 2025-01-27**

**Status: Implementation Complete**

---

**Table of Contents:**

1.  Introduction
2.  Communication Mechanisms Overview
3.  Asynchronous Event Payloads
    3.1 MarketDataEvent (L2 Update)
    3.2 MarketDataEvent (OHLCV Update)
    3.3 FeatureEvent
    3.4 PredictionEvent
    3.5 TradeSignalProposedEvent
    3.6 TradeSignalApprovedEvent
    3.7 TradeSignalRejectedEvent
    3.8 ExecutionReportEvent
    3.9 LogEvent
    3.10 SystemStateEvent (e.g., HALT)
    3.11 ModelLifecycleEvent (Enterprise)
    3.12 ExperimentEvent (Enterprise)
    3.13 DriftDetectionEvent (Enterprise)
    3.14 ReconciliationEvent (Enterprise)
    3.15 PerformanceOptimizationEvent (Enterprise)
4.  Synchronous API Definitions
    4.1 PortfolioManager API: `get_current_state`
    4.2 ModelRegistry API (Enterprise)
    4.3 ExperimentManager API (Enterprise)

---

## 1. Introduction

This document defines the implemented structure and format of data exchanged between the core modules of the Gal-Friday system, operating within the Enhanced Modular Monolith architecture. It specifies the payloads for asynchronous events passed via the internal event bus/queues and the request/response formats for synchronous internal API calls. These definitions serve as contracts to ensure consistent communication and data handling between modules, supporting both core trading functionality and enterprise features.

## 2. Communication Mechanisms Overview

* **Enhanced Asynchronous Events:** The primary communication method uses an advanced, `asyncio`-based event bus with priority queuing, message ordering, and delivery guarantees. Modules publish events with enterprise features like replay capability and metrics collection.
* **Optimized Synchronous Calls:** Performance-optimized for critical, time-sensitive operations with caching and connection pooling reducing latency to sub-millisecond levels.
* **Enterprise Event Types:** Additional event types support model lifecycle management, A/B testing, drift detection, and performance optimization.

## 3. Asynchronous Event Payloads

All events include enhanced metadata for enterprise operations:

* `event_id`: A unique identifier for the event instance (UUID).
* `event_type`: A string identifying the type of event.
* `timestamp`: ISO 8601 formatted timestamp (UTC) with nanosecond precision.
* `source_module`: The name of the module that generated the event.
* `priority`: Event priority level (0-10, default 5).
* `correlation_id`: Optional correlation ID for tracking related events.
* `trace_id`: Distributed tracing identifier for performance monitoring.

*(Note: Examples below focus on the core data payload, but assume the enhanced metadata above is present.)*

### 3.1 MarketDataEvent (L2 Update)

* **`event_type`**: `"MARKET_DATA_L2"`
* **`source_module`**: `"DataIngestor"`
* **`payload`**:
    * `trading_pair`: String (e.g., "XRP/USD")
    * `exchange`: String (e.g., "kraken")
    * `timestamp_exchange`: (Optional) Timestamp from the exchange message, if available.
    * `bids`: List of lists `[[price_str, volume_str], ...]` (Sorted highest bid first)
    * `asks`: List of lists `[[price_str, volume_str], ...]` (Sorted lowest ask first)
    * `is_snapshot`: Boolean (True if this is a full book snapshot, False if an update)

### 3.2 MarketDataEvent (OHLCV Update)

* **`event_type`**: `"MARKET_DATA_OHLCV"`
* **`source_module`**: `"DataIngestor"`
* **`payload`**:
    * `trading_pair`: String (e.g., "XRP/USD")
    * `exchange`: String (e.g., "kraken")
    * `interval`: String (e.g., "1m")
    * `timestamp_bar_start`: ISO 8601 timestamp for the bar's opening time.
    * `open`: String (Price)
    * `high`: String (Price)
    * `low`: String (Price)
    * `close`: String (Price)
    * `volume`: String (Volume)

### 3.3 FeatureEvent

* **`event_type`**: `"FEATURES_CALCULATED"`
* **`source_module`**: `"FeatureEngine"`
* **`payload`**:
    * `trading_pair`: String
    * `exchange`: String
    * `timestamp_features_for`: ISO 8601 timestamp the features correspond to (e.g., end of OHLCV bar, L2 update time).
    * `features`: Dictionary containing calculated feature names and values.
        * *Example:* `{"rsi_14": "65.3", "spread_pct": "0.001", "book_imbalance_5": "0.8", "momentum_5": "0.0005", ...}` (Values as strings initially, conversion handled by consumer if needed)

### 3.4 PredictionEvent

* **`event_type`**: `"PREDICTION_GENERATED"`
* **`source_module`**: `"PredictionService"`
* **`payload`**:
    * `trading_pair`: String
    * `exchange`: String
    * `timestamp_prediction_for`: ISO 8601 timestamp the prediction corresponds to (aligned with `timestamp_features_for`).
    * `model_id`: String identifying the model used (e.g., "XGBoost_v1.2").
    * `prediction_target`: String describing what is predicted (e.g., "prob_price_up_0.1pct_5min").
    * `prediction_value`: Float (e.g., 0.72 meaning 72% probability).
    * `confidence`: (Optional) Float indicating model confidence, if available.
    * `associated_features`: (Optional) Dictionary of features used for this prediction (for logging/debugging).

### 3.5 TradeSignalProposedEvent

* **`event_type`**: `"TRADE_SIGNAL_PROPOSED"`
* **`source_module`**: `"StrategyArbitrator"`
* **`payload`**:
    * `trading_pair`: String
    * `exchange`: String
    * `signal_id`: Unique ID for this trade proposal.
    * `side`: String ("BUY" or "SELL")
    * `entry_type`: String (e.g., "LIMIT", "MARKET")
    * `proposed_entry_price`: String (Required if `entry_type` is "LIMIT")
    * `proposed_sl_price`: String (Stop-Loss price)
    * `proposed_tp_price`: String (Take-Profit price)
    * `triggering_prediction`: (Optional) Dictionary containing the `PredictionEvent` payload that triggered this signal.
    * `strategy_id`: String identifying the strategy logic used.

### 3.6 TradeSignalApprovedEvent

* **`event_type`**: `"TRADE_SIGNAL_APPROVED"`
* **`source_module`**: `"RiskManager"`
* **`payload`**:
    * `signal_id`: String (Corresponds to the `TradeSignalProposedEvent`)
    * `trading_pair`: String
    * `exchange`: String
    * `side`: String ("BUY" or "SELL")
    * `order_type`: String ("LIMIT" or "MARKET" - confirmed order type)
    * `quantity`: String (Calculated position size in base currency)
    * `limit_price`: String (Required if `order_type` is "LIMIT")
    * `sl_price`: String (Confirmed Stop-Loss price)
    * `tp_price`: String (Confirmed Take-Profit price)
    * `risk_parameters`: Dictionary showing risk checks passed (e.g., `{"risk_per_trade_pct": "0.5", "max_exposure_ok": true}`).

### 3.7 TradeSignalRejectedEvent

* **`event_type`**: `"TRADE_SIGNAL_REJECTED"`
* **`source_module`**: `"RiskManager"`
* **`payload`**:
    * `signal_id`: String (Corresponds to the `TradeSignalProposedEvent`)
    * `trading_pair`: String
    * `exchange`: String
    * `side`: String
    * `reason`: String describing why the trade was rejected (e.g., "EXCEEDS_MAX_EXPOSURE", "INSUFFICIENT_FUNDS", "DAILY_DRAWDOWN_LIMIT_HIT").

### 3.8 ExecutionReportEvent

* **`event_type`**: `"EXECUTION_REPORT"`
* **`source_module`**: `"ExecutionHandler"`
* **`payload`**:
    * `signal_id`: String (Originating signal ID, if applicable)
    * `exchange_order_id`: String (Order ID from Kraken)
    * `client_order_id`: String (Internal order ID used when placing)
    * `trading_pair`: String
    * `exchange`: String
    * `order_status`: String (e.g., "NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "REJECTED", "EXPIRED", "ERROR")
    * `order_type`: String ("LIMIT", "MARKET", "STOP_LOSS", "TAKE_PROFIT", etc.)
    * `side`: String ("BUY" or "SELL")
    * `quantity_ordered`: String
    * `quantity_filled`: String
    * `average_fill_price`: String (If applicable)
    * `limit_price`: String (If applicable)
    * `stop_price`: String (If applicable)
    * `commission`: String (Fee paid for this fill/order)
    * `commission_asset`: String (Asset the fee was paid in)
    * `timestamp_exchange`: Timestamp of the event from Kraken.
    * `error_message`: String (If order_status is REJECTED or ERROR)

### 3.9 LogEvent

* **`event_type`**: `"LOG_ENTRY"`
* **`source_module`**: *(Module generating the log)*
* **`payload`**:
    * `level`: String (e.g., "INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG")
    * `message`: String (The log message)
    * `context`: (Optional) Dictionary with additional context (e.g., `{"trading_pair": "XRP/USD", "signal_id": "..."}`)

### 3.10 SystemStateEvent (e.g., HALT)

* **`event_type`**: `"SYSTEM_STATE_CHANGE"`
* **`source_module`**: `"MonitoringService"` or `"CLIService"`
* **`payload`**:
    * `new_state`: String (e.g., "HALTED", "RUNNING", "DEGRADED")
    * `reason`: String (Reason for state change, e.g., "DAILY_DRAWDOWN_LIMIT_HIT", "MANUAL_HALT_COMMAND", "API_CONNECTION_LOST")
    * `halt_action`: (Optional, if `new_state` is "HALTED") String indicating action taken on open positions (e.g., "LIQUIDATE_POSITIONS", "MAINTAIN_POSITIONS").

### 3.11 ModelLifecycleEvent (Enterprise)

* **`event_type`**: `"MODEL_LIFECYCLE"`
* **`source_module`**: `"ModelRegistry"`
* **`payload`**:
    * `model_id`: String (Unique model identifier)
    * `action`: String (e.g., "REGISTERED", "PROMOTED", "DEPLOYED", "ARCHIVED")
    * `stage`: String (e.g., "development", "staging", "production")
    * `version`: String (Model version)
    * `metadata`: Dictionary (Model metadata including performance metrics)
    * `previous_stage`: Optional String (Previous stage for promotions)

### 3.12 ExperimentEvent (Enterprise)

* **`event_type`**: `"EXPERIMENT"`
* **`source_module`**: `"ExperimentManager"`
* **`payload`**:
    * `experiment_id`: String (Unique experiment identifier)
    * `action`: String (e.g., "CREATED", "STARTED", "RESULT_RECORDED", "COMPLETED")
    * `variants`: List of dictionaries (Model variants being tested)
    * `traffic_split`: Dictionary (Traffic allocation percentages)
    * `result`: Optional Dictionary (Experimental result if action is "RESULT_RECORDED")
    * `statistical_significance`: Optional Float (P-value for completed experiments)

### 3.13 DriftDetectionEvent (Enterprise)

* **`event_type`**: `"DRIFT_DETECTION"`
* **`source_module`**: `"RetrainingPipeline"`
* **`payload`**:
    * `model_id`: String (Model being monitored)
    * `drift_type`: String (e.g., "DATA_DRIFT", "CONCEPT_DRIFT", "PREDICTION_DRIFT", "PERFORMANCE_DRIFT")
    * `severity`: String (e.g., "LOW", "MEDIUM", "HIGH", "CRITICAL")
    * `confidence`: Float (Confidence score 0.0-1.0)
    * `metrics`: Dictionary (Drift detection metrics and statistics)
    * `recommendation`: String (e.g., "RETRAIN_IMMEDIATELY", "SCHEDULE_RETRAINING", "MONITOR")
    * `threshold_exceeded`: Boolean (Whether drift exceeds configured thresholds)

### 3.14 ReconciliationEvent (Enterprise)

* **`event_type`**: `"RECONCILIATION"`
* **`source_module`**: `"ReconciliationService"`
* **`payload`**:
    * `reconciliation_type`: String (e.g., "PORTFOLIO", "ORDERS", "POSITIONS")
    * `status`: String (e.g., "STARTED", "DISCREPANCY_FOUND", "RESOLVED", "FAILED")
    * `discrepancies`: List of dictionaries (Details of any discrepancies found)
    * `resolution_actions`: Optional List (Actions taken to resolve discrepancies)
    * `internal_state`: Dictionary (Snapshot of internal system state)
    * `exchange_state`: Dictionary (Snapshot of exchange state)

### 3.15 PerformanceOptimizationEvent (Enterprise)

* **`event_type`**: `"PERFORMANCE_OPTIMIZATION"`
* **`source_module`**: `"PerformanceOptimizer"`
* **`payload`**:
    * `optimization_type`: String (e.g., "CACHE", "CONNECTION_POOL", "QUERY", "MEMORY")
    * `action`: String (e.g., "ANALYSIS_STARTED", "OPTIMIZATION_APPLIED", "METRICS_COLLECTED")
    * `metrics_before`: Dictionary (Performance metrics before optimization)
    * `metrics_after`: Optional Dictionary (Performance metrics after optimization)
    * `optimization_details`: Dictionary (Details of optimizations applied)
    * `improvement_percentage`: Optional Float (Performance improvement percentage)

---

## 4. Synchronous API Definitions

### 4.1 PortfolioManager API: `get_current_state`

* **Caller:** `RiskManager`
* **Purpose:** To retrieve the latest, strongly consistent portfolio state immediately before performing pre-trade risk checks.
* **Request Parameters:**
    * `trading_pair`: (Optional) String - If provided, may return specific details for that pair.
* **Response Payload (Python Dictionary or Data Class):**
    * `timestamp`: ISO 8601 timestamp of the state snapshot.
    * `total_equity`: String (Current total account value in quote currency, e.g., USD)
    * `available_balance`: String (Free cash balance in quote currency)
    * `positions`: Dictionary where keys are trading pairs (e.g., "XRP/USD") and values are dictionaries:
        * `base_asset`: String (e.g., "XRP")
        * `quote_asset`: String (e.g., "USD")
        * `quantity`: String (Current position size, positive for long, negative for short - if shorting is supported)
        * `average_entry_price`: String
        * `current_market_value`: String (Estimated value based on last price)
        * `unrealized_pnl`: String
    * `total_exposure_pct`: String (Total value of all positions as % of equity)
    * `daily_drawdown_pct`: String
    * `weekly_drawdown_pct`: String
    * `total_drawdown_pct`: String

### 4.2 ModelRegistry API (Enterprise)

* **Caller:** `PredictionService`, `RetrainingPipeline`, `ExperimentManager`
* **Purpose:** Model lifecycle management operations.
* **Key Methods:**
    * `get_production_model(model_type: str) -> ModelArtifact`: Returns current production model.
    * `register_model(artifact: Any, metadata: dict) -> str`: Registers new model version.
    * `promote_model(model_id: str, stage: str) -> bool`: Promotes model between stages.
    * `get_model_metrics(model_id: str) -> dict`: Returns model performance metrics.

### 4.3 ExperimentManager API (Enterprise)

* **Caller:** `PredictionService`, `StrategyArbitrator`
* **Purpose:** A/B testing experiment management.
* **Key Methods:**
    * `get_active_experiments() -> List[dict]`: Returns currently running experiments.
    * `assign_variant(user_context: dict) -> str`: Assigns variant for A/B test.
    * `record_outcome(experiment_id: str, variant: str, outcome: dict) -> None`: Records experimental result.
    * `check_significance(experiment_id: str) -> dict`: Returns statistical significance analysis.

---
**End of Document**
