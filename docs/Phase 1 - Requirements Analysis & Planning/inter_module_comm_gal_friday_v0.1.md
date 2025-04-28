# Inter-Module Communication Definitions

**Project: Gal-Friday**

**Version: 0.1**

**Date: 2025-04-27**

**Status: Draft**

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
4.  Synchronous API Definitions
    4.1 PortfolioManager API: `get_current_state`

---

## 1. Introduction

This document defines the structure and format of data exchanged between the core modules of the Gal-Friday system, operating within the Modular Monolith architecture. It specifies the payloads for asynchronous events passed via the internal event bus/queues and the request/response formats for necessary synchronous internal API calls. These definitions serve as contracts to ensure consistent communication and data handling between modules.

## 2. Communication Mechanisms Overview

* **Asynchronous Events:** The primary communication method uses an internal, `asyncio`-based event bus or queue system. Modules publish events, and interested modules subscribe to consume them. This promotes loose coupling. Payloads are typically represented as Python dictionaries or data classes.
* **Synchronous Calls:** Used sparingly for critical, time-sensitive operations requiring immediate, consistent state information where asynchronous event propagation latency is unacceptable. Primarily used by the `RiskManager` to query the `PortfolioManager`.

## 3. Asynchronous Event Payloads

All events should ideally include common metadata:

* `event_id`: A unique identifier for the event instance (e.g., UUID).
* `event_type`: A string identifying the type of event (e.g., "MARKET_DATA_L2", "PREDICTION").
* `timestamp`: An ISO 8601 formatted timestamp (UTC) indicating when the event was generated, with millisecond precision.
* `source_module`: The name of the module that generated the event (e.g., "DataIngestor", "PredictionService").

*(Note: Examples below focus on the core data payload for brevity, but assume the metadata above is present.)*

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

---
**End of Document**
