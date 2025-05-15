# Backtester Design Document

**Project: Gal-Friday**

**Version: 0.1**

**Date: 2025-04-27**

**Status: Draft**

---

**Table of Contents:**

1.  Introduction
2.  Backtester Architecture & Core Principles
3.  Data Handling
    3.1 Data Sources
    3.2 Data Loading & Preprocessing
    3.3 Preventing Look-ahead Bias
4.  Simulation Loop
    4.1 Event Generation
    4.2 Module Interaction
5.  Execution Simulation
    5.1 Order Matching Logic
    5.2 Slippage Modeling
    5.3 Commission/Fee Modeling
    5.4 Latency Simulation (Optional)
6.  State Management
7.  Configuration
8.  Output & Reporting
9.  Assumptions & Limitations

---

## 1. Introduction

This document details the design for the Backtesting Engine module of the Gal-Friday system, as specified in the Project Plan (`project_plan_gal_friday_v0.1` [R2]) and SRS (`srs_gal_friday_v0.1` [R1], Feature 3.10). The purpose of the backtester is to simulate the execution of trading strategies using historical market data, providing performance metrics to evaluate strategy viability before live deployment. This design emphasizes realism by incorporating models for fees, slippage, and ensuring correct data handling to avoid common backtesting pitfalls.

## 2. Backtester Architecture & Core Principles

* **Integration:** The backtester will operate within the Modular Monolith structure. It will instantiate and utilize the *same* core logic modules (`FeatureEngine`, `PredictionService`, `StrategyArbitrator`, `RiskManager`, `PortfolioManager`) as the live trading system. This ensures that the logic being tested is identical to the logic that will be deployed.
* **Simulation Core:** The `BacktestingEngine` class itself will orchestrate the simulation. It will manage the flow of historical time, load data, simulate the event loop, and interact with a specialized `SimulatedExecutionHandler`.
* **Simulated Execution:** A `SimulatedExecutionHandler` module will replace the live `ExecutionHandler`. It will receive approved order events from the `RiskManager` and simulate their execution against the historical data stream according to the logic defined in Section 5. It will publish simulated `ExecutionReportEvent`s.
* **Deterministic:** The backtester must be deterministic. Given the same historical data and configuration, it must produce the exact same results every time. This requires careful handling of random elements (if any) and consistent data processing.

## 3. Data Handling

### 3.1 Data Sources
* **Primary:** Historical OHLCV data (1-minute interval minimum) for the target trading pairs (XRP/USD, DOGE/USD) sourced from Kraken or a reliable historical data provider. Data should be stored locally (e.g., in CSV files, Parquet files, or queried from InfluxDB/PostgreSQL).
* **Secondary (Optional/Future):** Historical Level 2 order book data (snapshots or tick-by-tick updates) if available and deemed necessary for strategy accuracy. Handling L2 data significantly increases complexity and storage requirements.
* **Tertiary (Optional/Future):** Historical data for other inputs like funding rates, news events, or sentiment scores, if these are incorporated into the strategy.

### 3.2 Data Loading & Preprocessing
* The `BacktestingEngine` will be responsible for loading the required historical data for the specified date range and trading pairs.
* Data needs to be cleaned and preprocessed into a consistent format suitable for feeding into the simulation loop (e.g., pandas DataFrames or iterators yielding timestamped data points).
* Timestamps must be handled carefully, ensuring they are timezone-aware (preferably UTC) and sorted correctly.

### 3.3 Preventing Look-ahead Bias
* This is the most critical principle. The simulation loop must *only* provide data to the strategy modules that would have been available at that specific point in historical time.
* **Mechanism:** The simulation loop iterates through time step by time step (e.g., based on OHLCV bar timestamps). At each step `t`, only data up to and including time `t` is made available to the `FeatureEngine`, `PredictionService`, `StrategyArbitrator`, and `RiskManager` for decision-making. Order execution simulation (Section 5) for time `t` must use data from `t` or `t+1` (depending on the model) to determine fills, never future data.

## 4. Simulation Loop

* **Time Iteration:** The core loop iterates through the sorted historical timestamps (e.g., each 1-minute bar).
* **Event Generation:** At each timestamp `t`:
    * The `BacktestingEngine` generates simulated `MarketDataEvent`s (OHLCV, potentially L2) based on the historical data for time `t`.
    * These events are fed into the system, typically via the same internal event bus/queue mechanism used in live trading, or by directly calling the handler methods of the subscribed modules (`FeatureEngine.handle_market_data_event`).
* **Module Interaction:**
    * The `FeatureEngine` calculates features based on data up to time `t`.
    * The `PredictionService` generates predictions based on features at time `t`.
    * The `StrategyArbitrator` evaluates predictions and generates proposed signals based on state at time `t`.
    * The `RiskManager` evaluates proposed signals using portfolio state *before* any potential fills at time `t`.
    * If a signal is approved, it's passed to the `SimulatedExecutionHandler`.
    * The `SimulatedExecutionHandler` processes the order based on market conditions at time `t` (or `t+1`, see Section 5) and generates simulated `ExecutionReportEvent`(s).
    * The `PortfolioManager` consumes the simulated execution reports to update the portfolio state *after* the simulated execution for time `t`.
    * The `LoggerService` records events as configured.

## 5. Execution Simulation

The `SimulatedExecutionHandler` is responsible for realistically simulating how orders would have been filled.

### 5.1 Order Matching Logic
* **Data Granularity:** The realism depends heavily on the granularity of historical data available.
    * **OHLCV Data Only:** This is common but less accurate.
        * *Market Orders:* Assume filled at the `open` price of the *next* bar (`t+1`), or potentially a less favorable price like the `high` (for buys) or `low` (for sells) of the current bar `t` or next bar `t+1` to simulate worst-case execution within that period. Add slippage (see 5.2).
        * *Limit Orders (Buy):* Filled if the `low` price of the current bar `t` (or potentially next bar `t+1`) is less than or equal to the limit price. Fill price is typically the limit price.
        * *Limit Orders (Sell):* Filled if the `high` price of the current bar `t` (or potentially next bar `t+1`) is greater than or equal to the limit price. Fill price is typically the limit price.
        * *Stop-Loss Orders (Sell):* Triggered if the `low` price of the current bar `t` (or `t+1`) is less than or equal to the stop price. Fill price is simulated at the stop price or worse (e.g., stop price + slippage).
        * *Stop-Loss Orders (Buy):* Triggered if the `high` price of the current bar `t` (or `t+1`) is greater than or equal to the stop price. Fill price is simulated at the stop price or worse (e.g., stop price - slippage).
        * *Take-Profit Orders:* Handled similarly to Limit orders.
    * **Tick Data / L2 Data:** Allows for much more realistic simulation. Orders can be matched against historical bid/ask prices and volumes. Limit orders fill if the price crosses the limit level. Market orders "walk the book," consuming available volume at progressively worse prices until filled. This naturally incorporates slippage based on historical liquidity. *This is significantly more complex to implement.*

* **Assumption for MVP:** Assume simulation based on **OHLCV data only** for simplicity, using conservative fill assumptions (e.g., market orders fill at next bar open + slippage, limit/stop orders trigger if the bar's high/low range crosses the price).

### 5.2 Slippage Modeling
* Slippage is the difference between the expected fill price and the actual fill price. It's crucial for market orders and stop-loss orders.
* **Models (for OHLCV backtesting):**
    * *Fixed Slippage:* Add/subtract a fixed amount or percentage to the assumed fill price (e.g., 0.05% of price). Simple but unrealistic.
    * *Volatility-Based Slippage:* Calculate slippage based on recent price volatility (e.g., a fraction of the Average True Range - ATR). More realistic.
    * *Volume-Based Slippage (Requires Volume Data):* Estimate slippage based on the order size relative to the bar's volume (larger orders relative to volume experience more slippage).
* **Assumption for MVP:** Implement **Volatility-Based Slippage** using ATR as a configurable parameter.

### 5.3 Commission/Fee Modeling
* The `SimulatedExecutionHandler` must apply trading fees based on Kraken's fee structure.
* **Model:**
    * Use configurable maker/taker fee percentages based on Kraken's tiers.
    * Determine if a simulated limit order fill would likely be a "maker" (providing liquidity) or "taker" (removing liquidity) based on the simulated fill price relative to the bar's open/close or assumed spread. Market orders are always "takers".
    * Subtract the calculated commission from the trade's P&L and the portfolio's cash balance upon simulated fill.

### 5.4 Latency Simulation (Optional)
* Simulate delays between signal generation, order placement, and fill confirmation.
* **Model:** Add a configurable fixed or random delay (e.g., 50-200ms) before an order is considered "sent" to the simulated exchange and before a fill confirmation is generated. This can impact strategies sensitive to execution speed.
* **Assumption for MVP:** No explicit latency simulation initially, but acknowledge its potential impact.

## 6. State Management

* The `PortfolioManager` instance used in the backtest will track the simulated portfolio state (cash, positions, equity, P&L, drawdown) based on the simulated fills generated by the `SimulatedExecutionHandler`.
* It starts with the configured `initial_capital`.
* Its state is used by the `RiskManager` for pre-trade checks during the simulation.

## 7. Configuration

The backtester requires specific configuration inputs:
* Historical data source/path.
* Trading pair(s) to simulate.
* Date range (start and end dates).
* Initial capital.
* Commission fee percentages (maker/taker).
* Slippage model parameters (e.g., ATR multiplier for volatility-based slippage).
* Strategy and risk parameters (loaded via `ConfigurationManager`).

## 8. Output & Reporting

Upon completion, the `BacktestingEngine` will generate a performance report containing metrics specified in the SRS (FR-1006):
* Total Return (%)
* Annualized Return (%)
* Sharpe Ratio
* Sortino Ratio
* Maximum Drawdown (Value and %)
* Win Rate (%)
* Profit Factor (Gross Profit / Gross Loss)
* Average Trade P&L
* Total Number of Trades
* Average Holding Period
* Equity Curve Data (Timestamp, Equity) for plotting.
* List of executed trades (entry/exit times, prices, P&L).
* Output format: Console summary and detailed CSV/JSON files.

## 9. Assumptions & Limitations

* **Data Quality:** Assumes historical data is accurate and reasonably complete. Gaps or errors in data can significantly affect results.
* **OHLCV Limitations:** Backtesting solely on OHLCV data provides limited insight into intra-bar price movements and true liquidity, making execution simulation less precise than with tick/L2 data.
* **Slippage/Fill Models:** Simulated slippage and fill logic are approximations of real market behavior. Actual results may vary.
* **Static Environment:** Backtests assume a static environment (e.g., fixed fees, no API changes). Real markets evolve.
* **No Market Impact:** Assumes the simulated trades do not impact the market price (generally true for smaller capital sizes, but not for very large orders).
* **Overfitting Risk:** Good backtest performance does not guarantee future results. Strategies can be overfit to historical data. Requires out-of-sample testing and forward testing (paper trading) for validation.

---
**End of Document**
