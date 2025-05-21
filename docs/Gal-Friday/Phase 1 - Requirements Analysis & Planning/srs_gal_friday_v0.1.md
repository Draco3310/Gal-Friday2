# Software Requirements Specification (SRS)

**Project: Gal-Friday**

**Version: 0.1**

**Date: 2025-04-27**

**Status: Draft**

---

**Table of Contents:**

1.  **Introduction**
    1.1 Purpose
    1.2 Document Conventions
    1.3 Intended Audience and Reading Suggestions
    1.4 Project Scope
    1.5 References
2.  **Overall Description**
    2.1 Product Perspective
    2.2 Product Features Summary
    2.3 User Classes and Characteristics
    2.4 Operating Environment
    2.5 Design and Implementation Constraints
    2.6 User Documentation (Assumptions)
    2.7 Assumptions and Dependencies
3.  **System Features (Functional Requirements)**
    3.1 Feature 1: Market Data Ingestion
    3.2 Feature 2: Feature Engineering
    3.3 Feature 3: Predictive Modeling
    3.4 Feature 4: Strategy Arbitration & Signal Generation
    3.5 Feature 5: Risk Management
    3.6 Feature 6: Order Execution
    3.7 Feature 7: Portfolio Management
    3.8 Feature 8: Logging & Auditing
    3.9 Feature 9: Monitoring & HALT/Recovery
    3.10 Feature 10: Backtesting & Simulation
4.  **External Interface Requirements**
    4.1 User Interfaces
    4.2 Hardware Interfaces
    4.3 Software Interfaces
    4.4 Communications Interfaces
5.  **Non-functional Requirements**
    5.1 Performance Requirements
    5.2 Safety Requirements
    5.3 Security Requirements
    5.4 Software Quality Attributes
    5.5 Business Rules
6.  **Other Requirements**
    6.1 Glossary
    6.2 Analysis Models (Reference)

---

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for Project Gal-Friday. Its purpose is to provide a detailed description of the system's functional and non-functional requirements. Gal-Friday is an automated cryptocurrency trading bot designed to execute high-frequency scalping and day trading strategies for XRP and DOGE on the Kraken exchange, utilizing AI/ML predictive models, with the primary objective of generating significant, consistent revenue ($75k/year target). This SRS serves as the primary input for the design, implementation, and testing phases of the project.

### 1.2 Document Conventions
The keywords "shall", "must", "should", "may" are used as defined in RFC 2119. "Shall" indicates a mandatory requirement. "Should" indicates a recommendation. "May" indicates an optional feature.
Requirements are uniquely identified with prefixes FR (Functional Requirement) and NFR (Non-Functional Requirement) followed by a number.
Priority levels (e.g., High, Medium, Low) are assigned. tag indicates requirement is part of the initial Minimum Viable Product scope.

### 1.3 Intended Audience and Reading Suggestions
This document is intended for:
* **Project Lead/Creative Director (User):** For understanding system capabilities, scope, and providing feedback.
* **Development Team (AI):** As the specification for design and implementation.
* **Testing Team:** As the basis for creating test plans and test cases.
* **Future Maintainers:** For understanding system functionality.
Readers should start with the Introduction and Overall Description for context, then dive into System Features (Section 3) and Non-functional Requirements (Section 5) for detailed specifications.

### 1.4 Project Scope
The Gal-Friday system will automate the trading process for XRP/USD and DOGE/USD pairs on the Kraken cryptocurrency exchange. Key capabilities include:
* Real-time ingestion of Level 2 order book and OHLCV market data.
* Calculation of relevant technical indicators and order book features.
* Utilization of Machine Learning models (XGBoost, RandomForest, LSTM) for price movement prediction.
* Strategy logic for generating trade signals based on predictions.
* Rigorous pre-trade risk management including position sizing based on defined capital and risk tolerance ($100k capital, 15% max drawdown, 0.5-1% risk/trade).
* Automated order execution via Kraken APIs (Limit and Market orders).
* Comprehensive logging, auditing, and performance tracking.
* System monitoring with automated HALT mechanisms.
* Realistic backtesting and paper trading simulation capabilities.

**Out of Scope (Initially):** Trading on exchanges other than Kraken, trading assets other than XRP/USD and DOGE/USD, development of a graphical user interface (GUI), strategies not based on the specified ML models, integration with external portfolio management tools outside the bot's internal tracking.

### 1.5 References
* Project Charter / PID (To be created - TBC)
* Risk Analysis Document (To be created - TBC)
* Kraken API Documentation (REST and WebSocket) - [Link to specific Kraken API documentation pages TBD]
* RFC 2119 (Keywords for Requirements) - [https://www.ietf.org/rfc/rfc2119.txt](https://www.ietf.org/rfc/rfc2119.txt)

---

## 2. Overall Description

### 2.1 Product Perspective
Gal-Friday is a self-contained, automated trading system. It operates as a client interacting with the external Kraken exchange via its public APIs. It does not require integration into larger existing systems beyond the operating system and database services it utilizes. It is a new product.

### 2.2 Product Features Summary
The major features of Gal-Friday include:
* Real-time Market Data Handling
* Data Feature Calculation Engine
* Multi-Model Predictive Analytics Engine
* Trading Strategy & Signal Generation Logic
* Comprehensive Risk Management Layer (Pre-trade & Portfolio Level)
* Automated Trade Execution Interface (Kraken)
* Real-time Portfolio & P&L Tracking
* Lifecycle Logging & Auditing System
* System Health Monitoring & Safety Halts
* Historical Simulation (Backtesting) & Paper Trading

### 2.3 User Classes and Characteristics
* **Administrator/Monitor (User - Project Lead):** Responsible for starting/stopping the bot, configuring parameters (risk settings, strategy thresholds), monitoring performance and logs, intervening during HALT conditions, managing API keys, and initiating model retraining. Assumed to have significant trading domain knowledge but not necessarily deep programming expertise (interaction via logs/config files/CLI initially).
* **System (The Bot itself):** Executes all automated functions, interacts directly with external APIs and internal modules.

### 2.4 Operating Environment
* **NFR-101:** Gal-Friday **shall** operate on a Linux-based Cloud Virtual Machine (e.g., AWS EC2, GCP Compute Engine, Azure VM). Specific distribution TBD (e.g., Ubuntu LTS). `[High]`
* **NFR-102:** The system **shall** utilize Python 3.9+ environment. `[High]`
* **NFR-103:** The system requires network connectivity to Kraken API endpoints (REST and WebSocket). `[High]`
* **NFR-104:** The system requires access to running PostgreSQL (v13+) and InfluxDB (v2.x+) database instances. `[High]`
* **NFR-105:** The cloud VM **should** be located geographically near Kraken's servers (e.g., EU or US East, depending on Kraken infrastructure) to minimize latency. `[Medium]`

### 2.5 Design and Implementation Constraints
* **NFR-106:** The system **must** use the official Kraken APIs for all exchange interactions. `[High]`
* **NFR-107:** The system **shall** be developed primarily in Python 3. `[High]`
* **NFR-108:** Specific Python libraries agreed upon (e.g., `ccxt`, `pandas`, `numpy`, `scikit-learn`, `tensorflow` or `pytorch`, `websockets`, `influxdb-client`, `psycopg2`/`SQLAlchemy`) **should** be used where appropriate. `[High]`
* **NFR-109:** API keys and sensitive configuration **must not** be hardcoded; they **shall** be managed securely (e.g., environment variables, configuration files with restricted permissions, secrets management service). `[High]`
* **NFR-110:** Development **shall** follow the principles outlined in the Project Plan (TBC) and adhere to SDLC best practices. `[High]`
* **NFR-111:** The system design **should** prioritize modularity, testability, and extensibility using an event-driven architecture. `[High]`

### 2.6 User Documentation (Assumptions)
Initial user documentation will be limited to configuration file explanations (`README.md`), instructions for running the bot, guidance on interpreting logs, and potentially a simple command-line interface (CLI) for basic control. No dedicated user manual or GUI is planned for the MVP.

### 2.7 Assumptions and Dependencies
* **A-01:** Kraken APIs (REST & WebSocket) are available, stable, and function as documented during market hours. Rate limits are manageable.
* **A-02:** Market data provided by Kraken is accurate and timely enough for the intended strategies (scalping/day trading).
* **A-03:** The chosen cloud provider offers reliable VM performance and network connectivity.
* **A-04:** The selected Python libraries function correctly and are maintained.
* **A-05:** Historical data used for backtesting and model training is reasonably representative of future market conditions (this is a significant assumption with inherent risks).
* **A-06:** The $100k trading capital is available and allocated.
* **D-01:** The system depends on the availability and correctness of external Python libraries.
* **D-02:** The system depends on the availability and performance of the Kraken exchange platform and APIs.
* **D-03:** The system depends on the availability of the underlying operating system, database services, and network infrastructure.

---

## 3. System Features (Functional Requirements)

### 3.1 Feature 1: Market Data Ingestion
* **FR-101:** The system **shall** establish and maintain persistent WebSocket connections to the Kraken API. `[High]`
* **FR-102:** The system **shall** subscribe to the following WebSocket feeds for configured trading pairs (XRP/USD, DOGE/USD):
    * Level 2 Order Book updates ('book' feed). `[High]`
    * OHLCV data at 1-minute intervals ('ohlc-1' feed). `[High]`
* **FR-103:** The system **shall** parse incoming WebSocket messages for L2 book updates (asks, bids, checksums if provided) and OHLCV data. `[High]`
* **FR-104:** The system **shall** reconstruct and maintain a local, real-time representation of the L2 order book for each subscribed pair. `[High]`
* **FR-105:** The system **shall** detect WebSocket disconnections or errors and implement an automated reconnection strategy with exponential backoff. `[High]`
* **FR-106:** The system **shall** standardize the ingested data into a consistent internal format (e.g., Python objects or dictionaries) before further processing. `[High]`
* **FR-107:** The system **shall** publish or make available the standardized, real-time L2 book state and OHLCV data to downstream modules (Feature Engine, Logger) via an internal event bus or queue. `[High]`
* **FR-108:** The system **should** handle potential data integrity issues (e.g., sequence gaps, checksum failures if applicable) and log warnings. `[Medium]`
* **FR-109:** (Future) The system **may** allow integration with news/sentiment API providers. `[Low]`

### 3.2 Feature 2: Feature Engineering
* **FR-201:** The system **shall** consume standardized OHLCV data (1-min). `[High]`
* **FR-202:** The system **shall** calculate a configurable set of technical indicators based on OHLCV data, including but not limited to:
    * Relative Strength Index (RSI). `[High]`
    * Moving Average Convergence Divergence (MACD). `[Medium]`
    * Bollinger Bands. `[Medium]`
    * Volume Weighted Average Price (VWAP) - Requires trade data if not in OHLCV. `[Medium]`
    * Price Rate of Change / Momentum. `[High]`
    * Volatility measures (e.g., ATR, standard deviation of returns). `[Medium]`
* **FR-203:** The system **shall** consume the local L2 order book state. `[High]`
* **FR-204:** The system **shall** calculate a configurable set of order book features, including but not limited to:
    * Bid-Ask Spread (absolute and percentage). `[High]`
    * Order Book Imbalance (ratio of volume within N levels or price range). `[High]`
    * Weighted Average Price (WAP) for bid/ask sides. `[Medium]`
    * Depth at N levels. `[Medium]`
* **FR-205:** The system **shall** calculate volume/trade flow indicators (e.g., Volume Delta - requires real-time trade feed if available, or estimated from OHLCV). `[Medium]`
* **FR-206:** The system **shall** make the calculated features available in a structured format (e.g., feature vector event) for the Predictive Modeling module. `[High]`
* **FR-207:** The system **shall** allow configuration of parameters for all calculated features (e.g., RSI period, Bollinger Band deviation) via a configuration file. `[High]`

### 3.3 Feature 3: Predictive Modeling
* **FR-301:** The system **shall** load pre-trained machine learning models from persistent storage upon startup. `[High]`
* **FR-302:** The system **shall** support loading models compatible with Scikit-learn (for RF/XGBoost) and TensorFlow/PyTorch (for LSTM). `[High]`
* **FR-303:** The system **shall** consume the latest calculated features from the Feature Engineering module. `[High]`
* **FR-304:** The system **shall** preprocess features as required by the specific loaded model(s) (e.g., scaling, normalization). `[High]`
* **FR-305:** The system **shall** use the loaded models to generate predictions. The primary prediction target is the probability of the price moving up or down by a configurable threshold (e.g., 0.1%) within a configurable future time window (e.g., 5 minutes). `[High]`
* **FR-306:** The system **shall** support using multiple models concurrently (e.g., one XGBoost, one LSTM). `[Medium]`
* **FR-307:** The system **shall** implement a mechanism to combine predictions from multiple models into a single actionable signal (e.g., weighted averaging based on confidence, simple voting). `[Medium]` (MVP will use single model output initially)
* **FR-308:** The system **shall** publish the final prediction probability (or combined signal) for the Strategy module. `[High]`
* **FR-309:** The system **shall** include a pipeline script or functionality for retraining models using stored historical feature data. `[High]`
* **FR-310:** The retraining pipeline **shall** allow configuration of the training data window (e.g., use last 90 days of data). `[High]`
* **FR-311:** The system **shall** implement model validation procedures (e.g., walk-forward validation using historical data) to assess model performance before deployment. `[High]`
* **FR-312:** The retraining pipeline **should** run on a configurable schedule (e.g., daily, weekly) via external scheduling (e.g., cron). `[Medium]`

### 3.4 Feature 4: Strategy Arbitration & Signal Generation
* **FR-401:** The system **shall** consume the final prediction probability from the Predictive Modeling module. `[High]`
* **FR-402:** The system **shall** apply configurable strategy rules to generate trade entry signals (BUY or SELL). A primary rule **shall** be based on the prediction probability crossing a configurable threshold (e.g., > 65% for BUY, < 35% [or >65% for DOWN] for SELL). `[High]`
* **FR-403:** The system **may** incorporate secondary confirmation conditions using features from the Feature Engine (e.g., require positive momentum for BUY signal). `[Medium]`
* **FR-404:** The system **shall**, upon generating an entry signal, determine preliminary Stop-Loss (SL) and Take-Profit (TP) price levels based on configurable rules (e.g., fixed percentage below/above entry, ATR multiple, price target implied by prediction). `[High]`
* **FR-405:** The system **shall** generate a candidate trade order event containing: Asset (XRP/USD or DOGE/USD), Side (BUY/SELL), preliminary SL price, preliminary TP price. `[High]`
* **FR-406:** The system **shall** publish the candidate trade order event to the Risk Management module for assessment. `[High]`
* **FR-407:** The system **shall** define logic for trade exits based on monitoring market price relative to active SL/TP levels associated with open positions. `[High]`
* **FR-408:** The system **may** support additional exit conditions:
    * Time-based exit (configurable maximum holding period). `[Medium]`
    * Exit based on model prediction reversal (configurable). `[Medium]`
* **FR-409:** (Future) The system **may** dynamically adjust strategy parameters (e.g., probability thresholds, risk/reward ratios) based on real-time market volatility or model confidence metrics. `[Low]`

### 3.5 Feature 5: Risk Management
* **FR-501:** The system **shall** consume candidate trade order events from the Strategy module. `[High]`
* **FR-502:** The system **shall** access the current portfolio state (equity, open positions, overall exposure) from the Portfolio Management module. `[High]`
* **FR-503:** The system **shall** enforce the maximum portfolio drawdown limits (Total: 15%, Daily: 2%, Weekly: 5% of starting capital for that period). If a limit is breached, no new trades **shall** be initiated (HALT condition FR-905). `[High]`
* **FR-504:** The system **shall** enforce a limit on the maximum number of consecutive losing trades (configurable, e.g., 5). If breached, no new trades **shall** be initiated (HALT condition FR-905). `[Medium]`
* **FR-505:** The system **shall** calculate the final position size for the candidate trade using the Fixed Fractional method: `PositionSizeInQuoteCurrency = (CurrentEquity * RiskPercentPerTrade)` and `PositionSizeInBaseCurrency = PositionSizeInQuoteCurrency / EstimatedEntryPrice`. The actual position size placed will be `PositionSizeInQuoteCurrency` divided by the distance to the stop-loss in quote currency per unit of base currency: `PositionSize = (CurrentEquity * RiskPercentPerTrade) / StopLossDistancePerUnit`. RiskPercentPerTrade **shall** be configurable (0.5% - 1.0%). `[High]`
* **FR-506:** The system **shall** perform the following pre-trade checks before approving any order:
    * Check if calculated position size exceeds a maximum configurable percentage of equity per asset (e.g., 10%). `[High]`
    * Check if the new position would cause total portfolio exposure (sum of all open positions' value) to exceed a configurable percentage of equity (e.g., 25%). `[High]`
    * Check if sufficient free balance/margin is available on Kraken to open the position. `[High]`
    * Check if the calculated position size exceeds a maximum allowable order size (sanity check, e.g., max $10,000 per order). `[High]`
    * Perform a 'fat finger' check: Ensure the proposed entry price is within a reasonable percentage (e.g., +/- 5%) of the current market price. `[High]`
* **FR-507:** If all checks pass, the system **shall** approve the trade, attaching the calculated position size, and generate an approved order event for the Execution Handler. `[High]`
* **FR-508:** If any check fails, the system **shall** reject the trade signal and log the reason. `[High]`

### 3.6 Feature 6: Order Execution
* **FR-601:** The system **shall** consume approved order events (Asset, Side, Size, Entry Target, SL, TP) from the Risk Management module. `[High]`
* **FR-602:** The system **shall** interact with the Kraken REST API for placing, canceling, and querying orders. `[High]`
* **FR-603:** The system **should** utilize the Kraken private WebSocket feed (if available and suitable) for receiving real-time updates on its own orders and fills to minimize latency. `[Medium]`
* **FR-604:** For trade entry, the system **shall** primarily attempt to place Limit orders at a price slightly better than the current market (e.g., join bid for BUY, join ask for SELL) or at the price suggested by the strategy. `[High]`
* **FR-605:** The system **shall** implement a configurable timeout for entry Limit orders (e.g., 15 seconds). If not filled within the timeout, the system **should** either cancel the order or switch to a Market order (configurable behavior). `[Medium]`
* **FR-606:** Upon successful entry fill (or partial fill), the system **shall** place associated exit orders:
    * Stop-Loss order: **Shall** use a Market order type (e.g., `stop-loss` order type on Kraken) triggered at the SL price. `[High]`
    * Take-Profit order: **Shall** use a Limit order type (e.g., `take-profit` order type on Kraken) at the TP price. `[High]`
    * (The system **should** use Order-Cancels-Other (OCO) functionality if Kraken API supports it reliably for this; otherwise, it must manage the cancellation of the remaining order when one is filled). `[Medium]`
* **FR-607:** The system **shall** handle partial fills for entry orders: update the current position state accordingly, place proportionally sized SL/TP orders for the filled amount, and potentially adjust the remaining open entry order based on configuration. `[High]`
* **FR-608:** The system **shall** monitor the status of open orders and positions to trigger SL/TP logic based on market price movements and fill events. `[High]`
* **FR-609:** The system **shall** handle API errors (e.g., insufficient funds, invalid parameters, rate limits) and order rejections gracefully, log the event, and potentially trigger alerts or HALT conditions. `[High]`
* **FR-610:** The system **shall** publish detailed execution report events (order submissions, cancellations, fills with price/quantity/fees, errors) for logging and portfolio updates. `[High]`

### 3.7 Feature 7: Portfolio Management
* **FR-701:** The system **shall** consume execution report events from the Execution Handler. `[High]`
* **FR-702:** The system **shall** maintain an accurate, real-time internal record of:
    * Current cash balance (USD). `[High]`
    * Current positions for each traded asset (XRP, DOGE), including quantity and average entry price. `[High]`
* **FR-703:** The system **shall** calculate and track the current equity of the portfolio (Cash Balance + Market Value of Open Positions based on latest market price). `[High]`
* **FR-704:** The system **shall** calculate and track realized P&L (from closed trades) and unrealized P&L (for open positions). `[High]`
* **FR-705:** The system **shall** provide the current portfolio state (equity, positions, exposure) to the Risk Management module accurately and promptly when needed for pre-trade checks. `[High]`
* **FR-706:** The system **should** periodically (e.g., every hour) reconcile its internal state with the actual account balance and positions reported by the Kraken API via REST calls. `[Medium]`

### 3.8 Feature 8: Logging & Auditing
* **FR-801:** The system **shall** log all major system events: startup, shutdown, connections established/lost, configuration loaded, HALT triggers, errors encountered. `[High]`
* **FR-802:** The system **shall** log the complete lifecycle of every trade attempt:
    * Signal generation (prediction value, threshold, decision). `[High]`
    * Risk assessment results (checks performed, position size calculated, approval/rejection). `[High]`
    * Order submission details (API request). `[High]`
    * Order status updates (acknowledgements, fills, cancellations, rejections) including API responses. `[High]`
    * Final realized P&L for closed trades. `[High]`
* **FR-803:** The system **shall** log key features and model prediction values associated with each trade signal. `[High]`
* **FR-804:** Logs **shall** be timestamped accurately (UTC recommended) with millisecond precision. `[High]`
* **FR-805:** Logs **shall** be written in a structured format (e.g., JSON) to rolling log files (e.g., daily rotation, max size). `[High]`
* **FR-806:** Critical data points (e.g., trade details, order fills, final P&L, daily performance summaries) **shall** be persisted to the PostgreSQL database for analysis and auditing. `[High]`
* **FR-807:** Time-series data relevant for analysis (e.g., selected market data snapshots, calculated features, model predictions over time, portfolio equity over time) **should** be stored in the InfluxDB database. `[Medium]`

### 3.9 Feature 9: Monitoring & HALT/Recovery
* **FR-901:** The system **shall** continuously monitor the status of its connection to Kraken APIs (WebSocket and REST availability via periodic checks). `[High]`
* **FR-902:** The system **shall** monitor the freshness of incoming market data feeds. If no data is received for a configurable period (e.g., 60 seconds), it indicates a problem. `[High]`
* **FR-903:** The system **should** monitor basic host system resource usage (CPU, RAM, Disk) and log warnings if thresholds are exceeded. `[Medium]`
* **FR-904:** The system **shall** monitor portfolio equity and drawdown against the defined daily, weekly, and total limits (see FR-503). `[High]`
* **FR-905:** The system **shall** implement automated HALT triggers that immediately cease the placement of *new* trade entries under the following conditions:
    * Breaching daily, weekly, or total drawdown limits. `[High]`
    * Experiencing a configurable number of consecutive losing trades (e.g., 5). `[Medium]`
    * Repeated critical API errors (e.g., 3 consecutive order placement failures, authentication failures). `[High]`
    * Loss of reliable market data feed (staleness detected per FR-902). `[High]`
    * Detection of excessive market volatility (e.g., price change % exceeding configurable threshold in short period). `[Medium]`
    * Receiving an external HALT command (e.g., via CLI or signal). `[High]`
* **FR-906:** The behavior upon HALT regarding *existing* open positions **shall** be configurable:
    * Option 1 (Default): Close all open positions immediately via Market orders. `[High]`
    * Option 2: Let existing positions run with their predefined SL/TP orders active. `[Medium]`
* **FR-907:** Upon triggering a HALT condition, the system **shall** log the reason clearly and send a notification alert (initially via detailed log messages; email/SMS integration is a future enhancement). `[High]`
* **FR-908:** Following a HALT, the system **shall** require manual intervention (e.g., review by Administrator/User, explicit restart command via CLI/signal) before resuming the placement of new trades. No automated recovery for MVP. `[High]`

### 3.10 Feature 10: Backtesting & Simulation
* **FR-1001:** The system **shall** include a backtesting engine capable of simulating the trading strategy using historical market data (minimum OHLCV; L2 preferred if available). `[High]`
* **FR-1002:** The backtester **shall** process historical data sequentially, ensuring no look-ahead bias (using only data available up to the simulation point in time). `[High]`
* **FR-1003:** The backtester **shall** simulate trade execution realistically:
    * Account for Kraken trading fees (configurable maker/taker rates). `[High]`
    * Simulate slippage on market orders (using a configurable model, e.g., fixed percentage or volatility-based). `[High]`
    * Simulate potential delays in order acknowledgement/fills (configurable latency). `[Medium]`
    * Simulate limit order fill probability based on price interaction (e.g., must touch or cross limit price). `[High]`
* **FR-1004:** The backtester **shall** use the same Feature Engineering, Prediction (using historical models or retraining logic), Strategy, and Risk Management modules/logic as the live trading system to ensure consistency. `[High]`
* **FR-1005:** The backtester **shall** allow users to specify the historical date range, initial capital, trading pairs, and configuration parameters for the simulation via command-line arguments or a configuration file. `[High]`
* **FR-1006:** The backtester **shall** generate a detailed performance report including: P&L curve data (for plotting), total return %, Sharpe ratio, Sortino ratio, maximum drawdown (value and percentage), win rate, profit factor, average profit/loss per trade, trade count, exposure time. Output **should** be available in both console summary and CSV/JSON file formats. `[High]`
* **FR-1007:** The system **shall** support a 'paper trading' mode that connects to the live Kraken data feeds but uses the Kraken Sandbox/Demo environment API for order execution, or simulates fills internally if a sandbox is unavailable/unsuitable. `[High]`
* **FR-1008:** The system **should** allow replaying historical scenarios using detailed logs to debug past behavior. `[Medium]`

---

## 4. External Interface Requirements

### 4.1 User Interfaces
* **NFR-201:** The primary user interface for the Administrator/Monitor **shall** be via configuration files (e.g., YAML or JSON) for setting parameters and a command-line interface (CLI) for starting, stopping, checking status, and initiating HALT/Resume commands. `[High]`
* **NFR-202:** System output for the user **shall** primarily be through structured log files and database entries. `[High]`
* **NFR-203:** No Graphical User Interface (GUI) is required for the MVP. `[High]`

### 4.2 Hardware Interfaces
* **NFR-204:** The system interfaces with standard cloud VM hardware (CPU, RAM, Disk, Network Interface Card). No specialized hardware is required. `[High]`

### 4.3 Software Interfaces
* **NFR-301:** The system **shall** interface with the **Kraken REST API** for:
    * Placing Orders (New, Cancel)
    * Querying Order Status
    * Querying Account Balances & Positions
    * Querying Exchange Status / Trading Pair Info
    * Fetching Historical Data (for backtesting/retraining) `[High]`
* **NFR-302:** The system **shall** interface with the **Kraken WebSocket API** for:
    * Receiving real-time L2 Order Book data.
    * Receiving real-time OHLCV data.
    * (Optional/Should) Receiving real-time private order/fill updates. `[High]`
* **NFR-303:** The system **shall** interface with a **PostgreSQL database server** via standard SQL protocols (e.g., using psycopg2 or SQLAlchemy) for storing relational data (trades, orders, logs). `[High]`
* **NFR-304:** The system **shall** interface with an **InfluxDB database server** via its HTTP API (e.g., using influxdb-client) for storing time-series data. `[Medium]`
* **NFR-305:** The system **shall** operate on a **Linux Operating System**, interacting via standard POSIX system calls and interfaces. `[High]`
* **NFR-306:** The system **shall** utilize the **Python 3 interpreter** and its standard libraries, plus specified third-party libraries. `[High]`

### 4.4 Communications Interfaces
* **NFR-401:** All communication with the Kraken REST API **shall** use HTTPS. `[High]`
* **NFR-402:** All communication with the Kraken WebSocket API **shall** use Secure WebSockets (WSS). `[High]`
* **NFR-403:** Communication with PostgreSQL and InfluxDB databases **should** occur over a secure network (e.g., within a VPC or using TLS if databases are exposed). `[High]`
* **NFR-404:** The system requires outbound internet access on ports 443 (HTTPS) and potentially others specified by Kraken for WSS. `[High]`

---

## 5. Non-functional Requirements

### 5.1 Performance Requirements
* **NFR-501:** **Latency (Data Processing):** The time from receiving a market data event (WebSocket message) to generating a prediction based on it **should** be minimized, ideally under 100 milliseconds on average under normal load. `[Medium]`
* **NFR-502:** **Latency (Order Placement):** The time from an approved trade signal event to submitting the corresponding order via the Kraken REST API **should** be minimized, ideally under 50 milliseconds on average. `[Medium]`
* **NFR-503:** **Throughput:** The system **shall** be capable of processing the expected volume of L2 updates and OHLCV data from Kraken for two active trading pairs without falling behind or consuming excessive resources. `[High]`
* **NFR-504:** **Resource Usage:** The system **should** operate within reasonable CPU (<75% average utilization), RAM (<4GB recommended allocation initially), and Network I/O limits of the chosen cloud VM instance under normal operating conditions. `[Medium]`
* **NFR-505:** **Scalability:** While the MVP targets 2 pairs, the architecture **should** allow for potential future scaling to handle more trading pairs or data sources with additional resources (vertical scaling) or potentially horizontal scaling of stateless components. `[Medium]`

### 5.2 Safety Requirements
* **NFR-601:** The HALT mechanisms defined in FR-905 **must** function reliably to prevent excessive losses or trading under adverse conditions. `[High]`
* **NFR-602:** Risk management checks (FR-506) **must** be performed before every trade execution attempt. `[High]`
* **NFR-603:** In case of unhandled exceptions or critical component failures, the system **should** default to a safe state (e.g., cease new trading, log the error, potentially trigger HALT). `[High]`

### 5.3 Security Requirements
* **NFR-701:** Kraken API keys **must** be stored securely and not exposed in source code or logs. Use environment variables, secure configuration files, or a secrets management service. `[High]`
* **NFR-702:** API keys used for the bot **should** have permissions restricted to trading and querying balances/orders only; withdrawal permissions **must** be disabled. `[High]`
* **NFR-703:** Access to the host machine, databases, and logs **should** be restricted using standard security practices (firewalls, strong passwords, SSH keys, limited user privileges). `[High]`
* **NFR-704:** Dependencies (Python libraries, OS packages) **should** be kept up-to-date to patch known vulnerabilities. `[Medium]`

### 5.4 Software Quality Attributes
* **NFR-801:** **Reliability:** The system **shall** operate continuously during configured trading hours with minimal downtime. Automated reconnection (FR-105) and robust error handling (FR-609) are essential. Target uptime > 99.5% during active trading periods. `[High]`
* **NFR-802:** **Maintainability:** The code **shall** be well-commented, follow consistent coding standards (e.g., PEP 8), and be organized into logical modules to facilitate understanding and future modifications. `[High]`
* **NFR-803:** **Testability:** The system design **shall** support unit testing of individual modules and integration testing of workflows. Backtesting (Feature 10) serves as a key system-level test. `[High]`
* **NFR-804:** **Configurability:** Key parameters (risk settings, strategy thresholds, API endpoints, feature parameters, file paths) **shall** be externalized into configuration files, not hardcoded. `[High]`
* **NFR-805:** **Extensibility:** The modular design **should** allow for adding new prediction models, feature calculations, strategy components, or exchange interfaces with reasonable effort in the future. `[Medium]`

### 5.5 Business Rules
* **NFR-901:** Trading **shall** only occur on the specified pairs: XRP/USD, DOGE/USD. `[High]`
* **NFR-902:** Trading **shall** only occur on the Kraken exchange. `[High]`
* **NFR-903:** All risk parameters (drawdown limits, risk per trade %) **must** be adhered to as defined (FR-503, FR-505). `[High]`
* **NFR-904:** The system **must** account for Kraken's trading fees when calculating P&L and potentially in decision-making (e.g., ensuring profit targets exceed fees). `[High]`

---

## 6. Other Requirements

### 6.1 Glossary
*(To be populated during design and development)*
* **API:** Application Programming Interface
* **ATR:** Average True Range
* **CLI:** Command-Line Interface
* **DFD:** Data Flow Diagram
* **HALT:** A state where the bot ceases initiating new trades due to predefined risk or error conditions.
* **HFT:** High-Frequency Trading
* **InfluxDB:** Time-series database.
* **JSON:** JavaScript Object Notation
* **L2 Data:** Level 2 Order Book Data (bids and asks)
* **LSTM:** Long Short-Term Memory (a type of recurrent neural network)
* **MACD:** Moving Average Convergence Divergence
* **MVP:** Minimum Viable Product
* **OHLCV:** Open, High, Low, Close, Volume (candlestick data)
* **P&L:** Profit and Loss
* **PID:** Project Initiation Document
* **PostgreSQL:** Relational database.
* **REST:** Representational State Transfer (API style)
* **RF:** Random Forest (machine learning model)
* **RSI:** Relative Strength Index
* **RTM:** Requirements Traceability Matrix
* **SDLC:** Software Development Life Cycle
* **SL:** Stop-Loss
* **SRS:** Software Requirements Specification
* **TP:** Take-Profit
* **UTC:** Coordinated Universal Time
* **VM:** Virtual Machine
* **VWAP:** Volume Weighted Average Price
* **WAP:** Weighted Average Price
* **WebSocket:** A persistent, full-duplex communication protocol.
* **WSS:** Secure WebSocket
* **XGBoost:** Extreme Gradient Boosting (machine learning model)
* **YAML:** YAML Ain't Markup Language (configuration file format)

### 6.2 Analysis Models (Reference)
*(This section will reference diagrams created during design, e.g., Use Case Diagrams, Data Flow Diagrams (DFDs), Architecture Diagrams)*
* Use Case Diagrams: (Link/Reference TBD)
* Data Flow Diagrams (Level 0, Level 1): (Link/Reference TBD)
* System Architecture Diagram: (Link/Reference TBD)

---
**End of Document**
