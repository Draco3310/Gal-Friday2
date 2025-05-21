\# Database Schema Definition

\*\*Project: Gal-Friday\*\*

\*\*Version: 0.1\*\*

\*\*Date: 2025-04-27\*\*

\*\*Status: Draft\*\*

\---

\*\*Table of Contents:\*\*

1\.  Introduction
2\.  PostgreSQL Schema (Relational Data)
    2.1 Table: \`signals\`
    2.2 Table: \`orders\`
    2.3 Table: \`fills\`
    2.4 Table: \`trades\`
    2.5 Table: \`system\_logs\`
    2.6 Table: \`portfolio\_snapshots\`
    2.7 Table: \`configurations\` (Optional)
3\.  InfluxDB Schema (Time-Series Data)
    3.1 Measurement: \`market\_data\_l2\`
    3.2 Measurement: \`market\_data\_ohlcv\`
    3.3 Measurement: \`features\`
    3.4 Measurement: \`predictions\`
    3.5 Measurement: \`portfolio\_metrics\`
    3.6 Measurement: \`system\_metrics\`
4\.  Data Types and Conventions

\---

\#\# 1\. Introduction

This document defines the database schemas for Project Gal-Friday. It details the structure for storing persistent data required for operation, auditing, analysis, and recovery.
\* \*\*PostgreSQL:\*\* Used for relational and transactional data that requires strong consistency, such as trade records, order lifecycle details, and system logs.
\* \*\*InfluxDB:\*\* Used for time-series data, optimized for high write/query performance on time-stamped events like market data, calculated features, predictions, and performance metrics.

\#\# 2\. PostgreSQL Schema (Relational Data)

\*(Note: SQL syntax is illustrative. \`TEXT\` can be replaced with \`VARCHAR(n)\` where appropriate. \`NUMERIC\` precision should be defined based on expected price/quantity scales. Timestamps should ideally be \`TIMESTAMPTZ\` \- timestamp with time zone, stored in UTC.)\*

\#\#\# 2.1 Table: \`signals\`
Stores information about trade signals proposed by the strategy arbitrator and their outcome from the risk manager.

\`\`\`sql
CREATE TABLE signals (
    signal\_id UUID PRIMARY KEY,               \-- Unique identifier for the signal proposal (from event payload)
    trading\_pair VARCHAR(16) NOT NULL,        \-- e.g., 'XRP/USD'
    exchange VARCHAR(32) NOT NULL,           \-- e.g., 'kraken'
    strategy\_id VARCHAR(64) NOT NULL,        \-- Identifier for the strategy generating the signal
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')), \-- 'BUY' or 'SELL'
    entry\_type VARCHAR(10) NOT NULL,         \-- e.g., 'LIMIT', 'MARKET'
    proposed\_entry\_price NUMERIC,            \-- Nullable if market order
    proposed\_sl\_price NUMERIC NOT NULL,
    proposed\_tp\_price NUMERIC NOT NULL,
    prediction\_event\_id UUID,                \-- FK to a potential prediction event log (optional)
    prediction\_value REAL,                   \-- Prediction probability/value that triggered signal
    status VARCHAR(10) NOT NULL CHECK (status IN ('PROPOSED', 'APPROVED', 'REJECTED')), \-- Outcome from RiskManager
    rejection\_reason TEXT,                   \-- Null if status is not 'REJECTED'
    risk\_check\_details JSONB,                \-- Details of risk checks performed (optional)
    created\_at TIMESTAMPTZ NOT NULL DEFAULT NOW() \-- Timestamp when the signal was proposed
);
CREATE INDEX idx\_signals\_created\_at ON signals(created\_at);
CREATE INDEX idx\_signals\_trading\_pair\_status ON signals(trading\_pair, status);

### **2.2 Table: orders**

Stores details about orders submitted to the exchange.

CREATE TABLE orders (
    order\_pk SERIAL PRIMARY KEY,             \-- Internal primary key
    client\_order\_id UUID UNIQUE NOT NULL,    \-- Internal unique order ID sent to exchange (if possible/used)
    exchange\_order\_id VARCHAR(64) UNIQUE,    \-- Order ID received from Kraken (may not be available immediately)
    signal\_id UUID REFERENCES signals(signal\_id), \-- Link back to the originating signal
    trading\_pair VARCHAR(16) NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order\_type VARCHAR(16) NOT NULL,         \-- e.g., 'LIMIT', 'MARKET', 'STOP\_LOSS', 'TAKE\_PROFIT'
    quantity\_ordered NUMERIC NOT NULL,       \-- Size of the order in base currency
    limit\_price NUMERIC,                     \-- Limit price (if applicable)
    stop\_price NUMERIC,                      \-- Stop price (if applicable)
    status VARCHAR(20) NOT NULL,             \-- Current known status (e.g., 'PENDING\_NEW', 'NEW', 'PARTIALLY\_FILLED', 'FILLED', 'PENDING\_CANCEL', 'CANCELED', 'REJECTED', 'EXPIRED')
    error\_message TEXT,                      \-- Error details if rejected by exchange or internal error
    created\_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), \-- Timestamp when the order was created internally
    submitted\_at TIMESTAMPTZ,                \-- Timestamp when the order was sent to the exchange
    last\_updated\_at TIMESTAMPTZ NOT NULL DEFAULT NOW() \-- Timestamp of the last status update
);
CREATE INDEX idx\_orders\_signal\_id ON orders(signal\_id);
CREATE INDEX idx\_orders\_exchange\_order\_id ON orders(exchange\_order\_id);
CREATE INDEX idx\_orders\_status ON orders(status);
CREATE INDEX idx\_orders\_created\_at ON orders(created\_at);

### **2.3 Table: fills**

Stores details about individual fills received for orders. An order can have multiple fills.

CREATE TABLE fills (
    fill\_pk SERIAL PRIMARY KEY,              \-- Internal primary key
    fill\_id VARCHAR(64),                     \-- Unique fill/trade ID from Kraken (if available)
    order\_pk INTEGER NOT NULL REFERENCES orders(order\_pk), \-- Link to the parent order
    exchange\_order\_id VARCHAR(64),           \-- Redundant for easier querying, link to orders table
    trading\_pair VARCHAR(16) NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity\_filled NUMERIC NOT NULL,        \-- Quantity filled in this specific execution
    fill\_price NUMERIC NOT NULL,             \-- Price at which this fill occurred
    commission NUMERIC NOT NULL,             \-- Fee paid for this fill
    commission\_asset VARCHAR(16) NOT NULL,   \-- Currency of the fee
    liquidity\_type VARCHAR(10),              \-- 'MAKER' or 'TAKER' (if provided by exchange)
    filled\_at TIMESTAMPTZ NOT NULL           \-- Timestamp of the fill from the exchange
);
CREATE INDEX idx\_fills\_order\_pk ON fills(order\_pk);
CREATE INDEX idx\_fills\_exchange\_order\_id ON fills(exchange\_order\_id);
CREATE INDEX idx\_fills\_filled\_at ON fills(filled\_at);
ALTER TABLE fills ADD CONSTRAINT uq\_fills\_exchange\_fill UNIQUE (exchange, fill\_id); \-- If exchange provides unique fill IDs

### **2.4 Table: trades**

Represents a completed trade cycle (entry fill(s) to exit fill(s)), summarizing P\&L.

CREATE TABLE trades (
    trade\_pk SERIAL PRIMARY KEY,             \-- Internal primary key
    trade\_id UUID UNIQUE NOT NULL,           \-- Unique internal ID for the trade cycle
    signal\_id UUID REFERENCES signals(signal\_id), \-- Link back to the originating signal
    trading\_pair VARCHAR(16) NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    strategy\_id VARCHAR(64) NOT NULL,
    side VARCHAR(4) NOT NULL,                \-- Side of the entry ('BUY' or 'SELL')
    entry\_order\_pk INTEGER REFERENCES orders(order\_pk), \-- Link to the entry order
    exit\_order\_pk INTEGER REFERENCES orders(order\_pk),  \-- Link to the exit order (SL, TP, or other)
    entry\_timestamp TIMESTAMPTZ NOT NULL,    \-- Timestamp of the first entry fill
    exit\_timestamp TIMESTAMPTZ NOT NULL,     \-- Timestamp of the last exit fill
    quantity NUMERIC NOT NULL,               \-- Total quantity traded (base currency)
    average\_entry\_price NUMERIC NOT NULL,
    average\_exit\_price NUMERIC NOT NULL,
    total\_commission NUMERIC NOT NULL,       \-- Sum of commissions for entry and exit fills (in quote currency)
    realized\_pnl NUMERIC NOT NULL,           \-- Profit or loss in quote currency (Exit Value \- Entry Value \- Commissions)
    realized\_pnl\_pct REAL NOT NULL,          \-- P\&L as a percentage of entry value (approx)
    exit\_reason VARCHAR(32) NOT NULL         \-- e.g., 'TP\_HIT', 'SL\_HIT', 'TIME\_EXIT', 'MANUAL\_CLOSE'
);
CREATE INDEX idx\_trades\_signal\_id ON trades(signal\_id);
CREATE INDEX idx\_trades\_entry\_timestamp ON trades(entry\_timestamp);
CREATE INDEX idx\_trades\_exit\_timestamp ON trades(exit\_timestamp);
CREATE INDEX idx\_trades\_trading\_pair ON trades(trading\_pair);

### **2.5 Table: system\_logs**

Stores structured log entries for important system events and errors.

CREATE TABLE system\_logs (
    log\_pk BIGSERIAL PRIMARY KEY,            \-- Use BIGSERIAL for high volume potential
    log\_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source\_module VARCHAR(64) NOT NULL,      \-- Module generating the log
    log\_level VARCHAR(10) NOT NULL CHECK (log\_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    trading\_pair VARCHAR(16),                \-- Context: Trading pair (if applicable)
    signal\_id UUID,                          \-- Context: Signal ID (if applicable)
    order\_pk INTEGER,                        \-- Context: Order PK (if applicable)
    exception\_type TEXT,                     \-- If log\_level is ERROR/CRITICAL
    stack\_trace TEXT,                        \-- If log\_level is ERROR/CRITICAL
    context JSONB                            \-- Additional arbitrary context
);
CREATE INDEX idx\_system\_logs\_log\_timestamp ON system\_logs(log\_timestamp);
CREATE INDEX idx\_system\_logs\_log\_level ON system\_logs(log\_level);
CREATE INDEX idx\_system\_logs\_source\_module ON system\_logs(source\_module);

### **2.6 Table: portfolio\_snapshots**

Stores periodic snapshots of the overall portfolio state for analysis and potential recovery points.

CREATE TABLE portfolio\_snapshots (
    snapshot\_pk SERIAL PRIMARY KEY,
    snapshot\_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total\_equity NUMERIC NOT NULL,           \-- In quote currency (e.g., USD)
    available\_balance NUMERIC NOT NULL,      \-- In quote currency
    total\_exposure\_pct REAL NOT NULL,        \-- Total value of positions / total\_equity
    daily\_drawdown\_pct REAL NOT NULL,
    weekly\_drawdown\_pct REAL NOT NULL,
    total\_drawdown\_pct REAL NOT NULL,
    positions JSONB NOT NULL                 \-- JSON object detailing current positions (pair, qty, entry\_price, market\_value, unrealized\_pnl)
);
CREATE UNIQUE INDEX idx\_portfolio\_snapshots\_timestamp ON portfolio\_snapshots(snapshot\_timestamp);

### **2.7 Table: configurations (Optional)**

Stores versions of the system configuration used, allowing correlation between trades/logs and specific settings.

CREATE TABLE configurations (
    config\_pk SERIAL PRIMARY KEY,
    config\_hash VARCHAR(64) UNIQUE NOT NULL, \-- SHA-256 hash of the configuration content
    config\_content JSONB NOT NULL,           \-- The actual configuration parameters
    loaded\_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is\_active BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE INDEX idx\_configurations\_loaded\_at ON configurations(loaded\_at);
\-- Add logic to ensure only one config is active at a time if needed.

## **3\. InfluxDB Schema (Time-Series Data)**

InfluxDB stores data in "measurements," which are analogous to SQL tables. Data points have a timestamp, "tags" (indexed metadata used for grouping/filtering), and "fields" (the actual data values, not indexed).

### **3.1 Measurement: market\_data\_l2**

Stores Level 2 order book updates or snapshots.

* **Measurement Name:** market\_data\_l2
* **Tags:**
  * exchange (String, e.g., "kraken")
  * trading\_pair (String, e.g., "XRP/USD")
* **Fields:**
  * bid\_price\_level\_N (Float, e.g., bid\_price\_level\_0, bid\_price\_level\_1, ...)
  * bid\_volume\_level\_N (Float)
  * ask\_price\_level\_N (Float)
  * ask\_volume\_level\_N (Float)
  * is\_snapshot (Boolean)
  * spread (Float)
  * mid\_price (Float)
  * wap (Float, Weighted Average Price)
  * imbalance\_N (Float, Order book imbalance calculated over N levels)
* Timestamp: Precision likely nanoseconds or milliseconds (from exchange or ingestion time).
  (Note: Storing full L2 data can be very high volume. Consider storing only key derived metrics or sampling.)

### **3.2 Measurement: market\_data\_ohlcv**

Stores OHLCV candlestick data.

* **Measurement Name:** market\_data\_ohlcv
* **Tags:**
  * exchange (String)
  * trading\_pair (String)
  * interval (String, e.g., "1m", "5m")
* **Fields:**
  * open (Float)
  * high (Float)
  * low (Float)
  * close (Float)
  * volume (Float)
* **Timestamp:** Start time of the OHLCV bar.

### **3.3 Measurement: features**

Stores calculated features used for prediction models.

* **Measurement Name:** features
* **Tags:**
  * exchange (String)
  * trading\_pair (String)
  * feature\_set\_id (String, e.g., "default\_v1")
* **Fields:**
  * rsi\_14 (Float)
  * spread\_pct (Float)
  * book\_imbalance\_5 (Float)
  * momentum\_5 (Float)
  * *(... other calculated features as floats)*
* **Timestamp:** Timestamp the features correspond to (aligned with market data).

### **3.4 Measurement: predictions**

Stores the output of the prediction models.

* **Measurement Name:** predictions
* **Tags:**
  * exchange (String)
  * trading\_pair (String)
  * model\_id (String)
  * prediction\_target (String)
* **Fields:**
  * prediction\_value (Float)
  * confidence (Float, optional)
* **Timestamp:** Timestamp the prediction corresponds to (aligned with features).

### **3.5 Measurement: portfolio\_metrics**

Stores time-series data about the portfolio's state and performance.

* **Measurement Name:** portfolio\_metrics
* **Tags:**
  * account\_id (String, if managing multiple accounts in future)
* **Fields:**
  * total\_equity (Float)
  * available\_balance (Float)
  * unrealized\_pnl (Float)
  * realized\_pnl\_session (Float, P\&L since bot start/day start)
  * total\_exposure\_pct (Float)
  * daily\_drawdown\_pct (Float)
  * weekly\_drawdown\_pct (Float)
  * total\_drawdown\_pct (Float)
  * position\_qty\_XRPUSD (Float, example field per asset)
  * position\_value\_XRPUSD (Float, example field per asset)
* **Timestamp:** Time of the metric calculation.

### **3.6 Measurement: system\_metrics**

Stores performance and health metrics of the bot itself.

* **Measurement Name:** system\_metrics
* **Tags:**
  * host\_id (String, identifier for the VM instance)
  * module\_id (String, optional, specific module if metric is module-specific)
* **Fields:**
  * cpu\_usage\_pct (Float)
  * memory\_usage\_mb (Float)
  * event\_queue\_depth (Integer)
  * api\_call\_latency\_ms (Float, potentially per endpoint)
  * prediction\_latency\_ms (Float)
  * end\_to\_end\_latency\_ms (Float)
  * websocket\_connected (Boolean)
  * api\_error\_rate (Float)
* **Timestamp:** Time the metric was recorded.

## **4\. Data Types and Conventions**

* **Timestamps:** Use TIMESTAMPTZ in PostgreSQL, store in UTC. Use standard InfluxDB timestamp precision (default nanoseconds). ISO 8601 format for representation.
* **Prices/Quantities:** Use NUMERIC in PostgreSQL for precise financial calculations. Use Float (float64) in InfluxDB fields. Define appropriate precision for PostgreSQL NUMERIC types.
* **Identifiers:** Use UUID for internally generated unique IDs where possible. Use VARCHAR for external IDs (exchange order IDs, etc.).
* **Enums:** Use CHECK constraints in PostgreSQL for columns with fixed allowed values (e.g., side, status, log\_level).
* **JSONB:** Use PostgreSQL's JSONB type for storing semi-structured data like context, risk check details, or position snapshots.
* **Indexing:** Define appropriate indexes in PostgreSQL on columns frequently used in WHERE clauses, JOIN conditions, or \`
