# Models Folder (`gal_friday/models`) Documentation

## Folder Overview

The `gal_friday/models` folder is a critical part of the Data Access Layer (DAL) within the Gal-Friday trading system. Its primary purpose is to define the structure of data as it is persisted in the relational database (e.g., PostgreSQL) using SQLAlchemy Object Relational Mapper (ORM) models. These Python classes directly map to database tables and their columns, providing an object-oriented interface for all database interactions.

It's important to distinguish these SQLAlchemy ORM models from other types of models that might exist in the system, such as:
-   Pydantic models: Used for data validation, serialization/deserialization (e.g., in API request/response cycles or for event payloads).
-   Machine Learning models: The actual predictive models (e.g., scikit-learn, XGBoost, TensorFlow/Keras files) managed by the `model_lifecycle` components.

The models defined in this folder are exclusively for defining the database schema and interacting with the relational data store.

## Key Modules and Their Roles (SQLAlchemy Models)

Each Python file within this folder typically defines one or more SQLAlchemy ORM models, representing different entities within the Gal-Friday system.

### `base.py` (`Base`)

-   **Purpose:** This module (or a similarly named one like `models_base.py` if located in `dal/models/`) defines the `declarative_base()` from SQLAlchemy.
-   **Key Component:**
    -   `Base = declarative_base()`: This `Base` object serves as the foundation for all SQLAlchemy ORM models defined in this folder. All table models inherit from this `Base`. It collects metadata about all defined tables, which is then used by SQLAlchemy to interact with the database and by Alembic to generate schema migrations.

### `configuration.py` (`Configuration` model)

-   **Purpose:** Represents stored versions or snapshots of the application's configuration (e.g., content from `config.yaml` or dynamic configurations). This allows for auditing configuration changes or potentially rolling back to previous settings.
-   **Key Fields (Columns):**
    -   `config_pk (Integer, Primary Key)`: Unique identifier for the configuration record.
    -   `config_hash (String)`: A hash (e.g., SHA256) of the configuration content, used for quick comparisons and identifying changes.
    -   `config_content (JSON/Text)`: The actual configuration content, typically stored as a JSON string or text.
    -   `loaded_at (DateTime)`: Timestamp indicating when this configuration version was loaded or recorded.
    -   `is_active (Boolean)`: Flag indicating if this configuration version is currently the active one.

### `fill.py` (`Fill` model)

-   **Purpose:** Represents an individual trade execution (a "fill") that occurs as part of fulfilling an order. An order can have multiple fills, especially if it's large or executed as a "taker" order.
-   **Key Fields (Columns):**
    -   `fill_pk (Integer, Primary Key)`: Unique identifier for the fill record.
    -   `fill_id (String)`: The unique fill identifier provided by the exchange.
    -   `order_pk (Integer, ForeignKey("orders.order_pk"))`: Foreign key linking this fill to its parent `Order`.
    -   `exchange_order_id (String)`: The exchange's identifier for the parent order.
    -   `trading_pair (String)`: The trading pair (e.g., "BTC/USD").
    -   `exchange (String)`: The exchange where the fill occurred.
    -   `side (String)`: The side of the trade (e.g., "BUY", "SELL").
    -   `quantity_filled (Decimal)`: The amount of the base asset filled in this execution.
    -   `fill_price (Decimal)`: The price at which this portion of the order was filled.
    -   `commission (Decimal)`: The commission amount paid for this fill.
    -   `commission_asset (String)`: The asset in which the commission was paid.
    -   `liquidity_type (String, Optional)`: Indicates if the fill was a "MAKER" or "TAKER" of liquidity.
    -   `filled_at (DateTime)`: Timestamp of when the fill occurred.
-   **Relationships:**
    -   Belongs to an `Order` (many-to-one).

### `log.py` (`Log` model)

-   **Purpose:** Defines the database schema for storing structured application log entries. This model is primarily used by the `AsyncPostgresHandler` within the `LoggerService`.
-   **Key Fields (Columns):**
    -   `id (Integer, Primary Key)`: Unique identifier for the log entry.
    -   `timestamp (DateTime)`: Time when the log record was created.
    -   `logger_name (String)`: Name of the logger that emitted the record (often the module name).
    -   `level_name (String)`: Human-readable log level (e.g., "INFO", "ERROR").
    -   `level_no (Integer)`: Numeric log level.
    -   `message (Text)`: The main log message.
    -   `pathname (String)`: Path to the source file where the logging call was made.
    -   `filename (String)`: Name of the source file.
    -   `lineno (Integer)`: Line number in the source file.
    -   `func_name (String)`: Name of the function that made the logging call.
    -   `context_json (JSONB/Text)`: Additional contextual information, stored as a JSON string.
    -   `exception_text (Text, Optional)`: Full traceback if an exception was logged.

### `order.py` (`Order` model)

-   **Purpose:** Represents trading orders placed by the Gal-Friday system. It tracks the state and details of each order throughout its lifecycle.
-   **Key Fields (Columns):**
    -   `order_pk (Integer, Primary Key)`: Unique identifier for the order record in the database.
    -   `client_order_id (UUID/String, Unique)`: A unique identifier generated by Gal-Friday for tracking the order internally.
    -   `exchange_order_id (String, Optional, Index)`: The unique identifier assigned to the order by the exchange once it's accepted.
    -   `signal_id (UUID/String, ForeignKey("signals.signal_id"), Optional)`: Foreign key linking to the `Signal` that triggered this order.
    -   `trading_pair (String)`: The trading pair (e.g., "BTC/USD").
    -   `exchange (String)`: The exchange where the order was placed.
    -   `side (String)`: "BUY" or "SELL".
    -   `order_type (String)`: Type of order (e.g., "LIMIT", "MARKET", "STOP_LOSS_LIMIT").
    -   `quantity_ordered (Decimal)`: The original quantity of the base asset ordered.
    -   `limit_price (Decimal, Optional)`: The limit price for LIMIT orders.
    -   `stop_price (Decimal, Optional)`: The stop price for STOP orders.
    -   `status (String, Index)`: Current status of the order (e.g., "PENDING_SUBMIT", "OPEN", "FILLED", "PARTIALLY_FILLED", "CANCELLED", "REJECTED", "EXPIRED").
    -   `error_message (Text, Optional)`: Any error message received from the exchange or system related to this order.
    -   `created_at (DateTime)`: Timestamp when the order was created in Gal-Friday.
    -   `submitted_at (DateTime, Optional)`: Timestamp when the order was successfully submitted to the exchange.
    -   `last_updated_at (DateTime)`: Timestamp of the last update to this order record.
-   **Relationships:**
    -   May belong to a `Signal` (many-to-one, if orders are always tied to signals).
    -   Has many `Fill`s (one-to-many).

### `portfolio_snapshot.py` (`PortfolioSnapshot` model)

-   **Purpose:** Stores periodic or event-driven snapshots of the overall portfolio state, including equity, balances, exposure, and drawdown metrics. Useful for performance tracking and historical analysis of portfolio health.
-   **Key Fields (Columns):**
    -   `snapshot_pk (Integer, Primary Key)`: Unique identifier for the snapshot.
    -   `snapshot_timestamp (DateTime, Index)`: Timestamp when the snapshot was taken.
    -   `total_equity (Decimal)`: Total value of the portfolio in the valuation currency.
    -   `available_balance (JSONB/Text)`: JSON storing available balances per currency.
    -   `total_exposure_pct (Decimal)`: Overall market exposure as a percentage of equity.
    -   Drawdown percentages (e.g., `daily_drawdown_pct`, `weekly_drawdown_pct`, `max_drawdown_pct` as `Decimal`).
    -   `positions (JSONB/Text)`: JSON representation of all open positions at the time of the snapshot, including their market value and unrealized P&L.

### `signal.py` (`Signal` model)

-   **Purpose:** Represents trading signals generated by the system's strategies, before they are necessarily acted upon or converted into orders. Tracks the signal's parameters and its outcome (e.g., whether it was approved or rejected by risk management).
-   **Key Fields (Columns):**
    -   `signal_id (UUID/String, Primary Key)`: Unique identifier for the signal.
    -   `trading_pair (String)`: The target trading pair.
    -   `exchange (String)`: The target exchange.
    -   `strategy_id (String)`: Identifier of the strategy that generated the signal.
    -   `side (String)`: Proposed trade direction ("BUY" or "SELL").
    -   `entry_type (String)`: Proposed order type ("LIMIT" or "MARKET").
    -   Proposed prices (`proposed_entry_price`, `proposed_sl_price`, `proposed_tp_price` as `Decimal`).
    -   `prediction_event_id (UUID/String, Optional)`: ID of the `PredictionEvent` that led to this signal.
    -   `prediction_value (JSONB/Text, Optional)`: Key prediction values or confidence scores.
    -   `status (String, Index)`: Status of the signal (e.g., "PROPOSED", "APPROVED", "REJECTED_RISK", "REJECTED_ERROR").
    -   `rejection_reason (Text, Optional)`: Reason if the signal was rejected.
    -   `risk_check_details (JSONB/Text, Optional)`: Details of risk checks performed.
    -   `created_at (DateTime)`: Timestamp of signal generation.
-   **Relationships:**
    -   May have one or more `Order`s (one-to-many, if a signal can lead to multiple orders like entry and SL/TP).
    -   May lead to one or more `Trade`s.

### `system_log.py` (`SystemLog` model)

-   **Purpose:** Defines a schema for more detailed or specialized system-level logs, potentially for specific subsystems or critical events that require more structured contextual information than general application logs.
-   **Key Fields (Columns):**
    -   `log_pk (Integer, Primary Key)`: Unique identifier.
    -   `log_timestamp (DateTime, Index)`: Timestamp of the log event.
    -   `source_module (String)`: The module or service that generated this log.
    -   `log_level (String)`: Severity level (e.g., "INFO", "CRITICAL", "AUDIT").
    -   `message (Text)`: The primary log message.
    -   Contextual foreign keys (Optional): `trading_pair (String)`, `signal_id (UUID/String)`, `order_pk (Integer)`.
    -   `exception_type (String, Optional)`: If the log is related to an error.
    -   `stack_trace (Text, Optional)`: Full stack trace for errors.
    -   `context (JSONB/Text)`: Rich structured context specific to the event.

### `trade.py` (`Trade` model)

-   **Purpose:** Represents a completed trade cycle, typically linking an entry execution (or set of fills) with one or more exit executions. This model is crucial for performance analysis and P&L calculation.
-   **Key Fields (Columns):**
    -   `trade_pk (Integer, Primary Key)`: Unique identifier for the trade record.
    -   `trade_id (UUID/String, Unique)`: A system-generated unique ID for the trade.
    -   `signal_id (UUID/String, ForeignKey("signals.signal_id"), Optional)`: The signal that initiated this trade.
    -   `trading_pair (String)`: The traded instrument.
    -   `exchange (String)`: The exchange where the trade occurred.
    -   `strategy_id (String)`: The strategy responsible for the trade.
    -   `side (String)`: The direction of the entry part of the trade (e.g., "LONG" if entered with a BUY, "SHORT" if entered with a SELL).
    -   `entry_order_pk (Integer, ForeignKey("orders.order_pk"), Optional)`: Link to the primary entry order.
    -   `exit_order_pk (Integer, ForeignKey("orders.order_pk"), Optional)`: Link to the primary exit order.
    -   Timestamps: `entry_timestamp (DateTime)`, `exit_timestamp (DateTime, Optional)`.
    -   `quantity (Decimal)`: The quantity of the base asset traded.
    -   Prices: `average_entry_price (Decimal)`, `average_exit_price (Decimal, Optional)`.
    -   `total_commission (Decimal)`: Sum of commissions for all fills related to this trade.
    -   `realized_pnl (Decimal, Optional)`: Profit or Loss for this completed trade.
    -   `realized_pnl_pct (Decimal, Optional)`: P&L as a percentage of invested capital/risk.
    -   `exit_reason (String, Optional)`: Reason for exiting the trade (e.g., "TAKE_PROFIT_HIT", "STOP_LOSS_HIT", "MANUAL_CLOSE").
-   **Relationships:**
    -   Associated with a `Signal`.
    -   Links to entry and exit `Order`s (or directly to `Fill`s if more granular).

### `__init__.py`

-   **Purpose:** Marks the `models` directory as a Python package.
-   **Key Aspects:**
    -   Allows modules within this directory to be imported using package notation (e.g., `from gal_friday.models.order import Order`).
    -   Crucially, it **exports all the defined ORM model classes** (e.g., `Order`, `Log`, `Signal`, `Fill`, `Trade`, `Configuration`, `PortfolioSnapshot`, `SystemLog`) and the `Base` object. This makes them easily accessible for other parts of the application, especially for the DAL repositories and Alembic migrations (e.g., `from gal_friday.models import Order, Base`).

## Database Interaction

-   **Data Access Layer (DAL):** These SQLAlchemy ORM models are the core of how the application interacts with its relational database (assumed to be PostgreSQL, given typical `asyncpg` usage with SQLAlchemy's async features). Services and components do not write raw SQL queries; instead, they use these models and SQLAlchemy sessions (obtained via `AsyncSessionFactory` from `gal_friday.database` or `DatabaseConnectionPool` from `gal_friday.dal.connection_pool`) to perform database operations.
-   **Repositories:** Specific repositories within the `gal_friday.dal.repositories/` directory (e.g., `OrderRepository`, `SignalRepository`) use these models to abstract the actual query logic. For example, `OrderRepository` would use the `Order` model to fetch, add, or update order records.
-   **Services:**
    -   `LoggerService` (via `AsyncPostgresHandler`) uses the `Log` model to persist log entries.
    -   `EventStore` (if implemented for persisting all system events) would use an `EventLog` model.
    -   `PortfolioManager` might use `PortfolioSnapshot` to save periodic states.
-   **Alembic Migrations:** The Alembic migration scripts located in `gal_friday/dal/alembic_env/versions/` are generated based on changes detected in these ORM model definitions. When a model is added, removed, or a column is changed, running `alembic revision -m "description" --autogenerate` compares the current model definitions against the database schema (as recorded by Alembic) and generates the necessary upgrade/downgrade scripts.

## Importance

-   **Structured Data Persistence:** These ORM models provide a clear, Pythonic, and structured way to define and interact with the database schema. They are fundamental for ensuring that all critical operational data, historical records, configuration snapshots, and system state are reliably stored and can be retrieved.
-   **Object-Oriented Database Interaction:** Developers can work with Python objects rather than writing SQL strings, which can reduce errors, improve code readability, and leverage Python's type system.
-   **Foundation for Data Integrity:** Relationships, constraints (like primary keys, foreign keys, uniqueness), and data types defined in the models help maintain data integrity at the database level.
-   **Centralized Schema Definition:** Having all database table definitions in one `models` folder provides a single source of truth for the relational database schema.

## Adherence to Standards

Using a robust ORM like SQLAlchemy for defining data models and interacting with the database is a standard practice in modern application development. It promotes:
-   **Maintainability:** Changes to the database schema can be managed through Python code and version-controlled.
-   **Database Agnosticism (to a degree):** While specific database features might be used, SQLAlchemy provides a layer of abstraction that can make it easier to switch underlying database systems if needed (though this is rarely trivial for complex systems).
-   **Reduced Boilerplate:** The ORM handles much of the repetitive SQL generation and result mapping.

This structured approach to data persistence is key to building a reliable and evolvable trading system.
