# DAL (Data Access Layer) Folder (`gal_friday/dal`) Documentation

## Folder Overview

The `gal_friday/dal` (Data Access Layer) folder is responsible for managing all aspects of data persistence, retrieval, and interaction with various data stores within the Gal-Friday trading system. It provides a structured and abstracted interface for other application services to interact with databases, primarily a relational database (likely PostgreSQL) through SQLAlchemy ORM, and a time-series database (InfluxDB) for specific data types. The DAL ensures that data access logic is centralized, maintainable, and consistent.

## Key Components and Their Roles

The `dal` folder is structured to separate concerns related to different aspects of data management:

### SQLAlchemy ORM Usage

The primary mechanism for interacting with the relational database is SQLAlchemy's Object Relational Mapper (ORM).

-   **`models/` directory:**
    -   **Purpose:** This directory contains all SQLAlchemy ORM model definitions. Each Python class in this directory typically maps to a database table.
    -   **Examples:**
        -   `Log`: Represents log entries stored in the database by `LoggerService`.
        -   `Order`: Represents trading orders (e.g., their status, price, quantity).
        -   `Position`: Represents open or closed trading positions.
        -   `EventLog`: Represents persisted system events from `EventStore`.
        -   `TradeSignal`: Could store details of proposed or approved trade signals.
        -   Other models for application entities like user configurations, strategy parameters, backtest results, etc.
    -   **Structure:** Each model class inherits from a declarative base (e.g., `Base`) and defines table columns as class attributes using SQLAlchemy's `Column` type and other constructs. Relationships between tables (e.g., one-to-many, many-to-many) are also defined here.

-   **`models/models_base.py` (or similar convention):**
    -   **Purpose:** Defines the `declarative_base` that all SQLAlchemy ORM models in the `models/` directory inherit from.
    -   **Example:** `Base = declarative_base()`
    -   This base class holds metadata about the tables and their mappings.

-   **`database.py` (located at `gal_friday/database.py`, but central to DAL operations):**
    -   **Purpose:** Initializes and provides access to the global asynchronous SQLAlchemy engine (`async_engine`) and a session factory (`AsyncSessionFactory`) for creating `AsyncSession` instances.
    -   **(Reference:** See `docs/Gal-Friday/Phase 4 - Full Implementation/System Documents/modules/database.md` for detailed documentation.)

### Connection Management

-   **`connection_pool.py` (class `DatabaseConnectionPool`):**
    -   **Purpose:** Centralizes the creation and management of the SQLAlchemy `AsyncEngine` and provides an `async_sessionmaker` instance.
    -   **Key Aspects:**
        -   Takes database connection parameters (URL, pool size, max overflow, echo flag) typically from `ConfigManager`.
        -   Initializes the `AsyncEngine` once.
        -   Provides a method to get an `async_sessionmaker` instance, which is then used by repositories or services to obtain `AsyncSession`s.
        -   May include methods to gracefully close the engine's connection pool during application shutdown.
    -   **Importance:** Ensures that database connection configurations are managed in one place and that all parts of the application use the same engine and session creation mechanism.

### Repository Pattern

The DAL employs the Repository pattern to abstract data access logic for specific entities (ORM models).

-   **`base.py` (class `BaseRepository`):**
    -   **Purpose:** A generic base class that provides common asynchronous CRUD (Create, Read, Update, Delete) operations for SQLAlchemy models.
    -   **Key Aspects:**
        -   Takes an `async_sessionmaker` in its constructor.
        -   Provides methods like `add(instance)`, `get(id)`, `update(instance)`, `delete(id)`, `find_all(criteria)`, `find_one_by(criteria)`. These methods internally use an `AsyncSession` obtained from the sessionmaker.
        -   Handles session management (begin, commit, rollback, close) within these methods or expects the calling service to manage the session lifecycle if a session is passed in.
    -   **Importance:** Reduces boilerplate code in specific repositories and enforces a consistent data access interface.

-   **`repositories/` directory:**
    -   **Purpose:** Contains concrete repository implementations for each major SQLAlchemy ORM model/entity.
    -   **Examples:**
        -   `OrderRepository(BaseRepository[Order])`: Provides methods specific to `Order` entities, such as `find_by_status(status)` or `get_open_orders_for_pair(pair)`.
        -   `PositionRepository(BaseRepository[Position])`: For `Position` specific queries.
        -   `ModelRepository` (if ML models or their metadata are stored in DB): For accessing ML model information.
        -   `LogRepository(BaseRepository[Log])`: For querying persisted log entries.
    -   **Structure:** Each repository class inherits from `BaseRepository`, specifying the SQLAlchemy model it manages. It then implements additional methods tailored to the query requirements for that entity, often using SQLAlchemy's expression language for complex queries.

### Database Migrations (Alembic)

Alembic is used for managing and applying database schema migrations.

-   **`alembic.ini`:**
    -   **Purpose:** The main configuration file for Alembic.
    -   **Content:** Specifies the database connection URL (for migration operations, can be synchronous), path to migration scripts, and other Alembic settings.

-   **`alembic_env/` directory:**
    -   **Purpose:** Contains the Alembic environment setup and migration scripts.
    -   **`env.py`:** Configures how Alembic connects to the database and discovers model metadata for generating autogenerated migration scripts. It's set up to use the application's SQLAlchemy models (`Base.metadata`).
    -   **`script.py.mako`:** Template for new migration scripts.
    -   **`versions/` directory:** Stores individual, ordered migration scripts generated by Alembic (e.g., `xxxx_add_new_column_to_orders.py`). Each script contains `upgrade()` and `downgrade()` functions.

-   **`migrations/migration_manager.py` (class `MigrationManager`):**
    -   **Purpose:** Provides a programmatic Python interface to manage Alembic database schema migrations from within the application (e.g., at startup).
    -   **Key Aspects:**
        -   Uses Alembic's command API (`alembic.command`).
        -   Offers methods like:
            -   `run_migrations()`: Applies all pending migrations (equivalent to `alembic upgrade head`).
            -   `stamp_db(revision='head')`: Sets the database revision without running migrations.
            -   `generate_revision(message: str, autogenerate: bool = True)`: Programmatically generates a new migration script (though typically done via Alembic CLI).
            -   Potentially methods for downgrading or checking current revision.
    -   **Importance:** Automates the process of keeping the database schema synchronized with the application's models, crucial for deployment and development consistency.

### Time-Series Database (InfluxDB)

For data that is best stored and queried as time-series (e.g., high-frequency market data, performance metrics over time), InfluxDB is used.

-   **`influxdb_client.py` (class `TimeSeriesDB` or similar wrapper):**
    -   **Purpose:** Provides a client wrapper for interacting with an InfluxDB instance.
    -   **Key Aspects:**
        -   Initializes the `influxdb_client.InfluxDBClient` and `WriteApi`/`QueryApi` using connection details from `ConfigManager` (URL, token, org, bucket).
        -   Offers methods for:
            -   Writing data points (e.g., `write_point(measurement, tags, fields, timestamp)`).
            -   Querying data using Flux query language (e.g., `query_data(flux_query)`).
        -   Handles client setup and teardown.
        -   Used by services like `LoggerService` (for performance metrics) or potentially a dedicated metrics collection service.
    -   **Importance:** Provides an efficient way to store and analyze large volumes of timestamped data, which is common in trading systems.

### `__init__.py`

-   **Purpose:** An empty or nearly empty file that marks the `dal` directory (and its subdirectories like `models` and `repositories`) as Python packages.
-   **Key Aspects:** Enables modules within the DAL to be imported using package notation (e.g., `from gal_friday.dal.repositories.order_repository import OrderRepository`). It might also selectively expose key classes at the `gal_friday.dal` level for convenience.

## Data Persistence Strategy

-   **Primary Relational Store:** The main data store for structured application data (configurations, trade records, positions, logs, event history) is a relational database, typically PostgreSQL, due to its robustness, transactional integrity, and support for complex queries and JSONB types (useful for storing context or unstructured metadata). Interaction is managed via the **SQLAlchemy ORM**, providing an object-oriented way to work with database tables.
-   **Schema Evolution:** Database schema changes are managed using **Alembic**. This ensures that the schema evolves in a controlled, versioned manner alongside the application code. Migrations are typically applied automatically at application startup via `MigrationManager`.
-   **Time-Series Data:** For high-volume, high-frequency time-stamped data such as granular market data ticks (if stored), detailed system performance metrics, or continuous feature values, **InfluxDB** is employed. This database is optimized for time-series workloads, offering efficient storage, querying, and downsampling capabilities.
-   **Asynchronous Operations:** All interactions with the primary relational database (via SQLAlchemy) and potentially with InfluxDB are designed to be **asynchronous** (`async/await`) to ensure that database I/O does not block the main application's event loop, which is critical for responsiveness in a trading system.

## Interactions and Importance

The Data Access Layer (DAL) is a cornerstone of the Gal-Friday system, providing several key benefits:

-   **Abstraction & Decoupling:** It provides a structured and abstracted interface for other services to interact with data stores. Services do not need to embed raw SQL queries or be aware of the specific database connection details (beyond what SQLAlchemy or the InfluxDB client exposes). This decouples application logic from data storage specifics.
-   **Centralized Data Logic:** The use of **repositories** centralizes query logic related to specific entities. This makes queries easier to find, maintain, optimize, and test. If the underlying data schema changes, modifications are often localized within the repository and its corresponding model.
-   **Schema Consistency:** The `MigrationManager` and Alembic ensure that the database schema is consistent with the application's expectations across different environments (development, testing, production) and throughout the development lifecycle. This prevents errors caused by schema mismatches.
-   **Maintainability & Testability:** By clearly separating data access concerns, the DAL makes the overall system more maintainable. Repositories and data access methods can be unit-tested independently (often with a test database or mocks).
-   **Support for Multiple Data Stores:** The DAL structure allows for the integration of different types of databases (SQLAlchemy for RDBMS, a dedicated client for InfluxDB) based on the nature of the data and query requirements, all managed under a common `dal` umbrella.

## Adherence to Standards

The design of the Data Access Layer in Gal-Friday aims to adhere to established software engineering best practices, including:
-   **Separation of Concerns:** Isolating data access logic from business logic.
-   **Repository Pattern:** Abstracting data retrieval and persistence mechanisms.
-   **ORM Usage:** Leveraging Object-Relational Mapping for easier and more Pythonic database interaction.
-   **Migration Management:** Ensuring controlled and versioned schema evolution.
These practices contribute to a more robust, maintainable, and scalable data management solution.
