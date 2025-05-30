# LoggerService Module Documentation

## Module Overview

The `gal_friday.logger_service.py` module provides a centralized and comprehensive logging facility for the Gal-Friday trading system. It is designed to be highly configurable and supports multiple output targets, including the console, rotating log files, a PostgreSQL database (via SQLAlchemy), and an InfluxDB time-series database. Key features include structured JSON logging for files, asynchronous processing for database logs, automatic filtering of sensitive data from log contexts, and the ability to centralize logs from other application modules via `PubSubManager`.

## Key Features

-   **Multiple Log Handlers:** Supports various logging outputs simultaneously:
    -   **Console:** Human-readable formatted output, including context information, suitable for live monitoring.
    -   **Rotating Log Files:** Logs messages to files that rotate based on size, using a structured JSON format (via `python-json-logger`) for easier parsing and analysis by log management systems.
    -   **PostgreSQL Database (Asynchronous):** Logs messages to a PostgreSQL database table using SQLAlchemy's asynchronous capabilities. A custom `AsyncPostgresHandler` manages an internal `asyncio.Queue` for non-blocking database writes.
    -   **InfluxDB Time-Series Database:** Logs structured time-series data (e.g., metrics, specific events) to InfluxDB. The InfluxDB client is initialized on demand.
-   **Configuration-Driven:** All aspects of logging (levels, formats, output paths, database connection details, InfluxDB settings) are configurable via a `ConfigManagerProtocol`-compliant configuration object, typically loaded from `config.yaml`.
-   **Structured Logging:** Encourages logging with extra context information (passed as a dictionary), which is included in structured log outputs (JSON file, database).
-   **Sensitive Data Filtering:** Automatically redacts sensitive information (matching predefined patterns for API keys, passwords, etc.) from the `context` dictionary before logging to prevent accidental exposure.
-   **Asynchronous Database Logging:** The `AsyncPostgresHandler` processes log records asynchronously, preventing database I/O from blocking the main application thread or event loop.
-   **Thread-Safe Synchronous Logging:** For calls made from synchronous code (or asynchronous code not directly using the async methods), logs are put onto a standard `queue.Queue` and processed by a dedicated worker thread, which then schedules the async logging operations on the event loop.
-   **Centralized Event Bus Logging:** Subscribes to `LogEvent`s published on the `PubSubManager` event bus, allowing other services to delegate their logging through this central service.
-   **Standard Logging Interface:** Provides standard convenience methods (`debug`, `info`, `warning`, `error`, `exception`, `critical`) familiar from Python's built-in `logging` module.

## Protocols

-   **`ConfigManagerProtocol`**: An informal protocol defining the expected interface for the configuration manager object. It should provide methods like `get(key, default)` and `get_bool(key, default)` to access logging configurations.
-   **`DBConnection`**: (Defined in the snippet but its direct usage seems superseded by the SQLAlchemy session management within `AsyncPostgresHandler`). It originally might have represented a more generic database connection interface.

## Custom Formatters & Handlers

-   **`ContextFormatter(logging.Formatter)`**:
    -   An extension of the standard `logging.Formatter`.
    -   It enhances log records by ensuring that a `context` dictionary (if provided in the logging call via `extra={'context': ...}`) is included in the log output, making it accessible to handlers like the JSON file handler or console handler.
-   **`AsyncPostgresHandler(logging.Handler)`**:
    -   A custom logging handler designed for asynchronous writes to a PostgreSQL database.
    -   It uses an internal `asyncio.Queue` to buffer log records.
    -   A dedicated asyncio task consumes records from this queue and writes them to the database using an SQLAlchemy `AsyncSession` obtained from a provided `async_sessionmaker`.
    -   It maps `LogRecord` attributes to the fields of a `Log` SQLAlchemy model (expected to be defined in `dal.models.log`).
    -   Handles batching or individual writes depending on implementation details.
    -   Provides a `close()` method to gracefully shut down, ensuring all buffered logs are processed.

## Class `LoggerService`

### Initialization (`__init__`)

-   **Parameters:**
    -   `config_manager (ConfigManagerProtocol)`: An instance of a configuration manager.
    -   `pubsub_manager (Optional[PubSubManager])`: An optional `PubSubManager` instance. If provided, `LoggerService` will subscribe to `LogEvent`s.
    -   `db_session_maker (Optional[async_sessionmaker[AsyncSession]])`: An optional SQLAlchemy asynchronous sessionmaker. If provided and database logging is enabled, it's used by `AsyncPostgresHandler`. If not provided but DB logging is enabled, `LoggerService` will attempt to initialize its own SQLAlchemy engine and sessionmaker during `start()`.
-   **Actions:**
    -   Stores `config_manager`, `pubsub_manager`, and `db_session_maker`.
    -   Loads logging configurations (levels, formats, paths, etc.) using `_config_manager`.
    -   Initializes `_sensitive_keys_pattern` for data filtering.
    -   Calls `_setup_logging()` to configure the root logger, console handler, and file handler.
    -   Initializes `_log_queue (queue.Queue)` for thread-safe synchronous log call handling.
    -   Starts `_log_processing_thread` which runs `_process_log_queue()`.
    -   Sets `_db_handler`, `_influxdb_client`, `_influxdb_write_api`, `_async_pg_handler_task` to `None`; these are initialized later if enabled and needed.

### Core Logging Logic

-   **`_setup_logging() -> None`**:
    -   Configures the Python root logger's level based on `logging.level` from config.
    -   Sets up the **Console Handler** if `logging.console.enabled` is true:
        -   Uses `ContextFormatter` with configured format and date format.
        -   Adds it to the root logger.
    -   Sets up the **Rotating File Handler** if `logging.file.enabled` is true:
        -   Uses `pythonjsonlogger.jsonlogger.JsonFormatter` for structured JSON logs.
        -   Configures file path, max bytes, and backup count from settings.
        -   Adds it to the root logger.

-   **`_process_log_queue() -> None`**:
    -   The target function for `_log_processing_thread`.
    -   Continuously gets log call arguments (level, message, args, context, etc.) from `_log_queue`.
    -   If a "STOP" signal is received, the loop terminates.
    -   Otherwise, it uses `asyncio.run_coroutine_threadsafe()` to schedule the actual `self.log()` coroutine on the main asyncio event loop. This bridges synchronous log calls to the asynchronous logging infrastructure.

-   **`_filter_sensitive_data(context: Dict[str, Any]) -> Dict[str, Any]`**:
    -   Recursively iterates through the `context` dictionary (and any nested dictionaries or lists).
    -   If a key matches `_sensitive_keys_pattern` (e.g., "api_key", "password", "secret") or if a string value matches, it redacts the value to "[SENSITIVE_DATA_REDACTED]".
    -   Returns the sanitized context dictionary.

-   **`async log(level: int, message: str, *args, source_module: Optional[str] = None, context: Optional[Dict[str, Any]] = None, exc_info=None, stack_info=False, stacklevel: int = 1) -> None`**:
    -   The main asynchronous logging method.
    -   Determines the `source_module` if not provided (e.g., by inspecting the call stack, though this can be fragile; explicit passing is better).
    -   Creates a `log_context` dictionary, populating it with `source_module` and other standard fields.
    -   If `context` is provided, it's filtered using `_filter_sensitive_data()` and then merged into `log_context`.
    -   Gets the appropriate Python logger instance (e.g., `logging.getLogger(source_module or self._service_name)`).
    -   Calls the logger's `log()` method (e.g., `python_logger.log(level, message, *args, extra={'context': log_context}, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel + 1)`).

-   **Convenience Methods (`debug`, `info`, `warning`, `error`, `exception`, `critical`)**:
    -   These methods (`async def debug(...)`, `async def info(...)`, etc.) provide a familiar interface.
    -   They simply call the main `await self.log(...)` method with the appropriate `level` and pass through other arguments.
    -   Synchronous versions of these methods are also provided (e.g., `def info_sync(...)`) which put the log request onto `_log_queue` for thread-safe processing.

### Database Logging (SQLAlchemy)

-   **`_initialize_sqlalchemy() -> None`**:
    -   Called by `start()` if database logging (`logging.database.enabled`) is true and `_db_session_maker` was not provided during `__init__`.
    -   Retrieves `database.connection_string`, `pool_size`, `max_overflow`, `echo_sql` from config.
    -   Creates an SQLAlchemy `async_engine` using `create_async_engine`.
    -   Creates an `async_sessionmaker` (`self._db_session_maker`) bound to this engine.
    -   This sessionmaker is then used by `AsyncPostgresHandler`.

-   The `AsyncPostgresHandler` (if added to the root logger) will:
    -   Receive `LogRecord` objects.
    -   Format them into a `Log` SQLAlchemy model instance (expected to be defined in `gal_friday.dal.models.log` or a similar path). This model would have columns for level, message, timestamp, source module, context (as JSONB), etc.
    -   Use a session from `self._db_session_maker` to add and commit the `Log` instance to the database. This happens in its internal asyncio task consuming from its internal queue.

### Time-Series Logging (InfluxDB)

-   **`_initialize_influxdb_client() -> None`**:
    -   Called on the first attempt to use `log_timeseries` if InfluxDB logging (`logging.influxdb.enabled`) is true and the client isn't already initialized.
    -   Retrieves InfluxDB connection parameters (`url`, `token`, `org`, `bucket`) from config.
    -   Initializes `influxdb_client.InfluxDBClient` and `influxdb_client.WriteApi` (with `SYNCHRONOUS` or `ASYNCHRONOUS` write options, typically async).
    -   Stores these as `self._influxdb_client` and `self._influxdb_write_api`.

-   **`_prepare_influxdb_point(measurement: str, tags: Dict[str, str], fields: Dict[str, Any], timestamp: Optional[datetime] = None) -> influxdb_client.Point`**:
    -   Creates an `influxdb_client.Point` object for writing to InfluxDB.
    -   Sets the `measurement` name.
    -   Adds all `tags` and `fields` to the point.
    -   Sets the `timestamp` (defaults to `datetime.utcnow()` if not provided).

-   **`async log_timeseries(measurement: str, tags: Dict[str, str], fields: Dict[str, Any], timestamp: Optional[datetime] = None) -> None`**:
    -   The public method for logging data to InfluxDB.
    -   Ensures the InfluxDB client is initialized by calling `_initialize_influxdb_client()` if needed.
    -   Calls `_prepare_influxdb_point()` to create the data point.
    -   Uses `self._influxdb_write_api.write(bucket=self._influxdb_bucket, org=self._influxdb_org, record=point)` to send the data. If the write API is asynchronous, this will be an `await` call.
    -   Handles potential errors during the write operation.

### Service Lifecycle & Event Handling

-   **`async start() -> None`**:
    -   If database logging is enabled and `_db_session_maker` was not provided at init, calls `_initialize_sqlalchemy()`.
    -   If database logging is enabled and `_db_session_maker` is available:
        -   Creates an instance of `AsyncPostgresHandler` using `_db_session_maker`.
        -   Adds this handler to the Python root logger.
        -   Starts the `AsyncPostgresHandler`'s internal processing task (`self._async_pg_handler_task = asyncio.create_task(self._db_handler.process_queue_async())`).
    -   If `pubsub_manager` is available, subscribes `_handle_log_event` to `EventType.LOG_EVENT`.
    -   Logs that the LoggerService has started.

-   **`async stop() -> None`**:
    -   If `pubsub_manager` is available, unsubscribes from `LogEvent`.
    -   Signals the synchronous log processing thread to stop by putting "STOP" on `_log_queue` and then joins the thread.
    -   If `_db_handler` (AsyncPostgresHandler) is active:
        -   Calls `await self._db_handler.close()` which signals its internal queue processor to finish and then closes resources.
        -   Waits for `_async_pg_handler_task` to complete.
    -   If `_influxdb_write_api` is active, calls `_influxdb_write_api.close()`.
    -   If `_influxdb_client` is active, calls `_influxdb_client.close()`.
    -   Logs that the LoggerService is stopping.

-   **`async _handle_log_event(event: LogEvent) -> None`**:
    -   Receives `LogEvent` objects from the `PubSubManager`. A `LogEvent` typically encapsulates log level, message, source module, and context.
    -   Calls `await self.log()` with the information extracted from the `LogEvent`, effectively centralizing logs from other services that publish these events.

## Dependencies

-   **Standard Libraries:**
    -   `asyncio`: For asynchronous operations.
    -   `json`: Used by `python-json-logger`.
    -   `logging`, `logging.handlers`: Core Python logging framework.
    -   `queue.Queue`: For thread-safe synchronous log queuing.
    -   `re`: For regular expressions used in sensitive data filtering.
    -   `sys`: For console output (stderr/stdout).
    -   `threading.Thread`: For the synchronous log processing thread.
    -   `datetime.datetime`, `datetime.timezone`: For timestamping.
    -   `decimal.Decimal`: Potentially for formatting numbers in context, though primary conversion is to string.
    -   `pathlib.Path`: For file path manipulation.
-   **Third-Party Libraries:**
    -   `pythonjsonlogger.jsonlogger.JsonFormatter`: For structured JSON file logging.
    -   `sqlalchemy.ext.asyncio.create_async_engine`, `sqlalchemy.ext.asyncio.AsyncSession`, `sqlalchemy.orm.sessionmaker`, `sqlalchemy.exc.SQLAlchemyError`: For asynchronous database interaction with PostgreSQL.
    -   `influxdb_client` (Optional, imported on demand): The client library for InfluxDB v2.
-   **Core Application Modules:**
    -   `gal_friday.core.pubsub.PubSubManager` (Optional).
    -   `gal_friday.core.events.LogEvent`, `gal_friday.core.events.EventType`.
    -   `gal_friday.dal.models.log.Log` (SQLAlchemy model for log entries, path may vary).
    -   `ConfigManagerProtocol` (interface for configuration).

## Configuration (Key options from `logging` section of app config)

-   **`level (str)`**: Root logging level for the application (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
-   **`format (str)`**: The format string for console logs (uses standard `logging` format directives plus custom `%(context)s`).
-   **`date_format (str)`**: The date/time format string (e.g., "%Y-%m-%d %H:%M:%S,%f").
-   **`console` (dict)**:
    -   `enabled (bool)`: Whether to enable console logging.
-   **`file` (dict)**:
    -   `enabled (bool)`: Whether to enable file logging.
    -   `directory (str)`: Directory to store log files.
    -   `filename (str)`: Base name for log files (e.g., "galfriday.log").
    -   `max_bytes (int)`: Maximum size of a log file before rotation.
    -   `backup_count (int)`: Number of backup log files to keep.
-   **`database` (dict)**:
    -   `enabled (bool)`: Whether to enable database logging.
    -   `connection_string (str)`: SQLAlchemy asynchronous connection string (e.g., "postgresql+asyncpg://user:pass@host/db").
    -   `table_name (str)`: (Note: The table name is typically defined by the `Log` SQLAlchemy model's `__tablename__`, e.g., "logs").
    -   `level (str)`: Minimum log level to send to the database (e.g., "INFO").
    -   `pool_size (int)`: SQLAlchemy engine pool size for the logger's dedicated engine (if created internally).
    -   `max_overflow (int)`: SQLAlchemy engine max overflow for the logger's engine.
    -   `echo_sql (bool)`: Whether the logger's SQLAlchemy engine should echo SQL.
-   **`influxdb` (dict)**:
    -   `enabled (bool)`: Whether to enable InfluxDB logging.
    -   `url (str)`: URL of the InfluxDB instance (e.g., "http://localhost:8086").
    -   `token (str)`: Authentication token for InfluxDB.
    -   `org (str)`: InfluxDB organization name/ID.
    -   `bucket (str)`: InfluxDB bucket name to write to.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `LoggerService` module.
