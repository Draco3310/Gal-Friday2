# Logger Service Module
"""Logging service module providing centralized logging capabilities with multiple output targets.

This module implements a comprehensive logging service that handles logging to console,
files, databases and time-series databases. It provides thread-safety, async support,
and context-rich logging capabilities.
"""

import asyncio
import contextlib  # Added for SIM105
import json
import logging
import logging.handlers
import queue
import re
import threading
import types  # Added for exc_info typing
from asyncio import QueueFull  # Import QueueFull from asyncio, not asyncio.exceptions
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,  # For AsyncpgPoolAdapter
)
from datetime import datetime
from decimal import Decimal
from random import SystemRandom
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    TypeVar,
)

import asyncpg

# Runtime imports
from influxdb_client import Point as InfluxDBPoint

# Import JSON Formatter
from .core.events import EventType, LogEvent

if TYPE_CHECKING:
    from .core.pubsub import PubSubManager

# Import custom exceptions
from .exceptions import DatabaseError, InvalidLoggerTableNameError, UnsupportedParamsTypeError

# Type variables for generic protocols
_T = TypeVar("_T")
_RT = TypeVar("_RT")

# Define a Protocol for ConfigManager to properly type hint its interface


class ConfigManagerProtocol(Protocol):
    """Protocol defining the interface for configuration management.

    This protocol establishes the required methods for accessing
    configuration values from various sources.
    """

    def get(self, key: str, default: _T | None = None) -> _T:
        """Get a configuration value by key.

        Args:
        ----
            key: The configuration key to retrieve
            default: Value to return if key is not found

        Returns:
        -------
            The configuration value or default if not found
        """
        ...

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value by key.

        Args:
        ----
            key: The configuration key to retrieve
            default: Integer value to return if key is not found

        Returns:
        -------
            The integer configuration value or default if not found
        """
        ...


# Define a Protocol for database connection


class DBConnection(Protocol):
    """Protocol defining a database connection interface.

    Establishes the required methods for interacting with a database.
    """

    async def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> _RT:
        """Execute a database query.

        Args:
        ----
            query: SQL query string to execute
            params: Optional sequence or mapping of parameters for the query

        Returns:
        -------
            Query result
        """
        ...


# Define a Protocol for the Pool interface we need


class PoolProtocol(Protocol):
    """Protocol defining a connection pool interface.

    Specifies the methods required for connection pooling functionality.
    """

    def acquire(self) -> AbstractAsyncContextManager[DBConnection]:
        """Acquire a connection from the pool.

        Returns:
        -------
            A connection context manager
        """
        ...

    async def release(self, conn: DBConnection) -> None:
        """Release a connection back to the pool.

        Args:
        ----
            conn: The connection to release
        """
        ...

    async def close(self) -> None:
        """Close the connection pool."""
        ...


# Define a proper type alias for the Pool type
PoolType = TypeVar("PoolType", bound=PoolProtocol)

# Placeholder Type Hints (Refine later if needed)
if TYPE_CHECKING:
    # Use actual type if stubs were available
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import WriteApi
    AsyncPostgresHandlerType = logging.Handler  # Placeholder for typing
else:
    # Define placeholders if not type checking to avoid runtime errors
    AsyncPostgresHandlerType = logging.Handler




class ContextFormatter(logging.Formatter):
    """Format log records with context dictionary information.

    This formatter extends the standard logging formatter to include
    additional context information from the record's context dictionary.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with context information.

        Args:
        ----
            record: The log record to format

        Returns:
        -------
            Formatted log record as string with context information included
        """
        # Default formatting first
        s = super().format(record)
        # Check if context exists in the record's extra dict
        context_str = ""
        if hasattr(record, "context") and record.context:
            if isinstance(record.context, dict):
                context_items = [f"{k}={v}" for k, v in record.context.items()]
                context_str = ", ".join(context_items)
            else:
                context_str = str(record.context)  # Fallback if not a dict

        # Replace the placeholder [%(context)s] - handle case where it might be
        # missing
        if self._fmt is not None and "[%(context)s]" in self._fmt:
            if context_str:
                s = s.replace("[%(context)s]", f"[{context_str}]")
            else:
                s = s.replace(
                    " - [%(context)s]",
                    "",
                )  # Remove placeholder and separator if no context
                # Remove just placeholder if at start/end
                s = s.replace("[%(context)s]", "")

        return s


# --- Custom Async Database Handler ---
class AsyncPostgresHandler(logging.Handler, Generic[PoolType]):
    """Asynchronous handler for logging to PostgreSQL database."""

    ALLOWED_TABLE_NAMES: ClassVar[set[str]] = {
        "logs",
        "application_logs",
        "system_logs",
    }  # Add your allowed table names here

    def __init__(self, pool: PoolType, table_name: str, loop: asyncio.AbstractEventLoop) -> None:
        """Initialize the handler with database pool and table name.

        Args:
        ----
            pool: Database connection pool
            table_name: Name of the table to log to (must be in ALLOWED_TABLE_NAMES)
            loop: Event loop to use for async operations
        """
        super().__init__()
        if table_name not in self.ALLOWED_TABLE_NAMES:
            raise InvalidLoggerTableNameError(table_name, self.ALLOWED_TABLE_NAMES)
        self._table_name = table_name
        self._pool = pool
        self._loop = loop
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._task = asyncio.create_task(self._process_queue())

    def emit(self, record: logging.LogRecord) -> None:
        """Format record and place it in the queue for async processing.

        Implements the emit method required by the logging.Handler interface.

        Args:
        ----
            record: The log record to be processed and stored
        """
        if self._closed:
            return
        try:
            # Prepare data for the queue
            data = self._format_record(record)
            # Use call_soon_threadsafe for thread safety
            self._loop.call_soon_threadsafe(
                (lambda d: self._queue.put_nowait(d)) if not self._closed else (lambda _: None),
                data,
            )
        except (ValueError, TypeError, RuntimeError, QueueFull):
            self.handleError(record)

    def _format_record(self, record: logging.LogRecord) -> dict[str, Any]:
        """Format the log record into a dictionary suitable for DB insertion."""
        context_json = None
        if hasattr(record, "context") and record.context:
            try:
                context_json = json.dumps(record.context)
            except TypeError:
                # Fallback for non-serializable
                context_json = json.dumps(str(record.context))

        exc_text = None
        if record.exc_info and not record.exc_text:
            if self.formatter:
                record.exc_text = self.formatter.formatException(record.exc_info)
            else:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            exc_text = record.exc_text

        # Ensure message is formatted
        self.format(record)

        return {
            "timestamp": datetime.fromtimestamp(record.created),
            "logger_name": record.name,
            "level_name": record.levelname,
            "level_no": record.levelno,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "filename": record.filename,
            "lineno": record.lineno,
            "func_name": record.funcName,
            "context_json": context_json,
            "exception_text": exc_text,
        }

    async def _attempt_db_insert(self, record_data: dict[str, Any]) -> bool:
        """Attempt to insert a single log record into the database.

        Note: The table name is validated against ALLOWED_TABLE_NAMES in __init__
        and again here for defense in depth.
        """
        try:
            # Double-check table name against allowed values
            if self._table_name not in self.ALLOWED_TABLE_NAMES:
                raise ValueError(f"Invalid table name: {self._table_name}")

            # Define query templates for each allowed table
            query_templates = {
                "logs": """
                    INSERT INTO logs (
                        timestamp, logger_name, level_name, level_no, message,
                        pathname, filename, lineno, func_name, context_json,
                        exception_text
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                "application_logs": """
                    INSERT INTO application_logs (
                        timestamp, logger_name, level_name, level_no, message,
                        pathname, filename, lineno, func_name, context_json,
                        exception_text
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                "system_logs": """
                    INSERT INTO system_logs (
                        timestamp, logger_name, level_name, level_no, message,
                        pathname, filename, lineno, func_name, context_json,
                        exception_text
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
            }


            # Get the pre-defined query for this table
            query = query_templates[self._table_name]

            async with self._pool.acquire() as conn:
                try:
                    await conn.execute(
                        query,
                        record_data["timestamp"],
                        record_data["logger_name"],
                        record_data["level_name"],
                        record_data["level_no"],
                        record_data["message"],
                        record_data["pathname"],
                        record_data["filename"],
                        record_data["lineno"],
                        record_data["func_name"],
                        record_data["context_json"],
                        record_data["exception_text"],
                    )
                    return True
                except (
                    asyncpg.exceptions.ConnectionDoesNotExistError,
                    asyncpg.exceptions.ConnectionIsClosedError,
                    asyncpg.exceptions.InterfaceError,
                    OSError,
                ) as e:
                    # These are retryable errors
                    logging.getLogger(__name__).debug(
                        "Retryable error in _attempt_db_insert: %s", str(e),
                    )
                    raise
                except Exception as e:
                    # Non-retryable error during DB operation
                    logging.getLogger(__name__).error(
                        "Non-retryable error in _attempt_db_insert: %s",
                        str(e),
                        exc_info=True,
                    )
                    return False
        except Exception as e:
            logging.getLogger(__name__).error(
                "Unexpected error in _attempt_db_insert: %s",
                str(e),
                exc_info=True,
            )
            return False

    async def _process_queue_with_retry(self, record_data: dict[str, Any]) -> None:
        """Process a single record with retry logic."""
        max_retries = 3
        base_backoff = 1.0  # seconds
        attempt = 0
        while attempt < max_retries:
            try:
                if await self._attempt_db_insert(record_data):
                    return  # Success
            except (
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.ConnectionIsClosedError,
                asyncpg.exceptions.InterfaceError,
                OSError,
            ) as conn_err:
                attempt += 1
                if attempt >= max_retries:
                    logging.getLogger(__name__).error(
                        "AsyncPostgresHandler: DB connection error failed after %d attempts: %s",
                        max_retries,
                        str(conn_err),
                        exc_info=True,
                    )
                else:
                    # Exponential backoff with jitter
                    wait_time = min(base_backoff * (2**attempt), 30.0)  # Cap at 30s
                    wait_time += SystemRandom().uniform(0, wait_time * 0.1)
                    logging.getLogger(__name__).warning(
                        "AsyncPostgresHandler: DB connection error (Attempt %d/%d). "
                        "Retrying in %.2fs. Error: %s",
                        attempt,
                        max_retries,
                        wait_time,
                        str(conn_err),
                    )
                    await asyncio.sleep(wait_time)
            except (RuntimeError, ValueError, TypeError, DatabaseError, asyncpg.PostgresError):
                # Catches non-retryable errors from _attempt_db_insert
                return  # Stop processing this record

    async def _process_queue(self) -> None:
        """Continuously processes log records from the queue."""
        while True:
            record_data = None
            try:
                record_data = await self._queue.get()
                if record_data is None:  # Sentinel value to stop
                    self._queue.task_done()
                    break

                await self._process_queue_with_retry(record_data)
                self._queue.task_done()

            except asyncio.CancelledError:
                logging.getLogger(__name__).info(
                    "AsyncPostgresHandler queue processing cancelled.",
                )
                if record_data is not None:
                    with contextlib.suppress(ValueError):
                        self._queue.task_done()
                break
            except Exception as e:
                logging.getLogger(__name__).error(
                    "AsyncPostgresHandler: Error in outer processing loop: %s",
                    e,
                    exc_info=True,
                )
                if record_data is not None:
                    with contextlib.suppress(ValueError):
                        self._queue.task_done()
                await asyncio.sleep(1)

    def close(self) -> None:
        """Close the handler, signaling the queue processor to finish."""
        if not self._closed:
            self._closed = True
            self._loop.call_soon_threadsafe(lambda: self._queue.put_nowait(None))
        super().close()

    async def wait_closed(self) -> None:
        """Wait for the queue processing task to finish.

        This method should be awaited after calling close() to ensure
        all queued log records are processed before the application exits.
        """
        if self._task is not None:
            await self._task


# -------------------------------------



class LoggerService(Generic[PoolType]):
    """Handle logging configuration and provides interfaces for logging messages/time-series data.

    Configures and manages logging to multiple destinations including console,
    files, databases and time-series databases with support for context information.
    """

    _influx_client: Optional["InfluxDBClient"] = None
    _influx_write_api: Optional["WriteApi"] = None

    def __init__(
        self,
        config_manager: ConfigManagerProtocol,
        pubsub_manager: "PubSubManager",
    ) -> None:
        """Initialize the logger service.

        Sets up logging handlers based on configuration and starts the log processing thread.

        Args:
        ----
            config_manager: Configuration provider for logger settings
            pubsub_manager: Publish-subscribe manager for event handling
        """
        self._config_manager = config_manager
        self._pubsub = pubsub_manager
        self._log_level = self._config_manager.get("logging.level", "INFO").upper()
        self._log_format = self._config_manager.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context_json)s",
        )
        self._log_date_format = self._config_manager.get(
            "logging.date_format",
            "%Y-%m-%d %H:%M:%S",
        )

        # Async DB Handler setup
        self._db_config: dict[str, Any] = self._config_manager.get("logging.database", {})
        self._db_enabled = bool(self._db_config.get("enabled", False))
        self._async_handler: AsyncPostgresHandler[PoolType] | None = None
        self._db_pool: PoolType | None = None

        # Queue and thread for handling synchronous logging calls from async context
        self._queue: queue.Queue[tuple[Callable[..., None], tuple, dict]] = queue.Queue()
        self._thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self._stop_event = threading.Event()
        self._loggers: dict[str, logging.Logger] = {}
        self._root_logger: logging.Logger = logging.getLogger("gal_friday")

        self._setup_logging()
        self._thread.start()
        self.info("LoggerService initialized.", source_module="LoggerService")

    def _filter_sensitive_data(
        self,
        context: dict[str, object] | None,
    ) -> dict[str, object] | None:
        """Recursively filter sensitive data from log context.

        Args:
            context: Dictionary containing log context data to be filtered

        Returns:
            Filtered dictionary with sensitive data redacted, or None if input is None/empty
        """
        if not context:  # Handle None or empty dict
            return None

        filtered: dict[str, object] = {}
        sensitive_keys = [
            "api_key",
            "secret",
            "password",
            "token",
            "credentials",
            "private_key",
            "auth",
            "access_key",
            "secret_key",
            # Add other sensitive key names or patterns
        ]
        # Regex for things that look like keys/secrets
        sensitive_value_pattern = re.compile(
            r"^[A-Za-z0-9/+]{20,}$",
        )  # Example: Base64-like strings > 20 chars

        for key, value in context.items():
            key_lower = str(key).lower()
            is_sensitive = any(pattern in key_lower for pattern in sensitive_keys)

            if isinstance(value, dict):
                # Recurse into nested dictionaries
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, list):
                # Recurse into lists (filter dicts within lists)
                filtered[key] = [
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif is_sensitive or (isinstance(value, str) and sensitive_value_pattern.match(value)):
                # Mask if key is sensitive or value looks sensitive
                filtered[key] = "********"
            else:
                filtered[key] = value
        return filtered

    # Define a type alias for exc_info to improve readability and manage line length
    ExcInfoType = (
        bool |
        tuple[type[BaseException], BaseException, types.TracebackType] |
        BaseException |
        None
    )

    def log(
        self,
        level: int,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
        exc_info: ExcInfoType = None,
    ) -> None:
        """Log a message to the configured handlers.

        Args:
        ----
            level: The logging level (e.g., logging.INFO, logging.WARNING)
            message: The primary log message string (can be a format string)
            *args: Arguments for the format string in 'message'
            source_module: Optional name of the module generating the log
            context: Optional dictionary of key-value pairs for extra context
            exc_info: Optional exception info (e.g., True or exception tuple)
        """
        # Get the specific logger for the source module, or root if None
        logger_name = f"gal_friday.{source_module}" if source_module else "gal_friday"
        logger = logging.getLogger(logger_name)

        # Filter context BEFORE passing it as 'extra'
        filtered_context = self._filter_sensitive_data(context)

        # Prepare extra dictionary for context
        extra_data = {"context": filtered_context} if filtered_context else {}

        # Log the message using the standard logging interface
        logger.log(
            level, message, *args, exc_info=exc_info, extra=extra_data, stacklevel=2,
        )  # Pass *args
        # stacklevel=2 ensures filename/lineno are from the caller of this
        # method

    # --- Convenience Helper Methods --- #

    def debug(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Log a message with DEBUG level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.DEBUG, message, *args, source_module=source_module, context=context,
        )  # Pass *args

    def info(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Log a message with INFO level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.INFO, message, *args, source_module=source_module, context=context,
        )  # Pass *args

    def warning(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Log a message with WARNING level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.WARNING, message, *args, source_module=source_module, context=context,
        )  # Pass *args

    def error(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
        exc_info: ExcInfoType = None,
    ) -> None:
        """Log a message with ERROR level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
            exc_info: Optional exception info
        """
        self.log(
            logging.ERROR,
            message,
            *args,
            source_module=source_module,
            context=context,
            exc_info=exc_info,
        )  # Pass *args

    def exception(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Log a message with ERROR level and include exception information.

        This method should only be called from an exception handler.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.ERROR,
            message,
            *args,
            source_module=source_module,
            context=context,
            exc_info=True,
        )  # Pass *args

    def critical(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: dict[str, object] | None = None,
        exc_info: ExcInfoType = None,
    ) -> None:
        """Log a message with CRITICAL level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
            exc_info: Optional exception info
        """
        self.log(
            logging.CRITICAL,
            message,
            *args,
            source_module=source_module,
            context=context,
            exc_info=exc_info,
        )  # Pass *args

    # --- Placeholder for Time-Series Logging --- #
    async def _initialize_influxdb_client(self) -> bool:
        """Initialize the InfluxDB client if not already initialized.

        Returns:
        -------
            bool: True if initialization was successful, False otherwise
        """
        if self._influx_client is not None:  # Check instance variable
            return True  # Already initialized

        url: str | None = self._config_manager.get("logging.influxdb.url")
        token: str | None = self._config_manager.get("logging.influxdb.token")
        org: str | None = self._config_manager.get("logging.influxdb.org")
        if not all([url, token, org]):
            self.warning(
                "InfluxDB config incomplete (url/token/org). Cannot log timeseries.",
                source_module="LoggerService",
            )
            return False
        try:
            import influxdb_client  # Import here to keep dependency optional
            from influxdb_client.client.write_api import SYNCHRONOUS
        except ImportError:
            self.error(
                "InfluxDB client library not installed ('pip install influxdb-client'). "
                "Cannot log timeseries.",
                source_module="LoggerService",
            )
            self._influx_client = None
            self._influx_write_api = None  # Ensure write_api is also cleared
            return False
        except Exception as e:
            self.error(
                "Failed to initialize InfluxDB client: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )
            self._influx_client = None
            self._influx_write_api = None  # Ensure write_api is also cleared
            return False
        else:
            # Since we've already checked that all([url, token, org]) is True,
            # we can safely check these are strings
            if not isinstance(token, str):
                raise TypeError("Token must be a string")
            if not isinstance(org, str):
                raise TypeError("Organization must be a string")
            self._influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
            self._influx_write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)
            self.info(
                "InfluxDB client initialized for timeseries logging.",
                source_module="LoggerService",
            )
            return True

    def _prepare_influxdb_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: datetime,
    ) -> InfluxDBPoint | None:  # Returns InfluxDB Point or None
        """Prepare a data point for InfluxDB.

        Args:
        ----
            measurement: Name of the measurement
            tags: Dictionary of tag keys and values
            fields: Dictionary of field keys and values
            timestamp: Timestamp for the data point

        Returns:
        -------
            Optional influxdb_client.Point object or None if preparation fails
        """
        try:
            from influxdb_client import Point, WritePrecision  # Import here
        except ImportError:
            # This case should ideally be caught by _initialize_influxdb_client
            self.error(
                "InfluxDB client library not found during point preparation.",
                source_module="LoggerService",
            )
            return None
        except Exception as e:
            self.error(
                "Error preparing InfluxDB point: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )
            return None
        else:
            point = Point(measurement).time(timestamp, WritePrecision.MS)

            for key, value in tags.items():
                point = point.tag(key, str(value))

            valid_fields = {}
            for key, value in fields.items():
                if isinstance(value, float | int | bool | str):
                    valid_fields[key] = value
                elif isinstance(value, Decimal):
                    valid_fields[key] = float(value)
                else:
                    self.warning(
                        "Unsupported type for InfluxDB field '%s': %s. " "Converting to string.",
                        key,
                        type(value),
                        source_module="LoggerService",
                    )
                    valid_fields[key] = str(value)

            if not valid_fields:
                self.warning(
                    "No valid fields for timeseries point in '%s'. Skipping.",
                    measurement,
                    source_module="LoggerService",
                )
                return None

            for key, value in valid_fields.items():
                point = point.field(key, value)
            # Explicitly annotate the return type
            from influxdb_client import Point

            return point  # type: ignore[no-any-return]

    async def log_timeseries(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> None:
        """Log time-series data to InfluxDB.

        Args:
        ----
            measurement: Name of the measurement
            tags: Dictionary of tag keys and values
            fields: Dictionary of field keys and values
            timestamp: Optional timestamp for the data point, current time used if not provided
        """
        log_time = timestamp if timestamp else datetime.utcnow()

        self.debug(
            "[TimeSeries] M=%s, T=%s, F=%s, TS=%s",
            measurement,
            tags,
            fields,
            log_time.isoformat(),
            source_module="LoggerService",
        )

        if not self._config_manager.get("logging.influxdb.enabled", default=False):
            return

        if not await self._initialize_influxdb_client():
            return  # Initialization failed or not configured

        if not hasattr(self, "_influx_write_api") or self._influx_write_api is None:
            return  # Should have been caught by initialize, but as a safeguard

        point = self._prepare_influxdb_point(measurement, tags, fields, log_time)
        if not point:
            return

        try:
            bucket: str | None = self._config_manager.get("logging.influxdb.bucket")
            if not bucket:
                self.warning(
                    "InfluxDB bucket not configured. Cannot log timeseries.",
                    source_module="LoggerService",
                )
                return
            self._influx_write_api.write(bucket=bucket, record=point)
        except Exception as e:
            self.error(
                "Failed to write time-series data to InfluxDB: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )

    async def start(self) -> None:
        """Initialize the logger service and set up required connections.

        Initializes database connection pool if configured and subscribes to log events.
        """
        self.info("LoggerService start sequence initiated.", source_module="LoggerService")

        # Subscribe to LOG events from the event bus
        try:
            self._pubsub.subscribe(EventType.LOG_ENTRY, self._handle_log_event)  # Removed await
            self.info(
                "Subscribed to LOG_ENTRY events from event bus.",
                source_module="LoggerService",
            )
        except Exception as e:
            self.error(
                "Failed to subscribe to LOG_ENTRY events: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )

        if self._db_enabled:
            await self._initialize_db_pool()
            # Attempt to configure DB handler again if pool initialization
            # succeeded now
            if self._db_pool and not self._async_handler:
                if not hasattr(self, "_db_handler"):
                    self.info(
                        "Re-attempting to configure database log handler "
                        "after pool initialization...",
                    )
                self._setup_logging()  # Re-run config to add the handler
            elif not self._db_pool:
                logging.error(
                    "Database logging enabled but pool initialization failed. "
                    "DB logging inactive.",
                )

    async def stop(self) -> None:
        """Shut down the logger service gracefully.

        Closes the database handler, connection pool, and logging threads.
        """
        self.info("LoggerService stop sequence initiated.", source_module="LoggerService")

        # Unsubscribe from LOG events
        try:
            self._pubsub.unsubscribe(EventType.LOG_ENTRY, self._handle_log_event)  # Removed await
            self.info("Unsubscribed from LOG_ENTRY events.", source_module="LoggerService")
        except Exception as e:
            self.error(
                "Error unsubscribing from LOG_ENTRY events: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )

        # Close the custom handler first to allow queue processing
        if self._async_handler:
            self.info("Closing database log handler...", source_module="LoggerService")
            self._async_handler.close()
            try:
                # Check if handler has the wait_closed method (our custom
                # AsyncPostgresHandler does)
                if hasattr(self._async_handler, "wait_closed"):
                    await asyncio.wait_for(
                        self._async_handler.wait_closed(),
                        timeout=5.0,
                    )  # Wait for queue to empty
                    self.info(
                        "Database log handler closed gracefully.",
                        source_module="LoggerService",
                    )
            except TimeoutError:
                self.warning(
                    "Timeout waiting for database log handler queue to empty.",
                    source_module="LoggerService",
                )
            except Exception as e:
                self.error(
                    "Error waiting for database log handler closure: %s",
                    e,
                    source_module="LoggerService",
                    exc_info=True,
                )

        # Close the pool
        await self._close_db_pool()

        # Signal the log processing thread to stop
        self._stop_event.set()
        self._thread.join(timeout=2.0)  # Wait briefly for thread to exit
        if self._thread.is_alive():
            self.warning(
                "Log processing thread did not exit cleanly.",
                source_module="LoggerService",
            )

    async def _initialize_db_pool(self) -> None:
        """Initialize the asyncpg connection pool for database logging.

        Creates a new connection pool if database logging is enabled in the configuration.
        """
        if not self._db_enabled:
            return

        db_dsn = str(self._config_manager.get("logging.database.connection_string"))
        if not db_dsn:
            self.error(
                "Database logging enabled but connection_string is missing.",
                source_module="LoggerService",
            )
            return

        try:
            min_size = int(self._config_manager.get("logging.database.min_pool_size", default=1))
            max_size = int(self._config_manager.get("logging.database.max_pool_size", default=5))
            self.info(
                "Initializing database connection pool " "(min: %s, max: %s) for logging...",
                min_size,
                max_size,
                source_module="LoggerService",
            )
            # Create the actual asyncpg pool
            actual_pool = await asyncpg.create_pool(
                dsn=db_dsn,
                min_size=min_size,
                max_size=max_size,
            )
            # Wrap it with our adapter
            self._db_pool = AsyncpgPoolAdapter(actual_pool)  # type: ignore[assignment]
            # The type: ignore[assignment] might be needed if PoolType is strictly PoolProtocol
            # and the linter can't infer AsyncpgPoolAdapter is a PoolProtocol.
            # Alternatively, refine PoolType or ensure AsyncpgPoolAdapter explicitly inherits
            # (it does structurally).

            self.info(
                "Database connection pool (via adapter) initialized successfully.",
                source_module="LoggerService",
            )
        except (
            asyncpg.exceptions.InvalidConnectionParametersError,
            asyncpg.exceptions.CannotConnectNowError,
            OSError,
            Exception,  # Catching generic Exception last
        ) as e:
            self.critical(
                "Failed to initialize database connection pool: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )
            self._db_pool = None

    async def _close_db_pool(self) -> None:
        """Close the database connection pool.

        Safely shuts down the asyncpg connection pool if it exists.
        """
        if self._db_pool:
            self.info("Closing database connection pool...", source_module="LoggerService")
            try:
                await self._db_pool.close()
                self.info("Database connection pool closed.", source_module="LoggerService")
            except Exception as e:
                self.error(
                    "Error closing database connection pool: %s",
                    e,
                    source_module="LoggerService",
                    exc_info=True,
                )
            finally:
                self._db_pool = None

    async def _handle_log_event(self, event: LogEvent) -> None:
        """Handle a LogEvent received from the event bus.

        Process the event and send it to the appropriate logging handlers.

        Args:
        ----
            event: The LogEvent object containing log information
        """
        if not isinstance(event, LogEvent):
            self.warning(
                "Received non-LogEvent on LOG_ENTRY topic: %s",
                type(event),
                source_module="LoggerService",
            )
            return

        # Map event level string to logging level integer
        level_name = event.level.upper() if hasattr(event, "level") else "INFO"
        level = getattr(logging, level_name, logging.INFO)  # Default to INFO if invalid

        # Call the standard logging method
        self.log(
            level=level,
            message=event.message,
            # source_module and context are already keyword arguments for self.log
            # No *args needed here as event.message is expected to be a complete string
            source_module=event.source_module,  # Use source from event
            context=event.context if hasattr(event, "context") else None,
            exc_info=None,  # Or derive from context/level if needed
        )


# --- Adapter for asyncpg.Connection to match DBConnection protocol ---
class AsyncpgConnectionAdapter(DBConnection):
    """Adapts an asyncpg.Connection to the DBConnection protocol."""

    def __init__(self, actual_connection: asyncpg.Connection) -> None:
        """Initialize the adapter with an asyncpg connection.

        Args:
            actual_connection: The asyncpg connection to adapt
        """
        self._conn = actual_connection

    @property
    def actual_connection(self) -> asyncpg.Connection:
        """Provides access to the underlying asyncpg.Connection."""
        return self._conn

    async def execute(
        self,
        query: str,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> _RT:  # Changed Any to _RT
        """Execute a query using the wrapped asyncpg.Connection."""
        if params is None:
            return await self._conn.execute(query)  # type: ignore[no-any-return]
        if isinstance(params, Sequence):
            # Ensure not passing a string as a sequence of characters for multiple params
            if isinstance(params, str | bytes):
                # If a single string/byte is the *only* param, wrap it in a list for asyncpg
                return await self._conn.execute(query, params)  # type: ignore[no-any-return]
            return await self._conn.execute(query, *params)  # type: ignore[no-any-return]
        if isinstance(params, Mapping):
            # asyncpg's basic execute(*args) doesn't directly support named params via **kwargs.
            # More complex logic or different asyncpg features would be needed.
            raise NotImplementedError(
                "Mapping parameters are not directly supported by this adapter's "
                "execute method for asyncpg.",
            )
        # Should not happen with Union type hint, but as a safeguard:
        raise UnsupportedParamsTypeError(type(params))

    # Potentially delegate other methods like close, is_closed, etc. if needed
    # and if they are part of DBConnection protocol


# --- Adapter for asyncpg.Pool to match PoolProtocol ---
class AsyncpgPoolAdapter(PoolProtocol):
    """Adapts an asyncpg.Pool to the PoolProtocol."""

    def __init__(self, actual_pool: asyncpg.Pool) -> None:
        """Initialize the adapter with an asyncpg pool.

        Args:
            actual_pool: The asyncpg pool to adapt
        """
        self._pool = actual_pool

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DBConnection]:  # type: ignore[override]
        """Acquires an adapted connection from the pool."""
        # The type: ignore[override] can be needed if type checker has trouble with
        # AsyncContextManager variance or our adapter vs DBConnection in this context.
        async with self._pool.acquire() as actual_conn:
            yield AsyncpgConnectionAdapter(actual_conn)

    async def release(self, conn: DBConnection) -> None:
        """Releases an adapted connection back to the pool."""
        # This is tricky. The `conn` here will be our AsyncpgConnectionAdapter.
        # We need to unwrap it to release the actual asyncpg.Connection.
        # This will be addressed by adding a property to AsyncpgConnectionAdapter.
        if isinstance(conn, AsyncpgConnectionAdapter):
            # Assuming _conn is accessible; consider making it a public property.
            await self._pool.release(conn.actual_connection)  # Changed from conn._conn
        else:
            # This case should ideally not happen if acquire always returns the adapter.
            # For now, we assume conn will be our adapter.
            # If this adapter is used with other DBConnection types, this release is problematic.
            pass  # Or raise TypeError("Cannot release non-adapter to AsyncpgPoolAdapter")

    async def close(self) -> None:
        """Close the underlying connection pool."""
        await self._pool.close()
