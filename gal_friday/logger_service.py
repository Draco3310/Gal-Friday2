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

# SQLAlchemy imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError # For error handling

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


# Define a proper type alias for the Pool type - No longer needed for SQLAlchemy with LoggerService
# PoolType = TypeVar("PoolType", bound=PoolProtocol) # REMOVING

# Placeholder Type Hints (Refine later if needed)
if TYPE_CHECKING:
    # Use actual type if stubs were available
    from influxdb_client import InfluxDBClient
    from influxdb_client import Point as InfluxDBPoint
    from influxdb_client.client.write_api import WriteApi
    from sqlalchemy.ext.asyncio import AsyncSession  # For AsyncSessionFactory type hint
    from sqlalchemy.orm import sessionmaker  # For AsyncSessionFactory type hint
    from typing import AsyncContextManager
    
    # Define type aliases for better type hints
    AsyncPostgresHandlerType = logging.Handler
    AsyncSessionFactoryType = Callable[[], AsyncContextManager[AsyncSession]]
else:
    # Define placeholders if not type checking to avoid runtime errors
    AsyncPostgresHandlerType = logging.Handler
    AsyncSessionFactoryType = Callable[..., Any]  # Placeholder for runtime




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
class AsyncPostgresHandler(logging.Handler):
    """Asynchronous handler for logging to PostgreSQL database using SQLAlchemy."""

    ALLOWED_TABLE_NAMES: ClassVar[set[str]] = {
        "logs",
    }

    def __init__(
        self, session_factory: AsyncSessionFactoryType, table_name: str
    ) -> None:
        """Initialize the handler with SQLAlchemy session factory and table name.

        Args:
        ----
            session_factory: SQLAlchemy async session factory.
            table_name: Name of the table to log to (must be 'logs').
        """
        super().__init__()
        if table_name not in self.ALLOWED_TABLE_NAMES:
            raise InvalidLoggerTableNameError(table_name, self.ALLOWED_TABLE_NAMES)

        self._session_factory = session_factory
        self._table_name = table_name
        # Ensure event loop is acquired correctly, especially if LoggerService is init before loop starts
        self._loop: asyncio.AbstractEventLoop | None = None # Will be set in start_processing or when loop is available

        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._closed = False
        self._task: asyncio.Task[None] | None = None

    def _ensure_loop(self) -> None:
        """Ensures the event loop is available."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # This case might happen if the handler is initialized early.
                # The loop will be picked up by create_task if it's running then.
                # Or, LoggerService.start() can explicitly call this.
                self._loop = asyncio.get_event_loop_policy().get_event_loop()


    def start_processing(self) -> None:
        """Starts the queue processing task. Should be called when an event loop is available."""
        self._ensure_loop() # Ensure loop is set
        if self._task is None and not self._closed and self._loop:
            self._task = self._loop.create_task(self._process_queue())


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
        context_data = None # Changed from context_json
        if hasattr(record, "context") and record.context:
            if isinstance(record.context, dict) or isinstance(record.context, list):
                context_data = record.context # Keep as Python dict/list for SQLAlchemy
            else:
                try:
                    # Attempt to load if it's a JSON string
                    context_data = json.loads(str(record.context))
                except json.JSONDecodeError:
                    # Fallback for non-serializable or non-JSON string
                    context_data = {"raw_context": str(record.context)}


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
            "context_json": context_data, # Renamed from context_json
            "exception_text": exc_text,
        }

<<<<<<< HEAD
    async def _insert_log_sqlalchemy(self, record_data: dict[str, Any]) -> bool:
        """Insert a single log record into the database using SQLAlchemy."""
        # Late import to avoid circular dependency if models are in the same dir/package
        # and also to ensure models are loaded.
        # A better place might be a global import at the top of the file if structure allows.
        # For now, let's assume it's fine here or move it up if there are issues.
        from ..models.log import Log # Adjusted relative import
        from sqlalchemy.exc import SQLAlchemyError # For catching SQLAlchemy errors

        try:
            async with self._session_factory() as session: # type: ignore # session_factory might be Any at runtime
                async with session.begin(): # Handles commit/rollback
                    log_entry = Log(
                        timestamp=record_data["timestamp"],
                        logger_name=record_data["logger_name"],
                        level_name=record_data["level_name"],
                        level_no=record_data["level_no"],
                        message=record_data["message"],
                        pathname=record_data["pathname"],
                        filename=record_data["filename"],
                        lineno=record_data["lineno"],
                        func_name=record_data["func_name"],
                        context_json=record_data["context_json"],
                        exception_text=record_data["exception_text"],
                    )
                    session.add(log_entry)
                # session.commit() is called by session.begin() context manager on success
            return True
        except SQLAlchemyError as e: # Catch generic SQLAlchemy errors
            # Specific connection errors for retry could be (OperationalError, InterfaceError, etc.)
            # from sqlalchemy.exc
            # For now, a generic catch. This will be handled by retry logic.
            print(
                f"AsyncPostgresHandler: SQLAlchemy error inserting log record: {e}",
                file=sys.stderr,
            )
            # Re-raise to be caught by the retry logic
            # Important: Ensure the specific errors re-raised are what the retry logic expects
            # For now, let's assume SQLAlchemyError itself can be a trigger for retry
            # if it's a connection issue. This might need refinement.
            from sqlalchemy.exc import OperationalError # Example of a retryable error
            if isinstance(e, OperationalError): # Check if it's a connection-like error
                 raise # Re-raise to trigger retry
            return False # Indicate non-retryable failure for other SQLAlchemy errors
        except Exception as e: # Catch any other unexpected errors
            print(
                f"AsyncPostgresHandler: Unexpected error inserting log record: {e}",
                file=sys.stderr,
            )
            return False # Indicate non-retryable failure
=======
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
>>>>>>> main

    async def _process_queue_with_retry(self, record_data: dict[str, Any]) -> None:
        """Process a single record with retry logic using SQLAlchemy."""
        from sqlalchemy.exc import OperationalError # For specific retryable exceptions

        max_retries = 3
        base_backoff = 1.0  # seconds
        attempt = 0
        while attempt < max_retries:
            try:
                if await self._insert_log_sqlalchemy(record_data):
                    return  # Success
                # If _insert_log_sqlalchemy returns False (non-retryable SQLAlchemy error), exit loop
                print(
                    f"AsyncPostgresHandler: Non-retryable SQLAlchemy error for record. Skipping.",
                    file=sys.stderr
                )
                return # Stop processing this record
            except OperationalError as conn_err: # Catch specific SQLAlchemy connection errors
                attempt += 1
                if attempt >= max_retries:
<<<<<<< HEAD
                    print(
                        f"AsyncPostgresHandler: SQLAlchemy DB connection error failed after {max_retries} "
                        f"attempts: {conn_err}",
                        file=sys.stderr,
=======
                    logging.getLogger(__name__).error(
                        "AsyncPostgresHandler: DB connection error failed after %d attempts: %s",
                        max_retries,
                        str(conn_err),
                        exc_info=True,
>>>>>>> main
                    )
                else:
                    wait_time = min(base_backoff * (2**attempt), 30.0)
                    wait_time += SystemRandom().uniform(0, wait_time * 0.1)
<<<<<<< HEAD
                    print(
                        f"AsyncPostgresHandler: SQLAlchemy DB connection error (Attempt {attempt}/"
                        f"{max_retries}). Retrying in {wait_time:.2f}s. Error: {conn_err}",
                        file=sys.stderr,
=======
                    logging.getLogger(__name__).warning(
                        "AsyncPostgresHandler: DB connection error (Attempt %d/%d). "
                        "Retrying in %.2fs. Error: %s",
                        attempt,
                        max_retries,
                        wait_time,
                        str(conn_err),
>>>>>>> main
                    )
                    await asyncio.sleep(wait_time)
            except Exception as e: # Catch other unexpected errors from _insert_log_sqlalchemy
                print(
                    f"AsyncPostgresHandler: Unexpected error during retry processing: {e}. Skipping record.",
                    file=sys.stderr
                )
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
            # Ensure task is created before trying to put None on queue if it wasn't started
            if self._task is None: # Ensure the task is running or has run
                 self.start_processing() # Try to start it if it never did

            if self._loop.is_running(): # Check if loop is available for call_soon_threadsafe
                 self._loop.call_soon_threadsafe(lambda: self._queue.put_nowait(None))
            else: # Fallback if loop is closed, try to put directly (might fail if full)
                try:
                    self._queue.put_nowait(None)
                except QueueFull:
                    print("AsyncPostgresHandler: Queue full while trying to add sentinel in close().", file=sys.stderr)

        super().close() # Calls Handler.close()

    async def wait_closed(self) -> None:
        """Wait for the queue processing task to finish.

        This method should be awaited after calling close() to ensure
        all queued log records are processed before the application exits.
        """
        if self._task is not None:
            await self._task


# -------------------------------------


<<<<<<< HEAD
class LoggerService: # Removed Generic[PoolType]
=======

class LoggerService(Generic[PoolType]):
>>>>>>> main
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
            # Using a format that ContextFormatter can work with if context is present
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(context)s]",
        )
        self._log_date_format = self._config_manager.get(
            "logging.date_format",
            "%Y-%m-%d %H:%M:%S",
        )

        # SQLAlchemy specific attributes for database logging
        self._db_config: dict[str, Any] = self._config_manager.get("logging.database", {})
        self._db_enabled: bool = bool(self._db_config.get("enabled", False))
        self._sqlalchemy_engine: Any = None  # Will hold the SQLAlchemy async engine
        self._sqlalchemy_session_factory: AsyncSessionFactoryType | None = None
        self._async_handler: AsyncPostgresHandler | None = None # Type hint for SQLAlchemy handler

        # Queue and thread for handling synchronous logging calls from async context
<<<<<<< HEAD
        # (This is for the main logger, not the DB handler's internal queue)
=======
>>>>>>> main
        self._queue: queue.Queue[tuple[Callable[..., None], tuple, dict]] = queue.Queue()
        self._thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self._stop_event = threading.Event()
        self._loggers: dict[str, logging.Logger] = {}
        self._root_logger: logging.Logger = logging.getLogger("gal_friday")

        self._setup_logging()
        self._thread.start()
        self.info("LoggerService initialized.", source_module="LoggerService")

<<<<<<< HEAD
    def _process_log_queue(self) -> None:
        """Worker thread target to process log messages from the queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for an item with a timeout to allow checking the stop
                # event
                log_func, args, kwargs = self._queue.get(timeout=0.1)
                log_func(*args, **kwargs)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Log error using root logger to avoid recursion if self.error
                # uses the queue
                logging.exception("Error processing log queue item")

    def _setup_logging(self) -> None:
        """Configure root logger and handlers."""
        self._root_logger.setLevel(self._log_level)

        # Remove existing handlers
        for handler in self._root_logger.handlers[:]:
            self._root_logger.removeHandler(handler)
            handler.close()

        # --- Console Handler (Human-Readable) ---
        use_console = bool(self._config_manager.get("logging.console.enabled", default=True))
        if use_console:
            console_formatter = ContextFormatter(self._log_format, datefmt=self._log_date_format)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(self._log_level)
            self._root_logger.addHandler(console_handler)

        # --- File Handler (JSON Format) ---
        use_file = bool(self._config_manager.get("logging.file.enabled", default=True))
        if use_file:
            log_dir = str(self._config_manager.get("logging.file.directory", default="logs"))
            log_filename = str(
                self._config_manager.get("logging.file.filename", default="gal_friday.log"),
            )
            Path(log_dir).mkdir(parents=True, exist_ok=True)  # Shorter comment
            log_path = str(Path(log_dir) / log_filename)  # Shorter comment

            # Configure JSON Formatter
            json_formatter = jsonlogger.JsonFormatter(
                self._log_format,
                datefmt=self._log_date_format,
                rename_fields={"levelname": "level"},
            )

            max_bytes = int(
                self._config_manager.get("logging.file.max_bytes", default=10 * 1024 * 1024),
            )
            backup_count = int(self._config_manager.get("logging.file.backup_count", default=5))
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(json_formatter)
            file_handler.setLevel(self._log_level)
            self._root_logger.addHandler(file_handler)

        # --- Database Handler (SQLAlchemy) ---
        use_db = self._db_enabled
        if use_db:
            # SQLAlchemy engine and session factory should be initialized by start()
            # _setup_logging is called in __init__ before start, so factory might not be ready here.
            # The handler will be fully configured/added in start() if necessary.
            if self._sqlalchemy_session_factory:
                db_table = str(
                    self._config_manager.get("logging.database.table_name", default="logs"),
                )
                # Enforce 'logs' table name to match the Log model
                if db_table != "logs":
                    logging.warning(
                        "Database table name in config ('%s') differs from expected 'logs'. Using 'logs'.",
                        db_table,
                    )
                    db_table = "logs"

                db_level_str = str(
                    self._config_manager.get("logging.database.level", default="INFO"),
                ).upper()
                db_level = getattr(logging, db_level_str, logging.INFO)

                try:
                    self._async_handler = AsyncPostgresHandler(
                        self._sqlalchemy_session_factory, db_table
                    )
                    self._async_handler.setLevel(db_level)
                    self._root_logger.addHandler(self._async_handler)
                    # Handler's start_processing() will be called in LoggerService.start()
                    logging.info(
                        "SQLAlchemy Database logging handler configured for table '%s'. Level: %s",
                        db_table,
                        db_level_str,
                    )
                except InvalidLoggerTableNameError as e:
                    logging.error(f"Failed to initialize AsyncPostgresHandler due to table name: {e}")
                except Exception:
                    logging.exception("Failed to create or add AsyncPostgresHandler (SQLAlchemy)")
            else:
                logging.warning(
                    "Database logging enabled, but SQLAlchemy session factory not yet initialized "
                    "during _setup_logging. Handler will be set up in start().",
                )

=======
>>>>>>> main
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

        # Initialize SQLAlchemy engine and session factory if DB logging is enabled
        if self._db_enabled:
            await self._initialize_sqlalchemy()
            # If _setup_logging ran in __init__ before factory was ready,
            # _async_handler might be None. We need to set it up now.
            if self._async_handler is None and self._sqlalchemy_session_factory:
                self.info(
                    "SQLAlchemy initialized. Re-evaluating database log handler setup.",
                    source_module="LoggerService"
                )
                # Temporarily remove and re-add handlers to ensure correct setup
                # This is a bit heavy-handed; a more refined approach might be better
                # but this ensures correctness if _setup_logging is complex.
                # For now, directly try to add the handler if missing.
                current_handlers = self._root_logger.handlers[:]
                for handler in current_handlers:
                    if isinstance(handler, AsyncPostgresHandler): # Remove any old/stale one
                        self._root_logger.removeHandler(handler)
                        handler.close()
                # Re-run the DB handler part of _setup_logging logic
                db_table = str(self._config_manager.get("logging.database.table_name", default="logs"))
                if db_table != "logs": db_table = "logs" # Enforce
                db_level_str = str(self._config_manager.get("logging.database.level", default="INFO")).upper()
                db_level = getattr(logging, db_level_str, logging.INFO)
                try:
                    self._async_handler = AsyncPostgresHandler(self._sqlalchemy_session_factory, db_table)
                    self._async_handler.setLevel(db_level)
                    self._root_logger.addHandler(self._async_handler)
                    logging.info("SQLAlchemy Database logging handler added/updated in start().")
                except Exception:
                    logging.exception("Failed to add/update AsyncPostgresHandler in start()")


        # Subscribe to LOG events from the event bus
        try:
            self._pubsub.subscribe(EventType.LOG_ENTRY, self._handle_log_event)
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

        # Start the DB handler's internal processing task
        if self._async_handler:
            self._async_handler.start_processing()
            self.info("AsyncPostgresHandler processing started.", source_module="LoggerService")
        elif self._db_enabled:
            self.error("DB logging enabled, but AsyncPostgresHandler not initialized in start().", source_module="LoggerService")


    async def stop(self) -> None:
        """Shut down the logger service gracefully.

        Closes the database handler, SQLAlchemy engine, and logging threads.
        """
        self.info("LoggerService stop sequence initiated.", source_module="LoggerService")

        # Unsubscribe from LOG events
        try:
            self._pubsub.unsubscribe(EventType.LOG_ENTRY, self._handle_log_event)
            self.info("Unsubscribed from LOG_ENTRY events.", source_module="LoggerService")
        except Exception as e:
            self.error(
                "Error unsubscribing from LOG_ENTRY events: %s",
                e,
                source_module="LoggerService",
                exc_info=True,
            )

        # Close the custom SQLAlchemy handler first
        if self._async_handler:
            self.info("Closing SQLAlchemy database log handler...", source_module="LoggerService")
            self._async_handler.close()
            try:
                if hasattr(self._async_handler, "wait_closed"):
                    await asyncio.wait_for(self._async_handler.wait_closed(), timeout=10.0) # Increased timeout
                    self.info(
                        "SQLAlchemy database log handler closed gracefully.",
                        source_module="LoggerService",
                    )
            except TimeoutError:
                self.warning(
                    "Timeout waiting for SQLAlchemy database log handler queue to empty.",
                    source_module="LoggerService",
                )
            except Exception as e:
                self.error(
                    "Error waiting for SQLAlchemy database log handler closure: %s",
                    e,
                    source_module="LoggerService",
                    exc_info=True,
                )

        # Dispose of the SQLAlchemy engine
        if self._sqlalchemy_engine:
            self.info("Disposing SQLAlchemy engine...", source_module="LoggerService")
            try:
                await self._sqlalchemy_engine.dispose()
                self.info("SQLAlchemy engine disposed.", source_module="LoggerService")
            except Exception as e:
                self.error(
                    "Error disposing SQLAlchemy engine: %s", e, source_module="LoggerService", exc_info=True
                )

        # Signal the main log processing thread to stop (for non-DB logs if any are queued)
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self.warning(
                "Main log processing thread did not exit cleanly.",
                source_module="LoggerService",
            )

    async def _initialize_sqlalchemy(self) -> None:
        """Initialize the SQLAlchemy engine and session factory for database logging."""
        if not self._db_enabled:
            self.info("Database logging is disabled. SQLAlchemy setup skipped.", source_module="LoggerService")
            return

        db_url = self._config_manager.get("logging.database.connection_string")
        if not db_url:
            self.error(
                "Database logging enabled but connection_string is missing. SQLAlchemy setup failed.",
                source_module="LoggerService",
            )
            self._db_enabled = False # Disable DB logging if URL is missing
            return

        try:
            self.info(
                "Initializing SQLAlchemy async engine for logging...",
                source_module="LoggerService",
                context={"database_url": str(db_url)[:str(db_url).find("@")] + "@********" if "@" in str(db_url) else str(db_url)} # Mask credentials
            )
            pool_size = self._config_manager.get_int("logging.database.pool_size", 5)
            max_overflow = self._config_manager.get_int("logging.database.max_overflow", 10)
            echo_sql = self._config_manager.get("logging.database.echo_sql", False)

            self._sqlalchemy_engine = create_async_engine(
                str(db_url), # Ensure db_url is a string
                pool_size=pool_size,
                max_overflow=max_overflow,
                echo=echo_sql,
            )
            self._sqlalchemy_session_factory = sessionmaker(
                self._sqlalchemy_engine, class_=AsyncSession, expire_on_commit=False
            )
            self.info(
                "SQLAlchemy async engine and session factory initialized successfully.",
                source_module="LoggerService",
            )
        except SQLAlchemyError as e: # Catch SQLAlchemy specific errors
            self.critical(
                "Failed to initialize SQLAlchemy engine: %s. Disabling DB logging.",
                e,
                source_module="LoggerService",
                exc_info=True,
            )
            self._sqlalchemy_engine = None
            self._sqlalchemy_session_factory = None
            self._db_enabled = False # Disable DB logging on error
        except Exception as e: # Catch any other unexpected errors
            self.critical(
                "An unexpected error occurred during SQLAlchemy engine initialization: %s. Disabling DB logging.",
                e,
                source_module="LoggerService",
                exc_info=True,
            )
            self._sqlalchemy_engine = None
            self._sqlalchemy_session_factory = None
            self._db_enabled = False # Disable DB logging on error


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


<<<<<<< HEAD
# Removing AsyncpgConnectionAdapter and AsyncpgPoolAdapter as they are no longer needed.
# Also removing unused DBConnection and PoolProtocol if they were solely for this.
# If they are used elsewhere, they should remain. Assuming they are not for now.
=======
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
>>>>>>> main
