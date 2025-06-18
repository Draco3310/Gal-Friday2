# Logger Service Module
"""Logging service module providing centralized logging capabilities with multiple output targets.

This module implements a comprehensive logging service that handles logging to console,
files, databases and time-series databases. It provides thread-safety, async support,
and context-rich logging capabilities with enterprise-grade handler implementations.
"""

import asyncio
import contextlib  # Added for SIM105
import json
import logging
import logging.handlers
import queue
import re
import sys
import threading
import time
import types  # Added for exc_info typing
from abc import ABC, abstractmethod
from asyncio import QueueFull  # Import QueueFull from asyncio, not asyncio.exceptions
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path  # Add missing import
from random import SystemRandom
from typing import (
    # Add Point type for type checking
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast)
from typing import (
    TypeAlias as TypingTypeAlias)

from .interfaces.service_protocol import ServiceProtocol

from pythonjsonlogger import jsonlogger  # Add missing import for JSON logging

# Runtime imports
# from influxdb_client import Point as InfluxDBPoint # Moved into methods

if TYPE_CHECKING:
    # For type hinting InfluxDBClient and WriteApi if needed at class/method signature level
    from asyncpg import Connection as AsyncpgConnection
    from asyncpg import Pool as AsyncpgPool
    from asyncpg.exceptions import (
        ConnectionDoesNotExistError as AsyncpgConnectionDoesNotExistError)
    from asyncpg.exceptions import ConnectionIsClosedError as AsyncpgConnectionIsClosedError
    from asyncpg.exceptions import InterfaceError as AsyncpgInterfaceError
    from asyncpg.exceptions import PostgresError as AsyncpgPostgresError
    from influxdb_client import InfluxDBClient
    from influxdb_client import Point as InfluxDBPoint
    from influxdb_client.client.write_api import WriteApi
else:
    # Define placeholders if asyncpg is not available at runtime for type checking purposes
    # This helps prevent ModuleNotFoundError if asyncpg is not installed during e.g. linting
    # or if a part of the code is executed where asyncpg is not strictly needed.
    AsyncpgConnection = Any
    AsyncpgPool = Any
    AsyncpgConnectionDoesNotExistError = Exception
    AsyncpgConnectionIsClosedError = Exception
    AsyncpgInterfaceError = Exception
    AsyncpgPostgresError = Exception

# SQLAlchemy imports for refactored DB logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from gal_friday.dal.models.log import Log  # Import the Log model

# Import JSON Formatter
from .core.events import EventType, LogEvent

if TYPE_CHECKING:
    from .core.pubsub import PubSubManager

# Import custom exceptions
# SQLAlchemy imports
from sqlalchemy.ext.asyncio import create_async_engine

from .exceptions import DatabaseError

# Type[Any] variables for generic protocols
_T = TypeVar("_T")
_RT = TypeVar("_RT")

# Define a type alias for exc_info to improve readability and manage line length
# Moved to module level and updated to use Union, Tuple
ExcInfoType: TypingTypeAlias = (
    bool | tuple[type[BaseException], BaseException, types.TracebackType] | BaseException | None
)


# ========================================
# Enterprise-Grade Handler Infrastructure
# ========================================


class LogLevel(str, Enum):
    """Log levels for enterprise logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HandlerType(str, Enum):
    """Types of logging handlers supported by the enterprise logging system."""

    CONSOLE = "console"
    FILE = "file"
    ROTATING_FILE = "rotating_file"
    TIMED_ROTATING_FILE = "timed_rotating_file"
    SYSLOG = "syslog"
    HTTP = "http"
    SMTP = "smtp"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"
    KAFKA = "kafka"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class HandlerConfig:
    """Configuration for enterprise logging handlers."""

    handler_type: HandlerType
    name: str
    level: LogLevel = LogLevel.INFO
    format_string: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict[str, Any])
    enabled: bool = True
    filters: List[str] = field(default_factory=list[Any])


class LogHandlerProtocol(Protocol):
    """Protocol defining the interface for logging handlers."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        ...

    def close(self) -> None:
        """Close the handler."""
        ...

    def flush(self) -> None:
        """Flush any pending output."""
        ...


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
        params: Sequence[Any] | Mapping[str, Any] | None = None) -> _RT:
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


# Define a proper type alias for the Pool type
# PoolType = TypeVar("PoolType", bound=PoolProtocol) # Replaced by async_sessionmaker
# PoolProtocol and DBConnection might be removable if AsyncpgPoolAdapter is removed.

# ========================================
# Enterprise Handler Implementations
# ========================================


class BaseLogHandler(logging.Handler, ABC):
    """Base class for enterprise logging handlers with performance tracking."""

    def __init__(self, config: HandlerConfig) -> None:
        super().__init__()
        self.config = config
        self.setLevel(getattr(logging, config.level.value))

        # Performance tracking
        self.emit_count = 0
        self.error_count = 0
        self.last_emit_time = 0.0

        # Setup formatter
        if config.format_string:
            formatter = logging.Formatter(config.format_string)
            self.setFormatter(formatter)

    @abstractmethod
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record - must be implemented by subclasses."""
        pass

    def handle_emit_error(self, record: logging.LogRecord, error: Exception) -> None:
        """Handle errors during record emission."""
        self.error_count += 1
        self.handleError(record)


class EnterpriseConsoleLogHandler(BaseLogHandler):
    """Enterprise console logging handler with color support."""

    def __init__(self, config: HandlerConfig) -> None:
        super().__init__(config)
        self.stream = self.config.parameters.get("stream", "stdout")

        if self.stream == "stderr":
            self.target_stream = sys.stderr
        else:
            self.target_stream = sys.stdout

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to console with optional color coding."""
        try:
            self.emit_count += 1
            self.last_emit_time = time.time()

            msg = self.format(record)

            # Add color coding if enabled
            if self.config.parameters.get("colorize", False):
                msg = self._colorize_message(msg, record.levelno)

            self.target_stream.write(msg + "\n")
            self.target_stream.flush()

        except Exception as e:
            self.handle_emit_error(record, e)

    def _colorize_message(self, message: str, level: int) -> str:
        """Add ANSI color codes based on log level."""
        color_map = {
            logging.DEBUG: "\033[36m",  # Cyan
            logging.INFO: "\033[32m",  # Green
            logging.WARNING: "\033[33m",  # Yellow
            logging.ERROR: "\033[31m",  # Red
            logging.CRITICAL: "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        color = color_map.get(level, "")
        return f"{color}{message}{reset}"


class EnterpriseRotatingFileLogHandler(BaseLogHandler):
    """Enterprise rotating file logging handler."""

    def __init__(self, config: HandlerConfig) -> None:
        super().__init__(config)

        # Extract file rotation parameters
        self.filename = config.parameters.get("filename", "app.log")
        self.max_bytes = config.parameters.get("max_bytes", 10 * 1024 * 1024)  # 10MB
        self.backup_count = config.parameters.get("backup_count", 5)

        # Create the actual rotating file handler
        self.file_handler = logging.handlers.RotatingFileHandler(
            filename=self.filename,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8")

        if self.formatter:
            self.file_handler.setFormatter(self.formatter)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to rotating file."""
        try:
            self.emit_count += 1
            self.last_emit_time = time.time()

            self.file_handler.emit(record)

        except Exception as e:
            self.handle_emit_error(record, e)

    def close(self) -> None:
        """Close the file handler."""
        self.file_handler.close()
        super().close()


class EnterpriseElasticsearchLogHandler(BaseLogHandler):
    """Enterprise Elasticsearch logging handler with batching."""

    def __init__(self, config: HandlerConfig) -> None:
        super().__init__(config)

        # Elasticsearch configuration
        self.hosts = config.parameters.get("hosts", ["localhost:9200"])
        self.index_pattern = config.parameters.get("index_pattern", "logs-%Y.%m.%d")
        self.doc_type = config.parameters.get("doc_type", "_doc")

        # Initialize Elasticsearch client
        self.es_client = self._create_es_client()

        # Batch processing
        self.batch_size = config.parameters.get("batch_size", 100)
        self.batch_timeout = config.parameters.get("batch_timeout", 5.0)
        self.batch_buffer: List[Dict[str, Any]] = []
        self.last_flush_time = time.time()

    def _create_es_client(self) -> Any:
        """Create Elasticsearch client if available."""
        try:
            from elasticsearch import Elasticsearch

            return Elasticsearch(self.hosts)
        except ImportError:
            logging.getLogger(__name__).warning(
                "Elasticsearch client not available. Install 'elasticsearch' package."
            )
            return None

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to Elasticsearch."""
        if not self.es_client:
            return

        try:
            self.emit_count += 1
            self.last_emit_time = time.time()

            # Convert log record to document
            doc = self._record_to_document(record)

            # Add to batch buffer
            self.batch_buffer.append(doc)

            # Check if batch should be flushed
            if (
                len(self.batch_buffer) >= self.batch_size
                or time.time() - self.last_flush_time > self.batch_timeout
            ):
                self._flush_batch()

        except Exception as e:
            self.handle_emit_error(record, e)

    def _record_to_document(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert log record to Elasticsearch document."""
        doc = {
            "@timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add exception info if present
        if record.exc_info:
            doc["exception"] = self.format(record)

        # Add extra fields
        if hasattr(record, "context"):
            doc["context"] = record.context

        return doc

    def _flush_batch(self) -> None:
        """Flush batch buffer to Elasticsearch."""
        if not self.batch_buffer or not self.es_client:
            return

        try:
            # Create bulk body for Elasticsearch
            bulk_body = []
            for doc in self.batch_buffer:
                index_name = time.strftime(self.index_pattern)
                bulk_body.append({"index": {"_index": index_name, "_type": self.doc_type}})
                bulk_body.append(doc)

            # Execute bulk operation
            self.es_client.bulk(body=bulk_body)

            self.batch_buffer.clear()
            self.last_flush_time = time.time()

        except Exception as e:
            self.error_count += 1
            logging.getLogger(__name__).error(f"Failed to flush batch to Elasticsearch: {e}")


class EnterpriseInfluxDBLogHandler(BaseLogHandler):
    """Enterprise InfluxDB logging handler for time-series log data."""

    def __init__(self, config: HandlerConfig) -> None:
        super().__init__(config)

        # InfluxDB configuration
        self.host = config.parameters.get("host", "localhost")
        self.port = config.parameters.get("port", 8086)
        self.database = config.parameters.get("database", "logs")
        self.measurement = config.parameters.get("measurement", "application_logs")

        # Initialize InfluxDB client
        self.influx_client: Any = self._create_influx_client()

        # Batch processing
        self.batch_size = config.parameters.get("batch_size", 100)
        self.batch_buffer: List[Dict[str, Any]] = []

    def _create_influx_client(self) -> Any:
        """Create InfluxDB client if available."""
        try:
            from influxdb_client import InfluxDBClient

            token = self.config.parameters.get("token", "")
            org = self.config.parameters.get("org", "")
            return InfluxDBClient(
                url=f"http://{self.host}:{self.port}",
                token=token,
                org=org)
        except ImportError:
            logging.getLogger(__name__).warning(
                "InfluxDB client not available. Install 'influxdb-client' package."
            )
            return None

    def emit(self, record: logging.LogRecord) -> None:
        """Emit record to InfluxDB."""
        if not self.influx_client:
            return

        try:
            self.emit_count += 1
            self.last_emit_time = time.time()

            # Convert log record to InfluxDB point
            point = self._record_to_point(record)

            # Add to batch buffer
            self.batch_buffer.append(point)

            # Flush if batch is full
            if len(self.batch_buffer) >= self.batch_size:
                self._flush_batch()

        except Exception as e:
            self.handle_emit_error(record, e)

    def _record_to_point(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Convert log record to InfluxDB point."""
        return {
            "measurement": self.measurement,
            "time": int(record.created * 1000000000),  # Nanoseconds
            "tags": {
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
            },
            "fields": {
                "message": record.getMessage(),
                "line": record.lineno,
                "thread": record.thread,
                "process": record.process,
            },
        }

    def _flush_batch(self) -> None:
        """Flush batch buffer to InfluxDB."""
        if not self.batch_buffer or not self.influx_client:
            return

        try:
            write_api = self.influx_client.write_api()
            write_api.write(
                bucket=self.config.parameters.get("bucket", "logs"), record=self.batch_buffer
            )

            self.batch_buffer.clear()

        except Exception as e:
            self.error_count += 1
            logging.getLogger(__name__).error(f"Failed to flush batch to InfluxDB: {e}")


# LogHandlerFactory will be defined after all handler classes


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
        # Ensure 'context' attribute exists on the record before super().format()
        # if the main format string self._fmt (used by super().format()) contains '%(context)s'.
        if not hasattr(record, "context"):
            record.context = {}  # Default to an empty dict[str, Any] if not present

        # Default formatting first
        s = super().format(record)

        # The following logic is for custom post-processing of a literal "[%(context)s]"
        # string that might have been in the original format string, or further refines
        # the output of what %(context)s produced.
        context_str = ""
        if (
            hasattr(record, "context") and record.context
        ):  # record.context is now guaranteed to exist
            if isinstance(record.context, dict):
                context_items = [f"{k}={v}" for k, v in record.context.items()]
                context_str = ", ".join(context_items)
            else:
                context_str = str(record.context)  # Fallback if not a dict[str, Any]

        # Replace the placeholder [%(context)s] - handle case where it might be
        # missing
        if self._fmt is not None and "[%(context)s]" in self._fmt:
            if context_str:
                s = s.replace("[%(context)s]", f"[{context_str}]")
            else:
                s = s.replace(
                    " - [%(context)s]",
                    "")  # Remove placeholder and separator if no context
                # Remove just placeholder if at start/end
                s = s.replace("[%(context)s]", "")

        return s


# --- Enterprise Async Database Handler ---
class EnterpriseAsyncPostgresHandler(BaseLogHandler):
    """Enterprise asynchronous handler for logging to PostgreSQL database using SQLAlchemy."""

    def __init__(
        self,
        config: HandlerConfig,
        session_maker: async_sessionmaker[AsyncSession],
        loop: asyncio.AbstractEventLoop) -> None:
        """Initialize the handler with SQLAlchemy session maker and event loop.

        Args:
        ----
            config: Handler configuration for the enterprise system.
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            loop: Event loop to use for async operations.
        """
        super().__init__(config)
        self._session_maker = session_maker
        self._loop = loop  # Still needed for call_soon_threadsafe
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None  # Type[Any] hint for task
        self._closed = False

    def start_processing(self) -> None:
        """Starts the queue processing task. Should be called when an event loop is available."""
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
                data)
        except (ValueError, TypeError, RuntimeError, QueueFull):
            self.handleError(record)

    def _format_record(self, record: logging.LogRecord) -> dict[str, Any]:
        """Format the log record into a dictionary suitable for DB insertion."""
        context_data = None  # Changed from context_json
        if hasattr(record, "context") and record.context:
            if isinstance(record.context, dict) or isinstance(record.context, list):
                context_data = record.context  # Keep as Python dict[str, Any]/list[Any] for SQLAlchemy
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
            "context_json": context_data,  # Renamed from context_json
            "exception_text": exc_text,
        }

    async def _attempt_db_insert(self, record_data: dict[str, Any]) -> bool:
        """Attempt to insert a single log record into the database.

        Note: The table name is implicitly handled by the Log model.

        Returns:
            bool: True if successful, False if failed (caller determines retry)
        """
        try:
            # Map record_data keys to Log model attributes if they differ.
            # Assuming _format_record produces keys matching Log model attributes.
            log_entry = Log(**record_data)

            async with self._session_maker() as session:
                async with session.begin():  # Start a transaction
                    session.add(log_entry)
                # Commit happens automatically with session.begin() context manager,
                # or call await session.commit() if not using begin()
            return True
        except SQLAlchemyError as e:  # Catch specific SQLAlchemy errors
            # Improved error analysis to determine retryability
            error_msg = str(e)

            # Check for specific retryable conditions
            retryable_conditions = [
                "connection",
                "timeout",
                "deadlock",
                "lock",
                "could not connect",
                "connection reset",
                "broken pipe",
                "resource temporarily unavailable",
                "too many connections",
            ]

            is_retryable = any(
                condition in error_msg.lower() for condition in retryable_conditions
            )

            # Check for specific non-retryable conditions
            non_retryable_conditions = [
                "syntax error",
                "column",
                "constraint",
                "relation",
                "does not exist",
                "invalid",
                "permission denied",
                "authentication failed",
            ]

            is_non_retryable = any(
                condition in error_msg.lower() for condition in non_retryable_conditions
            )

            # Log with appropriate level based on retryability
            if is_non_retryable or (not is_retryable and hasattr(e, "orig")):
                # Check original database error if available
                if hasattr(e, "orig"):
                    orig_error = str(e.orig)
                    if any(condition in orig_error.lower() for condition in retryable_conditions):
                        is_retryable = True

            log_level = logging.WARNING if is_retryable else logging.ERROR
            logging.getLogger(__name__).log(
                log_level,
                "SQLAlchemy error in _attempt_db_insert: %s (retryable=%s)",
                str(e),
                is_retryable,
                exc_info=True)

            # Raise only if retryable to trigger retry logic
            if is_retryable:
                raise
            return False  # Non-retryable error
        except Exception as e:  # Catch any other unexpected errors
            logging.getLogger(__name__).error(
                "Unexpected error in _attempt_db_insert: %s",
                str(e),
                exc_info=True)
            return False

    async def _process_queue_with_retry(self, record_data: dict[str, Any]) -> None:
        """Process a single record with retry logic using SQLAlchemy."""
        max_retries = 3
        base_backoff = 1.0  # seconds
        attempt = 0

        while attempt < max_retries:
            try:
                if await self._attempt_db_insert(record_data):
                    return  # Success
                else:
                    # Non-retryable error, stop trying
                    return
            except SQLAlchemyError as db_err:
                # This exception is only raised for retryable errors
                attempt += 1
                if attempt >= max_retries:
                    logging.getLogger(__name__).error(
                        "AsyncPostgresHandler: Database operation failed after %d attempts: %s",
                        max_retries,
                        str(db_err),
                        exc_info=True)
                    return
                else:
                    # Exponential backoff with jitter
                    wait_time = min(base_backoff * (2 ** (attempt - 1)), 30.0)
                    wait_time += SystemRandom().uniform(0, wait_time * 0.1)
                    logging.getLogger(__name__).warning(
                        "AsyncPostgresHandler: Retryable database error (Attempt %d/%d). "
                        "Retrying in %.2fs. Error: %s",
                        attempt,
                        max_retries,
                        wait_time,
                        str(db_err))
                    await asyncio.sleep(wait_time)
            except OSError as conn_err:  # Network/connection issues
                attempt += 1
                if attempt >= max_retries:
                    logging.getLogger(__name__).error(
                        "AsyncPostgresHandler: Network/OS error failed after %d attempts: %s",
                        max_retries,
                        str(conn_err),
                        exc_info=True)
                    return
                else:
                    wait_time = min(base_backoff * (2 ** (attempt - 1)), 30.0)
                    wait_time += SystemRandom().uniform(0, wait_time * 0.1)
                    logging.getLogger(__name__).warning(
                        "AsyncPostgresHandler: Network/OS error (Attempt %d/%d). "
                        "Retrying in %.2fs. Error: %s",
                        attempt,
                        max_retries,
                        wait_time,
                        str(conn_err))
                    await asyncio.sleep(wait_time)
            except (RuntimeError, ValueError, TypeError) as e:
                # Non-database errors, likely programming errors - don't retry
                logging.getLogger(__name__).error(
                    "AsyncPostgresHandler: Non-retryable error for record: %s. Error: %s",
                    record_data.get("message", "N/A"),
                    e,
                    exc_info=True)
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
                    "AsyncPostgresHandler queue processing cancelled.")
                if record_data is not None:
                    with contextlib.suppress(ValueError):
                        self._queue.task_done()
                break
            except Exception as e:
                logging.getLogger(__name__).error(
                    "AsyncPostgresHandler: Error in outer processing loop: %s",
                    e,
                    exc_info=True)
                if record_data is not None:
                    with contextlib.suppress(ValueError):
                        self._queue.task_done()
                await asyncio.sleep(1)

    def close(self) -> None:
        """Close the handler, signaling the queue processor to finish."""
        if not self._closed:
            self._closed = True
            # Ensure task is created before trying to put None on queue if it wasn't started
            if self._task is None and self._loop:  # Ensure the task is running or has run
                self.start_processing()  # Try to start it if it never did

            if (
                self._loop and self._loop.is_running()
            ):  # Check if loop is available for call_soon_threadsafe
                self._loop.call_soon_threadsafe(lambda: self._queue.put_nowait(None))
            else:  # Fallback if loop is closed, try to put directly (might fail if full)
                try:
                    self._queue.put_nowait(None)
                except QueueFull:
                    logging.getLogger(__name__).error(
                        "AsyncPostgresHandler: Queue full while trying to add sentinel in close()."
                    )

        super().close()  # Calls Handler.close()

    async def wait_closed(self) -> None:
        """Wait for the queue processing task to finish.

        This method should be awaited after calling close() to ensure
        all queued log records are processed before the application exits.
        """
        if self._task is not None:
            await self._task


# -------------------------------------


class LogHandlerFactory:
    """Factory for creating enterprise logging handlers with proper type annotations."""

    # Mapping[Any, Any] of handler types to concrete classes
    HANDLER_CLASSES: Dict[HandlerType, Type[BaseLogHandler]] = {
        HandlerType.CONSOLE: EnterpriseConsoleLogHandler,
        HandlerType.ROTATING_FILE: EnterpriseRotatingFileLogHandler,
        HandlerType.ELASTICSEARCH: EnterpriseElasticsearchLogHandler,
        HandlerType.INFLUXDB: EnterpriseInfluxDBLogHandler,
        HandlerType.DATABASE: EnterpriseAsyncPostgresHandler,
    }

    @classmethod
    def create_handler(cls, config: HandlerConfig) -> BaseLogHandler:
        """Create logging handler from configuration.

        Replaces placeholder handler types with concrete implementations.

        Args:
            config: Handler configuration specifying type and parameters

        Returns:
            Concrete handler implementation

        Raises:
            ValueError: If handler type is not supported
            RuntimeError: If handler creation fails
        """
        handler_class = cls.HANDLER_CLASSES.get(config.handler_type)

        if not handler_class:
            raise ValueError(f"Unsupported handler type: {config.handler_type}")

        try:
            handler = handler_class(config)

            # Apply filters if configured
            for filter_name in config.filters:
                log_filter = cls._create_filter(filter_name)
                if log_filter:
                    handler.addFilter(log_filter)

            return handler

        except Exception as e:
            raise RuntimeError(f"Failed to create handler {config.name}: {e}")

    @classmethod
    def register_handler_class(
        cls, handler_type: HandlerType, handler_class: Type[BaseLogHandler]
    ) -> None:
        """Register custom handler class for extensibility."""
        cls.HANDLER_CLASSES[handler_type] = handler_class

    @classmethod
    def _create_filter(cls, filter_name: str) -> Optional[logging.Filter]:
        """Create logging filter by name."""
        # Placeholder for filter creation logic - can be extended
        return None


class LoggerService(ServiceProtocol):
    """Enterprise logging service with comprehensive handler management.

    Configures and manages logging to multiple destinations including console,
    files, databases and time-series databases with support for context information
    and enterprise-grade handler implementations.
    """

    _influx_client: Optional[Any] = None
    _influx_write_api: Optional[Any] = None
    _sqlalchemy_engine: AsyncEngine | None = None  # Added
    _sqlalchemy_session_factory: async_sessionmaker[AsyncSession] | None = None  # Added

    def __init__(
        self,
        config_manager: ConfigManagerProtocol,
        pubsub_manager: "PubSubManager",
        # Add db_session_maker for SQLAlchemy
        db_session_maker: async_sessionmaker[AsyncSession] | None = None) -> None:
        """Initialize the logger service.

        Sets up logging handlers based on configuration and starts the log processing thread.

        Args:
        ----
            config_manager: Configuration provider for logger settings.
            pubsub_manager: Publish-subscribe manager for event handling.
            db_session_maker: Optional SQLAlchemy async_sessionmaker for database logging.
        """
        self._config_manager = config_manager
        self._pubsub = pubsub_manager
        self._log_level = self._config_manager.get("logging.level", "INFO").upper()
        self._log_format = self._config_manager.get(
            "logging.format",
            # Using a format that ContextFormatter can work with if context is present
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(context)s]")
        self._log_date_format = self._config_manager.get(
            "logging.date_format",
            "%Y-%m-%d %H:%M:%S")

        # Enterprise handler registry with proper type annotations
        self._enterprise_handlers: Dict[str, Union[BaseLogHandler, EnterpriseAsyncPostgresHandler]] = {}
        self._handler_configs: Dict[str, HandlerConfig] = {}
        self._handler_stats: Dict[str, Dict[str, Any]] = {}

        # SQLAlchemy DB Handler setup
        self._db_config: dict[str, Any] = self._config_manager.get("logging.database", {})
        self._db_enabled = bool(self._db_config.get("enabled", False))
        self._db_session_maker = db_session_maker  # Store the session_maker
        self._async_handler: EnterpriseAsyncPostgresHandler | None = (
            None  # Updated to enterprise handler
        )
        # self._db_pool: PoolType | None = None # Removed, using session_maker

        # Queue and thread for handling synchronous logging calls from async context
        self._queue: queue.Queue[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]] = (
            queue.Queue()
        )  # Keep this for sync calls
        self._thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self._stop_event = threading.Event()
        self._loggers: dict[str, logging.Logger] = {}
        self._root_logger: logging.Logger = logging.getLogger("gal_friday")

        self._setup_logging()
        self._thread.start()
        self.info("LoggerService initialized.", source_module="LoggerService")

    async def initialize(self) -> None:
        """Async initialization hook for compatibility with ServiceProtocol."""
        # No asynchronous setup currently required
        return None

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
                self._config_manager.get("logging.file.filename", default="gal_friday.log"))
            Path(log_dir).mkdir(parents=True, exist_ok=True)  # Shorter comment
            log_path = str(Path(log_dir) / log_filename)  # Shorter comment

            # Configure JSON Formatter
            json_formatter = jsonlogger.JsonFormatter(
                self._log_format,
                datefmt=self._log_date_format,
                rename_fields={"levelname": "level"})

            max_bytes = int(
                self._config_manager.get("logging.file.max_bytes", default=10 * 1024 * 1024))
            backup_count = int(self._config_manager.get("logging.file.backup_count", default=5))
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8")
            file_handler.setFormatter(json_formatter)
            file_handler.setLevel(self._log_level)
            self._root_logger.addHandler(file_handler)

        # --- Database Handler (Legacy/SQLAlchemy) ---
        # Note: Enterprise handlers are initialized separately in start()
        # This section handles legacy database logging configuration
        use_db = self._db_enabled
        if use_db:
            # Database handler will be set up in start() when session factory is available
            # Either through enterprise handlers or legacy fallback
            logging.info(
                "Database logging enabled. Handler will be configured in start() method "
                "using enterprise or legacy approach based on configuration.")

    def _filter_sensitive_data(
        self,
        context: Mapping[str, object] | None,  # Changed to Mapping[Any, Any]
    ) -> dict[str, object] | None:
        """Recursively filter sensitive data from log context.

        Args:
            context: Dictionary containing log context data to be filtered

        Returns:
            Filtered dictionary with sensitive data redacted, or None if input is None/empty
        """
        if not context:  # Handle None or empty dict[str, Any]
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
            r"^[A-Za-z0-9/+]{20,}$")  # Example: Base64-like strings > 20 chars

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

    # ExcInfoType is now defined at the module level

    def log(
        self,
        level: int,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
        exc_info: ExcInfoType = None) -> None:
        """Log a message to the configured handlers.

        Args:
        ----
            level: The logging level (e.g., logging.INFO, logging.WARNING)
            message: The primary log message string (can be a format string)
            *args: Arguments for the format string in 'message'
            source_module: Optional name of the module generating the log
            context: Optional dictionary of key-value pairs for extra context
            exc_info: Optional exception info (e.g., True or exception tuple[Any, ...])
        """
        # Get the specific logger for the source module, or root if None
        logger_name = f"gal_friday.{source_module}" if source_module else "gal_friday"
        logger = logging.getLogger(logger_name)

        # Filter context BEFORE passing it as 'extra'
        filtered_context = self._filter_sensitive_data(context)

        # Prepare extra dictionary for context
        # Ensure 'context' key always exists if the formatter string expects '%(context)s'
        extra_data = {"context": filtered_context if filtered_context is not None else {}}

        # Log the message using the standard logging interface
        logger.log(
            level,
            message,
            *args,
            exc_info=exc_info,
            extra=extra_data,
            stacklevel=2)  # Pass *args
        # stacklevel=2 ensures filename/lineno are from the caller of this
        # method

    # --- Convenience Helper Methods --- #

    def debug(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None) -> None:
        """Log a message with DEBUG level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.DEBUG,
            message,
            *args,
            source_module=source_module,
            context=context)  # Pass *args

    def info(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None) -> None:
        """Log a message with INFO level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.INFO,
            message,
            *args,
            source_module=source_module,
            context=context)  # Pass *args

    def warning(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None) -> None:
        """Log a message with WARNING level.

        Args:
        ----
            message: The message to log
            *args: Arguments for the format string in 'message'
            source_module: Optional module name generating the log
            context: Optional context information
        """
        self.log(
            logging.WARNING,
            message,
            *args,
            source_module=source_module,
            context=context)  # Pass *args

    def error(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
        exc_info: ExcInfoType = None) -> None:
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
            exc_info=exc_info)  # Pass *args

    def exception(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None) -> None:
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
            exc_info=True)  # Pass *args

    def critical(
        self,
        message: str,
        *args: object,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
        exc_info: ExcInfoType = None) -> None:
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
            exc_info=exc_info)  # Pass *args

    # ========================================
    # Enterprise Handler Management
    # ========================================

    def initialize_enterprise_handlers(self) -> None:
        """Initialize enterprise handlers from configuration."""

        handlers_config: list[dict[str, Any]] = self._config_manager.get("logging.enterprise_handlers", [])

        for handler_config_data in handlers_config:
            try:
                # Create handler configuration
                config = HandlerConfig(
                    handler_type=HandlerType(handler_config_data["type"]),
                    name=handler_config_data["name"],
                    level=LogLevel(handler_config_data.get("level", "INFO")),
                    format_string=handler_config_data.get("format"),
                    parameters=handler_config_data.get("parameters", {}),
                    enabled=handler_config_data.get("enabled", True),
                    filters=handler_config_data.get("filters", []))

                # Create and register handler
                if config.enabled:
                    handler: BaseLogHandler
                    if config.handler_type == HandlerType.DATABASE:
                        # Special handling for database handler
                        if self._db_session_maker:
                            handler = EnterpriseAsyncPostgresHandler(
                                config, self._db_session_maker, asyncio.get_event_loop()
                            )
                        else:
                            self.warning(
                                "Database handler requested but session_maker not available",
                                source_module="LoggerService")
                            continue
                    else:
                        handler = LogHandlerFactory.create_handler(config)

                    self._enterprise_handlers[config.name] = handler
                    self._handler_configs[config.name] = config
                    self._root_logger.addHandler(handler)

                    self.info(
                        f"Initialized enterprise handler: {config.name} ({config.handler_type.value})"
                    )

            except Exception as e:
                self.error(f"Failed to initialize enterprise handler: {e}")

    def get_enterprise_handler(self, name: str) -> Optional[BaseLogHandler]:
        """Get enterprise handler by name with proper type annotation."""
        return self._enterprise_handlers.get(name)

    def add_enterprise_handler(self, config: HandlerConfig) -> bool:
        """Add new enterprise handler at runtime."""

        try:
            if config.name in self._enterprise_handlers:
                self.warning(f"Enterprise handler {config.name} already exists")
                return False

            handler: BaseLogHandler
            if config.handler_type == HandlerType.DATABASE:
                # Special handling for database handler
                if self._db_session_maker:
                    handler = EnterpriseAsyncPostgresHandler(
                        config, self._db_session_maker, asyncio.get_event_loop()
                    )
                else:
                    self.error("Database handler requested but session_maker not available")
                    return False
            else:
                handler = LogHandlerFactory.create_handler(config)

            self._enterprise_handlers[config.name] = handler
            self._handler_configs[config.name] = config
            self._root_logger.addHandler(handler)

            self.info(f"Added enterprise handler: {config.name}")
            return True

        except Exception as e:
            self.error(f"Failed to add enterprise handler {config.name}: {e}")
            return False

    def remove_enterprise_handler(self, name: str) -> bool:
        """Remove enterprise handler by name."""

        if name in self._enterprise_handlers:
            handler = self._enterprise_handlers[name]
            self._root_logger.removeHandler(handler)
            handler.close()

            del self._enterprise_handlers[name]
            del self._handler_configs[name]

            self.info(f"Removed enterprise handler: {name}")
            return True

        return False

    def get_enterprise_handler_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all enterprise handlers."""

        stats = {}

        for name, handler in self._enterprise_handlers.items():
            stats[name] = {
                "type": self._handler_configs[name].handler_type.value,
                "emit_count": handler.emit_count,
                "error_count": handler.error_count,
                "last_emit_time": handler.last_emit_time,
                "enabled": self._handler_configs[name].enabled,
            }

        return stats

    def flush_all_enterprise_handlers(self) -> None:
        """Flush all enterprise handlers."""

        for handler in self._enterprise_handlers.values():
            try:
                handler.flush()
            except Exception as e:
                self.error(f"Error flushing enterprise handler: {e}")

    def close_all_enterprise_handlers(self) -> None:
        """Close all enterprise handlers."""

        for handler in self._enterprise_handlers.values():
            try:
                handler.close()
            except Exception as e:
                self.error(f"Error closing enterprise handler: {e}")

    # --- Placeholder for Time-Series[Any] Logging --- #
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
                source_module="LoggerService")
            return False

        assert url is not None, "URL should be a string after 'all' check"

        try:
            # Import influxdb_client specifics here, only when actually trying to initialize
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.exceptions import InfluxDBError
            from influxdb_client.client.write_api import SYNCHRONOUS
        except ImportError:
            self.error(
                "InfluxDB client library not installed ('pip install influxdb-client'). "
                "Cannot log timeseries.",
                source_module="LoggerService")
            self._influx_client = None
            self._influx_write_api = None
            return False
        except Exception as e:
            self.error(
                "Failed to initialize InfluxDB client: %s",
                e,
                source_module="LoggerService",
                exc_info=True)
            self._influx_client = None
            self._influx_write_api = None  # Ensure write_api is also cleared
            return False
        else:
            # Since we've already checked that all([url, token, org]) is True,
            # and asserted non-None, these isinstance checks are less critical
            # but don't hurt if config_manager.get could return non-str for some reason.
            # However, the primary issue for mypy is None vs str.
            if not isinstance(token, str):  # This check might be deemed redundant by mypy now
                raise TypeError("Token must be a string")
            if not isinstance(org, str):  # This check might be deemed redundant by mypy now
                raise TypeError("Organization must be a string")

            # Now use the imported InfluxDBClient
            self._influx_client = InfluxDBClient(url=url, token=token, org=org)
            self._influx_write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)
            self.info(
                "InfluxDB client initialized for timeseries logging.",
                source_module="LoggerService")
            return True

    def _prepare_influxdb_point(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: datetime) -> Optional["InfluxDBPoint"]:  # Returns InfluxDB Point or None
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
            # Import Point and WritePrecision here
            from influxdb_client import Point, WritePrecision
        except ImportError:
            # This case should ideally be caught by _initialize_influxdb_client
            self.error(
                "InfluxDB client library not found during point preparation.",
                source_module="LoggerService")
            return None
        except Exception as e:
            self.error(
                "Error preparing InfluxDB point: %s",
                e,
                source_module="LoggerService",
                exc_info=True)
            return None
        else:
            point = Point(measurement).time(timestamp, WritePrecision.MS)

            for key, value in tags.items():
                point = point.tag(key, str(value))

            valid_fields: dict[str, Any] = {}
            for key, value in fields.items():
                # Explicit type guard to help mypy understand value can be Any type
                field_value: Any = value
                
                # Handle specific types - check string last since it's the broadest type
                if isinstance(field_value, bool):  # bool must come before int since bool is a subclass of int
                    valid_fields[key] = field_value
                elif isinstance(field_value, (int, float)):
                    valid_fields[key] = field_value
                elif isinstance(field_value, Decimal):
                    valid_fields[key] = float(field_value)
                elif isinstance(field_value, str):
                    valid_fields[key] = field_value
                else:
                    self.warning(
                        "Unsupported type for InfluxDB field '%s': %s. Converting to string.",
                        key,
                        type(field_value),
                        source_module="LoggerService")
                    valid_fields[key] = str(field_value)

            if not valid_fields:
                self.warning(
                    "No valid fields for timeseries point in '%s'. Skipping.",
                    measurement,
                    source_module="LoggerService")
                return None

            # Create the point with valid fields
            for key, value in valid_fields.items():
                point = point.field(key, value)

            return cast("InfluxDBPoint", point)

    async def log_timeseries(
        self,
        measurement: str,
        tags: dict[str, str],
        fields: dict[str, Any],
        timestamp: datetime | None = None) -> None:
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
            source_module="LoggerService")

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
                    source_module="LoggerService")
                return
            self._influx_write_api.write(bucket=bucket, record=point)
        except Exception as e:
            self.error(
                "Failed to write time-series data to InfluxDB: %s",
                e,
                source_module="LoggerService",
                exc_info=True)

    async def start(self) -> None:
        """Initialize the logger service and set up required connections.

        Initializes database connection pool if configured and subscribes to log events.
        """
        self.info("LoggerService start sequence initiated.", source_module="LoggerService")

        # Initialize SQLAlchemy engine and session factory if DB logging is enabled
        if self._db_enabled:
            await self._initialize_sqlalchemy()

        # Initialize enterprise handlers (includes database handler if configured)
        self.initialize_enterprise_handlers()

        # Legacy database handler setup (if not using enterprise handlers)
        if self._db_enabled and self._async_handler is None and self._db_session_maker:
            self.info(
                "Setting up legacy database handler. Consider migrating to enterprise handlers.",
                source_module="LoggerService")
            # Create legacy database handler configuration
            db_config = HandlerConfig(
                handler_type=HandlerType.DATABASE,
                name="legacy_database",
                level=LogLevel(
                    str(self._config_manager.get("logging.database.level", "INFO")).upper()
                ),
                enabled=True)
            try:
                self._async_handler = EnterpriseAsyncPostgresHandler(
                    db_config, self._db_session_maker, asyncio.get_event_loop()
                )
                self._async_handler.setLevel(getattr(logging, db_config.level.value))
                self._root_logger.addHandler(self._async_handler)
                logging.info("Legacy SQLAlchemy Database logging handler added in start().")
            except Exception:
                logging.exception("Failed to add legacy EnterpriseAsyncPostgresHandler in start()")

        # Subscribe to LOG events from the event bus
        try:
            self._pubsub.subscribe(EventType.LOG_ENTRY, self._handle_log_event)
            self.info(
                "Subscribed to LOG_ENTRY events from event bus.",
                source_module="LoggerService")
        except Exception as e:
            self.error(
                "Failed to subscribe to LOG_ENTRY events: %s",
                e,
                source_module="LoggerService",
                exc_info=True)

        # Start enterprise database handlers' internal processing tasks
        for handler in self._enterprise_handlers.values():
            if isinstance(handler, EnterpriseAsyncPostgresHandler):
                handler.start_processing()
                self.info(
                    f"Enterprise database handler {handler.config.name} processing started.",
                    source_module="LoggerService")

        # Start the legacy DB handler's internal processing task
        if self._async_handler:
            self._async_handler.start_processing()
            self.info(
                "Legacy AsyncPostgresHandler processing started.", source_module="LoggerService"
            )
        elif self._db_enabled and not any(
            isinstance(h, EnterpriseAsyncPostgresHandler)
            for h in self._enterprise_handlers.values()
        ):
            self.error(
                "DB logging enabled, but no database handlers initialized in start().",
                source_module="LoggerService")

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
                exc_info=True)

        # Close enterprise handlers first
        self.info("Closing enterprise handlers...", source_module="LoggerService")
        for name, handler in self._enterprise_handlers.items():
            try:
                if isinstance(handler, EnterpriseAsyncPostgresHandler):
                    self.info(
                        f"Closing enterprise database handler {name}...",
                        source_module="LoggerService")
                    handler.close()
                    if hasattr(handler, "wait_closed"):
                        await asyncio.wait_for(handler.wait_closed(), timeout=10.0)
                        self.info(
                            f"Enterprise database handler {name} closed gracefully.",
                            source_module="LoggerService")
                else:
                    handler.close()
                    self.info(f"Enterprise handler {name} closed.", source_module="LoggerService")
            except TimeoutError:
                self.warning(
                    f"Timeout waiting for enterprise handler {name} to close.",
                    source_module="LoggerService")
            except Exception as e:
                self.error(
                    f"Error closing enterprise handler {name}: {e}",
                    source_module="LoggerService",
                    exc_info=True)

        # Close the legacy SQLAlchemy handler
        if self._async_handler:
            self.info(
                "Closing legacy SQLAlchemy database log handler...", source_module="LoggerService"
            )
            self._async_handler.close()
            try:
                if hasattr(self._async_handler, "wait_closed"):
                    await asyncio.wait_for(
                        self._async_handler.wait_closed(), timeout=10.0
                    )  # Increased timeout
                    self.info(
                        "Legacy SQLAlchemy database log handler closed gracefully.",
                        source_module="LoggerService")
            except TimeoutError:
                self.warning(
                    "Timeout waiting for legacy SQLAlchemy database log handler queue to empty.",
                    source_module="LoggerService")
            except Exception as e:
                self.error(
                    "Error waiting for legacy SQLAlchemy database log handler closure: %s",
                    e,
                    source_module="LoggerService",
                    exc_info=True)

        # Signal the log processing thread to stop (This thread is for the python logging queue, keep it)
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self.warning(
                "Main log processing thread did not exit cleanly.",
                source_module="LoggerService")

    async def _initialize_sqlalchemy(self) -> None:
        """Initialize the SQLAlchemy engine and session factory for database logging."""
        if not self._db_enabled:
            self.info(
                "Database logging is disabled. SQLAlchemy setup skipped.",
                source_module="LoggerService")
            return

        db_url: str | None = self._config_manager.get("logging.database.connection_string")
        if not db_url:
            self.error(
                "Database logging enabled but connection_string is missing. SQLAlchemy setup failed.",
                source_module="LoggerService")
            self._db_enabled = False  # Disable DB logging if URL is missing
            return

        try:
            self.info(
                "Initializing SQLAlchemy async engine for logging...",
                source_module="LoggerService",
                context={
                    "database_url": str(db_url)[: str(db_url).find("@")] + "@********"
                    if "@" in str(db_url)
                    else str(db_url)
                },  # Mask credentials
            )
            pool_size = self._config_manager.get_int("logging.database.pool_size", 5)
            max_overflow = self._config_manager.get_int("logging.database.max_overflow", 10)
            echo_sql = self._config_manager.get("logging.database.echo_sql", False)

            self._sqlalchemy_engine = create_async_engine(
                str(db_url),  # Ensure db_url is a string
                pool_size=pool_size,
                max_overflow=max_overflow,
                echo=echo_sql)
            # Use async_sessionmaker for AsyncEngine and AsyncSession
            self._sqlalchemy_session_factory = async_sessionmaker(
                bind=self._sqlalchemy_engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False)
            self.info(
                "SQLAlchemy async engine and session factory initialized successfully.",
                source_module="LoggerService")
        except SQLAlchemyError as e:  # Catch SQLAlchemy specific errors
            self.critical(
                "Failed to initialize SQLAlchemy engine: %s. Disabling DB logging.",
                e,
                source_module="LoggerService",
                exc_info=True)
            self._sqlalchemy_engine = None
            self._sqlalchemy_session_factory = None
            self._db_enabled = False  # Disable DB logging on error
        except Exception as e:  # Catch any other unexpected errors
            self.critical(
                "An unexpected error occurred during SQLAlchemy engine initialization: %s. Disabling DB logging.",
                e,
                source_module="LoggerService",
                exc_info=True)
            self._sqlalchemy_engine = None
            self._sqlalchemy_session_factory = None
            self._db_enabled = False  # Disable DB logging on error

    async def _handle_log_event(self, event: LogEvent) -> None:
        """Handle a LogEvent received from the event bus.

        Process the event and send it to the appropriate logging handlers.

        Args:
        ----
            event: The LogEvent object containing log information
        """
        # Event is already typed as LogEvent, no need for runtime check

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