# Logger Service Module
import logging
import logging.handlers
import sys
from typing import (
    Optional,
    Dict,
    Any,
    Callable,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
    Protocol,
    Generic,
    AsyncContextManager,
)
from datetime import datetime
import os
import json
import asyncio
import asyncpg  # type: ignore[import-untyped]
import queue
import threading
import random
import re
from decimal import Decimal

from .core.pubsub import PubSubManager
from .core.events import EventType, LogEvent

# Import JSON Formatter
from pythonjsonlogger import jsonlogger

# Define a Protocol for ConfigManager to properly type hint its interface


class ConfigManagerProtocol(Protocol):
    def get(self, key: str, default: Optional[Any] = None) -> Any: ...
    def get_int(self, key: str, default: int = 0) -> int: ...


# Define a Protocol for database connection


class DBConnection(Protocol):
    async def execute(self, query: str, *args: Any) -> Any: ...


# Define a Protocol for the Pool interface we need


class PoolProtocol(Protocol):
    def acquire(self) -> AsyncContextManager[DBConnection]: ...
    async def release(self, conn: DBConnection) -> None: ...
    async def close(self) -> None: ...


# Define a proper type alias for the Pool type
PoolType = TypeVar("PoolType", bound=PoolProtocol)

# Placeholder Type Hints (Refine later if needed)
if TYPE_CHECKING:
    # Use actual type if stubs were available
    AsyncPostgresHandlerType = logging.Handler  # Placeholder for typing
    from influxdb_client import InfluxDBClient
    from influxdb_client.client.write_api import WriteApi
else:
    # Define placeholders if not type checking to avoid runtime errors
    AsyncPostgresHandlerType = logging.Handler


class ContextFormatter(logging.Formatter):
    """Custom formatter to include context dictionary in log messages."""

    def format(self, record: logging.LogRecord) -> str:
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
        if "[%(context)s]" in self._style._fmt:
            if context_str:
                s = s.replace("[%(context)s]", f"[{context_str}]")
            else:
                s = s.replace(
                    " - [%(context)s]", ""
                )  # Remove placeholder and separator if no context
                # Remove just placeholder if at start/end
                s = s.replace("[%(context)s]", "")

        return s


# --- Custom Async Database Handler ---
class AsyncPostgresHandler(logging.Handler, Generic[PoolType]):
    """A logging handler that asynchronously writes logs to a PostgreSQL database."""

    def __init__(self, pool: PoolType, table_name: str, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._pool = pool
        self._table_name = table_name
        self._loop = loop
        self._queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._closed = False
        self._task = asyncio.create_task(self._process_queue())

    def emit(self, record: logging.LogRecord) -> None:
        """Format record and place it in the queue for async processing."""
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
        except Exception:
            self.handleError(record)

    def _format_record(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Formats the log record into a dictionary suitable for DB insertion."""
        context_json = None
        if hasattr(record, "context") and record.context:
            try:
                context_json = json.dumps(record.context)
            except TypeError:
                # Fallback for non-serializable
                context_json = json.dumps(str(record.context))

        exc_text = None
        if record.exc_info:
            if not record.exc_text:
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

    async def _attempt_db_insert(self, record_data: Dict[str, Any]) -> bool:
        """Attempts to insert a single log record into the database."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self._table_name} (
                        timestamp, logger_name, level_name, level_no, message,
                        pathname, filename, lineno, func_name, context_json,
                        exception_text
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
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
            asyncpg.exceptions.InterfaceError,  # Can indicate connection issues
            OSError,  # Can occur on network issues
        ) as conn_err:
            # This will be handled by the retry logic in _process_queue_with_retry
            raise conn_err  # Re-raise to be caught by the retry mechanism
        except Exception as e:
            # Non-retryable error during DB operation
            print(
                f"AsyncPostgresHandler: Non-retryable error inserting log record: {e}",
                file=sys.stderr,
            )
            return False  # Indicate non-retryable failure for this record

    async def _process_queue_with_retry(self, record_data: Dict[str, Any]) -> None:
        """Processes a single record with retry logic."""
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
                    print(
                        f"AsyncPostgresHandler: DB connection error failed after {max_retries} "
                        f"attempts: {conn_err}",
                        file=sys.stderr,
                    )
                else:
                    # Exponential backoff with jitter
                    wait_time = min(base_backoff * (2**attempt), 30.0)  # Cap at 30s
                    wait_time += random.uniform(0, wait_time * 0.1)
                    print(
                        f"AsyncPostgresHandler: DB connection error (Attempt {attempt}/"
                        f"{max_retries}). Retrying in {wait_time:.2f}s. Error: {conn_err}",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait_time)
            except Exception:  # Catches non-retryable errors from _attempt_db_insert
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
                print("AsyncPostgresHandler queue processing cancelled.", file=sys.stderr)
                if record_data is not None:
                    try:
                        self._queue.task_done()
                    except ValueError:
                        pass
                break
            except Exception as e:
                print(
                    f"AsyncPostgresHandler: Error in outer processing loop: {e}", file=sys.stderr
                )
                if record_data is not None:
                    try:
                        self._queue.task_done()
                    except ValueError:
                        pass
                await asyncio.sleep(1)

    def close(self) -> None:
        """Closes the handler, signaling the queue processor to finish."""
        if not self._closed:
            self._closed = True
            self._loop.call_soon_threadsafe(lambda: self._queue.put_nowait(None))
        super().close()

    async def wait_closed(self) -> None:
        """Wait for the queue processing task to finish."""
        if self._task is not None:
            await self._task


# -------------------------------------


class LoggerService(Generic[PoolType]):
    """Handles logging configuration and provides interfaces for logging messages
    and time-series data to configured destinations (file, console, database).
    """

    _influx_client: Optional["InfluxDBClient"] = None
    _influx_write_api: Optional["WriteApi"] = None

    def __init__(
        self, config_manager: ConfigManagerProtocol, pubsub_manager: "PubSubManager"
    ) -> None:
        """Initialize LoggerService."""
        self._config_manager = config_manager
        self._pubsub = pubsub_manager
        self._log_level = self._config_manager.get("logging.level", "INFO").upper()
        self._log_format = self._config_manager.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context_json)s",
        )
        self._log_date_format = self._config_manager.get(
            "logging.date_format", "%Y-%m-%d %H:%M:%S"
        )

        # Async DB Handler setup
        self._db_config = self._config_manager.get("logging.database", {})
        self._db_enabled = bool(self._db_config.get("enabled", False))
        self._async_handler: Optional[AsyncPostgresHandler[PoolType]] = None
        self._db_pool: Optional[PoolType] = None

        # Queue and thread for handling synchronous logging calls from async
        # context
        self._queue: queue.Queue[Tuple[Callable[..., None], tuple, dict]] = queue.Queue()
        self._thread = threading.Thread(target=self._process_log_queue, daemon=True)
        self._stop_event = threading.Event()
        self._loggers: Dict[str, logging.Logger] = {}
        self._root_logger: logging.Logger = logging.getLogger("gal_friday")

        self._setup_logging()
        self._thread.start()
        self.info("LoggerService initialized.", source_module="LoggerService")

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
            except Exception as e:
                # Log error using root logger to avoid recursion if self.error
                # uses the queue
                logging.error(f"Error processing log queue item: {e}", exc_info=True)

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
                self._config_manager.get("logging.file.filename", default="gal_friday.log")
            )
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, log_filename)

            # Configure JSON Formatter
            json_formatter = jsonlogger.JsonFormatter(
                self._log_format,
                datefmt=self._log_date_format,
                rename_fields={"levelname": "level"},
            )

            max_bytes = int(
                self._config_manager.get("logging.file.max_bytes", default=10 * 1024 * 1024)
            )
            backup_count = int(self._config_manager.get("logging.file.backup_count", default=5))
            file_handler = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(json_formatter)
            file_handler.setLevel(self._log_level)
            self._root_logger.addHandler(file_handler)

        # --- Database Handler (PostgreSQL) ---
        use_db = self._db_enabled
        if use_db:
            if self._db_pool:
                db_table = str(
                    self._config_manager.get("logging.database.table_name", default="logs")
                )
                db_level_str = str(
                    self._config_manager.get("logging.database.level", default="INFO")
                ).upper()
                db_level = getattr(logging, db_level_str, logging.INFO)

                try:
                    loop = asyncio.get_running_loop()
                    self._async_handler = AsyncPostgresHandler(self._db_pool, db_table, loop)
                    self._async_handler.setLevel(db_level)
                    self._root_logger.addHandler(self._async_handler)
                    logging.info(
                        f"Database logging handler added for table '{db_table}'. "
                        f"Level: {db_level_str}",
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to create or add AsyncPostgresHandler: {e}",
                        exc_info=True,
                    )
            else:
                logging.warning(
                    "Database logging enabled, but pool not yet initialized. "
                    "Handler will be added after pool connection.",
                )

    def _filter_sensitive_data(
        self, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Recursively filter sensitive data from log context."""
        if (
            not context
        ):  # Simplified check for None or empty dict, though type hint is Optional[Dict]
            return None

        filtered: Dict[str, Any] = {}
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
            r"^[A-Za-z0-9/+]{20,}$"
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

    def log(
        self,
        level: int,
        message: str,
        source_module: Optional[str] = None,
        context: Optional[Dict] = None,
        exc_info: Optional[Any] = None,
    ) -> None:
        """Logs a message to the configured handlers.

        Args:
            level: The logging level (e.g., logging.INFO, logging.WARNING).
            message: The primary log message string.
            source_module: Optional name of the module generating the log.
            context: Optional dictionary of key-value pairs for extra context.
            exc_info: Optional exception info (e.g., True or exception tuple).
        """
        # Get the specific logger for the source module, or root if None
        logger_name = f"gal_friday.{source_module}" if source_module else "gal_friday"
        logger = logging.getLogger(logger_name)

        # Filter context BEFORE passing it as 'extra'
        filtered_context = self._filter_sensitive_data(context)

        # Prepare extra dictionary for context
        extra_data = {"context": filtered_context} if filtered_context else {}

        # Log the message using the standard logging interface
        logger.log(level, message, exc_info=exc_info, extra=extra_data, stacklevel=2)
        # stacklevel=2 ensures filename/lineno are from the caller of this
        # method

    # --- Convenience Helper Methods --- #

    def debug(
        self,
        message: str,
        source_module: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Logs a message with level DEBUG."""
        self.log(logging.DEBUG, message, source_module, context)

    def info(
        self,
        message: str,
        source_module: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Logs a message with level INFO."""
        self.log(logging.INFO, message, source_module, context)

    def warning(
        self,
        message: str,
        source_module: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> None:
        """Logs a message with level WARNING."""
        self.log(logging.WARNING, message, source_module, context)

    def error(
        self,
        message: str,
        source_module: Optional[str] = None,
        context: Optional[Dict] = None,
        exc_info: Optional[Any] = None,
    ) -> None:
        """Logs a message with level ERROR."""
        self.log(logging.ERROR, message, source_module, context, exc_info=exc_info)

    def critical(
        self,
        message: str,
        source_module: Optional[str] = None,
        context: Optional[Dict] = None,
        exc_info: Optional[Any] = None,
    ) -> None:
        """Logs a message with level CRITICAL."""
        self.log(logging.CRITICAL, message, source_module, context, exc_info=exc_info)

    # --- Placeholder for Time-Series Logging --- #
    async def _initialize_influxdb_client(self) -> bool:
        """Initializes the InfluxDB client if not already initialized. Returns True on success."""
        if self._influx_client is not None:  # Check instance variable
            return True  # Already initialized

        url = self._config_manager.get("logging.influxdb.url")
        token = self._config_manager.get("logging.influxdb.token")
        org = self._config_manager.get("logging.influxdb.org")
        if not all([url, token, org]):
            self.warning(
                "InfluxDB config incomplete (url/token/org). Cannot log timeseries.",
                source_module="LoggerService",
            )
            return False
        try:
            import influxdb_client  # Import here to keep dependency optional
            from influxdb_client.client.write_api import SYNCHRONOUS

            self._influx_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
            self._influx_write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)
            self.info(
                "InfluxDB client initialized for timeseries logging.",
                source_module="LoggerService",
            )
            return True
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
                f"Failed to initialize InfluxDB client: {e}",
                source_module="LoggerService",
                exc_info=True,
            )
            self._influx_client = None
            self._influx_write_api = None  # Ensure write_api is also cleared
            return False

    def _prepare_influxdb_point(
        self, measurement: str, tags: Dict[str, str], fields: Dict[str, Any], timestamp: datetime
    ) -> Optional[Any]:  # Actually returns influxdb_client.Point but avoiding import at top level
        """Prepares a data point for InfluxDB."""
        try:
            from influxdb_client import Point, WritePrecision  # Import here

            point = Point(measurement).time(timestamp, WritePrecision.MS)

            for key, value in tags.items():
                point = point.tag(key, str(value))

            valid_fields = {}
            for key, value in fields.items():
                if isinstance(value, (float, int, bool, str)):
                    valid_fields[key] = value
                elif isinstance(value, Decimal):
                    valid_fields[key] = float(value)
                else:
                    self.warning(
                        f"Unsupported type for InfluxDB field '{key}': {type(value)}. "
                        "Converting to string.",
                        source_module="LoggerService",
                    )
                    valid_fields[key] = str(value)

            if not valid_fields:
                self.warning(
                    f"No valid fields for timeseries point in '{measurement}'. Skipping.",
                    source_module="LoggerService",
                )
                return None

            for key, value in valid_fields.items():
                point = point.field(key, value)
            return point
        except ImportError:
            # This case should ideally be caught by _initialize_influxdb_client
            self.error(
                "InfluxDB client library not found during point preparation.",
                source_module="LoggerService",
            )
            return None
        except Exception as e:
            self.error(
                f"Error preparing InfluxDB point: {e}",
                source_module="LoggerService",
                exc_info=True,
            )
            return None

    async def log_timeseries(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Logs time-series data to InfluxDB if configured."""
        log_time = timestamp if timestamp else datetime.utcnow()

        self.debug(
            f"[TimeSeries] M={measurement}, T={tags}, F={fields}, TS={log_time.isoformat()}",
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
            bucket = self._config_manager.get("logging.influxdb.bucket")
            if not bucket:
                self.warning(
                    "InfluxDB bucket not configured. Cannot log timeseries.",
                    source_module="LoggerService",
                )
                return
            self._influx_write_api.write(bucket=bucket, record=point)
        except Exception as e:
            self.error(
                f"Failed to write time-series data to InfluxDB: {e}",
                source_module="LoggerService",
                exc_info=True,
            )

    async def start(self) -> None:
        """Initializes database connection pool if needed."""
        self.info("LoggerService start sequence initiated.", source_module="LoggerService")

        # Subscribe to LOG events from the event bus
        try:
            self._pubsub.subscribe(EventType.LOG_ENTRY, self._handle_log_event)  # Removed await
            self.info(
                "Subscribed to LOG_ENTRY events from event bus.", source_module="LoggerService"
            )
        except Exception as e:
            self.error(
                f"Failed to subscribe to LOG_ENTRY events: {e}",
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
                        "after pool initialization..."
                    )
                self._setup_logging()  # Re-run config to add the handler
            elif not self._db_pool:
                logging.error(
                    "Database logging enabled but pool initialization failed. "
                    "DB logging inactive.",
                )

    async def stop(self) -> None:
        """Closes the database handler and connection pool."""
        self.info("LoggerService stop sequence initiated.", source_module="LoggerService")

        # Unsubscribe from LOG events
        try:
            self._pubsub.unsubscribe(EventType.LOG_ENTRY, self._handle_log_event)  # Removed await
            self.info("Unsubscribed from LOG_ENTRY events.", source_module="LoggerService")
        except Exception as e:
            self.error(
                f"Error unsubscribing from LOG_ENTRY events: {e}",
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
                        self._async_handler.wait_closed(), timeout=5.0
                    )  # Wait for queue to empty
                    self.info(
                        "Database log handler closed gracefully.", source_module="LoggerService"
                    )
            except asyncio.TimeoutError:
                self.warning(
                    "Timeout waiting for database log handler queue to empty.",
                    source_module="LoggerService",
                )
            except Exception as e:
                self.error(
                    f"Error waiting for database log handler closure: {e}",
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
                "Log processing thread did not exit cleanly.", source_module="LoggerService"
            )

    async def _initialize_db_pool(self) -> None:
        """Initializes the asyncpg connection pool if DB logging is enabled."""
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
                f"Initializing database connection pool "
                f"(min: {min_size}, max: {max_size}) for logging...",
                source_module="LoggerService",
            )
            self._db_pool = await asyncpg.create_pool(
                dsn=db_dsn, min_size=min_size, max_size=max_size
            )
            self.info(
                "Database connection pool initialized successfully.", source_module="LoggerService"
            )
        except (
            asyncpg.exceptions.InvalidConnectionParametersError,
            asyncpg.exceptions.CannotConnectNowError,
            OSError,
            Exception,
        ) as e:
            self.critical(
                f"Failed to initialize database connection pool: {e}",
                source_module="LoggerService",
                exc_info=True,
            )
            self._db_pool = None

    async def _close_db_pool(self) -> None:
        """Closes the asyncpg connection pool."""
        if self._db_pool:
            self.info("Closing database connection pool...", source_module="LoggerService")
            try:
                await self._db_pool.close()
                self.info("Database connection pool closed.", source_module="LoggerService")
            except Exception as e:
                self.error(
                    f"Error closing database connection pool: {e}",
                    source_module="LoggerService",
                    exc_info=True,
                )
            finally:
                self._db_pool = None

    async def _handle_log_event(self, event: LogEvent) -> None:
        """Handles LogEvent received from the event bus."""
        if not isinstance(event, LogEvent):
            self.warning(
                f"Received non-LogEvent on LOG_ENTRY topic: {type(event)}",
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
            source_module=event.source_module,  # Use source from event
            context=event.context if hasattr(event, "context") else None,
            exc_info=None,  # Or derive from context/level if needed
        )


# Example Usage (Typically done within other modules):
# from gal_friday.logger_service import LoggerService
# from gal_friday.config_manager import ConfigManager
# import logging
#
# if __name__ == "__main__":
#     # Minimal example setup
#     # In real app, ConfigManager would load from file
#     mock_config_data = {
#         'logging': {
#             'level': 'DEBUG',
#             'file': {'enabled': True},
#             'console': {'enabled': True}
#         }
#     }
#     config = ConfigManager(config_path=None) # Need a way to inject mock data or load dummy file
#     config._config = mock_config_data # Hack for example
#
#     logger_service = LoggerService(config)
#
#     # Direct logging call from another module (e.g., DataIngestor)
#     logger_service.log(logging.INFO, "Data ingestion started.", source_module="DataIngestor")
#     logger_service.log(logging.WARNING, "Low API rate limit remaining.",
#                       source_module="ExecutionHandler", context={"remaining": 50})
#     try:
#         x = 1 / 0
#     except ZeroDivisionError:
#         logger_service.log(
#             logging.ERROR,
#             "Calculation failed.",
#             source_module="FeatureEngine",
#             exc_info=True
#         )
#
#     # Example of getting a standard logger after configuration
#     std_logger = logging.getLogger("gal_friday.MyOtherModule")
#     std_logger.debug(
#         "This uses the handlers configured by LoggerService."
#     )
