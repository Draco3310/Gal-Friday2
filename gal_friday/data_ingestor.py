"""Retrieve and process market data from the Kraken WebSocket API.

This module implements a data ingestion service that connects to Kraken WebSocket API v2,
subscribes to L2 order book and OHLCV data streams, handles parsing, validation and state
management, and publishes standardized market data events to the application's event bus.
"""

# src/gal_friday/data_ingestor.py

from collections import defaultdict
import contextlib  # Added for SIM105 fix
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import logging
from typing import TYPE_CHECKING, Any, ClassVar
import uuid

import asyncio
from sortedcontainers import SortedDict
import websockets

# Import necessary event classes from core module
from .core.events import (
    MarketDataL2Event,
    MarketDataTradeEvent,
    PotentialHaltTriggerEvent,
    SystemStateEvent,
)
from .logger_service import LoggerService

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .core.pubsub import PubSubManager

# Initialize logger for the module
logger = logging.getLogger(__name__)

# --- Event Payloads (for constructing core events) ---


@dataclass
class MarketDataL2Payload:
    """Payload for L2 market data."""

    trading_pair: str
    exchange: str
    # Timestamp from the book update message
    timestamp_exchange: str | None = None
    # [(price_str, volume_str), ...] sorted desc
    bids: list[tuple[str, str]] = field(default_factory=list[Any])
    # [(price_str, volume_str), ...] sorted asc
    asks: list[tuple[str, str]] = field(default_factory=list[Any])
    is_snapshot: bool = False
    # Add checksum to event for potential downstream validation
    checksum: int | None = None


@dataclass
class MarketDataOHLCVPayload:
    """Payload for OHLCV market data."""

    trading_pair: str
    exchange: str
    interval: str  # e.g., "1m", "5m"
    timestamp_bar_start: str  # ISO 8601
    open: str
    high: str
    low: str
    close: str
    volume: str


@dataclass
class SystemStatusPayload:
    """Payload for system status updates."""

    system_status: str  # e.g., "online", "cancel_only"
    connection_id: int | None = None


# --- DataIngestor Class ---


class DataIngestor:
    """Connect to Kraken WebSocket API v2, manage subscriptions for L2 book and OHLCV data.

    This class handles parsing messages, maintaining L2 order book state with
    checksum validation, and publishes standardized market data events.
    """

    # Map Kraken interval integers to readable strings
    INTERVAL_MAP: ClassVar[dict[int, str]] = {
        1: "1m",
        5: "5m",
        15: "15m",
        30: "30m",
        60: "1h",
        240: "4h",
        1440: "1d",
        10080: "1w",
        21600: "15d",  # Kraken uses 15d for 21600
    }
    # Map readable strings back to integers for context lookups
    INTERVAL_INT_MAP: ClassVar[dict[str, int]] = {v: k for k, v in INTERVAL_MAP.items()}

    def __init__(
        self,
        config: "ConfigManager",
        pubsub_manager: "PubSubManager",
        logger_service: LoggerService) -> None:
        """Initialize the DataIngestor.

        Args:
        ----
            config: The application's configuration manager
            pubsub_manager: The application's PubSubManager instance
            logger_service: The shared logger service instance
        """
        self._config = config
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self._source_module = self.__class__.__name__

        # Load configuration
        self._load_configuration()

        # Initialize state
        self._connection: Any | None = None
        self._is_running: bool = False
        self._is_stopping: bool = False
        self._last_message_received_time: datetime | None = None
        self._last_heartbeat_received_time: datetime | None = None  # For heartbeat tracking
        self._connection_established_time: datetime | None = (
            None  # For initial connection tracking
        )
        self._liveness_task: asyncio.Task[Any] | None = None
        self._subscriptions: dict[str, dict[str, Any]] = {}
        self._connection_id: int | None = None
        self._system_status: str | None = None

        # Initialize book state
        self._l2_books: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"bids": SortedDict(), "asks": SortedDict(), "checksum": None})

        # Validate configuration
        self._validate_initial_config()

    def _load_configuration(self) -> None:
        """Load configuration values from ConfigManager."""
        data_config = self._config.get("data_ingestor", {})
        kraken_config = self._config.get("kraken", {})

        # WebSocket configuration
        self._websocket_url = data_config.get("kraken_ws_url",
            kraken_config.get("websocket", {}).get("url", "wss://ws.kraken.com/v2"))
        self._connection_timeout = data_config.get("connection_timeout_s", 15)
        self._max_heartbeat_interval = data_config.get("max_heartbeat_interval_s", 60)
        self._reconnect_delay = data_config.get("reconnect_delay_s", 5)
        self._max_reconnect_attempts = data_config.get("max_reconnect_attempts", 5)
        self._max_reconnect_delay = data_config.get("max_reconnect_delay_s", 60)

        # Data processing configuration
        self._book_depth = data_config.get("book_depth", 10)
        self._ohlc_intervals = data_config.get("ohlc_intervals", [1, 5, 15, 60])
        self._expected_ohlc_item_length = data_config.get("expected_ohlc_item_length", 7)
        self._min_qty_threshold = data_config.get("min_qty_threshold", 1e-12)

        # Error handling configuration
        self._critical_error_threshold = data_config.get("critical_error_threshold", 3)

        # Trading pairs configuration
        self._trading_pairs = self._config.get("trading", {}).get("pairs", ["XRP/USD"])

        # Initialize error tracking for HALT conditions
        self._consecutive_errors = 0

    def _validate_initial_config(self) -> None:
        """Validate initial configuration settings."""
        if not self._trading_pairs:
            raise ValueError("DataIngestor: 'trading_pairs' configuration cannot be empty.")

        valid_book_depths = [0, 10, 25, 100, 500, 1000]
        if self._book_depth not in valid_book_depths:
            self.logger.warning(
                "Invalid book_depth, defaulting to 10.",
                source_module=self._source_module,
                context={"book_depth": self._book_depth, "valid_depths": valid_book_depths})
            self._book_depth = 10

        valid_intervals = [i for i in self._ohlc_intervals if i in self.INTERVAL_MAP]
        if len(valid_intervals) != len(self._ohlc_intervals):
            invalid_intervals = set(self._ohlc_intervals) - set(valid_intervals)
            self.logger.warning(
                "Invalid ohlc_intervals found. Using only valid intervals.",
                source_module=self._source_module,
                context={
                    "invalid_intervals": invalid_intervals,
                    "valid_intervals": valid_intervals,
                })
        self._ohlc_intervals = valid_intervals

        # Validate WebSocket URL format
        if not self._websocket_url.startswith(("ws://", "wss://")):
            raise ValueError(f"Invalid WebSocket URL format: {self._websocket_url}")

    def _build_subscription_message(self) -> str | None:
        """Build the Kraken WebSocket v2 subscription message.

        Returns:
        -------
            Optional[str]: The subscription message JSON string or None if no subscriptions
        """
        subscriptions_params = []
        self._subscriptions = {}

        # Book Subscription
        if self._book_depth > 0:
            book_params = {
                "channel": "book",
                "symbol": self._trading_pairs,
                "depth": self._book_depth,
                "snapshot": True,
            }
            subscriptions_params.append(book_params)
            # Store subscription details
            for pair in self._trading_pairs:
                sub_key = f"book_{pair}"
                self._subscriptions[sub_key] = {
                    "channel": "book",
                    "symbol": pair,
                    "depth": self._book_depth,
                }

        # OHLC Subscription
        if self._ohlc_intervals:
            ohlc_params = {
                "channel": "ohlc",
                "symbol": self._trading_pairs,
                "interval": self._ohlc_intervals,
                "snapshot": True,
            }
            subscriptions_params.append(ohlc_params)
            # Store subscription details
            for pair in self._trading_pairs:
                for interval in self._ohlc_intervals:
                    sub_key = f"ohlc_{pair}_{interval}"
                    self._subscriptions[sub_key] = {
                        "channel": "ohlc",
                        "symbol": pair,
                        "interval": interval,
                    }

        # Trade Subscription (NEW)
        if self._trading_pairs:  # Only add if there are pairs to subscribe to
            trade_params = {
                "channel": "trade",
                "symbol": self._trading_pairs,
            }
            subscriptions_params.append(trade_params)
            for pair in self._trading_pairs:
                sub_key = f"trade_{pair}"
                self._subscriptions[sub_key] = {
                    "channel": "trade",
                    "symbol": pair,
                }

        if not subscriptions_params:
            self.logger.exception(
                "No valid subscriptions configured.",
                source_module=self.__class__.__name__)
            return None

        try:
            return json.dumps({"method": "subscribe", "params": subscriptions_params})
        except Exception as error:
            self.logger.error(
                "Error building subscription message.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"error": str(error)})
            return None

    async def start(self) -> None:
        """Start the data ingestion process with reconnection logic."""
        self._is_running = True
        self._is_stopping = False
        self.logger.info("Starting Data Ingestor...", source_module=self.__class__.__name__)

        subscription_msg = self._build_subscription_message()
        if not subscription_msg:
            self.logger.error(
                "No subscriptions configured. Data Ingestor cannot start.",
                source_module=self.__class__.__name__)
            self._is_running = False
            return

        while self._is_running:
            connected_and_setup = False
            try:
                if await self._establish_connection():
                    if await self._setup_connection(subscription_msg):
                        connected_and_setup = True
                        # Listen loop
                        await self._message_listen_loop()
                    else:
                        await self._cleanup_connection()  # Setup failed
                # If establish or setup failed, connected_and_setup remains False
            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosedError) as e:
                close_code = getattr(e, "code", None)
                close_reason = getattr(e, "reason", "Unknown reason")
                self.logger.warning(
                    "WebSocket connection closed.",
                    source_module=self.__class__.__name__,
                    context={"code": close_code, "reason": close_reason})
                # Expected closure or error, proceed to reconnect logic
            except Exception as e:
                self._handle_connection_error(e)
            finally:
                await self._cleanup_connection()

            # Reconnect logic only if running and connection failed/closed
            if (
                self._is_running
                and not connected_and_setup
                and not await self._reconnect_with_backoff()
            ):
                break  # Stop if reconnect fails permanently

        self.logger.info("Data Ingestor stopped.", source_module=self.__class__.__name__)

    async def _establish_connection(self) -> bool:
        """Establish WebSocket connection.

        Returns:
        -------
            bool: True if connection was established successfully
        """
        self.logger.info(
            "Attempting to connect.",
            source_module=self.__class__.__name__,
            context={"url": self._websocket_url})
        try:
            # Add timeout to connect attempt itself
            async with asyncio.timeout(self._connection_timeout + 5):
                # Connect to the WebSocket
                self._connection = await websockets.connect(self._websocket_url)
                self._last_message_received_time = datetime.now(UTC)
                self._connection_established_time = datetime.now(
                    UTC)  # Record connection time
                self.logger.info("WebSocket connected.", source_module=self.__class__.__name__)
                return True
        except TimeoutError:
            self.logger.warning(
                "Connection attempt timed out. Retrying...",
                source_module=self.__class__.__name__,
                context={"timeout": self._connection_timeout + 5})
            return False
        except (
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.InvalidStatus,
            ConnectionRefusedError,
            OSError) as e:
            if e is not None:  # Filter out 'None' errors which can happen during shutdown
                self.logger.warning(
                    "WebSocket connection error. Reconnecting.",
                    source_module=self.__class__.__name__,
                    context={"error": str(e), "delay": self._reconnect_delay})
            return False

    async def _setup_connection(self, subscription_msg: str) -> bool:
        """Set up the connection by starting liveness monitor and subscribing.

        Args:
        ----
            subscription_msg: The subscription message to send

        Returns:
        -------
            bool: True if setup was successful
        """
        success = False  # Default to False
        try:
            await self._start_liveness_monitor()
            if self._connection is not None:
                await self._connection.send(subscription_msg)
                self.logger.info(
                    "Sent subscription request.",
                    source_module=self.__class__.__name__)
                self.logger.debug(
                    "Subscription message content.",
                    source_module=self.__class__.__name__,
                    context={"message": subscription_msg})
                success = True  # Set to True only if all steps in try succeed
            # If self._connection was None, success remains False
        except Exception as e:
            self.logger.error(
                "Error during connection setup.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"error": str(e)})
            # success remains False
        return success  # Return the final state

    async def _start_liveness_monitor(self) -> None:
        """Start the liveness monitor task."""
        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):  # SIM105
                await self._liveness_task
        self._liveness_task = asyncio.create_task(self._monitor_connection_liveness_loop())

    async def _message_listen_loop(self) -> None:
        """Listen for and process incoming messages."""
        if self._connection is None:
            return

        async for message in self._connection:
            self._last_message_received_time = datetime.now(UTC)
            try:
                await self._process_message(message)
            except Exception as error:
                self.logger.error(  # - LoggerService uses .error with exc_info
                    "Error processing incoming message.",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                    context={"error": str(error)})

    def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors.

        Args:
        ----
            error: The error that occurred
        """
        self.logger.error(
            "Unexpected error in Data Ingestor loop. Reconnecting.",
            source_module=self.__class__.__name__,
            context={"error": str(error), "reconnect_delay_s": self._reconnect_delay})

    async def stop(self) -> None:
        """Stop the data ingestion process gracefully."""
        self._is_running = False
        self._is_stopping = True
        self.logger.info("Stopping Data Ingestor...", source_module=self.__class__.__name__)
        await self._cleanup_connection()

    async def _cleanup_connection(self) -> None:
        """Clean up connection resources."""
        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
            try:
                await self._liveness_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.warning(
                    "Error closing WebSocket connection.",
                    source_module=self.__class__.__name__,
                    context={"error": str(e)})

        self._connection = None
        self._connection_id = None  # Reset connection specific state

    async def _reconnect_with_backoff(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns:
        -------
            bool: True if reconnection was successful, False if max retries exceeded
        """
        self.logger.info(
            "Attempting to reconnect to WebSocket...", source_module=self._source_module)
        await self._cleanup_connection()

        # Start with initial delay and increase exponentially
        delay = self._reconnect_delay

        for attempt in range(1, self._max_reconnect_attempts + 1):
            try:
                self.logger.info(
                    "Reconnection attempt %d/%d (delay: %ds)",
                    attempt,
                    self._max_reconnect_attempts,
                    delay,
                    source_module=self._source_module)

                # Wait before retrying
                await asyncio.sleep(delay)

                # Attempt to establish connection
                if await self._establish_connection():
                    subscription_msg = self._build_subscription_message()
                    if subscription_msg and await self._setup_connection(subscription_msg):
                        self.logger.info(
                            "Successfully reconnected on attempt %d",
                            attempt,
                            source_module=self._source_module)
                        self._consecutive_errors = 0  # Reset error counter on success
                        return True

                # Increase delay with exponential backoff (cap at max_delay)
                delay = min(delay * 2, self._max_reconnect_delay)

            except Exception as e:
                # Log the specific reconnection error
                self.logger.error(
                    "Error during reconnection attempt %d: %s",
                    attempt,
                    str(e),
                    source_module=self._source_module)

                # If this is the last attempt, trigger HALT consideration
                if attempt == self._max_reconnect_attempts:
                    context = {
                        "reconnection_attempts": attempt,
                        "last_error": str(e),
                    }
                    await self._trigger_halt_if_needed(e, context)

        self.logger.error(
            "Failed to reconnect after %d attempts",
            self._max_reconnect_attempts,
            source_module=self._source_module)
        return False

    async def _monitor_connection_liveness_loop(self) -> None:
        """Periodically checks if messages are being received."""
        monitor_msg = (
            f"Starting connection liveness monitor "
            f"(timeout: {self._connection_timeout}s, "
            f"heartbeat timeout: {self._max_heartbeat_interval}s)..."
        )
        self.logger.info(monitor_msg, source_module=self.__class__.__name__)
        check_interval = max(1, min(self._connection_timeout, self._max_heartbeat_interval) / 2)

        while self._is_running and self._connection and not self._connection.closed:
            # Check if task was cancelled externally (e.g., during shutdown)
            current_task = asyncio.current_task()
            if current_task and current_task.cancelled():
                break
            await asyncio.sleep(check_interval)

            now = datetime.now(UTC)
            general_timeout = False
            heartbeat_timeout = False

            # Check general message timeout
            if self._last_message_received_time:
                time_since_last = now - self._last_message_received_time
                if time_since_last > timedelta(seconds=self._connection_timeout):
                    self.logger.warning(
                        "No messages received for a period. Assuming connection loss.",
                        source_module=self.__class__.__name__,
                        context={
                            "time_since_last_message_seconds":
                                f"{time_since_last.total_seconds():.1f}",
                            "timeout_seconds": self._connection_timeout,
                        })
                    general_timeout = True
            elif self._connection_established_time:
                # No messages received *at all* yet after connecting
                time_since_connect = now - self._connection_established_time
                if time_since_connect > timedelta(seconds=self._connection_timeout):
                    self.logger.warning(
                        "No messages received since connecting. Triggering reconnect.",
                        source_module=self.__class__.__name__,
                        context={
                            "time_since_connect_seconds":
                                f"{time_since_connect.total_seconds():.1f}",
                            "timeout_seconds": self._connection_timeout,
                        })
                    general_timeout = True

            # Check specific heartbeat timeout
            if self._last_heartbeat_received_time:
                time_since_last_hb = now - self._last_heartbeat_received_time
                if time_since_last_hb > timedelta(seconds=self._max_heartbeat_interval):
                    self.logger.warning(
                        "No heartbeat received recently. Assuming connection issue.",
                        source_module=self.__class__.__name__,
                        context={
                            "time_since_last_hb_seconds":
                                f"{time_since_last_hb.total_seconds():.1f}",
                            "max_heartbeat_interval_seconds": self._max_heartbeat_interval,
                        })
                    heartbeat_timeout = True
            elif (
                self._connection_established_time
                and now - self._connection_established_time
                > timedelta(seconds=self._max_heartbeat_interval * 2)
            ):
                # If we've been connected for a while but never received a heartbeat
                self.logger.warning(
                    "No heartbeat received for an extended period after connecting.",
                    source_module=self.__class__.__name__,
                    context={
                        "max_heartbeat_interval_seconds_doubled": self._max_heartbeat_interval * 2,
                    })
                heartbeat_timeout = True

            # Trigger reconnect if either timeout occurs
            if general_timeout or heartbeat_timeout:
                if self._connection and not self._connection.closed:
                    # Use create_task to avoid blocking the monitor loop
                    self._cleanup_connection_task = asyncio.create_task(self._cleanup_connection())
                break  # Exit monitor loop, main loop will handle reconnect

        self.logger.info(
            "Connection liveness monitor stopped.",
            source_module=self.__class__.__name__)

    async def _process_message(self, message: str | bytes) -> None:
        """Parse and route incoming WebSocket messages."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        self.logger.debug(
            "Received message snippet.",
            source_module=self.__class__.__name__,
            context={"message_start": message[:200]})

        try:
            data = json.loads(message)
            if isinstance(data, dict):
                if "method" in data:
                    await self._handle_method_message(data)
                elif "channel" in data:
                    await self._handle_channel_message(data)
                else:
                    self.logger.warning(
                        "Unknown message format.",
                        source_module=self.__class__.__name__,
                        context={"message_start": message[:200]})
        except Exception:
            self.logger.exception(
                "Error processing message",
                source_module=self.__class__.__name__)

    async def _handle_method_message(self, data: dict[str, Any]) -> None:
        """Handle method-type messages (e.g., subscription acknowledgments).

        Args:
            data: The message data
        """
        method = data.get("method")
        if method == "subscribe":
            await self._handle_subscribe_ack(data)
        elif method == "unsubscribe":
            self.logger.debug(
                "Unsubscribe acknowledgment received.",
                source_module=self.__class__.__name__,
                context={"data": data})
        else:
            self.logger.debug(
                "Unhandled method message: %s",
                method,
                source_module=self.__class__.__name__)

    async def _handle_channel_message(self, data: dict[str, Any]) -> None:
        """Handle channel-type messages (e.g., market data).

        Args:
        ----
            data: The message data
        """
        channel = data["channel"]
        msg_type = data.get("type")

        try:
            if channel == "status":
                await self._handle_status_update(data)
            elif channel == "heartbeat":
                self._last_heartbeat_received_time = datetime.now(UTC)
                self.logger.debug("Received heartbeat.", source_module=self.__class__.__name__)
            elif channel == "book" and msg_type in ["snapshot", "update"]:
                await self._handle_book_data(data)
            elif channel == "ohlc" and msg_type in ["snapshot", "update"]:
                # Explicitly cast data to Dict[str, Any] to satisfy mypy
                typed_data: dict[str, Any] = data
                await self._handle_ohlc_data(typed_data)
            # Kraken trade messages are typically of type 'update'
            elif channel == "trade" and msg_type == "update":
                await self._handle_trade_data(data)  # Use dedicated trade handler
            else:
                self.logger.warning(
                    "Unknown channel or message type.",
                    source_module=self.__class__.__name__,
                    context={"channel": channel, "type": msg_type})
        except Exception as error:
            self.logger.error(  # - LoggerService uses .error with exc_info
                "Error handling channel message.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"channel": channel, "error": str(error)})

    async def _handle_subscribe_ack(self, data: dict[str, Any]) -> None:
        """Handle the acknowledgment message for subscriptions."""
        try:
            success = data.get("success")
            result = data.get("result", {})
            channel = result.get("channel")
            symbol = result.get("symbol")  # Can be None if top-level ack
            error = data.get("error")
            req_id = data.get("req_id")

            log_prefix = (
                "Subscription Ack (ReqID: {req_id}): " if req_id else "Subscription Ack: "
            ).format(req_id=req_id)

            if success:
                success_msg = f"{log_prefix}Success for Channel={channel}, Symbol={symbol}"
                self.logger.info(
                    success_msg,
                    source_module=self.__class__.__name__)
                # Can potentially update self._subscriptions based on acks if
                # needed
            else:
                error_msg = (
                    f"{log_prefix}FAILED! Error: {error}, Channel={channel}, Symbol={symbol}"
                )
                self.logger.error(
                    error_msg,
                    source_module=self.__class__.__name__)
                # Consider specific error handling (e.g., stop if critical
                # subscription fails)

        except Exception as e:
            self.logger.error(  # - LoggerService uses .error with exc_info
                "Error processing subscription ack.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"data": str(data), "error": str(e)})

    async def _handle_status_update(self, data: dict[str, Any]) -> None:
        """Handle connection or system status updates."""
        status = data.get("status")
        connection_id = data.get("connectionID")
        # Ensure version is string or None for consistent logging/processing
        raw_version = data.get("version")
        version_str = str(raw_version) if raw_version is not None else None

        log_payload = {"status": status, "connection_id": connection_id, "version": version_str}

        if status == "online":
            self.logger.info(
                "WebSocket connection is online.",
                source_module=self._source_module,
                context=log_payload)
            # Potentially publish a system state event if needed
        elif status == "error":
            error_msg = data.get("message", "Unknown error")
            self.logger.error(
                "WebSocket status error.",
                source_module=self._source_module,
                context={**log_payload, "error_message": error_msg},  # Merge contexts
            )
            # Trigger reconnection or halt?
        else:
            self.logger.info(
                "Received status update.",
                source_module=self._source_module,
                context={**log_payload, "status_value": status},  # Merge contexts
            )

        # Example: Publish SystemStateEvent (adjust fields as needed based on
        # core.events)
        event = SystemStateEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            # Map Kraken status to internal state if needed
            new_state=str(status) if status is not None else "unknown", # Ensure str
            reason=(
                f"Kraken WS Status Update: "
                f"{str(status) if status is not None else 'unknown'}"
            ),  # Ensure str in f-string part
        )
        try:
            await self.pubsub.publish(event)
            self.logger.debug("Published SystemStateEvent.", context={"status": status})
        except Exception as e:
            self.logger.error(  # - LoggerService uses .error with exc_info
                "Failed to publish SystemStateEvent.",
                exc_info=True,
                context={"error": str(e)})

    async def _handle_book_data(self, data: dict[str, Any]) -> bool:
        """Handle book data message.

        Args:
        ----
            data: The book data message

        Returns:
        -------
            bool: True if successful, False otherwise
        """
        if not self._validate_book_message(data):
            return False

        is_snapshot = data.get("type") == "snapshot"
        processed_ok = True

        # Book data structure: data: [ { book_obj } ]
        for book_item in data.get("data", []):
            if not self._validate_book_item(book_item):
                processed_ok = False
                continue  # Skip this item

            symbol = book_item.get("symbol")
            book_state = self._l2_books[symbol]
            received_checksum = book_item.get("checksum")
            update_timestamp = book_item.get("timestamp")

            try:
                # Process the book item
                valid_after_apply = await self._process_book_item(
                    book_state,
                    book_item,
                    symbol,
                    received_checksum,
                    is_snapshot)

                if not valid_after_apply:
                    continue

                # Publish the update
                await self._publish_book_event(symbol, book_state, is_snapshot, update_timestamp)

            except Exception as error:
                self.logger.error(  # - LoggerService uses .error with exc_info
                    "Error processing book data.",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                    context={"error": str(error)})
                return False

        return processed_ok

    def _validate_book_message(self, data: dict[str, Any]) -> bool:
        """Validate the overall book message structure.

        Args:
        ----
            data: The book data message

        Returns:
        -------
            bool: True if valid, False otherwise
        """
        if not isinstance(data.get("data"), list):
            self.logger.warning(
                "Invalid book message: 'data' is not a list[Any].",
                source_module=self._source_module,
                context={"message_snippet": str(data)[:200]})
            return False

        msg_type = data.get("type")
        if msg_type not in ["snapshot", "update"]:
            self.logger.warning(
                "Invalid book message type.",
                source_module=self._source_module,
                context={"type": msg_type, "message_snippet": str(data)[:200]})
            return False

        return True

    def _validate_book_item(self, book_item: dict[str, Any]) -> bool:
        """Validate that a book item has the expected structure.

        Args:
        ----
            book_item: The book item to validate

        Returns:
        -------
            bool: True if valid, False otherwise
        """
        required_fields = ["symbol", "bids", "asks"]

        for field_name in required_fields:
            if field_name not in book_item:
                self.logger.warning(
                    "Book item missing required field: %s",
                    field,
                    source_module=self._source_module,
                    context={"book_item_keys": list[Any](book_item.keys())})
                return False

        # Validate bids and asks are lists
        if not isinstance(book_item["bids"], list) or not isinstance(book_item["asks"], list):
            self.logger.warning(
                "Book item bids/asks must be lists",
                source_module=self._source_module,
                context={
                    "bids_type": type(book_item["bids"]).__name__,
                    "asks_type": type(book_item["asks"]).__name__,
                })
            return False

        return True

    async def _process_book_item(
        self,
        book_state: dict[str, Any],
        book_item: dict[str, Any],
        symbol: str,
        received_checksum: int | None,
        is_snapshot: bool) -> bool:
        """Process a validated book item.

        Args:
        ----
            book_state: Current book state to update
            book_item: Book data from exchange
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange
            is_snapshot: Whether this is a snapshot or update

        Returns:
        -------
            bool: True if processing was successful, False otherwise
        """
        try:
            if is_snapshot:
                valid_after_apply = self._apply_book_snapshot(
                    book_state,
                    book_item,
                    symbol,
                    received_checksum)
            else:
                valid_after_apply = self._apply_book_update(book_state, book_item, symbol)

            # Truncate book to subscribed depth
            valid_after_apply = self._truncate_book_to_depth(book_state, symbol, valid_after_apply)

            # Validate checksum and update book state
            return self._validate_and_update_checksum(
                book_state,
                symbol,
                received_checksum,
                valid_after_apply)

        except Exception as e:
            self.logger.error(  # - LoggerService uses .error with exc_info
                "Error in book item processing.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"error": str(e)})
            return False

    def _apply_book_snapshot(
        self,
        book_state: dict[str, Any],
        book_item: dict[str, Any],
        symbol: str,
        received_checksum: int | None) -> bool:
        """Apply a book snapshot to the book state.

        Args:
        ----
            book_state: Current book state to update
            book_item: Snapshot data from exchange
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange

        Returns:
        -------
            bool: True if snapshot was applied successfully
        """
        book_state["asks"].clear()
        book_state["bids"].clear()

        # Process asks
        for level in book_item.get("asks", []):
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) > self._min_qty_threshold:
                book_state["asks"][price_str] = qty_str

        # Process bids
        for level in book_item.get("bids", []):
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) > self._min_qty_threshold:
                book_state["bids"][price_str] = qty_str

        self.logger.info(
            "Processed L2 snapshot.",
            source_module=self.__class__.__name__,
            context={
                "symbol": symbol,
                "ask_levels": len(book_state["asks"]),
                "bid_levels": len(book_state["bids"]),
            })
        book_state["checksum"] = received_checksum
        return True

    def _apply_book_update(self, book_state: dict[str, Any], book_item: dict[str, Any], symbol: str) -> bool:
        """Apply book updates to the book state.

        Args:
        ----
            book_state: Current book state to update
            book_item: Update data from exchange
            symbol: Trading pair symbol

        Returns:
        -------
            bool: True if any updates were applied
        """
        asks_updated = self._update_price_levels(
            book_state["asks"],
            book_item.get("asks", []),
            symbol,
            "ask")
        bids_updated = self._update_price_levels(
            book_state["bids"],
            book_item.get("bids", []),
            symbol,
            "bid")

        valid_after_apply = asks_updated or bids_updated
        if not valid_after_apply:
            self.logger.debug(
                "Received book update with no effective changes.",
                source_module=self.__class__.__name__,
                context={"symbol": symbol})
            return True  # Treat no change as valid for checksum check

        return valid_after_apply

    def _update_price_levels(
        self,
        book_side: Any,  # SortedDict instance
        updates: list[Any],
        symbol: str,
        side: str) -> bool:
        """Update price levels for one side of the book.

        Args:
        ----
            book_side: SortedDict containing the side's price levels
            updates: List of updates from exchange
            symbol: Trading pair symbol
            side: Side being updated ("bid" or "ask")

        Returns:
        -------
            bool: True if any updates were applied
        """
        updated = False
        for level in updates:
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) < self._min_qty_threshold:  # Remove level
                if book_side.pop(price_str, None):
                    updated = True
                    self.logger.debug(
                        "Removed level.",
                        source_module=self.__class__.__name__,
                        context={"side": side, "price": price_str, "symbol": symbol})
            elif book_side.get(price_str) != qty_str:
                book_side[price_str] = qty_str
                updated = True
                self.logger.debug(
                    "Updated level.",
                    source_module=self.__class__.__name__,
                    context={
                        "side": side,
                        "price": price_str,
                        "qty": qty_str,
                        "symbol": symbol,
                    })
        return updated

    def _truncate_book_to_depth(
        self,
        book_state: dict[str, Any],
        symbol: str,
        valid_after_apply: bool) -> bool:
        """Truncate book to subscribed depth.

        Args:
        ----
            book_state: Current book state to truncate
            symbol: Trading pair symbol
            valid_after_apply: Current validity state

        Returns:
        -------
            bool: Updated validity state
        """
        # Bids: Remove lowest price bids if count > depth
        while len(book_state["bids"]) > self._book_depth:
            removed_bid = book_state["bids"].popitem(0)
            self.logger.debug(
                "Truncated bid level due to depth limit.",
                source_module=self.__class__.__name__,
                context={"level": removed_bid[0], "symbol": symbol})
            valid_after_apply = True

        # Asks: Remove highest price asks if count > depth
        while len(book_state["asks"]) > self._book_depth:
            removed_ask = book_state["asks"].popitem(-1)
            self.logger.debug(
                "Truncated ask level due to depth limit.",
                source_module=self.__class__.__name__,
                context={"level": removed_ask[0], "symbol": symbol})
            valid_after_apply = True

        return valid_after_apply

    def _calculate_book_checksum(self, book_state: dict[str, Any]) -> int | None:
        """Calculate the checksum for the order book.

        Args:
            book_state: The current book state containing bids and asks

        Returns:
        -------
            The calculated checksum or None if calculation fails
        """
        try:
            # This is a simplified implementation - the actual algorithm would depend on
            # the specific exchange's checksum calculation method
            checksum_str = ""

            # Process bids (sort in descending order - highest bid first)
            bids = book_state["bids"]
            for price in list[Any](bids.keys())[:10]:  # Use top 10 levels
                qty = bids[price]
                checksum_str += f"{float(price):.8f}:{float(qty):.8f}:"

            # Process asks (sort in ascending order - lowest ask first)
            asks = book_state["asks"]
            for price in list[Any](asks.keys())[:10]:  # Use top 10 levels
                qty = asks[price]
                checksum_str += f"{float(price):.8f}:{float(qty):.8f}:"

            # Use CRC32 algorithm to generate checksum
            import zlib

            return zlib.crc32(checksum_str.encode())
        except Exception as e:
            self.logger.exception(
                "Error calculating checksum",
                source_module=self._source_module,
                exc_info=False,
                context={"error": str(e)})
            return None

    def _validate_and_update_checksum(
        self,
        book_state: dict[str, Any],
        symbol: str,
        received_checksum: int | None,
        valid_after_apply: bool) -> bool:
        """Validate and update book checksum.

        Args:
        ----
            book_state: Current book state
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange
            valid_after_apply: Whether updates were applied

        Returns:
        -------
            bool: True if checksum is valid
        """
        if received_checksum is not None and valid_after_apply:
            local_checksum = self._calculate_book_checksum(book_state)
            if local_checksum is None:
                error_msg = (
                    f"Failed to calculate local checksum for {symbol}. "
                    "Skipping validation and event publishing."
                )
                self.logger.error(error_msg, source_module=self.__class__.__name__)
                book_state["checksum"] = None
                return False

            if local_checksum != received_checksum:
                self._handle_checksum_mismatch(
                    book_state,
                    symbol,
                    local_checksum,
                    received_checksum)
                return False

            book_state["checksum"] = received_checksum
            self.logger.debug(
                "Checksum validation passed.",
                source_module=self.__class__.__name__,
                context={"symbol": symbol})
            return True

        if not valid_after_apply:
            return self._handle_no_updates_case(book_state, symbol, received_checksum)

        if received_checksum is None:
            warning_msg = f"No checksum received in book message for {symbol}. Cannot validate."
            self.logger.warning(warning_msg, source_module=self.__class__.__name__)
            return True

        return True

    def _handle_checksum_mismatch(
        self: "DataIngestor",
        book_state: dict[str, Any],
        symbol: str,
        local_checksum: int,
        received_checksum: int) -> None:
        """Handle checksum mismatch case.

        Args:
        ----
            book_state: Current book state
            symbol: Trading pair symbol
            local_checksum: Locally calculated checksum
            received_checksum: Checksum from exchange
        """
        self.logger.error(
            "Checksum mismatch. Book is potentially corrupt.",
            source_module=self.__class__.__name__,
            context={
                "symbol": symbol,
                "local_checksum": local_checksum,
                "received_checksum": received_checksum,
            })
        self.logger.error(
            "Bids (Top 3).",
            source_module=self.__class__.__name__,
            context={"bids_top_3": list[Any](reversed(book_state["bids"].items()))[:3]})
        self.logger.error(
            "Asks (Top 3).",
            source_module=self.__class__.__name__,
            context={"asks_top_3": list[Any](book_state["asks"].items())[:3]})
        book_state["checksum"] = None

    def _handle_no_updates_case(
        self: "DataIngestor",
        book_state: dict[str, Any],
        symbol: str,
        received_checksum: int | None) -> bool:
        """Handle case where no updates were applied.

        Args:
        ----
            book_state: Current book state
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange

        Returns:
        -------
            bool: True if state is valid
        """
        if book_state["checksum"] == received_checksum:
            self.logger.debug(
                "No book changes, checksum still matches.",
                source_module=self.__class__.__name__,
                context={"symbol": symbol, "checksum": received_checksum})
            return True

        if received_checksum is not None:
            self.logger.warning(
                "Book checksum changed but no updates applied locally. Recalculating.",
                source_module=self.__class__.__name__,
                context={
                    "symbol": symbol,
                    "previous_checksum": book_state["checksum"],
                    "new_checksum": received_checksum,
                })
            local_checksum = self._calculate_book_checksum(book_state)
            if local_checksum == received_checksum:
                self.logger.info(
                    "Recalculated checksum matches received.",
                    source_module=self.__class__.__name__,
                    context={
                        "symbol": symbol,
                        "checksum": local_checksum,
                    })
                book_state["checksum"] = received_checksum
                return True

            error_msg = (
                f"Recalculated checksum {local_checksum} still does not match "
                f"received {received_checksum} for {symbol} despite no updates."
            )
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            book_state["checksum"] = None
            return False

        return True

    async def _publish_book_event(
        self: "DataIngestor",
        symbol: str,
        book_state: dict[str, Any],
        is_snapshot: bool,
        update_timestamp: str | None,  # Timestamp from message if available
    ) -> None:
        """Create and publish a MarketDataL2Event."""
        try:
            # Extract bids and asks from the book_state (SortedDict)
            # Ensure they are formatted as List[Tuple[str, str]]
            bids_list = [(str(price), str(vol)) for price, vol in book_state["bids"].items()]
            asks_list = [(str(price), str(vol)) for price, vol in book_state["asks"].items()]

            # Parse the exchange timestamp if provided
            exchange_ts = None
            if update_timestamp:
                try:
                    # Kraken v2 uses float seconds with nanosecond precision
                    exchange_ts = datetime.fromtimestamp(float(update_timestamp), tz=UTC)
                except (ValueError, TypeError):
                    self.logger.warning(
                        "Could not parse book update timestamp.",
                        context={"timestamp": update_timestamp})

            event = MarketDataL2Event(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),  # Event creation time
                trading_pair=symbol,
                exchange=self._config.get("data_ingestion.default_exchange", "kraken"),
                bids=bids_list,
                asks=asks_list,
                is_snapshot=is_snapshot,
                timestamp_exchange=exchange_ts,  # Timestamp from Kraken message
            )
            await self.pubsub.publish(event)
        except Exception:
            self.logger.exception(
                "Unexpected error publishing OHLCV event.",
                source_module=self._source_module)

    def _validate_ohlc_item(self, ohlc_item: Any) -> bool:
        """Validate OHLC item structure.

        Args:
        ----
            ohlc_item: OHLC data item to validate

        Returns:
        -------
            bool: True if valid, False otherwise
        """
        if not isinstance(ohlc_item, list):
            self.logger.warning(
                "OHLC item must be a list[Any]",
                source_module=self._source_module,
                context={"item_type": type(ohlc_item).__name__})
            return False

        if len(ohlc_item) != self._expected_ohlc_item_length:
            self.logger.warning(
                "OHLC item has unexpected length",
                source_module=self._source_module,
                context={
                    "expected_length": self._expected_ohlc_item_length,
                    "actual_length": len(ohlc_item),
                })
            return False

        return True

    async def _handle_ohlc_data(self, data: dict[str, Any]) -> None:
        """Handle OHLC data messages and validate them properly.

        Args:
        ----
            data: OHLC data from WebSocket
        """
        try:
            if "data" not in data:
                self.logger.warning(
                    "OHLC message missing data field",
                    source_module=self._source_module)
                return

            ohlc_data = data["data"]

            # Validate each OHLC item
            if isinstance(ohlc_data, list):
                for item in ohlc_data:
                    if not self._validate_ohlc_item(item):
                        self.logger.warning(
                            "Skipping invalid OHLC item",
                            source_module=self._source_module,
                            context={"item_preview": str(item)[:100]})
                        continue

                    # Process valid OHLC item
                    # (Existing OHLC processing logic would go here)

        except Exception as e:
            context = {"data_preview": str(data)[:200]}
            await self._trigger_halt_if_needed(e, context)
            self.logger.error(
                "Critical error processing OHLC data",
                source_module=self._source_module,
                exc_info=True,
                context=context)

    async def _handle_trade_data(self, data: dict[str, Any]) -> bool:
        """Handle trade data message.

        Trade messages contain information about executed trades including
        price, volume, time, and side (buy/sell).

        Args:
        ----
            data: The trade data message from Kraken

        Returns:
        -------
            bool: True if successful, False otherwise
        """
        msg_type = data.get("type")
        if msg_type not in ["update"]:  # Trades are typically updates
            self.logger.warning(
                "Unexpected trade message type: %s",
                msg_type,
                source_module=self._source_module,
                context={"message_type": msg_type})
            return False

        # Trade data structure: data: [ { trade_obj } ]
        trade_items = data.get("data", [])
        if not isinstance(trade_items, list):
            self.logger.warning(
                "Invalid trade message: 'data' is not a list[Any].",
                source_module=self._source_module,
                context={"message_snippet": str(data)[:200]})
            return False

        processed_ok = True
        for trade_item in trade_items:
            if not self._validate_trade_item(trade_item):
                processed_ok = False
                continue

            try:
                symbol = trade_item.get("symbol")
                if not symbol:
                    self.logger.warning(
                        "Trade item missing symbol",
                        source_module=self._source_module,
                        context={"trade_item": trade_item})
                    continue

                # Process the trade
                await self._process_and_publish_trade(symbol, trade_item)

            except Exception as error:
                self.logger.error(
                    "Error processing trade data.",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                    context={"error": str(error), "symbol": symbol})
                processed_ok = False

        return processed_ok

    def _validate_trade_item(self, trade_item: dict[str, Any]) -> bool:
        """Validate that a trade item has the expected structure.

        Args:
        ----
            trade_item: The trade item to validate

        Returns:
        -------
            bool: True if valid, False otherwise
        """
        required_fields = ["symbol", "price", "qty", "timestamp"]

        for field_name in required_fields:
            if field_name not in trade_item:
                self.logger.warning(
                    "Trade item missing required field: %s",
                    field_name,
                    source_module=self._source_module,
                    context={"trade_item_keys": list[Any](trade_item.keys())})
                return False

        return True

    async def _process_and_publish_trade(
        self,
        symbol: str,
        trade_item: dict[str, Any]) -> None:
        """Process a trade item and publish as event.

        Args:
        ----
            symbol: Trading pair symbol
            trade_item: Trade data from exchange
        """
        try:
            # Extract trade data
            price_str = str(trade_item.get("price", "0"))
            qty_str = str(trade_item.get("qty", "0"))
            timestamp_str = trade_item.get("timestamp")
            side = trade_item.get("side", "").lower()  # buy/sell
            trade_id = trade_item.get("trade_id", "")

            # Additional validation
            if float(qty_str) <= 0:
                self.logger.debug(
                    "Ignoring zero or negative quantity trade for %s",
                    symbol,
                    source_module=self._source_module)
                return

            # Parse timestamp
            try:
                # Kraken uses Unix timestamp (float seconds)
                timestamp_exchange = datetime.fromtimestamp(
                    float(timestamp_str) if timestamp_str is not None else 0,
                    tz=UTC,
                )
            except (ValueError, TypeError):
                self.logger.warning(
                    "Could not parse trade timestamp, using current time",
                    source_module=self._source_module,
                    context={"timestamp": timestamp_str})
                timestamp_exchange = datetime.now(UTC)

            # Validate side (MarketDataTradeEvent requires "buy" or "sell")
            if side not in ["buy", "sell"]:
                # Default to "buy" for unknown sides
                side = "buy"

            # Create trade event
            trade_event = MarketDataTradeEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair=symbol,
                exchange="kraken",
                timestamp_exchange=timestamp_exchange,
                price=Decimal(price_str),
                volume=Decimal(qty_str),
                side=side,
                trade_id=str(trade_id) if trade_id else None)

            try:
                await self.pubsub.publish(trade_event)
                self.logger.debug(
                    "Published trade event for %s: price=%s, qty=%s, side=%s",
                    symbol,
                    price_str,
                    qty_str,
                    side,
                    source_module=self._source_module)
            except Exception as e:
                self.logger.error(
                    "Failed to publish trade event.",
                    exc_info=True,
                    context={"error": str(e), "symbol": symbol})

        except Exception:
            self.logger.exception(
                "Error processing trade for %s",
                symbol,
                source_module=self._source_module,
                context={"trade_item": trade_item})

    async def _trigger_halt_if_needed(self, error: Exception, context: dict[str, Any]) -> None:
        """Trigger system HALT if error conditions warrant it.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        self._consecutive_errors += 1

        # Trigger HALT for critical errors or too many consecutive errors
        should_halt = (
            isinstance(error, ConnectionError | TimeoutError | OSError) and
            self._consecutive_errors >= self._critical_error_threshold # Changed from self.MAX_CONSECUTIVE_ERRORS
        )

        if should_halt:
            # Create halt event with formatted reason
            reason = f"Critical data ingestor error: {error!s}"
            if context:
                reason += f" | Context: {context}"

            halt_event = PotentialHaltTriggerEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                reason=reason)

            await self.pubsub.publish(halt_event)

            self.logger.critical(
                "Triggered potential system HALT due to data ingestor errors",
                source_module=self._source_module,
                context={
                    "error_type": type(error).__name__,
                    "consecutive_errors": self._consecutive_errors,
                    **context,
                })
