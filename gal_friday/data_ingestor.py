"""Retrieve and process market data from the Kraken WebSocket API.

This module implements a data ingestion service that connects to Kraken WebSocket API v2,
subscribes to L2 order book and OHLCV data streams, handles parsing, validation and state
management, and publishes standardized market data events to the application's event bus.
"""

# src/gal_friday/data_ingestor.py

import asyncio
from collections import defaultdict
import contextlib  # Added for SIM105 fix
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import random  # Add random for jitter in backoff
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union
import uuid

from sortedcontainers import SortedDict
import websockets

# Import necessary event classes from core module
from .core.events import (
    MarketDataL2Event,
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
    timestamp_exchange: Optional[str] = None
    # [(price_str, volume_str), ...] sorted desc
    bids: list[tuple[str, str]] = field(default_factory=list)
    # [(price_str, volume_str), ...] sorted asc
    asks: list[tuple[str, str]] = field(default_factory=list)
    is_snapshot: bool = False
    # Add checksum to event for potential downstream validation
    checksum: Optional[int] = None


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
    connection_id: Optional[int] = None


# --- DataIngestor Class ---


class DataIngestor:
    """
    Connect to Kraken WebSocket API v2, manage subscriptions for L2 book and OHLCV data.

    This class handles parsing messages, maintaining L2 order book state with
    checksum validation, and publishes standardized market data events.
    """

    KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
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
    _EXPECTED_OHLC_ITEM_LENGTH = 7  # PLR2004: For OHLC item validation
    _MIN_QTY_THRESHOLD = 1e-12  # PLR2004: For minimum quantity checks

    def __init__(
        self,
        config: "ConfigManager",
        pubsub_manager: "PubSubManager",
        logger_service: LoggerService,
    ) -> None:
        """Initialize the DataIngestor.

        Args
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
        self._websocket_url = config.get("kraken_ws_url", self.KRAKEN_WS_URL)
        self._trading_pairs = config.get("trading_pairs", ["XRP/USD"])
        self._book_depth = config.get("book_depth", 10)
        self._ohlc_intervals = config.get("ohlc_intervals", [1])
        self._reconnect_delay = config.get("reconnect_delay_s", 5)
        self._connection_timeout = config.get("connection_timeout_s", 15)
        self._max_heartbeat_interval = config.get("max_heartbeat_interval_s", 60)

        # Initialize state
        self._connection: Optional[Any] = None
        self._is_running: bool = False
        self._is_stopping: bool = False
        self._last_message_received_time: Optional[datetime] = None
        self._last_heartbeat_received_time: Optional[datetime] = None  # For heartbeat tracking
        self._connection_established_time: Optional[datetime] = (
            None  # For initial connection tracking
        )
        self._liveness_task: Optional[asyncio.Task] = None
        self._subscriptions: dict[str, dict[str, Any]] = {}
        self._connection_id: Optional[int] = None
        self._system_status: Optional[str] = None

        # Initialize book state
        self._l2_books: dict[str, dict[str, Union[SortedDict, Optional[int]]]] = defaultdict(
            lambda: {"bids": SortedDict(), "asks": SortedDict(), "checksum": None}
        )

        # Validate configuration
        self._validate_initial_config()

    def _validate_initial_config(self) -> None:
        """Validate initial configuration settings."""
        if not self._trading_pairs:
            raise ValueError("DataIngestor: 'trading_pairs' configuration cannot be empty.")  # noqa: TRY003

        if self._book_depth not in [0, 10, 25, 100, 500, 1000]:
            self.logger.warning(
                "Invalid book_depth, defaulting to 10.",
                source_module=self._source_module,
                context={"book_depth": self._book_depth}
            )
            self._book_depth = 10

        valid_intervals = [i for i in self._ohlc_intervals if i in self.INTERVAL_MAP]
        if len(valid_intervals) != len(self._ohlc_intervals):
            invalid_intervals = set(self._ohlc_intervals) - set(valid_intervals)
            self.logger.warning(
                "Invalid ohlc_intervals found. Using only valid intervals.",
                source_module=self._source_module,
                context={
                    "invalid_intervals": invalid_intervals,
                    "valid_intervals": valid_intervals
                }
            )
        self._ohlc_intervals = valid_intervals

    def _build_subscription_message(self) -> Optional[str]:
        """Build the Kraken WebSocket v2 subscription message.

        Returns
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
            self.logger.error(
                "No valid subscriptions configured.", source_module=self.__class__.__name__
            )
            return None

        try:
            return json.dumps({"method": "subscribe", "params": subscriptions_params})
        except Exception as error:
            self.logger.error(  # noqa: G201
                "Error building subscription message.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"error": str(error)}
            )
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
                source_module=self.__class__.__name__,
            )
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
                websockets.exceptions.ConnectionClosedError,
            ) as e:
                close_code = getattr(e, "code", None)
                close_reason = getattr(e, "reason", "Unknown reason")
                self.logger.warning(
                    "WebSocket connection closed.",
                    source_module=self.__class__.__name__,
                    context={"code": close_code, "reason": close_reason}
                )
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

        Returns
        -------
            bool: True if connection was established successfully
        """
        self.logger.info(
            "Attempting to connect.",
            source_module=self.__class__.__name__,
            context={"url": self._websocket_url}
        )
        try:
            # Add timeout to connect attempt itself
            async with asyncio.timeout(self._connection_timeout + 5):
                # Connect to the WebSocket
                self._connection = await websockets.connect(self._websocket_url)
                self._last_message_received_time = datetime.now(timezone.utc)
                self._connection_established_time = datetime.now(
                    timezone.utc
                )  # Record connection time
                self.logger.info("WebSocket connected.", source_module=self.__class__.__name__)
                return True
        except TimeoutError:
            self.logger.warning(
                "Connection attempt timed out. Retrying...",
                source_module=self.__class__.__name__,
                context={"timeout": self._connection_timeout + 5}
            )
            return False
        except (
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.InvalidStatus,
            ConnectionRefusedError,
            OSError,
        ) as e:
            if e is not None:  # Filter out 'None' errors which can happen during shutdown
                self.logger.warning(
                    "WebSocket connection error. Reconnecting.",
                    source_module=self.__class__.__name__,
                    context={"error": str(e), "delay": self._reconnect_delay}
                )
            return False

    async def _setup_connection(self, subscription_msg: str) -> bool:
        """Set up the connection by starting liveness monitor and subscribing.

        Args
        ----
            subscription_msg: The subscription message to send

        Returns
        -------
            bool: True if setup was successful
        """
        success = False  # Default to False
        try:
            await self._start_liveness_monitor()
            if self._connection is not None:
                await self._connection.send(subscription_msg)
                self.logger.info(
                    "Sent subscription request.", source_module=self.__class__.__name__
                )
                self.logger.debug(
                    "Subscription message content.",
                    source_module=self.__class__.__name__,
                    context={"message": subscription_msg}
                )
                success = True # Set to True only if all steps in try succeed
            # If self._connection was None, success remains False
        except Exception as e:
            self.logger.error(  # noqa: G201
                "Error during connection setup.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"error": str(e)}
            )
            # success remains False
        return success # Return the final state

    async def _start_liveness_monitor(self) -> None:
        """Start the liveness monitor task."""
        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
            with contextlib.suppress(asyncio.CancelledError): # SIM105
                await self._liveness_task
        self._liveness_task = asyncio.create_task(self._monitor_connection_liveness_loop())

    async def _message_listen_loop(self) -> None:
        """Listen for and process incoming messages."""
        if self._connection is None:
            return

        async for message in self._connection:
            self._last_message_received_time = datetime.now(timezone.utc)
            try:
                await self._process_message(message)
            except Exception as error:
                self.logger.error( # noqa: G201 - LoggerService uses .error with exc_info
                    "Error processing incoming message.",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                    context={"error": str(error)}
                )

    def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors.

        Args
        ----
            error: The error that occurred
        """
        self.logger.error(
            "Unexpected error in Data Ingestor loop. Reconnecting.",
            source_module=self.__class__.__name__,
            exc_info=True,
            context={"error": str(error), "reconnect_delay_s": self._reconnect_delay}
        )

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
                    context={"error": str(e)}
                )

        self._connection = None
        self._connection_id = None  # Reset connection specific state

    async def _reconnect_with_backoff(self) -> bool:
        """Attempt to reconnect with exponential backoff.

        Returns
        -------
            bool: True if reconnection was successful, False if max retries exceeded
        """
        self.logger.info("Attempting to reconnect to WebSocket...", source_module=self._source_module)
        await self._cleanup_connection()

        # Start with initial delay and increase exponentially
        delay = self._reconnect_delay
        max_delay = 60  # Maximum delay between reconnection attempts (1 minute)
        max_retries = 5  # Maximum number of reconnection attempts

        for attempt in range(1, max_retries + 1):
            self.logger.info(
                "Reconnection attempt %d/%d (delay: %ds)",
                attempt, max_retries, delay,
                source_module=self._source_module
            )

            # Wait before retrying
            await asyncio.sleep(delay)

            # Attempt to establish connection
            if await self._establish_connection():
                subscription_msg = self._build_subscription_message()
                if subscription_msg and await self._setup_connection(subscription_msg):
                    self.logger.info(
                        "Successfully reconnected on attempt %d",
                        attempt,
                        source_module=self._source_module
                    )
                    return True

            # Increase delay with exponential backoff (cap at max_delay)
            delay = min(delay * 2, max_delay)

        self.logger.error(
            "Failed to reconnect after %d attempts",
            max_retries,
            source_module=self._source_module
        )
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

            now = datetime.now(timezone.utc)
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
                        }
                    )
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
                        },
                    )
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
                        },
                    )
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
                        "max_heartbeat_interval_seconds_doubled":
                            self._max_heartbeat_interval * 2
                    }
                )
                heartbeat_timeout = True

            # Trigger reconnect if either timeout occurs
            if general_timeout or heartbeat_timeout:
                if self._connection and not self._connection.closed:
                    # Use create_task to avoid blocking the monitor loop
                    self._cleanup_connection_task = asyncio.create_task(self._cleanup_connection())
                break  # Exit monitor loop, main loop will handle reconnect

        self.logger.info(
            "Connection liveness monitor stopped.", source_module=self.__class__.__name__
        )

    async def _process_message(self, message: Union[str, bytes]) -> None:
        """Parse and route incoming WebSocket messages."""
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        self.logger.debug(
            "Received message snippet.",
            source_module=self.__class__.__name__,
            context={"message_start": message[:200]}
        )

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
                        context={"message_start": message[:200]}
                    )
        except Exception:
            self.logger.exception(
                "Error processing message",
                source_module=self.__class__.__name__
            )

    async def _handle_method_message(self, data: dict) -> None:
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
                context={"data": data}
            )
        else:
            self.logger.debug(
                "Unhandled method message: %s",
                method,
                source_module=self.__class__.__name__
            )

    async def _handle_channel_message(self, data: dict) -> None:
        """Handle channel-type messages (e.g., market data).

        Args
        ----
            data: The message data
        """
        channel = data["channel"]
        msg_type = data.get("type")

        try:
            if channel == "status":
                await self._handle_status_update(data)
            elif channel == "heartbeat":
                self._last_heartbeat_received_time = datetime.now(timezone.utc)
                self.logger.debug("Received heartbeat.", source_module=self.__class__.__name__)
            elif channel == "book" and msg_type in ["snapshot", "update"]:
                await self._handle_book_data(data)
            elif channel == "ohlc" and msg_type in ["snapshot", "update"]:
                # Explicitly cast data to Dict[str, Any] to satisfy mypy
                typed_data: dict[str, Any] = data
                await self._handle_book_data(typed_data)  # Reuse book data handler for now
            # Kraken trade messages are typically of type 'update'
            elif channel == "trade" and msg_type == "update":
                await self._handle_book_data(data)  # Reuse book data handler for now
            else:
                self.logger.warning(
                    "Unknown channel or message type.",
                    source_module=self.__class__.__name__,
                    context={"channel": channel, "type": msg_type}
                )
        except Exception as error:
            self.logger.error( # noqa: G201 - LoggerService uses .error with exc_info
                "Error handling channel message.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"channel": channel, "error": str(error)}
            )

    async def _handle_subscribe_ack(self, data: dict) -> None:
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
                    source_module=self.__class__.__name__,
                )
                # Can potentially update self._subscriptions based on acks if
                # needed
            else:
                error_msg = (
                    f"{log_prefix}FAILED! Error: {error}, Channel={channel}, Symbol={symbol}"
                )
                self.logger.error(
                    error_msg,
                    source_module=self.__class__.__name__,
                )
                # Consider specific error handling (e.g., stop if critical
                # subscription fails)

        except Exception as e:
            self.logger.error( # noqa: G201 - LoggerService uses .error with exc_info
                "Error processing subscription ack.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"data": str(data), "error": str(e)}
            )

    async def _handle_status_update(self, data: dict) -> None:
        """Handle connection or system status updates."""
        status = data.get("status")
        connection_id = data.get("connectionID")
        version = data.get("version")

        log_payload = {"status": status, "connection_id": connection_id, "version": version}

        if status == "online":
            self.logger.info(
                "WebSocket connection is online.",
                source_module=self._source_module,
                context=log_payload,
            )
            # Potentially publish a system state event if needed
        elif status == "error":
            error_msg = data.get("message", "Unknown error")
            self.logger.error(
                "WebSocket status error.",
                source_module=self._source_module,
                context={**log_payload, "error_message": error_msg} # Merge contexts
            )
            # Trigger reconnection or halt?
        else:
            self.logger.info(
                "Received status update.",
                source_module=self._source_module,
                context={**log_payload, "status_value": status} # Merge contexts
            )

        # Example: Publish SystemStateEvent (adjust fields as needed based on
        # core.events)
        event = SystemStateEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            # Map Kraken status to internal state if needed
            new_state=status if status is not None else "unknown",
            reason=f"Kraken WS Status Update: {status}",
        )
        try:
            await self.pubsub.publish(event)
            self.logger.debug("Published SystemStateEvent.", context={"status": status})
        except Exception as e:
            self.logger.error( # noqa: G201 - LoggerService uses .error with exc_info
                "Failed to publish SystemStateEvent.",
                exc_info=True,
                context={"error": str(e)}
            )

    async def _handle_book_data(self, data: dict) -> bool:
        """Handle book data message.

        Args
        ----
            data: The book data message

        Returns
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
                    book_state, book_item, symbol, received_checksum, is_snapshot
                )

                if not valid_after_apply:
                    continue

                # Publish the update
                await self._publish_book_event(symbol, book_state, is_snapshot, update_timestamp)

            except Exception as error:
                self.logger.error( # noqa: G201 - LoggerService uses .error with exc_info
                    "Error processing book data.",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                    context={"error": str(error)}
                )
                return False

        return processed_ok

    def _validate_book_message(self, data: dict) -> bool:
        """Validate the overall book message structure.

        Args
        ----
            data: The book data message

        Returns
        -------
            bool: True if valid, False otherwise
        """
        if not isinstance(data.get("data"), list):
            self.logger.warning(
                "Invalid book message: 'data' is not a list.",
                source_module=self._source_module,
                context={"message_snippet": str(data)[:200]}
            )
            return False

        msg_type = data.get("type")
        if msg_type not in ["snapshot", "update"]:
            self.logger.warning(
                "Invalid book message type.",
                source_module=self._source_module,
                context={"type": msg_type, "message_snippet": str(data)[:200]}
            )
            return False

        return True

    def _validate_book_item(self, book_item: dict) -> bool:
        """Validate an individual book item.

        Args
        ----
            book_item: The book item to validate

        Returns
        -------
            bool: True if valid, False otherwise
        """
        if not isinstance(book_item, dict):
            self.logger.warning(
                "Invalid book item: not a dict.",
                source_module=self._source_module,
                context={"item_snippet": str(book_item)[:200]}
            )
            return False

        symbol = book_item.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            self.logger.warning(
                "Book item missing/invalid symbol.",
                source_module=self._source_module,
                context={"item_snippet": str(book_item)[:200]}
            )
            return False

        # Validate bids/asks structure (list of dicts with price/qty)
        for side_key in ["asks", "bids"]:
            side_data = book_item.get(side_key)
            if side_data is not None:  # It's okay if a side is missing in an update
                if not isinstance(side_data, list):
                    self.logger.warning(
                        "Book item side is not a list.",
                        source_module=self._source_module,
                        context={
                            "side_key": side_key,
                            "symbol": symbol,
                            "item_snippet": str(book_item)[:200]
                        }
                    )
                    return False

                for level in side_data:
                    if not isinstance(level, dict) or "price" not in level or "qty" not in level:
                        self.logger.warning(
                            "Invalid level in book side.",
                            source_module=self._source_module,
                            context={
                                "side_key": side_key,
                                "symbol": symbol,
                                "level_snippet": str(level)[:100]
                            }
                        )
                        return False

        return True

    async def _process_book_item(
        self,
        book_state: dict,
        book_item: dict,
        symbol: str,
        received_checksum: Optional[int],
        is_snapshot: bool,
    ) -> bool:
        """Process a validated book item.

        Args
        ----
            book_state: Current book state to update
            book_item: Book data from exchange
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange
            is_snapshot: Whether this is a snapshot or update

        Returns
        -------
            bool: True if processing was successful, False otherwise
        """
        try:
            if is_snapshot:
                valid_after_apply = self._apply_book_snapshot(
                    book_state, book_item, symbol, received_checksum
                )
            else:
                valid_after_apply = self._apply_book_update(book_state, book_item, symbol)

            # Truncate book to subscribed depth
            valid_after_apply = self._truncate_book_to_depth(book_state, symbol, valid_after_apply)

            # Validate checksum and update book state
            return self._validate_and_update_checksum(
                book_state, symbol, received_checksum, valid_after_apply
            )

        except Exception as e:
            self.logger.error(  # noqa: G201 - LoggerService uses .error with exc_info
                "Error in book item processing.",
                source_module=self.__class__.__name__,
                exc_info=True,
                context={"error": str(e)}
            )
            return False

    def _apply_book_snapshot(
        self, book_state: dict, book_item: dict, symbol: str, received_checksum: Optional[int]
    ) -> bool:
        """Apply a book snapshot to the book state.

        Args
        ----
            book_state: Current book state to update
            book_item: Snapshot data from exchange
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange

        Returns
        -------
            bool: True if snapshot was applied successfully
        """
        book_state["asks"].clear()
        book_state["bids"].clear()

        # Process asks
        for level in book_item.get("asks", []):
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) > self._MIN_QTY_THRESHOLD:
                book_state["asks"][price_str] = qty_str

        # Process bids
        for level in book_item.get("bids", []):
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) > self._MIN_QTY_THRESHOLD:
                book_state["bids"][price_str] = qty_str

        self.logger.info(
            "Processed L2 snapshot.",
            source_module=self.__class__.__name__,
            context={
                "symbol": symbol,
                "ask_levels": len(book_state["asks"]),
                "bid_levels": len(book_state["bids"])
            }
        )
        book_state["checksum"] = received_checksum
        return True

    def _apply_book_update(self, book_state: dict, book_item: dict, symbol: str) -> bool:
        """Apply book updates to the book state.

        Args
        ----
            book_state: Current book state to update
            book_item: Update data from exchange
            symbol: Trading pair symbol

        Returns
        -------
            bool: True if any updates were applied
        """
        asks_updated = self._update_price_levels(
            book_state["asks"], book_item.get("asks", []), symbol, "ask"
        )
        bids_updated = self._update_price_levels(
            book_state["bids"], book_item.get("bids", []), symbol, "bid"
        )

        valid_after_apply = asks_updated or bids_updated
        if not valid_after_apply:
            self.logger.debug(
                "Received book update with no effective changes.",
                source_module=self.__class__.__name__,
                context={"symbol": symbol}
            )
            return True  # Treat no change as valid for checksum check

        return valid_after_apply

    def _update_price_levels(
        self, book_side: SortedDict, updates: list, symbol: str, side: str
    ) -> bool:
        """Update price levels for one side of the book.

        Args
        ----
            book_side: SortedDict containing the side's price levels
            updates: List of updates from exchange
            symbol: Trading pair symbol
            side: Side being updated ("bid" or "ask")

        Returns
        -------
            bool: True if any updates were applied
        """
        updated = False
        for level in updates:
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) < self._MIN_QTY_THRESHOLD:  # Remove level
                if book_side.pop(price_str, None):
                    updated = True
                    self.logger.debug(
                        "Removed level.",
                        source_module=self.__class__.__name__,
                        context={"side": side, "price": price_str, "symbol": symbol}
                    )
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
                    },
                )
        return updated

    def _truncate_book_to_depth(
        self, book_state: dict, symbol: str, valid_after_apply: bool
    ) -> bool:
        """Truncate book to subscribed depth.

        Args
        ----
            book_state: Current book state to truncate
            symbol: Trading pair symbol
            valid_after_apply: Current validity state

        Returns
        -------
            bool: Updated validity state
        """
        # Bids: Remove lowest price bids if count > depth
        while len(book_state["bids"]) > self._book_depth:
            removed_bid = book_state["bids"].popitem(0)
            self.logger.debug(
                "Truncated bid level due to depth limit.",
                source_module=self.__class__.__name__,
                context={"level": removed_bid[0], "symbol": symbol}
            )
            valid_after_apply = True

        # Asks: Remove highest price asks if count > depth
        while len(book_state["asks"]) > self._book_depth:
            removed_ask = book_state["asks"].popitem(-1)
            self.logger.debug(
                "Truncated ask level due to depth limit.",
                source_module=self.__class__.__name__,
                context={"level": removed_ask[0], "symbol": symbol}
            )
            valid_after_apply = True

        return valid_after_apply

    def _calculate_book_checksum(self, book_state: dict) -> Optional[int]:
        """Calculate the checksum for the order book.
        
        Args:
            book_state: The current book state containing bids and asks
            
        Returns
        -------
            The calculated checksum or None if calculation fails
        """
        try:
            # This is a simplified implementation - the actual algorithm would depend on
            # the specific exchange's checksum calculation method
            checksum_str = ""

            # Process bids (sort in descending order - highest bid first)
            bids = book_state["bids"]
            for price in list(bids.keys())[:10]:  # Use top 10 levels
                qty = bids[price]
                checksum_str += f"{float(price):.8f}:{float(qty):.8f}:"

            # Process asks (sort in ascending order - lowest ask first)
            asks = book_state["asks"]
            for price in list(asks.keys())[:10]:  # Use top 10 levels
                qty = asks[price]
                checksum_str += f"{float(price):.8f}:{float(qty):.8f}:"

            # Use CRC32 algorithm to generate checksum
            import zlib
            return zlib.crc32(checksum_str.encode())
        except Exception as e:
            self.logger.error(
                "Error calculating checksum",
                source_module=self._source_module,
                exc_info=False,
                context={"error": str(e)}
            )
            return None

    def _validate_and_update_checksum(
        self,
        book_state: dict,
        symbol: str,
        received_checksum: Optional[int],
        valid_after_apply: bool,
    ) -> bool:
        """Validate and update book checksum.

        Args
        ----
            book_state: Current book state
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange
            valid_after_apply: Whether updates were applied

        Returns
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
                    book_state, symbol, local_checksum, received_checksum
                )
                return False

            book_state["checksum"] = received_checksum
            self.logger.debug(
                "Checksum validation passed.",
                source_module=self.__class__.__name__,
                context={"symbol": symbol}
            )
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
        book_state: dict,
        symbol: str,
        local_checksum: int,
        received_checksum: int,
    ) -> None:
        """Handle checksum mismatch case.

        Args
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
            },
        )
        self.logger.error(
            "Bids (Top 3).",
            source_module=self.__class__.__name__,
            context={"bids_top_3": list(reversed(book_state["bids"].items()))[:3]}
        )
        self.logger.error(
            "Asks (Top 3).",
            source_module=self.__class__.__name__,
            context={"asks_top_3": list(book_state["asks"].items())[:3]}
        )
        book_state["checksum"] = None

    def _handle_no_updates_case(
        self: "DataIngestor", book_state: dict, symbol: str, received_checksum: Optional[int]
    ) -> bool:
        """Handle case where no updates were applied.

        Args
        ----
            book_state: Current book state
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange

        Returns
        -------
            bool: True if state is valid
        """
        if book_state["checksum"] == received_checksum:
            self.logger.debug(
                "No book changes, checksum still matches.",
                source_module=self.__class__.__name__,
                context={"symbol": symbol, "checksum": received_checksum}
            )
            return True

        if received_checksum is not None:
            self.logger.warning(
                "Book checksum changed but no updates applied locally. Recalculating.",
                source_module=self.__class__.__name__,
                context={
                    "symbol": symbol,
                    "previous_checksum": book_state["checksum"],
                    "new_checksum": received_checksum,
                },
            )
            local_checksum = self._calculate_book_checksum(book_state)
            if local_checksum == received_checksum:
                self.logger.info(
                    "Recalculated checksum matches received.",
                    source_module=self.__class__.__name__,
                    context={
                        "symbol": symbol,
                        "checksum": local_checksum,
                    },
                )
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
        book_state: dict,
        is_snapshot: bool,
        update_timestamp: Optional[str],  # Timestamp from message if available
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
                    exchange_ts = datetime.fromtimestamp(float(update_timestamp), tz=timezone.utc)
                except (ValueError, TypeError):
                    self.logger.warning(
                        "Could not parse book update timestamp.",
                        context={"timestamp": update_timestamp}
                    )

            event = MarketDataL2Event(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),  # Event creation time
                trading_pair=symbol,
                exchange="kraken",  # Hardcoded for now
                bids=bids_list,
                asks=asks_list,
                is_snapshot=is_snapshot,
                timestamp_exchange=exchange_ts,  # Timestamp from Kraken message
            )
            await self.pubsub.publish(event)
        except Exception:
            self.logger.exception(
                "Unexpected error publishing OHLCV event.",
                source_module=self._source_module
            )

def _handle_checksum_mismatch(
    self: "DataIngestor",
    book_state: dict,
    symbol: str,
    local_checksum: int,
    received_checksum: int,
) -> None:
    """Handle checksum mismatch case.

    Args
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
        },
    )
    self.logger.error(
        "Bids (Top 3).",
        source_module=self.__class__.__name__,
        context={"bids_top_3": list(reversed(book_state["bids"].items()))[:3]}
    )
    self.logger.error(
        "Asks (Top 3).",
        source_module=self.__class__.__name__,
        context={"asks_top_3": list(book_state["asks"].items())[:3]}
    )
    book_state["checksum"] = None

def _handle_no_updates_case(
    self: "DataIngestor",
    book_state: dict,
    symbol: str,
    received_checksum: Optional[int],
) -> bool:
    """Handle case where no updates were applied.

    Args
    ----
        book_state: Current book state
        symbol: Trading pair symbol
        received_checksum: Checksum from exchange

    Returns
    -------
        bool: True if state is valid
    """
    if book_state["checksum"] == received_checksum:
        self.logger.debug(
            "No book changes, checksum still matches.",
            source_module=self.__class__.__name__,
            context={"symbol": symbol, "checksum": received_checksum}
        )
        return True

    if received_checksum is not None:
        self.logger.warning(
            "Book checksum changed but no updates applied locally. Recalculating.",
            source_module=self.__class__.__name__,
            context={
                "symbol": symbol,
                "previous_checksum": book_state["checksum"],
                "new_checksum": received_checksum,
            },
        )
        local_checksum = self._calculate_book_checksum(book_state)
        if local_checksum == received_checksum:
            self.logger.info(
                "Recalculated checksum matches received.",
                source_module=self.__class__.__name__,
                context={
                    "symbol": symbol,
                    "checksum": local_checksum,
                },
            )
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
    book_state: dict,
    is_snapshot: bool,
    update_timestamp: Optional[str],  # Timestamp from message if available
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
                exchange_ts = datetime.fromtimestamp(float(update_timestamp), tz=timezone.utc)
            except (ValueError, TypeError):
                self.logger.warning(
                    "Invalid timestamp format: %s",
                    update_timestamp,
                    source_module=self._source_module,
                    context={"timestamp": str(update_timestamp)}
                )
                exchange_ts = datetime.utcnow()

        # Create and publish the market data event
        event_id = uuid.uuid4()
        event = MarketDataL2Event(
            event_id=event_id,
            source_module=self._source_module,
            timestamp=datetime.utcnow(),  # Event creation time
            trading_pair=symbol,
            exchange="kraken",
            bids=bids_list,
            asks=asks_list,
            is_snapshot=is_snapshot,
            timestamp_exchange=exchange_ts,
        )
        await self.pubsub.publish(event)
    except Exception:
        self.logger.exception(
            "Unexpected error publishing book event",
            source_module=self._source_module,
            context={"symbol": symbol, "is_snapshot": is_snapshot}
        )

    async def _reconnect_with_backoff(self: "DataIngestor") -> bool:
        """Attempt reconnection with exponential backoff and jitter.

        Returns
        -------
            bool: True if reconnection was successful, False if max retries exceeded
        """
        retry_count = 0
        # Get reconnection parameters (could be from config in the future)
        max_retries = 5  # self._config.get("websocket.max_retries", 5)
        base_delay = 2.0  # self._config.get("websocket.base_delay_seconds", 2.0)
        max_delay = 60.0  # self._config.get("websocket.max_delay_seconds", 60.0)

        while self._is_running and retry_count < max_retries:
            retry_count += 1
            delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
            total_delay = delay + jitter

            self.logger.warning(
                "WebSocket disconnected. Attempting reconnect.",
                source_module=self.__class__.__name__,
                context={
                    "attempt": f"{retry_count}/{max_retries}",
                    "delay_seconds": f"{total_delay:.2f}",
                },
            )
            await asyncio.sleep(total_delay)

            if not self._is_running:
                break  # Check if stop was called during sleep

            # Try to establish connection again
            if await self._establish_connection():
                # If successful, try to setup (subscribe, etc.)
                subscription_msg = self._build_subscription_message()
                if subscription_msg and await self._setup_connection(subscription_msg):
                    self.logger.info(
                        "WebSocket reconnected and setup successfully.",
                        source_module=self.__class__.__name__,
                    )
                    return True  # Reconnect successful
                await self._cleanup_connection()  # Setup failed

        self.logger.error(
            "Failed to reconnect WebSocket after attempts. Stopping.",
            source_module=self.__class__.__name__,
            context={"max_retries": max_retries},
        )
        self._is_running = False  # Stop the main loop
        return False


# Example Usage (for testing purposes - requires libraries installed)
async def main() -> None:
    """Execute the main standalone testing function."""
    await _setup_logging()
    config = _create_test_config()
    event_bus: asyncio.Queue[Any] = asyncio.Queue()
    logger_service = MockLoggerService()

    try:
        await _run_test(config, event_bus, logger_service)
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        print("Exiting...")


def _create_test_config() -> dict:
    """Create test configuration.

    Returns
    -------
        dict: Test configuration
    """
    return {
        "trading_pairs": ["XRP/USD", "DOGE/USD"],
        "book_depth": 10,
        "ohlc_intervals": [1, 5],  # 1m and 5m
        "reconnect_delay_s": 5,
        "connection_timeout_s": 15,
    }


async def _setup_logging() -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def _run_test(
    config: dict, event_bus: asyncio.Queue, logger_service: "LoggerService[Any]"
) -> None:
    """Run the test with the given configuration.

    Args
    ----
        config: Test configuration
        event_bus: Event bus for communication
        logger_service: Logger service instance
    """
    from .core.pubsub import PubSubManager

    logger = logging.getLogger("test")
    # Create a ConfigManager-compatible dictionary wrapper first
    from .config_manager import ConfigManager

    class TestConfigManager(ConfigManager):
        def __init__(self, config_dict: dict[str, Any]) -> None:
            self._config = config_dict

        def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
            if self._config is None:
                return default
            return self._config.get(key, default)

    config_manager = TestConfigManager(config)

    # Create PubSubManager with the config_manager parameter
    pubsub = PubSubManager(logger, config_manager)

    # Create and start the data ingestor with proper types
    data_ingestor = DataIngestor(config_manager, pubsub, logger_service)

    # Create a separate event consumer for testing
    test_consumer_task = asyncio.create_task(_run_event_consumer(event_bus))

    try:
        await data_ingestor.start()
    except asyncio.CancelledError:
        print("Test cancelled.")
    finally:
        await data_ingestor.stop()
        test_consumer_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await test_consumer_task


async def _run_event_consumer(event_bus: asyncio.Queue) -> None:
    """Run the event consumer.

    Args
    ----
        event_bus: Event bus to consume from
    """
    while True:
        try:
            event = await event_bus.get()
            print(f"Received event: {event}")
            event_bus.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in consumer: {e}")
            break


class MockLoggerService(LoggerService[Any]):
    """Provide a mock logger service for testing."""

    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        pubsub_manager: Optional["PubSubManager"] = None,
    ) -> None:
        """Initialize the mock logger service."""

    def log(
        self,
        level: int,
        message: str,
        *args: Any,
        source_module: Optional[str] = None,
        context: Optional[dict[Any, Any]] = None,
        exc_info: Optional[Union[bool, tuple[type[BaseException], BaseException, TracebackType], BaseException]] = None,
    ) -> None:
        """Log a message."""
        level_name = {50: "CRITICAL", 40: "ERROR", 30: "WARNING", 20: "INFO", 10: "DEBUG"}.get(
            level, "UNKNOWN"
        )
        print(f"[{level_name}] {message}")

    def debug(
        self,
        message: str,
        *args: Any,
        source_module: Optional[str] = None,
        context: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Log a debug message."""
        print(f"[DEBUG] {message}")

    def info(
        self,
        message: str,
        *args: Any,
        source_module: Optional[str] = None,
        context: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Log an info message."""
        print(f"[INFO] {message}")

    def warning(
        self,
        message: str,
        *args: Any,
        source_module: Optional[str] = None,
        context: Optional[dict[Any, Any]] = None,
    ) -> None:
        """Log a warning message."""
        print(f"[WARNING] {message}")

    def error(
        self,
        message: str,
        *args: Any,
        source_module: Optional[str] = None,
        context: Optional[dict[Any, Any]] = None,
        exc_info: Optional[Union[bool, tuple[type[BaseException], BaseException, TracebackType], BaseException]] = None,
    ) -> None:
        """Log an error message."""
        print(f"[ERROR] {message}")

    def critical(
        self,
        message: str,
        *args: Any,
        source_module: Optional[str] = None,
        context: Optional[dict[Any, Any]] = None,
        exc_info: Optional[Union[bool, tuple[type[BaseException], BaseException, TracebackType], BaseException]] = None,
    ) -> None:
        """Log a critical message."""
        print(f"[CRITICAL] {message}")


if __name__ == "__main__":
    asyncio.run(main())
