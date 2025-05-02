# src/gal_friday/data_ingestor.py

import asyncio
import websockets
import json
import uuid
import binascii  # For CRC32 checksum
import logging
from collections import defaultdict
from sortedcontainers import SortedDict
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any, Union, cast
# Import ClientProtocol for type checking but use Any for the actual connection
from websockets.client import ClientProtocol  # For type annotation only

# Import necessary event classes from core module
from .core.events import (
    Event, 
    EventType, 
    MarketDataL2Event, 
    MarketDataOHLCVEvent, 
    SystemStateEvent 
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
    """Payload for L2 market data"""
    trading_pair: str
    exchange: str
    timestamp_exchange: Optional[str] = None  # Timestamp from the book update message
    bids: List[Tuple[str, str]] = field(default_factory=list)  # [(price_str, volume_str), ...] sorted desc
    asks: List[Tuple[str, str]] = field(default_factory=list)  # [(price_str, volume_str), ...] sorted asc
    is_snapshot: bool = False
    checksum: Optional[int] = None  # Add checksum to event for potential downstream validation


@dataclass
class MarketDataOHLCVPayload:
    """Payload for OHLCV market data"""
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
    """Payload for system status updates"""
    system_status: str  # e.g., "online", "cancel_only"
    connection_id: Optional[int] = None


# --- DataIngestor Class ---

class DataIngestor:
    """
    Connects to Kraken WebSocket API v2, manages subscriptions for L2 book
    and OHLCV data, parses messages, maintains L2 order book state with
    checksum validation, and publishes standardized market data events.
    """

    KRAKEN_WS_URL = "wss://ws.kraken.com/v2"
    # Map Kraken interval integers to readable strings
    INTERVAL_MAP = {
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
    INTERVAL_INT_MAP = {v: k for k, v in INTERVAL_MAP.items()}

    def __init__(
        self, 
        config: "ConfigManager", 
        pubsub_manager: "PubSubManager", 
        logger_service: LoggerService
    ):
        """Initialize the DataIngestor.

        Args:
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

        # Initialize state
        self._connection: Optional[Any] = None
        self._is_running: bool = False
        self._is_stopping: bool = False
        self._last_message_received_time: Optional[datetime] = None
        self._liveness_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._connection_id: Optional[int] = None
        self._system_status: Optional[str] = None

        # Initialize book state
        self._l2_books: Dict[str, Dict[str, Union[SortedDict, Optional[int]]]] = defaultdict(
            lambda: {"bids": SortedDict(), "asks": SortedDict(), "checksum": None}
        )

        # Validate configuration
        self._validate_initial_config()

    def _validate_initial_config(self) -> None:
        """Validate initial configuration settings."""
        if not self._trading_pairs:
            raise ValueError("DataIngestor: 'trading_pairs' configuration cannot be empty.")

        if self._book_depth not in [0, 10, 25, 100, 500, 1000]:
            self.logger.warning(
                f"Invalid book_depth {self._book_depth}, defaulting to 10.",
                source_module=self._source_module,
            )
            self._book_depth = 10

        valid_intervals = [i for i in self._ohlc_intervals if i in self.INTERVAL_MAP]
        if len(valid_intervals) != len(self._ohlc_intervals):
            invalid_intervals = set(self._ohlc_intervals) - set(valid_intervals)
            self.logger.warning(
                f"Invalid ohlc_intervals found: {invalid_intervals}. "
                f"Using only valid intervals: {valid_intervals}",
                source_module=self._source_module,
            )
        self._ohlc_intervals = valid_intervals

    def _build_subscription_message(self) -> Optional[str]:
        """Build the Kraken WebSocket v2 subscription message.

        Returns:
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

        if not subscriptions_params:
            self.logger.error(
                "No valid subscriptions configured.", source_module=self.__class__.__name__
            )
            return None

        try:
            return json.dumps({"method": "subscribe", "params": subscriptions_params})
        except Exception as error:
            self.logger.error(
                f"Error building subscription message: {error}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            return None

    async def start(self) -> None:
        """Starts the data ingestion process with reconnection logic."""
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
            try:
                if not await self._establish_connection():
                    continue

                if not await self._setup_connection(subscription_msg):
                    continue

                # Listen loop
                await self._message_listen_loop()

            except Exception as e:
                self._handle_connection_error(e)

            finally:
                await self._cleanup_connection()

            if self._is_running:
                # Prevent rapid spin if connect fails instantly
                await asyncio.sleep(self._reconnect_delay)

        self.logger.info("Data Ingestor stopped.", source_module=self.__class__.__name__)

    async def _establish_connection(self) -> bool:
        """Establish WebSocket connection.

        Returns:
            bool: True if connection was established successfully
        """
        self.logger.info(
            f"Attempting to connect to {self._websocket_url}...",
            source_module=self.__class__.__name__,
        )
        try:
            # Add timeout to connect attempt itself
            async with asyncio.timeout(self._connection_timeout + 5):
                # Connect to the WebSocket
                self._connection = await websockets.connect(self._websocket_url)
                self._last_message_received_time = datetime.now(timezone.utc)
                self.logger.info("WebSocket connected.", source_module=self.__class__.__name__)
                return True
        except TimeoutError:
            self.logger.warning(
                f"Connection attempt timed out after {self._connection_timeout + 5}s. "
                "Retrying...",
                source_module=self.__class__.__name__,
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
                    f"WebSocket connection error: {e}. "
                    f"Reconnecting in {self._reconnect_delay}s...",
                    source_module=self.__class__.__name__,
                )
            return False

    async def _setup_connection(self, subscription_msg: str) -> bool:
        """Set up the connection by starting liveness monitor and subscribing.

        Args:
            subscription_msg: The subscription message to send

        Returns:
            bool: True if setup was successful
        """
        try:
            # Start background liveness check
            await self._start_liveness_monitor()

            # Subscribe
            if self._connection is not None:
                await self._connection.send(subscription_msg)
                self.logger.info("Sent subscription request.", source_module=self.__class__.__name__)
                self.logger.debug(
                    f"Subscription message: {subscription_msg}", source_module=self.__class__.__name__
                )
                return True
            return False
        except Exception as e:
            self.logger.error(
                f"Error during connection setup: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            return False

    async def _start_liveness_monitor(self) -> None:
        """Start the liveness monitor task."""
        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
            try:
                await self._liveness_task
            except asyncio.CancelledError:
                pass
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
                error_msg = f"Error processing message: {error}"
                self.logger.error(error_msg, source_module=self.__class__.__name__, exc_info=True)

    def _handle_connection_error(self, error: Exception) -> None:
        """Handle connection errors.

        Args:
            error: The error that occurred
        """
        self.logger.error(
            f"Unexpected error in Data Ingestor loop: {error}. "
            f"Reconnecting in {self._reconnect_delay}s...",
            source_module=self.__class__.__name__,
            exc_info=True,
        )

    async def stop(self) -> None:
        """Stops the data ingestion process gracefully."""
        self._is_running = False
        self._is_stopping = True
        self.logger.info("Stopping Data Ingestor...", source_module=self.__class__.__name__)
        await self._cleanup_connection()

    async def _cleanup_connection(self) -> None:
        """Cleans up connection resources."""
        if self._liveness_task and not self._liveness_task.done():
            self._liveness_task.cancel()
            try:
                await self._liveness_task
            except asyncio.CancelledError:
                pass  # Expected
            self._liveness_task = None
            self.logger.debug(
                "Liveness monitor task cancelled.", source_module=self.__class__.__name__
            )

        if self._connection and not self._connection.closed:
            try:
                await self._connection.close()
                self.logger.info(
                    "WebSocket connection closed.", source_module=self.__class__.__name__
                )
            except Exception as e:
                self.logger.warning(
                    f"Error closing WebSocket connection: {e}",
                    source_module=self.__class__.__name__,
                )

        self._connection = None
        self._connection_id = None  # Reset connection specific state

    async def _monitor_connection_liveness_loop(self) -> None:
        """Periodically checks if messages are being received."""
        monitor_msg = (
            f"Starting connection liveness monitor " f"(timeout: {self._connection_timeout}s)..."
        )
        self.logger.info(monitor_msg, source_module=self.__class__.__name__)
        check_interval = max(1, self._connection_timeout / 2)

        while self._is_running and self._connection and not self._connection.closed:
            # Check if task was cancelled externally (e.g., during shutdown)
            current_task = asyncio.current_task()
            if current_task and current_task.cancelled():
                break
            await asyncio.sleep(check_interval)
            if self._last_message_received_time:
                time_since_last = datetime.now(timezone.utc) - self._last_message_received_time
                if time_since_last > timedelta(seconds=self._connection_timeout):
                    timeout_msg = (
                        f"No messages received for {time_since_last.total_seconds():.1f}s "
                        f"(> {self._connection_timeout}s timeout). "
                        "Assuming connection loss, triggering reconnect."
                    )
                    self.logger.warning(timeout_msg, source_module=self.__class__.__name__)
                    # Trigger reconnect by cleanly closing the current connection
                    if self._connection and not self._connection.closed:
                        asyncio.create_task(self._cleanup_connection())
                    break  # Exit loop
            else:
                # Can happen briefly between connect and first message
                self.logger.debug(
                    "Liveness check running but _last_message_received_time " "is not yet set.",
                    source_module=self.__class__.__name__,
                )

        self.logger.info(
            "Connection liveness monitor stopped.", source_module=self.__class__.__name__
        )

    async def _process_message(self, message: Union[str, bytes]) -> None:
        """Parse and route incoming WebSocket messages."""
        # Kraken v2 sends JSON text frames
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        self.logger.debug(
            f"Received message: {message[:200]}...", source_module=self.__class__.__name__
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
                        f"Unknown message format: {message[:200]}...",
                        source_module=self.__class__.__name__,
                    )
        except Exception as e:
            self.logger.error(
                f"Error processing message: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    async def _handle_method_message(self, data: dict) -> None:
        """Handle method-type messages (e.g., subscription acknowledgments).

        Args:
            data: The message data
        """
        if data.get("method") == "subscribe":
            await self._handle_subscribe_ack(data)
        # Handle other method responses if needed

    async def _handle_channel_message(self, data: dict) -> None:
        """Handle channel-type messages (e.g., market data).

        Args:
            data: The message data
        """
        channel = data["channel"]
        msg_type = data.get("type")

        try:
            if channel == "status":
                await self._handle_status_update(data)
            elif channel == "heartbeat":
                self.logger.debug("Received heartbeat.", source_module=self.__class__.__name__)
            elif channel == "book" and msg_type in ["snapshot", "update"]:
                await self._handle_book_data(data)
            elif channel == "ohlc" and msg_type in ["snapshot", "update"]:
                # Explicitly cast data to Dict[str, Any] to satisfy mypy
                typed_data: Dict[str, Any] = data
                await self._handle_ohlc_data(typed_data)
            else:
                self.logger.warning(
                    f"Unknown channel or message type: {channel} - {msg_type}",
                    source_module=self.__class__.__name__,
                )
        except Exception as error:
            self.logger.error(
                f"Error handling {channel} message: {error}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    async def _handle_subscribe_ack(self, data: dict) -> None:
        """Handles the acknowledgment message for subscriptions."""
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
                success_msg = f"{log_prefix}Success for Channel={channel}, " f"Symbol={symbol}"
                self.logger.info(
                    success_msg,
                    source_module=self.__class__.__name__,
                )
                # Can potentially update self._subscriptions based on acks if needed
            else:
                error_msg = (
                    f"{log_prefix}FAILED! Error: {error}, " f"Channel={channel}, Symbol={symbol}"
                )
                self.logger.error(
                    error_msg,
                    source_module=self.__class__.__name__,
                )
                # Consider specific error handling (e.g., stop if critical subscription fails)

        except Exception as e:
            self.logger.error(
                f"Error processing subscription ack: {data} - {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    async def _handle_status_update(self, data: dict) -> None:
        """Handles connection or system status updates."""
        status = data.get("status")
        connection_id = data.get("connectionID")
        version = data.get("version")
        
        log_payload = {
            "status": status,
            "connection_id": connection_id,
            "version": version
        }

        if status == "online":
            self.logger.info("WebSocket connection is online.", source_module=self._source_module, context=log_payload)
            # Potentially publish a system state event if needed
        elif status == "error":
            error_msg = data.get("message", "Unknown error")
            self.logger.error(f"WebSocket status error: {error_msg}", source_module=self._source_module, context=log_payload)
            # Trigger reconnection or halt?
        else:
            self.logger.info(f"Received status update: {status}", source_module=self._source_module, context=log_payload)
            
        # Example: Publish SystemStateEvent (adjust fields as needed based on core.events)
        event = SystemStateEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            new_state=status if status is not None else "unknown", # Map Kraken status to internal state if needed
            reason=f"Kraken WS Status Update: {status}",
        )
        try:
            await self.pubsub.publish(event)
            self.logger.debug(f"Published SystemStateEvent: {status}")
        except Exception as e:
             self.logger.error(f"Failed to publish SystemStateEvent: {e}", exc_info=True)

    async def _handle_book_data(self, data: dict) -> bool:
        """Handle book data message.

        Args:
            data: The book data message

        Returns:
            bool: True if successful, False otherwise
        """
        msg_type = data["type"]
        is_snapshot = msg_type == "snapshot"

        # Book data structure: data: [ { book_obj } ]
        for book_item in data.get("data", []):
            symbol = book_item.get("symbol")
            if not symbol:
                self.logger.warning(
                    f"Book data missing symbol: {book_item}", source_module=self._source_module
                )
                continue

            book_state = self._l2_books[symbol]
            received_checksum = book_item.get("checksum")
            update_timestamp = book_item.get("timestamp")

            try:
                if is_snapshot:
                    valid_after_apply = self._apply_book_snapshot(
                        book_state, book_item, symbol, received_checksum
                    )
                else:
                    valid_after_apply = self._apply_book_update(book_state, book_item, symbol)

                # Truncate book to subscribed depth
                valid_after_apply = self._truncate_book_to_depth(
                    book_state, symbol, valid_after_apply
                )

                # Validate checksum and update book state
                if not self._validate_and_update_checksum(
                    book_state, symbol, received_checksum, valid_after_apply
                ):
                    continue

                # Publish the book update event
                await self._publish_book_event(symbol, book_state, is_snapshot, update_timestamp)

            except Exception as error:
                error_msg = f"Error processing book data: {error}"
                self.logger.error(error_msg, source_module=self.__class__.__name__, exc_info=True)
                return False

        return True

    def _apply_book_snapshot(
        self, book_state: dict, book_item: dict, symbol: str, received_checksum: Optional[int]
    ) -> bool:
        """Apply a book snapshot to the book state.

        Args:
            book_state: Current book state to update
            book_item: Snapshot data from exchange
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange

        Returns:
            bool: True if snapshot was applied successfully
        """
        book_state["asks"].clear()
        book_state["bids"].clear()

        # Process asks
        for level in book_item.get("asks", []):
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) > 1e-12:
                book_state["asks"][price_str] = qty_str

        # Process bids
        for level in book_item.get("bids", []):
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) > 1e-12:
                book_state["bids"][price_str] = qty_str

        self.logger.info(
            f"Processed L2 snapshot for {symbol}. "
            f"Ask levels: {len(book_state['asks'])}, "
            f"Bid levels: {len(book_state['bids'])}",
            source_module=self.__class__.__name__,
        )
        book_state["checksum"] = received_checksum
        return True

    def _apply_book_update(self, book_state: dict, book_item: dict, symbol: str) -> bool:
        """Apply book updates to the book state.

        Args:
            book_state: Current book state to update
            book_item: Update data from exchange
            symbol: Trading pair symbol

        Returns:
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
                f"Received book update for {symbol} with no effective changes.",
                source_module=self.__class__.__name__,
            )
            return True  # Treat no change as valid for checksum check

        return valid_after_apply

    def _update_price_levels(
        self, book_side: SortedDict, updates: list, symbol: str, side: str
    ) -> bool:
        """Update price levels for one side of the book.

        Args:
            book_side: SortedDict containing the side's price levels
            updates: List of updates from exchange
            symbol: Trading pair symbol
            side: Side being updated ("bid" or "ask")

        Returns:
            bool: True if any updates were applied
        """
        updated = False
        for level in updates:
            price_str = str(level["price"])
            qty_str = str(level["qty"])
            if float(qty_str) < 1e-12:  # Remove level
                if book_side.pop(price_str, None):
                    updated = True
                    self.logger.debug(
                        f"Removed {side} level {price_str} for {symbol}",
                        source_module=self.__class__.__name__,
                    )
            else:  # Update level
                if book_side.get(price_str) != qty_str:
                    book_side[price_str] = qty_str
                    updated = True
                    self.logger.debug(
                        f"Updated {side} level {price_str} to {qty_str} for {symbol}",
                        source_module=self.__class__.__name__,
                    )
        return updated

    def _truncate_book_to_depth(
        self, book_state: dict, symbol: str, valid_after_apply: bool
    ) -> bool:
        """Truncate book to subscribed depth.

        Args:
            book_state: Current book state to truncate
            symbol: Trading pair symbol
            valid_after_apply: Current validity state

        Returns:
            bool: Updated validity state
        """
        # Bids: Remove lowest price bids if count > depth
        while len(book_state["bids"]) > self._book_depth:
            removed_bid = book_state["bids"].popitem(0)
            self.logger.debug(
                f"Truncated bid level {removed_bid[0]} for {symbol} due to depth limit.",
                source_module=self.__class__.__name__,
            )
            valid_after_apply = True

        # Asks: Remove highest price asks if count > depth
        while len(book_state["asks"]) > self._book_depth:
            removed_ask = book_state["asks"].popitem(-1)
            self.logger.debug(
                f"Truncated ask level {removed_ask[0]} for {symbol} due to depth limit.",
                source_module=self.__class__.__name__,
            )
            valid_after_apply = True

        return valid_after_apply

    def _validate_and_update_checksum(
        self, book_state: dict, symbol: str, received_checksum: Optional[int], valid_after_apply: bool
    ) -> bool:
        """Validate and update book checksum.

        Args:
            book_state: Current book state
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange
            valid_after_apply: Whether updates were applied

        Returns:
            bool: True if checksum is valid
        """
        if received_checksum is not None and valid_after_apply:
            local_checksum = self._calculate_book_checksum(book_state["bids"], book_state["asks"])
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
                f"Checksum validation passed for {symbol}.", source_module=self.__class__.__name__
            )
            return True

        elif not valid_after_apply:
            return self._handle_no_updates_case(book_state, symbol, received_checksum)

        elif received_checksum is None:
            warning_msg = f"No checksum received in book message for {symbol}. " "Cannot validate."
            self.logger.warning(warning_msg, source_module=self.__class__.__name__)
            return True

        return True

    def _handle_checksum_mismatch(
        self, book_state: dict, symbol: str, local_checksum: int, received_checksum: int
    ) -> None:
        """Handle checksum mismatch case.

        Args:
            book_state: Current book state
            symbol: Trading pair symbol
            local_checksum: Locally calculated checksum
            received_checksum: Checksum from exchange
        """
        self.logger.error(
            f"Checksum mismatch for {symbol}! "
            f"Local: {local_checksum}, Received: {received_checksum}. "
            "Book is potentially corrupt.",
            source_module=self.__class__.__name__,
        )
        self.logger.error(
            f"Bids (Top 3): {list(reversed(book_state['bids'].items()))[:3]}",
            source_module=self.__class__.__name__,
        )
        self.logger.error(
            f"Asks (Top 3): {list(book_state['asks'].items())[:3]}",
            source_module=self.__class__.__name__,
        )
        book_state["checksum"] = None

    def _handle_no_updates_case(
        self, book_state: dict, symbol: str, received_checksum: Optional[int]
    ) -> bool:
        """Handle case where no updates were applied.

        Args:
            book_state: Current book state
            symbol: Trading pair symbol
            received_checksum: Checksum from exchange

        Returns:
            bool: True if state is valid
        """
        if book_state["checksum"] == received_checksum:
            self.logger.debug(
                f"No book changes for {symbol}, checksum still matches {received_checksum}.",
                source_module=self.__class__.__name__,
            )
            return True

        if received_checksum is not None:
            self.logger.warning(
                f"Book checksum changed from {book_state['checksum']} to "
                f"{received_checksum} for {symbol} but no updates were "
                "applied locally? Recalculating.",
                source_module=self.__class__.__name__,
            )
            local_checksum = self._calculate_book_checksum(book_state["bids"], book_state["asks"])
            if local_checksum == received_checksum:
                self.logger.info(
                    f"Recalculated checksum {local_checksum} matches received "
                    f"{received_checksum} for {symbol}.",
                    source_module=self.__class__.__name__,
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
        self,
        symbol: str,
        book_state: dict,
        is_snapshot: bool,
        update_timestamp: Optional[str], # Timestamp from message if available
    ) -> None:
        """Creates and publishes a MarketDataL2Event."""
        try:
            # Extract bids and asks from the book_state (SortedDict)
            # Ensure they are formatted as List[Tuple[str, str]]
            bids_list = [ (str(price), str(vol)) for price, vol in book_state['bids'].items() ]
            asks_list = [ (str(price), str(vol)) for price, vol in book_state['asks'].items() ]

            # Parse the exchange timestamp if provided
            exchange_ts = None
            if update_timestamp:
                try:
                    # Kraken v2 uses float seconds with nanosecond precision
                    exchange_ts = datetime.fromtimestamp(float(update_timestamp), tz=timezone.utc)
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse book update timestamp: {update_timestamp}")

            event = MarketDataL2Event(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(), # Event creation time
                trading_pair=symbol,
                exchange="kraken", # Hardcoded for now
                bids=bids_list,
                asks=asks_list,
                is_snapshot=is_snapshot,
                timestamp_exchange=exchange_ts # Timestamp from Kraken message
            )
            await self.pubsub.publish(event)
            # Optional: Log after successful publish
            # self.logger.debug(f"Published L2 book event for {symbol}")
        except Exception as e:
            self.logger.error(
                f"Error creating/publishing L2 book event for {symbol}: {e}",
                source_module=self._source_module,
                exc_info=True,
            )

    def _format_level_value(self, value_str: str) -> str:
        """Formats price or quantity string for checksum calculation (Kraken v2 spec)."""
        # Remove decimal point
        formatted = value_str.replace(".", "")
        # Remove leading zeros, but handle "0" or "0.0..." correctly
        if len(formatted) > 1:
            # Find first non-zero digit
            first_digit_index = -1
            for i, char in enumerate(formatted):
                if char != "0":
                    first_digit_index = i
                    break
            if first_digit_index != -1:
                formatted = formatted[first_digit_index:]
            else:  # String was all zeros
                formatted = "0"
        elif formatted == "" or formatted == "0":  # Handle cases like "0", "0.0", "."
            formatted = "0"
        # Ensure final result is "0" if original float value was 0
        try:
            if float(value_str) == 0.0:
                formatted = "0"
        except ValueError:
            self.logger.warning(
                f"Could not convert value '{value_str}' to float for zero check.",
                source_module=self.__class__.__name__,
            )
            # Keep formatted value as is if float conversion fails

        if not formatted:  # Final safety check
            self.logger.warning(
                f"Unexpected empty string after formatting '{value_str}'. Defaulting to '0'.",
                source_module=self.__class__.__name__,
            )
            return "0"

        return formatted

    def _calculate_book_checksum(self, bids: SortedDict, asks: SortedDict) -> Optional[int]:
        """
        Calculates the CRC32 checksum for the top 10 bids and asks
        according to Kraken's v2 specification. Returns unsigned 32-bit int or None on error.
        """
        checksum_str_parts = []
        levels_to_checksum = 10

        # Top 10 Asks (lowest price first)
        ask_items = list(asks.items())[:levels_to_checksum]
        for price_str, qty_str in ask_items:
            try:
                formatted_price = self._format_level_value(price_str)
                formatted_qty = self._format_level_value(qty_str)
                checksum_str_parts.append(formatted_price)
                checksum_str_parts.append(formatted_qty)
            except Exception:
                self.logger.error(
                    f"Error formatting ask level for checksum: " f"P='{price_str}', Q='{qty_str}'",
                    source_module=self.__class__.__name__,
                )
                return None  # Cannot calculate if formatting fails

        # Top 10 Bids (highest price first)
        bid_items = list(reversed(bids.items()))[:levels_to_checksum]
        for price_str, qty_str in bid_items:
            try:
                formatted_price = self._format_level_value(price_str)
                formatted_qty = self._format_level_value(qty_str)
                checksum_str_parts.append(formatted_price)
                checksum_str_parts.append(formatted_qty)
            except Exception:
                self.logger.error(
                    f"Error formatting bid level for checksum: " f"P='{price_str}', Q='{qty_str}'",
                    source_module=self.__class__.__name__,
                )
                return None  # Cannot calculate if formatting fails

        if not checksum_str_parts:
            self.logger.debug(
                "Calculating checksum for empty top 10 bids/asks.",
                source_module=self.__class__.__name__,
            )
            # Per Kraken examples, checksum exists even for sparse books.
            # If top 10 is empty, the string should be empty. Let CRC32 handle empty string.

        final_checksum_str = "".join(checksum_str_parts)
        self.logger.debug(
            f"Checksum string (first 100 chars): " f"{final_checksum_str[:100]}...",
            source_module=self.__class__.__name__,
        )

        try:
            # Calculate CRC32 checksum and ensure it's treated as unsigned 32-bit
            checksum = binascii.crc32(final_checksum_str.encode("utf-8")) & 0xFFFFFFFF
            return checksum
        except Exception:
            self.logger.error(
                f"Error in binascii.crc32 for string '{final_checksum_str[:100]}...'",
                source_module=self.__class__.__name__,
            )
            return None

    async def _handle_ohlc_data(self, data: Dict[str, Any]) -> None:
        """Handles incoming OHLC data."""
        # Safely get nested dictionary values with None checks
        params = data.get('params', {})  # Use empty dict as default instead of None
        symbol = params.get('symbol')
        interval_int = params.get('interval') # Kraken uses integer intervals
        
        # Only try to get interval string if interval_int is not None
        interval_str = self.INTERVAL_MAP.get(interval_int, f"unknown({interval_int})") if interval_int is not None else "unknown"
        
        # Safely get data list
        ohlc_list = data.get('data', [])  # Use empty list as default instead of None

        if not symbol or not ohlc_list:
            self.logger.warning(f"Received incomplete OHLC data: {data}")
            return

        for ohlc_item in ohlc_list:
            if len(ohlc_item) < 7:
                self.logger.warning(f"Malformed OHLC item for {symbol}: {ohlc_item}")
                continue

            try:
                # Kraken v2 OHLC format: [ timestamp, open, high, low, close, volume, trades ]
                ts_float = float(ohlc_item[0])
                bar_start_dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)
                open_p, high_p, low_p, close_p, volume_v = (
                    str(ohlc_item[1]),
                    str(ohlc_item[2]),
                    str(ohlc_item[3]),
                    str(ohlc_item[4]),
                    str(ohlc_item[5]),
                )
                # trades_count = int(ohlc_item[6]) # Optional

                event = MarketDataOHLCVEvent(
                    source_module=self._source_module,
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow(), # Event creation time
                    trading_pair=symbol,
                    exchange="kraken",
                    interval=interval_str,
                    timestamp_bar_start=bar_start_dt,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=volume_v
                )
                await self.pubsub.publish(event)

                # self.logger.debug(f"Published OHLCV event for {symbol} {interval_str}")

            except (ValueError, TypeError, IndexError) as e:
                self.logger.error(
                    f"Error processing OHLC item for {symbol}: {e} - Item: {ohlc_item}",
                    source_module=self._source_module,
                    exc_info=True
                )
            except Exception as e:
                 self.logger.error(
                    f"Unexpected error publishing OHLCV event for {symbol}: {e}",
                    source_module=self._source_module,
                    exc_info=True
                 )

    async def _reconnect(self) -> None:
        """Attempts to reconnect to the WebSocket after a delay."""
        if self._is_stopping:
            return
        self.logger.info(
            f"Attempting WebSocket reconnection in {self._reconnect_delay} seconds...",
            source_module=self._source_module,
        )
        await asyncio.sleep(self._reconnect_delay)
        if not self._is_stopping:
             await self.start() # Re-run the start logic to connect and subscribe
        else:
            self.logger.info("Stop signal received during reconnect delay, aborting reconnect.", source_module=self._source_module)


# Example Usage (for testing purposes - requires libraries installed)
async def main() -> None:
    """Main function for standalone testing."""
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

    Returns:
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
    config: dict, 
    event_bus: asyncio.Queue, 
    logger_service: "LoggerService[Any]"
) -> None:
    """Run the test with the given configuration.

    Args:
        config: Test configuration
        event_bus: Event bus for communication
        logger_service: Logger service instance
    """
    from .core.pubsub import PubSubManager
    logger = logging.getLogger("test")
    pubsub = PubSubManager(logger)
    
    # Create a ConfigManager-compatible dictionary wrapper
    from .config_manager import ConfigManager
    
    class TestConfigManager(ConfigManager):
        def __init__(self, config_dict: Dict[str, Any]):
            self._config = config_dict
            
        def get(self, key: str, default: Any = None) -> Any:
            if self._config is None:
                return default
            return self._config.get(key, default)
    
    config_manager = TestConfigManager(config)
    
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
        try:
            await test_consumer_task
        except asyncio.CancelledError:
            pass


async def _run_event_consumer(event_bus: asyncio.Queue) -> None:
    """Run the event consumer.

    Args:
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
    """Mock logger service for testing."""

    def __init__(self, config_manager: Optional["ConfigManager"] = None, pubsub_manager: Optional["PubSubManager"] = None) -> None:
        """Initialize the mock logger service."""
        pass

    def log(
        self,
        level: int,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
        exc_info: Optional[bool] = None,
    ) -> None:
        """Log a message."""
        level_name = {
            50: "CRITICAL",
            40: "ERROR",
            30: "WARNING",
            20: "INFO",
            10: "DEBUG"
        }.get(level, "UNKNOWN")
        print(f"[{level_name}] {msg}")

    def debug(
        self, msg: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None
    ) -> None:
        """Log a debug message."""
        self.log(10, msg, source_module, context)

    def info(
        self, msg: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None
    ) -> None:
        """Log an info message."""
        self.log(20, msg, source_module, context)

    def warning(
        self, msg: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None
    ) -> None:
        """Log a warning message."""
        self.log(30, msg, source_module, context)

    def error(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
        exc_info: Optional[bool] = None,
    ) -> None:
        """Log an error message."""
        self.log(40, msg, source_module, context, exc_info)

    def critical(
        self,
        msg: str,
        source_module: Optional[str] = None,
        context: Optional[Dict[Any, Any]] = None,
        exc_info: Optional[bool] = None,
    ) -> None:
        """Log a critical message."""
        self.log(50, msg, source_module, context, exc_info)


if __name__ == "__main__":
    asyncio.run(main())
