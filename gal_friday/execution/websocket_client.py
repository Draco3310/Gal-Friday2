"""Kraken WebSocket client for real-time data."""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Sequence, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import aiohttp
import websockets
from websockets import ClientConnection

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import (
    ExecutionReportEvent,
    FillEvent,
    MarketDataL2Event,
    MarketDataOHLCVEvent,
    MarketDataTickerEvent,
    MarketDataTradeEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class TokenCache(TypedDict):
    """Type for WebSocket token cache."""
    token: str
    expires_at: float


@dataclass
class WebSocketMessage:
    """Parsed WebSocket message."""
    channel: str
    data: dict[str, Any]
    sequence: int | None = None
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


class OrderBookSide(str, Enum):
    """Order book sides."""
    BID = "bid"
    ASK = "ask"


class ProcessingError(str, Enum):
    """Types of order book processing errors."""
    INVALID_PRICE = "invalid_price"
    INVALID_QUANTITY = "invalid_quantity"
    SEQUENCE_GAP = "sequence_gap"
    MALFORMED_MESSAGE = "malformed_message"
    STALE_DATA = "stale_data"
    MEMORY_LIMIT = "memory_limit"
    PROCESSING_TIMEOUT = "processing_timeout"
    CROSSED_MARKET = "crossed_market"


@dataclass
class OrderBookLevel:
    """Individual price level in order book."""
    price: Decimal
    quantity: Decimal
    timestamp: float
    
    def is_empty(self) -> bool:
        """Check if level should be removed."""
        return self.quantity <= 0


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get highest bid price."""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get lowest ask price."""
        return self.asks[0] if self.asks else None
    
    def get_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None


@dataclass
class ProcessingMetrics:
    """Metrics for order book processing performance."""
    messages_processed: int = 0
    errors_encountered: int = 0
    average_processing_time: float = 0.0
    last_update_time: float = 0.0
    sequence_gaps: int = 0
    recovery_attempts: int = 0


# Constants for message parsing
MESSAGE_MIN_LENGTH = 4  # Minimum length of a valid message list
MESSAGE_CHANNEL_INDEX = 2  # Index of channel name in message list
MESSAGE_CHANNEL_NAME_INDEX = 2  # Index of channel name in message list
MESSAGE_DATA_INDEX = 1  # Index of data in message list
MESSAGE_PAIR_INDEX = 3  # Index of trading pair in message list


class ExecutionHandlerAuthenticationError(ValueError):
    """Raised when an API credential is not in the expected format or other auth issue.

    Args:
        message: Custom error message. Defaults to API secret format error.
        *args: Additional arguments to pass to the parent class.
    """
    def __init__(self, message: str = "API secret must be base64 encoded.", *args: object) -> None:
        """Initialize the error with a custom message.

        Args:
            message: Custom error message. Defaults to API secret format error.
            *args: Additional arguments to pass to the parent class.
        """
        super().__init__(message, *args)


class KrakenWebSocketClient:
    """WebSocket client for Kraken real-time data."""

    def __init__(
        self,
        config: ConfigManager,
        pubsub: PubSubManager,
        logger: LoggerService,
    ) -> None:
        """Initialize Kraken WebSocket client.

        Args:
            config: Configuration manager instance
            pubsub: PubSub manager for event publishing
            logger: Logger service instance
        """
        self.config = config
        self.pubsub = pubsub
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Connection settings
        self.ws_url = config.get("kraken.ws_url", "wss://ws.kraken.com")
        self.ws_auth_url = config.get("kraken.ws_auth_url", "wss://ws-auth.kraken.com")
        self.api_key = config.get("kraken.api_key")
        self.api_secret = base64.b64decode(config.get("kraken.secret_key"))

        # Add REST API base URL for token retrieval
        self.api_base_url = config.get("kraken.api_url", "https://api.kraken.com")

        # Cache for WebSocket token with expiry
        self._ws_token_cache: TokenCache = {
            "token": "",
            "expires_at": 0,
        }

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.ws_public: ClientConnection | None = None
        self.ws_private: ClientConnection | None = None

        # Subscriptions
        self.public_subscriptions: set[str] = set()
        self.private_subscriptions: set[str] = set()

        # Message handling
        self.sequence_numbers: dict[str, int] = {}
        self.message_handlers: dict[str, Callable] = {
            "ticker": self._handle_ticker,
            "book": self._handle_orderbook,
            "trade": self._handle_trades,
            "ohlc": self._handle_ohlc,
            "ownTrades": self._handle_own_trades,
            "openOrders": self._handle_open_orders,
        }

        # Connection management
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0
        self.heartbeat_interval = 30.0
        self._connection_tasks: list[asyncio.Task] = []

        # Order book state management
        self.order_books: dict[str, OrderBookSnapshot] = {}
        self.last_sequence: dict[str, int] = {}
        self.error_counts: dict[ProcessingError, int] = {}
        
        # Order book processing configuration
        self.max_order_book_depth = config.get("order_book.max_depth", 100)
        self.stale_data_threshold = config.get("order_book.stale_data_threshold_seconds", 30)
        self.max_processing_time = config.get("order_book.max_processing_time_seconds", 1.0)
        self.error_recovery_enabled = config.get("order_book.error_recovery_enabled", True)
        self.validation_enabled = config.get("order_book.validation_enabled", True)
        self.max_sequence_gap = config.get("order_book.max_sequence_gap", 10)
        
        # Performance tracking
        self.processing_metrics = ProcessingMetrics()
        self.processing_times = deque(maxlen=1000)

    async def connect(self) -> None:
        """Establish WebSocket connections."""
        self.state = ConnectionState.CONNECTING

        try:
            # Connect to public WebSocket
            self._connection_tasks.append(
                asyncio.create_task(self._connect_public()),
            )

            # Connect to private WebSocket
            self._connection_tasks.append(
                asyncio.create_task(self._connect_private()),
            )

            # Start heartbeat
            self._connection_tasks.append(
                asyncio.create_task(self._heartbeat_loop()),
            )

            self.state = ConnectionState.CONNECTED
            self.logger.info(
                "WebSocket connections established",
                source_module=self._source_module,
            )

        except Exception:
            self.state = ConnectionState.ERROR
            self.logger.exception(
                "Failed to establish WebSocket connections",
                source_module=self._source_module,
            )
            await self._reconnect()

    async def _connect_public(self) -> None:
        """Connect to public WebSocket."""
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.ws_public = ws
                    self.logger.info(
                        "Public WebSocket connected",
                        source_module=self._source_module,
                    )

                    # Resubscribe to channels
                    await self._resubscribe_public()

                    # Handle messages
                    async for message in ws:
                        await self._process_public_message(str(message))

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(
                    "Public WebSocket connection closed",
                    source_module=self._source_module,
                )
                await asyncio.sleep(self.reconnect_delay)

            except Exception:
                self.logger.exception(
                    "Error in public WebSocket",
                    source_module=self._source_module,
                )
                await asyncio.sleep(self.reconnect_delay)

    async def _connect_private(self) -> None:
        """Connect to authenticated private WebSocket."""
        while True:
            try:
                # Get authentication token
                token = await self._get_ws_token()

                async with websockets.connect(self.ws_auth_url) as ws:
                    self.ws_private = ws

                    # Authenticate
                    auth_message = {
                        "event": "subscribe",
                        "subscription": {
                            "name": "*",
                            "token": token,
                        },
                    }
                    await ws.send(json.dumps(auth_message))

                    # Wait for authentication confirmation
                    response = await ws.recv()
                    auth_response = json.loads(str(response))

                    if auth_response.get("status") == "ok":
                        self.state = ConnectionState.AUTHENTICATED
                        self.logger.info(
                            "Private WebSocket authenticated",
                            source_module=self._source_module,
                        )

                        # Handle messages
                        async for message in ws:
                            await self._process_private_message(str(message))
                    else:
                        raise Exception(f"Authentication failed: {auth_response}")

            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(
                    "Private WebSocket connection closed",
                    source_module=self._source_module,
                )
                await asyncio.sleep(self.reconnect_delay)

            except Exception:
                self.logger.exception(
                    "Error in private WebSocket",
                    source_module=self._source_module,
                )
                await asyncio.sleep(self.reconnect_delay)

    async def _get_ws_token(self) -> str:
        """Get WebSocket authentication token from Kraken API.
        
        Implements caching to avoid unnecessary API calls.
        Tokens are valid for 900 seconds (15 minutes).
        
        Returns:
            Valid WebSocket authentication token
            
        Raises:
            ExecutionHandlerAuthenticationError: If token retrieval fails
        """
        current_time = time.time()

        # Check if cached token is still valid (with 60s buffer)
        if (self._ws_token_cache["token"] and
            self._ws_token_cache["expires_at"] > current_time + 60):
            self.logger.debug(
                "Using cached WebSocket token",
                source_module=self._source_module,
                context={"expires_in": self._ws_token_cache["expires_at"] - current_time},
            )
            return self._ws_token_cache["token"]  # This is definitely a string based on our TokenCache type

        # Generate new token
        try:
            # Prepare API request
            endpoint = "/0/private/GetWebSocketsToken"
            nonce = str(int(time.time() * 1000))

            # Create signature
            post_data = f"nonce={nonce}"
            message = endpoint.encode() + hashlib.sha256(
                f"{nonce}{post_data}".encode(),
            ).digest()
            signature = hmac.new(
                self.api_secret,
                message,  # message is already bytes
                hashlib.sha512,
            ).digest()

            # Make API request
            headers = {
                "API-Key": self.api_key,
                "API-Sign": base64.b64encode(signature).decode(),
                "Content-Type": "application/x-www-form-urlencoded",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base_url}{endpoint}",
                    headers=headers,
                    data=post_data,
                ) as response:
                    result = await response.json()

            # Handle response
            if result.get("error"):
                raise ExecutionHandlerAuthenticationError(
                    f"Failed to get WebSocket token: {result['error']}",
                )

            token = cast("str", result["result"]["token"])

            # Cache token (valid for 900 seconds)
            self._ws_token_cache = {
                "token": token,
                "expires_at": current_time + 900,
            }

            self.logger.info(
                "Successfully retrieved new WebSocket token",
                source_module=self._source_module,
                context={"expires_in": 900},
            )

            return token

        except Exception as e:
            self.logger.error(
                "Failed to retrieve WebSocket token: %s",
                str(e),
                source_module=self._source_module,
                context={"error_type": type(e).__name__},
            )
            raise ExecutionHandlerAuthenticationError(
                f"WebSocket token retrieval failed: {e}",
            ) from e

    async def subscribe_market_data(self, pairs: list[str], channels: list[str]) -> None:
        """Subscribe to market data channels."""
        if not self.ws_public:
            self.logger.error(
                "Cannot subscribe - public WebSocket not connected",
                source_module=self._source_module,
            )
            return

        for channel in channels:
            subscription = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": channel},
            }

            await self.ws_public.send(json.dumps(subscription))
            self.public_subscriptions.add(f"{channel}:{','.join(pairs)}")

        self.logger.info(
            f"Subscribed to {channels} for {pairs}",
            source_module=self._source_module,
        )

    async def _resubscribe_public(self) -> None:
        """Resubscribe to all public channels after reconnection."""
        for subscription in self.public_subscriptions:
            channel, pairs_str = subscription.split(":", 1)
            pairs = pairs_str.split(",")
            await self.subscribe_market_data(pairs, [channel])

    async def _process_public_message(self, raw_message: str) -> None:
        """Process public WebSocket message."""
        try:
            message = json.loads(raw_message)

            # Handle different message types
            if isinstance(message, dict):
                if "event" in message:
                    await self._handle_event_message(message)
                elif "errorMessage" in message:
                    self.logger.error(
                        f"WebSocket error: {message['errorMessage']}",
                        source_module=self._source_module,
                    )
            elif isinstance(message, list) and len(message) >= MESSAGE_MIN_LENGTH:
                # Data message format: [channelID, data, channelName, pair]
                channel_name = (
                    message[MESSAGE_CHANNEL_INDEX]
                    if len(message) > MESSAGE_CHANNEL_INDEX
                    else None
                )

                if channel_name is not None and channel_name in self.message_handlers:
                    handler = self.message_handlers[channel_name]
                    await handler(message)

        except json.JSONDecodeError as e:
            self.logger.error(
                f"Invalid JSON in WebSocket message: {raw_message}, error: {e!s}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.exception(
                f"Error processing public WebSocket message: {e!s}",
                source_module=self._source_module,
            )

    async def _process_private_message(self, raw_message: str) -> None:
        """Process private WebSocket message."""
        try:
            message = json.loads(raw_message)

            if isinstance(message, list) and len(message) > MESSAGE_CHANNEL_NAME_INDEX:
                channel_name = (
                    message[MESSAGE_CHANNEL_NAME_INDEX]
                    if isinstance(message[MESSAGE_CHANNEL_NAME_INDEX], str)
                    else None
                )

                if channel_name in ["ownTrades", "openOrders"]:
                    handler = self.message_handlers[channel_name]
                    await handler(message)

        except Exception:
            self.logger.exception(
                "Error processing private WebSocket message",
                source_module=self._source_module,
            )

    async def _handle_event_message(self, message: dict) -> None:
        """Handle WebSocket event messages."""
        try:
            event = message.get("event")

            if event == "systemStatus":
                self.logger.info(
                    f"System status: {message.get('status')}",
                    source_module=self._source_module,
                )
            elif event == "subscriptionStatus":
                status = message.get("status")
                if status == "subscribed":
                    self.logger.info(
                        f"Successfully subscribed to {message.get('channelName')}",
                        source_module=self._source_module,
                    )
                elif status == "error":
                    self.logger.error(
                        f"Subscription error: {message.get('errorMessage')}",
                        source_module=self._source_module,
                    )
            elif isinstance(message, list) and len(message) >= MESSAGE_MIN_LENGTH:
                # Data message format: [channelID, data, channelName, pair]
                channel_name = (
                    message[MESSAGE_CHANNEL_INDEX]
                    if len(message) > MESSAGE_CHANNEL_INDEX
                    else None
                )

                if channel_name in self.message_handlers:
                    handler = self.message_handlers[channel_name]
                    await handler(message)
        except Exception as e:
            self.logger.exception(
                f"Error in _handle_event_message: {e!s}",
                source_module=self._source_module,
            )

    async def _handle_open_orders(self, message: list) -> None:
        """Handle open orders updates."""
        orders_data = message[0]

        for order_data in orders_data:
            order_id = order_data.get("orderid")

            # Map Kraken status to internal status
            status_map = {
                "pending": "NEW",
                "open": "OPEN",
                "closed": "CLOSED",
                "canceled": "CANCELLED",
                "expired": "EXPIRED",
            }

            status = status_map.get(order_data.get("status", ""), "UNKNOWN")

            # Create execution report event
            event = ExecutionReportEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                exchange_order_id=order_id,
                client_order_id=order_data.get("userref", ""),
                order_status=status,
                order_type="LIMIT",  # Default order type, should be extracted from order data
                trading_pair=order_data.get("descr", {}).get("pair", ""),
                exchange="kraken",
                side=order_data.get("descr", {}).get("type", "").upper(),
                quantity_ordered=Decimal(order_data.get("vol", "0")),
                quantity_filled=Decimal(order_data.get("vol_exec", "0")),
                average_fill_price=(
                    Decimal(order_data["avg_price"])
                    if order_data.get("avg_price")
                    else None
                ),
                commission=Decimal(order_data.get("fee", "0")) if order_data.get("fee") else None,
                signal_id=uuid.UUID(order_data.get("userref", str(uuid.uuid4()))),
                error_message=order_data.get("reason") if status == "CANCELLED" else None,
            )

            await self.pubsub.publish(event)

    async def _handle_orderbook(self, message: list) -> None:
        """Handle order book updates with comprehensive processing and error handling."""
        start_time = time.time()
        
        try:
            # Validate basic message structure
            if not self._validate_orderbook_message(message):
                await self._handle_processing_error(
                    ProcessingError.MALFORMED_MESSAGE,
                    "Invalid order book message structure",
                    raw_data=message
                )
                return
            
            # Extract message components
            # Kraken format: [channelID, data, "book", pair]
            data = message[MESSAGE_DATA_INDEX]
            pair = message[MESSAGE_PAIR_INDEX] if len(message) > MESSAGE_PAIR_INDEX else "UNKNOWN"
            
            # Map Kraken pair to internal format
            symbol = self._map_kraken_pair(pair)
            
            # Check processing time threshold
            timeout_task = asyncio.create_task(asyncio.sleep(self.max_processing_time))
            processing_task = asyncio.create_task(
                self._process_order_book_data(symbol, data)
            )
            
            done, pending = await asyncio.wait(
                [processing_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any remaining tasks
            for task in pending:
                task.cancel()
            
            # Check if processing completed successfully
            if processing_task in done:
                success = await processing_task
                if success:
                    # Update processing metrics
                    processing_time = time.time() - start_time
                    await self._update_processing_metrics(processing_time)
                    
                    self.logger.debug(
                        f"Successfully processed order book update for {symbol}",
                        source_module=self._source_module,
                        context={"processing_time": f"{processing_time:.4f}s"}
                    )
                else:
                    self.logger.warning(
                        f"Order book processing failed for {symbol}",
                        source_module=self._source_module
                    )
            else:
                await self._handle_processing_error(
                    ProcessingError.PROCESSING_TIMEOUT,
                    f"Processing timeout exceeded {self.max_processing_time}s for {symbol}",
                    symbol=symbol
                )
                
        except asyncio.TimeoutError:
            await self._handle_processing_error(
                ProcessingError.PROCESSING_TIMEOUT,
                f"Processing timeout for order book update",
                symbol=symbol if 'symbol' in locals() else None
            )
        except MemoryError:
            await self._handle_processing_error(
                ProcessingError.MEMORY_LIMIT,
                "Memory limit exceeded during order book processing",
                symbol=symbol if 'symbol' in locals() else None
            )
        except Exception as e:
            await self._handle_processing_error(
                ProcessingError.MALFORMED_MESSAGE,
                f"Unexpected error processing order book: {e}",
                symbol=symbol if 'symbol' in locals() else None,
                raw_data=message
            )

    async def _handle_own_trades(self, message: list) -> None:
        """Handle own trades updates and publish :class:`FillEvent`."""
        trades_data = message[0]

        for trade_id, trade_data in trades_data.items():
            order_id = trade_data.get("ordertxid") or trade_data.get("orderid", "")

            fill_event = FillEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                order_id=order_id,
                fill_id=str(trade_id),
                trading_pair=self._map_kraken_pair(trade_data.get("pair", "")),
                side=str(trade_data.get("type", "")).upper(),
                price=Decimal(trade_data.get("price", "0")),
                quantity=Decimal(trade_data.get("vol", "0")),
                fee=Decimal(trade_data.get("fee", "0")),
            )

            await self.pubsub.publish(fill_event)

    async def _handle_ticker(self, message: list) -> None:
        """Handle ticker updates."""
        try:
            if len(message) < 4:
                self.logger.warning(
                    "Invalid ticker message format",
                    source_module=self._source_module,
                    context={"message_length": len(message)},
                )
                return

            # Kraken ticker format: [channelID, data, "ticker", pair]
            data = message[1]
            pair = message[3]

            # Extract ticker data
            ticker_event = MarketDataTickerEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair=self._map_kraken_pair(pair),
                exchange="kraken",
                bid=Decimal(data["b"][0]),  # Best bid price
                bid_size=Decimal(data["b"][1]),  # Best bid size
                ask=Decimal(data["a"][0]),  # Best ask price
                ask_size=Decimal(data["a"][1]),  # Best ask size
                last_price=Decimal(data["c"][0]),  # Last trade price
                last_size=Decimal(data["c"][1]),  # Last trade size
                volume_24h=Decimal(data["v"][1]),  # 24h volume
                vwap_24h=Decimal(data["p"][1]),  # 24h VWAP
                high_24h=Decimal(data["h"][1]),  # 24h high
                low_24h=Decimal(data["l"][1]),  # 24h low
                trades_24h=int(data["t"][1]),  # 24h trade count
                timestamp_exchange=datetime.now(UTC),
            )

            await self.pubsub.publish(ticker_event)

            self.logger.debug(
                f"Published ticker event for {pair}",
                source_module=self._source_module,
                context={
                    "bid": str(ticker_event.bid),
                    "ask": str(ticker_event.ask),
                    "last": str(ticker_event.last_price),
                },
            )

        except Exception:
            self.logger.exception(
                "Error handling ticker message",
                source_module=self._source_module,
            )

    async def _handle_trades(self, message: list) -> None:
        """Handle public trades."""
        try:
            if len(message) < 4:
                self.logger.warning(
                    "Invalid trade message format",
                    source_module=self._source_module,
                    context={"message_length": len(message)},
                )
                return

            # Kraken trade format: [channelID, [[price, volume, time, side, orderType, misc]], "trade", pair]
            trades_data = message[1]
            pair = message[3]

            for trade in trades_data:
                if len(trade) < 4:
                    continue

                trade_event = MarketDataTradeEvent(
                    source_module=self._source_module,
                    event_id=uuid.uuid4(),
                    timestamp=datetime.now(UTC),
                    trading_pair=self._map_kraken_pair(pair),
                    exchange="kraken",
                    trade_id=str(uuid.uuid4()),  # Kraken doesn't provide trade IDs in stream
                    price=Decimal(trade[0]),
                    volume=Decimal(trade[1]),
                    side="buy" if trade[3] == "b" else "sell",
                    timestamp_exchange=datetime.fromtimestamp(float(trade[2]), tz=UTC),
                )

                await self.pubsub.publish(trade_event)

            self.logger.debug(
                f"Published {len(trades_data)} trade events for {pair}",
                source_module=self._source_module,
            )

        except Exception:
            self.logger.exception(
                "Error handling trades message",
                source_module=self._source_module,
            )

    async def _handle_ohlc(self, message: list) -> None:
        """Handle OHLC candle updates."""
        try:
            if len(message) < 4:
                self.logger.warning(
                    "Invalid OHLC message format",
                    source_module=self._source_module,
                    context={"message_length": len(message)},
                )
                return

            # Kraken OHLC format: [channelID, [time, etime, open, high, low, close, vwap, volume, count], "ohlc-interval", pair]
            ohlc_data = message[1]
            pair = message[3]

            # Extract interval from channel name (e.g., "ohlc-5" for 5 minutes)
            channel_name = message[2] if len(message) > 2 else ""
            interval = "1m"  # Default
            if "-" in channel_name:
                interval_str = channel_name.split("-")[1]
                interval = f"{interval_str}m"

            ohlc_event = MarketDataOHLCVEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair=self._map_kraken_pair(pair),
                exchange="kraken",
                interval=interval,
                timestamp_bar_start=datetime.fromtimestamp(float(ohlc_data[0]), tz=UTC),
                open=str(ohlc_data[2]),
                high=str(ohlc_data[3]),
                low=str(ohlc_data[4]),
                close=str(ohlc_data[5]),
                volume=str(ohlc_data[7]),
            )

            await self.pubsub.publish(ohlc_event)

            self.logger.debug(
                f"Published OHLC event for {pair} ({interval})",
                source_module=self._source_module,
                context={
                    "open": str(ohlc_event.open),
                    "close": str(ohlc_event.close),
                    "volume": str(ohlc_event.volume),
                },
            )

        except Exception:
            self.logger.exception(
                "Error handling OHLC message",
                source_module=self._source_module,
            )

    # Order book processing methods
    
    def _validate_orderbook_message(self, message: list) -> bool:
        """Validate basic order book message structure."""
        try:
            if len(message) < MESSAGE_MIN_LENGTH:
                return False
            
            # Check if we have data and pair information
            if len(message) <= MESSAGE_PAIR_INDEX:
                return False
                
            # Validate that data is present
            data = message[MESSAGE_DATA_INDEX]
            if not isinstance(data, dict):
                return False
                
            return True
            
        except Exception:
            return False
    
    async def _process_order_book_data(self, symbol: str, data: dict) -> bool:
        """Process order book data with comprehensive error handling."""
        try:
            # Check for stale data
            current_time = time.time()
            if self._is_stale_data(current_time):
                await self._handle_processing_error(
                    ProcessingError.STALE_DATA,
                    f"Stale data detected for {symbol}",
                    symbol=symbol
                )
                return False
            
            # Determine if this is a snapshot or update
            is_snapshot = "bs" in data and "as" in data  # Kraken snapshot format
            
            if is_snapshot:
                success = await self._process_snapshot_data(symbol, data)
            else:
                success = await self._process_incremental_data(symbol, data)
            
            if success:
                # Validate order book consistency
                if self.validation_enabled:
                    consistency_valid = await self._validate_order_book_consistency(symbol)
                    if not consistency_valid:
                        self.logger.error(f"Order book consistency check failed for {symbol}")
                        return False
                
                # Publish order book event
                await self._publish_order_book_event(symbol, is_snapshot)
                
            return success
            
        except Exception as e:
            await self._handle_processing_error(
                ProcessingError.MALFORMED_MESSAGE,
                f"Error processing order book data for {symbol}: {e}",
                symbol=symbol
            )
            return False
    
    async def _process_snapshot_data(self, symbol: str, data: dict) -> bool:
        """Process order book snapshot data."""
        try:
            # Create new order book snapshot
            order_book = OrderBookSnapshot(symbol=symbol)
            
            # Process bids (bs = bid snapshot)
            if "bs" in data:
                bids_success = await self._process_bid_levels(order_book, data["bs"])
                if not bids_success:
                    return False
            
            # Process asks (as = ask snapshot)
            if "as" in data:
                asks_success = await self._process_ask_levels(order_book, data["as"])
                if not asks_success:
                    return False
            
            # Update order book state
            self.order_books[symbol] = order_book
            
            self.logger.debug(
                f"Processed snapshot for {symbol}",
                source_module=self._source_module,
                context={
                    "bid_levels": len(order_book.bids),
                    "ask_levels": len(order_book.asks)
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing snapshot data for {symbol}: {e}")
            return False
    
    async def _process_incremental_data(self, symbol: str, data: dict) -> bool:
        """Process incremental order book updates."""
        try:
            # Initialize order book if not exists
            if symbol not in self.order_books:
                self.order_books[symbol] = OrderBookSnapshot(symbol=symbol)
            
            order_book = self.order_books[symbol]
            
            # Process bid updates (b = bid updates)
            if "b" in data:
                bids_success = await self._process_bid_updates(order_book, data["b"])
                if not bids_success:
                    return False
            
            # Process ask updates (a = ask updates)
            if "a" in data:
                asks_success = await self._process_ask_updates(order_book, data["a"])
                if not asks_success:
                    return False
            
            # Update timestamp
            order_book.timestamp = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing incremental data for {symbol}: {e}")
            return False
    
    async def _process_bid_levels(self, order_book: OrderBookSnapshot, bids: list) -> bool:
        """Process bid levels for snapshot data."""
        try:
            processed_count = 0
            
            for bid_data in bids:
                try:
                    price, quantity = await self._extract_price_quantity(bid_data)
                    
                    if price is None or quantity is None:
                        continue
                    
                    if quantity > 0:  # Only add non-zero quantities
                        level = OrderBookLevel(
                            price=price,
                            quantity=quantity,
                            timestamp=time.time()
                        )
                        order_book.bids.append(level)
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing bid level: {e}")
                    continue
            
            # Sort bids in descending order (highest price first)
            order_book.bids.sort(key=lambda x: x.price, reverse=True)
            
            # Trim to max depth
            if len(order_book.bids) > self.max_order_book_depth:
                order_book.bids = order_book.bids[:self.max_order_book_depth]
            
            return processed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing bid levels: {e}")
            return False
    
    async def _process_ask_levels(self, order_book: OrderBookSnapshot, asks: list) -> bool:
        """Process ask levels for snapshot data."""
        try:
            processed_count = 0
            
            for ask_data in asks:
                try:
                    price, quantity = await self._extract_price_quantity(ask_data)
                    
                    if price is None or quantity is None:
                        continue
                    
                    if quantity > 0:  # Only add non-zero quantities
                        level = OrderBookLevel(
                            price=price,
                            quantity=quantity,
                            timestamp=time.time()
                        )
                        order_book.asks.append(level)
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing ask level: {e}")
                    continue
            
            # Sort asks in ascending order (lowest price first)
            order_book.asks.sort(key=lambda x: x.price)
            
            # Trim to max depth
            if len(order_book.asks) > self.max_order_book_depth:
                order_book.asks = order_book.asks[:self.max_order_book_depth]
            
            return processed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing ask levels: {e}")
            return False
    
    async def _process_bid_updates(self, order_book: OrderBookSnapshot, updates: list) -> bool:
        """Process bid updates for incremental data."""
        try:
            processed_count = 0
            
            for update_data in updates:
                try:
                    price, quantity = await self._extract_price_quantity(update_data)
                    
                    if price is None or quantity is None:
                        continue
                    
                    # Apply update to bid side
                    await self._apply_bid_update(order_book, price, quantity)
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing bid update: {e}")
                    continue
            
            return processed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing bid updates: {e}")
            return False
    
    async def _process_ask_updates(self, order_book: OrderBookSnapshot, updates: list) -> bool:
        """Process ask updates for incremental data."""
        try:
            processed_count = 0
            
            for update_data in updates:
                try:
                    price, quantity = await self._extract_price_quantity(update_data)
                    
                    if price is None or quantity is None:
                        continue
                    
                    # Apply update to ask side
                    await self._apply_ask_update(order_book, price, quantity)
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing ask update: {e}")
                    continue
            
            return processed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing ask updates: {e}")
            return False
    
    async def _extract_price_quantity(self, level_data: list) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Extract and validate price and quantity from level data."""
        try:
            if not isinstance(level_data, list) or len(level_data) < 2:
                return None, None
            
            # Extract price
            try:
                price = Decimal(str(level_data[0]))
                if price < 0:
                    self.logger.warning(f"Negative price detected: {price}")
                    return None, None
            except (ValueError, InvalidOperation):
                self.logger.warning(f"Invalid price format: {level_data[0]}")
                return None, None
            
            # Extract quantity
            try:
                quantity = Decimal(str(level_data[1]))
                if quantity < 0:
                    self.logger.warning(f"Negative quantity detected: {quantity}")
                    return None, None
            except (ValueError, InvalidOperation):
                self.logger.warning(f"Invalid quantity format: {level_data[1]}")
                return None, None
            
            return price, quantity
            
        except Exception as e:
            self.logger.error(f"Error extracting price/quantity: {e}")
            return None, None
    
    async def _apply_bid_update(self, order_book: OrderBookSnapshot, price: Decimal, quantity: Decimal) -> None:
        """Apply bid update to order book."""
        # Find existing level or insertion point
        insertion_index = 0
        found_existing = False
        
        for i, level in enumerate(order_book.bids):
            if level.price == price:
                # Update existing level
                if quantity <= 0:
                    # Remove level
                    order_book.bids.pop(i)
                else:
                    # Update quantity
                    level.quantity = quantity
                    level.timestamp = time.time()
                found_existing = True
                break
            elif level.price < price:
                insertion_index = i
                break
            else:
                insertion_index = i + 1
        
        # Insert new level if not updating existing and quantity > 0
        if not found_existing and quantity > 0:
            new_level = OrderBookLevel(
                price=price,
                quantity=quantity,
                timestamp=time.time()
            )
            order_book.bids.insert(insertion_index, new_level)
        
        # Trim to max depth
        if len(order_book.bids) > self.max_order_book_depth:
            order_book.bids = order_book.bids[:self.max_order_book_depth]
    
    async def _apply_ask_update(self, order_book: OrderBookSnapshot, price: Decimal, quantity: Decimal) -> None:
        """Apply ask update to order book."""
        # Find existing level or insertion point
        insertion_index = len(order_book.asks)
        found_existing = False
        
        for i, level in enumerate(order_book.asks):
            if level.price == price:
                # Update existing level
                if quantity <= 0:
                    # Remove level
                    order_book.asks.pop(i)
                else:
                    # Update quantity
                    level.quantity = quantity
                    level.timestamp = time.time()
                found_existing = True
                break
            elif level.price > price:
                insertion_index = i
                break
        
        # Insert new level if not updating existing and quantity > 0
        if not found_existing and quantity > 0:
            new_level = OrderBookLevel(
                price=price,
                quantity=quantity,
                timestamp=time.time()
            )
            order_book.asks.insert(insertion_index, new_level)
        
        # Trim to max depth
        if len(order_book.asks) > self.max_order_book_depth:
            order_book.asks = order_book.asks[:self.max_order_book_depth]
    
    async def _validate_order_book_consistency(self, symbol: str) -> bool:
        """Validate order book consistency after updates."""
        try:
            order_book = self.order_books.get(symbol)
            if not order_book:
                return True  # Empty order book is consistent
            
            # Check bid/ask spread sanity
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if best_bid and best_ask:
                if best_bid.price >= best_ask.price:
                    await self._handle_processing_error(
                        ProcessingError.CROSSED_MARKET,
                        f"Crossed order book detected for {symbol}: bid={best_bid.price}, ask={best_ask.price}",
                        symbol=symbol
                    )
                    return False
                
                spread = best_ask.price - best_bid.price
                spread_percentage = spread / best_ask.price
                if spread_percentage > Decimal('0.1'):  # 10% spread threshold
                    self.logger.warning(
                        f"Wide spread detected for {symbol}: {spread} ({spread_percentage:.2%})",
                        source_module=self._source_module
                    )
            
            # Check for duplicate price levels
            bid_prices = [level.price for level in order_book.bids]
            ask_prices = [level.price for level in order_book.asks]
            
            if len(bid_prices) != len(set(bid_prices)):
                self.logger.error(f"Duplicate bid prices detected for {symbol}")
                return False
            
            if len(ask_prices) != len(set(ask_prices)):
                self.logger.error(f"Duplicate ask prices detected for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order book consistency for {symbol}: {e}")
            return False
    
    async def _publish_order_book_event(self, symbol: str, is_snapshot: bool) -> None:
        """Publish MarketDataL2Event to PubSub."""
        try:
            order_book = self.order_books.get(symbol)
            if not order_book:
                return
            
            # Convert order book levels to event format
            bids_list: Sequence[tuple[str, str]] = [
                (str(level.price), str(level.quantity))
                for level in order_book.bids
            ]
            
            asks_list: Sequence[tuple[str, str]] = [
                (str(level.price), str(level.quantity))
                for level in order_book.asks
            ]
            
            # Create and publish MarketDataL2Event
            event = MarketDataL2Event(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                trading_pair=symbol,
                exchange="kraken",
                bids=bids_list,
                asks=asks_list,
                is_snapshot=is_snapshot,
                timestamp_exchange=datetime.fromtimestamp(order_book.timestamp, tz=UTC)
            )
            
            await self.pubsub.publish(event)
            
            self.logger.debug(
                f"Published L2 order book event for {symbol}",
                source_module=self._source_module,
                context={
                    "is_snapshot": is_snapshot,
                    "bid_levels": len(bids_list),
                    "ask_levels": len(asks_list)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error publishing order book event for {symbol}: {e}")
    
    def _is_stale_data(self, current_time: float) -> bool:
        """Check if data is considered stale."""
        if hasattr(self, '_last_data_time'):
            return current_time - self._last_data_time > self.stale_data_threshold
        
        self._last_data_time = current_time
        return False
    
    async def _handle_processing_error(
        self, 
        error_type: ProcessingError, 
        message: str,
        symbol: Optional[str] = None,
        sequence: Optional[int] = None,
        raw_data: Optional[Any] = None
    ) -> None:
        """Centralized error handling with recovery strategies."""
        
        # Track error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.processing_metrics.errors_encountered += 1
        
        # Log error with appropriate level
        if error_type in [ProcessingError.SEQUENCE_GAP, ProcessingError.STALE_DATA]:
            self.logger.warning(
                f"Processing warning ({error_type.value}): {message}",
                source_module=self._source_module,
                context={"symbol": symbol, "sequence": sequence}
            )
        else:
            self.logger.error(
                f"Processing error ({error_type.value}): {message}",
                source_module=self._source_module,
                context={"symbol": symbol, "sequence": sequence}
            )
        
        # Apply recovery strategy if enabled
        if self.error_recovery_enabled:
            try:
                await self._apply_error_recovery(error_type, symbol)
                self.processing_metrics.recovery_attempts += 1
            except Exception as e:
                self.logger.error(f"Error recovery failed for {error_type.value}: {e}")
    
    async def _apply_error_recovery(self, error_type: ProcessingError, symbol: Optional[str]) -> None:
        """Apply error recovery strategies."""
        if error_type == ProcessingError.CROSSED_MARKET and symbol:
            # Clear order book state for crossed market
            if symbol in self.order_books:
                del self.order_books[symbol]
                self.logger.info(f"Cleared order book state for {symbol} due to crossed market")
        
        elif error_type == ProcessingError.STALE_DATA and symbol:
            # Clear stale order book data
            if symbol in self.order_books:
                del self.order_books[symbol]
                self.logger.info(f"Cleared stale order book data for {symbol}")
        
        elif error_type == ProcessingError.MEMORY_LIMIT:
            # Clear oldest order books to free memory
            if len(self.order_books) > 10:
                # Keep only the 10 most recently updated order books
                sorted_books = sorted(
                    self.order_books.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True
                )
                self.order_books = dict(sorted_books[:10])
                self.logger.info("Cleared old order book data to manage memory usage")
    
    async def _update_processing_metrics(self, processing_time: float) -> None:
        """Update processing performance metrics."""
        self.processing_metrics.messages_processed += 1
        self.processing_metrics.last_update_time = time.time()
        
        # Add to processing times deque
        self.processing_times.append(processing_time)
        
        # Calculate average processing time
        if self.processing_times:
            self.processing_metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
    
    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot for a symbol."""
        return self.order_books.get(symbol)
    
    def get_processing_statistics(self) -> dict[str, Any]:
        """Get comprehensive order book processing statistics."""
        return {
            "metrics": {
                "messages_processed": self.processing_metrics.messages_processed,
                "errors_encountered": self.processing_metrics.errors_encountered,
                "average_processing_time": self.processing_metrics.average_processing_time,
                "sequence_gaps": self.processing_metrics.sequence_gaps,
                "recovery_attempts": self.processing_metrics.recovery_attempts
            },
            "error_counts": dict(self.error_counts),
            "symbols_tracked": len(self.order_books),
            "configuration": {
                "max_depth": self.max_order_book_depth,
                "validation_enabled": self.validation_enabled,
                "error_recovery_enabled": self.error_recovery_enabled,
                "stale_data_threshold": self.stale_data_threshold
            }
        }

    def _map_kraken_pair(self, kraken_pair: str) -> str:
        """Map Kraken pair format to internal format.
        
        Args:
            kraken_pair: Kraken pair (e.g., "XBT/USD" or "XXBTZUSD")
            
        Returns:
            Internal pair format (e.g., "BTC/USD")
        """
        # Common mappings
        mappings = {
            "XBT": "BTC",
            "XXBT": "BTC",
            "XXRP": "XRP",
            "XDOGE": "DOGE",
            "ZUSD": "USD",
        }

        # Handle different formats
        if "/" in kraken_pair:
            # Format: XBT/USD
            base, quote = kraken_pair.split("/")
            base = mappings.get(base, base)
            quote = mappings.get(quote, quote)
            return f"{base}/{quote}"
        # Format: XXBTZUSD
        for old, new in mappings.items():
            kraken_pair = kraken_pair.replace(old, new)

        # Try to split (assumes 3-letter currencies)
        if len(kraken_pair) >= 6:
            return f"{kraken_pair[:3]}/{kraken_pair[3:6]}"

        return kraken_pair

    def _get_next_sequence(self, channel: str) -> int:
        """Get next sequence number for channel."""
        current = self.sequence_numbers.get(channel, 0)
        self.sequence_numbers[channel] = current + 1
        return current + 1

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to keep connections alive."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Send ping to both connections
                if self.ws_public:
                    with contextlib.suppress(Exception):
                        await self.ws_public.ping()

                if self.ws_private:
                    with contextlib.suppress(Exception):
                        await self.ws_private.ping()

            except Exception:
                self.logger.exception(
                    "Error in heartbeat loop",
                    source_module=self._source_module,
                )

    async def _reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        self.state = ConnectionState.RECONNECTING

        self.logger.info(
            f"Attempting reconnection in {self.reconnect_delay} seconds",
            source_module=self._source_module,
        )

        await asyncio.sleep(self.reconnect_delay)

        # Exponential backoff
        self.reconnect_delay = min(
            self.reconnect_delay * 2,
            self.max_reconnect_delay,
        )

        await self.connect()

    async def disconnect(self) -> None:
        """Close WebSocket connections."""
        self.state = ConnectionState.DISCONNECTED

        # Cancel connection tasks
        for task in self._connection_tasks:
            task.cancel()

        # Close connections
        if self.ws_public:
            await self.ws_public.close()

        if self.ws_private:
            await self.ws_private.close()

        self.logger.info(
            "WebSocket connections closed",
            source_module=self._source_module,
        )
