"""Kraken WebSocket client for real-time data."""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import aiohttp
import websockets
from websockets import ClientConnection

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import (
    ExecutionReportEvent,
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
        """Handle order book updates. (Placeholder)"""
        # Placeholder for actual order book handling logic
        # For now, just log that the message was received
        pair_info = message[3] if len(message) > 3 else "UnknownPair"
        self.logger.info(
            f"Order book update received for {pair_info}. Processing not yet implemented.",
            source_module=self._source_module,
            context={"message_snippet": str(message)[:200]},
        )
        # TODO: Implement full order book processing logic here, including:
        # - Parsing snapshot and update messages
        # - Maintaining local order book state (bids, asks)
        # - Validating checksums if provided by Kraken
        # - Publishing MarketDataL2Event to PubSub

    async def _handle_own_trades(self, message: list) -> None:
        """Handle own trades updates."""
        trades_data = message[0]

        for trade_id, trade_data in trades_data.items():
            # This represents a fill, update the order
            order_id = trade_data.get("orderid")

            # We'll need to maintain order state to properly track fills
            # For now, publish as execution report
            self.logger.info(
                f"Trade executed: {trade_id} for order {order_id}",
                source_module=self._source_module,
                context={"trade_data": trade_data},
            )

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
