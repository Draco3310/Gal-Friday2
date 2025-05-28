# High Priority Solution: WebSocket Order Updates Implementation

## Overview
This document provides the complete implementation plan for adding WebSocket connectivity to the Gal-Friday system for real-time order updates and market data streaming. Currently, the system uses HTTP polling which introduces latency and wastes API rate limits.

## Current State Problems

1. **Polling Inefficiencies**
   - Order status updates have 1-5 second delay
   - Wastes API rate limit on repeated queries
   - Potential to miss rapid status changes
   - Higher network overhead

2. **Missing Real-Time Features**
   - No instant order fill notifications
   - No real-time market data streaming
   - No immediate position updates
   - Delayed risk calculations

3. **Technical Limitations**
   - No WebSocket infrastructure
   - No message sequencing
   - No connection resilience
   - No event stream processing

## Solution Architecture

### 1. Kraken WebSocket Client Implementation

#### 1.1 Base WebSocket Client
```python
# gal_friday/execution/websocket_client.py
"""Kraken WebSocket client for real-time data."""

import asyncio
import websockets
import json
import hmac
import hashlib
import base64
import time
from datetime import datetime, UTC
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import uuid

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.core.pubsub import PubSubManager
from gal_friday.core.events import (
    ExecutionReportEvent,
    MarketDataL2Event,
    EventType
)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """Parsed WebSocket message."""
    channel: str
    data: Dict[str, Any]
    sequence: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)


class KrakenWebSocketClient:
    """WebSocket client for Kraken real-time data."""
    
    def __init__(self, 
                 config: ConfigManager,
                 pubsub: PubSubManager,
                 logger: LoggerService):
        self.config = config
        self.pubsub = pubsub
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Connection settings
        self.ws_url = config.get("kraken.ws_url", "wss://ws.kraken.com")
        self.ws_auth_url = config.get("kraken.ws_auth_url", "wss://ws-auth.kraken.com")
        self.api_key = config.get("kraken.api_key")
        self.api_secret = base64.b64decode(config.get("kraken.secret_key"))
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.ws_public: Optional[websockets.WebSocketClientProtocol] = None
        self.ws_private: Optional[websockets.WebSocketClientProtocol] = None
        
        # Subscriptions
        self.public_subscriptions: Set[str] = set()
        self.private_subscriptions: Set[str] = set()
        
        # Message handling
        self.sequence_numbers: Dict[str, int] = {}
        self.message_handlers: Dict[str, Callable] = {
            "ticker": self._handle_ticker,
            "book": self._handle_orderbook,
            "trade": self._handle_trades,
            "ohlc": self._handle_ohlc,
            "ownTrades": self._handle_own_trades,
            "openOrders": self._handle_open_orders
        }
        
        # Connection management
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 60.0
        self.heartbeat_interval = 30.0
        self._connection_tasks: List[asyncio.Task] = []
        
    async def connect(self):
        """Establish WebSocket connections."""
        self.state = ConnectionState.CONNECTING
        
        try:
            # Connect to public WebSocket
            self._connection_tasks.append(
                asyncio.create_task(self._connect_public())
            )
            
            # Connect to private WebSocket
            self._connection_tasks.append(
                asyncio.create_task(self._connect_private())
            )
            
            # Start heartbeat
            self._connection_tasks.append(
                asyncio.create_task(self._heartbeat_loop())
            )
            
            self.state = ConnectionState.CONNECTED
            self.logger.info(
                "WebSocket connections established",
                source_module=self._source_module
            )
            
        except Exception:
            self.state = ConnectionState.ERROR
            self.logger.exception(
                "Failed to establish WebSocket connections",
                source_module=self._source_module
            )
            await self._reconnect()
            
    async def _connect_public(self):
        """Connect to public WebSocket."""
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self.ws_public = ws
                    self.logger.info(
                        "Public WebSocket connected",
                        source_module=self._source_module
                    )
                    
                    # Resubscribe to channels
                    await self._resubscribe_public()
                    
                    # Handle messages
                    async for message in ws:
                        await self._process_public_message(message)
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(
                    "Public WebSocket connection closed",
                    source_module=self._source_module
                )
                await asyncio.sleep(self.reconnect_delay)
                
            except Exception:
                self.logger.exception(
                    "Error in public WebSocket",
                    source_module=self._source_module
                )
                await asyncio.sleep(self.reconnect_delay)
                
    async def _connect_private(self):
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
                            "token": token
                        }
                    }
                    await ws.send(json.dumps(auth_message))
                    
                    # Wait for authentication confirmation
                    response = await ws.recv()
                    auth_response = json.loads(response)
                    
                    if auth_response.get("status") == "ok":
                        self.state = ConnectionState.AUTHENTICATED
                        self.logger.info(
                            "Private WebSocket authenticated",
                            source_module=self._source_module
                        )
                        
                        # Handle messages
                        async for message in ws:
                            await self._process_private_message(message)
                    else:
                        raise Exception(f"Authentication failed: {auth_response}")
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(
                    "Private WebSocket connection closed",
                    source_module=self._source_module
                )
                await asyncio.sleep(self.reconnect_delay)
                
            except Exception:
                self.logger.exception(
                    "Error in private WebSocket",
                    source_module=self._source_module
                )
                await asyncio.sleep(self.reconnect_delay)
                
    async def _get_ws_token(self) -> str:
        """Get WebSocket authentication token."""
        # In production, this would call Kraken's GetWebSocketsToken endpoint
        # For now, return a placeholder
        return "test_token"
        
    async def subscribe_market_data(self, pairs: List[str], channels: List[str]):
        """Subscribe to market data channels."""
        if not self.ws_public:
            self.logger.error(
                "Cannot subscribe - public WebSocket not connected",
                source_module=self._source_module
            )
            return
            
        for channel in channels:
            subscription = {
                "event": "subscribe",
                "pair": pairs,
                "subscription": {"name": channel}
            }
            
            await self.ws_public.send(json.dumps(subscription))
            self.public_subscriptions.add(f"{channel}:{','.join(pairs)}")
            
        self.logger.info(
            f"Subscribed to {channels} for {pairs}",
            source_module=self._source_module
        )
        
    async def _resubscribe_public(self):
        """Resubscribe to all public channels after reconnection."""
        for subscription in self.public_subscriptions:
            channel, pairs_str = subscription.split(":", 1)
            pairs = pairs_str.split(",")
            await self.subscribe_market_data(pairs, [channel])
            
    async def _process_public_message(self, raw_message: str):
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
                        source_module=self._source_module
                    )
            elif isinstance(message, list) and len(message) >= 3:
                # Data message format: [channelID, data, channelName, pair]
                channel_name = message[2] if len(message) > 2 else None
                
                if channel_name in self.message_handlers:
                    handler = self.message_handlers[channel_name]
                    await handler(message)
                    
        except json.JSONDecodeError:
            self.logger.error(
                f"Invalid JSON in WebSocket message: {raw_message}",
                source_module=self._source_module
            )
        except Exception:
            self.logger.exception(
                "Error processing public WebSocket message",
                source_module=self._source_module
            )
            
    async def _process_private_message(self, raw_message: str):
        """Process private WebSocket message."""
        try:
            message = json.loads(raw_message)
            
            if isinstance(message, list) and len(message) >= 2:
                channel_name = message[1] if isinstance(message[1], str) else None
                
                if channel_name in ["ownTrades", "openOrders"]:
                    handler = self.message_handlers[channel_name]
                    await handler(message)
                    
        except Exception:
            self.logger.exception(
                "Error processing private WebSocket message",
                source_module=self._source_module
            )
            
    async def _handle_event_message(self, message: Dict):
        """Handle WebSocket event messages."""
        event = message.get("event")
        
        if event == "systemStatus":
            self.logger.info(
                f"System status: {message.get('status')}",
                source_module=self._source_module
            )
        elif event == "subscriptionStatus":
            status = message.get("status")
            if status == "subscribed":
                self.logger.info(
                    f"Successfully subscribed to {message.get('channelName')}",
                    source_module=self._source_module
                )
            elif status == "error":
                self.logger.error(
                    f"Subscription error: {message.get('errorMessage')}",
                    source_module=self._source_module
                )
                
    async def _handle_orderbook(self, message: List):
        """Handle order book updates."""
        data = message[1]
        pair = message[3] if len(message) > 3 else "UNKNOWN"
        
        # Extract bids and asks
        bids = []
        asks = []
        
        if "bs" in data:  # Snapshot
            bids = [[Decimal(p), Decimal(v)] for p, v in data.get("bs", [])]
            asks = [[Decimal(p), Decimal(v)] for p, v in data.get("as", [])]
        else:  # Update
            if "b" in data:
                bids = [[Decimal(p), Decimal(v)] for p, v in data.get("b", [])]
            if "a" in data:
                asks = [[Decimal(p), Decimal(v)] for p, v in data.get("a", [])]
                
        # Create and publish event
        event = MarketDataL2Event(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair=pair,
            exchange="kraken",
            bids=bids,
            asks=asks,
            timestamp_exchange=datetime.now(UTC),
            sequence_number=self._get_next_sequence("orderbook")
        )
        
        await self.pubsub.publish(event)
        
    async def _handle_open_orders(self, message: List):
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
                "expired": "EXPIRED"
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
                trading_pair=order_data.get("descr", {}).get("pair", ""),
                exchange="kraken",
                side=order_data.get("descr", {}).get("type", "").upper(),
                quantity_ordered=Decimal(order_data.get("vol", "0")),
                quantity_filled=Decimal(order_data.get("vol_exec", "0")),
                average_fill_price=Decimal(order_data.get("avg_price", "0")) if order_data.get("avg_price") else None,
                commission=Decimal(order_data.get("fee", "0")) if order_data.get("fee") else None,
                signal_id=uuid.UUID(order_data.get("userref", str(uuid.uuid4()))),
                error_message=order_data.get("reason") if status == "CANCELLED" else None
            )
            
            await self.pubsub.publish(event)
            
    async def _handle_own_trades(self, message: List):
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
                context={"trade_data": trade_data}
            )
            
    async def _handle_ticker(self, message: List):
        """Handle ticker updates."""
        # Implement ticker handling if needed
        pass
        
    async def _handle_trades(self, message: List):
        """Handle public trades."""
        # Implement public trade handling if needed
        pass
        
    async def _handle_ohlc(self, message: List):
        """Handle OHLC candle updates."""
        # Implement OHLC handling if needed
        pass
        
    def _get_next_sequence(self, channel: str) -> int:
        """Get next sequence number for channel."""
        current = self.sequence_numbers.get(channel, 0)
        self.sequence_numbers[channel] = current + 1
        return current + 1
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeat to keep connections alive."""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Send ping to both connections
                if self.ws_public and not self.ws_public.closed:
                    await self.ws_public.ping()
                    
                if self.ws_private and not self.ws_private.closed:
                    await self.ws_private.ping()
                    
            except Exception:
                self.logger.exception(
                    "Error in heartbeat loop",
                    source_module=self._source_module
                )
                
    async def _reconnect(self):
        """Handle reconnection with exponential backoff."""
        self.state = ConnectionState.RECONNECTING
        
        self.logger.info(
            f"Attempting reconnection in {self.reconnect_delay} seconds",
            source_module=self._source_module
        )
        
        await asyncio.sleep(self.reconnect_delay)
        
        # Exponential backoff
        self.reconnect_delay = min(
            self.reconnect_delay * 2,
            self.max_reconnect_delay
        )
        
        await self.connect()
        
    async def disconnect(self):
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
            source_module=self._source_module
        )
```

#### 1.2 Message Processor and Sequencing
```python
# gal_friday/execution/websocket_processor.py
"""WebSocket message processing and sequencing."""

import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Set, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
import json

from gal_friday.logger_service import LoggerService


@dataclass
class SequencedMessage:
    """Message with sequence tracking."""
    sequence: int
    channel: str
    data: Dict
    timestamp: datetime
    processed: bool = False


class SequenceTracker:
    """Track message sequences for gap detection."""
    
    def __init__(self, logger: LoggerService):
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Track sequences per channel
        self.sequences: Dict[str, int] = {}
        self.gaps: Dict[str, List[tuple]] = defaultdict(list)
        
    def check_sequence(self, channel: str, sequence: int) -> Optional[List[tuple]]:
        """Check for sequence gaps.
        
        Returns:
            List of gap ranges if gaps detected, None otherwise
        """
        expected = self.sequences.get(channel, 0) + 1
        
        if sequence == expected:
            # Normal sequence
            self.sequences[channel] = sequence
            return None
            
        elif sequence > expected:
            # Gap detected
            gap = (expected, sequence - 1)
            self.gaps[channel].append(gap)
            self.sequences[channel] = sequence
            
            self.logger.warning(
                f"Sequence gap detected in {channel}: {gap}",
                source_module=self._source_module
            )
            return [gap]
            
        else:
            # Out of order or duplicate
            self.logger.warning(
                f"Out of order message in {channel}: expected {expected}, got {sequence}",
                source_module=self._source_module
            )
            return None
            
    def get_gaps(self, channel: str) -> List[tuple]:
        """Get all gaps for a channel."""
        return self.gaps.get(channel, [])
        
    def clear_gap(self, channel: str, start: int, end: int):
        """Clear a gap after recovery."""
        if channel in self.gaps:
            self.gaps[channel] = [
                (s, e) for s, e in self.gaps[channel]
                if not (s == start and e == end)
            ]


class MessageCache:
    """Cache for deduplication and replay."""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 10000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Deque[SequencedMessage] = deque(maxlen=max_size)
        self.message_ids: Set[str] = set()
        
    def add(self, message: SequencedMessage) -> bool:
        """Add message to cache.
        
        Returns:
            True if new message, False if duplicate
        """
        # Create unique ID
        msg_id = f"{message.channel}:{message.sequence}"
        
        if msg_id in self.message_ids:
            return False
            
        # Add to cache
        self.cache.append(message)
        self.message_ids.add(msg_id)
        
        # Clean old messages
        self._cleanup()
        
        return True
        
    def get_unprocessed(self, channel: str, 
                       start_seq: int, 
                       end_seq: int) -> List[SequencedMessage]:
        """Get unprocessed messages in sequence range."""
        messages = []
        
        for msg in self.cache:
            if (msg.channel == channel and 
                start_seq <= msg.sequence <= end_seq and
                not msg.processed):
                messages.append(msg)
                
        return sorted(messages, key=lambda m: m.sequence)
        
    def mark_processed(self, channel: str, sequence: int):
        """Mark message as processed."""
        for msg in self.cache:
            if msg.channel == channel and msg.sequence == sequence:
                msg.processed = True
                break
                
    def _cleanup(self):
        """Remove expired messages."""
        cutoff = datetime.now(UTC) - timedelta(seconds=self.ttl_seconds)
        
        while self.cache and self.cache[0].timestamp < cutoff:
            old_msg = self.cache.popleft()
            msg_id = f"{old_msg.channel}:{old_msg.sequence}"
            self.message_ids.discard(msg_id)


class WebSocketMessageProcessor:
    """Process and validate WebSocket messages."""
    
    def __init__(self, logger: LoggerService):
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        self.sequence_tracker = SequenceTracker(logger)
        self.message_cache = MessageCache()
        
        # Message validation rules
        self.required_fields = {
            "openOrders": ["orderid", "status", "descr"],
            "ownTrades": ["orderid", "pair", "vol", "price"],
            "book": ["pair"],
            "ticker": ["pair", "c", "v"]
        }
        
    async def process_message(self, 
                            channel: str,
                            data: Dict,
                            sequence: Optional[int] = None) -> Optional[SequencedMessage]:
        """Process and validate WebSocket message."""
        try:
            # Validate message format
            if not self._validate_message(channel, data):
                return None
                
            # Create sequenced message
            message = SequencedMessage(
                sequence=sequence or 0,
                channel=channel,
                data=data,
                timestamp=datetime.now(UTC)
            )
            
            # Check for duplicates
            if not self.message_cache.add(message):
                self.logger.debug(
                    f"Duplicate message ignored: {channel}:{sequence}",
                    source_module=self._source_module
                )
                return None
                
            # Check sequence if provided
            if sequence:
                gaps = self.sequence_tracker.check_sequence(channel, sequence)
                if gaps:
                    # Handle gaps asynchronously
                    asyncio.create_task(self._handle_gaps(channel, gaps))
                    
            return message
            
        except Exception:
            self.logger.exception(
                "Error processing WebSocket message",
                source_module=self._source_module
            )
            return None
            
    def _validate_message(self, channel: str, data: Dict) -> bool:
        """Validate message has required fields."""
        required = self.required_fields.get(channel, [])
        
        for field in required:
            if field not in data:
                self.logger.error(
                    f"Missing required field '{field}' in {channel} message",
                    source_module=self._source_module
                )
                return False
                
        return True
        
    async def _handle_gaps(self, channel: str, gaps: List[tuple]):
        """Handle sequence gaps."""
        for start, end in gaps:
            self.logger.info(
                f"Attempting to recover gap {start}-{end} in {channel}",
                source_module=self._source_module
            )
            
            # Check cache for missing messages
            cached = self.message_cache.get_unprocessed(channel, start, end)
            
            if len(cached) == (end - start + 1):
                # All messages found in cache
                self.logger.info(
                    f"Recovered gap from cache: {channel} {start}-{end}",
                    source_module=self._source_module
                )
                
                # Mark gap as resolved
                self.sequence_tracker.clear_gap(channel, start, end)
            else:
                # Need to request missing messages
                # In production, this would trigger a recovery mechanism
                self.logger.warning(
                    f"Unable to recover gap from cache: {channel} {start}-{end}",
                    source_module=self._source_module
                )
```

### 2. Integration with Existing System

#### 2.1 Updated Execution Handler
```python
# Updates to gal_friday/execution_handler.py

class ExecutionHandler:
    """Execution handler with WebSocket support."""
    
    def __init__(self, config, pubsub, monitoring, logger):
        # ... existing init code ...
        
        # Initialize WebSocket client
        self.ws_client = KrakenWebSocketClient(config, pubsub, logger)
        self.ws_processor = WebSocketMessageProcessor(logger)
        
        # Order tracking
        self.active_orders: Dict[str, Dict] = {}
        self.order_updates_queue: asyncio.Queue = asyncio.Queue()
        
    async def start(self):
        """Start execution handler with WebSocket."""
        await super().start()
        
        # Connect WebSocket
        await self.ws_client.connect()
        
        # Subscribe to private channels
        # Note: Kraken private channels auto-subscribe with authentication
        
        # Start order update processor
        asyncio.create_task(self._process_order_updates())
        
    async def place_order(self, signal: TradeSignalApprovedEvent):
        """Place order with WebSocket tracking."""
        # Place order via REST API as before
        order_id = await self._place_order_rest(signal)
        
        # Track order for WebSocket updates
        self.active_orders[order_id] = {
            "signal_id": signal.signal_id,
            "status": "NEW",
            "placed_at": datetime.now(UTC)
        }
        
        return order_id
        
    async def _process_order_updates(self):
        """Process order updates from WebSocket."""
        while self.running:
            try:
                # WebSocket client publishes ExecutionReportEvent directly
                # This method handles any additional processing needed
                
                # Check for stale orders (fallback to REST)
                await self._check_stale_orders()
                
                await asyncio.sleep(1)
                
            except Exception:
                self.logger.exception(
                    "Error processing order updates",
                    source_module=self._source_module
                )
                
    async def _check_stale_orders(self):
        """Check for orders without recent updates."""
        stale_threshold = datetime.now(UTC) - timedelta(seconds=30)
        
        for order_id, order_info in self.active_orders.items():
            if (order_info["status"] not in ["CLOSED", "CANCELLED", "EXPIRED"] and
                order_info.get("last_update", order_info["placed_at"]) < stale_threshold):
                
                # Fallback to REST API query
                self.logger.warning(
                    f"Order {order_id} stale, querying via REST",
                    source_module=self._source_module
                )
                
                await self._query_order_status_rest(order_id)
```

#### 2.2 Market Data Integration
```python
# gal_friday/data_ingestion/websocket_market_data.py
"""WebSocket market data ingestion."""

from typing import List, Set
import asyncio

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.core.pubsub import PubSubManager
from gal_friday.execution.websocket_client import KrakenWebSocketClient


class WebSocketMarketDataService:
    """Market data ingestion via WebSocket."""
    
    def __init__(self,
                 config: ConfigManager,
                 pubsub: PubSubManager,
                 logger: LoggerService):
        self.config = config
        self.pubsub = pubsub
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # WebSocket client
        self.ws_client = KrakenWebSocketClient(config, pubsub, logger)
        
        # Subscriptions
        self.pairs: Set[str] = set(config.get_list("trading.pairs", ["XRP/USD", "DOGE/USD"]))
        self.channels = ["book", "ticker", "trade", "ohlc"]
        
    async def start(self):
        """Start market data service."""
        self.logger.info(
            "Starting WebSocket market data service",
            source_module=self._source_module
        )
        
        # Connect WebSocket
        await self.ws_client.connect()
        
        # Subscribe to market data
        await self.ws_client.subscribe_market_data(
            list(self.pairs),
            self.channels
        )
        
    async def stop(self):
        """Stop market data service."""
        await self.ws_client.disconnect()
        
    async def add_pair(self, pair: str):
        """Add a trading pair subscription."""
        if pair not in self.pairs:
            self.pairs.add(pair)
            await self.ws_client.subscribe_market_data([pair], self.channels)
            
    async def remove_pair(self, pair: str):
        """Remove a trading pair subscription."""
        if pair in self.pairs:
            self.pairs.remove(pair)
            # Note: Kraken doesn't support unsubscribe, would need to reconnect
```

### 3. Connection Resilience and Recovery

#### 3.1 Connection Manager
```python
# gal_friday/execution/websocket_connection_manager.py
"""WebSocket connection management and resilience."""

import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from gal_friday.logger_service import LoggerService


class ConnectionHealth(Enum):
    """Connection health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ConnectionMetrics:
    """WebSocket connection metrics."""
    messages_received: int = 0
    messages_sent: int = 0
    last_message_time: Optional[datetime] = None
    connection_time: Optional[datetime] = None
    disconnections: int = 0
    errors: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate connection uptime."""
        if self.connection_time:
            return (datetime.now(UTC) - self.connection_time).total_seconds()
        return 0
        
    @property
    def message_rate(self) -> float:
        """Calculate messages per second."""
        if self.uptime_seconds > 0:
            return self.messages_received / self.uptime_seconds
        return 0


class WebSocketConnectionManager:
    """Manages WebSocket connection health and recovery."""
    
    def __init__(self, logger: LoggerService):
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Connection tracking
        self.connections: Dict[str, ConnectionMetrics] = {}
        self.health_check_interval = 10.0
        self.unhealthy_threshold = 30.0  # No messages for 30 seconds
        
        # Recovery strategies
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 5
        
    def register_connection(self, connection_id: str):
        """Register a new connection for monitoring."""
        self.connections[connection_id] = ConnectionMetrics(
            connection_time=datetime.now(UTC)
        )
        self.recovery_attempts[connection_id] = 0
        
    def record_message(self, connection_id: str, direction: str = "received"):
        """Record message activity."""
        if connection_id in self.connections:
            metrics = self.connections[connection_id]
            
            if direction == "received":
                metrics.messages_received += 1
            else:
                metrics.messages_sent += 1
                
            metrics.last_message_time = datetime.now(UTC)
            
    def record_error(self, connection_id: str):
        """Record connection error."""
        if connection_id in self.connections:
            self.connections[connection_id].errors += 1
            
    def record_disconnection(self, connection_id: str):
        """Record disconnection event."""
        if connection_id in self.connections:
            self.connections[connection_id].disconnections += 1
            self.recovery_attempts[connection_id] += 1
            
    def check_health(self, connection_id: str) -> ConnectionHealth:
        """Check connection health."""
        if connection_id not in self.connections:
            return ConnectionHealth.UNHEALTHY
            
        metrics = self.connections[connection_id]
        
        # Check last message time
        if metrics.last_message_time:
            time_since_message = (datetime.now(UTC) - metrics.last_message_time).total_seconds()
            
            if time_since_message > self.unhealthy_threshold:
                return ConnectionHealth.UNHEALTHY
            elif time_since_message > self.unhealthy_threshold / 2:
                return ConnectionHealth.DEGRADED
                
        # Check error rate
        if metrics.messages_received > 0:
            error_rate = metrics.errors / metrics.messages_received
            if error_rate > 0.1:  # More than 10% errors
                return ConnectionHealth.DEGRADED
                
        return ConnectionHealth.HEALTHY
        
    def should_reconnect(self, connection_id: str) -> bool:
        """Determine if connection should be reconnected."""
        attempts = self.recovery_attempts.get(connection_id, 0)
        return attempts < self.max_recovery_attempts
        
    def reset_recovery_attempts(self, connection_id: str):
        """Reset recovery attempts after successful reconnection."""
        self.recovery_attempts[connection_id] = 0
        
    async def monitor_connections(self):
        """Monitor all connections and trigger recovery."""
        while True:
            try:
                for conn_id, metrics in self.connections.items():
                    health = self.check_health(conn_id)
                    
                    if health == ConnectionHealth.UNHEALTHY:
                        self.logger.warning(
                            f"Connection {conn_id} is unhealthy",
                            source_module=self._source_module,
                            context={
                                "metrics": {
                                    "uptime": metrics.uptime_seconds,
                                    "messages": metrics.messages_received,
                                    "errors": metrics.errors
                                }
                            }
                        )
                        
                await asyncio.sleep(self.health_check_interval)
                
            except Exception:
                self.logger.exception(
                    "Error monitoring connections",
                    source_module=self._source_module
                )
```

### 4. Testing Infrastructure

#### 4.1 WebSocket Mock Server
```python
# tests/mocks/mock_websocket_server.py
"""Mock WebSocket server for testing."""

import asyncio
import websockets
import json
from typing import Dict, List, Set
import uuid


class MockKrakenWebSocketServer:
    """Mock Kraken WebSocket server for testing."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.subscriptions: Dict[str, Set[str]] = {}
        self.order_updates: asyncio.Queue = asyncio.Queue()
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        
        try:
            # Send system status
            await websocket.send(json.dumps({
                "event": "systemStatus",
                "connectionID": str(uuid.uuid4()),
                "status": "online",
                "version": "1.0.0"
            }))
            
            # Handle messages
            async for message in websocket:
                await self.process_message(websocket, message)
                
        finally:
            self.clients.remove(websocket)
            
    async def process_message(self, websocket, message: str):
        """Process client messages."""
        try:
            data = json.loads(message)
            
            if data.get("event") == "subscribe":
                # Handle subscription
                channel = data["subscription"]["name"]
                pairs = data.get("pair", [])
                
                await websocket.send(json.dumps({
                    "event": "subscriptionStatus",
                    "channelName": channel,
                    "pair": pairs,
                    "status": "subscribed",
                    "subscription": data["subscription"]
                }))
                
                # Start sending mock data
                if channel == "book":
                    asyncio.create_task(self.send_orderbook_updates(websocket, pairs))
                elif channel == "openOrders":
                    asyncio.create_task(self.send_order_updates(websocket))
                    
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "event": "error",
                "errorMessage": "Invalid JSON"
            }))
            
    async def send_orderbook_updates(self, websocket, pairs: List[str]):
        """Send mock orderbook updates."""
        channel_id = 100
        
        while websocket in self.clients:
            for pair in pairs:
                # Send snapshot
                message = [
                    channel_id,
                    {
                        "bs": [["0.5000", "1000"], ["0.4999", "2000"]],
                        "as": [["0.5001", "1000"], ["0.5002", "2000"]]
                    },
                    "book-10",
                    pair
                ]
                
                await websocket.send(json.dumps(message))
                
            await asyncio.sleep(1)
            
    async def send_order_updates(self, websocket):
        """Send mock order updates."""
        while websocket in self.clients:
            try:
                # Check for queued updates
                update = await asyncio.wait_for(
                    self.order_updates.get(), 
                    timeout=1.0
                )
                
                await websocket.send(json.dumps(update))
                
            except asyncio.TimeoutError:
                continue
                
    async def simulate_order_fill(self, order_id: str):
        """Simulate an order fill."""
        update = [
            [{
                "orderid": order_id,
                "status": "closed",
                "vol_exec": "1000",
                "avg_price": "0.5000",
                "descr": {
                    "pair": "XRP/USD",
                    "type": "buy"
                }
            }],
            "openOrders"
        ]
        
        await self.order_updates.put(update)
        
    async def start(self):
        """Start mock server."""
        async with websockets.serve(self.handler, "localhost", self.port):
            await asyncio.Future()  # Run forever
```

#### 4.2 WebSocket Integration Tests
```python
# tests/test_websocket_integration.py
"""Integration tests for WebSocket functionality."""

import pytest
import asyncio
import uuid
from decimal import Decimal

from tests.mocks.mock_websocket_server import MockKrakenWebSocketServer
from gal_friday.execution.websocket_client import KrakenWebSocketClient


class TestWebSocketIntegration:
    """Test WebSocket integration."""
    
    @pytest.fixture
    async def mock_server(self):
        """Start mock WebSocket server."""
        server = MockKrakenWebSocketServer()
        server_task = asyncio.create_task(server.start())
        
        yield server
        
        server_task.cancel()
        
    @pytest.mark.asyncio
    async def test_connection_establishment(self, mock_server, test_config, pubsub_manager, mock_logger):
        """Test WebSocket connection establishment."""
        # Configure for mock server
        test_config.config["kraken"]["ws_url"] = "ws://localhost:8765"
        
        client = KrakenWebSocketClient(test_config, pubsub_manager, mock_logger)
        
        await client.connect()
        await asyncio.sleep(0.5)  # Allow connection
        
        assert client.state == ConnectionState.CONNECTED
        
        await client.disconnect()
        
    @pytest.mark.asyncio
    async def test_market_data_subscription(self, mock_server, test_config, pubsub_manager, mock_logger):
        """Test market data subscription."""
        test_config.config["kraken"]["ws_url"] = "ws://localhost:8765"
        
        client = KrakenWebSocketClient(test_config, pubsub_manager, mock_logger)
        await client.connect()
        
        # Subscribe to market data
        await client.subscribe_market_data(["XRP/USD"], ["book"])
        
        # Wait for data
        await asyncio.sleep(2)
        
        # Should have received orderbook updates
        # Check via pubsub events
        
        await client.disconnect()
        
    @pytest.mark.asyncio
    async def test_order_update_processing(self, mock_server, test_config, pubsub_manager, mock_logger):
        """Test order update processing."""
        order_updates = []
        
        async def capture_updates(event):
            if isinstance(event, ExecutionReportEvent):
                order_updates.append(event)
                
        pubsub_manager.subscribe(EventType.EXECUTION_REPORT, capture_updates)
        
        test_config.config["kraken"]["ws_auth_url"] = "ws://localhost:8765"
        
        client = KrakenWebSocketClient(test_config, pubsub_manager, mock_logger)
        await client.connect()
        
        # Simulate order fill
        order_id = "TEST-ORDER-123"
        await mock_server.simulate_order_fill(order_id)
        
        await asyncio.sleep(0.5)
        
        # Check received update
        assert len(order_updates) > 0
        assert order_updates[0].exchange_order_id == order_id
        assert order_updates[0].order_status == "CLOSED"
        
        await client.disconnect()
        
    @pytest.mark.asyncio
    async def test_connection_recovery(self, test_config, pubsub_manager, mock_logger):
        """Test connection recovery after disconnect."""
        # This would test reconnection logic
        # Implementation depends on being able to simulate disconnects
        pass
```

## Implementation Steps

### Phase 1: Core WebSocket Client (3 days)
1. Implement base WebSocket client
2. Add authentication for private channels
3. Create message handlers for each channel type
4. Add connection state management

### Phase 2: Message Processing (2 days)
1. Implement sequence tracking
2. Add message caching and deduplication
3. Create gap detection and recovery
4. Add message validation

### Phase 3: System Integration (3 days)
1. Update ExecutionHandler for WebSocket
2. Integrate market data streaming
3. Update monitoring for WebSocket metrics
4. Add fallback to REST API

### Phase 4: Resilience and Testing (2 days)
1. Implement connection health monitoring
2. Add automatic reconnection logic
3. Create mock WebSocket server
4. Write comprehensive tests

## Success Criteria

1. **Latency Reduction**: < 100ms order update latency (from 1-5 seconds)
2. **Reliability**: 99.9% message delivery rate
3. **Recovery Time**: Automatic reconnection within 5 seconds
4. **Message Ordering**: 100% correct sequence processing
5. **Resource Efficiency**: 90% reduction in API calls for order status

## Monitoring and Maintenance

1. **Connection Metrics**:
   - Connection uptime percentage
   - Message rates (in/out)
   - Reconnection frequency
   - Message latency

2. **Data Quality**:
   - Sequence gap count
   - Duplicate message rate
   - Validation failure rate

3. **Performance**:
   - CPU usage for message processing
   - Memory usage for message cache
   - Network bandwidth utilization

4. **Alerts**:
   - Connection health degradation
   - High message latency
   - Excessive reconnections
   - Sequence gaps detected 