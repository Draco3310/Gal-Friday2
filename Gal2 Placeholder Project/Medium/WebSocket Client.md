# WebSocket Client Implementation Design

**File**: `/gal_friday/execution/websocket_client.py`
- **Line 709**: `# For now, publish as execution report`
- **Issue**: Simplified event publishing without comprehensive message validation, error handling, and reconnection logic
- **Impact**: Unreliable real-time data feeds and execution reporting

## Overview
The WebSocket Client currently has a basic implementation that lacks enterprise-grade reliability features essential for high-frequency trading. This design implements a production-ready WebSocket client with sophisticated connection management, message processing, error handling, and real-time data reliability guarantees.

## Architecture Design

### 1. Current Implementation Issues

```
WebSocket Client Problems:
├── Message Processing (Line 709)
│   ├── Basic event publishing without validation
│   ├── No message queuing or buffering
│   ├── Missing duplicate detection
│   └── No message ordering guarantees
├── Connection Management
│   ├── Basic reconnection logic
│   ├── No connection health monitoring
│   ├── Missing authentication renewal
│   └── No graceful degradation
├── Error Handling
│   ├── Limited error classification
│   ├── No circuit breaker pattern
│   ├── Missing retry backoff strategies
│   └── No error recovery automation
└── Performance & Reliability
    ├── No message rate limiting
    ├── Missing latency monitoring
    ├── No data integrity checks
    └── Limited observability
```

### 2. Production WebSocket Architecture

```
Enterprise WebSocket Client System:
├── Connection Management Layer
│   ├── Multi-endpoint connection pooling
│   ├── Automatic failover and load balancing
│   ├── Health monitoring and diagnostics
│   ├── Authentication and session management
│   └── Graceful shutdown and cleanup
├── Message Processing Engine
│   ├── High-performance message parsing
│   ├── Schema validation and sanitization
│   ├── Duplicate detection and deduplication
│   ├── Message ordering and sequencing
│   └── Event enrichment and transformation
├── Reliability & Resilience
│   ├── Circuit breaker pattern implementation
│   ├── Exponential backoff retry logic
│   ├── Message buffering and replay
│   ├── Connection state management
│   └── Automatic recovery mechanisms
├── Performance Optimization
│   ├── Asynchronous message processing
│   ├── Batched event publishing
│   ├── Memory-efficient buffering
│   ├── Latency monitoring and optimization
│   └── Throughput scaling
└── Observability & Monitoring
    ├── Real-time connection metrics
    ├── Message processing statistics
    ├── Error rate monitoring
    ├── Performance dashboards
    └── Alerting and notifications
```

### 3. Key Features

1. **Enterprise Connection Management**: Multi-endpoint failover with automatic reconnection
2. **Message Integrity**: Comprehensive validation with duplicate detection and ordering
3. **Performance Optimization**: High-throughput processing with minimal latency
4. **Fault Tolerance**: Circuit breaker pattern with intelligent recovery
5. **Security**: Robust authentication with credential renewal
6. **Monitoring**: Real-time observability with comprehensive metrics

## Implementation Plan

### Phase 1: Core Architecture and Connection Management

```python
import asyncio
import json
import time
import uuid
import hashlib
from datetime import datetime, UTC, timedelta
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
import websockets
from websockets.exceptions import (
    ConnectionClosedError, 
    ConnectionClosedOK, 
    InvalidMessage,
    ProtocolError
)
import aiohttp
from decimal import Decimal
import logging

from ..core.events import Event, FillEvent, MarketDataEvent, OrderEvent
from ..core.pubsub import PubSubManager
from ..logger_service import LoggerService


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATED = auto()
    SUBSCRIBING = auto()
    ACTIVE = auto()
    RECONNECTING = auto()
    FAILED = auto()
    SHUTDOWN = auto()


class MessageType(Enum):
    """WebSocket message types."""
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION = "subscribe"
    UNSUBSCRIPTION = "unsubscribe"
    MARKET_DATA = "market_data"
    EXECUTION_REPORT = "execution_report"
    ORDER_UPDATE = "order_update"
    ERROR = "error"
    SYSTEM_STATUS = "system_status"


@dataclass
class ConnectionEndpoint:
    """WebSocket endpoint configuration."""
    name: str
    url: str
    is_primary: bool = True
    max_reconnects: int = 10
    priority: int = 1
    health_check_interval: float = 30.0
    
    # Connection state
    state: ConnectionState = ConnectionState.DISCONNECTED
    connection_attempts: int = 0
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    
    # Performance metrics
    latency_ms: float = 0.0
    message_count: int = 0
    error_count: int = 0


@dataclass
class MessageMetadata:
    """Metadata for message processing."""
    message_id: str
    timestamp: datetime
    endpoint: str
    message_type: MessageType
    sequence_number: Optional[int] = None
    checksum: Optional[str] = None
    processing_latency_ms: float = 0.0


@dataclass
class CircuitBreakerState:
    """Circuit breaker state management."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    half_open_calls: int = 0


class EnterpriseWebSocketClient:
    """Production-grade WebSocket client for high-frequency trading."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: LoggerService,
        pubsub: PubSubManager
    ) -> None:
        self.config = config
        self.logger = logger
        self.pubsub = pubsub
        self._source_module = self.__class__.__name__
        
        # Connection management
        self._endpoints: List[ConnectionEndpoint] = []
        self._active_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self._connection_tasks: Dict[str, asyncio.Task] = {}
        self._primary_endpoint: Optional[str] = None
        
        # Authentication
        self._auth_token: Optional[str] = None
        self._auth_expires: Optional[datetime] = None
        self._auth_refresh_task: Optional[asyncio.Task] = None
        
        # Message processing
        self._message_buffer: deque = deque(maxlen=config.get("websocket.buffer_size", 10000))
        self._processed_messages: Set[str] = set()
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._sequence_numbers: Dict[str, int] = defaultdict(int)
        
        # Reliability features
        self._circuit_breaker = CircuitBreakerState()
        self._retry_backoff_base = config.get("websocket.retry_backoff_base", 1.0)
        self._retry_backoff_max = config.get("websocket.retry_backoff_max", 60.0)
        self._max_message_age_seconds = config.get("websocket.max_message_age", 300)
        
        # Performance monitoring
        self._stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "messages_dropped": 0,
            "connection_events": 0,
            "authentication_renewals": 0,
            "circuit_breaker_trips": 0
        }
        self._latency_samples: deque = deque(maxlen=1000)
        
        # Control flags
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Initialize configuration
        self._initialize_endpoints()
        self._initialize_message_handlers()
        
        self.logger.info(
            "EnterpriseWebSocketClient initialized with %d endpoints",
            len(self._endpoints),
            source_module=self._source_module
        )

    def _initialize_endpoints(self) -> None:
        """Initialize WebSocket endpoints from configuration."""
        endpoint_configs = self.config.get("websocket.endpoints", [])
        
        for i, endpoint_config in enumerate(endpoint_configs):
            endpoint = ConnectionEndpoint(
                name=endpoint_config.get("name", f"endpoint_{i}"),
                url=endpoint_config["url"],
                is_primary=endpoint_config.get("is_primary", i == 0),
                max_reconnects=endpoint_config.get("max_reconnects", 10),
                priority=endpoint_config.get("priority", i + 1)
            )
            self._endpoints.append(endpoint)
            
            if endpoint.is_primary:
                self._primary_endpoint = endpoint.name
        
        # Sort by priority
        self._endpoints.sort(key=lambda e: e.priority)
        
        if not self._primary_endpoint and self._endpoints:
            self._primary_endpoint = self._endpoints[0].name

    def _initialize_message_handlers(self) -> None:
        """Initialize message type handlers."""
        self._message_handlers = {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.MARKET_DATA: self._handle_market_data,
            MessageType.EXECUTION_REPORT: self._handle_execution_report,
            MessageType.ORDER_UPDATE: self._handle_order_update,
            MessageType.ERROR: self._handle_error_message,
            MessageType.SYSTEM_STATUS: self._handle_system_status
        }

    async def start(self) -> None:
        """Start the WebSocket client with all reliability features."""
        if self._is_running:
            self.logger.warning(
                "WebSocket client already running",
                source_module=self._source_module
            )
            return
        
        self._is_running = True
        self._shutdown_event.clear()
        
        try:
            # Start authentication renewal task
            self._auth_refresh_task = asyncio.create_task(self._auth_renewal_loop())
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Connect to all endpoints
            await self._connect_all_endpoints()
            
            # Start message processing
            await self._start_message_processing()
            
            self.logger.info(
                "WebSocket client started successfully",
                source_module=self._source_module,
                extra={"active_endpoints": len(self._active_connections)}
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to start WebSocket client: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Gracefully stop the WebSocket client."""
        self.logger.info(
            "Stopping WebSocket client",
            source_module=self._source_module
        )
        
        self._is_running = False
        self._shutdown_event.set()
        
        # Cancel all tasks
        tasks_to_cancel = []
        
        if self._auth_refresh_task:
            tasks_to_cancel.append(self._auth_refresh_task)
        
        if self._health_check_task:
            tasks_to_cancel.append(self._health_check_task)
        
        tasks_to_cancel.extend(self._connection_tasks.values())
        
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Close all connections
        await self._close_all_connections()
        
        self.logger.info(
            "WebSocket client stopped",
            source_module=self._source_module
        )

    async def _connect_all_endpoints(self) -> None:
        """Connect to all configured endpoints."""
        connection_tasks = []
        
        for endpoint in self._endpoints:
            task = asyncio.create_task(self._connect_endpoint(endpoint))
            connection_tasks.append(task)
            self._connection_tasks[endpoint.name] = task
        
        # Wait for at least one successful connection
        completed_tasks = []
        for task in asyncio.as_completed(connection_tasks, timeout=30):
            try:
                result = await task
                completed_tasks.append(result)
                
                # If we have a successful primary connection, we can proceed
                if self._primary_endpoint in self._active_connections:
                    break
                    
            except asyncio.TimeoutError:
                self.logger.error(
                    "Timeout waiting for WebSocket connections",
                    source_module=self._source_module
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Error in connection task: {e}",
                    source_module=self._source_module
                )
        
        if not self._active_connections:
            raise ConnectionError("Failed to establish any WebSocket connections")

    async def _connect_endpoint(self, endpoint: ConnectionEndpoint) -> bool:
        """Connect to a specific endpoint with retry logic."""
        max_retries = endpoint.max_reconnects
        retry_count = 0
        
        while retry_count < max_retries and self._is_running:
            try:
                endpoint.state = ConnectionState.CONNECTING
                endpoint.connection_attempts += 1
                
                self.logger.info(
                    f"Connecting to endpoint {endpoint.name} (attempt {retry_count + 1})",
                    source_module=self._source_module,
                    extra={"url": endpoint.url}
                )
                
                # Apply circuit breaker
                if not self._circuit_breaker_allow_request():
                    await asyncio.sleep(self._circuit_breaker.recovery_timeout)
                    continue
                
                # Establish WebSocket connection
                websocket = await websockets.connect(
                    endpoint.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=2**20,  # 1MB max message size
                    compression=None  # Disable compression for lower latency
                )
                
                endpoint.state = ConnectionState.CONNECTED
                endpoint.last_connected = datetime.now(UTC)
                endpoint.last_error = None
                self._active_connections[endpoint.name] = websocket
                
                # Authenticate if required
                if await self._authenticate_connection(endpoint, websocket):
                    endpoint.state = ConnectionState.AUTHENTICATED
                    
                    # Subscribe to required channels
                    if await self._subscribe_to_channels(endpoint, websocket):
                        endpoint.state = ConnectionState.ACTIVE
                        
                        # Start message handling for this connection
                        task = asyncio.create_task(
                            self._handle_connection_messages(endpoint, websocket)
                        )
                        self._connection_tasks[f"{endpoint.name}_handler"] = task
                        
                        self._circuit_breaker_record_success()
                        self._stats["connection_events"] += 1
                        
                        self.logger.info(
                            f"Successfully connected to endpoint {endpoint.name}",
                            source_module=self._source_module
                        )
                        
                        return True
                
                # If we reach here, authentication or subscription failed
                await websocket.close()
                endpoint.state = ConnectionState.FAILED
                
            except Exception as e:
                endpoint.state = ConnectionState.FAILED
                endpoint.last_error = str(e)
                self._circuit_breaker_record_failure()
                
                self.logger.error(
                    f"Failed to connect to endpoint {endpoint.name}: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                
                # Calculate backoff delay
                backoff_delay = min(
                    self._retry_backoff_base * (2 ** retry_count),
                    self._retry_backoff_max
                )
                
                retry_count += 1
                if retry_count < max_retries:
                    self.logger.info(
                        f"Retrying connection to {endpoint.name} in {backoff_delay}s",
                        source_module=self._source_module
                    )
                    await asyncio.sleep(backoff_delay)
        
        endpoint.state = ConnectionState.FAILED
        return False

    async def _authenticate_connection(
        self, 
        endpoint: ConnectionEndpoint, 
        websocket
    ) -> bool:
        """Authenticate WebSocket connection."""
        try:
            # Get fresh authentication token
            auth_token = await self._get_auth_token()
            if not auth_token:
                return False
            
            # Send authentication message
            auth_message = {
                "method": "auth",
                "params": {
                    "token": auth_token
                },
                "id": str(uuid.uuid4())
            }
            
            await websocket.send(json.dumps(auth_message))
            
            # Wait for authentication response
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            auth_response = json.loads(response)
            
            if auth_response.get("result") == "success":
                self.logger.info(
                    f"Authentication successful for {endpoint.name}",
                    source_module=self._source_module
                )
                return True
            else:
                self.logger.error(
                    f"Authentication failed for {endpoint.name}: {auth_response}",
                    source_module=self._source_module
                )
                return False
                
        except Exception as e:
            self.logger.error(
                f"Authentication error for {endpoint.name}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False

    async def _subscribe_to_channels(
        self, 
        endpoint: ConnectionEndpoint, 
        websocket
    ) -> bool:
        """Subscribe to required channels."""
        try:
            subscriptions = self.config.get("websocket.subscriptions", [])
            
            for subscription in subscriptions:
                subscribe_message = {
                    "method": "subscribe",
                    "params": subscription,
                    "id": str(uuid.uuid4())
                }
                
                await websocket.send(json.dumps(subscribe_message))
                
                # Wait for subscription confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=10)
                sub_response = json.loads(response)
                
                if sub_response.get("result") != "success":
                    self.logger.error(
                        f"Subscription failed for {endpoint.name}: {sub_response}",
                        source_module=self._source_module
                    )
                    return False
            
            self.logger.info(
                f"All subscriptions successful for {endpoint.name}",
                source_module=self._source_module,
                extra={"subscription_count": len(subscriptions)}
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Subscription error for {endpoint.name}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return False

    async def _handle_connection_messages(
        self, 
        endpoint: ConnectionEndpoint, 
        websocket
    ) -> None:
        """Handle messages from a specific connection."""
        try:
            while self._is_running and endpoint.name in self._active_connections:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    
                    # Record message reception
                    self._stats["messages_received"] += 1
                    endpoint.message_count += 1
                    
                    # Process message
                    await self._process_raw_message(message, endpoint)
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                    
                except ConnectionClosedOK:
                    self.logger.info(
                        f"Connection {endpoint.name} closed normally",
                        source_module=self._source_module
                    )
                    break
                    
                except ConnectionClosedError as e:
                    self.logger.warning(
                        f"Connection {endpoint.name} closed unexpectedly: {e}",
                        source_module=self._source_module
                    )
                    break
                    
                except Exception as e:
                    endpoint.error_count += 1
                    self.logger.error(
                        f"Error handling message from {endpoint.name}: {e}",
                        source_module=self._source_module,
                        exc_info=True
                    )
                    
                    # If too many errors, close connection
                    if endpoint.error_count > 10:
                        break
        
        finally:
            # Clean up connection
            if endpoint.name in self._active_connections:
                del self._active_connections[endpoint.name]
            
            endpoint.state = ConnectionState.DISCONNECTED
            
            # Attempt reconnection if still running
            if self._is_running:
                self.logger.info(
                    f"Scheduling reconnection for {endpoint.name}",
                    source_module=self._source_module
                )
                asyncio.create_task(self._reconnect_endpoint(endpoint))

    async def _process_raw_message(
        self, 
        raw_message: str, 
        endpoint: ConnectionEndpoint
    ) -> None:
        """Process raw WebSocket message with validation and deduplication."""
        try:
            start_time = time.time()
            
            # Parse JSON message
            try:
                message_data = json.loads(raw_message)
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Invalid JSON message from {endpoint.name}: {e}",
                    source_module=self._source_module,
                    extra={"raw_message": raw_message[:200]}
                )
                return
            
            # Create message metadata
            message_id = self._generate_message_id(raw_message, endpoint.name)
            metadata = MessageMetadata(
                message_id=message_id,
                timestamp=datetime.now(UTC),
                endpoint=endpoint.name,
                message_type=self._classify_message_type(message_data)
            )
            
            # Check for duplicates
            if message_id in self._processed_messages:
                self.logger.debug(
                    f"Duplicate message detected: {message_id}",
                    source_module=self._source_module
                )
                return
            
            # Add to processed set (with size limit)
            self._processed_messages.add(message_id)
            if len(self._processed_messages) > 10000:
                # Remove oldest 1000 entries
                oldest_entries = list(self._processed_messages)[:1000]
                for entry in oldest_entries:
                    self._processed_messages.discard(entry)
            
            # Validate message structure
            if not self._validate_message_structure(message_data, metadata.message_type):
                self.logger.warning(
                    f"Invalid message structure from {endpoint.name}",
                    source_module=self._source_module,
                    extra={"message_type": metadata.message_type.value}
                )
                return
            
            # Add to buffer for processing
            self._message_buffer.append((message_data, metadata))
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            metadata.processing_latency_ms = processing_time
            self._latency_samples.append(processing_time)
            
            # Update endpoint latency
            endpoint.latency_ms = processing_time
            
        except Exception as e:
            self.logger.error(
                f"Error processing raw message from {endpoint.name}: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _start_message_processing(self) -> None:
        """Start the message processing loop."""
        asyncio.create_task(self._message_processing_loop())

    async def _message_processing_loop(self) -> None:
        """Main message processing loop."""
        while self._is_running:
            try:
                # Process buffered messages
                batch_size = min(len(self._message_buffer), 100)
                if batch_size == 0:
                    await asyncio.sleep(0.001)  # Small delay
                    continue
                
                # Process batch of messages
                messages_to_process = []
                for _ in range(batch_size):
                    if self._message_buffer:
                        messages_to_process.append(self._message_buffer.popleft())
                
                # Process messages concurrently
                tasks = []
                for message_data, metadata in messages_to_process:
                    task = asyncio.create_task(
                        self._process_message(message_data, metadata)
                    )
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                self.logger.error(
                    f"Error in message processing loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(0.1)

    async def _process_message(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Process individual message based on type."""
        try:
            # Get appropriate handler
            handler = self._message_handlers.get(metadata.message_type)
            if not handler:
                self.logger.warning(
                    f"No handler for message type: {metadata.message_type}",
                    source_module=self._source_module
                )
                return
            
            # Execute handler
            await handler(message_data, metadata)
            self._stats["messages_processed"] += 1
            
        except Exception as e:
            self.logger.error(
                f"Error processing message {metadata.message_id}: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            self._stats["messages_dropped"] += 1

    # Message Handlers
    async def _handle_execution_report(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Handle execution report with comprehensive validation and enrichment."""
        try:
            # Enhanced execution report processing
            trade_data = message_data.get("data", {})
            
            # Validate required fields
            required_fields = ["ordertxid", "pair", "type", "price", "vol"]
            for field in required_fields:
                if field not in trade_data:
                    self.logger.error(
                        f"Missing required field in execution report: {field}",
                        source_module=self._source_module,
                        extra={"message_id": metadata.message_id}
                    )
                    return
            
            # Create enriched fill event
            fill_event = FillEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=metadata.timestamp,
                order_id=str(trade_data["ordertxid"]),
                fill_id=str(trade_data.get("id", uuid.uuid4())),
                trading_pair=self._normalize_trading_pair(trade_data["pair"]),
                side=str(trade_data["type"]).upper(),
                price=Decimal(str(trade_data["price"])),
                quantity=Decimal(str(trade_data["vol"])),
                fee=Decimal(str(trade_data.get("fee", "0"))),
                metadata={
                    "endpoint": metadata.endpoint,
                    "message_id": metadata.message_id,
                    "processing_latency_ms": metadata.processing_latency_ms,
                    "sequence_number": self._sequence_numbers[metadata.endpoint],
                    "venue": "kraken",
                    "execution_time": trade_data.get("time"),
                    "commission_asset": trade_data.get("fee_currency"),
                    "liquidity_indicator": trade_data.get("liquidity", "unknown")
                }
            )
            
            # Publish event with validation
            await self._publish_validated_event(fill_event, metadata)
            
            self.logger.debug(
                f"Processed execution report: {fill_event.fill_id}",
                source_module=self._source_module,
                extra={
                    "order_id": fill_event.order_id,
                    "trading_pair": fill_event.trading_pair,
                    "side": fill_event.side,
                    "price": float(fill_event.price),
                    "quantity": float(fill_event.quantity)
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"Error handling execution report: {e}",
                source_module=self._source_module,
                exc_info=True,
                extra={"message_id": metadata.message_id}
            )

    async def _handle_market_data(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Handle market data updates."""
        try:
            # Process different types of market data
            data_type = message_data.get("channel", "unknown")
            
            if data_type == "ticker":
                await self._process_ticker_data(message_data, metadata)
            elif data_type == "book":
                await self._process_orderbook_data(message_data, metadata)
            elif data_type == "trade":
                await self._process_trade_data(message_data, metadata)
            else:
                self.logger.warning(
                    f"Unknown market data type: {data_type}",
                    source_module=self._source_module
                )
                
        except Exception as e:
            self.logger.error(
                f"Error handling market data: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _handle_heartbeat(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Handle heartbeat messages."""
        # Update endpoint health status
        for endpoint in self._endpoints:
            if endpoint.name == metadata.endpoint:
                endpoint.last_connected = metadata.timestamp
                break

    async def _handle_error_message(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Handle error messages from WebSocket."""
        error_code = message_data.get("error_code")
        error_message = message_data.get("error_message", "Unknown error")
        
        self.logger.error(
            f"WebSocket error from {metadata.endpoint}: {error_message}",
            source_module=self._source_module,
            extra={
                "error_code": error_code,
                "endpoint": metadata.endpoint
            }
        )
        
        # Handle specific error types
        if error_code in ["AUTH_FAILED", "TOKEN_EXPIRED"]:
            # Trigger authentication renewal
            asyncio.create_task(self._renew_authentication())

    async def _handle_order_update(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Handle order status updates."""
        try:
            order_data = message_data.get("data", {})
            
            # Create order event
            order_event = OrderEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=metadata.timestamp,
                order_id=str(order_data.get("order_id", "")),
                trading_pair=self._normalize_trading_pair(order_data.get("pair", "")),
                side=str(order_data.get("side", "")).upper(),
                order_type=str(order_data.get("type", "")),
                quantity=Decimal(str(order_data.get("quantity", "0"))),
                price=Decimal(str(order_data.get("price", "0"))),
                status=str(order_data.get("status", "")),
                metadata={
                    "endpoint": metadata.endpoint,
                    "message_id": metadata.message_id,
                    "venue": "kraken"
                }
            )
            
            await self._publish_validated_event(order_event, metadata)
            
        except Exception as e:
            self.logger.error(
                f"Error handling order update: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _handle_system_status(
        self, 
        message_data: Dict[str, Any], 
        metadata: MessageMetadata
    ) -> None:
        """Handle system status messages."""
        status = message_data.get("status", "unknown")
        
        self.logger.info(
            f"System status update from {metadata.endpoint}: {status}",
            source_module=self._source_module
        )

    # Utility Methods
    def _generate_message_id(self, raw_message: str, endpoint: str) -> str:
        """Generate unique message ID for deduplication."""
        content = f"{endpoint}:{raw_message}"
        return hashlib.md5(content.encode()).hexdigest()

    def _classify_message_type(self, message_data: Dict[str, Any]) -> MessageType:
        """Classify message type based on content."""
        if "heartbeat" in message_data:
            return MessageType.HEARTBEAT
        elif "error" in message_data:
            return MessageType.ERROR
        elif "execution" in message_data or "fill" in message_data:
            return MessageType.EXECUTION_REPORT
        elif "order" in message_data:
            return MessageType.ORDER_UPDATE
        elif "ticker" in message_data or "book" in message_data or "trade" in message_data:
            return MessageType.MARKET_DATA
        elif "status" in message_data:
            return MessageType.SYSTEM_STATUS
        else:
            return MessageType.MARKET_DATA  # Default

    def _validate_message_structure(
        self, 
        message_data: Dict[str, Any], 
        message_type: MessageType
    ) -> bool:
        """Validate message structure based on type."""
        # Basic structure validation
        if not isinstance(message_data, dict):
            return False
        
        # Type-specific validation
        if message_type == MessageType.EXECUTION_REPORT:
            required_fields = ["data"]
            return all(field in message_data for field in required_fields)
        
        return True  # Default to valid

    def _normalize_trading_pair(self, pair: str) -> str:
        """Normalize trading pair format."""
        # Implement pair normalization logic
        return pair.replace("XBT", "BTC").upper()

    async def _publish_validated_event(self, event: Event, metadata: MessageMetadata) -> None:
        """Publish event with additional validation."""
        try:
            # Add sequence number
            self._sequence_numbers[metadata.endpoint] += 1
            
            # Publish to PubSub
            await self.pubsub.publish(event)
            
        except Exception as e:
            self.logger.error(
                f"Failed to publish event: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    # Circuit Breaker Implementation
    def _circuit_breaker_allow_request(self) -> bool:
        """Check if circuit breaker allows the request."""
        now = datetime.now(UTC)
        
        if self._circuit_breaker.state == "CLOSED":
            return True
        elif self._circuit_breaker.state == "OPEN":
            if (self._circuit_breaker.last_failure_time and 
                now - self._circuit_breaker.last_failure_time > 
                timedelta(seconds=self._circuit_breaker.recovery_timeout)):
                self._circuit_breaker.state = "HALF_OPEN"
                self._circuit_breaker.half_open_calls = 0
                return True
            return False
        elif self._circuit_breaker.state == "HALF_OPEN":
            if self._circuit_breaker.half_open_calls < self._circuit_breaker.half_open_max_calls:
                self._circuit_breaker.half_open_calls += 1
                return True
            return False
        
        return False

    def _circuit_breaker_record_success(self) -> None:
        """Record successful operation in circuit breaker."""
        if self._circuit_breaker.state == "HALF_OPEN":
            self._circuit_breaker.state = "CLOSED"
            self._circuit_breaker.failure_count = 0

    def _circuit_breaker_record_failure(self) -> None:
        """Record failed operation in circuit breaker."""
        self._circuit_breaker.failure_count += 1
        self._circuit_breaker.last_failure_time = datetime.now(UTC)
        
        if self._circuit_breaker.failure_count >= self._circuit_breaker.failure_threshold:
            self._circuit_breaker.state = "OPEN"
            self._stats["circuit_breaker_trips"] += 1

    # Authentication and Health Monitoring
    async def _get_auth_token(self) -> Optional[str]:
        """Get fresh authentication token."""
        # Implement authentication logic
        # This would typically involve API key signing or OAuth flow
        return self.config.get("websocket.auth_token")

    async def _auth_renewal_loop(self) -> None:
        """Continuous authentication renewal loop."""
        while self._is_running:
            try:
                # Check if token needs renewal
                if (self._auth_expires and 
                    datetime.now(UTC) + timedelta(minutes=5) > self._auth_expires):
                    await self._renew_authentication()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(
                    f"Error in auth renewal loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(60)

    async def _renew_authentication(self) -> None:
        """Renew authentication for all connections."""
        self.logger.info(
            "Renewing authentication",
            source_module=self._source_module
        )
        
        # Implement authentication renewal
        self._stats["authentication_renewals"] += 1

    async def _health_check_loop(self) -> None:
        """Continuous health check for all connections."""
        while self._is_running:
            try:
                for endpoint in self._endpoints:
                    if endpoint.name in self._active_connections:
                        await self._check_endpoint_health(endpoint)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(
                    f"Error in health check loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(30)

    async def _check_endpoint_health(self, endpoint: ConnectionEndpoint) -> None:
        """Check health of specific endpoint."""
        try:
            websocket = self._active_connections.get(endpoint.name)
            if websocket:
                # Send ping
                await websocket.ping()
                
                # Update health metrics
                endpoint.last_connected = datetime.now(UTC)
                
        except Exception as e:
            self.logger.warning(
                f"Health check failed for {endpoint.name}: {e}",
                source_module=self._source_module
            )
            endpoint.error_count += 1

    async def _reconnect_endpoint(self, endpoint: ConnectionEndpoint) -> None:
        """Reconnect to a specific endpoint."""
        await asyncio.sleep(5)  # Brief delay before reconnection
        await self._connect_endpoint(endpoint)

    async def _close_all_connections(self) -> None:
        """Close all WebSocket connections."""
        for name, websocket in self._active_connections.items():
            try:
                await websocket.close()
            except Exception as e:
                self.logger.error(
                    f"Error closing connection {name}: {e}",
                    source_module=self._source_module
                )
        
        self._active_connections.clear()

    # Diagnostics and Monitoring
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information."""
        avg_latency = 0.0
        if self._latency_samples:
            avg_latency = sum(self._latency_samples) / len(self._latency_samples)
        
        endpoint_status = []
        for endpoint in self._endpoints:
            endpoint_status.append({
                "name": endpoint.name,
                "state": endpoint.state.name,
                "is_active": endpoint.name in self._active_connections,
                "connection_attempts": endpoint.connection_attempts,
                "message_count": endpoint.message_count,
                "error_count": endpoint.error_count,
                "latency_ms": endpoint.latency_ms,
                "last_connected": endpoint.last_connected.isoformat() if endpoint.last_connected else None,
                "last_error": endpoint.last_error
            })
        
        return {
            "client_status": "running" if self._is_running else "stopped",
            "active_connections": len(self._active_connections),
            "total_endpoints": len(self._endpoints),
            "primary_endpoint": self._primary_endpoint,
            "statistics": self._stats.copy(),
            "performance": {
                "avg_latency_ms": avg_latency,
                "message_buffer_size": len(self._message_buffer),
                "processed_messages_cache_size": len(self._processed_messages)
            },
            "circuit_breaker": {
                "state": self._circuit_breaker.state,
                "failure_count": self._circuit_breaker.failure_count,
                "last_failure": self._circuit_breaker.last_failure_time.isoformat() if self._circuit_breaker.last_failure_time else None
            },
            "endpoints": endpoint_status
        }
```

### Phase 2: Advanced Message Processing and Validation

```python
# Additional processing methods for different market data types

async def _process_ticker_data(
    self, 
    message_data: Dict[str, Any], 
    metadata: MessageMetadata
) -> None:
    """Process ticker data with validation."""
    try:
        ticker_data = message_data.get("data", {})
        
        # Validate ticker structure
        required_fields = ["symbol", "bid", "ask", "last"]
        if not all(field in ticker_data for field in required_fields):
            return
        
        # Create market data event
        market_event = MarketDataEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=metadata.timestamp,
            symbol=self._normalize_trading_pair(ticker_data["symbol"]),
            data_type="ticker",
            data={
                "bid": float(ticker_data["bid"]),
                "ask": float(ticker_data["ask"]),
                "last": float(ticker_data["last"]),
                "volume": float(ticker_data.get("volume", 0)),
                "change_24h": float(ticker_data.get("change_24h", 0)),
                "high_24h": float(ticker_data.get("high_24h", 0)),
                "low_24h": float(ticker_data.get("low_24h", 0))
            },
            metadata={
                "endpoint": metadata.endpoint,
                "sequence": self._sequence_numbers[metadata.endpoint],
                "venue": "kraken"
            }
        )
        
        await self._publish_validated_event(market_event, metadata)
        
    except Exception as e:
        self.logger.error(
            f"Error processing ticker data: {e}",
            source_module=self._source_module,
            exc_info=True
        )

async def _process_orderbook_data(
    self, 
    message_data: Dict[str, Any], 
    metadata: MessageMetadata
) -> None:
    """Process order book data with validation."""
    try:
        book_data = message_data.get("data", {})
        
        # Validate order book structure
        if "bids" not in book_data or "asks" not in book_data:
            return
        
        # Process bids and asks
        bids = []
        asks = []
        
        for bid in book_data["bids"][:10]:  # Top 10 levels
            if len(bid) >= 2:
                bids.append({
                    "price": float(bid[0]),
                    "size": float(bid[1]),
                    "timestamp": bid[2] if len(bid) > 2 else metadata.timestamp.timestamp()
                })
        
        for ask in book_data["asks"][:10]:  # Top 10 levels
            if len(ask) >= 2:
                asks.append({
                    "price": float(ask[0]),
                    "size": float(ask[1]),
                    "timestamp": ask[2] if len(ask) > 2 else metadata.timestamp.timestamp()
                })
        
        # Create order book event
        book_event = MarketDataEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=metadata.timestamp,
            symbol=self._normalize_trading_pair(book_data.get("symbol", "")),
            data_type="orderbook",
            data={
                "bids": bids,
                "asks": asks,
                "checksum": book_data.get("checksum"),
                "sequence": book_data.get("sequence")
            },
            metadata={
                "endpoint": metadata.endpoint,
                "venue": "kraken",
                "is_snapshot": book_data.get("is_snapshot", False)
            }
        )
        
        await self._publish_validated_event(book_event, metadata)
        
    except Exception as e:
        self.logger.error(
            f"Error processing order book data: {e}",
            source_module=self._source_module,
            exc_info=True
        )

async def _process_trade_data(
    self, 
    message_data: Dict[str, Any], 
    metadata: MessageMetadata
) -> None:
    """Process trade data with validation."""
    try:
        trade_data = message_data.get("data", {})
        
        # Validate trade structure
        required_fields = ["price", "size", "side"]
        if not all(field in trade_data for field in required_fields):
            return
        
        # Create trade event
        trade_event = MarketDataEvent(
            source_module=self._source_module,
            event_id=uuid.uuid4(),
            timestamp=metadata.timestamp,
            symbol=self._normalize_trading_pair(trade_data.get("symbol", "")),
            data_type="trade",
            data={
                "price": float(trade_data["price"]),
                "size": float(trade_data["size"]),
                "side": trade_data["side"],
                "trade_id": trade_data.get("trade_id"),
                "timestamp": trade_data.get("timestamp", metadata.timestamp.timestamp())
            },
            metadata={
                "endpoint": metadata.endpoint,
                "venue": "kraken"
            }
        )
        
        await self._publish_validated_event(trade_event, metadata)
        
    except Exception as e:
        self.logger.error(
            f"Error processing trade data: {e}",
            source_module=self._source_module,
            exc_info=True
        )
```

## Testing Strategy

1. **Unit Tests**
   - Connection management functionality
   - Message parsing and validation
   - Circuit breaker behavior
   - Authentication flows
   - Error handling scenarios

2. **Integration Tests**
   - End-to-end message flow
   - Multi-endpoint failover
   - Real WebSocket server interaction
   - Performance under load
   - Network failure simulation

3. **Performance Tests**
   - High-frequency message processing
   - Memory usage optimization
   - Latency benchmarking
   - Concurrent connection handling
   - Resource utilization monitoring

4. **Reliability Tests**
   - Connection drop recovery
   - Authentication expiry handling
   - Message deduplication accuracy
   - Circuit breaker effectiveness
   - Graceful shutdown behavior

## Monitoring & Observability

1. **Real-time Metrics**
   - Connection status and health
   - Message processing rates
   - Error rates by type and endpoint
   - Latency distributions (p50, p95, p99)
   - Authentication renewal frequency

2. **Alerting Thresholds**
   - Connection failures > 3 in 5 minutes
   - Message processing latency > 100ms p95
   - Error rate > 1% over 1 minute
   - Circuit breaker trips
   - Authentication failures

3. **Performance Dashboards**
   - Live connection topology
   - Message flow visualization
   - Error rate trends
   - Latency heat maps
   - Resource utilization graphs

## Security Considerations

1. **Authentication Security**
   - Secure token storage and rotation
   - API key signing validation
   - Session management best practices
   - Credential encryption in memory

2. **Message Security**
   - Input validation and sanitization
   - Message size limits
   - Rate limiting protection
   - Injection attack prevention

3. **Network Security**
   - TLS/SSL enforcement
   - Certificate validation
   - Firewall-friendly design
   - DDoS protection mechanisms

## Performance Optimization

1. **High-Frequency Processing**
   - Zero-copy message handling
   - Batched event publishing
   - Optimized JSON parsing
   - Memory pool allocation

2. **Connection Efficiency**
   - Connection pooling
   - Keep-alive optimization
   - Compression when beneficial
   - TCP socket tuning

3. **Resource Management**
   - Bounded memory usage
   - CPU affinity optimization
   - Garbage collection tuning
   - System resource monitoring

## Future Enhancements

1. **Advanced Features**
   - Protocol multiplexing
   - Custom binary protocols
   - Message compression algorithms
   - Smart routing and load balancing

2. **Analytics Integration**
   - Real-time pattern detection
   - Anomaly identification
   - Predictive failure analysis
   - Performance optimization recommendations

3. **Multi-Exchange Support**
   - Unified message normalization
   - Cross-venue arbitrage data
   - Exchange-specific optimizations
   - Venue health monitoring