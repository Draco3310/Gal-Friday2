"""WebSocket protocols for real-time market data streaming.

This module provides protocol definitions for WebSocket-based market data streaming,
enabling real-time data processing and event-driven architectures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from datetime import datetime


@runtime_checkable
class WebSocketConnectionProtocol(Protocol):
    """Protocol for WebSocket connection management.

    This protocol defines the interface for WebSocket connections,
    handling connection lifecycle and basic message operations.
    """

    async def connect(self, url: str, headers: dict[str, str] | None = None) -> None:
        """Establish WebSocket connection.

        Args:
            url: WebSocket endpoint URL.
            headers: Optional HTTP headers for connection.
        """
        ...

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        ...

    async def send(self, message: str | bytes) -> None:
        """Send message through WebSocket.

        Args:
            message: Message to send (text or binary).
        """
        ...

    async def receive(self) -> str | bytes:
        """Receive message from WebSocket.

        Returns:
            Received message (text or binary).
        """
        ...

    async def ping(self) -> None:
        """Send ping frame to keep connection alive."""
        ...

    async def pong(self, data: bytes = b"") -> None:
        """Send pong frame in response to ping.

        Args:
            data: Optional pong data.
        """
        ...

    def is_connected(self) -> bool:
        """Check if connection is active.

        Returns:
            True if connected, False otherwise.
        """
        ...

    def get_connection_info(self) -> dict[str, Any]:
        """Get connection information.

        Returns:
            Dictionary containing connection details.
        """
        ...


@runtime_checkable
class WebSocketMessageHandlerProtocol(Protocol):
    """Protocol for WebSocket message handling.

    This protocol defines the interface for processing WebSocket messages
    and converting them to appropriate data structures.
    """

    async def handle_message(self, message: str | bytes) -> dict[str, Any] | None:
        """Process incoming WebSocket message.

        Args:
            message: Raw message from WebSocket.

        Returns:
            Processed message as dictionary or None if invalid.
        """
        ...

    async def handle_error(self, error: Exception) -> None:
        """Handle WebSocket errors.

        Args:
            error: Exception that occurred.
        """
        ...

    async def handle_close(self, code: int, reason: str) -> None:
        """Handle WebSocket connection close.

        Args:
            code: Close code.
            reason: Close reason.
        """
        ...

    def get_message_types(self) -> list[str]:
        """Get list of supported message types.

        Returns:
            List of message type identifiers.
        """
        ...

    def can_handle_message(self, message: dict[str, Any]) -> bool:
        """Check if handler can process message type.

        Args:
            message: Message to check.

        Returns:
            True if handler can process message, False otherwise.
        """
        ...


@runtime_checkable
class WebSocketStreamProtocol(Protocol):
    """Protocol for WebSocket streaming services.

    This protocol defines the interface for WebSocket streaming services
    that provide real-time data streams.
    """

    async def start_stream(self, symbols: list[str]) -> None:
        """Start streaming for specified symbols.

        Args:
            symbols: List of symbols to stream.
        """
        ...

    async def stop_stream(self) -> None:
        """Stop all streaming."""
        ...

    async def subscribe(self, symbol: str, channel: str) -> None:
        """Subscribe to a specific channel for a symbol.

        Args:
            symbol: Symbol to subscribe to.
            channel: Channel type (e.g., "ticker", "trades", "book").
        """
        ...

    async def unsubscribe(self, symbol: str, channel: str) -> None:
        """Unsubscribe from a specific channel for a symbol.

        Args:
            symbol: Symbol to unsubscribe from.
            channel: Channel type.
        """
        ...

    async def get_stream_data(self) -> AsyncIterator[dict[str, Any]]:
        """Get async iterator of streaming data.

        Yields:
            Streaming data as dictionaries.
        """
        ...

    def is_streaming(self) -> bool:
        """Check if streaming is active.

        Returns:
            True if streaming, False otherwise.
        """
        ...

    def get_active_subscriptions(self) -> dict[str, list[str]]:
        """Get active subscriptions.

        Returns:
            Dictionary mapping symbols to list of channels.
        """
        ...

    def get_stream_statistics(self) -> dict[str, Any]:
        """Get streaming statistics.

        Returns:
            Dictionary containing streaming statistics.
        """
        ...


@runtime_checkable
class WebSocketReconnectProtocol(Protocol):
    """Protocol for WebSocket reconnection handling.

    This protocol defines the interface for handling WebSocket reconnection
    logic and maintaining connection resilience.
    """

    async def enable_reconnect(self, max_retries: int = -1, retry_delay: float = 1.0) -> None:
        """Enable automatic reconnection.

        Args:
            max_retries: Maximum number of retry attempts (-1 for infinite).
            retry_delay: Delay between retry attempts in seconds.
        """
        ...

    async def disable_reconnect(self) -> None:
        """Disable automatic reconnection."""
        ...

    async def force_reconnect(self) -> None:
        """Force immediate reconnection."""
        ...

    def is_reconnect_enabled(self) -> bool:
        """Check if reconnection is enabled.

        Returns:
            True if reconnection is enabled, False otherwise.
        """
        ...

    def get_reconnect_stats(self) -> dict[str, Any]:
        """Get reconnection statistics.

        Returns:
            Dictionary containing reconnection statistics.
        """
        ...

    async def on_connection_lost(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for connection lost events.

        Args:
            callback: Function to call when connection is lost.
        """
        ...

    async def on_reconnect_success(self, callback: Callable[[], None]) -> None:
        """Register callback for successful reconnection.

        Args:
            callback: Function to call when reconnection succeeds.
        """
        ...

    async def on_reconnect_failed(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for failed reconnection.

        Args:
            callback: Function to call when reconnection fails.
        """
        ...


@runtime_checkable
class WebSocketAuthenticationProtocol(Protocol):
    """Protocol for WebSocket authentication.

    This protocol defines the interface for WebSocket authentication
    and authorization mechanisms.
    """

    async def authenticate(self, credentials: dict[str, Any]) -> bool:
        """Authenticate with WebSocket server.

        Args:
            credentials: Authentication credentials.

        Returns:
            True if authentication successful, False otherwise.
        """
        ...

    async def refresh_token(self) -> bool:
        """Refresh authentication token.

        Returns:
            True if token refresh successful, False otherwise.
        """
        ...

    def is_authenticated(self) -> bool:
        """Check if client is authenticated.

        Returns:
            True if authenticated, False otherwise.
        """
        ...

    def get_auth_status(self) -> dict[str, Any]:
        """Get authentication status.

        Returns:
            Dictionary containing authentication status.
        """
        ...

    async def logout(self) -> None:
        """Logout and invalidate authentication."""
        ...


@runtime_checkable
class WebSocketHeartbeatProtocol(Protocol):
    """Protocol for WebSocket heartbeat management.

    This protocol defines the interface for managing WebSocket heartbeat
    and keep-alive mechanisms.
    """

    async def start_heartbeat(self, interval: float = 30.0) -> None:
        """Start heartbeat with specified interval.

        Args:
            interval: Heartbeat interval in seconds.
        """
        ...

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat."""
        ...

    async def send_heartbeat(self) -> None:
        """Send heartbeat message."""
        ...

    def is_heartbeat_active(self) -> bool:
        """Check if heartbeat is active.

        Returns:
            True if heartbeat is active, False otherwise.
        """
        ...

    def get_last_heartbeat(self) -> datetime | None:
        """Get timestamp of last heartbeat.

        Returns:
            Datetime of last heartbeat or None if not available.
        """
        ...

    def get_heartbeat_stats(self) -> dict[str, Any]:
        """Get heartbeat statistics.

        Returns:
            Dictionary containing heartbeat statistics.
        """
        ...

    async def on_heartbeat_timeout(self, callback: Callable[[], None]) -> None:
        """Register callback for heartbeat timeout.

        Args:
            callback: Function to call on heartbeat timeout.
        """
        ...


@runtime_checkable
class WebSocketRateLimitProtocol(Protocol):
    """Protocol for WebSocket rate limiting.

    This protocol defines the interface for managing WebSocket rate limiting
    to comply with API limits and avoid throttling.
    """

    async def set_rate_limit(self, requests_per_second: float) -> None:
        """Set rate limit for outgoing messages.

        Args:
            requests_per_second: Maximum requests per second.
        """
        ...

    async def wait_for_rate_limit(self) -> None:
        """Wait if rate limit would be exceeded."""
        ...

    def is_rate_limited(self) -> bool:
        """Check if currently rate limited.

        Returns:
            True if rate limited, False otherwise.
        """
        ...

    def get_rate_limit_info(self) -> dict[str, Any]:
        """Get rate limit information.

        Returns:
            Dictionary containing rate limit info.
        """
        ...

    def get_remaining_requests(self) -> int:
        """Get remaining requests in current window.

        Returns:
            Number of remaining requests.
        """
        ...

    def get_reset_time(self) -> datetime | None:
        """Get time when rate limit resets.

        Returns:
            Datetime when rate limit resets or None.
        """
        ...


@runtime_checkable
class WebSocketMetricsProtocol(Protocol):
    """Protocol for WebSocket metrics collection.

    This protocol defines the interface for collecting and reporting
    WebSocket performance and usage metrics.
    """

    def record_message_sent(self, message_type: str, size: int) -> None:
        """Record outgoing message metrics.

        Args:
            message_type: Type of message sent.
            size: Size of message in bytes.
        """
        ...

    def record_message_received(self, message_type: str, size: int) -> None:
        """Record incoming message metrics.

        Args:
            message_type: Type of message received.
            size: Size of message in bytes.
        """
        ...

    def record_connection_event(self, event_type: str, duration: float | None = None) -> None:
        """Record connection event metrics.

        Args:
            event_type: Type of connection event.
            duration: Duration of event in seconds (if applicable).
        """
        ...

    def record_error(self, error_type: str, details: str | None = None) -> None:
        """Record error metrics.

        Args:
            error_type: Type of error.
            details: Additional error details.
        """
        ...

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary containing metrics summary.
        """
        ...

    def get_connection_metrics(self) -> dict[str, Any]:
        """Get connection-specific metrics.

        Returns:
            Dictionary containing connection metrics.
        """
        ...

    def get_message_metrics(self) -> dict[str, Any]:
        """Get message-specific metrics.

        Returns:
            Dictionary containing message metrics.
        """
        ...

    def get_error_metrics(self) -> dict[str, Any]:
        """Get error-specific metrics.

        Returns:
            Dictionary containing error metrics.
        """
        ...

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        ...


@runtime_checkable
class WebSocketOrderBookProtocol(Protocol):
    """Protocol for WebSocket order book streaming.

    This protocol defines the interface for streaming order book data
    and maintaining real-time order book state.
    """

    async def subscribe_order_book(self, symbol: str, depth: int = 10) -> None:
        """Subscribe to order book updates for a symbol.

        Args:
            symbol: Trading pair symbol.
            depth: Order book depth (number of levels).
        """
        ...

    async def unsubscribe_order_book(self, symbol: str) -> None:
        """Unsubscribe from order book updates.

        Args:
            symbol: Trading pair symbol.
        """
        ...

    async def get_order_book_stream(self, symbol: str) -> AsyncIterator[dict[str, Any]]:
        """Get order book update stream for a symbol.

        Args:
            symbol: Trading pair symbol.

        Yields:
            Order book update data.
        """
        ...

    def get_current_order_book(self, symbol: str) -> dict[str, Any] | None:
        """Get current order book snapshot.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Current order book data or None if not available.
        """
        ...

    def get_order_book_timestamp(self, symbol: str) -> datetime | None:
        """Get timestamp of last order book update.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Timestamp of last update or None.
        """
        ...

    def is_order_book_subscribed(self, symbol: str) -> bool:
        """Check if subscribed to order book for symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            True if subscribed, False otherwise.
        """
        ...

    def get_order_book_stats(self, symbol: str) -> dict[str, Any]:
        """Get order book statistics for symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dictionary containing order book statistics.
        """
        ...


@runtime_checkable
class WebSocketTradeStreamProtocol(Protocol):
    """Protocol for WebSocket trade streaming.

    This protocol defines the interface for streaming trade data
    and real-time trade execution information.
    """

    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade updates for a symbol.

        Args:
            symbol: Trading pair symbol.
        """
        ...

    async def unsubscribe_trades(self, symbol: str) -> None:
        """Unsubscribe from trade updates.

        Args:
            symbol: Trading pair symbol.
        """
        ...

    async def get_trade_stream(self, symbol: str) -> AsyncIterator[dict[str, Any]]:
        """Get trade update stream for a symbol.

        Args:
            symbol: Trading pair symbol.

        Yields:
            Trade data.
        """
        ...

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent trades for a symbol.

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of trades to return.

        Returns:
            List of recent trade data.
        """
        ...

    def get_trade_statistics(self, symbol: str) -> dict[str, Any]:
        """Get trade statistics for symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dictionary containing trade statistics.
        """
        ...

    def is_trades_subscribed(self, symbol: str) -> bool:
        """Check if subscribed to trades for symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            True if subscribed, False otherwise.
        """
        ...

    def get_last_trade_time(self, symbol: str) -> datetime | None:
        """Get timestamp of last trade.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Timestamp of last trade or None.
        """
        ...
