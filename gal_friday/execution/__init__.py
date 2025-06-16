"""Execution module for handling trade execution across supported exchanges.

This package contains implementations for various exchanges and provides
a consistent interface for order management across platforms.
"""

from .websocket_client import ConnectionState, KrakenWebSocketClient
from .websocket_connection_manager import ConnectionHealth, WebSocketConnectionManager
from .websocket_processor import MessageCache, SequenceTracker, WebSocketMessageProcessor

__all__ = [
    "ConnectionHealth",
    "ConnectionState",
    "KrakenWebSocketClient",
    "MessageCache",
    "SequenceTracker",
    "WebSocketConnectionManager",
    "WebSocketMessageProcessor",
]