"""Publish-subscribe manager implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from typing_extensions import Protocol

T = TypeVar("T")


class PubSubManager:
    """Manager for publish-subscribe pattern."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the pubsub manager."""
        self._subscribers: dict[str, set[Callable[..., Any]]] = {}

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Subscribe to an event type.

        Args:
            event_type: The event type to subscribe to
            callback: The callback function to call when the event is published
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = set()
        self._subscribers[event_type].add(callback)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: The event type to unsubscribe from
            callback: The callback function to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].discard(callback)

    def publish(self, event_type: str, *args, **kwargs) -> None:
        """Publish an event.

        Args:
            event_type: The event type to publish
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
        """
        if event_type in self._subscribers:
            for callback in list(self._subscribers[event_type]):
                callback(*args, **kwargs)


class PubSubManagerProtocol(Protocol):
    """Protocol defining the required interface for pubsub managers."""

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Subscribe to an event type."""
        ...

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Unsubscribe from an event type."""
        ...

    def publish(self, event_type: str, *args, **kwargs) -> None:
        """Publish an event."""
        ...
