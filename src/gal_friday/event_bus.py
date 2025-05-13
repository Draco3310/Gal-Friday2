"""Compatibility module for legacy imports.

This module re-exports event classes from core.events with legacy names to maintain
backward compatibility with existing test code.
"""

from typing import Any, Dict

# Re-export event classes with legacy names
from gal_friday.core.events import EventType
from gal_friday.core.events import ExecutionReportEvent as FillEvent
from gal_friday.core.events import MarketDataL2Event as MarketDataEvent
from gal_friday.core.events import TradeSignalApprovedEvent as OrderEvent
from gal_friday.core.events import TradeSignalProposedEvent as SignalEvent

# Re-export other items that may be needed
from gal_friday.core.pubsub import PubSubManager as EventBus


class BackpressureStrategy:
    """Base class for backpressure strategies to control event flow."""

    async def should_accept(self, event_type: EventType, queue_size: int) -> bool:
        """Determine if an event should be accepted based on event type and queue size.

        Args
        ----
            event_type: Type of the event
            queue_size: Current size of the event queue

        Returns
        -------
            True if the event should be accepted, False otherwise
        """
        return True  # Default implementation accepts all events


class SimpleThresholdBackpressure(BackpressureStrategy):
    """Threshold-based backpressure strategy with different levels per event type."""

    def __init__(self, config_manager: Any) -> None:
        """Initialize with config manager to access thresholds.

        Args
        ----
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.thresholds: Dict[int, int] = {}  # EventType.value -> queue size threshold
        self.default_threshold: int = 100  # Default threshold

    async def should_accept(self, event_type: EventType, queue_size: int) -> bool:
        """Implement threshold-based acceptance logic.

        Args
        ----
            event_type: Type of the event
            queue_size: Current size of the event queue

        Returns
        -------
            True if the event should be accepted, False otherwise
        """
        # Get threshold for this event type (by value)
        threshold = self.thresholds.get(event_type.value, self.default_threshold)
        return queue_size < threshold


__all__ = [
    "FillEvent",
    "MarketDataEvent",
    "OrderEvent",
    "SignalEvent",
    "EventBus",
    "BackpressureStrategy",
    "SimpleThresholdBackpressure",
]
