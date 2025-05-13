"""Compatibility module for legacy imports.

This module re-exports event classes from core.events with legacy names to maintain
backward compatibility with existing test code.
"""

# Re-export event classes with legacy names
from gal_friday.core.events import ExecutionReportEvent as FillEvent
from gal_friday.core.events import MarketDataL2Event as MarketDataEvent
from gal_friday.core.events import TradeSignalApprovedEvent as OrderEvent
from gal_friday.core.events import TradeSignalProposedEvent as SignalEvent

# Re-export other items that may be needed
from gal_friday.core.pubsub import PubSubManager as EventBus

__all__ = ["FillEvent", "MarketDataEvent", "OrderEvent", "SignalEvent", "EventBus"]
