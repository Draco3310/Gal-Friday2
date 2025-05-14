"""Compatibility module for legacy imports.

This module re-exports event classes from core.events with legacy names to maintain
backward compatibility with existing test code.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, cast, overload

# Re-export event classes with legacy names
from gal_friday.core.events import EventType
from gal_friday.core.events import ExecutionReportEvent as ExecutionReportEventBase
from gal_friday.core.events import MarketDataL2Event as MarketDataL2EventBase
from gal_friday.core.events import TradeSignalApprovedEvent as TradeSignalApprovedEventBase
from gal_friday.core.events import TradeSignalProposedEvent as TradeSignalProposedEventBase

# Re-export other items that may be needed
from gal_friday.core.pubsub import PubSubManager as EventBus

# Add missing event types needed by tests
# These will be available as EventType.PORTFOLIO_UPDATE, etc.
setattr(EventType, "PORTFOLIO_UPDATE", EventType.SYSTEM_STATE_CHANGE)
setattr(EventType, "PORTFOLIO_RECONCILIATION", EventType.SYSTEM_STATE_CHANGE)
setattr(EventType, "PORTFOLIO_DISCREPANCY", EventType.SYSTEM_STATE_CHANGE)
setattr(EventType, "RISK_LIMIT_ALERT", EventType.POTENTIAL_HALT_TRIGGER)
setattr(EventType, "MARKET_DATA_RAW", EventType.MARKET_DATA_L2)
setattr(EventType, "FEATURE_CALCULATED", EventType.FEATURES_CALCULATED)


# Create wrapper classes for backward compatibility with test code
class MarketDataEvent:
    """Legacy MarketDataEvent for backward compatibility.

    This wraps the new MarketDataL2Event class with more flexible initialization.
    """

    @overload
    def __new__(
        cls,
        *,
        bids: Optional[List[Tuple[Any, Any]]] = None,
        asks: Optional[List[Tuple[Any, Any]]] = None,
        **kwargs: Any,
    ) -> "MarketDataEvent": ...

    @overload
    def __new__(cls, trading_pair: str, **kwargs: Any) -> "MarketDataEvent": ...

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new MarketDataEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        kwargs = MarketDataEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = MarketDataL2EventBase(**kwargs)

        # Return the instance as the wrapper class type
        return cast("MarketDataEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process kwargs with defaults and type conversions."""
        # Set default metadata
        if "source_module" not in kwargs:
            kwargs["source_module"] = "test_compatibility"
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now()
        if "event_id" not in kwargs:
            kwargs["event_id"] = uuid.uuid4()

        # Ensure orderbook data is properly formatted
        MarketDataEvent._prepare_orderbook_data(kwargs)

        return kwargs

    @staticmethod
    def _prepare_orderbook_data(kwargs: Dict[str, Any]) -> None:
        """Ensure bids and asks are properly formatted."""
        # Ensure bids and asks are properly formatted
        if "bids" not in kwargs:
            kwargs["bids"] = []
        if "asks" not in kwargs:
            kwargs["asks"] = []
        if "is_snapshot" not in kwargs:
            kwargs["is_snapshot"] = True

        # Convert string values to Decimal if needed
        for order_book in ["bids", "asks"]:
            if order_book in kwargs and kwargs[order_book]:
                for i, (price, quantity) in enumerate(kwargs[order_book]):
                    if isinstance(price, str):
                        price = Decimal(price)
                    if isinstance(quantity, str):
                        quantity = Decimal(quantity)
                    kwargs[order_book][i] = (price, quantity)


class FillEvent:
    """Legacy FillEvent for backward compatibility.

    This wraps the new ExecutionReportEvent class.
    """

    @overload
    def __new__(
        cls, *, exchange_order_id: str, trading_pair: str, **kwargs: Any
    ) -> "FillEvent": ...

    @overload
    def __new__(cls, *, fill_price: Any, **kwargs: Any) -> "FillEvent": ...

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new FillEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        kwargs = FillEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = ExecutionReportEventBase(**kwargs)

        # Return the instance as the wrapper class type
        return cast("FillEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process kwargs with defaults and type conversions."""
        # Set default metadata
        if "source_module" not in kwargs:
            kwargs["source_module"] = "test_compatibility"
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now()
        if "event_id" not in kwargs:
            kwargs["event_id"] = uuid.uuid4()

        # Convert decimal fields
        FillEvent._convert_decimal_fields(kwargs)

        # Set defaults
        FillEvent._set_default_values(kwargs)

        return kwargs

    @staticmethod
    def _convert_decimal_fields(kwargs: Dict[str, Any]) -> None:
        """Convert string fields to Decimal."""
        decimal_fields = [
            "quantity_ordered",
            "quantity_filled",
            "average_fill_price",
            "limit_price",
            "commission",
        ]

        for field in decimal_fields:
            if field in kwargs and kwargs[field] is not None:
                if isinstance(kwargs[field], str):
                    kwargs[field] = Decimal(kwargs[field])

    @staticmethod
    def _set_default_values(kwargs: Dict[str, Any]) -> None:
        """Set default values if fields are missing."""
        # Set status to FILLED for FillEvents
        if "order_status" not in kwargs:
            kwargs["order_status"] = "FILLED"

        # Add required fields with sensible defaults if missing
        defaults = {
            "exchange_order_id": f"mock-order-{uuid.uuid4()}",
            "trading_pair": "BTC/USD",
            "exchange": "mock_exchange",
            "order_type": "MARKET",
            "side": "BUY",
            "quantity_ordered": Decimal("1.0"),
        }

        for field, default_value in defaults.items():
            if field not in kwargs:
                kwargs[field] = default_value


class OrderEvent:
    """Legacy OrderEvent for backward compatibility.

    This wraps the new TradeSignalApprovedEvent class.
    """

    @overload
    def __new__(cls, *, trading_pair: str, side: str, **kwargs: Any) -> "OrderEvent": ...

    @overload
    def __new__(cls, *, signal_id: Any, **kwargs: Any) -> "OrderEvent": ...

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new OrderEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        kwargs = OrderEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = TradeSignalApprovedEventBase(**kwargs)

        # Return the instance as the wrapper class type
        return cast("OrderEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process kwargs with defaults and type conversions."""
        # Set default metadata
        if "source_module" not in kwargs:
            kwargs["source_module"] = "test_compatibility"
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now()
        if "event_id" not in kwargs:
            kwargs["event_id"] = uuid.uuid4()

        # Convert decimal fields
        OrderEvent._convert_decimal_fields(kwargs)

        # Set defaults
        OrderEvent._set_default_values(kwargs)

        return kwargs

    @staticmethod
    def _convert_decimal_fields(kwargs: Dict[str, Any]) -> None:
        """Convert string fields to Decimal."""
        decimal_fields = ["quantity", "sl_price", "tp_price", "limit_price"]

        for field in decimal_fields:
            if field in kwargs and kwargs[field] is not None:
                if isinstance(kwargs[field], str):
                    kwargs[field] = Decimal(kwargs[field])

    @staticmethod
    def _set_default_values(kwargs: Dict[str, Any]) -> None:
        """Set default values if fields are missing."""
        # Add required fields with sensible defaults if missing
        if "risk_parameters" not in kwargs:
            kwargs["risk_parameters"] = {}
        if "signal_id" not in kwargs:
            kwargs["signal_id"] = uuid.uuid4()

        defaults = {
            "trading_pair": "BTC/USD",
            "side": "BUY",
            "order_type": "MARKET",
            "quantity": Decimal("1.0"),
        }

        for field, default_value in defaults.items():
            if field not in kwargs:
                kwargs[field] = default_value


class SignalEvent:
    """Legacy SignalEvent for backward compatibility.

    This wraps the new TradeSignalProposedEvent class.
    """

    @overload
    def __new__(cls, *, trading_pair: str, side: str, **kwargs: Any) -> "SignalEvent": ...

    @overload
    def __new__(cls, *, strategy_id: str, **kwargs: Any) -> "SignalEvent": ...

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create a new SignalEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        kwargs = SignalEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = TradeSignalProposedEventBase(**kwargs)

        # Return the instance as the wrapper class type
        return cast("SignalEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Process kwargs with defaults and type conversions."""
        # Set default metadata
        if "source_module" not in kwargs:
            kwargs["source_module"] = "test_compatibility"
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now()
        if "event_id" not in kwargs:
            kwargs["event_id"] = uuid.uuid4()

        # Convert decimal fields
        SignalEvent._convert_decimal_fields(kwargs)

        # Set defaults
        SignalEvent._set_default_values(kwargs)

        return kwargs

    @staticmethod
    def _convert_decimal_fields(kwargs: Dict[str, Any]) -> None:
        """Convert string fields to Decimal."""
        decimal_fields = ["proposed_sl_price", "proposed_tp_price", "proposed_entry_price"]

        for field in decimal_fields:
            if field in kwargs and kwargs[field] is not None:
                if isinstance(kwargs[field], str):
                    kwargs[field] = Decimal(kwargs[field])

    @staticmethod
    def _set_default_values(kwargs: Dict[str, Any]) -> None:
        """Set default values if fields are missing."""
        # Handle strategy field
        if "signal_id" not in kwargs:
            kwargs["signal_id"] = uuid.uuid4()
        if "strategy_id" not in kwargs and "strategy" in kwargs:
            kwargs["strategy_id"] = kwargs.pop("strategy")
        elif "strategy_id" not in kwargs:
            kwargs["strategy_id"] = "test_strategy"

        # Set defaults for required fields
        defaults = {"trading_pair": "BTC/USD", "side": "BUY"}

        for field, default_value in defaults.items():
            if field not in kwargs:
                kwargs[field] = default_value


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
    "EventType",
]
