"""Compatibility module for legacy imports.

This module re-exports event classes from core.events with legacy names to maintain
backward compatibility with existing test code.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, TypedDict, Union, cast, overload
import uuid

from gal_friday.config_manager import ConfigManager

# Re-export event classes with legacy names
from gal_friday.core.events import EventType
from gal_friday.core.events import ExecutionReportEvent as ExecutionReportEventBase
from gal_friday.core.events import MarketDataL2Event as MarketDataL2EventBase
from gal_friday.core.events import TradeSignalApprovedEvent as TradeSignalApprovedEventBase
from gal_friday.core.events import TradeSignalProposedEvent as TradeSignalProposedEventBase

# Re-export other items that may be needed
from gal_friday.core.pubsub import PubSubManager as EventBus

# Note: EventType enum values are now properly defined in core.events
# No need to dynamically add attributes here anymore


# --- TypedDicts for **kwargs to address ANN401 ---

class MarketDataEventKwargs(TypedDict, total=False):
    trading_pair: str
    exchange: str
    bids: Sequence[tuple[str, str]]
    asks: Sequence[tuple[str, str]]
    is_snapshot: bool
    timestamp_exchange: Optional[datetime]
    source_module: str
    event_id: uuid.UUID
    timestamp: datetime

class FillEventKwargs(TypedDict, total=False): # Corresponds to ExecutionReportEventBase
    exchange_order_id: str
    trading_pair: str
    exchange: str
    order_status: str
    order_type: str
    side: str
    quantity_ordered: Decimal
    signal_id: Optional[uuid.UUID]
    client_order_id: Optional[str]
    quantity_filled: Optional[Decimal] # Base has Decimal(0) default
    average_fill_price: Optional[Decimal]
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    commission: Optional[Decimal]
    commission_asset: Optional[str]
    timestamp_exchange: Optional[datetime]
    error_message: Optional[str]
    source_module: str
    event_id: uuid.UUID
    timestamp: datetime
    fill_price: Optional[Union[str, Decimal]]

class OrderEventKwargs(TypedDict, total=False): # Corresponds to TradeSignalApprovedEventBase
    signal_id: uuid.UUID
    trading_pair: str
    exchange: str
    side: str
    order_type: str
    quantity: Decimal
    sl_price: Decimal
    tp_price: Decimal
    risk_parameters: dict # dict as per base model
    limit_price: Optional[Decimal]
    source_module: str
    event_id: uuid.UUID
    timestamp: datetime

class SignalEventKwargs(TypedDict, total=False): # Corresponds to TradeSignalProposedEventBase
    signal_id: uuid.UUID
    trading_pair: str
    exchange: str
    side: str
    entry_type: str # Field name in base model
    proposed_sl_price: Decimal
    proposed_tp_price: Decimal
    strategy_id: str # Field name in base model
    proposed_entry_price: Optional[Decimal]
    triggering_prediction_event_id: Optional[uuid.UUID]
    triggering_prediction: Optional[dict] # dict as per base model
    source_module: str
    event_id: uuid.UUID
    timestamp: datetime
    strategy: str # Legacy field, will be mapped to strategy_id


# Create wrapper classes for backward compatibility with test code
class MarketDataEvent:
    """Legacy MarketDataEvent for backward compatibility.

    This wraps the new MarketDataL2Event class with more flexible initialization.
    """

    @overload
    def __new__(
        cls,
        *,
        bids: Optional[list[tuple[Any, Any]]] = None,
        asks: Optional[list[tuple[Any, Any]]] = None,
        **kwargs: MarketDataEventKwargs,
    ) -> "MarketDataEvent": ...

    @overload
    def __new__(cls, trading_pair: str, **kwargs: MarketDataEventKwargs) -> "MarketDataEvent": ...

    def __new__(cls, **kwargs: MarketDataEventKwargs) -> Any:
        """Create a new MarketDataEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        prepared_kwargs = MarketDataEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = MarketDataL2EventBase(**prepared_kwargs)

        # Return the instance as the wrapper class type
        return cast("MarketDataEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
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
    def _prepare_orderbook_data(kwargs: dict[str, Any]) -> None:
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
            if kwargs.get(order_book):
                updated_entries = []
                for loop_price, loop_quantity in kwargs[order_book]:
                    if isinstance(loop_price, str):
                        price_to_store = Decimal(loop_price)
                    else:
                        price_to_store = loop_price

                    if isinstance(loop_quantity, str):
                        quantity_to_store = Decimal(loop_quantity)
                    else:
                        quantity_to_store = loop_quantity
                    updated_entries.append((price_to_store, quantity_to_store))
                kwargs[order_book] = updated_entries


class FillEvent:
    """Legacy FillEvent for backward compatibility.

    This wraps the new ExecutionReportEvent class.
    """

    @overload
    def __new__(
        cls, *, exchange_order_id: str, trading_pair: str, **kwargs: FillEventKwargs
    ) -> "FillEvent": ...

    @overload
    def __new__(
        cls,
        *,
        fill_price: Optional[Union[str, Decimal]],
        **kwargs: FillEventKwargs
    ) -> "FillEvent": ...

    def __new__(cls, **kwargs: FillEventKwargs) -> Any:
        """Create a new FillEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        prepared_kwargs = FillEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = ExecutionReportEventBase(**prepared_kwargs)

        # Return the instance as the wrapper class type
        return cast("FillEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
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

        # Handle legacy fill_price if present
        if "fill_price" in kwargs and kwargs["fill_price"] is not None:
            if "average_fill_price" not in kwargs: # Prioritize average_fill_price if already set
                 kwargs["average_fill_price"] = Decimal(str(kwargs["fill_price"]))
            # Assume fully filled if not specified
            if ("quantity_filled" not in kwargs and
                    "quantity_ordered" in kwargs):
                 kwargs["quantity_filled"] = kwargs["quantity_ordered"]
            del kwargs["fill_price"]

        return kwargs

    @staticmethod
    def _convert_decimal_fields(kwargs: dict[str, Any]) -> None:
        """Convert string fields to Decimal."""
        decimal_fields = [
            "quantity_ordered",
            "quantity_filled",
            "average_fill_price",
            "limit_price",
            "commission",
        ]

        for field in decimal_fields:
            if field in kwargs and kwargs[field] is not None and isinstance(kwargs[field], str):
                kwargs[field] = Decimal(kwargs[field])

    @staticmethod
    def _set_default_values(kwargs: dict[str, Any]) -> None:
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

        # Ensure quantity_filled defaults to quantity_ordered if status is FILLED
        if (kwargs.get("order_status") == "FILLED" and
                "quantity_filled" not in kwargs and
                "quantity_ordered" in kwargs):
            kwargs["quantity_filled"] = kwargs["quantity_ordered"]
        elif "quantity_filled" not in kwargs: # Default if quantity_ordered not set
            kwargs["quantity_filled"] = Decimal("0")

        for field, default_value in defaults.items():
            if field not in kwargs:
                kwargs[field] = default_value


class OrderEvent:
    """Legacy OrderEvent for backward compatibility.

    This wraps the new TradeSignalApprovedEvent class.
    """

    @overload
    def __new__(
        cls, *, trading_pair: str, side: str, **kwargs: OrderEventKwargs
    ) -> "OrderEvent": ...

    @overload
    def __new__(cls, *, signal_id: uuid.UUID, **kwargs: OrderEventKwargs) -> "OrderEvent": ...

    def __new__(cls, **kwargs: OrderEventKwargs) -> Any:
        """Create a new OrderEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        prepared_kwargs = OrderEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = TradeSignalApprovedEventBase(**prepared_kwargs)

        # Return the instance as the wrapper class type
        return cast("OrderEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
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
    def _convert_decimal_fields(kwargs: dict[str, Any]) -> None:
        """Convert string fields to Decimal."""
        decimal_fields = ["quantity", "sl_price", "tp_price", "limit_price"]

        for field in decimal_fields:
            if field in kwargs and kwargs[field] is not None and isinstance(kwargs[field], str):
                kwargs[field] = Decimal(kwargs[field])

    @staticmethod
    def _set_default_values(kwargs: dict[str, Any]) -> None:
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

        # Default sl_price and tp_price if not provided, using dummy values for testing
        if "sl_price" not in kwargs:
            kwargs["sl_price"] = Decimal("0.01") # Placeholder
        if "tp_price" not in kwargs:
            kwargs["tp_price"] = Decimal("100000") # Placeholder

        for field, default_value in defaults.items():
            if field not in kwargs:
                kwargs[field] = default_value


class SignalEvent:
    """Legacy SignalEvent for backward compatibility.

    This wraps the new TradeSignalProposedEvent class.
    """

    @overload
    def __new__(
        cls, *, trading_pair: str, side: str, **kwargs: SignalEventKwargs
    ) -> "SignalEvent": ...

    @overload
    def __new__(cls, *, strategy_id: str, **kwargs: SignalEventKwargs) -> "SignalEvent": ...

    def __new__(cls, **kwargs: SignalEventKwargs) -> Any:
        """Create a new SignalEvent instance with default values where needed."""
        # Prepare kwargs with defaults and type conversions
        prepared_kwargs = SignalEvent._prepare_kwargs(kwargs)

        # Create the actual instance using the base class
        instance = TradeSignalProposedEventBase(**prepared_kwargs)

        # Return the instance as the wrapper class type
        return cast("SignalEvent", instance)

    @staticmethod
    def _prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
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

        # Set defaults (includes handling legacy 'strategy' field)
        SignalEvent._set_default_values(kwargs)

        return kwargs

    @staticmethod
    def _convert_decimal_fields(kwargs: dict[str, Any]) -> None:
        """Convert string fields to Decimal."""
        decimal_fields = ["proposed_sl_price", "proposed_tp_price", "proposed_entry_price"]

        for field in decimal_fields:
            if field in kwargs and kwargs[field] is not None and isinstance(kwargs[field], str):
                kwargs[field] = Decimal(kwargs[field])

    @staticmethod
    def _set_default_values(kwargs: dict[str, Any]) -> None:
        """Set default values if fields are missing."""
        # Handle strategy field (map legacy 'strategy' to 'strategy_id')
        if "strategy_id" not in kwargs and "strategy" in kwargs:
            kwargs["strategy_id"] = kwargs.pop("strategy")
        elif "strategy_id" not in kwargs:
            kwargs["strategy_id"] = "test_strategy" # Default if neither provided

        if "signal_id" not in kwargs:
            kwargs["signal_id"] = uuid.uuid4()

        # Set defaults for required fields in TradeSignalProposedEvent
        defaults = {
            "trading_pair": "BTC/USD",
            "side": "BUY",
            "entry_type": "MARKET", # Default entry_type for proposed signal
             # For testing, make sure these are present if not set by user
            "proposed_sl_price": Decimal("0.01"),
            "proposed_tp_price": Decimal("100000"),
        }

        for field, default_value in defaults.items():
            if field not in kwargs:
                kwargs[field] = default_value


class BackpressureStrategy(ABC):
    """Base class for backpressure strategies to control event flow."""

    @abstractmethod
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


class SimpleThresholdBackpressure(BackpressureStrategy):
    """Threshold-based backpressure strategy with different levels per event type."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize with config manager to access thresholds.

        Args
        ----
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.thresholds: dict[int, int] = {}  # EventType.value -> queue size threshold
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
    "BackpressureStrategy",
    "EventBus",
    "EventType",
    "FillEvent",
    "MarketDataEvent",
    "OrderEvent",
    "SignalEvent",
    "SimpleThresholdBackpressure",
    # Add TypedDicts to __all__ if they are meant to be importable,
    # otherwise keep them as internal types. For now, keeping internal.
    # "MarketDataEventKwargs",
    # "FillEventKwargs",
    # "OrderEventKwargs",
    # "SignalEventKwargs",
]
