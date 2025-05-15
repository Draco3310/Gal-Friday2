"""Placeholder classes for use during runtime when TYPE_CHECKING is False.

This module provides lightweight placeholders for classes that would otherwise
create circular imports when not being used for actual type checking.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Callable, Optional, Union


# --- Event System ---
class EventType(Enum):
    """Enumeration of event types in the system."""

    MARKET_DATA_L2 = auto()
    MARKET_DATA_TRADE = auto()
    MARKET_DATA_OHLCV = auto()
    PREDICTION_GENERATED = auto()
    TRADE_SIGNAL_PROPOSED = auto()
    TRADE_SIGNAL_APPROVED = auto()
    TRADE_SIGNAL_REJECTED = auto()
    EXECUTION_REPORT = auto()
    SYSTEM_HALT_TRIGGERED = auto()
    SYSTEM_RESUME_TRIGGERED = auto()
    SYSTEM_STATE_CHANGED = auto()
    SYSTEM_ERROR = auto()


class Event:
    """Base class for all events in the system."""

    def __init__(
        self,
        source_module: str = "",
        event_id: object = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Initialize a base Event.

        Args
        ----
            source_module: The module that created this event
            event_id: Unique identifier for this event
            timestamp: When the event was created, defaults to current time
        """
        self.source_module = source_module
        self.event_id = event_id
        self.timestamp = timestamp or datetime.utcnow()


@dataclass
class ExecutionReportDetails:
    """Details specific to an execution report."""

    signal_id: Optional[object] = None
    exchange_order_id: str = ""
    client_order_id: str = ""
    trading_pair: str = ""
    exchange: str = ""
    order_status: str = ""
    order_type: str = ""
    side: str = ""
    quantity_ordered: Decimal = Decimal(0)
    quantity_filled: Decimal = Decimal(0)
    average_fill_price: Optional[Decimal] = None
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    commission_asset: Optional[str] = None
    timestamp_exchange: Optional[datetime] = None
    error_message: Optional[str] = None


class ExecutionReportEvent(Event):
    """Event representing an execution report from an exchange."""

    def __init__(
        self,
        *,
        source_module: str = "",
        event_id: object = None,
        timestamp: Optional[datetime] = None,
        details: ExecutionReportDetails,
    ) -> None:
        """Initialize an execution report event.

        Args
        ----
            source_module: Module that created this event
            event_id: Unique identifier for this event
            timestamp: When the event was created
            details: An object containing detailed information about the execution report.
        """
        super().__init__(source_module, event_id, timestamp)
        self.signal_id = details.signal_id
        self.exchange_order_id = details.exchange_order_id
        self.client_order_id = details.client_order_id
        self.trading_pair = details.trading_pair
        self.exchange = details.exchange
        self.order_status = details.order_status
        self.order_type = details.order_type
        self.side = details.side
        self.quantity_ordered = details.quantity_ordered
        self.quantity_filled = details.quantity_filled
        self.average_fill_price = details.average_fill_price
        self.limit_price = details.limit_price
        self.stop_price = details.stop_price
        self.commission = details.commission
        self.commission_asset = details.commission_asset
        self.timestamp_exchange = details.timestamp_exchange or self.timestamp
        self.error_message = details.error_message


# --- Config Manager ---
class ConfigManager:
    """Placeholder for ConfigManager to avoid circular imports."""

    def get(self, _key: str, default: object = None) -> object:
        """Get config value by key path."""
        return default

    def get_int(self, _key: str, default: int = 0) -> int:
        """Get integer config value by key path."""
        return default

    def get_bool(self, _key: str, default: bool = False) -> bool:
        """Get boolean config value by key path."""
        return default

    def get_decimal(self, _key: str, default: Union[Decimal, str, int, float] = 0) -> Decimal:
        """Get Decimal config value by key path."""
        if isinstance(default, Decimal):
            return default
        return Decimal(str(default))


# --- Logger Service ---
class LoggerService:
    """Placeholder for LoggerService to avoid circular imports."""

    def info(self, message: str, source_module: str = "", **_kwargs: object) -> None:
        """Log an info message."""

    def debug(self, message: str, source_module: str = "", **_kwargs: object) -> None:
        """Log a debug message."""

    def warning(self, message: str, source_module: str = "", **_kwargs: object) -> None:
        """Log a warning message."""

    def error(self, message: str, source_module: str = "", **_kwargs: object) -> None:
        """Log an error message."""

    def critical(self, message: str, source_module: str = "", **_kwargs: object) -> None:
        """Log a critical message."""


# --- PubSub Manager ---
class PubSubManager:
    """Placeholder for PubSubManager to avoid circular imports."""

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to an event type."""

    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe from an event type."""

    async def publish(self, event: Event) -> None:
        """Publish an event."""


# --- Market Price Service ---
class MarketPriceService:
    """Placeholder for MarketPriceService to avoid circular imports."""

    async def get_latest_price(self, _trading_pair: str) -> Optional[Decimal]:
        """Get latest price for a trading pair."""
        return None

    async def get_bid_ask_spread(self, _trading_pair: str) -> Optional[tuple[Decimal, Decimal]]:
        """Get bid-ask spread for a trading pair."""
        return None


# --- Execution Handler ---
class ExecutionHandler:
    """Placeholder for ExecutionHandler to avoid circular imports."""

    async def get_account_balances(self) -> dict[str, Decimal]:
        """Get account balances."""
        return {}

    async def submit_order(self, _order: object) -> str:
        """Submit an order."""
        return ""

    async def cancel_order(self, _order_id: str) -> bool:
        """Cancel an order."""
        return True
