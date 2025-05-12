"""Placeholder classes for use during runtime when TYPE_CHECKING is False.

This module provides lightweight placeholders for classes that would otherwise
create circular imports when not being used for actual type checking.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union


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
        event_id: Any = None,
        timestamp: Optional[datetime] = None
    ):
        self.source_module = source_module
        self.event_id = event_id
        self.timestamp = timestamp or datetime.utcnow()


class ExecutionReportEvent(Event):
    """Event representing an execution report from an exchange."""
    
    def __init__(
        self,
        *,
        source_module: str = "",
        event_id: Any = None,
        timestamp: Optional[datetime] = None,
        signal_id: Optional[Any] = None,
        exchange_order_id: str = "",
        client_order_id: str = "",
        trading_pair: str = "",
        exchange: str = "",
        order_status: str = "",
        order_type: str = "",
        side: str = "",
        quantity_ordered: Decimal = Decimal(0),
        quantity_filled: Decimal = Decimal(0),
        average_fill_price: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        commission: Optional[Decimal] = None,
        commission_asset: Optional[str] = None,
        timestamp_exchange: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ):
        super().__init__(source_module, event_id, timestamp)
        self.signal_id = signal_id
        self.exchange_order_id = exchange_order_id
        self.client_order_id = client_order_id
        self.trading_pair = trading_pair
        self.exchange = exchange
        self.order_status = order_status
        self.order_type = order_type
        self.side = side
        self.quantity_ordered = quantity_ordered
        self.quantity_filled = quantity_filled
        self.average_fill_price = average_fill_price
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.commission = commission
        self.commission_asset = commission_asset
        self.timestamp_exchange = timestamp_exchange or timestamp
        self.error_message = error_message


# --- Config Manager ---
class ConfigManager:
    """Placeholder for ConfigManager to avoid circular imports."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key path."""
        return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer config value by key path."""
        return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean config value by key path."""
        return default
    
    def get_decimal(self, key: str, default: Union[Decimal, str, int, float] = 0) -> Decimal:
        """Get Decimal config value by key path."""
        if isinstance(default, Decimal):
            return default
        return Decimal(str(default))


# --- Logger Service ---
class LoggerService:
    """Placeholder for LoggerService to avoid circular imports."""
    
    def info(self, message: str, source_module: str = "", **kwargs: Any) -> None:
        """Log an info message."""
        pass
    
    def debug(self, message: str, source_module: str = "", **kwargs: Any) -> None:
        """Log a debug message."""
        pass
    
    def warning(self, message: str, source_module: str = "", **kwargs: Any) -> None:
        """Log a warning message."""
        pass
    
    def error(self, message: str, source_module: str = "", **kwargs: Any) -> None:
        """Log an error message."""
        pass
    
    def critical(self, message: str, source_module: str = "", **kwargs: Any) -> None:
        """Log a critical message."""
        pass


# --- PubSub Manager ---
class PubSubManager:
    """Placeholder for PubSubManager to avoid circular imports."""
    
    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """Subscribe to an event type."""
        pass
    
    def unsubscribe(self, event_type: EventType, handler: Callable) -> None:
        """Unsubscribe from an event type."""
        pass
    
    async def publish(self, event: Event) -> None:
        """Publish an event."""
        pass


# --- Market Price Service ---
class MarketPriceService:
    """Placeholder for MarketPriceService to avoid circular imports."""
    
    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """Get latest price for a trading pair."""
        return None
    
    async def get_bid_ask_spread(
        self, trading_pair: str
    ) -> Optional[tuple[Decimal, Decimal]]:
        """Get bid-ask spread for a trading pair."""
        return None


# --- Execution Handler ---
class ExecutionHandler:
    """Placeholder for ExecutionHandler to avoid circular imports."""
    
    async def get_account_balances(self) -> Dict[str, Decimal]:
        """Get account balances."""
        return {}
    
    async def submit_order(self, order: Any) -> str:
        """Submit an order."""
        return ""
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return True 