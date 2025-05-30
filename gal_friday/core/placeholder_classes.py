"""Placeholder classes for use during runtime when TYPE_CHECKING is False.

This module provides lightweight placeholders for classes that would otherwise
create circular imports when not being used for actual type checking.
"""

from collections.abc import Callable
from decimal import Decimal


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

    def get_decimal(self, _key: str, default: Decimal | str | int | float = 0) -> Decimal:
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

    def subscribe(self, event_type: object, handler: Callable) -> None:
        """Subscribe to an event type."""

    def unsubscribe(self, event_type: object, handler: Callable) -> None:
        """Unsubscribe from an event type."""

    async def publish(self, event: object) -> None:
        """Publish an event."""


# --- Market Price Service ---
class MarketPriceService:
    """Placeholder for MarketPriceService to avoid circular imports."""

    async def get_latest_price(self, _trading_pair: str) -> Decimal | None:
        """Get latest price for a trading pair."""
        return None

    async def get_bid_ask_spread(self, _trading_pair: str) -> tuple[Decimal, Decimal] | None:
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
