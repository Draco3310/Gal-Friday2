"""Core component interfaces for dependency injection.

This module contains Protocol definitions for core system components
to enable dependency injection and break circular dependencies.
"""

from decimal import Decimal
from typing import Any, Protocol, runtime_checkable
import uuid


@runtime_checkable
class PortfolioManagerInterface(Protocol):
    """Interface for portfolio management functionality."""

    def get_current_state(self) -> dict[str, Any]:
        """Get the current portfolio state including positions, balances, and equity."""
        ...

    def get_position(self, trading_pair: str) -> dict[str, Any] | None:
        """Get position information for a specific trading pair."""
        ...

    def get_total_equity_usd(self) -> Decimal:
        """Get total equity in USD."""
        ...

    def get_available_balance(self, currency: str) -> Decimal:
        """Get available balance for a specific currency."""
        ...

    async def update_position(self, trading_pair: str, quantity: Decimal, side: str) -> None:
        """Update position for a trading pair."""
        ...


@runtime_checkable
class RiskManagerInterface(Protocol):
    """Interface for risk management functionality."""

    async def validate_signal(
        self,
        signal_id: uuid.UUID,
        trading_pair: str,
        side: str,
        proposed_entry_price: Decimal | None,
        proposed_sl_price: Decimal | None,
        proposed_tp_price: Decimal | None,
    ) -> tuple[bool, str | None]:
        """Validate a trading signal against risk rules.

        Returns:
            Tuple of (is_valid, rejection_reason)
        """
        ...

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        ...

    def check_drawdown_limits(self) -> tuple[bool, str | None]:
        """Check if drawdown limits are exceeded.

        Returns:
            Tuple of (is_within_limits, violation_message)
        """
        ...


@runtime_checkable
class MonitoringServiceInterface(Protocol):
    """Interface for system monitoring functionality."""

    async def record_metric(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric value."""
        ...

    async def check_system_health(self) -> dict[str, Any]:
        """Check overall system health."""
        ...

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of recent metrics."""
        ...


@runtime_checkable
class ConfigManagerInterface(Protocol):
    """Interface for configuration management."""

    def get_config(self) -> dict[str, Any]:
        """Get the complete configuration."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        ...

    def reload_config(self) -> None:
        """Reload configuration from source."""
        ...


@runtime_checkable
class PubSubManagerInterface(Protocol):
    """Interface for publish/subscribe event management."""

    async def publish(self, event: Any) -> None:
        """Publish an event to subscribers."""
        ...

    async def subscribe(self, event_type: str, handler: Any) -> None:
        """Subscribe to events of a specific type."""
        ...

    async def unsubscribe(self, event_type: str, handler: Any) -> None:
        """Unsubscribe from events."""
        ...


@runtime_checkable
class LoggerServiceInterface(Protocol):
    """Interface for logging functionality."""

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        ...

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        ...

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message."""
        ...

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        ...


@runtime_checkable
class ExchangeInfoServiceInterface(Protocol):
    """Interface for exchange information service."""

    def get_step_size(self, trading_pair: str) -> Decimal | None:
        """Get the minimum order quantity step size for a trading pair."""
        ...

    def get_tick_size(self, trading_pair: str) -> Decimal | None:
        """Get the minimum price tick size for a trading pair."""
        ...

    def get_min_order_size(self, trading_pair: str) -> Decimal | None:
        """Get the minimum order size for a trading pair."""
        ...

    def round_price(self, price: Decimal, trading_pair: str) -> Decimal:
        """Round a price to the correct tick size for a trading pair."""
        ...

    def round_quantity(self, quantity: Decimal, trading_pair: str) -> Decimal:
        """Round a quantity to the correct step size for a trading pair."""
        ...


@runtime_checkable
class StrategyArbitratorInterface(Protocol):
    """Interface for strategy arbitration functionality."""

    async def process_prediction(self, prediction_event: Any) -> None:
        """Process a prediction event and potentially generate trading signals."""
        ...

    def get_active_strategies(self) -> list[str]:
        """Get list of currently active strategy IDs."""
        ...

    async def update_strategy_config(self, strategy_id: str, config: dict[str, Any]) -> None:
        """Update configuration for a specific strategy."""
        ...
