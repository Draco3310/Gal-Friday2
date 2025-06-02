"""Core type definitions for the Gal-Friday trading system.

This module contains type hints and protocols that define the interfaces for various
components in the trading system. These types are used throughout the codebase to
ensure type safety and consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import (
        Awaitable,  # For PubSubManagerProtocol
        Callable,
    )
    from datetime import datetime
    from decimal import Decimal

    import pandas as pd

# Type variables for generic types
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
PredictionOutput_co = TypeVar("PredictionOutput_co", covariant=True) # For PredictionServiceProtocol


# Protocol for configuration management
@runtime_checkable
class ConfigManager(Protocol[T_co]):
    """Protocol for configuration management."""

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get a configuration value."""
        ...

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401 # Config values can be truly dynamic.
        """Get a configuration value using dictionary-style access."""
        ...


# Protocol for logging service
class LoggerService(Protocol):
    """Protocol for logging service."""

    def log(self, message: str, level: str = "info", **kwargs: object) -> None:
        """Log a message at a specific level."""
        ...

    def debug(self, message: str, **kwargs: object) -> None:
        """Log a debug message."""
        ...

    def info(self, message: str, **kwargs: object) -> None:
        """Log an info message."""
        ...

    def warning(self, message: str, **kwargs: object) -> None:
        """Log a warning message."""
        ...

    def error(self, message: str, **kwargs: object) -> None:
        """Log an error message."""
        ...

    def exception(self, message: str, **kwargs: object) -> None:
        """Log an exception message."""
        ...

    def critical(self, message: str, **kwargs: object) -> None:
        """Log a critical message."""
        ...


# Protocol for market price service
class MarketPriceService(Protocol):
    """Protocol for market price service."""

    def get_price(self, symbol: str) -> Decimal | None:
        """Get the current price for a symbol."""
        ...

    def get_prices(self, symbols: list[str]) -> dict[str, Decimal | None]:
        """Get current prices for multiple symbols."""
        ...


# Protocol for portfolio management
class PortfolioManager(Protocol):
    """Protocol for portfolio management."""

    def get_balance(self, currency: str = "USD") -> Decimal:
        """Get the balance for a specific currency."""
        ...

    def get_position(self, symbol: str) -> Decimal:
        """Get the current position for a symbol."""
        ...

    def get_positions(self) -> dict[str, Decimal]:
        """Get all current positions."""
        ...


# Protocol for feature engineering
class FeatureEngine(Protocol):
    """Protocol for feature engineering."""

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features to the input data."""
        ...


# Protocol for prediction service
class PredictionService(Protocol[PredictionOutput_co]):
    """Protocol for prediction service."""

    def predict(self, features: pd.DataFrame) -> PredictionOutput_co:
        """Generate predictions based on input features."""
        ...


# Protocol for risk management
class RiskManager(Protocol):
    """Protocol for risk management."""

    def check_risk(self, order: dict[str, object]) -> bool:
        """Check the risk associated with an order."""
        ...


# Protocol for strategy arbitration
class StrategyArbitrator(Protocol):
    """Protocol for strategy arbitration."""

    def decide_action(self, signals: dict[str, object]) -> dict[str, object]:
        """Decide the next trading action based on input signals."""
        ...


# Protocol for exchange information
class ExchangeInfoService(Protocol):
    """Protocol for exchange information service."""

    def get_symbol_info(self, symbol: str) -> dict[str, object]:
        """Get information about a trading symbol."""
        ...


# Protocol for execution handling
class ExecutionHandler(Protocol):
    """Protocol for order execution."""

    async def execute_order(self, order: dict[str, object]) -> dict[str, object]:
        """Execute a trading order."""
        ...


# Protocol for pub/sub management
class PubSubManager(Protocol):
    """Protocol for pub/sub management."""

    def subscribe(self, channel: str, callback: Callable[[object], Awaitable[None]]) -> None:
        """Subscribe to a pub/sub channel."""
        ...

    def publish(self, channel: str, message: object) -> None:
        """Publish a message to a pub/sub channel."""
        ...


# Protocol for historical data provider
class BacktestHistoricalDataProvider(Protocol):
    """Protocol for historical data access during backtesting."""

    def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Get historical OHLCV data."""
        ...

    def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Get historical trade data."""
        ...


# Type aliases for better readability
LoggerServiceType = LoggerService
SimulatedMarketPriceServiceType = MarketPriceService
PortfolioManagerType = PortfolioManager
FeatureEngineType = FeatureEngine
PredictionServiceType = PredictionService
RiskManagerType = RiskManager
StrategyArbitratorType = StrategyArbitrator
