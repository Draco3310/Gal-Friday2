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
PredictionOutput_co = TypeVar("PredictionOutput_co", covariant=True)  # For PredictionServiceProtocol


# Protocol for configuration management
@runtime_checkable
class ConfigManager(Protocol[T_co]):
    """Protocol for configuration management."""

    def get(self, key: str, default: T | None = None) -> T | None:
        """Retrieve configuration value by key with optional default.

        Args:
            key: Configuration key to retrieve.
            default: Default value if key not found.

        Returns:
            Configuration value or default if not found.
        """
        ...

    def __getitem__(self, key: str) -> Any:  # Config values can be truly dynamic.
        """Get configuration value using dictionary-style access.

        Args:
            key: Configuration key to retrieve.

        Returns:
            Configuration value.

        Raises:
            KeyError: If key not found.
        """
        ...


# Protocol for logging service
class LoggerService(Protocol):
    """Protocol for logging service."""

    def log(self, message: str, level: str = "info", **kwargs: object) -> None:
        """Log a message at the specified level.

        Args:
            message: Message to log.
            level: Log level (default: "info").
            **kwargs: Additional context parameters.
        """
        ...

    def debug(self, message: str, **kwargs: object) -> None:
        """Log a debug-level message.

        Args:
            message: Debug message to log.
            **kwargs: Additional context parameters.
        """
        ...

    def info(self, message: str, **kwargs: object) -> None:
        """Log an info-level message.

        Args:
            message: Info message to log.
            **kwargs: Additional context parameters.
        """
        ...

    def warning(self, message: str, **kwargs: object) -> None:
        """Log a warning-level message.

        Args:
            message: Warning message to log.
            **kwargs: Additional context parameters.
        """
        ...

    def error(self, message: str, **kwargs: object) -> None:
        """Log an error-level message.

        Args:
            message: Error message to log.
            **kwargs: Additional context parameters.
        """
        ...

    def exception(self, message: str, **kwargs: object) -> None:
        """Log an exception with traceback information.

        Args:
            message: Exception message to log.
            **kwargs: Additional context parameters.
        """
        ...

    def critical(self, message: str, **kwargs: object) -> None:
        """Log a critical-level message.

        Args:
            message: Critical message to log.
            **kwargs: Additional context parameters.
        """
        ...


# Protocol for market price service
class MarketPriceService(Protocol):
    """Protocol for market price service."""

    def get_price(self, symbol: str) -> Decimal | None:
        """Get current market price for a single symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'XRP/USD').

        Returns:
            Current price as Decimal or None if unavailable.
        """
        ...

    def get_prices(self, symbols: list[str]) -> dict[str, Decimal | None]:
        """Get current market prices for multiple symbols.

        Args:
            symbols: List of trading pair symbols.

        Returns:
            Dictionary mapping symbols to their current prices.
        """
        ...


# Protocol for portfolio management
class PortfolioManager(Protocol):
    """Protocol for portfolio management."""

    def get_balance(self, currency: str = "USD") -> Decimal:
        """Get account balance for specified currency.

        Args:
            currency: Currency code (default: 'USD').

        Returns:
            Current balance as Decimal.
        """
        ...

    def get_position(self, symbol: str) -> Decimal:
        """Get position size for a specific symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Position size as Decimal (0 if no position).
        """
        ...

    def get_positions(self) -> dict[str, Decimal]:
        """Get all current positions.

        Returns:
            Dictionary mapping symbols to position sizes.
        """
        ...


# Protocol for feature engineering
class FeatureEngine(Protocol):
    """Protocol for feature engineering."""

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and derived features to market data.

        Args:
            data: DataFrame with OHLCV market data.

        Returns:
            DataFrame with additional feature columns.
        """
        ...


# Protocol for prediction service
class PredictionService(Protocol[PredictionOutput_co]):
    """Protocol for prediction service."""

    def predict(self, features: pd.DataFrame) -> PredictionOutput_co:
        """Generate predictions from feature data.

        Args:
            features: DataFrame containing feature columns.

        Returns:
            Prediction output (type varies by implementation).
        """
        ...


# Protocol for risk management
class RiskManager(Protocol):
    """Protocol for risk management."""

    def check_risk(self, order: dict[str, object]) -> bool:
        """Evaluate if an order passes risk management criteria.

        Args:
            order: Order details dictionary.

        Returns:
            True if order is approved, False if rejected.
        """
        ...


# Protocol for strategy arbitration
class StrategyArbitrator(Protocol):
    """Protocol for strategy arbitration."""

    def decide_action(self, signals: dict[str, object]) -> dict[str, object]:
        """Decide trading action based on strategy signals.

        Args:
            signals: Dictionary of signals from various strategies.

        Returns:
            Dictionary containing the decided action details.
        """
        ...


# Protocol for exchange information
class ExchangeInfoService(Protocol):
    """Protocol for exchange information service."""

    def get_symbol_info(self, symbol: str) -> dict[str, object]:
        """Get trading symbol information from exchange.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dictionary containing symbol details (lot size, tick size, etc.).
        """
        ...


# Protocol for execution handling
class ExecutionHandler(Protocol):
    """Protocol for order execution."""

    async def execute_order(self, order: dict[str, object]) -> dict[str, object]:
        """Execute a trading order on the exchange.

        Args:
            order: Order details dictionary.

        Returns:
            Dictionary containing execution result.
        """
        ...


# Protocol for pub/sub management
class PubSubManager(Protocol):
    """Protocol for pub/sub management."""

    def subscribe(self, channel: str, callback: Callable[..., Awaitable[None]]) -> None:
        """Subscribe to a pub/sub channel.

        Args:
            channel: Channel name to subscribe to.
            callback: Async callback function to handle messages.
        """
        ...

    def publish(self, channel: str, message: object) -> None:
        """Publish a message to a channel.

        Args:
            channel: Channel name to publish to.
            message: Message object to publish.
        """
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
        """Retrieve historical OHLCV data for backtesting.

        Args:
            trading_pair: Trading pair symbol.
            start_time: Start of data range.
            end_time: End of data range.
            interval: Time interval (default: '1d').

        Returns:
            DataFrame with OHLCV data or None if unavailable.
        """
        ...

    def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Retrieve historical trade data for backtesting.

        Args:
            trading_pair: Trading pair symbol.
            start_time: Start of data range.
            end_time: End of data range.

        Returns:
            DataFrame with trade data or None if unavailable.
        """
        ...


class ServiceProtocol(Protocol):
    """Protocol for application services."""

    async def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the service."""

    async def start(self) -> None:
        """Start the service."""


# Type[Any] aliases for better readability
LoggerServiceType = LoggerService
SimulatedMarketPriceServiceType = MarketPriceService
PortfolioManagerType = PortfolioManager
FeatureEngineType = FeatureEngine
PredictionServiceType = PredictionService
RiskManagerType = RiskManager
StrategyArbitratorType = StrategyArbitrator
