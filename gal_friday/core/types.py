"""Core type definitions for the Gal-Friday trading system.

This module contains type hints and protocols that define the interfaces for various
components in the trading system. These types are used throughout the codebase to
ensure type safety and consistency.
"""

from __future__ import annotations

from collections.abc import (
    Awaitable,  # For PubSubManagerProtocol
    Callable,
)
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, TypeVar, runtime_checkable

import pandas as pd

# Type variables for generic types
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
PredictionOutput = TypeVar("PredictionOutput", covariant=True) # For PredictionServiceProtocol


# Protocol for configuration management
@runtime_checkable
class ConfigManager(Protocol[T_co]):
    """Protocol for configuration management."""

    def get(self, key: str, default: T | None = None) -> T | None:
        ...

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401 # Config values can be truly dynamic.
        ...


# Protocol for logging service
class LoggerService(Protocol):
    """Protocol for logging service."""

    def log(self, message: str, level: str = "info", **kwargs: object) -> None:
        ...

    def debug(self, message: str, **kwargs: object) -> None:
        ...

    def info(self, message: str, **kwargs: object) -> None:
        ...

    def warning(self, message: str, **kwargs: object) -> None:
        ...

    def error(self, message: str, **kwargs: object) -> None:
        ...

    def exception(self, message: str, **kwargs: object) -> None:
        ...

    def critical(self, message: str, **kwargs: object) -> None:
        ...


# Protocol for market price service
class MarketPriceService(Protocol):
    """Protocol for market price service."""

    def get_price(self, symbol: str) -> Decimal | None:
        ...

    def get_prices(self, symbols: list[str]) -> dict[str, Decimal | None]:
        ...


# Protocol for portfolio management
class PortfolioManager(Protocol):
    """Protocol for portfolio management."""

    def get_balance(self, currency: str = "USD") -> Decimal:
        ...

    def get_position(self, symbol: str) -> Decimal:
        ...

    def get_positions(self) -> dict[str, Decimal]:
        ...


# Protocol for feature engineering
class FeatureEngine(Protocol):
    """Protocol for feature engineering."""

    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


# Protocol for prediction service
class PredictionService(Protocol[PredictionOutput]):
    """Protocol for prediction service."""

    def predict(self, features: pd.DataFrame) -> PredictionOutput:
        ...


# Protocol for risk management
class RiskManager(Protocol):
    """Protocol for risk management."""

    def check_risk(self, order: dict[str, object]) -> bool:
        ...


# Protocol for strategy arbitration
class StrategyArbitrator(Protocol):
    """Protocol for strategy arbitration."""

    def decide_action(self, signals: dict[str, object]) -> dict[str, object]:
        ...


# Protocol for exchange information
class ExchangeInfoService(Protocol):
    """Protocol for exchange information service."""

    def get_symbol_info(self, symbol: str) -> dict[str, object]:
        ...


# Protocol for execution handling
class ExecutionHandler(Protocol):
    """Protocol for order execution."""

    async def execute_order(self, order: dict[str, object]) -> dict[str, object]:
        ...


# Protocol for pub/sub management
class PubSubManager(Protocol):
    """Protocol for pub/sub management."""

    def subscribe(self, channel: str, callback: Callable[[object], Awaitable[None]]) -> None:
        ...

    def publish(self, channel: str, message: object) -> None:
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
        ...

    def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        ...


# Type aliases for better readability
LoggerServiceType = LoggerService
SimulatedMarketPriceServiceType = MarketPriceService
PortfolioManagerType = PortfolioManager
FeatureEngineType = FeatureEngine
PredictionServiceType = PredictionService
RiskManagerType = RiskManager
StrategyArbitratorType = StrategyArbitrator
