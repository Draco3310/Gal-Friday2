"""Core type definitions for the Gal-Friday trading system.

This module contains type hints and protocols that define the interfaces for various
components in the trading system. These types are used throughout the codebase to
ensure type safety and consistency.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable, Optional, Dict, List, Union
from datetime import datetime
from decimal import Decimal
import pandas as pd

# Type variables for generic types
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

# Protocol for configuration management
@runtime_checkable
class ConfigManager(Protocol[T_co]):
    """Protocol for configuration management."""
    def get(self, key: str, default: T | None = None) -> T | None: ...
    def __getitem__(self, key: str) -> Any: ...

# Protocol for logging service
class LoggerService(Protocol):
    """Protocol for logging service."""
    def log(self, message: str, level: str = "info", **kwargs: Any) -> None: ...
    def debug(self, message: str, **kwargs: Any) -> None: ...
    def info(self, message: str, **kwargs: Any) -> None: ...
    def warning(self, message: str, **kwargs: Any) -> None: ...
    def error(self, message: str, **kwargs: Any) -> None: ...
    def exception(self, message: str, **kwargs: Any) -> None: ...
    def critical(self, message: str, **kwargs: Any) -> None: ...

# Protocol for market price service
class MarketPriceService(Protocol):
    """Protocol for market price service."""
    def get_price(self, symbol: str) -> Optional[Decimal]: ...
    def get_prices(self, symbols: List[str]) -> Dict[str, Optional[Decimal]]: ...

# Protocol for portfolio management
class PortfolioManager(Protocol):
    """Protocol for portfolio management."""
    def get_balance(self, currency: str = "USD") -> Decimal: ...
    def get_position(self, symbol: str) -> Decimal: ...
    def get_positions(self) -> Dict[str, Decimal]: ...

# Protocol for feature engineering
class FeatureEngine(Protocol):
    """Protocol for feature engineering."""
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame: ...

# Protocol for prediction service
class PredictionService(Protocol):
    """Protocol for prediction service."""
    def predict(self, features: pd.DataFrame) -> Any: ...

# Protocol for risk management
class RiskManager(Protocol):
    """Protocol for risk management."""
    def check_risk(self, order: Dict[str, Any]) -> bool: ...

# Protocol for strategy arbitration
class StrategyArbitrator(Protocol):
    """Protocol for strategy arbitration."""
    def decide_action(self, signals: Dict[str, Any]) -> Dict[str, Any]: ...

# Protocol for exchange information
class ExchangeInfoService(Protocol):
    """Protocol for exchange information service."""
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]: ...

# Protocol for execution handling
class ExecutionHandler(Protocol):
    """Protocol for order execution."""
    async def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]: ...

# Protocol for pub/sub management
class PubSubManager(Protocol):
    """Protocol for pub/sub management."""
    def subscribe(self, channel: str, callback: Any) -> None: ...
    def publish(self, channel: str, message: Any) -> None: ...

# Protocol for historical data provider
class BacktestHistoricalDataProvider(Protocol):
    """Protocol for historical data access during backtesting."""
    def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]: ...
    
    def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[pd.DataFrame]: ...

# Type aliases for better readability
LoggerServiceType = LoggerService
SimulatedMarketPriceServiceType = MarketPriceService
PortfolioManagerType = PortfolioManager
FeatureEngineType = FeatureEngine
PredictionServiceType = PredictionService
RiskManagerType = RiskManager
StrategyArbitratorType = StrategyArbitrator
