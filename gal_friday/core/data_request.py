"""Common data request types for the trading system.

This module contains shared data structures used across different
components to avoid circular imports.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DataSource(str, Enum):
    """Supported historical data sources."""

    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    BINANCE = "binance"
    KRAKEN = "kraken"
    LOCAL_FILES = "local_files"
    DATABASE = "database"


@dataclass
class DataRequest:
    """Request for historical market data.

    This data structure is used by various components including
    backtesting engine, data providers, and simulated services.
    """

    symbol: str
    start_date: datetime
    end_date: datetime
    frequency: str = "1m"
    source: str | None = None
    data_source: DataSource | None = None
    include_volume: bool = True
    validate_data: bool = True
    cache_result: bool = True

    def __post_init__(self) -> None:
        """Validate the data request."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        valid_frequencies = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if self.frequency not in valid_frequencies:
            raise ValueError(f"Invalid frequency: {self.frequency}. Must be one of {valid_frequencies}")


@dataclass
class HistoricalDataPoint:
    """Single historical data point."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
