"""Interface definition for retrieving historical market data."""

from datetime import datetime
from decimal import Decimal
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class HistoricalDataService(Protocol):
    """Protocol for components providing historical market data."""

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,  # e.g., "1m", "5m", "1h"
    ) -> pd.DataFrame | None:
        """Get historical OHLCV data for a given pair, time range, and interval."""
        # DataFrame should have columns like
        # 'timestamp', 'open', 'high', 'low', 'close', 'volume'
        # Timestamp should ideally be the index and timezone-aware (UTC)
        ...

    async def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Get historical trade data for a given pair and time range."""
        # DataFrame should have columns like ['timestamp', 'price', 'volume',
        # 'side']
        ...

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> pd.Series | None:
        """Get the next available OHLCV bar after the given timestamp.

        Args:
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD")
            timestamp: The reference timestamp

        Returns:
        -------
            A pandas Series[Any] containing the OHLCV data for the next bar,
            or None if no next bar is available
        """
        ...

    def get_atr(self, trading_pair: str, timestamp: datetime, period: int = 14) -> Decimal | None:
        """Get the Average True Range indicator value at the given timestamp.

        Args:
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD")
            timestamp: The reference timestamp
            period: The ATR calculation period, default is 14

        Returns:
        -------
            The ATR value as a Decimal, or None if it cannot be calculated
        """
        ...
