"""Backtest historical data provider implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


class BacktestHistoricalDataProvider:
    """Provides historical data for backtesting."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the backtest historical data provider."""

    def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Get historical data for the given symbol and time range.

        Args:
            symbol: The trading symbol to get data for
            start_time: Start of the time range
            end_time: End of the time range
            interval: Data interval (e.g., '1d', '1h', '1m')

        Returns:
        -------
            DataFrame with historical data or None if not available
        """
        return None
