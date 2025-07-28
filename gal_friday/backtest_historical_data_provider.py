"""Backtest historical data provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd


class BacktestHistoricalDataProvider:
    """Provides historical data for backtesting."""

    def __init__(self) -> None:
        """Initialize the backtest historical data provider."""

    def get_historical_data(
        self,
        _symbol: str,
        _start_time: datetime,
        _end_time: datetime,
        _interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Get historical data for the given symbol and time range.

        Args:
            _symbol: The trading symbol to get data for
            _start_time: Start of the time range
            _end_time: End of the time range
            _interval: Data interval (e.g., '1d', '1h', '1m')

        Returns:
        -------
            DataFrame with historical data or None if not available
        """
        return None
