from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gal_friday.dal.influxdb_client import TimeSeriesDB
    from gal_friday.logger_service import LoggerService


class HistoryRepository:
    """Repository for historical OHLCV data stored in InfluxDB."""

    def __init__(self, ts_db: TimeSeriesDB, logger: LoggerService) -> None:
        """Initialize the instance."""
        self.ts_db = ts_db
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def get_recent_ohlcv(
        self, trading_pair: str, limit: int, interval: str,
    ) -> pd.DataFrame | None:
        """Fetch recent OHLCV candles for a trading pair."""
        try:
            interval_minutes = int(interval.rstrip("m"))
        except ValueError:
            self.logger.exception(
                "Invalid interval '%s' for get_recent_ohlcv", interval, source_module=self._source_module,
            )
            return None

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(minutes=interval_minutes * limit)
        try:
            raw = await self.ts_db.query_ohlcv(trading_pair, interval, start_time, end_time)
        except Exception as exc:  # pragma: no cover - network/DB errors
            self.logger.exception(
                "Failed to query OHLCV for %s: %s", trading_pair, exc, source_module=self._source_module)
            return None

        if not raw:
            return None
        df = pd.DataFrame(raw)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df.tail(limit)
