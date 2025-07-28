"""History repository for accessing historical market data.

This module provides repository classes for retrieving and managing
historical market data from time-series databases.
"""

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
        self,
        trading_pair: str,
        limit: int,
        interval: str,
    ) -> pd.DataFrame | None:
        """Fetch recent OHLCV candles for a trading pair."""
        try:
            interval_minutes = int(interval.rstrip("m"))
        except ValueError:
            self.logger.exception(
                "Invalid interval '%s' for get_recent_ohlcv",
                interval,
                source_module=self._source_module,
            )
            return None

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(minutes=interval_minutes * limit)
        try:
            raw = await self.ts_db.query_ohlcv(trading_pair, interval, start_time, end_time)
        except Exception:  # pragma: no cover - network/DB errors
            self.logger.exception("Failed to query OHLCV for %s: %s", trading_pair, source_module=self._source_module)
            return None

        if not raw:
            return None
        df = pd.DataFrame(raw)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df.tail(limit)

    async def get_feature_history(
        self,
        trading_pair: str,
        lookback: int,
        interval: str = "1h",
    ) -> pd.DataFrame | None:
        """Fetch historical feature values for a trading pair.

        Args:
            trading_pair: The trading pair to fetch features for
            lookback: Number of data points to retrieve
            interval: Time interval (default: "1h")

        Returns:
            DataFrame with columns: timestamp, feature_name, value
            or None if no data is available
        """
        try:
            # For now, we'll return OHLCV data transformed into feature format
            # In a production system, this would query a dedicated feature store
            ohlcv_data = await self.get_recent_ohlcv(trading_pair, lookback, interval)

            if ohlcv_data is None or ohlcv_data.empty:
                return None

            # Transform OHLCV data into feature format
            feature_data = []

            for timestamp, row in ohlcv_data.iterrows():
                # Add basic price features
                feature_data.extend(
                    [
                        {
                            "timestamp": timestamp,
                            "feature_name": f"{col}_price" if col != "volume" else col,
                            "value": float(row[col]),
                        }
                        for col in ["open", "high", "low", "close", "volume"]
                        if col in row
                    ],
                )

                # Add calculated features
                if all(col in row for col in ["high", "low"]):
                    feature_data.append(
                        {
                            "timestamp": timestamp,
                            "feature_name": "price_range",
                            "value": float(row["high"] - row["low"]),
                        },
                    )

                if all(col in row for col in ["close", "open"]):
                    feature_data.append(
                        {
                            "timestamp": timestamp,
                            "feature_name": "price_change",
                            "value": float(row["close"] - row["open"]),
                        },
                    )

            if not feature_data:
                return None

            return pd.DataFrame(feature_data)

        except Exception:
            self.logger.exception(
                "Failed to get feature history for %s",
                trading_pair,
                source_module=self._source_module,
            )
            return None
