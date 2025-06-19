import logging
from pathlib import Path
from typing import Any

import pandas as pd

from gal_friday.simulated_market_price_service import DataRequest, HistoricalDataPoint, HistoricalDataProvider


class LocalFileDataProvider(HistoricalDataProvider):
    """Load historical data from a local file."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        """Initialize the instance."""
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def fetch_data(self, request: DataRequest) -> list[HistoricalDataPoint]:
        """Fetch data for ``request`` from the configured local file."""
        path_str = self.config.get("local_file_path")
        if not path_str:
            self.logger.error("No local_file_path configured for LocalFileDataProvider")
            return []

        path = Path(path_str)
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, parse_dates=["timestamp"])
            elif path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                self.logger.error("Unsupported file format: %s", path.suffix)
                return []
        except Exception as exc:  # pragma: no cover - simple logging
            self.logger.exception("Failed reading %s:", exc)
            return []

        if "symbol" in df.columns:
            df = df[df["symbol"] == request.symbol]
        if "timestamp" in df.columns:
            df = df[(df["timestamp"] >= request.start_date) & (df["timestamp"] <= request.end_date)]

        data: list[HistoricalDataPoint] = [
            HistoricalDataPoint(
                timestamp=row["timestamp"],
                symbol=request.symbol,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0.0))
            for _, row in df.iterrows()
        ]
        return data

    async def validate_symbol(self, symbol: str) -> bool:
        """Basic check that the file exists and contains ``symbol``."""
        path_str = self.config.get("local_file_path")
        if not path_str:
            return False
        path = Path(path_str)
        if not path.exists():
            return False
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:  # pragma: no cover - best effort
            return False
        return "symbol" in df.columns
