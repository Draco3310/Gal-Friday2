"""Service for computing asset correlation matrices."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from gal_friday.logger_service import LoggerService  # noqa: TC001
from gal_friday.market_price_service import MarketPriceService  # noqa: TC001

if TYPE_CHECKING:  # pragma: no cover - typing-only imports
    from collections.abc import Sequence

MIN_SERIES_REQUIRED = 2


class CorrelationService:
    """Calculate correlations between trading pairs."""

    def __init__(self, price_service: MarketPriceService, logger: LoggerService) -> None:
        """Initialize the service."""
        self.price_service = price_service
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def get_portfolio_correlation_matrix(
        self, trading_pairs: Sequence[str], lookback_hours: int = 24,
    ) -> dict[str, dict[str, float]]:
        """Return a price correlation matrix for the given trading pairs."""
        since = datetime.now(UTC) - timedelta(hours=lookback_hours)
        series: dict[str, list[float]] = {}

        for pair in trading_pairs:
            try:
                data = await self.price_service.get_historical_ohlcv(pair, "1h", since)
            except Exception as exc:  # pragma: no cover - network failures
                self.logger.error(
                    f"Failed to fetch OHLCV for {pair}: {exc}",
                    source_module=self._source_module,
                )
                continue
            if not data:
                continue
            closes = [float(candle["close"]) for candle in data]
            series[pair] = closes

        if len(series) < MIN_SERIES_REQUIRED:
            return {pair: {pair: 1.0} for pair in trading_pairs}

        df = pd.DataFrame(series)
        matrix = df.pct_change().dropna().corr().fillna(0.0)
        return matrix.to_dict()
