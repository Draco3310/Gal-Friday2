import logging
from typing import Any, Dict, List

from ..simulated_market_price_service import (
    DataRequest,
    HistoricalDataPoint,
    HistoricalDataProvider,
)


class APIDataProvider(HistoricalDataProvider):
    """Placeholder provider fetching data from an external API."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

    async def fetch_data(self, request: DataRequest) -> List[HistoricalDataPoint]:
        self.logger.info("APIDataProvider not implemented")
        return []

    async def validate_symbol(self, symbol: str) -> bool:
        # Assume all symbols are valid until API integration exists
        return True
