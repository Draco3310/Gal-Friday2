"""Tests for HistoricalDataLoader provider selection and error handling."""

from datetime import datetime

import pytest

from gal_friday.simulated_market_price_service import (
    DataLoadingError,
    DataRequest,
    DataSource,
    HistoricalDataLoader,
    HistoricalDataPoint,
    HistoricalDataProvider,
)


class MockProvider(HistoricalDataProvider):
    """Simple provider used for testing."""

    def __init__(self, symbols: list[str], result: list[HistoricalDataPoint], *, raise_error: bool = False) -> None:
        self.symbols = symbols
        self.result = result
        self.raise_error = raise_error
        self.fetch_called = 0
        self.validate_called = 0

    async def fetch_data(self, request: DataRequest) -> list[HistoricalDataPoint]:
        self.fetch_called += 1
        if self.raise_error:
            raise ValueError("failed")
        return self.result

    async def validate_symbol(self, symbol: str) -> bool:
        self.validate_called += 1
        return symbol in self.symbols


@pytest.mark.asyncio
async def test_select_by_data_source() -> None:
    """Provider is chosen based on request.data_source."""
    data = [HistoricalDataPoint(datetime.utcnow(), "BTC", 1, 1, 1, 1, 1)]
    p1 = MockProvider(["ETH"], [])
    p2 = MockProvider(["BTC"], data)

    loader = HistoricalDataLoader({})
    loader.providers = {DataSource.YAHOO_FINANCE: p1, DataSource.KRAKEN: p2}

    request = DataRequest(
        symbol="BTC",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow(),
        frequency="1m",
        data_source=DataSource.KRAKEN,
    )

    result = await loader._load_from_provider(request)

    assert result == data
    assert p2.fetch_called == 1
    assert loader.cache_stats["provider_requests"] == 1
    assert p1.fetch_called == 0


@pytest.mark.asyncio
async def test_auto_provider_selection() -> None:
    """First provider validating the symbol is used when data_source is None."""
    data = [HistoricalDataPoint(datetime.utcnow(), "BTC", 1, 1, 1, 1, 1)]
    p1 = MockProvider(["ETH"], [])
    p2 = MockProvider(["BTC"], data)

    loader = HistoricalDataLoader({})
    loader.providers = {DataSource.YAHOO_FINANCE: p1, DataSource.KRAKEN: p2}

    request = DataRequest(
        symbol="BTC",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow(),
        frequency="1m",
    )

    result = await loader._load_from_provider(request)

    assert result == data
    assert p2.fetch_called == 1
    assert p1.fetch_called == 0
    assert p1.validate_called == 1
    assert loader.cache_stats["provider_requests"] == 1


@pytest.mark.asyncio
async def test_missing_provider_raises() -> None:
    """DataLoadingError is raised when no provider is found."""
    loader = HistoricalDataLoader({})
    loader.providers = {}

    request = DataRequest(
        symbol="BTC",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow(),
        frequency="1m",
        data_source=DataSource.KRAKEN,
    )

    with pytest.raises(DataLoadingError):
        await loader._load_from_provider(request)

    assert loader.cache_stats["provider_requests"] == 0


@pytest.mark.asyncio
async def test_fetch_error_wrapped() -> None:
    """Errors from provider.fetch_data are wrapped in DataLoadingError."""
    p = MockProvider(["BTC"], [], raise_error=True)
    loader = HistoricalDataLoader({})
    loader.providers = {DataSource.KRAKEN: p}

    request = DataRequest(
        symbol="BTC",
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow(),
        frequency="1m",
        data_source=DataSource.KRAKEN,
    )

    with pytest.raises(DataLoadingError):
        await loader._load_from_provider(request)

    assert loader.cache_stats["provider_requests"] == 1

