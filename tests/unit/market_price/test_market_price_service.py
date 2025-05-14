"""Tests for the market_price_service module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.event_bus import MarketDataEvent
from tests.unit.market_price.fixtures.mock_market_price_service import MockMarketPriceService
from tests.unit.market_price.fixtures.mock_event_bus import MockEventBus


@pytest.fixture
def market_data_config():
    """Fixture for market data configuration."""
    return {
        "markets": {
            "kraken": {
                "symbols": ["BTC/USD", "ETH/USD"],
                "timeframes": ["1m", "5m", "1h"],
                "api_key": "test_key",
                "api_secret": "test_secret",
            }
        }
    }


def test_market_price_service_initialization(market_data_config):
    """Test that the MarketPriceService initializes correctly."""
    event_bus = MockEventBus()
    service = MockMarketPriceService(market_data_config, event_bus)

    assert service is not None
    assert service.symbols == market_data_config["markets"]["kraken"]["symbols"]
    assert service.exchange_name == "kraken"


@patch("ccxt.kraken")
def test_market_price_service_connect(mock_ccxt_kraken, market_data_config):
    """Test connecting to the exchange."""
    # Set up mock
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange

    # Initialize service
    event_bus = MockEventBus()
    with patch("tests.unit.market_price.fixtures.mock_market_price_service.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        service = MockMarketPriceService(market_data_config, event_bus)
        service.connect()

    # Verify exchange connection
    mock_ccxt_kraken.assert_called_once_with(
        {
            "apiKey": "test_key",
            "secret": "test_secret",
        }
    )
    assert service.exchange == mock_exchange
    mock_exchange.load_markets.assert_called_once()


@patch("ccxt.kraken")
def test_market_price_service_get_ticker(mock_ccxt_kraken, market_data_config):
    """Test fetching ticker data."""
    # Set up mock
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {
        "symbol": "BTC/USD",
        "bid": 50000.0,
        "ask": 50100.0,
        "last": 50050.0,
        "datetime": "2025-01-01T00:00:00.000Z",
    }
    mock_ccxt_kraken.return_value = mock_exchange

    # Initialize service
    event_bus = MockEventBus()
    with patch("tests.unit.market_price.fixtures.mock_market_price_service.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        service = MockMarketPriceService(market_data_config, event_bus)
        service.connect()
        ticker = service.get_ticker("BTC/USD")

    # Verify ticker data
    assert ticker["symbol"] == "BTC/USD"
    assert ticker["bid"] == 50000.0
    assert ticker["ask"] == 50100.0
    assert ticker["last"] == 50050.0
    mock_exchange.fetch_ticker.assert_called_once_with("BTC/USD")


@patch("ccxt.kraken")
def test_market_price_service_subscribe(mock_ccxt_kraken, market_data_config):
    """Test subscribing to market data events."""
    # Set up mock
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange

    # Initialize service
    event_bus = MockEventBus()
    mock_handler = MagicMock()

    with patch("tests.unit.market_price.fixtures.mock_market_price_service.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        service = MockMarketPriceService(market_data_config, event_bus)
        service.connect()

        # Subscribe to market data events
        event_bus.subscribe(MarketDataEvent, mock_handler)

        # Simulate market data update
        service._handle_market_update(
            {
                "symbol": "BTC/USD",
                "bid": 50000.0,
                "ask": 50100.0,
                "last": 50050.0,
                "timestamp": datetime.now().timestamp() * 1000,
            }
        )

    # Verify event was published
    assert mock_handler.call_count == 1
    event = mock_handler.call_args[0][0]
    assert isinstance(event, MarketDataEvent)
    assert event.trading_pair == "BTC/USD"
    assert event.is_snapshot is True
