"""Tests for the data_ingestor module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import MarketDataEvent
from gal_friday.data_ingestor import DataIngestor


@pytest.fixture
def data_config():
    """Fixture providing data ingestor configuration."""
    return {
        "data_collection": {
            "storage_path": "./data",
            "exchanges": ["kraken"],
            "symbols": ["BTC/USD", "ETH/USD"],
            "timeframes": ["1m", "5m", "1h", "1d"],
            "batch_size": 1000,
            "max_retries": 3,
            "retry_delay_seconds": 5,
            "update_interval_seconds": 60,
            "historical_start_date": "2023-01-01",
        },
        "database": {"connection_string": "sqlite:///:memory:", "table_prefix": "market_data_"},
    }


def test_data_ingestor_initialization(data_config, event_bus):
    """Test that the DataIngestor initializes correctly."""
    config = ConfigManager(config_dict=data_config)
    data_ingestor = DataIngestor(config, event_bus)

    assert data_ingestor is not None
    assert data_ingestor.symbols == data_config["data_collection"]["symbols"]
    assert data_ingestor.timeframes == data_config["data_collection"]["timeframes"]
    assert data_ingestor.batch_size == data_config["data_collection"]["batch_size"]
    assert data_ingestor.max_retries == data_config["data_collection"]["max_retries"]


@patch("ccxt.kraken")
def test_data_ingestor_fetch_historical_data(mock_ccxt_kraken, data_config, event_bus):
    """Test fetching historical data."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange

    # Create sample OHLCV data
    now = datetime.now().timestamp() * 1000
    hour_ms = 60 * 60 * 1000
    sample_data = [
        [now - (i * hour_ms), 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i]
        for i in range(10, 0, -1)
    ]
    mock_exchange.fetch_ohlcv.return_value = sample_data

    # Initialize ingestor
    config = ConfigManager(config_dict=data_config)
    with patch("gal_friday.data_ingestor.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken

        data_ingestor = DataIngestor(config, event_bus)
        data_ingestor.exchange = mock_exchange

        # Fetch historical data
        ohlcv_data = data_ingestor.fetch_historical_data(
            symbol="BTC/USD", timeframe="1h", start_date=datetime.now() - timedelta(days=1)
        )

    # Verify data was fetched
    assert mock_exchange.fetch_ohlcv.call_count == 1
    assert len(ohlcv_data) == 10
    assert isinstance(ohlcv_data, pd.DataFrame)
    assert all(col in ohlcv_data.columns for col in ["open", "high", "low", "close", "volume"])


@patch("ccxt.kraken")
def test_data_ingestor_process_data(mock_ccxt_kraken, data_config, event_bus):
    """Test processing raw data."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange

    # Create sample OHLCV data
    now = datetime.now().timestamp() * 1000
    hour_ms = 60 * 60 * 1000
    sample_data = [
        [now - (i * hour_ms), 100 + i, 105 + i, 95 + i, 102 + i, 1000 + i]
        for i in range(10, 0, -1)
    ]

    # Initialize ingestor
    config = ConfigManager(config_dict=data_config)
    with patch("gal_friday.data_ingestor.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken

        data_ingestor = DataIngestor(config, event_bus)

        # Process raw data
        processed_data = data_ingestor.process_data(sample_data, symbol="BTC/USD", timeframe="1h")

    # Verify data was processed correctly
    assert len(processed_data) == 10
    assert isinstance(processed_data, pd.DataFrame)
    assert all(col in processed_data.columns for col in ["open", "high", "low", "close", "volume"])
    assert processed_data["symbol"].iloc[0] == "BTC/USD"
    assert processed_data["timeframe"].iloc[0] == "1h"


@patch("ccxt.kraken")
def test_data_ingestor_store_data(mock_ccxt_kraken, data_config, event_bus):
    """Test storing data."""
    # Set up mock exchange and database
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange
    mock_db = MagicMock()

    # Create sample dataframe
    sample_df = pd.DataFrame(
        {
            "timestamp": [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)],
            "open": [100 + i for i in range(10, 0, -1)],
            "high": [105 + i for i in range(10, 0, -1)],
            "low": [95 + i for i in range(10, 0, -1)],
            "close": [102 + i for i in range(10, 0, -1)],
            "volume": [1000 + i for i in range(10, 0, -1)],
            "symbol": ["BTC/USD"] * 10,
            "timeframe": ["1h"] * 10,
        }
    )

    # Initialize ingestor
    config = ConfigManager(config_dict=data_config)
    with patch("gal_friday.data_ingestor.ccxt") as mock_ccxt:
        with patch("gal_friday.data_ingestor.create_engine") as mock_create_engine:
            mock_ccxt.kraken = mock_ccxt_kraken
            mock_create_engine.return_value = mock_db

            data_ingestor = DataIngestor(config, event_bus)
            data_ingestor.db_engine = mock_db

            # Store data
            data_ingestor.store_data(sample_df)

    # Verify data was stored
    assert mock_db.execute.call_count >= 1 or mock_db.connect.call_count >= 1


@patch("ccxt.kraken")
def test_data_ingestor_publish_market_data(mock_ccxt_kraken, data_config, event_bus):
    """Test publishing market data events."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange

    # Set up mock event bus
    mock_event_bus = MagicMock()

    # Initialize ingestor
    config = ConfigManager(config_dict=data_config)
    with patch("gal_friday.data_ingestor.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken

        data_ingestor = DataIngestor(config, mock_event_bus)

        # Create sample latest data
        latest_data = {
            "BTC/USD": {"price": 50000.0, "timestamp": datetime.now()},
            "ETH/USD": {"price": 3000.0, "timestamp": datetime.now()},
        }

        # Store sample data
        data_ingestor.latest_market_data = latest_data

        # Publish market data
        data_ingestor.publish_market_data()

    # Verify events were published
    assert mock_event_bus.publish.call_count == 2

    # Check event types
    for call in mock_event_bus.publish.call_args_list:
        event = call.args[0]
        assert isinstance(event, MarketDataEvent)
        assert event.symbol in ["BTC/USD", "ETH/USD"]
        assert event.price in [50000.0, 3000.0]


@patch("ccxt.kraken")
def test_data_ingestor_update_data(mock_ccxt_kraken, data_config, event_bus):
    """Test updating market data."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.side_effect = lambda symbol: {
        "BTC/USD": {
            "symbol": "BTC/USD",
            "last": 51000.0,
            "timestamp": datetime.now().timestamp() * 1000,
        },
        "ETH/USD": {
            "symbol": "ETH/USD",
            "last": 3100.0,
            "timestamp": datetime.now().timestamp() * 1000,
        },
    }[symbol]
    mock_ccxt_kraken.return_value = mock_exchange

    # Initialize ingestor
    config = ConfigManager(config_dict=data_config)
    with patch("gal_friday.data_ingestor.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken

        data_ingestor = DataIngestor(config, event_bus)
        data_ingestor.exchange = mock_exchange

        # Update data
        data_ingestor.update_market_data()

    # Verify data was updated
    assert len(data_ingestor.latest_market_data) == 2
    assert "BTC/USD" in data_ingestor.latest_market_data
    assert "ETH/USD" in data_ingestor.latest_market_data
    assert data_ingestor.latest_market_data["BTC/USD"]["price"] == 51000.0
    assert data_ingestor.latest_market_data["ETH/USD"]["price"] == 3100.0


@patch("ccxt.kraken")
def test_data_ingestor_error_handling(mock_ccxt_kraken, data_config, event_bus):
    """Test error handling in data fetching."""
    # Set up mock exchange with error
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.side_effect = Exception("API rate limit exceeded")
    mock_ccxt_kraken.return_value = mock_exchange

    # Set up mock logger
    mock_logger = MagicMock()

    # Initialize ingestor
    config = ConfigManager(config_dict=data_config)
    with patch("gal_friday.data_ingestor.ccxt") as mock_ccxt:
        with patch("gal_friday.data_ingestor.logger", mock_logger):
            mock_ccxt.kraken = mock_ccxt_kraken

            data_ingestor = DataIngestor(config, event_bus)
            data_ingestor.exchange = mock_exchange

            # Attempt to fetch data (should handle the error)
            result = data_ingestor.fetch_historical_data(
                symbol="BTC/USD", timeframe="1h", start_date=datetime.now() - timedelta(days=1)
            )

    # Verify error was logged
    assert mock_logger.error.call_count >= 1
    assert result is None or isinstance(result, pd.DataFrame)
