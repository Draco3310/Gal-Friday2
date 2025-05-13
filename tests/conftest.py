"""
Shared pytest fixtures for Gal-Friday2 tests.

This file contains fixtures that can be used across multiple test files.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager


@pytest.fixture
def event_bus():
    """Fixture providing a clean PubSubManager instance."""
    return PubSubManager()


@pytest.fixture
def base_config():
    """Fixture providing a basic configuration."""
    return {
        "app_name": "Gal-Friday2",
        "environment": "test",
        "log_level": "INFO",
        "exchanges": {
            "kraken": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "symbols": ["BTC/USD", "ETH/USD"],
                "timeframes": ["1m", "5m", "1h", "1d"],
            }
        },
        "database": {"connection_string": "sqlite:///:memory:"},
        "backtesting": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000.0,
            "symbols": ["BTC/USD", "ETH/USD"],
            "strategy": "momentum",
            "strategy_params": {"lookback_period": 14, "threshold": 0.05},
        },
        "trading": {
            "max_position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.05,  # 5% stop loss
            "take_profit_pct": 0.1,  # 10% take profit
            "max_open_trades": 5,
            "default_timeframe": "1h",
        },
    }


@pytest.fixture
def config_manager(base_config):
    """Fixture providing a ConfigManager instance."""
    return ConfigManager(config_dict=base_config)


@pytest.fixture
def mock_ohlcv_data():
    """Fixture providing mock OHLCV price data for testing."""
    # Create sample dates
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(100)]

    # Create sample BTC/USD data with some realistic price movements
    btc_base_price = 50000
    btc_close = [btc_base_price + (np.sin(i / 10) * 1000) + (i * 10) for i in range(100)]
    btc_open = [close - (np.random.randn() * 100) for close in btc_close]
    btc_high = [
        max(open_price, close) + (np.random.randn() * 100)
        for open_price, close in zip(btc_open, btc_close)
    ]
    btc_low = [
        min(open_price, close) - (np.random.randn() * 100)
        for open_price, close in zip(btc_open, btc_close)
    ]
    btc_volume = [10000 + (np.random.randn() * 5000) for _ in range(100)]

    btc_df = pd.DataFrame(
        {
            "open": btc_open,
            "high": btc_high,
            "low": btc_low,
            "close": btc_close,
            "volume": btc_volume,
        },
        index=dates,
    )

    # Create sample ETH/USD data correlated with BTC but with some differences
    eth_base_price = 3000
    eth_close = [
        eth_base_price + (btc_close[i] - btc_base_price) * 0.05 + (np.random.randn() * 50)
        for i in range(100)
    ]
    eth_open = [close - (np.random.randn() * 20) for close in eth_close]
    eth_high = [
        max(open_price, close) + (np.random.randn() * 30)
        for open_price, close in zip(eth_open, eth_close)
    ]
    eth_low = [
        min(open_price, close) - (np.random.randn() * 30)
        for open_price, close in zip(eth_open, eth_close)
    ]
    eth_volume = [50000 + (np.random.randn() * 10000) for _ in range(100)]

    eth_df = pd.DataFrame(
        {
            "open": eth_open,
            "high": eth_high,
            "low": eth_low,
            "close": eth_close,
            "volume": eth_volume,
        },
        index=dates,
    )

    return {"BTC/USD": btc_df, "ETH/USD": eth_df}


@pytest.fixture
def mock_exchange():
    """Fixture providing a mock exchange instance."""
    mock_exchange = MagicMock()

    # Mock the fetch_ticker method
    def mock_fetch_ticker(symbol):
        if symbol == "BTC/USD":
            return {
                "symbol": symbol,
                "bid": 50000.0,
                "ask": 50100.0,
                "last": 50050.0,
                "datetime": datetime.now().isoformat(),
            }
        elif symbol == "ETH/USD":
            return {
                "symbol": symbol,
                "bid": 3000.0,
                "ask": 3010.0,
                "last": 3005.0,
                "datetime": datetime.now().isoformat(),
            }
        else:
            raise Exception(f"Symbol {symbol} not found")

    mock_exchange.fetch_ticker.side_effect = mock_fetch_ticker

    # Mock the fetch_ohlcv method
    def mock_fetch_ohlcv(symbol, timeframe="1h", since=None, limit=None):
        # Return 10 candles of mock data
        now = datetime.now().timestamp() * 1000
        hour_ms = 60 * 60 * 1000

        if symbol == "BTC/USD":
            base_price = 50000
            return [
                [
                    now - (i * hour_ms),
                    base_price + (i * 100),
                    base_price + (i * 100) + 200,
                    base_price + (i * 100) - 100,
                    base_price + (i * 100) + 50,
                    10 + i,
                ]
                for i in range(10, 0, -1)
            ]
        elif symbol == "ETH/USD":
            base_price = 3000
            return [
                [
                    now - (i * hour_ms),
                    base_price + (i * 10),
                    base_price + (i * 10) + 20,
                    base_price + (i * 10) - 10,
                    base_price + (i * 10) + 5,
                    100 + i,
                ]
                for i in range(10, 0, -1)
            ]
        else:
            return []

    mock_exchange.fetch_ohlcv.side_effect = mock_fetch_ohlcv

    # Add other necessary methods
    mock_exchange.create_order.return_value = {
        "id": "12345",
        "timestamp": datetime.now().timestamp() * 1000,
        "status": "open",
        "symbol": "BTC/USD",
        "type": "limit",
        "side": "buy",
        "price": 50000.0,
        "amount": 1.0,
    }

    mock_exchange.fetch_order.return_value = {
        "id": "12345",
        "timestamp": datetime.now().timestamp() * 1000,
        "status": "closed",
        "symbol": "BTC/USD",
        "type": "limit",
        "side": "buy",
        "price": 50000.0,
        "amount": 1.0,
        "filled": 1.0,
        "cost": 50000.0,
    }

    mock_exchange.fetch_balance.return_value = {
        "total": {"USD": 100000.0, "BTC": 2.0, "ETH": 10.0},
        "free": {"USD": 50000.0, "BTC": 1.0, "ETH": 5.0},
        "used": {"USD": 50000.0, "BTC": 1.0, "ETH": 5.0},
    }

    mock_exchange.has = {"fetchOHLCV": True, "fetchTicker": True}
    mock_exchange.load_markets.return_value = {
        "BTC/USD": {"id": "XBTUSD", "symbol": "BTC/USD", "base": "BTC", "quote": "USD"},
        "ETH/USD": {"id": "ETHUSD", "symbol": "ETH/USD", "base": "ETH", "quote": "USD"},
    }

    return mock_exchange
