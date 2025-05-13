"""
Tests for the HistoricalDataService interface contract.

These tests verify that implementations of HistoricalDataService correctly
follow the interface contract and behavior requirements.
"""

from abc import ABC
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from gal_friday.interfaces.historical_data_service_interface import HistoricalDataService
from gal_friday.kraken_historical_data_service import KrakenHistoricalDataService


class MockHistoricalDataService(HistoricalDataService):
    """A minimal implementation of HistoricalDataService for testing."""

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Implement abstract method for test."""
        if trading_pair != "XRP/USD":
            return None

        # Create a mock DataFrame with OHLCV data
        dates = pd.date_range(start=start_time, end=end_time, freq=interval.replace("m", "T"))
        if len(dates) == 0:
            return None

        data = {
            "timestamp": dates,
            "open": np.random.random(len(dates)) + 0.5,
            "high": np.random.random(len(dates)) + 0.55,
            "low": np.random.random(len(dates)) + 0.45,
            "close": np.random.random(len(dates)) + 0.5,
            "volume": np.random.random(len(dates)) * 1000,
        }

        return pd.DataFrame(data)

    async def get_historical_trades(
        self, trading_pair: str, start_time: datetime, end_time: datetime
    ) -> Optional[pd.DataFrame]:
        """Implement abstract method for test."""
        if trading_pair != "XRP/USD":
            return None

        # Create a mock DataFrame with trade data
        num_trades = 100
        timestamps = [start_time + timedelta(seconds=i * 30) for i in range(num_trades)]
        prices = np.random.random(num_trades) + 0.5
        volumes = np.random.random(num_trades) * 10
        sides = np.random.choice(["buy", "sell"], num_trades)

        data = {"timestamp": timestamps, "price": prices, "volume": volumes, "side": sides}

        return pd.DataFrame(data)

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> Optional[pd.Series]:
        """Implement abstract method for test."""
        if trading_pair != "XRP/USD":
            return None

        # Create a mock Series representing the next bar
        next_timestamp = timestamp + timedelta(minutes=1)
        return pd.Series(
            {
                "timestamp": next_timestamp,
                "open": 0.52,
                "high": 0.54,
                "low": 0.51,
                "close": 0.53,
                "volume": 1000.0,
            }
        )

    def get_atr(
        self, trading_pair: str, timestamp: datetime, period: int = 14
    ) -> Optional[Decimal]:
        """Implement abstract method for test."""
        if trading_pair != "XRP/USD":
            return None

        return Decimal("0.015")


class IncompleteHistoricalDataService(ABC):
    """A class that inherits from HistoricalDataService but doesn't implement all methods."""

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
    ) -> Optional[pd.DataFrame]:
        """Implement one abstract method but not others."""


def test_historical_data_service_is_abstract():
    """Test that HistoricalDataService cannot be instantiated directly."""
    with pytest.raises(TypeError):
        HistoricalDataService()


def test_historical_data_service_requires_implementation():
    """Test that a class inheriting from HistoricalDataService.

    must implement all abstract methods.
    """
    with pytest.raises(TypeError):
        IncompleteHistoricalDataService()


def test_can_instantiate_concrete_implementation():
    """Test that a concrete implementation can be instantiated."""
    service = MockHistoricalDataService()
    assert isinstance(service, HistoricalDataService)


@pytest.mark.asyncio
async def test_get_historical_ohlcv_contract():
    """Test that the get_historical_ohlcv method follows the contract."""
    service = MockHistoricalDataService()

    # Set up test parameters
    trading_pair = "XRP/USD"
    start_time = datetime.now() - timedelta(days=7)
    end_time = datetime.now()
    interval = "1m"

    # Should return a DataFrame for a valid request
    df = await service.get_historical_ohlcv(trading_pair, start_time, end_time, interval)
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "open" in df.columns
    assert "high" in df.columns
    assert "low" in df.columns
    assert "close" in df.columns
    assert "volume" in df.columns

    # Should return None for an invalid pair
    df = await service.get_historical_ohlcv("INVALID/PAIR", start_time, end_time, interval)
    assert df is None


@pytest.mark.asyncio
async def test_get_historical_trades_contract():
    """Test that the get_historical_trades method follows the contract."""
    service = MockHistoricalDataService()

    # Set up test parameters
    trading_pair = "XRP/USD"
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()

    # Should return a DataFrame for a valid request
    df = await service.get_historical_trades(trading_pair, start_time, end_time)
    assert isinstance(df, pd.DataFrame)
    assert "timestamp" in df.columns
    assert "price" in df.columns
    assert "volume" in df.columns

    # Should return None for an invalid pair
    df = await service.get_historical_trades("INVALID/PAIR", start_time, end_time)
    assert df is None


def test_get_next_bar_contract():
    """Test that the get_next_bar method follows the contract."""
    service = MockHistoricalDataService()

    # Set up test parameters
    trading_pair = "XRP/USD"
    timestamp = datetime.now() - timedelta(minutes=5)

    # Should return a Series for a valid request
    series = service.get_next_bar(trading_pair, timestamp)
    assert isinstance(series, pd.Series)
    assert "timestamp" in series.index or "timestamp" in series
    assert "open" in series.index or "open" in series
    assert "high" in series.index or "high" in series
    assert "low" in series.index or "low" in series
    assert "close" in series.index or "close" in series

    # Should return None for an invalid pair
    series = service.get_next_bar("INVALID/PAIR", timestamp)
    assert series is None


def test_get_atr_contract():
    """Test that the get_atr method follows the contract."""
    service = MockHistoricalDataService()

    # Set up test parameters
    trading_pair = "XRP/USD"
    timestamp = datetime.now() - timedelta(minutes=5)

    # Should return a Decimal for a valid request
    atr = service.get_atr(trading_pair, timestamp)
    assert isinstance(atr, Decimal)

    # Should accept a period parameter
    atr = service.get_atr(trading_pair, timestamp, period=7)
    assert isinstance(atr, Decimal)

    # Should return None for an invalid pair
    atr = service.get_atr("INVALID/PAIR", timestamp)
    assert atr is None


@pytest.mark.parametrize("implementation", [KrakenHistoricalDataService])
def test_real_implementations_conform_to_interface(implementation):
    """Test that real implementations conform to the interface."""
    # This test verifies that our actual implementations properly implement the interface
    assert issubclass(implementation, HistoricalDataService)

    # Verify that the implementations have the required methods
    assert hasattr(implementation, "get_historical_ohlcv")
    assert hasattr(implementation, "get_historical_trades")
    assert hasattr(implementation, "get_next_bar")
    assert hasattr(implementation, "get_atr")
