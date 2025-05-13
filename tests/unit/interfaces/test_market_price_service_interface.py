"""
Tests for the MarketPriceService interface contract.

These tests verify that implementations of MarketPriceService correctly
follow the interface contract and behavior requirements.
"""

from abc import ABC
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Tuple

import pytest

from gal_friday.interfaces.market_price_service_interface import MarketPriceService
from gal_friday.market_price.kraken_service import KrakenMarketPriceService


class MockMarketPriceService(MarketPriceService):
    """A minimal implementation of MarketPriceService for testing."""

    async def start(self) -> None:
        """Implement abstract method for test."""

    async def stop(self) -> None:
        """Implement abstract method for test."""

    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """Implement abstract method for test."""
        if trading_pair == "XRP/USD":
            return Decimal("0.5123")
        return None

    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Implement abstract method for test."""
        if trading_pair == "XRP/USD":
            return Decimal("0.5120"), Decimal("0.5125")
        return None

    async def get_price_timestamp(self, trading_pair: str) -> Optional[datetime]:
        """Implement abstract method for test."""
        if trading_pair == "XRP/USD":
            return datetime.now() - timedelta(seconds=5)
        return None

    async def is_price_fresh(self, trading_pair: str, max_age_seconds: float = 60.0) -> bool:
        """Implement abstract method for test."""
        if trading_pair == "XRP/USD":
            return True
        return False

    async def convert_amount(
        self, from_amount: Decimal, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """Implement abstract method for test."""
        if from_currency == "XRP" and to_currency == "USD":
            return from_amount * Decimal("0.5123")
        return None


class IncompleteMarketPriceService(ABC):
    """A class that inherits from MarketPriceService but doesn't implement all methods."""

    async def start(self) -> None:
        """Implement one abstract method but not others."""


def test_market_price_service_is_abstract():
    """Test that MarketPriceService cannot be instantiated directly."""
    with pytest.raises(TypeError):
        MarketPriceService()


def test_market_price_service_requires_implementation():
    """Test that a class inheriting from MarketPriceService must implement all abstract methods."""
    with pytest.raises(TypeError):
        IncompleteMarketPriceService()


def test_can_instantiate_concrete_implementation():
    """Test that a concrete implementation can be instantiated."""
    service = MockMarketPriceService()
    assert isinstance(service, MarketPriceService)


@pytest.mark.asyncio
async def test_get_latest_price_contract():
    """Test that the get_latest_price method follows the contract."""
    service = MockMarketPriceService()

    # Should return a Decimal for a valid pair
    price = await service.get_latest_price("XRP/USD")
    assert isinstance(price, Decimal)

    # Should return None for an invalid pair
    price = await service.get_latest_price("INVALID/PAIR")
    assert price is None


@pytest.mark.asyncio
async def test_get_bid_ask_spread_contract():
    """Test that the get_bid_ask_spread method follows the contract."""
    service = MockMarketPriceService()

    # Should return a tuple of Decimals for a valid pair
    spread = await service.get_bid_ask_spread("XRP/USD")
    assert isinstance(spread, tuple)
    assert len(spread) == 2
    assert all(isinstance(price, Decimal) for price in spread)
    assert spread[0] <= spread[1]  # Bid should be less than or equal to ask

    # Should return None for an invalid pair
    spread = await service.get_bid_ask_spread("INVALID/PAIR")
    assert spread is None


@pytest.mark.asyncio
async def test_get_price_timestamp_contract():
    """Test that the get_price_timestamp method follows the contract."""
    service = MockMarketPriceService()

    # Should return a datetime for a valid pair
    timestamp = await service.get_price_timestamp("XRP/USD")
    assert isinstance(timestamp, datetime)

    # Should return None for an invalid pair
    timestamp = await service.get_price_timestamp("INVALID/PAIR")
    assert timestamp is None


@pytest.mark.asyncio
async def test_is_price_fresh_contract():
    """Test that the is_price_fresh method follows the contract."""
    service = MockMarketPriceService()

    # Should return a boolean
    is_fresh = await service.is_price_fresh("XRP/USD")
    assert isinstance(is_fresh, bool)

    # Should accept a max_age_seconds parameter
    is_fresh = await service.is_price_fresh("XRP/USD", max_age_seconds=30.0)
    assert isinstance(is_fresh, bool)


@pytest.mark.asyncio
async def test_convert_amount_contract():
    """Test that the convert_amount method follows the contract."""
    service = MockMarketPriceService()

    # Should convert correctly for valid currencies
    amount = await service.convert_amount(Decimal("10"), "XRP", "USD")
    assert isinstance(amount, Decimal)

    # Should return None for invalid currencies
    amount = await service.convert_amount(Decimal("10"), "INVALID", "USD")
    assert amount is None


@pytest.mark.parametrize("implementation", [KrakenMarketPriceService])
def test_real_implementations_conform_to_interface(implementation):
    """Test that real implementations conform to the interface."""
    # This test verifies that our actual implementations properly implement the interface
    assert issubclass(implementation, MarketPriceService)

    # Verify that the implementations have the required methods
    assert hasattr(implementation, "start")
    assert hasattr(implementation, "stop")
    assert hasattr(implementation, "get_latest_price")
    assert hasattr(implementation, "get_bid_ask_spread")
    assert hasattr(implementation, "get_price_timestamp")
    assert hasattr(implementation, "is_price_fresh")
    assert hasattr(implementation, "convert_amount")
