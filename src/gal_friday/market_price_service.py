"""Abstract interface for retrieving real-time market price information."""

import abc
from decimal import Decimal
from typing import Optional, Tuple


class MarketPriceService(abc.ABC):
    """Defines the interface for components providing real-time market prices."""

    @abc.abstractmethod
    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """Get the latest known market price for a trading pair."""
        # This could be the mid-price, last trade price, etc., depending on implementation.
        raise NotImplementedError

    @abc.abstractmethod
    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Get the current best bid and ask prices."""
        # Returns a tuple (bid, ask)
        raise NotImplementedError

    # Add other methods as needed, e.g., get_vwap, get_last_trade 