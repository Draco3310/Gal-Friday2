"""Abstract interface for retrieving real-time market price information."""

import abc
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional


class MarketPriceService(abc.ABC):
    """
    Abstract Base Class for components providing real-time market prices.

    Implementations should:
    1. Handle connection setup/teardown via start()/stop().
    2. Fetch and cache price data efficiently.
    3. Implement logic for get_latest_price and get_bid_ask_spread.
    4. Provide data freshness checks via is_price_fresh().
    5. Handle errors gracefully (e.g., network issues, unknown pairs)
       and return None when data is unavailable or stale.
    6. Be implemented asynchronously.
    """

    @abc.abstractmethod
    async def start(self) -> None:
        """
        Initialize the service, establish connections, and start any background tasks.

        Should be called once during application startup.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def stop(self) -> None:
        """
        Clean up resources, close connections, and stop background tasks.

        Should be called once during application shutdown.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """
        Get the latest known market price for a trading pair.

        This could be the mid-price, last trade price, or other relevant price
        depending on the implementation and data source.

        Args
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns
        -------
            The latest price as a Decimal, or None if the price is
            unavailable, stale, or the pair is not supported.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[tuple[Decimal, Decimal]]:
        """
        Get the current best bid and ask prices from the data source.

        Args
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns
        -------
            A tuple containing (best_bid, best_ask) as Decimals,
            or None if the spread is unavailable, stale, or the pair
            is not supported. Returns None if bid >= ask (crossed book).
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_price_timestamp(self, trading_pair: str) -> Optional[datetime]:
        """Get the timestamp of the latest price data for a trading pair.

        Used by get_latest_price() and get_bid_ask_spread() to determine the
        freshness of the cached market data.

        Args
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns
        -------
            The UTC datetime of the last price update, or None if no data exists.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def is_price_fresh(self, trading_pair: str, max_age_seconds: float = 60.0) -> bool:
        """Check if the price data for a trading pair is recent enough.

        Determines if the most recent price data for the specified trading pair
        is fresh enough based on the maximum age threshold.

        Args
        ----
            trading_pair: The trading pair symbol (e.g., "XRP/USD").
            max_age_seconds: The maximum allowed age in seconds for the data
                           to be considered fresh. Defaults to 60 seconds.

        Returns
        -------
            True if data exists and its timestamp is within max_age_seconds
            from the current time (UTC), False otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def convert_amount(
        self, from_amount: Decimal, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """
        Convert an amount from one currency to another.

        Args
        ----
            from_amount: The amount to convert.
            from_currency: The currency of the from_amount (e.g., "BTC").
            to_currency: The target currency (e.g., "USD").

        Returns
        -------
            The converted amount as a Decimal, or None if conversion
            is not possible (e.g., unknown currency, no exchange rate).
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        timeframe: str, # e.g., "1d" for daily, "1h" for hourly (Kraken uses minutes)
        since: datetime, # Start timestamp (UTC)
        limit: Optional[int] = None # Number of data points
    ) -> Optional[list[dict[str, Any]]]: # List of OHLCV candles
        """
        Fetch historical OHLCV data for a trading pair.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").
            timeframe: The timeframe for the candles (e.g., "1m", "1h", "1d" - map to Kraken's minute values).
            since: Python datetime object indicating the start time for fetching data (UTC).
                   Kraken 'since' is exclusive start of time slice, returns data *after* this time.
            limit: The maximum number of candles to return. Kraken's OHLC 'limit' is not directly supported,
                   it returns up to 720 data points. We might need to handle this.

        Returns
        -------
            A list of dictionaries, where each dictionary represents an OHLCV candle:
            {'timestamp': datetime_obj, 'open': Decimal, 'high': Decimal, 
             'low': Decimal, 'close': Decimal, 'volume': Decimal},
            or None if data is unavailable or an error occurs. Timestamps are UTC.
        """
        raise NotImplementedError
