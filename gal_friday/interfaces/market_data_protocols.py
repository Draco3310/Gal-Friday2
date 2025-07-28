"""Market data and price service protocols for type-safe, protocol-based architecture.

This module provides protocol definitions for market data services, enabling
duck-typing compatibility and better architectural separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import datetime
    from decimal import Decimal

    from gal_friday.core.events import (
        MarketDataL2Event,
        MarketDataOHLCVEvent,
        MarketDataTickerEvent,
        MarketDataTradeEvent,
    )

    # Union type for all market data events
    MarketDataEvent = MarketDataL2Event | MarketDataOHLCVEvent | MarketDataTradeEvent | MarketDataTickerEvent


@runtime_checkable
class MarketPriceServiceProtocol(Protocol):
    """Protocol for components providing real-time market prices.

    This protocol defines the interface for market price services that handle
    connection management, price retrieval, and data freshness validation.
    """

    async def start(self) -> None:
        """Initialize the service.

        Establish connections, and start any background tasks needed
        for fetching prices. Should be called once during application startup.
        """
        ...

    async def stop(self) -> None:
        """Clean up resources.

        Close connections, and stop background tasks.
        Should be called once during application shutdown.
        """
        ...

    async def get_latest_price(self, trading_pair: str) -> Decimal | None:
        """Get the latest known market price for a trading pair.

        This could be the mid-price, last trade price, or other relevant price
        depending on the implementation and data source.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns:
            The latest price as a Decimal, or None if the price is
            unavailable, stale, or the pair is not supported.
        """
        ...

    async def get_bid_ask_spread(self, trading_pair: str) -> tuple[Decimal, Decimal] | None:
        """Get the current best bid and ask prices from the data source.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns:
            A tuple containing (best_bid, best_ask) as Decimals,
            or None if the spread is unavailable, stale, or the pair
            is not supported. Returns None if bid >= ask (crossed book).
        """
        ...

    async def get_price_timestamp(self, trading_pair: str) -> datetime | None:
        """Get the timestamp (UTC) associated with the latest price data.

        Used for get_latest_price() and get_bid_ask_spread().

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns:
            The UTC datetime of the last price update, or None if no data exists.
        """
        ...

    async def is_price_fresh(self, trading_pair: str, max_age_seconds: float = 60.0) -> bool:
        """Check if the price data for a trading pair is recent enough.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").
            max_age_seconds: The maximum allowed age in seconds for the data
                           to be considered fresh. Defaults to 60 seconds.

        Returns:
            True if data exists and its timestamp is within max_age_seconds
            from the current time (UTC), False otherwise.
        """
        ...

    async def convert_amount(self, from_amount: Decimal, from_currency: str, to_currency: str) -> Decimal | None:
        """Convert an amount from one currency to another.

        Args:
            from_amount: The amount to convert.
            from_currency: The currency of the from_amount (e.g., "BTC").
            to_currency: The target currency (e.g., "USD").

        Returns:
            The converted amount as a Decimal, or None if conversion
            is not possible (e.g., unknown currency, no exchange rate).
        """
        ...

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        timeframe: str,
        since: datetime,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Fetch historical OHLCV data for a trading pair.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").
            timeframe: The timeframe for the candles (e.g., "1m", "1h", "1d").
            since: Python datetime object indicating the start time for
                   fetching data (UTC). Exclusive start of time slice.
            limit: The maximum number of candles to return.

        Returns:
            A list of dictionaries, where each dictionary represents an OHLCV candle:
            {'timestamp': datetime_obj, 'open': Decimal, 'high': Decimal,
             'low': Decimal, 'close': Decimal, 'volume': Decimal},
            or None if data is unavailable or an error occurs. Timestamps are UTC.
        """
        ...


@runtime_checkable
class MarketDataProviderProtocol(Protocol):
    """Protocol for market data providers that fetch and process market data.

    This protocol defines the interface for components that provide market data
    from various sources (exchanges, APIs, etc.) and handle data processing.
    """

    async def start(self) -> None:
        """Initialize the market data provider.

        Setup connections, authentication, and start data collection processes.
        """
        ...

    async def stop(self) -> None:
        """Shutdown the market data provider.

        Clean up resources, close connections, and stop data collection.
        """
        ...

    async def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading symbols.

        Returns:
            List of supported trading pair symbols.
        """
        ...

    async def subscribe_to_ticker(self, symbol: str) -> None:
        """Subscribe to real-time ticker data for a symbol.

        Args:
            symbol: Trading pair symbol to subscribe to.
        """
        ...

    async def unsubscribe_from_ticker(self, symbol: str) -> None:
        """Unsubscribe from real-time ticker data for a symbol.

        Args:
            symbol: Trading pair symbol to unsubscribe from.
        """
        ...

    async def get_ticker_data(self, symbol: str) -> dict[str, Any] | None:
        """Get current ticker data for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dictionary containing ticker data (price, volume, etc.) or None.
        """
        ...

    async def get_order_book(self, symbol: str, depth: int = 10) -> dict[str, Any] | None:
        """Get order book data for a symbol.

        Args:
            symbol: Trading pair symbol.
            depth: Number of price levels to include.

        Returns:
            Dictionary containing order book data (bids, asks) or None.
        """
        ...

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]] | None:
        """Get recent trade data for a symbol.

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of trades to return.

        Returns:
            List of trade dictionaries or None.
        """
        ...


@runtime_checkable
class MarketDataStreamProtocol(Protocol):
    """Protocol for streaming market data services.

    This protocol defines the interface for components that provide real-time
    streaming market data with event-based updates.
    """

    async def start_stream(self, symbols: list[str]) -> None:
        """Start streaming market data for specified symbols.

        Args:
            symbols: List of trading pair symbols to stream.
        """
        ...

    async def stop_stream(self) -> None:
        """Stop the market data stream."""
        ...

    async def add_symbol(self, symbol: str) -> None:
        """Add a symbol to the active stream.

        Args:
            symbol: Trading pair symbol to add.
        """
        ...

    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from the active stream.

        Args:
            symbol: Trading pair symbol to remove.
        """
        ...

    async def get_stream_events(self) -> AsyncIterator[MarketDataEvent]:
        """Get async iterator of market data events.

        Yields:
            MarketDataEvent instances as they are received.
        """
        ...

    def is_streaming(self) -> bool:
        """Check if the stream is currently active.

        Returns:
            True if streaming is active, False otherwise.
        """
        ...

    def get_active_symbols(self) -> list[str]:
        """Get list of currently streaming symbols.

        Returns:
            List of active trading pair symbols.
        """
        ...


@runtime_checkable
class MarketDataCacheProtocol(Protocol):
    """Protocol for market data caching services.

    This protocol defines the interface for components that cache market data
    to improve performance and reduce API calls.
    """

    async def get_cached_price(self, symbol: str) -> Decimal | None:
        """Get cached price for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Cached price as Decimal or None if not cached/expired.
        """
        ...

    async def cache_price(self, symbol: str, price: Decimal, timestamp: datetime) -> None:
        """Cache a price for a symbol.

        Args:
            symbol: Trading pair symbol.
            price: Price to cache.
            timestamp: Timestamp of the price data.
        """
        ...

    async def get_cached_ticker(self, symbol: str) -> dict[str, Any] | None:
        """Get cached ticker data for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Cached ticker data dictionary or None.
        """
        ...

    async def cache_ticker(self, symbol: str, ticker_data: dict[str, Any]) -> None:
        """Cache ticker data for a symbol.

        Args:
            symbol: Trading pair symbol.
            ticker_data: Ticker data to cache.
        """
        ...

    async def get_cached_order_book(self, symbol: str) -> dict[str, Any] | None:
        """Get cached order book data for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Cached order book data dictionary or None.
        """
        ...

    async def cache_order_book(self, symbol: str, order_book_data: dict[str, Any]) -> None:
        """Cache order book data for a symbol.

        Args:
            symbol: Trading pair symbol.
            order_book_data: Order book data to cache.
        """
        ...

    async def invalidate_cache(self, symbol: str) -> None:
        """Invalidate cached data for a symbol.

        Args:
            symbol: Trading pair symbol.
        """
        ...

    async def clear_cache(self) -> None:
        """Clear all cached data."""
        ...

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics.
        """
        ...


@runtime_checkable
class MarketDataAggregatorProtocol(Protocol):
    """Protocol for market data aggregation services.

    This protocol defines the interface for components that aggregate market data
    from multiple sources and provide unified access.
    """

    async def add_provider(self, provider: MarketDataProviderProtocol, priority: int = 0) -> None:
        """Add a market data provider to the aggregator.

        Args:
            provider: Market data provider instance.
            priority: Provider priority (higher = more preferred).
        """
        ...

    async def remove_provider(self, provider: MarketDataProviderProtocol) -> None:
        """Remove a market data provider from the aggregator.

        Args:
            provider: Market data provider instance to remove.
        """
        ...

    async def get_best_price(self, symbol: str) -> Decimal | None:
        """Get the best available price from all providers.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Best available price as Decimal or None.
        """
        ...

    async def get_aggregated_ticker(self, symbol: str) -> dict[str, Any] | None:
        """Get aggregated ticker data from all providers.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Aggregated ticker data dictionary or None.
        """
        ...

    async def get_best_order_book(self, symbol: str) -> dict[str, Any] | None:
        """Get the best order book from all providers.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Best order book data dictionary or None.
        """
        ...

    def get_active_providers(self) -> list[MarketDataProviderProtocol]:
        """Get list of active market data providers.

        Returns:
            List of active provider instances.
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all providers.

        Returns:
            Dictionary containing health status for each provider.
        """
        ...


@runtime_checkable
class MarketDataValidatorProtocol(Protocol):
    """Protocol for market data validation services.

    This protocol defines the interface for components that validate market data
    quality and detect anomalies.
    """

    async def validate_price(self, symbol: str, price: Decimal, timestamp: datetime) -> bool:
        """Validate a price data point.

        Args:
            symbol: Trading pair symbol.
            price: Price to validate.
            timestamp: Timestamp of the price data.

        Returns:
            True if price is valid, False otherwise.
        """
        ...

    async def validate_ticker(self, symbol: str, ticker_data: dict[str, Any]) -> bool:
        """Validate ticker data.

        Args:
            symbol: Trading pair symbol.
            ticker_data: Ticker data to validate.

        Returns:
            True if ticker data is valid, False otherwise.
        """
        ...

    async def validate_order_book(self, symbol: str, order_book_data: dict[str, Any]) -> bool:
        """Validate order book data.

        Args:
            symbol: Trading pair symbol.
            order_book_data: Order book data to validate.

        Returns:
            True if order book data is valid, False otherwise.
        """
        ...

    async def detect_anomalies(self, symbol: str, data: dict[str, Any]) -> list[str]:
        """Detect anomalies in market data.

        Args:
            symbol: Trading pair symbol.
            data: Market data to analyze.

        Returns:
            List of anomaly descriptions found.
        """
        ...

    def get_validation_rules(self) -> dict[str, Any]:
        """Get current validation rules.

        Returns:
            Dictionary containing validation rules.
        """
        ...

    async def update_validation_rules(self, rules: dict[str, Any]) -> None:
        """Update validation rules.

        Args:
            rules: New validation rules to apply.
        """
        ...
