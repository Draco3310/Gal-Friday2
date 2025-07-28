"""Data provider protocols for historical market data access.

This module contains protocol definitions for various data providers used
throughout the Gal-Friday trading system for accessing historical market data.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gal_friday.simulated_market_price_service import DataRequest, HistoricalDataPoint


@runtime_checkable
class HistoricalDataProviderProtocol(Protocol):
    """Protocol for historical data providers."""

    async def fetch_data(self, request: "DataRequest") -> list["HistoricalDataPoint"]:
        """Fetch historical data for the given request.

        Args:
            request: Data request containing symbol, time range, and other parameters

        Returns:
            List of historical data points
        """
        ...

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by this provider.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol is supported, False otherwise
        """
        ...


@runtime_checkable
class CacheLayerProtocol(Protocol):
    """Protocol for cache layers."""

    async def get(self, key: str) -> list["HistoricalDataPoint"] | None:
        """Get data from cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached data points if found, None otherwise
        """
        ...

    async def set(self, key: str, data: list["HistoricalDataPoint"], ttl: int = 3600) -> None:
        """Store data in cache with TTL.

        Args:
            key: Cache key to store under
            data: Data points to cache
            ttl: Time to live in seconds (default: 3600)
        """
        ...


@runtime_checkable
class DataProviderRegistryProtocol(Protocol):
    """Protocol for data provider registry implementations."""

    async def fetch_historical_data(
        self,
        trading_pair: str,
        start_time: Any,
        end_time: Any,
        interval: str,
    ) -> Any:
        """Fetch historical data for a trading pair.

        Args:
            trading_pair: Trading pair symbol
            start_time: Start time for data range
            end_time: End time for data range
            interval: Data interval (e.g., '1m', '1h', '1d')

        Returns:
            Historical data in provider-specific format
        """
        ...

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names.

        Returns:
            List of available provider identifiers
        """
        ...


# Enhanced data provider protocol with additional methods
@runtime_checkable
class EnhancedDataProviderProtocol(HistoricalDataProviderProtocol, Protocol):
    """Enhanced protocol for data providers with additional capabilities."""

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a trading symbol.

        Args:
            symbol: Trading symbol to get info for

        Returns:
            Dictionary containing symbol information
        """
        ...

    async def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about provider health.

        Returns:
            Dictionary containing diagnostic metrics and status
        """
        ...

    async def initialize(self) -> None:
        """Initialize the data provider."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources used by the data provider."""
        ...


# Cache-specific protocols
@runtime_checkable
class CacheMetricsProtocol(Protocol):
    """Protocol for cache metrics and monitoring."""

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache metrics
        """
        ...

    def clear_cache(self, pattern: str | None = None) -> None:
        """Clear cache entries.

        Args:
            pattern: Optional pattern to match keys (None clears all)
        """
        ...


@runtime_checkable
class AdvancedCacheProtocol(CacheLayerProtocol, CacheMetricsProtocol, Protocol):
    """Advanced cache protocol with metrics and management capabilities."""

    async def get_or_set(self, key: str, factory: Any, ttl: int = 3600) -> list["HistoricalDataPoint"]:
        """Get data from cache or set it using factory function.

        Args:
            key: Cache key
            factory: Function to generate data if not in cache
            ttl: Time to live in seconds

        Returns:
            Data points from cache or factory
        """
        ...

    async def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        ...
