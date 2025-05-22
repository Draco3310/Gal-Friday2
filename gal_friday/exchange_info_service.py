"""Exchange information service implementation."""

from __future__ import annotations

from typing import Any


class ExchangeInfoService:
    """Service for retrieving exchange information."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the exchange info service."""

    def get_exchange_info(self) -> dict[str, Any]:
        """Get general exchange information.

        Returns:
        -------
            Dictionary containing exchange information
        """
        return {}

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get information for a specific symbol.

        Args:
            symbol: The trading symbol to get info for

        Returns:
        -------
            Dictionary with symbol information or None if not found
        """
        return None

    def get_trading_pairs(self) -> list[str]:
        """Get list of all available trading pairs.

        Returns:
        -------
            List of trading pair symbols
        """
        return []
