"""Abstract interface for retrieving exchange information."""

import abc
from decimal import Decimal
from typing import Any


class ExchangeInfoService(abc.ABC):
    """Abstract Base Class for components providing exchange information.

    Implementations should handle fetching and providing details about
    exchange-level configurations, symbol specifications, trading limits, etc.
    """

    @abc.abstractmethod
    def get_exchange_info(self) -> dict[str, Any]:
        """Get general exchange information.

        Returns:
        -------
            Dictionary containing exchange information.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_symbol_info(self, trading_pair: str) -> dict[str, Any] | None:
        """Get information for a specific symbol.

        Args:
        ----
            trading_pair: The trading symbol to get info for (e.g., "XRP/USD").

        Returns:
        -------
            Dictionary with symbol information (e.g., lot size, tick size, limits),
            or None if the pair is not supported.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_trading_pairs(self) -> list[str]:
        """Get list[Any] of all available trading pairs on the exchange.

        Returns:
        -------
            List of trading pair symbols.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_tick_size(self, trading_pair: str) -> Decimal | None:
        """Get the minimum price movement (tick size) for a trading pair.

        Args:
        ----
            trading_pair: The trading pair symbol.

        Returns:
        -------
            The tick size as a Decimal, or None if not available.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_step_size(self, trading_pair: str) -> Decimal | None:
        """Get the minimum order quantity increment (step size or lot size) for a trading pair.

        Args:
        ----
            trading_pair: The trading pair symbol.

        Returns:
        -------
            The step size as a Decimal, or None if not available.
        """
        raise NotImplementedError
