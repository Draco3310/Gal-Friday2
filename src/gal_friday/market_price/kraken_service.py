"""Kraken exchange market price service implementation.

This module provides real-time market price data retrieval from the Kraken
cryptocurrency exchange, including price, spread information, and currency conversion.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, Tuple

import aiohttp

from ..config_manager import ConfigManager
from ..logger_service import LoggerService
from ..market_price_service import MarketPriceService


class KrakenMarketPriceService(MarketPriceService):
    """
    Kraken-specific implementation of the MarketPriceService.

    Fetches real-time price data from the Kraken cryptocurrency exchange.
    """

    def __init__(self, config_manager: ConfigManager, logger_service: LoggerService) -> None:
        """
        Initialize the Kraken market price service.

        Args
        ----
            config_manager: Configuration manager instance
            logger_service: Logger service for logging
        """
        self.config = config_manager
        self.logger = logger_service
        self._api_url = self.config.get("kraken.api_url", "https://api.kraken.com")
        self._session: Optional[aiohttp.ClientSession] = None
        self._source_module = self.__class__.__name__
        self._price_timestamps: Dict[str, datetime] = {}
        self.logger.info(
            "KrakenMarketPriceService initialized.", source_module=self._source_module
        )

    async def start(self) -> None:
        """Start the service and create an HTTP session."""
        self._session = aiohttp.ClientSession()
        self.logger.info(
            "KrakenMarketPriceService started, session created.", source_module=self._source_module
        )

    async def stop(self) -> None:
        """Stop the service and close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info(
                "KrakenMarketPriceService stopped, session closed.",
                source_module=self._source_module,
            )

    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """
        Get latest price using Kraken public Ticker endpoint.

        Args
        ----
            trading_pair: The trading pair to get the price for (e.g., "XBT/USD")

        Returns
        -------
            The latest price as a Decimal or None if it couldn't be retrieved.
        """
        if not self._session:
            self.logger.warning(
                "Session not initialized. Call start() before using the service.",
                source_module=self._source_module,
            )
            return None

        try:
            # Map internal pair to Kraken format
            kraken_pair = self._map_internal_to_kraken_pair(trading_pair)
            if not kraken_pair:
                self.logger.error(
                    f"Could not map trading pair {trading_pair} to Kraken format",
                    source_module=self._source_module,
                )
                return None

            # Construct URL and make request
            url = f"{self._api_url}/0/public/Ticker?pair={kraken_pair}"
            self.logger.debug(
                f"Requesting ticker for {kraken_pair}", source_module=self._source_module
            )

            async with self._session.get(url) as response:
                if response.status != 200:
                    self.logger.error(
                        f"Error fetching price for {trading_pair}: HTTP {response.status}",
                        source_module=self._source_module,
                    )
                    return None

                data = await response.json()

                # Check for errors in response
                if data.get("error") and len(data["error"]) > 0:
                    self.logger.error(
                        f"Kraken API error: {data['error']}", source_module=self._source_module
                    )
                    return None

                # Extract price from response
                if "result" in data and kraken_pair in data["result"]:
                    # c[0] is the last trade closed price
                    price_str = data["result"][kraken_pair]["c"][0]
                    self._price_timestamps[kraken_pair] = datetime.utcnow()
                    return Decimal(price_str)
                else:
                    self.logger.error(
                        f"Unexpected response format for {trading_pair}",
                        source_module=self._source_module,
                    )
                    return None

        except Exception as e:
            self.logger.error(
                f"Error fetching price for {trading_pair}: {str(e)}",
                source_module=self._source_module,
            )
            return None

    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
        """
        Get best bid/ask using Kraken public Ticker endpoint.

        Args
        ----
            trading_pair: The trading pair to get the spread for (e.g., "XBT/USD")

        Returns
        -------
            A tuple of (bid_price, ask_price) as Decimals, or None if not retrieved.
        """
        if not self._session:
            self.logger.warning(
                "Session not initialized. Call start() before using the service.",
                source_module=self._source_module,
            )
            return None

        try:
            # Map internal pair to Kraken format
            kraken_pair = self._map_internal_to_kraken_pair(trading_pair)
            if not kraken_pair:
                self.logger.error(
                    f"Could not map trading pair {trading_pair} to Kraken format",
                    source_module=self._source_module,
                )
                return None

            # Construct URL and make request
            url = f"{self._api_url}/0/public/Ticker?pair={kraken_pair}"
            self.logger.debug(
                f"Requesting ticker for {kraken_pair}", source_module=self._source_module
            )

            async with self._session.get(url) as response:
                if response.status != 200:
                    self.logger.error(
                        f"Error fetching spread for {trading_pair}: HTTP {response.status}",
                        source_module=self._source_module,
                    )
                    return None

                data = await response.json()

                # Check for errors in response
                if data.get("error") and len(data["error"]) > 0:
                    self.logger.error(
                        f"Kraken API error: {data['error']}", source_module=self._source_module
                    )
                    return None

                # Extract bid and ask from response
                if "result" in data and kraken_pair in data["result"]:
                    # b[0] is the best bid price, a[0] is the best ask price
                    bid_str = data["result"][kraken_pair]["b"][0]
                    ask_str = data["result"][kraken_pair]["a"][0]
                    self._price_timestamps[kraken_pair] = datetime.utcnow()
                    return (Decimal(bid_str), Decimal(ask_str))
                else:
                    self.logger.error(
                        f"Unexpected response format for {trading_pair}",
                        source_module=self._source_module,
                    )
                    return None

        except Exception as e:
            self.logger.error(
                f"Error fetching spread for {trading_pair}: {str(e)}",
                source_module=self._source_module,
            )
            return None

    def _map_internal_to_kraken_pair(self, internal_pair: str) -> Optional[str]:
        """
        Map internal trading pair format to Kraken's format.

        Args
        ----
            internal_pair: Trading pair in internal format (e.g., "BTC/USD")

        Returns
        -------
            Trading pair in Kraken format (e.g., "XXBTZUSD"), or None if mapping failed
        """
        if not internal_pair:
            return None

        # Map common symbols to Kraken's format
        # Kraken often uses X prefix for crypto and Z for fiat
        symbol_map = {
            "BTC": "XBT",  # Bitcoin is XBT in Kraken
            "DOGE": "XDG",  # Dogecoin is XDG in Kraken
        }

        # Map base currency
        base, quote = internal_pair.split("/")
        base = symbol_map.get(base, base)
        quote = symbol_map.get(quote, quote)

        # For major cryptocurrencies and fiat, Kraken often adds X/Z prefix
        if base in ["BTC", "XBT", "ETH", "LTC", "XRP"]:
            base = "X" + base
        if quote in ["USD", "EUR", "GBP", "JPY"]:
            quote = "Z" + quote

        # Construct Kraken pair
        kraken_pair = base + quote

        return kraken_pair

    async def get_price_timestamp(self, trading_pair: str) -> Optional[datetime]:
        """Get the timestamp (UTC) associated with the latest price data."""
        kraken_pair = self._map_internal_to_kraken_pair(trading_pair)
        if not kraken_pair:
            self.logger.warning(
                f"Could not map trading pair {trading_pair} for timestamp lookup.",
                source_module=self._source_module,
            )
            return None
        return self._price_timestamps.get(kraken_pair)

    async def is_price_fresh(self, trading_pair: str, max_age_seconds: float = 60.0) -> bool:
        """Check if the price data for a trading pair is recent enough."""
        timestamp = await self.get_price_timestamp(trading_pair)
        if timestamp is None:
            return False

        is_fresh = (datetime.utcnow() - timestamp) < timedelta(seconds=max_age_seconds)
        if not is_fresh:
            self.logger.debug(
                (
                    f"Price data for {trading_pair} (timestamp: {timestamp}) "
                    f"is older than {max_age_seconds}s."
                ),
                source_module=self._source_module,
            )
        return is_fresh

    async def _get_safe_price(self, pair: str) -> Optional[Decimal]:
        """Get price, return None if price is 0 or None to avoid division by zero."""
        price_decimal = await self.get_latest_price(pair)
        if price_decimal is not None and price_decimal > 0:
            return price_decimal
        self.logger.debug(
            f"Safe price for {pair} is None or zero.", source_module=self._source_module
        )
        return None

    async def _try_conversion_step(
        self, amount: Decimal, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """
        Attempt a single conversion from_currency to to_currency.

        Tries both direct and reverse pairs.
        """
        # 1. Try direct conversion (FROM/TO)
        direct_pair = f"{from_currency}/{to_currency}"
        price_direct = await self._get_safe_price(direct_pair)
        if price_direct:
            self.logger.debug(
                (
                    f"Converting {amount} {from_currency} to {to_currency} "
                    f"via direct pair {direct_pair} with price {price_direct}"
                ),
                source_module=self._source_module,
            )
            return amount * price_direct

        # 2. Try reverse conversion (TO/FROM)
        reverse_pair = f"{to_currency}/{from_currency}"
        price_reverse = await self._get_safe_price(reverse_pair)
        if price_reverse:
            self.logger.debug(
                (
                    f"Converting {amount} {from_currency} to {to_currency} "
                    f"via reverse pair {reverse_pair} with price {price_reverse}"
                ),
                source_module=self._source_module,
            )
            return amount / price_reverse

        return None

    async def convert_amount(
        self, from_amount: Decimal, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """
        Convert an amount from one currency to another using Kraken.

        Args
        ----
            from_amount: The amount to convert.
            from_currency: The currency of the from_amount (e.g., "BTC").
            to_currency: The target currency (e.g., "USD").

        Returns
        -------
            The converted amount as a Decimal, or None if conversion
            is not possible.
        """
        if from_currency == to_currency:
            return from_amount

        # Attempt direct or reverse conversion first
        converted_amount = await self._try_conversion_step(from_amount, from_currency, to_currency)
        if converted_amount is not None:
            return converted_amount

        # If direct/reverse fails, try cross-conversion via intermediaries
        intermediaries = ["USD", "USDT", "EUR"]  # Prioritize USD, then USDT, then EUR

        # Log if trying to convert between two intermediaries failed directly
        if (
            from_currency in intermediaries
            and to_currency in intermediaries
            and from_currency != to_currency
        ):
            self.logger.warning(
                (
                    f"Direct/reverse conversion failed for intermediaries {from_currency} "
                    f"to {to_currency}. This is unexpected."
                ),
                source_module=self._source_module,
            )

        for intermediary in intermediaries:
            # Skip if the intermediary is the source or target currency itself,
            # as that would have been covered by the initial direct/reverse attempt.
            if from_currency == intermediary or to_currency == intermediary:
                continue

            self.logger.debug(
                f"Attempting conversion via intermediary: {intermediary}",
                source_module=self._source_module,
            )

            # First leg: from_currency -> intermediary
            amount_in_intermediary = await self._try_conversion_step(
                from_amount, from_currency, intermediary
            )

            if amount_in_intermediary is not None:
                self.logger.debug(
                    f"Converted {from_amount} {from_currency} to "
                    f"{amount_in_intermediary} {intermediary}",
                    source_module=self._source_module,
                )
                # Second leg: intermediary -> to_currency
                final_amount = await self._try_conversion_step(
                    amount_in_intermediary, intermediary, to_currency
                )
                if final_amount is not None:
                    self.logger.debug(
                        (
                            f"Successfully converted {from_amount} {from_currency} "
                            f"to {final_amount} {to_currency} via {intermediary}"
                        ),
                        source_module=self._source_module,
                    )
                    return final_amount
                else:
                    self.logger.debug(
                        f"Failed second leg of conversion: {intermediary} to {to_currency}",
                        source_module=self._source_module,
                    )
            else:
                self.logger.debug(
                    f"Failed first leg of conversion: {from_currency} to {intermediary}",
                    source_module=self._source_module,
                )

        self.logger.warning(
            (
                f"Could not convert {from_amount} {from_currency} to {to_currency}. "
                f"No direct, reverse, or intermediary path found."
            ),
            source_module=self._source_module,
        )
        return None
