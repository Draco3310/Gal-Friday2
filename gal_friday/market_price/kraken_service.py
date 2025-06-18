"""Kraken exchange market price service implementation.

This module provides real-time market price data retrieval from the Kraken
cryptocurrency exchange, including price, spread information, and currency conversion.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast  # Added cast

import aiohttp

from ..config_manager import ConfigManager
from ..logger_service import LoggerService
from ..market_price_service import MarketPriceService


class KrakenMarketPriceService(MarketPriceService):
    """Kraken-specific implementation of the MarketPriceService.

    Fetches real-time price data from the Kraken cryptocurrency exchange.
    """

    HTTP_OK = 200

    def __init__(self, config_manager: ConfigManager, logger_service: LoggerService) -> None:
        """Initialize the Kraken market price service.

        Args:
        ----
            config_manager: Configuration manager instance
            logger_service: Logger service for logging
        """
        self.config = config_manager
        self.logger = logger_service
        self._api_url = self.config.get("kraken.api_url", "https://api.kraken.com")
        self._session: aiohttp.ClientSession | None = None
        self._source_module = self.__class__.__name__
        self._price_timestamps: dict[str, datetime] = {}
        self.logger.info(
            "KrakenMarketPriceService initialized.",
            source_module=self._source_module)

    async def start(self) -> None:
        """Start the service and create an HTTP session."""
        self._session = aiohttp.ClientSession()
        self.logger.info(
            "KrakenMarketPriceService started, session created.",
            source_module=self._source_module)

    async def stop(self) -> None:
        """Stop the service and close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info(
                "KrakenMarketPriceService stopped, session closed.",
                source_module=self._source_module)

    async def get_latest_price(self, trading_pair: str) -> Decimal | None:
        """Get latest price using Kraken public Ticker endpoint.

        Args:
        ----
            trading_pair: The trading pair to get the price for (e.g., "XBT/USD")

        Returns:
        -------
            The latest price as a Decimal or None if it couldn't be retrieved.
        """
        price_to_return: Decimal | None = None

        if not self._session:
            self.logger.warning(
                "Session not initialized. Call start() before using the service.",
                source_module=self._source_module)
            return price_to_return

        try:
            kraken_pair = self._map_internal_to_kraken_pair(trading_pair)
            if not kraken_pair:
                self.logger.error(
                    "Could not map trading pair %s to Kraken format",
                    trading_pair,
                    source_module=self._source_module)
            else:
                url = f"{self._api_url}/0/public/Ticker?pair={kraken_pair}"
                self.logger.debug(
                    "Requesting ticker for %s",
                    kraken_pair,
                    source_module=self._source_module)

                async with self._session.get(url) as response:
                    if response.status != self.HTTP_OK:
                        self.logger.error(
                            "Error fetching price for %s: HTTP %s",
                            trading_pair,
                            response.status,
                            source_module=self._source_module)
                    else:
                        data = await response.json()
                        if data.get("error") and len(data["error"]) > 0:
                            self.logger.error(
                                "Kraken API error: %s",
                                data["error"],
                                source_module=self._source_module)
                        elif "result" in data and kraken_pair in data["result"]:
                            # c[0] is the last trade closed price
                            price_str = data["result"][kraken_pair]["c"][0]
                            self._price_timestamps[kraken_pair] = datetime.utcnow()
                            price_to_return = Decimal(price_str)
                        else:
                            self.logger.error(
                                "Unexpected response format for %s",
                                trading_pair,
                                source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error fetching price for %s",
                trading_pair,
                source_module=self._source_module)

        return price_to_return

    async def get_bid_ask_spread(self, trading_pair: str) -> tuple[Decimal, Decimal] | None:
        """Get best bid/ask using Kraken public Ticker endpoint.

        Args:
        ----
            trading_pair: The trading pair to get the spread for (e.g., "XBT/USD")

        Returns:
        -------
            A tuple[Any, ...] of (bid_price, ask_price) as Decimals, or None if not retrieved.
        """
        spread_to_return: tuple[Decimal, Decimal] | None = None

        if not self._session:
            self.logger.warning(
                "Session not initialized. Call start() before using the service.",
                source_module=self._source_module)
            return spread_to_return

        try:
            kraken_pair = self._map_internal_to_kraken_pair(trading_pair)
            if not kraken_pair:
                self.logger.error(
                    "Could not map trading pair %s to Kraken format",
                    trading_pair,
                    source_module=self._source_module)
            else:
                url = f"{self._api_url}/0/public/Ticker?pair={kraken_pair}"
                self.logger.debug(
                    "Requesting ticker for %s",
                    kraken_pair,
                    source_module=self._source_module)

                async with self._session.get(url) as response:
                    if response.status != self.HTTP_OK:
                        self.logger.error(
                            "Error fetching spread for %s: HTTP %s",
                            trading_pair,
                            response.status,
                            source_module=self._source_module)
                    else:
                        data = await response.json()
                        if data.get("error") and len(data["error"]) > 0:
                            self.logger.error(
                                "Kraken API error: %s",
                                data["error"],
                                source_module=self._source_module)
                        elif "result" in data and kraken_pair in data["result"]:
                            # b[0] is the best bid price, a[0] is the best ask price
                            bid_str = data["result"][kraken_pair]["b"][0]
                            ask_str = data["result"][kraken_pair]["a"][0]
                            self._price_timestamps[kraken_pair] = datetime.utcnow()
                            spread_to_return = (Decimal(bid_str), Decimal(ask_str))
                        else:
                            self.logger.error(
                                "Unexpected response format for %s",
                                trading_pair,
                                source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error fetching spread for %s",
                trading_pair,
                source_module=self._source_module)

        return spread_to_return

    def _map_internal_to_kraken_pair(self, internal_pair: str) -> str | None:
        """Map internal trading pair format to Kraken's format.

        Args:
        ----
            internal_pair: Trading pair in internal format (e.g., "BTC/USD")

        Returns:
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

        return base + quote

    async def get_price_timestamp(self, trading_pair: str) -> datetime | None:
        """Get the timestamp (UTC) associated with the latest price data."""
        kraken_pair = self._map_internal_to_kraken_pair(trading_pair)
        if not kraken_pair:
            self.logger.warning(
                "Could not map trading pair %s for timestamp lookup.",
                trading_pair,
                source_module=self._source_module)
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
                "Price data for %s (timestamp: %s) is older than %ss.",
                trading_pair,
                timestamp,
                max_age_seconds,
                source_module=self._source_module)
        return is_fresh

    async def _get_safe_price(self, pair: str) -> Decimal | None:
        """Get price, return None if price is 0 or None to avoid division by zero."""
        price_decimal = await self.get_latest_price(pair)
        if price_decimal is not None and price_decimal > 0:
            return price_decimal
        self.logger.debug(
            "Safe price for %s is None or zero.",
            pair,
            source_module=self._source_module)
        return None

    async def _try_conversion_step(
        self,
        amount: Decimal,
        from_currency: str,
        to_currency: str) -> Decimal | None:
        """Attempt a single conversion from_currency to to_currency.

        Tries both direct and reverse pairs.
        """
        # 1. Try direct conversion (FROM/TO)
        direct_pair = f"{from_currency}/{to_currency}"
        price_direct = await self._get_safe_price(direct_pair)
        if price_direct:
            self.logger.debug(
                "Converting %s %s to %s via direct pair %s with price %s",
                amount,
                from_currency,
                to_currency,
                direct_pair,
                price_direct,
                source_module=self._source_module)
            return amount * price_direct

        # 2. Try reverse conversion (TO/FROM)
        reverse_pair = f"{to_currency}/{from_currency}"
        price_reverse = await self._get_safe_price(reverse_pair)
        if price_reverse:
            self.logger.debug(
                "Converting %s %s to %s via reverse pair %s with price %s",
                amount,
                from_currency,
                to_currency,
                reverse_pair,
                price_reverse,
                source_module=self._source_module)
            return amount / price_reverse

        return None

    async def convert_amount(
        self,
        from_amount: Decimal,
        from_currency: str,
        to_currency: str) -> Decimal | None:
        """Convert an amount from one currency to another using Kraken.

        Args:
        ----
            from_amount: The amount to convert.
            from_currency: The currency of the from_amount (e.g., "BTC").
            to_currency: The target currency (e.g., "USD").

        Returns:
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
                "Direct/reverse conversion failed for intermediaries %s to %s. Unexpected.",
                from_currency,
                to_currency,
                source_module=self._source_module)

        for intermediary in intermediaries:
            # Skip if the intermediary is the source or target currency itself,
            # as that would have been covered by the initial direct/reverse attempt.
            if intermediary in (from_currency, to_currency):
                continue

            self.logger.debug(
                "Attempting conversion via intermediary: %s",
                intermediary,
                source_module=self._source_module)

            # First leg: from_currency -> intermediary
            amount_in_intermediary = await self._try_conversion_step(
                from_amount,
                from_currency,
                intermediary)

            if amount_in_intermediary is not None:
                self.logger.debug(
                    "Converted %s %s to %s %s",
                    from_amount,
                    from_currency,
                    amount_in_intermediary,
                    intermediary,
                    source_module=self._source_module)
                # Second leg: intermediary -> to_currency
                final_amount = await self._try_conversion_step(
                    amount_in_intermediary,
                    intermediary,
                    to_currency)
                if final_amount is not None:
                    self.logger.debug(
                        "Successfully converted %s %s to %s %s via %s",
                        from_amount,
                        from_currency,
                        final_amount,
                        to_currency,
                        intermediary,
                        source_module=self._source_module)
                    return final_amount
                self.logger.debug(
                    "Failed second leg of conversion: %s to %s",
                    intermediary,
                    to_currency,
                    source_module=self._source_module)
            else:
                self.logger.debug(
                    "Failed first leg of conversion: %s to %s",
                    from_currency,
                    intermediary,
                    source_module=self._source_module)

        self.logger.warning(
            "Could not convert %s %s to %s. No direct, reverse, or intermediary path found.",
            from_amount,
            from_currency,
            to_currency,
            source_module=self._source_module)
        return None

    async def _process_ohlc_response(
        self,
        data: dict[str, Any],
        kraken_pair: str,
        trading_pair: str,
        timeframe: str,
        limit: int | None = None) -> list[dict[str, Any]] | None:
        """Process the OHLC response from Kraken API."""
        if "result" not in data or kraken_pair not in data["result"]:
            self.logger.warning(
                "Unexpected response format or pair not found in result for "
                f"OHLC {trading_pair} ({kraken_pair}). Data: {str(data)[:500]}",
                source_module=self._source_module)
            return None

        ohlcv_data = []
        raw_candles = data["result"][kraken_pair]
        for candle in raw_candles:
            # Kraken candle format:
            # [<time>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <count>]
            # We need: timestamp, open, high, low, close, volume
            ohlcv_data.append(
                {
                    "timestamp": datetime.fromtimestamp(int(candle[0]), tz=UTC),
                    "open": Decimal(str(candle[1])),
                    "high": Decimal(str(candle[2])),
                    "low": Decimal(str(candle[3])),
                    "close": Decimal(str(candle[4])),
                    "volume": Decimal(str(candle[6])),  # volume is at index 6
                })

        # Apply limit if specified
        if limit is not None and len(ohlcv_data) > limit:
            ohlcv_data = ohlcv_data[:limit]

        self.logger.info(
            f"Fetched {len(ohlcv_data)} OHLCV candles for {trading_pair} "
            f"({kraken_pair}) timeframe {timeframe}",
            source_module=self._source_module)
        return ohlcv_data

    def _get_timeframe_map(self) -> dict[str, int]:
        """Return the mapping of timeframes to Kraken interval values."""
        return {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,  # 1 day
            "7d": 10080,  # 1 week
            "15d": 21600,  # 15 days (Kraken actually has 21600 for this, not 14d)
        }

    def _get_kraken_pair(self, trading_pair: str) -> str | None:
        """Convert internal trading pair to Kraken's format for OHLC."""
        try:
            base, quote = trading_pair.split("/")
            kraken_ohlc_pair_map = {
                "BTC": "XBT",
                "DOGE": "XDG",  # Kraken uses XDG for Dogecoin
            }
            mapped_base = kraken_ohlc_pair_map.get(base.upper(), base.upper())
            return mapped_base + quote.upper()
        except ValueError:
            self.logger.error(
                f"Invalid trading pair format: {trading_pair}",
                source_module=self._source_module)
            return None

    async def _fetch_ohlc_data(
        self, url: str, trading_pair: str, kraken_pair: str) -> dict[str, Any] | None:
        """Fetch OHLC data from Kraken API."""
        if not self._session or self._session.closed:
            self.logger.warning(
                "AIOHTTP session is not available for OHLC fetch. Ensure start() was called.",
                source_module=self._source_module,
                context={"trading_pair": trading_pair})
            return None
        try:
            async with self._session.get(url) as response:
                if response.status != self.HTTP_OK:
                    self.logger.error(
                        f"Error fetching OHLC for {trading_pair} ({kraken_pair}): "
                        f"HTTP {response.status} - {await response.text()}",
                        source_module=self._source_module)
                    return None
                return cast("dict[str, Any]", await response.json())
        except aiohttp.ClientError as e:
            self.logger.exception(
                f"HTTP Client error fetching OHLC for {trading_pair} "
                f"({kraken_pair}): {e}",
                source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                f"Error fetching OHLC data: {e}",
                source_module=self._source_module)
        return None

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        timeframe: str,
        since: datetime,
        limit: int | None = None) -> list[dict[str, Any]] | None:
        """Fetch historical OHLCV data for a trading pair from Kraken."""
        if not self._session:
            self.logger.warning(
                "Session not initialized. Call start() before using the service.",
                source_module=self._source_module)
            return None

        # Get timeframe mapping and validate
        timeframe_map = self._get_timeframe_map()
        kraken_interval = timeframe_map.get(timeframe.lower())
        if not kraken_interval:
            self.logger.error(
                f"Invalid timeframe '{timeframe}' for Kraken. "
                f"Supported: {list[Any](timeframe_map.keys())}",
                source_module=self._source_module)
            return None

        # Convert trading pair to Kraken format
        kraken_pair_for_ohlc = self._get_kraken_pair(trading_pair)
        if not kraken_pair_for_ohlc:
            return None

        # Prepare request parameters
        since_timestamp = int(since.timestamp())
        url = (
            f"{self._api_url}/0/public/OHLC?"
            f"pair={kraken_pair_for_ohlc}&interval={kraken_interval}&since={since_timestamp}"
        )

        self.logger.debug(
            f"Requesting OHLC for {kraken_pair_for_ohlc} (from {trading_pair}) "
            f"with interval {kraken_interval}, since {since_timestamp}",
            source_module=self._source_module)

        # Fetch and process data
        data = await self._fetch_ohlc_data(url, trading_pair, kraken_pair_for_ohlc)
        if not data:
            return None

        if data.get("error") and len(data["error"]) > 0:
            self.logger.error(
                f"Kraken API error for OHLC {trading_pair} "
                f"({kraken_pair_for_ohlc}): {data['error']}",
                source_module=self._source_module)
            return None

        return await self._process_ohlc_response(
            data, kraken_pair_for_ohlc, trading_pair, timeframe, limit)

    async def get_volatility(
        self,
        trading_pair: str,
        lookback_hours: int = 24) -> float | None:
        """Calculate price volatility for a trading pair.

        Args:
        ----
            trading_pair: The trading pair to calculate volatility for
            lookback_hours: Number of hours to look back for volatility calculation

        Returns:
        -------
            The calculated volatility as a float (percentage), or None if calculation fails
        """
        try:
            # Calculate appropriate timeframe based on lookback hours
            timeframe = "1H" if lookback_hours <= 24 else "4H"
            
            # Get OHLC data for the lookback period
            since = datetime.now(UTC) - timedelta(hours=lookback_hours)
            candles = await self.get_historical_ohlcv(
                trading_pair=trading_pair,
                timeframe=timeframe,
                since=since
            )
            
            if not candles or len(candles) < 2:
                self.logger.warning(
                    f"Insufficient data for volatility calculation: {trading_pair}",
                    source_module=self._source_module)
                return None
            
            # Calculate returns from close prices
            import numpy as np
            closes = [float(candle["close"]) for candle in candles]
            returns = np.diff(np.log(closes))
            
            # Calculate standard deviation of returns (volatility)
            if len(returns) > 0:
                volatility = float(np.std(returns) * 100)  # Convert to percentage
                
                # Annualize the volatility based on timeframe
                periods_per_year = 8760 if timeframe == "1H" else 2190  # Hours in a year / timeframe
                annualized_volatility = volatility * np.sqrt(periods_per_year / len(returns))
                
                self.logger.debug(
                    f"Calculated volatility for {trading_pair}: {annualized_volatility:.2f}%",
                    source_module=self._source_module)
                
                return float(annualized_volatility)
            
            return None
            
        except Exception as e:
            self.logger.error(
                f"Failed to calculate volatility for {trading_pair}: {e}",
                source_module=self._source_module,
                exc_info=True)
            return None