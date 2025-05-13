"""Portfolio valuation and drawdown calculation functionality."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from ..exceptions import PriceNotAvailableError


class ValuationService:
    """
    Handles portfolio valuation and drawdown calculations.

    Calculates total portfolio value, equity peaks, drawdowns,
    and portfolio exposure metrics.
    """

    def __init__(
        self, logger_service: Any, market_price_service: Any, valuation_currency: str = "USD"
    ):
        """
        Initialize the valuation service.

        Args:
            logger_service: Logger service for logging
            market_price_service: Service to get market prices
            valuation_currency: Base currency for portfolio valuation
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self.market_price_service = market_price_service
        self.valuation_currency = valuation_currency

        # Valuation metrics
        self._total_equity: Decimal = Decimal(0)
        self._peak_equity: Decimal = Decimal(0)
        self._total_drawdown_pct: Decimal = Decimal(0)

        # Time-based drawdown tracking
        self._daily_peak_equity: Decimal = Decimal(0)
        self._weekly_peak_equity: Decimal = Decimal(0)
        self._daily_drawdown_pct: Decimal = Decimal(0)
        self._weekly_drawdown_pct: Decimal = Decimal(0)
        self._last_daily_reset_time: Optional[datetime] = None
        self._last_weekly_reset_time: Optional[datetime] = None

        # Configuration for drawdown reset times
        self._daily_reset_hour_utc: int = 0
        self._weekly_reset_day: int = 0  # 0=Monday

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Cache for price conversion rates
        self._price_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        self._price_cache_ttl_seconds: int = 60  # Cache prices for 1 minute

    @property
    def total_equity(self) -> Decimal:
        """Get the current total portfolio equity value."""
        return self._total_equity

    @property
    def peak_equity(self) -> Decimal:
        """Get the peak equity value."""
        return self._peak_equity

    @property
    def total_drawdown_pct(self) -> Decimal:
        """Get the current total drawdown percentage."""
        return self._total_drawdown_pct

    @property
    def daily_drawdown_pct(self) -> Decimal:
        """Get the current daily drawdown percentage."""
        return self._daily_drawdown_pct

    @property
    def weekly_drawdown_pct(self) -> Decimal:
        """Get the current weekly drawdown percentage."""
        return self._weekly_drawdown_pct

    def configure_drawdown_resets(self, daily_reset_hour_utc: int, weekly_reset_day: int) -> None:
        """
        Configure when drawdown metrics are reset.

        Args:
            daily_reset_hour_utc: Hour (0-23) in UTC when daily drawdown resets
            weekly_reset_day: Day of week (0=Monday, 6=Sunday) when weekly drawdown resets
        """
        if not 0 <= daily_reset_hour_utc <= 23:
            raise ValueError(f"Invalid daily reset hour: {daily_reset_hour_utc}, must be 0-23")

        if not 0 <= weekly_reset_day <= 6:
            raise ValueError(f"Invalid weekly reset day: {weekly_reset_day}, must be 0-6")

        self._daily_reset_hour_utc = daily_reset_hour_utc
        self._weekly_reset_day = weekly_reset_day

        # Use a list lookup instead of long string for day names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        self.logger.info(
            f"Configured drawdown resets: daily at {daily_reset_hour_utc}:00 UTC, "
            f"weekly on {day_names[weekly_reset_day]}",
            source_module=self._source_module,
        )

    async def _try_direct_conversion(
        self, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """
        Try to directly convert between currencies using market price.

        Args:
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Conversion rate if available, None otherwise
        """
        try:
            # Try direct market price
            pair = f"{from_currency}/{to_currency}"
            rate = await self.market_price_service.get_latest_price(pair)
            if rate is not None:
                # Ensure the rate is a Decimal to satisfy the type checker
                decimal_rate = Decimal(str(rate))
                now = datetime.utcnow()
                cache_key = f"{from_currency}/{to_currency}"
                self._price_cache[cache_key] = (decimal_rate, now)
                return decimal_rate
        except Exception as e:
            self.logger.debug(
                f"Direct price not available for {pair}: {e}", source_module=self._source_module
            )
        return None

    async def _try_inverse_conversion(
        self, from_currency: str, to_currency: str
    ) -> Optional[Decimal]:
        """
        Try to convert between currencies using inverse market price.

        Args:
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Conversion rate if available, None otherwise
        """
        try:
            inverse_pair = f"{to_currency}/{from_currency}"
            inverse_rate = await self.market_price_service.get_latest_price(inverse_pair)
            if inverse_rate is not None and inverse_rate > Decimal("0"):
                # Ensure we're working with Decimal values
                decimal_inverse_rate = Decimal(str(inverse_rate))
                rate = Decimal("1") / decimal_inverse_rate
                now = datetime.utcnow()
                cache_key = f"{from_currency}/{to_currency}"
                self._price_cache[cache_key] = (rate, now)
                return rate
        except Exception as e:
            self.logger.debug(
                f"Inverse price not available for {inverse_pair}: {e}",
                source_module=self._source_module,
            )
        return None

    async def _try_usd_conversion(self, from_currency: str, to_currency: str) -> Optional[Decimal]:
        """
        Try to convert between currencies using USD as an intermediate.

        Args:
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Conversion rate if available, None otherwise
        """
        if from_currency == "USD" or to_currency == "USD":
            return None

        try:
            from_usd_rate = await self.get_currency_conversion_rate(from_currency, "USD")
            usd_to_rate = await self.get_currency_conversion_rate("USD", to_currency)
            if from_usd_rate is not None and usd_to_rate is not None:
                rate = from_usd_rate * usd_to_rate
                now = datetime.utcnow()
                cache_key = f"{from_currency}/{to_currency}"
                self._price_cache[cache_key] = (rate, now)
                return rate
        except PriceNotAvailableError:
            self.logger.debug(
                f"USD conversion path failed for {from_currency}->{to_currency}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.debug(
                f"Error during USD conversion for {from_currency}->{to_currency}: {e}",
                source_module=self._source_module,
            )
        return None

    async def get_currency_conversion_rate(self, from_currency: str, to_currency: str) -> Decimal:
        """
        Get the conversion rate between two currencies.

        Args:
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Conversion rate as a Decimal

        Raises:
            PriceNotAvailableError: If conversion rate cannot be determined
        """
        # Check for identity conversion
        if from_currency == to_currency:
            return Decimal("1.0")

        # Check cache first
        cache_key = f"{from_currency}/{to_currency}"
        now = datetime.utcnow()

        if cache_key in self._price_cache:
            rate, timestamp = self._price_cache[cache_key]
            age_seconds = (now - timestamp).total_seconds()

            if age_seconds < self._price_cache_ttl_seconds:
                # Ensure we always return a Decimal type from cache
                return Decimal(str(rate)) if rate is not None else Decimal("0")

        # Try different conversion methods in sequence
        direct_rate = await self._try_direct_conversion(from_currency, to_currency)
        if direct_rate is not None:
            # Ensure we return a Decimal type
            return Decimal(str(direct_rate))

        inverse_rate = await self._try_inverse_conversion(from_currency, to_currency)
        if inverse_rate is not None:
            # Ensure we return a Decimal type
            return Decimal(str(inverse_rate))

        usd_rate = await self._try_usd_conversion(from_currency, to_currency)
        if usd_rate is not None:
            # Ensure we return a Decimal type
            return Decimal(str(usd_rate))

        raise PriceNotAvailableError(
            f"Cannot determine conversion rate: {from_currency} to {to_currency}"
        )

    async def calculate_position_value(
        self, positions: Dict[str, Any], valuation_currency: Optional[str] = None
    ) -> Tuple[Decimal, bool, Dict[str, Decimal]]:
        """
        Calculate total value of all positions in the valuation currency.

        Args:
            positions: Dictionary of position information
            valuation_currency: Currency for valuation (defaults to instance default)

        Returns:
            Tuple of (total_value, has_missing_prices, price_dict)
        """
        if valuation_currency is None:
            valuation_currency = self.valuation_currency

        total_value = Decimal("0")
        has_missing_prices = False
        calculated_prices = {}

        for pair, position in positions.items():
            if position.quantity == Decimal("0"):
                continue  # Skip zero positions

            # Extract information from position
            quote_asset = position.quote_asset
            quantity = position.quantity

            try:
                # Get the latest price for this pair
                latest_price = await self.market_price_service.get_latest_price(pair)

                if latest_price is None:
                    has_missing_prices = True
                    self.logger.warning(
                        f"Could not get price for {pair}", source_module=self._source_module
                    )
                    continue

                # Store price for later use
                calculated_prices[pair] = latest_price

                # Calculate position value in quote currency
                position_value_quote = quantity * latest_price

                # Convert to valuation currency if needed
                if quote_asset != valuation_currency:
                    try:
                        conversion_rate = await self.get_currency_conversion_rate(
                            quote_asset, valuation_currency
                        )
                        position_value_val_curr = position_value_quote * conversion_rate
                    except PriceNotAvailableError:
                        has_missing_prices = True
                        self.logger.warning(
                            f"Could not convert {quote_asset} to {valuation_currency}",
                            source_module=self._source_module,
                        )
                        continue
                else:
                    position_value_val_curr = position_value_quote

                # Add to total (long positions add, short positions subtract)
                total_value += position_value_val_curr

            except Exception as e:
                has_missing_prices = True
                self.logger.error(
                    f"Error calculating position value for {pair}: {e}",
                    source_module=self._source_module,
                    exc_info=True,
                )

        return total_value, has_missing_prices, calculated_prices

    async def calculate_cash_value(
        self, funds: Dict[str, Decimal], valuation_currency: Optional[str] = None
    ) -> Tuple[Decimal, bool]:
        """
        Calculate the value of cash balances in valuation currency.

        Args:
            funds: Dictionary of currency balances
            valuation_currency: Currency for valuation (defaults to instance default)

        Returns:
            Tuple of (total_value, has_missing_prices)
        """
        if valuation_currency is None:
            valuation_currency = self.valuation_currency

        total_value = Decimal("0")
        has_missing_prices = False

        for currency, amount in funds.items():
            if amount == Decimal("0"):
                continue  # Skip zero balances

            if currency == valuation_currency:
                total_value += amount
            else:
                try:
                    conversion_rate = await self.get_currency_conversion_rate(
                        currency, valuation_currency
                    )
                    currency_value = amount * conversion_rate
                    total_value += currency_value
                except PriceNotAvailableError:
                    has_missing_prices = True
                    self.logger.warning(
                        f"Could not convert {currency} to {valuation_currency}",
                        source_module=self._source_module,
                    )

        return total_value, has_missing_prices

    async def update_portfolio_value(
        self, funds: Dict[str, Decimal], positions: Dict[str, Any]
    ) -> Tuple[Decimal, Dict[str, Decimal], Decimal]:
        """
        Update total portfolio value, drawdowns, and exposure metrics.

        Args:
            funds: Dictionary of currency balances
            positions: Dictionary of positions

        Returns:
            Tuple of (total_equity, latest_prices, exposure_percentage)
        """
        # Calculate cash value
        cash_value, missing_prices_cash = await self.calculate_cash_value(funds)

        # Calculate positions value
        position_value, missing_prices_pos, latest_prices = await self.calculate_position_value(
            positions
        )

        # Total value and missing prices flag
        total_value = cash_value + position_value
        has_missing_prices = missing_prices_cash or missing_prices_pos

        if has_missing_prices:
            self.logger.warning(
                "Portfolio value calculation has missing prices - result may be inaccurate",
                source_module=self._source_module,
            )

        # Calculate exposure percentage
        exposure_pct = await self._calculate_exposure_percentage(
            positions, latest_prices, total_value
        )

        # Update metrics with lock for thread safety
        async with self._lock:
            self._total_equity = total_value

            # Check for peak equity updates and calculate drawdowns
            await self._update_drawdown_metrics()

        return total_value, latest_prices, exposure_pct

    async def _calculate_exposure_percentage(
        self, positions: Dict[str, Any], latest_prices: Dict[str, Decimal], total_value: Decimal
    ) -> Decimal:
        """Calculate the exposure percentage of positions relative to total portfolio value."""
        exposure_pct = Decimal("0")

        if total_value > Decimal("0"):
            # Use absolute position value for exposure calculation
            abs_position_exposure = Decimal(0)
            for pair, pos_info in positions.items():
                if pos_info.quantity != Decimal("0") and pair in latest_prices:
                    pair_val_in_quote = abs(pos_info.quantity) * latest_prices[pair]
                    if pos_info.quote_asset == self.valuation_currency:
                        abs_position_exposure += pair_val_in_quote
                    else:
                        try:
                            rate = await self.get_currency_conversion_rate(
                                pos_info.quote_asset, self.valuation_currency
                            )
                            # Ensure rate is a Decimal
                            decimal_rate = Decimal(str(rate))
                            abs_position_exposure += pair_val_in_quote * decimal_rate
                        except PriceNotAvailableError:
                            pass  # Already logged

            exposure_pct = (abs_position_exposure / total_value) * Decimal("100")

        return exposure_pct

    async def _update_drawdown_metrics(self) -> None:
        """Update peak equity values and calculate drawdown metrics."""
        now = datetime.utcnow()

        # Check if we need to reset daily/weekly peaks
        await self._check_reset_periods(now)

        # Update all-time peak if we're at a new high
        if self._total_equity > self._peak_equity:
            self._peak_equity = self._total_equity
            self._total_drawdown_pct = Decimal("0")
        else:
            # Calculate drawdown if we have a valid peak
            if self._peak_equity > Decimal("0"):
                self._total_drawdown_pct = (
                    (self._peak_equity - self._total_equity) / self._peak_equity
                ) * Decimal("100")
            else:
                self._total_drawdown_pct = Decimal("0")

        # Update daily peak
        if self._total_equity > self._daily_peak_equity:
            self._daily_peak_equity = self._total_equity
            self._daily_drawdown_pct = Decimal("0")
        else:
            # Calculate daily drawdown if we have a valid peak
            if self._daily_peak_equity > Decimal("0"):
                self._daily_drawdown_pct = (
                    (self._daily_peak_equity - self._total_equity) / self._daily_peak_equity
                ) * Decimal("100")
            else:
                self._daily_drawdown_pct = Decimal("0")

        # Update weekly peak
        if self._total_equity > self._weekly_peak_equity:
            self._weekly_peak_equity = self._total_equity
            self._weekly_drawdown_pct = Decimal("0")
        else:
            # Calculate weekly drawdown if we have a valid peak
            if self._weekly_peak_equity > Decimal("0"):
                self._weekly_drawdown_pct = (
                    (self._weekly_peak_equity - self._total_equity) / self._weekly_peak_equity
                ) * Decimal("100")
            else:
                self._weekly_drawdown_pct = Decimal("0")

    async def _check_reset_periods(self, current_time: datetime) -> None:
        """
        Check if daily/weekly peak values should be reset.

        Args:
            current_time: Current UTC datetime
        """
        # Handle daily reset
        if self._last_daily_reset_time is None:
            # First run, set initial reset time
            self._last_daily_reset_time = current_time
            self._daily_peak_equity = self._total_equity
        else:
            # Check if we've passed the reset hour
            if (
                current_time.date() > self._last_daily_reset_time.date()
                and current_time.hour >= self._daily_reset_hour_utc
                and (
                    self._last_daily_reset_time.hour < self._daily_reset_hour_utc
                    or current_time.date() > self._last_daily_reset_time.date() + timedelta(days=0)
                )
                # Corrected timedelta, should be same day or next day after reset hour
            ):
                # Reset daily peak
                self._daily_peak_equity = self._total_equity
                self._daily_drawdown_pct = Decimal("0")
                self._last_daily_reset_time = current_time

                self.logger.info(
                    f"Reset daily drawdown peak to {self._daily_peak_equity}",
                    source_module=self._source_module,
                )

        # Handle weekly reset
        if self._last_weekly_reset_time is None:
            # First run, set initial reset time
            self._last_weekly_reset_time = current_time
            self._weekly_peak_equity = self._total_equity
        else:
            # Check if we've passed the weekly reset day and hour
            current_weekday = current_time.weekday()  # 0=Monday

            if (
                current_weekday == self._weekly_reset_day
                and current_time.hour >= self._daily_reset_hour_utc
                and (current_time - self._last_weekly_reset_time >= timedelta(days=6))
                # Ensure at least ~a week has passed since last weekly reset on this day
            ):
                # Reset weekly peak
                self._weekly_peak_equity = self._total_equity
                self._weekly_drawdown_pct = Decimal("0")
                self._last_weekly_reset_time = current_time

                self.logger.info(
                    f"Reset weekly drawdown peak to {self._weekly_peak_equity}",
                    source_module=self._source_module,
                )
