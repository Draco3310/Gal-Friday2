"""Portfolio valuation and drawdown calculation functionality."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from ..exceptions import PriceNotAvailableError
from ..interfaces import MarketPriceService
from ..logger_service import LoggerService

# Constants for magic numbers
MAX_HOURS_IN_DAY = 23
MAX_DAYS_IN_WEEK = 6  # 0=Monday, 6=Sunday
PAIR_SPLIT_EXPECTED_PARTS = 2


class PositionLike(Protocol):
    """A protocol for objects that represent a financial position."""

    quantity: Decimal
    base_asset: str
    quote_asset: str | None  # Made optional as it's not always used by all consumers


PositionInput = PositionLike | dict[str, Any]


class ValuationService:
    """Handles portfolio valuation and drawdown calculations.

    Calculates total portfolio value, equity peaks, drawdowns,
    and portfolio exposure metrics.
    """

    def __init__(
        self,
        logger_service: LoggerService,
        market_price_service: MarketPriceService,
        valuation_currency: str = "USD",
    ) -> None:
        """Initialize the valuation service.

        Args:
        ----
            logger_service: Service for logging
            market_price_service: Service to fetch market prices
            valuation_currency: The currency for overall portfolio valuation (default: "USD")
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self.market_price_service = market_price_service
        self.valuation_currency = valuation_currency.upper()

        # Valuation metrics
        self._total_equity: Decimal = Decimal(0)
        self._peak_equity: Decimal = Decimal(0)
        self._total_drawdown_pct: Decimal = Decimal(0)

        # Time-based drawdown tracking
        self._daily_peak_equity: Decimal = Decimal(0)
        self._weekly_peak_equity: Decimal = Decimal(0)
        self._daily_drawdown_pct: Decimal = Decimal(0)
        self._weekly_drawdown_pct: Decimal = Decimal(0)
        self._last_daily_reset_time: datetime | None = None
        self._last_weekly_reset_time: datetime | None = None

        # Configuration for drawdown reset times
        self._daily_reset_hour_utc: int = 0
        self._weekly_reset_day: int = 0  # 0=Monday

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Cache for price conversion rates
        self._price_cache: dict[str, tuple[Decimal, datetime]] = {}
        self._price_cache_ttl_seconds: int = 60  # Cache prices for 1 minute

        self.logger.info(
            "ValuationService initialized. Valuation currency: %s",
            self.valuation_currency,
            source_module=self._source_module,
        )

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
        """Configure when drawdown metrics are reset.

        Args:
        ----
            daily_reset_hour_utc: Hour (0-23) in UTC when daily drawdown resets
            weekly_reset_day: Day of week (0=Monday, 6=Sunday) when weekly drawdown resets
        """
        if not 0 <= daily_reset_hour_utc <= MAX_HOURS_IN_DAY:
            raise ValueError

        if not 0 <= weekly_reset_day <= MAX_DAYS_IN_WEEK:
            raise ValueError

        self._daily_reset_hour_utc = daily_reset_hour_utc
        self._weekly_reset_day = weekly_reset_day

        # Use a list lookup instead of long string for day names
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        self.logger.info(
            "Configured drawdown resets: daily at %s:00 UTC, weekly on %s",
            daily_reset_hour_utc,
            day_names[weekly_reset_day],
            source_module=self._source_module,
        )

    async def _try_direct_conversion(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Decimal | None:
        """Try to directly convert between currencies using market price.

        Args:
        ----
            from_currency: Source currency
            to_currency: Target currency

        Returns:
        -------
            Conversion rate if available, None otherwise
        """
        try:
            # Try direct market price
            pair = f"{from_currency}/{to_currency}"
            rate = await self.market_price_service.get_latest_price(pair)
            if rate is not None:
                # Ensure the rate is a Decimal to satisfy the type checker
                decimal_rate = Decimal(str(rate))
                now = datetime.now(UTC)
                cache_key = f"{from_currency}/{to_currency}"
                self._price_cache[cache_key] = (decimal_rate, now)
                return decimal_rate
        except Exception as e:
            self.logger.debug(
                "Direct price not available for %s: %s",
                pair,
                e,
                source_module=self._source_module,
            )
        return None

    async def _try_inverse_conversion(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Decimal | None:
        """Try to convert between currencies using inverse market price.

        Args:
        ----
            from_currency: Source currency
            to_currency: Target currency

        Returns:
        -------
            Conversion rate if available, None otherwise
        """
        try:
            inverse_pair = f"{to_currency}/{from_currency}"
            inverse_rate = await self.market_price_service.get_latest_price(inverse_pair)
            if inverse_rate is not None and inverse_rate > Decimal("0"):
                # Ensure we're working with Decimal values
                decimal_inverse_rate = Decimal(str(inverse_rate))
                rate = Decimal("1") / decimal_inverse_rate
                now = datetime.now(UTC)
                cache_key = f"{from_currency}/{to_currency}"
                self._price_cache[cache_key] = (rate, now)
                return rate
        except Exception as e:
            self.logger.debug(
                "Inverse price not available for %s: %s",
                inverse_pair,
                e,
                source_module=self._source_module,
            )
        return None

    async def _try_usd_conversion(self, from_currency: str, to_currency: str) -> Decimal | None:
        """Try to convert between currencies using USD as an intermediate.

        Args:
        ----
            from_currency: Source currency
            to_currency: Target currency

        Returns:
        -------
            Conversion rate if available, None otherwise
        """
        if from_currency == "USD" or to_currency == "USD":
            return None

        try:
            from_usd_rate = await self.get_currency_conversion_rate(from_currency, "USD")
            usd_to_rate = await self.get_currency_conversion_rate("USD", to_currency)
            if from_usd_rate is not None and usd_to_rate is not None:
                rate = from_usd_rate * usd_to_rate
                now = datetime.now(UTC)
                cache_key = f"{from_currency}/{to_currency}"
                self._price_cache[cache_key] = (rate, now)
                return rate
        except PriceNotAvailableError:
            self.logger.debug(
                "USD conversion path failed for %s->%s",
                from_currency,
                to_currency,
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.debug(
                "Error during USD conversion for %s->%s: %s",
                from_currency,
                to_currency,
                e,
                source_module=self._source_module,
            )
        return None

    async def get_currency_conversion_rate(
        self,
        from_currency: str,
        to_currency: str,
    ) -> Decimal | None:
        """Get the conversion rate between two currencies.

        Args:
        ----
            from_currency: Source currency
            to_currency: Target currency

        Returns:
        -------
            Conversion rate as a Decimal

        Raises:
        ------
            PriceNotAvailableError: If conversion rate cannot be determined
        """
        # Check for identity conversion
        if from_currency == to_currency:
            return Decimal("1.0")

        # Check cache first
        cache_key = f"{from_currency}/{to_currency}"
        now = datetime.now(UTC)

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

        raise PriceNotAvailableError

    def _extract_position_details(
        self,
        pair: str,
        pos_data_any: PositionInput,
    ) -> tuple[Decimal, str, str] | None:
        """Extract quantity, base_asset, and quote_asset from position data."""
        if (
            hasattr(pos_data_any, "quantity")
            and hasattr(pos_data_any, "base_asset")
            and hasattr(pos_data_any, "quote_asset")
        ):
            obj_quantity = Decimal(str(pos_data_any.quantity))  # Ensure Decimal type
            obj_base_asset = str(pos_data_any.base_asset)  # Ensure str type
            obj_quote_asset = str(pos_data_any.quote_asset)  # Ensure str type
            return obj_quantity, obj_base_asset, obj_quote_asset
        if isinstance(pos_data_any, dict):
            dict_quantity = Decimal(str(pos_data_any.get("quantity", 0)))
            if "base_asset" in pos_data_any and "quote_asset" in pos_data_any:
                dict_base_asset = str(pos_data_any["base_asset"])
                dict_quote_asset = str(pos_data_any["quote_asset"])
                return dict_quantity, dict_base_asset, dict_quote_asset
            self.logger.warning(
                "Cannot determine base/quote for position %s from dict.",
                pair,
            )
            return None
        self.logger.warning("Unsupported position data type for %s.", pair)
        return None

    async def calculate_position_value(
        self,
        positions: dict[str, PositionInput],
        valuation_currency: str | None = None,
    ) -> tuple[Decimal, bool, dict[str, Decimal]]:
        """Calculate total value of all positions in the valuation currency.

        Args:
        ----
            positions: A dictionary of PositionInfo objects or similar dicts
                       keyed by trading_pair. Each must have 'quantity',
                       'base_asset', 'quote_asset'.
            valuation_currency: Optional currency to value in. Defaults to service default.

        Returns:
        -------
            A tuple: (total_value, has_missing_prices, position_specific_values_dict).
            `position_specific_values_dict` contains individual position values.
        """
        target_valuation_currency = (valuation_currency or self.valuation_currency).upper()
        total_value = Decimal(0)
        has_missing_prices = False
        position_values: dict[str, Decimal] = {}

        for pair, pos_data_any in positions.items():
            position_details = self._extract_position_details(pair, pos_data_any)
            if position_details is None:
                has_missing_prices = True
                position_values[pair] = Decimal("NaN")  # Mark as NaN if details can't be extracted
                continue

            quantity, base_asset, quote_asset = position_details

            if quantity == Decimal(0):
                position_values[pair] = Decimal(0)
                continue

            try:
                current_price = await self._try_direct_conversion(base_asset, quote_asset)
                if current_price is None:
                    self.logger.warning("Could not get price for %s", pair)
                    has_missing_prices = True
                    position_values[pair] = Decimal("NaN")
                    continue

                value_in_quote_asset = quantity * current_price
                value_in_valuation_currency: Decimal

                if quote_asset.upper() == target_valuation_currency:
                    value_in_valuation_currency = value_in_quote_asset
                else:
                    conversion_rate = await self.get_currency_conversion_rate(
                        quote_asset.upper(),
                        target_valuation_currency,
                    )
                    if conversion_rate is None:
                        self.logger.warning(
                            "Could not convert %s to %s",
                            quote_asset,
                            target_valuation_currency,
                        )
                        has_missing_prices = True
                        value_in_valuation_currency = Decimal("NaN")
                    else:
                        value_in_valuation_currency = value_in_quote_asset * conversion_rate

                if not value_in_valuation_currency.is_nan():
                    total_value += value_in_valuation_currency
                position_values[pair] = value_in_valuation_currency

            except Exception:  # Removed 'e' as it's unused with self.logger.exception
                has_missing_prices = True
                self.logger.exception(
                    "Error calculating position value for %s",
                    pair,
                    source_module=self._source_module,
                )
                position_values[pair] = Decimal("NaN")

        return total_value, has_missing_prices, position_values

    async def calculate_cash_value(
        self,
        funds: dict[str, Decimal],
        valuation_currency: str | None = None,
    ) -> tuple[Decimal, bool]:
        """Calculate the value of cash balances in valuation currency.

        Args:
        ----
            funds: Dict of currency symbols to Decimal amounts.
            valuation_currency: Optional target currency. Defaults to service default.

        Returns:
        -------
            Tuple of (total_cash_value, has_missing_prices).
        """
        target_valuation_currency = (valuation_currency or self.valuation_currency).upper()
        total_cash_val = Decimal(0)
        has_missing_prices = False

        for currency, amount in funds.items():
            currency_upper = currency.upper()
            if amount == Decimal(0):
                continue

            if currency_upper == target_valuation_currency:
                total_cash_val += amount
            else:
                rate = await self.get_currency_conversion_rate(
                    currency_upper,
                    target_valuation_currency,
                )
                if rate is not None:
                    total_cash_val += amount * rate
                else:
                    self.logger.warning(
                        "Could not convert %s to %s",
                        currency,
                        target_valuation_currency,
                    )
                    has_missing_prices = True

        return total_cash_val, has_missing_prices

    async def update_portfolio_value(
        self,
        _funds: dict[str, Decimal],
        positions: dict[str, PositionInput],
    ) -> tuple[Decimal, dict[str, Decimal], Decimal]:
        """Update total portfolio value, drawdowns, and exposure metrics.

        This is the main method to call periodically for portfolio valuation.

        Args:
        ----
            funds: Current cash balances (Dict[currency_symbol, Decimal]).
            positions: Current positions (Dict[trading_pair, PositionInfo-like]).

        Returns:
        -------
            A tuple: (total_portfolio_value, position_specific_values, exposure_percentage)
        """
        async with self._lock:
            current_time = datetime.now(UTC)
            position_value_result = await self.calculate_position_value(positions)
            self._total_equity = position_value_result[0]
            self._peak_equity = max(self._peak_equity, self._total_equity)
            self._total_drawdown_pct = (
                (self._peak_equity - self._total_equity) / self._peak_equity
            ) * Decimal("100")

            # Update daily and weekly drawdown metrics
            await self._update_drawdown_metrics(self._total_equity, current_time)

            # Calculate exposure percentage
            exposure_pct = await self._calculate_exposure_percentage(positions, self._total_equity)

            self.logger.info(
                "Portfolio value updated: Total=%.2f %s, Exposure=%.2f%%",
                self._total_equity,
                self.valuation_currency,
                exposure_pct,
                source_module=self._source_module,
            )

            position_values = await self.calculate_position_value(positions)
            return (
                self._total_equity,
                position_values[2],
                exposure_pct,
            )

    async def _get_rate_direct_to_valuation_currency(self, base_asset: str) -> Decimal | None:
        """Try to get conversion rate directly from base_asset to valuation_currency."""
        return await self.get_currency_conversion_rate(base_asset, self.valuation_currency)

    async def _get_rate_via_pair_quote_asset(
        self,
        base_asset: str,
        pair_str: str,
    ) -> Decimal | None:
        """Try to get rate via the pair's quote asset as an intermediary."""
        split_pair = pair_str.split("/")
        if len(split_pair) != PAIR_SPLIT_EXPECTED_PARTS:
            self.logger.debug(
                "Cannot get quote asset from pair '%s' for indirect valuation "
                "(invalid format).",
                pair_str,
                source_module=self._source_module,
            )
            return None

        quote_asset_from_pair = split_pair[1]
        price_base_in_quote = await self._try_direct_conversion(base_asset, quote_asset_from_pair)

        if price_base_in_quote is None:
            self.logger.debug(
                "Failed to get price of %s in %s for indirect valuation (pair: %s).",
                base_asset,
                quote_asset_from_pair,
                pair_str,
                source_module=self._source_module,
            )
            return None

        if quote_asset_from_pair.upper() == self.valuation_currency:
            return price_base_in_quote  # Already in valuation currency

        # Convert the value from quote_asset_from_pair to valuation_currency
        conversion_to_valuation_curr = await self.get_currency_conversion_rate(
            quote_asset_from_pair,
            self.valuation_currency,
        )

        if conversion_to_valuation_curr is None:
            self.logger.debug(
                "Failed to convert %s (quote of %s) to %s for indirect valuation.",
                quote_asset_from_pair,
                pair_str,
                self.valuation_currency,
                source_module=self._source_module,
            )
            return None

        return price_base_in_quote * conversion_to_valuation_curr

    async def _get_position_base_asset_value_in_valuation_currency(
        self,
        pair: str,
        pos_data_any: PositionInput,  # pair is used for the indirect strategy
    ) -> Decimal | None:
        """Get the value of a position's base asset in the valuation currency."""
        quantity: Decimal
        base_asset: str

        # 1. Initial Parsing & Validation
        if hasattr(pos_data_any, "quantity") and hasattr(pos_data_any, "base_asset"):
            quantity = pos_data_any.quantity
            base_asset = pos_data_any.base_asset
        elif isinstance(pos_data_any, dict):
            raw_quantity = pos_data_any.get("quantity", 0)
            raw_base_asset = pos_data_any.get("base_asset")
            if raw_base_asset is None:
                self.logger.debug(
                    "Skipping value calculation for %s, missing 'base_asset' in dict.",
                    pair,
                    source_module=self._source_module,
                )
                return None
            quantity = Decimal(str(raw_quantity))
            base_asset = str(raw_base_asset)
        else:
            self.logger.debug(
                "Unsupported position data type for %s: %s.",
                pair,
                type(pos_data_any).__name__,
                source_module=self._source_module,
            )
            return None

        if quantity == Decimal(0):
            return Decimal(0)

        # 2. Attempt valuation strategies
        rate_in_valuation_currency: Decimal | None = None

        # Strategy 1: Direct to valuation currency
        rate_in_valuation_currency = await self._get_rate_direct_to_valuation_currency(base_asset)

        # Strategy 2: Via pair's quote asset (if direct failed)
        if rate_in_valuation_currency is None:
            self.logger.debug(
                "Direct valuation failed for %s. Trying via pair %s.",
                base_asset,
                pair,
                source_module=self._source_module,
            )
            rate_in_valuation_currency = await self._get_rate_via_pair_quote_asset(
                base_asset,
                pair,
            )

        # 3. Calculate final value if rate was found
        if rate_in_valuation_currency is not None:
            return abs(quantity * rate_in_valuation_currency)

        self.logger.warning(
            "Could not determine value for base asset %s (pair: %s) in %s after all strategies.",
            base_asset,
            pair,
            self.valuation_currency,
            source_module=self._source_module,
        )
        return None

    async def _calculate_exposure_percentage(
        self,
        positions: dict[str, PositionInput],
        total_portfolio_value: Decimal,
    ) -> Decimal:
        """Calculate the exposure percentage of positions relative to total portfolio value."""
        if total_portfolio_value == Decimal(0) or total_portfolio_value.is_nan():
            return Decimal(0)

        total_position_abs_value_in_valuation_currency = Decimal(0)

        for pair, pos_data_any in positions.items():
            try:
                position_value = await self._get_position_base_asset_value_in_valuation_currency(
                    pair,
                    pos_data_any,
                )
                if position_value is not None:
                    total_position_abs_value_in_valuation_currency += position_value
            except Exception as e:
                self.logger.debug(
                    "Error calculating exposure component for %s: %s",
                    pair,
                    e,
                    source_module=self._source_module,
                )
                continue

        if total_portfolio_value.is_nan() or total_portfolio_value == Decimal(0):
            return Decimal(0)  # Avoid division by zero or NaN propagation

        return total_position_abs_value_in_valuation_currency / total_portfolio_value

    async def _update_daily_drawdown(
        self,
        current_total_equity: Decimal,
        current_time: datetime,
    ) -> None:
        """Update daily drawdown metrics."""
        should_reset_daily = False
        if self._last_daily_reset_time is None:
            should_reset_daily = True
        else:
            cond1 = (
                current_time.date() > self._last_daily_reset_time.date()
                and current_time.hour >= self._daily_reset_hour_utc
            )
            cond2 = (
                current_time.date() == self._last_daily_reset_time.date()
                and current_time.hour >= self._daily_reset_hour_utc
                and self._last_daily_reset_time.hour < self._daily_reset_hour_utc
            )
            cond3 = (
                current_time.date() > self._last_daily_reset_time.date()
                and self._last_daily_reset_time.hour < self._daily_reset_hour_utc
            )
            if cond1 or cond2 or cond3:
                should_reset_daily = True

        if should_reset_daily:
            self._daily_peak_equity = current_total_equity
            potential_reset_time = current_time.replace(
                hour=self._daily_reset_hour_utc,
                minute=0,
                second=0,
                microsecond=0,
            )
            if current_time >= potential_reset_time:
                self._last_daily_reset_time = potential_reset_time
            else:
                self._last_daily_reset_time = potential_reset_time - timedelta(days=1)
            self.logger.info(
                "Reset daily drawdown peak to %s",
                self._daily_peak_equity,
                source_module=self._source_module,
            )

        self._daily_peak_equity = max(self._daily_peak_equity, current_total_equity)
        if self._daily_peak_equity > 0:
            self._daily_drawdown_pct = (
                (self._daily_peak_equity - current_total_equity) / self._daily_peak_equity
            ) * Decimal("100")  # Ensure percentage is calculated correctly
        else:
            self._daily_drawdown_pct = Decimal(0)

    async def _update_weekly_drawdown(
        self,
        current_total_equity: Decimal,
        current_time: datetime,
    ) -> None:
        """Update weekly drawdown metrics."""
        should_reset_weekly = False
        if self._last_weekly_reset_time is None:
            should_reset_weekly = True
        else:
            days_since_configured_reset_day_this_week = (
                current_time.weekday() - self._weekly_reset_day + 7
            ) % 7
            date_of_this_week_reset_day = (
                current_time - timedelta(days=days_since_configured_reset_day_this_week)
            ).date()
            this_week_reset_event_dt = datetime.combine(
                date_of_this_week_reset_day,
                datetime.min.time(),
                tzinfo=UTC,
            ).replace(hour=self._daily_reset_hour_utc)

            if (
                self._last_weekly_reset_time < this_week_reset_event_dt
                and current_time >= this_week_reset_event_dt
            ) or (current_time - self._last_weekly_reset_time) >= timedelta(days=7):
                should_reset_weekly = True

        if should_reset_weekly:
            self._weekly_peak_equity = current_total_equity
            days_to_subtract = (current_time.weekday() - self._weekly_reset_day + 7) % 7
            actual_reset_day_this_cycle = current_time - timedelta(days=days_to_subtract)
            self._last_weekly_reset_time = actual_reset_day_this_cycle.replace(
                hour=self._daily_reset_hour_utc,
                minute=0,
                second=0,
                microsecond=0,
            )
            if current_time < self._last_weekly_reset_time:
                self._last_weekly_reset_time -= timedelta(weeks=1)
            self.logger.info(
                "Reset weekly drawdown peak to %s",
                self._weekly_peak_equity,
                source_module=self._source_module,
            )

        self._weekly_peak_equity = max(self._weekly_peak_equity, current_total_equity)
        if self._weekly_peak_equity > 0:
            self._weekly_drawdown_pct = (
                (self._weekly_peak_equity - current_total_equity) / self._weekly_peak_equity
            ) * Decimal("100")  # Ensure percentage is calculated correctly
        else:
            self._weekly_drawdown_pct = Decimal(0)

    async def _update_drawdown_metrics(
        self,
        current_total_equity: Decimal,
        current_time: datetime,
    ) -> None:
        """Update daily, weekly, and all-time drawdown metrics."""
        if current_total_equity.is_nan():
            self.logger.warning(
                "Current total equity is NaN, skipping drawdown update.",
                source_module=self._source_module,
            )
            return

        await self._update_daily_drawdown(current_total_equity, current_time)
        await self._update_weekly_drawdown(current_total_equity, current_time)

        # All-Time Drawdown (remains in the main method as it's simpler)
        self._peak_equity = max(self._peak_equity, current_total_equity)
        if self._peak_equity > 0:
            self._total_drawdown_pct = (
                (self._peak_equity - current_total_equity) / self._peak_equity
            ) * Decimal("100")  # Ensure percentage is calculated correctly
        else:
            self._total_drawdown_pct = Decimal(0)
