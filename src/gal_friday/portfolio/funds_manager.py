"""Funds management functionality for the portfolio system."""

import asyncio
from decimal import Decimal
from typing import Any, Optional

from ..exceptions import DataValidationError, InsufficientFundsError
from ..logger_service import LoggerService


class FundsManager:
    """
    Manages available funds for trading, handling deposits, withdrawals,
    and updates based on trade executions and commission payments.
    Ensures thread-safe operations on fund balances.
    """

    def __init__(self, logger_service: LoggerService, valuation_currency: str = "USD") -> None:
        """
        Initialize the funds manager.

        Args
        ----
            logger_service: Service for logging.
            valuation_currency: The currency for overall valuation (default: "USD").
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self._available_funds: dict[str, Decimal] = {}
        self.valuation_currency = valuation_currency
        self._lock = asyncio.Lock()

    @property
    def available_funds(self) -> dict[str, Decimal]:
        """Get a copy of available funds."""
        return self._available_funds.copy()

    async def initialize_funds(self, initial_capital: dict[str, Any]) -> None:
        """
        Initialize funds from configuration.

        Args
        ----
            initial_capital: A dictionary where keys are currency symbols (e.g., "USD")
                             and values are the initial amounts.
        """
        async with self._lock:
            self._available_funds.clear()
            for currency, amount_any in initial_capital.items():
                try:
                    amount = Decimal(str(amount_any))
                    if amount < 0:
                        self.logger.warning("Initial capital for %s is negative (%s), setting to 0.", currency, amount, source_module=self._source_module)
                        amount = Decimal(0)
                    self._available_funds[currency.upper()] = amount
                except Exception as e:
                    self.logger.error("Error processing initial capital for %s: %s. Setting to 0.", currency, e, source_module=self._source_module)
                    self._available_funds[currency.upper()] = Decimal(0)
            self.logger.info(
                "Initialized funds: %s", self._available_funds,
                source_module=self._source_module,
            )

    async def update_funds_for_trade(
        self,
        base_asset: str,
        quote_asset: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        cost_or_proceeds: Decimal,
    ) -> None:
        """
        Update available funds based on a trade execution.

        Args
        ----
            base_asset: The asset being bought or sold.
            quote_asset: The asset used for payment or received.
            side: "BUY" or "SELL".
            quantity: Amount of base_asset traded.
            price: Price of the trade.
            cost_or_proceeds: Net amount in quote_asset. For BUYs, this is a positive
                              value representing cost. For SELLs, it's positive, representing proceeds.

        Raises
        ------
            DataValidationError: If side is invalid
            InsufficientFundsError: If not enough funds for a buy
        """
        if side not in ("BUY", "SELL"):
            raise DataValidationError("Invalid trade side")

        base_asset_upper = base_asset.upper()
        quote_asset_upper = quote_asset.upper()

        async with self._lock:
            current_balance_quote = self._available_funds.get(quote_asset_upper, Decimal(0))
            current_balance_base = self._available_funds.get(base_asset_upper, Decimal(0))

            if side == "BUY":
                if current_balance_quote < cost_or_proceeds:
                    raise InsufficientFundsError(
                        "Insufficient %s funds for trade. Required: %s, Available: %s" % (quote_asset_upper, cost_or_proceeds, current_balance_quote)
                    )
                self._available_funds[quote_asset_upper] = current_balance_quote - cost_or_proceeds
                self._available_funds[base_asset_upper] = current_balance_base + quantity
                self.logger.debug(
                    "Decreased %s by %s for BUY. New balance: %s. Increased %s by %s. New balance: %s",
                    quote_asset_upper, cost_or_proceeds, self._available_funds[quote_asset_upper],
                    base_asset_upper, quantity, self._available_funds[base_asset_upper],
                    source_module=self._source_module,
                )
            elif side == "SELL":
                self._available_funds[quote_asset_upper] = current_balance_quote + cost_or_proceeds
                self._available_funds[base_asset_upper] = current_balance_base - quantity
                self.logger.debug(
                    "Increased %s by %s for SELL. New balance: %s. Decreased %s by %s. New balance: %s",
                    quote_asset_upper, cost_or_proceeds, self._available_funds[quote_asset_upper],
                    base_asset_upper, quantity, self._available_funds[base_asset_upper],
                    source_module=self._source_module,
                )

    async def handle_commission(
        self,
        commission: Decimal,
        commission_asset: Optional[str],
    ) -> None:
        """
        Update funds to account for trading commission.

        Args
        ----
            commission: Commission amount
            commission_asset: Commission currency symbol
        """
        if commission <= Decimal(0) or not commission_asset:
            return  # No commission or no asset specified

        async with self._lock:
            commission_asset_upper = commission_asset.upper()

            # Ensure currency exists in our funds tracking
            current_balance = self._available_funds.get(commission_asset_upper, Decimal(0))

            # Deduct commission
            self._available_funds[commission_asset_upper] = current_balance - commission

            self.logger.debug(
                f"Deducted {commission} {commission_asset_upper} for commission. "
                f"New balance: {self._available_funds.get(commission_asset_upper, Decimal(0))}",
                source_module=self._source_module,
            )

    async def deposit(self, currency: str, amount: Decimal) -> None:
        """
        Record a deposit of funds.

        Args
        ----
            currency: Currency symbol
            amount: Deposit amount

        Raises
        ------
            DataValidationError: If amount is invalid
        """
        if amount <= Decimal(0):
            raise DataValidationError(f"Invalid deposit amount: {amount}")

        async with self._lock:
            currency_upper = currency.upper()
            current_balance = self._available_funds.get(currency_upper, Decimal(0))
            self._available_funds[currency_upper] = current_balance + amount

            self.logger.info(
                f"Deposited {amount} {currency_upper}. "
                f"New balance: {self._available_funds[currency_upper]}",
                source_module=self._source_module,
            )

    async def withdraw(self, currency: str, amount: Decimal) -> None:
        """
        Record a withdrawal of funds.

        Args
        ----
            currency: Currency symbol
            amount: Withdrawal amount

        Raises
        ------
            DataValidationError: If amount is invalid
            InsufficientFundsError: If not enough funds
        """
        if amount <= Decimal(0):
            raise DataValidationError(f"Invalid withdrawal amount: {amount}")

        async with self._lock:
            currency_upper = currency.upper()
            current_balance = self._available_funds.get(currency_upper, Decimal(0))

            if current_balance < amount:
                raise InsufficientFundsError(
                    f"Insufficient {currency_upper} funds for withdrawal. "
                    f"Requested: {amount}, "
                    f"Available: {current_balance}"
                )

            self._available_funds[currency_upper] = current_balance - amount

            self.logger.info(
                f"Withdrew {amount} {currency_upper}. "
                f"New balance: {self._available_funds[currency_upper]}",
                source_module=self._source_module,
            )

    async def reconcile_with_exchange_balances(
        self, exchange_balances: dict[str, Decimal]
    ) -> None:
        """
        Reconciles internal fund balances with exchange-reported balances.

        Args
        ----
            exchange_balances: Dictionary of currency to balance from exchange API
        """
        self.logger.info(
            "Reconciling funds with exchange balances",
            source_module=self._source_module,
        )

        async with self._lock:
            # Track which currencies we've processed
            processed_currencies = set()

            # For each currency at the exchange
            for currency, exchange_amount in exchange_balances.items():
                currency_upper = currency.upper()
                processed_currencies.add(currency_upper)

                internal_amount = self._available_funds.get(currency_upper, Decimal(0))

                # If there's a discrepancy
                if internal_amount != exchange_amount:
                    self.logger.info(
                        f"Reconciling {currency_upper}: Internal {internal_amount} -> "
                        f"Exchange {exchange_amount} (Diff: {exchange_amount - internal_amount})",
                        source_module=self._source_module,
                    )

                    # Update to match exchange
                    self._available_funds[currency_upper] = exchange_amount

            # Check for currencies we have internally that aren't at the exchange
            for currency in list(self._available_funds.keys()):
                if currency not in processed_currencies:
                    # If we track a currency not at exchange, set to zero or remove
                    if self._available_funds[currency] != Decimal(0):
                        self.logger.info(
                            f"Zeroing {currency} balance not found at exchange "
                            f"(was {self._available_funds[currency]})",
                            source_module=self._source_module,
                        )
                        self._available_funds[currency] = Decimal(0)

        self.logger.info(
            f"Reconciliation complete: {self._available_funds}",
            source_module=self._source_module,
        )
