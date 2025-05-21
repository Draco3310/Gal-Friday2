"""Funds management functionality for the portfolio system."""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from ..exceptions import DataValidationError, InsufficientFundsError
from ..logger_service import LoggerService


@dataclass
class TradeParams:
    """Parameters for a trade operation."""

    base_asset: str
    quote_asset: str
    side: str
    quantity: Decimal
    price: Decimal
    cost_or_proceeds: Decimal


class FundsManager:
    """Manage available funds for trading.

    Handles deposits, withdrawals, and updates based on trade executions and
    commission payments. Ensures thread-safe operations on fund balances.
    """

    # Error messages as class variables for TRY003
    INVALID_DEPOSIT_AMOUNT = "Invalid deposit amount: %s"
    INVALID_WITHDRAWAL_AMOUNT = "Invalid withdrawal amount: %s"
    INVALID_TRADE_SIDE = "Trade side must be either 'BUY' or 'SELL'"
    INSUFFICIENT_QUOTE_FUNDS = "Insufficient {} funds for trade. Required: {}, Available: {}"
    INSUFFICIENT_BASE_FUNDS = "Insufficient {} to sell. Required: {}, Available: {}"

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
                        self.logger.warning(
                            "Initial capital for %s is negative (%s), setting to 0.",
                            currency,
                            amount,
                            source_module=self._source_module,
                        )
                        amount = Decimal(0)
                    self._available_funds[currency.upper()] = amount
                except Exception:
                    self.logger.exception(
                        "Error processing initial capital for %s. Setting to 0.",
                        currency,
                        source_module=self._source_module,
                    )
                    self._available_funds[currency.upper()] = Decimal(0)
            self.logger.info(
                "Initialized funds: %s",
                self._available_funds,
                source_module=self._source_module,
            )

    async def update_funds_for_trade(
        self,
        trade: TradeParams,
    ) -> None:
        """
        Update available funds based on a trade execution.

        Args
        ----
            trade: Trade parameters
        """
        if trade.side not in ("BUY", "SELL"):
            raise DataValidationError(self.INVALID_TRADE_SIDE)

        base_asset_upper = trade.base_asset.upper()
        quote_asset_upper = trade.quote_asset.upper()

        # Initialize balances if they don't exist
        current_balance_quote = self._available_funds.get(quote_asset_upper, Decimal(0))
        current_balance_base = self._available_funds.get(base_asset_upper, Decimal(0))

        async with self._lock:
            if trade.side == "BUY":
                if current_balance_quote < trade.cost_or_proceeds:
                    raise InsufficientFundsError(
                        self.INSUFFICIENT_QUOTE_FUNDS.format(
                            quote_asset_upper, trade.cost_or_proceeds, current_balance_quote
                        )
                    )
                self._available_funds[quote_asset_upper] = (
                    current_balance_quote - trade.cost_or_proceeds
                )
                self._available_funds[base_asset_upper] = current_balance_base + trade.quantity
                self.logger.debug(
                    "Decreased %s by %s for BUY. New balance: %s. "
                    "Increased %s by %s. New balance: %s",
                    quote_asset_upper,
                    trade.cost_or_proceeds,
                    self._available_funds[quote_asset_upper],
                    base_asset_upper,
                    trade.quantity,
                    self._available_funds[base_asset_upper],
                    source_module=self._source_module,
                )
            else:  # SELL
                if current_balance_base < trade.quantity:
                    raise InsufficientFundsError(
                        self.INSUFFICIENT_BASE_FUNDS.format(
                            base_asset_upper, trade.quantity, current_balance_base
                        )
                    )
                self._available_funds[base_asset_upper] = current_balance_base - trade.quantity
                self._available_funds[quote_asset_upper] = (
                    current_balance_quote + trade.cost_or_proceeds
                )
                self.logger.debug(
                    "Increased %s by %s for SELL. New balance: %s. "
                    "Decreased %s by %s. New balance: %s",
                    quote_asset_upper,
                    trade.cost_or_proceeds,
                    self._available_funds[quote_asset_upper],
                    base_asset_upper,
                    trade.quantity,
                    self._available_funds[base_asset_upper],
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
            new_balance = current_balance - commission

            # Deduct commission
            self._available_funds[commission_asset_upper] = new_balance

            self.logger.debug(
                "Deducted %s %s for commission. New balance: %s",
                commission,
                commission_asset_upper,
                new_balance,
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
            raise DataValidationError(self.INVALID_DEPOSIT_AMOUNT % amount)

        async with self._lock:
            currency_upper = currency.upper()
            current_balance = self._available_funds.get(currency_upper, Decimal(0))
            new_balance = current_balance + amount
            self._available_funds[currency_upper] = new_balance

            self.logger.info(
                "Deposited %s %s. New balance: %s",
                amount,
                currency_upper,
                new_balance,
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
            raise DataValidationError(self.INVALID_WITHDRAWAL_AMOUNT % amount)


        async with self._lock:
            currency_upper = currency.upper()
            current_balance = self._available_funds.get(currency_upper, Decimal(0))

            if current_balance < amount:
                raise InsufficientFundsError(
                    self.INSUFFICIENT_FUNDS % (currency_upper, amount, current_balance)
                )

            new_balance = current_balance - amount
            self._available_funds[currency_upper] = new_balance

            self.logger.info(
                "Withdrew %s %s. New balance: %s",
                amount,
                currency_upper,
                new_balance,
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
                        "Reconciling %s: Internal %s -> Exchange %s (Diff: %s)",
                        currency_upper,
                        internal_amount,
                        exchange_amount,
                        exchange_amount - internal_amount,
                        source_module=self._source_module,
                    )

                    # Update to match exchange
                    self._available_funds[currency_upper] = exchange_amount

            # Check for currencies we have internally that aren't at the exchange
            for currency, balance in list(self._available_funds.items()):
                if currency not in processed_currencies and balance != Decimal(0):
                    # If we track a currency not at exchange, set to zero or remove
                    self.logger.info(
                        "Zeroing %s balance not found at exchange (was %s)",
                        currency,
                        balance,
                        source_module=self._source_module,
                    )
                    self._available_funds[currency] = Decimal(0)

        self.logger.info(
            "Reconciliation complete: %s",
            self._available_funds,
            source_module=self._source_module,
        )
