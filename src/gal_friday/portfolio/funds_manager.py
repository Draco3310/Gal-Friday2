"""Funds management functionality for the portfolio system."""

import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional

from ..exceptions import DataValidationError, InsufficientFundsError


class FundsManager:
    """
    Manages available funds in different currencies.

    Handles updates to funds based on trades, deposits, withdrawals,
    and converts between different currencies for valuation.
    """

    def __init__(self, logger_service, valuation_currency: str = "USD"):
        """
        Initialize the funds manager.

        Args:
            logger_service: Logger service for logging
            valuation_currency: Base currency for portfolio valuation
        """
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        self._available_funds: Dict[str, Decimal] = {}
        self.valuation_currency = valuation_currency
        self._lock = asyncio.Lock()

    @property
    def available_funds(self) -> Dict[str, Decimal]:
        """Get a copy of available funds."""
        return self._available_funds.copy()

    async def initialize_funds(self, initial_capital: Dict[str, Any]) -> None:
        """
        Initialize funds from configuration.

        Args:
            initial_capital: Dictionary of initial capital by currency
        """
        async with self._lock:
            # Convert to Decimal and uppercase currency codes
            self._available_funds = {
                k.upper(): Decimal(str(v)) for k, v in initial_capital.items()
            }

            # Ensure valuation currency is present, default to 0
            if self.valuation_currency not in self._available_funds:
                self._available_funds[self.valuation_currency] = Decimal("0")

            self.logger.info(
                f"Initialized funds: {self._available_funds}",
                source_module=self._source_module,
            )

    async def update_funds_for_trade(
        self,
        quote_asset: str,
        side: str,
        cost_or_proceeds: Decimal,
    ) -> None:
        """
        Updates available funds based on trade execution.

        Args:
            quote_asset: Quote currency symbol
            side: Trade side ("BUY" or "SELL")
            cost_or_proceeds: Total cost or proceeds amount

        Raises:
            DataValidationError: If side is invalid
            InsufficientFundsError: If not enough funds for a buy
        """
        if side not in ("BUY", "SELL"):
            raise DataValidationError(f"Invalid side '{side}' in trade")

        async with self._lock:
            # Ensure currency exists in our funds tracking
            current_balance = self._available_funds.get(quote_asset, Decimal(0))

            if side == "BUY":
                if current_balance < cost_or_proceeds:
                    raise InsufficientFundsError(
                        f"Insufficient {quote_asset} funds for trade. "
                        f"Required: {cost_or_proceeds}, "
                        f"Available: {current_balance}"
                    )
                self._available_funds[quote_asset] = current_balance - cost_or_proceeds
                self.logger.debug(
                    f"Decreased {quote_asset} by {cost_or_proceeds} for BUY. "
                    f"New balance: {self._available_funds[quote_asset]}",
                    source_module=self._source_module,
                )
            elif side == "SELL":
                self._available_funds[quote_asset] = current_balance + cost_or_proceeds
                self.logger.debug(
                    f"Increased {quote_asset} by {cost_or_proceeds} for SELL. "
                    f"New balance: {self._available_funds[quote_asset]}",
                    source_module=self._source_module,
                )

    async def handle_commission(
        self,
        commission: Decimal,
        commission_asset: Optional[str],
    ) -> None:
        """
        Updates funds to account for trading commission.

        Args:
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
        Records a deposit of funds.

        Args:
            currency: Currency symbol
            amount: Deposit amount

        Raises:
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
        Records a withdrawal of funds.

        Args:
            currency: Currency symbol
            amount: Withdrawal amount

        Raises:
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
            
    async def reconcile_with_exchange_balances(self, exchange_balances: Dict[str, Decimal]) -> None:
        """
        Reconciles internal fund balances with exchange-reported balances.
        
        Args:
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
