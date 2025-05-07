# Portfolio Manager Module

import json
from typing import (  # Reformatted import
    Dict,
    Optional,
    Any,
    TYPE_CHECKING,
    Tuple,
    List,
    Callable,
    Coroutine,
    cast,
    Protocol,
)
from decimal import Decimal, getcontext
from dataclasses import dataclass, field
import asyncio
from datetime import datetime
from collections import defaultdict
import uuid  # Add missing import for uuid

# Set Decimal precision
getcontext().prec = 28  # Set precision for Decimal calculations

# Import type hints when type checking
if TYPE_CHECKING:
    from .core.events import Event, EventType, ExecutionReportEvent
    from .core.pubsub import PubSubManager
    from .config_manager import ConfigManager
    from .logger_service import LoggerService
    from .market_price_service import MarketPriceService
    from .execution_handler import ExecutionHandler
else:
    # Create placeholder classes for runtime
    from .core.placeholder_classes import (  # noqa: F401
        Event,
        EventType,
        ExecutionReportEvent,
        ConfigManager,
        PubSubManager,
        MarketPriceService,
        ExecutionHandler,
    )


@dataclass
class TradeInfo:
    """Stores information about a single trade."""

    timestamp: datetime
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal = Decimal(0)
    commission_asset: str = ""
    realized_pnl: Decimal = Decimal(0)


@dataclass
class PositionInfo:
    """Stores information about a specific asset position."""

    trading_pair: str
    base_asset: str
    quote_asset: str
    quantity: Decimal = Decimal(0)
    average_entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)
    trade_history: List[TradeInfo] = field(default_factory=list)
    # Add other fields if needed, e.g., unrealized_pnl, last_update_time


# Define a protocol for the methods expected by _reconcile_with_exchange
class ReconcilableExecutionHandler(Protocol):
    async def get_account_balances(self) -> Dict[str, Decimal]: ...
    async def get_open_positions(self) -> Dict[str, PositionInfo]: ...


class PortfolioManager:
    """Manages the state of the trading portfolio, including positions, cash balances,
    equity, and drawdown.

    Consumes ExecutionReportEvents to update its internal state.
    Provides a synchronous method to query the current portfolio state.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: "PubSubManager",
        market_price_service: "MarketPriceService",
        logger_service: "LoggerService",
        execution_handler: Optional["ExecutionHandler"] = None,
    ):
        self.logger = logger_service
        self.config_manager = config_manager
        self.pubsub = pubsub_manager
        self.market_price_service = market_price_service
        self._execution_handler = execution_handler
        self._subscription_queue: Optional[asyncio.Queue] = None

        # --- State Variables ---
        # { currency_code: Decimal(amount) } - Represents available cash
        self._available_funds: Dict[str, Decimal] = {}
        # { trading_pair: PositionInfo }
        self._positions: Dict[str, PositionInfo] = {}
        # Current total portfolio value in valuation currency
        self._total_equity: Decimal = Decimal(0)
        # Highest recorded total equity
        self._peak_equity: Decimal = Decimal(0)
        # Current total drawdown percentage
        self._total_drawdown_pct: Decimal = Decimal(0)

        # Daily and weekly drawdown tracking
        self._daily_peak_equity: Decimal = Decimal(0)
        self._weekly_peak_equity: Decimal = Decimal(0)
        self._daily_drawdown_pct: Decimal = Decimal(0)
        self._weekly_drawdown_pct: Decimal = Decimal(0)
        self._last_daily_reset_time: Optional[datetime] = None
        self._last_weekly_reset_time: Optional[datetime] = None

        # Order tracking for cancellations
        self._pending_orders: Dict[str, Dict[str, Any]] = {}

        # Load drawdown reset configuration
        self._daily_reset_hour_utc = self.config_manager.get_int(
            "portfolio.drawdown.daily_reset_hour_utc", 0
        )
        self._weekly_reset_day = self.config_manager.get_int(
            "portfolio.drawdown.weekly_reset_day", 0
        )  # 0=Monday

        # Exchange reconciliation configuration
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._reconciliation_interval = self.config_manager.get_int(
            "portfolio.reconciliation.interval_seconds", 3600
        )  # Default 1hr
        self._reconciliation_threshold = Decimal(
            self.config_manager.get("portfolio.reconciliation.threshold", "0.01")
        )  # Default 1%
        self._auto_reconcile = self.config_manager.get_bool(
            "portfolio.reconciliation.auto_update", False
        )

        # Lock for potential future concurrent access (not strictly needed with
        # single listener)
        self._lock = asyncio.Lock()
        # Store the execution report handler for unsubscribing
        self._execution_report_handler: Optional[
            Callable[[ExecutionReportEvent], Coroutine[Any, Any, Any]]
        ] = None

        # Cache for latest prices and state update timestamp
        self._last_known_prices: Dict[str, Decimal] = {}
        self._last_state_update_time: Optional[datetime] = None
        self._last_total_exposure_pct: Decimal = Decimal(0)

        self.valuation_currency = self.config_manager.get("portfolio.valuation_currency", "USD")

        self._load_initial_state()

        # Initialize equity based on initial state
        asyncio.create_task(self._update_portfolio_value_async())  # Initial calculation async

        self.logger.info("PortfolioManager initialized.", source_module=self.__class__.__name__)
        self.logger.info(
            f"Valuation Currency: {self.valuation_currency}",
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            f"Initial Available Funds: {self._available_funds}",
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            f"Initial Positions: {self._positions}",
            source_module=self.__class__.__name__,
        )

    def _load_initial_state(self) -> None:
        """Loads initial portfolio state, prioritizing config."""
        # Load available funds (initial capital)
        initial_funds = self.config_manager.get("portfolio.initial_capital", {})
        self._available_funds = {k.upper(): Decimal(str(v)) for k, v in initial_funds.items()}

        # Ensure valuation currency is present in funds, default to 0 if not
        # explicitly set
        if self.valuation_currency not in self._available_funds:
            self._available_funds[self.valuation_currency] = Decimal("0")

        # TODO: Load initial positions from config or exchange if needed (MVP
        # starts flat)
        initial_positions_config = self.config_manager.get("portfolio.initial_positions", {})
        for pair, pos_data in initial_positions_config.items():
            try:
                base, quote = self._split_symbol(pair)
                self._positions[pair] = PositionInfo(
                    trading_pair=pair,
                    base_asset=base,
                    quote_asset=quote,
                    quantity=Decimal(str(pos_data.get("quantity", 0))),
                    average_entry_price=Decimal(str(pos_data.get("average_entry_price", 0))),
                )
            except ValueError:
                self.logger.error(
                    f"Invalid trading pair format in initial_positions: {pair}",
                    source_module=self.__class__.__name__,
                )
            except Exception as e:
                self.logger.error(
                    f"Error loading initial position for {pair}: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )

        self.logger.debug(
            f"Loaded initial state: Funds={self._available_funds}, Positions={self._positions}",
            source_module=self.__class__.__name__,
        )

    async def start(self) -> None:
        """Subscribes to execution reports and starts reconciliation task if configured."""
        # Store the handler to enable unsubscribing later
        self._execution_report_handler = self._handle_execution_report

        # Subscribe to execution reports
        self.pubsub.subscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)

        self.logger.info(
            "Subscribed to EXECUTION_REPORT events.", source_module=self.__class__.__name__
        )

        # Start reconciliation task if configured and execution handler is available
        if self._reconciliation_interval > 0 and self._execution_handler is not None:
            self.logger.info(
                f"Starting periodic reconciliation every {self._reconciliation_interval}s.",
                source_module=self.__class__.__name__,
            )
            self._reconciliation_task = asyncio.create_task(self._run_periodic_reconciliation())
        else:
            if self._reconciliation_interval <= 0:
                self.logger.info(
                    "Exchange reconciliation disabled by configuration.",
                    source_module=self.__class__.__name__,
                )
            elif self._execution_handler is None:
                self.logger.warning(
                    "Exchange reconciliation disabled because execution handler is not available.",
                    source_module=self.__class__.__name__,
                )

    async def stop(self) -> None:
        """Unsubscribes from execution reports and stops reconciliation task."""
        # Use the stored handler to unsubscribe
        if self._execution_report_handler is not None:
            try:
                self.pubsub.unsubscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
                self.logger.info(
                    "Unsubscribed from EXECUTION_REPORT events.",
                    source_module=self.__class__.__name__,
                )
                self._execution_report_handler = None
            except Exception as e:
                self.logger.error(
                    f"Failed to unsubscribe PortfolioManager from EXECUTION_REPORT events: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )
        else:
            self.logger.info(
                "PortfolioManager was not subscribed or already stopped.",
                source_module=self.__class__.__name__,
            )

        # Stop reconciliation task if running
        if self._reconciliation_task and not self._reconciliation_task.done():
            self.logger.info(
                "Stopping reconciliation task...", source_module=self.__class__.__name__
            )
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                self.logger.info(
                    "Reconciliation task cancelled.", source_module=self.__class__.__name__
                )
            except Exception as e:
                self.logger.error(
                    f"Error stopping reconciliation task: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )
            self._reconciliation_task = None

    def _validate_execution_report(self, event: "ExecutionReportEvent") -> bool:
        """Validates the execution report event data."""
        # Handle order cancellations
        if event.order_status == "CANCELED":
            self._handle_order_cancellation(event)
            return False  # Not processing cancellations for position updates

        if event.order_status not in ["FILLED", "PARTIALLY_FILLED"]:
            self.logger.debug(
                f"Ignoring execution report with status: {event.order_status}",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    def _handle_order_cancellation(self, event: "ExecutionReportEvent") -> None:
        """Handles cancellation of an order."""
        self.logger.info(
            f"Order {event.exchange_order_id} for {event.trading_pair} was cancelled.",
            source_module=self.__class__.__name__,
        )

        # Remove from pending orders if being tracked
        if event.exchange_order_id in self._pending_orders:
            del self._pending_orders[event.exchange_order_id]
            self.logger.debug(
                f"Removed cancelled order {event.exchange_order_id} from pending orders tracking.",
                source_module=self.__class__.__name__,
            )

    def _parse_execution_values(
        self, event: "ExecutionReportEvent"
    ) -> Tuple[str, str, Decimal, Decimal, Decimal, Optional[str]]:
        """Parses and validates values from the execution report."""
        pair = event.trading_pair
        side = event.side.upper()

        # Ensure values from event are Decimals, handle potential None
        quantity_filled = (
            Decimal(str(event.quantity_filled))
            if event.quantity_filled is not None
            else Decimal(0)
        )
        avg_fill_price = (
            Decimal(str(event.average_fill_price))
            if event.average_fill_price is not None
            else Decimal(0)
        )
        commission = Decimal(str(event.commission)) if event.commission is not None else Decimal(0)
        commission_asset = event.commission_asset.upper() if event.commission_asset else None

        if quantity_filled <= 0 or avg_fill_price <= 0:
            self.logger.warning(
                f"Ignoring execution report with zero or negative quantity/price: {event}",
                source_module=self.__class__.__name__,
            )
            raise ValueError("Invalid quantity or price")

        return pair, side, quantity_filled, avg_fill_price, commission, commission_asset

    def _update_funds_for_trade(
        self,
        quote_asset: str,
        side: str,
        cost_or_proceeds: Decimal,
        event_side: str,
    ) -> bool:
        """Updates available funds based on trade execution."""
        self._available_funds[quote_asset] = self._available_funds.get(quote_asset, Decimal(0))
        if side == "BUY":
            self._available_funds[quote_asset] -= cost_or_proceeds
        elif side == "SELL":
            self._available_funds[quote_asset] += cost_or_proceeds
        else:
            self.logger.error(
                f"Invalid side '{event_side}' in execution report. Funds not updated.",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    def _update_position_for_trade(
        self,
        pair: str,
        base_asset: str,
        quote_asset: str,
        side: str,
        quantity_filled: Decimal,
        price: Decimal,
        cost_or_proceeds: Decimal,
        timestamp: datetime,
        commission: Decimal,
        commission_asset: Optional[str],
    ) -> None:
        """Updates position information based on trade execution."""
        position = self._positions.get(pair)
        if position is None:
            position = PositionInfo(
                trading_pair=pair, base_asset=base_asset, quote_asset=quote_asset
            )
            self._positions[pair] = position

        current_quantity = position.quantity
        current_avg_price = position.average_entry_price

        # Create trade record
        trade = TradeInfo(
            timestamp=timestamp,
            side=side,
            quantity=quantity_filled,
            price=price,
            commission=commission,
            commission_asset=commission_asset if commission_asset else "",
        )

        realized_pnl = Decimal(0)

        if side == "BUY":
            # If this is opening or adding to a long position
            if current_quantity >= 0:
                new_total_cost = (current_quantity * current_avg_price) + cost_or_proceeds
                position.quantity += quantity_filled
                position.average_entry_price = (
                    new_total_cost / position.quantity if position.quantity != 0 else Decimal(0)
                )
            # If this is reducing a short position, calculate realized P&L
            else:
                # When covering a short position (BUY while having negative quantity)
                # Realized P&L = (entry price - exit price) * quantity being covered
                realized_pnl = (current_avg_price - price) * min(
                    abs(current_quantity), quantity_filled
                )

                # Update position
                position.quantity += quantity_filled
                # If position direction changes or closes, reset avg price
                if position.quantity >= 0:
                    # If completely covered with leftover, set new avg price for the remainder
                    if position.quantity > 0:
                        # Calculate excess quantity (new long position)
                        position.average_entry_price = (
                            price  # New entry price for the long position
                        )
                # Add realized P&L to position total
                position.realized_pnl += realized_pnl
                trade.realized_pnl = realized_pnl

                self.logger.info(
                    f"Realized P&L for covering short {pair}: {realized_pnl:.4f} {quote_asset}",
                    source_module=self.__class__.__name__,
                )

        elif side == "SELL":
            # If this is opening or adding to a short position
            if current_quantity <= 0:
                new_total_cost = (abs(current_quantity) * current_avg_price) + cost_or_proceeds
                position.quantity -= quantity_filled
                position.average_entry_price = (
                    new_total_cost / abs(position.quantity)
                    if position.quantity != 0
                    else Decimal(0)
                )
            # If this is reducing a long position, calculate realized P&L
            else:
                # When selling a long position
                # Realized P&L = (exit price - entry price) * quantity being sold
                realized_pnl = (price - current_avg_price) * min(current_quantity, quantity_filled)

                # Update position
                position.quantity -= quantity_filled
                # If position direction changes or closes, reset avg price
                if position.quantity <= 0:
                    # If completely liquidated with new short position
                    if position.quantity < 0:
                        # Calculate excess quantity (new short position)
                        position.average_entry_price = (
                            price  # New entry price for the short position
                        )
                    else:
                        # Position closed exactly
                        position.average_entry_price = Decimal(0)

                # Add realized P&L to position total
                position.realized_pnl += realized_pnl
                trade.realized_pnl = realized_pnl

                self.logger.info(
                    f"Realized P&L for selling long {pair}: {realized_pnl:.4f} {quote_asset}",
                    source_module=self.__class__.__name__,
                )

        # Add trade to history
        position.trade_history.append(trade)

    def _handle_commission(
        self,
        commission: Decimal,
        commission_asset: Optional[str],
    ) -> None:
        """Handles commission deduction from available funds."""
        if commission > 0 and commission_asset:
            self._available_funds[commission_asset] = self._available_funds.get(
                commission_asset, Decimal(0)
            )
            self._available_funds[commission_asset] -= commission
            self.logger.debug(
                (f"Deducted commission {commission} {commission_asset} " "from available funds."),
                source_module=self.__class__.__name__,
            )
            if self._available_funds[commission_asset] < 0:
                self.logger.warning(
                    (
                        f"Available funds for {commission_asset} went negative "
                        "after commission deduction."
                    ),
                    source_module=self.__class__.__name__,
                )

    async def _handle_execution_report(self, event: "ExecutionReportEvent") -> None:
        """Processes incoming execution report events."""
        try:
            # Make sure the handler accepts the correct Event type
            if not isinstance(event, ExecutionReportEvent):
                self.logger.warning(
                    f"Received non-ExecutionReportEvent in handler: {type(event)}",
                    source_module=self.__class__.__name__,
                )
                return

            self.logger.debug(
                f"Handling execution report: {event.exchange_order_id} - {event.order_status}",
                source_module=self.__class__.__name__,
            )

            if not self._validate_execution_report(event):
                return

            try:
                # Parse and validate values from event
                pair, side, qty_filled, avg_price, commission, commission_asset = (
                    self._parse_execution_values(event)
                )

                # Extract base and quote assets
                base_asset, quote_asset = self._split_symbol(pair)

                # Calculate cost/proceeds in quote currency
                cost_or_proceeds = qty_filled * avg_price

                # Update state atomically using lock
                async with self._lock:
                    # Update funds
                    if not self._update_funds_for_trade(
                        quote_asset, side, cost_or_proceeds, event.side
                    ):
                        return

                    # Update position with realized P&L tracking
                    self._update_position_for_trade(
                        pair,
                        base_asset,
                        quote_asset,
                        side,
                        qty_filled,
                        avg_price,
                        cost_or_proceeds,
                        event.timestamp,
                        commission,
                        commission_asset,
                    )

                    # Handle commission
                    self._handle_commission(commission, commission_asset)

                # Recalculate portfolio value - now async
                await self._update_portfolio_value_async()

            except ValueError as e:
                self.logger.warning(
                    f"Invalid execution report values: {e}", source_module=self.__class__.__name__
                )

        except Exception as e:
            self.logger.error(
                f"Error processing execution report: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    def _log_updated_state(self) -> None:
        """Logs the updated portfolio state after changes."""
        funds_str = {
            k: str(v.quantize(Decimal("0.0001"))) for k, v in self._available_funds.items()
        }
        self.logger.info(
            f"State after trade: Funds={funds_str}, Positions={self._positions}",
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            (
                f"Current Equity: {self._total_equity:.2f} {self.valuation_currency}, "
                f"Peak Equity: {self._peak_equity:.2f}, "
                f"Total Drawdown: {self._total_drawdown_pct:.2f}%, "
                f"Daily Drawdown: {self._daily_drawdown_pct:.2f}%, "
                f"Weekly Drawdown: {self._weekly_drawdown_pct:.2f}%"
            ),
            source_module=self.__class__.__name__,
        )

    async def _calculate_cash_value(self) -> Tuple[Decimal, bool]:
        """Calculates the value of cash balances in valuation currency."""
        current_total_value = Decimal(0)
        missing_prices = False

        for currency, amount in self._available_funds.items():
            if amount == 0:
                continue  # Skip zero balances

            if currency == self.valuation_currency:
                current_total_value += amount
            else:
                value, has_missing = await self._convert_currency_value(currency, amount)
                current_total_value += value
                missing_prices = missing_prices or has_missing

        return current_total_value, missing_prices

    async def _convert_currency_value(
        self, currency: str, amount: Decimal
    ) -> Tuple[Decimal, bool]:
        """Converts a currency amount to valuation currency."""
        # Construct pairs for conversion
        conversion_pair = f"{currency}/{self.valuation_currency}"
        inv_pair = f"{self.valuation_currency}/{currency}"

        # Try direct conversion
        conversion_rate = await self._get_latest_price_async(conversion_pair)
        if conversion_rate is not None and conversion_rate > 0:
            return amount * conversion_rate, False

        # Try inverse conversion
        inv_price = await self._get_latest_price_async(inv_pair)
        if inv_price is not None and inv_price > 0:
            return amount / inv_price, False

        # Log warning if no conversion possible
        self.logger.warning(
            (
                f"Missing price for fund {currency} to "
                f"{self.valuation_currency}. Cannot accurately value."
            ),
            source_module=self.__class__.__name__,
        )
        return Decimal(0), True

    async def _calculate_position_value(self) -> Tuple[Decimal, bool, Dict[str, Decimal]]:
        """Calculates the value of positions in valuation currency."""
        current_total_value = Decimal(0)
        missing_prices = False
        calculated_prices_dict = {}

        for pair, position in self._positions.items():
            if position.quantity == 0:
                continue  # Skip closed positions

            value, has_missing, price = await self._calculate_single_position_value(pair, position)
            current_total_value += value
            missing_prices = missing_prices or has_missing

            if price is not None:
                calculated_prices_dict[pair] = price

        return current_total_value, missing_prices, calculated_prices_dict

    async def _calculate_single_position_value(
        self, pair: str, position: PositionInfo
    ) -> Tuple[Decimal, bool, Optional[Decimal]]:
        """Calculates the value of a single position in valuation currency."""
        market_price = await self._get_latest_price_async(pair)
        if market_price is None or market_price <= 0:
            self.logger.warning(
                f"Missing market price for position {pair}. Cannot accurately value.",
                source_module=self.__class__.__name__,
            )
            return Decimal(0), True, None

        market_value_base = position.quantity * market_price

        # If quote asset is valuation currency, return directly
        if position.quote_asset == self.valuation_currency:
            return market_value_base, False, market_price

        # Convert to valuation currency
        converted_value, has_missing = await self._convert_currency_value(
            position.quote_asset, market_value_base
        )
        return converted_value, has_missing, market_price

    async def _get_latest_price_async(self, pair: str) -> Optional[Decimal]:
        """Get the latest price asynchronously from the market price service."""
        try:
            # Correctly use await for the asynchronous method
            price = await self.market_price_service.get_latest_price(pair)

            if isinstance(price, Decimal):
                # Cache the price for use in get_current_state
                self._last_known_prices[pair] = price
                return price
            elif price is not None:
                # Convert to Decimal if needed
                try:
                    decimal_price = Decimal(str(price))
                    # Cache the price
                    self._last_known_prices[pair] = decimal_price
                    return decimal_price
                except (ValueError, TypeError):
                    self.logger.error(
                        f"Invalid price format returned for {pair}: {price}",
                        source_module=self.__class__.__name__,
                    )
        except Exception as e:
            self.logger.error(
                f"Error getting latest price for {pair}: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

        return None

    def _update_drawdown_metrics(self) -> None:
        """Updates total, daily, and weekly drawdown metrics."""
        now = datetime.utcnow()

        # --- Total Drawdown ---
        if self._total_equity > self._peak_equity:
            self._peak_equity = self._total_equity
        # Calculate total drawdown % (ensure peak > 0)
        if self._peak_equity > Decimal(0):
            self._total_drawdown_pct = (
                (self._peak_equity - self._total_equity) / self._peak_equity
            ) * 100
        else:
            self._total_drawdown_pct = Decimal(0)

        # --- Daily Drawdown ---
        # Check for reset condition (new day based on UTC reset hour)
        if (
            self._last_daily_reset_time is None
            or (now.date() > self._last_daily_reset_time.date())
            or (
                now.date() == self._last_daily_reset_time.date()
                and now.hour >= self._daily_reset_hour_utc
                and self._last_daily_reset_time.hour < self._daily_reset_hour_utc
            )
        ):

            self.logger.info(
                f"Resetting daily peak equity at {now}. Previous peak: {self._daily_peak_equity}",
                source_module=self.__class__.__name__,
            )
            self._daily_peak_equity = self._total_equity  # Reset to current equity
            self._last_daily_reset_time = now
            self._daily_drawdown_pct = Decimal(0)  # Reset drawdown too

        # Update daily peak
        if self._total_equity > self._daily_peak_equity:
            self._daily_peak_equity = self._total_equity

        # Calculate daily drawdown %
        if self._daily_peak_equity > Decimal(0):
            self._daily_drawdown_pct = (
                (self._daily_peak_equity - self._total_equity) / self._daily_peak_equity
            ) * 100
        else:
            self._daily_drawdown_pct = Decimal(0)

        # --- Weekly Drawdown ---
        # Check for reset condition (new week, reset on configured day)
        is_reset_day = now.weekday() == self._weekly_reset_day
        is_new_week = (
            self._last_weekly_reset_time is None
            or now.isocalendar()[1] != self._last_weekly_reset_time.isocalendar()[1]
            or now.year != self._last_weekly_reset_time.year
        )

        if is_reset_day and is_new_week:
            self.logger.info(
                f"Resetting weekly peak equity at {now}. "
                f"Previous peak: {self._weekly_peak_equity}",
                source_module=self.__class__.__name__,
            )
            self._weekly_peak_equity = self._total_equity  # Reset to current equity
            self._last_weekly_reset_time = now
            self._weekly_drawdown_pct = Decimal(0)  # Reset drawdown

        # Update weekly peak
        if self._total_equity > self._weekly_peak_equity:
            self._weekly_peak_equity = self._total_equity

        # Calculate weekly drawdown %
        if self._weekly_peak_equity > Decimal(0):
            self._weekly_drawdown_pct = (
                (self._weekly_peak_equity - self._total_equity) / self._weekly_peak_equity
            ) * 100
        else:
            self._weekly_drawdown_pct = Decimal(0)

        self.logger.debug(
            f"Drawdown Update: Total={self._total_drawdown_pct:.2f}%, "
            f"Daily={self._daily_drawdown_pct:.2f}% (Peak={self._daily_peak_equity:.4f}), "
            f"Weekly={self._weekly_drawdown_pct:.2f}% (Peak={self._weekly_peak_equity:.4f})",
            source_module=self.__class__.__name__,
        )

    async def _update_portfolio_value_async(self) -> None:
        """Recalculates the total portfolio value in the base currency,
        updates peak equity, and calculates drawdown.
        Uses the market_price_service to get latest prices asynchronously.
        """
        # Calculate value of cash balances
        cash_value, missing_prices_cash = await self._calculate_cash_value()

        # Calculate value of positions
        position_value, missing_prices_pos, calculated_prices_dict = (
            await self._calculate_position_value()
        )

        # Total value and missing prices flag
        current_total_value = cash_value + position_value
        missing_prices = missing_prices_cash or missing_prices_pos

        if missing_prices:
            self.logger.error(
                "Could not calculate total portfolio value accurately due to missing prices.",
                source_module=self.__class__.__name__,
            )

        # Calculate total exposure
        total_exposure_pct = Decimal(0)
        if self._total_equity > 0:
            # We should calculate exposure based on converted position values
            total_exposure_pct = (abs(position_value) / current_total_value) * 100

        async with self._lock:
            # Update total equity
            self._total_equity = current_total_value

            # Update drawdown metrics
            self._update_drawdown_metrics()

            # Cache values for get_current_state
            self._last_known_prices.update(calculated_prices_dict)
            self._last_total_exposure_pct = total_exposure_pct
            self._last_state_update_time = datetime.utcnow()

            # Log updated state
            self._log_updated_state()

        self.logger.debug(
            (
                f"Updated Portfolio Value: Equity={self._total_equity:.4f}, "
                f"Peak={self._peak_equity:.4f}, "
                f"Drawdown={self._total_drawdown_pct:.2f}%"
            ),
            source_module=self.__class__.__name__,
        )

    def get_current_state(self) -> Dict[str, Any]:
        """**Synchronous Method**
        Returns the latest known portfolio state snapshot.
        NOTE: Relies on cached prices and equity values that were last calculated
              by _update_portfolio_value_async after execution reports.
        """
        self.logger.debug("get_current_state called.", source_module=self.__class__.__name__)

        positions_dict = {}
        for pair, pos_info in self._positions.items():
            if pos_info.quantity != 0:  # Only include open positions
                # Get cached price
                latest_price = self._last_known_prices.get(pair)
                market_value = (
                    pos_info.quantity * latest_price if latest_price is not None else None
                )
                unrealized_pnl = (
                    (latest_price - pos_info.average_entry_price) * pos_info.quantity
                    if latest_price is not None
                    else None
                )

                positions_dict[pair] = {
                    "base_asset": pos_info.base_asset,
                    "quote_asset": pos_info.quote_asset,
                    "quantity": str(pos_info.quantity),
                    "average_entry_price": str(pos_info.average_entry_price),
                    "current_market_value": (
                        str(market_value) if market_value is not None else None
                    ),
                    "unrealized_pnl": (
                        str(unrealized_pnl) if unrealized_pnl is not None else None
                    ),
                    "realized_pnl": str(pos_info.realized_pnl),
                    "trade_count": len(pos_info.trade_history),
                }

        # Use cached exposure percentage
        total_exposure_pct = self._last_total_exposure_pct

        return {
            "timestamp": (
                self._last_state_update_time.isoformat() + "Z"
                if self._last_state_update_time
                else datetime.utcnow().isoformat() + "Z"
            ),
            "valuation_currency": self.valuation_currency,
            "total_equity": str(self._total_equity),
            "available_funds": {k: str(v) for k, v in self._available_funds.items()},
            "positions": positions_dict,
            "total_exposure_pct": str(total_exposure_pct),
            "daily_drawdown_pct": str(self._daily_drawdown_pct),
            "weekly_drawdown_pct": str(self._weekly_drawdown_pct),
            "total_drawdown_pct": str(self._total_drawdown_pct),
        }

    def get_available_funds(self, currency: str) -> Decimal:
        """Returns the available funds for a specific currency."""
        return self._available_funds.get(currency.upper(), Decimal(0))

    def get_current_equity(self) -> Decimal:
        """Returns the current equity value."""
        return self._total_equity

    def get_position_history(self, pair: str) -> List[Dict[str, Any]]:
        """Returns the trade history for a specific pair."""
        position = self._positions.get(pair)
        if not position:
            return []

        result = []
        for trade in position.trade_history:
            result.append(
                {
                    "timestamp": trade.timestamp.isoformat() + "Z",
                    "side": trade.side,
                    "quantity": str(trade.quantity),
                    "price": str(trade.price),
                    "commission": str(trade.commission),
                    "commission_asset": trade.commission_asset,
                    "realized_pnl": str(trade.realized_pnl),
                }
            )

        return result

    def get_open_positions(self) -> List[PositionInfo]:
        """Returns a list of open positions."""
        return [pos for pos in self._positions.values() if pos.quantity != 0]

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        """Splits a trading symbol (e.g., 'XRP/USD') into base and quote assets."""
        parts = symbol.split("/")
        if len(parts) == 2:
            return parts[0].upper(), parts[1].upper()
        raise ValueError(f"Invalid symbol format: {symbol}. Expected 'BASE/QUOTE'.")

    async def _run_periodic_reconciliation(self) -> None:
        """Periodically reconciles internal state with the exchange."""
        while True:
            try:
                await asyncio.sleep(self._reconciliation_interval)
                self.logger.info(
                    "Running periodic exchange reconciliation...",
                    source_module=self.__class__.__name__,
                )
                await self._reconcile_with_exchange()
            except asyncio.CancelledError:
                self.logger.info(
                    "Reconciliation loop cancelled.",
                    source_module=self.__class__.__name__,
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Error in reconciliation loop: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )
                # Avoid tight loop on error, wait before retrying
                await asyncio.sleep(self._reconciliation_interval)

    def _execution_handler_available_for_reconciliation(self) -> bool:
        """Checks if the execution handler is available and has necessary methods."""
        if (
            not self._execution_handler
            or not hasattr(self._execution_handler, "get_account_balances")
            or not hasattr(self._execution_handler, "get_open_positions")
        ):
            self.logger.warning(
                "Execution handler not available or missing required methods for reconciliation.",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    def _reconcile_single_balance(self, currency: str, exchange_bal: Decimal) -> bool:
        """Reconciles a single currency balance. Returns True if discrepancy found."""
        internal_bal = self._available_funds.get(currency, Decimal(0))
        diff = abs(internal_bal - exchange_bal)

        if diff > self._reconciliation_threshold:
            self.logger.warning(
                f"Reconciliation: Balance mismatch for {currency}. "
                f"Internal={internal_bal:.8f}, Exchange={exchange_bal:.8f}, Diff={diff:.8f}",
                source_module=self.__class__.__name__,
            )
            if self._auto_reconcile:
                self.logger.info(
                    f"Auto-reconciling {currency} balance to exchange value: {exchange_bal:.8f}",
                    source_module=self.__class__.__name__,
                )
                self._available_funds[currency] = exchange_bal
            return True
        return False

    def _reconcile_all_balances(self, exchange_balances: Dict[str, Decimal]) -> bool:
        """Reconciles all currency balances. Returns True if any discrepancy found."""
        discrepancies_found = False
        all_currencies = set(self._available_funds.keys()) | set(exchange_balances.keys())
        for currency in all_currencies:
            exchange_balance_for_currency = exchange_balances.get(currency, Decimal(0))
            if self._reconcile_single_balance(currency, exchange_balance_for_currency):
                discrepancies_found = True
        return discrepancies_found

    def _reconcile_single_position(self, pair: str, exchange_pos: Optional[PositionInfo]) -> bool:
        """Reconciles a single asset position. Returns True if discrepancy found."""
        internal_pos = self._positions.get(pair)
        internal_qty = internal_pos.quantity if internal_pos else Decimal(0)
        exchange_qty = exchange_pos.quantity if exchange_pos else Decimal(0)
        qty_diff = abs(internal_qty - exchange_qty)
        qty_threshold = Decimal("1e-8")  # Small threshold for quantity differences

        if qty_diff > qty_threshold:
            self.logger.warning(
                f"Reconciliation: Position quantity mismatch for {pair}. "
                f"Internal={internal_qty:.8f}, Exchange={exchange_qty:.8f}, Diff={qty_diff:.8f}",
                source_module=self.__class__.__name__,
            )
            if self._auto_reconcile:
                self.logger.info(
                    f"Auto-reconciling {pair} position to exchange value.",
                    source_module=self.__class__.__name__,
                )
                if exchange_pos:
                    # Preserve realized P&L and trade history from internal state if available
                    realized_pnl = internal_pos.realized_pnl if internal_pos else Decimal(0)
                    trade_history = internal_pos.trade_history if internal_pos else []

                    # Apply to the exchange_pos object before assigning it
                    exchange_pos.realized_pnl = realized_pnl
                    exchange_pos.trade_history = trade_history
                    self._positions[pair] = exchange_pos
                elif internal_pos:  # Exchange shows no position, but internal one exists
                    self.logger.info(
                        f"Exchange shows no position for {pair}, removing internal position.",
                        source_module=self.__class__.__name__,
                    )
                    del self._positions[pair]
            return True
        return False

    def _reconcile_all_positions(self, exchange_positions: Dict[str, PositionInfo]) -> bool:
        """Reconciles all asset positions. Returns True if any discrepancy found."""
        discrepancies_found = False
        all_pairs = set(self._positions.keys()) | set(exchange_positions.keys())
        for pair in all_pairs:
            if self._reconcile_single_position(pair, exchange_positions.get(pair)):
                discrepancies_found = True
        return discrepancies_found

    async def _reconcile_with_exchange(self) -> None:
        """Fetches exchange state and compares/updates internal state."""
        if not self._execution_handler_available_for_reconciliation():
            return

        # At this point, _execution_handler_available_for_reconciliation has confirmed
        # that self._execution_handler is not None and has the required attributes.
        # We assert and cast to make this explicit for mypy.
        assert self._execution_handler is not None
        reconcilable_handler = cast(ReconcilableExecutionHandler, self._execution_handler)

        try:
            exchange_balances = await reconcilable_handler.get_account_balances()
            exchange_positions = await reconcilable_handler.get_open_positions()

            async with self._lock:
                discrepancies_in_balances = self._reconcile_all_balances(exchange_balances)
                discrepancies_in_positions = self._reconcile_all_positions(exchange_positions)

                discrepancies_found = discrepancies_in_balances or discrepancies_in_positions

                if not discrepancies_found:
                    self.logger.info(
                        "Reconciliation complete. No significant discrepancies found.",
                        source_module=self.__class__.__name__,
                    )
                elif self._auto_reconcile:
                    self.logger.info(
                        "Discrepancies found and auto-reconciled. Updating portfolio value.",
                        source_module=self.__class__.__name__,
                    )
                    await self._update_portfolio_value_async()
                else:  # Discrepancies found, but auto_reconcile is off
                    self.logger.warning(
                        "Reconciliation found discrepancies, auto-reconciliation is disabled. "
                        "Manual review advised.",
                        source_module=self.__class__.__name__,
                    )

        except Exception as e:
            self.logger.error(
                f"Error during exchange reconciliation: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )


# Example Usage (for testing purposes, remove in production)
async def example_usage() -> None:  # noqa: C901
    # --- Mocks & Setup ---
    from typing import Protocol, runtime_checkable

    # Define protocol classes that are compatible with the expected interfaces
    @runtime_checkable
    class ConfigManagerProtocol(Protocol):
        def get(self, key: str, default: Any = None) -> Any: ...
        def get_int(self, key: str, default: int = 0) -> int: ...
        def get_bool(self, key: str, default: bool = False) -> bool: ...

    @runtime_checkable
    class LoggerServiceProtocol(Protocol):  # Remove Generic[T]
        def info(self, msg: str, source_module: str = "?", **kwargs: Any) -> None: ...

        def debug(self, msg: str, source_module: str = "?", **kwargs: Any) -> None: ...

        def warning(self, msg: str, source_module: str = "?", **kwargs: Any) -> None: ...

        def error(self, msg: str, source_module: str = "?", **kwargs: Any) -> None: ...

        def critical(self, msg: str, source_module: str = "?", **kwargs: Any) -> None: ...

    @runtime_checkable
    class PubSubManagerProtocol(Protocol):
        def subscribe(self, event_type: EventType, handler: Callable) -> None: ...

        def unsubscribe(self, event_type: EventType, handler: Callable) -> bool: ...

        async def publish(self, event: Event) -> None: ...

    @runtime_checkable
    class MarketPriceServiceProtocol(Protocol):
        async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]: ...

        async def get_bid_ask_spread(
            self, trading_pair: str
        ) -> Optional[Tuple[Decimal, Decimal]]: ...

    @runtime_checkable
    class ExecutionHandlerProtocol(Protocol):
        async def get_account_balances(self) -> Dict[str, Decimal]: ...

        async def get_open_positions(self) -> Dict[str, PositionInfo]: ...

    # Import the actual types for type casting
    from .config_manager import ConfigManager
    from .logger_service import LoggerService
    from .core.pubsub import PubSubManager
    from .market_price_service import MarketPriceService
    from .execution_handler import ExecutionHandler

    class MockConfigManager:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "portfolio.initial_capital":
                return {"USD": 100000}
            if key == "portfolio.valuation_currency":
                return "USD"
            if key == "portfolio.reconciliation.threshold":
                return "0.01"  # 1% threshold
            return default

        def get_int(self, key: str, default: int = 0) -> int:
            if key == "portfolio.drawdown.daily_reset_hour_utc":
                return 0  # midnight UTC
            if key == "portfolio.drawdown.weekly_reset_day":
                return 0  # Monday
            if key == "portfolio.reconciliation.interval_seconds":
                return 3600  # hourly
            return default

        def get_bool(self, key: str, default: bool = False) -> bool:
            if key == "portfolio.reconciliation.auto_update":
                return True  # Auto-reconcile for demo
            return default

    class MockLoggerService:
        def info(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
            print(f"INFO [{source_module}]: {msg}")

        def debug(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
            print(f"DEBUG [{source_module}]: {msg}")

        def warning(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
            print(f"WARN [{source_module}]: {msg}")

        def error(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
            print(f"ERROR [{source_module}]: {msg}")

        def critical(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
            print(f"CRITICAL [{source_module}]: {msg}")

    class MockPubSubManager:
        def __init__(self, logger: Any) -> None:
            self._logger = logger
            self._subscriptions: Dict[EventType, List[Callable]] = defaultdict(list)

        async def publish(self, event: Event) -> None:
            print(f"MockPublish: {event}")
            # Extract event_type safely using getattr with EventType type
            # checking
            event_type = getattr(event, "event_type", None)
            if event_type is not None:
                handlers = self._subscriptions.get(event_type, [])
                for handler in handlers:
                    asyncio.create_task(handler(event))

        def subscribe(self, event_type: EventType, handler: Callable) -> None:
            print(f"MockSubscribe: {handler.__name__} to {event_type.name}")
            self._subscriptions[event_type].append(handler)

        def unsubscribe(self, event_type: EventType, handler: Callable) -> bool:
            print(f"MockUnsubscribe: {handler.__name__} from {event_type.name}")
            try:
                self._subscriptions[event_type].remove(handler)
                return True
            except ValueError:
                return False

    class MockMarketPriceService:
        async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
            if trading_pair == "BTC/USD":
                return Decimal("50000.0")
            return None

        async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
            return (Decimal("49999.0"), Decimal("50001.0"))

    class MockExecutionHandler:
        async def get_account_balances(self) -> Dict[str, Decimal]:
            # Simulate exchange balances - slightly different from internal
            return {
                "USD": Decimal("95000.0"),  # Internal has 95005.0 after commission
                "BTC": Decimal("0.1"),
            }

        async def get_open_positions(self) -> Dict[str, PositionInfo]:
            # Simulate exchange positions
            return {
                "BTC/USD": PositionInfo(
                    trading_pair="BTC/USD",
                    base_asset="BTC",
                    quote_asset="USD",
                    quantity=Decimal("0.1"),
                    average_entry_price=Decimal("50000.0"),
                )
            }

    # --- Import event classes for the example function ---
    from .core.events import EventType, ExecutionReportEvent

    # --- Initialization ---
    # Create mock instances and explicitly cast them to the required types
    mock_config = MockConfigManager()
    mock_logger = MockLoggerService()
    mock_pubsub = MockPubSubManager(logger=mock_logger)
    mock_market_price = MockMarketPriceService()
    mock_execution_handler = MockExecutionHandler()

    # Use type casts to satisfy the type checker
    portfolio_manager = PortfolioManager(
        config_manager=cast(ConfigManager, mock_config),
        pubsub_manager=cast(PubSubManager, mock_pubsub),
        market_price_service=cast(MarketPriceService, mock_market_price),
        logger_service=cast(LoggerService[Any], mock_logger),
        execution_handler=cast(ExecutionHandler, mock_execution_handler),
    )

    # --- Test Execution --- #
    await portfolio_manager.start()

    # Simulate an execution report event
    exec_event = ExecutionReportEvent(
        source_module="MockExecHandler",
        event_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        exchange_order_id="ORDER123",
        trading_pair="BTC/USD",
        exchange="SIM",
        order_status="FILLED",
        order_type="MARKET",
        side="BUY",
        quantity_ordered=Decimal("0.1"),
        quantity_filled=Decimal("0.1"),
        average_fill_price=Decimal("50000.0"),
        commission=Decimal("5.0"),
        commission_asset="USD",
        timestamp_exchange=datetime.utcnow(),
    )

    await mock_pubsub.publish(exec_event)

    await asyncio.sleep(0.1)  # Allow time for handler to process

    # --- Check State ---
    current_state = portfolio_manager.get_current_state()
    print("\nCurrent Portfolio State:")
    print(json.dumps(current_state, indent=2, default=str))

    # Force a reconciliation (would normally happen on a timer)
    print("\nForcing reconciliation with exchange...")
    await portfolio_manager._reconcile_with_exchange()

    # Check state after reconciliation
    current_state_after = portfolio_manager.get_current_state()
    print("\nPortfolio State After Reconciliation:")
    print(json.dumps(current_state_after, indent=2, default=str))

    await portfolio_manager.stop()


if __name__ == "__main__":
    # asyncio.run(example_usage())
    pass
