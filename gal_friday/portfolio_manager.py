"""Portfolio Manager for maintaining comprehensive portfolio state tracking.

This module orchestrates the FundsManager, PositionManager, and ValuationService
to provide a unified interface for portfolio operations, state tracking, and
exchange reconciliation.
"""

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import asyncio

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from .execution_handler import ExecutionHandler

from .config_manager import ConfigManager
from .core.events import EventType, ExecutionReportEvent
from .core.pubsub import PubSubManager
from .dal.models.position import Position as PositionModel
from .exceptions import (
    DataValidationError,
    InsufficientFundsError,
    PriceNotAvailableError,
)
from .interfaces import MarketPriceService
from .logger_service import LoggerService
from .portfolio.funds_manager import FundsManager, TradeParams
from .portfolio.position_manager import PositionManager
from .portfolio.trade_history_service import TradeHistoryService
from .portfolio.valuation_service import (
    PositionInput,  # Added import
    ValuationService,
)

# Set Decimal precision
getcontext().prec = 28

# Define a protocol for the methods expected by _reconcile_with_exchange
@runtime_checkable
class ReconcilableExecutionHandler(Protocol):
    """Protocol for execution handlers that support account reconciliation."""

    async def get_account_balances(self) -> dict[str, Decimal]:
        """Retrieve account balances from exchange.

        Returns:
        -------
            Dictionary of currency to available balance amount
        """
        ...

    async def get_open_positions(self) -> dict[str, PositionModel]:
        """Retrieve open positions from exchange.

        Returns:
        -------
            Dictionary of trading pair to position information
        """
        ...


class PortfolioManager:
    """Coordinate portfolio state updates and provide state snapshots.

    Orchestrates FundsManager, PositionManager, and ValuationService.
    Consumes ExecutionReportEvents to update the portfolio state.
    Provides a synchronous method to query the current portfolio state snapshot.
    Handles periodic reconciliation with the exchange.
    """

    def __init__(
        self,
        *,  # Force keyword arguments for better readability
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        market_price_service: MarketPriceService,
        logger_service: LoggerService,
        execution_handler: "ExecutionHandler | None" = None,
        session_maker: "async_sessionmaker[AsyncSession]", # Added session_maker
    ) -> None:
        """Initialize the PortfolioManager with required dependencies.

        Args:
        ----
            config_manager: Configuration manager for system settings
            pubsub_manager: Publisher/subscriber manager for event handling
            market_price_service: Service for retrieving market prices
            logger_service: Logging service for system messages
            execution_handler: Optional handler for exchange execution
        """
        self.logger = logger_service
        self.config_manager = config_manager
        self.pubsub = pubsub_manager
        self.market_price_service = market_price_service
        self._execution_handler = execution_handler
        self._source_module = self.__class__.__name__

        self.valuation_currency = self.config_manager.get("portfolio.valuation_currency", "USD")

        # Instantiate components
        self.funds_manager = FundsManager(logger_service, self.valuation_currency)
        self.position_manager = PositionManager(
            logger_service, session_maker, config_manager,
        )  # Added session_maker and config_manager
        self.valuation_service = ValuationService(
            logger_service,
            market_price_service,
            self.valuation_currency)
        self.trade_history_service = TradeHistoryService(
            session_maker,
            logger_service,
            cache_size=config_manager.get_int("portfolio.trade_history.cache_size", 500),
            cache_ttl_seconds=config_manager.get_int("portfolio.trade_history.cache_ttl_seconds", 300))

        # --- Load Initial State & Config ---
        self._initialization_task = asyncio.create_task(self._initialize_state())
        self._configure_reconciliation()
        self._configure_drawdown_resets()

        # --- Internal State & Cache ---
        self._lock = asyncio.Lock()  # Lock for managing cached state updates
        self._cached_positions: dict[str, PositionModel] = {} # Cache for positions
        self._reconciliation_task: asyncio.Task[Any] | None = None
        self._execution_report_handler: None | (
            Callable[..., Coroutine[Any, Any, Any]]
        ) = None

        # Cache for the latest state snapshot (used by get_current_state)
        self._last_known_prices: dict[str, Decimal] = {}
        self._last_state_update_time: datetime | None = None
        self._last_total_exposure_pct: Decimal = Decimal(0)

        self.logger.info("PortfolioManager initialized.", source_module=self._source_module)
        self.logger.info(
            "Valuation Currency: %s",
            self.valuation_currency,
            source_module=self._source_module)

    async def _initialize_state(self) -> None:
        """Initialize funds and positions from configuration.

        Loads initial capital and positions from config, sets up the portfolio
        valuation, and logs the initial state.
        """
        try:
            initial_capital = self.config_manager.get("portfolio.initial_capital", {})
            await self.funds_manager.initialize_funds(initial_capital)

            self.config_manager.get("portfolio.initial_positions", {})
            await self.position_manager.initialize_positions() # Removed arguments

            # Perform initial valuation
            await self._update_portfolio_value_and_cache()

            self.logger.info(
                "Initial Available Funds: %s",
                self.funds_manager.available_funds,
                source_module=self._source_module)
            initial_db_positions = await self.position_manager.get_open_positions()
            async with self._lock:
                self._cached_positions = {pos.trading_pair: pos for pos in initial_db_positions}

            self.logger.info(
                "Initial Positions: %s",
                {
                    p: {"qty": str(info.quantity), "aep": str(info.entry_price)}
                    for p, info in self._cached_positions.items()
                },
                source_module=self._source_module)
            self.logger.info(
                "Initial Equity: %.2f %s",
                self.valuation_service.total_equity,
                self.valuation_currency,
                source_module=self._source_module)

        except Exception as e:
            self.logger.critical(
                "CRITICAL: Failed to initialize portfolio state:",
                source_module=self._source_module,
                exc_info=True)
            # Consider raising a specific exception or halting
            raise RuntimeError from e

    def _configure_reconciliation(self) -> None:
        """Load reconciliation configuration from ConfigManager.

        Sets up parameters for comparing internal state with exchange state
        and whether to auto-reconcile differences.
        """
        try:
            self._reconciliation_interval = self.config_manager.get_int(
                "portfolio.reconciliation.interval_seconds",
                3600)
            self._reconciliation_threshold = Decimal(
                self.config_manager.get("portfolio.reconciliation.threshold", "0.01"))
            self._auto_reconcile = self.config_manager.get_bool(
                "portfolio.reconciliation.auto_update",
                default=False)
            self.logger.info("Reconciliation configured.", source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error loading reconciliation config. Using defaults.",
                source_module=self._source_module)
            self._reconciliation_interval = 3600
            self._reconciliation_threshold = Decimal("0.01")
            self._auto_reconcile = False

    def _configure_drawdown_resets(self) -> None:
        """Configure drawdown reset times in the ValuationService.

        Sets up when daily and weekly drawdowns should be reset based on
        configuration parameters.
        """
        try:
            daily_reset_hour = self.config_manager.get_int(
                "portfolio.drawdown.daily_reset_hour_utc",
                0)
            weekly_reset_day = self.config_manager.get_int(
                "portfolio.drawdown.weekly_reset_day",
                0)
            self.valuation_service.configure_drawdown_resets(daily_reset_hour, weekly_reset_day)
        except ValueError:
            self.logger.exception(
                "Invalid drawdown reset config. Using defaults (0, 0).",
                source_module=self._source_module)
            self.valuation_service.configure_drawdown_resets(0, 0)
        except Exception:
            self.logger.exception(
                "Error configuring drawdown resets. Using defaults.",
                source_module=self._source_module)
            self.valuation_service.configure_drawdown_resets(0, 0)  # Ensure defaults are set

    async def start(self) -> None:
        """Subscribe to events and start background tasks.

        Sets up event handlers and starts reconciliation tasks if configured.
        """
        if self._execution_report_handler:
            self.logger.warning(
                "PortfolioManager already started.",
                source_module=self._source_module)
            return

        self._execution_report_handler = self._handle_execution_report
        self.pubsub.subscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
        self.logger.info(
            "Subscribed to EXECUTION_REPORT events.",
            source_module=self._source_module)

        if self._reconciliation_interval > 0 and self._execution_handler is not None:
            self.logger.info(
                "Starting periodic reconciliation every %ss.",
                self._reconciliation_interval,
                source_module=self._source_module)
            self._reconciliation_task = asyncio.create_task(self._run_periodic_reconciliation())
        # Log reasons for not starting reconciliation
        elif self._reconciliation_interval <= 0:
            self.logger.info(
                "Reconciliation disabled by interval config.",
                source_module=self._source_module)
        elif self._execution_handler is None:
            self.logger.warning(
                "Reconciliation disabled (no execution handler).",
                source_module=self._source_module)

    async def stop(self) -> None:
        """Unsubscribe from events and stop background tasks.

        Cleans up event subscriptions and stops reconciliation tasks.
        """
        # Unsubscribe from execution reports
        if self._execution_report_handler:
            try:
                self.pubsub.unsubscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
                self.logger.info(
                    "Unsubscribed from EXECUTION_REPORT events.",
                    source_module=self._source_module)
                self._execution_report_handler = None
            except Exception:
                self.logger.exception(
                    "Error unsubscribing from EXECUTION_REPORT:",
                    source_module=self._source_module)

        # Stop reconciliation task if running
        if self._reconciliation_task and not self._reconciliation_task.done():
            self.logger.info("Stopping reconciliation task...", source_module=self._source_module)
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
                self.logger.info(
                    "Reconciliation task cancelled.",
                    source_module=self._source_module)
            except Exception:
                self.logger.exception(
                    "Error stopping reconciliation task:",
                    source_module=self._source_module)
            finally:
                self._reconciliation_task = None

    async def _handle_execution_report(self, event: ExecutionReportEvent) -> None:
        """Process incoming execution report events.

        Updates portfolio state based on trade execution reports from the exchange.

        Args:
        ----
            event: The execution report event containing trade details
        """
        # In runtime, the event might not be exactly ExecutionReportEvent due to
        # differences between the placeholder class and actual implementation
        if not hasattr(event, "order_status") or not hasattr(event, "exchange_order_id"):
            self.logger.warning(
                "Received event missing required attributes: %s",
                type(event),
                source_module=self._source_module)
            return

        self.logger.debug(
            "Handling exec report: %s - %s",
            event.exchange_order_id,
            event.order_status,
            source_module=self._source_module)

        if event.order_status == "CANCELED":
            self._handle_order_cancellation(event)
            return

        if event.order_status not in ["FILLED", "PARTIALLY_FILLED"]:
            self.logger.debug(
                "Ignoring exec report status: %s",
                event.order_status,
                source_module=self._source_module)
            return

        try:
            # 1. Parse and Validate Event Data
            (
                pair,
                side,
                qty_filled,
                avg_price,
                commission,
                commission_asset) = self._parse_execution_values(event)
            base_asset, quote_asset = self._split_symbol(pair)
            cost_or_proceeds = qty_filled * avg_price

            # 2. Update Funds (FundsManager)
            await self.funds_manager.update_funds_for_trade(
                TradeParams(
                    base_asset=base_asset,
                    quote_asset=quote_asset,
                    side=side,
                    quantity=qty_filled,
                    price=avg_price,
                    cost_or_proceeds=cost_or_proceeds))

            # 3. Update Position (PositionManager)
            # Store and log the realized PNL from the position update
            pnl, updated_pos_model = await self.position_manager.update_position_for_trade(
                trading_pair=pair,
                side=side,
                quantity=qty_filled,
                price=avg_price,
                timestamp=event.timestamp,
                trade_id=event.exchange_order_id,
                order_id=event.client_order_id or event.exchange_order_id,
                commission=commission,
                commission_asset=commission_asset)

            # Update cache with the result from PositionManager
            if updated_pos_model:
                async with self._lock:
                    if updated_pos_model.is_active and updated_pos_model.quantity != Decimal(0):
                        self._cached_positions[updated_pos_model.trading_pair] = updated_pos_model
                    else: # Position closed or inactive
                        self._cached_positions.pop(updated_pos_model.trading_pair, None)
            elif event.order_status in ["FILLED", "PARTIALLY_FILLED"]:
                # If update failed but should have happened
                self.logger.warning(
                    f"Position model for {event.trading_pair} was not updated in cache "
                    f"after trade {event.exchange_order_id}.",
                    source_module=self._source_module)

            if pnl != Decimal(0):
                self.logger.info(
                    "Realized PNL from trade: %s",
                    pnl,
                    source_module=self._source_module)

            # 4. Handle Commission (FundsManager)
            if commission > 0 and commission_asset:
                await self.funds_manager.handle_commission(commission, commission_asset)

            # 5. Update Portfolio Value (ValuationService) and Cache State
            await self._update_portfolio_value_and_cache()

            # 6. Log Updated State
            self._log_updated_state()

        except (DataValidationError, InsufficientFundsError):
            self.logger.exception(
                "Data/Funds error processing execution report %s:",
                event.exchange_order_id,
                source_module=self._source_module)
        except ValueError:  # Catch specific parsing/symbol errors
            self.logger.exception(
                "Value error processing execution report %s:",
                event.exchange_order_id,
                source_module=self._source_module)
        except Exception:  # Catch all other unexpected errors
            self.logger.exception(
                "Unexpected error processing execution report %s:",
                event.exchange_order_id,
                source_module=self._source_module)

    def _handle_order_cancellation(self, event: ExecutionReportEvent) -> None:
        """Handle cancellation of an order.

        Logs the cancellation event for tracking purposes.

        Args:
        ----
            event: The execution report event with cancellation details
        """
        self.logger.info(
            "Order %s for %s was cancelled.",
            event.exchange_order_id,
            event.trading_pair,
            source_module=self._source_module)
        # Optional: Could track pending orders locally if needed

    def _raise_for_invalid_parsed_values(self) -> None:
        """Raise ValueError for invalid parsed execution values."""
        raise ValueError

    def _parse_execution_values(
        self,
        event: ExecutionReportEvent) -> tuple[str, str, Decimal, Decimal, Decimal, str | None]:
        """Parse and validate numeric values from the execution report.

        Extracts and validates key trade data from execution reports.

        Args:
        ----
            event: The execution report event to parse

        Returns:
        -------
            Tuple containing pair, side, quantity, price, commission and commission asset

        Raises:
        ------
            ValueError: If quantity or price values are invalid
        """
        try:
            pair = event.trading_pair
            side = event.side.upper()
            qty_filled = Decimal(str(event.quantity_filled))
            avg_price = Decimal(str(event.average_fill_price))
            commission = Decimal(str(event.commission or 0))
            commission_asset = event.commission_asset.upper() if event.commission_asset else None

            # Check for the valid condition first and return if met
            if qty_filled > 0 and avg_price > 0:
                return pair, side, qty_filled, avg_price, commission, commission_asset

            # If the above condition wasn't met (i.e., values are not valid), call helper to raise.
            self._raise_for_invalid_parsed_values()
            # The helper function will always raise, so code execution stops here for this path.

        except (TypeError, ValueError, ArithmeticError) as e:
            raise ValueError from e

        # This line should never be reached due to the exceptions above,
        # but we add it to satisfy the type checker
        raise ValueError

    async def _update_portfolio_value_and_cache(self) -> None:
        """Trigger valuation update and cache the results.

        Updates portfolio valuation and stores results for synchronous access.
        """
        try:
            current_funds = self.funds_manager.available_funds
            # Fetch fresh positions from DB for valuation
            db_positions_list = await self.position_manager.get_open_positions()
            current_positions_dict = {pos.trading_pair: pos for pos in db_positions_list}

            _, latest_prices, exposure_pct = await self.valuation_service.update_portfolio_value(
                current_funds,
                cast("dict[str, PositionInput]", current_positions_dict), # Refined cast
            )

            # Update local cache for get_current_state
            async with self._lock:
                self._last_known_prices = latest_prices
                self._last_total_exposure_pct = exposure_pct
                self._last_state_update_time = datetime.now(UTC)

        except PriceNotAvailableError:
            self.logger.exception(
                "Valuation failed due to missing prices:",
                source_module=self._source_module)
            # Keep old cached values, but log the error
        except Exception:
            self.logger.exception(
                "Error updating portfolio value:",
                source_module=self._source_module)

    def _log_updated_state(self) -> None:
        """Log the updated portfolio state after changes.

        Provides a detailed log of funds, positions, and valuation metrics
        after state changes occur.
        """
        # Get data from managers
        funds_str = {
            k: str(v.quantize(Decimal("0.0001")))
            for k, v in self.funds_manager.available_funds.items()
        }
        positions_str = {}
        # Create a shallow copy for safe iteration, as self._cached_positions might be updated elsewhere
        # by async methods. This is a synchronous method.
        cached_positions_copy = self._cached_positions.copy()
        for pair, pos in cached_positions_copy.items(): # Use cache
            positions_str[pair] = (
                f"Qty={pos.quantity.quantize(Decimal('1e-8'))}, "
                f"AvgPx={pos.entry_price.quantize(Decimal('0.0001'))}" # Changed to entry_price
            )
        equity = self.valuation_service.total_equity
        peak_equity = self.valuation_service.peak_equity
        total_dd = self.valuation_service.total_drawdown_pct
        daily_dd = self.valuation_service.daily_drawdown_pct
        weekly_dd = self.valuation_service.weekly_drawdown_pct

        self.logger.info(
            "State Update: Funds=%s, Positions=%s",
            funds_str,
            positions_str,
            source_module=self._source_module)
        self.logger.info(
            "Valuation: Equity=%.2f %s, Peak=%.2f, DD(Total=%.2f%%, Daily=%.2f%%, Weekly=%.2f%%)",
            equity,
            self.valuation_currency,
            peak_equity,
            total_dd,
            daily_dd,
            weekly_dd,
            source_module=self._source_module)

    # --- State Retrieval ---

    def get_current_state(self) -> dict[str, Any]:
        """Return the latest known portfolio state snapshot using cached data.

        This is a synchronous method that provides a snapshot of the portfolio
        including positions, funds, and valuation metrics.

        Returns:
        -------
            Dictionary containing comprehensive portfolio state information
        """
        self.logger.debug("get_current_state called.", source_module=self._source_module)

        # Note: This method intentionally uses cached values updated by
        # _update_portfolio_value_and_cache to remain synchronous.

        positions_dict = {}
        # Create a shallow copy for safe iteration if self._cached_positions can be modified by async code
        cached_positions_copy = self._cached_positions.copy()
        for pair, pos_model in cached_positions_copy.items(): # Use cache
            if pos_model.quantity != 0:  # Only include open positions
                # Use cached price for market value/unrealized PNL
                latest_price = self._last_known_prices.get(pair)
                market_value = None
                unrealized_pnl = None

                if latest_price is not None:
                    market_value = pos_model.quantity * latest_price
                    if (
                        pos_model.entry_price > 0 or pos_model.quantity != 0 # Changed to entry_price
                    ):  # Avoid PNL calc on zero avg price unless qty exists
                        unrealized_pnl = (
                            latest_price - pos_model.entry_price # Changed to entry_price
                        ) * pos_model.quantity

                base_asset, quote_asset = self._split_symbol(pair)
                positions_dict[pair] = {
                    "base_asset": base_asset,
                    "quote_asset": quote_asset,
                    "quantity": str(pos_model.quantity),
                    "average_entry_price": str(pos_model.entry_price), # Changed to entry_price
                    "current_market_value": (
                        str(market_value) if market_value is not None else None
                    ),
                    "unrealized_pnl": str(unrealized_pnl) if unrealized_pnl is not None else None,
                    "realized_pnl": str(pos_model.realized_pnl),
                    "trade_count": 0, # PositionModel does not have trade_history, using placeholder
                }

        # Get latest values from ValuationService and FundsManager
        # Note: Using internal attributes directly for sync access relies on them
        # being updated correctly by async methods. Consider locks if needed,
        # but for reads, it might be acceptable if staleness is tolerated.
        current_equity = self.valuation_service.total_equity
        daily_dd = self.valuation_service.daily_drawdown_pct
        weekly_dd = self.valuation_service.weekly_drawdown_pct
        total_dd = self.valuation_service.total_drawdown_pct
        available_funds = self.funds_manager.available_funds

        # Use cached exposure percentage
        total_exposure_pct = self._last_total_exposure_pct

        return {
            "timestamp": (
                self._last_state_update_time.isoformat() + "Z"
                if self._last_state_update_time
                else datetime.now(UTC).isoformat() + "Z"  # Fallback timestamp
            ),
            "valuation_currency": self.valuation_currency,
            "total_equity": str(current_equity),
            "available_funds": {k: str(v) for k, v in available_funds.items()},
            "positions": positions_dict,
            "total_exposure_pct": str(total_exposure_pct),
            "daily_drawdown_pct": str(daily_dd),
            "weekly_drawdown_pct": str(weekly_dd),
            "total_drawdown_pct": str(total_dd),
        }

    # --- Utility and Delegate Methods ---

    def get_available_funds(self, currency: str) -> Decimal:
        """Return the available funds for a specific currency.

        Args:
        ----
            currency: The currency code to check funds for

        Returns:
        -------
            The available amount of the specified currency
        """
        # Delegate to FundsManager (synchronous access is okay for read)
        return self.funds_manager.available_funds.get(currency.upper(), Decimal(0))

    def get_current_equity(self) -> Decimal:
        """Return the current equity value.

        Returns:
        -------
            The total portfolio equity in the valuation currency
        """
        # Delegate to ValuationService (synchronous access okay for read)
        return self.valuation_service.total_equity

    async def get_position_history(
        self,
        pair: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
        offset: int = 0) -> list[dict[str, Any]]:
        """Return the trade history for a specific pair.

        Retrieves actual trade history from the database using the TradeHistoryService.
        Provides comprehensive filtering, pagination, and caching capabilities.

        Args:
        ----
            pair: Trading pair to get history for
            start_date: Optional start date filter for trade history
            end_date: Optional end date filter for trade history
            limit: Maximum number of trades to return (default: 1000)
            offset: Number of trades to skip for pagination (default: 0)

        Returns:
        -------
            List of historical trades for the specified pair

        Raises:
        ------
            DataValidationError: If request parameters are invalid
        """
        try:
            # Check if position exists (optional validation)
            position_model = await self.position_manager.get_position(pair)
            if not position_model:
                self.logger.debug(
                    f"No active position found for {pair}, but retrieving trade history anyway",
                    source_module=self._source_module)

            # Use TradeHistoryService to get actual trade history
            trade_history = await self.trade_history_service.get_trade_history_for_pair(
                trading_pair=pair,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset)

            self.logger.info(
                f"Retrieved {len(trade_history)} trades for {pair} "
                f"(limit: {limit}, offset: {offset})",
                source_module=self._source_module)

        except Exception:
            self.logger.exception(
                f"Error retrieving trade history for {pair}: ",
                source_module=self._source_module)
            # Return empty list[Any] on error to maintain API compatibility
            return []
        else:
            return trade_history

    async def get_trade_analytics(
        self,
        pair: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None) -> dict[str, Any]:
        """Get comprehensive trade analytics for a specific trading pair.

        Provides aggregated statistics including volume, commission costs,
        trade counts, and performance metrics.

        Args:
        ----
            pair: Trading pair to analyze
            start_date: Optional start date filter for analytics
            end_date: Optional end date filter for analytics

        Returns:
        -------
            Dictionary containing comprehensive trade analytics

        Raises:
        ------
            DataValidationError: If request parameters are invalid
        """
        try:
            analytics = await self.trade_history_service.get_analytics_summary(
                trading_pair=pair,
                start_date=start_date,
                end_date=end_date)

            self.logger.info(
                f"Generated trade analytics for {pair}: {analytics['total_trades']} trades, "
                f"volume: {analytics['total_volume']}",
                source_module=self._source_module)

        except Exception as e:
            self.logger.exception(
                f"Error generating trade analytics for {pair}: ",
                source_module=self._source_module)
            # Return empty analytics on error
            return {
                "total_trades": 0,
                "total_volume": "0",
                "total_commission": "0",
                "avg_trade_size": "0",
                "buy_trades": 0,
                "sell_trades": 0,
                "error": str(e),
            }
        else:
            return analytics

    async def clear_trade_history_cache(self) -> None:
        """Clear the trade history cache.

        Useful for troubleshooting or when fresh data is required.
        """
        try:
            await self.trade_history_service.clear_cache()
            self.logger.info(
                "Trade history cache cleared successfully",
                source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error clearing trade history cache: ",
                source_module=self._source_module)

    def get_trade_history_cache_stats(self) -> dict[str, Any]:
        """Get trade history cache performance statistics.

        Returns:
        -------
            Dictionary containing cache performance metrics
        """
        return self.trade_history_service.get_cache_stats()

    async def get_open_positions(self) -> list[PositionModel]:
        """Return a list[Any] of open positions.

        Returns:
        -------
            List of current open position information objects
        """
        # Since we're making this method async and the position manager likely has an async method
        # we can properly await it. If position_manager.get_open_positions is sync, we'd need to adjust.
        # Based on the file structure, this is likely async
        return list[Any](await self.position_manager.get_open_positions())

    EXPECTED_SYMBOL_PARTS = 2

    def _split_symbol(self, symbol: str) -> tuple[str, str]:
        """Split a trading symbol into base and quote assets.

        Args:
        ----
            symbol: Trading symbol in format 'BASE/QUOTE'

        Returns:
        -------
            Tuple of (base_asset, quote_asset)

        Raises:
        ------
            ValueError: If symbol format is invalid
        """
        # Keep this utility method local or move to a shared utils module
        parts = symbol.split("/")
        if len(parts) == self.EXPECTED_SYMBOL_PARTS:
            return parts[0].upper(), parts[1].upper()
        raise ValueError

    # --- Reconciliation ---

    async def _run_periodic_reconciliation(self) -> None:
        """Periodically reconcile internal state with the exchange.

        Runs a background task that checks for discrepancies between internal
        portfolio state and exchange data at configured intervals.
        """
        while True:
            try:
                await asyncio.sleep(self._reconciliation_interval)
                self.logger.info(
                    "Running periodic exchange reconciliation...",
                    source_module=self._source_module)
                await self._reconcile_with_exchange()
            except asyncio.CancelledError:
                self.logger.info(
                    "Reconciliation loop cancelled.",
                    source_module=self._source_module)
                break
            except Exception:
                self.logger.exception(
                    "Error in reconciliation loop:",
                    source_module=self._source_module)
                await asyncio.sleep(self._reconciliation_interval)  # Avoid tight loop

    def _execution_handler_available_for_reconciliation(self) -> bool:
        """Check if the execution handler is suitable for reconciliation.

        Verifies the execution handler implements the required methods
        for account reconciliation.

        Returns:
        -------
            True if execution handler can be used for reconciliation
        """
        # Since we're using runtime_checkable Protocol, we need a more careful check
        # to prevent mypy unreachable code error
        if self._execution_handler is None:
            self.logger.warning(
                "No execution handler available for reconciliation.",
                source_module=self._source_module)
            return False

        # Check if the handler has the required methods
        has_get_balances = hasattr(self._execution_handler, "get_account_balances")
        has_get_positions = hasattr(self._execution_handler, "get_open_positions")

        if not (has_get_balances and has_get_positions):
            self.logger.warning(
                "Execution handler missing required methods for reconciliation.",
                source_module=self._source_module)
            return False

        return True

    async def _reconcile_with_exchange(self) -> None:
        """Fetch exchange state and compare/update internal state.

        Compares internal portfolio state with exchange data and
        reconciles differences based on configuration.
        """
        if not self._execution_handler_available_for_reconciliation():
            return

        reconcilable_handler = cast("ReconcilableExecutionHandler", self._execution_handler)

        try:
            # Fetch exchange state
            exchange_balances = await reconcilable_handler.get_account_balances()
            exchange_positions_raw = await reconcilable_handler.get_open_positions()
            # Ensure exchange_positions is Dict[str, PositionModel]
            exchange_positions = {
                pair: pos
                for pair, pos in exchange_positions_raw.items()
                if isinstance(pos, PositionModel)
            }

            # Get internal state
            internal_balances = self.funds_manager.available_funds
            # Get current positions from cache
            internal_positions = self._cached_positions.copy()

            discrepancies_found = False

            # Reconcile Balances
            bal_discrepancies = self._compare_balances(internal_balances, exchange_balances)
            if bal_discrepancies:
                discrepancies_found = True
                self.logger.warning(
                    "Reconciliation: Balance mismatches: %s",
                    bal_discrepancies,
                    source_module=self._source_module)
                if self._auto_reconcile:
                    await self._auto_reconcile_balances(exchange_balances)

            # Reconcile Positions
            pos_discrepancies = self._compare_positions(internal_positions, exchange_positions)
            if pos_discrepancies:
                discrepancies_found = True
                self.logger.warning(
                    "Reconciliation: Position mismatches: %s",
                    pos_discrepancies,
                    source_module=self._source_module)
                if self._auto_reconcile:
                    await self._auto_reconcile_positions(exchange_positions)

            # Log final status and update value if reconciled
            if not discrepancies_found:
                self.logger.info(
                    "Reconciliation complete. No significant discrepancies found.",
                    source_module=self._source_module)
            elif self._auto_reconcile:
                self.logger.info(
                    "Discrepancies found and auto-reconciled. Updating portfolio value.",
                    source_module=self._source_module)
                await self._update_portfolio_value_and_cache()
            else:
                self.logger.warning(
                    "Reconciliation found discrepancies; auto-reconcile is OFF.",
                    source_module=self._source_module)

        except Exception:
            self.logger.exception(
                "Error during exchange reconciliation:",
                source_module=self._source_module)

    async def _auto_reconcile_balances(self, exchange_balances: dict[str, Decimal]) -> None:
        """Auto-reconcile balances with exchange data.

        Updates internal fund balances to match exchange reported balances.

        Args:
        ----
            exchange_balances: Dictionary of currency to balance amount from exchange
        """
        self.logger.info("Auto-reconciling balances...", source_module=self._source_module)
        # Delegate to FundsManager
        await self.funds_manager.reconcile_with_exchange_balances(exchange_balances)
        self.logger.info("Balances auto-reconciled.", source_module=self._source_module)

    async def _auto_reconcile_positions(
        self,
        exchange_positions: dict[str, PositionModel]) -> None:
        """Auto-reconcile positions with exchange data.

        Updates internal positions to match exchange reported positions.

        Args:
        ----
            exchange_positions: Dictionary of positions reported by exchange
        """
        self.logger.info("Auto-reconciling positions...", source_module=self._source_module)
        # Delegate to PositionManager instead of implementing directly
        await self._reconcile_positions_with_exchange(exchange_positions)
        self.logger.info("Positions auto-reconciled.", source_module=self._source_module)

    async def _reconcile_positions_with_exchange(
        self,
        exchange_positions: dict[str, PositionModel]) -> None:
        """Reconcile positions with exchange data.

        Adjusts internal positions to match exchange positions by creating
        reconciliation trades as needed.

        Args:
        ----
            exchange_positions: Dictionary of positions reported by exchange
        """
        # First, iterate through exchange positions
        for pair, ex_pos in exchange_positions.items():
            int_pos = await self.position_manager.get_position(pair)
            if int_pos:
                # Position exists in both - check if quantities match
                if int_pos.quantity != ex_pos.quantity:
                    self.logger.info(
                        "Adjusting position for %s: %s → %s",
                        pair,
                        int_pos.quantity,
                        ex_pos.quantity,
                        source_module=self._source_module)
                    base_asset, quote_asset = self._split_symbol(pair)
                    # Create a reconciliation trade to adjust the quantity
                    await self._create_reconciliation_trade(
                        pair,
                        base_asset,
                        quote_asset,
                        int_pos,
                        ex_pos)
            else:
                # Position exists on exchange but not internally - create it
                base_asset, quote_asset = self._split_symbol(pair)
                self.logger.info(
                    "Creating missing position for %s with qty %s",
                    pair,
                    ex_pos.quantity,
                    source_module=self._source_module)
                # Create new position through a mock trade
                await self._create_position_from_exchange(pair, base_asset, quote_asset, ex_pos)

        # Now check for positions that exist internally but not on exchange
        internal_positions = self._cached_positions.copy()
        for pair, int_pos in internal_positions.items():
            if pair not in exchange_positions and int_pos.quantity != 0:
                self.logger.info(
                    "Closing position %s that doesn't exist on exchange",
                    pair,
                    source_module=self._source_module)
                base_asset, quote_asset = self._split_symbol(pair)
                # Create a reconciliation trade to close the position
                # Create a dummy PositionModel for target with 0 quantity
                dummy_pos = PositionModel(
                    trading_pair=pair,
                    quantity=Decimal(0),
                    entry_price=Decimal(0))
                await self._create_reconciliation_trade(
                    pair,
                    base_asset,
                    quote_asset,
                    int_pos,
                    dummy_pos)

    async def _create_reconciliation_trade(
        self,
        pair: str,
        base_asset: str,
        quote_asset: str,
        current_pos: PositionModel,
        target_pos: PositionModel) -> None:
        """Create a reconciliation trade to adjust position to match target.

        Generates a synthetic trade to adjust position quantity to match
        the exchange-reported quantity.

        Args:
        ----
            pair: Trading pair symbol
            base_asset: Base asset of the pair
            quote_asset: Quote asset of the pair
            current_pos: Current internal position
            target_pos: Target position from exchange
        """
        if current_pos.quantity == target_pos.quantity:
            return  # No adjustment needed

        # Calculate difference and direction
        diff_qty = target_pos.quantity - current_pos.quantity
        if diff_qty == 0:
            return

        side = "BUY" if diff_qty > 0 else "SELL"
        abs_qty = abs(diff_qty)

        # Use average entry price or a fallback price
        price = target_pos.entry_price
        if price <= 0:
            # Try to get current market price as fallback
            try:
                current_price = await self.market_price_service.get_latest_price(pair)
                if current_price is not None and current_price > 0:
                    price = current_price
                else:
                    # Last resort - use internal position's price or 1.0
                    price = (
                        current_pos.entry_price
                        if current_pos.entry_price > 0
                        else Decimal("1.0")
                    )
            except Exception:
                # If all else fails
                price = Decimal("1.0")

        # Calculate cost or proceeds
        cost = abs_qty * price

        # Create reconciliation trade
        timestamp = datetime.now(UTC)
        description = (
            f"Reconciliation trade to adjust {pair} from "
            f"{current_pos.quantity} to {target_pos.quantity}"
        )
        self.logger.info(description, source_module=self._source_module)

        # Use PositionManager to apply the adjustment
        await self.position_manager.update_position_for_trade(
            trading_pair=pair,
            side=side,
            quantity=abs_qty,
            price=price,
            timestamp=timestamp,
            trade_id="reconciliation")

        # Also update funds for the trade if needed
        await self.funds_manager.update_funds_for_trade(
            TradeParams(
                base_asset=base_asset,
                quote_asset=quote_asset,
                side=side,
                quantity=abs_qty,
                price=price,
                cost_or_proceeds=cost))

        # Update portfolio value
        await self._update_portfolio_value_and_cache()

    async def _create_position_from_exchange(
        self,
        pair: str,
        base_asset: str,
        quote_asset: str,
        exchange_pos: PositionModel) -> None:
        """Create a new position from exchange data.

        Generates a synthetic position to match exchange-reported position.

        Args:
        ----
            pair: Trading pair symbol
            base_asset: Base asset of the pair
            quote_asset: Quote asset of the pair
            exchange_pos: Position information from exchange
        """
        if exchange_pos.quantity == 0:
            return  # Don't create zero positions

        # Determine side based on position direction
        side = "BUY" if exchange_pos.quantity > 0 else "SELL"
        abs_qty = abs(exchange_pos.quantity)

        # Use exchange position's average entry price or fallback
        price = exchange_pos.entry_price
        if price <= 0:
            try:
                current_price = await self.market_price_service.get_latest_price(pair)
                price = current_price if current_price and current_price > 0 else Decimal("1.0")
            except Exception:
                price = Decimal("1.0")

        # Calculate cost
        cost = abs_qty * price

        # Create mock trade to establish the position
        timestamp = datetime.now(UTC)
        self.logger.info(
            "Creating position from exchange: %s %s %s @ %s",
            pair,
            side,
            abs_qty,
            price,
            source_module=self._source_module)

        # Use PositionManager to create the position
        await self.position_manager.update_position_for_trade(
            trading_pair=pair,
            side=side,
            quantity=abs_qty,
            price=price,
            timestamp=timestamp,
            trade_id="reconciliation")

        # Update funds to reflect the position
        await self.funds_manager.update_funds_for_trade(
            TradeParams(
                base_asset=base_asset,
                quote_asset=quote_asset,
                side=side,
                quantity=abs_qty,
                price=price,
                cost_or_proceeds=cost))

        # Update portfolio value
        await self._update_portfolio_value_and_cache()

    def _compare_balances(
        self,
        internal: dict[str, Decimal],
        exchange: dict[str, Decimal]) -> dict[str, dict[str, Decimal | str]]:
        """Compare internal and exchange balances, return discrepancies.

        Identifies balance differences that exceed configured thresholds.
        Uses both absolute and relative thresholds to better handle balances
        of different magnitudes.

        Args:
        ----
            internal: Dictionary of internal balances
            exchange: Dictionary of exchange-reported balances

        Returns:
        -------
            Dictionary of currencies with discrepancies
        """
        discrepancies = {}
        all_currencies = set(internal.keys()) | set(exchange.keys())

        # Define relative threshold for larger balances (0.1% = 0.001)
        relative_threshold = Decimal("0.001")

        for currency in all_currencies:
            internal_bal = internal.get(currency, Decimal(0))
            exchange_bal = exchange.get(currency, Decimal(0))
            diff = abs(internal_bal - exchange_bal)

            # Use the larger balance as the reference for relative threshold
            reference_balance = max(internal_bal, exchange_bal)

            # Calculate dynamic threshold: max of absolute threshold or relative threshold
            if reference_balance > 0:
                dynamic_threshold = max(
                    self._reconciliation_threshold,
                    reference_balance * relative_threshold)
            else:
                dynamic_threshold = self._reconciliation_threshold

            if diff > dynamic_threshold:
                discrepancies[currency] = {
                    "internal": internal_bal,
                    "exchange": exchange_bal,
                    "diff": diff,
                    "threshold_used": dynamic_threshold,
                    "threshold_type": "relative" if dynamic_threshold > self._reconciliation_threshold else "absolute",
                }
        return discrepancies

    def _compare_positions(
        self,
        internal: dict[str, PositionModel],
        exchange: dict[str, PositionModel]) -> dict[str, dict[str, Decimal]]:
        """Compare internal and exchange positions, return discrepancies.

        Identifies position differences that exceed configured thresholds.

        Args:
        ----
            internal: Dictionary of internal positions
            exchange: Dictionary of exchange-reported positions

        Returns:
        -------
            Dictionary of trading pairs with discrepancies
        """
        discrepancies = {}
        all_pairs = set(internal.keys()) | set(exchange.keys())
        qty_threshold = Decimal("1e-8")  # Threshold for quantity comparison

        for pair in all_pairs:
            internal_pos = internal.get(pair)
            exchange_pos = exchange.get(pair)
            internal_qty = internal_pos.quantity if internal_pos else Decimal(0)
            exchange_qty = exchange_pos.quantity if exchange_pos else Decimal(0)
            qty_diff = abs(internal_qty - exchange_qty)

            if qty_diff > qty_threshold:
                discrepancies[pair] = {
                    "internal_qty": internal_qty,
                    "exchange_qty": exchange_qty,
                    "diff": qty_diff,
                }

        return discrepancies
