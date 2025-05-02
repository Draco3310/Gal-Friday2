# Portfolio Manager Module

import logging
import json
from typing import Dict, Optional, Any, TYPE_CHECKING, Tuple, List, Callable, Coroutine, cast, TypeVar
from decimal import Decimal, getcontext
from dataclasses import dataclass
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
else:
    # Create placeholder classes for runtime
    from .core.placeholder_classes import (  # noqa: F401
        Event,
        EventType,
        ExecutionReportEvent,
        ConfigManager,
        PubSubManager,
        MarketPriceService,
    )


@dataclass
class PositionInfo:
    """Stores information about a specific asset position."""
    trading_pair: str
    base_asset: str
    quote_asset: str
    quantity: Decimal = Decimal(0)
    average_entry_price: Decimal = Decimal(0)
    # Add other fields if needed, e.g., unrealized_pnl, last_update_time


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
    ):
        self.logger = logger_service
        self.config_manager = config_manager
        self.pubsub = pubsub_manager
        self.market_price_service = market_price_service
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
        # Lock for potential future concurrent access (not strictly needed with single listener)
        self._lock = asyncio.Lock()
        # Store the execution report handler for unsubscribing
        self._execution_report_handler: Optional[Callable[[ExecutionReportEvent], Coroutine[Any, Any, Any]]] = None

        self.valuation_currency = self.config_manager.get("portfolio.valuation_currency", "USD")

        self._load_initial_state()

        # Initialize equity based on initial state
        self._update_portfolio_value()  # Initial calculation
        self._peak_equity = self._total_equity  # Start peak at initial equity

        self.logger.info("PortfolioManager initialized.", source_module=self.__class__.__name__)
        self.logger.info(
            f"Valuation Currency: {self.valuation_currency}", source_module=self.__class__.__name__
        )
        self.logger.info(
            f"Initial Available Funds: {self._available_funds}",
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            f"Initial Positions: {self._positions}", source_module=self.__class__.__name__
        )
        self.logger.info(
            f"Initial Equity: {self._total_equity:.2f} {self.valuation_currency}",
            source_module=self.__class__.__name__,
        )

    def _load_initial_state(self) -> None:
        """Loads initial portfolio state, prioritizing config."""
        # Load available funds (initial capital)
        initial_funds = self.config_manager.get("portfolio.initial_capital", {})
        self._available_funds = {k.upper(): Decimal(str(v)) for k, v in initial_funds.items()}

        # Ensure valuation currency is present in funds, default to 0 if not explicitly set
        if self.valuation_currency not in self._available_funds:
            self._available_funds[self.valuation_currency] = Decimal("0")

        # TODO: Load initial positions from config or exchange if needed (MVP starts flat)
        initial_positions_config = self.config_manager.get("portfolio.initial_positions", {})
        for pair, pos_data in initial_positions_config.items():
            try:
                base, quote = self._split_symbol(pair)
                self._positions[pair] = PositionInfo(
                    trading_pair=pair,
                    base_asset=base,
                    quote_asset=quote,
                    quantity=Decimal(str(pos_data.get("quantity", 0))),
                    average_entry_price = Decimal(str(pos_data.get("average_entry_price", 0)))
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
        """Subscribes to execution reports."""
        # Store the handler to enable unsubscribing later
        self._execution_report_handler = self._handle_execution_report
        
        # Subscribe to execution reports
        self.pubsub.subscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
        
        self.logger.info(
            "Subscribed to EXECUTION_REPORT events.", source_module=self.__class__.__name__
        )

    async def stop(self) -> None:
        """Unsubscribes from execution reports."""
        # Use the stored handler to unsubscribe
        if self._execution_report_handler is not None:
            try:
                self.pubsub.unsubscribe(
                    EventType.EXECUTION_REPORT, self._execution_report_handler
                )
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

    def _validate_execution_report(self, event: "ExecutionReportEvent") -> bool:
        """Validates the execution report event data."""
        if event.order_status not in ["FILLED", "PARTIALLY_FILLED"]:
            self.logger.debug(
                f"Ignoring execution report with status: {event.order_status}",
                source_module=self.__class__.__name__,
            )
            return False
        return True

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
        commission = (
            Decimal(str(event.commission))
            if event.commission is not None
            else Decimal(0)
        )
        commission_asset = (
            event.commission_asset.upper()
            if event.commission_asset
            else None
        )

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
        self._available_funds[quote_asset] = self._available_funds.get(
            quote_asset, Decimal(0)
        )
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
        cost_or_proceeds: Decimal,
    ) -> None:
        """Updates position information based on trade execution."""
        position = self._positions.get(pair)
        if position is None:
            position = PositionInfo(
                trading_pair=pair,
                base_asset=base_asset,
                quote_asset=quote_asset
            )
            self._positions[pair] = position

        current_quantity = position.quantity
        current_avg_price = position.average_entry_price

        if side == "BUY":
            new_total_cost = (current_quantity * current_avg_price) + cost_or_proceeds
            position.quantity += quantity_filled
            position.average_entry_price = (
                new_total_cost / position.quantity
                if position.quantity != 0
                else Decimal(0)
            )
        elif side == "SELL":
            position.quantity -= quantity_filled
            if position.quantity < 0:
                self.logger.warning(
                    (
                        f"Position quantity for {pair} went negative "
                        f"({position.quantity}) after sell. Reconciliation needed?"
                    ),
                    source_module=self.__class__.__name__,
                )
            if position.quantity == 0:
                position.average_entry_price = Decimal(0)

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
                (
                    f"Deducted commission {commission} {commission_asset} "
                    "from available funds."
                ),
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
                    source_module=self.__class__.__name__
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
                pair, side, qty_filled, avg_price, commission, commission_asset = self._parse_execution_values(event)
                
                # Extract base and quote assets
                base_asset, quote_asset = self._split_symbol(pair)
                
                # Calculate cost/proceeds in quote currency
                cost_or_proceeds = qty_filled * avg_price

                # Update state atomically using lock
                async with self._lock:
                    # Update funds
                    if not self._update_funds_for_trade(quote_asset, side, cost_or_proceeds, event.side):
                        return
                    
                    # Update position
                    self._update_position_for_trade(pair, base_asset, quote_asset, side, qty_filled, cost_or_proceeds)
                    
                    # Handle commission
                    self._handle_commission(commission, commission_asset)
                    
                    # Recalculate portfolio value
                    self._update_portfolio_value()
                    
                    # Log updated state
                    self._log_updated_state()

            except ValueError as e:
                self.logger.warning(
                    f"Invalid execution report values: {e}", 
                    source_module=self.__class__.__name__
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
            k: str(v.quantize(Decimal("0.0001")))
            for k, v in self._available_funds.items()
        }
        self.logger.info(
            f"State after trade: Funds={funds_str}, Positions={self._positions}",
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            (
                f"Current Equity: {self._total_equity:.2f} {self.valuation_currency}, "
                f"Peak Equity: {self._peak_equity:.2f}, "
                f"Drawdown: {self._total_drawdown_pct:.2f}%"
            ),
            source_module=self.__class__.__name__,
        )

    def _calculate_cash_value(self) -> Tuple[Decimal, bool]:
        """Calculates the value of cash balances in valuation currency."""
        current_total_value = Decimal(0)
        missing_prices = False

        for currency, amount in self._available_funds.items():
            if amount == 0:
                continue  # Skip zero balances

            if currency == self.valuation_currency:
                current_total_value += amount
            else:
                value, has_missing = self._convert_currency_value(currency, amount)
                current_total_value += value
                missing_prices = missing_prices or has_missing

        return current_total_value, missing_prices

    def _convert_currency_value(
        self, currency: str, amount: Decimal
    ) -> Tuple[Decimal, bool]:
        """Converts a currency amount to valuation currency."""
        # Construct pairs for conversion
        conversion_pair = f"{currency}/{self.valuation_currency}"
        inv_pair = f"{self.valuation_currency}/{currency}"

        # Try direct conversion
        conversion_rate = self._get_latest_price_sync(conversion_pair)
        if conversion_rate is not None and conversion_rate > 0:
            return amount * conversion_rate, False

        # Try inverse conversion
        inv_price = self._get_latest_price_sync(inv_pair)
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

    def _calculate_position_value(self) -> Tuple[Decimal, bool]:
        """Calculates the value of positions in valuation currency."""
        current_total_value = Decimal(0)
        missing_prices = False

        for pair, position in self._positions.items():
            if position.quantity == 0:
                continue  # Skip closed positions

            value, has_missing = self._calculate_single_position_value(pair, position)
            current_total_value += value
            missing_prices = missing_prices or has_missing

        return current_total_value, missing_prices

    def _calculate_single_position_value(
        self, pair: str, position: PositionInfo
    ) -> Tuple[Decimal, bool]:
        """Calculates the value of a single position in valuation currency."""
        market_price = self._get_latest_price_sync(pair)
        if market_price is None or market_price <= 0:
            self.logger.warning(
                f"Missing market price for position {pair}. Cannot accurately value.",
                source_module=self.__class__.__name__,
            )
            return Decimal(0), True

        market_value_base = position.quantity * market_price

        # If quote asset is valuation currency, return directly
        if position.quote_asset == self.valuation_currency:
            return market_value_base, False

        # Convert to valuation currency
        return self._convert_currency_value(
            position.quote_asset, market_value_base
        )

    def _get_latest_price_sync(self, pair: str) -> Optional[Decimal]:
        """Get the latest price synchronously from the market price service.
        
        This is a wrapper to handle the async->sync transition.
        In a real implementation, you'd want a non-blocking solution.
        """
        # Ensure we're actually getting a Decimal value, not a coroutine
        price = self.market_price_service.get_latest_price(pair)  # Not a coroutine, no await needed
        if isinstance(price, Decimal):
            return price
        elif price is not None:
            # Convert to Decimal if needed
            try:
                return Decimal(str(price))
            except (ValueError, TypeError):
                self.logger.error(
                    f"Invalid price format returned for {pair}: {price}",
                    source_module=self.__class__.__name__,
                )
        return None

    def _update_portfolio_value(self) -> None:
        """Recalculates the total portfolio value in the base currency,
        updates peak equity, and calculates drawdown.
        Uses the market_price_service to get latest prices.
        """
        # Calculate value of cash balances
        cash_value, missing_prices_cash = self._calculate_cash_value()

        # Calculate value of positions
        position_value, missing_prices_pos = self._calculate_position_value()

        # Total value and missing prices flag
        current_total_value = cash_value + position_value
        missing_prices = missing_prices_cash or missing_prices_pos

        if missing_prices:
            self.logger.error(
                "Could not calculate total portfolio value accurately due to missing prices.",
                source_module=self.__class__.__name__,
            )

        # Update total equity
        self._total_equity = current_total_value

        # Update peak equity and drawdown
        if self._total_equity > self._peak_equity:
            self._peak_equity = self._total_equity
            self._total_drawdown_pct = Decimal(0)
        elif self._peak_equity > 0:
            drawdown = (self._peak_equity - self._total_equity) / self._peak_equity
            self._total_drawdown_pct = drawdown * 100
        else:
            self._total_drawdown_pct = Decimal(0)

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
        NOTE: Relies on internally calculated equity/drawdown, which are updated
              after execution reports. It does NOT fetch live prices itself.
        """
        self.logger.debug("get_current_state called.", source_module=self.__class__.__name__)
        # Note: Accessing state vars directly here assumes they are kept up-to-date
        # by the async handler. No lock needed for read if running in single thread.

        positions_dict = {}
        for pair, pos_info in self._positions.items():
            if pos_info.quantity != 0:  # Only include open positions
                # Get latest price synchronously (from placeholder or real service)
                latest_price = self._get_latest_price_sync(pair)
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
                }

        # Calculate total exposure %
        # Sum market values (needs conversion to valuation currency)
        total_position_value_in_valuation_ccy = Decimal(0)
        for pair_data in positions_dict.values():
            if pair_data["current_market_value"] is not None:
                market_val = Decimal(pair_data["current_market_value"])
                quote_asset = pair_data["quote_asset"]
                if quote_asset == self.valuation_currency:
                    total_position_value_in_valuation_ccy += market_val
                else:
                    # Convert market value (in quote asset) to valuation currency
                    conv_pair = f"{quote_asset}/{self.valuation_currency}"
                    conv_price = self._get_latest_price_sync(conv_pair)
                    if conv_price is not None and conv_price > 0:
                        total_position_value_in_valuation_ccy += market_val * conv_price
                    else:
                        # Try inverse conversion price
                        inv_conv_pair = (
                            f"{self.valuation_currency}/{quote_asset}"
                        )
                        inv_conv_price = self._get_latest_price_sync(
                            inv_conv_pair
                        )
                        if inv_conv_price is not None and inv_conv_price > 0:
                            total_position_value_in_valuation_ccy += (
                                market_val / inv_conv_price
                            )
                        else:
                            self.logger.warning(
                                (
                                    f"Missing conversion price for exposure calc "
                                    f"{quote_asset} to {self.valuation_currency}"
                                ),
                                source_module=self.__class__.__name__,
                            )

        total_exposure_pct = (
            (abs(total_position_value_in_valuation_ccy) / self._total_equity) * 100
            if self._total_equity > 0
            else Decimal(0)
        )

        # TODO: Add calculation for daily/weekly drawdown if needed

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "valuation_currency": self.valuation_currency,
            "total_equity": str(self._total_equity),
            "available_funds": {k: str(v) for k, v in self._available_funds.items()},
            "positions": positions_dict,
            "total_exposure_pct": str(total_exposure_pct),
            "daily_drawdown_pct": "0.00",  # Placeholder
            "weekly_drawdown_pct": "0.00",  # Placeholder
            "total_drawdown_pct": str(self._total_drawdown_pct),
        }

    def get_available_funds(self, currency: str) -> Decimal:
        """Returns the available funds for a specific currency."""
        return self._available_funds.get(currency.upper(), Decimal(0))
        
    def get_current_equity(self) -> Decimal:
        """Returns the current equity value."""
        return self._total_equity

    def get_open_positions(self) -> List[PositionInfo]:
        """Returns a list of open positions."""
        return [pos for pos in self._positions.values() if pos.quantity != 0]

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        """Splits a trading symbol (e.g., 'XRP/USD') into base and quote assets."""
        parts = symbol.split("/")
        if len(parts) == 2:
            return parts[0].upper(), parts[1].upper()
        raise ValueError(f"Invalid symbol format: {symbol}. Expected 'BASE/QUOTE'.")


# Example Usage (for testing purposes, remove in production)
async def example_usage() -> None:
    # --- Mocks & Setup ---
    from typing import Protocol, TypeVar, Generic, runtime_checkable

    T = TypeVar('T')
    
    # Define protocol classes that are compatible with the expected interfaces
    @runtime_checkable
    class ConfigManagerProtocol(Protocol):
        def get(self, key: str, default: Any = None) -> Any: ...
        
    @runtime_checkable
    class LoggerServiceProtocol(Protocol):  # Remove Generic[T] since it causes invariant/covariant type error
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
        def get_latest_price(self, trading_pair: str) -> Optional[Decimal]: ...
        def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]: ...

    # Import the actual types for type casting
    from .config_manager import ConfigManager
    from .logger_service import LoggerService
    from .core.pubsub import PubSubManager
    from .market_price_service import MarketPriceService

    class MockConfigManager:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "portfolio.initial_capital":
                return {"USD": 100000}
            if key == "portfolio.valuation_currency":
                return "USD"
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
            # Extract event_type safely using getattr with EventType type checking
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
        def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
            if trading_pair == "BTC/USD":
                return Decimal("50000.0")
            return None
            
        def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
             return (Decimal("49999.0"), Decimal("50001.0"))

    # --- Import event classes for the example function ---
    from .core.events import EventType, ExecutionReportEvent

    # --- Initialization ---
    # Create mock instances and explicitly cast them to the required types
    mock_config = MockConfigManager()
    mock_logger = MockLoggerService()
    mock_pubsub = MockPubSubManager(logger=mock_logger)
    mock_market_price = MockMarketPriceService()

    # Use type casts to satisfy the type checker
    portfolio_manager = PortfolioManager(
        config_manager=cast(ConfigManager, mock_config),
        pubsub_manager=cast(PubSubManager, mock_pubsub),
        market_price_service=cast(MarketPriceService, mock_market_price),
        logger_service=cast(LoggerService[Any], mock_logger),
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

    await portfolio_manager.stop()


if __name__ == "__main__":
    # asyncio.run(example_usage())
    pass
