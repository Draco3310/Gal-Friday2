"""Simulator for order execution during backtesting using historical OHLCV data.

This module provides a simulated execution environment for backtesting trading strategies.
It handles market and limit orders, slippage simulation, and SL/TP order processing
without requiring a connection to an actual exchange.
"""

import asyncio  # Required for test coroutines
import decimal  # noqa: F401 # Used in error handling
import logging
import uuid  # Add missing uuid import
from collections import defaultdict  # Added for _active_sl_tp
from dataclasses import dataclass, field  # noqa: F401 # Used in placeholder classes
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import TYPE_CHECKING  # Used in type hints and conditional imports
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

import pandas as pd  # Used in type hints for pd.Series

# Set Decimal precision
getcontext().prec = 28

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .core.events import Event  # Base class for events
    from .core.events import EventType  # Event type enumeration
    from .core.events import ExecutionReportEvent, TradeSignalApprovedEvent
    from .core.pubsub import PubSubManager
    from .historical_data_service import HistoricalDataService
    from .logger_service import LoggerService
else:
    # Import Event-related classes from core placeholders
    from .core.placeholder_classes import (  # noqa: F401 # Required for event system
        ConfigManager,
        Event,
        EventType,
        ExecutionReportEvent,
        HistoricalDataService,
        PubSubManager,
        TradeSignalApprovedEvent,
    )

# log = logging.getLogger(__name__) # Removed module-level logger


class SimulatedExecutionHandler:
    """Simulates order execution based on historical OHLCV data for backtesting."""

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: "PubSubManager",
        data_service: "HistoricalDataService",
        logger_service: "LoggerService",
    ):  # Added logger_service
        """Initialize the simulation execution handler with required services.

        Args
        ----
            config_manager: Configuration provider for slippage, fees, etc.
            pubsub_manager: Publish-subscribe manager for event communication
            data_service: Service providing access to historical data
            logger_service: Service for structured logging
        """
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.data_service = data_service
        self.logger = logger_service  # Assigned injected logger

        # Load configuration
        self.taker_fee_pct = self.config.get_decimal(
            "backtest.commission_taker_pct", Decimal("0.0026")
        )
        self.maker_fee_pct = self.config.get_decimal(
            "backtest.commission_maker_pct", Decimal("0.0016")
        )
        self.slippage_model = self.config.get("backtest.slippage_model", "volatility")
        self.slip_atr_multiplier = self.config.get_decimal(
            "backtest.slippage_atr_multiplier", Decimal("0.1")
        )
        self.slip_fixed_pct = self.config.get_decimal(
            "backtest.slippage_fixed_pct", Decimal("0.0005")
        )
        self.valuation_currency = self.config.get("portfolio.valuation_currency", "USD").upper()

        # New configurations based on whiteboard
        self._active_sl_tp: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._fill_liquidity_ratio = self.config.get_decimal(
            "backtest.fill_liquidity_ratio", Decimal("0.1")  # Default 10% of bar volume
        )
        self.limit_order_timeout_bars = self.config.get_int(  # Not fully used in this iteration
            "backtest.limit_order_timeout_bars", 3  # Example: check up to 3 future bars
        )
        # Slippage model: market_impact parameters
        self.slip_market_impact_base_pct = self.config.get_decimal(
            "backtest.slippage_market_impact_base_pct", Decimal("0.0001")  # 0.01% base
        )
        self.slip_market_impact_factor = self.config.get_decimal(
            "backtest.slippage_market_impact_factor", Decimal("0.1")
        )
        self.slip_market_impact_exponent = self.config.get_decimal(
            "backtest.slippage_market_impact_exponent", Decimal("1.0")  # Linear impact initially
        )

        # Placeholder for min/max order sizes (needs actual config keys)
        # self.min_order_sizes = self.config.get_dict("backtest.min_order_sizes", {})
        # self.max_order_sizes = self.config.get_dict("backtest.max_order_sizes", {})

        self.logger.info(
            "SimulatedExecutionHandler initialized.", source_module=self.__class__.__name__
        )
        self.logger.info(
            (
                f" Fees: Taker={self.taker_fee_pct * 100:.4f}%, "
                f"Maker={self.maker_fee_pct * 100:.4f}% (Maker not used in MVP)"
            ),
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            (
                f" Slippage: Model={self.slippage_model}, "
                f"ATR Mult={self.slip_atr_multiplier}, "
                f"Fixed Pct={self.slip_fixed_pct * 100:.4f}%, "
                f"Market Impact Base Pct: {self.slip_market_impact_base_pct * 100:.4f}%"
            ),
            source_module=self.__class__.__name__,
        )
        self.logger.info(
            f" Partial Fills: Liquidity Ratio={self._fill_liquidity_ratio}",
            source_module=self.__class__.__name__,
        )

    async def start(self) -> None:  # Interface consistency
        """Start the simulated execution handler.

        For API consistency with other execution handlers; no external connections needed.
        """
        # log.info("SimulatedExecutionHandler started.") # Replaced
        self.logger.info(
            "SimulatedExecutionHandler started.", source_module=self.__class__.__name__
        )
        # No external connections to establish

    async def stop(self) -> None:  # Interface consistency
        """Stop the simulated execution handler.

        For API consistency with other execution handlers; no external connections needed.
        """
        # log.info("SimulatedExecutionHandler stopped.") # Replaced
        self.logger.info(
            "SimulatedExecutionHandler stopped.", source_module=self.__class__.__name__
        )
        # No external connections to close

    async def handle_trade_signal_approved(self, event: "TradeSignalApprovedEvent") -> None:
        """Process an approved signal and simulate fill based on next bar data."""
        self.logger.debug(
            (f"SimExec received approved signal: " f"{event.signal_id} at {event.timestamp}"),
            source_module=self.__class__.__name__,
        )

        # Validate order parameters
        if not await self._validate_order_parameters(event):
            return  # Validation failed, report already published

        # Get next bar data
        next_bar = await self._get_next_bar_data(event)
        if next_bar is None:
            return

        try:
            # Initialize simulation parameters
            fill_result = await self._simulate_order_fill(event, next_bar)
            if not fill_result:
                return

            # Store SL/TP if entry order is filled
            if fill_result["status"] in ["FILLED", "PARTIALLY_FILLED"] and fill_result[
                "quantity"
            ] > Decimal(0):
                position_id = str(event.signal_id)  # Or a more unique ID if needed
                self._active_sl_tp[position_id] = {
                    "sl": event.sl_price,
                    "tp": event.tp_price,
                    "side": event.side,
                    "pair": event.trading_pair,
                    "entry_qty": fill_result["quantity"],  # Filled quantity
                    "entry_ts": fill_result["timestamp"],  # Fill timestamp of entry
                    "entry_event": event,  # Original event for reporting exits
                    "exchange": event.exchange,  # Store exchange
                    "order_type": event.order_type,  # Store original order type
                }
                self.logger.info(
                    f"Stored SL/TP for position {position_id} ({event.trading_pair}): "
                    f"SL={event.sl_price}, TP={event.tp_price}, Qty={fill_result['quantity']}",
                    source_module=self.__class__.__name__,
                )

            # Publish the execution report
            await self._publish_simulated_report(
                event,
                fill_result["status"],
                fill_result["quantity"],
                fill_result["fill_price"],
                fill_result["commission"],
                fill_result["commission_asset"],
                fill_result.get("error_msg"),
                fill_result["timestamp"],
            )

        except Exception as e:
            error_msg = (
                f"Error during fill simulation for signal {event.signal_id}. "
                f"Event: {event}, Next Bar: {next_bar}"
            )
            self.logger.error(
                error_msg,
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            await self._publish_simulated_report(
                event,
                "ERROR",
                Decimal(0),
                None,
                Decimal(0),
                None,
                f"Simulation error: {e}",
            )

    async def _validate_order_parameters(self, event: "TradeSignalApprovedEvent") -> bool:
        """Validate incoming order parameters."""
        if not isinstance(event.quantity, Decimal) or event.quantity <= Decimal(0):
            error_msg = f"Invalid order quantity: {event.quantity} for signal {event.signal_id}."
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event, "REJECTED", Decimal(0), None, Decimal(0), None, error_msg
            )
            return False

        if event.side.upper() not in ["BUY", "SELL"]:
            error_msg = f"Invalid order side: {event.side} for signal {event.signal_id}."
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event, "REJECTED", Decimal(0), None, Decimal(0), None, error_msg
            )
            return False

        supported_order_types = ["MARKET", "LIMIT"]  # Extend as more types are supported
        if event.order_type.upper() not in supported_order_types:
            error_msg = (
                f"Unsupported order type '{event.order_type}' for simulation. "
                f"Signal {event.signal_id}."
            )
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event, "REJECTED", Decimal(0), None, Decimal(0), None, error_msg
            )
            return False

        if event.order_type.upper() == "LIMIT":
            if (
                event.limit_price is None
                or not isinstance(event.limit_price, Decimal)
                or event.limit_price <= Decimal(0)
            ):
                error_msg = (
                    f"Invalid or missing limit price for LIMIT order: {event.limit_price} "
                    f"for signal {event.signal_id}."
                )
                self.logger.error(error_msg, source_module=self.__class__.__name__)
                await self._publish_simulated_report(
                    event, "REJECTED", Decimal(0), None, Decimal(0), None, error_msg
                )
                return False

        # Placeholder for min/max order size validation
        # based on self.min_order_sizes, self.max_order_sizes
        # pair_min_size = self.min_order_sizes.get(event.trading_pair)
        # if pair_min_size and event.quantity < Decimal(str(pair_min_size)):
        #     # ... publish rejected ...
        #     return False

        return True

    async def _get_next_bar_data(
        self,
        event: "TradeSignalApprovedEvent",
    ) -> Optional[pd.Series]:
        """Get and validate the next bar data for simulation."""
        try:
            next_bar = self.data_service.get_next_bar(event.trading_pair, event.timestamp)

            if next_bar is None:
                error_msg = (
                    f"No next bar data found for {event.trading_pair} after "
                    f"{event.timestamp}. Cannot simulate fill for {event.signal_id}."
                )
                self.logger.warning(
                    error_msg,
                    source_module=self.__class__.__name__,
                )
                next_time = event.timestamp + timedelta(minutes=1)
                error_detail = f"No historical data for fill simulation at {next_time}"
                await self._publish_simulated_report(
                    event,
                    "REJECTED",
                    Decimal(0),
                    None,
                    Decimal(0),
                    None,
                    error_detail,
                )
                return None

            return next_bar
        except Exception as e:
            self.logger.error(
                f"Error retrieving next bar data: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            return None

    async def _simulate_order_fill(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """Simulate order fill based on order type and market conditions."""
        # fill_qty will be determined by simulation logic, considering partial fills
        commission_pct = self.taker_fee_pct  # Assume taker for MVP

        # Convert index name to datetime if needed
        if hasattr(next_bar, "name") and next_bar.name is not None:
            # Make sure fill_timestamp is a datetime object, not just a
            # Hashable index
            if isinstance(next_bar.name, datetime):
                fill_timestamp = next_bar.name
            else:
                # Try to convert to datetime or use current time
                # First convert to a specific type that to_datetime can handle
                try:
                    # Convert the Hashable to a string first if needed
                    name_str = str(next_bar.name)
                    fill_timestamp = pd.to_datetime(name_str)
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Could not convert bar index {next_bar.name} to datetime. "
                        f"Using current time.",
                        source_module=self.__class__.__name__,
                    )
                    fill_timestamp = datetime.utcnow()
        else:
            self.logger.warning(
                "Bar has no name/index. Using current time for fill timestamp.",
                source_module=self.__class__.__name__,
            )
            fill_timestamp = datetime.utcnow()

        # Handle different order types
        if event.order_type.upper() == "MARKET":
            return await self._simulate_market_order(
                event,
                next_bar,
                commission_pct,
                fill_timestamp,
            )
        elif event.order_type.upper() == "LIMIT":
            return await self._simulate_limit_order(
                event,
                next_bar,
                commission_pct,
                fill_timestamp,
            )
        else:
            error_msg = (
                f"Unsupported order type '{event.order_type}' for simulation. "
                f"Signal {event.signal_id}."
            )
            self.logger.error(
                error_msg,
                source_module=self.__class__.__name__,
            )
            await self._publish_simulated_report(
                event,
                "REJECTED",
                Decimal(0),
                None,
                Decimal(0),
                None,
                f"Unsupported order type: {event.order_type}",
            )
            return None

    async def _simulate_market_order(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series,
        commission_pct: Decimal,
        fill_timestamp: datetime,
    ) -> dict:
        """Simulate a market order fill."""
        fill_price_base = next_bar["open"]  # Assume fill at next bar's open

        # Calculate slippage using potentially next_bar volume data
        next_bar_volume = Decimal(str(next_bar.get("volume", "0")))  # Ensure volume exists
        slippage = self._calculate_slippage(
            event.trading_pair,
            event.side,
            Decimal(fill_price_base),
            event.timestamp,  # Timestamp of signal generation
            order_quantity=event.quantity,
            bar_volume=next_bar_volume,
        )

        # Calculate fill price with slippage
        simulated_fill_price = (
            Decimal(fill_price_base) + slippage
            if event.side.upper() == "BUY"
            else Decimal(fill_price_base) - slippage
        )

        self.logger.debug(
            (
                f" Market fill sim: Base={fill_price_base}, "
                f"Slip={slippage:.6f}, Final={simulated_fill_price:.6f}"
            ),
            source_module=self.__class__.__name__,
        )

        # --- Partial Fill Logic ---
        available_volume_at_bar = Decimal(str(next_bar.get("volume", "0")))
        # Max fillable based on a fraction of the bar's total volume
        max_fillable_qty_liquidity = available_volume_at_bar * self._fill_liquidity_ratio

        simulated_fill_qty = min(event.quantity, max_fillable_qty_liquidity)

        status = "REJECTED"
        error_message = None

        if simulated_fill_qty > Decimal("1e-12"):  # Check against a small threshold
            if simulated_fill_qty < event.quantity:
                status = "PARTIALLY_FILLED"
                self.logger.info(
                    f"Market order {event.signal_id} ({event.trading_pair}) partially filled: "
                    f"{simulated_fill_qty}/{event.quantity} based on bar volume "
                    f"{available_volume_at_bar} and liquidity ratio {self._fill_liquidity_ratio}",
                    source_module=self.__class__.__name__,
                )
            else:
                status = "FILLED"
        else:
            simulated_fill_qty = Decimal(0)
            # Ensure it's precisely zero if rejected
            error_message = "Zero fillable quantity based on available liquidity"
            self.logger.warning(
                f"Market order {event.signal_id} ({event.trading_pair}) rejected due to zero "
                f"fillable quantity. Requested: {event.quantity}, Bar Volume: "
                f"{available_volume_at_bar}, Liquidity Ratio: {self._fill_liquidity_ratio}",
                source_module=self.__class__.__name__,
            )

        # Calculate commission based on actual filled quantity
        fill_value = simulated_fill_qty * simulated_fill_price
        commission_amount = abs(fill_value * commission_pct)
        _, quote_asset = event.trading_pair.split("/")
        commission_asset = quote_asset.upper() if simulated_fill_qty > Decimal(0) else None

        final_fill_price = simulated_fill_price if simulated_fill_qty > Decimal(0) else None

        # Convert fill_timestamp if needed to ensure it's a datetime object
        timestamp = fill_timestamp
        if not isinstance(timestamp, datetime):
            self.logger.warning(
                f"Converting fill_timestamp from {type(timestamp)} to datetime",
                source_module=self.__class__.__name__,
            )
            # Default to current time if conversion fails
            timestamp = datetime.utcnow()

        return {
            "status": status,
            "quantity": simulated_fill_qty,
            "fill_price": final_fill_price,
            "commission": commission_amount,
            "commission_asset": commission_asset,
            "timestamp": timestamp,
            "error_msg": error_message,
        }

    async def _simulate_limit_order(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series,
        commission_pct: Decimal,
        fill_timestamp: datetime,
    ) -> Optional[dict]:
        """Simulate a limit order fill."""
        limit_price = event.limit_price
        if limit_price is None:
            error_msg = f"Limit price missing for signal {event.signal_id}. " "Cannot simulate."
            self.logger.error(
                error_msg,
                source_module=self.__class__.__name__,
            )
            await self._publish_simulated_report(
                event,
                "REJECTED",
                Decimal(0),
                None,
                Decimal(0),
                None,
                "Limit price missing",
            )
            return None

        # Check if order would be filled
        filled = self._check_limit_order_fill(event.side, limit_price, next_bar)
        if not filled:
            bar_info = f"Limit={limit_price}, " f"Bar H/L={next_bar['high']}/{next_bar['low']}"
            self.logger.debug(
                f" Limit order {event.signal_id} NOT filled. {bar_info}",
                source_module=self.__class__.__name__,
            )
            await self._publish_simulated_report(
                event,
                "REJECTED",
                Decimal(0),
                None,
                Decimal(0),
                None,
                "Limit price not reached",
            )
            return None

        # Calculate commission
        fill_value = event.quantity * limit_price
        commission_amount = abs(fill_value * commission_pct)
        _, quote_asset = event.trading_pair.split("/")
        commission_asset = quote_asset.upper()

        self.logger.debug(
            f" Limit fill sim: Price={limit_price:.6f}",
            source_module=self.__class__.__name__,
        )

        # Convert fill_timestamp if needed to ensure it's a datetime object
        timestamp = fill_timestamp
        if not isinstance(timestamp, datetime):
            self.logger.warning(
                f"Converting fill_timestamp from {type(timestamp)} to datetime",
                source_module=self.__class__.__name__,
            )
            # Default to current time if conversion fails
            timestamp = datetime.utcnow()

        return {
            "status": "FILLED",
            "quantity": event.quantity,
            "fill_price": limit_price,
            "commission": commission_amount,
            "commission_asset": commission_asset,
            "timestamp": timestamp,
        }

    def _check_limit_order_fill(
        self,
        side: str,
        limit_price: Decimal,
        next_bar: pd.Series,
    ) -> bool:
        """Check if a limit order would be filled based on price levels."""
        try:
            if side.upper() == "BUY":
                return bool(next_bar["low"] <= limit_price)
            else:  # SELL
                return bool(next_bar["high"] >= limit_price)
        except Exception as e:
            self.logger.error(
                f"Error checking limit order fill: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            return False

    def _calculate_slippage(
        self,
        trading_pair: str,
        side: str,
        base_price: Decimal,
        signal_timestamp: datetime,  # Timestamp of the original signal event
        order_quantity: Optional[Decimal] = None,  # Original order quantity
        bar_volume: Optional[Decimal] = None,  # Volume of the bar against which fill is attempted
    ) -> Decimal:
        """Calculate slippage based on configured model."""
        slippage = Decimal(0)
        try:
            if self.slippage_model == "fixed":
                slippage = base_price * self.slip_fixed_pct
            elif self.slippage_model == "market_impact":
                if (
                    order_quantity is None
                    or order_quantity <= Decimal(0)
                    or bar_volume is None
                    or bar_volume <= Decimal(0)
                ):
                    self.logger.warning(
                        f"Market impact slippage model requires valid order_quantity and "
                        f"bar_volume. Got Qty: {order_quantity}, Vol: {bar_volume}. "
                        f"Using zero slippage.",
                        source_module=self.__class__.__name__,
                    )
                    return Decimal(0)

                try:
                    # Ensure order_quantity and bar_volume are positive for calculation
                    qty_ratio = order_quantity / bar_volume
                    impact_component_pct = self.slip_market_impact_factor * (
                        qty_ratio**self.slip_market_impact_exponent
                    )
                    total_slippage_pct = self.slip_market_impact_base_pct + impact_component_pct
                    slippage = base_price * total_slippage_pct
                except (
                    decimal.InvalidOperation
                ) as e:  # Catch potential math errors like negative power
                    self.logger.error(
                        f"Error in market_impact slippage calculation "
                        f"(ratio: {order_quantity}/{bar_volume}): {e}. "
                        f"Using base fixed slippage.",
                        source_module=self.__class__.__name__,
                        exc_info=True,
                    )
                    slippage = base_price * self.slip_fixed_pct  # Fallback

            elif self.slippage_model == "volatility":
                # Get ATR for the bar the signal was generated on
                atr = self.data_service.get_atr(trading_pair, signal_timestamp)
                if atr is not None and atr > 0:
                    slippage = atr * self.slip_atr_multiplier
                else:
                    self.logger.warning(
                        (
                            f"Could not get ATR for {trading_pair} at {signal_timestamp} "
                            "for slippage calc. Using zero."
                        ),
                        source_module=self.__class__.__name__,
                    )
            else:
                self.logger.warning(
                    f"Unknown slippage model: {self.slippage_model}. Using zero.",
                    source_module=self.__class__.__name__,
                )
        except Exception as e:
            self.logger.error(
                f"Error calculating slippage: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

        # Slippage is always adverse
        return abs(slippage)

    async def _publish_simulated_report(
        self,
        originating_event: "TradeSignalApprovedEvent",
        status: str,
        qty_filled: Decimal,
        avg_price: Optional[Decimal],
        commission: Decimal,
        commission_asset: Optional[str],
        error_msg: Optional[str] = None,
        fill_timestamp: Optional[datetime] = None,
        # Optional overrides for exit orders (SL/TP)
        custom_exchange_order_id: Optional[str] = None,
        custom_client_order_id: Optional[str] = None,
        custom_order_type: Optional[str] = None,
        custom_side: Optional[str] = None,
    ) -> None:
        """Create and publish a simulated ExecutionReportEvent."""
        try:
            # Generate unique simulation order ID if not provided
            if custom_exchange_order_id:
                exchange_order_id = custom_exchange_order_id
                client_order_id = (
                    custom_client_order_id or f"sim_{originating_event.signal_id}_exit"
                )
            else:  # For entry orders
                timestamp_micros = int(datetime.utcnow().timestamp() * 1e6)
                exchange_order_id = f"sim_{originating_event.signal_id}_{timestamp_micros}"
                client_order_id = f"sim_{originating_event.signal_id}"

            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal_id=originating_event.signal_id,
                exchange_order_id=exchange_order_id,
                client_order_id=client_order_id,
                trading_pair=originating_event.trading_pair,
                exchange=originating_event.exchange,
                order_status=status,
                order_type=custom_order_type or originating_event.order_type,
                side=custom_side or originating_event.side,
                quantity_ordered=Decimal(
                    originating_event.quantity
                ),  # Original quantity for entry
                quantity_filled=qty_filled,
                average_fill_price=avg_price,
                limit_price=originating_event.limit_price,
                stop_price=None,  # Not applicable for MVP fills
                commission=commission,
                commission_asset=commission_asset,
                timestamp_exchange=(fill_timestamp if fill_timestamp else datetime.utcnow()),
                error_message=error_msg,
            )
            await self.pubsub.publish(report)
            self.logger.debug(
                f"Published simulated report: {status} for {report.signal_id}",
                source_module=self.__class__.__name__,
            )
        except Exception as e:
            self.logger.error(
                (
                    f"Failed to publish simulated execution report for "
                    f"signal {originating_event.signal_id}: {e}"
                ),
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    async def check_active_sl_tp(self, current_bar: pd.Series, bar_timestamp: datetime) -> None:
        """Check SL/TP triggers for active positions with each new bar.

        Called by the backtesting engine for each new bar to monitor active positions.

        Args
        ----
            current_bar: The OHLCV data for the current simulation time
            bar_timestamp: The timestamp of the current_bar (usually its open time)
        """
        if not self._is_valid_bar_for_sl_tp(current_bar, bar_timestamp):
            return

        try:
            bar_high = Decimal(str(current_bar["high"]))
            bar_low = Decimal(str(current_bar["low"]))
        except (TypeError, ValueError, decimal.InvalidOperation) as e:
            self.logger.error(
                f"Could not convert current bar high/low to Decimal for SL/TP check at "
                f"{bar_timestamp}. Bar: {current_bar.to_dict()}. Error: {e}",
                source_module=self.__class__.__name__,
            )
            return

        triggered_position_ids = []
        for position_id, sl_tp_data in list(self._active_sl_tp.items()):
            if bar_timestamp <= sl_tp_data["entry_ts"]:
                continue

            exit_details = self._check_sl_tp_trigger(sl_tp_data, bar_high, bar_low, bar_timestamp)

            if exit_details:
                await self._process_sl_tp_exit(
                    position_id, sl_tp_data, exit_details, bar_timestamp
                )
                triggered_position_ids.append(position_id)

        for pos_id in triggered_position_ids:
            if pos_id in self._active_sl_tp:
                del self._active_sl_tp[pos_id]
                self.logger.debug(
                    f"Removed SL/TP monitoring for position {pos_id}",
                    source_module=self.__class__.__name__,
                )

    def _is_valid_bar_for_sl_tp(self, current_bar: pd.Series, bar_timestamp: datetime) -> bool:
        """Check if the current bar is valid for SL/TP processing."""
        if (
            not hasattr(current_bar, "high")
            or not hasattr(current_bar, "low")
            or current_bar.get("high") is None
            or current_bar.get("low") is None
        ):
            self.logger.warning(
                f"Current bar for SL/TP check at {bar_timestamp} missing high/low data or "
                f"data is None. Bar: {current_bar.to_dict()}",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    def _check_sl_tp_trigger(
        self, sl_tp_data: dict, bar_high: Decimal, bar_low: Decimal, bar_timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Check if SL or TP conditions are met for a position."""
        sl_price = sl_tp_data["sl"]
        tp_price = sl_tp_data["tp"]
        original_side = sl_tp_data["side"]

        exit_price: Optional[Decimal] = None
        exit_reason: Optional[str] = None
        exit_order_type = "MARKET"  # Default for SL

        # Check Stop Loss (SL)
        if sl_price is not None:
            if original_side.upper() == "BUY" and bar_low <= sl_price:
                exit_price = sl_price
                exit_reason = f"Stop Loss triggered at {sl_price} (Bar Low: {bar_low})"
            elif original_side.upper() == "SELL" and bar_high >= sl_price:
                exit_price = sl_price
                exit_reason = f"Stop Loss triggered at {sl_price} (Bar High: {bar_high})"

        # Check Take Profit (TP) - only if SL not already triggered
        if exit_price is None and tp_price is not None:
            if original_side.upper() == "BUY" and bar_high >= tp_price:
                exit_price = tp_price
                exit_reason = f"Take Profit triggered at {tp_price} (Bar High: {bar_high})"
                exit_order_type = "LIMIT"
            elif original_side.upper() == "SELL" and bar_low <= tp_price:
                exit_price = tp_price
                exit_reason = f"Take Profit triggered at {tp_price} (Bar Low: {bar_low})"
                exit_order_type = "LIMIT"

        if exit_price is not None and exit_reason is not None:
            return {
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "exit_order_type": exit_order_type,
                "original_side": original_side,
                "entry_qty": sl_tp_data["entry_qty"],
                "trading_pair": sl_tp_data["pair"],
                "originating_event": sl_tp_data["entry_event"],
            }
        return None

    async def _process_sl_tp_exit(
        self,
        position_id: str,
        sl_tp_data: dict,
        exit_details: Dict[str, Any],
        bar_timestamp: datetime,
    ) -> None:
        """Process the exit of a position due to SL/TP trigger."""
        log_msg = (
            f"{exit_details['exit_reason']} for position {position_id} "
            f"({exit_details['trading_pair']}) Exit Price: {exit_details['exit_price']}, "
            f"Qty: {exit_details['entry_qty']}, Bar: {bar_timestamp}"
        )
        self.logger.info(log_msg, source_module=self.__class__.__name__)

        exit_value = exit_details["entry_qty"] * exit_details["exit_price"]
        commission_amount = abs(exit_value * self.taker_fee_pct)
        _, quote_asset = exit_details["trading_pair"].split("/")
        commission_asset = quote_asset.upper()

        exit_timestamp_micros = int(datetime.utcnow().timestamp() * 1e6)
        originating_event_signal_id = exit_details["originating_event"].signal_id
        exit_exchange_order_id = f"sim_exit_{originating_event_signal_id}_{exit_timestamp_micros}"
        exit_client_order_id = f"sim_exit_{originating_event_signal_id}"
        exit_side = "SELL" if exit_details["original_side"].upper() == "BUY" else "BUY"

        await self._publish_simulated_report(
            originating_event=exit_details["originating_event"],
            status="FILLED",
            qty_filled=exit_details["entry_qty"],
            avg_price=exit_details["exit_price"],
            commission=commission_amount,
            commission_asset=commission_asset,
            error_msg=exit_details["exit_reason"],
            fill_timestamp=bar_timestamp,
            custom_exchange_order_id=exit_exchange_order_id,
            custom_client_order_id=exit_client_order_id,
            custom_order_type=exit_details["exit_order_type"],
            custom_side=exit_side,
        )


# Example Usage
async def _setup_services_for_main_example() -> (
    Tuple["ConfigManager", "PubSubManager", "HistoricalDataService", "LoggerService"]
):
    """Set up required services for the main example function."""
    logger = logging.getLogger("sim_exec_example")
    config = ConfigManager()
    pubsub = PubSubManager(config_manager=config, logger=logger)

    class MockHistoricalDataService(HistoricalDataService):
        def get_next_bar(self, trading_pair: str, timestamp: datetime) -> Optional[pd.Series]:
            idx = pd.date_range(start="2023-01-01 00:01:00", periods=5, freq="1min", tz="UTC")
            dummy_data = {
                "open": [0.495, 0.496, 0.497, 0.498, 0.499],
                "high": [0.50, 0.505, 0.51, 0.515, 0.52],
                "low": [0.4951, 0.4952, 0.4953, 0.4954, 0.4955],
                "close": [0.497, 0.498, 0.499, 0.50, 0.51],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
            df = pd.DataFrame(dummy_data, index=idx)
            return df.iloc[1]

        def get_atr(
            self, trading_pair: str, timestamp: datetime, period: int = 14
        ) -> Optional[Decimal]:
            return Decimal("0.0025")

        async def get_historical_ohlcv(
            self, trading_pair: str, start_time: datetime, end_time: datetime, interval: str
        ) -> Optional[pd.DataFrame]:
            return pd.DataFrame()

        async def get_historical_trades(
            self, trading_pair: str, start_time: datetime, end_time: datetime
        ) -> Optional[pd.DataFrame]:
            return pd.DataFrame()

    data_service = MockHistoricalDataService()
    from .logger_service import LoggerService  # Local import for example

    class MockLoggerService(LoggerService):
        def __init__(self) -> None:
            pass

        def log(
            self,
            level: int,
            msg: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
            exc_info: Optional[bool] = None,
        ) -> None:
            logger.log(level, f"[{source_module}] {msg}", exc_info=exc_info)

        def info(
            self,
            msg: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            logger.info(f"[{source_module}] {msg}")

        def debug(
            self,
            msg: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            logger.debug(f"[{source_module}] {msg}")

        def warning(
            self,
            msg: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            logger.warning(f"[{source_module}] {msg}")

        def error(
            self,
            msg: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
            exc_info: Optional[bool] = None,
        ) -> None:
            logger.error(f"[{source_module}] {msg}", exc_info=exc_info)

        def critical(
            self,
            msg: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
            exc_info: Optional[bool] = None,
        ) -> None:
            logger.critical(f"[{source_module}] {msg}", exc_info=exc_info)

    logger_service = MockLoggerService()
    return config, pubsub, data_service, logger_service


async def _run_market_order_test(sim_exec: SimulatedExecutionHandler) -> None:
    """Run the market order test scenario."""
    signal_market_buy_ts = datetime.utcnow() - timedelta(minutes=10)
    signal_market_buy = TradeSignalApprovedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=signal_market_buy_ts,
        signal_id=uuid.uuid4(),
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        order_type="MARKET",
        quantity=Decimal("500"),
        limit_price=None,
        sl_price=Decimal("0.48"),
        tp_price=Decimal("0.52"),
        risk_parameters={"source": "example_main_market_buy"},
    )
    print("\n--- Test Market Buy ---")
    await sim_exec.handle_trade_signal_approved(signal_market_buy)


async def _run_sl_tp_check_test(sim_exec: SimulatedExecutionHandler) -> None:
    """Run the SL/TP check test scenario if positions are active."""
    if not sim_exec._active_sl_tp:
        return

    pos_id_to_check = list(sim_exec._active_sl_tp.keys())[0]
    entry_ts = sim_exec._active_sl_tp[pos_id_to_check]["entry_ts"]

    sl_tp_check_ts = entry_ts + timedelta(minutes=1)
    sl_trigger_bar_data = {
        "open": Decimal("0.475"),
        "high": Decimal("0.478"),
        "low": Decimal("0.47"),
        "close": Decimal("0.472"),
        "volume": Decimal("500"),
    }
    sl_trigger_bar = pd.Series(sl_trigger_bar_data, name=sl_tp_check_ts)
    print(
        f"\n--- Checking SL/TP with bar at {sl_tp_check_ts} " f"(Low: {sl_trigger_bar['low']}) ---"
    )
    await sim_exec.check_active_sl_tp(sl_trigger_bar, sl_tp_check_ts)

    if pos_id_to_check in sim_exec._active_sl_tp:
        tp_check_ts = entry_ts + timedelta(minutes=2)
        tp_trigger_bar_data = {
            "open": Decimal("0.518"),
            "high": Decimal("0.525"),
            "low": Decimal("0.517"),
            "close": Decimal("0.522"),
            "volume": Decimal("600"),
        }
        tp_trigger_bar = pd.Series(tp_trigger_bar_data, name=tp_check_ts)
        print(
            f"\n--- Checking SL/TP with bar at {tp_check_ts} "
            f"(High: {tp_trigger_bar['high']}) ---"
        )
        await sim_exec.check_active_sl_tp(tp_trigger_bar, tp_check_ts)


async def _run_limit_order_tests(sim_exec: SimulatedExecutionHandler) -> None:
    """Run limit order test scenarios."""
    signal_limit_buy_fill_ts = datetime.utcnow() - timedelta(minutes=5)
    signal_limit_buy_fill = TradeSignalApprovedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=signal_limit_buy_fill_ts,
        signal_id=uuid.uuid4(),
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("200"),
        limit_price=Decimal("0.496"),
        sl_price=Decimal("0.48"),
        tp_price=Decimal("0.52"),
        risk_parameters={"source": "example_main_limit_fill"},
    )
    print("\n--- Test Limit Buy (Fill) ---")
    await sim_exec.handle_trade_signal_approved(signal_limit_buy_fill)

    signal_limit_buy_nofill_ts = datetime.utcnow() - timedelta(minutes=3)
    signal_limit_buy_nofill = TradeSignalApprovedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=signal_limit_buy_nofill_ts,
        signal_id=uuid.uuid4(),
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("200"),
        limit_price=Decimal("0.40"),
        sl_price=Decimal("0.38"),
        tp_price=Decimal("0.45"),
        risk_parameters={"source": "example_main_limit_nofill"},
    )
    print("\n--- Test Limit Buy (No Fill) ---")
    await sim_exec.handle_trade_signal_approved(signal_limit_buy_nofill)


async def main() -> None:  # C901 'main' is too complex (14) -> reduced by refactoring
    """Run example simulation tests demonstrating the SimulatedExecutionHandler functionality."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    config, pubsub, data_service, logger_service = await _setup_services_for_main_example()
    sim_exec = SimulatedExecutionHandler(config, pubsub, data_service, logger_service)
    await sim_exec.start()

    await _run_market_order_test(sim_exec)
    await _run_sl_tp_check_test(sim_exec)
    await _run_limit_order_tests(sim_exec)

    await sim_exec.stop()


if __name__ == "__main__":
    asyncio.run(main())
