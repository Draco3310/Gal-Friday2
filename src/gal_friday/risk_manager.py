# Risk Manager Module

import asyncio
from decimal import Decimal, getcontext, InvalidOperation
from typing import Tuple, Optional, TYPE_CHECKING, Dict, Any, Union
from dataclasses import dataclass
import uuid
from datetime import datetime

# Event Definitions
from .core.events import (
    Event, 
    EventType, 
    TradeSignalProposedEvent, 
    TradeSignalApprovedEvent, 
    TradeSignalRejectedEvent,
    SystemStateEvent,
    PotentialHaltTriggerEvent
)

# Import PubSubManager
from .core.pubsub import PubSubManager
# Import logger service
from .logger_service import LoggerService


# Custom exceptions
class RiskManagerError(Exception):
    """Base exception class for RiskManager errors."""
    pass


# Type hint for PortfolioManager without circular import
if TYPE_CHECKING:
    from .portfolio_manager import PortfolioManager


# Set Decimal precision
getcontext().prec = 28


# --- Event Payloads ---
@dataclass
class TradeSignalProposedPayload:
    """Payload for trade signal proposals"""
    signal_id: uuid.UUID
    trading_pair: str
    exchange: str
    side: str
    entry_type: str
    proposed_entry_price: Optional[str] = None
    proposed_sl_price: Optional[str] = None
    proposed_tp_price: Optional[str] = None
    strategy_id: str = "default"
    triggering_prediction_event_id: Optional[uuid.UUID] = None


@dataclass
class SystemHaltPayload:
    """Payload for system halt events"""
    reason: str
    details: Dict[str, Any]


# --- RiskManager Class ---
class RiskManager:
    """
    Consumes proposed trade signals, performs pre-trade risk checks against
    portfolio state, and publishes approved/rejected signals or triggers HALT.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pubsub_manager: PubSubManager,
        portfolio_manager: "PortfolioManager",
        logger_service: LoggerService,
    ):
        """
        Initializes the RiskManager.

        Args:
            config: Configuration settings.
            pubsub_manager: The application PubSubManager instance.
            portfolio_manager: The PortfolioManager instance.
            logger_service: Shared logger instance.
        """
        self._config = config.get("risk_manager", {})
        self.pubsub = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self.logger = logger_service
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self._periodic_check_task: Optional[asyncio.Task] = None
        self._source_module = self.__class__.__name__
        
        # Store handler for unsubscribing
        self._signal_proposal_handler = self._handle_trade_signal_proposed

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Loads risk parameters from configuration."""
        limits = self._config.get("limits", {})
        self._max_total_drawdown_pct = Decimal(
            str(limits.get("max_total_drawdown_pct", 15.0))
        )
        self._max_daily_drawdown_pct = Decimal(
            str(limits.get("max_daily_drawdown_pct", 2.0))
        )
        self._max_weekly_drawdown_pct = Decimal(
            str(limits.get("max_weekly_drawdown_pct", 5.0))
        )
        self._max_consecutive_losses = int(limits.get("max_consecutive_losses", 5))
        self._max_exposure_per_asset_pct = Decimal(
            str(limits.get("max_exposure_per_asset_pct", 10.0))
        )
        self._max_total_exposure_pct = Decimal(
            str(limits.get("max_total_exposure_pct", 25.0))
        )
        self._max_order_size_usd = Decimal(str(limits.get("max_order_size_usd", 10000)))
        self._risk_per_trade_pct = Decimal(
            str(self._config.get("sizing", {}).get("risk_per_trade_pct", 0.5))
        )
        self._check_interval_s = int(self._config.get("check_interval_s", 60))
        self._min_sl_distance_pct = Decimal(
            str(self._config.get("min_sl_distance_pct", 0.01))
        )
        self._max_single_position_pct = Decimal(
            str(self._config.get("max_single_position_pct", 100.0))
        )

        self.logger.info("RiskManager configured.", source_module=self._source_module)

    async def start(self) -> None:
        """Starts listening for trade signals and periodic checks."""
        if self._is_running:
            self.logger.warning("RiskManager already running.", source_module=self._source_module)
            return
        self._is_running = True
        
        # Subscribe to proposed signals
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_PROPOSED, self._signal_proposal_handler)

        # Start periodic checks
        if self._check_interval_s > 0:
            self._periodic_check_task = asyncio.create_task(self._run_periodic_checks())
            
        self.logger.info("RiskManager started.", source_module=self._source_module)

    async def stop(self) -> None:
        """Stops the RiskManager."""
        if not self._is_running:
            return
        self._is_running = False
        
        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.TRADE_SIGNAL_PROPOSED, self._signal_proposal_handler)
            self.logger.info("Unsubscribed from TRADE_SIGNAL_PROPOSED.", source_module=self._source_module)
        except Exception as e:
            self.logger.error(f"Error unsubscribing RiskManager: {e}", exc_info=True, source_module=self._source_module)

        # Stop periodic checks
        if self._periodic_check_task and not self._periodic_check_task.done():
            self._periodic_check_task.cancel()
            try:
                await self._periodic_check_task
            except asyncio.CancelledError:
                pass
            self._periodic_check_task = None
            
        self.logger.info("RiskManager stopped.", source_module=self._source_module)
        
    async def _handle_trade_signal_proposed(self, event: TradeSignalProposedEvent) -> None:
        """Handles incoming trade signal proposal events."""
        if not isinstance(event, TradeSignalProposedEvent):
            self.logger.warning(f"Received non-TradeSignalProposedEvent: {type(event)}", source_module=self._source_module)
            return
            
        if not self._is_running:
            return # Don't process if stopped
            
        self.logger.debug(f"Received trade signal proposal: {event.signal_id}")
        
        # Perform checks
        is_approved, reason, approved_payload_data = self._perform_pre_trade_checks(event)
        
        if is_approved and approved_payload_data:
             await self._publish_trade_signal_approved(approved_payload_data)
        else:
            rejection_payload_data = {
                "signal_id": event.signal_id,
                "trading_pair": event.trading_pair,
                "exchange": event.exchange,
                "side": event.side,
                "reason": reason or "Unknown rejection reason",
            }
            await self._publish_trade_signal_rejected(rejection_payload_data)

    async def _run_periodic_checks(self) -> None:
        """Periodically checks portfolio-level risk limits."""
        self.logger.info("Starting periodic risk checks.", source_module=self._source_module)
        while self._is_running:
            try:
                await asyncio.sleep(self._check_interval_s)
                if not self._is_running: break # Check again after sleep
                
                self.logger.debug("Running periodic risk check...", source_module=self._source_module)
                portfolio_state = self._get_portfolio_state()
                
                if portfolio_state is not None:
                    halt_reason = self._check_drawdown_limits(portfolio_state)
                    if halt_reason:
                        self.logger.warning(f"Drawdown limit breached: {halt_reason}", source_module=self._source_module)
                        # Create event payload
                        details = {
                            "current_drawdown": str(portfolio_state.get("total_drawdown_pct", "N/A")),
                            "max_allowed": str(self._max_total_drawdown_pct),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        # Publish PotentialHaltTriggerEvent
                        halt_event = PotentialHaltTriggerEvent(
                            source_module=self._source_module,
                            event_id=uuid.uuid4(),
                            timestamp=datetime.utcnow(),
                            reason=halt_reason
                        )
                        await self.pubsub.publish(halt_event)

            except asyncio.CancelledError:
                self.logger.info("Periodic risk check task cancelled.", source_module=self._source_module)
                break
            except Exception as e:
                self.logger.error(f"Error in periodic risk check loop: {e}", exc_info=True, source_module=self._source_module)
                await asyncio.sleep(self._check_interval_s) # Avoid tight loop on error
                
        self.logger.info("Stopped periodic risk checks.", source_module=self._source_module)

    def _get_portfolio_state(self) -> Optional[Dict[str, Any]]:
        """Get portfolio state from the portfolio manager."""
        try:
            portfolio_state = self._portfolio_manager.get_current_state()
            if not portfolio_state:
                self.logger.error(
                    "Failed to retrieve portfolio state from PortfolioManager.",
                    source_module=self._source_module,
                )
                return None
            return portfolio_state
        except Exception as exc:
            self.logger.error(
                f"Exception calling PortfolioManager.get_current_state(): {exc}",
                source_module=self._source_module,
                exc_info=True,
            )
            return None

    def _validate_portfolio_state_values(
        self,
        portfolio_state: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Decimal]], Optional[str]]:
        """Extract and validate necessary values from portfolio state."""
        try:
            state_values = {
                "current_equity": Decimal(portfolio_state["total_equity"]),
                "total_dd": Decimal(portfolio_state["total_drawdown_pct"]),
                "daily_dd": Decimal(portfolio_state["daily_drawdown_pct"]),
                "weekly_dd": Decimal(portfolio_state["weekly_drawdown_pct"]),
            }
            return state_values, None
        except (KeyError, InvalidOperation, TypeError) as exc:
            self.logger.error(
                f"Error parsing portfolio state: {exc}. State: {portfolio_state}",
                source_module=self._source_module,
            )
            return None, "INVALID_PORTFOLIO_STATE"

    def _check_drawdown_limits(self, portfolio_state: Dict[str, Any]) -> Optional[str]:
        """Check if any drawdown limits are exceeded."""
        try:
            state_values, error = self._validate_portfolio_state_values(portfolio_state)
            if error or state_values is None:
                return None
                
            if state_values["total_dd"] > self._max_total_drawdown_pct:
                msg = (
                    f"MAX_TOTAL_DRAWDOWN_LIMIT ({state_values['total_dd']}% > {self._max_total_drawdown_pct}%)"
                )
                return msg
                
            if state_values["daily_dd"] > self._max_daily_drawdown_pct:
                msg = (
                    f"MAX_DAILY_DRAWDOWN_LIMIT ({state_values['daily_dd']}% > {self._max_daily_drawdown_pct}%)"
                )
                return msg
                
            if state_values["weekly_dd"] > self._max_weekly_drawdown_pct:
                msg = (
                    f"MAX_WEEKLY_DRAWDOWN_LIMIT ({state_values['weekly_dd']}% > {self._max_weekly_drawdown_pct}%)"
                )
                return msg
                
            return None
        except (KeyError, TypeError) as e:
            self.logger.error(f"Error checking drawdown limits: {e}", exc_info=True, source_module=self._source_module)
            return None

    def _perform_pre_trade_checks(
        self,
        event: TradeSignalProposedEvent
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Performs all pre-trade risk checks.
        Returns: (is_approved, rejection_reason, approved_payload_dict)
        """
        signal_id = event.signal_id
        self.logger.debug(
            f"Performing pre-trade checks for signal: {signal_id} ({event.side} {event.trading_pair})",
            source_module=self._source_module,
        )

        # Get and validate portfolio state
        portfolio_state = self._get_portfolio_state()
        if portfolio_state is None:
            return False, "PORTFOLIO_STATE_UNAVAILABLE", None

        # Extract and validate state values
        state_values, error = self._validate_portfolio_state_values(portfolio_state)
        if error or state_values is None:
            return False, error, None

        # Check drawdown limits
        drawdown_error = self._check_drawdown_limits(portfolio_state)
        if drawdown_error:
            return False, drawdown_error, None

        # Validate and get prices
        entry_price_str = self._get_entry_reference_price(event)
        if not entry_price_str:
            return False, "MISSING_ENTRY_REFERENCE_PRICE", None
            
        # Check if sl_price exists as an attribute or use proposed_sl_price
        sl_price_str = None
        if hasattr(event, 'sl_price') and event.sl_price:
            sl_price_str = event.sl_price
        elif hasattr(event, 'proposed_sl_price') and event.proposed_sl_price:
            sl_price_str = event.proposed_sl_price
            
        if not sl_price_str:
            return False, "MISSING_SL_PRICE", None
            
        try:
            entry_price = Decimal(entry_price_str)
            sl_price = Decimal(sl_price_str)
        except (InvalidOperation, ValueError, TypeError) as e:
            error_msg = f"Invalid price format: Entry={entry_price_str}, SL={sl_price_str}. Error: {e}"
            self.logger.error(error_msg, source_module=self._source_module)
            return False, "INVALID_PRICE_FORMAT", None

        # Validate stop loss price
        sl_validation_error = self._validate_sl_price(signal_id, event.side, entry_price, sl_price)
        if sl_validation_error:
            return False, sl_validation_error, None

        # Calculate position size
        calculated_qty = self._calculate_position_size(
            state_values["current_equity"],
            self._risk_per_trade_pct,
            entry_price,
            sl_price
        )
        if calculated_qty is None or calculated_qty <= 0:
            return False, "POSITION_SIZE_CALCULATION_FAILED", None

        # Check trade-level limits
        estimated_trade_value_quote = calculated_qty * entry_price
        trade_value_pct = (
            estimated_trade_value_quote / state_values["current_equity"]
        ) * 100

        if trade_value_pct > self._max_single_position_pct:
            msg = f"MAX_SINGLE_POSITION_LIMIT ({self._max_single_position_pct}%)"
            self.logger.warning(f"Signal {signal_id} rejected: {msg}", source_module=self._source_module)
            return False, msg, None

        # Prepare approved payload
        qty_str = str(calculated_qty)
        self.logger.info(f"Signal {signal_id} approved. Calculated Qty: {qty_str}", source_module=self._source_module)

        approved_payload = {
            "signal_id": str(signal_id),
            "trading_pair": event.trading_pair,
            "exchange": event.exchange,
            "side": event.side,
            "order_type": event.entry_type,
            "quantity": qty_str,
            "limit_price": event.proposed_entry_price if hasattr(event, 'proposed_entry_price') else None,
            "sl_price": sl_price_str,
            "tp_price": event.tp_price if hasattr(event, 'tp_price') else None,
            "risk_parameters": {
                "risk_per_trade_pct": str(self._risk_per_trade_pct),
                "calculated_qty": qty_str,
                "entry_ref_price": entry_price_str,
                "sl_price": sl_price_str,
                "equity_at_check": str(state_values["current_equity"]),
            },
        }
        return True, None, approved_payload

    def _get_entry_reference_price(self, event: TradeSignalProposedEvent) -> Optional[str]:
        """Get the entry reference price based on order type."""
        if hasattr(event, 'proposed_entry_price') and event.proposed_entry_price:
            # Convert Decimal to string before returning
            return str(event.proposed_entry_price)
            
        # For market orders, try to get price from associated data
        # This depends on your TradeSignalProposedEvent structure
        return None

    def _validate_sl_price(
        self,
        signal_id: uuid.UUID,
        side: str,
        entry_price: Decimal,
        sl_price: Decimal
    ) -> Optional[str]:
        """
        Validate stop loss price relative to entry price.

        Args:
            signal_id: The ID of the trade signal
            side: Trade side ('BUY' or 'SELL')
            entry_price: The entry price
            sl_price: The stop loss price

        Returns:
            Error message if validation fails, None if successful
        """
        if side == "BUY" and sl_price >= entry_price:
            msg = f"INVALID_SL_PRICE (SL {sl_price} >= Entry {entry_price} for BUY)"
            self.logger.warning(f"Signal {signal_id} rejected: {msg}", source_module=self._source_module)
            return msg

        if side == "SELL" and sl_price <= entry_price:
            msg = f"INVALID_SL_PRICE (SL {sl_price} <= Entry {entry_price} for SELL)"
            self.logger.warning(f"Signal {signal_id} rejected: {msg}", source_module=self._source_module)
            return msg

        price_diff_pct = (abs(entry_price - sl_price) / entry_price) * 100 if entry_price > 0 else Decimal(0)

        if price_diff_pct < self._min_sl_distance_pct:
            msg = f"SL_TOO_CLOSE ({price_diff_pct:.4f}% < {self._min_sl_distance_pct}%)"
            self.logger.warning(f"Signal {signal_id} rejected: {msg}", source_module=self._source_module)
            return msg

        return None

    def _calculate_position_size(
        self,
        current_equity: Decimal,
        risk_per_trade_pct: Decimal,
        entry_price: Decimal,
        sl_price: Decimal,
    ) -> Optional[Decimal]:
        """
        Calculates position size using fixed fractional risk based on equity.
        Assumes prices are in quote currency per unit of base currency.
        Returns quantity in base currency.
        """
        if current_equity <= 0:
            self.logger.warning(
                "Cannot calculate position size: Equity is zero or negative.",
                source_module=self._source_module,
            )
            return None
            
        if risk_per_trade_pct <= 0:
            self.logger.warning(
                "Cannot calculate position size: Risk per trade is zero or negative.",
                source_module=self._source_module,
            )
            return None

        risk_amount_quote = current_equity * (risk_per_trade_pct / 100)
        price_diff_per_unit = abs(entry_price - sl_price)

        if price_diff_per_unit <= 0:
            self.logger.error(
                f"Cannot calculate position size: Entry ({entry_price}) and SL ({sl_price}) "
                "prices are identical or invalid.",
                source_module=self._source_module,
            )
            return None

        # Quantity (Base) = (Amount to Risk in Quote) / (Risk per Unit in Quote)
        quantity = risk_amount_quote / price_diff_per_unit

        if quantity <= 0:
            self.logger.warning(
                f"Calculated position size is zero or negative ({quantity}) "
                f"after potential rounding. RiskAmt={risk_amount_quote}, PriceDiff={price_diff_per_unit}",
                source_module=self._source_module,
            )
            return None

        self.logger.debug(
            f"Calculated size: Qty={quantity} (Equity={current_equity}, Risk%={risk_per_trade_pct}, "
            f"RiskAmt={risk_amount_quote}, Entry={entry_price}, SL={sl_price}, Diff={price_diff_per_unit})",
            source_module=self._source_module,
        )
        return quantity

    async def _publish_trade_signal_approved(self, approved_payload_dict: Dict[str, Any]) -> None:
        """Constructs and publishes the approved trade signal event."""
        try:
            # Make sure sl_price exists - it should be guaranteed by _perform_pre_trade_checks
            if not approved_payload_dict.get("sl_price"):
                self.logger.error("Cannot publish approved signal: missing sl_price", source_module=self._source_module)
                return
                
            # Ensure we have a tp_price even if None was provided in payload
            tp_price_value = None
            if approved_payload_dict.get("tp_price"):
                tp_price_value = Decimal(approved_payload_dict["tp_price"])
            else:
                # Using a default tp_price based on entry price (maybe 2x the risk)
                if approved_payload_dict.get("limit_price"):
                    entry_price = Decimal(approved_payload_dict["limit_price"])
                else:
                    entry_price = Decimal(approved_payload_dict["risk_parameters"]["entry_ref_price"])
                    
                sl_price = Decimal(approved_payload_dict["sl_price"])
                price_diff = abs(entry_price - sl_price)
                
                # Set TP at 2x the risk distance or some other reasonable default
                if approved_payload_dict["side"].upper() == "BUY":
                    tp_price_value = entry_price + (price_diff * 2)
                else:
                    tp_price_value = entry_price - (price_diff * 2)
            
            # Now create the event with guaranteed values
            event = TradeSignalApprovedEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal_id=uuid.UUID(approved_payload_dict["signal_id"]),
                trading_pair=approved_payload_dict["trading_pair"],
                exchange=approved_payload_dict["exchange"],
                side=approved_payload_dict["side"],
                order_type=approved_payload_dict["order_type"],
                quantity=Decimal(approved_payload_dict["quantity"]),
                limit_price=Decimal(approved_payload_dict["limit_price"]) if approved_payload_dict.get("limit_price") else None,
                sl_price=Decimal(approved_payload_dict["sl_price"]),  # This is now guaranteed to exist
                tp_price=tp_price_value  # This is now guaranteed to have a value
            )
            
            await self.pubsub.publish(event)
            self.logger.info(f"Published TRADE_SIGNAL_APPROVED: {approved_payload_dict['signal_id']}", source_module=self._source_module)
        except Exception as e:
            self.logger.error(f"Error publishing approved signal: {e}", source_module=self._source_module, exc_info=True)

    async def _publish_trade_signal_rejected(self, rejected_payload_dict: Dict[str, Any]) -> None:
        """Constructs and publishes the rejected trade signal event."""
        try:
            event = TradeSignalRejectedEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal_id=uuid.UUID(rejected_payload_dict["signal_id"]) if isinstance(rejected_payload_dict["signal_id"], str) else rejected_payload_dict["signal_id"],
                trading_pair=rejected_payload_dict["trading_pair"],
                exchange=rejected_payload_dict["exchange"],
                side=rejected_payload_dict["side"],
                reason=rejected_payload_dict["reason"]
            )
            await self.pubsub.publish(event)
            self.logger.warning(
                f"Published TRADE_SIGNAL_REJECTED: {rejected_payload_dict['signal_id']}, Reason: {rejected_payload_dict['reason']}",
                source_module=self._source_module
            )
        except Exception as e:
            self.logger.error(f"Error publishing rejected signal: {e}", source_module=self._source_module, exc_info=True)
