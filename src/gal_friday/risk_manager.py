# Risk Manager Module
"""Risk management module for trading operations.

This module provides risk management functionality for trading operations,
including position sizing, drawdown limits, and trade validation.
"""

import asyncio
import time  # Added for retry delay
import uuid
from dataclasses import dataclass  # Added import
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

# Event Definitions
from .core.events import (
    EventType,
    ExecutionReportEvent,
    PotentialHaltTriggerEvent,
    TradeSignalApprovedEvent,
    TradeSignalProposedEvent,
    TradeSignalRejectedEvent,
)

# Import PubSubManager
from .core.pubsub import PubSubManager

# Import logger service
from .logger_service import LoggerService


# Custom exceptions
class RiskManagerError(Exception):
    """Custom exception for risk management errors.

    Used to indicate errors in risk management operations, such as
    invalid configurations or trade validation failures.
    """

    pass


# Type hint for PortfolioManager without circular import
if TYPE_CHECKING:
    from .market_price_service import MarketPriceService
    from .portfolio_manager import PortfolioManager
else:
    # Define placeholder or attempt runtime import carefully
    try:
        from .portfolio_manager import PortfolioManager
    except ImportError:
        # Define a minimal placeholder if import fails at runtime
        # This allows basic script execution but will fail if methods are called
        class PortfolioManager:  # type: ignore
            """Placeholder for PortfolioManager when not available at runtime.

            Provides minimal implementations of methods used by RiskManager.
            """

            def get_current_state(self) -> Dict[str, Any]:
                """Return empty portfolio state dictionary.

                Returns
                -------
                    Empty dictionary representing portfolio state
                """
                return {}

            # Add other methods used by RiskManager if necessary, e.g.:
            # def get_asset_balance(self, asset_symbol: str) -> Decimal: return Decimal(0)
            # def get_trade_history(self, pair: str, limit: int) -> List[Any]: return []

    try:  # Added for MarketPriceService
        from .market_price_service import MarketPriceService
    except ImportError:

        class MarketPriceService:  # type: ignore
            """Placeholder for MarketPriceService when not available at runtime.

            Provides minimal implementations of methods used by RiskManager.
            """

            async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
                """Return None as placeholder for latest price.

                Args
                ----
                    trading_pair: The trading pair to get price for

                Returns
                -------
                    None as placeholder
                """
                return None

            async def convert_amount(
                self, from_amount: Decimal, from_currency: str, to_currency: str
            ) -> Optional[Decimal]:
                """Return None as placeholder for currency conversion.

                Args
                ----
                    from_amount: Amount to convert
                    from_currency: Source currency
                    to_currency: Target currency

                Returns
                -------
                    None as placeholder
                """
                return None


# Set Decimal precision
# getcontext().prec = 28 # Removed global precision setting


# --- Event Payloads ---
@dataclass
class TradeSignalProposedPayload:
    """Payload for trade signal proposals."""

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
    """Payload for system halt events."""

    reason: str
    details: Dict[str, Any]


# --- RiskManager Class ---
class RiskManager:
    """Assess trade signals against risk parameters and portfolio state.

    Consumes proposed trade signals, performs pre-trade risk checks against
    portfolio state, and publishes approved/rejected signals or triggers HALT.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pubsub_manager: PubSubManager,
        portfolio_manager: "PortfolioManager",
        logger_service: LoggerService,
        market_price_service: "MarketPriceService",
    ):
        """Initialize the RiskManager with configuration and dependencies.

        Args
        ----
            config: Configuration settings.
            pubsub_manager: The application PubSubManager instance.
            portfolio_manager: The PortfolioManager instance.
            logger_service: Shared logger instance.
            market_price_service: MarketPriceService instance.
        """
        self._config = config.get("risk_manager", {})
        self.pubsub = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self._market_price_service = market_price_service
        self.logger = logger_service
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self._periodic_check_task: Optional[asyncio.Task] = None
        self._source_module = self.__class__.__name__

        # Store handler for unsubscribing
        self._signal_proposal_handler = self._handle_trade_signal_proposed
        self._exec_report_handler = self._handle_execution_report_for_losses

        # State for consecutive losses
        self._consecutive_loss_count: int = 0

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load risk parameters from configuration.

        Extracts and initializes risk limits, position sizing parameters,
        and other configuration settings used for risk checks.
        """
        limits = self._config.get("limits", {})
        self._max_total_drawdown_pct = Decimal(str(limits.get("max_total_drawdown_pct", 15.0)))
        self._max_daily_drawdown_pct = Decimal(str(limits.get("max_daily_drawdown_pct", 2.0)))
        self._max_weekly_drawdown_pct = Decimal(str(limits.get("max_weekly_drawdown_pct", 5.0)))
        self._max_consecutive_losses = int(limits.get("max_consecutive_losses", 5))
        self._max_exposure_per_asset_pct = Decimal(
            str(limits.get("max_exposure_per_asset_pct", 10.0))
        )
        self._max_total_exposure_pct = Decimal(str(limits.get("max_total_exposure_pct", 25.0)))
        self._max_order_size_usd = Decimal(str(limits.get("max_order_size_usd", 10000)))
        self._risk_per_trade_pct = Decimal(
            str(self._config.get("sizing", {}).get("risk_per_trade_pct", 0.5))
        )
        # Added default_tp_rr_ratio from config or defaults to 2.0
        self._default_tp_rr_ratio = Decimal(
            str(self._config.get("sizing", {}).get("default_tp_rr_ratio", "2.0"))
        )
        self._check_interval_s = int(self._config.get("check_interval_s", 60))
        self._min_sl_distance_pct = Decimal(str(self._config.get("min_sl_distance_pct", 0.01)))
        self._max_single_position_pct = Decimal(
            str(self._config.get("max_single_position_pct", 100.0))
        )
        # New config values for pre-trade checks
        self._fat_finger_max_deviation_pct = Decimal(
            str(self._config.get("fat_finger_max_deviation_pct", "5.0"))
        )
        self._exchange_taker_fee_pct = Decimal(  # General exchange fee
            str(self._config.get("exchange", {}).get("taker_fee_pct", "0.26"))
        )
        # Portfolio valuation currency (e.g., "USD")
        self._valuation_currency = str(self._config.get("portfolio_valuation_currency", "USD"))

        self.logger.info("RiskManager configured.", source_module=self._source_module)
        self._validate_config()  # Added call to validate_config

    def _validate_config(self) -> None:
        """Validate risk parameters from configuration.

        Checks that all required parameters are present and have valid values.
        Raises RiskManagerError if validation fails.
        """
        config_errors = []

        # Helper to check percentage values
        def check_percentage(
            value: Decimal,
            name: str,
            lower_bound: Decimal = Decimal("0"),
            upper_bound: Decimal = Decimal("100"),
        ) -> None:
            if not (lower_bound <= value <= upper_bound):
                config_errors.append(
                    f"{name} ({value}%) is outside the valid range [{lower_bound}-{upper_bound}]."
                )

        check_percentage(self._max_total_drawdown_pct, "max_total_drawdown_pct")
        check_percentage(self._max_daily_drawdown_pct, "max_daily_drawdown_pct")
        check_percentage(self._max_weekly_drawdown_pct, "max_weekly_drawdown_pct")
        check_percentage(self._max_exposure_per_asset_pct, "max_exposure_per_asset_pct")
        check_percentage(self._max_total_exposure_pct, "max_total_exposure_pct")
        # risk_per_trade_pct typically small, but can be up to 100
        check_percentage(
            self._risk_per_trade_pct, "risk_per_trade_pct", upper_bound=Decimal("100")
        )
        # min_sl_distance_pct must be > 0
        check_percentage(
            self._min_sl_distance_pct, "min_sl_distance_pct", lower_bound=Decimal("0.001")
        )
        check_percentage(self._max_single_position_pct, "max_single_position_pct")

        if self._default_tp_rr_ratio <= 0:
            config_errors.append(
                f"default_tp_rr_ratio ({self._default_tp_rr_ratio}) must be positive."
            )

        if self._max_consecutive_losses < 1:
            config_errors.append(
                f"max_consecutive_losses ({self._max_consecutive_losses}) must be at least 1."
            )

        if config_errors:
            for error_msg in config_errors:
                self.logger.error(
                    f"Configuration Error: {error_msg}", source_module=self._source_module
                )
            # Log critical message and continue as per whiteboard,
            # rather than raising an exception.
            self.logger.critical(
                f"RiskManager has configuration errors. Review settings: "
                f"{'; '.join(config_errors)}",
                source_module=self._source_module,
            )
        else:
            self.logger.info(
                "RiskManager configuration validated successfully.",
                source_module=self._source_module,
            )

    async def start(self) -> None:
        """Start the risk manager.

        Subscribes to trade signal proposals and execution reports.
        Starts periodic portfolio checks.
        """
        if self._is_running:
            self.logger.warning("RiskManager already running.", source_module=self._source_module)
            return
        self._is_running = True

        # Subscribe to proposed signals
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_PROPOSED, self._signal_proposal_handler)
        # Subscribe to execution reports for loss tracking
        self.pubsub.subscribe(EventType.EXECUTION_REPORT, self._exec_report_handler)

        # Start periodic checks
        if self._check_interval_s > 0:
            self._periodic_check_task = asyncio.create_task(self._run_periodic_checks())

        self.logger.info("RiskManager started.", source_module=self._source_module)

    async def stop(self) -> None:
        """Stop the risk manager.

        Unsubscribes from events and stops periodic checks.
        """
        if not self._is_running:
            return
        self._is_running = False

        # Unsubscribe
        try:
            self.pubsub.unsubscribe(EventType.TRADE_SIGNAL_PROPOSED, self._signal_proposal_handler)
            self.pubsub.unsubscribe(EventType.EXECUTION_REPORT, self._exec_report_handler)
            self.logger.info(
                "Unsubscribed from TRADE_SIGNAL_PROPOSED and EXECUTION_REPORT.",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.error(
                f"Error unsubscribing RiskManager: {e}",
                exc_info=True,
                source_module=self._source_module,
            )

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
        """Handle a proposed trade signal.

        Performs risk checks and publishes approval/rejection.

        Args
        ----
            event: The proposed trade signal event
        """
        if not isinstance(event, TradeSignalProposedEvent):
            self.logger.warning(
                f"Received non-TradeSignalProposedEvent: {type(event)}",
                source_module=self._source_module,
            )
            return

        if not self._is_running:
            return  # Don't process if stopped

        self.logger.debug(f"Received trade signal proposal: {event.signal_id}")

        # Perform checks (now awaited)
        is_approved, reason, approved_payload_data = await self._perform_pre_trade_checks(event)

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
        """Run periodic portfolio risk checks.

        Checks drawdown limits and other risk parameters at regular intervals.
        """
        self.logger.info("Starting periodic risk checks.", source_module=self._source_module)
        while self._is_running:
            try:
                await asyncio.sleep(self._check_interval_s)
                if not self._is_running:
                    break  # Check again after sleep

                self.logger.debug(
                    "Running periodic risk check...", source_module=self._source_module
                )
                portfolio_state = self._get_portfolio_state()

                if portfolio_state is not None:
                    halt_reason = self._check_drawdown_limits(portfolio_state)
                    if halt_reason:
                        self.logger.warning(
                            f"Drawdown limit breached: {halt_reason}",
                            source_module=self._source_module,
                        )
                        # Create halt event without using details variable
                        # Publish PotentialHaltTriggerEvent
                        halt_event = PotentialHaltTriggerEvent(
                            source_module=self._source_module,
                            event_id=uuid.uuid4(),
                            timestamp=datetime.utcnow(),
                            reason=halt_reason,
                        )
                        await self.pubsub.publish(halt_event)

            except asyncio.CancelledError:
                self.logger.info(
                    "Periodic risk check task cancelled.", source_module=self._source_module
                )
                break
            except Exception as e:
                self.logger.error(
                    f"Error in periodic risk check loop: {e}",
                    exc_info=True,
                    source_module=self._source_module,
                )
                # Avoid tight loop on error
                await asyncio.sleep(self._check_interval_s)

        self.logger.info("Stopped periodic risk checks.", source_module=self._source_module)

    def _get_portfolio_state(self) -> Optional[Dict[str, Any]]:
        """Get current portfolio state.

        Returns
        -------
            Dictionary with portfolio state or None if unavailable
        """
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

    def _get_portfolio_state_with_retry(
        self, max_retries: int = 2, retry_delay_s: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """Get portfolio state with retries on failure.

        Args
        ----
            max_retries: Maximum number of retry attempts
            retry_delay_s: Delay between retries in seconds

        Returns
        -------
            Dictionary with portfolio state or None if unavailable
        """
        for attempt in range(max_retries + 1):
            try:
                state = self._portfolio_manager.get_current_state()
                if state:  # Check if state is not None or empty
                    return state
                # Log warning if state is None/empty even without exception
                self.logger.warning(
                    f"PortfolioManager returned empty/None state "
                    f"(Attempt {attempt + 1}/{max_retries + 1})",
                    source_module=self._source_module,
                )
            except Exception as e:
                self.logger.warning(
                    f"Error getting portfolio state "
                    f"(Attempt {attempt + 1}/{max_retries + 1}): {e}",
                    source_module=self._source_module,
                )
            # Wait before retrying (use time.sleep as this method is synchronous)
            if attempt < max_retries:
                time.sleep(retry_delay_s)

        self.logger.error(
            f"Failed to get valid portfolio state after {max_retries + 1} attempts.",
            source_module=self._source_module,
        )
        return None

    def _validate_portfolio_state_values(
        self, portfolio_state: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Decimal]], Optional[str]]:
        """Validate portfolio state values.

        Args
        ----
            portfolio_state: Portfolio state dictionary to validate

        Returns
        -------
            Tuple of (validated state dict, error message) where error is None if valid
        """
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
        """Check if drawdown limits are exceeded.

        Args
        ----
            portfolio_state: Current portfolio state

        Returns
        -------
            Error message if limits exceeded, None otherwise
        """
        try:
            state_values, error = self._validate_portfolio_state_values(portfolio_state)
            if error or state_values is None:
                return None

            if state_values["total_dd"] > self._max_total_drawdown_pct:
                msg = (
                    f"MAX_TOTAL_DRAWDOWN_LIMIT ({state_values['total_dd']}% > "
                    f"{self._max_total_drawdown_pct}%)"
                )
                return msg

            if state_values["daily_dd"] > self._max_daily_drawdown_pct:
                msg = (
                    f"MAX_DAILY_DRAWDOWN_LIMIT ({state_values['daily_dd']}% > "
                    f"{self._max_daily_drawdown_pct}%)"
                )
                return msg

            if state_values["weekly_dd"] > self._max_weekly_drawdown_pct:
                msg = (
                    f"MAX_WEEKLY_DRAWDOWN_LIMIT ({state_values['weekly_dd']}% > "
                    f"{self._max_weekly_drawdown_pct}%)"
                )
                return msg

            return None
        except (KeyError, TypeError) as e:
            self.logger.error(
                f"Error checking drawdown limits: {e}",
                exc_info=True,
                source_module=self._source_module,
            )
            return None

    async def _perform_pre_trade_checks(
        self, event: TradeSignalProposedEvent
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Perform pre-trade risk checks.

        Args
        ----
            event: The proposed trade signal event

        Returns
        -------
            Tuple of (passed, error_message, trade_params)
        """
        signal_id = event.signal_id
        self.logger.debug(
            f"Performing pre-trade checks for signal: {signal_id} "
            f"({event.side} {event.trading_pair})",
            source_module=self._source_module,
        )

        # Validate portfolio state
        passed, error, portfolio_state, state_values = await self._validate_portfolio_for_trade(
            event
        )
        if not passed:
            return False, error, None

        if portfolio_state is None:
            return False, "PORTFOLIO_STATE_IS_NONE", None

        if state_values is None:
            return False, "STATE_VALUES_IS_NONE", None

        current_equity = state_values["current_equity"]

        # Validate price values and check fat finger risk
        passed, error, prices = await self._check_trade_prices(event, signal_id)
        if not passed:
            return False, error, None

        entry_price, sl_price = prices["entry_price"], prices["sl_price"]

        # Calculate position size
        calculated_qty = self._calculate_position_size(
            current_equity, self._risk_per_trade_pct, entry_price, sl_price
        )
        if calculated_qty is None or calculated_qty <= 0:
            return False, "POSITION_SIZE_CALCULATION_FAILED", None

        # Perform exposure and balance checks
        base_asset, quote_asset = self._split_symbol(event.trading_pair)
        trade_value_quote = calculated_qty * entry_price

        # Check exposure and balance
        passed, error, trade_value_valuation_ccy = await self._check_trade_exposure_and_balance(
            event, signal_id, current_equity, trade_value_quote, quote_asset, portfolio_state
        )
        if not passed:
            return False, error, None

        # Prepare approved payload
        qty_str = str(calculated_qty)
        self.logger.info(
            f"Signal {signal_id} approved. Calculated Qty: {qty_str}",
            source_module=self._source_module,
        )

        approved_payload = self._prepare_approved_payload(
            event, signal_id, qty_str, entry_price, sl_price, state_values
        )
        return True, None, approved_payload

    async def _check_trade_prices(
        self, event: TradeSignalProposedEvent, signal_id: uuid.UUID
    ) -> Tuple[bool, Optional[str], Dict[str, Decimal]]:
        """Check trade prices and validate them.

        Args
        ----
            event: The proposed trade signal event
            signal_id: The trade signal ID

        Returns
        -------
            Tuple of (passed, error_message, prices)
        """
        # Validate price values
        passed, error, prices = self._validate_trade_prices(event)
        if not passed:
            return False, error, {}

        entry_price, sl_price = prices["entry_price"], prices["sl_price"]

        # Check fat finger risk
        passed, error = await self._check_fat_finger_risk(event, signal_id, entry_price)
        if not passed:
            return False, error, {}

        # Validate stop loss price
        sl_validation_error = self._validate_sl_price(signal_id, event.side, entry_price, sl_price)
        if sl_validation_error:
            return False, sl_validation_error, {}

        return True, None, prices

    async def _check_trade_exposure_and_balance(
        self,
        event: TradeSignalProposedEvent,
        signal_id: uuid.UUID,
        current_equity: Decimal,
        trade_value_quote: Decimal,
        quote_asset: str,
        portfolio_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Decimal]]:
        """Check portfolio exposure, balance, and other limits.

        Args
        ----
            event: The proposed trade signal event
            signal_id: The trade signal ID
            current_equity: Current portfolio equity
            trade_value_quote: Trade value in quote currency
            quote_asset: Quote asset symbol
            portfolio_state: Portfolio state dictionary

        Returns
        -------
            Tuple of (passed, error_message, trade_value_in_valuation_currency)
        """
        # Check portfolio exposure
        passed, error, trade_value_valuation_ccy = await self._check_portfolio_exposure(
            event, signal_id, current_equity, trade_value_quote, quote_asset, portfolio_state
        )
        if not passed:
            return False, error, None

        # Check sufficient balance
        if event.side.upper() == "BUY":
            if portfolio_state is None:
                return False, "PORTFOLIO_STATE_IS_NONE", None

            passed, error = self._check_sufficient_balance(
                signal_id, trade_value_quote, quote_asset, portfolio_state
            )
            if not passed:
                return False, error, None

        # Check consecutive losses
        loss_check_ok, loss_reason = self._check_consecutive_losses()
        if not loss_check_ok:
            return False, loss_reason, None

        # Check single position limit
        if trade_value_valuation_ccy is None:
            return False, "TRADE_VALUE_IN_VALUATION_CURRENCY_IS_NONE", None

        passed, error = self._check_single_position_limit(
            signal_id, trade_value_valuation_ccy, current_equity
        )
        if not passed:
            return False, error, None

        return True, None, trade_value_valuation_ccy

    async def _validate_portfolio_for_trade(
        self, event: TradeSignalProposedEvent
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Decimal]]]:
        """Validate portfolio state for trade.

        Args
        ----
            event: The proposed trade signal event

        Returns
        -------
            Tuple of (passed, error_message, portfolio_state, state_values)
        """
        portfolio_state = self._get_portfolio_state_with_retry()
        if portfolio_state is None:
            return False, "PORTFOLIO_STATE_UNAVAILABLE_AFTER_RETRIES", None, None

        state_values, error = self._validate_portfolio_state_values(portfolio_state)
        if error or state_values is None:
            return False, error, None, None

        drawdown_error = self._check_drawdown_limits(portfolio_state)
        if drawdown_error:
            return False, drawdown_error, None, None

        return True, None, portfolio_state, state_values

    def _validate_trade_prices(
        self, event: TradeSignalProposedEvent
    ) -> Tuple[bool, Optional[str], Dict[str, Decimal]]:
        """Validate trade entry and stop loss prices.

        Args
        ----
            event: The proposed trade signal event

        Returns
        -------
            Tuple of (passed, error_message, prices_dict)
        """
        entry_price_str = self._get_entry_reference_price(event)
        if not entry_price_str:
            return False, "MISSING_ENTRY_REFERENCE_PRICE", {}

        sl_price_str = None
        if hasattr(event, "sl_price") and event.sl_price:
            sl_price_str = str(event.sl_price)
        elif hasattr(event, "proposed_sl_price") and event.proposed_sl_price:
            sl_price_str = str(event.proposed_sl_price)

        if not sl_price_str:
            return False, "MISSING_SL_PRICE", {}

        try:
            entry_price = Decimal(entry_price_str)
            sl_price = Decimal(sl_price_str)
            return (
                True,
                None,
                {
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                },
            )
        except (InvalidOperation, ValueError, TypeError) as e:
            self.logger.error(
                f"Invalid price format: Entry={entry_price_str}, SL={sl_price_str}. Error: {e}",
                source_module=self._source_module,
            )
            return False, "INVALID_PRICE_FORMAT", {}

    async def _check_fat_finger_risk(
        self, event: TradeSignalProposedEvent, signal_id: uuid.UUID, entry_price: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """Check for fat finger mistakes in the entry price.

        Args
        ----
            event: The proposed trade signal event
            signal_id: The trade signal ID
            entry_price: The trade entry price

        Returns
        -------
            Tuple of (passed, error_message)
        """
        if not self._market_price_service:
            self.logger.warning(
                "MarketPriceService not available, skipping fat finger check.",
                source_module=self._source_module,
            )
            return True, None

        current_market_price = await self._market_price_service.get_latest_price(
            event.trading_pair
        )

        if current_market_price is None:
            self.logger.warning(
                f"Signal {signal_id}: Market price unavailable for fat finger check on "
                f"{event.trading_pair}. Check skipped.",
                source_module=self._source_module,
            )
            return True, None

        if current_market_price <= 0:
            self.logger.warning(
                f"Signal {signal_id}: Current market price for {event.trading_pair} "
                f"is {current_market_price}, skipping fat finger check.",
                source_module=self._source_module,
            )
            return True, None

        deviation_pct = abs(entry_price - current_market_price) / current_market_price * 100
        if deviation_pct > self._fat_finger_max_deviation_pct:
            reason = (
                f"FAT_FINGER_CHECK_FAILED ({deviation_pct:.2f}% > "
                f"{self._fat_finger_max_deviation_pct}%)"
            )
            self.logger.warning(
                f"Signal {signal_id} rejected: {reason}", source_module=self._source_module
            )
            return False, reason

        return True, None

    async def _check_portfolio_exposure(
        self,
        event: TradeSignalProposedEvent,
        signal_id: uuid.UUID,
        current_equity: Decimal,
        trade_value_quote: Decimal,
        quote_asset: str,
        portfolio_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[Decimal]]:
        """Check portfolio exposure limits.

        Args
        ----
            event: The proposed trade signal event
            signal_id: The trade signal ID
            current_equity: Current portfolio equity
            trade_value_quote: Trade value in quote currency
            quote_asset: Quote asset symbol
            portfolio_state: Current portfolio state

        Returns
        -------
            Tuple of (passed, error_message, trade_value_in_valuation_currency)
        """
        trade_value_valuation_ccy, conversion_error = await self._convert_to_valuation_ccy(
            trade_value_quote, quote_asset
        )

        if conversion_error:
            return False, f"CURRENCY_CONVERSION_FAILED_FOR_EXPOSURE ({conversion_error})", None

        if trade_value_valuation_ccy is None:
            self.logger.error(
                f"Signal {signal_id}: Trade value in valuation ccy is None "
                f"post-conversion without error string. Critical logic flaw.",
                source_module=self._source_module,
            )
            return False, "EXPOSURE_CHECK_CCY_CONVERSION_UNEXPECTED_NONE", None

        if current_equity > 0:
            current_total_exposure_pct_str = portfolio_state.get("total_exposure_pct", "0")
            try:
                current_total_exposure_pct = Decimal(current_total_exposure_pct_str)
            except InvalidOperation:
                self.logger.error(
                    f"Signal {signal_id}: Invalid total_exposure_pct format: "
                    f"{current_total_exposure_pct_str}",
                    source_module=self._source_module,
                )
                return False, "INVALID_TOTAL_EXPOSURE_PCT_FORMAT", None

            new_exposure_increment_pct = (abs(trade_value_valuation_ccy) / current_equity) * 100
            new_total_exposure_pct = current_total_exposure_pct + new_exposure_increment_pct

            if new_total_exposure_pct > self._max_total_exposure_pct:
                reason = (
                    f"MAX_TOTAL_EXPOSURE_LIMIT ({new_total_exposure_pct:.2f}% > "
                    f"{self._max_total_exposure_pct}%)"
                )
                self.logger.warning(
                    f"Signal {signal_id} rejected: {reason}", source_module=self._source_module
                )
                return False, reason, None
        elif abs(trade_value_valuation_ccy) > 0:
            self.logger.warning(
                f"Signal {signal_id} rejected: Attempting to add exposure "
                f"with zero/negative equity ({current_equity}).",
                source_module=self._source_module,
            )
            return False, "EXPOSURE_ADD_WITH_ZERO_NEGATIVE_EQUITY", None

        return True, None, trade_value_valuation_ccy

    def _check_sufficient_balance(
        self,
        signal_id: uuid.UUID,
        trade_value_quote: Decimal,
        quote_asset: str,
        portfolio_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Check if there's sufficient balance for the trade.

        Args
        ----
            signal_id: The trade signal ID
            trade_value_quote: Trade value in quote currency
            quote_asset: Quote asset symbol
            portfolio_state: Current portfolio state

        Returns
        -------
            Tuple of (passed, error_message)
        """
        fee_rate = self._exchange_taker_fee_pct / 100
        estimated_cost_quote = trade_value_quote * (1 + fee_rate)

        available_quote_funds_str = portfolio_state.get("available_funds", {}).get(
            quote_asset, "0"
        )
        try:
            available_quote_funds = Decimal(available_quote_funds_str)
        except InvalidOperation:
            self.logger.error(
                f"Signal {signal_id}: Invalid available funds format for "
                f"{quote_asset}: {available_quote_funds_str}",
                source_module=self._source_module,
            )
            return False, f"INVALID_AVAILABLE_FUNDS_FORMAT_FOR_{quote_asset.upper()}"

        if available_quote_funds < estimated_cost_quote:
            reason = (
                f"INSUFFICIENT_FUNDS ({quote_asset}: {available_quote_funds:.4f} < "
                f"{estimated_cost_quote:.4f}) TradeValQuote: {trade_value_quote:.4f} "
                f"FeeRate: {fee_rate}"
            )
            self.logger.warning(
                f"Signal {signal_id} rejected: {reason}", source_module=self._source_module
            )
            return False, reason

        return True, None

    def _check_single_position_limit(
        self, signal_id: uuid.UUID, trade_value_valuation_ccy: Decimal, current_equity: Decimal
    ) -> Tuple[bool, Optional[str]]:
        """Check if trade exceeds single position limit.

        Args
        ----
            signal_id: The trade signal ID
            trade_value_valuation_ccy: Trade value in valuation currency
            current_equity: Current portfolio equity

        Returns
        -------
            Tuple of (passed, error_message)
        """
        if current_equity > 0:
            trade_value_vs_equity_pct = (abs(trade_value_valuation_ccy) / current_equity) * 100
            if trade_value_vs_equity_pct > self._max_single_position_pct:
                msg = (
                    f"MAX_SINGLE_POSITION_LIMIT ({trade_value_vs_equity_pct:.2f}% of equity > "
                    f"{self._max_single_position_pct}%). Value: {trade_value_valuation_ccy}, "
                    f"Equity: {current_equity}"
                )
                self.logger.warning(
                    f"Signal {signal_id} rejected: {msg}", source_module=self._source_module
                )
                return False, msg
        elif abs(trade_value_valuation_ccy) > 0:
            self.logger.warning(
                f"Signal {signal_id} rejected: Attempting position with zero/negative equity "
                f"for max_single_position_pct check.",
                source_module=self._source_module,
            )
            return False, "POSITION_WITH_ZERO_NEGATIVE_EQUITY"

        return True, None

    def _prepare_approved_payload(
        self,
        event: TradeSignalProposedEvent,
        signal_id: uuid.UUID,
        qty_str: str,
        entry_price: Decimal,
        sl_price: Decimal,
        state_values: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """Prepare payload for approved trade signal.

        Args
        ----
            event: The proposed trade signal event
            signal_id: The trade signal ID
            qty_str: Position quantity as string
            entry_price: Entry price
            sl_price: Stop loss price
            state_values: Portfolio state values

        Returns
        -------
            Approved payload dictionary
        """
        # These string conversions are used in the returned payload
        sl_price_str = str(sl_price)

        return {
            "signal_id": str(signal_id),
            "trading_pair": event.trading_pair,
            "exchange": event.exchange,
            "side": event.side,
            "order_type": event.entry_type,
            "quantity": qty_str,
            "limit_price": (
                event.proposed_entry_price if hasattr(event, "proposed_entry_price") else None
            ),
            "sl_price": sl_price_str,
            "tp_price": event.tp_price if hasattr(event, "tp_price") else None,
            "risk_parameters": {
                "risk_per_trade_pct": self._risk_per_trade_pct,
                "calculated_qty": Decimal(qty_str),
                "entry_ref_price": entry_price,
                "sl_price": sl_price,
                "equity_at_check": state_values["current_equity"],
            },
        }

    def _get_entry_reference_price(self, event: TradeSignalProposedEvent) -> Optional[str]:
        """Get reference price for entry.

        Args
        ----
            event: The proposed trade signal event

        Returns
        -------
            Reference price as string or None if unavailable
        """
        if hasattr(event, "proposed_entry_price") and event.proposed_entry_price:
            # Convert Decimal to string before returning
            return str(event.proposed_entry_price)

        # For market orders, try to get price from associated data
        # This depends on your TradeSignalProposedEvent structure
        return None

    def _validate_sl_price(
        self, signal_id: uuid.UUID, side: str, entry_price: Decimal, sl_price: Decimal
    ) -> Optional[str]:
        """Validate stop loss price.

        Args
        ----
            signal_id: Trade signal ID
            side: Trade side (BUY/SELL)
            entry_price: Entry price
            sl_price: Stop loss price

        Returns
        -------
            Error message if invalid, None if valid
        """
        if side == "BUY" and sl_price >= entry_price:
            msg = f"INVALID_SL_PRICE (SL {sl_price} >= Entry {entry_price} for BUY)"
            self.logger.warning(
                f"Signal {signal_id} rejected: {msg}", source_module=self._source_module
            )
            return msg

        if side == "SELL" and sl_price <= entry_price:
            msg = f"INVALID_SL_PRICE (SL {sl_price} <= Entry {entry_price} for SELL)"
            self.logger.warning(
                f"Signal {signal_id} rejected: {msg}", source_module=self._source_module
            )
            return msg

        price_diff_pct = (
            (abs(entry_price - sl_price) / entry_price) * 100 if entry_price > 0 else Decimal(0)
        )

        if price_diff_pct < self._min_sl_distance_pct:
            msg = f"SL_TOO_CLOSE ({price_diff_pct:.4f}% < {self._min_sl_distance_pct}%)"
            self.logger.warning(
                f"Signal {signal_id} rejected: {msg}", source_module=self._source_module
            )
            return msg

        return None

    def _calculate_position_size(
        self,
        current_equity: Decimal,
        risk_per_trade_pct: Decimal,
        entry_price: Decimal,
        sl_price: Decimal,
    ) -> Optional[Decimal]:
        """Calculate position size based on risk parameters.

        Args
        ----
            current_equity: Current portfolio equity
            risk_per_trade_pct: Risk percentage per trade
            entry_price: Entry price
            sl_price: Stop loss price

        Returns
        -------
            Position size or None if calculation fails
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

        # Quantity (Base) = (Amount to Risk in Quote) / (Risk per Unit in
        # Quote)
        quantity = risk_amount_quote / price_diff_per_unit

        if quantity <= 0:
            self.logger.warning(
                f"Calculated position size is zero or negative ({quantity}) "
                f"after potential rounding. RiskAmt={risk_amount_quote}, "
                f"PriceDiff={price_diff_per_unit}",
                source_module=self._source_module,
            )
            return None

        self.logger.debug(
            f"Calculated size: Qty={quantity} (Equity={current_equity}, "
            f"Risk%={risk_per_trade_pct}, RiskAmt={risk_amount_quote}, "
            f"Entry={entry_price}, SL={sl_price}, Diff={price_diff_per_unit})",
            source_module=self._source_module,
        )
        return quantity

    async def _publish_trade_signal_approved(self, approved_payload_dict: Dict[str, Any]) -> None:
        """Publish trade signal approved event.

        Args
        ----
            approved_payload_dict: Approved trade signal payload
        """
        try:
            # Make sure sl_price exists - it should be guaranteed by
            # _perform_pre_trade_checks
            if not approved_payload_dict.get("sl_price"):
                self.logger.error(
                    "Cannot publish approved signal: missing sl_price",
                    source_module=self._source_module,
                )
                return

            # Ensure we have a tp_price even if None was provided in payload
            tp_price_value = None
            if approved_payload_dict.get("tp_price"):
                tp_price_value = Decimal(approved_payload_dict["tp_price"])
            else:
                # Using a default tp_price based on entry price
                # (using configured self._default_tp_rr_ratio)
                if approved_payload_dict.get("limit_price"):
                    entry_price = Decimal(approved_payload_dict["limit_price"])
                else:
                    entry_price = Decimal(
                        approved_payload_dict["risk_parameters"]["entry_ref_price"]
                    )

                sl_price = Decimal(approved_payload_dict["sl_price"])
                price_diff = abs(entry_price - sl_price)

                # Set TP at self._default_tp_rr_ratio times the risk distance
                if approved_payload_dict["side"].upper() == "BUY":
                    tp_price_value = entry_price + (
                        price_diff * self._default_tp_rr_ratio
                    )  # Use configured ratio
                else:
                    tp_price_value = entry_price - (
                        price_diff * self._default_tp_rr_ratio
                    )  # Use configured ratio

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
                limit_price=(
                    Decimal(approved_payload_dict["limit_price"])
                    if approved_payload_dict.get("limit_price")
                    else None
                ),
                sl_price=Decimal(approved_payload_dict["sl_price"]),
                tp_price=tp_price_value,
                risk_parameters=approved_payload_dict["risk_parameters"],  # Added missing argument
            )

            await self.pubsub.publish(event)
            self.logger.info(
                f"Published TRADE_SIGNAL_APPROVED: {approved_payload_dict['signal_id']}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.error(
                f"Error publishing approved signal: {e}",
                source_module=self._source_module,
                exc_info=True,
            )

    async def _publish_trade_signal_rejected(self, rejected_payload_dict: Dict[str, Any]) -> None:
        """Publish trade signal rejected event.

        Args
        ----
            rejected_payload_dict: Rejected trade signal payload
        """
        try:
            event = TradeSignalRejectedEvent(
                source_module=self._source_module,
                event_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                signal_id=(
                    uuid.UUID(rejected_payload_dict["signal_id"])
                    if isinstance(rejected_payload_dict["signal_id"], str)
                    else rejected_payload_dict["signal_id"]
                ),
                trading_pair=rejected_payload_dict["trading_pair"],
                exchange=rejected_payload_dict["exchange"],
                side=rejected_payload_dict["side"],
                reason=rejected_payload_dict["reason"],
            )
            await self.pubsub.publish(event)
            self.logger.warning(
                f"Published TRADE_SIGNAL_REJECTED: {rejected_payload_dict['signal_id']}, "
                f"Reason: {rejected_payload_dict['reason']}",
                source_module=self._source_module,
            )
        except Exception as e:
            self.logger.error(
                f"Error publishing rejected signal: {e}",
                source_module=self._source_module,
                exc_info=True,
            )

    # --- Helper Methods Needed by Pre-Trade Checks ---
    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        """Split trading pair symbol into base and quote.

        Args
        ----
            symbol: Trading pair symbol (e.g. "BTCUSD")

        Returns
        -------
            Tuple of (base_currency, quote_currency)
        """
        parts = symbol.split("/")
        if len(parts) == 2:
            return parts[0].upper(), parts[1].upper()
        if len(symbol) > 3:
            base = symbol[:-3]
            quote = symbol[-3:]
            if quote.upper() in [
                "USD",
                "USDT",
                "EUR",
                "GBP",
                "JPY",
                "BTC",
                "ETH",
                "BNB",
            ]:  # Expanded common quotes
                return base.upper(), quote.upper()
        self.logger.warning(
            f"Could not robustly split symbol: {symbol}. "
            f"Using fallback to {self._valuation_currency}.",
            source_module=self._source_module,
        )
        return symbol.upper(), self._valuation_currency

    async def _convert_to_valuation_ccy(
        self, amount: Decimal, currency: str
    ) -> Tuple[Optional[Decimal], Optional[str]]:
        """Convert amount to valuation currency.

        Args
        ----
            amount: Amount to convert
            currency: Source currency

        Returns
        -------
            Tuple of (converted_amount, error_message)
        """
        target_valuation_currency = self._valuation_currency.upper()
        if currency.upper() == target_valuation_currency:
            return amount, None

        # Ensure MarketPriceService is available
        if self._market_price_service is None:
            error_msg = "MarketPriceService not available for currency conversion."
            self.logger.error(error_msg, source_module=self._source_module)
            return None, error_msg

        converted_amount = await self._market_price_service.convert_amount(  # Await
            from_amount=amount,
            from_currency=currency.upper(),
            to_currency=target_valuation_currency,
        )
        if converted_amount is None:
            error_msg = (
                f"Failed to convert {amount} {currency} to {target_valuation_currency} "
                f"via MarketPriceService."
            )
            self.logger.error(error_msg, source_module=self._source_module)
            return None, error_msg
        return converted_amount, None

    # --- End Helper Methods ---

    # --- Methods for Consecutive Loss Tracking ---
    async def _handle_execution_report_for_losses(self, event: ExecutionReportEvent) -> None:
        """Handle execution report for loss tracking.

        Args
        ----
            event: Execution report event
        """
        if not self._is_running:
            return

        # Simplified logic: Assume event indicates a filled order relevant to a closed trade
        # and might have PnL. A real implementation
        # needs robust trade tracking from PortfolioManager
        # or a dedicated TradeClosedEvent.

        # Filter for relevant events, e.g.
        # 'FILLED' status if it implies trade closure or PnL realization.
        # This depends heavily on the structure and meaning of ExecutionReportEvent.
        # For this placeholder, we'll assume if 'realized_pnl' is present, it's relevant.

        realized_pnl_attr = getattr(event, "realized_pnl", None)

        if realized_pnl_attr is None:
            # self.logger.debug(
            #    f"ExecutionReport {event.order_id if hasattr(event, 'order_id') else 'N/A'} "
            #    f"has no 'realized_pnl', skipping for loss tracking.",
            #    source_module=self._source_module
            # )
            return  # Cannot determine PnL from this event alone

        try:
            realized_pnl = Decimal(str(realized_pnl_attr))
        except (InvalidOperation, ValueError, TypeError):
            self.logger.warning(
                f"Could not parse 'realized_pnl' ({realized_pnl_attr}) "
                f"from ExecutionReportEvent to Decimal.",
                source_module=self._source_module,
            )
            return

        if realized_pnl < 0:
            self._consecutive_loss_count += 1
            self.logger.info(
                f"Consecutive loss count increased to {self._consecutive_loss_count}. "
                f"PnL: {realized_pnl:.2f}",
                source_module=self._source_module,
            )

            # Check limit immediately and trigger halt if needed
            if self._consecutive_loss_count >= self._max_consecutive_losses:
                reason = (
                    f"Consecutive loss limit reached: {self._consecutive_loss_count} >= "
                    f"{self._max_consecutive_losses}"
                )
                self.logger.critical(
                    f"HALT Trigger Condition (from ExecutionReport): {reason}",
                    source_module=self._source_module,
                )
                halt_event = PotentialHaltTriggerEvent(
                    source_module=self._source_module,
                    event_id=uuid.uuid4(),
                    timestamp=datetime.utcnow(),
                    reason=reason,
                )
                await self.pubsub.publish(halt_event)  # Ensure await

        elif realized_pnl > 0:  # Reset on profit
            if self._consecutive_loss_count > 0:
                self.logger.info(
                    f"Consecutive loss streak broken (PnL: {realized_pnl:.2f}). "
                    f"Resetting count from {self._consecutive_loss_count}.",
                    source_module=self._source_module,
                )
                self._consecutive_loss_count = 0
        # else: Zero PnL, count doesn't change. Or could be treated as non-loss.

    def _check_consecutive_losses(self) -> Tuple[bool, Optional[str]]:
        """Check if consecutive loss limit is exceeded.

        Returns
        -------
            Tuple of (limit_exceeded, error_message)
        """
        if self._consecutive_loss_count >= self._max_consecutive_losses:
            reason = (
                f"MAX_CONSECUTIVE_LOSSES_LIMIT_REACHED ({self._consecutive_loss_count} >= "
                f"{self._max_consecutive_losses})"
            )
            self.logger.warning(f"Trade rejected: {reason}", source_module=self._source_module)
            # Publishing PotentialHaltTriggerEvent here as well, as per whiteboard, although
            # the _handle_execution_report_for_losses might have already done it.
            # This ensures a halt is signaled if a trade is proposed while already at the limit.
            # Consider if this could lead to duplicate halt signals if not managed carefully
            # by downstream.
            # For now, following the whiteboard.

            # To avoid awaiting in a sync method, we can't publish directly here
            # if pubsub.publish is async.
            # The whiteboard implies _check_consecutive_losses is called in
            # _perform_pre_trade_checks (async).
            # However, the _check_consecutive_losses itself is defined as sync.
            # For now, we will just return the reason. The halt event based on this check
            # should be published from the async _perform_pre_trade_checks if this check fails.
            # OR, if this check leads to a HALT, no new trades should be approved anyway.
            # The halt from execution report is more immediate.
            return False, reason
        return True, None
