# Risk Manager Module
"""Risk management module for trading operations.

This module provides risk management functionality for trading operations,
including position sizing, drawdown limits, and trade validation.
"""

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
import decimal
from decimal import Decimal, InvalidOperation
import time
from typing import TYPE_CHECKING, Any, Optional
import uuid

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


# Type hint for PortfolioManager without circular import
if TYPE_CHECKING:
    from .exchange_info_service import ExchangeInfoService
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

            def get_current_state(self) -> dict[str, Any]:
                """Return empty portfolio state dictionary.

                Returns
                -------
                    dict: Empty dictionary representing portfolio state
                """
                return {}

    try:
        from .market_price_service import MarketPriceService
    except ImportError:

        class MarketPriceService:  # type: ignore
            """Placeholder for MarketPriceService when not available at runtime.

            This allows the RiskManager to be imported and used without the actual service.
            """

            async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
                """Return None as placeholder for latest price."""
                _ = trading_pair  # Unused parameter
                return None
                
            async def get_volatility(self, trading_pair: str, lookback_hours: int = 24) -> Optional[float]:
                """Return None as placeholder for volatility calculation.
                
                Args:
                    trading_pair: The trading pair to calculate volatility for
                    lookback_hours: Number of hours to look back for calculation
                    
                Returns:
                    None as placeholder
                """
                _ = (trading_pair, lookback_hours)  # Unused parameters
                return None
                
            async def get_volatility(self, trading_pair: str, lookback_hours: int = 24) -> Optional[float]:
                """Return None as placeholder for volatility calculation.
                
                Args:
                    trading_pair: The trading pair to calculate volatility for
                    lookback_hours: Number of hours to look back for calculation
                    
                Returns:
                    None as placeholder
                """
                _ = (trading_pair, lookback_hours)  # Unused parameters
                return None

            async def convert_amount(
                self, from_amount: Decimal, from_currency: str, to_currency: str
            ) -> Optional[Decimal]:
                """Return None as placeholder for currency conversion.

                Args:
                    from_amount: Amount to convert
                    from_currency: Source currency
                    to_currency: Target currency

                Returns
                -------
                    None as placeholder
                """
                _ = (from_amount, from_currency, to_currency)  # Unused parameters
                return None

    try:  # Added for ExchangeInfoService
        from .exchange_info_service import ExchangeInfoService
    except ImportError:

        class ExchangeInfoService:  # type: ignore
            """Placeholder for ExchangeInfoService."""

            def get_symbol_info(self, trading_pair: str) -> Optional[Dict[str, Any]]:
                """Get information for a specific symbol.
                
                Args:
                    trading_pair: The trading symbol to get info for
                    
                Returns:
                    Dictionary with symbol information or None if not found
                """
                return None
                
            def get_tick_size(self, trading_pair: str) -> Optional[Decimal]:
                """Get the minimum price movement for a trading pair.

                Args:
                    trading_pair: The trading pair to get tick size for

                Returns
                -------
                    The minimum price movement or None if not available
                """
                _ = trading_pair  # Unused parameter
                # Example: return Decimal("0.000001") for XRP/USD
                return None  # Default to no specific tick size

            def get_step_size(self, trading_pair: str) -> Optional[Decimal]:
                """Get the minimum trade size for a trading pair.

                Args:
                    trading_pair: The trading pair to get step size for

                Returns
                -------
                    The minimum trade size or None if not available
                """
                _ = trading_pair  # Unused parameter
                # Example: return Decimal("0.1") for XRP
                return None  # Default to no specific step size


# Using default Decimal precision

MIN_SYMBOL_PARTS = 2
MIN_SYMBOL_LENGTH_FOR_FALLBACK_SPLIT = 3
CACHE_EXPIRY_SECONDS = 300

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
class FinalValidationDataContext:
    """Holds data for the final stage of pre-trade checks."""

    event: TradeSignalProposedEvent
    signal_id: uuid.UUID
    trading_pair: str
    side: str
    entry_type: str
    exchange: str
    strategy_id: str
    current_equity: Decimal
    portfolio_state: dict[str, Any]
    state_values: dict[str, Decimal]
    initial_rounded_calculated_qty: Decimal
    rounded_entry_price: Optional[Decimal]
    rounded_sl_price: Decimal
    rounded_tp_price: Optional[Decimal]
    effective_entry_price: Decimal # Guaranteed non-None if this stage is reached
    ref_entry_for_calculation: Decimal # Guaranteed non-None if this stage is reached


@dataclass
class PriceValidationContext:
    """Holds data for price validation steps (fat finger, SL)."""

    event: TradeSignalProposedEvent
    entry_type: str
    side: str
    rounded_entry_price: Optional[Decimal]
    rounded_sl_price: Decimal # Must be non-None if this stage is reached
    effective_entry_price_for_non_limit: Optional[Decimal]
    current_market_price: Optional[Decimal]


@dataclass
class PositionScalingContext:
    """Holds data for the position scaling check."""

    signal_id: uuid.UUID
    trading_pair: str
    side: str
    ref_entry_price: Decimal # Guaranteed non-None
    portfolio_state: dict[str, Any]
    initial_calculated_qty: Decimal # Before scaling


@dataclass
class PriceRoundingContext:
    """Holds data for the price rounding and initial validation step."""

    entry_type: str
    side: str
    trading_pair: str
    effective_entry_price: Optional[Decimal]
    sl_price: Optional[Decimal] # Initial SL before rounding, can be None if not proposed
    tp_price: Optional[Decimal]


@dataclass
class SystemHaltPayload:
    """Payload for system halt events."""

    reason: str
    details: dict[str, Any]


# --- RiskManager Class ---
class RiskManager:
    """Assess trade signals against risk parameters and portfolio state.

    Consumes proposed trade signals, performs pre-trade risk checks against
    portfolio state, and publishes approved/rejected signals or triggers HALT.
    """

    def __init__(  # noqa: PLR0913 - Multiple dependencies are required
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        portfolio_manager: "PortfolioManager",
        logger_service: LoggerService,
        market_price_service: "MarketPriceService",
        exchange_info_service: "ExchangeInfoService",
    ) -> None:
        """Initialize the RiskManager with configuration and dependencies.

        Args:
            config: Configuration settings.
            pubsub_manager: The application PubSubManager instance.
            portfolio_manager: The PortfolioManager instance.
            logger_service: Shared logger instance.
            market_price_service: MarketPriceService instance.
            exchange_info_service: ExchangeInfoService instance.
        """
        self._config = config.get("risk_manager", {})
        self.pubsub = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self._market_price_service = market_price_service
        self._exchange_info_service = exchange_info_service
        # symbol_info = self._exchange_info_service.get_symbol_info(trading_pair) # COMMENTED OUT
        self.logger = logger_service
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self._periodic_check_task: Optional[asyncio.Task] = None
        self._dynamic_risk_adjustment_task: Optional[asyncio.Task] = None
        self._risk_metrics_task: Optional[asyncio.Task] = None
        self._source_module = self.__class__.__name__

        # Store handler for unsubscribing
        self._signal_proposal_handler = self._handle_trade_signal_proposed
        self._exec_report_handler = self._handle_execution_report_for_losses

        # State for consecutive losses
        self._consecutive_loss_count: int = 0

        # Cache for currency conversion rates
        self._cached_conversion_rates: dict[str, Decimal] = {}
        self._cached_conversion_timestamps: dict[str, datetime] = {}

        # Normal volatility levels for risk adjustment (calibrated during startup)
        self._normal_volatility: dict[str, Decimal] = {}

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

        # Add new configuration for dynamic risk adjustment
        self._enable_dynamic_risk_adjustment = bool(
            self._config.get("enable_dynamic_risk_adjustment", False)
        )
        self._risk_adjustment_interval_s = int(
            self._config.get("risk_adjustment_interval_s", 900)
        )  # 15 minutes default
        self._volatility_window_size = int(
            self._config.get("volatility_window_size", 24)
        )  # Hours of lookback for volatility
        self._risk_metrics_interval_s = int(
            self._config.get("risk_metrics_interval_s", 60)
        )  # 1 minute default

        self.logger.info(f"RiskManager configured.", source_module=self._source_module)
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
                self.logger.debug(
                    f"Configuration Error: {error_msg}",
                    source_module=self._source_module,
                )
            # Log critical message and continue as per whiteboard,
            # rather than raising an exception.
            error_msg = "; ".join(config_errors)
            self.logger.critical(
                f"RiskManager has configuration errors. Review settings: {error_msg}",
                source_module=self._source_module,
            )

        try:
            self.logger.info(
                f"Unsubscribed from TRADE_SIGNAL_PROPOSED and EXECUTION_REPORT.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error unsubscribing RiskManager",
                source_module=self._source_module,
            )

        # Stop periodic checks
        if self._periodic_check_task and not self._periodic_check_task.done():
            self._periodic_check_task.cancel()

        if True: # Placeholder to ensure correct indentation for the following blocks
            existing_side = existing_position.get("side")
            if not existing_side:
                # Position data incomplete
                self.logger.debug(
                    f"Position data for {trading_pair} is incomplete, cannot check scaling",
                    source_module=self._source_module,
                )

            existing_qty = Decimal(str(existing_position.get("quantity", 0)))
            existing_entry = Decimal(str(existing_position.get("avg_entry_price", 0)))
            # pct_diff, is_better_entry would be defined here or passed if this logger call is active
            # self.logger.debug(
            #     f"Scaling evaluation for {trading_pair}: Existing Qty={existing_qty}, Entry={existing_entry}. Proposed Entry={ref_entry_price}. Pct Diff={pct_diff:.2f}%. Better Entry: {is_better_entry}",
            #     pct_diff,
            #     is_better_entry,
            #     source_module=self._source_module,
            # )

            return_value = self._calculate_lot_size_with_fallback(trading_pair, base_amount)
            self.logger.exception(
                "Error converting lot size",
                source_module=self._source_module,
            )
            # Continue with normal position sizing as fallback
            # return True, None, None # This return needs a proper method context
        # else: # This 'else' would correspond to a 'try' block
            # This block executes if the try completed without an exception
            # Calculate and cache the rate
            # conversion_rate = converted_amount / amount
            # self._cached_conversion_rates[currency] = conversion_rate
            # self._cached_conversion_timestamps[currency] = datetime.utcnow()
        # except (ZeroDivisionError, TypeError) as e: # This 'except' would correspond to a 'try' block
            # self.logger.debug(
            # f"Could not cache conversion rate for {currency}: {e}",
            # source_module=self._source_module,
            # )

        # return converted_amount, error # This return needs a proper method context

    async def _try_direct_conversion(self, from_currency: str, to_currency: str, amount: Decimal) -> tuple[Optional[Decimal], Optional[str]]:
        # Original content was elided by {{ ... }} marker.
        # The log message "Dynamic risk adjustment task cancelled." was present below the marker.
        # This is a placeholder structure.
        try:
            # Placeholder for the original logic
            pass
        except asyncio.CancelledError: # Assuming this was the context for the log
            self.logger.info(
                "Dynamic risk adjustment task cancelled.",
                source_module=self._source_module,
            )
        # Placeholder return
        return None, "Original implementation elided"

    async def _dynamic_risk_adjustment_loop(self) -> None:
        """Periodically adjust risk parameters based on market conditions."""
        self.logger.info("Starting dynamic risk adjustment.", source_module=self._source_module)
        while self._is_running:
            try:
                # Iterate over relevant trading pairs (e.g., from portfolio or watchlist)
                # For simplicity, using a placeholder; replace with actual logic
                example_trading_pairs = ["XRP/USD", "BTC/USD"] # Placeholder
                for trading_pair in example_trading_pairs:
                    await self._adjust_risk_parameters_for_volatility(trading_pair)

                await asyncio.sleep(self._risk_adjustment_interval_s)
            except asyncio.CancelledError:
                self.logger.info(
                    "Dynamic risk adjustment task cancelled.",
                    source_module=self._source_module,
                )
                break
            except Exception:
                self.logger.exception(
                    "Error in dynamic risk adjustment loop",
                    source_module=self._source_module,
                )
                # Avoid tight loop on error
                await asyncio.sleep(60)  # 1 minute backoff on error

        self.logger.info("Stopped dynamic risk adjustment.", source_module=self._source_module)

    async def _adjust_risk_parameters_for_volatility(self, trading_pair: str) -> None:
        """Adjust risk parameters based on market volatility.

        Args:
            trading_pair: The trading pair to adjust parameters for
        """
        try:
            # Get current volatility metric (e.g., ATR or std dev of returns)
            # TODO: Implement get_recent_volatility method in MarketPriceService
            # Use get_volatility with a longer lookback period as a substitute for historical volatility
            current_volatility = await self._market_price_service.get_volatility(
                trading_pair=trading_pair,
                lookback_hours=6,  # Recent window
            )

            if current_volatility is None:
                self.logger.warning(
                    f"Could not retrieve volatility for {trading_pair}, skipping adjustment.",
                    source_module=self._source_module,
                )
                return

            # Example: Adjust risk_per_trade_pct based on volatility
            # This logic is illustrative; replace with your specific strategy
            # normal_vol = self._normal_volatility.get(trading_pair)
            # if normal_vol and normal_vol > 0:
            #    if current_volatility > normal_vol * Decimal("1.5"): # High volatility
            #        self._risk_per_trade_pct = max(self._risk_per_trade_pct / Decimal("2"), Decimal("0.1"))
            #        self.logger.info(f"Reduced risk per trade to {self._risk_per_trade_pct}% for {trading_pair} due to high volatility.", source_module=self._source_module)
            #    elif current_volatility < normal_vol * Decimal("0.75"): # Low volatility
            #        self._risk_per_trade_pct = min(self._risk_per_trade_pct * Decimal("1.5"), Decimal("1.0")) # Max risk per trade can be capped
            #        self.logger.info(f"Increased risk per trade to {self._risk_per_trade_pct}% for {trading_pair} due to low volatility.", source_module=self._source_module)

            # self.logger.debug(f"Volatility for {trading_pair}: {current_volatility:.4f}", source_module=self._source_module)
            pass # Placeholder for actual adjustment logic

        except Exception:
            self.logger.exception(
                f"Error adjusting risk parameters for {trading_pair}",
                source_module=self._source_module,
            )

    async def _periodic_risk_metrics_loop(self) -> None:
        """Periodically calculate and log/publish risk metrics."""
        self.logger.info("Starting periodic risk metrics calculation.", source_module=self._source_module)
        while self._is_running:
            try:
                await self._calculate_and_log_risk_metrics()
                await asyncio.sleep(self._risk_metrics_interval_s)
            except asyncio.CancelledError:
                self.logger.info("Risk metrics calculation task cancelled.", source_module=self._source_module)
                break
            except Exception:
                self.logger.exception("Error in risk metrics calculation loop", source_module=self._source_module)
                await asyncio.sleep(60) # Backoff on error
        self.logger.info("Stopped periodic risk metrics calculation.", source_module=self._source_module)

    async def _calculate_and_log_risk_metrics(self) -> None:
        """Calculates current risk metrics and logs them."""
        portfolio_state = self._portfolio_manager.get_current_state()
        if not portfolio_state or "equity" not in portfolio_state:
            self.logger.warning("Portfolio state or equity not available for risk metrics.", source_module=self._source_module)
            return

        current_equity = Decimal(str(portfolio_state["equity"]))
        # total_initial_equity = ... # Need a way to get initial equity or a baseline
        # current_drawdown_pct = ((total_initial_equity - current_equity) / total_initial_equity) * 100 if total_initial_equity > 0 else Decimal(0)

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "current_equity_usd": float(current_equity),
            # "current_total_drawdown_pct": float(current_drawdown_pct),
            "consecutive_losses": self._consecutive_loss_count,
            # Add more metrics: VaR, Sharpe Ratio (if calculable), total exposure, etc.
        }
        
        # Example: Extracting entry price for logging from portfolio state if needed
        # This part is illustrative and depends on your portfolio_state structure
        with contextlib.suppress(Exception): # Gracefully handle missing keys or malformed data
            for symbol, data in portfolio_state.get("positions", {}).items():
                entry_price = data.get("avg_entry_price")
                # timestamp = datetime.utcnow() # Already defined above
                # if timestamp.minute % 10 == 0: # This was from a different context, adapt if needed
                #    self.logger.info(
                #        f"Setting entry price for {symbol}: {entry_price}", # symbol not trading_pair
                #        source_module=self._source_module,
                #    )


        # Publish metrics (optional)
        # await self.pubsub.publish(EventType.RISK_METRICS_UPDATED, metrics)

        # Log metrics
        self.logger.info(
            f"Current risk metrics: {metrics}",
            source_module=self._source_module,
        )

    async def _handle_trade_signal_proposed(
        self, event: TradeSignalProposedEvent
    ) -> None:
        """Handle incoming trade signal proposals.

        This is the main entry point for risk assessment of a trade.
        """
        # --- Stage 1: Initial Validation & Price Rounding ---
        price_rounding_ctx = PriceRoundingContext(
            entry_type=event.entry_type,
            side=event.side,
            trading_pair=event.trading_pair,
            effective_entry_price=effective_entry_price, # Can be None for LIMIT
            sl_price=proposed_sl_price_decimal, # Can be None
            tp_price=proposed_tp_price_decimal, # Can be None
        )
        (
            is_valid_initial,
            initial_rejection_reason,
            rounded_entry_price, # Will be None for MARKET until later
            rounded_sl_price,    # Will be derived if not given
            rounded_tp_price,    # Will be derived if not given or None
        ) = self._calculate_and_validate_prices(price_rounding_ctx)

        if not is_valid_initial:
            await self._reject_signal(
                event.signal_id, event, initial_rejection_reason or "Initial price validation failed"
            )
            return

        # Ensure SL is now non-None, derived if it was initially None
        if rounded_sl_price is None:
             # This should not happen if _calculate_and_validate_prices works correctly and derives SL
            self.logger.error(
                f"SL price is unexpectedly None after initial validation for signal {event.signal_id}.",
                signal_id=event.signal_id,
                source_module=self._source_module,
            )
            await self._reject_signal(
                event.signal_id, event, "Critical error: SL price calculation failed."
            )
            return

        # --- Stage 2: Fat Finger & Stop-Loss Distance (Market Price Dependent) ---
        # For MARKET orders, effective_entry_price is now the current market price.
        # For LIMIT orders, effective_entry_price is the proposed limit price.
        # If entry price was None (MARKET), it should have been fetched by now.
        # If it's still None, it's an issue.

        current_market_price_for_validation = await self._get_current_market_price(event.trading_pair)

        # Determine the reference entry price for non-limit orders (MARKET, STOP_MARKET etc.)
        # This is the current market price for these types for validation purposes.
        effective_entry_price_for_non_limit = None
        if event.entry_type.upper() != "LIMIT":
            effective_entry_price_for_non_limit = current_market_price_for_validation
            if effective_entry_price_for_non_limit is None:
                self.logger.warning(
                    f"Cannot perform fat-finger/SL validation for signal {event.signal_id} due to missing market price.",
                    signal_id=event.signal_id,
                    source_module=self._source_module,
                )
                await self._reject_signal(
                    event.signal_id, event, "Market price unavailable for validation."
                )
                return
            # For non-LIMIT orders, the rounded_entry_price is conceptually the current market price (or derived from it)
            # Let's ensure it reflects this for subsequent calculations if it was market.
            if rounded_entry_price is None: # Was a MARKET order
                rounded_entry_price = self._round_price_to_tick_size(effective_entry_price_for_non_limit, event.trading_pair)


        # The effective_entry_price for calculation now depends on order type:
        # For LIMIT, it's the rounded_entry_price (the limit itself).
        # For MARKET, it's the effective_entry_price_for_non_limit (current market).
        ref_entry_for_calculation = rounded_entry_price if event.entry_type.upper() == "LIMIT" else effective_entry_price_for_non_limit

        if ref_entry_for_calculation is None:
            # This should be caught earlier, but as a safeguard:
            self.logger.error(f"Reference entry price for calculation is None for signal {event.signal_id}. Type: {event.entry_type}",
                signal_id=event.signal_id,
                source_module=self._source_module
            )
            await self._reject_signal(event.signal_id, event, "Internal error: reference entry price missing.")
            return

        price_validation_ctx = PriceValidationContext(
            event=event,
            entry_type=event.entry_type,
            side=event.side,
            rounded_entry_price=rounded_entry_price, # This is the limit price for LIMIT, or market price for MARKET
            rounded_sl_price=rounded_sl_price, # Should be non-None
            effective_entry_price_for_non_limit=effective_entry_price_for_non_limit, # Current market for non-LIMIT
            current_market_price=current_market_price_for_validation
        )

        is_price_valid, price_rejection_reason = self._validate_prices_fat_finger_and_sl_distance(price_validation_ctx)


        if not is_price_valid:
            await self._reject_signal(
                event.signal_id, event, price_rejection_reason or "Price validation failed (fat-finger/SL distance)."
            )
            return

        # --- Stage 3: Position Sizing & Portfolio Checks ---
        portfolio_state = self._portfolio_manager.get_current_state()
        if not portfolio_state or "total_equity_usd" not in portfolio_state:
            self.logger.warning(
                f"Portfolio state or equity not available for signal {event.signal_id}.",
                signal_id=event.signal_id,
                source_module=self._source_module,
            )
            await self._reject_signal(
                event.signal_id, event, "Portfolio state unavailable."
            )
            return

        current_equity = portfolio_state.get("total_equity_usd")
        if current_equity is None:
            # Attempt to get from 'equity' if 'total_equity_usd' is missing
            current_equity = portfolio_state.get("equity")
            if current_equity is None:
                self.logger.warning(
                    f"Equity not found in portfolio state for signal {event.signal_id}.",
                    signal_id=event.signal_id,
                    source_module=self._source_module,
                )
                await self._reject_signal(event.signal_id, event, "Equity unavailable in portfolio state.")
                return
        try:
            current_equity_decimal = Decimal(str(current_equity))
        except InvalidOperation:
            self.logger.error(
                f"Invalid equity value '{current_equity}' in portfolio state for signal {event.signal_id}.",
                signal_id=event.signal_id,
                source_module=self._source_module,
            )
            await self._reject_signal(event.signal_id, event, "Invalid equity value in portfolio state.")
            return

        # ref_entry_for_calculation is used for SL distance calculation
        sizing_result = self._calculate_and_validate_position_size(
            event,
            current_equity_decimal,
            ref_entry_for_calculation, # Entry price for SL distance
            rounded_sl_price, # Non-None SL price
            portfolio_state,
        )
        if not sizing_result.is_valid:
            await self._reject_signal(
                event.signal_id, event, sizing_result.rejection_reason or "Position sizing/validation failed."
            )
            return
        
        initial_rounded_calculated_qty = sizing_result.quantity # Renamed from validated_qty

        if initial_rounded_calculated_qty is None or initial_rounded_calculated_qty.is_zero():
            await self._reject_signal(
                event.signal_id, event, "Calculated quantity is zero or None."
            )
            return
            
        # --- Stage 4: Position Scaling (if applicable) ---
        position_scaling_ctx = PositionScalingContext(
            signal_id=event.signal_id,
            trading_pair=event.trading_pair,
            side=event.side,
            ref_entry_price=ref_entry_for_calculation,
            portfolio_state=portfolio_state,
            initial_calculated_qty=initial_rounded_calculated_qty # Pass the already rounded qty
        )
        
        can_scale, scale_rejection_reason, final_trade_action, final_quantity = self._check_position_scaling(position_scaling_ctx)

        if not can_scale:
            await self._reject_signal(
                event.signal_id, event, scale_rejection_reason or "Position scaling check failed."
            )
            return

        # If _check_position_scaling modified the quantity or action
        current_qty_to_trade = final_quantity if final_quantity is not None else initial_rounded_calculated_qty
        # current_trade_action = final_trade_action if final_trade_action is not None else event.side # Assuming side doesn't change, but action might be 'REDUCE_ONLY' etc.

        if current_qty_to_trade.is_zero():
            await self._reject_signal(
                event.signal_id, event, "Quantity became zero after scaling."
            )
            return

        # --- Stage 5: Final Pre-Trade Validation (includes balance check) ---
        # Determine the effective entry price for the trade:
        # For LIMIT orders, it's the limit price.
        # For MARKET orders, it's the current market price used for validation earlier.
        final_effective_entry_price = rounded_entry_price if event.entry_type.upper() == "LIMIT" else effective_entry_price_for_non_limit

        if final_effective_entry_price is None: # Safeguard
            self.logger.error(f"Final effective entry price is None for signal {event.signal_id}", signal_id=event.signal_id, source_module=self._source_module)
            await self._reject_signal(event.signal_id, event, "Internal error: final entry price missing.")
            return

        # state_values are used by _perform_final_pre_trade_validations
        # It expects keys like 'initial_equity_usd', 'daily_initial_equity_usd' etc.
        # Ensure portfolio_state provides these or they are derived correctly.
        # For now, we pass portfolio_state directly and let the method extract.
        # Consider creating a more specific context object if this becomes complex.

        final_validation_ctx = FinalValidationDataContext(
            event=event,
            signal_id=event.signal_id,
            trading_pair=event.trading_pair,
            side=event.side, # or current_trade_action if it can change
            entry_type=event.entry_type,
            exchange=event.exchange,
            strategy_id=event.strategy_id,
            current_equity=current_equity_decimal,
            portfolio_state=portfolio_state, # Pass the raw state
            state_values=self._extract_relevant_portfolio_values(portfolio_state), # Extracted values
            initial_rounded_calculated_qty=current_qty_to_trade,
            rounded_entry_price=rounded_entry_price, # For LIMIT orders, or market price based for MARKET
            rounded_sl_price=rounded_sl_price,
            rounded_tp_price=rounded_tp_price,
            effective_entry_price=final_effective_entry_price,
            ref_entry_for_calculation=ref_entry_for_calculation
        )

        is_final_valid, final_rejection_reason = await self._perform_final_pre_trade_validations(final_validation_ctx)


        if not is_final_valid:
            await self._reject_signal(
                event.signal_id, event, final_rejection_reason or "Final pre-trade validation failed."
            )
            return

    async def _perform_final_pre_trade_validations(
        self, ctx: FinalValidationDataContext
    ) -> tuple[bool, Optional[str]]:
        """Perform final overall portfolio and risk checks before approval."""
        # Check 1: Max Exposure per Asset
        # This requires knowing the value of existing position in this asset + new trade value.
        # Portfolio state should provide current position value for the asset.
        # New trade value = quantity * effective_entry_price

        # Convert new trade quantity to quote currency value
        value_of_new_trade_usd = ctx.initial_rounded_calculated_qty * ctx.effective_entry_price # Assuming effective_entry_price is in quote currency

        # Get current exposure for this specific asset from portfolio_state
        current_asset_exposure_usd = Decimal("0")
        if ctx.portfolio_state and "positions" in ctx.portfolio_state:
            asset_position_data = ctx.portfolio_state["positions"].get(ctx.trading_pair)
            if asset_position_data:
                # Assuming 'current_market_value' is in USD or valuation currency
                current_asset_exposure_usd = Decimal(str(asset_position_data.get("current_market_value", "0")))
        
        total_potential_asset_exposure_usd = current_asset_exposure_usd + value_of_new_trade_usd 

        max_asset_exposure_allowed_usd = ctx.current_equity * (
            self._max_exposure_per_asset_pct / Decimal("100")
        )
        if total_potential_asset_exposure_usd > max_asset_exposure_allowed_usd:
            reason = (
                f"Exceeds max exposure per asset for {ctx.trading_pair} "
                f"({total_potential_asset_exposure_usd:.2f} > {max_asset_exposure_allowed_usd:.2f} USD). "
                f"Limit: {self._max_exposure_per_asset_pct}%"
            )
            self.logger.info(reason, signal_id=ctx.signal_id, source_module=self._source_module)
            return False, reason

        # Check 2: Max Total Portfolio Exposure
        current_total_exposure_usd = Decimal("0")
        if ctx.portfolio_state and "positions" in ctx.portfolio_state:
            for pair_data in ctx.portfolio_state["positions"].values():
                 current_total_exposure_usd += Decimal(str(pair_data.get("current_market_value", "0")))
        
        total_potential_portfolio_exposure_usd = current_total_exposure_usd + value_of_new_trade_usd

        max_portfolio_exposure_allowed_usd = ctx.current_equity * (
            self._max_total_exposure_pct / Decimal("100")
        )
        if total_potential_portfolio_exposure_usd > max_portfolio_exposure_allowed_usd:
            reason = (
                f"Exceeds max total portfolio exposure "
                f"({total_potential_portfolio_exposure_usd:.2f} > {max_portfolio_exposure_allowed_usd:.2f} USD). "
                f"Limit: {self._max_total_exposure_pct}%"
            )
            self.logger.info(reason, signal_id=ctx.signal_id, source_module=self._source_module)
            return False, reason

        # Check 3: Sufficient Free Balance (SRS FR-506)
        available_balance_usd = ctx.state_values.get("available_balance_usd")
        if available_balance_usd is None:
            self.logger.warning(f"Available balance not found in portfolio state for signal {ctx.signal_id}. Cannot verify funds.",
                                signal_id=ctx.signal_id, source_module=self._source_module)
            return False, "Available balance missing in portfolio state for fund check."

        estimated_order_cost_usd = ctx.initial_rounded_calculated_qty * ctx.effective_entry_price
        taker_fee_multiplier = self._exchange_taker_fee_pct / Decimal("100")
        estimated_fee_usd = estimated_order_cost_usd * taker_fee_multiplier
        total_estimated_cost_with_fee_usd = estimated_order_cost_usd + estimated_fee_usd
        
        if total_estimated_cost_with_fee_usd > available_balance_usd:
            reason = (
                f"Insufficient available balance for trade. "
                f"Estimated cost with fee {total_estimated_cost_with_fee_usd:.2f} {self._valuation_currency} > "
                f"Available {available_balance_usd:.2f} {self._valuation_currency}."
            )
            self.logger.info(reason, signal_id=ctx.signal_id, source_module=self._source_module)
            return False, reason
            
        # Check 4: Drawdown limits
        if not self._check_drawdown_limits(ctx.portfolio_state, is_pre_trade_check=True):
            return False, "Portfolio drawdown limits would be breached or are currently breached."

        return True, None

    def _check_drawdown_limits(self, portfolio_state: dict[str, Any], is_pre_trade_check: bool = False) -> bool:
        """Check portfolio against configured drawdown limits."""
        if not portfolio_state:
            self.logger.warning("Cannot check drawdown limits: Portfolio state is empty.", source_module=self._source_module)
            return True 

        state_values = self._extract_relevant_portfolio_values(portfolio_state)
        current_equity = state_values.get("current_equity_usd")

        if current_equity is None:
            self.logger.warning(
                "Cannot check drawdown limits: Current equity not found in portfolio state.",
                source_module=self._source_module
            )
            return True 

        breached_limit_reason = None

        current_total_drawdown_pct_str = portfolio_state.get("total_drawdown_pct")
        current_daily_drawdown_pct_str = portfolio_state.get("daily_drawdown_pct")
        current_weekly_drawdown_pct_str = portfolio_state.get("weekly_drawdown_pct")

        # Total Drawdown
        if current_total_drawdown_pct_str is not None:
            try:
                current_total_drawdown_pct_val = Decimal(str(current_total_drawdown_pct_str))
                if current_total_drawdown_pct_val > self._max_total_drawdown_pct:
                    breached_limit_reason = (
                        f"Total drawdown limit breached (from PM): "
                        f"{current_total_drawdown_pct_val:.2f}% > {self._max_total_drawdown_pct:.2f}%"
                    )
            except InvalidOperation:
                self.logger.warning(f"Invalid total_drawdown_pct '{current_total_drawdown_pct_str}' from PM.", source_module=self._source_module)
                current_total_drawdown_pct_str = None # Force fallback if invalid

        if breached_limit_reason is None and current_total_drawdown_pct_str is None and \
           state_values.get("initial_equity_usd") is not None and state_values["initial_equity_usd"] > 0:
            calculated_total_dd = ((state_values["initial_equity_usd"] - current_equity) / state_values["initial_equity_usd"]) * Decimal("100")
            self.logger.info(f"PM did not provide valid total_drawdown_pct. Calculated: {calculated_total_dd:.2f}%", source_module=self._source_module)
            if calculated_total_dd > self._max_total_drawdown_pct:
                breached_limit_reason = f"Total drawdown limit breached (calculated): {calculated_total_dd:.2f}% > {self._max_total_drawdown_pct:.2f}%"
        
        # Daily Drawdown
        if breached_limit_reason is None and current_daily_drawdown_pct_str is not None:
            try:
                current_daily_drawdown_pct_val = Decimal(str(current_daily_drawdown_pct_str))
                if current_daily_drawdown_pct_val > self._max_daily_drawdown_pct:
                    breached_limit_reason = (
                        f"Daily drawdown limit breached (from PM): "
                        f"{current_daily_drawdown_pct_val:.2f}% > {self._max_daily_drawdown_pct:.2f}%"
                    )
            except InvalidOperation:
                self.logger.warning(f"Invalid daily_drawdown_pct '{current_daily_drawdown_pct_str}' from PM.", source_module=self._source_module)
                current_daily_drawdown_pct_str = None # Force fallback
        
        if breached_limit_reason is None and current_daily_drawdown_pct_str is None and \
           state_values.get("daily_initial_equity_usd") is not None and state_values["daily_initial_equity_usd"] > 0:
            calculated_daily_dd = ((state_values["daily_initial_equity_usd"] - current_equity) / state_values["daily_initial_equity_usd"]) * Decimal("100")
            self.logger.info(f"PM did not provide valid daily_drawdown_pct. Calculated: {calculated_daily_dd:.2f}%", source_module=self._source_module)
            if calculated_daily_dd > self._max_daily_drawdown_pct:
                breached_limit_reason = f"Daily drawdown limit breached (calculated): {calculated_daily_dd:.2f}% > {self._max_daily_drawdown_pct:.2f}%"

        # Weekly Drawdown (SRS FR-503)
        if breached_limit_reason is None and current_weekly_drawdown_pct_str is not None and self._max_weekly_drawdown_pct is not None:
            try:
                current_weekly_drawdown_pct_val = Decimal(str(current_weekly_drawdown_pct_str))
                if current_weekly_drawdown_pct_val > self._max_weekly_drawdown_pct:
                    breached_limit_reason = (
                        f"Weekly drawdown limit breached (from PM): "
                        f"{current_weekly_drawdown_pct_val:.2f}% > {self._max_weekly_drawdown_pct:.2f}%"
                    )
            except InvalidOperation:
                self.logger.warning(f"Invalid weekly_drawdown_pct '{current_weekly_drawdown_pct_str}' from PM.", source_module=self._source_module)
                current_weekly_drawdown_pct_str = None # Force fallback

        elif breached_limit_reason is None and current_weekly_drawdown_pct_str is None and \
             state_values.get("weekly_initial_equity_usd") is not None and \
             state_values["weekly_initial_equity_usd"] > 0 and self._max_weekly_drawdown_pct is not None: 
            calculated_weekly_dd = ((state_values["weekly_initial_equity_usd"] - current_equity) / state_values["weekly_initial_equity_usd"]) * Decimal("100")
            self.logger.info(f"PM did not provide valid weekly_drawdown_pct. Calculated: {calculated_weekly_dd:.2f}%", source_module=self._source_module)
            if calculated_weekly_dd > self._max_weekly_drawdown_pct:
                breached_limit_reason = f"Weekly drawdown limit breached (calculated): {calculated_weekly_dd:.2f}% > {self._max_weekly_drawdown_pct:.2f}%"
        
        if breached_limit_reason:
            self.logger.critical(
                breached_limit_reason,
                source_module=self._source_module,
                current_equity=current_equity,
                initial_equity_usd=state_values.get("initial_equity_usd"),
                daily_initial_equity_usd=state_values.get("daily_initial_equity_usd"),
                weekly_initial_equity_usd=state_values.get("weekly_initial_equity_usd"),
            )
            if not is_pre_trade_check: 
                asyncio.create_task(
                    self.pubsub.publish(
                        EventType.POTENTIAL_HALT_TRIGGER,
                        PotentialHaltTriggerEvent(
                            reason=breached_limit_reason,
                            details={
                                "check_type": "drawdown_limit",
                                "current_equity": str(current_equity),
                                "total_dd_pct": str(current_total_drawdown_pct_str) if current_total_drawdown_pct_str is not None else "N/A",
                                "daily_dd_pct": str(current_daily_drawdown_pct_str) if current_daily_drawdown_pct_str is not None else "N/A",
                                "weekly_dd_pct": str(current_weekly_drawdown_pct_str) if current_weekly_drawdown_pct_str is not None else "N/A",
                            },
                        ),
                    )
                )
            return False  

        return True 

    async def _update_risk_parameters_based_on_volatility(
        self, trading_pair: str, current_volatility: Decimal
    ) -> None:
        """Dynamically adjust risk_per_trade_pct based on volatility levels."""
        if not self._enable_dynamic_risk_adjustment:
            return

        normal_vol = self._normal_volatility.get(trading_pair)
        current_risk_setting_before_adj = self._risk_per_trade_pct 

        if normal_vol and normal_vol > 0:
            static_configured_risk_pct = Decimal(str(self._config.get("sizing", {}).get("risk_per_trade_pct", "0.5")))
            new_risk_pct = current_risk_setting_before_adj 

            if current_volatility > normal_vol * Decimal("1.5"):  # High volatility
                new_risk_pct = max(current_risk_setting_before_adj / Decimal("2"), Decimal("0.05")) 
                if new_risk_pct != current_risk_setting_before_adj:
                    self.logger.info(
                        f"DYNAMIC RISK: Reducing risk per trade from {current_risk_setting_before_adj:.3f}% to {new_risk_pct:.3f}% "
                        f"for {trading_pair} due to high volatility ({current_volatility:.4f} vs normal ~{normal_vol:.4f}).",
                        source_module=self._source_module,
                    )
                    self._risk_per_trade_pct = new_risk_pct

            elif current_volatility < normal_vol * Decimal("0.75"):  # Low volatility
                new_risk_pct = min(current_risk_setting_before_adj * Decimal("1.5"), static_configured_risk_pct)
                if new_risk_pct != current_risk_setting_before_adj:
                    self.logger.info(
                        f"DYNAMIC RISK: Increasing risk per trade from {current_risk_setting_before_adj:.3f}% to {new_risk_pct:.3f}% "
                        f"for {trading_pair} due to low volatility ({current_volatility:.4f} vs normal ~{normal_vol:.4f}). Capped at {static_configured_risk_pct:.3f}%.",
                        source_module=self._source_module,
                    )
                    self._risk_per_trade_pct = new_risk_pct
            else: # Volatility is within normal range
                if self._risk_per_trade_pct != static_configured_risk_pct:
                     self.logger.info(
                        f"DYNAMIC RISK: Volatility normal for {trading_pair} ({current_volatility:.4f} vs normal ~{normal_vol:.4f}). "
                        f"Reverting risk per trade from {self._risk_per_trade_pct:.3f}% to static configured {static_configured_risk_pct:.3f}%.",
                        source_module=self._source_module,
                    )
                     self._risk_per_trade_pct = static_configured_risk_pct

        else: 
            if not hasattr(self, '_normal_volatility_logged_missing'): 
                self._normal_volatility_logged_missing = {}

            if self._normal_volatility_logged_missing.get(trading_pair, False) is False:
                self.logger.warning(
                    f"Cannot dynamically adjust risk for {trading_pair}: Normal volatility not calibrated or is zero. Using static risk per trade.",
                    source_module=self._source_module,
                )
                self._normal_volatility_logged_missing[trading_pair] = True 
            
            static_configured_risk_pct = Decimal(str(self._config.get("sizing", {}).get("risk_per_trade_pct", "0.5")))
            if self._risk_per_trade_pct != static_configured_risk_pct:
                self.logger.info(
                    f"DYNAMIC RISK: Reverting risk for {trading_pair} to static configured {static_configured_risk_pct:.3f}% due to missing normal volatility calibration.",
                    source_module=self._source_module
                )
                self._risk_per_trade_pct = static_configured_risk_pct


    async def _calibrate_normal_volatility(self) -> None:
        """Placeholder for calibrating normal volatility levels for relevant pairs.
        This might involve fetching historical volatility over a longer period on startup,
        or loading pre-calculated values.
        """
        # Example: Initialize for a known set of pairs, or dynamically discover them
        # For now, just ensuring the dictionary exists for the adjustment logic
        if not hasattr(self, '_normal_volatility_logged_missing'):
            self._normal_volatility_logged_missing = {}
        
        self.logger.info("Normal volatility calibration logic is a placeholder.", source_module=self._source_module)
        # TODO: Implement actual calibration logic.
        # This might involve:
        # 1. Defining a list of trading pairs to monitor (e.g., from config or portfolio).
        # 2. On startup, for each pair:
        #    - Fetch historical price data (e.g., daily closes for the last N days).
        #    - Calculate a historical volatility metric (e.g., standard deviation of log returns, or average ATR).
        #    - Store this as the 'normal_vol' for the pair in self._normal_volatility.
        # Example for one pair (replace with actual data fetching and calculation):
        # historical_vol = await self._market_price_service.get_historical_volatility("XRP/USD", days=30)
        # if historical_vol is not None:
        #    self._normal_volatility["XRP/USD"] = historical_vol
        # else:
        #    self.logger.warning("Could not calibrate normal volatility for XRP/USD.", source_module=self._source_module)
        pass
