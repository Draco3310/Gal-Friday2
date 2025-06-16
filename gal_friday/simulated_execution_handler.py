"""Simulated execution handler for backtesting and paper trading."""

import decimal
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import (
    ExecutionReportEvent,
    PotentialHaltTriggerEvent,
    TradeSignalApprovedEvent)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService


class ValidationError(ValueError):
    """Raised when validation fails."""


@dataclass
class PortfolioState:
    """Portfolio state information."""


@dataclass
class Position:
    """Position information."""


class HistoricalDataService:
    """Service for providing historical market data."""

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> pd.Series[Any] | None:
        """Get the next bar after the given timestamp."""
        return None

    def get_atr(self, trading_pair: str, timestamp: datetime) -> float | None:
        """Get Average True Range for the given trading pair and timestamp."""
        return None


@dataclass
class CustomReportOverrides:
    """Helper dataclass to group custom override parameters for execution reports."""

    exchange_order_id: str | None = None
    client_order_id: str | None = None
    order_type: str | None = None
    side: str | None = None


@dataclass
class SimulatedReportParams:
    """Parameters for creating a simulated execution report."""

    status: str
    qty_filled: Decimal
    avg_price: Decimal | None
    commission: Decimal
    commission_asset: str | None
    error_msg: str | None = None
    fill_timestamp: datetime | None = None
    liquidity_type: str | None = None


@dataclass
class FillDetails:
    """Container for order fill details to reduce parameter count in methods."""

    qty_increment: Decimal
    timestamp: datetime
    trading_pair: str
    side: str
    exchange: str
    order_type: str
    event: "TradeSignalApprovedEvent"


@dataclass
class MarketExitParams:
    """Parameters for simulating a market order exit (e.g., for SL)."""

    originating_event: "TradeSignalApprovedEvent"  # Event that setup the SL/TP
    quantity: Decimal
    side: str  # Side of the exit order
    trading_pair: str
    exchange: str
    trigger_timestamp: datetime  # Timestamp of the bar that triggered the SL


class SimulatedExecutionHandler:
    """Simulates order execution based on historical OHLCV data for backtesting."""

    def __init__(
        self,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        data_service: HistoricalDataService,
        logger_service: LoggerService) -> None:
        """Initialize the simulation execution handler with required services.

        Args:
        ----
            config_manager: Configuration provider for slippage, fees, etc.
            pubsub_manager: Publish-subscribe manager for event communication
            data_service: Service providing access to historical data
            logger_service: Service for structured logging
        """
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.data_service = data_service
        self.logger = logger_service

        # Load all configuration values to eliminate hardcoding
        self._load_configuration()

        # Initialize state tracking
        self._active_sl_tp: dict[str, dict[str, Any]] = defaultdict(dict[str, Any])
        self._active_limit_orders: dict[str, dict[str, Any]] = {}

        # Log initialization details
        self._log_initialization_summary()

    def _load_configuration(self) -> None:
        """Load all configuration values from ConfigManager to eliminate hardcoding."""
        # Load backtesting configuration
        backtest_config = self.config.get("backtest", {})
        portfolio_config = self.config.get("portfolio", {})

        # Commission configuration
        self.taker_fee_pct = Decimal(str(backtest_config.get("commission_taker_pct", 0.0026)))
        self.maker_fee_pct = Decimal(str(backtest_config.get("commission_maker_pct", 0.0016)))

        # Slippage configuration
        self.slippage_model = backtest_config.get("slippage_model", "volatility")
        self.slip_atr_multiplier = Decimal(
            str(backtest_config.get("slippage_atr_multiplier", 0.1)))
        self.slip_fixed_pct = Decimal(str(backtest_config.get("slippage_fixed_pct", 0.0005)))

        # Market impact slippage parameters
        self.slip_market_impact_base_pct = Decimal(
            str(backtest_config.get("slippage_market_impact_base_pct", 0.0001)))
        self.slip_market_impact_factor = Decimal(
            str(backtest_config.get("slippage_market_impact_factor", 0.1)))
        self.slip_market_impact_exponent = Decimal(
            str(backtest_config.get("slippage_market_impact_exponent", 1.0)))

        # Portfolio configuration
        self.valuation_currency = portfolio_config.get("valuation_currency", "USD").upper()

        # Order execution configuration
        self._fill_liquidity_ratio = Decimal(str(backtest_config.get("fill_liquidity_ratio", 0.1)))
        self.limit_order_timeout_bars = backtest_config.get("limit_order_timeout_bars", 3)
        self.limit_order_timeout_action = backtest_config.get(
            "limit_order_timeout_action", "CANCEL")
        self.processing_delay_bars = backtest_config.get("processing_delay_bars", 0)

        # Data requirements configuration
        self._min_data_points = backtest_config.get("min_data_points", 100)

        # Error handling configuration
        self._max_consecutive_errors = 3  # Could be made configurable
        self._consecutive_errors = 0

    def _log_initialization_summary(self) -> None:
        """Log initialization summary with all configuration values."""
        self.logger.info(
            "SimulatedExecutionHandler initialized with full configuration",
            source_module=self.__class__.__name__)
        self.logger.info(
            "Fees: Taker=%.4f%%, Maker=%.4f%%",
            self.taker_fee_pct * 100,
            self.maker_fee_pct * 100,
            source_module=self.__class__.__name__)
        self.logger.info(
            "Slippage: Model=%s, ATR Mult=%s, Fixed Pct=%.4f%%, Market Impact Base Pct: %.4f%%",
            self.slippage_model,
            self.slip_atr_multiplier,
            self.slip_fixed_pct * 100,
            self.slip_market_impact_base_pct * 100,
            source_module=self.__class__.__name__)
        self.logger.info(
            "Execution: Liquidity Ratio=%s, "
            "Limit Timeout: %s bars (%s), "
            "Processing Delay: %s bars",
            self._fill_liquidity_ratio,
            self.limit_order_timeout_bars,
            self.limit_order_timeout_action,
            self.processing_delay_bars,
            source_module=self.__class__.__name__)

    async def start(self) -> None:  # Interface consistency
        """Start the simulated execution handler.

        For API consistency with other execution handlers; no external connections needed.
        """
        # Start the simulated execution handler
        self.logger.info(
            "SimulatedExecutionHandler started.",
            source_module=self.__class__.__name__)
        # No external connections to establish

    async def stop(self) -> None:  # Interface consistency
        """Stop the simulated execution handler.

        For API consistency with other execution handlers; no external connections needed.
        """
        # Stop the simulated execution handler
        self.logger.info(
            "SimulatedExecutionHandler stopped.",
            source_module=self.__class__.__name__)
        # No external connections to close

    async def handle_trade_signal_approved(self, event: "TradeSignalApprovedEvent") -> None:
        """Process an approved signal and simulate fill based on next bar data."""
        try:
            self.logger.debug(
                "SimExec received approved signal: %s at %s",
                event.signal_id,
                event.timestamp,
                source_module=self.__class__.__name__)

            # Validate order parameters
            if not await self._validate_order_parameters(event):
                return  # Validation failed, report already published

            # Get next bar data
            next_bar = await self._get_next_bar_data(event)
            if next_bar is None:
                return

            # Reset consecutive errors on successful processing start
            self._consecutive_errors = 0

            # Simulate order execution
            fill_result = await self._simulate_order_fill(event, next_bar)

            if fill_result:  # Market order fill or immediate limit fill
                await self._handle_successful_fill(event, fill_result)

            # If fill_result is None, limit order was added to _active_limit_orders
            # and will be processed by check_active_limit_orders in subsequent calls

        except Exception as e:
            await self._handle_execution_error(event, e)

    async def _handle_successful_fill(
        self, event: "TradeSignalApprovedEvent", fill_result: dict[str, Any]) -> None:
        """Handle successful order fill processing."""
        try:
            # Store SL/TP if entry order is filled
            if (fill_result["status"] in ["FILLED", "PARTIALLY_FILLED"] and
                    fill_result["quantity"] > Decimal(0)):
                pos_id_for_sl_tp = str(event.signal_id)
                if (event.risk_parameters and
                        event.risk_parameters.get("original_signal_id_for_sl_tp")):
                    orig_id = event.risk_parameters["original_signal_id_for_sl_tp"]
                    pos_id_for_sl_tp = str(orig_id)
                    self.logger.info(
                        "Using original signal ID %s for SL/TP tracking of converted market order",
                        pos_id_for_sl_tp,
                        source_module=self.__class__.__name__)

                self._register_or_update_sl_tp(
                    initial_event_signal_id_str=pos_id_for_sl_tp,
                    fill_details=FillDetails(
                        qty_increment=fill_result["quantity"],
                        timestamp=fill_result["timestamp"],
                        trading_pair=event.trading_pair,
                        side=event.side,
                        exchange=event.exchange,
                        order_type=event.order_type,
                        event=event))

            # Publish the execution report
            report_params = SimulatedReportParams(
                status=fill_result["status"],
                qty_filled=fill_result["quantity"],
                avg_price=fill_result["fill_price"],
                commission=fill_result["commission"],
                commission_asset=fill_result["commission_asset"],
                error_msg=fill_result.get("error_msg"),
                fill_timestamp=fill_result["timestamp"],
                liquidity_type=fill_result.get("liquidity_type"))

            await self._publish_simulated_report(
                originating_event=event,
                params=report_params,
                overrides=None)

        except Exception as e:
            await self._handle_execution_error(event, e)

    async def _handle_execution_error(
        self, event: "TradeSignalApprovedEvent", error: Exception) -> None:
        """Handle execution errors and publish error reports."""
        self._consecutive_errors += 1

        error_msg = f"Error during fill simulation for signal {event.signal_id}: {error!s}"
        self.logger.exception(
            error_msg,
            source_module=self.__class__.__name__)

        # Trigger HALT if too many consecutive errors
        if self._consecutive_errors >= self._max_consecutive_errors:
            halt_event = PotentialHaltTriggerEvent(
                source_module=self.__class__.__name__,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                reason=(
                    f"Too many consecutive execution simulation errors: "
                    f"{self._consecutive_errors}"
                ))
            await self.pubsub.publish(halt_event)

        # Publish error report
        error_params = SimulatedReportParams(
            status="ERROR",
            qty_filled=Decimal(0),
            avg_price=None,
            commission=Decimal(0),
            commission_asset=None,
            error_msg="Simulation error",
            fill_timestamp=None)

        await self._publish_simulated_report(
            originating_event=event,
            params=error_params,
            overrides=None)

    async def _validate_order_parameters(self, event: "TradeSignalApprovedEvent") -> bool:
        """Validate incoming order parameters."""
        if not isinstance(event.quantity, Decimal) or event.quantity <= Decimal(0):
            error_msg = f"Invalid order quantity: {event.quantity} for signal {event.signal_id}."
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        if event.side.upper() not in ["BUY", "SELL"]:
            error_msg = f"Invalid order side: {event.side} for signal {event.signal_id}."
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        supported_order_types = ["MARKET", "LIMIT"]  # Extend as more types are supported
        if event.order_type.upper() not in supported_order_types:
            error_msg = (
                f"Unsupported order type '{event.order_type}' for simulation. "
                f"Signal {event.signal_id}."
            )
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        if event.order_type.upper() == "LIMIT" and (
            event.limit_price is None
            or not isinstance(event.limit_price, Decimal)
            or event.limit_price <= Decimal(0)
        ):
            error_msg = (
                f"Invalid or missing limit price for LIMIT order: {event.limit_price} "
                f"for signal {event.signal_id}."
            )
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            # f-string in error message is acceptable
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        # Enterprise-grade min/max order size validation
        # Get exchange-specific order size limits from configuration
        exchange_config = self.config.get(f"exchanges.{event.exchange.lower()}", {})
        pair_limits = exchange_config.get("order_limits", {}).get(event.trading_pair, {})

        # Get min/max values with defaults
        min_order_size = Decimal(str(pair_limits.get("min_size", "0.00001")))
        max_order_size = Decimal(str(pair_limits.get("max_size", "1000000.0")))
        min_notional_value = Decimal(str(pair_limits.get("min_notional", "1.0")))
        max_notional_value = Decimal(str(pair_limits.get("max_notional", "10000000.0")))

        # Validate order size
        if event.quantity < min_order_size:
            error_msg = (
                f"Order size {event.quantity} below minimum {min_order_size} "
                f"for {event.trading_pair}"
            )
            self.logger.warning(
                error_msg,
                source_module=self.__class__.__name__,
                context={
                    "signal_id": str(event.signal_id),
                    "quantity": str(event.quantity),
                    "min_size": str(min_order_size),
                })
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        if event.quantity > max_order_size:
            error_msg = (
                f"Order size {event.quantity} exceeds maximum {max_order_size} "
                f"for {event.trading_pair}"
            )
            self.logger.warning(
                error_msg,
                source_module=self.__class__.__name__,
                context={
                    "signal_id": str(event.signal_id),
                    "quantity": str(event.quantity),
                    "max_size": str(max_order_size),
                })
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        # Validate notional value if price is available
        if hasattr(event, "limit_price") and event.limit_price is not None:
            notional_value = event.quantity * event.limit_price
        else:
            # For market orders, try to get current price from data service
            try:
                current_bar = self.data_service.get_next_bar(
                    event.trading_pair,
                    event.timestamp)
                if current_bar is not None and "close" in current_bar:
                    notional_value = event.quantity * Decimal(str(current_bar["close"]))
                else:
                    # Skip notional validation if we can't get price
                    self.logger.debug(
                        "Cannot determine notional value for market order validation",
                        source_module=self.__class__.__name__)
                    return True
            except Exception:
                self.logger.exception(
                    "Error getting price for notional validation",
                    source_module=self.__class__.__name__)
                return True  # Don't reject on price lookup failure

        # Check notional limits
        if notional_value < min_notional_value:
            error_msg = (
                f"Order notional value {notional_value:.2f} below minimum "
                f"{min_notional_value:.2f} for {event.trading_pair}"
            )
            self.logger.warning(
                error_msg,
                source_module=self.__class__.__name__,
                context={
                    "signal_id": str(event.signal_id),
                    "notional": str(notional_value),
                    "min_notional": str(min_notional_value),
                })
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        if notional_value > max_notional_value:
            error_msg = (
                f"Order notional value {notional_value:.2f} exceeds maximum "
                f"{max_notional_value:.2f} for {event.trading_pair}"
            )
            self.logger.warning(
                error_msg,
                source_module=self.__class__.__name__,
                context={
                    "signal_id": str(event.signal_id),
                    "notional": str(notional_value),
                    "max_notional": str(max_notional_value),
                })
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=error_msg))
            return False

        self.logger.debug(
            "Order passed size validation: quantity=%s, notional=%s",
            event.quantity,
            notional_value if "notional_value" in locals() else "N/A",
            source_module=self.__class__.__name__)

        return True

    async def _get_next_bar_data(
        self,
        event: "TradeSignalApprovedEvent") -> pd.Series[Any] | None:
        """Get and validate the next bar data for simulation."""
        try:
            # Adjust timestamp for processing_delay_bars
            if self.processing_delay_bars > 0:
                current_bar_timestamp = event.timestamp
                # This is the close of the signal bar usually
                target_fill_bar_open_timestamp = current_bar_timestamp
                for _ in range(1 + self.processing_delay_bars):
                    # This is a placeholder for advancing bar time correctly.
                    # Actual implementation depends on how HistoricalDataService
                    # allows indexed fetching.
                    # Assuming get_next_bar advances one bar from the provided timestamp.
                    temp_bar = self.data_service.get_next_bar(
                        event.trading_pair,
                        target_fill_bar_open_timestamp)
                    if temp_bar is None:
                        error_msg = (
                            f"No bar data found for {event.trading_pair} at "
                            f"{target_fill_bar_open_timestamp} while looking for "
                            f"{1 + self.processing_delay_bars}-th bar after signal "
                            f"{event.signal_id} at {event.timestamp}."
                        )
                        self.logger.warning(error_msg, source_module=self.__class__.__name__)
                        await self._publish_simulated_report(
                            event,
                            SimulatedReportParams(
                                status="REJECTED",
                                qty_filled=Decimal(0),
                                avg_price=None,
                                commission=Decimal(0),
                                commission_asset=None,
                                error_msg=(
                                    f"No data for delayed fill at {target_fill_bar_open_timestamp}"
                                )))
                        return None
                    target_fill_bar_open_timestamp = temp_bar.name  # type: ignore

            # The HistoricalDataService must be able to look ahead N bars from the
            # event's timestamp bar.
            # The `event.timestamp` is the timestamp of the bar ON WHICH the signal was
            # generated.
            # Orders are typically simulated against the NEXT bar
            # (event.timestamp + 1 bar interval).
            # If there's a processing_delay_bars, it means we use the bar that is
            # (1 + processing_delay_bars) from the signal bar.
            # We can't use lookahead_bars parameter as it's not in the interface
            # Instead, we'll need to call get_next_bar multiple times if needed
            next_bar = self.data_service.get_next_bar(
                event.trading_pair,
                event.timestamp)

            # If we need to look ahead more bars due to processing delay
            current_timestamp = event.timestamp
            for _ in range(self.processing_delay_bars):
                if next_bar is None:
                    break
                # Get timestamp from bar name or keep current one
                current_timestamp = (
                    next_bar.name if hasattr(next_bar, "name") else current_timestamp
                )
                next_bar = self.data_service.get_next_bar(
                    event.trading_pair,
                    current_timestamp)

            if next_bar is None:
                error_msg = (
                    f"No next bar data found for {event.trading_pair} after "
                    f"{event.timestamp} (considering {self.processing_delay_bars} delay bars). "
                    f"Cannot simulate fill for {event.signal_id}."
                )
                self.logger.warning(
                    error_msg,
                    source_module=self.__class__.__name__)
                next_time = event.timestamp + timedelta(minutes=1)
                error_detail = f"No historical data for fill simulation at {next_time}"
                await self._publish_simulated_report(
                    event,
                    SimulatedReportParams(
                        status="REJECTED",
                        qty_filled=Decimal(0),
                        avg_price=None,
                        commission=Decimal(0),
                        commission_asset=None,
                        error_msg=error_detail))
                return None
        except Exception:
            self.logger.exception(
                "Error retrieving next bar data: %s",
                source_module=self.__class__.__name__)
            return None
        else:
            return next_bar

    async def _simulate_order_fill(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series[Any]) -> dict[str, Any] | None:  # UP007: Optional[Dict[str, Any]] -> dict[str, Any] | None
        """Simulate order fill based on order type and market conditions."""
        # fill_qty will be determined by simulation logic, considering partial fills
        # commission_pct will be set based on maker/taker logic

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
                        "Could not convert bar index %s to datetime. Using current time.",
                        next_bar.name,
                        source_module=self.__class__.__name__)
                    fill_timestamp = datetime.now(UTC)  # DTZ003 corrected
        else:
            self.logger.warning(
                "Bar has no name/index. Using current time for fill timestamp.",
                source_module=self.__class__.__name__)
            fill_timestamp = datetime.now(UTC)  # DTZ003 corrected

        # Handle different order types
        if event.order_type.upper() == "MARKET":
            return await self._simulate_market_order(
                event,
                next_bar,
                self.taker_fee_pct,
                fill_timestamp)
        if event.order_type.upper() == "LIMIT":
            return await self._handle_limit_order_placement(  # Renamed from _simulate_limit_order
                event,
                next_bar,
                fill_timestamp)
        error_msg = (
            f"Unsupported order type '{event.order_type}' for simulation. "
            f"Signal {event.signal_id}."
        )
        self.logger.error(
            error_msg,
            source_module=self.__class__.__name__)
        await self._publish_simulated_report(
            event,
            SimulatedReportParams(
                status="REJECTED",
                qty_filled=Decimal(0),
                avg_price=None,
                commission=Decimal(0),
                commission_asset=None,
                error_msg=f"Unsupported order type: {event.order_type}"))
        return None

    async def _simulate_market_order(
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series[Any],
        commission_pct: Decimal,
        fill_timestamp: datetime) -> dict[str, Any]:
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
            bar_volume=next_bar_volume)

        # Calculate fill price with slippage
        simulated_fill_price = (
            Decimal(fill_price_base) + slippage
            if event.side.upper() == "BUY"
            else Decimal(fill_price_base) - slippage
        )

        self.logger.debug(
            (
                " Market fill sim: Base=%s, "  # G004 fix
                "Slip=%.6f, Final=%.6f"
            ),
            fill_price_base,
            slippage,
            simulated_fill_price,
            source_module=self.__class__.__name__)

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
                    (
                        "Market order %s (%s) partially filled: "  # G004, E501 fix
                        "%s/%s based on bar volume %s and liquidity ratio %s"
                    ),
                    event.signal_id,
                    event.trading_pair,
                    simulated_fill_qty,
                    event.quantity,
                    available_volume_at_bar,
                    self._fill_liquidity_ratio,
                    source_module=self.__class__.__name__)
            else:
                status = "FILLED"
        else:
            simulated_fill_qty = Decimal(0)
            # Ensure it's precisely zero if rejected
            error_message = "Zero fillable quantity based on available liquidity"
            self.logger.warning(
                (
                    "Market order %s (%s) rejected due to zero fillable quantity. "
                    "Requested: %s, Bar Volume: %s, Liquidity Ratio: %s"
                ),
                event.signal_id,
                event.trading_pair,
                event.quantity,
                available_volume_at_bar,
                self._fill_liquidity_ratio,
                source_module=self.__class__.__name__)

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
                "Converting fill_timestamp from %s to datetime",
                type(timestamp),  # G004 fix
                source_module=self.__class__.__name__)
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
            "liquidity_type": "TAKER",  # Market orders are takers
        }

    async def _handle_limit_order_placement(  # Renamed from _simulate_limit_order
        self,
        event: "TradeSignalApprovedEvent",
        next_bar: pd.Series[Any],
        fill_timestamp: datetime) -> dict[str, Any] | None:
        """Handle the initial placement of a limit order.

        It checks for immediate fill potential on the next_bar.
        If not fully filled immediately, and persistence is enabled,
        it adds the order (or its remainder) to _active_limit_orders.
        """
        limit_price = event.limit_price
        if limit_price is None:  # Should have been caught by _validate_order_parameters
            error_msg = (
                f"Limit price missing for signal {event.signal_id} "
                "in _handle_limit_order_placement."
            )
            self.logger.error(error_msg, source_module=self.__class__.__name__)
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg="Limit price missing"))
            return None  # Indicates terminal rejection

        # Check for immediate fill potential
        fill_price_if_met, can_fill_this_bar = self._check_limit_order_fill_on_bar(
            event.side,
            limit_price,
            next_bar)

        order_fully_processed_immediately = False
        simulated_fill_qty_immediate = Decimal(0)

        if can_fill_this_bar and fill_price_if_met is not None:
            available_volume_at_bar = Decimal(str(next_bar.get("volume", "0")))
            max_fillable_qty_liquidity = available_volume_at_bar * self._fill_liquidity_ratio
            simulated_fill_qty_immediate = min(event.quantity, max_fillable_qty_liquidity)

            if simulated_fill_qty_immediate > Decimal("1e-12"):
                liquidity_type = self._determine_limit_order_liquidity(
                    event.side,
                    fill_price_if_met,
                    next_bar)
                commission_pct = (
                    self.maker_fee_pct if liquidity_type == "MAKER" else self.taker_fee_pct
                )
                fill_value = simulated_fill_qty_immediate * fill_price_if_met
                commission_amount = abs(fill_value * commission_pct)
                _, quote_asset_str = event.trading_pair.split("/")
                commission_asset = quote_asset_str.upper()

                status = (
                    "PARTIALLY_FILLED"
                    if simulated_fill_qty_immediate < event.quantity
                    else "FILLED"
                )

                await self._publish_simulated_report(
                    originating_event=event,
                    params=SimulatedReportParams(
                        status=status,
                        qty_filled=simulated_fill_qty_immediate,
                        avg_price=fill_price_if_met,
                        commission=commission_amount,
                        commission_asset=commission_asset,
                        fill_timestamp=fill_timestamp,
                        liquidity_type=liquidity_type))

                # Register SL/TP for the filled portion if it's an entry order
                self._register_or_update_sl_tp(
                    initial_event_signal_id_str=str(event.signal_id),
                    fill_details=FillDetails(
                        qty_increment=simulated_fill_qty_immediate,
                        timestamp=fill_timestamp,
                        trading_pair=event.trading_pair,
                        side=event.side,
                        exchange=event.exchange,
                        order_type=event.order_type,
                        event=event,  # Pass the original event
                    ))

                if status == "FILLED":
                    order_fully_processed_immediately = True

                # Return structure for immediate fill (even if partial, this part is done)
                # The calling handle_trade_signal_approved expects this structure
                # for immediate fills
                return {
                    "status": status,
                    "quantity": simulated_fill_qty_immediate,
                    "fill_price": fill_price_if_met,
                    "commission": commission_amount,
                    "commission_asset": commission_asset,
                    "timestamp": fill_timestamp,
                    "liquidity_type": liquidity_type,
                }
            # else: fillable quantity is effectively zero despite price being met

        # If not fully processed immediately (either no fill, zero liquidity fill,
        # or partial fill)
        if not order_fully_processed_immediately:
            remaining_qty_for_active_order = event.quantity - simulated_fill_qty_immediate
            if (
                remaining_qty_for_active_order > Decimal("1e-12")
                and self.limit_order_timeout_bars > 0
            ):
                # Add the original event (or a modified one for remaining qty) to active orders
                event_for_active = event
                if simulated_fill_qty_immediate > Decimal(0):  # If it was a partial fill
                    # Create a new event for the remaining quantity
                    # Convert dataclass to dict[str, Any] instead of using model_dump which is for Pydantic
                    import dataclasses

                    remaining_payload = dataclasses.asdict(event)
                    remaining_payload["quantity"] = remaining_qty_for_active_order
                    # Ensure the new signal_id for the remainder is unique if needed,
                    # but _add_to_active_limit_orders uses its own internal_sim_order_id.
                    # The original signal_id is still important for SL/TP tracking.
                    from gal_friday.core.events import (
                        TradeSignalApprovedEvent as ConcreteEvent,  # type: ignore
                    )

                    event_for_active = ConcreteEvent(**remaining_payload)

                self._add_to_active_limit_orders(
                    original_event=event_for_active,  # Contains current remaining qty
                    initial_event_signal_id_str=str(event.signal_id),  # Original ID for SL/TP
                )
                # Report NEW for the part that went active
                await self._publish_simulated_report(
                    originating_event=event_for_active,  # Report for the quantity going active
                    params=SimulatedReportParams(
                        status="NEW",  # Or PENDING
                        qty_filled=Decimal(0),  # No fill for this report (active part)
                        avg_price=None,
                        commission=Decimal(0),
                        commission_asset=None,
                        fill_timestamp=fill_timestamp,  # Timestamp of this decision
                    ))
                return None  # Signal that it's active, not terminally processed by this call

            if remaining_qty_for_active_order > Decimal("1e-12"):  # Not filled and no persistence
                self.logger.debug(
                    "Limit order %s for %s not filled and no persistence. Rejecting.",
                    event.signal_id,
                    event.trading_pair,
                    source_module=self.__class__.__name__)
                await self._publish_simulated_report(
                    event,
                    SimulatedReportParams(
                        status="REJECTED",
                        qty_filled=Decimal(0),
                        avg_price=None,
                        commission=Decimal(0),
                        commission_asset=None,
                        error_msg="Limit price not reached and no persistence"))
                return None  # Terminal rejection
            # If remaining_qty is zero, it was handled by immediate fill.

        # This path should ideally be covered by immediate fill or active order block.
        # Fallback if logic above isn't complete.
        if not order_fully_processed_immediately:  # and not (active order condition)
            self.logger.warning(
                "Limit order %s fell through. Assuming rejection.",
                event.signal_id)
            await self._publish_simulated_report(
                event,
                SimulatedReportParams(
                    status="REJECTED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg="Limit order processing fell through"))

        return None  # Default case if no immediate full fill dict[str, Any] was returned

    def _add_to_active_limit_orders(
        self,
        original_event: "TradeSignalApprovedEvent",
        initial_event_signal_id_str: str) -> str:
        """Add a limit order to the active tracking dictionary."""
        internal_sim_order_id = str(uuid.uuid4())
        self._active_limit_orders[internal_sim_order_id] = {
            "original_event": original_event,  # This event contains the current remaining quantity
            "remaining_qty": original_event.quantity,
            "bar_count_waited": 0,
            "status": "NEW",  # Initial status when added
            "initial_event_signal_id_str": initial_event_signal_id_str,
            "placement_timestamp": datetime.now(UTC),  # When it was added to active list[Any]
        }
        self.logger.info(
            "Limit order (orig_signal_id: %s, qty: %s) added to active tracking. Internal ID: %s",
            initial_event_signal_id_str,
            original_event.quantity,
            internal_sim_order_id,
            source_module=self.__class__.__name__)
        return internal_sim_order_id

    def _check_limit_order_fill_on_bar(
        self,
        side: str,
        limit_price: Decimal,
        bar: pd.Series[Any]) -> tuple[Decimal | None, bool]:
        """Check if a limit order fills on a given bar.

        Returns the fill price (or better) and a boolean indicating if filled.
        Limit orders fill at the limit_price, or at the bar's open price if
        it's more favorable and the limit price was touched within the bar.

        Args:
            side: Order side ("BUY" or "SELL").
            limit_price: The order's limit price.
            bar: OHLCV bar data for the current period.

        Returns:
        -------
            A tuple[Any, ...] of (fill_price, is_filled) where:
              - fill_price: The price at which the order filled, or None if not filled
              - is_filled: True if the order filled, False otherwise
        """
        # Safe extraction of bar data with early return on errors
        bar_data = self._extract_bar_prices(bar)
        if bar_data is None:
            return None, False  # Error in bar data, cannot process

        bar_open, bar_low, bar_high = bar_data

        # Check fill conditions based on side
        side_upper = side.upper()
        if side_upper == "BUY":
            if bar_low <= limit_price:  # Buy limit price was reached
                # Return the better price (open if lower than limit, otherwise limit)
                fill_price = min(bar_open, limit_price)
                return fill_price, True
            # Buy limit price not reached
            return None, False
        if side_upper == "SELL":
            if bar_high >= limit_price:  # Sell limit price was reached
                # Return the better price (open if higher than limit, otherwise limit)
                fill_price = max(bar_open, limit_price)
                return fill_price, True
            # Sell limit price not reached
            return None, False
        # Invalid side
        self.logger.warning(
            "Invalid side '%s' in _check_limit_order_fill_on_bar. Expected 'BUY' or 'SELL'.",
            side,
            source_module=self.__class__.__name__)
        return None, False

    def _extract_bar_prices(self, bar: pd.Series[Any]) -> tuple[Decimal, Decimal, Decimal] | None:
        """Extract and convert open, low, high prices from a bar.

        Args:
            bar: OHLCV bar data

        Returns:
        -------
            Tuple of (open, low, high) prices as Decimal, or None if extraction failed
        """
        # Initialize result to None
        extracted_values = None

        # Attempt to extract and convert bar data
        try:
            open_val = Decimal(str(bar["open"]))
            low_val = Decimal(str(bar["low"]))
            high_val = Decimal(str(bar["high"]))
            # Store the values but don't return directly from the try block
            extracted_values = (open_val, low_val, high_val)
        except KeyError:
            self.logger.exception(
                "Bar data missing a required key (open/low/high) in _extract_bar_prices. Bar: %s",
                bar.to_dict(),
                source_module=self.__class__.__name__)
        except (TypeError, ValueError, decimal.InvalidOperation):
            self.logger.exception(
                "Data conversion error for OHLC values in _extract_bar_prices. Bar: %s",
                bar.to_dict(),
                source_module=self.__class__.__name__)
        except Exception:
            self.logger.exception(
                "Unexpected error in _extract_bar_prices. Bar: %s",
                bar.to_dict(),
                source_module=self.__class__.__name__)

        # Return the extracted values (or None if an exception occurred)
        return extracted_values

    def _determine_limit_order_liquidity(
        self,
        side: str,
        fill_price: Decimal,  # This is usually the limit_price
        bar: pd.Series[Any]) -> str:
        """Determine if a limit order fill is MAKER or TAKER.

        This is a heuristic for OHLCV data.
        """  # D205 addressed
        # Default to TAKER if logic is ambiguous, as it's usually more conservative for fees.
        try:
            bar_open = Decimal(str(bar["open"]))
            bar_high = Decimal(str(bar["high"]))
            bar_low = Decimal(str(bar["low"]))

            if side.upper() == "BUY":
                # If limit buy price was below the bar's open and market dropped to fill it,
                # likely MAKER.
                # If limit buy was at or above open but below high, and low touched it,
                # more like TAKER (aggressive limit).
                if fill_price < bar_open and fill_price >= bar_low:  # Touched from above
                    return "MAKER"
                return "TAKER"  # More aggressive or filled within typical spread
            if side.upper() == "SELL":
                # If limit sell price was above the bar's open and market rose to fill it,
                # likely MAKER.
                if fill_price > bar_open and fill_price <= bar_high:  # Touched from below
                    return "MAKER"
                return "TAKER"  # Default case for SELL orders
        except (TypeError, ValueError, decimal.InvalidOperation):
            self.logger.warning(
                "Could not determine maker/taker for limit order, "
                "defaulting to TAKER. Fill: %s, Bar: %s",
                fill_price,
                bar.to_dict(),  # E501: This can be long, but acceptable for a warning.
                source_module=self.__class__.__name__)
            # Removed exc_info parameter as it's not supported by the LoggerService interface
            return "TAKER"  # Default to TAKER in case of exceptions

        # Default return for any other case (invalid side)
        return "TAKER"  # Default to TAKER for unknown side values

    def _calculate_slippage(
        self,
        trading_pair: str,
        _side: str,  # ARG002: Prefixed unused 'side'
        base_price: Decimal,
        signal_timestamp: datetime,  # Timestamp of the original signal event
        order_quantity: Decimal | None = None,  # Original order quantity
        bar_volume: Decimal | None = None,  # Volume of the bar against which fill is attempted
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
                        (
                            "Market impact slippage model requires valid order_quantity and "
                            "bar_volume. Got Qty: %s, Vol: %s. Using zero slippage."
                        ),  # G004 fix
                        order_quantity,
                        bar_volume,
                        source_module=self.__class__.__name__)
                    return Decimal(0)

                try:
                    # Ensure order_quantity and bar_volume are positive for calculation
                    qty_ratio = order_quantity / bar_volume
                    impact_component_pct = self.slip_market_impact_factor * (
                        qty_ratio**self.slip_market_impact_exponent
                    )
                    total_slippage_pct = self.slip_market_impact_base_pct + impact_component_pct
                    slippage = base_price * total_slippage_pct
                except decimal.InvalidOperation:  # Catch potential math errors like negative power
                    self.logger.exception(  # G201/TRY400: Changed to .exception()
                        (
                            "Error in market_impact slippage calculation "
                            "(ratio: %s/%s): %s. Using base fixed slippage."
                        ),  # G004 fix
                        order_quantity,
                        bar_volume,
                        # TRY401: Removed str(e) as exc_info is True with .exception()
                        source_module=self.__class__.__name__)
                    slippage = base_price * self.slip_fixed_pct  # Fallback

            elif self.slippage_model == "volatility":
                # Get ATR for the bar the signal was generated on
                atr = self.data_service.get_atr(trading_pair, signal_timestamp)
                if atr is not None and atr > 0:
                    slippage = Decimal(str(atr)) * self.slip_atr_multiplier
                else:
                    self.logger.warning(
                        (
                            "Could not get ATR for %s at %s for slippage calc. Using zero."
                        ),  # G004 fix
                        trading_pair,
                        signal_timestamp,
                        source_module=self.__class__.__name__)
            else:
                self.logger.warning(
                    "Unknown slippage model: %s. Using zero.",
                    self.slippage_model,  # G004 fix
                    source_module=self.__class__.__name__)
        except Exception:  # Keep generic Exception for unknown errors in calculation
            self.logger.exception(  # G201/TRY400: Changed to .exception()
                "Error calculating slippage: %s",  # G004 fix, TRY401: Removed str(e)
                source_module=self.__class__.__name__)

        # Slippage is always adverse
        return abs(slippage)

    async def _publish_simulated_report(
        self,
        originating_event: "TradeSignalApprovedEvent",
        params: SimulatedReportParams,  # Use the new dataclass
        overrides: CustomReportOverrides | None = None) -> None:
        """Create and publish a simulated ExecutionReportEvent."""
        try:
            # Generate unique simulation order ID if not provided
            timestamp_micros = int(datetime.now(UTC).timestamp() * 1e6)  # DTZ003 fix

            if overrides and overrides.exchange_order_id:
                exchange_order_id = overrides.exchange_order_id
                client_order_id = (
                    overrides.client_order_id or f"sim_{originating_event.signal_id}_exit"
                )
            else:  # For entry orders or if overrides not provided / not complete
                exchange_order_id = f"sim_{originating_event.signal_id}_{timestamp_micros}"
                client_order_id = f"sim_{originating_event.signal_id}"

            report_order_type = originating_event.order_type
            if overrides and overrides.order_type:
                report_order_type = overrides.order_type

            report_side = originating_event.side
            if overrides and overrides.side:
                report_side = overrides.side

            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),  # DTZ003
                signal_id=originating_event.signal_id,
                exchange_order_id=exchange_order_id,
                client_order_id=client_order_id,
                trading_pair=originating_event.trading_pair,
                exchange=originating_event.exchange,
                order_status=params.status,  # Access via params
                order_type=report_order_type,  # Use resolved order type
                side=report_side,  # Use resolved side
                quantity_ordered=Decimal(
                    originating_event.quantity),  # Original quantity for entry
                quantity_filled=params.qty_filled,  # Access via params
                average_fill_price=params.avg_price,  # Access via params
                limit_price=originating_event.limit_price,
                stop_price=None,  # Not applicable for MVP fills
                commission=params.commission,  # Access via params
                commission_asset=params.commission_asset,  # Access via params
                timestamp_exchange=(
                    params.fill_timestamp
                    if params.fill_timestamp
                    else datetime.now(UTC)  # Access via params
                ),  # DTZ003
                error_message=params.error_msg,  # Access via params
                # Note: liquidity_type is stored in context for logging but not passed to
                # ExecutionReportEvent
            )
            await self.pubsub.publish(report)
            self.logger.debug(
                "Published simulated report: %s for %s",
                params.status,
                report.signal_id,  # G004 fix, Access via params
                source_module=self.__class__.__name__)
        except Exception:  # Keep generic Exception for unknown errors in publishing
            self.logger.exception(  # G201/TRY400: Changed to .exception()
                ("Failed to publish simulated execution report for signal %s"),  # G004 fix
                originating_event.signal_id,
                source_module=self.__class__.__name__)

    def _is_valid_bar_for_sl_tp(self, bar: pd.Series[Any], bar_timestamp: datetime) -> bool:
        """Validate that the bar is suitable for SL/TP checking.

        Args:
            bar: The OHLCV data for the current simulation time
            bar_timestamp: The timestamp of the current_bar

        Returns:
        -------
            bool: True if the bar is valid for SL/TP checks, False otherwise
        """
        if bar is None or bar_timestamp is None:
            self.logger.warning(
                "Invalid bar or timestamp provided for SL/TP check",
                source_module=self.__class__.__name__)
            return False

        # Check if bar has required OHLC data
        required_columns = ["high", "low"]
        for col in required_columns:
            if col not in bar:
                self.logger.warning(
                    "Bar missing required column %s for SL/TP check",
                    col,
                    source_module=self.__class__.__name__)
                return False

        return True

    def _check_sl_tp_trigger(
        self, sl_tp_data: dict[str, Any], bar_high: Decimal, bar_low: Decimal, bar_timestamp: datetime) -> dict[str, Any] | None:
        """Check if the current bar triggers any SL/TP conditions.

        Args:
            sl_tp_data: Dictionary containing SL/TP details
            bar_high: High price of the current bar
            bar_low: Low price of the current bar
            bar_timestamp: Timestamp of the current bar

        Returns:
        -------
            Optional[dict[str, Any]]: Dictionary with exit details if triggered, None otherwise
        """
        sl_price = sl_tp_data.get("sl")
        tp_price = sl_tp_data.get("tp")
        original_side = sl_tp_data.get("side")

        exit_price: Decimal | None = None
        exit_reason: str | None = None
        exit_order_type = "MARKET"  # Default for SL
        is_sl_trigger = False

        # Check Stop Loss (SL)
        if sl_price is not None and original_side is not None:
            side_upper = original_side.upper()
            if side_upper == "BUY" and bar_low <= sl_price:
                # SL price is the trigger, fill on next bar market conditions
                exit_price = sl_price
                exit_reason = f"Stop Loss triggered at {sl_price} (Bar Low: {bar_low})"
                exit_order_type = "MARKET"  # Explicitly market for SL
                is_sl_trigger = True
            elif side_upper == "SELL" and bar_high >= sl_price:
                exit_price = sl_price  # SL price is the trigger
                exit_reason = f"Stop Loss triggered at {sl_price} (Bar High: {bar_high})"
                exit_order_type = "MARKET"  # Explicitly market for SL
                is_sl_trigger = True

        # Check Take Profit (TP) - only if SL not already triggered
        if not is_sl_trigger and tp_price is not None and original_side is not None:
            side_upper = original_side.upper()
            if side_upper == "BUY" and bar_high >= tp_price:
                exit_price = tp_price
                exit_reason = f"Take Profit triggered at {tp_price} (Bar High: {bar_high})"
                exit_order_type = "LIMIT"  # TP is a limit order
            elif side_upper == "SELL" and bar_low <= tp_price:
                exit_price = tp_price
                exit_reason = f"Take Profit triggered at {tp_price} (Bar Low: {bar_low})"
                exit_order_type = "LIMIT"  # TP is a limit order

        if exit_price is not None and exit_reason is not None:
            return {
                "trigger_price": exit_price,  # This is the SL or TP price level
                "exit_reason": exit_reason,
                "exit_order_type": exit_order_type,
                "original_side": original_side,
                "entry_qty": sl_tp_data.get("entry_qty"),
                "trading_pair": sl_tp_data.get("pair"),
                "originating_event": sl_tp_data.get("entry_event"),
                "is_sl_trigger": is_sl_trigger,  # Pass this info
            }

        return None

    async def check_active_sl_tp(self, current_bar: pd.Series[Any], bar_timestamp: datetime) -> None:
        """Check SL/TP triggers for active positions with each new bar.

        Called by the backtesting engine for each new bar to monitor active positions.

        Args:
        ----
            current_bar: The OHLCV data for the current simulation time
            bar_timestamp: The timestamp of the current_bar (usually its open time)
        """
        if not self._is_valid_bar_for_sl_tp(current_bar, bar_timestamp):
            return

        try:
            bar_high = Decimal(str(current_bar["high"]))
            bar_low = Decimal(str(current_bar["low"]))
        except (TypeError, ValueError, decimal.InvalidOperation):  # Removed 'as e'
            self.logger.exception(  # G201/TRY400: Changed to .exception()
                (
                    "Could not convert current bar high/low to Decimal for SL/TP check at "
                    "%s. Bar: %s."
                ),  # G004 fix, TRY401: Removed Error: %s and str(e)
                bar_timestamp,
                current_bar.to_dict(),  # E501: This might still be long
                source_module=self.__class__.__name__)
            return

        triggered_position_ids = []
        for position_id, sl_tp_data in list[Any](self._active_sl_tp.items()):
            if bar_timestamp <= sl_tp_data["entry_ts"]:
                continue

            exit_details = self._check_sl_tp_trigger(sl_tp_data, bar_high, bar_low, bar_timestamp)

            if exit_details:
                # Pass current_bar (trigger bar for TP/SL).
                if exit_details["is_sl_trigger"]:
                    await self._process_sl_exit(
                        position_id,
                        sl_tp_data,
                        exit_details,
                        current_bar,
                        bar_timestamp)
                else:  # TP Trigger
                    await self._process_tp_exit(
                        position_id,
                        sl_tp_data,
                        exit_details,
                        current_bar,
                        bar_timestamp)
                triggered_position_ids.append(position_id)

        for pos_id in triggered_position_ids:
            if pos_id in self._active_sl_tp:
                del self._active_sl_tp[pos_id]
                self.logger.debug(
                    "Removed SL/TP monitoring for position %s",
                    pos_id,  # G004 fix
                    source_module=self.__class__.__name__)

    # Commented out code and duplicate method implementations have been removed

    async def _process_tp_exit(
        self,
        position_id: str,
        sl_tp_data: dict[str, Any],  # Contains 'first_entry_event_object'
        exit_details: dict[str, Any],
        tp_trigger_bar: pd.Series[Any],  # Bar on which TP triggered
        tp_trigger_bar_timestamp: datetime) -> None:
        """Process the exit of a position due to a Take Profit trigger."""
        self.logger.info(
            "TP TRIGGERED: %s for position %s (%s). Details: %s",
            exit_details["exit_reason"],
            position_id,
            exit_details["trading_pair"],
            exit_details,
            source_module=self.__class__.__name__)

        originating_event_of_position = sl_tp_data["first_entry_event_object"]
        fill_price = exit_details["trigger_price"]  # TP fills at its price
        filled_qty = exit_details["entry_qty"]
        exit_side = "SELL" if exit_details["original_side"].upper() == "BUY" else "BUY"

        # Determine liquidity for TP fill (usually MAKER if it was resting)
        liquidity_type = self._determine_limit_order_liquidity(
            exit_side,
            fill_price,
            tp_trigger_bar)
        commission_pct = self.maker_fee_pct if liquidity_type == "MAKER" else self.taker_fee_pct
        commission_amount = abs(filled_qty * fill_price * commission_pct)
        _, quote_asset = exit_details["trading_pair"].split("/")
        commission_asset = quote_asset.upper()

        report_overrides = CustomReportOverrides(
            exchange_order_id=(
                f"sim_tp_exit_{position_id}_" f"{int(datetime.now(UTC).timestamp() * 1e6)}"
            ),
            client_order_id=f"sim_tp_exit_{position_id}",
            order_type=exit_details["exit_order_type"],  # Should be "LIMIT" or "TAKE_PROFIT_LIMIT"
            side=exit_side)

        await self._publish_simulated_report(
            originating_event=originating_event_of_position,
            params=SimulatedReportParams(
                status="FILLED",
                qty_filled=filled_qty,
                avg_price=fill_price,
                commission=commission_amount,
                commission_asset=commission_asset,
                fill_timestamp=tp_trigger_bar_timestamp,  # TP fills on the trigger bar
                liquidity_type=liquidity_type),
            overrides=report_overrides)

    async def _process_sl_exit(
        self,
        position_id: str,
        sl_tp_data: dict[str, Any],  # Contains 'first_entry_event_object'
        exit_details: dict[str, Any],
        _sl_trigger_bar: pd.Series[Any],  # Bar on which SL triggered, passed for context if needed
        sl_trigger_bar_timestamp: datetime) -> None:
        """Process the exit of a position due to a Stop Loss trigger."""
        self.logger.info(
            (
                "SL TRIGGERED: %s for position %s (%s). "
                "Simulating market exit on next bar. Details: %s"
            ),
            exit_details["exit_reason"],
            position_id,
            exit_details["trading_pair"],
            exit_details,
            source_module=self.__class__.__name__)

        originating_event_of_position = sl_tp_data["first_entry_event_object"]
        exit_qty = exit_details["entry_qty"]
        exit_side = "SELL" if exit_details["original_side"].upper() == "BUY" else "BUY"

        # Construct a temporary event for simulating the SL market order
        from gal_friday.core.events import (
            TradeSignalApprovedEvent as ConcreteEvent,  # type: ignore
        )

        sl_market_exit_payload = {
            "event_id": uuid.uuid4(),
            "timestamp": sl_trigger_bar_timestamp,  # SL triggered on this bar, market fill on next
            "signal_id": f"sl_exit_{position_id}_{uuid.uuid4()}",
            "trading_pair": exit_details["trading_pair"],
            "exchange": sl_tp_data["exchange"],
            "side": exit_side,
            "order_type": "MARKET",
            "quantity": exit_qty,
            "limit_price": None,
            "sl_price": None,  # No SL for the SL exit order itself
            "tp_price": None,  # No TP for the SL exit order itself
            "source_module": self.__class__.__name__,
            "risk_parameters": {
                "reason": "SL_EXIT_MARKET_SIMULATION",
                "original_position_id": position_id,
            },
        }
        sl_market_exit_event = ConcreteEvent(**sl_market_exit_payload)

        next_bar_for_sl_fill = await self._get_next_bar_data(sl_market_exit_event)

        if next_bar_for_sl_fill is None:
            self.logger.error(
                (
                    "Cannot simulate SL market exit for position %s: No subsequent "
                    "bar data found after trigger bar %s."
                ),
                position_id,
                sl_trigger_bar_timestamp,
                source_module=self.__class__.__name__)
            # Publish an error report for the SL exit attempt
            await self._publish_simulated_report(
                originating_event=originating_event_of_position,
                params=SimulatedReportParams(
                    status="ERROR",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=(
                        f"No data for SL market fill after trigger at {sl_trigger_bar_timestamp}"
                    ),
                    fill_timestamp=sl_trigger_bar_timestamp,  # Timestamp of error recognition
                ),
                overrides=CustomReportOverrides(
                    order_type="MARKET",
                    side=exit_side,
                    exchange_order_id=(
                        f"sim_sl_err_{position_id}_" f"{int(datetime.now(UTC).timestamp() * 1e6)}"
                    ),
                    client_order_id=f"sim_sl_err_{position_id}"))
            return

        sl_fill_timestamp = next_bar_for_sl_fill.name
        if not isinstance(sl_fill_timestamp, datetime):
            try:
                sl_fill_timestamp = pd.to_datetime(str(sl_fill_timestamp)).to_pydatetime(
                    warn=False)
                if sl_fill_timestamp.tzinfo is None:
                    sl_fill_timestamp = sl_fill_timestamp.replace(tzinfo=UTC)

            except Exception:
                self.logger.warning(
                    "SL fill timestamp conversion error. Using current UTC.",
                    source_module=self.__class__.__name__)
                sl_fill_timestamp = datetime.now(UTC)

        # Simulate the market order fill
        # (slippage, partial fill due to liquidity, taker commission)
        market_fill_result = await self._simulate_market_order(
            sl_market_exit_event,  # Use the specifically created event for this market order
            next_bar_for_sl_fill,
            self.taker_fee_pct,
            sl_fill_timestamp)

        report_overrides = CustomReportOverrides(
            exchange_order_id=(
                f"sim_sl_exit_{position_id}_" f"{int(datetime.now(UTC).timestamp() * 1e6)}"
            ),
            client_order_id=f"sim_sl_exit_{position_id}",
            order_type="MARKET",  # Or specific like "STOP_MARKET"
            side=exit_side)

        await self._publish_simulated_report(
            # Report against the original position
            originating_event=originating_event_of_position,
            params=SimulatedReportParams(
                status=market_fill_result["status"],
                qty_filled=market_fill_result["quantity"],
                avg_price=market_fill_result["fill_price"],
                commission=market_fill_result["commission"],
                commission_asset=market_fill_result["commission_asset"],
                error_msg=market_fill_result.get("error_msg") or exit_details["exit_reason"],
                fill_timestamp=market_fill_result["timestamp"],  # Actual fill time from market sim
                liquidity_type=market_fill_result.get("liquidity_type", "TAKER")),
            overrides=report_overrides)

        if market_fill_result["quantity"] < exit_qty:
            self.logger.warning(
                "SL Market Exit for position %s was PARTIALLY FILLED (%s/%s). "
                "Portfolio must handle this discrepancy. Current simulation assumes "
                "this partial fill closes the tracked SL/TP state.",
                position_id,
                market_fill_result["quantity"],
                exit_qty,
                source_module=self.__class__.__name__)

    async def _process_sl_tp_exit(
        self,
        position_id: str,
        _sl_tp_data: dict[str, Any],  # ARG002: Prefixed unused sl_tp_data
        exit_details: dict[str, Any],  # UP006: Dict -> dict[str, Any]
        bar_timestamp: datetime,  # This is the timestamp of the bar that triggered the SL/TP
    ) -> None:
        """Process the exit of a position due to SL/TP trigger."""
        originating_event = exit_details["originating_event"]
        exit_qty = exit_details["entry_qty"]
        exit_side = "SELL" if exit_details["original_side"].upper() == "BUY" else "BUY"
        exit_order_type_from_trigger = exit_details["exit_order_type"]
        initial_exit_reason = exit_details["exit_reason"]
        trigger_price = exit_details["trigger_price"]
        is_sl_trigger = exit_details["is_sl_trigger"]
        # Ensure trading_pair is defined from exit_details
        trading_pair = exit_details["trading_pair"]

        log_msg_base = (
            f"{initial_exit_reason} for position {position_id} "
            f"({trading_pair}) Trigger Price: {trigger_price}, "
            f"Qty: {exit_qty}, Exit Side: {exit_side}, "
            f"Order Type: {exit_order_type_from_trigger}, "  # Use exit_order_type_from_trigger
            f"Trigger Bar: {bar_timestamp}"
        )
        self.logger.info(log_msg_base, source_module=self.__class__.__name__)

        outcome_details: dict[str, Any] = {  # Changed Dict to dict[str, Any]
            "status": "ERROR",
            "fill_price": None,
            "commission": Decimal(0),
            "commission_asset": None,
            "error_msg": "Unknown SL/TP processing error",
            "liquidity_type": None,
            "actual_fill_timestamp": bar_timestamp,
            "qty_filled_for_report": Decimal(0),
        }
        sl_market_exit_signal_id_for_report = None

        if is_sl_trigger:
            market_exit_params = MarketExitParams(
                originating_event=originating_event,
                quantity=exit_qty,
                side=exit_side,
                trading_pair=trading_pair,
                exchange=originating_event.exchange,
                trigger_timestamp=bar_timestamp)
            sl_outcome = await self._simulate_sl_market_order_for_exit(
                position_id,
                market_exit_params)
            outcome_details.update(sl_outcome)
            sl_market_exit_signal_id_for_report = sl_outcome.get("sl_market_exit_signal_id")
            outcome_details["error_msg"] = sl_outcome.get("error_msg") or initial_exit_reason

        elif exit_order_type_from_trigger == "LIMIT":  # Use exit_order_type_from_trigger
            tp_outcome = self._calculate_tp_limit_fill_for_exit(
                position_id,
                trigger_price,
                exit_qty,
                trading_pair,
                bar_timestamp,
                # Use defined trading_pair
            )
            outcome_details.update(tp_outcome)
            outcome_details["error_msg"] = initial_exit_reason

        else:
            unknown_type_error = (
                f"Unknown exit_order_type '{exit_order_type_from_trigger}' for SL/TP "
                f"processing of {position_id}. Expected MARKET (for SL) or LIMIT (for TP)."
            )
            self.logger.error(unknown_type_error, source_module=self.__class__.__name__)
            outcome_details["error_msg"] = unknown_type_error
            # outcome_details default status is ERROR, qty_filled is 0

        # Determine final quantity for the report based on status
        final_qty_to_report = outcome_details["qty_filled_for_report"]
        if outcome_details["status"] not in ["FILLED", "PARTIALLY_FILLED"]:
            final_qty_to_report = Decimal(0)
        elif (
            outcome_details["status"] == "FILLED"
            and final_qty_to_report == Decimal(0)
            and not is_sl_trigger
        ):  # TP path, condition split
            final_qty_to_report = exit_qty

        # Prepare and publish report
        custom_client_order_id_val = (
            sl_market_exit_signal_id_for_report
            if is_sl_trigger and sl_market_exit_signal_id_for_report
            else f"sim_exit_{originating_event.signal_id}"
        )
        exit_exchange_order_id = (
            f"sim_exit_{originating_event.signal_id}_"
            f"{int(datetime.now(UTC).timestamp() * 1e6)}"  # Split f-string
        )

        report_overrides = CustomReportOverrides(
            exchange_order_id=exit_exchange_order_id,
            client_order_id=custom_client_order_id_val,
            order_type=exit_order_type_from_trigger,  # Use exit_order_type_from_trigger
            side=exit_side)

        # Error message for report should only be set if status indicates an issue
        report_error_msg = None
        if outcome_details["status"] not in ["FILLED", "PARTIALLY_FILLED"]:
            report_error_msg = outcome_details["error_msg"]
        elif outcome_details["error_msg"] and outcome_details["status"] in [
            "FILLED",
            "PARTIALLY_FILLED",
        ]:  # Condition split
            report_error_msg = initial_exit_reason  # Use initial_exit_reason

        await self._publish_simulated_report(
            originating_event=originating_event,
            params=SimulatedReportParams(
                status=outcome_details["status"],
                qty_filled=final_qty_to_report,
                avg_price=outcome_details["fill_price"],
                commission=outcome_details["commission"],
                commission_asset=outcome_details["commission_asset"],
                error_msg=report_error_msg,
                fill_timestamp=outcome_details["actual_fill_timestamp"],
                liquidity_type=outcome_details["liquidity_type"]),
            overrides=report_overrides)

    async def _simulate_sl_market_order_for_exit(
        self,
        position_id: str,  # For logging and identification
        params: MarketExitParams) -> dict[str, Any]:
        """Simulate an SL market order exit and return fill details."""
        self.logger.debug(
            "Simulating SL market exit for %s on bar after %s.",
            position_id,
            params.trigger_timestamp,  # Use param field
            source_module=self.__class__.__name__)

        from gal_friday.core.events import (
            TradeSignalApprovedEvent as ConcreteEvent,  # Local import
        )

        sl_market_exit_signal_id = f"sl_exit_{params.originating_event.signal_id}_{uuid.uuid4()}"
        # Create SL market exit signal ID for the event
        # (removed unused dummy_event_payload dictionary)
        # Create a proper TradeSignalApprovedEvent instead of using **kwargs directly
        sl_exit_event_for_sim = ConcreteEvent(
            signal_id=params.originating_event.signal_id,  # Use the original event's signal_id
            timestamp=params.trigger_timestamp,
            trading_pair=params.trading_pair,
            exchange=params.exchange,
            side=params.side,
            order_type="MARKET",  # Always market for SL
            quantity=params.quantity,
            sl_price=Decimal("0"),  # Required field, set to 0 for SL exit event
            tp_price=Decimal("0"),  # Required field, set to 0 for SL exit event
            risk_parameters={
                "reason": "SL_EXIT_INTERNAL_SIM",
                "original_signal_id": str(params.originating_event.signal_id),
            },
            limit_price=None,  # Market order has no limit price
            source_module=self.__class__.__name__,  # Add missing required parameter
            event_id=uuid.uuid4(),  # Add missing required parameter with a new UUID
        )

        next_bar_for_sl_fill = await self._get_next_bar_data(sl_exit_event_for_sim)

        if next_bar_for_sl_fill is not None:
            actual_fill_timestamp = next_bar_for_sl_fill.name
            if not isinstance(actual_fill_timestamp, datetime):
                try:
                    actual_fill_timestamp = pd.to_datetime(
                        str(actual_fill_timestamp)).to_pydatetime(warn=False)
                    if actual_fill_timestamp.tzinfo is None:
                        actual_fill_timestamp = actual_fill_timestamp.replace(tzinfo=UTC)
                except ValueError:
                    self.logger.warning(
                        "SL fill timestamp conversion error for %s. Using current UTC.",
                        actual_fill_timestamp,
                        source_module=self.__class__.__name__)
                    actual_fill_timestamp = datetime.now(UTC)

            market_fill_result = await self._simulate_market_order(
                sl_exit_event_for_sim,
                next_bar_for_sl_fill,
                self.taker_fee_pct,
                actual_fill_timestamp)

            if (
                market_fill_result["quantity"] < params.quantity  # Use param
                and market_fill_result["status"] != "ERROR"
            ):
                self.logger.error(
                    "SL Market Exit for %s was only PARTIALLY FILLED (%s/%s). "
                    "Portfolio must handle this. Reporting actual filled quantity.",
                    position_id,
                    market_fill_result["quantity"],
                    params.quantity,  # Use param
                    source_module=self.__class__.__name__)

            return {
                "status": market_fill_result["status"],
                "fill_price": market_fill_result["fill_price"],
                "commission": market_fill_result["commission"],
                "commission_asset": market_fill_result["commission_asset"],
                "error_msg": market_fill_result.get("error_msg"),
                "liquidity_type": market_fill_result.get("liquidity_type", "TAKER"),
                "actual_fill_timestamp": market_fill_result["timestamp"],
                "sl_market_exit_signal_id": sl_market_exit_signal_id,
                "qty_filled_for_report": market_fill_result["quantity"],
            }
        error_msg = (
            f"No next bar data to simulate SL market exit for {position_id} "
            f"after trigger bar {params.trigger_timestamp}."  # Use param
        )
        self.logger.error(error_msg, source_module=self.__class__.__name__)
        return {
            "status": "ERROR",
            "fill_price": None,
            "commission": Decimal(0),
            "commission_asset": None,
            "error_msg": error_msg,
            "liquidity_type": None,
            "actual_fill_timestamp": params.trigger_timestamp,  # Use param
            "sl_market_exit_signal_id": sl_market_exit_signal_id,
            "qty_filled_for_report": Decimal(0),
        }

    def _calculate_tp_limit_fill_for_exit(
        self,
        position_id: str,  # For logging
        trigger_price: Decimal,
        exit_qty: Decimal,
        trading_pair: str,
        bar_timestamp: datetime,  # TP Fill timestamp (trigger bar)
    ) -> dict[str, Any]:
        """Calculate TP limit order fill details. Assume fill at trigger_price."""
        self.logger.info(
            "TP for position %s to be filled as LIMIT on bar %s at price %s.",
            position_id,
            bar_timestamp,
            trigger_price,
            source_module=self.__class__.__name__)

        final_fill_price = trigger_price
        exit_value = exit_qty * final_fill_price
        # Assuming TAKER fees for TP as a simplification, matching original logic.
        # Could be enhanced to use _determine_limit_order_liquidity if bar data passed.
        commission_amount = abs(exit_value * self.taker_fee_pct)
        final_liquidity_type = "TAKER"
        _, quote_asset = trading_pair.split("/")
        commission_asset = quote_asset.upper()

        return {
            "status": "FILLED",
            "fill_price": final_fill_price,
            "commission": commission_amount,
            "commission_asset": commission_asset,
            "error_msg": None,  # TP assumed to fill if this path is taken
            "liquidity_type": final_liquidity_type,
            "actual_fill_timestamp": bar_timestamp,
            "qty_filled_for_report": exit_qty,  # TP fills fully
        }

    # --- Public methods for test inspection ---
    def is_sl_tp_active(self, position_id: str) -> bool:
        """Check if SL/TP is active for a given position ID. For test inspection."""
        return position_id in self._active_sl_tp

    def get_active_sl_tp_entry_details(self, position_id: str) -> dict[str, Any] | None:
        """Get active SL/TP entry details for a given position ID. For test inspection."""
        if position_id in self._active_sl_tp:
            # Return a copy to prevent external modification of internal state
            return self._active_sl_tp[position_id].copy()
        return None

    def get_active_limit_order_count(self) -> int:
        """Return the number of active limit orders. For test inspection."""
        return len(self._active_limit_orders)

    def _register_or_update_sl_tp(
        self,
        initial_event_signal_id_str: str,
        fill_details: FillDetails) -> None:
        """Register a new SL/TP or update an existing one for a position."""
        position_id = initial_event_signal_id_str

        if fill_details.event.sl_price is None and fill_details.event.tp_price is None:
            self.logger.debug(
                "No SL or TP price provided for signal %s. Skipping SL/TP registration.",
                position_id)
            return

        if position_id not in self._active_sl_tp:
            if fill_details.qty_increment <= Decimal(0):  # Do not register if no quantity filled
                return

            self._active_sl_tp[position_id] = {
                "sl_price": fill_details.event.sl_price,
                "tp_price": fill_details.event.tp_price,
                "original_entry_side": fill_details.side,
                "pair": fill_details.trading_pair,
                "cumulative_filled_entry_qty": fill_details.qty_increment,
                "first_entry_ts": fill_details.timestamp,
                "first_entry_event_object": fill_details.event,
                # Store the whole event
                "exchange": fill_details.exchange,
                "last_entry_order_type": fill_details.order_type,
                "sl_tp_triggered_reason": None,  # Will be set if SL/TP processing occurs
            }
            self.logger.info(
                "New SL/TP registered for position %s (%s %s at %s). Qty: %s. SL: %s, TP: %s",
                position_id,
                fill_details.side,
                fill_details.trading_pair,
                fill_details.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                fill_details.qty_increment,
                fill_details.event.sl_price,
                fill_details.event.tp_price,
                source_module=self.__class__.__name__)
        else:
            self._active_sl_tp[position_id][
                "cumulative_filled_entry_qty"
            ] += fill_details.qty_increment
            self._active_sl_tp[position_id]["last_entry_order_type"] = fill_details.order_type
            self.logger.info(
                "Additional quantity %s added to SL/TP position %s. New total: %s.",
                fill_details.qty_increment,
                position_id,
                self._active_sl_tp[position_id]["cumulative_filled_entry_qty"],
                source_module=self.__class__.__name__)

    async def _handle_limit_order_timeout(
        self,
        internal_sim_order_id: str,
        order_details: dict[str, Any],  # Contains original_event, remaining_qty, etc.
        _current_bar: pd.Series[Any],  # Bar on which timeout is detected (unused)
        bar_timestamp: datetime,  # Timestamp of current_bar
    ) -> None:
        """Handle a limit order that has timed out."""
        original_event = order_details["original_event"]
        remaining_qty = order_details["remaining_qty"]
        initial_event_signal_id_str = order_details["initial_event_signal_id_str"]

        self.logger.info(
            "Limit order (internal_id: %s, orig_signal_id: %s, qty: %s) timed out on bar %s.",
            internal_sim_order_id,
            initial_event_signal_id_str,
            remaining_qty,
            bar_timestamp,
            source_module=self.__class__.__name__)

        if self.limit_order_timeout_action.upper() == "CANCEL":
            await self._publish_simulated_report(
                originating_event=original_event,  # Report against the event for this limit order
                params=SimulatedReportParams(
                    status="CANCELED",
                    qty_filled=Decimal(0),  # No quantity filled for the cancellation itself
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=f"Limit order timed out after {self.limit_order_timeout_bars} bars.",
                    fill_timestamp=bar_timestamp,  # Timestamp of cancellation decision
                ))
        elif self.limit_order_timeout_action.upper() == "MARKET":
            self.logger.info(
                "Converting timed-out limit order %s to Market order.",
                internal_sim_order_id,
                source_module=self.__class__.__name__)
            # Publish EXPIRED for the original limit order part
            await self._publish_simulated_report(
                originating_event=original_event,
                params=SimulatedReportParams(
                    status="EXPIRED",  # Or "REPLACED" if there's a link_id concept
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg="Limit order timed out, converting to market.",
                    fill_timestamp=bar_timestamp))

            # Create a new market TradeSignalApprovedEvent for the remaining quantity
            from gal_friday.core.events import (
                TradeSignalApprovedEvent as ConcreteEvent,  # type: ignore
            )

            market_conversion_payload = {
                "event_id": uuid.uuid4(),
                "timestamp": bar_timestamp,
                # Signal effectively generated now for market conversion
                "signal_id": f"market_conv_{initial_event_signal_id_str}_{uuid.uuid4()}",
                "trading_pair": original_event.trading_pair,
                "exchange": original_event.exchange,
                "side": original_event.side,
                "order_type": "MARKET",
                "quantity": remaining_qty,
                "limit_price": None,
                "sl_price": original_event.sl_price,  # Carry over SL/TP from original intent
                "tp_price": original_event.tp_price,
                "source_module": self.__class__.__name__,  # Indication of internal generation
                "risk_parameters": {
                    "reason": "LIMIT_TIMEOUT_MARKET_CONVERSION",
                    "original_internal_limit_order_id": internal_sim_order_id,
                    "original_signal_id_for_sl_tp": initial_event_signal_id_str,
                    # CRITICAL for SL/TP
                },
            }
            market_conversion_event = ConcreteEvent(**market_conversion_payload)

            # Simulate this new market order
            # The fill will occur on the bar *after* current_bar (due to _get_next_bar_data logic)
            next_bar_for_market_conv = await self._get_next_bar_data(market_conversion_event)
            if next_bar_for_market_conv is not None:
                fill_ts_market_conv = next_bar_for_market_conv.name
                if not isinstance(fill_ts_market_conv, datetime):
                    try:
                        fill_ts_market_conv_dt = pd.to_datetime(str(fill_ts_market_conv))
                        fill_ts_market_conv = fill_ts_market_conv_dt.to_pydatetime(warn=False)
                        if fill_ts_market_conv.tzinfo is None:
                            fill_ts_market_conv = fill_ts_market_conv.replace(tzinfo=UTC)
                    except Exception:
                        fill_ts_market_conv = datetime.now(UTC)

                market_fill_result = await self._simulate_market_order(
                    market_conversion_event,
                    next_bar_for_market_conv,
                    self.taker_fee_pct,
                    fill_ts_market_conv)

                # Publish report for the market fill
                # The originating event for this report is the new market_conversion_event
                # SL/TP will be set based on this market_conversion_event
                # which carries SL/TP prices
                # and the crucial 'original_signal_id_for_sl_tp' in risk_parameters.

                if market_fill_result["status"] in [
                    "FILLED",
                    "PARTIALLY_FILLED",
                ] and market_fill_result["quantity"] > Decimal(0):
                    self._register_or_update_sl_tp(
                        initial_event_signal_id_str=initial_event_signal_id_str,
                        fill_details=FillDetails(
                            qty_increment=market_fill_result["quantity"],
                            timestamp=market_fill_result["timestamp"],
                            trading_pair=market_conversion_event.trading_pair,
                            side=market_conversion_event.side,
                            exchange=market_conversion_event.exchange,
                            order_type=market_conversion_event.order_type,
                            event=market_conversion_event,  # Pass the market_conversion_event
                        ))

                await self._publish_simulated_report(
                    originating_event=market_conversion_event,
                    # Report against the new market order
                    params=SimulatedReportParams(
                        status=market_fill_result["status"],
                        qty_filled=market_fill_result["quantity"],
                        avg_price=market_fill_result["fill_price"],
                        commission=market_fill_result["commission"],
                        commission_asset=market_fill_result["commission_asset"],
                        error_msg=market_fill_result.get("error_msg"),
                        fill_timestamp=market_fill_result["timestamp"],
                        liquidity_type=market_fill_result.get("liquidity_type")))
            else:
                self.logger.error(
                    "No next bar data to execute market conversion for timed-out limit order %s.",
                    internal_sim_order_id,
                    source_module=self.__class__.__name__)
                # Report failure of market conversion
                await self._publish_simulated_report(
                    originating_event=market_conversion_event,
                    params=SimulatedReportParams(
                        status="ERROR",
                        qty_filled=Decimal(0),
                        avg_price=None,
                        commission=Decimal(0),
                        commission_asset=None,
                        error_msg="No data for market conversion of timed-out limit.",
                        fill_timestamp=bar_timestamp))
        else:
            self.logger.warning(
                "Unknown limit_order_timeout_action: %s for order %s.",
                self.limit_order_timeout_action,
                internal_sim_order_id,
                source_module=self.__class__.__name__)
            # Default to CANCEL if action is unknown
            await self._publish_simulated_report(
                originating_event=original_event,
                params=SimulatedReportParams(
                    status="CANCELED",
                    qty_filled=Decimal(0),
                    avg_price=None,
                    commission=Decimal(0),
                    commission_asset=None,
                    error_msg=(
                        f"Unknown timeout action '{self.limit_order_timeout_action}', "
                        "order canceled."
                    ),
                    fill_timestamp=bar_timestamp))

    async def check_active_limit_orders(
        self,
        current_bar: pd.Series[Any],
        bar_timestamp: datetime) -> None:
        """Check active limit orders against the current bar for fills or timeouts.

        This method should be called by the backtesting engine for each new bar.
        """
        if not self._active_limit_orders:
            return

        # Iterate over a copy of keys to allow modification of the dict[str, Any] during iteration
        for internal_sim_order_id in list[Any](self._active_limit_orders.keys()):
            order_details = self._active_limit_orders.get(internal_sim_order_id)
            if not order_details:  # Should not happen if keys are from the dict[str, Any]
                continue

            order_details["bar_count_waited"] += 1
            # Event for this specific limit order part
            original_event = order_details["original_event"]
            current_remaining_qty = order_details["remaining_qty"]
            initial_event_signal_id_str = order_details["initial_event_signal_id_str"]

            fill_price_if_met, can_fill_this_bar = self._check_limit_order_fill_on_bar(
                original_event.side,
                original_event.limit_price,
                current_bar)

            if can_fill_this_bar and fill_price_if_met is not None:
                available_volume_at_bar = Decimal(str(current_bar.get("volume", "0")))
                max_fillable_qty_liquidity = available_volume_at_bar * self._fill_liquidity_ratio

                qty_filled_this_bar = min(current_remaining_qty, max_fillable_qty_liquidity)

                if qty_filled_this_bar > Decimal("1e-12"):
                    liquidity_type = self._determine_limit_order_liquidity(
                        original_event.side,
                        fill_price_if_met,
                        current_bar)
                    commission_pct = (
                        self.maker_fee_pct if liquidity_type == "MAKER" else self.taker_fee_pct
                    )
                    fill_value = qty_filled_this_bar * fill_price_if_met
                    commission_amount = abs(fill_value * commission_pct)
                    _, quote_asset_str = original_event.trading_pair.split("/")
                    commission_asset = quote_asset_str.upper()

                    order_details["remaining_qty"] -= qty_filled_this_bar
                    order_details["status"] = (
                        "ACTIVE_PARTIAL"
                        if order_details["remaining_qty"] > Decimal("1e-12")
                        else "FILLED"
                    )

                    report_status = (
                        "PARTIALLY_FILLED"
                        if order_details["status"] == "ACTIVE_PARTIAL"
                        else "FILLED"
                    )

                    await self._publish_simulated_report(
                        originating_event=original_event,
                        # Reports the fill for this part of the order
                        params=SimulatedReportParams(
                            status=report_status,
                            qty_filled=qty_filled_this_bar,
                            avg_price=fill_price_if_met,
                            commission=commission_amount,
                            commission_asset=commission_asset,
                            fill_timestamp=bar_timestamp,  # Fill occurs on this bar
                            liquidity_type=liquidity_type))

                    # Register SL/TP for the filled portion
                    # The originating_event_for_sl_tp_setup should be the original_event
                    # as it contains the SL/TP prices from the strategy.
                    self._register_or_update_sl_tp(
                        initial_event_signal_id_str=initial_event_signal_id_str,
                        fill_details=FillDetails(
                            qty_increment=qty_filled_this_bar,
                            timestamp=bar_timestamp,
                            trading_pair=original_event.trading_pair,
                            side=original_event.side,
                            exchange=original_event.exchange,
                            order_type=original_event.order_type,
                            event=original_event,  # Pass the original_event for this limit part
                        ))

                    if order_details["status"] == "FILLED":
                        self.logger.info(
                            ("Active limit order (internal_id: %s) fully filled."),
                            internal_sim_order_id,
                            source_module=self.__class__.__name__)
                        del self._active_limit_orders[internal_sim_order_id]
                        continue  # Move to next active order
                    # Partially filled, update status and continue monitoring
                    self._active_limit_orders[internal_sim_order_id] = order_details

            # Check for timeout if not fully filled and removed this bar
            # Re-check because it might have been deleted
            if (
                internal_sim_order_id in self._active_limit_orders
                and order_details["bar_count_waited"] >= self.limit_order_timeout_bars
            ):
                await self._handle_limit_order_timeout(
                    internal_sim_order_id,
                    order_details,
                    current_bar,
                    bar_timestamp)
                # Ensure timeout handler removed it
                if internal_sim_order_id in self._active_limit_orders:
                    del self._active_limit_orders[internal_sim_order_id]


def _check_bar_for_trigger(
    original_side: str,
    trigger_type: str,  # "SL" or "TP"
    target_price: Decimal,
    bar_low: Decimal,
    bar_high: Decimal) -> bool:
    """Check if the given bar's low/high prices trigger the specified SL/TP condition.

    Args:
        original_side: The side of the original trade ("BUY" or "SELL").
        trigger_type: The type of trigger to check ("SL" or "TP").
        target_price: The SL or TP price to check against.
        bar_low: The low price of the current bar.
        bar_high: The high price of the current bar.

    Returns:
    -------
        True if the bar triggers the condition, False otherwise.
    """
    tt_upper = trigger_type.upper()

    if tt_upper == "SL":
        return (original_side == "BUY" and bar_low <= target_price) or (
            original_side == "SELL" and bar_high >= target_price
        )
    if tt_upper == "TP":  # Changed from elif to if, satisfying SIM102 implicitly by structure
        return (original_side == "BUY" and bar_high >= target_price) or (
            original_side == "SELL" and bar_low <= target_price
        )
    return False


async def _find_and_test_single_trigger(
    sim_exec: "SimulatedExecutionHandler",  # Forward reference for type hint
    pos_id_to_check: str,
    position_details: dict[str, Any],
    trigger_type: str,  # "SL" or "TP"
    source_module_suffix: str = "TriggerTestHelper") -> None:
    """Find a trigger bar for SL or TP and test it."""
    source_module = f"{trigger_type.upper()}{source_module_suffix}"

    entry_ts = position_details["entry_ts"]
    target_price_key = trigger_type.lower()
    if target_price_key not in position_details:
        sim_exec.logger.error(
            "%s check for %s: Key '%s' not found in position_details. Details: %s",
            trigger_type.upper(),
            pos_id_to_check,
            target_price_key,
            position_details,
            source_module=source_module)
        return
    target_price = position_details[target_price_key]
    original_side = position_details["side"].upper()
    test_trading_pair = position_details.get("pair", "XRP/USD")

    if target_price is None:
        sim_exec.logger.info(
            "%s check for %s: No target price (%s is None). Skipping.",
            trigger_type.upper(),
            pos_id_to_check,
            trigger_type.lower(),
            source_module=source_module)
        return

    sim_exec.logger.info(
        "%s Check for %s (%s on %s): Entry TS: %s, Target %s Price: %s",
        trigger_type.upper(),
        pos_id_to_check,
        original_side,
        test_trading_pair,
        entry_ts,
        trigger_type.upper(),
        target_price,
        source_module=source_module)

    trigger_bar_data = None
    trigger_bar_ts = None
    # Since we don't have direct access to the data length, we'll use a reasonable
    # number of bars to check (e.g., 100 bars) and rely on None returns to stop
    max_bars_to_check = 100  # A reasonable number to check

    # Use a date-based approach rather than index-based approach
    current_time = entry_ts
    for _ in range(max_bars_to_check):
        # Get the next bar after the current time
        potential_bar = sim_exec.data_service.get_next_bar(test_trading_pair, current_time)
        if potential_bar is None:
            break  # No more data available

        # Update current_time for the next iteration
        current_time = potential_bar.name if hasattr(potential_bar, "name") else current_time
        if (
            potential_bar is None
            or not hasattr(potential_bar, "name")
            or potential_bar.name <= entry_ts
        ):
            continue

        if not all(k in potential_bar for k in ["low", "high"]):
            sim_exec.logger.warning(
                "Bar at %s missing low/high data for %s test.",
                potential_bar.name,
                trigger_type.upper(),
                source_module=source_module)
            continue
        try:
            bar_low = Decimal(str(potential_bar["low"]))
            bar_high = Decimal(str(potential_bar["high"]))
        except Exception:  # Removed 'as e'
            sim_exec.logger.exception(
                "Error converting bar low/high to Decimal. Bar: %s",  # Simplified log message
                potential_bar,
                source_module=source_module)
            continue

        if _check_bar_for_trigger(original_side, trigger_type, target_price, bar_low, bar_high):
            trigger_bar_data = potential_bar
            trigger_bar_ts = trigger_bar_data.name
            break

    if trigger_bar_data is not None and trigger_bar_ts is not None:
        sim_exec.logger.info(
            ("--- Checking %s for %s with bar at %s (Low: %s, High: %s, Target Price: %s) ---"),
            trigger_type.upper(),
            pos_id_to_check,
            trigger_bar_ts,
            trigger_bar_data["low"],  # Assuming these keys exist
            trigger_bar_data["high"],  # Assuming these keys exist
            target_price,
            source_module=source_module)
        await sim_exec.check_active_sl_tp(trigger_bar_data, trigger_bar_ts)
    else:
        log_message_skip = (
            "Could not find a suitable %s trigger bar for %s after %s for target price %s. "
            "Skipping this %s trigger test part."
        )
        sim_exec.logger.info(
            log_message_skip,
            trigger_type.upper(),
            pos_id_to_check,
            entry_ts,
            target_price,
            trigger_type.upper(),
            source_module=source_module)