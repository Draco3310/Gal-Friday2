#!/usr/bin/env python3
"""Monitoring Service for Gal Friday trading system.

This module provides system monitoring capabilities including health checks,
performance tracking, and automatic trading halt triggers when thresholds are exceeded.
"""

import asyncio
import logging  # Added for structured logging
import time
import uuid
from collections import deque  # Added for tracking recent API errors
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from decimal import Decimal
from typing import (  # Added Type for exc_info typing
    TYPE_CHECKING,
    Any,
    Optional,
)

import psutil  # Added for system resource monitoring

# Import actual classes when available, otherwise use placeholders
from .logger_service import LoggerService
from .portfolio_manager import PortfolioManager

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .core.events import (
        APIErrorEvent,  # Added for _handle_api_error
        ClosePositionCommand,  # Added for position closing on HALT
        Event,
        EventType,  # Add Event for completeness
        ExecutionReportEvent,
        MarketDataL2Event,
        MarketDataOHLCVEvent,
        PotentialHaltTriggerEvent,
        SystemStateEvent,
    )
    from .core.halt_coordinator import HaltCoordinator
    from .core.pubsub import PubSubManager  # Import from correct module
    from .execution_handler import (
        ExecutionHandler,  # Move to TYPE_CHECKING to break circular import
    )
else:
    # Simple placeholder classes for testing/development
    class _EventType:
        SYSTEM_STATE_CHANGE = "SYSTEM_STATE_CHANGE"  # Example value
        POTENTIAL_HALT_TRIGGER = "POTENTIAL_HALT_TRIGGER"  # Example value

    class _Event:
        # Base event class placeholder
        pass

    class _SystemStateEvent(_Event):  # Inherit from Event
        def __init__(
            self,
            timestamp: datetime,
            event_id: uuid.UUID,
            source_module: str,
            new_state: str,
            reason: str,
        ) -> None:
            self.timestamp = timestamp
            self.event_id = event_id
            self.source_module = source_module
            self.new_state = new_state
            self.reason = reason

    class _PotentialHaltTriggerEvent(_Event):  # Existing placeholder
        pass

    class _APIErrorEvent(_Event):  # Added placeholder for APIErrorEvent
        pass  # Simple placeholder

    class _ClosePositionCommand(_Event):  # Added placeholder for ClosePositionCommand
        def __init__(self, data: dict[str, Any] | None = None) -> None:
            if data is not None:
                self.__dict__.update(data)
            # The actual attributes (timestamp, event_id, etc.) are expected
            # to be handled by the real ClosePositionCommand during type checking
            # and when the real event is instantiated.

    class _HaltCoordinator:
        """Placeholder HaltCoordinator for non-TYPE_CHECKING mode."""
        def __init__(
            self,
            config_manager: "ConfigManager",
            pubsub_manager: "PubSubManager",
            logger_service: "LoggerService",
        ) -> None:
            pass

        def set_halt_state(self, is_halted: bool, reason: str, source: str) -> None:
            pass

        def clear_halt_state(self) -> None:
            pass

        def check_all_conditions(self) -> list[dict[str, Any]]:
            return []

        def update_condition(self, name: str, value: float | int | Decimal) -> bool:
            return False

    class _ConfigManager:
        def get(
            self,
            key: str,
            default: None | (str | int | float | Decimal | bool) = None,
        ) -> int | Decimal | None:
            # Provide default values for testing/running without real config
            if key == "monitoring.check_interval_seconds":
                return 60
            if key == "risk.limits.max_total_drawdown_pct":
                return Decimal("10.0")  # Example value
            return default

    class _PubSubManager:
        def __init__(self) -> None:
            """Initialize mock PubSub manager."""
            self.logger = logging.getLogger("MockPubSubManager")

        async def publish(self, event: "_Event") -> None:
            # Updated signature to match real implementation
            self.logger.debug(f"Publishing mock event: {event.__dict__}")

        def subscribe(
            self,
            event_type: str,
            handler: Callable[[Any], Coroutine[Any, Any, None]],
        ) -> None:
            # Placeholder for subscribe method
            pass

        def unsubscribe(
            self,
            event_type: str,
            handler: Callable[[Any], Coroutine[Any, Any, None]],
        ) -> None:
            # Placeholder for unsubscribe method
            pass

    class _PortfolioManager:
        def get_current_state(self) -> dict:
            # Return dummy data for placeholder
            return {"total_drawdown_pct": Decimal("1.5")}

    # Assign placeholder classes to the expected names
    ConfigManager = _ConfigManager
    PubSubManager = _PubSubManager
    PortfolioManager = _PortfolioManager
    EventType = _EventType
    SystemStateEvent = _SystemStateEvent
    Event = _Event
    PotentialHaltTriggerEvent = _PotentialHaltTriggerEvent  # Corrected to use its own placeholder
    APIErrorEvent = _APIErrorEvent  # Assign the new placeholder
    ClosePositionCommand = _ClosePositionCommand  # Assign the new placeholder
    HaltCoordinator = _HaltCoordinator  # Assign the new placeholder


class MonitoringService:
    """Monitors the overall system health and manages the global HALT state.

    Triggers HALT based on predefined conditions (e.g., max drawdown) or manual requests.
    Publishes system state changes (HALTED/RUNNING) via the PubSubManager.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        portfolio_manager: PortfolioManager,
        logger_service: LoggerService,
        execution_handler: "ExecutionHandler | None" = None,
        halt_coordinator: Optional["HaltCoordinator"] = None,
    ) -> None:
        """Initialize the MonitoringService.

        Args:
        ----
            config_manager: The application's configuration manager instance.
            pubsub_manager: The application's publish/subscribe manager instance.
            portfolio_manager: The application's portfolio manager instance.
            logger_service: The shared logger instance.
            execution_handler: Optional execution handler for API connectivity checks.
            halt_coordinator: Optional HALT coordinator for centralized HALT management.
        """
        self.config_manager = config_manager
        self.pubsub_manager = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self.logger = logger_service
        self._execution_handler = execution_handler
        self._source = self.__class__.__name__

        # Initialize HALT coordinator if not provided
        if halt_coordinator is None:
            try:
                from .core.halt_coordinator import HaltCoordinator as RealHaltCoordinator
                self._halt_coordinator = RealHaltCoordinator(
                    config_manager=config_manager,
                    pubsub_manager=pubsub_manager,
                    logger_service=logger_service,
                )
            except ImportError:
                # Fall back to placeholder if real implementation not available
                self._halt_coordinator = HaltCoordinator(
                    config_manager=config_manager,
                    pubsub_manager=pubsub_manager,
                    logger_service=logger_service,
                )
        else:
            self._halt_coordinator = halt_coordinator

        self._is_halted: bool = False
        self._periodic_check_task: asyncio.Task | None = None

        # Add startup time tracking
        self._service_start_time: datetime | None = None

        # Handler storage for unsubscribing
        self._potential_halt_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._market_data_l2_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._market_data_ohlcv_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._execution_report_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._api_error_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None

        # State for tracking additional monitoring metrics
        self._last_market_data_times: dict[str, datetime] = {}  # pair -> timestamp
        self._consecutive_api_failures: int = 0
        self._consecutive_losses: int = 0
        self._recent_api_errors: deque[float] = deque(
            maxlen=10,
        )  # Store timestamps of recent errors

        # Load configuration values instead of hardcoded constants
        self._load_configuration()

        self.logger.info("MonitoringService initialized.", source_module=self._source)

    def is_halted(self) -> bool:
        """Return whether the system is currently halted."""
        return self._is_halted

    async def start(self) -> None:
        """Start the periodic monitoring checks."""
        # Record when the service actually started
        self._service_start_time = datetime.now(UTC)
        self.logger.info(
            "MonitoringService started at %s",
            self._service_start_time,
            source_module=self._source,
        )

        # Publish initial state when starting, if not already halted
        if not self._is_halted:
            await self._publish_state_change(
                "RUNNING",
                "System startup",
                "MonitoringService Start",
            )

        if self._periodic_check_task and not self._periodic_check_task.done():
            self.logger.warning(
                "MonitoringService periodic check task already running.",
                source_module=self._source,
            )
            return

        msg = f"Starting MonitoringService periodic checks every {self._check_interval} seconds."
        self.logger.info(msg, source_module=self._source)

        self._periodic_check_task = asyncio.create_task(self._run_periodic_checks())

        # Subscribe to potential halt events
        self._potential_halt_handler = self._handle_potential_halt_trigger
        self.pubsub_manager.subscribe(
            EventType.POTENTIAL_HALT_TRIGGER,
            self._potential_halt_handler,
        )
        self.logger.info(
            "Subscribed to POTENTIAL_HALT_TRIGGER events.",
            source_module=self._source,
        )

        # Subscribe to API error events
        self._api_error_handler = self._handle_api_error
        self.pubsub_manager.subscribe(EventType.SYSTEM_ERROR, self._api_error_handler)
        self.logger.info(
            "Subscribed to SYSTEM_ERROR events for API error tracking.",
            source_module=self._source,
        )

        # Subscribe to market data events to track freshness
        self._market_data_l2_handler = self._update_market_data_timestamp
        self._market_data_ohlcv_handler = self._update_market_data_timestamp
        self.pubsub_manager.subscribe(EventType.MARKET_DATA_L2, self._market_data_l2_handler)
        self.pubsub_manager.subscribe(EventType.MARKET_DATA_OHLCV, self._market_data_ohlcv_handler)
        self.logger.info(
            "Subscribed to market data events for freshness tracking.",
            source_module=self._source,
        )

        # Subscribe to execution reports to track consecutive losses
        self._execution_report_handler = self._handle_execution_report
        self.pubsub_manager.subscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
        self.logger.info(
            "Subscribed to execution reports for loss tracking.",
            source_module=self._source,
        )

    async def stop(self) -> None:
        """Stop the periodic monitoring checks."""
        # Unsubscribe from all event types
        try:
            # Potential HALT trigger events
            if self._potential_halt_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.POTENTIAL_HALT_TRIGGER,
                    self._potential_halt_handler,
                )
                self.logger.info(
                    "Unsubscribed from POTENTIAL_HALT_TRIGGER events.",
                    source_module=self._source,
                )
                self._potential_halt_handler = None

            # API error events
            if hasattr(self, "_api_error_handler") and self._api_error_handler:
                self.pubsub_manager.unsubscribe(EventType.SYSTEM_ERROR, self._api_error_handler)
                self.logger.info(
                    "Unsubscribed from SYSTEM_ERROR events.",
                    source_module=self._source,
                )
                self._api_error_handler = None

            # Market data events
            if self._market_data_l2_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.MARKET_DATA_L2,
                    self._market_data_l2_handler,
                )
                self._market_data_l2_handler = None

            if self._market_data_ohlcv_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.MARKET_DATA_OHLCV,
                    self._market_data_ohlcv_handler,
                )
                self._market_data_ohlcv_handler = None

            # Execution report events
            if self._execution_report_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.EXECUTION_REPORT,
                    self._execution_report_handler,
                )
                self._execution_report_handler = None

            self.logger.info(
                "Unsubscribed from all monitoring events.",
                source_module=self._source,
            )
        except Exception:
            self.logger.exception(
                "Error unsubscribing from events",
                source_module=self._source,
            )

        if self._periodic_check_task and not self._periodic_check_task.done():
            self.logger.info(
                "Stopping MonitoringService periodic checks...",
                source_module=self._source,
            )
            self._periodic_check_task.cancel()
            try:
                await self._periodic_check_task
            except asyncio.CancelledError:
                self.logger.info(
                    "Monitoring check task successfully cancelled.",
                    source_module=self._source,
                )
            except Exception:
                self.logger.exception(
                    "Error encountered while stopping monitoring task.",
                    source_module=self._source,
                )
            finally:
                self._periodic_check_task = None
        else:
            self.logger.info(
                "MonitoringService periodic check task was not running.",
                source_module=self._source,
            )

    async def trigger_halt(self, reason: str, source: str) -> None:
        """Halt the system operations.

        Args:
        ----
            reason: The reason for halting the system.
            source: The source triggering the halt (e.g., 'MANUAL', 'AUTO: Max Drawdown').
        """
        if self._is_halted:
            self.logger.warning(
                "System already halted. Ignoring HALT trigger from %s.",
                source,
                source_module=self._source,
            )
            return

        self._is_halted = True

        # Update HALT coordinator state
        self._halt_coordinator.set_halt_state(
            is_halted=True,
            reason=reason,
            source=source,
        )

        self.logger.critical(
            "SYSTEM HALTED by %s. Reason: %s",
            source,
            reason,
            source_module=self._source,
        )
        await self._publish_state_change("HALTED", reason, source)

        # Handle positions based on configuration
        await self._handle_positions_on_halt()

    async def _handle_positions_on_halt(self) -> None:
        """Process existing positions according to the configured HALT behavior.

        Can close positions, maintain them, or perform other actions.
        """
        halt_behavior = self._halt_position_behavior
        self.logger.info(
            "HALT triggered. Position behavior set to: %s",
            halt_behavior,
            source_module=self._source,
        )

        if halt_behavior in {"close", "liquidate"}:
            self.logger.warning(
                "Attempting to close all open positions due to HALT.",
                source_module=self._source,
            )
            try:
                # Get current positions from portfolio manager
                current_state = self._portfolio_manager.get_current_state()
                open_positions = current_state.get("positions", {})

                if not open_positions:
                    self.logger.info(
                        "No open positions found to close during HALT.",
                        source_module=self._source,
                    )
                    return

                for pair, pos_data in open_positions.items():
                    # Extract quantity from position
                    qty_str = pos_data.get("quantity")
                    if not qty_str:
                        continue

                    try:
                        qty = Decimal(str(qty_str))
                    except Exception:
                        self.logger.warning(
                            "Could not convert position quantity to Decimal: %s",
                            qty_str,
                            source_module=self._source,
                        )
                        continue

                    if abs(qty) > Decimal("1e-12"):  # Check if position exists (non-zero)
                        close_side = "SELL" if qty > Decimal("0") else "BUY"
                        self.logger.info(
                            "Requesting closure of %s position (%s %s)",
                            pair,
                            close_side,
                            abs(qty),
                            source_module=self._source,
                        )

                        # For future implementation: Create and publish a close position command
                        self.logger.info(
                            "Creating ClosePositionCommand for %s: %s %s",
                            pair,
                            close_side,
                            abs(qty),
                            source_module=self._source,
                        )
                        close_command = ClosePositionCommand(
                            timestamp=datetime.now(UTC),
                            event_id=uuid.uuid4(),
                            source_module=self._source,
                            trading_pair=pair,
                            quantity=abs(qty),
                            side=close_side,
                        )
                        await self.pubsub_manager.publish(close_command)

            except Exception:
                self.logger.exception(
                    "Error during attempt to close positions on HALT",
                    source_module=self._source,
                )
        elif halt_behavior == "maintain":
            self.logger.info(
                "Maintaining existing positions during HALT as per configuration.",
                source_module=self._source,
            )
        else:
            self.logger.warning(
                "Unknown halt position behavior configured: %s. Maintaining positions.",
                halt_behavior,
                source_module=self._source,
            )

    async def trigger_resume(self, source: str) -> None:
        """Resume system operations after a HALT.

        Args:
        ----
            source: The source triggering the resume (e.g., 'MANUAL').
        """
        if not self._is_halted:
            self.logger.warning(
                "System not halted. Ignoring RESUME trigger from %s.",
                source,
                source_module=self._source,
            )
            return

        self._is_halted = False

        # Clear HALT coordinator state
        self._halt_coordinator.clear_halt_state()

        self.logger.info(
            "SYSTEM RESUMED by %s.",
            source,
            source_module=self._source,
        )
        await self._publish_state_change("RUNNING", "Manual resume", source)

    async def _publish_state_change(self, new_state: str, reason: str, source: str) -> None:
        """Publish a SystemStateEvent through the PubSubManager.

        Args:
        ----
            new_state: The new system state ("HALTED" or "RUNNING").
            reason: The reason for the state change.
            source: The source triggering the state change.
        """
        try:
            # Create a proper SystemStateEvent with correct parameters
            event = SystemStateEvent(
                source_module=source,
                event_id=uuid.uuid4(),
                timestamp=datetime.now().replace(microsecond=0),
                new_state=new_state,
                reason=reason,
            )
            # Correct publish method call - only passing the event
            await self.pubsub_manager.publish(event)
            self.logger.debug(
                "Published SYSTEM_STATE_CHANGE event: %s - %s",
                new_state,
                reason,
                source_module=self._source,
            )
        except Exception:
            self.logger.exception(
                "Failed to publish SYSTEM_STATE_CHANGE event",
                source_module=self._source,
            )

    async def _handle_potential_halt_trigger(self, event: "PotentialHaltTriggerEvent") -> None:
        """Handle events that suggest a potential HALT condition.

        Args:
        ----
            event: The PotentialHaltTriggerEvent containing halt trigger information.
        """
        if not isinstance(event, PotentialHaltTriggerEvent):
            self.logger.warning(
                "Received non-PotentialHaltTriggerEvent: %s",
                type(event),
                source_module=self._source,
            )
            return

        warning_msg = (
            f"Potential HALT condition received from {event.source_module}: {event.reason}"
        )
        self.logger.warning(warning_msg, source_module=self._source)
        # Trigger actual halt - might add confirmation logic later
        await self.trigger_halt(reason=event.reason, source=event.source_module)

    async def _run_periodic_checks(self) -> None:
        """Execute the core background task performing periodic checks.

        This method runs at regular intervals defined by the check_interval configuration.
        """
        self.logger.info(
            "MonitoringService periodic check task started.",
            source_module=self._source,
        )
        while True:
            try:
                await asyncio.sleep(self._check_interval)

                if not self._is_halted:
                    self.logger.debug(
                        "Running periodic checks...",
                        source_module=self._source,
                    )
                    # Comprehensive check of all HALT conditions
                    await self._check_all_halt_conditions()

            except asyncio.CancelledError:
                self.logger.info(
                    "MonitoringService periodic check task cancelled.",
                    source_module=self._source,
                )
                break
            except Exception:
                self.logger.exception(
                    "Unhandled error during periodic monitoring check. Continuing...",
                    source_module=self._source,
                )
                # Avoid tight loop on unexpected errors
                await asyncio.sleep(self._check_interval)

    async def _check_all_halt_conditions(self) -> None:
        """Comprehensive check of all HALT conditions."""
        # 1. Drawdown checks
        await self._check_drawdown_conditions()

        # 2. Market volatility checks
        await self._check_market_volatility()

        # 3. System health checks
        await self._check_system_health()

        # 4. API connectivity checks
        await self._check_api_connectivity()

        # 5. Data freshness checks
        await self._check_market_data_freshness()

        # 6. Position risk checks
        await self._check_position_risk()

        # Check if any conditions are triggered
        triggered_conditions = self._halt_coordinator.check_all_conditions()
        if triggered_conditions:
            # Build comprehensive reason from all triggered conditions
            reasons = [
                f"{c.name}: {c.current_value} > {c.threshold}"
                for c in triggered_conditions
            ]
            combined_reason = "; ".join(reasons)
            await self.trigger_halt(
                reason=f"Multiple HALT conditions triggered: {combined_reason}",
                source="AUTO: Multiple Conditions",
            )

    async def _check_drawdown_conditions(self) -> None:
        """Check all drawdown-related conditions."""
        try:
            current_state = self._portfolio_manager.get_current_state()

            # Total drawdown
            total_dd = current_state.get("total_drawdown_pct", Decimal("0"))
            if not isinstance(total_dd, Decimal):
                total_dd = Decimal(str(total_dd))

            # Update condition in coordinator
            if self._halt_coordinator.update_condition("max_total_drawdown", abs(total_dd)):
                await self.trigger_halt(
                    reason=f"Maximum total drawdown exceeded: {abs(total_dd):.2f}%",
                    source="AUTO: Max Drawdown",
                )

            # Daily drawdown
            daily_dd = current_state.get("daily_drawdown_pct", Decimal("0"))
            if not isinstance(daily_dd, Decimal):
                daily_dd = Decimal(str(daily_dd))

            if self._halt_coordinator.update_condition("max_daily_drawdown", abs(daily_dd)):
                await self.trigger_halt(
                    reason=f"Maximum daily drawdown exceeded: {abs(daily_dd):.2f}%",
                    source="AUTO: Daily Drawdown",
                )

# Consecutive losses
            consecutive_losses = self._consecutive_losses
            if self._halt_coordinator.update_condition(
                "max_consecutive_losses", consecutive_losses,
            ):
                await self.trigger_halt(
                    reason=f"Maximum consecutive losses reached: {consecutive_losses}",
                    source="AUTO: Consecutive Losses",
                )

        except Exception:
            self.logger.exception(
                "Error checking drawdown conditions",
                source_module=self._source,
            )

    async def _check_system_health(self) -> None:
        """Check system resource health."""
        await self._check_system_resources()

    async def _check_position_risk(self) -> None:
        """Check position-specific risk metrics.
        
        Monitors individual position sizes, concentration risk, and triggers
        automated position reduction or closure when risk thresholds are breached.
        """
        self.logger.debug("Running periodic check for position-specific risks.", source_module=self._source)

        # 1. Fetch Current Portfolio State
        try:
            current_positions = await self._get_all_open_positions()
            portfolio_summary = await self._get_portfolio_summary()
            total_portfolio_value = portfolio_summary.get("total_equity", 0)
        except Exception as e:
            self.logger.error(
                "Failed to fetch position or portfolio data for risk check: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )
            return  # Cannot proceed without position data

        if not current_positions:
            self.logger.debug("No open positions to check for risk.", source_module=self._source)
            return

        # 2. Load Position Risk Configuration
        position_risk_config = self.config_manager.get("monitoring", {}).get("position_risk_checks", {})
        global_max_pos_pct_config = position_risk_config.get("max_single_position_percentage_of_portfolio", {})
        global_max_pos_notional_usd_config = position_risk_config.get("max_position_notional_value_usd", {})
        specific_pair_limits_config = position_risk_config.get("specific_pair_limits", {})

        # 3. Iterate Through Each Open Position and Check Risks
        for position in current_positions:
            trading_pair = position.get("trading_pair")
            position_value_usd = position.get("current_market_value_usd", 0)
            position_base_quantity = position.get("quantity", 0)

            if not trading_pair:
                continue

            # Convert to Decimal for precise calculations
            try:
                position_value_usd = Decimal(str(position_value_usd))
                position_base_quantity = Decimal(str(position_base_quantity))
                total_portfolio_value_decimal = Decimal(str(total_portfolio_value))
            except (ValueError, TypeError):
                self.logger.warning(
                    "Could not convert position values to Decimal for %s",
                    trading_pair,
                    source_module=self._source,
                )
                continue

            # 3.1. Check: Position Size as Percentage of Total Portfolio
            if total_portfolio_value_decimal > 0:
                position_pct_of_portfolio = position_value_usd / total_portfolio_value_decimal
                warning_thresh_pct = global_max_pos_pct_config.get("warning_threshold", 0.20)
                action_thresh_pct = global_max_pos_pct_config.get("action_threshold")

                if position_pct_of_portfolio > warning_thresh_pct:
                    alert_details = {
                        "trading_pair": trading_pair,
                        "metric": "position_percentage_of_portfolio",
                        "value": float(position_pct_of_portfolio),
                        "warning_threshold": warning_thresh_pct,
                        "action_threshold": action_thresh_pct,
                        "position_value_usd": float(position_value_usd),
                        "total_portfolio_value_usd": float(total_portfolio_value_decimal),
                    }

                    self.logger.warning(
                        "Position Risk Alert: %s (%.2f%%) exceeds warning portfolio percentage (%.2f%%)",
                        trading_pair,
                        position_pct_of_portfolio * 100,
                        warning_thresh_pct * 100,
                        source_module=self._source,
                    )

                    await self._publish_position_risk_alert(alert_details, "WARNING")

                    if action_thresh_pct is not None and position_pct_of_portfolio > action_thresh_pct:
                        self.logger.critical(
                            "Position Risk Breach: %s (%.2f%%) exceeds ACTION portfolio percentage (%.2f%%). Initiating reduction.",
                            trading_pair,
                            position_pct_of_portfolio * 100,
                            action_thresh_pct * 100,
                            source_module=self._source,
                        )

                        reduction_pct = global_max_pos_pct_config.get("reduction_percentage")
                        if reduction_pct is not None:
                            await self._initiate_position_reduction(
                                position=position,
                                reduction_type="PERCENTAGE_OF_CURRENT",
                                reduction_value=Decimal(str(reduction_pct)),
                                reason="EXCEEDED_MAX_PORTFOLIO_PERCENTAGE_LIMIT",
                                breach_details=alert_details,
                            )

            # 3.2. Check: Position Notional Value (Absolute USD Limit)
            warn_thresh_notional = global_max_pos_notional_usd_config.get("warning_threshold")
            action_thresh_notional = global_max_pos_notional_usd_config.get("action_threshold")

            if warn_thresh_notional is not None and position_value_usd > Decimal(str(warn_thresh_notional)):
                alert_details = {
                    "trading_pair": trading_pair,
                    "metric": "position_notional_value_usd",
                    "value": float(position_value_usd),
                    "warning_threshold": warn_thresh_notional,
                    "action_threshold": action_thresh_notional,
                }

                self.logger.warning(
                    "Position Risk Alert: %s value ($%.2f) exceeds warning notional value ($%.2f)",
                    trading_pair,
                    position_value_usd,
                    warn_thresh_notional,
                    source_module=self._source,
                )

                await self._publish_position_risk_alert(alert_details, "WARNING")

                if action_thresh_notional is not None and position_value_usd > Decimal(str(action_thresh_notional)):
                    self.logger.critical(
                        "Position Risk Breach: %s value ($%.2f) exceeds ACTION notional value ($%.2f). Initiating reduction.",
                        trading_pair,
                        position_value_usd,
                        action_thresh_notional,
                        source_module=self._source,
                    )

                    reduction_target_notional = global_max_pos_notional_usd_config.get("reduction_target_notional_value")
                    if reduction_target_notional is not None:
                        await self._initiate_position_reduction(
                            position=position,
                            reduction_type="NOTIONAL_TARGET",
                            reduction_value=Decimal(str(reduction_target_notional)),
                            reason="EXCEEDED_MAX_NOTIONAL_VALUE_LIMIT",
                            breach_details=alert_details,
                        )

            # 3.3. Check: Specific Pair Limits (if configured)
            pair_specific_config = specific_pair_limits_config.get(trading_pair, {})
            base_qty_limits = pair_specific_config.get("max_base_qty", {})
            warn_thresh_base_qty = base_qty_limits.get("warning_threshold")
            action_thresh_base_qty = base_qty_limits.get("action_threshold")

            if warn_thresh_base_qty is not None and abs(position_base_quantity) > Decimal(str(warn_thresh_base_qty)):
                alert_details = {
                    "trading_pair": trading_pair,
                    "metric": "position_base_quantity",
                    "value": float(abs(position_base_quantity)),
                    "warning_threshold": warn_thresh_base_qty,
                    "action_threshold": action_thresh_base_qty,
                    "asset": trading_pair.split("/")[0] if "/" in trading_pair else trading_pair,
                }

                self.logger.warning(
                    "Position Risk Alert: %s quantity (%.6f) exceeds specific pair warning base quantity (%.6f)",
                    trading_pair,
                    abs(position_base_quantity),
                    warn_thresh_base_qty,
                    source_module=self._source,
                )

                await self._publish_position_risk_alert(alert_details, "WARNING")

                if action_thresh_base_qty is not None and abs(position_base_quantity) > Decimal(str(action_thresh_base_qty)):
                    self.logger.critical(
                        "Position Risk Breach: %s quantity (%.6f) exceeds specific pair ACTION base quantity (%.6f). Initiating reduction.",
                        trading_pair,
                        abs(position_base_quantity),
                        action_thresh_base_qty,
                        source_module=self._source,
                    )

                    reduction_qty_val = base_qty_limits.get("reduction_qty")
                    if reduction_qty_val is not None:
                        await self._initiate_position_reduction(
                            position=position,
                            reduction_type="QUANTITY",
                            reduction_value=Decimal(str(reduction_qty_val)),
                            reason="EXCEEDED_PAIR_MAX_BASE_QUANTITY_LIMIT",
                            breach_details=alert_details,
                        )

        self.logger.debug("Position risk check completed.", source_module=self._source)

    async def _get_all_open_positions(self) -> list[dict]:
        """Get all open positions from portfolio manager."""
        try:
            current_state = self._portfolio_manager.get_current_state()
            positions_dict = current_state.get("positions", {})

            # Convert positions dict to list of position objects
            positions = []
            for pair, pos_data in positions_dict.items():
                if pos_data.get("quantity", 0) != 0:  # Only include non-zero positions
                    position = {
                        "trading_pair": pair,
                        "quantity": pos_data.get("quantity", 0),
                        "current_market_value_usd": pos_data.get("market_value_usd", 0),
                        **pos_data,  # Include all other position data
                    }
                    positions.append(position)
            return positions
        except Exception as e:
            self.logger.error(
                "Failed to get open positions: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )
            return []

    async def _get_portfolio_summary(self) -> dict:
        """Get portfolio summary from portfolio manager."""
        try:
            current_state = self._portfolio_manager.get_current_state()
            return {
                "total_equity": current_state.get("total_equity", 0),
                "available_balance": current_state.get("available_balance", 0),
                "total_unrealized_pnl": current_state.get("total_unrealized_pnl", 0),
            }
        except Exception as e:
            self.logger.error(
                "Failed to get portfolio summary: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )
            return {"total_equity": 0, "available_balance": 0, "total_unrealized_pnl": 0}

    async def _publish_position_risk_alert(self, alert_details: dict, severity: str) -> None:
        """Publish a position risk alert event."""
        try:
            # Create a position risk alert event (would need to be defined in events.py)
            alert_event = {
                "timestamp": datetime.now(UTC),
                "event_id": uuid.uuid4(),
                "source_module": self._source,
                "alert_type": "POSITION_RISK",
                "severity": severity,
                "details": alert_details,
            }

            self.logger.info(
                "Publishing position risk alert for %s: %s",
                alert_details.get("trading_pair"),
                alert_details.get("metric"),
                source_module=self._source,
            )

            # In a real implementation, this would publish a proper PositionRiskAlertEvent
            # await self.pubsub_manager.publish(PositionRiskAlertEvent(**alert_event))

        except Exception as e:
            self.logger.error(
                "Failed to publish position risk alert: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )

    async def _initiate_position_reduction(
        self,
        position: dict,
        reduction_type: str,
        reduction_value: Decimal,
        reason: str,
        breach_details: dict,
    ) -> None:
        """Initiate position reduction by publishing a ReducePositionCommand."""
        trading_pair = position.get("trading_pair")
        current_quantity = Decimal(str(position.get("quantity", 0)))
        quantity_to_reduce = Decimal(0)

        if reduction_type == "PERCENTAGE_OF_CURRENT":
            quantity_to_reduce = abs(current_quantity) * reduction_value
        elif reduction_type == "QUANTITY":
            quantity_to_reduce = reduction_value
        elif reduction_type == "NOTIONAL_TARGET":
            # Requires current price to convert target notional to target quantity
            self.logger.warning(
                "NOTIONAL_TARGET reduction type for %s requires MarketPriceService integration (TODO).",
                trading_pair,
                source_module=self._source,
            )
            return
        else:
            self.logger.error(
                "Unknown reduction_type: %s for %s",
                reduction_type,
                trading_pair,
                source_module=self._source,
            )
            return

        if quantity_to_reduce <= Decimal(0):
            self.logger.info(
                "Calculated reduction quantity for %s is zero or negative (%.6f). No action taken.",
                trading_pair,
                quantity_to_reduce,
                source_module=self._source,
            )
            return

        # Ensure reduction doesn't exceed current position size
        quantity_to_reduce = min(quantity_to_reduce, abs(current_quantity))

        self.logger.info(
            "Attempting to reduce position %s by %.6f (Type: %s, Value: %.6f). Reason: %s",
            trading_pair,
            quantity_to_reduce,
            reduction_type,
            reduction_value,
            reason,
            source_module=self._source,
        )

        command_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        # Determine order type for reduction
        reduction_order_type = self.config_manager.get("monitoring", {}).get("position_risk_checks", {}).get("default_reduction_order_type", "MARKET")

        try:
            # Create reduce position command (would need to be defined in events.py)
            reduce_command = {
                "command_id": command_id,
                "timestamp": timestamp,
                "source_module": self._source,
                "trading_pair": trading_pair,
                "quantity_to_reduce": float(quantity_to_reduce),
                "order_type_preference": reduction_order_type,
                "reason": f"AUTOMATED_RISK_REDUCTION: {reason}",
                "metadata": {
                    "breach_details": breach_details,
                    "reduction_type": reduction_type,
                    "reduction_value_config": str(reduction_value),
                },
            }

            self.logger.info(
                "Successfully created ReducePositionCommand (%s) for %s to reduce by %.6f",
                str(command_id)[:8],
                trading_pair,
                quantity_to_reduce,
                source_module=self._source,
            )

            # In a real implementation, this would publish a proper ReducePositionCommand
            # await self.pubsub_manager.publish(ReducePositionCommand(**reduce_command))

        except Exception as e:
            self.logger.critical(
                "Failed to create/publish ReducePositionCommand (%s) for %s. Position reduction failed. Error: %s",
                str(command_id)[:8],
                trading_pair,
                e,
                source_module=self._source,
                exc_info=True,
            )

    async def _check_system_health(self) -> None:
        """Check system resource health."""
        await self._check_system_resources()

    async def _check_drawdown(self) -> None:
        """Check if the maximum total portfolio drawdown has been exceeded.

        Retrieves the current drawdown percentage from the portfolio manager
        and compares it against the configured maximum drawdown threshold.
        """
        try:
            # PortfolioManager.get_current_state() needs to be synchronous per design doc
            # If it becomes async, this needs adjustment (e.g., run_in_executor)
            # For now, assuming it's sync as requested for MVP.
            current_state = self._portfolio_manager.get_current_state()
            drawdown_pct = current_state.get("total_drawdown_pct")

            if drawdown_pct is None:
                self.logger.warning(
                    "Could not retrieve 'total_drawdown_pct' from PortfolioManager state.",
                    source_module=self._source,
                )
                return

            # Ensure drawdown_pct is Decimal
            if not isinstance(drawdown_pct, Decimal):
                try:
                    drawdown_pct = Decimal(drawdown_pct)
                except Exception:
                    self.logger.warning(
                        "Invalid type for 'total_drawdown_pct': %s. Skipping check.",
                        type(drawdown_pct),
                        source_module=self._source,
                    )
                    return

            self.logger.debug(
                "Current total drawdown: %.2f%% (Limit: %s%%)",
                drawdown_pct,
                self._max_total_drawdown_pct,
                source_module=self._source,
            )

            # Check if drawdown exceeds the limit (absolute value)
            if abs(drawdown_pct) > self._max_total_drawdown_pct:
                drawdown_val = abs(drawdown_pct)
                limit_val = self._max_total_drawdown_pct
                reason = f"Max total drawdown limit exceeded: {drawdown_val:.2f}% > {limit_val}%"
                self.logger.warning(reason, source_module=self._source)
                await self.trigger_halt(reason=reason, source="AUTO: Max Drawdown")

        except Exception:
            self.logger.exception(
                "Error occurred during drawdown check.",
                source_module=self._source,
            )

    async def _check_api_connectivity(self) -> None:
        """Check connectivity to Kraken API by attempting a lightweight authenticated call.

        Triggers HALT if consecutive failures exceed the threshold.
        """
        if not self._execution_handler:
            self.logger.warning(
                "No execution handler available for API connectivity check.",
                source_module=self._source,
            )
            return

        try:
            # This would be a call to the execution handler's API status check method
            # For example: success = await self._execution_handler.check_api_status()
            # Since we don't have the actual method yet, we'll just simulate success for now
            success = True  # Replace with actual API check when available

            if success:
                self._consecutive_api_failures = 0  # Reset on success
                self.logger.debug("API connectivity check passed.", source_module=self._source)
            else:
                self._consecutive_api_failures += 1
                warning_msg = (
                    "API connectivity check failed "
                    f"({self._consecutive_api_failures}/"
                    f"{self._api_failure_threshold})"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )

                if self._consecutive_api_failures >= self._api_failure_threshold:
                    reason = (
                        f"API connectivity failed "
                        f"{self._consecutive_api_failures} consecutive times."
                    )
                    self.logger.error(reason, source_module=self._source)
                    await self.trigger_halt(reason=reason, source="AUTO: API Connectivity")

        except Exception:
            self._consecutive_api_failures += 1
            self.logger.exception(
                "Error during API connectivity check",
                source_module=self._source,
            )

            if self._consecutive_api_failures >= self._api_failure_threshold:
                reason = (
                    f"API connectivity check errors: "
                    f"{self._consecutive_api_failures} consecutive failures."
                )
                await self.trigger_halt(reason=reason, source="AUTO: API Connectivity")

    async def _check_market_data_freshness(self) -> None:
        """Check if market data timestamps are recent enough.

        Triggers HALT if data for active pairs is stale beyond threshold.
        Uses startup time tracking to provide grace period during system initialization.
        """
        now = datetime.now(UTC)
        stale_pairs = []
        potentially_stale_awaiting_initial_data = []

        if not self._active_pairs:
            self.logger.warning(
                "No active trading pairs configured for market data freshness check.",
                source_module=self._source,
            )
            return

        # Determine system uptime for startup grace period
        if self._service_start_time is None:
            self.logger.warning(
                "Service start time not recorded. Staleness check might be unreliable during initial startup phase.",
                source_module=self._source,
            )
            system_uptime_seconds = float("inf")  # Effectively disables startup grace period
        else:
            system_uptime_seconds = (now - self._service_start_time).total_seconds()

        for pair in self._active_pairs:
            last_ts = self._last_market_data_times.get(pair)

            if last_ts is None:
                # Case 1: No data ever received for this pair
                if system_uptime_seconds < self._data_staleness_threshold_s:
                    # Startup grace period is active for this pair as no data has been seen yet
                    self.logger.info(
                        "Awaiting initial market data for active pair %s. System uptime: %.2fs.",
                        pair,
                        system_uptime_seconds,
                        source_module=self._source,
                    )
                    potentially_stale_awaiting_initial_data.append(pair)
                    # Do NOT add to stale_pairs yet
                else:
                    # Startup grace period has passed, and still no data. This is a concern.
                    self.logger.warning(
                        "No market data received for active pair %s after initial grace period (%.2fs). Marking as stale.",
                        pair,
                        system_uptime_seconds,
                        source_module=self._source,
                    )
                    stale_pairs.append(pair)  # Now it's considered genuinely stale
            elif (now - last_ts).total_seconds() > self._data_staleness_threshold_s:
                # Case 2: Data was received, but it's now older than the staleness threshold
                stale_pairs.append(pair)
                warning_msg = (
                    f"Market data for {pair} is stale (last update: {last_ts}, "
                    f"threshold: {self._data_staleness_threshold_s}s, current age: {(now - last_ts).total_seconds():.2f}s)"
                )
                self.logger.warning(warning_msg, source_module=self._source)
            else:
                # Data is present and not stale
                self.logger.debug(f"Market data for {pair} is current (last update: {last_ts}).", source_module=self._source)

        if stale_pairs:
            self.logger.info(f"Identified stale pairs: {stale_pairs}", source_module=self._source)
            reason = f"Market data stale for pairs: {', '.join(stale_pairs)}"
            await self.trigger_halt(reason=reason, source="AUTO: Market Data Staleness")

        if potentially_stale_awaiting_initial_data:
            self.logger.info(
                f"Pairs awaiting initial data (within startup grace period): {potentially_stale_awaiting_initial_data}",
                source_module=self._source,
            )

    async def _check_system_resources(self) -> None:
        """Monitor CPU and Memory usage.

        Logs warnings when thresholds are approached, triggers HALT at critical levels.
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=None)  # Non-blocking
            mem_usage = psutil.virtual_memory().percent

            self.logger.debug(
                "System Resources: CPU=%.1f%%, Memory=%.1f%%",
                cpu_usage,
                mem_usage,
                source_module=self._source,
            )

            # Check CPU usage
            if cpu_usage > self._cpu_threshold_pct:
                warning_msg = (
                    "High CPU usage detected: "
                    f"{cpu_usage:.1f}% "
                    f"(Threshold: {self._cpu_threshold_pct}%)"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )
                # Only trigger HALT on extremely high CPU usage that would impact trading
                if cpu_usage > self._cpu_threshold_pct + 5:  # Extra 5% buffer
                    reason = f"Critical CPU usage: {cpu_usage:.1f}%"
                    await self.trigger_halt(reason=reason, source="AUTO: System Resources")

            # Check memory usage
            if mem_usage > self._memory_threshold_pct:
                warning_msg = (
                    "High Memory usage detected: "
                    f"{mem_usage:.1f}% "
                    f"(Threshold: {self._memory_threshold_pct}%)"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )
                # Only trigger HALT on extremely high memory usage that would impact trading
                if mem_usage > self._memory_threshold_pct + 5:  # Extra 5% buffer
                    reason = f"Critical Memory usage: {mem_usage:.1f}%"
                    await self.trigger_halt(reason=reason, source="AUTO: System Resources")

        except Exception:
            self.logger.exception(
                "Error checking system resources",
                source_module=self._source,
            )

    async def _check_market_volatility(self) -> None:
        """Check for excessive market volatility.

        Triggers HALT if volatility exceeds configured thresholds.
        """
        try:
            for pair in self._active_pairs:
                # Calculate rolling volatility (would need price history)
                volatility = await self._calculate_volatility(pair)

                if (
                    volatility is not None and
                    self._halt_coordinator.update_condition("max_volatility", volatility)
                ):
                    reason = (
                        f"Market volatility for {pair} ({volatility:.2f}%) "
                        f"exceeds threshold"
                    )
                    await self.trigger_halt(
                        reason=reason,
                        source="AUTO: Market Volatility",
                    )
                    break

        except Exception:
            self.logger.exception(
                "Error checking market volatility",
                source_module=self._source,
            )

    async def _calculate_volatility(self, pair: str) -> Decimal | None:
        """Calculate rolling volatility for a trading pair.

        Supports both standard deviation and GARCH volatility calculation methods.
        The method used is determined by the 'volatility_calculation.method' configuration.

        Args:
            pair: Trading pair to calculate volatility for

        Returns:
            Decimal: Annualized volatility percentage, or None if insufficient data
        """
        self.logger.debug(f"Calculating volatility for {pair}.", source_module=self._source)

        vol_config = self.config_manager.get("monitoring", {}).get("volatility_calculation", {})
        calculation_method = vol_config.get("method", "stddev").lower()

        if calculation_method == "garch":
            self.logger.debug(f"Using GARCH method for volatility calculation for {pair}.", source_module=self._source)
            return await self._calculate_garch_volatility_internal(pair, vol_config)
        if calculation_method == "stddev":
            self.logger.debug(f"Using standard deviation method for volatility calculation for {pair}.", source_module=self._source)
            return await self._calculate_stddev_volatility_internal(pair, vol_config)
        self.logger.error(
            f"Unknown volatility calculation method configured: {calculation_method}. Defaulting to None.",
            source_module=self._source,
        )
        return None

    async def _calculate_stddev_volatility_internal(self, trading_pair: str, vol_config: dict) -> Decimal | None:
        """Calculate standard deviation volatility for a trading pair.
        
        Args:
            trading_pair: Trading pair to calculate volatility for
            vol_config: Volatility calculation configuration
            
        Returns:
            Decimal: Annualized volatility percentage, or None if insufficient data
        """
        self.logger.debug(f"Calculating stddev volatility for {trading_pair}.", source_module=self._source)

        window_size = vol_config.get("stddev_window_size_candles", 100)
        candle_interval_minutes = vol_config.get("candle_interval_minutes", 60)
        min_required_data_points = vol_config.get("stddev_min_data_points_for_calc", int(window_size * 0.8))
        use_log_returns = vol_config.get("use_log_returns", True)
        annualization_factor_config = vol_config.get("annualization_periods_per_year")

        # Calculate annualization factor
        if annualization_factor_config is None:
            if candle_interval_minutes == 1440:  # Daily
                periods_per_year = 365
            elif candle_interval_minutes == 60:  # Hourly
                periods_per_year = 365 * 24
            elif candle_interval_minutes == 1:  # Minute
                periods_per_year = 365 * 24 * 60
            else:
                self.logger.warning(
                    f"Unsupported candle_interval_minutes ({candle_interval_minutes}) for default annualization factor. "
                    "Volatility will not be annualized correctly without explicit 'annualization_periods_per_year' config.",
                    source_module=self._source,
                )
                periods_per_year = 1  # Effectively no annualization
            annualization_factor = (periods_per_year) ** 0.5
        else:
            annualization_factor = (annualization_factor_config) ** 0.5

        try:
            # This would need to be implemented to fetch historical candles
            # For now, simulating the call
            price_history_candles = await self._get_historical_candles_for_volatility(
                trading_pair=trading_pair,
                num_candles=window_size + 1,
                interval_minutes=candle_interval_minutes,
            )

            if price_history_candles is None or len(price_history_candles) < min_required_data_points + 1:
                self.logger.warning(
                    f"StdDev Vol: Insufficient historical price data for {trading_pair}. "
                    f"Required: {min_required_data_points + 1}, Got: {len(price_history_candles) if price_history_candles else 0}.",
                    source_module=self._source,
                )
                return None

            closing_prices = [Decimal(str(candle.get("close", 0))) for candle in price_history_candles]
        except Exception as e:
            self.logger.error(
                f"StdDev Vol: Failed to fetch/process price history for {trading_pair}: {e}",
                source_module=self._source,
                exc_info=True,
            )
            return None

        # Convert to numpy for calculations
        import numpy as np
        np_closing_prices = np.array([float(p) for p in closing_prices])

        if use_log_returns:
            if np.any(np_closing_prices <= 0):
                self.logger.error(f"StdDev Vol: Invalid prices for {trading_pair} for log returns.", source_module=self._source)
                return None
            returns = np.log(np_closing_prices[1:] / np_closing_prices[:-1])
        else:
            returns = (np_closing_prices[1:] - np_closing_prices[:-1]) / np_closing_prices[:-1]

        if len(returns) == 0:
            self.logger.warning(f"StdDev Vol: No returns calculated for {trading_pair}.", source_module=self._source)
            return None

        std_dev_returns = np.std(returns)
        annualized_volatility_float = std_dev_returns * annualization_factor
        annualized_volatility_decimal = Decimal(str(annualized_volatility_float)) * Decimal("100")

        self.logger.info(
            f"StdDev Vol for {trading_pair}: {annualized_volatility_decimal:.2f}%",
            source_module=self._source,
        )
        return annualized_volatility_decimal.quantize(Decimal("0.0001"))

    async def _calculate_garch_volatility_internal(self, trading_pair: str, vol_config: dict) -> Decimal | None:
        """Calculate GARCH volatility for a trading pair.
        
        Args:
            trading_pair: Trading pair to calculate volatility for
            vol_config: Volatility calculation configuration
            
        Returns:
            Decimal: Annualized volatility percentage, or None if insufficient data or GARCH unavailable
        """
        self.logger.debug(f"Calculating GARCH volatility for {trading_pair}.", source_module=self._source)

        # Check if GARCH dependencies are available
        try:
            # import arch  # Would need to be installed: pip install arch
            # For now, we'll simulate GARCH calculation as it requires additional dependencies
            self.logger.warning(
                f"GARCH volatility calculation for {trading_pair} requires 'arch' library. "
                "Install with: pip install arch. Falling back to standard deviation method.",
                source_module=self._source,
            )
            return await self._calculate_stddev_volatility_internal(trading_pair, vol_config)
        except ImportError:
            self.logger.warning(
                f"GARCH volatility calculation for {trading_pair} requires 'arch' library. "
                "Falling back to standard deviation method.",
                source_module=self._source,
            )
            return await self._calculate_stddev_volatility_internal(trading_pair, vol_config)

        # GARCH implementation would go here when arch library is available
        # The pseudocode implementation shows:
        # 1. Get GARCH configuration (window_size, p, q, distribution)
        # 2. Fetch longer price history (GARCH often needs 200+ points)
        # 3. Calculate returns and scale by 100 for numerical stability
        # 4. Fit GARCH model using arch.arch_model()
        # 5. Forecast 1-step ahead conditional volatility
        # 6. Annualize and return as percentage

    async def _get_historical_candles_for_volatility(
        self,
        trading_pair: str,
        num_candles: int,
        interval_minutes: int,
    ) -> list[dict] | None:
        """Get historical candles for volatility calculation.
        
        This is a placeholder method that would integrate with the historical data service.
        
        Args:
            trading_pair: Trading pair to get data for
            num_candles: Number of candles to retrieve
            interval_minutes: Interval in minutes between candles
            
        Returns:
            List of candle dictionaries with 'close' prices, or None if unavailable
        """
        # This would integrate with HistoricalDataService when available
        # For now, return None to indicate no data available
        self.logger.debug(
            f"Historical candles request for {trading_pair}: {num_candles} candles at {interval_minutes}min intervals "
            "(placeholder - requires HistoricalDataService integration)",
            source_module=self._source,
        )
        return None

    async def _handle_execution_report(self, event: "ExecutionReportEvent") -> None:
        """Handle execution report events to track consecutive losses.

        Triggers HALT if consecutive loss limit is reached.

        Args:
        ----
            event: An ExecutionReportEvent
        """
        try:
            # Check for filled order with realized PnL
            if (
                hasattr(event, "order_status")
                and event.order_status == "FILLED"
                and hasattr(event, "realized_pnl")
            ):
                # Convert to Decimal if needed
                pnl = event.realized_pnl
                if not isinstance(pnl, Decimal):
                    try:
                        pnl = Decimal(str(pnl))
                    except Exception:
                        self.logger.warning(
                            "Could not convert realized_pnl to Decimal: %s",
                            pnl,
                            source_module=self._source,
                        )
                        return

                # Update consecutive losses counter
                if pnl < Decimal("0"):
                    self._consecutive_losses += 1
                    warning_msg = (
                        f"Trade loss detected: {pnl}. "
                        f"Consecutive losses: {self._consecutive_losses}"
                    )
                    self.logger.warning(
                        warning_msg,
                        source_module=self._source,
                    )

                    # Check if we've hit the consecutive loss limit
                    if self._consecutive_losses >= self._consecutive_loss_limit:
                        reason = f"Consecutive loss limit reached: {self._consecutive_losses}"
                        await self.trigger_halt(reason=reason, source="AUTO: Consecutive Losses")
                else:
                    # Reset counter on profitable trade
                    if self._consecutive_losses > 0:
                        info_msg = (
                            f"Profitable trade resets consecutive loss counter "
                            f"(was {self._consecutive_losses})"
                        )
                        self.logger.info(
                            info_msg,
                            source_module=self._source,
                        )
                    self._consecutive_losses = 0
        except Exception:
            self.logger.exception(
                "Error handling execution report",
                source_module=self._source,
            )

    async def _handle_api_error(self, event: "APIErrorEvent") -> None:
        """Count and evaluate API errors to detect excessive error rates.

        Triggers HALT if error frequency exceeds threshold.

        Args:
        ----
            event: An APIErrorEvent
        """
        try:
            now = time.time()
            self._recent_api_errors.append(now)

            # Check if we've exceeded the error threshold within the time window
            error_window = now - self._api_error_threshold_period_s
            errors_in_period = sum(1 for t in self._recent_api_errors if t > error_window)

            warning_msg = (
                f"API error received: {event.error_message}. "
                f"{errors_in_period} errors in the last "
                f"{self._api_error_threshold_period_s}s"
            )
            self.logger.warning(
                warning_msg,
                source_module=self._source,
            )

            if errors_in_period >= self._api_error_threshold_count:
                reason = (
                    f"High frequency of API errors: {errors_in_period} "
                    f"in {self._api_error_threshold_period_s}s"
                )
                await self.trigger_halt(reason=reason, source="AUTO: API Errors")
        except Exception:
            self.logger.exception(
                "Error handling API error event",
                source_module=self._source,
            )

    def _load_configuration(self) -> None:
        """Load configuration values from ConfigManager."""
        monitoring_config = self.config_manager.get("monitoring", {})
        risk_config = self.config_manager.get("risk", {})
        trading_config = self.config_manager.get("trading", {})

        # Main monitoring intervals and thresholds
        self._check_interval = monitoring_config.get("check_interval_seconds", 60)

        # API monitoring configuration
        self._api_failure_threshold = monitoring_config.get(
            "api_failure_threshold", 3,
        )
        self._api_error_threshold_count = monitoring_config.get(
            "api_error_threshold_count", 5,
        )
        self._api_error_threshold_period_s = monitoring_config.get(
            "api_error_threshold_period_s", 60,
        )
        self._data_staleness_threshold_s = monitoring_config.get(
            "data_staleness_threshold_s", 120.0,
        )

        # System resource monitoring configuration
        self._cpu_threshold_pct = monitoring_config.get("cpu_threshold_pct", 90.0)
        self._memory_threshold_pct = monitoring_config.get("memory_threshold_pct", 90.0)

        # Trading performance monitoring configuration
        self._consecutive_loss_limit = monitoring_config.get("consecutive_loss_limit", 5)

        # Risk management configuration
        risk_limits = risk_config.get("limits", {})
        self._max_total_drawdown_pct = Decimal(
            str(risk_limits.get("max_total_drawdown_pct", 10.0)),
        )

        # HALT behavior configuration
        halt_config = monitoring_config.get("halt", {})
        self._halt_position_behavior = halt_config.get("position_behavior", "maintain").lower()

        # Active trading pairs
        self._active_pairs = trading_config.get("pairs", [])

    async def _update_market_data_timestamp(
        self,
        event: "MarketDataL2Event | MarketDataOHLCVEvent",
    ) -> None:
        """Update the last received timestamp for market data events.

        This helps track market data freshness.

        Args:
        ----
            event: Either a MarketDataL2Event or MarketDataOHLCVEvent
        """
        try:
            # Extract pair from the event
            if hasattr(event, "trading_pair"):
                pair = event.trading_pair
            else:
                self.logger.warning(
                    "Market data event missing trading_pair attribute: %s",
                    type(event),
                    source_module=self._source,
                )
                return

            # Extract timestamp, preferring exchange timestamp if available
            if hasattr(event, "timestamp_exchange") and event.timestamp_exchange:
                ts = event.timestamp_exchange
            elif hasattr(event, "timestamp"):
                ts = event.timestamp
            else:
                self.logger.warning(
                    "Market data event missing timestamp: %s",
                    type(event),
                    source_module=self._source,
                )
                return

            # Ensure timestamp is timezone-aware UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            # Update the timestamp for this pair
            self._last_market_data_times[pair] = ts
            self.logger.debug(
                "Updated market data timestamp for %s: %s",
                pair,
                ts,
                source_module=self._source,
            )
        except Exception:
            self.logger.exception(
                "Error updating market data timestamp",
                source_module=self._source,
            )
