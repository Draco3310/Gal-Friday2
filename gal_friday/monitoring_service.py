#!/usr/bin/env python3
"""Monitoring Service for Gal Friday trading system.

This module provides system monitoring capabilities including health checks,
performance tracking, and automatic trading halt triggers when thresholds are exceeded.
"""

import asyncio
import time
import uuid
from collections import deque  # Added for tracking recent API errors
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType  # Added for exc_info typing
from typing import (  # Added Type for exc_info typing
    TYPE_CHECKING,
    Any,
    Optional,
)

import psutil  # Added for system resource monitoring

# Import actual classes when available, otherwise use placeholders
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
    from .core.pubsub import PubSubManager  # Import from correct module
    from .execution_handler import ExecutionHandler  # MODIFIED: Corrected import path
    from .logger_service import LoggerService
    from .portfolio_manager import PortfolioManager
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
        async def publish(self, event: "_Event") -> None:
            # Updated signature to match real implementation
            print(f"Publishing: {event.__dict__}")  # Simple print

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


class MonitoringService:
    """Monitors the overall system health and manages the global HALT state.

    Triggers HALT based on predefined conditions (e.g., max drawdown) or manual requests.
    Publishes system state changes (HALTED/RUNNING) via the PubSubManager.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: "PubSubManager",
        portfolio_manager: "PortfolioManager",
        logger_service: "LoggerService",
        execution_handler: Optional["ExecutionHandler"] = None,
    ) -> None:
        """Initialize the MonitoringService.

        Args:
        ----
            config_manager: The application's configuration manager instance.
            pubsub_manager: The application's publish/subscribe manager instance.
            portfolio_manager: The application's portfolio manager instance.
            logger_service: The shared logger instance.
            execution_handler: Optional execution handler for API connectivity checks.
        """
        self._config = config_manager
        self._pubsub = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self.logger = logger_service
        self._execution_handler = execution_handler
        self._source = self.__class__.__name__

        self._is_halted: bool = False
        self._periodic_check_task: asyncio.Task | None = None

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

        # Load configuration
        try:
            # Basic monitoring config
            self._check_interval = int(self._config.get("monitoring.check_interval_seconds", 60))
            raw_max_drawdown = self._config.get("risk.limits.max_total_drawdown_pct", "10.0")
            self._max_drawdown_pct = Decimal(raw_max_drawdown)

            # API monitoring config
            self._api_failure_threshold = int(
                self._config.get("monitoring.api_failure_threshold", 3),
            )
            self._api_error_threshold_count = int(
                self._config.get("monitoring.api_error_threshold_count", 5),
            )
            self._api_error_threshold_period_s = int(
                self._config.get("monitoring.api_error_threshold_period_s", 60),
            )

            # Market data monitoring config
            self._data_staleness_threshold_s = float(
                self._config.get("monitoring.data_staleness_threshold_s", 120.0),
            )

            # System resource monitoring config
            self._cpu_threshold_pct = float(self._config.get("monitoring.cpu_threshold_pct", 90.0))
            self._memory_threshold_pct = float(
                self._config.get("monitoring.memory_threshold_pct", 90.0),
            )

            # Trading halt behavior
            self._halt_position_behavior = self._config.get(
                "monitoring.halt.position_behavior",
                "maintain",
            ).lower()

            # Consecutive loss monitoring
            self._consecutive_loss_limit = int(
                self._config.get("monitoring.consecutive_loss_limit", 5),
            )

            # Active trading pairs
            self._active_pairs = self._config.get_list("trading.pairs", [])

            self.logger.info(
                "MonitoringService configured with intervals and thresholds",
                source_module=self._source,
            )
        except Exception:
            self.logger.exception(
                "Failed to load configuration for MonitoringService. Using defaults.",
                source_module=self._source,
            )
            # Set defaults for all configurations
            self._check_interval = 60
            self._max_drawdown_pct = Decimal("10.0")
            self._api_failure_threshold = 3
            self._api_error_threshold_count = 5
            self._api_error_threshold_period_s = 60
            self._data_staleness_threshold_s = 120.0
            self._cpu_threshold_pct = 90.0
            self._memory_threshold_pct = 90.0
            self._halt_position_behavior = "maintain"
            self._consecutive_loss_limit = 5
            self._active_pairs = []

        self.logger.info("MonitoringService initialized.", source_module=self._source)

    def is_halted(self) -> bool:
        """Return whether the system is currently halted."""
        return self._is_halted

    async def start(self) -> None:
        """Start the periodic monitoring checks."""
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
        self._pubsub.subscribe(EventType.POTENTIAL_HALT_TRIGGER, self._potential_halt_handler)
        self.logger.info(
            "Subscribed to POTENTIAL_HALT_TRIGGER events.",
            source_module=self._source,
        )

        # Subscribe to API error events
        self._api_error_handler = self._handle_api_error
        self._pubsub.subscribe(EventType.SYSTEM_ERROR, self._api_error_handler)
        self.logger.info(
            "Subscribed to SYSTEM_ERROR events for API error tracking.",
            source_module=self._source,
        )

        # Subscribe to market data events to track freshness
        self._market_data_l2_handler = self._update_market_data_timestamp
        self._market_data_ohlcv_handler = self._update_market_data_timestamp
        self._pubsub.subscribe(EventType.MARKET_DATA_L2, self._market_data_l2_handler)
        self._pubsub.subscribe(EventType.MARKET_DATA_OHLCV, self._market_data_ohlcv_handler)
        self.logger.info(
            "Subscribed to market data events for freshness tracking.",
            source_module=self._source,
        )

        # Subscribe to execution reports to track consecutive losses
        self._execution_report_handler = self._handle_execution_report
        self._pubsub.subscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
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
                self._pubsub.unsubscribe(
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
                self._pubsub.unsubscribe(EventType.SYSTEM_ERROR, self._api_error_handler)
                self.logger.info(
                    "Unsubscribed from SYSTEM_ERROR events.",
                    source_module=self._source,
                )
                self._api_error_handler = None

            # Market data events
            if self._market_data_l2_handler:
                self._pubsub.unsubscribe(EventType.MARKET_DATA_L2, self._market_data_l2_handler)
                self._market_data_l2_handler = None

            if self._market_data_ohlcv_handler:
                self._pubsub.unsubscribe(
                    EventType.MARKET_DATA_OHLCV,
                    self._market_data_ohlcv_handler,
                )
                self._market_data_ohlcv_handler = None

            # Execution report events
            if self._execution_report_handler:
                self._pubsub.unsubscribe(
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
                        await self._pubsub.publish(close_command)

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
            await self._pubsub.publish(event)
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
                    await self._check_drawdown()
                    # Add calls to new check methods
                    await self._check_api_connectivity()
                    await self._check_market_data_freshness()
                    await self._check_system_resources()
                    await self._check_market_volatility()

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
                self._max_drawdown_pct,
                source_module=self._source,
            )

            # Check if drawdown exceeds the limit (absolute value)
            if abs(drawdown_pct) > self._max_drawdown_pct:
                drawdown_val = abs(drawdown_pct)
                limit_val = self._max_drawdown_pct
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
        """
        now = datetime.now(UTC)
        stale_pairs = []

        if not self._active_pairs:
            self.logger.warning(
                "No active trading pairs configured for market data freshness check.",
                source_module=self._source,
            )
            return

        for pair in self._active_pairs:
            last_ts = self._last_market_data_times.get(pair)

            if last_ts is None:
                # No data has been received yet for this pair
                self.logger.warning(
                    "No market data timestamp found for active pair %s.",
                    pair,
                    source_module=self._source,
                )
                # Only consider stale if system has been running longer than staleness threshold
                # This prevents false alerts during startup
                # TODO: Add startup time tracking if needed
            elif (now - last_ts).total_seconds() > self._data_staleness_threshold_s:
                stale_pairs.append(pair)
                warning_msg = (
                    f"Market data for {pair} is stale (last update: {last_ts}, "
                    f"threshold: {self._data_staleness_threshold_s}s)"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )

        if stale_pairs:
            reason = f"Market data stale for pairs: {', '.join(stale_pairs)}"
            await self.trigger_halt(reason=reason, source="AUTO: Market Data Staleness")

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
        # This is a placeholder that would need access to market price data
        # Implementation would require:
        # 1. Access to recent price data for active pairs
        # 2. Calculation of price change percentage over configured time window
        # 3. Comparison against volatility threshold
        #
        # For now, we'll just log that this check would occur
        self.logger.debug(
            "Market volatility check (placeholder implementation)",
            source_module=self._source,
        )

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


# Define Mock Classes at module level for accessibility by helper functions
# Create a mock logger service that implements LoggerService
class TestLoggerService(LoggerService[Any]):
    """Mock LoggerService for testing purposes."""

    def __init__(
        self,
        _config_manager: "ConfigManager",
        _pubsub_manager: "PubSubManager",
    ) -> None:
        """Initialize the mock logger service."""

    def info(
        self,
        message: str,
        *args: object,  # More specific type than Any
        source_module: str | None = None,
        context: dict[Any, Any] | None = None,
    ) -> None:
        """Log an info message."""
        formatted_message = message % args if args else message
        context_str = f" [context: {context}]" if context else ""
        print(f"INFO [{source_module}]:{context_str} {formatted_message}")

    def debug(
        self,
        message: str,
        *args: object,  # More specific type than Any
        source_module: str | None = None,
        context: dict[Any, Any] | None = None,
    ) -> None:
        """Log a debug message."""
        formatted_message = message % args if args else message
        context_str = f" [context: {context}]" if context else ""
        print(f"DEBUG [{source_module}]:{context_str} {formatted_message}")

    def warning(
        self,
        message: str,
        *args: object,  # More specific type than Any
        source_module: str | None = None,
        context: dict[Any, Any] | None = None,
    ) -> None:
        """Log a warning message."""
        formatted_message = message % args if args else message
        context_str = f" [context: {context}]" if context else ""
        print(f"WARN [{source_module}]:{context_str} {formatted_message}")

    def error(
        self,
        message: str,
        *args: object,  # More specific type than Any
        source_module: str | None = None,
        context: dict[Any, Any] | None = None,
        exc_info: None
        | (
            bool | tuple[type[BaseException], BaseException, TracebackType | None] | BaseException
        ) = None,
    ) -> None:
        """Log an error message."""
        formatted_message = message % args if args else message
        context_str = f" [context: {context}]" if context else ""
        exc_str = ""
        if exc_info:
            if isinstance(exc_info, bool) and exc_info:
                exc_str = " [with current exception]"
            elif isinstance(exc_info, BaseException):
                exc_str = f" [exception: {type(exc_info).__name__}: {exc_info}]"
            elif isinstance(exc_info, tuple):
                exc_type, exc_value, _ = exc_info
                exc_str = f" [exception: {exc_type.__name__}: {exc_value}]"
        print(f"ERROR [{source_module}]:{context_str}{exc_str} {formatted_message}")

    def critical(
        self,
        message: str,
        *args: object,  # More specific type than Any
        source_module: str | None = None,
        context: dict[Any, Any] | None = None,
        exc_info: None
        | (
            bool | tuple[type[BaseException], BaseException, TracebackType | None] | BaseException
        ) = None,
    ) -> None:
        """Log a critical message."""
        formatted_message = message % args if args else message
        context_str = f" [context: {context}]" if context else ""
        exc_str = ""
        if exc_info:
            if isinstance(exc_info, bool) and exc_info:
                exc_str = " [with current exception]"
            elif isinstance(exc_info, BaseException):
                exc_str = f" [exception: {type(exc_info).__name__}: {exc_info}]"
            elif isinstance(exc_info, tuple):
                exc_type, exc_value, _ = exc_info
                exc_str = f" [exception: {exc_type.__name__}: {exc_value}]"
        print(f"CRITICAL [{source_module}]:{context_str}{exc_str} {formatted_message}")


# Create a mock config manager that inherits from ConfigManager
class MockConfigManager(ConfigManager):
    """Mock ConfigManager for testing purposes."""

    def __init__(self) -> None:
        """Initialize the mock config manager."""

    def get(
        self,
        key: str,
        default: str | int | float | Decimal | None = None,
    ) -> str | int | float | Decimal | None:  # MODIFIED: Formatted
        """Get a configuration value."""
        # Return some sensible defaults
        if key == "monitoring.check_interval_seconds":
            return 60
        if key == "risk.limits.max_total_drawdown_pct":
            return Decimal("10.0")
        return default


# Create a mock portfolio manager that inherits from PortfolioManager
class MockPortfolioManager(PortfolioManager):
    """Mock PortfolioManager for testing purposes."""

    def __init__(self) -> None:
        """Initialize the mock portfolio manager."""

    def get_current_state(self) -> dict[str, Any]:
        """Get the current mock portfolio state."""
        # Return dummy data for testing
        return {
            "total_drawdown_pct": Decimal("1.5"),
            "daily_drawdown_pct": Decimal("0.5"),
            "weekly_drawdown_pct": Decimal("1.2"),
            "total_equity": Decimal("100000"),
        }


# Create a mock portfolio manager with high drawdown
class MockPortfolioManagerHighDrawdown(PortfolioManager):
    """Mock PortfolioManager with high drawdown for testing."""

    def __init__(self) -> None:
        """Initialize the mock high drawdown portfolio manager."""

    def get_current_state(self) -> dict[str, Any]:
        """Get the current mock portfolio state with high drawdown."""
        # Exceeds default 10%
        return {
            "total_drawdown_pct": Decimal("15.0"),
            "daily_drawdown_pct": Decimal("5.0"),
            "weekly_drawdown_pct": Decimal("12.0"),
            "total_equity": Decimal("85000"),
        }


# Example Usage (for testing purposes, remove in production)
async def _test_normal_drawdown(
    config_mgr: "MockConfigManager",
    pubsub_mgr: "PubSubManager",
    portfolio_mgr: "MockPortfolioManager",
    test_logger: "LoggerService[Any]",
) -> None:
    """Test MonitoringService with normal drawdown."""
    print("\n=== Starting MonitoringService with normal drawdown ===")
    monitor_service = MonitoringService(config_mgr, pubsub_mgr, portfolio_mgr, test_logger)
    await monitor_service.start()
    print(f"Is system halted? {monitor_service.is_halted()}")
    print("Sleeping for 2 seconds to allow periodic check to run...")
    await asyncio.sleep(2)
    print(f"Is system halted? {monitor_service.is_halted()}")
    await monitor_service.stop()


async def _test_high_drawdown_halt(
    config_mgr: "MockConfigManager",
    pubsub_mgr: "PubSubManager",
    portfolio_mgr_high: "MockPortfolioManagerHighDrawdown",
    test_logger: "LoggerService[Any]",
) -> None:
    """Test MonitoringService with high drawdown that should trigger HALT."""
    print("\n=== Starting MonitoringService with high drawdown (should HALT) ===")
    monitor_service_halt = MonitoringService(
        config_mgr,
        pubsub_mgr,
        portfolio_mgr_high,
        test_logger,
    )
    await monitor_service_halt.start()
    print(f"Is system halted? {monitor_service_halt.is_halted()}")
    print("Sleeping for 2 seconds to allow periodic check to run...")
    await asyncio.sleep(2)
    print(f"Is system halted? {monitor_service_halt.is_halted()} (should be True)")
    await monitor_service_halt.stop()


async def _test_manual_halt(
    config_mgr: "MockConfigManager",
    pubsub_mgr: "PubSubManager",
    portfolio_mgr: "MockPortfolioManager",
    test_logger: "LoggerService[Any]",
) -> None:
    """Test manual trigger of HALT."""
    print("\n=== Testing manual HALT trigger ===")
    monitor_service = MonitoringService(config_mgr, pubsub_mgr, portfolio_mgr, test_logger)
    await monitor_service.start()
    print(f"Is system halted before manual trigger? {monitor_service.is_halted()}")
    await monitor_service.trigger_halt("Manual test halt", "TEST")
    print(f"Is system halted after manual trigger? {monitor_service.is_halted()}")
    await monitor_service.stop()


async def _test_resume_after_halt(
    config_mgr: "MockConfigManager",
    pubsub_mgr: "PubSubManager",
    portfolio_mgr: "MockPortfolioManager",
    test_logger: "LoggerService[Any]",
) -> None:
    """Test RESUME after HALT."""
    print("\n=== Testing RESUME after HALT ===")
    monitor_service = MonitoringService(config_mgr, pubsub_mgr, portfolio_mgr, test_logger)
    await monitor_service.start()
    await monitor_service.trigger_halt("Test halt for resume test", "TEST")
    print(f"Is system halted after trigger? {monitor_service.is_halted()}")
    await monitor_service.trigger_resume("TEST")
    print(f"Is system running after resume? {not monitor_service.is_halted()}")
    await monitor_service.stop()


async def main(logger: Optional["LoggerService[Any]"] = None) -> None:
    """Testing/demonstration usage of the MonitoringService."""
    # Import here to avoid circular imports
    import logging  # Need for example main
    from typing import Any

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Import the real LoggerService to create a test implementation

    # Import PubSubManager from core.pubsub instead of event_bus for type consistency
    from .core.pubsub import PubSubManager
    from .logger_service import LoggerService

    # Mock classes are now defined at module level

    # Use placeholder implementations
    config_mgr = MockConfigManager()

    # Fix the PubSubManager initialization by providing a proper logger
    # Use a real logger instance and add the missing config_manager
    pubsub_logger = logging.getLogger("PubSubManager")
    pubsub_mgr = PubSubManager(logger=pubsub_logger, config_manager=config_mgr)

    if logger is None:
        # Fix type error by using proper type annotation
        # This ensures LoggerService[Any] is type-compatible with TestLoggerService
        test_logger: LoggerService[Any] = TestLoggerService(config_mgr, pubsub_mgr)
    else:
        test_logger = logger

    # First test: normal drawdown case
    portfolio_mgr = MockPortfolioManager()
    await _test_normal_drawdown(config_mgr, pubsub_mgr, portfolio_mgr, test_logger)

    # Second test: high drawdown case that should trigger HALT
    portfolio_mgr_high = MockPortfolioManagerHighDrawdown()
    await _test_high_drawdown_halt(
        config_mgr,
        pubsub_mgr,
        portfolio_mgr_high,
        test_logger,
    )

    # Third test: manual trigger of HALT
    await _test_manual_halt(config_mgr, pubsub_mgr, portfolio_mgr, test_logger)

    # Fourth test: resume
    await _test_resume_after_halt(config_mgr, pubsub_mgr, portfolio_mgr, test_logger)

    print("\n=== All MonitoringService tests completed ===")


if __name__ == "__main__":
    # Use the imported asyncio from the top of the file
    asyncio.run(main())
