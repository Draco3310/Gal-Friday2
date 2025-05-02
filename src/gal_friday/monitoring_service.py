# Monitoring Service Module

import asyncio
from typing import TYPE_CHECKING, Optional, Callable, Any, Coroutine
from datetime import datetime
from decimal import Decimal
import uuid

# Import actual classes when available, otherwise use placeholders
if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .core.pubsub import PubSubManager
    from .portfolio_manager import PortfolioManager
    from .core.events import EventType, SystemStateEvent, PotentialHaltTriggerEvent
    from .logger_service import LoggerService, PoolProtocol
else:
    # Simple placeholder classes for testing/development
    class _EventType:
        SYSTEM_STATE_CHANGE = "SYSTEM_STATE_CHANGE"  # Example value
        POTENTIAL_HALT_TRIGGER = "POTENTIAL_HALT_TRIGGER"  # Example value

    class _SystemStateEvent:
        def __init__(self, timestamp: datetime, new_state: str, reason: str, source: str):
            self.timestamp = timestamp
            self.new_state = new_state
            self.reason = reason
            self.source = source

    class _ConfigManager:
        def get(self, key: str, default=None):
            # Provide default values for testing/running without real config
            if key == "monitoring.check_interval_seconds":
                return 60
            if key == "risk.limits.max_total_drawdown_pct":
                return Decimal("10.0")  # Example value
            return default

    class _PubSubManager:
        async def publish(self, event_type: str, event: object):
            print(f"Publishing {event_type}: {event.__dict__}")  # Simple print

        def subscribe(self, event_type: str, handler: Callable[[Any], Coroutine[Any, Any, None]]):
            # Placeholder for subscribe method
            pass

        def unsubscribe(self, event_type: str, handler: Callable[[Any], Coroutine[Any, Any, None]]):
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
    PotentialHaltTriggerEvent = _SystemStateEvent  # Assuming the same structure

# log = logging.getLogger(__name__) # Removed module-level logger

# Remove the conflicting LoggerService definition below
# from typing import TypeVar, Generic
# from .logger_service import PoolProtocol
# 
# T = TypeVar('T', bound='PoolProtocol')
# 
# class LoggerService(Generic[T]):
#     def info(self, message: str, source_module: Optional[str] = None, **kwargs: Any) -> None:
#         pass
# 
#     def debug(self, message: str, source_module: Optional[str] = None, **kwargs: Any) -> None:
#         pass
# 
#     def warning(self, message: str, source_module: Optional[str] = None, **kwargs: Any) -> None:
#         pass
# 
#     def error(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None, exc_info: Optional[Any] = None) -> None:
#         pass
# 
#     def critical(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None, exc_info: Optional[Any] = None) -> None:
#         pass


class MonitoringService:
    """
    Monitors the overall system health and manages the global HALT state.
    Triggers HALT based on predefined conditions (e.g., max drawdown) or manual requests.
    Publishes system state changes (HALTED/RUNNING) via the PubSubManager.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: "PubSubManager",
        portfolio_manager: "PortfolioManager",
        logger_service: "LoggerService",
    ):
        """
        Initializes the MonitoringService.

        Args:
            config_manager: The application's configuration manager instance.
            pubsub_manager: The application's publish/subscribe manager instance.
            portfolio_manager: The application's portfolio manager instance.
            logger_service: The shared logger instance.
        """
        self._config = config_manager
        self._pubsub = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self.logger = logger_service
        self._source = self.__class__.__name__

        self._is_halted: bool = False
        self._periodic_check_task: Optional[asyncio.Task] = None
        # Handler storage for unsubscribing
        self._potential_halt_handler: Optional[Callable[[Any], Coroutine[Any, Any, None]]] = None 

        # Load configuration
        try:
            self._check_interval = int(
                self._config.get("monitoring.check_interval_seconds", 60)
            )
            raw_max_drawdown = self._config.get(
                "risk.limits.max_total_drawdown_pct",
                "10.0"
            )
            self._max_drawdown_pct = Decimal(raw_max_drawdown)
            self.logger.info(
                "MonitoringService configured: "
                f"Check Interval={self._check_interval}s, "
                f"Max Drawdown={self._max_drawdown_pct}%",
                source_module=self._source,
            )
        except Exception:
            self.logger.error(
                "Failed to load configuration for MonitoringService. "
                "Using defaults.",
                source_module=self._source,
                exc_info=True,
            )
            self._check_interval = 60
            self._max_drawdown_pct = Decimal("10.0")  # Default fallback

        self.logger.info("MonitoringService initialized.", source_module=self._source)

    def is_halted(self) -> bool:
        """Synchronously checks if the system is currently halted."""
        return self._is_halted

    async def start(self) -> None:
        """Starts the periodic monitoring checks."""
        # Publish initial state when starting, if not already halted
        if not self._is_halted:
            await self._publish_state_change("RUNNING", "System startup", "MonitoringService Start")
            
        if self._periodic_check_task and not self._periodic_check_task.done():
            self.logger.warning(
                "MonitoringService periodic check task already running.",
                source_module=self._source,
            )
            return

# log.info(f"Starting MonitoringService
# periodic checks every {self._check_interval} seconds.")
# # Removed
        msg = (
            "Starting MonitoringService periodic checks "
            "every {} seconds."
        ).format(self._check_interval)
        self.logger.info(
            msg,
            source_module=self._source
        )

        self._periodic_check_task = asyncio.create_task(self._run_periodic_checks())
        
        # Subscribe to potential halt events
        self._potential_halt_handler = self._handle_potential_halt_trigger
        self._pubsub.subscribe(EventType.POTENTIAL_HALT_TRIGGER, self._potential_halt_handler)
        self.logger.info("Subscribed to POTENTIAL_HALT_TRIGGER events.", source_module=self._source)

    async def stop(self) -> None:
        """Stops the periodic monitoring checks."""
        # Unsubscribe first
        if self._potential_halt_handler:
            try:
                self._pubsub.unsubscribe(EventType.POTENTIAL_HALT_TRIGGER, self._potential_halt_handler)
                self.logger.info("Unsubscribed from POTENTIAL_HALT_TRIGGER events.", source_module=self._source)
                self._potential_halt_handler = None
            except Exception as e:
                self.logger.error(f"Error unsubscribing from POTENTIAL_HALT_TRIGGER: {e}", exc_info=True, source_module=self._source)
                
        if self._periodic_check_task and not self._periodic_check_task.done():
            # log.info("Stopping MonitoringService periodic checks...") # Removed
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
                self.logger.error(
                    "Error encountered while stopping monitoring task.",
                    source_module=self._source,
                    exc_info=True,
                )
            finally:
                self._periodic_check_task = None
        else:
            # log.info("MonitoringService periodic check task was not running.") # Removed
            self.logger.info(
                "MonitoringService periodic check task was not running.",
                source_module=self._source,
            )

    async def trigger_halt(self, reason: str, source: str) -> None:
        """
        Halts the system operations.

        Args:
            reason: The reason for halting the system.
            source: The source triggering the halt (e.g., 'MANUAL', 'AUTO: Max Drawdown').
        """
        if self._is_halted:
            # log.warning(f"System already halted. Ignoring HALT trigger from {source}.") # Removed
            self.logger.warning(
                f"System already halted. Ignoring HALT trigger from {source}.",
                source_module=self._source,
            )
            return

        self._is_halted = True
        # log.critical(f"SYSTEM HALTED by {source}. Reason: {reason}") # Removed
        self.logger.critical(
            f"SYSTEM HALTED by {source}. Reason: {reason}",
            source_module=self._source,
        )
        await self._publish_state_change("HALTED", reason, source)

    async def trigger_resume(self, source: str) -> None:
        """
        Resumes system operations after a HALT.

        Args:
            source: The source triggering the resume (e.g., 'MANUAL').
        """
        if not self._is_halted:
            self.logger.warning(
                f"System is not halted. Ignoring RESUME trigger from {source}.",
                source_module=self._source,
            )
            return

        self._is_halted = False
        self.logger.info(
            f"SYSTEM RESUMED by {source}.",
            source_module=self._source,
        )
        await self._publish_state_change("RUNNING", "System resumed", source)

    async def _publish_state_change(self, new_state: str, reason: str, source: str) -> None:
        """Helper method to publish SystemStateEvent."""
        # Create the specific event object
        event = SystemStateEvent(
            source_module=source, # Use the trigger source here
            event_id=uuid.uuid4(),  # Generate a UUID for the event
            timestamp=datetime.utcnow(),  # Use current time
            new_state=new_state,
            reason=reason,
            # halt_action might be relevant for HALTED state, needs logic
        )
        try:
            # Publish the event object
            await self._pubsub.publish(event)
            self.logger.info(
                f"Published SystemStateEvent: {new_state} (Source: {source})",
                source_module=self._source,
            )
        except Exception:
            self.logger.error(
                f"Failed to publish SystemStateEvent ({new_state}).",
                source_module=self._source,
                exc_info=True,
            )
            
    async def _handle_potential_halt_trigger(self, event: "PotentialHaltTriggerEvent") -> None:
        """Handles events that suggest a potential HALT condition."""
        if not isinstance(event, PotentialHaltTriggerEvent):
            self.logger.warning(f"Received non-PotentialHaltTriggerEvent: {type(event)}", source_module=self._source)
            return
            
        self.logger.warning(f"Potential HALT condition received from {event.source_module}: {event.reason}", source_module=self._source)
        # Trigger actual halt - might add confirmation logic later
        await self.trigger_halt(reason=event.reason, source=event.source_module)

    async def _run_periodic_checks(self) -> None:
        """The core background task performing periodic checks."""
        # log.info("MonitoringService periodic check task started.") # Removed
        self.logger.info(
            "MonitoringService periodic check task started.",
            source_module=self._source,
        )
        while True:
            try:
                await asyncio.sleep(self._check_interval)

                if not self._is_halted:
                    # log.debug("Running periodic checks...") # Removed
                    self.logger.debug(
                        "Running periodic checks...",
                        source_module=self._source,
                    )
                    await self._check_drawdown()
                    # Add calls to other check methods here in the future
                    # e.g., await self._check_connectivity()
                    # e.g., await self._check_api_limits()

            except asyncio.CancelledError:
                # log.info("MonitoringService periodic check task cancelled.") # Removed
                self.logger.info(
                    "MonitoringService periodic check task cancelled.",
                    source_module=self._source,
                )
                break
            except Exception:
                self.logger.error(
                    "Unhandled error during periodic monitoring check. "
                    "Continuing...",
                    source_module=self._source,
                    exc_info=True,
                )
                # Avoid tight loop on unexpected errors
                await asyncio.sleep(self._check_interval)

    async def _check_drawdown(self) -> None:
        """Checks if the maximum total portfolio drawdown has been exceeded."""
        try:
            # PortfolioManager.get_current_state() needs to be synchronous per design doc
            # If it becomes async, this needs adjustment (e.g., run_in_executor)
            # For now, assuming it's sync as requested for MVP.
            current_state = self._portfolio_manager.get_current_state()
            drawdown_pct = current_state.get("total_drawdown_pct")

            if drawdown_pct is None:
                self.logger.warning(
                    "Could not retrieve 'total_drawdown_pct' from "
                    "PortfolioManager state.",
                    source_module=self._source,
                )
                return

            # Ensure drawdown_pct is Decimal
            if not isinstance(drawdown_pct, Decimal):
                try:
                    drawdown_pct = Decimal(drawdown_pct)
                except Exception:
                    self.logger.warning(
                        "Invalid type for 'total_drawdown_pct': "
                        f"{type(drawdown_pct)}. Skipping check.",
                        source_module=self._source,
                    )
                    return

            self.logger.debug(
                "Current total drawdown: {:.2f}% (Limit: {}%)".format(
                    drawdown_pct, self._max_drawdown_pct
                ),
                source_module=self._source,
            )

            # Check if drawdown exceeds the limit (absolute value)
            if abs(drawdown_pct) > self._max_drawdown_pct:
                reason = (
                    "Maximum total drawdown limit exceeded: "
                    "{:.2f}% > {}%".format(
                        abs(drawdown_pct), self._max_drawdown_pct
                    )
                )
                self.logger.warning(reason, source_module=self._source)
                await self.trigger_halt(reason=reason, source="AUTO: Max Drawdown")

        except Exception:
            self.logger.error(
                "Error occurred during drawdown check.",
                source_module=self._source,
                exc_info=True,
            )


# Example Usage (for testing purposes, remove in production)
async def main(logger: Optional["LoggerService[Any]"] = None) -> None:
    """Main function for testing purposes, to be removed in production."""
    import logging  # Need for example main
    from typing import Any, cast, Dict, Union

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    from .logger_service import LoggerService
    from .config_manager import ConfigManager
    from .portfolio_manager import PortfolioManager
    from .event_bus import PubSubManager  # Changed import to match expected type

    # Create a mock logger service that implements LoggerService
    class TestLoggerService(LoggerService[Any]):
        def __init__(self, config_manager: "ConfigManager", pubsub_manager: "PubSubManager") -> None:
            pass
            
        # Updated method signatures to match base LoggerService
        def info(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None) -> None:
            print(f"INFO [{source_module}]: {message}")
        def debug(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None) -> None:
            print(f"DEBUG [{source_module}]: {message}")
        def warning(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None) -> None:
            print(f"WARN [{source_module}]: {message}")
        def error(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None, exc_info: Optional[Any] = None) -> None:
            print(f"ERROR [{source_module}]: {message}")
        def critical(self, message: str, source_module: Optional[str] = None, context: Optional[dict[Any, Any]] = None, exc_info: Optional[Any] = None) -> None:
            print(f"CRITICAL [{source_module}]: {message}")

    # Create a mock config manager that inherits from ConfigManager
    class MockConfigManager(ConfigManager):
        def __init__(self) -> None:
            pass
            
        def get(self, key: str, default: Any = None) -> Any:
            # Return some sensible defaults
            if key == "monitoring.check_interval_seconds":
                return 60
            if key == "risk.limits.max_total_drawdown_pct":
                return Decimal("10.0")
            return default

    # Create a mock portfolio manager that inherits from PortfolioManager
    class MockPortfolioManager(PortfolioManager):
        def __init__(self) -> None:
            pass
            
        def get_current_state(self) -> Dict[str, Any]:
            # Return dummy data for testing
            return {"total_drawdown_pct": Decimal("1.5"),
                    "daily_drawdown_pct": Decimal("0.5"),
                    "weekly_drawdown_pct": Decimal("1.2"),
                    "total_equity": Decimal("100000")}

    # Create a mock for high drawdown that properly inherits from PortfolioManager
    class MockPortfolioManagerHighDrawdown(PortfolioManager):
        def __init__(self) -> None:
            pass
            
        def get_current_state(self) -> Dict[str, Any]:
            # Exceeds default 10%
            return {"total_drawdown_pct": Decimal("15.0"),
                    "daily_drawdown_pct": Decimal("3.0"),
                    "weekly_drawdown_pct": Decimal("8.0"),
                    "total_equity": Decimal("85000")}

    # Import the real Logger type to ensure proper type compatibility
    from logging import Logger
    
    # Use placeholder implementations
    config_mgr = MockConfigManager()
    
    # Need a real logger instance for logging purposes
    temp_logger = logging.getLogger("mock_pubsub_logger")
    temp_logger.addHandler(logging.NullHandler()) # Avoid "No handlers could be found"
    
    # Initialize PubSubManager without the logger parameter
    pubsub_mgr = PubSubManager()
    
    # Create logger service instance
    logger_service = TestLoggerService(config_manager=config_mgr, pubsub_manager=pubsub_mgr)
    
    portfolio_mgr = MockPortfolioManager()

    # Create the monitoring service
    monitor_service = MonitoringService(
        config_manager=config_mgr,
        pubsub_manager=pubsub_mgr,  # type: ignore # The actual type from event_bus is compatible
        portfolio_manager=portfolio_mgr,
        logger_service=logger_service
    )

    print("Starting monitor service...")
    await monitor_service.start()

    print("Simulating work for 15 seconds...")
    await asyncio.sleep(15)

    print("Triggering manual HALT...")
    await monitor_service.trigger_halt(
        reason="Manual intervention test",
        source="TEST_SCRIPT"
    )
    print(f"System halted: {monitor_service.is_halted()}")
    await asyncio.sleep(5)

    print("Triggering RESUME...")
    await monitor_service.trigger_resume(source="TEST_SCRIPT")
    print(f"System halted: {monitor_service.is_halted()}")
    await asyncio.sleep(5)

    print("Simulating drawdown breach...")

    monitor_service._portfolio_manager = MockPortfolioManagerHighDrawdown()
    monitor_service._check_interval = 2  # Check faster for test
    print("Waiting for drawdown check...")
    await asyncio.sleep(5)  # Wait longer than check interval
    print(f"System halted after drawdown check: {monitor_service.is_halted()}")

    print("Stopping monitor service...")
    await monitor_service.stop()
    print("Monitor service stopped.")


if __name__ == "__main__":
    # Remove the # if you want to run this file directly for testing
    # asyncio.run(main())
    pass
