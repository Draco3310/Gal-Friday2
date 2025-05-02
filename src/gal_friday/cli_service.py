# CLI Service Module

import asyncio
import sys
import logging  # For example_main logging configuration
from typing import TYPE_CHECKING, Any, Optional, Dict

if TYPE_CHECKING:
    from .monitoring_service import MonitoringService
    from .logger_service import LoggerService  # Added import

    # Define MainAppController interface for type hinting
    class MainAppController:
        async def shutdown(self) -> None:
            raise NotImplementedError

else:
    # Placeholders if not type checking
    class MonitoringService:  # Placeholder
        def is_halted(self) -> bool:
            return False

        async def trigger_halt(self, reason: str, source: str):
            print(f"HALT triggered by {source}: {reason}")

        async def trigger_resume(self, source: str):
            print(f"RESUME triggered by {source}")

    class MainAppController:  # Placeholder
        async def shutdown(self) -> None:
            print("Shutdown requested by CLI.")


# log = logging.getLogger(__name__) # Removed module-level logger


class CLIService:
    """Handles Command-Line Interface interactions for runtime control."""

    def __init__(
        self,
        monitoring_service: "MonitoringService",
        main_app_controller: "MainAppController",
        logger_service: "LoggerService",
    ):  # Added logger_service
        """
        Initializes the CLIService.

        Args:
            monitoring_service: Instance of the MonitoringService.
            main_app_controller: Instance of the main application controller/orchestrator
                                  which must have an async shutdown() method.
            logger_service: The shared logger instance.
        """
        self.monitoring_service = monitoring_service
        self.main_app_controller = main_app_controller
        self.logger = logger_service  # Assigned injected logger
        self._running = False
        # log.info("CLIService initialized.") # Replaced
        self.logger.info("CLIService initialized.", source_module=self.__class__.__name__)

    async def start(self) -> None:
        """Starts listening for commands on stdin."""
        if self._running:
            # log.warning("CLIService already running.") # Replaced
            self.logger.warning(
                "CLIService already running.", source_module=self.__class__.__name__
            )
            return

        # log.info("Starting CLIService input listener...") # Replaced
        self.logger.info(
            "Starting CLIService input listener...", source_module=self.__class__.__name__
        )
        try:
            loop = asyncio.get_running_loop()
            # Attempt to add reader. Note: May have limitations on Windows.
            loop.add_reader(sys.stdin.fileno(), self._handle_stdin_input)
            self._running = True
            # Give user initial prompt only once
            print("\n---")
            print("Gal-Friday CLI Ready. Type command and press Enter.")
            print("Available commands: halt, resume, status, stop")
            print("---")
        except NotImplementedError:
            self.logger.error(
                "Failed to start CLI listener: loop.add_reader not supported on this "
                "platform/loop implementation (e.g., standard Windows asyncio). "
                "CLI commands unavailable.",
                source_module=self.__class__.__name__,
            )
        except Exception as e:
            self.logger.error(
                f"Error starting CLIService listener: {str(e)}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    async def stop(self) -> None:
        """Stops listening for commands on stdin."""
        if not self._running:
            # log.info("CLIService was not running.") # Replaced
            self.logger.info("CLIService was not running.", source_module=self.__class__.__name__)
            return

        # log.info("Stopping CLIService input listener...") # Replaced
        self.logger.info(
            "Stopping CLIService input listener...", source_module=self.__class__.__name__
        )
        try:
            loop = asyncio.get_running_loop()
            # Check if remove_reader exists and is callable, handle potential NotImplementedError
            if hasattr(loop, "remove_reader") and callable(loop.remove_reader):
                loop.remove_reader(sys.stdin.fileno())
            else:
                self.logger.warning(
                    "loop.remove_reader not available or not callable. "
                    "Cannot remove stdin listener cleanly.",
                    source_module=self.__class__.__name__,
                )
            self._running = False
            print("CLIService stopped listening for commands.")
        except NotImplementedError:
            self.logger.error(
                "Failed to stop CLI listener: loop.remove_reader not supported on this "
                "platform/loop implementation.",
                source_module=self.__class__.__name__,
            )
        except Exception as e:
            self.logger.error(
                f"Error stopping CLIService listener: {str(e)}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    def _handle_stdin_input(self) -> None:
        """Synchronous callback triggered when stdin is readable.
        Reads a command and schedules its processing asynchronously.
        """
        try:
            command = sys.stdin.readline().strip().lower()
            if not command:  # Ignore empty input
                return

            self.logger.info(
                f"CLI command received: '{command}'"
            )  # Simplified - context implies source is CLI

            if command == "halt":
                print(">>> Issuing HALT command...")
                # Schedule the async task without awaiting it here
                asyncio.create_task(
                    self.monitoring_service.trigger_halt(
                        reason="Manual user command via CLI", source=self.__class__.__name__
                    )
                )
            elif command == "resume":
                print(">>> Issuing RESUME command...")
                asyncio.create_task(
                    self.monitoring_service.trigger_resume(source=self.__class__.__name__)
                )
            elif command == "status":
                # This part is synchronous as is_halted() is synchronous
                halted_status = self.monitoring_service.is_halted()
                print(f">>> System Status: {'HALTED' if halted_status else 'RUNNING'}")
                # TODO: Future - Add call to an async method to get more detailed status
            elif command == "stop" or command == "exit":
                print(">>> Issuing STOP command... Initiating graceful shutdown.")
                # Schedule the main application shutdown
                asyncio.create_task(self.main_app_controller.shutdown())
            else:
                print(f">>> Unknown command: '{command}'. Available: halt, resume, status, stop")

        except Exception as e:
            # Catch errors within the callback itself (e.g., issues reading stdin)
            self.logger.error(
                f"Error processing CLI input: {str(e)}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )


# Example Usage (requires running in a context where stdin is available)
async def example_main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a mock logger service
    from typing import TypeVar, Generic, Any
    from .logger_service import LoggerService, PoolProtocol
    
    T = TypeVar('T', bound=PoolProtocol)
    
    class MockLoggerService(LoggerService[T]):
        def info(self, message: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None) -> None:
            print(f"INFO [{source_module}]: {message}")
        def debug(self, message: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None) -> None:
            print(f"DEBUG [{source_module}]: {message}")
        def warning(self, message: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None) -> None:
            print(f"WARN [{source_module}]: {message}")
        def error(self, message: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None, exc_info: Optional[Any] = None) -> None:
            print(f"ERROR [{source_module}]: {message}")
        def critical(self, message: str, source_module: Optional[str] = None, context: Optional[Dict[Any, Any]] = None, exc_info: Optional[Any] = None) -> None:
            print(f"CRITICAL [{source_module}]: {message}")

    # Create a mock PubSubManager
    from .event_bus import PubSubManager  # Import the correct PubSubManager
    from asyncio import Queue
    from typing import Callable, Awaitable, Any, Coroutine
    
    class MockPubSubManager(PubSubManager):  # Inherit from the correct PubSubManager
        def __init__(self) -> None:
            # PubSubManager doesn't accept logger parameter
            super().__init__()
            
            # If we need custom logging, we can set up a module-level logger
            self._mock_logger = logging.getLogger("mock_pubsub")
            self._mock_logger.addHandler(logging.NullHandler())

        async def publish(self, event: Any) -> None:
            # Simple implementation that just logs the event
            if hasattr(event, 'event_id'):
                print(f"MockPubSubManager: Would publish event {event.event_id}")
            else:
                print(f"MockPubSubManager: Would publish event {event}")

    # Create mock config manager
    from .config_manager import ConfigManager
    
    class MockConfigManager(ConfigManager):
        def __init__(self) -> None:
            pass
        def get(self, key: str, default: Any = None) -> Any:
            return default

    # Create a mock portfolio manager
    from .portfolio_manager import PortfolioManager
    
    class MockPortfolioManager(PortfolioManager):
        def __init__(self) -> None:
            pass
        def get_current_state(self) -> Dict[str, Any]:
            return {"total_drawdown_pct": 1.5}

    # Use placeholders
    config_manager = MockConfigManager()
    pubsub_manager = MockPubSubManager()
    
    # Create logger service with required parameters and type annotation
    logger_service: LoggerService[Any] = MockLoggerService(
        config_manager=config_manager,
        pubsub_manager=pubsub_manager
    )
    
    # Create a mock portfolio manager for MonitoringService
    portfolio_manager = MockPortfolioManager()
    
    # Create the monitoring service with all required arguments
    monitor = MonitoringService(
        config_manager=config_manager,
        pubsub_manager=pubsub_manager,  # type: ignore # MockPubSubManager inherits from PubSubManager and is compatible
        portfolio_manager=portfolio_manager,
        logger_service=logger_service
    )
    
    app_controller = MainAppController()
    
    # Create CLIService with required logger_service parameter
    cli = CLIService(
        monitoring_service=monitor,
        main_app_controller=app_controller,
        logger_service=logger_service
    )

    await cli.start()

    print("Example main loop running. Waiting for commands or shutdown signal...")
    # Simulate the application running and waiting for shutdown
    # In a real app, this would be the main event loop or a shutdown event
    shutdown_event = asyncio.Event()

    # Example of how shutdown might be triggered elsewhere
    async def trigger_example_shutdown() -> None:
        await asyncio.sleep(60)  # Shutdown after 60 seconds unless stopped by CLI
        if not shutdown_event.is_set():
            print("\nExample shutdown trigger fired.")
            shutdown_event.set()

    # We need a way to replace the app_controller's shutdown to signal our local event
    async def mock_shutdown() -> None:
        print("Mock shutdown called by CLI!")
        shutdown_event.set()

    # Fix method assignment by defining a new method on the class instead of replacing it
    # This addresses the "Cannot assign to a method" error
    setattr(app_controller.__class__, 'shutdown', mock_shutdown)

    # Run example shutdown trigger concurrently
    shutdown_task = asyncio.create_task(trigger_example_shutdown())

    # Wait until shutdown is signaled
    await shutdown_event.wait()

    print("Shutdown signaled. Stopping CLI...")
    await cli.stop()
    # Ensure shutdown trigger task is cancelled if it didn't complete
    shutdown_task.cancel()
    try:
        await shutdown_task
    except asyncio.CancelledError:
        pass
    print("Example finished.")


if __name__ == "__main__":
    # Note: Running this directly might behave unexpectedly depending on the
    # terminal environment and how stdin is handled.
    # It's best tested as part of the integrated application.
    # try:
    #     asyncio.run(example_main())
    # except KeyboardInterrupt:
    #     print("\nKeyboardInterrupt received, exiting.")
    pass
