"""Provide command-line interface functionality for runtime control of the trading system.

This module implements a CLI service that allows users to interact with and control
the trading system through terminal commands. It handles commands for checking system status,
halting/resuming trading, and gracefully shutting down the application.
"""

# CLI Service Module

import asyncio
import logging  # For example_main logging configuration
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    import typer
    from rich.console import Console
    from rich.table import Table
    from .logger_service import LoggerService
    from .main import GalFridayApp
    from .monitoring_service import MonitoringService
    from .portfolio_manager import PortfolioManager

    # Define MainAppController interface for type hinting
    class MainAppController:
        """Interface for the main application controller."""

        async def stop(self) -> None:
            """Stop the application."""
            raise NotImplementedError

    # Allow GalFridayApp to be used where MainAppController is expected
    MainAppControllerType = Union[MainAppController, "GalFridayApp"]

else:
    # Use stub module when not type checking
    from .typer_stubs import Typer as typer

    # Mock rich modules - these would normally be installed dependencies
    class Console:
        """Mock implementation of rich.console.Console for non-type-checking mode."""

        def print(self, *args, **kwargs):
            """Print to console, simplified version of rich.console.Console.print."""
            print(*args)

    class Table:
        """Mock implementation of rich.table.Table for non-type-checking mode."""

        def __init__(self, *args, **kwargs):
            """Initialize a mock table."""
            pass

        def add_column(self, *args, **kwargs):
            """Add a column to the table."""
            pass

        def add_row(self, *args, **kwargs):
            """Add a row to the table."""
            pass

    # Placeholders for other imports
    class MonitoringService:  # Placeholder
        """Placeholder for MonitoringService when not type checking."""

        def is_halted(self) -> bool:
            """Return whether the system is halted."""
            return False

        async def trigger_halt(self, reason: str, source: str):
            """Trigger a halt of the trading system."""
            print(f"HALT triggered by {source}: {reason}")

        async def trigger_resume(self, source: str):
            """Resume trading after a halt."""
            print(f"RESUME triggered by {source}")

    class MainAppController:  # Placeholder
        """Placeholder for MainAppController when not type checking."""

        async def stop(self) -> None:
            """Stop the application."""
            print("Shutdown requested by CLI.")

    class PortfolioManager:  # Placeholder
        """Placeholder for PortfolioManager when not type checking."""

        def get_current_state(self) -> Dict[str, Any]:
            """Return the current state of the portfolio."""
            return {"total_drawdown_pct": 1.5}

    class LoggerService:
        """Placeholder for LoggerService when not type checking."""

        def info(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            """Log info message."""
            print(f"INFO: {message}")

        def warning(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            """Log warning message."""
            print(f"WARNING: {message}")

        def error(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
            exc_info: Optional[Any] = None,
        ) -> None:
            """Log error message."""
            print(f"ERROR: {message}")

    # Define GalFridayApp as alias to MainAppController for non-type-checking mode
    GalFridayApp = MainAppController

    # Allow GalFridayApp to be used where MainAppController is expected
    MainAppControllerType = Union[MainAppController, GalFridayApp]

# Create Typer application instance
app = typer.Typer(help="Gal-Friday Trading System Control CLI")
console = Console()


class CLIService:
    """Handle Command-Line Interface interactions for runtime control through Typer."""

    def __init__(
        self,
        monitoring_service: "MonitoringService",
        main_app_controller: "MainAppControllerType",
        logger_service: "LoggerService",
        portfolio_manager: Optional["PortfolioManager"] = None,
    ):
        """
        Initialize the CLIService.

        Args
        ----
            monitoring_service: Instance of the MonitoringService.
            main_app_controller: Instance of the main application controller/orchestrator
                                 which must have an async shutdown() method.
            logger_service: The shared logger instance.
            portfolio_manager: Optional portfolio manager for detailed status information.
        """
        self.monitoring_service = monitoring_service
        self.main_app_controller = main_app_controller
        self.logger = logger_service
        self.portfolio_manager = portfolio_manager
        self._running = False
        self._stop_event = asyncio.Event()
        self._input_thread: Optional[threading.Thread] = None
        self.logger.info("CLIService initialized.", source_module=self.__class__.__name__)

    async def start(self) -> None:
        """Start listening for commands on stdin."""
        if self._running:
            self.logger.warning(
                "CLIService already running.", source_module=self.__class__.__name__
            )
            return

        self.logger.info(
            "Starting CLIService input listener...", source_module=self.__class__.__name__
        )
        self._running = True
        self._stop_event.clear()

        try:
            loop = asyncio.get_running_loop()
            # Try using add_reader for POSIX systems
            loop.add_reader(sys.stdin.fileno(), self._handle_input_posix)
            self.logger.info(
                "Using asyncio.add_reader for CLI input.", source_module=self.__class__.__name__
            )
            print("\n--- Gal-Friday CLI Ready (POSIX Mode) ---")
            print("Type a command (e.g., 'status', 'halt', 'stop') or '--help' and press Enter.")
            print("---")
        except (NotImplementedError, AttributeError):
            # Fallback for Windows or other environments where add_reader isn't suitable for stdin
            self.logger.warning(
                "asyncio.add_reader not supported for stdin, falling back to threaded input.",
                source_module=self.__class__.__name__,
            )
            print("\n--- Gal-Friday CLI Ready (Fallback Mode) ---")
            print("Type a command (e.g., 'status', 'halt', 'stop') or '--help' and press Enter.")
            print("(Note: CLI runs in a separate thread)")
            print("---")
            self._input_thread = threading.Thread(target=self._threaded_input_loop, daemon=True)
            self._input_thread.start()

    def _handle_input_posix(self) -> None:
        """Process input received through the POSIX stdin reader."""
        try:
            line = sys.stdin.readline()
            if not line:  # Handle EOF or empty line gracefully
                self.logger.info(
                    "EOF received on stdin, stopping CLI listener.",
                    source_module=self.__class__.__name__,
                )
                asyncio.create_task(self.stop())  # Trigger graceful stop
                return
            command_args = line.strip().split()
            if command_args:
                # Schedule Typer app execution in the event loop
                asyncio.create_task(self._run_typer_command(command_args))
        except Exception as e:
            self.logger.error(
                f"Error reading/parsing CLI input (POSIX): {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    def _threaded_input_loop(self) -> None:
        """Run input loop in a separate thread for Windows compatibility."""
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                # Blocking input call in the thread
                line = input("gal-friday> ")  # Basic prompt
                if line:
                    command_args = line.strip().split()
                    if command_args:
                        # Schedule the async command execution from the thread
                        asyncio.run_coroutine_threadsafe(
                            self._run_typer_command(command_args), loop
                        )
            except EOFError:
                self.logger.info(
                    "EOF received on stdin (threaded), stopping CLI.",
                    source_module=self.__class__.__name__,
                )
                asyncio.run_coroutine_threadsafe(
                    self.main_app_controller.stop(), loop
                )  # Trigger stop
                break  # Exit thread loop
            except Exception as e:
                self.logger.error(
                    f"Error in threaded CLI input loop: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )
                # Avoid busy-looping on persistent errors
                time.sleep(0.5)

    async def _run_typer_command(self, args: List[str]) -> None:
        """Run the Typer app with the given arguments."""
        try:
            # Register instance with global state for command access
            global_cli_instance.set_instance(self)

            # Run the Typer command
            app(args=args, prog_name="gal-friday")
        except SystemExit as e:
            # Typer uses SystemExit for --help, completion, etc. This is normal.
            if e.code != 0:
                self.logger.warning(
                    f"Typer exited with code {e.code}", source_module=self.__class__.__name__
                )
        except Exception as e:
            self.logger.error(
                f"Error executing Typer command '{' '.join(args)}': {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            print("Error executing command. Check logs for details.")

    async def stop(self) -> None:
        """Stop listening for commands on stdin."""
        if not self._running:
            return

        self.logger.info(
            "Stopping CLIService input listener...", source_module=self.__class__.__name__
        )
        self._running = False
        self._stop_event.set()  # Signal thread loop to stop

        # Clean up add_reader if it was used
        try:
            loop = asyncio.get_running_loop()
            if hasattr(loop, "remove_reader"):
                try:
                    loop.remove_reader(sys.stdin.fileno())
                    self.logger.info(
                        "Removed stdin reader.", source_module=self.__class__.__name__
                    )
                except ValueError:  # Handle case where fd was not registered
                    pass
        except Exception as e:
            self.logger.error(
                f"Error removing stdin reader: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

        # Join the input thread if it exists
        if self._input_thread and self._input_thread.is_alive():
            self.logger.info(
                "Waiting for input thread to finish...", source_module=self.__class__.__name__
            )
            # Since it's a daemon thread, it might just exit when the main app exits
            self._input_thread.join(timeout=1.0)  # Wait briefly
            if self._input_thread.is_alive():
                self.logger.warning(
                    "Input thread did not exit cleanly.", source_module=self.__class__.__name__
                )

        print("CLIService stopped.")


# Global state to allow Typer commands to access the CLIService instance
class GlobalCLIInstance:
    """Singleton class to maintain a global reference to the CLI service instance."""

    def __init__(self) -> None:
        """Initialize the GlobalCLIInstance with an empty reference."""
        self._instance: Optional[CLIService] = None

    def set_instance(self, instance: CLIService) -> None:
        """Set the global CLI service instance.

        Args
        ----
            instance: The CLIService instance to store globally.
        """
        self._instance = instance

    def get_instance(self) -> Optional[CLIService]:
        """Retrieve the global CLI service instance.

        Returns
        -------
            The globally stored CLIService instance, or None if not set.
        """
        return self._instance


global_cli_instance = GlobalCLIInstance()


# Define Typer commands
@app.command()
def status() -> None:
    """Display the current operational status of the system."""
    cli = global_cli_instance.get_instance()
    if not cli:
        print("Error: CLI service not initialized")
        return

    halted = cli.monitoring_service.is_halted()

    # Create a rich table for better formatting
    table = Table(title="Gal-Friday System Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("System State", "[bold red]HALTED[/]" if halted else "[bold green]RUNNING[/]")

    # Add portfolio info if available
    if cli.portfolio_manager:
        try:
            portfolio_state = cli.portfolio_manager.get_current_state()
            if portfolio_state:
                # Add portfolio metrics to the table
                if "total_drawdown_pct" in portfolio_state:
                    table.add_row(
                        "Portfolio Drawdown",
                        f"{portfolio_state.get('total_drawdown_pct', 'N/A')}%",
                    )
                # Add more portfolio metrics as needed
        except Exception as e:
            cli.logger.error(
                f"Error fetching portfolio state: {e}", source_module=cli.__class__.__name__
            )

    console.print(table)


@app.command()
def halt(
    reason: str = typer.Option("Manual user command via CLI", help="Reason for halting trading.")
) -> None:
    """Temporarily halt trading activity."""
    cli = global_cli_instance.get_instance()
    if not cli:
        print("Error: CLI service not initialized")
        return

    if cli.monitoring_service.is_halted():
        print("System is already halted.")
        return

    if typer.confirm("Are you sure you want to HALT trading?"):
        print(">>> Issuing HALT command...")
        asyncio.create_task(
            cli.monitoring_service.trigger_halt(reason=reason, source=cli.__class__.__name__)
        )
    else:
        print("Halt command cancelled.")


@app.command()
def resume() -> None:
    """Resume trading activity if halted."""
    cli = global_cli_instance.get_instance()
    if not cli:
        print("Error: CLI service not initialized")
        return

    if not cli.monitoring_service.is_halted():
        print("System is already running.")
        return

    print(">>> Issuing RESUME command...")
    asyncio.create_task(cli.monitoring_service.trigger_resume(source=cli.__class__.__name__))


@app.command(name="stop")
def stop_command() -> None:
    """Initiate a graceful shutdown of the application."""
    cli = global_cli_instance.get_instance()
    if not cli:
        print("Error: CLI service not initialized")
        return

    if typer.confirm("Are you sure you want to STOP the application?"):
        print(">>> Issuing STOP command... Initiating graceful shutdown.")
        asyncio.create_task(cli.main_app_controller.stop())
        cli._stop_event.set()  # Signal input loop to stop
    else:
        print("Stop command cancelled.")


# Example Usage (requires running in a context where stdin is available)
async def example_main() -> None:  # noqa: C901
    """Run an example CLI service for testing purposes.

    This function creates mock services needed for the CLI service and runs a simple
    demonstration of its functionality.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a mock logger service
    from typing import TypeVar

    from .core.pubsub import PubSubManager  # Import moved here to fix undefined name error
    from .logger_service import LoggerService, PoolProtocol

    T = TypeVar("T", bound=PoolProtocol)

    class MockLoggerService(LoggerService[T]):
        def __init__(self, config_manager: "ConfigManager", pubsub_manager: PubSubManager) -> None:
            # We don't need to do anything with these parameters in this mock class
            # They're just required by the interface
            pass

        def info(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            print(f"INFO [{source_module}]: {message}")

        def debug(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            print(f"DEBUG [{source_module}]: {message}")

        def warning(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
        ) -> None:
            print(f"WARN [{source_module}]: {message}")

        def error(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
            exc_info: Optional[Any] = None,
        ) -> None:
            print(f"ERROR [{source_module}]: {message}")

        def critical(
            self,
            message: str,
            source_module: Optional[str] = None,
            context: Optional[Dict[Any, Any]] = None,
            exc_info: Optional[Any] = None,
        ) -> None:
            print(f"CRITICAL [{source_module}]: {message}")

    # Create a mock PubSubManager
    # Import PubSubManager from core.pubsub instead of event_bus for type consistency
    class MockPubSubManager(PubSubManager):  # Inherit from the correct PubSubManager
        def __init__(self, logger: logging.Logger, config_manager: "ConfigManager") -> None:
            super().__init__(logger=logger, config_manager=config_manager)

            # We can still use our custom logger for additional logging
            # self._mock_logger = logger
            # # This might be redundant if super init already stores logger

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
    config_manager_instance = MockConfigManager()
    # MockPubSubManager needs logger and config_manager
    mock_pubsub_logger = logging.getLogger("mock_pubsub_cli")
    pubsub_manager = MockPubSubManager(
        logger=mock_pubsub_logger, config_manager=config_manager_instance
    )

    # Create logger service with required parameters and type annotation
    logger_service: LoggerService[Any] = MockLoggerService(
        config_manager=config_manager_instance, pubsub_manager=pubsub_manager
    )

    # Create a mock portfolio manager for MonitoringService
    portfolio_manager = MockPortfolioManager()

    # Create the monitoring service with all required arguments
    monitor = MonitoringService(
        config_manager=config_manager_instance,
        pubsub_manager=pubsub_manager,
        portfolio_manager=portfolio_manager,
        logger_service=logger_service,
    )

    app_controller = MainAppController()

    # Create CLIService with required logger_service parameter
    cli = CLIService(
        monitoring_service=monitor,
        main_app_controller=app_controller,
        logger_service=logger_service,
        portfolio_manager=portfolio_manager,
    )

    await cli.start()

    print("Example main loop running. Waiting for commands or shutdown signal...")
    # Simulate the application running and waiting for shutdown
    # In a real app, this would be the main event loop or a shutdown event
    shutdown_event = asyncio.Event()

    # Example of how shutdown might be triggered elsewhere
    async def trigger_example_shutdown() -> None:
        # Shutdown after 60 seconds unless stopped by CLI
        await asyncio.sleep(60)
        if not shutdown_event.is_set():
            print("\nExample shutdown trigger fired.")
            shutdown_event.set()

    # We need a way to replace the app_controller's shutdown to signal our
    # local event
    async def mock_shutdown() -> None:
        print("Mock shutdown called by CLI!")
        shutdown_event.set()

    # Fix method assignment by defining a new method on the class instead of replacing it
    # This addresses the "Cannot assign to a method" error
    setattr(app_controller.__class__, "stop", mock_shutdown)

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
