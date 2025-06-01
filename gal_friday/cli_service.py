"""Provide command-line interface functionality for runtime control of the trading system.

This module implements a CLI service that allows users to interact with and control
the trading system through terminal commands. It handles commands for checking system status,
halting/resuming trading, and gracefully shutting down the application.
"""

# CLI Service Module

import asyncio
import logging
import os
import sys
import threading
import time
from collections.abc import Coroutine, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    TypeVar,
    Union,
)
from rich import print as rich_print

# Create TYPE_CHECKING specific imports
if TYPE_CHECKING:
    import typer
    from rich.console import Console
    from rich.table import Table

    from .config_manager import ConfigManager
    from .core.halt_recovery import HaltRecoveryManager
    from .core.pubsub import PubSubManager
    from .logger_service import ExcInfoType, LoggerService
    # Define a protocol for connection pools
    class PoolProtocol(Protocol):
        """Protocol for connection pools."""

    # Type variable for connection pools
    T_Pool = TypeVar("T_Pool", bound=PoolProtocol)
    from .main import GalFridayApp
    from .monitoring_service import MonitoringService
    from .portfolio_manager import PortfolioManager

    # Make Typer available at runtime as well
    Typer = typer.Typer

    T = TypeVar("T", bound=PoolProtocol)

    # Define MainAppController interface for type hinting
    class MainAppController:
        """Interface for the main application controller."""

        async def stop(self) -> None:
            """Stop the application."""
            raise NotImplementedError

    # Allow GalFridayApp to be used where MainAppController is expected
    MainAppControllerType = Union[MainAppController, "GalFridayApp"]
else:
    # Non-type checking imports
    # Import mock implementations
    from .cli_service_mocks import (
        Console,
        LoggerService,
        MainAppController,
        MonitoringService,
        PortfolioManager,
        Table,
    )
    from .typer_stubs import Typer

    # For non-type checking compatibility
    GalFridayApp = MainAppController
    MainAppControllerType = Union[MainAppController, GalFridayApp]
    T = TypeVar("T")

# Create Typer application instance
app = Typer(help="Gal-Friday Trading System Control CLI")
console = Console()


class CLIService:
    """Handle Command-Line Interface interactions for runtime control through Typer."""

    def __init__(
        self,
        monitoring_service: "MonitoringService",
        main_app_controller: "MainAppControllerType",
        logger_service: "LoggerService",
        portfolio_manager: Optional["PortfolioManager"] = None,
        recovery_manager: Optional["HaltRecoveryManager"] = None,
    ) -> None:
        """Initialize the CLIService.

        Args:
        ----
            monitoring_service: Instance of the MonitoringService.
            main_app_controller: Instance of the main application controller/orchestrator
                                 which must have an async shutdown() method.
            logger_service: The shared logger instance.
            portfolio_manager: Optional portfolio manager for detailed status information.
            recovery_manager: Optional recovery manager for HALT recovery procedures.
        """
        self.monitoring_service = monitoring_service
        self.main_app_controller = main_app_controller
        self.logger = logger_service
        self.portfolio_manager = portfolio_manager
        self.recovery_manager = recovery_manager
        self._running = False
        self._stop_event = asyncio.Event()
        self._input_thread: threading.Thread | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self.logger.info("CLIService initialized.", source_module=self.__class__.__name__)

    def launch_background_task(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Create a background task, add it to the tracking set, and set a done callback."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._handle_task_completion)

    def _handle_task_completion(self, task: asyncio.Task) -> None:
        """Handle completion of a background task (log exceptions, remove from set)."""
        self._background_tasks.discard(task)
        try:
            task.result()  # This will raise an exception if the task failed
        except asyncio.CancelledError:
            self.logger.debug(
                "Task %s was cancelled.",
                source_module=self.__class__.__name__,
                context={"task_name": task.get_name()},
            )
        except Exception:
            self.logger.exception(
                "Background task failed",
                source_module=self.__class__.__name__,
                context={"task_name": task.get_name()},
            )

    def signal_input_loop_stop(self) -> None:
        """Signal the input loop to stop."""
        self._stop_event.set()

    async def start(self) -> None:
        """Start listening for commands on stdin."""
        if self._running:
            self.logger.warning(
                "CLIService already running.",
                source_module=self.__class__.__name__,
            )
            return

        self.logger.info(
            "Starting CLIService input listener...",
            source_module=self.__class__.__name__,
        )
        self._running = True
        self._stop_event.clear()

        try:
            if os.name == "posix" and sys.stdin.isatty():
                # Use asyncio event loop on POSIX systems with TTY
                self.logger.info(
                    "CLI Ready (POSIX Mode) - Type commands or '--help'",
                    source_module=self.__class__.__name__,
                )
                loop = asyncio.get_running_loop()
                loop.add_reader(sys.stdin.fileno(), self._handle_input_posix)
            else:
                # Use threading on Windows or non-TTY
                self.logger.info(
                    "CLI Ready (Fallback Mode) - Commands available via threading",
                    source_module=self.__class__.__name__,
                )
                self._input_thread = threading.Thread(
                    target=self._threaded_input_loop,
                    daemon=True,
                )
                self._input_thread.start()
        except (NotImplementedError, AttributeError):
            # Fallback for Windows or other environments where add_reader isn't suitable for stdin
            self.logger.warning(
                "asyncio.add_reader not supported for stdin, falling back to threaded input.",
                source_module=self.__class__.__name__,
            )
            console.print("\n--- Gal-Friday CLI Ready (Fallback Mode) ---")
            console.print("Type a command (e.g., 'status', 'halt', 'stop') or '--help' and press Enter.")
            console.print("(Note: CLI runs in a separate thread)")
            console.print("---")
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
                self.launch_background_task(self.stop())  # Use public method
                return
            command_args = line.strip().split()
            if command_args:
                # Schedule Typer app execution in the event loop
                self.launch_background_task(
                    self._run_typer_command(command_args),
                )
        except Exception:
            self.logger.exception(
                "Error reading/parsing CLI input (POSIX)",
                source_module=self.__class__.__name__,
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
                            self._run_typer_command(command_args),
                            loop,
                        )
            except EOFError:
                self.logger.info(
                    "EOF received on stdin (threaded), stopping CLI.",
                    source_module=self.__class__.__name__,
                )
                asyncio.run_coroutine_threadsafe(
                    self.main_app_controller.stop(),
                    loop,
                )  # Trigger stop
                break  # Exit thread loop
            except Exception:
                self.logger.exception(
                    "Error in threaded CLI input loop",
                    source_module=self.__class__.__name__,
                )
                # Avoid busy-looping on persistent errors
                time.sleep(0.5)

    async def _run_typer_command(self, args: list[str]) -> None:
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
                    "Typer exited with code",
                    source_module=self.__class__.__name__,
                    context={"exit_code": str(e.code)},
                )
        except Exception:
            self.logger.exception(
                "Error executing Typer command",
                source_module=self.__class__.__name__,
                context={"command": " ".join(args)},
            )
            self.logger.error(
                "Command execution failed - check logs for details",
                source_module=self.__class__.__name__,
            )

    async def stop(self) -> None:
        """Stop listening for commands on stdin."""
        if not self._running:
            return

        self.logger.info(
            "Stopping CLIService input listener...",
            source_module=self.__class__.__name__,
        )
        self._running = False
        self._stop_event.set()  # Signal thread loop to stop

        # Cancel any outstanding background tasks
        tasks_to_cancel = list(self._background_tasks)  # Iterate over a copy
        if tasks_to_cancel:
            self.logger.info(
                "Cancelling outstanding background tasks",
                source_module=self.__class__.__name__,
                context={"task_count": len(tasks_to_cancel)},
            )
            for task in tasks_to_cancel:
                task.cancel()
            # Allow time for tasks to process cancellation
            # Their _handle_task_completion callbacks will log issues and remove from set.
            await asyncio.sleep(0.1)

        # Clean up add_reader if it was used
        try:
            loop = asyncio.get_running_loop()
            if hasattr(loop, "remove_reader"):
                try:
                    loop.remove_reader(sys.stdin.fileno())
                    self.logger.info(
                        "Removed stdin reader",
                        source_module=self.__class__.__name__,
                    )
                except ValueError:  # Handle case where fd was not registered
                    pass
        except Exception:
            self.logger.exception(
                "Error removing stdin reader",
                source_module=self.__class__.__name__,
            )

        # Join the input thread if it exists
        if self._input_thread and self._input_thread.is_alive():
            self.logger.info(
                "Waiting for input thread to finish",
                source_module=self.__class__.__name__,
            )
            # Since it's a daemon thread, it might just exit when the main app exits
            self._input_thread.join(timeout=1.0)  # Wait briefly
            if self._input_thread.is_alive():
                self.logger.warning(
                    "Input thread did not exit cleanly",
                    source_module=self.__class__.__name__,
                )

        self.logger.info(
            "CLIService stopped",
            source_module=self.__class__.__name__,
        )
        self._running = False


# Global state to allow Typer commands to access the CLIService instance
class GlobalCLIInstance:
    """Singleton class to maintain a global reference to the CLI service instance."""

    def __init__(self) -> None:
        """Initialize the GlobalCLIInstance with an empty reference."""
        self._instance: CLIService | None = None

    def set_instance(self, instance: CLIService) -> None:
        """Set the global CLI service instance.

        Args:
        ----
            instance: The CLIService instance to store globally.
        """
        self._instance = instance

    def get_instance(self) -> CLIService | None:
        """Retrieve the global CLI service instance.

        Returns:
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
        cli.logger.error(
            "CLI service not initialized",
            source_module="CLI_Command",
        ) if cli else console.print("Error: CLI service not initialized")
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
            if portfolio_state and "total_drawdown_pct" in portfolio_state:
                table.add_row(
                    "Portfolio Drawdown",
                    f"{portfolio_state.get('total_drawdown_pct', 'N/A')}%",
                )
                # Add more portfolio metrics as needed
        except Exception:  # Catching general Exception, specific error message in log
            cli.logger.exception(
                "Error fetching portfolio state for status display",
                source_module=cli.__class__.__name__,
            )

    console.print(table)


@app.command()
def halt(
    reason: str = typer.Option("Manual user command via CLI", help="Reason for halting trading."),
) -> None:
    """Temporarily halt trading activity."""
    cli = global_cli_instance.get_instance()
    if not cli:
        cli.logger.error(
            "CLI service not initialized for halt command",
            source_module="CLI_Command",
        ) if cli else console.print("Error: CLI service not initialized")
        return

    if cli.monitoring_service.is_halted():
        cli.logger.info(
            "System already halted - no action taken",
            source_module="CLI_Command",
        )
        return

    if typer.confirm("Are you sure you want to HALT trading?"):
        cli.logger.info(
            "User confirmed HALT command",
            source_module="CLI_Command",
            context={"reason": reason},
        )
        cli.launch_background_task(
            cli.monitoring_service.trigger_halt(reason=reason, source=cli.__class__.__name__),
        )
    else:
        cli.logger.info(
            "HALT command cancelled by user",
            source_module="CLI_Command",
        )


@app.command()
def resume() -> None:
    """Resume trading activity if halted."""
    cli = global_cli_instance.get_instance()
    if not cli:
        cli.logger.error(
            "CLI service not initialized for resume command",
            source_module="CLI_Command",
        ) if cli else console.print("Error: CLI service not initialized")
        return

    if not cli.monitoring_service.is_halted():
        cli.logger.info(
            "System already running - no action taken",
            source_module="CLI_Command",
        )
        return

    cli.logger.info(
        "Issuing RESUME command",
        source_module="CLI_Command",
    )
    cli.launch_background_task(
        cli.monitoring_service.trigger_resume(source=cli.__class__.__name__),
    )


@app.command(name="stop")
def stop_command() -> None:
    """Initiate a graceful shutdown of the application."""
    cli = global_cli_instance.get_instance()
    if not cli:
        cli.logger.error(
            "CLI service not initialized for stop command",
            source_module="CLI_Command",
        ) if cli else console.print("Error: CLI service not initialized")
        return

    if typer.confirm("Are you sure you want to STOP the application?"):
        cli.logger.info(
            "User confirmed STOP command - initiating graceful shutdown",
            source_module="CLI_Command",
        )
        cli.launch_background_task(cli.main_app_controller.stop())
        cli.signal_input_loop_stop()  # Use public method
    else:
        cli.logger.info(
            "STOP command cancelled by user",
            source_module="CLI_Command",
        )


@app.command()
def recovery_status() -> None:
    """Show HALT recovery checklist status."""
    cli = global_cli_instance.get_instance()
    if not cli:
        console.print("Error: CLI service not initialized")
        return

    if not cli.recovery_manager:
        console.print("Recovery manager not initialized")
        return

    table = Table(title="HALT Recovery Checklist")
    table.add_column("Status", style="cyan")
    table.add_column("Item", style="white")
    table.add_column("Completed By", style="green")

    for item in cli.recovery_manager.checklist:
        status = "✓" if item.is_completed else "✗"
        completed_by = item.completed_by or "-"
        table.add_row(status, item.description, completed_by)

    console.print(table)

    if cli.recovery_manager.is_recovery_complete():
        console.print("\n[green]All recovery items complete. Safe to resume.[/green]")
    else:
        incomplete = len(cli.recovery_manager.get_incomplete_items())
        console.print(f"\n[yellow]{incomplete} items remaining.[/yellow]")


@app.command()
def complete_recovery_item(
    item_id: str,
    completed_by: str = typer.Option(..., prompt="Your name"),
) -> None:
    """Mark a recovery checklist item as complete."""
    cli = global_cli_instance.get_instance()
    if not cli:
        console.print("Error: CLI service not initialized")
        return

    if not cli.recovery_manager:
        console.print("Recovery manager not initialized")
        return

    if cli.recovery_manager.complete_item(item_id, completed_by):
        console.print(f"✓ Item '{item_id}' marked complete by {completed_by}")
    else:
        console.print(f"✗ Item '{item_id}' not found")


# --- Mock Implementations ---


class MockLoggerService(LoggerService):
    """Mock implementation of LoggerService for testing."""

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: Optional["PubSubManager"],
    ) -> None:
        """Initialize the mock logger."""
        # Minimal implementation for mocks; avoid super().__init__ if it has side effects
        # Store args if needed by other methods, or just pass.

    def info(
        self,
        message: str,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Log info message."""
        rich_print(f"INFO [{source_module}]: {message}")

    def debug(
        self,
        message: str,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Log debug message."""
        rich_print(f"DEBUG [{source_module}]: {message}")

    def warning(
        self,
        message: str,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Log warning message."""
        rich_print(f"WARN [{source_module}]: {message}")

    def error(
        self,
        message: str,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
        exc_info: ExcInfoType = None,
    ) -> None:
        """Log error message."""
        rich_print(f"ERROR [{source_module}]: {message}")
        if exc_info and not isinstance(exc_info, bool):
            rich_print(f"Exception: {exc_info}")

    def exception(
        self,
        message: str,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> None:
        """Log exception message with traceback."""
        self.error(message, source_module=source_module, context=context, exc_info=True)

    def critical(
        self,
        message: str,
        source_module: str | None = None,
        context: Mapping[str, object] | None = None,
        exc_info: ExcInfoType = None,
    ) -> None:
        """Log critical message."""
        rich_print(f"CRITICAL [{source_module}]: {message}")
        if exc_info and not isinstance(exc_info, bool):
            rich_print(f"Exception: {exc_info}")


class MockPubSubManager(PubSubManager):
    """Mock implementation of PubSubManager for testing."""

    def __init__(self, logger: "MockLoggerService", config_manager: ConfigManager) -> None:
        """Initialize the mock pubsub manager."""
        self._logger = logger  # type: ignore[assignment]
        self._config_manager = config_manager
        self._running = False


class MockConfigManager(ConfigManager):
    """Mock implementation of ConfigManager for testing."""

    def __init__(self) -> None:
        """Initialize the mock config manager."""

    def get(self, _key: str, default: object | None = None) -> object:
        """Get a configuration value."""
        return default


class MockPortfolioManager(PortfolioManager):
    """Mock implementation of PortfolioManager for testing."""

    def __init__(self) -> None:
        """Initialize the mock portfolio manager."""

    def get_current_state(self) -> dict[str, Any]:
        """Get the current portfolio state."""
        return {
            "total_value": 105000.0,
            "cash": 50000.0,
            "positions": {"BTC/USD": 1.0, "ETH/USD": 10.0},
            "unrealized_pnl": 5000.0,
            "total_drawdown_pct": 3.5,
            "max_drawdown_pct": 8.2,
        }


class MockMonitoringService(MonitoringService):
    """Mock implementation of MonitoringService for testing."""

    def __init__(self) -> None:
        """Initialize the mock monitoring service."""
        self._halted = False
        self._halt_reason = ""

    def is_halted(self) -> bool:
        """Check if the system is halted."""
        return self._halted

    async def trigger_halt(self, reason: str, source: str) -> None:
        """Trigger a system halt."""
        console.print(f"HALTING SYSTEM - Source: {source}, Reason: {reason}")
        self._halted = True
        self._halt_reason = reason

    async def trigger_resume(self, source: str) -> None:
        """Resume the system from a halt."""
        console.print(f"RESUMING SYSTEM - Source: {source}")
        self._halted = False
        self._halt_reason = ""


class MockMainAppController:
    """Mock implementation of MainAppController for testing."""

    async def stop(self) -> None:
        """Stop the application."""
        console.print("SHUTTING DOWN APPLICATION")


def _create_mock_logger(
    config_manager: ConfigManager,
    pubsub_manager: MockPubSubManager | None = None,
) -> MockLoggerService:
    """Create a mock logger for testing."""
    return MockLoggerService(config_manager, pubsub_manager)


def _create_mock_pubsub(
    logger: MockLoggerService,
    config_manager: ConfigManager,
) -> MockPubSubManager:
    """Create a mock PubSub manager."""
    return MockPubSubManager(logger, config_manager)


def _create_mock_config() -> MockConfigManager:
    """Create a mock configuration manager."""
    return MockConfigManager()


def _create_mock_portfolio() -> MockPortfolioManager:
    """Create a mock portfolio manager."""
    return MockPortfolioManager()


def _create_mock_monitoring() -> MockMonitoringService:
    """Create a mock monitoring service."""
    return MockMonitoringService()


def _create_mock_app_controller() -> MockMainAppController:
    """Create a mock application controller."""
    return MockMainAppController()


def _create_mock_services() -> (
    tuple[
        MockLoggerService,
        MockConfigManager,
        MockPubSubManager,
        MockMonitoringService,
        MockMainAppController,
        MockPortfolioManager,
    ]
):
    """Create all mock services needed for testing.

    Returns:
    -------
        A tuple containing mock instances of:
        (logger, config, pubsub, monitoring, app_controller, portfolio)
    """
    # Create mock components in the right order
    config_manager = _create_mock_config()
    logger = _create_mock_logger(config_manager)
    pubsub = _create_mock_pubsub(logger, config_manager)
    monitoring = _create_mock_monitoring()
    app_controller = _create_mock_app_controller()
    portfolio = _create_mock_portfolio()

    return logger, config_manager, pubsub, monitoring, app_controller, portfolio


async def _run_example_cli(cli_service: CLIService, duration: int = 60) -> None:
    """Run the example CLI service for a specified duration.

    Args:
    ----
        cli_service: The CLI service to run
        duration: How long to run the example in seconds
    """
    shutdown_task = asyncio.create_task(_trigger_example_shutdown(cli_service, duration))

    try:
        # Start the CLI service
        await cli_service.start()

        # Wait until the shutdown task completes
        await shutdown_task
    finally:
        # Clean up
        await cli_service.stop()
        console.print("Example CLI service stopped.")


async def _trigger_example_shutdown(cli_service: CLIService, duration: int) -> None:
    """Shutdown after the specified duration unless stopped by CLI.

    Args:
    ----
        cli_service: The CLI service to shut down
        duration: How long to wait before automatic shutdown in seconds
    """
    console.print(f"\nExample will automatically shut down after {duration} seconds.")
    console.print("Use commands like 'status', 'halt', or 'stop' to interact with the system.")
    console.print("Press Ctrl+C to exit early.")

    try:
        await asyncio.sleep(duration)
        console.print(f"\n{duration} seconds elapsed, triggering automatic shutdown...")
        await _mock_shutdown(cli_service)
    except asyncio.CancelledError:
        console.print("Shutdown task cancelled.")


async def _mock_shutdown(cli_service: CLIService) -> None:
    """Perform a mock shutdown of the application.

    Args:
    ----
        cli_service: The CLI service being shut down
    """
    console.print("Example shutting down...")
    await cli_service.main_app_controller.stop()
    await cli_service.stop()


async def example_main() -> None:
    """Run an example CLI service for testing purposes.

    This function creates mock services needed for the CLI service and runs a simple
    demonstration of its functionality.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create mock services - only unpack what we actually use
    mock_services = _create_mock_services()
    mock_logger: LoggerService = mock_services[0]  # type: ignore[assignment]  # Used in CLIService initialization
    mock_monitoring = mock_services[3]  # Used in CLIService initialization
    mock_app_controller: MainAppControllerType = mock_services[4]  # type: ignore[assignment]  # Used in CLIService initialization
    mock_portfolio = mock_services[5]  # Used in CLIService initialization

    # Create the CLI service
    cli_service = CLIService(
        monitoring_service=mock_monitoring,
        main_app_controller=mock_app_controller,
        logger_service=mock_logger,
        portfolio_manager=mock_portfolio,
    )

    # Run the example
    await _run_example_cli(cli_service)


if __name__ == "__main__":
    # Note: Running this directly might behave unexpectedly depending on the
    # terminal environment and how stdin is handled.
    # It's best tested as part of the integrated application.
    pass
