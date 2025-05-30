# CLIService Module Documentation

## Module Overview

The `gal_friday.cli_service.py` module provides a Command-Line Interface (CLI) that allows for runtime interaction with and control of the Gal-Friday trading system. It leverages the `Typer` library for defining commands and parsing arguments, and `Rich` for producing formatted and user-friendly console output. The CLI enables users to perform actions such as checking system status, manually halting or resuming trading operations, initiating a graceful application shutdown, and interacting with the HALT recovery checklist.

## Key Features

-   **Interactive System Control:** Offers a suite of commands for real-time monitoring and control of the application.
    -   Check overall system status (running, halted).
    -   Manually trigger a trading HALT with a specified reason.
    -   Resume trading activity after a HALT.
    -   Initiate a graceful shutdown of the entire application.
-   **HALT Recovery Checklist Interaction:**
    -   Display the status of the HALT recovery checklist.
    -   Mark specific recovery checklist items as complete.
-   **Modern CLI Framework:**
    -   Uses `Typer` for robust command definition, argument parsing, and help message generation.
    -   Employs `Rich` for enhanced console output, including tables and styled text, improving readability.
-   **Asynchronous Input Handling:**
    -   For POSIX-compliant TTY environments (like Linux, macOS terminals), it uses `asyncio.get_event_loop().add_reader()` for efficient, non-blocking stdin processing.
    -   For other environments (e.g., Windows, non-TTY pipes), it falls back to a threaded input loop to prevent blocking the main asyncio event loop.
-   **Service Integration:**
    -   Interacts with `MonitoringService` to query and set the system's halt/resume status.
    -   Communicates with `MainAppController` (via a protocol) to trigger application shutdown.
    -   Optionally integrates with `PortfolioManager` to display detailed portfolio status, like drawdown metrics.
    -   Optionally integrates with `HaltRecoveryManager` to manage and display the HALT recovery checklist.
-   **Background Task Management:** Manages asyncio background tasks that might be spawned by CLI commands (though current commands are mostly synchronous in their core action after confirmation).

## Global State

-   **`GlobalCLIInstance: class`**:
    -   A simple singleton-like class designed to hold a global reference to the active `CLIService` instance.
    -   This allows the Typer command functions (which are typically defined at the module level) to access the methods and state of the `CLIService` instance (e.g., `GlobalCLIInstance.instance.monitoring_service`).

## Class `CLIService`

### Initialization (`__init__`)

-   **Parameters:**
    -   `monitoring_service (MonitoringService)`: An instance of the `MonitoringService` for halt/resume control.
    -   `main_app_controller (MainAppControllerType)`: An object conforming to the `MainAppControllerType` protocol, used to request application shutdown.
    -   `logger_service (LoggerService)`: An instance of `LoggerService` for logging CLI activities.
    -   `portfolio_manager (Optional[PortfolioManager])`: An optional instance of `PortfolioManager` to fetch detailed portfolio status.
    -   `recovery_manager (Optional[HaltRecoveryManager])`: An optional instance of `HaltRecoveryManager` for interacting with the recovery checklist.
-   **Actions:**
    -   Stores references to all provided service instances.
    -   Initializes internal attributes:
        -   `_input_thread (Optional[threading.Thread])`: For the fallback input mechanism.
        -   `_should_stop_input_loop (bool)`: Flag to signal the threaded input loop to terminate.
        -   `_background_tasks (Set[asyncio.Task])`: To keep track of tasks launched by the CLI.
        -   `_loop (asyncio.AbstractEventLoop)`: The asyncio event loop.
    -   Sets the global `CLIService` instance: `GlobalCLIInstance.instance = self`.

### Service Lifecycle & Input Handling

-   **`async start() -> None`**:
    -   Starts the CLI input listener.
    -   Determines the appropriate input handling mechanism:
        -   If POSIX TTY (`os.isatty(sys.stdin.fileno())` and not Windows): Uses `self._loop.add_reader(sys.stdin.fileno(), self._handle_input_posix)` to listen for stdin events asynchronously.
        -   Otherwise (Windows, non-TTY): Starts `_threaded_input_loop()` in a separate daemon thread.
    -   Logs that the CLI service has started.

-   **`async stop() -> None`**:
    -   Stops the CLI input listener and cleans up resources.
    -   If using POSIX TTY reader, calls `self._loop.remove_reader(sys.stdin.fileno())`.
    -   If using the threaded input loop, calls `signal_input_loop_stop()` and joins the thread.
    -   Cancels all tracked background tasks in `_background_tasks`.
    -   Logs that the CLI service is stopping.

-   **`_handle_input_posix() -> None`**: (Synchronous callback for `add_reader`)
    -   Reads a line from `sys.stdin`.
    -   If input is received, it schedules `_run_typer_command` to be executed in the event loop using `asyncio.create_task` or `loop.call_soon_threadsafe` if called from a different thread (though `add_reader` callback is in the loop's thread).
    -   Input is processed as a list of arguments.

-   **`_threaded_input_loop() -> None`**:
    -   The main loop for the fallback threaded input mechanism.
    -   Continuously prompts for user input using `input()`.
    -   Runs until `_should_stop_input_loop` is `True`.
    -   On receiving input, it splits the input string into arguments and uses `self._loop.call_soon_threadsafe(asyncio.create_task, self._run_typer_command(args))` to schedule the Typer command execution on the main asyncio event loop.
    -   Handles `EOFError` and `KeyboardInterrupt` gracefully.

-   **`async _run_typer_command(args: list[str]) -> None`**:
    -   Executes a Typer command by invoking `app.main(args=args, standalone_mode=False)`.
    -   `standalone_mode=False` is crucial as it prevents Typer from calling `sys.exit()` after command execution, allowing the application to continue running.
    -   Catches and logs any exceptions that occur during Typer command processing.

-   **`signal_input_loop_stop() -> None`**:
    -   Sets the `_should_stop_input_loop` flag to `True`, signaling the threaded input loop to terminate.

### Task Management

-   **`launch_background_task(coro: Coroutine) -> None`**:
    -   Creates an asyncio background task from the provided coroutine `coro`.
    -   Adds a completion callback (`_handle_task_completion`) to the task.
    -   Stores the task in the `_background_tasks` set for tracking.

-   **`_handle_task_completion(task: asyncio.Task) -> None`**:
    -   A callback function that is called when a background task managed by `launch_background_task` completes.
    -   Checks if the task raised an exception and logs it if it did.
    -   Removes the completed task from the `_background_tasks` set.

## Typer Commands (`@app.command()`)

The following commands are defined using Typer decorators. They are accessible when the CLI is active.

-   **`status()`**:
    -   Displays the current operational status of the Gal-Friday system.
    -   Indicates whether trading is "RUNNING" or "HALTED" (obtained from `MonitoringService`).
    -   If `PortfolioManager` is available, it also fetches and displays current portfolio drawdown metrics (daily, weekly, total).
    -   Output is formatted using `rich.table.Table`.

-   **`halt(reason: str)`**:
    -   Initiates a trading HALT.
    -   `reason (str)`: A mandatory argument specifying the reason for the halt.
    -   Prompts the user for confirmation ("Are you sure you want to halt trading? [y/N]: ").
    -   If confirmed, calls `monitoring_service.update_system_halt_status(True, reason)`.
    -   Prints a confirmation message.

-   **`resume()`**:
    -   Resumes trading activity if the system is currently HALTED.
    -   Prompts for confirmation.
    -   If confirmed, calls `monitoring_service.update_system_halt_status(False, "Trading resumed via CLI")`.
    -   Prints a confirmation message.

-   **`stop_command()` (exposed as `stop` in the CLI)**:
    -   Initiates a graceful shutdown of the entire Gal-Friday application.
    -   Prompts for confirmation.
    -   If confirmed, calls `main_app_controller.request_shutdown("CLI stop command received")`.
    -   Prints a message indicating shutdown has been initiated.

-   **`recovery_status()`**:
    -   Displays the current status of the HALT recovery checklist.
    -   Requires `HaltRecoveryManager` to be available. If not, prints an error message.
    -   Fetches checklist items from `recovery_manager.get_checklist_status()`.
    -   Displays the items, their status (pending/completed), completed_by, and completion_time in a `rich.table.Table`.

-   **`complete_recovery_item(item_id: str, completed_by: str)`**:
    -   Marks a specific item in the HALT recovery checklist as complete.
    -   `item_id (str)`: The unique identifier of the checklist item to mark as complete.
    -   `completed_by (str)`: The name or identifier of the user completing the item.
    -   Requires `HaltRecoveryManager`.
    -   Prompts for confirmation.
    -   If confirmed, calls `recovery_manager.complete_checklist_item(item_id, completed_by)`.
    -   Prints a success or error message.

## Mock Implementations (for testing)

The `cli_service.py` file also contains mock implementations of the services it depends on (e.g., `MockLoggerService`, `MockMonitoringService`, `MockMainAppController`, `MockPortfolioManager`, `MockHaltRecoveryManager`). These are typically defined within an `if __name__ == "__main__":` block or imported from a `cli_service_mocks.py` file (especially if `TYPE_CHECKING` is false).

An `example_main()` asynchronous function is also usually present, demonstrating how to instantiate and run the `CLIService` with these mock dependencies. This setup is primarily for standalone testing and development of the CLI functionalities without needing to run the entire Gal-Friday application.

## Dependencies

-   **Standard Libraries:**
    -   `asyncio`: For asynchronous programming and event loop management.
    -   `logging`: For standard logging (though often wrapped by `LoggerService`).
    -   `os`: For OS-specific checks like `os.name`.
    -   `sys`: For stdin/stdout operations.
    -   `threading`: For the fallback threaded input loop.
    -   `time`: Potentially for sleeps or timing, though less prominent in async code.
-   **Third-Party Libraries:**
    -   `typer`: The CLI framework for defining commands and parsing arguments.
    -   `rich`: For creating rich text and beautifully formatted output in the console (tables, styled text).
-   **Core Application Modules:**
    -   `gal_friday.monitoring_service.MonitoringService`
    -   `gal_friday.main_app_controller.MainAppControllerType` (protocol/interface)
    -   `gal_friday.logger_service.LoggerService`
    -   `gal_friday.portfolio_manager.PortfolioManager` (Optional)
    -   `gal_friday.halt_recovery_manager.HaltRecoveryManager` (Optional)
-   **Development/Type Checking Utilities:**
    -   `cli_service_mocks` (or similar): Contains mock implementations for testing when `TYPE_CHECKING` is false.
    -   `typer_stubs` (or similar, often `typer.main` for type hints): Stubs or type information for Typer.

## Adherence to Standards

This documentation aims to align with best practices for software documentation, drawing inspiration from principles found in standards such as:

-   **ISO/IEC/IEEE 26512:2018** (Acquirers and suppliers of information for users)
-   **ISO/IEC/IEEE 12207** (Software life cycle processes)
-   **ISO/IEC/IEEE 15288** (System life cycle processes)

The documentation endeavors to provide clear, comprehensive, and accurate information to facilitate the development, use, and maintenance of the `CLIService` module.
