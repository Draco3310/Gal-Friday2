# **Whiteboard: Refactoring Plan for CLIService**

This document outlines a proposed solution to address the findings from the manual code review of src/gal\_friday/cli\_service.py. The core idea is to refactor the service using the Typer library for a more robust, maintainable, and user-friendly command-line interface, while also tackling platform compatibility issues.

## **1\. Core Refactoring: Adopt Typer**

**Problem:** The current implementation manually parses commands from sys.stdin using basic string comparisons (if/elif). This is brittle, lacks features like argument parsing, and requires manual help messages.

**Solution:** Introduce Typer to manage CLI commands, arguments, and help text generation.

**Implementation Sketch:**

\# Simplified conceptual structure within cli\_service.py
import typer
import asyncio
import sys
import threading
from typing import Optional \# ... other imports

\# Typer application instance
app \= typer.Typer(help="Gal-Friday Trading System Control CLI")

class CLIService:
    def \_\_init\_\_(self, monitoring\_service, main\_app\_controller, logger\_service, portfolio\_manager): \# Add portfolio\_manager if needed for status
        self.monitoring\_service \= monitoring\_service
        self.main\_app\_controller \= main\_app\_controller
        self.logger \= logger\_service
        self.portfolio\_manager \= portfolio\_manager \# Store if needed
        self.\_running \= False
        self.\_stop\_event \= asyncio.Event() \# For signaling shutdown to input loop
        self.\_input\_thread: Optional\[threading.Thread\] \= None \# For Windows fallback

        \# \--- Typer Command Definitions \---
        \# Note: These methods need access to 'self' (the CLIService instance).
        \# Typer doesn't directly support methods as commands easily.
        \# A common pattern is to pass the service instance or necessary dependencies
        \# to standalone functions, or use a class-based approach if Typer supports it well.
        \# For simplicity here, we'll assume access to 'self' is handled.
        \# A more robust way is needed in actual implementation (e.g., context passing).

        @app.command()
        def status():
            """Displays the current operational status of the system."""
            \# Fetch detailed status (potentially async)
            \# Use 'rich' for better formatting (see UX section)
            halted \= self.monitoring\_service.is\_halted()
            \# portfolio\_state \= self.portfolio\_manager.get\_current\_state() \# Example
            print(f"System Status: {'HALTED' if halted else 'RUNNING'}")
            \# print(f"Portfolio Drawdown: {portfolio\_state.get('total\_drawdown\_pct', 'N/A')}%") \# Example

        @app.command()
        def halt(
            reason: str \= typer.Option("Manual user command via CLI", help="Reason for halting trading.")
        ):
            """Temporarily halts trading activity."""
            if typer.confirm("Are you sure you want to HALT trading?"):
                print("\>\>\> Issuing HALT command...")
                asyncio.create\_task(
                    self.monitoring\_service.trigger\_halt(reason=reason, source=self.\_\_class\_\_.\_\_name\_\_)
                )
            else:
                print("Halt command cancelled.")

        @app.command()
        def resume():
            """Resumes trading activity if halted."""
            \# Check if already running?
            if not self.monitoring\_service.is\_halted():
                 print("System is already running.")
                 return
            print("\>\>\> Issuing RESUME command...")
            asyncio.create\_task(
                self.monitoring\_service.trigger\_resume(source=self.\_\_class\_\_.\_\_name\_\_)
            )

        @app.command(name="stop", help="Initiates a graceful shutdown of the application.") \# Add alias 'exit'?
        def stop\_command():
            """Alias for the stop command."""
            if typer.confirm("Are you sure you want to STOP the application?"):
                print("\>\>\> Issuing STOP command... Initiating graceful shutdown.")
                asyncio.create\_task(self.main\_app\_controller.stop())
                self.\_stop\_event.set() \# Signal input loop to stop
            else:
                print("Stop command cancelled.")

        \# Add more commands here (e.g., portfolio details, specific trade info)

    \# \--- Service Lifecycle and Input Handling \---

    async def start(self) \-\> None:
        if self.\_running:
            self.logger.warning("CLIService already running.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
            return

        self.logger.info("Starting CLIService input listener...", source\_module=self.\_\_class\_\_.\_\_name\_\_)
        self.\_running \= True
        self.\_stop\_event.clear()

        try:
            loop \= asyncio.get\_running\_loop()
            \# Try using add\_reader for POSIX systems
            loop.add\_reader(sys.stdin.fileno(), self.\_handle\_input\_posix)
            self.logger.info("Using asyncio.add\_reader for CLI input.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
            print("\\n--- Gal-Friday CLI Ready (POSIX Mode) \---")
            print("Type a command (e.g., 'status', 'halt', 'stop') or '--help' and press Enter.")
            print("---")
        except (NotImplementedError, AttributeError):
            \# Fallback for Windows or other environments where add\_reader isn't suitable for stdin
            self.logger.warning(
                "asyncio.add\_reader not supported for stdin, falling back to threaded input.",
                source\_module=self.\_\_class\_\_.\_\_name\_\_
            )
            print("\\n--- Gal-Friday CLI Ready (Fallback Mode) \---")
            print("Type a command (e.g., 'status', 'halt', 'stop') or '--help' and press Enter.")
            print("(Note: CLI runs in a separate thread)")
            print("---")
            self.\_input\_thread \= threading.Thread(target=self.\_threaded\_input\_loop, daemon=True)
            self.\_input\_thread.start()

    def \_handle\_input\_posix(self):
        """Callback for add\_reader (POSIX)."""
        try:
            line \= sys.stdin.readline()
            if not line: \# Handle EOF or empty line gracefully
                self.logger.info("EOF received on stdin, stopping CLI listener.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
                asyncio.create\_task(self.stop()) \# Trigger graceful stop
                return
            command\_args \= line.strip().split()
            if command\_args:
                 \# Schedule Typer app execution in the event loop
                 asyncio.create\_task(self.\_run\_typer\_command(command\_args))
        except Exception as e:
            self.logger.error(f"Error reading/parsing CLI input (POSIX): {e}", source\_module=self.\_\_class\_\_.\_\_name\_\_, exc\_info=True)

    def \_threaded\_input\_loop(self):
        """Input loop running in a separate thread for Windows compatibility."""
        loop \= asyncio.get\_running\_loop()
        while not self.\_stop\_event.is\_set():
            try:
                \# Blocking input call in the thread
                line \= input("gal-friday\> ") \# Basic prompt
                if line:
                    command\_args \= line.strip().split()
                    if command\_args:
                        \# Schedule the async command execution from the thread
                        asyncio.run\_coroutine\_threadsafe(self.\_run\_typer\_command(command\_args), loop)
            except EOFError:
                self.logger.info("EOF received on stdin (threaded), stopping CLI.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
                asyncio.run\_coroutine\_threadsafe(self.main\_app\_controller.stop(), loop) \# Trigger stop
                break \# Exit thread loop
            except Exception as e:
                self.logger.error(f"Error in threaded CLI input loop: {e}", source\_module=self.\_\_class\_\_.\_\_name\_\_, exc\_info=True)
                \# Avoid busy-looping on persistent errors
                time.sleep(0.5)

    async def \_run\_typer\_command(self, args: list\[str\]):
        """Runs the Typer app with the given arguments."""
        try:
            \# Note: Typer's main entry point isn't directly async.
            \# If commands themselves need to be async, Typer needs configuration
            \# or wrappers to handle the event loop correctly.
            \# For now, we assume commands schedule async tasks as needed (like halt/resume).
            \# This might need refinement depending on Typer's async capabilities.
            \# A simple approach: Typer calls synchronous functions which then use asyncio.create\_task.
            app(args=args, prog\_name="gal-friday")
        except SystemExit as e:
             \# Typer uses SystemExit for \--help, completion, etc. This is normal.
             \# We might want to suppress exit code 0 messages or handle specific codes.
             if e.code \!= 0:
                 self.logger.warning(f"Typer exited with code {e.code}", source\_module=self.\_\_class\_\_.\_\_name\_\_)
        except Exception as e:
            self.logger.error(f"Error executing Typer command '{' '.join(args)}': {e}", source\_module=self.\_\_class\_\_.\_\_name\_\_, exc\_info=True)
            \# Optionally print a user-friendly error message
            print(f"Error executing command. Check logs for details.")

    async def stop(self) \-\> None:
        if not self.\_running:
            return

        self.logger.info("Stopping CLIService input listener...", source\_module=self.\_\_class\_\_.\_\_name\_\_)
        self.\_running \= False
        self.\_stop\_event.set() \# Signal thread loop to stop

        \# Clean up add\_reader if it was used
        try:
            loop \= asyncio.get\_running\_loop()
            if hasattr(loop, "remove\_reader"):
                 \# Check if stdin is actually registered before removing
                 \# This requires tracking if add\_reader was successful.
                 \# For simplicity, we try/except, but tracking state is better.
                 try:
                     loop.remove\_reader(sys.stdin.fileno())
                     self.logger.info("Removed stdin reader.", source\_module=self.\_\_class\_\_.\_\_name\_\_)
                 except ValueError: \# Handle case where fd was not registered
                      pass
        except Exception as e:
             self.logger.error(f"Error removing stdin reader: {e}", source\_module=self.\_\_class\_\_.\_\_name\_\_, exc\_info=True)

        \# Join the input thread if it exists
        if self.\_input\_thread and self.\_input\_thread.is\_alive():
            self.logger.info("Waiting for input thread to finish...", source\_module=self.\_\_class\_\_.\_\_name\_\_)
            \# We might need to interrupt the blocking input() call here if possible/necessary
            \# On POSIX, sending a signal might work. On Windows, it's harder.
            \# Since it's a daemon thread, it might just exit abruptly when the main app exits.
            \# A cleaner way involves using select/poll on stdin in the thread with a timeout.
            self.\_input\_thread.join(timeout=1.0) \# Wait briefly
            if self.\_input\_thread.is\_alive():
                 self.logger.warning("Input thread did not exit cleanly.", source\_module=self.\_\_class\_\_.\_\_name\_\_)

        print("CLIService stopped.")

\# \--- Typer needs to be runnable outside the class context \---
\# This part needs careful design. How does Typer access the CLIService instance?
\# Option 1: Global instance (less ideal)
\# Option 2: Pass instance via context (Typer might support this)
\# Option 3: Make commands static/module-level functions that get the instance passed in.

\# Placeholder for how Typer app might be run (needs refinement)
\# if \_\_name\_\_ \== "\_\_main\_\_":
\#    \# This example won't work directly as Typer needs the CLIService context
\#    \# In the real app, CLIService.start() manages running Typer commands.
\#    \# app()

## **2\. Addressing Platform Compatibility (Windows Fallback)**

**Problem:** asyncio.loop.add\_reader is not reliably implemented for sys.stdin on Windows.

**Solution:**

1. **Attempt add\_reader:** Try using it first, as it integrates best with the asyncio loop on POSIX systems.
2. **Fallback to Threading:** If add\_reader fails (raises NotImplementedError or similar), start a separate threading.Thread.
3. **Blocking Input in Thread:** Inside the thread, use the standard blocking input() function to read user commands.
4. **Communicate with Async Loop:** When a command is received in the thread, use asyncio.run\_coroutine\_threadsafe(coro, loop) to schedule the \_run\_typer\_command coroutine (or another async handler) on the main event loop.
5. **Shutdown Signal:** Use an asyncio.Event or threading.Event (self.\_stop\_event in the sketch) to signal the input thread to terminate during graceful shutdown.

**Alternative:** Consider prompt\_toolkit for a more sophisticated, cross-platform asynchronous input handling solution, potentially enabling features like history and autocompletion later. However, this adds a heavier dependency.

## **3\. Addressing Robustness**

**Problem:** Lack of confirmation for critical actions, limited input validation, broad exception handling.

**Solution:**

1. **Confirmation:** Use typer.confirm("Are you sure?") within the halt and stop command functions. (See sketch above).
2. **Input Validation:** Typer automatically handles basic type conversions (e.g., for options/arguments) and reports errors. Add custom validation logic within command functions for more complex requirements.
3. **Error Handling:**
   * Wrap calls within command functions (e.g., self.monitoring\_service.trigger\_halt) in more specific try...except blocks if needed, catching potential application-specific exceptions.
   * The \_run\_typer\_command wrapper catches general errors during command execution and logs them.
   * Typer handles errors related to incorrect command usage (wrong arguments, etc.).

## **4\. Addressing Functionality Gaps**

**Problem:** Basic status, no help, limited command set.

**Solution:**

1. **Enhanced status:** Modify the status command function (decorated with @app.command()) to:
   * Fetch data from MonitoringService (halt status).
   * Fetch relevant data from PortfolioManager (e.g., P\&L, drawdown, positions). This might require adding methods to PortfolioManager or injecting it into CLIService.
   * Format the output clearly (see UX section).
2. **help Command:** Typer automatically generates \--help output based on the application structure, command function docstrings, and argument/option help texts.
3. **New Commands:** Add new functions decorated with @app.command() (e.g., portfolio, show\_trades \<symbol\>). Define necessary arguments and options using Typer's syntax.

## **5\. Addressing User Experience**

**Problem:** Basic text output, no history/aliases (lower priority).

**Solution:**

1. **Output Formatting:** Integrate the rich library. Typer has good integration with rich.
   * Use rich.print() for colored/styled output.
   * Use rich.table.Table to display the enhanced status information in a structured way.
   * Use rich.panel.Panel to frame output sections.

\# Example using rich within a Typer command
from rich.console import Console
from rich.table import Table

console \= Console()

@app.command()
def status():
    """Displays the current operational status of the system."""
    halted \= self.monitoring\_service.is\_halted()
    \# portfolio\_state \= self.portfolio\_manager.get\_current\_state() \# Fetch data

    table \= Table(title="Gal-Friday System Status")
    table.add\_column("Metric", style="cyan")
    table.add\_column("Value", style="magenta")

    table.add\_row("System State", "\[bold red\]HALTED\[/\]" if halted else "\[bold green\]RUNNING\[/\]")
    \# table.add\_row("Portfolio Drawdown", f"{portfolio\_state.get('total\_drawdown\_pct', 'N/A')}%")
    \# Add more rows...

    console.print(table)

2. **Command History/Aliases:** These are typically handled by the user's shell. Implementing them within the app would require a more complex REPL (Read-Eval-Print Loop) using libraries like prompt\_toolkit. Recommend deferring this unless a full REPL becomes a requirement. Typer itself doesn't provide built-in history. Aliases can sometimes be added via Typer's command naming/decorators.

## **Conclusion**

Refactoring CLIService with Typer provides a structured foundation that addresses most of the review findings: improved command handling, automatic help, easier argument parsing, and integration points for better error handling and UX (with rich). The threaded input fallback ensures better cross-platform compatibility. This approach modernizes the CLI and makes it significantly more extensible.
