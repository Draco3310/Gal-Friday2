"""Mock implementations for CLI Service runtime dependencies.

This module provides mock implementations for dependencies required by cli_service.py
when not in type checking mode. These mocks allow the module to run without
requiring all dependencies to be installed.
"""

from typing import Any
from rich import print as rich_print
import builtins


class Console:
    """Mock implementation of rich.console.Console for non-type-checking mode."""

    def print(self, *args: str) -> None:
        """Print to console, simplified version of rich.console.Console.print."""
        builtins.print(*args)  # noqa: T201 - Mocking built-in print for console testing


class Table:
    """Mock implementation of rich.table.Table for non-type-checking mode."""

    MIN_ARGS_FOR_ROW_PRINT = 2 # Not strictly needed anymore with buffer

    def __init__(self, title: str | None = None) -> None:
        """Initialize a mock table."""
        self.title = title
        self.output_buffer: list[str] = []
        if title:
            # This print is for immediate feedback during mock usage, not part of table content
            rich_print(f"TABLE: {title}")

    def add_column(self, title: str, style: str | None = None) -> None:
        """Add a column to the table."""
        self.output_buffer.append(f"COLUMN: {title} (style={style})")

    def add_row(self, *args: str) -> None:
        """Add a row to the table."""
        # Representing the row as a tuple of strings
        self.output_buffer.append(f"ROW: {args}")


class MonitoringService:
    """Placeholder for MonitoringService when not type checking."""

    def is_halted(self) -> bool:
        """Return whether the system is halted."""
        return False

    async def trigger_halt(self, reason: str, source: str) -> None:
        """Trigger a halt of the trading system."""
        rich_print(f"HALT triggered by {source}: {reason}")

    async def trigger_resume(self, source: str) -> None:
        """Resume trading after a halt."""
        rich_print(f"RESUME triggered by {source}")


class MainAppController:
    """Placeholder for MainAppController when not type checking."""

    async def stop(self) -> None:
        """Stop the application."""
        rich_print("Shutdown requested by CLI.")


class PortfolioManager:
    """Placeholder for PortfolioManager when not type checking."""

    def get_current_state(self) -> dict[str, Any]:
        """Return the current state of the portfolio."""
        return {"total_drawdown_pct": 1.5}


class LoggerService:
    """Placeholder for LoggerService when not type checking."""

    def info(
        self,
        message: str,
    ) -> None:
        """Log info message."""
        rich_print(f"INFO: {message}")

    def warning(
        self,
        message: str,
    ) -> None:
        """Log warning message."""
        rich_print(f"WARNING: {message}")

    def error(
        self,
        message: str,
        exc_info: BaseException | None = None,
    ) -> None:
        """Log error message."""
        rich_print(f"ERROR: {message}")
        if exc_info:
            rich_print(f"Exception: {exc_info}")

    def exception(
        self,
        message: str,
        source_module: str | None = None,
        context: dict | None = None,
    ) -> None:
        """Log error message with exception info."""
        rich_print(f"EXCEPTION: {message}")
        if source_module:
            rich_print(f"Source: {source_module}")
        if context:
            rich_print(f"Context: {context}")
        import sys

        rich_print(f"Exception: {sys.exc_info()[1]}")

    def debug(
        self,
        message: str,
    ) -> None:
        """Log debug message."""
        rich_print(f"DEBUG: {message}")

    def critical(
        self,
        message: str,
        exc_info: BaseException | None = None,
    ) -> None:
        """Log critical message."""
        rich_print(f"CRITICAL: {message}")
        if exc_info:
            rich_print(f"Exception: {exc_info}")
