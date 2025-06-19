"""Mock implementations for CLI Service runtime dependencies.

This module provides mock implementations for dependencies required by cli_service.py
when not in type checking mode. These mocks allow the module to run without
requiring all dependencies to be installed.

These mocks are designed to be:
- Realistic in behavior for testing scenarios
- Configurable for different test cases
- Thread-safe where applicable
- Properly typed and documented
"""

import builtins
from collections.abc import Callable, Coroutine
import contextlib
from datetime import UTC, datetime
from decimal import Decimal
import threading
from typing import Any

from rich import print as rich_print


class Console:
    """Mock implementation of rich.console.Console for non-type-checking mode."""

    def __init__(self, *, force_terminal: bool = True, legacy_windows: bool = False) -> None:
        """Initialize console with optional configuration."""
        self.force_terminal = force_terminal
        self.legacy_windows = legacy_windows
        self._output_history: list[str] = []
        self._lock = threading.Lock()

    def print(self, *args: str, style: str | None = None, **kwargs: Any) -> None:
        """Print to console with optional styling."""
        message = " ".join(str(arg) for arg in args)

        # Thread-safe output history tracking
        with self._lock:
            self._output_history.append(message)

        # Apply basic styling if provided
        if style:
            message = f"[{style}]{message}[/{style}]"

        builtins.print(message, **kwargs)  # noqa: T201

    def get_output_history(self) -> list[str]:
        """Get the history of printed messages (useful for testing)."""
        with self._lock:
            return self._output_history.copy()

    def clear_output_history(self) -> None:
        """Clear the output history."""
        with self._lock:
            self._output_history.clear()


class Table:
    """Mock implementation of rich.table.Table for non-type-checking mode."""

    def __init__(self, title: str | None = None, **kwargs: Any) -> None:
        """Initialize a mock table."""
        self.title = title
        self.columns: list[dict[str, Any]] = []
        self.rows: list[list[str]] = []
        self._lock = threading.Lock()

        if title:
            rich_print(f"TABLE: {title}")

    def add_column(
        self,
        header: str,
        *,
        style: str | None = None,
        width: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Add a column to the table."""
        with self._lock:
            self.columns.append({
                "header": header,
                "style": style,
                "width": width,
                **kwargs,
            })

    def add_row(self, *cells: str, **kwargs: Any) -> None:
        """Add a row to the table."""
        with self._lock:
            self.rows.append(list[Any](cells))
            rich_print(f"ROW: {cells}")

    def get_data(self) -> dict[str, Any]:
        """Get table data (useful for testing)."""
        with self._lock:
            return {
                "title": self.title,
                "columns": self.columns.copy(),
                "rows": [row.copy() for row in self.rows],
            }


class MonitoringService:
    """Enhanced mock for MonitoringService with realistic behavior."""

    def __init__(self, *, initial_halt_state: bool = False) -> None:
        """Initialize monitoring service mock.

        Args:
            initial_halt_state: Whether system should start in halted state
        """
        self._is_halted = initial_halt_state
        self._halt_reason = ""
        self._halt_timestamp: datetime | None = None
        self._resume_timestamp: datetime | None = None
        self._halt_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def is_halted(self) -> bool:
        """Return whether the system is halted."""
        with self._lock:
            return self._is_halted

    async def trigger_halt(self, reason: str, source: str) -> None:
        """Trigger a halt of the trading system."""
        with self._lock:
            if not self._is_halted:
                self._is_halted = True
                self._halt_reason = reason
                self._halt_timestamp = datetime.now(UTC)

                # Record halt event
                self._halt_history.append({
                    "action": "halt",
                    "reason": reason,
                    "source": source,
                    "timestamp": self._halt_timestamp,
                })

                rich_print(f"HALT triggered by {source}: {reason}")
            else:
                rich_print(f"System already halted. Current reason: {self._halt_reason}")

    async def trigger_resume(self, source: str) -> None:
        """Resume trading after a halt."""
        with self._lock:
            if self._is_halted:
                self._is_halted = False
                self._resume_timestamp = datetime.now(UTC)

                # Record resume event
                self._halt_history.append({
                    "action": "resume",
                    "source": source,
                    "timestamp": self._resume_timestamp,
                })

                rich_print(f"RESUME triggered by {source}")
            else:
                rich_print("System is not currently halted")

    def get_halt_history(self) -> list[dict[str, Any]]:
        """Get the history of halt/resume events."""
        with self._lock:
            return self._halt_history.copy()

    def get_halt_status(self) -> dict[str, Any]:
        """Get detailed halt status information."""
        with self._lock:
            return {
                "is_halted": self._is_halted,
                "halt_reason": self._halt_reason,
                "halt_timestamp": self._halt_timestamp,
                "resume_timestamp": self._resume_timestamp,
            }


class MainAppController:
    """Enhanced mock for MainAppController with realistic lifecycle management."""

    def __init__(self) -> None:
        """Initialize the main app controller mock."""
        self._running = True
        self._shutdown_callbacks: list[Callable[[], Coroutine[Any, Any, None]]] = []
        self._shutdown_timestamp: datetime | None = None
        self._lock = threading.Lock()

    async def stop(self) -> None:
        """Stop the application."""
        with self._lock:
            if self._running:
                self._running = False
                self._shutdown_timestamp = datetime.now(UTC)
                rich_print("SHUTDOWN requested - Main application stopping")

                # Execute shutdown callbacks
                for callback in self._shutdown_callbacks:
                    try:
                        await callback()
                    except Exception as e:
                        rich_print(f"Error in shutdown callback: {e}")
            else:
                rich_print("Application is already stopped")

    def is_running(self) -> bool:
        """Check if the application is running."""
        with self._lock:
            return self._running

    def add_shutdown_callback(self, callback: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Add a callback to be executed during shutdown."""
        with self._lock:
            self._shutdown_callbacks.append(callback)

    def get_status(self) -> dict[str, Any]:
        """Get application status information."""
        with self._lock:
            return {
                "running": self._running,
                "shutdown_timestamp": self._shutdown_timestamp,
            }


class PortfolioManager:
    """Enhanced mock for PortfolioManager with configurable portfolio data."""

    def __init__(self, *, initial_state: dict[str, Any] | None = None) -> None:
        """Initialize portfolio manager mock.

        Args:
            initial_state: Initial portfolio state data
        """
        self._default_state = {
            "total_value": Decimal("105000.00"),
            "cash": Decimal("50000.00"),
            "positions": {"BTC/USD": Decimal("1.0"), "ETH/USD": Decimal("10.0")},
            "unrealized_pnl": Decimal("5000.00"),
            "total_drawdown_pct": Decimal("1.5"),
            "max_drawdown_pct": Decimal("8.2"),
            "winning_trades": 15,
            "losing_trades": 5,
            "total_trades": 20,
            "last_updated": datetime.now(UTC),
        }

        if initial_state:
            self._default_state.update(initial_state)

        self._current_state = self._default_state.copy()
        self._lock = threading.Lock()

    def get_current_state(self) -> dict[str, Any]:
        """Return the current state of the portfolio."""
        with self._lock:
            # Update timestamp on each call
            self._current_state["last_updated"] = datetime.now(UTC)
            return self._current_state.copy()

    def update_state(self, updates: dict[str, Any]) -> None:
        """Update portfolio state (useful for testing scenarios)."""
        with self._lock:
            self._current_state.update(updates)
            self._current_state["last_updated"] = datetime.now(UTC)

    def reset_to_default(self) -> None:
        """Reset portfolio to default state."""
        with self._lock:
            self._current_state = self._default_state.copy()

    def simulate_trade_result(self, profit_loss: Decimal, symbol: str = "BTC/USD") -> None:
        """Simulate a trade result for testing."""
        with self._lock:
            current_pnl_raw = self._current_state.get("unrealized_pnl", Decimal(0))
            current_pnl = current_pnl_raw if isinstance(current_pnl_raw, Decimal) else Decimal(str(current_pnl_raw))
            self._current_state["unrealized_pnl"] = current_pnl + profit_loss

            current_total_raw = self._current_state.get("total_trades", 0)
            current_total = current_total_raw if isinstance(current_total_raw, int) else int(str(current_total_raw))
            self._current_state["total_trades"] = current_total + 1

            if profit_loss > 0:
                current_winning_raw = self._current_state.get("winning_trades", 0)
                current_winning = (
                    current_winning_raw if isinstance(current_winning_raw, int)
                    else int(str(current_winning_raw))
                )
                self._current_state["winning_trades"] = current_winning + 1
            else:
                current_losing_raw = self._current_state.get("losing_trades", 0)
                current_losing = (
                    current_losing_raw if isinstance(current_losing_raw, int)
                    else int(str(current_losing_raw))
                )
                self._current_state["losing_trades"] = current_losing + 1


class LoggerService:
    """Enhanced mock for LoggerService with configurable logging behavior."""

    def __init__(self, *, log_level: str = "INFO", capture_logs: bool = True) -> None:
        """Initialize logger service mock.

        Args:
            log_level: Minimum log level to process
            capture_logs: Whether to capture logs for testing
        """
        self.log_level = log_level
        self.capture_logs = capture_logs
        self._captured_logs: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Log level hierarchy
        self._log_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }

    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on level."""
        return self._log_levels.get(level, 0) >= self._log_levels.get(self.log_level, 1)

    def _log(
        self,
        level: str,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None,
        exc_info: BaseException | None = None,
    ) -> None:
        """Internal logging method."""
        if not self._should_log(level):
            return

        # Format message with args
        try:
            formatted_message = message % args if args else message
        except (TypeError, ValueError):
            formatted_message = f"{message} {args}" if args else message

        # Create log entry
        log_entry = {
            "level": level,
            "message": formatted_message,
            "source_module": source_module,
            "context": context,
            "exc_info": exc_info,
            "timestamp": datetime.now(UTC),
        }

        # Capture logs if enabled
        if self.capture_logs:
            with self._lock:
                self._captured_logs.append(log_entry)

        # Print to console
        prefix = f"[{source_module}]" if source_module else ""
        rich_print(f"{level} {prefix}: {formatted_message}")

        if exc_info:
            rich_print(f"Exception: {exc_info}")

        if context:
            rich_print(f"Context: {context}")

    def info(
        self,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None) -> None:
        """Log info message."""
        self._log("INFO", message, *args, source_module=source_module, context=context)

    def warning(
        self,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None) -> None:
        """Log warning message."""
        self._log("WARNING", message, *args, source_module=source_module, context=context)

    def error(
        self,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None,
        exc_info: BaseException | None = None) -> None:
        """Log error message."""
        self._log("ERROR", message, *args, source_module=source_module, context=context, exc_info=exc_info)

    def exception(
        self,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None) -> None:
        """Log error message with exception info."""
        import sys
        exc_info = sys.exc_info()[1]
        self._log("EXCEPTION", message, *args, source_module=source_module, context=context, exc_info=exc_info)

    def debug(
        self,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None) -> None:
        """Log debug message."""
        self._log("DEBUG", message, *args, source_module=source_module, context=context)

    def critical(
        self,
        message: str,
        *args: Any,
        source_module: str | None = None,
        context: dict[str, Any] | None = None,
        exc_info: BaseException | None = None) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, *args, source_module=source_module, context=context, exc_info=exc_info)

    def get_captured_logs(self, level: str | None = None) -> list[dict[str, Any]]:
        """Get captured logs, optionally filtered by level."""
        with self._lock:
            if level:
                return [log for log in self._captured_logs if log["level"] == level]
            return self._captured_logs.copy()

    def clear_captured_logs(self) -> None:
        """Clear captured logs."""
        with self._lock:
            self._captured_logs.clear()

    def set_log_level(self, level: str) -> None:
        """Set the minimum log level."""
        if level in self._log_levels:
            self.log_level = level
        else:
            self.warning(f"Invalid log level: {level}. Using INFO instead.")
            self.log_level = "INFO"


class ConfigManager:
    """Enhanced mock for ConfigManager with realistic configuration behavior."""

    def __init__(self, *, config_data: dict[str, Any] | None = None) -> None:
        """Initialize config manager mock.

        Args:
            config_data: Initial configuration data
        """
        self._config = config_data or {
            "cli": {
                "port": 8080,
                "host": "localhost",
                "timeout": 30,
            },
            "portfolio": {
                "valuation_currency": "USD",
                "max_positions": 10,
            },
            "risk_manager": {
                "max_drawdown_pct": 10.0,
                "position_size_pct": 2.0,
            },
            "monitoring": {
                "check_interval_seconds": 60,
                "halt_on_errors": True,
            },
        }
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        with self._lock:
            try:
                keys = key.split(".")
                value = self._config
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value (useful for testing)."""
        with self._lock:
            keys = key.split(".")
            config = self._config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value

    def get_all(self) -> dict[str, Any]:
        """Get all configuration data."""
        with self._lock:
            return self._config.copy()

    def update(self, updates: dict[str, Any]) -> None:
        """Update configuration with new values."""
        with self._lock:
            self._config.update(updates)


class PubSubManager:
    """Enhanced mock for PubSubManager with realistic pub/sub behavior."""

    def __init__(self) -> None:
        """Initialize pub/sub manager mock."""
        self._subscribers: dict[str, list[Callable[..., Any]]] = {}
        self._published_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    async def publish(self, event: Any) -> None:
        """Publish an event to subscribers."""
        event_type = getattr(event, "event_type", str(type(event).__name__))

        # Record published event
        with self._lock:
            self._published_events.append({
                "event_type": event_type,
                "event": event,
                "timestamp": datetime.now(UTC),
            })

            subscribers = self._subscribers.get(event_type, [])

        # Call subscribers
        for handler in subscribers:
            try:
                if callable(handler):
                    result = handler(event)
                    if hasattr(result, "__await__"):
                        await result
            except Exception as e:
                rich_print(f"Error in event handler: {e}")

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        """Subscribe to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        """Unsubscribe from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                with contextlib.suppress(ValueError):
                    self._subscribers[event_type].remove(handler)

    def get_published_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get published events, optionally filtered by type."""
        with self._lock:
            if event_type:
                return [e for e in self._published_events if e["event_type"] == event_type]
            return self._published_events.copy()

    def clear_published_events(self) -> None:
        """Clear published events history."""
        with self._lock:
            self._published_events.clear()

    def get_subscribers(self, event_type: str) -> list[Callable[..., Any]]:
        """Get subscribers for an event type."""
        with self._lock:
            return self._subscribers.get(event_type, []).copy()


class HaltRecoveryManager:
    """Enhanced mock for HaltRecoveryManager with realistic recovery workflow."""

    def __init__(self) -> None:
        """Initialize halt recovery manager mock."""
        self._recovery_items: dict[str, dict[str, Any]] = {}
        self._completed_items: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_recovery_status(self) -> dict[str, Any]:
        """Get current recovery status."""
        with self._lock:
            total_items = len(self._recovery_items)
            completed_items = len(self._completed_items)

            return {
                "total_items": total_items,
                "completed_items": completed_items,
                "pending_items": total_items - completed_items,
                "recovery_items": list[Any](self._recovery_items.values()),
                "completed_items": list[Any](self._completed_items.values()),
            }

    def complete_item(self, item_id: str, completed_by: str) -> bool:
        """Mark a recovery item as completed."""
        with self._lock:
            if item_id in self._recovery_items:
                item = self._recovery_items.pop(item_id)
                item.update({
                    "completed_by": completed_by,
                    "completed_at": datetime.now(UTC),
                })
                self._completed_items[item_id] = item
                return True
            return False

    def add_recovery_item(
        self,
        item_id: str,
        description: str,
        priority: str = "medium",
    ) -> None:
        """Add a recovery item (useful for testing)."""
        with self._lock:
            self._recovery_items[item_id] = {
                "id": item_id,
                "description": description,
                "priority": priority,
                "created_at": datetime.now(UTC),
            }

    def reset_recovery_state(self) -> None:
        """Reset recovery state for testing."""
        with self._lock:
            self._recovery_items.clear()
            self._completed_items.clear()
