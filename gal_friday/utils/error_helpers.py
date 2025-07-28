"""Enterprise-grade error handling utilities for Gal-Friday.

This module provides helper functions to abstract raise statements from try/except blocks,
addressing Ruff TRY301 linting issues while maintaining production-ready error handling.
"""

import sys
from typing import Any, Protocol


class LoggerProtocol(Protocol):
    """Protocol for logger types."""

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        ...

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        ...


def log_and_reraise(
    logger: LoggerProtocol,
    error_msg: str,
    *,
    source_module: str | None = None,
    context: dict[str, Any] | None = None,
    exc_info: bool = True,
) -> None:
    """Log an error and re-raise the current exception.

    This helper function abstracts the common pattern of logging an error
    and then re-raising the exception, addressing TRY301 linting issues.

    Args:
        logger: Logger instance to use
        error_msg: Error message to log
        source_module: Optional source module name for structured logging
        context: Optional additional context for structured logging
        exc_info: Whether to include exception info in log (default True)

    Raises:
        The current exception from sys.exc_info()

    Example:
        try:
            risky_operation()
        except Exception:
            log_and_reraise(
                self.logger,
                "Failed to perform risky operation",
                source_module=self._source_module,
                context={"operation_id": op_id}
            )
    """
    # Check if we're in an exception handler
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type is None:
        # Not in an exception handler, create a RuntimeError
        raise RuntimeError(f"log_and_reraise called outside exception handler: {error_msg}")

    # Handle different logger interfaces
    if hasattr(logger, "source_module"):
        # LoggerService interface - supports source_module and context directly
        logger.exception(error_msg, source_module=source_module, context=context)
    else:
        # Standard logging.Logger interface - use extra dict
        extra = {}
        if source_module:
            extra["source_module"] = source_module
        if context:
            extra.update(context)

        if extra:
            logger.exception(error_msg, extra=extra, exc_info=exc_info)
        else:
            logger.exception(error_msg, exc_info=exc_info)

    # Re-raise the current exception
    # Since we already checked exc_type is not None, we can safely raise
    if exc_value is not None:
        raise exc_value.with_traceback(exc_traceback)
    # Fallback if exc_value is None for some reason
    raise RuntimeError(error_msg)


def log_and_transform_exception(
    logger: LoggerProtocol,
    error_msg: str,
    target_exception: type[Exception],
    *,
    exception_msg: str | None = None,
    source_module: str | None = None,
    context: dict[str, Any] | None = None,
    include_cause: bool = True,
    exc_info: bool = True,
) -> None:
    """Log an error and raise a new exception, preserving the original cause.

    This helper function abstracts the pattern of catching an exception,
    logging it, and raising a different exception type.

    Args:
        logger: Logger instance to use
        error_msg: Error message for logging
        target_exception: Exception class to raise
        exception_msg: Optional message for the new exception (defaults to error_msg)
        source_module: Optional source module name for structured logging
        context: Optional additional context for structured logging
        include_cause: Whether to chain exceptions with 'from' (default True)
        exc_info: Whether to include exception info in log (default True)

    Raises:
        An instance of target_exception with the provided message

    Example:
        try:
            database_operation()
        except Exception:
            log_and_transform_exception(
                self.logger,
                "Database operation failed",
                DatabaseError,
                exception_msg="Failed to save data",
                source_module=self._source_module
            )
    """
    # Get current exception for chaining
    current_exception = sys.exc_info()[1]

    # Handle different logger interfaces
    if hasattr(logger, "source_module"):
        # LoggerService interface - supports source_module and context directly
        logger.exception(error_msg, source_module=source_module, context=context)
    else:
        # Standard logging.Logger interface - use extra dict
        extra = {}
        if source_module:
            extra["source_module"] = source_module
        if context:
            extra.update(context)

        if extra:
            logger.exception(error_msg, extra=extra, exc_info=exc_info)
        else:
            logger.exception(error_msg, exc_info=exc_info)

    # Determine the message for the new exception
    new_exception_msg = exception_msg or error_msg

    # If we need to include the original error in the message
    if include_cause and current_exception and "{" in new_exception_msg:
        # Format the message with the original exception
        new_exception_msg = new_exception_msg.format(e=current_exception, error=current_exception)

    # Raise the new exception with proper chaining
    if include_cause and current_exception:
        raise target_exception(new_exception_msg) from current_exception
    raise target_exception(new_exception_msg)


def log_error_without_raise(
    logger: LoggerProtocol,
    error_msg: str,
    *,
    source_module: str | None = None,
    context: dict[str, Any] | None = None,
    exc_info: bool = True,
) -> None:
    """Log an error without raising an exception.

    This helper is useful when you want to log an error but continue execution.

    Args:
        logger: Logger instance to use
        error_msg: Error message to log
        source_module: Optional source module name for structured logging
        context: Optional additional context for structured logging
        exc_info: Whether to include exception info in log (default True)
    """
    # Handle different logger interfaces
    if hasattr(logger, "source_module"):
        # LoggerService interface - supports source_module and context directly
        logger.error(error_msg, source_module=source_module, context=context)
    else:
        # Standard logging.Logger interface - use extra dict
        extra = {}
        if source_module:
            extra["source_module"] = source_module
        if context:
            extra.update(context)

        if extra:
            logger.error(error_msg, extra=extra, exc_info=exc_info)
        else:
            logger.error(error_msg, exc_info=exc_info)
