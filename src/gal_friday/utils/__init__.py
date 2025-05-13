"""Utility functions for the Gal Friday application."""

import logging
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar("T")


def handle_exceptions(
    logger: logging.Logger,
    *,  # Force keyword arguments for all parameters after this
    specific_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    default_return: Any = None,
    message: str = "An error occurred: {error}",
    include_traceback: bool = True,
    source_module: str = "",
    re_raise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Handle exceptions in a standardized way.

    Args:
        logger: The logger to use for error reporting
        specific_exceptions: Tuple of exception types to catch (defaults to Exception)
        default_return: Value to return on exception
        message: Message template for the error
        include_traceback: Whether to include traceback in logs
        source_module: Source module name for logging
        re_raise: Whether to re-raise the exception after logging

    Returns:
        Decorated function
    """
    if specific_exceptions is None:
        specific_exceptions = (Exception,)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except specific_exceptions as exc:
                formatted_message = message.format(error=str(exc))

                # Log with traceback if needed
                if include_traceback:
                    logger.error(formatted_message, source_module=source_module, exc_info=True)
                else:
                    logger.error(formatted_message, source_module=source_module)

                # Re-raise with proper chaining if required
                if re_raise:
                    # Get the original exception type to preserve the type
                    raise exc.__class__(formatted_message) from exc

                return default_return

        return wrapper

    return decorator


async def handle_exceptions_async(
    logger: logging.Logger,
    *,  # Force keyword arguments for all parameters after this
    specific_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    default_return: Any = None,
    message: str = "An error occurred: {error}",
    include_traceback: bool = True,
    source_module: str = "",
    re_raise: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Handle exceptions in a standardized way for async functions.

    Args:
        logger: The logger to use for error reporting
        specific_exceptions: Tuple of exception types to catch (defaults to Exception)
        default_return: Value to return on exception
        message: Message template for the error
        include_traceback: Whether to include traceback in logs
        source_module: Source module name for logging
        re_raise: Whether to re-raise the exception after logging

    Returns:
        Decorated function
    """
    if specific_exceptions is None:
        specific_exceptions = (Exception,)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except specific_exceptions as exc:
                formatted_message = message.format(error=str(exc))

                # Log with traceback if needed
                if include_traceback:
                    logger.error(formatted_message, source_module=source_module, exc_info=True)
                else:
                    logger.error(formatted_message, source_module=source_module)

                # Re-raise with proper chaining if required
                if re_raise:
                    # Get the original exception type to preserve the type
                    raise exc.__class__(formatted_message) from exc

                return default_return

        return wrapper

    return decorator
