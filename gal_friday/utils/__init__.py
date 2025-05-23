"""Utility functions for the Gal Friday application."""

import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class ExceptionHandlerConfig(Generic[T]):
    """Configuration for the exception handlers."""

    specific_exceptions: tuple[type[Exception], ...] | None = None
    default_return: T | None = None
    message: str = "An error occurred: {error}"
    include_traceback: bool = True
    source_module: str | None = None
    re_raise: bool = False
    # Ensure specific_exceptions is initialized as a tuple for the 'except' clause logic later
    # We'll handle the default (Exception,)
    # logic within the handler functions if it's None after init.
    # Or, we could use a field with a default_factory
    # if we want to initialize it to (Exception,) here.
    # For now, let the handlers manage the default if it remains None.


def handle_exceptions(
    logger: logging.Logger,
    config: ExceptionHandlerConfig[T],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Handle exceptions in a standardized way.

    Args:
    ----
        logger: The logger to use for error reporting
        config: Configuration for exception handling.

    Returns:
    -------
        Decorated function
    """
    effective_exceptions: tuple[type[Exception], ...]
    if (
        config.specific_exceptions
        and isinstance(config.specific_exceptions, tuple)
        and len(config.specific_exceptions) > 0
    ):
        effective_exceptions = config.specific_exceptions
    else:
        effective_exceptions = (Exception,)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: object, **kwargs: object) -> T:
            try:
                return func(*args, **kwargs)
            except effective_exceptions as e:
                formatted_message = config.message.format(error=e)
                module_name = config.source_module or func.__module__

                if config.include_traceback:
                    logger.exception("[%s] %s", module_name, formatted_message)
                else:
                    logger.exception("[%s] %s", module_name, formatted_message, exc_info=False)

                if config.re_raise:
                    raise e.__class__(formatted_message) from e

                return config.default_return  # type: ignore

        return wrapper

    return decorator


def handle_exceptions_async(
    logger: logging.Logger,
    config: ExceptionHandlerConfig[T],
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Handle exceptions in a standardized way for async functions.

    Args:
    ----
        logger: The logger to use for error reporting
        config: Configuration for exception handling.

    Returns:
    -------
        Decorated function
    """
    effective_exceptions: tuple[type[Exception], ...]
    if (
        config.specific_exceptions
        and isinstance(config.specific_exceptions, tuple)
        and len(config.specific_exceptions) > 0
    ):
        effective_exceptions = config.specific_exceptions
    else:
        effective_exceptions = (Exception,)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args: object, **kwargs: object) -> T:
            try:
                return await func(*args, **kwargs)
            except effective_exceptions as e:
                formatted_message = config.message.format(error=e)
                module_name = config.source_module or func.__module__

                if config.include_traceback:
                    logger.exception("[%s] %s", module_name, formatted_message)
                else:
                    logger.exception("[%s] %s", module_name, formatted_message, exc_info=False)

                if config.re_raise:
                    raise e.__class__(formatted_message) from e

                return config.default_return  # type: ignore

        return wrapper

    return decorator
