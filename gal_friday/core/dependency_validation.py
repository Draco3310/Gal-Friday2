"""Dependency validation and type safety utilities for the GalFriday application.

This module provides decorators and type guards to ensure that dependencies
are properly initialized before use, resolving MyPy type safety issues while
maintaining runtime safety.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import TYPE_CHECKING, Any, TypeVar, cast

from gal_friday.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gal_friday.config_manager import ConfigManager
    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.logger_service import LoggerService
    from gal_friday.monitoring_service import MonitoringService

F = TypeVar("F", bound=Callable[..., Any])


class DependencyValidator:
    """Dependency validation utilities for type-safe dependency injection."""

    @staticmethod
    def requires_dependencies(dependencies: Sequence[str]) -> Callable[[F], F]:
        """Decorator to ensure required dependencies are initialized.

        Args:
            dependencies: List of dependency attribute names that must be non-None

        Returns:
            Decorator function that validates dependencies before method execution

        Raises:
            DependencyMissingError: If any required dependency is None
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                # Validate all required dependencies
                for dep_name in dependencies:
                    dep_value = getattr(self, dep_name, None)
                    if dep_value is None:
                        raise DependencyMissingError(
                            component=func.__name__,
                            dependency=dep_name,
                            message=f"Ensure {dep_name} is initialized before calling {func.__name__}",
                        )

                return func(self, *args, **kwargs)

            return cast("F", wrapper)

        return decorator

    @staticmethod
    def assert_config_manager(config: ConfigManager | None) -> ConfigManager:
        """Type guard to assert ConfigManager is not None.

        Args:
            config: ConfigManager instance that may be None

        Returns:
            ConfigManager instance (guaranteed non-None)

        Raises:
            DependencyMissingError: If config is None
        """
        if config is None:
            raise DependencyMissingError(
                component="ConfigManager",
                dependency="config_manager",
                message="Ensure ConfigManager is initialized before use",
            )
        return config

    @staticmethod
    def assert_pubsub_manager(pubsub: PubSubManager | None) -> PubSubManager:
        """Type guard to assert PubSubManager is not None.

        Args:
            pubsub: PubSubManager instance that may be None

        Returns:
            PubSubManager instance (guaranteed non-None)

        Raises:
            DependencyMissingError: If pubsub is None
        """
        if pubsub is None:
            raise DependencyMissingError(
                component="PubSubManager",
                dependency="pubsub_manager",
                message="Ensure PubSubManager is initialized before use",
            )
        return pubsub

    @staticmethod
    def assert_logger_service(logger: LoggerService | None) -> LoggerService:
        """Type guard to assert LoggerService is not None.

        Args:
            logger: LoggerService instance that may be None

        Returns:
            LoggerService instance (guaranteed non-None)

        Raises:
            DependencyMissingError: If logger is None
        """
        if logger is None:
            raise DependencyMissingError(
                component="LoggerService",
                dependency="logger_service",
                message="Ensure LoggerService is initialized before use",
            )
        return logger

    @staticmethod
    def assert_monitoring_service(monitoring: MonitoringService | None) -> MonitoringService:
        """Type guard to assert MonitoringService is not None.

        Args:
            monitoring: MonitoringService instance that may be None

        Returns:
            MonitoringService instance (guaranteed non-None)

        Raises:
            DependencyMissingError: If monitoring is None
        """
        if monitoring is None:
            raise DependencyMissingError(
                component="MonitoringService",
                dependency="monitoring_service",
                message="Ensure MonitoringService is initialized before use",
            )
        return monitoring


# Convenience aliases for common patterns
requires_config = DependencyValidator.requires_dependencies(["config"])
requires_pubsub = DependencyValidator.requires_dependencies(["pubsub"])
requires_logger = DependencyValidator.requires_dependencies(["logger_service"])
requires_monitoring = DependencyValidator.requires_dependencies(["monitoring_service"])

# Common combinations
requires_config_and_pubsub = DependencyValidator.requires_dependencies(["config", "pubsub"])
requires_core_services = DependencyValidator.requires_dependencies(
    [
        "config",
        "pubsub",
        "logger_service",
        "monitoring_service",
    ],
)
