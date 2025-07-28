"""Service initialization builder for type-safe service creation.

This module provides a builder pattern for creating and initializing services
with proper type safety, dependency injection, and error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from gal_friday.core.type_guards import DependencyTypeGuards
from gal_friday.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gal_friday.cli_service import CLIService
    from gal_friday.config_manager import ConfigManager
    from gal_friday.core.service_container import ServiceContainer
    from gal_friday.logger_service import LoggerService
    from gal_friday.monitoring_service import MonitoringService

T = TypeVar("T")


class ServiceInitializationError(Exception):
    """Raised when service initialization fails."""

    def __init__(self, service_name: str, stage: str, message: str, cause: Exception | None = None) -> None:
        """Initialize service initialization error."""
        super().__init__(f"Service {service_name} failed at {stage}: {message}")
        self.service_name = service_name
        self.stage = stage
        self.cause = cause


class ServiceInitializationStage(Enum):
    """Stages of service initialization."""

    DEPENDENCY_RESOLUTION = "dependency_resolution"
    INSTANTIATION = "instantiation"
    CONFIGURATION = "configuration"
    STARTUP = "startup"
    VALIDATION = "validation"
    READY = "ready"


@dataclass
class ServiceInitializationContext(Generic[T]):
    """Context for service initialization."""

    service_name: str
    service_type: type[T]
    container: ServiceContainer
    config_manager: ConfigManager
    stage: ServiceInitializationStage = ServiceInitializationStage.DEPENDENCY_RESOLUTION
    dependencies: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None

    def update_stage(self, stage: ServiceInitializationStage) -> None:
        """Update the current initialization stage."""
        self.stage = stage


class ServiceInitializationBuilder(ABC, Generic[T]):
    """Abstract base class for service initialization builders."""

    def __init__(self, service_name: str, service_type: type[T], container: ServiceContainer) -> None:
        """Initialize service builder."""
        self.service_name = service_name
        self.service_type = service_type
        self.container = container
        self.context = ServiceInitializationContext(
            service_name=service_name,
            service_type=service_type,
            container=container,
            config_manager=self._get_config_manager(),
        )

    def _get_config_manager(self) -> ConfigManager:
        """Get config manager from container."""
        # Access the config manager through public interface
        config_manager = self.container.get_config_manager()
        if config_manager is not None:
            return DependencyTypeGuards.validate_config_dependency(config_manager)

        raise DependencyMissingError(
            component=self.service_name,
            dependency="ConfigManager",
            message="ConfigManager not available in container",
        )

    def _raise_validation_error(self, validation_errors: list[str]) -> None:
        """Raise ServiceInitializationError for validation failures."""
        raise ServiceInitializationError(
            self.service_name,
            "validation",
            f"Service validation failed: {validation_errors}",
        )

    @abstractmethod
    async def resolve_dependencies(self) -> dict[str, Any]:
        """Resolve service dependencies."""

    @abstractmethod
    async def create_instance(self, dependencies: dict[str, Any]) -> T:
        """Create service instance."""

    @abstractmethod
    async def configure_service(self, instance: T) -> T:
        """Configure the service instance."""

    @abstractmethod
    async def start_service(self, instance: T) -> T:
        """Start the service."""

    @abstractmethod
    async def validate_service(self, instance: T) -> list[str]:
        """Validate the service."""

    async def build(self) -> T:
        """Build and initialize the service."""
        try:
            # Stage 1: Dependency resolution
            self.context.update_stage(ServiceInitializationStage.DEPENDENCY_RESOLUTION)
            dependencies = await self.resolve_dependencies()
            self.context.dependencies = dependencies

            # Stage 2: Instantiation
            self.context.update_stage(ServiceInitializationStage.INSTANTIATION)
            instance = await self.create_instance(dependencies)

            # Stage 3: Configuration
            self.context.update_stage(ServiceInitializationStage.CONFIGURATION)
            configured_instance = await self.configure_service(instance)

            # Stage 4: Startup
            self.context.update_stage(ServiceInitializationStage.STARTUP)
            started_instance = await self.start_service(configured_instance)

            # Stage 5: Validation
            self.context.update_stage(ServiceInitializationStage.VALIDATION)
            validation_errors = await self.validate_service(started_instance)

            if validation_errors:
                self._raise_validation_error(validation_errors)

            # Stage 6: Ready
            self.context.update_stage(ServiceInitializationStage.READY)

        except Exception as e:
            raise ServiceInitializationError(
                self.service_name,
                self.context.stage.value,
                str(e),
                e,
            ) from e
        else:
            return started_instance


class LoggerServiceBuilder(ServiceInitializationBuilder["LoggerService"]):
    """Builder for LoggerService initialization."""

    async def resolve_dependencies(self) -> dict[str, Any]:
        """Resolve LoggerService dependencies."""
        dependencies: dict[str, Any] = {}

        # Resolve ConfigManager
        dependencies["config_manager"] = self.context.config_manager

        # Resolve PubSubManager
        try:
            pubsub_type = self.container.get_type("gal_friday.core.pubsub", "PubSubManager")
            dependencies["pubsub_manager"] = await self.container.resolve(pubsub_type)
        except Exception as e:
            raise DependencyMissingError(
                component="LoggerService",
                dependency="PubSubManager",
                message=f"Failed to resolve PubSubManager: {e}",
            ) from e

        # Optional: Database session maker
        # This might not be available yet, so we'll handle it gracefully
        try:
            # We'll set this to None for now, it can be set later
            dependencies["db_session_maker"] = None
        except Exception:
            dependencies["db_session_maker"] = None

        return dependencies

    async def create_instance(self, dependencies: dict[str, Any]) -> LoggerService:
        """Create LoggerService instance."""
        try:
            logger_service_type = self.container.get_type("gal_friday.logger_service", "LoggerService")
            instance = logger_service_type(
                config_manager=dependencies["config_manager"],
                pubsub_manager=dependencies["pubsub_manager"],
                db_session_maker=dependencies["db_session_maker"],
            )
            return cast("LoggerService", instance)
        except Exception as e:
            raise ServiceInitializationError(
                self.service_name,
                "instantiation",
                f"Failed to create LoggerService: {e}",
            ) from e

    async def configure_service(self, instance: LoggerService) -> LoggerService:
        """Configure LoggerService."""
        # LoggerService configuration happens during instantiation
        return instance

    async def start_service(self, instance: LoggerService) -> LoggerService:
        """Start LoggerService."""
        if hasattr(instance, "start"):
            await instance.start()
        return instance

    async def validate_service(self, instance: LoggerService) -> list[str]:
        """Validate LoggerService."""
        errors = []

        # Validate that it has required methods
        required_methods = ["info", "error", "debug", "warning"]
        errors.extend(
            [
                f"LoggerService missing required method: {method}"
                for method in required_methods
                if not hasattr(instance, method) or not callable(getattr(instance, method))
            ],
        )

        return errors


class MonitoringServiceBuilder(ServiceInitializationBuilder["MonitoringService"]):
    """Builder for MonitoringService initialization."""

    async def resolve_dependencies(self) -> dict[str, Any]:
        """Resolve MonitoringService dependencies."""
        dependencies = {}

        # Resolve ConfigManager
        dependencies["config_manager"] = self.context.config_manager

        # Resolve LoggerService
        try:
            logger_type = self.container.get_type("gal_friday.logger_service", "LoggerService")
            dependencies["logger_service"] = await self.container.resolve(logger_type)
        except Exception as e:
            raise DependencyMissingError(
                component="MonitoringService",
                dependency="LoggerService",
                message=f"Failed to resolve LoggerService: {e}",
            ) from e

        return dependencies

    async def create_instance(self, dependencies: dict[str, Any]) -> MonitoringService:
        """Create MonitoringService instance."""
        try:
            monitoring_service_type = self.container.get_type("gal_friday.monitoring_service", "MonitoringService")
            instance = monitoring_service_type(
                config_manager=dependencies["config_manager"],
                logger_service=dependencies["logger_service"],
            )
            return cast("MonitoringService", instance)
        except Exception as e:
            raise ServiceInitializationError(
                self.service_name,
                "instantiation",
                f"Failed to create MonitoringService: {e}",
            ) from e

    async def configure_service(self, instance: MonitoringService) -> MonitoringService:
        """Configure MonitoringService."""
        # MonitoringService configuration happens during instantiation
        return instance

    async def start_service(self, instance: MonitoringService) -> MonitoringService:
        """Start MonitoringService."""
        if hasattr(instance, "start"):
            await instance.start()
        return instance

    async def validate_service(self, instance: MonitoringService) -> list[str]:
        """Validate MonitoringService."""
        errors = []

        # Validate that it has required methods
        required_methods = ["start", "stop"]
        errors.extend(
            [
                f"MonitoringService missing required method: {method}"
                for method in required_methods
                if not hasattr(instance, method) or not callable(getattr(instance, method))
            ],
        )

        return errors


class CLIServiceBuilder(ServiceInitializationBuilder["CLIService"]):
    """Builder for CLIService initialization."""

    async def resolve_dependencies(self) -> dict[str, Any]:
        """Resolve CLIService dependencies."""
        dependencies: dict[str, Any] = {}

        # Resolve MonitoringService
        try:
            monitoring_type = self.container.get_type("gal_friday.monitoring_service", "MonitoringService")
            dependencies["monitoring_service"] = await self.container.resolve(monitoring_type)
        except Exception as e:
            raise DependencyMissingError(
                component="CLIService",
                dependency="MonitoringService",
                message=f"Failed to resolve MonitoringService: {e}",
            ) from e

        # Resolve LoggerService
        try:
            logger_type = self.container.get_type("gal_friday.logger_service", "LoggerService")
            dependencies["logger_service"] = await self.container.resolve(logger_type)
        except Exception as e:
            raise DependencyMissingError(
                component="CLIService",
                dependency="LoggerService",
                message=f"Failed to resolve LoggerService: {e}",
            ) from e

        # Note: PortfolioManager and main_app_controller will need to be resolved separately
        # as they may not be available in the container yet

        return dependencies

    async def create_instance(self, dependencies: dict[str, Any]) -> CLIService:
        """Create CLIService instance."""
        try:
            cli_service_type = self.container.get_type("gal_friday.cli_service", "CLIService")
            # This will need to be modified to handle the missing dependencies
            # For now, we'll create a partial instance
            instance = cli_service_type(
                monitoring_service=dependencies["monitoring_service"],
                logger_service=dependencies["logger_service"],
                main_app_controller=None,  # Will be set later
                portfolio_manager=None,  # Will be set later
            )
            return cast("CLIService", instance)
        except Exception as e:
            raise ServiceInitializationError(
                self.service_name,
                "instantiation",
                f"Failed to create CLIService: {e}",
            ) from e

    async def configure_service(self, instance: CLIService) -> CLIService:
        """Configure CLIService."""
        # CLIService configuration happens during instantiation
        return instance

    async def start_service(self, instance: CLIService) -> CLIService:
        """Start CLIService."""
        if hasattr(instance, "start"):
            await instance.start()
        return instance

    async def validate_service(self, instance: CLIService) -> list[str]:
        """Validate CLIService."""
        errors = []

        # Validate that it has required methods
        required_methods = ["start", "stop"]
        errors.extend(
            [
                f"CLIService missing required method: {method}"
                for method in required_methods
                if not hasattr(instance, method) or not callable(getattr(instance, method))
            ],
        )

        return errors


class ServiceInitializationDirector:
    """Director for coordinating service initialization."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize service initialization director."""
        self.container = container
        self._builders: dict[str, ServiceInitializationBuilder[Any]] = {}
        self._instances: dict[str, Any] = {}
        self._initialization_order = [
            "LoggerService",
            "MonitoringService",
            "CLIService",
        ]

    def register_builder(self, name: str, builder: ServiceInitializationBuilder[Any]) -> None:
        """Register a service builder."""
        self._builders[name] = builder

    def register_standard_builders(self) -> None:
        """Register standard service builders."""
        logger_service_type = self.container.get_type("gal_friday.logger_service", "LoggerService")
        monitoring_service_type = self.container.get_type("gal_friday.monitoring_service", "MonitoringService")
        cli_service_type = self.container.get_type("gal_friday.cli_service", "CLIService")

        self.register_builder(
            "LoggerService",
            LoggerServiceBuilder("LoggerService", logger_service_type, self.container),
        )
        self.register_builder(
            "MonitoringService",
            MonitoringServiceBuilder("MonitoringService", monitoring_service_type, self.container),
        )
        self.register_builder(
            "CLIService",
            CLIServiceBuilder("CLIService", cast("type[CLIService]", cli_service_type), self.container),
        )

    async def initialize_service(self, service_name: str) -> Any:
        """Initialize a specific service."""
        if service_name not in self._builders:
            raise ServiceInitializationError(
                service_name,
                "builder_resolution",
                f"No builder registered for service: {service_name}",
            )

        builder = self._builders[service_name]
        instance = await builder.build()
        self._instances[service_name] = instance

        return instance

    async def initialize_all_services(self) -> dict[str, Any]:
        """Initialize all registered services in order."""
        initialized_services = {}

        for service_name in self._initialization_order:
            if service_name in self._builders:
                try:
                    instance = await self.initialize_service(service_name)
                    initialized_services[service_name] = instance
                except Exception:
                    # Log error but continue with other services
                    import logging  # noqa: PLC0415 - architectural: fallback logging when service initialization fails and logger_service may be unavailable

                    logging.getLogger(__name__).exception(f"Failed to initialize service {service_name}")
                    continue

        return initialized_services

    def get_instance(self, service_name: str) -> Any:
        """Get an initialized service instance."""
        if service_name not in self._instances:
            raise ServiceInitializationError(
                service_name,
                "retrieval",
                f"Service {service_name} not initialized",
            )

        return self._instances[service_name]

    @asynccontextmanager
    async def service_lifecycle(self) -> AsyncIterator[dict[str, Any]]:
        """Context manager for service lifecycle management."""
        services = {}
        try:
            services = await self.initialize_all_services()
            yield services
        finally:
            # Clean up services in reverse order
            for service_name in reversed(self._initialization_order):
                if service_name in services:
                    instance = services[service_name]
                    if hasattr(instance, "stop"):
                        with suppress(Exception):
                            await instance.stop()


# Factory functions for common builders
def create_logger_service_builder(container: ServiceContainer) -> LoggerServiceBuilder:
    """Create a LoggerService builder."""
    logger_service_type = container.get_type("gal_friday.logger_service", "LoggerService")
    return LoggerServiceBuilder("LoggerService", logger_service_type, container)


def create_monitoring_service_builder(container: ServiceContainer) -> MonitoringServiceBuilder:
    """Create a MonitoringService builder."""
    monitoring_service_type = container.get_type("gal_friday.monitoring_service", "MonitoringService")
    return MonitoringServiceBuilder("MonitoringService", monitoring_service_type, container)


def create_cli_service_builder(container: ServiceContainer) -> CLIServiceBuilder:
    """Create a CLIService builder."""
    cli_service_type = container.get_type("gal_friday.cli_service", "CLIService")
    return CLIServiceBuilder("CLIService", cast("type[CLIService]", cli_service_type), container)


async def initialize_core_services(container: ServiceContainer) -> dict[str, Any]:
    """Initialize core services using builders."""
    director = ServiceInitializationDirector(container)
    director.register_standard_builders()

    return await director.initialize_all_services()
