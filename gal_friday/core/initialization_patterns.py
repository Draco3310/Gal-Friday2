"""Type-safe initialization patterns for the GalFriday application.

This module provides enterprise-grade initialization patterns that:
- Enforce type safety at compile time
- Provide proper error handling with structured exceptions
- Support dependency injection integration
- Enable configuration-driven initialization
- Offer validation and health checks
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from gal_friday.core.service_container import ServiceContainer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gal_friday.config_manager import ConfigManager
    from gal_friday.logger_service import LoggerService

T = TypeVar("T")
ServiceType = TypeVar("ServiceType")


class InitializationStage(Enum):
    """Stages of application initialization."""

    CONFIGURATION = "configuration"
    LOGGING = "logging"
    DEPENDENCY_INJECTION = "dependency_injection"
    DATABASE = "database"
    SERVICES = "services"
    VALIDATION = "validation"
    READY = "ready"


@dataclass
class InitializationResult(Generic[T]):
    """Result of an initialization operation."""

    success: bool
    instance: T | None = None
    error: Exception | None = None
    stage: InitializationStage | None = None
    validation_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Check if initialization was successful."""
        return self.success and self.instance is not None and self.error is None

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def get_instance(self) -> T:
        """Get the initialized instance with type safety."""
        if not self.is_success or self.instance is None:
            raise RuntimeError(f"Cannot get instance: initialization failed with error: {self.error}")
        return self.instance


class InitializationError(Exception):
    """Base exception for initialization errors."""

    def __init__(self, stage: InitializationStage, message: str, cause: Exception | None = None) -> None:
        """Initialize initialization error."""
        super().__init__(message)
        self.stage = stage
        self.cause = cause


class TypeSafeInitializer(ABC, Generic[T]):
    """Abstract base class for type-safe initializers."""

    def __init__(self, container: ServiceContainer | None = None) -> None:
        """Initialize the type-safe initializer."""
        self.container = container
        self._initialized = False
        self._instance: T | None = None

    @abstractmethod
    async def initialize(self) -> InitializationResult[T]:
        """Initialize the component and return the result."""

    @abstractmethod
    async def validate(self) -> list[str]:
        """Validate the component and return any validation errors."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""

    @property
    def is_initialized(self) -> bool:
        """Check if the component is initialized."""
        return self._initialized

    @property
    def instance(self) -> T | None:
        """Get the initialized instance."""
        return self._instance

    def get_instance_safe(self) -> T:
        """Get the initialized instance with type safety."""
        if not self._initialized or self._instance is None:
            raise RuntimeError("Component not initialized")
        return self._instance


class ConfigurationInitializer(TypeSafeInitializer["ConfigManager"]):
    """Type-safe initializer for configuration management."""

    def __init__(self, config_path: str, container: ServiceContainer | None = None) -> None:
        """Initialize configuration initializer."""
        super().__init__(container)
        self.config_path = config_path

    async def initialize(self) -> InitializationResult[ConfigManager]:
        """Initialize the configuration manager."""
        try:
            # Import here to avoid circular imports
            from gal_friday.config_manager import (  # noqa: PLC0415 - architectural: prevents circular dependency between initialization patterns and config manager
                ConfigManager,
            )

            config_manager = ConfigManager(
                config_path=self.config_path,
                logger_service=None,  # Will be set later
            )

            # Validate configuration
            validation_errors = await self.validate_config(config_manager)
            if validation_errors:
                return InitializationResult(
                    success=False,
                    instance=None,
                    error=InitializationError(
                        InitializationStage.CONFIGURATION,
                        f"Configuration validation failed: {validation_errors}",
                    ),
                    stage=InitializationStage.CONFIGURATION,
                    validation_errors=validation_errors,
                )

            self._instance = config_manager
            self._initialized = True

            return InitializationResult(
                success=True,
                instance=config_manager,
                stage=InitializationStage.CONFIGURATION,
            )

        except Exception as e:
            return InitializationResult(
                success=False,
                instance=None,
                error=InitializationError(InitializationStage.CONFIGURATION, str(e), e),
                stage=InitializationStage.CONFIGURATION,
            )

    async def validate_config(self, config_manager: ConfigManager) -> list[str]:
        """Validate configuration."""
        errors = []

        # Check if configuration is valid
        if not config_manager.is_valid():
            errors.append("Configuration is invalid")

        # Check required sections
        required_sections = ["logging", "database", "operational_modes"]
        errors.extend(
            [
                f"Missing required configuration section: {section}"
                for section in required_sections
                if not config_manager.get(section)
            ],
        )

        return errors

    async def validate(self) -> list[str]:
        """Validate the configuration."""
        if not self._initialized or not self._instance:
            return ["Configuration not initialized"]

        return await self.validate_config(self._instance)

    async def cleanup(self) -> None:
        """Clean up configuration resources."""
        if self._instance and hasattr(self._instance, "stop_watching"):
            self._instance.stop_watching()
        self._initialized = False
        self._instance = None


class ServiceContainerInitializer(TypeSafeInitializer[ServiceContainer]):
    """Type-safe initializer for the dependency injection container."""

    def __init__(self, config_manager: ConfigManager) -> None:
        """Initialize service container initializer."""
        super().__init__(None)
        self.config_manager = config_manager

    async def initialize(self) -> InitializationResult[ServiceContainer]:
        """Initialize the service container."""
        try:
            from gal_friday.core.application_services import (  # noqa: PLC0415 - architectural: prevents circular dependency in service container initialization
                create_application_container,
            )

            container = create_application_container(self.config_manager)

            # Validate service registrations
            validation_errors = container.validate_registrations()
            if validation_errors:
                return InitializationResult(
                    success=False,
                    instance=None,
                    error=InitializationError(
                        InitializationStage.DEPENDENCY_INJECTION,
                        f"Service registration validation failed: {validation_errors}",
                    ),
                    stage=InitializationStage.DEPENDENCY_INJECTION,
                    validation_errors=validation_errors,
                )

            self._instance = container
            self._initialized = True

            return InitializationResult(
                success=True,
                instance=container,
                stage=InitializationStage.DEPENDENCY_INJECTION,
            )

        except Exception as e:
            return InitializationResult(
                success=False,
                instance=None,
                error=InitializationError(InitializationStage.DEPENDENCY_INJECTION, str(e), e),
                stage=InitializationStage.DEPENDENCY_INJECTION,
            )

    async def validate(self) -> list[str]:
        """Validate the service container."""
        if not self._initialized or not self._instance:
            return ["Service container not initialized"]

        return self._instance.validate_registrations()

    async def cleanup(self) -> None:
        """Clean up service container resources."""
        if self._instance:
            self._instance.dispose()
        self._initialized = False
        self._instance = None


class DatabaseInitializer(TypeSafeInitializer[tuple[Any, Any]]):
    """Type-safe initializer for database components."""

    def __init__(self, config_manager: ConfigManager, logger: LoggerService | None = None) -> None:
        """Initialize database initializer."""
        super().__init__(None)
        self.config_manager = config_manager
        self.logger = logger

    def _validate_logger_for_database_init(self) -> None:
        """Validate that logger service is available for database initialization."""
        if not self.logger:
            raise InitializationError(
                InitializationStage.DATABASE,
                "Logger service required for database initialization",
            )

    async def initialize(self) -> InitializationResult[tuple[Any, Any]]:
        """Initialize database components."""
        try:
            # Import here to avoid circular imports
            from gal_friday.dal.connection_pool import (  # noqa: PLC0415 - architectural: prevents circular dependency with database initialization
                DatabaseConnectionPool,
            )

            self._validate_logger_for_database_init()

            # Create connection pool
            # Type narrowing: we validated above that logger is not None
            logger = cast("LoggerService", self.logger)
            db_pool = DatabaseConnectionPool(
                config=self.config_manager,
                logger=logger,
            )

            await db_pool.initialize()
            session_maker = db_pool.get_session_maker()

            if not session_maker:
                return InitializationResult(
                    success=False,
                    instance=None,
                    error=InitializationError(
                        InitializationStage.DATABASE,
                        "Failed to create database session maker",
                    ),
                    stage=InitializationStage.DATABASE,
                )

            instance = (db_pool, session_maker)
            self._instance = instance
            self._initialized = True

            return InitializationResult(
                success=True,
                instance=instance,
                stage=InitializationStage.DATABASE,
            )

        except Exception as e:
            return InitializationResult(
                success=False,
                instance=None,
                error=InitializationError(InitializationStage.DATABASE, str(e), e),
                stage=InitializationStage.DATABASE,
            )

    async def validate(self) -> list[str]:
        """Validate database components."""
        if not self._initialized or not self._instance:
            return ["Database not initialized"]

        errors = []
        db_pool, session_maker = self._instance

        # Test database connection
        try:
            async with session_maker() as session:
                await session.execute("SELECT 1")
        except Exception as e:
            errors.append(f"Database connection test failed: {e}")

        return errors

    async def cleanup(self) -> None:
        """Clean up database resources."""
        if self._instance:
            db_pool, _ = self._instance
            await db_pool.close()
        self._initialized = False
        self._instance = None


class TypeSafeInitializationManager:
    """Manager for type-safe initialization of application components."""

    def __init__(self) -> None:
        """Initialize the initialization manager."""
        self._initializers: dict[InitializationStage, TypeSafeInitializer[Any]] = {}
        self._results: dict[InitializationStage, InitializationResult[Any]] = {}
        self._current_stage = InitializationStage.CONFIGURATION

    def register_initializer(self, stage: InitializationStage, initializer: TypeSafeInitializer[T]) -> None:
        """Register an initializer for a specific stage."""
        self._initializers[stage] = initializer

    async def initialize_stage(self, stage: InitializationStage) -> InitializationResult[Any]:
        """Initialize a specific stage."""
        if stage not in self._initializers:
            return InitializationResult(
                success=False,
                instance=None,
                error=InitializationError(stage, f"No initializer registered for stage: {stage}"),
                stage=stage,
            )

        initializer = self._initializers[stage]
        result = await initializer.initialize()
        self._results[stage] = result

        if result.is_success:
            self._current_stage = stage

        return result

    async def initialize_all(self) -> dict[InitializationStage, InitializationResult[Any]]:
        """Initialize all registered stages in order."""
        stages_order = [
            InitializationStage.CONFIGURATION,
            InitializationStage.LOGGING,
            InitializationStage.DEPENDENCY_INJECTION,
            InitializationStage.DATABASE,
            InitializationStage.SERVICES,
            InitializationStage.VALIDATION,
        ]

        results = {}

        for stage in stages_order:
            if stage in self._initializers:
                result = await self.initialize_stage(stage)
                results[stage] = result

                if not result.is_success:
                    # Stop on first failure
                    break

        return results

    async def validate_all(self) -> dict[InitializationStage, list[str]]:
        """Validate all initialized components."""
        validation_results = {}

        for stage, initializer in self._initializers.items():
            if initializer.is_initialized:
                errors = await initializer.validate()
                validation_results[stage] = errors

        return validation_results

    async def cleanup_all(self) -> None:
        """Clean up all initialized components."""
        # Clean up in reverse order
        stages_order = list(
            reversed(
                [
                    InitializationStage.SERVICES,
                    InitializationStage.DATABASE,
                    InitializationStage.DEPENDENCY_INJECTION,
                    InitializationStage.LOGGING,
                    InitializationStage.CONFIGURATION,
                ],
            ),
        )

        for stage in stages_order:
            if stage in self._initializers:
                initializer = self._initializers[stage]
                if initializer.is_initialized:
                    await initializer.cleanup()

    def get_result(self, stage: InitializationStage) -> InitializationResult[Any] | None:
        """Get the result for a specific stage."""
        return self._results.get(stage)

    def get_instance(self, stage: InitializationStage) -> Any:
        """Get the instance for a specific stage with type safety."""
        result = self.get_result(stage)
        if result is None or not result.is_success:
            raise RuntimeError(f"Stage {stage} not initialized or failed")
        return result.get_instance()

    @asynccontextmanager
    async def initialization_context(self) -> AsyncIterator[TypeSafeInitializationManager]:
        """Context manager for safe initialization and cleanup."""
        try:
            yield self
        finally:
            await self.cleanup_all()


# Type-safe helper functions
def create_config_initializer(config_path: str) -> ConfigurationInitializer:
    """Create a configuration initializer."""
    return ConfigurationInitializer(config_path)


def create_container_initializer(config_manager: ConfigManager) -> ServiceContainerInitializer:
    """Create a service container initializer."""
    return ServiceContainerInitializer(config_manager)


def create_database_initializer(config_manager: ConfigManager) -> DatabaseInitializer:
    """Create a database initializer."""
    return DatabaseInitializer(config_manager)


async def initialize_application_safely(config_path: str) -> TypeSafeInitializationManager:
    """Initialize application with type safety and proper error handling."""
    manager = TypeSafeInitializationManager()

    # Register initializers
    config_init = create_config_initializer(config_path)
    manager.register_initializer(InitializationStage.CONFIGURATION, config_init)

    # Initialize configuration first
    config_result = await manager.initialize_stage(InitializationStage.CONFIGURATION)
    if not config_result.is_success:
        raise InitializationError(
            InitializationStage.CONFIGURATION,
            f"Configuration initialization failed: {config_result.error}",
        )

    config_manager = config_result.get_instance()

    # Register remaining initializers
    container_init = create_container_initializer(config_manager)
    manager.register_initializer(InitializationStage.DEPENDENCY_INJECTION, container_init)

    database_init = create_database_initializer(config_manager)
    manager.register_initializer(InitializationStage.DATABASE, database_init)

    return manager
