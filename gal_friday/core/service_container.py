"""Enterprise-grade dependency injection container for the GalFriday application.

This module provides a comprehensive dependency injection system with:
- Type-safe service registration and resolution
- Lifecycle management (singleton, transient, scoped)
- Async service support
- Configuration-driven service registration
- Circular dependency detection
- Factory pattern support
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Self,
    TypeVar,
    cast,
    get_args,
    get_origin,
)

import asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from gal_friday.config_manager import ConfigManager
    from gal_friday.logger_service import LoggerService

T = TypeVar("T")
ServiceType = TypeVar("ServiceType")


class ServiceLifecycle(Enum):
    """Service lifecycle management patterns."""

    SINGLETON = "singleton"  # One instance per container
    TRANSIENT = "transient"  # New instance per request
    SCOPED = "scoped"  # One instance per scope


class ServiceError(Exception):
    """Base exception for service container errors."""


class ServiceNotRegisteredError(ServiceError):
    """Raised when a requested service is not registered."""


class CircularDependencyError(ServiceError):
    """Raised when circular dependencies are detected."""


class ServiceRegistrationError(ServiceError):
    """Raised when service registration fails."""


class ServiceResolutionError(ServiceError):
    """Raised when service resolution fails."""


@dataclass
class ServiceDescriptor(Generic[T]):
    """Metadata for registered services."""

    service_type: type[T]
    implementation_type: type[T] | None
    factory: Callable[..., T] | Callable[..., asyncio.Task[T]] | None
    instance: T | None
    lifecycle: ServiceLifecycle
    dependencies: list[type] = field(default_factory=list)
    is_async: bool = False
    configuration: dict[str, Any] = field(default_factory=dict)
    conditions: list[Callable[[], bool]] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Validate service descriptor after initialization."""
        if self.implementation_type is None and self.factory is None and self.instance is None:
            raise ServiceRegistrationError(
                f"Service {self.service_type.__name__} must have implementation_type, factory, or instance",
            )


@dataclass
class ServiceScope:
    """Represents a service resolution scope."""

    scope_id: str
    instances: dict[type, Any] = field(default_factory=dict)
    is_active: bool = True


class ServiceContainer:
    """Enterprise-grade dependency injection container."""

    def __init__(self, config_manager: ConfigManager | None = None) -> None:
        """Initialize the service container.

        Args:
            config_manager: Optional configuration manager for config-driven service registration
        """
        self._services: dict[type, ServiceDescriptor[Any]] = {}
        self._singletons: dict[type, Any] = {}
        self._scopes: dict[str, ServiceScope] = {}
        self._resolution_stack: list[type] = []
        self._config_manager = config_manager
        self._logger: LoggerService | None = None
        self._is_disposed = False

    def register_singleton(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[..., T] | None = None,
        instance: T | None = None,
        **kwargs: Any,
    ) -> ServiceContainer:
        """Register a service with singleton lifecycle.

        Args:
            service_type: The service interface/type
            implementation_type: Concrete implementation type
            factory: Factory function to create the service
            instance: Pre-created instance
            **kwargs: Additional configuration

        Returns:
            Self for method chaining
        """
        return self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifecycle=ServiceLifecycle.SINGLETON,
            **kwargs,
        )

    def register_transient(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> ServiceContainer:
        """Register a service with transient lifecycle.

        Args:
            service_type: The service interface/type
            implementation_type: Concrete implementation type
            factory: Factory function to create the service
            **kwargs: Additional configuration

        Returns:
            Self for method chaining
        """
        return self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifecycle=ServiceLifecycle.TRANSIENT,
            **kwargs,
        )

    def register_scoped(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> ServiceContainer:
        """Register a service with scoped lifecycle.

        Args:
            service_type: The service interface/type
            implementation_type: Concrete implementation type
            factory: Factory function to create the service
            **kwargs: Additional configuration

        Returns:
            Self for method chaining
        """
        return self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifecycle=ServiceLifecycle.SCOPED,
            **kwargs,
        )

    def _register_service(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[..., T] | None = None,
        instance: T | None = None,
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        dependencies: list[type] | None = None,
        configuration: dict[str, Any] | None = None,
        conditions: list[Callable[[], bool]] | None = None,
        tags: set[str] | None = None,
    ) -> ServiceContainer:
        """Internal method to register a service.

        Args:
            service_type: The service interface/type
            implementation_type: Concrete implementation type
            factory: Factory function to create the service
            instance: Pre-created instance
            lifecycle: Service lifecycle pattern
            dependencies: Explicit dependencies
            configuration: Service configuration
            conditions: Registration conditions
            tags: Service tags

        Returns:
            Self for method chaining
        """
        if self._is_disposed:
            raise ServiceError("Cannot register services on disposed container")

        # Auto-detect if factory is async
        is_async = False
        if factory:
            is_async = asyncio.iscoroutinefunction(factory)

        # Auto-detect dependencies if not provided
        if dependencies is None:
            dependencies = self._extract_dependencies(implementation_type or factory)

        # Create service descriptor
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifecycle=lifecycle,
            dependencies=dependencies,
            is_async=is_async,
            configuration=configuration or {},
            conditions=conditions or [],
            tags=tags or set(),
        )

        # Validate registration
        self._validate_registration(descriptor)

        # Register the service
        self._services[service_type] = descriptor

        # If instance is provided and lifecycle is singleton, store it
        if instance is not None and lifecycle == ServiceLifecycle.SINGLETON:
            self._singletons[service_type] = instance

        return self

    def _extract_dependencies(self, implementation: type | Callable[..., Any] | None) -> list[type]:
        """Extract dependencies from constructor/factory signature.

        Args:
            implementation: Implementation type or factory function

        Returns:
            List of dependency types
        """
        if implementation is None:
            return []

        try:
            sig = self._get_signature(implementation)
            dependencies = []

            for param_name, param in sig.parameters.items():
                if self._should_skip_parameter(param_name, param):
                    continue

                param_type = self._extract_parameter_type(param)
                if param_type and inspect.isclass(param_type):
                    dependencies.append(param_type)

        except Exception:
            # If we can't extract dependencies, return empty list
            return []
        else:
            return dependencies

    def _get_signature(self, implementation: type | Callable[..., Any]) -> inspect.Signature:
        """Get signature from implementation."""
        if inspect.isclass(implementation):
            return inspect.signature(implementation.__init__)
        return inspect.signature(implementation)

    def _should_skip_parameter(self, param_name: str, param: inspect.Parameter) -> bool:
        """Check if parameter should be skipped."""
        return (
            param_name == "self"
            or param.annotation == inspect.Parameter.empty
            or get_origin(param.annotation) is type(None)
        )

    def _extract_parameter_type(self, param: inspect.Parameter) -> type | None:
        """Extract the actual type from parameter annotation."""
        param_type = param.annotation

        if hasattr(param_type, "__origin__") and param_type.__origin__ is not None:
            # Handle generic types
            args = get_args(param_type)
            if args:
                param_type = args[0]

        # Return None for empty annotations or non-type annotations
        if param_type == inspect.Parameter.empty or not isinstance(param_type, type):
            return None

        return param_type

    def _validate_registration(self, descriptor: ServiceDescriptor[Any]) -> None:
        """Validate service registration.

        Args:
            descriptor: Service descriptor to validate

        Raises:
            ServiceRegistrationError: If validation fails
        """
        # Check conditions
        for condition in descriptor.conditions:
            if not condition():
                raise ServiceRegistrationError(
                    f"Registration condition failed for service {descriptor.service_type.__name__}",
                )

        # Validate lifecycle constraints
        if (
            descriptor.lifecycle == ServiceLifecycle.SINGLETON
            and descriptor.instance is None
            and descriptor.implementation_type is None
            and descriptor.factory is None
        ):
            # Singleton must have implementation or factory
            raise ServiceRegistrationError(
                f"Singleton service {descriptor.service_type.__name__} must have implementation or factory",
            )

        # Check for circular dependencies (basic check)
        if descriptor.service_type in descriptor.dependencies:
            raise CircularDependencyError(
                f"Service {descriptor.service_type.__name__} has circular dependency on itself",
            )

    async def resolve(self, service_type: type[T]) -> T:
        """Resolve a service instance.

        Args:
            service_type: The service type to resolve

        Returns:
            Service instance

        Raises:
            ServiceNotRegisteredError: If service is not registered
            CircularDependencyError: If circular dependency is detected
            ServiceResolutionError: If resolution fails
        """
        if self._is_disposed:
            raise ServiceError("Cannot resolve services from disposed container")

        # Check for circular dependencies
        if service_type in self._resolution_stack:
            cycle = " -> ".join(t.__name__ for t in self._resolution_stack) + f" -> {service_type.__name__}"
            raise CircularDependencyError(f"Circular dependency detected: {cycle}")

        try:
            self._resolution_stack.append(service_type)
            return await self._resolve_service(service_type)
        finally:
            self._resolution_stack.pop()

    async def _resolve_service(self, service_type: type[T]) -> T:
        """Internal service resolution method.

        Args:
            service_type: The service type to resolve

        Returns:
            Service instance
        """
        # Check if service is registered
        if service_type not in self._services:
            raise ServiceNotRegisteredError(f"Service {service_type.__name__} is not registered")

        descriptor = self._services[service_type]

        # Handle different lifecycle patterns
        if descriptor.lifecycle == ServiceLifecycle.SINGLETON:
            return await self._resolve_singleton(descriptor)
        if descriptor.lifecycle == ServiceLifecycle.TRANSIENT:
            return await self._create_instance(descriptor)
        if descriptor.lifecycle == ServiceLifecycle.SCOPED:
            return await self._resolve_scoped(descriptor)
        raise ServiceResolutionError(f"Unknown lifecycle: {descriptor.lifecycle}")

    async def _resolve_singleton(self, descriptor: ServiceDescriptor[T]) -> T:
        """Resolve singleton service.

        Args:
            descriptor: Service descriptor

        Returns:
            Service instance
        """
        # Check if singleton already exists
        if descriptor.service_type in self._singletons:
            return cast("T", self._singletons[descriptor.service_type])

        # Create new singleton instance
        instance = await self._create_instance(descriptor)
        self._singletons[descriptor.service_type] = instance
        return instance

    async def _resolve_scoped(self, descriptor: ServiceDescriptor[T]) -> T:
        """Resolve scoped service.

        Args:
            descriptor: Service descriptor

        Returns:
            Service instance
        """
        # For now, use a default scope
        scope_id = "default"

        # Get or create scope
        if scope_id not in self._scopes:
            self._scopes[scope_id] = ServiceScope(scope_id=scope_id)

        scope = self._scopes[scope_id]

        # Check if instance already exists in scope
        if descriptor.service_type in scope.instances:
            return cast("T", scope.instances[descriptor.service_type])

        # Create new scoped instance
        instance = await self._create_instance(descriptor)
        scope.instances[descriptor.service_type] = instance
        return instance

    async def _create_instance(self, descriptor: ServiceDescriptor[T]) -> T:
        """Create a new service instance.

        Args:
            descriptor: Service descriptor

        Returns:
            Service instance
        """
        # If pre-created instance exists, return it
        if descriptor.instance is not None:
            return descriptor.instance

        # Resolve dependencies
        dependency_instances: list[Any] = []
        for dependency_type in descriptor.dependencies:
            dependency_instance: Any = await self.resolve(dependency_type)
            dependency_instances.append(dependency_instance)

        # Create instance using factory or constructor
        if descriptor.factory:
            if descriptor.is_async:
                # For async factories, the result should be awaitable
                factory_result = descriptor.factory(*dependency_instances)
                if asyncio.iscoroutine(factory_result):
                    instance = await factory_result
                else:
                    instance = cast("T", factory_result)
            else:
                instance = cast("T", descriptor.factory(*dependency_instances))
        elif descriptor.implementation_type:
            instance = descriptor.implementation_type(*dependency_instances)
        else:
            raise ServiceResolutionError(
                f"Cannot create instance of {descriptor.service_type.__name__}: no factory or implementation",
            )

        return cast("T", instance)

    def is_registered(self, service_type: type[T]) -> bool:
        """Check if a service type is registered.

        Args:
            service_type: The service type to check

        Returns:
            True if registered, False otherwise
        """
        return service_type in self._services

    def get_config_manager(self) -> ConfigManager | None:
        """Get the configuration manager instance.

        Returns:
            ConfigManager instance if available, None otherwise
        """
        return self._config_manager

    def get_services_by_tag(self, tag: str) -> list[ServiceDescriptor[Any]]:
        """Get all services with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of service descriptors
        """
        return [desc for desc in self._services.values() if tag in desc.tags]

    @contextmanager
    def create_scope(self, scope_id: str | None = None) -> Iterator[str]:
        """Create a new service scope.

        Args:
            scope_id: Optional scope identifier

        Yields:
            Scope identifier
        """
        if scope_id is None:
            scope_id = f"scope_{len(self._scopes)}"

        scope = ServiceScope(scope_id=scope_id)
        self._scopes[scope_id] = scope

        try:
            yield scope_id
        finally:
            # Clean up scope
            if scope_id in self._scopes:
                del self._scopes[scope_id]

    @asynccontextmanager
    async def create_async_scope(self, scope_id: str | None = None) -> AsyncIterator[str]:
        """Create a new async service scope.

        Args:
            scope_id: Optional scope identifier

        Yields:
            Scope identifier
        """
        if scope_id is None:
            scope_id = f"async_scope_{len(self._scopes)}"

        scope = ServiceScope(scope_id=scope_id)
        self._scopes[scope_id] = scope

        try:
            yield scope_id
        finally:
            # Clean up scope
            if scope_id in self._scopes:
                del self._scopes[scope_id]

    def clear_singletons(self) -> None:
        """Clear all singleton instances."""
        self._singletons.clear()

    def clear_scopes(self) -> None:
        """Clear all scoped instances."""
        self._scopes.clear()

    def dispose(self) -> None:
        """Dispose of the container and clean up resources."""
        if self._is_disposed:
            return

        self._is_disposed = True
        self.clear_singletons()
        self.clear_scopes()
        self._services.clear()
        self._resolution_stack.clear()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        """Context manager exit."""
        self.dispose()

    def validate_registrations(self) -> list[str]:
        """Validate all service registrations.

        Returns:
            List of validation errors
        """
        errors = []

        for service_type, descriptor in self._services.items():
            # Check if all dependencies are registered
            errors.extend(
                [
                    f"Service {service_type.__name__} depends on unregistered service {dependency_type.__name__}"
                    for dependency_type in descriptor.dependencies
                    if not self.is_registered(dependency_type)
                ],
            )

        return errors

    def get_type(self, module_name: str, class_name: str) -> type:
        """Get a type from module and class name.

        Args:
            module_name: Module name
            class_name: Class name

        Returns:
            Type object
        """
        try:
            module = __import__(module_name, fromlist=[class_name])
            return cast("type", getattr(module, class_name))
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import {class_name} from {module_name}: {e}") from e
