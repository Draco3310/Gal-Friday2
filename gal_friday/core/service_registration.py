"""Service registration system for the dependency injection container.

This module provides:
- Decorator-based service registration
- Configuration-driven service registration
- Auto-discovery of services
- Factory pattern support
- Conditional registration
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, TypeVar

from gal_friday.core.service_container import ServiceContainer, ServiceLifecycle

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

# Global registries for type-safe service registration metadata
_SERVICE_REGISTRY: dict[type, ServiceRegistrationConfig] = {}
_FACTORY_REGISTRY: dict[Callable[..., Any], ServiceRegistrationConfig] = {}


class RegistrationError(Exception):
    """Raised when service registration fails."""


@dataclass
class ServiceRegistrationConfig:
    """Configuration for service registration."""

    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    tags: set[str] | None = None
    dependencies: list[type] | None = None
    configuration: dict[str, Any] | None = None
    conditions: list[Callable[[], bool]] | None = None
    factory: Callable[..., Any] | None = None
    priority: int = 0


class ServiceRegistrar:
    """Service registration manager."""

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize the service registrar.

        Args:
            container: Service container to register services with
        """
        self._container = container
        self._registrations: dict[type, ServiceRegistrationConfig] = {}
        self._auto_register_enabled = True

    def register(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        factory: Callable[..., T] | None = None,
        instance: T | None = None,
        tags: set[str] | None = None,
        configuration: dict[str, Any] | None = None,
        conditions: list[Callable[[], bool]] | None = None,
    ) -> ServiceRegistrar:
        """Register a service with the container.

        Args:
            service_type: Service interface/type
            implementation_type: Concrete implementation
            lifecycle: Service lifecycle pattern
            factory: Factory function
            instance: Pre-created instance
            tags: Service tags
            configuration: Service configuration
            conditions: Registration conditions

        Returns:
            Self for method chaining
        """
        # Register with container based on lifecycle
        if lifecycle == ServiceLifecycle.SINGLETON:
            self._container.register_singleton(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                tags=tags,
                configuration=configuration,
                conditions=conditions,
            )
        elif lifecycle == ServiceLifecycle.TRANSIENT:
            self._container.register_transient(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                tags=tags,
                configuration=configuration,
                conditions=conditions,
            )
        elif lifecycle == ServiceLifecycle.SCOPED:
            self._container.register_scoped(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                tags=tags,
                configuration=configuration,
                conditions=conditions,
            )
        else:
            raise RegistrationError(f"Unknown lifecycle: {lifecycle}")

        return self

    def register_factory(
        self,
        service_type: type[T],
        factory: Callable[..., T],
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        tags: set[str] | None = None,
        configuration: dict[str, Any] | None = None,
        conditions: list[Callable[[], bool]] | None = None,
    ) -> ServiceRegistrar:
        """Register a service using a factory function.

        Args:
            service_type: Service interface/type
            factory: Factory function
            lifecycle: Service lifecycle pattern
            tags: Service tags
            configuration: Service configuration
            conditions: Registration conditions

        Returns:
            Self for method chaining
        """
        return self.register(
            service_type=service_type,
            factory=factory,
            lifecycle=lifecycle,
            tags=tags,
            configuration=configuration,
            conditions=conditions,
        )

    def register_instance(
        self,
        service_type: type[T],
        instance: T,
        tags: set[str] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> ServiceRegistrar:
        """Register a pre-created service instance.

        Args:
            service_type: Service interface/type
            instance: Pre-created instance
            tags: Service tags
            configuration: Service configuration

        Returns:
            Self for method chaining
        """
        return self.register(
            service_type=service_type,
            instance=instance,
            lifecycle=ServiceLifecycle.SINGLETON,
            tags=tags,
            configuration=configuration,
        )

    def register_conditional(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        condition: Callable[[], bool] | None = None,
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        factory: Callable[..., T] | None = None,
        tags: set[str] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> ServiceRegistrar:
        """Register a service with a condition.

        Args:
            service_type: Service interface/type
            implementation_type: Concrete implementation
            condition: Registration condition
            lifecycle: Service lifecycle pattern
            factory: Factory function
            tags: Service tags
            configuration: Service configuration

        Returns:
            Self for method chaining
        """
        conditions = [condition] if condition else None
        return self.register(
            service_type=service_type,
            implementation_type=implementation_type,
            lifecycle=lifecycle,
            factory=factory,
            tags=tags,
            configuration=configuration,
            conditions=conditions,
        )

    def auto_register_types(self, *types: type) -> ServiceRegistrar:
        """Auto-register types with their default configurations.

        Args:
            *types: Types to auto-register

        Returns:
            Self for method chaining
        """
        for service_type in types:
            # Check if already registered
            if not self._container.is_registered(service_type):
                # Get registration config from global registry or default
                config = _SERVICE_REGISTRY.get(service_type, ServiceRegistrationConfig())

                self.register(
                    service_type=service_type,
                    implementation_type=service_type if inspect.isclass(service_type) else None,
                    lifecycle=config.lifecycle,
                    factory=config.factory,
                    tags=config.tags,
                    configuration=config.configuration,
                    conditions=config.conditions,
                )

        return self

    def register_from_config(self, config: dict[str, Any]) -> ServiceRegistrar:
        """Register services from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Self for method chaining
        """
        services_config = config.get("services", {})

        for service_name, service_config in services_config.items():
            try:
                # Import the service type
                module_name = service_config.get("module")
                class_name = service_config.get("class")

                if not module_name or not class_name:
                    continue

                module = __import__(module_name, fromlist=[class_name])
                service_type = getattr(module, class_name)

                # Get implementation type if different
                impl_module_name = service_config.get("implementation_module")
                impl_class_name = service_config.get("implementation_class")

                implementation_type = None
                if impl_module_name and impl_class_name:
                    impl_module = __import__(impl_module_name, fromlist=[impl_class_name])
                    implementation_type = getattr(impl_module, impl_class_name)

                # Parse lifecycle
                lifecycle_str = service_config.get("lifecycle", "singleton")
                lifecycle = ServiceLifecycle(lifecycle_str)

                # Parse tags
                tags = set(service_config.get("tags", []))

                # Parse configuration
                service_configuration = service_config.get("configuration", {})

                # Register the service
                self.register(
                    service_type=service_type,
                    implementation_type=implementation_type,
                    lifecycle=lifecycle,
                    tags=tags,
                    configuration=service_configuration,
                )

            except Exception as e:
                raise RegistrationError(f"Failed to register service {service_name}: {e}") from e

        return self


def service(
    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
    tags: set[str] | None = None,
    configuration: dict[str, Any] | None = None,
    conditions: list[Callable[[], bool]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator for marking classes as services.

    Args:
        lifecycle: Service lifecycle pattern
        tags: Service tags
        configuration: Service configuration
        conditions: Registration conditions

    Returns:
        Decorated class
    """

    def decorator(cls: type[T]) -> type[T]:
        # Store registration metadata in global registry
        if cls not in _SERVICE_REGISTRY:
            _SERVICE_REGISTRY[cls] = ServiceRegistrationConfig(
                lifecycle=lifecycle,
                tags=tags,
                configuration=configuration,
                conditions=conditions,
            )

        return cls

    return decorator


def singleton(
    tags: set[str] | None = None,
    configuration: dict[str, Any] | None = None,
    conditions: list[Callable[[], bool]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator for singleton services.

    Args:
        tags: Service tags
        configuration: Service configuration
        conditions: Registration conditions

    Returns:
        Decorated class
    """
    return service(
        lifecycle=ServiceLifecycle.SINGLETON,
        tags=tags,
        configuration=configuration,
        conditions=conditions,
    )


def transient(
    tags: set[str] | None = None,
    configuration: dict[str, Any] | None = None,
    conditions: list[Callable[[], bool]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator for transient services.

    Args:
        tags: Service tags
        configuration: Service configuration
        conditions: Registration conditions

    Returns:
        Decorated class
    """
    return service(
        lifecycle=ServiceLifecycle.TRANSIENT,
        tags=tags,
        configuration=configuration,
        conditions=conditions,
    )


def scoped(
    tags: set[str] | None = None,
    configuration: dict[str, Any] | None = None,
    conditions: list[Callable[[], bool]] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Decorator for scoped services.

    Args:
        tags: Service tags
        configuration: Service configuration
        conditions: Registration conditions

    Returns:
        Decorated class
    """
    return service(
        lifecycle=ServiceLifecycle.SCOPED,
        tags=tags,
        configuration=configuration,
        conditions=conditions,
    )


def factory(
    service_type: type[T],
    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
    tags: set[str] | None = None,
    configuration: dict[str, Any] | None = None,
    conditions: list[Callable[[], bool]] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for factory functions.

    Args:
        service_type: Service type that factory creates
        lifecycle: Service lifecycle pattern
        tags: Service tags
        configuration: Service configuration
        conditions: Registration conditions

    Returns:
        Decorated factory function
    """

    def decorator(factory_func: Callable[..., T]) -> Callable[..., T]:
        # Store factory metadata in global registry
        if factory_func not in _FACTORY_REGISTRY:
            _FACTORY_REGISTRY[factory_func] = ServiceRegistrationConfig(
                lifecycle=lifecycle,
                tags=tags,
                configuration=configuration,
                conditions=conditions,
                factory=factory_func,
            )

        return factory_func

    return decorator


class ServiceDiscovery:
    """Service discovery utilities."""

    @staticmethod
    def discover_services(module_name: str) -> list[type]:
        """Discover services in a module.

        Args:
            module_name: Module to discover services in

        Returns:
            List of discovered service types
        """
        try:
            module = __import__(module_name, fromlist=[""])
            services = []

            for name in dir(module):
                obj = getattr(module, name)
                if inspect.isclass(obj) and obj in _SERVICE_REGISTRY:
                    services.append(obj)

        except Exception:
            return []
        else:
            return services

    @staticmethod
    def discover_factories(module_name: str) -> list[Callable[..., Any]]:
        """Discover factory functions in a module.

        Args:
            module_name: Module to discover factories in

        Returns:
            List of discovered factory functions
        """
        try:
            module = __import__(module_name, fromlist=[""])
            factories: list[Callable[..., Any]] = []

            for name in dir(module):
                obj = getattr(module, name)
                if inspect.isfunction(obj) and obj in _FACTORY_REGISTRY:
                    factories.append(obj)

        except Exception:
            return []
        else:
            return factories

    @staticmethod
    def auto_register_module(
        registrar: ServiceRegistrar,
        module_name: str,
        include_factories: bool = True,
    ) -> ServiceRegistrar:
        """Auto-register all services and factories in a module.

        Args:
            registrar: Service registrar
            module_name: Module to auto-register
            include_factories: Whether to include factory functions

        Returns:
            Service registrar for method chaining
        """
        # Register services
        services = ServiceDiscovery.discover_services(module_name)
        for service_type in services:
            config = _SERVICE_REGISTRY.get(service_type, ServiceRegistrationConfig())
            registrar.register(
                service_type=service_type,
                implementation_type=service_type,
                lifecycle=config.lifecycle,
                tags=config.tags,
                configuration=config.configuration,
                conditions=config.conditions,
            )

        # Register factories
        if include_factories:
            factories = ServiceDiscovery.discover_factories(module_name)
            for factory_func in factories:
                config = _FACTORY_REGISTRY.get(factory_func, ServiceRegistrationConfig())
                # Extract service type from factory config
                # This would need to be enhanced to properly determine service type
                registrar.register_factory(
                    service_type=object,  # Would need better type extraction
                    factory=factory_func,
                    lifecycle=config.lifecycle,
                    tags=config.tags,
                    configuration=config.configuration,
                    conditions=config.conditions,
                )

        return registrar
