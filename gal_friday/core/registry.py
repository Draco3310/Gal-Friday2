"""Plugin registry pattern for dynamic component loading.

This module provides a flexible registry system for managing pluggable
components, enabling configuration-based selection of implementations
without local imports.
"""

from collections.abc import Callable
import importlib
from typing import Any, TypeVar, cast

from gal_friday.interfaces.service_protocol import ServiceProtocol
from gal_friday.logger_service import LoggerService

T = TypeVar("T")


class ComponentRegistry:
    """Generic registry for pluggable components.

    Example:
        # Create a registry for execution handlers
        execution_registry = ComponentRegistry[ExecutionHandlerInterface](
            'ExecutionHandler',
            logger_service
        )

        # Register implementations
        execution_registry.register('kraken', KrakenExecutionHandler)
        execution_registry.register_lazy('simulated',
            'gal_friday.execution.simulated',
            'SimulatedExecutionHandler'
        )

        # Create instance based on config
        handler = execution_registry.create('kraken', config={'api_key': '...'})
    """

    def __init__(self, component_type: str, logger: LoggerService) -> None:
        """Initialize the component registry.

        Args:
            component_type: Human-readable name for the component type
            logger: Logger service instance
        """
        self.component_type = component_type
        self.logger = logger
        self._components: dict[str, type[Any]] = {}
        self._lazy_components: dict[str, tuple[str, str]] = {}
        self._factories: dict[str, Callable[..., Any] | tuple[str, str]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._source_module = self.__class__.__name__

    def register(
        self,
        name: str,
        component_class: type[T],
        factory: Callable[..., T] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component implementation.

        Args:
            name: Unique name for this implementation
            component_class: The class to instantiate
            factory: Optional factory function for complex initialization
            metadata: Optional metadata about the component
        """
        self._components[name] = component_class
        if factory:
            self._factories[name] = factory
        if metadata:
            self._metadata[name] = metadata

        self.logger.debug(
            f"Registered {self.component_type} '{name}'",
            source_module=self._source_module,
            context={"component": component_class.__name__},
        )

    def register_lazy(
        self,
        name: str,
        module_path: str,
        class_name: str,
        factory_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a component for lazy loading.

        Args:
            name: Unique name for this implementation
            module_path: Full module path (e.g., 'gal_friday.execution.kraken')
            class_name: Name of the class in the module
            factory_name: Optional factory function name in the module
            metadata: Optional metadata about the component
        """
        self._lazy_components[name] = (module_path, class_name)
        if factory_name:
            self._factories[name] = (module_path, factory_name)
        if metadata:
            self._metadata[name] = metadata

        self.logger.debug(
            f"Registered lazy {self.component_type} '{name}'",
            source_module=self._source_module,
            context={"module": module_path, "class": class_name},
        )

    def _load_lazy_component(self, name: str) -> type[Any]:
        """Load a lazily registered component."""
        if name not in self._lazy_components:
            raise KeyError(f"No lazy component registered with name '{name}'")

        module_path, class_name = self._lazy_components[name]

        try:
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            # Cache it for future use
            self._components[name] = component_class
            del self._lazy_components[name]

            # Load factory if it's lazy too
            if name in self._factories and isinstance(self._factories[name], tuple):
                factory_tuple = cast("tuple[str, str]", self._factories[name])
                factory_module_path, factory_name = factory_tuple
                if factory_module_path == module_path:
                    self._factories[name] = getattr(module, factory_name)
                else:
                    factory_module = importlib.import_module(factory_module_path)
                    self._factories[name] = getattr(factory_module, factory_name)

            self.logger.info(
                f"Lazy loaded {self.component_type} '{name}'",
                source_module=self._source_module,
                context={"module": module_path, "class": class_name},
            )

        except (ImportError, AttributeError) as e:
            self.logger.exception(
                f"Failed to load {self.component_type} '{name}'",
                source_module=self._source_module,
                context={"error": str(e), "module": module_path, "class": class_name},
            )
            raise ImportError(
                f"Cannot load {self.component_type} '{name}' from {module_path}.{class_name}: {e}",
            ) from e
        else:
            return cast("type[Any]", component_class)

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create an instance of a registered component.

        Args:
            name: Name of the component to create
            **kwargs: Arguments to pass to the constructor or factory

        Returns:
            Instance of the requested component

        Raises:
            KeyError: If no component is registered with the given name
            ImportError: If lazy loading fails
        """
        # Try to get already loaded component
        component_class = self._components.get(name)

        # If not found, try lazy loading
        if component_class is None and name in self._lazy_components:
            component_class = self._load_lazy_component(name)

        if component_class is None:
            available = self.get_available()
            raise KeyError(
                f"No {self.component_type} registered with name '{name}'. Available: {', '.join(available)}",
            )

        # Use factory if available
        if name in self._factories:
            factory = self._factories[name]
            # Ensure factory is loaded if it's still a tuple
            if isinstance(factory, tuple):
                self._load_lazy_component(name)
                factory = self._factories[name]
            instance = factory(**kwargs)  # type: ignore[operator]
        else:
            instance = component_class(**kwargs)

        self.logger.debug(
            f"Created {self.component_type} instance '{name}'",
            source_module=self._source_module,
            context={"class": component_class.__name__},
        )

        return instance

    def get_class(self, name: str) -> type[Any]:
        """Get the class for a registered component without instantiating it.

        Args:
            name: Name of the component

        Returns:
            The component class
        """
        component_class = self._components.get(name)

        if component_class is None and name in self._lazy_components:
            component_class = self._load_lazy_component(name)

        if component_class is None:
            raise KeyError(f"No {self.component_type} registered with name '{name}'")

        return component_class

    def get_available(self) -> list[str]:
        """Get list of all available component names."""
        return list(self._components.keys()) + list(self._lazy_components.keys())

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a component."""
        return self._metadata.get(name, {})

    def is_available(self, name: str) -> bool:
        """Check if a component is available."""
        return name in self._components or name in self._lazy_components

    def unregister(self, name: str) -> None:
        """Unregister a component."""
        self._components.pop(name, None)
        self._lazy_components.pop(name, None)
        self._factories.pop(name, None)
        self._metadata.pop(name, None)

        self.logger.debug(
            f"Unregistered {self.component_type} '{name}'",
            source_module=self._source_module,
        )


class ServiceRegistry(ComponentRegistry):
    """Specialized registry for ServiceProtocol implementations.

    Adds service-specific functionality like lifecycle management.
    """

    async def create_and_initialize(self, name: str, **kwargs: Any) -> ServiceProtocol:
        """Create and initialize a service.

        Args:
            name: Name of the service to create
            **kwargs: Arguments to pass to the constructor

        Returns:
            Initialized service instance
        """
        service = self.create(name, **kwargs)

        if hasattr(service, "initialize"):
            await service.initialize()
            self.logger.info(
                f"Initialized {self.component_type} service '{name}'",
                source_module=self._source_module,
            )

        return cast("ServiceProtocol", service)


class PluginManager:
    """Manager for multiple component registries.

    Provides a central place to manage all plugin registries in the application.
    """

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the plugin manager.

        Args:
            logger: Logger service instance
        """
        self.logger = logger
        self._registries: dict[str, ComponentRegistry] = {}
        self._source_module = self.__class__.__name__

    def create_registry(
        self,
        name: str,
        component_type: str | None = None,
    ) -> ComponentRegistry:
        """Create a new component registry.

        Args:
            name: Unique name for the registry
            component_type: Human-readable component type name

        Returns:
            The created registry
        """
        if name in self._registries:
            raise ValueError(f"Registry '{name}' already exists")

        registry = ComponentRegistry(
            component_type or name,
            self.logger,
        )
        self._registries[name] = registry

        self.logger.info(
            f"Created registry '{name}' for {component_type or name}",
            source_module=self._source_module,
        )

        return registry

    def get_registry(self, name: str) -> ComponentRegistry:
        """Get an existing registry.

        Args:
            name: Name of the registry

        Returns:
            The requested registry

        Raises:
            KeyError: If no registry exists with the given name
        """
        if name not in self._registries:
            raise KeyError(f"No registry found with name '{name}'")
        return self._registries[name]

    def list_registries(self) -> list[str]:
        """Get list of all registry names."""
        return list(self._registries.keys())
