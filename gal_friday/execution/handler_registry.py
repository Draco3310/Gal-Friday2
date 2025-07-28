"""Registry for execution handler implementations.

This module provides a registry for dynamically loading execution handlers
based on configuration, eliminating the need for local imports.
"""

from typing import Any, cast

from gal_friday.core.registry import ComponentRegistry
from gal_friday.interfaces.execution_handler_interface import ExecutionHandlerInterface
from gal_friday.logger_service import LoggerService


class ExecutionHandlerRegistry:
    """Registry for execution handler implementations."""

    _instance: "ExecutionHandlerRegistry | None" = None

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the execution handler registry.

        Args:
            logger: Logger service instance
        """
        self.logger = logger
        self._registry = ComponentRegistry("ExecutionHandler", logger)
        self._source_module = self.__class__.__name__

        # Register known execution handlers lazily
        self._register_default_handlers()

    @classmethod
    def get_instance(cls, logger: LoggerService) -> "ExecutionHandlerRegistry":
        """Get the singleton instance of the registry.

        Args:
            logger: Logger service instance

        Returns:
            The singleton registry instance
        """
        if cls._instance is None:
            cls._instance = cls(logger)
        return cls._instance

    def _register_default_handlers(self) -> None:
        """Register the default execution handler implementations."""
        # Register Kraken handler
        self._registry.register_lazy(
            name="kraken",
            module_path="gal_friday.execution.kraken",
            class_name="KrakenExecutionHandler",
            metadata={
                "description": "Kraken exchange execution handler",
                "requires_api_key": True,
                "supports_websocket": True,
            },
        )

        # Register simulated handler
        self._registry.register_lazy(
            name="simulated",
            module_path="gal_friday.simulated_execution_handler",
            class_name="SimulatedExecutionHandler",
            metadata={
                "description": "Simulated execution handler for testing",
                "requires_api_key": False,
                "supports_websocket": False,
            },
        )

        # Register backtesting handler
        self._registry.register_lazy(
            name="backtesting",
            module_path="gal_friday.backtesting_execution_handler",
            class_name="BacktestingExecutionHandler",
            metadata={
                "description": "Backtesting execution handler",
                "requires_api_key": False,
                "supports_websocket": False,
            },
        )

        self.logger.info(
            "Registered default execution handlers",
            source_module=self._source_module,
            context={"handlers": self._registry.get_available()},
        )

    def register_handler(
        self,
        name: str,
        handler_class: type[ExecutionHandlerInterface],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom execution handler.

        Args:
            name: Unique name for the handler
            handler_class: The handler class to register
            metadata: Optional metadata about the handler
        """
        self._registry.register(name, handler_class, metadata=metadata)

    def create_handler(
        self,
        name: str,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> ExecutionHandlerInterface:
        """Create an execution handler instance.

        Args:
            name: Name of the handler to create
            config: Configuration for the handler
            **kwargs: Additional arguments for the handler

        Returns:
            Configured execution handler instance

        Raises:
            KeyError: If no handler is registered with the given name
            ImportError: If the handler module cannot be imported
        """
        # Merge config into kwargs
        kwargs["config"] = config

        try:
            handler = self._registry.create(name, **kwargs)

            self.logger.info(
                f"Created execution handler '{name}'",
                source_module=self._source_module,
                context={
                    "handler_type": name,
                    "has_api_key": bool(config.get("api_key")),
                    "exchange": config.get("exchange", "unknown"),
                },
            )

        except (KeyError, ImportError) as e:
            self.logger.exception(
                f"Failed to create execution handler '{name}'",
                source_module=self._source_module,
                context={"error": str(e), "available": self._registry.get_available()},
            )
            raise
        else:
            return cast("ExecutionHandlerInterface", handler)

    def get_available_handlers(self) -> list[str]:
        """Get list of available execution handler names."""
        return self._registry.get_available()

    def get_handler_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a specific handler.

        Args:
            name: Name of the handler

        Returns:
            Handler metadata dictionary
        """
        return self._registry.get_metadata(name)

    def is_handler_available(self, name: str) -> bool:
        """Check if a handler is available.

        Args:
            name: Name of the handler

        Returns:
            True if the handler is registered
        """
        return self._registry.is_available(name)


def get_execution_handler_registry(logger: LoggerService) -> ExecutionHandlerRegistry:
    """Get the global execution handler registry.

    Args:
        logger: Logger service instance

    Returns:
        The global execution handler registry
    """
    return ExecutionHandlerRegistry.get_instance(logger)
