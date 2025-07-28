"""Registry for data provider implementations.

This module provides a registry pattern for data providers to avoid
circular imports in the backtesting engine.
"""

from typing import Any, cast

from gal_friday.core.registry import ComponentRegistry
from gal_friday.interfaces.data_provider_protocols import DataProviderRegistryProtocol, HistoricalDataProviderProtocol
from gal_friday.logger_service import LoggerService


class DataProviderRegistry(DataProviderRegistryProtocol):
    """Registry for data provider implementations."""

    _instance: "DataProviderRegistry | None" = None

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the data provider registry.

        Args:
            logger: Logger service instance
        """
        self.logger = logger
        self._registry = ComponentRegistry("DataProvider", logger)
        self._source_module = self.__class__.__name__

        # Register known providers lazily
        self._register_default_providers()

    @classmethod
    def get_instance(cls, logger: LoggerService) -> "DataProviderRegistry":
        """Get the singleton instance of the registry.

        Args:
            logger: Logger service instance

        Returns:
            The singleton registry instance
        """
        if cls._instance is None:
            cls._instance = cls(logger)
        return cls._instance

    def _register_default_providers(self) -> None:
        """Register the default data provider implementations."""
        # Register database provider
        self._registry.register_lazy(
            name="database",
            module_path="gal_friday.providers.database_provider",
            class_name="DatabaseDataProvider",
            metadata={
                "description": "Database-based historical data provider",
                "requires_database": True,
            },
        )

        # Register API provider
        self._registry.register_lazy(
            name="api",
            module_path="gal_friday.providers.api_provider",
            class_name="APIDataProvider",
            metadata={
                "description": "API-based historical data provider",
                "requires_api_key": True,
            },
        )

        self.logger.info(
            "Registered default data providers",
            source_module=self._source_module,
            context={"providers": self._registry.get_available()},
        )

    def create_provider(
        self,
        name: str,
        config: dict[str, Any],
        logger: LoggerService,
        **kwargs: Any,
    ) -> "HistoricalDataProviderProtocol":
        """Create a data provider instance.

        Args:
            name: Name of the provider to create
            config: Configuration for the provider
            logger: Logger service instance
            **kwargs: Additional arguments for the provider

        Returns:
            Configured data provider instance

        Raises:
            KeyError: If no provider is registered with the given name
            ImportError: If the provider module cannot be imported
        """
        try:
            provider = self._registry.create(
                name,
                config=config,
                logger=logger,
                **kwargs,
            )

            self.logger.info(
                f"Created data provider '{name}'",
                source_module=self._source_module,
                context={"provider_type": name},
            )

        except (KeyError, ImportError) as e:
            self.logger.exception(
                f"Failed to create data provider '{name}'",
                source_module=self._source_module,
                context={"error": str(e), "available": self._registry.get_available()},
            )
            raise
        else:
            return cast("HistoricalDataProviderProtocol", provider)

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return self._registry.get_available()

    async def fetch_historical_data(
        self,
        trading_pair: str,
        start_time: Any,
        end_time: Any,
        interval: str,
    ) -> Any:
        """Fetch historical data for a trading pair.

        This method is required by the DataProviderRegistryProtocol but is not
        implemented in the registry itself. Users should create specific provider
        instances using create_provider() method.
        """
        raise NotImplementedError("Use create_provider() to get a specific provider instance")


def get_data_provider_registry(logger: LoggerService) -> DataProviderRegistry:
    """Get the global data provider registry.

    Args:
        logger: Logger service instance

    Returns:
        The global data provider registry
    """
    return DataProviderRegistry.get_instance(logger)
