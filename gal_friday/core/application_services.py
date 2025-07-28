"""Application service registration for the GalFriday trading system.

This module provides centralized registration of all application services
with the dependency injection container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from gal_friday.core.service_container import ServiceContainer, ServiceLifecycle
from gal_friday.core.service_registration import ServiceRegistrar

if TYPE_CHECKING:
    from gal_friday.config_manager import ConfigManager


class ApplicationServiceRegistrar:
    """Centralized application service registration."""

    def __init__(self, container: ServiceContainer, config_manager: ConfigManager) -> None:
        """Initialize the application service registrar.

        Args:
            container: Service container
            config_manager: Configuration manager
        """
        self._container = container
        self._config_manager = config_manager
        self._registrar = ServiceRegistrar(container)

    def register_all_services(self) -> None:
        """Register all application services."""
        # Register core services first
        self._register_core_services()

        # Register data access services
        self._register_data_services()

        # Register business services
        self._register_business_services()

        # Register execution services
        self._register_execution_services()

        # Register monitoring services
        self._register_monitoring_services()

        # Register conditional services based on environment
        self._register_conditional_services()

    def _register_core_services(self) -> None:
        """Register core infrastructure services."""
        # ConfigManager (already exists)
        self._registrar.register_instance(
            service_type=type(self._config_manager),
            instance=self._config_manager,
            tags={"core", "infrastructure"},
        )

        # Register PubSubManager
        self._registrar.register(
            service_type=self._get_type("gal_friday.core.pubsub", "PubSubManager"),
            implementation_type=self._get_type("gal_friday.core.pubsub", "PubSubManager"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"core", "messaging"},
        )

        # Register LoggerService
        self._registrar.register(
            service_type=self._get_type("gal_friday.logger_service", "LoggerService"),
            implementation_type=self._get_type("gal_friday.logger_service", "LoggerService"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"core", "logging"},
        )

        # Register MonitoringService
        self._registrar.register(
            service_type=self._get_type("gal_friday.monitoring_service", "MonitoringService"),
            implementation_type=self._get_type("gal_friday.monitoring_service", "MonitoringService"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"core", "monitoring"},
        )

    def _register_data_services(self) -> None:
        """Register data access services."""
        # Database connection pool
        self._registrar.register(
            service_type=self._get_type("gal_friday.dal.connection_pool", "DatabaseConnectionPool"),
            implementation_type=self._get_type("gal_friday.dal.connection_pool", "DatabaseConnectionPool"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"data", "database"},
        )

        # Migration manager
        self._registrar.register(
            service_type=self._get_type("gal_friday.dal.migrations.migration_manager", "MigrationManager"),
            implementation_type=self._get_type("gal_friday.dal.migrations.migration_manager", "MigrationManager"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"data", "migration"},
        )

        # Data ingestor
        self._registrar.register(
            service_type=self._get_type("gal_friday.data_ingestor", "DataIngestor"),
            implementation_type=self._get_type("gal_friday.data_ingestor", "DataIngestor"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"data", "ingestion"},
        )

        # Historical data service
        self._register_historical_data_service()

    def _register_business_services(self) -> None:
        """Register business logic services."""
        # Feature engine
        self._registrar.register(
            service_type=self._get_type("gal_friday.feature_engine", "FeatureEngine"),
            implementation_type=self._get_type("gal_friday.feature_engine", "FeatureEngine"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"business", "features"},
        )

        # Prediction service
        self._registrar.register(
            service_type=self._get_type("gal_friday.prediction_service", "PredictionService"),
            implementation_type=self._get_type("gal_friday.prediction_service", "PredictionService"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"business", "prediction"},
        )

        # Strategy arbitrator
        self._registrar.register(
            service_type=self._get_type("gal_friday.strategy_arbitrator", "StrategyArbitrator"),
            implementation_type=self._get_type("gal_friday.strategy_arbitrator", "StrategyArbitrator"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"business", "strategy"},
        )

        # Risk manager
        self._registrar.register(
            service_type=self._get_type("gal_friday.risk_manager", "RiskManager"),
            implementation_type=self._get_type("gal_friday.risk_manager", "RiskManager"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"business", "risk"},
        )

        # Portfolio manager
        self._registrar.register(
            service_type=self._get_type("gal_friday.portfolio_manager", "PortfolioManager"),
            implementation_type=self._get_type("gal_friday.portfolio_manager", "PortfolioManager"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"business", "portfolio"},
        )

    def _register_execution_services(self) -> None:
        """Register execution services."""
        # Market price service
        self._register_market_price_service()

        # Execution handler
        self._register_execution_handler()

    def _register_monitoring_services(self) -> None:
        """Register monitoring and management services."""
        # CLI service
        self._registrar.register(
            service_type=self._get_type("gal_friday.cli_service", "CLIService"),
            implementation_type=self._get_type("gal_friday.cli_service", "CLIService"),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"monitoring", "cli"},
        )

    def _register_conditional_services(self) -> None:
        """Register services based on environment and configuration."""
        # Register environment-specific services
        environment = self._config_manager.get("environment", "development")

        if environment == "development":
            self._register_development_services()
        elif environment == "staging":
            self._register_staging_services()
        elif environment == "production":
            self._register_production_services()

    def _register_development_services(self) -> None:
        """Register development-specific services."""
        # Mock services for development

    def _register_staging_services(self) -> None:
        """Register staging-specific services."""
        # Staging-specific configurations

    def _register_production_services(self) -> None:
        """Register production-specific services."""
        # Production-specific configurations

    def _register_market_price_service(self) -> None:
        """Register market price service based on configuration."""
        mode = self._config_manager.get("operational_mode", "paper_trading")

        if mode in {"live_trading", "live"}:
            # Register live market price service
            self._registrar.register_conditional(
                service_type=self._get_type("gal_friday.market_price_service", "MarketPriceService"),
                implementation_type=self._get_type(
                    "gal_friday.market_price.kraken_service",
                    "KrakenMarketPriceService",
                ),
                condition=lambda: mode in {"live_trading", "live"},
                lifecycle=ServiceLifecycle.SINGLETON,
                tags={"market", "live"},
            )
        else:
            # Register simulated market price service
            self._registrar.register_conditional(
                service_type=self._get_type("gal_friday.market_price_service", "MarketPriceService"),
                implementation_type=self._get_type(
                    "gal_friday.simulated_market_price_service",
                    "SimulatedMarketPriceService",
                ),
                condition=lambda: mode not in {"live_trading", "live"},
                lifecycle=ServiceLifecycle.SINGLETON,
                tags={"market", "simulated"},
            )

    def _register_execution_handler(self) -> None:
        """Register execution handler based on configuration."""
        mode = self._config_manager.get("operational_mode", "paper_trading")

        if mode in {"live_trading", "live"}:
            # Register live execution handler
            self._registrar.register_conditional(
                service_type=self._get_type("gal_friday.execution_handler", "ExecutionHandler"),
                implementation_type=self._get_type("gal_friday.execution.kraken", "KrakenExecutionHandler"),
                condition=lambda: mode in {"live_trading", "live"},
                lifecycle=ServiceLifecycle.SINGLETON,
                tags={"execution", "live"},
            )
        else:
            # Register simulated execution handler
            self._registrar.register_conditional(
                service_type=self._get_type("gal_friday.execution_handler", "ExecutionHandler"),
                implementation_type=self._get_type(
                    "gal_friday.simulated_execution_handler",
                    "SimulatedExecutionHandler",
                ),
                condition=lambda: mode not in {"live_trading", "live"},
                lifecycle=ServiceLifecycle.SINGLETON,
                tags={"execution", "simulated"},
            )

    def _register_historical_data_service(self) -> None:
        """Register historical data service."""
        # Default to Kraken historical data service
        self._registrar.register(
            service_type=self._get_type("gal_friday.historical_data_service", "HistoricalDataService"),
            implementation_type=self._get_type(
                "gal_friday.kraken_historical_data_service",
                "KrakenHistoricalDataService",
            ),
            lifecycle=ServiceLifecycle.SINGLETON,
            tags={"data", "historical"},
        )

    def _get_type(self, module_name: str, class_name: str) -> type:
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

    def validate_registrations(self) -> list[str]:
        """Validate all service registrations.

        Returns:
            List of validation errors
        """
        return self._container.validate_registrations()


def create_application_container(config_manager: ConfigManager) -> ServiceContainer:
    """Create and configure the application service container.

    Args:
        config_manager: Configuration manager

    Returns:
        Configured service container
    """
    # Create container
    container = ServiceContainer(config_manager)

    # Create registrar and register all services
    registrar = ApplicationServiceRegistrar(container, config_manager)
    registrar.register_all_services()

    # Validate registrations
    errors = registrar.validate_registrations()
    if errors:
        error_msg = "Service registration validation failed:\n" + "\n".join(errors)
        raise RuntimeError(error_msg)

    return container


def register_core_services_only(config_manager: ConfigManager) -> ServiceContainer:
    """Register only core services for testing or minimal setups.

    Args:
        config_manager: Configuration manager

    Returns:
        Container with core services only
    """
    container = ServiceContainer(config_manager)
    registrar = ApplicationServiceRegistrar(container, config_manager)

    # Register only core services
    registrar._register_core_services()  # noqa: SLF001

    return container


# Service factory functions for common patterns
def create_logger_service_factory(container: ServiceContainer) -> Any:
    """Factory function for LoggerService.

    Args:
        container: Service container

    Returns:
        LoggerService instance
    """

    async def factory() -> Any:
        config_manager = await container.resolve(type(container._config_manager))  # noqa: SLF001
        pubsub_manager: Any = await container.resolve(
            container.get_type("gal_friday.core.pubsub", "PubSubManager"),
        )

        logger_service_type = container.get_type("gal_friday.logger_service", "LoggerService")
        return logger_service_type(
            config_manager=config_manager,
            pubsub_manager=pubsub_manager,
        )

    return factory


def create_monitoring_service_factory(container: ServiceContainer) -> Any:
    """Factory function for MonitoringService.

    Args:
        container: Service container

    Returns:
        MonitoringService instance
    """

    async def factory() -> Any:
        config_manager = await container.resolve(type(container._config_manager))  # noqa: SLF001
        logger_service: Any = await container.resolve(
            container.get_type("gal_friday.logger_service", "LoggerService"),
        )

        monitoring_service_type = container.get_type("gal_friday.monitoring_service", "MonitoringService")
        return monitoring_service_type(
            config_manager=config_manager,
            logger_service=logger_service,
        )

    return factory
