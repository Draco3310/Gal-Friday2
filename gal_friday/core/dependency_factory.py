"""Dependency injection factory for core components.

This module provides a factory pattern for creating and wiring up
application components with their dependencies, eliminating circular imports.
"""

from typing import TYPE_CHECKING, Any, cast

from gal_friday.interfaces.core_interfaces import (
    ConfigManagerInterface,
    ExchangeInfoServiceInterface,
    LoggerServiceInterface,
    MonitoringServiceInterface,
    PortfolioManagerInterface,
    PubSubManagerInterface,
    RiskManagerInterface,
    StrategyArbitratorInterface,
)
from gal_friday.utils.lazy_imports import LazyImport

if TYPE_CHECKING:
    from gal_friday.interfaces import (
        ExecutionHandlerInterface,
        FeatureEngineInterface,
        HistoricalDataService,
        MarketPriceService,
    )

# Lazy imports to avoid circular dependencies
_portfolio_manager = LazyImport("gal_friday.portfolio_manager.PortfolioManager")
_risk_manager = LazyImport("gal_friday.risk_manager.RiskManager")
_monitoring_service = LazyImport("gal_friday.monitoring_service.MonitoringService")
_strategy_arbitrator = LazyImport("gal_friday.strategy_arbitrator.StrategyArbitrator")
_backtesting_engine = LazyImport("gal_friday.backtesting_engine.BacktestingEngine")


class DependencyContainer:
    """Container for application dependencies.

    This container holds references to all major application components
    and provides them to other components that need them, breaking
    circular dependencies.
    """

    def __init__(self) -> None:
        """Initialize the dependency container."""
        # Core services
        self.config_manager: ConfigManagerInterface | None = None
        self.logger_service: LoggerServiceInterface | None = None
        self.pubsub_manager: PubSubManagerInterface | None = None

        # Trading components
        self.portfolio_manager: PortfolioManagerInterface | None = None
        self.risk_manager: RiskManagerInterface | None = None
        self.monitoring_service: MonitoringServiceInterface | None = None
        self.strategy_arbitrator: StrategyArbitratorInterface | None = None
        self.exchange_info_service: ExchangeInfoServiceInterface | None = None

        # Market data and execution
        self.market_price_service: MarketPriceService | None = None
        self.execution_handler: ExecutionHandlerInterface | None = None
        self.historical_data_service: HistoricalDataService | None = None
        self.feature_engine: FeatureEngineInterface | None = None

        # Additional services can be added here
        self._custom_services: dict[str, Any] = {}

    def register_service(self, name: str, service: Any) -> None:
        """Register a custom service.

        Args:
            name: Unique name for the service
            service: The service instance
        """
        self._custom_services[name] = service

    def get_service(self, name: str) -> Any:
        """Get a custom service by name.

        Args:
            name: Name of the service

        Returns:
            The service instance

        Raises:
            KeyError: If no service is registered with the given name
        """
        if name not in self._custom_services:
            raise KeyError(f"No service registered with name '{name}'")
        return self._custom_services[name]

    def validate(self) -> None:
        """Validate that all required dependencies are set.

        Raises:
            ValueError: If any required dependency is missing
        """
        required = {
            "config_manager": self.config_manager,
            "logger_service": self.logger_service,
            "pubsub_manager": self.pubsub_manager,
        }

        missing = [name for name, service in required.items() if service is None]
        if missing:
            raise ValueError(f"Missing required dependencies: {', '.join(missing)}")


class DependencyFactory:
    """Factory for creating application components with dependency injection."""

    def __init__(self, container: DependencyContainer) -> None:
        """Initialize the dependency factory.

        Args:
            container: The dependency container with core services
        """
        self.container = container
        container.validate()

    def create_portfolio_manager(self, **kwargs: Any) -> PortfolioManagerInterface:
        """Create a portfolio manager with injected dependencies.

        Args:
            **kwargs: Additional arguments for the portfolio manager

        Returns:
            Configured portfolio manager instance
        """
        portfolio_manager = _portfolio_manager.get_module()

        result = portfolio_manager(
            config=self.container.config_manager.get_config() if self.container.config_manager else {},
            pubsub_manager=self.container.pubsub_manager,
            logger_service=self.container.logger_service,
            **kwargs,
        )
        return cast("PortfolioManagerInterface", result)

    def create_risk_manager(self, **kwargs: Any) -> RiskManagerInterface:
        """Create a risk manager with injected dependencies.

        Args:
            **kwargs: Additional arguments for the risk manager

        Returns:
            Configured risk manager instance
        """
        risk_manager = _risk_manager.get_module()

        # Risk manager needs portfolio manager
        if not self.container.portfolio_manager:
            self.container.portfolio_manager = self.create_portfolio_manager()

        result = risk_manager(
            config=self.container.config_manager.get_config() if self.container.config_manager else {},
            portfolio_manager=self.container.portfolio_manager,
            pubsub_manager=self.container.pubsub_manager,
            logger_service=self.container.logger_service,
            market_price_service=self.container.market_price_service,
            exchange_info_service=self.container.exchange_info_service,
            **kwargs,
        )
        return cast("RiskManagerInterface", result)

    def create_monitoring_service(self, **kwargs: Any) -> MonitoringServiceInterface:
        """Create a monitoring service with injected dependencies.

        Args:
            **kwargs: Additional arguments for the monitoring service

        Returns:
            Configured monitoring service instance
        """
        monitoring_service = _monitoring_service.get_module()

        # Monitoring service needs portfolio manager
        if not self.container.portfolio_manager:
            self.container.portfolio_manager = self.create_portfolio_manager()

        result = monitoring_service(
            config_manager=self.container.config_manager,
            pubsub_manager=self.container.pubsub_manager,
            portfolio_manager=self.container.portfolio_manager,
            logger_service=self.container.logger_service,
            execution_handler=self.container.execution_handler,
            **kwargs,
        )
        return cast("MonitoringServiceInterface", result)

    def create_strategy_arbitrator(self, **kwargs: Any) -> StrategyArbitratorInterface:
        """Create a strategy arbitrator with injected dependencies.

        Args:
            **kwargs: Additional arguments for the strategy arbitrator

        Returns:
            Configured strategy arbitrator instance
        """
        strategy_arbitrator = _strategy_arbitrator.get_module()

        # Strategy arbitrator needs several dependencies
        if not self.container.portfolio_manager:
            self.container.portfolio_manager = self.create_portfolio_manager()

        if not self.container.risk_manager:
            self.container.risk_manager = self.create_risk_manager()

        result = strategy_arbitrator(
            config=self.container.config_manager.get_config() if self.container.config_manager else {},
            pubsub_manager=self.container.pubsub_manager,
            logger_service=self.container.logger_service,
            portfolio_manager=self.container.portfolio_manager,
            risk_manager=self.container.risk_manager,
            **kwargs,
        )
        return cast("StrategyArbitratorInterface", result)

    def create_backtesting_engine(self, **kwargs: Any) -> Any:
        """Create a backtesting engine with injected dependencies.

        Args:
            **kwargs: Additional arguments for the backtesting engine

        Returns:
            Configured backtesting engine instance
        """
        backtesting_engine = _backtesting_engine.get_module()

        # Ensure all required dependencies exist
        if not self.container.portfolio_manager:
            self.container.portfolio_manager = self.create_portfolio_manager()

        if not self.container.risk_manager:
            self.container.risk_manager = self.create_risk_manager()

        if not self.container.monitoring_service:
            self.container.monitoring_service = self.create_monitoring_service()

        if not self.container.strategy_arbitrator:
            self.container.strategy_arbitrator = self.create_strategy_arbitrator()

        return backtesting_engine(
            config=self.container.config_manager.get_config() if self.container.config_manager else {},
            logger_service=self.container.logger_service,
            pubsub_manager=self.container.pubsub_manager,
            portfolio_manager=self.container.portfolio_manager,
            risk_manager=self.container.risk_manager,
            monitoring_service=self.container.monitoring_service,
            strategy_arbitrator=self.container.strategy_arbitrator,
            market_price_service=self.container.market_price_service,
            execution_handler=self.container.execution_handler,
            historical_data_service=self.container.historical_data_service,
            feature_engine=self.container.feature_engine,
            **kwargs,
        )
