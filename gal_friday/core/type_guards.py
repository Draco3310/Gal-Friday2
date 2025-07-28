"""Enhanced type guards and validation patterns for type-safe initialization.

This module provides comprehensive type guards that go beyond basic None checks
to ensure type safety throughout the application lifecycle.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, cast

from gal_friday.exceptions import DependencyMissingError

if TYPE_CHECKING:
    from collections.abc import Callable

    from gal_friday.cli_service import CLIService
    from gal_friday.config_manager import ConfigManager
    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.core.service_container import ServiceContainer
    from gal_friday.interfaces.market_data_protocols import (
        MarketDataAggregatorProtocol,
        MarketDataCacheProtocol,
        MarketDataProviderProtocol,
        MarketDataStreamProtocol,
        MarketDataValidatorProtocol,
        MarketPriceServiceProtocol,
    )
    from gal_friday.logger_service import LoggerService
    from gal_friday.monitoring_service import MonitoringService
    from gal_friday.portfolio_manager import PortfolioManager

T = TypeVar("T")


class TypeGuardError(Exception):
    """Raised when a type guard fails."""

    def __init__(self, expected_type: type, actual_value: Any, context: str = "") -> None:
        """Initialize type guard error."""
        super().__init__(f"Expected {expected_type.__name__}, got {type(actual_value).__name__}")
        self.expected_type = expected_type
        self.actual_value = actual_value
        self.context = context


class EnhancedTypeGuards:
    """Enhanced type guards for common application types."""

    @staticmethod
    def is_config_manager(value: Any) -> TypeGuard[ConfigManager]:
        """Type guard for ConfigManager."""
        return (
            value is not None
            and hasattr(value, "get")
            and hasattr(value, "is_valid")
            and callable(value.get)
            and callable(value.is_valid)
        )

    @staticmethod
    def is_service_container(value: Any) -> TypeGuard[ServiceContainer]:
        """Type guard for ServiceContainer."""
        return (
            value is not None
            and hasattr(value, "resolve")
            and hasattr(value, "is_registered")
            and hasattr(value, "get_type")
            and callable(value.resolve)
            and callable(value.is_registered)
            and callable(value.get_type)
        )

    @staticmethod
    def is_pubsub_manager(value: Any) -> TypeGuard[PubSubManager]:
        """Type guard for PubSubManager."""
        return (
            value is not None
            and hasattr(value, "publish")
            and hasattr(value, "subscribe")
            and callable(value.publish)
            and callable(value.subscribe)
        )

    @staticmethod
    def is_logger_service(value: Any) -> TypeGuard[LoggerService]:
        """Type guard for LoggerService."""
        return (
            value is not None
            and hasattr(value, "info")
            and hasattr(value, "error")
            and hasattr(value, "debug")
            and callable(value.info)
            and callable(value.error)
            and callable(value.debug)
        )

    @staticmethod
    def is_monitoring_service(value: Any) -> TypeGuard[MonitoringService]:
        """Type guard for MonitoringService."""
        return (
            value is not None
            and hasattr(value, "start")
            and hasattr(value, "stop")
            and callable(value.start)
            and callable(value.stop)
        )

    @staticmethod
    def is_portfolio_manager(value: Any) -> TypeGuard[PortfolioManager]:
        """Type guard for PortfolioManager."""
        return (
            value is not None
            and hasattr(value, "get_portfolio")
            and hasattr(value, "update_position")
            and callable(value.get_portfolio)
            and callable(value.update_position)
        )

    @staticmethod
    def is_cli_service(value: Any) -> TypeGuard[CLIService]:
        """Type guard for CLIService."""
        return (
            value is not None
            and hasattr(value, "start")
            and hasattr(value, "stop")
            and callable(value.start)
            and callable(value.stop)
        )

    @staticmethod
    def is_callable_factory(value: Any) -> TypeGuard[Callable[..., Any]]:
        """Type guard for callable factory functions."""
        return callable(value)

    @staticmethod
    def is_service_with_lifecycle(value: Any) -> TypeGuard[Any]:
        """Type guard for services with start/stop lifecycle."""
        return (
            value is not None
            and hasattr(value, "start")
            and hasattr(value, "stop")
            and callable(value.start)
            and callable(value.stop)
        )

    @staticmethod
    def is_async_callable(value: Any) -> TypeGuard[Callable[..., Any]]:
        """Type guard for async callable functions."""
        return callable(value) and inspect.iscoroutinefunction(value)

    @staticmethod
    def is_class_type(value: Any) -> TypeGuard[type]:
        """Type guard for class types."""
        return inspect.isclass(value)

    @staticmethod
    def has_required_attributes(value: Any, attributes: list[str]) -> bool:
        """Check if value has all required attributes."""
        if value is None:
            return False

        return all(hasattr(value, attr) for attr in attributes)

    @staticmethod
    def has_required_methods(value: Any, methods: list[str]) -> bool:
        """Check if value has all required methods."""
        if value is None:
            return False

        return all(hasattr(value, method) and callable(getattr(value, method)) for method in methods)

    @staticmethod
    def is_market_price_service(value: Any) -> TypeGuard[MarketPriceServiceProtocol]:
        """Type guard for MarketPriceServiceProtocol."""
        return (
            value is not None
            and hasattr(value, "start")
            and hasattr(value, "stop")
            and hasattr(value, "get_latest_price")
            and hasattr(value, "get_bid_ask_spread")
            and hasattr(value, "get_price_timestamp")
            and hasattr(value, "is_price_fresh")
            and hasattr(value, "convert_amount")
            and hasattr(value, "get_historical_ohlcv")
            and callable(value.start)
            and callable(value.stop)
            and callable(value.get_latest_price)
            and callable(value.get_bid_ask_spread)
            and callable(value.get_price_timestamp)
            and callable(value.is_price_fresh)
            and callable(value.convert_amount)
            and callable(value.get_historical_ohlcv)
        )

    @staticmethod
    def is_market_data_provider(value: Any) -> TypeGuard[MarketDataProviderProtocol]:
        """Type guard for MarketDataProviderProtocol."""
        return (
            value is not None
            and hasattr(value, "start")
            and hasattr(value, "stop")
            and hasattr(value, "get_supported_symbols")
            and hasattr(value, "subscribe_to_ticker")
            and hasattr(value, "unsubscribe_from_ticker")
            and hasattr(value, "get_ticker_data")
            and hasattr(value, "get_order_book")
            and hasattr(value, "get_recent_trades")
            and callable(value.start)
            and callable(value.stop)
            and callable(value.get_supported_symbols)
            and callable(value.subscribe_to_ticker)
            and callable(value.unsubscribe_from_ticker)
            and callable(value.get_ticker_data)
            and callable(value.get_order_book)
            and callable(value.get_recent_trades)
        )

    @staticmethod
    def is_market_data_stream(value: Any) -> TypeGuard[MarketDataStreamProtocol]:
        """Type guard for MarketDataStreamProtocol."""
        return (
            value is not None
            and hasattr(value, "start_stream")
            and hasattr(value, "stop_stream")
            and hasattr(value, "add_symbol")
            and hasattr(value, "remove_symbol")
            and hasattr(value, "get_stream_events")
            and hasattr(value, "is_streaming")
            and hasattr(value, "get_active_symbols")
            and callable(value.start_stream)
            and callable(value.stop_stream)
            and callable(value.add_symbol)
            and callable(value.remove_symbol)
            and callable(value.get_stream_events)
            and callable(value.is_streaming)
            and callable(value.get_active_symbols)
        )

    @staticmethod
    def is_market_data_cache(value: Any) -> TypeGuard[MarketDataCacheProtocol]:
        """Type guard for MarketDataCacheProtocol."""
        return (
            value is not None
            and hasattr(value, "get_cached_price")
            and hasattr(value, "cache_price")
            and hasattr(value, "get_cached_ticker")
            and hasattr(value, "cache_ticker")
            and hasattr(value, "get_cached_order_book")
            and hasattr(value, "cache_order_book")
            and hasattr(value, "invalidate_cache")
            and hasattr(value, "clear_cache")
            and hasattr(value, "get_cache_stats")
            and callable(value.get_cached_price)
            and callable(value.cache_price)
            and callable(value.get_cached_ticker)
            and callable(value.cache_ticker)
            and callable(value.get_cached_order_book)
            and callable(value.cache_order_book)
            and callable(value.invalidate_cache)
            and callable(value.clear_cache)
            and callable(value.get_cache_stats)
        )

    @staticmethod
    def is_market_data_aggregator(value: Any) -> TypeGuard[MarketDataAggregatorProtocol]:
        """Type guard for MarketDataAggregatorProtocol."""
        return (
            value is not None
            and hasattr(value, "add_provider")
            and hasattr(value, "remove_provider")
            and hasattr(value, "get_best_price")
            and hasattr(value, "get_aggregated_ticker")
            and hasattr(value, "get_best_order_book")
            and hasattr(value, "get_active_providers")
            and hasattr(value, "health_check")
            and callable(value.add_provider)
            and callable(value.remove_provider)
            and callable(value.get_best_price)
            and callable(value.get_aggregated_ticker)
            and callable(value.get_best_order_book)
            and callable(value.get_active_providers)
            and callable(value.health_check)
        )

    @staticmethod
    def is_market_data_validator(value: Any) -> TypeGuard[MarketDataValidatorProtocol]:
        """Type guard for MarketDataValidatorProtocol."""
        return (
            value is not None
            and hasattr(value, "validate_price")
            and hasattr(value, "validate_ticker")
            and hasattr(value, "validate_order_book")
            and hasattr(value, "detect_anomalies")
            and hasattr(value, "get_validation_rules")
            and hasattr(value, "update_validation_rules")
            and callable(value.validate_price)
            and callable(value.validate_ticker)
            and callable(value.validate_order_book)
            and callable(value.detect_anomalies)
            and callable(value.get_validation_rules)
            and callable(value.update_validation_rules)
        )


class TypeSafeAssertion:
    """Type-safe assertion utilities."""

    @staticmethod
    def assert_config_manager(value: Any, context: str = "") -> ConfigManager:
        """Assert that value is a ConfigManager."""
        if not EnhancedTypeGuards.is_config_manager(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_service_container(value: Any, context: str = "") -> ServiceContainer:
        """Assert that value is a ServiceContainer."""
        if not EnhancedTypeGuards.is_service_container(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_pubsub_manager(value: Any, context: str = "") -> PubSubManager:
        """Assert that value is a PubSubManager."""
        if not EnhancedTypeGuards.is_pubsub_manager(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_logger_service(value: Any, context: str = "") -> LoggerService:
        """Assert that value is a LoggerService."""
        if not EnhancedTypeGuards.is_logger_service(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_monitoring_service(value: Any, context: str = "") -> MonitoringService:
        """Assert that value is a MonitoringService."""
        if not EnhancedTypeGuards.is_monitoring_service(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_portfolio_manager(value: Any, context: str = "") -> PortfolioManager:
        """Assert that value is a PortfolioManager."""
        if not EnhancedTypeGuards.is_portfolio_manager(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_cli_service(value: Any, context: str = "") -> CLIService:
        """Assert that value is a CLIService."""
        if not EnhancedTypeGuards.is_cli_service(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_not_none(value: T | None, context: str = "") -> T:
        """Assert that value is not None."""
        if value is None:
            raise TypeGuardError(type(value), value, context)
        return value

    @staticmethod
    def assert_type(value: Any, expected_type: type[T], context: str = "") -> T:
        """Assert that value is of expected type."""
        if not isinstance(value, expected_type):
            raise TypeGuardError(expected_type, value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_callable(value: Any, context: str = "") -> Callable[..., Any]:
        """Assert that value is callable."""
        if not callable(value):
            raise TypeGuardError(type(value), value, context)
        return cast("Callable[..., Any]", value)

    @staticmethod
    def assert_has_attributes(value: Any, attributes: list[str], context: str = "") -> Any:
        """Assert that value has all required attributes."""
        if not EnhancedTypeGuards.has_required_attributes(value, attributes):
            missing_attrs = [attr for attr in attributes if not hasattr(value, attr)]
            raise TypeGuardError(
                type(value),
                value,
                f"{context}: Missing attributes: {missing_attrs}"
                if context
                else f"Missing attributes: {missing_attrs}",
            )
        return value

    @staticmethod
    def assert_has_methods(value: Any, methods: list[str], context: str = "") -> Any:
        """Assert that value has all required methods."""
        if not EnhancedTypeGuards.has_required_methods(value, methods):
            missing_methods = [
                method for method in methods if not (hasattr(value, method) and callable(getattr(value, method)))
            ]
            raise TypeGuardError(
                type(value),
                value,
                f"{context}: Missing methods: {missing_methods}" if context else f"Missing methods: {missing_methods}",
            )
        return value

    @staticmethod
    def assert_market_price_service(value: Any, context: str = "") -> MarketPriceServiceProtocol:
        """Assert that value is a MarketPriceServiceProtocol."""
        if not EnhancedTypeGuards.is_market_price_service(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_market_data_provider(value: Any, context: str = "") -> MarketDataProviderProtocol:
        """Assert that value is a MarketDataProviderProtocol."""
        if not EnhancedTypeGuards.is_market_data_provider(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_market_data_stream(value: Any, context: str = "") -> MarketDataStreamProtocol:
        """Assert that value is a MarketDataStreamProtocol."""
        if not EnhancedTypeGuards.is_market_data_stream(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_market_data_cache(value: Any, context: str = "") -> MarketDataCacheProtocol:
        """Assert that value is a MarketDataCacheProtocol."""
        if not EnhancedTypeGuards.is_market_data_cache(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_market_data_aggregator(value: Any, context: str = "") -> MarketDataAggregatorProtocol:
        """Assert that value is a MarketDataAggregatorProtocol."""
        if not EnhancedTypeGuards.is_market_data_aggregator(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard

    @staticmethod
    def assert_market_data_validator(value: Any, context: str = "") -> MarketDataValidatorProtocol:
        """Assert that value is a MarketDataValidatorProtocol."""
        if not EnhancedTypeGuards.is_market_data_validator(value):
            raise TypeGuardError(type(value), value, context)
        return value  # Type already validated by type guard


class DependencyTypeGuards:
    """Type guards specifically for dependency injection scenarios."""

    @staticmethod
    def validate_service_dependency(service: Any, service_name: str, required_methods: list[str] | None = None) -> None:
        """Validate that a service meets dependency requirements."""
        if service is None:
            raise DependencyMissingError(
                component="Application",
                dependency=service_name,
                message=f"Service {service_name} is None",
            )

        if required_methods:
            missing_methods = [
                method
                for method in required_methods
                if not hasattr(service, method) or not callable(getattr(service, method))
            ]
            if missing_methods:
                raise DependencyMissingError(
                    component="Application",
                    dependency=service_name,
                    message=f"Service {service_name} missing required methods: {missing_methods}",
                )

    @staticmethod
    def validate_config_dependency(config: Any, required_sections: list[str] | None = None) -> ConfigManager:
        """Validate configuration dependency."""
        if not EnhancedTypeGuards.is_config_manager(config):
            raise DependencyMissingError(
                component="Application",
                dependency="ConfigManager",
                message="ConfigManager is not properly initialized",
            )

        config_manager = config  # Type already validated

        if required_sections:
            missing_sections = [section for section in required_sections if not config_manager.get(section)]
            if missing_sections:
                raise DependencyMissingError(
                    component="Application",
                    dependency="ConfigManager",
                    message=f"ConfigManager missing required sections: {missing_sections}",
                )

        return config_manager

    @staticmethod
    def validate_container_dependency(container: Any, required_services: list[str] | None = None) -> ServiceContainer:
        """Validate service container dependency."""
        if not EnhancedTypeGuards.is_service_container(container):
            raise DependencyMissingError(
                component="Application",
                dependency="ServiceContainer",
                message="ServiceContainer is not properly initialized",
            )

        service_container = container  # Type already validated

        if required_services:
            missing_services = []
            for service_name in required_services:
                try:
                    service_type = service_container.get_type("gal_friday." + service_name.lower(), service_name)
                    if not service_container.is_registered(service_type):
                        missing_services.append(service_name)
                except Exception:
                    missing_services.append(service_name)

            if missing_services:
                raise DependencyMissingError(
                    component="Application",
                    dependency="ServiceContainer",
                    message=f"ServiceContainer missing required services: {missing_services}",
                )

        return service_container


# Convenience functions
def ensure_config_manager(value: Any) -> ConfigManager:
    """Ensure value is a ConfigManager with proper error handling."""
    return TypeSafeAssertion.assert_config_manager(value, "ConfigManager validation")


def ensure_service_container(value: Any) -> ServiceContainer:
    """Ensure value is a ServiceContainer with proper error handling."""
    return TypeSafeAssertion.assert_service_container(value, "ServiceContainer validation")


def ensure_pubsub_manager(value: Any) -> PubSubManager:
    """Ensure value is a PubSubManager with proper error handling."""
    return TypeSafeAssertion.assert_pubsub_manager(value, "PubSubManager validation")


def ensure_logger_service(value: Any) -> LoggerService:
    """Ensure value is a LoggerService with proper error handling."""
    return TypeSafeAssertion.assert_logger_service(value, "LoggerService validation")


def ensure_not_none(value: T | None, name: str = "value") -> T:
    """Ensure value is not None with proper error handling."""
    return TypeSafeAssertion.assert_not_none(value, f"{name} must not be None")


def ensure_market_price_service(value: Any) -> MarketPriceServiceProtocol:
    """Ensure value is a MarketPriceServiceProtocol with proper error handling."""
    return TypeSafeAssertion.assert_market_price_service(value, "MarketPriceService validation")


def ensure_market_data_provider(value: Any) -> MarketDataProviderProtocol:
    """Ensure value is a MarketDataProviderProtocol with proper error handling."""
    return TypeSafeAssertion.assert_market_data_provider(value, "MarketDataProvider validation")


def ensure_market_data_stream(value: Any) -> MarketDataStreamProtocol:
    """Ensure value is a MarketDataStreamProtocol with proper error handling."""
    return TypeSafeAssertion.assert_market_data_stream(value, "MarketDataStream validation")


def ensure_market_data_cache(value: Any) -> MarketDataCacheProtocol:
    """Ensure value is a MarketDataCacheProtocol with proper error handling."""
    return TypeSafeAssertion.assert_market_data_cache(value, "MarketDataCache validation")


def ensure_market_data_aggregator(value: Any) -> MarketDataAggregatorProtocol:
    """Ensure value is a MarketDataAggregatorProtocol with proper error handling."""
    return TypeSafeAssertion.assert_market_data_aggregator(value, "MarketDataAggregator validation")


def ensure_market_data_validator(value: Any) -> MarketDataValidatorProtocol:
    """Ensure value is a MarketDataValidatorProtocol with proper error handling."""
    return TypeSafeAssertion.assert_market_data_validator(value, "MarketDataValidator validation")
