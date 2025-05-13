"""Example usage of the PortfolioManager class for testing purposes."""

import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, cast, runtime_checkable

from gal_friday.portfolio import PositionInfo


# Protocol definitions moved outside function
@runtime_checkable
class ConfigManagerProtocol(Protocol):
    """Protocol defining the configuration manager interface."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Args
        ----
            key: The configuration key to look up
            default: Default value if key not found

        Returns
        -------
            The configuration value or default if not found
        """
        ...

    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value by key.

        Args
        ----
            key: The configuration key to look up
            default: Default integer value if key not found

        Returns
        -------
            The integer configuration value or default if not found
        """
        ...

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value by key.

        Args
        ----
            key: The configuration key to look up
            default: Default boolean value if key not found

        Returns
        -------
            The boolean configuration value or default if not found
        """
        ...


@runtime_checkable
class LoggerServiceProtocol(Protocol):
    """Protocol defining the logger service interface."""

    def info(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Log an info message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        ...

    def debug(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Log a debug message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        ...

    def warning(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Log a warning message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        ...

    def error(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Log an error message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        ...

    def critical(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Log a critical message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        ...


@runtime_checkable
class PubSubManagerProtocol(Protocol):
    """Protocol defining the publish-subscribe manager interface."""

    def subscribe(self, event_type: Any, handler: Callable) -> None:
        """Subscribe a handler to an event type.

        Args
        ----
            event_type: The type of event to subscribe to
            handler: The callback function to handle the event
        """
        ...

    def unsubscribe(self, event_type: Any, handler: Callable) -> bool:
        """Unsubscribe a handler from an event type.

        Args
        ----
            event_type: The type of event to unsubscribe from
            handler: The callback function to remove

        Returns
        -------
            True if successfully unsubscribed, False otherwise
        """
        ...

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribed handlers.

        Args
        ----
            event: The event to publish
        """
        ...


@runtime_checkable
class MarketPriceServiceProtocol(Protocol):
    """Protocol defining the market price service interface."""

    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """Get the latest price for a trading pair.

        Args
        ----
            trading_pair: The trading pair to get price for

        Returns
        -------
            The latest price or None if not available
        """
        ...

    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Get the current bid-ask spread for a trading pair.

        Args
        ----
            trading_pair: The trading pair to get spread for

        Returns
        -------
            Tuple of (bid, ask) prices or None if not available
        """
        ...


@runtime_checkable
class ReconcilableExecutionHandler(Protocol):
    """Protocol defining the reconcilable execution handler interface."""

    async def get_account_balances(self) -> Dict[str, Decimal]:
        """Get current account balances from the exchange.

        Returns
        -------
            Dictionary mapping asset symbols to their balances
        """
        ...

    async def get_open_positions(self) -> Dict[str, PositionInfo]:
        """Get current open positions from the exchange.

        Returns
        -------
            Dictionary mapping trading pairs to their position information
        """
        ...


@runtime_checkable
class ExecutionHandlerProtocol(ReconcilableExecutionHandler, Protocol):
    """Protocol combining reconcilable execution handler with additional execution capabilities."""

    pass


# Mock implementations moved outside function
class MockConfigManager:
    """Mock implementation of the configuration manager for testing."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a mock configuration value.

        Args
        ----
            key: The configuration key to look up
            default: Default value if key not found

        Returns
        -------
            Predefined mock values for testing or default
        """
        if key == "portfolio.initial_capital":
            return {"USD": 100000}
        if key == "portfolio.valuation_currency":
            return "USD"
        if key == "portfolio.reconciliation.threshold":
            return "0.01"  # 1% threshold
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        """Get a mock integer configuration value.

        Args
        ----
            key: The configuration key to look up
            default: Default integer value if key not found

        Returns
        -------
            Predefined mock integer values for testing or default
        """
        if key == "portfolio.drawdown.daily_reset_hour_utc":
            return 0
        if key == "portfolio.drawdown.weekly_reset_day":
            return 0
        if key == "portfolio.reconciliation.interval_seconds":
            return 5  # Short interval for test
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a mock boolean configuration value.

        Args
        ----
            key: The configuration key to look up
            default: Default boolean value if key not found

        Returns
        -------
            Predefined mock boolean values for testing or default
        """
        if key == "portfolio.reconciliation.auto_update":
            return True
        return default


class MockLoggerService:
    """Mock implementation of the logger service for testing."""

    def info(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Print an info level log message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        print(f"INFO [{source_module}]: {msg}")

    def debug(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Print a debug level log message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        print(f"DEBUG [{source_module}]: {msg}")

    def warning(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Print a warning level log message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        print(f"WARN [{source_module}]: {msg}")

    def error(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Print an error level log message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        print(f"ERROR [{source_module}]: {msg}")

    def critical(self, msg: str, source_module: str = "?", **kwargs: Any) -> None:
        """Print a critical level log message.

        Args
        ----
            msg: The message to log
            source_module: The source module name
            **kwargs: Additional logging context
        """
        print(f"CRITICAL [{source_module}]: {msg}")


class MockPubSubManager:
    """Mock implementation of the publish-subscribe manager for testing."""

    def __init__(self, logger: Any) -> None:
        """Initialize the mock pubsub manager.

        Args
        ----
            logger: Logger instance for debug output
        """
        self._logger = logger
        self._subscriptions: Dict[Any, List[Callable]] = defaultdict(list)

    async def publish(self, event: Any) -> None:
        """Publish an event to all subscribed handlers.

        Args
        ----
            event: The event to publish
        """
        print(f"MockPublish: {event}")
        event_type = getattr(event, "event_type", None)
        if event_type is not None:
            handlers = self._subscriptions.get(event_type, [])
            print(f"Found handlers for {event_type}: {handlers}")
            for handler in handlers:
                print(f"Calling handler: {handler}")
                asyncio.create_task(handler(event))

    def subscribe(self, event_type: Any, handler: Callable) -> None:
        """Subscribe a handler to an event type.

        Args
        ----
            event_type: The type of event to subscribe to
            handler: The callback function to handle the event
        """
        etype_name = getattr(event_type, "name", str(event_type))
        print(f"MockSubscribe: {handler.__name__} to {etype_name}")
        self._subscriptions[event_type].append(handler)

    def unsubscribe(self, event_type: Any, handler: Callable) -> bool:
        """Unsubscribe a handler from an event type.

        Args
        ----
            event_type: The type of event to unsubscribe from
            handler: The callback function to remove

        Returns
        -------
            True if successfully unsubscribed, False otherwise
        """
        etype_name = getattr(event_type, "name", str(event_type))
        print(f"MockUnsubscribe: {handler.__name__} from {etype_name}")
        try:
            self._subscriptions[event_type].remove(handler)
            return True
        except (ValueError, KeyError):  # Handle KeyError if event_type not present
            return False


class MockMarketPriceService:
    """Mock implementation of the market price service for testing."""

    async def get_latest_price(self, trading_pair: str) -> Optional[Decimal]:
        """Get a mock latest price for a trading pair.

        Args
        ----
            trading_pair: The trading pair to get price for

        Returns
        -------
            Predefined mock price or None if not available
        """
        if trading_pair == "BTC/USD":
            return Decimal("50000.0")
        if trading_pair == "USD/USD":
            return Decimal("1.0")  # Needed for valuation
        return None

    async def get_bid_ask_spread(self, trading_pair: str) -> Optional[Tuple[Decimal, Decimal]]:
        """Get a mock bid-ask spread for a trading pair.

        Args
        ----
            trading_pair: The trading pair to get spread for

        Returns
        -------
            Tuple of predefined mock (bid, ask) prices
        """
        return (Decimal("49999.0"), Decimal("50001.0"))


class MockExecutionHandler:
    """Mock implementation of the execution handler for testing."""

    async def get_account_balances(self) -> Dict[str, Decimal]:
        """Get mock account balances.

        Returns
        -------
            Dictionary of predefined mock balances
        """
        return {"USD": Decimal("95000.0"), "BTC": Decimal("0.1")}

    async def get_open_positions(self) -> Dict[str, PositionInfo]:
        """Get mock open positions.

        Returns
        -------
            Dictionary of predefined mock positions
        """
        return {
            "BTC/USD": PositionInfo(
                trading_pair="BTC/USD",
                base_asset="BTC",
                quote_asset="USD",
                quantity=Decimal("0.1"),
                average_entry_price=Decimal("50000.0"),
            )
        }


def create_execution_report_event() -> Any:
    """Create a mock execution report event for testing."""
    from gal_friday.core.events import ExecutionReportEvent

    return ExecutionReportEvent(
        source_module="MockExecHandler",
        event_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        exchange_order_id="ORDER123",
        trading_pair="BTC/USD",
        exchange="SIM",
        order_status="FILLED",
        order_type="MARKET",
        side="BUY",
        quantity_ordered=Decimal("0.1"),
        quantity_filled=Decimal("0.1"),
        average_fill_price=Decimal("50000.0"),
        commission=Decimal("5.0"),
        commission_asset="USD",
        timestamp_exchange=datetime.utcnow(),
    )


def create_mock_services():
    """Create and return mock services needed for the portfolio manager."""
    mock_config = MockConfigManager()
    mock_logger = MockLoggerService()
    mock_pubsub = MockPubSubManager(logger=mock_logger)
    mock_market_price = MockMarketPriceService()
    mock_execution_handler = MockExecutionHandler()

    return mock_config, mock_logger, mock_pubsub, mock_market_price, mock_execution_handler


def initialize_portfolio_manager(mocks):
    """Initialize the portfolio manager with mock services."""
    from gal_friday.config_manager import ConfigManager
    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.execution_handler import ExecutionHandler
    from gal_friday.logger_service import LoggerService
    from gal_friday.market_price_service import MarketPriceService
    from gal_friday.portfolio_manager import PortfolioManager

    mock_config, mock_logger, mock_pubsub, mock_market_price, mock_execution_handler = mocks

    # Cast mocks to satisfy type checker
    return PortfolioManager(
        config_manager=cast(ConfigManager, mock_config),
        pubsub_manager=cast(PubSubManager, mock_pubsub),
        market_price_service=cast(MarketPriceService, mock_market_price),
        logger_service=cast(LoggerService, mock_logger),
        execution_handler=cast(ExecutionHandler, mock_execution_handler),
    )


async def run_portfolio_test(portfolio_manager, mock_pubsub):
    """Run the portfolio test scenario."""
    await portfolio_manager.start()
    await asyncio.sleep(0.1)  # Allow start tasks

    # Get the execution report event
    exec_event = create_execution_report_event()

    print("\n--- Publishing Execution Report ---")
    await mock_pubsub.publish(exec_event)
    await asyncio.sleep(0.2)  # Allow handler and valuation to process

    # Check state after execution
    print("\n--- Current Portfolio State (after trade) ---")
    current_state = portfolio_manager.get_current_state()
    print(json.dumps(current_state, indent=2, default=str))


async def run_reconciliation_test(portfolio_manager):
    """Run the reconciliation test scenario."""
    print("\n--- Running Reconciliation (expecting discrepancies) ---")
    # Reconciliation runs periodically, wait for it based on mock config interval
    await asyncio.sleep(6)  # Wait longer than the 5s interval in mock config

    # Check state after reconciliation
    print("\n--- Portfolio State After Reconciliation ---")
    current_state_after = portfolio_manager.get_current_state()
    print(json.dumps(current_state_after, indent=2, default=str))


async def example_usage():
    """
    Run an example of PortfolioManager class usage.

    Shows how to create and use a PortfolioManager with mock dependencies.
    """
    # Create mock services
    mocks = create_mock_services()
    mock_config, mock_logger, mock_pubsub, mock_market_price, mock_execution_handler = mocks

    # Initialize portfolio manager
    portfolio_manager = initialize_portfolio_manager(mocks)
    await asyncio.sleep(0.1)  # Allow _initialize_state task to run

    try:
        # Run the test scenarios
        await run_portfolio_test(portfolio_manager, mock_pubsub)
        await run_reconciliation_test(portfolio_manager)
    finally:
        # Cleanup
        await portfolio_manager.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())


class EventCapture:
    """Class for capturing and inspecting events during testing."""

    def __init__(self) -> None:
        """Initialize an empty event capture container."""
        self.events: List[Any] = []

    def capture_event(self, event: Any) -> None:
        """Capture an event for later inspection.

        Args
        ----
            event: The event to capture
        """
        print(f"Captured event: {event}")
        self.events.append(event)

    def get_events_of_type(self, event_type: Any) -> List[Any]:
        """Get all captured events of a specific type.

        Args
        ----
            event_type: The type of events to retrieve

        Returns
        -------
            List of captured events matching the specified type
        """
        return [e for e in self.events if getattr(e, "event_type", None) == event_type]


class MockEventBus:
    """Mock implementation of an event bus for testing."""

    def __init__(self) -> None:
        """Initialize the mock event bus."""
        self.events: List[Any] = []

    async def publish(self, event: Any) -> None:
        """Publish an event to the mock bus.

        Args
        ----
            event: The event to publish
        """
        print(f"Publishing event: {event}")
        self.events.append(event)
