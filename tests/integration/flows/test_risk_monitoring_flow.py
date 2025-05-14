"""
Integration tests for the risk monitoring and portfolio reconciliation flows.

These tests verify that the following critical system flows work correctly:
- Risk limit monitoring
- Position tracking
- Portfolio reconciliation with exchange
- Risk-based trade action (position liquidation)
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.core.events import EventType
from gal_friday.core.pubsub import PubSubManager
from gal_friday.event_bus import FillEvent, SignalEvent
from gal_friday.interfaces.execution_handler_interface import ExecutionHandlerInterface
from gal_friday.logger_service import LoggerService
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.risk_manager import RiskManager


class MockExecutionHandler(ExecutionHandlerInterface):
    """A mock execution handler for testing the risk monitoring flow."""

    def __init__(self, pubsub):
        """Initialize the mock execution handler.

        Args
        ----
            pubsub: The publish-subscribe manager to use for event communication
        """
        self.pubsub = pubsub
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.is_running = False

    async def start(self) -> None:
        """Start the execution handler."""
        self.is_running = True

    async def stop(self) -> None:
        """Stop the execution handler."""
        self.is_running = False

    async def handle_trade_signal_approved(self, event: Any) -> None:
        """Handle an approved trade signal by simulating order execution."""
        # Create a mock order ID
        order_id = f"mock-order-{datetime.now().timestamp()}"

        # Simulate order placement
        execution_report = FillEvent(
            event_id=f"exec-{order_id}",
            signal_id=event.event_id,
            exchange_order_id=order_id,
            client_order_id=f"client-{order_id}",
            trading_pair=event.trading_pair,
            exchange="kraken",
            order_status="NEW",
            order_type=event.order_type,
            side=event.side,
            quantity_ordered=event.quantity,
            quantity_filled="0",
            average_fill_price=None,
            limit_price=getattr(event, "limit_price", None),
            stop_price=None,
            commission=None,
            commission_asset=None,
        )

        # Publish the execution report
        await asyncio.sleep(0.1)  # Simulate network delay
        self.pubsub.publish(execution_report)

        # Simulate order fill after a short delay
        await asyncio.sleep(0.2)

        # Determine fill price (either limit price or a simulated market price)
        fill_price = getattr(event, "limit_price", "0.5123")

        # Create a fill execution report
        fill_report = FillEvent(
            event_id=f"fill-{order_id}",
            signal_id=event.event_id,
            exchange_order_id=order_id,
            client_order_id=f"client-{order_id}",
            trading_pair=event.trading_pair,
            exchange="kraken",
            order_status="FILLED",
            order_type=event.order_type,
            side=event.side,
            quantity_ordered=event.quantity,
            quantity_filled=event.quantity,
            average_fill_price=fill_price,
            limit_price=getattr(event, "limit_price", None),
            stop_price=None,
            commission="0.001",
            commission_asset="USD",
        )

        # Publish the fill report
        self.pubsub.publish(fill_report)

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an order."""
        if exchange_order_id in self.orders:
            # Get order details with defaults for safety
            order_details = self.orders[exchange_order_id]
            trading_pair = order_details.get("trading_pair", "BTC/USD")
            order_type = order_details.get("order_type", "LIMIT")
            side = order_details.get("side", "BUY")
            quantity = order_details.get("quantity", "1.0")

            # Simulate order cancellation
            cancel_report = FillEvent(
                event_id=f"cancel-{exchange_order_id}",
                signal_id=order_details.get("signal_id"),
                exchange_order_id=exchange_order_id,
                client_order_id=order_details.get("client_order_id"),
                trading_pair=trading_pair,
                exchange="kraken",
                order_status="CANCELED",
                order_type=order_type,
                side=side,
                quantity_ordered=quantity,
                quantity_filled="0",
                average_fill_price=None,
                limit_price=order_details.get("limit_price"),
                stop_price=None,
                commission=None,
                commission_asset=None,
            )

            # Publish the cancellation report
            self.pubsub.publish(cancel_report)
            return True

        return False


class EventCapture:
    """Utility class to capture and verify events in the integration tests."""

    def __init__(self):
        """Initialize an empty event capture container."""
        self.events = []

    def capture_event(self, event):
        """Store an event for later analysis."""
        self.events.append(event)

    def get_events_of_type(self, event_type):
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def clear(self):
        """Clear stored events."""
        self.events = []


@pytest.fixture
def pubsub():
    """Create a PubSubManager for testing."""
    return PubSubManager()


@pytest.fixture
def event_capture():
    """Create an EventCapture instance for testing."""
    return EventCapture()


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config_manager = MagicMock()

    # Set up the config to return appropriate values
    def get_config_section(section_path, default=None):
        if section_path == "risk_management":
            return {
                "max_drawdown_pct": 15.0,
                "daily_drawdown_pct": 2.0,
                "weekly_drawdown_pct": 5.0,
                "risk_per_trade_pct": 0.5,
                "max_exposure_pct": 25.0,
                "max_single_exposure_pct": 10.0,
                "max_consecutive_losses": 5,
                "liquidation_threshold_pct": 8.0,  # Liquidate positions when loss exceeds this
            }
        elif section_path == "portfolio":
            return {
                "reconciliation_interval_sec": 60,  # More frequent for testing
                "max_reconciliation_discrepancy_pct": 1.0,
            }

        return default

    config_manager.get.side_effect = get_config_section
    return config_manager


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=LoggerService)


@pytest.mark.asyncio
async def test_risk_limit_breach_flow(pubsub, event_capture, mock_config, mock_logger):
    """Test the risk monitoring flow when a risk limit is breached."""
    # Set up event capturing
    for event_type in [
        EventType.RISK_LIMIT_ALERT,
        EventType.TRADE_SIGNAL_PROPOSED,
        EventType.TRADE_SIGNAL_REJECTED,
        EventType.PORTFOLIO_UPDATE,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Initialize components
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Set up initial portfolio state
    portfolio._equity = Decimal("100000.0")
    portfolio._starting_equity = Decimal("100000.0")
    portfolio._cash = Decimal("50000.0")

    # Create a large position that's losing money to trigger risk alerts
    portfolio._positions = {
        "XRP/USD": {
            "quantity": Decimal("50000"),
            "average_entry_price": Decimal("0.5"),
            "current_price": Decimal("0.45"),  # 10% loss
            "unrealized_pnl": Decimal("-2500"),  # -$2,500 loss
        }
    }

    # Set up trading history with consecutive losses
    portfolio._trading_history = [
        {"trade_id": "1", "result": "loss", "amount": Decimal("-500")},
        {"trade_id": "2", "result": "loss", "amount": Decimal("-600")},
        {"trade_id": "3", "result": "loss", "amount": Decimal("-700")},
        {"trade_id": "4", "result": "loss", "amount": Decimal("-800")},
        {"trade_id": "5", "result": "loss", "amount": Decimal("-900")},
    ]

    risk_manager = RiskManager(mock_config, pubsub, mock_logger, portfolio)

    # Mock execution handler
    execution_handler = MockExecutionHandler(pubsub)

    # Start components
    await portfolio.start()
    await risk_manager.start()
    await execution_handler.start()

    try:
        # Create a trade signal that should be rejected due to risk limits
        trade_signal = SignalEvent(
            event_id="test-signal-1",
            timestamp=datetime.now(),
            source_module="test_risk_monitoring_flow",
            trading_pair="BTC/USD",
            side="BUY",
            proposed_entry_price="50000",  # High price
            strategy_id="test_strategy",
            proposed_quantity="2",  # Large quantity
        )

        # Publish the trade signal
        pubsub.publish(trade_signal)

        # Wait for risk evaluation
        await asyncio.sleep(0.5)

        # Verify events
        risk_limit_events = event_capture.get_events_of_type(EventType.RISK_LIMIT_ALERT)
        rejected_signals = event_capture.get_events_of_type(EventType.TRADE_SIGNAL_REJECTED)

        # There should be at least one risk limit event
        assert len(risk_limit_events) >= 1, "No risk limit events triggered"

        # The signal should be rejected
        assert len(rejected_signals) >= 1, "Trade signal was not rejected"
        assert rejected_signals[0].signal_id == trade_signal.event_id
        assert "risk" in rejected_signals[0].reason.lower()

    finally:
        # Stop all components
        await execution_handler.stop()
        await risk_manager.stop()
        await portfolio.stop()


@pytest.mark.asyncio
async def test_portfolio_reconciliation_flow(pubsub, event_capture, mock_config, mock_logger):
    """Test the portfolio reconciliation flow."""
    # Set up event capturing
    for event_type in [EventType.PORTFOLIO_UPDATE, EventType.PORTFOLIO_RECONCILIATION]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Initialize portfolio manager
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Set up initial portfolio state
    portfolio._equity = Decimal("100000.0")
    portfolio._starting_equity = Decimal("100000.0")
    portfolio._cash = Decimal("75000.0")

    # Create a position
    portfolio._positions = {
        "XRP/USD": {
            "quantity": Decimal("50000"),
            "average_entry_price": Decimal("0.5"),
            "current_price": Decimal("0.5"),
            "unrealized_pnl": Decimal("0"),
        }
    }

    # Create a mock fetch_exchange_positions method to simulate exchange data
    async def mock_fetch_exchange_positions():
        # Return slightly different values to simulate reconciliation needs
        return {
            "XRP/USD": {
                "quantity": Decimal("50100"),  # 100 more than we think we have
                "average_entry_price": Decimal("0.501"),
            },
            "ETH/USD": {  # A position we didn't know about
                "quantity": Decimal("0.5"),
                "average_entry_price": Decimal("1800"),
            },
        }

    # Create a mock fetch_exchange_balances method
    async def mock_fetch_exchange_balances():
        return {
            "USD": Decimal("74100"),  # Less cash than we think we have
            "XRP": Decimal("50100"),
            "ETH": Decimal("0.5"),
        }

    # Patch the portfolio methods
    with patch.object(portfolio, "_fetch_exchange_positions", mock_fetch_exchange_positions):
        with patch.object(portfolio, "_fetch_exchange_balances", mock_fetch_exchange_balances):
            # Start portfolio
            await portfolio.start()

            try:
                # Manually trigger a reconciliation
                await portfolio._reconcile_with_exchange()

                # Wait for events to be processed
                await asyncio.sleep(0.5)

                # Check for reconciliation events
                reconciliation_events = event_capture.get_events_of_type(
                    EventType.PORTFOLIO_RECONCILIATION
                )
                portfolio_update_events = event_capture.get_events_of_type(
                    EventType.PORTFOLIO_UPDATE
                )

                # Verify we got reconciliation events
                assert len(reconciliation_events) >= 1, "No reconciliation events triggered"

                # The reconciliation should have triggered portfolio updates
                assert (
                    len(portfolio_update_events) >= 1
                ), "No portfolio updates triggered by reconciliation"

                # Check that the portfolio was updated to match exchange data
                assert (
                    "ETH/USD" in portfolio._positions
                ), "New position was not added during reconciliation"
                assert portfolio._positions["XRP/USD"]["quantity"] == Decimal(
                    "50100"
                ), "Position quantity not updated"

                # Check cash was updated
                assert portfolio._cash == Decimal("74100"), "Cash balance not updated"

            finally:
                # Stop the portfolio manager
                await portfolio.stop()


@pytest.mark.asyncio
async def test_position_liquidation_flow(pubsub, event_capture, mock_config, mock_logger):
    """Test the automated position liquidation flow when risk limits are breached."""
    # Set up event capturing
    for event_type in [
        EventType.RISK_LIMIT_ALERT,
        EventType.TRADE_SIGNAL_APPROVED,
        EventType.EXECUTION_REPORT,
        EventType.PORTFOLIO_UPDATE,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Initialize components
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Set up initial portfolio state
    portfolio._equity = Decimal("100000.0")
    portfolio._starting_equity = Decimal("100000.0")
    portfolio._cash = Decimal("50000.0")

    # Create a position with significant losses to trigger liquidation
    portfolio._positions = {
        "XRP/USD": {
            "quantity": Decimal("50000"),
            "average_entry_price": Decimal("0.5"),
            "current_price": Decimal("0.4"),  # 20% loss
            "unrealized_pnl": Decimal("-5000"),  # -$5,000 loss
        }
    }

    # Initialize risk manager with auto-liquidation enabled
    risk_manager = RiskManager(mock_config, pubsub, mock_logger, portfolio)

    # Enable auto-liquidation for testing
    risk_manager._auto_liquidate_on_breach = True

    # Mock execution handler
    execution_handler = MockExecutionHandler(pubsub)

    # Start components
    await portfolio.start()
    await risk_manager.start()
    await execution_handler.start()

    try:
        # Trigger a risk evaluation that should lead to liquidation
        risk_manager._evaluate_risk_limits()

        # Wait for the full flow to complete
        await asyncio.sleep(1.0)

        # Verify events
        risk_limit_events = event_capture.get_events_of_type(EventType.RISK_LIMIT_ALERT)
        approved_signals = event_capture.get_events_of_type(EventType.TRADE_SIGNAL_APPROVED)
        execution_reports = event_capture.get_events_of_type(EventType.EXECUTION_REPORT)

        # There should be risk limit events
        assert len(risk_limit_events) >= 1, "No risk limit events triggered"

        # A liquidation order should be approved
        assert len(approved_signals) >= 1, "No liquidation order was approved"

        # Verify it's a liquidation order (should be a SELL for XRP/USD)
        liquidation_order = approved_signals[0]
        assert liquidation_order.trading_pair == "XRP/USD"
        assert liquidation_order.side == "SELL"

        # Execution reports should be generated
        assert len(execution_reports) >= 2, "Not enough execution reports generated"

        # Check for a FILLED report
        filled_reports = [r for r in execution_reports if r.order_status == "FILLED"]
        assert len(filled_reports) >= 1, "No filled execution reports"

        # After liquidation, position should be reduced or eliminated
        # Wait a bit for portfolio updates
        await asyncio.sleep(0.5)

        # Position should be eliminated or reduced
        if "XRP/USD" in portfolio._positions:
            assert portfolio._positions["XRP/USD"]["quantity"] < Decimal(
                "50000"
            ), "Position was not reduced"

    finally:
        # Stop all components
        await execution_handler.stop()
        await risk_manager.stop()
        await portfolio.stop()
