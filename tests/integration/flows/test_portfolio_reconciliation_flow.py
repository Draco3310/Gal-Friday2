"""
Integration tests for the portfolio reconciliation flow.

These tests verify that:
- The portfolio manager correctly fetches and compares positions/balances from the exchange
- Discrepancies are detected and resolved appropriately
- Portfolio state is updated to reflect the exchange state
- Appropriate events are emitted during reconciliation
"""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.core.events import EventType
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.portfolio_manager import PortfolioManager


class EventCapture:
    """Utility class to capture and verify events in the integration tests."""

    def __init__(self):
        """Initialize an empty event list."""
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
        if section_path == "portfolio":
            return {
                "reconciliation_interval_sec": 60,  # More frequent for testing
                "max_reconciliation_discrepancy_pct": 1.0,
                "reconciliation_threshold": 0.5,
                # Threshold for considering a discrepancy significant
            }

        return default

    config_manager.get.side_effect = get_config_section
    return config_manager


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return MagicMock(spec=LoggerService)


@pytest.mark.asyncio
async def test_portfolio_reconciliation_minor_discrepancy(
    pubsub, event_capture, mock_config, mock_logger
):
    """Test reconciliation with minor discrepancies that don't exceed thresholds."""
    # Set up event capturing
    for event_type in [
        EventType.PORTFOLIO_UPDATE,
        EventType.PORTFOLIO_RECONCILIATION,
        EventType.PORTFOLIO_DISCREPANCY,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Initialize portfolio manager
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Set up initial portfolio state
    portfolio._equity = Decimal("100000.0")
    portfolio._cash = Decimal("50000.0")

    # Create positions
    portfolio._positions = {
        "XRP/USD": {
            "quantity": Decimal("50000"),
            "average_entry_price": Decimal("0.5"),
            "current_price": Decimal("0.5"),
            "unrealized_pnl": Decimal("0"),
        },
        "BTC/USD": {
            "quantity": Decimal("1.5"),
            "average_entry_price": Decimal("30000"),
            "current_price": Decimal("30000"),
            "unrealized_pnl": Decimal("0"),
        },
    }

    # Create a mock fetch_exchange_positions method with minor discrepancies
    async def mock_fetch_exchange_positions():
        return {
            "XRP/USD": {
                "quantity": Decimal("50050"),  # 0.1% more
                "average_entry_price": Decimal("0.501"),  # 0.2% difference
            },
            "BTC/USD": {
                "quantity": Decimal("1.505"),  # 0.33% more
                "average_entry_price": Decimal("29950"),  # 0.17% difference
            },
        }

    # Create a mock fetch_exchange_balances method
    async def mock_fetch_exchange_balances():
        return {
            "USD": Decimal("49900"),  # 0.2% less cash
            "XRP": Decimal("50050"),
            "BTC": Decimal("1.505"),
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
                discrepancy_events = event_capture.get_events_of_type(
                    EventType.PORTFOLIO_DISCREPANCY
                )

                # Verify reconciliation happened
                assert len(reconciliation_events) >= 1, "No reconciliation events triggered"

                # Minor discrepancies shouldn't trigger discrepancy events if below threshold
                assert (
                    len(discrepancy_events) == 0
                ), "Unexpected discrepancy events for minor differences"

                # Portfolio should be updated to match exchange values
                assert portfolio._cash == Decimal("49900"), "Cash balance not updated"
                assert portfolio._positions["XRP/USD"]["quantity"] == Decimal(
                    "50050"
                ), "XRP position not updated"
                assert portfolio._positions["BTC/USD"]["quantity"] == Decimal(
                    "1.505"
                ), "BTC position not updated"

            finally:
                # Stop the portfolio manager
                await portfolio.stop()


@pytest.mark.asyncio
async def test_portfolio_reconciliation_major_discrepancy(
    pubsub, event_capture, mock_config, mock_logger
):
    """Test reconciliation with major discrepancies that exceed thresholds."""
    # Set up event capturing
    for event_type in [
        EventType.PORTFOLIO_UPDATE,
        EventType.PORTFOLIO_RECONCILIATION,
        EventType.PORTFOLIO_DISCREPANCY,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Initialize portfolio manager
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Set up initial portfolio state
    portfolio._equity = Decimal("100000.0")
    portfolio._cash = Decimal("50000.0")

    # Create positions
    portfolio._positions = {
        "XRP/USD": {
            "quantity": Decimal("50000"),
            "average_entry_price": Decimal("0.5"),
            "current_price": Decimal("0.5"),
            "unrealized_pnl": Decimal("0"),
        }
    }

    # Create a mock fetch_exchange_positions method with major discrepancies
    async def mock_fetch_exchange_positions():
        return {
            "XRP/USD": {
                "quantity": Decimal("55000"),  # 10% more - significant
                "average_entry_price": Decimal("0.52"),  # 4% difference
            },
            "ETH/USD": {  # Position that exists on exchange but not in our system
                "quantity": Decimal("10"),
                "average_entry_price": Decimal("1800"),
            },
        }

    # Create a mock fetch_exchange_balances method
    async def mock_fetch_exchange_balances():
        return {
            "USD": Decimal("40000"),  # 20% less cash - significant
            "XRP": Decimal("55000"),
            "ETH": Decimal("10"),
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
                discrepancy_events = event_capture.get_events_of_type(
                    EventType.PORTFOLIO_DISCREPANCY
                )

                # Verify reconciliation happened
                assert len(reconciliation_events) >= 1, "No reconciliation events triggered"

                # Major discrepancies should trigger discrepancy events
                assert (
                    len(discrepancy_events) >= 1
                ), "No discrepancy events triggered for major differences"

                # Portfolio should be updated despite discrepancies
                assert portfolio._cash == Decimal("40000"), "Cash balance not updated"
                assert portfolio._positions["XRP/USD"]["quantity"] == Decimal(
                    "55000"
                ), "XRP position not updated"
                assert "ETH/USD" in portfolio._positions, "Missing ETH position not added"

            finally:
                # Stop the portfolio manager
                await portfolio.stop()


@pytest.mark.asyncio
async def test_portfolio_reconciliation_missing_position(
    pubsub, event_capture, mock_config, mock_logger
):
    """Test reconciliation when a position exists in our system but not on the exchange."""
    # Set up event capturing
    for event_type in [
        EventType.PORTFOLIO_UPDATE,
        EventType.PORTFOLIO_RECONCILIATION,
        EventType.PORTFOLIO_DISCREPANCY,
    ]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Initialize portfolio manager
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Set up initial portfolio state
    portfolio._equity = Decimal("100000.0")
    portfolio._cash = Decimal("50000.0")

    # Create positions, including one that doesn't exist on the exchange
    portfolio._positions = {
        "XRP/USD": {
            "quantity": Decimal("50000"),
            "average_entry_price": Decimal("0.5"),
            "current_price": Decimal("0.5"),
            "unrealized_pnl": Decimal("0"),
        },
        "DOGE/USD": {  # This position doesn't exist on the exchange
            "quantity": Decimal("10000"),
            "average_entry_price": Decimal("0.1"),
            "current_price": Decimal("0.1"),
            "unrealized_pnl": Decimal("0"),
        },
    }

    # Create a mock fetch_exchange_positions method with missing position
    async def mock_fetch_exchange_positions():
        return {
            "XRP/USD": {"quantity": Decimal("50000"), "average_entry_price": Decimal("0.5")}
            # DOGE/USD is missing
        }

    # Create a mock fetch_exchange_balances method
    async def mock_fetch_exchange_balances():
        return {
            "USD": Decimal("50000"),
            "XRP": Decimal("50000"),
            # DOGE is missing
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
                discrepancy_events = event_capture.get_events_of_type(
                    EventType.PORTFOLIO_DISCREPANCY
                )

                # Verify reconciliation happened
                assert len(reconciliation_events) >= 1, "No reconciliation events triggered"

                # Missing position should trigger a discrepancy event
                assert (
                    len(discrepancy_events) >= 1
                ), "No discrepancy events triggered for missing position"

                # The position should be removed from our portfolio
                assert (
                    "DOGE/USD" not in portfolio._positions
                ), "Position that doesn't exist on exchange was not removed"

            finally:
                # Stop the portfolio manager
                await portfolio.stop()


@pytest.mark.asyncio
async def test_scheduled_portfolio_reconciliation(pubsub, event_capture, mock_config, mock_logger):
    """Test that portfolio reconciliation happens automatically at scheduled intervals."""
    # Set up event capturing
    for event_type in [EventType.PORTFOLIO_RECONCILIATION]:
        pubsub.subscribe(event_type, event_capture.capture_event)

    # Override the reconciliation interval to be very short for testing
    def get_config_with_short_interval(section_path, default=None):
        if section_path == "portfolio":
            return {
                "reconciliation_interval_sec": 1,  # Very frequent for testing
                "max_reconciliation_discrepancy_pct": 1.0,
            }
        return default

    mock_config.get.side_effect = get_config_with_short_interval

    # Initialize portfolio manager
    portfolio = PortfolioManager(mock_config, pubsub, mock_logger)

    # Mock the reconciliation method to avoid actual API calls
    with patch.object(portfolio, "_reconcile_with_exchange") as mock_reconcile:
        # Start the portfolio manager
        await portfolio.start()

        try:
            # Wait for scheduled reconciliation to occur
            await asyncio.sleep(3)  # Should be enough time for multiple reconciliations

            # Check that reconciliation was called multiple times
            assert mock_reconcile.call_count >= 2, "Scheduled reconciliation not occurring"

        finally:
            # Stop the portfolio manager
            await portfolio.stop()
