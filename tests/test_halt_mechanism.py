"""Tests for the HALT mechanism functionality.

This module tests the HALT coordinator, monitoring service integration,
and emergency position closure capabilities.
"""

from decimal import Decimal

import asyncio
import pytest

from gal_friday.core.events import ClosePositionCommand, EventType, SystemStateEvent
from gal_friday.core.halt_coordinator import HaltCondition, HaltCoordinator
from gal_friday.monitoring_service import MonitoringService


class TestHaltCoordinator:
    """Test suite for HaltCoordinator functionality."""

    def test_halt_coordinator_initialization(self, mock_config_manager, pubsub_manager, mock_logger):
        """Test HALT coordinator initializes with correct conditions."""
        coordinator = HaltCoordinator(mock_config_manager, pubsub_manager, mock_logger)

        # Check all expected conditions are registered
        expected_conditions = [
            "max_total_drawdown",
            "max_daily_drawdown",
            "max_consecutive_losses",
            "max_volatility",
            "api_error_rate",
            "data_staleness",
        ]

        for condition_id in expected_conditions:
            assert condition_id in coordinator.conditions
            assert isinstance(coordinator.conditions[condition_id], HaltCondition)

    def test_condition_update_numeric(self, mock_config_manager, pubsub_manager, mock_logger):
        """Test updating numeric conditions."""
        coordinator = HaltCoordinator(mock_config_manager, pubsub_manager, mock_logger)

        # Test drawdown condition
        assert not coordinator.update_condition("max_total_drawdown", Decimal("10.0"))
        assert not coordinator.conditions["max_total_drawdown"].is_triggered

        # Exceed threshold
        assert coordinator.update_condition("max_total_drawdown", Decimal("16.0"))
        assert coordinator.conditions["max_total_drawdown"].is_triggered

    def test_halt_state_management(self, mock_config_manager, pubsub_manager, mock_logger):
        """Test HALT state management."""
        coordinator = HaltCoordinator(mock_config_manager, pubsub_manager, mock_logger)

        # Initially not halted
        status = coordinator.get_halt_status()
        assert not status["is_halted"]
        assert status["halt_reason"] == ""

        # Set halt state
        coordinator.set_halt_state(True, "Test halt", "TEST")
        status = coordinator.get_halt_status()
        assert status["is_halted"]
        assert status["halt_reason"] == "Test halt"
        assert status["halt_source"] == "TEST"
        assert status["halt_timestamp"] is not None

        # Clear halt state
        coordinator.clear_halt_state()
        status = coordinator.get_halt_status()
        assert not status["is_halted"]
        assert all(not c.is_triggered for c in coordinator.conditions.values())


class TestMonitoringServiceHALT:
    """Test monitoring service HALT integration."""

    @pytest.fixture
    def mock_portfolio_manager(self, mock_portfolio_state):
        """Create a mock portfolio manager."""
        class MockPortfolioManager:
            def __init__(self, state):
                self.state = state

            def get_current_state(self):
                return self.state

        return MockPortfolioManager(mock_portfolio_state)

    @pytest.mark.asyncio
    async def test_halt_on_drawdown(self, mock_config_manager, pubsub_manager,
                                    mock_portfolio_manager, mock_logger):
        """Test HALT triggered by drawdown."""
        # Create portfolio with high drawdown
        mock_portfolio_manager.state["total_drawdown_pct"] = Decimal("20.0")

        monitoring = MonitoringService(
            mock_config_manager,
            pubsub_manager,
            mock_portfolio_manager,
            mock_logger,
        )

        # Subscribe to system state events
        state_events = []
        async def capture_state_event(event):
            if isinstance(event, SystemStateEvent):
                state_events.append(event)

        pubsub_manager.subscribe(EventType.SYSTEM_STATE_CHANGE, capture_state_event)

        # Start monitoring
        await monitoring.start()

        # Wait for periodic check
        await asyncio.sleep(1.5)

        # Should be halted
        assert monitoring.is_halted()
        assert len(state_events) >= 2  # RUNNING + HALTED
        assert state_events[-1].new_state == "HALTED"
        assert "drawdown" in state_events[-1].reason.lower()

        await monitoring.stop()

    @pytest.mark.asyncio
    async def test_position_closure_on_halt(self, mock_config_manager, pubsub_manager,
                                           mock_portfolio_manager, mock_logger):
        """Test positions are closed when HALT is triggered with close behavior."""
        monitoring = MonitoringService(
            mock_config_manager,
            pubsub_manager,
            mock_portfolio_manager,
            mock_logger,
        )

        # Subscribe to close position commands
        close_commands = []
        async def capture_close_command(event):
            if isinstance(event, ClosePositionCommand):
                close_commands.append(event)

        pubsub_manager.subscribe(EventType.TRADE_SIGNAL_APPROVED, capture_close_command)

        # Trigger HALT manually
        await monitoring.trigger_halt("Test HALT", "TEST")

        # Wait for async operations
        await asyncio.sleep(0.1)

        # Should have close command for XRP position
        assert len(close_commands) == 1
        assert close_commands[0].trading_pair == "XRP/USD"
        assert close_commands[0].quantity == Decimal(1000)
        assert close_commands[0].side == "SELL"  # Opposite of BUY position

    @pytest.mark.asyncio
    async def test_halt_coordinator_integration(self, mock_config_manager, pubsub_manager,
                                               mock_portfolio_manager, mock_logger):
        """Test monitoring service properly integrates with HALT coordinator."""
        monitoring = MonitoringService(
            mock_config_manager,
            pubsub_manager,
            mock_portfolio_manager,
            mock_logger,
        )

        # Access the HALT coordinator
        coordinator = monitoring._halt_coordinator
        assert coordinator is not None

        # Trigger HALT
        await monitoring.trigger_halt("Integration test", "TEST")

        # Check coordinator state is updated
        status = coordinator.get_halt_status()
        assert status["is_halted"]
        assert status["halt_reason"] == "Integration test"
        assert status["halt_source"] == "TEST"

        # Resume
        await monitoring.trigger_resume("TEST")

        # Check coordinator state is cleared
        status = coordinator.get_halt_status()
        assert not status["is_halted"]
        assert all(not c.is_triggered for c in coordinator.conditions.values())


class TestEmergencyPositionClosure:
    """Test emergency position closure functionality."""

    @pytest.mark.asyncio
    async def test_close_position_command_creation(self):
        """Test ClosePositionCommand event creation."""
        cmd = ClosePositionCommand.create(
            source_module="TEST",
            trading_pair="XRP/USD",
            quantity=Decimal(1000),
            side="SELL",
        )

        assert cmd.trading_pair == "XRP/USD"
        assert cmd.quantity == Decimal(1000)
        assert cmd.side == "SELL"
        assert cmd.order_type == "MARKET"
        assert cmd.event_type == EventType.TRADE_SIGNAL_APPROVED

    @pytest.mark.asyncio
    async def test_multiple_position_closure(self, mock_config_manager, pubsub_manager,
                                           mock_logger):
        """Test closing multiple positions on HALT."""
        # Create portfolio with multiple positions
        portfolio_state = {
            "total_drawdown_pct": Decimal("1.0"),
            "positions": {
                "XRP/USD": {
                    "quantity": Decimal(1000),
                    "side": "BUY",
                },
                "DOGE/USD": {
                    "quantity": Decimal(5000),
                    "side": "SELL",
                },
            },
        }

        class MultiPositionPortfolio:
            def get_current_state(self):
                return portfolio_state

        monitoring = MonitoringService(
            mock_config_manager,
            pubsub_manager,
            MultiPositionPortfolio(),
            mock_logger,
        )

        close_commands = []
        async def capture_close(event):
            if isinstance(event, ClosePositionCommand):
                close_commands.append(event)

        pubsub_manager.subscribe(EventType.TRADE_SIGNAL_APPROVED, capture_close)

        # Trigger HALT
        await monitoring.trigger_halt("Multi-position test", "TEST")
        await asyncio.sleep(0.1)

        # Should have commands for both positions
        assert len(close_commands) == 2

        # Check XRP position closure
        xrp_cmd = next(c for c in close_commands if c.trading_pair == "XRP/USD")
        assert xrp_cmd.quantity == Decimal(1000)
        assert xrp_cmd.side == "SELL"  # Opposite of BUY

        # Check DOGE position closure
        doge_cmd = next(c for c in close_commands if c.trading_pair == "DOGE/USD")
        assert doge_cmd.quantity == Decimal(5000)
        assert doge_cmd.side == "BUY"  # Opposite of SELL
