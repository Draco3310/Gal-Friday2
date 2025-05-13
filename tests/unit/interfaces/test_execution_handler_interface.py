"""
Tests for the ExecutionHandlerInterface contract.

These tests verify that implementations of ExecutionHandlerInterface correctly
follow the interface contract and behavior requirements.
"""

from abc import ABC
from typing import Any, Dict
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from gal_friday.core.events import TradeSignalApprovedEvent
from gal_friday.execution_handler import ExecutionHandler
from gal_friday.interfaces.execution_handler_interface import ExecutionHandlerInterface


class MockExecutionHandler(ExecutionHandlerInterface):
    """A minimal implementation of ExecutionHandlerInterface for testing."""

    def __init__(self) -> None:
        """Initialize a mock execution handler for testing."""
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.is_running = False

    async def start(self) -> None:
        """Implement abstract method for test."""
        self.is_running = True

    async def stop(self) -> None:
        """Implement abstract method for test."""
        self.is_running = False

    async def handle_trade_signal_approved(self, event: TradeSignalApprovedEvent) -> None:
        """Implement abstract method for test."""
        # Store the order in an internal dict for testing
        order_id = str(uuid4())
        self.orders[order_id] = {
            "symbol": event.trading_pair,
            "side": event.side,
            "quantity": event.quantity,
            "status": "OPEN",
        }

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Implement abstract method for test."""
        if exchange_order_id in self.orders:
            self.orders[exchange_order_id]["status"] = "CANCELED"
            return True
        return False


class IncompleteExecutionHandler(ABC):
    """A class that inherits from ExecutionHandlerInterface but doesn't implement all methods."""

    async def start(self) -> None:
        """Implement one abstract method but not others."""


def test_execution_handler_interface_is_abstract():
    """Test that ExecutionHandlerInterface cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ExecutionHandlerInterface()


def test_execution_handler_interface_requires_implementation():
    """Test that a class inheriting from ExecutionHandlerInterface.

    must implement all abstract methods.
    """
    with pytest.raises(TypeError):
        IncompleteExecutionHandler()


def test_can_instantiate_concrete_implementation():
    """Test that a concrete implementation can be instantiated."""
    handler = MockExecutionHandler()
    assert isinstance(handler, ExecutionHandlerInterface)


@pytest.mark.asyncio
async def test_start_stop_contract():
    """Test that the start/stop methods follow the contract."""
    handler = MockExecutionHandler()

    # Should be able to start
    await handler.start()
    assert handler.is_running is True

    # Should be able to stop
    await handler.stop()
    assert handler.is_running is False


@pytest.mark.asyncio
async def test_handle_trade_signal_approved_contract():
    """Test that the handle_trade_signal_approved method follows the contract."""
    handler = MockExecutionHandler()

    # Create a mock trade signal event
    event = MagicMock(spec=TradeSignalApprovedEvent)
    event.event_id = str(uuid4())
    event.trading_pair = "XRP/USD"
    event.side = "BUY"
    event.quantity = "100"

    # Handle the event
    await handler.handle_trade_signal_approved(event)

    # Verify an order was created
    assert len(handler.orders) == 1

    # Verify the order details match the event
    order = list(handler.orders.values())[0]
    assert order["symbol"] == "XRP/USD"
    assert order["side"] == "BUY"
    assert order["quantity"] == "100"
    assert order["status"] == "OPEN"


@pytest.mark.asyncio
async def test_cancel_order_contract():
    """Test that the cancel_order method follows the contract."""
    handler = MockExecutionHandler()

    # Create a mock order
    order_id = str(uuid4())
    handler.orders[order_id] = {
        "symbol": "XRP/USD",
        "side": "BUY",
        "quantity": "100",
        "status": "OPEN",
    }

    # Cancel the order
    result = await handler.cancel_order(order_id)

    # Verify cancellation was successful
    assert result is True
    assert handler.orders[order_id]["status"] == "CANCELED"

    # Canceling a non-existent order should return False
    result = await handler.cancel_order("non-existent-id")
    assert result is False


@pytest.mark.parametrize("implementation", [ExecutionHandler])
def test_real_implementations_conform_to_interface(implementation):
    """Test that real implementations conform to the interface."""
    # This test verifies that our actual implementations properly implement the interface
    assert issubclass(implementation, ExecutionHandlerInterface)

    # Verify that the implementations have the required methods
    assert hasattr(implementation, "start")
    assert hasattr(implementation, "stop")
    assert hasattr(implementation, "handle_trade_signal_approved")
    assert hasattr(implementation, "cancel_order")
