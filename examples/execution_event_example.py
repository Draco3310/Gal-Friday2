#!/usr/bin/env python3
"""Example demonstrating the enterprise-grade execution event creation and publishing system.

This example shows how to:
1. Create Fill instances with proper data
2. Generate ExecutionReportEvents using the new builder
3. Publish events to an event bus
4. Monitor publication statistics
5. Handle errors gracefully
"""

import contextlib
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
import uuid

import asyncio


# Mock event bus for demonstration
class MockEventBus:
    """Mock event bus for demonstration purposes."""

    def __init__(self):
        self.published_events = []
        self.should_fail = False

    async def publish(self, topic: str, event: Any) -> None:
        """Mock publish method."""
        if self.should_fail:
            raise Exception("Event bus connection failed")

        self.published_events.append({
            "topic": topic,
            "event": event,
            "timestamp": datetime.now(UTC),
        })


# Mock Order class for demonstration
class MockOrder:
    """Mock Order class for demonstration."""

    def __init__(self):
        self.order_pk = 456
        self.exchange_order_id = "KRAKEN_ORD_123456"
        self.client_order_id = uuid.uuid4()
        self.signal_id = uuid.uuid4()
        self.status = "PARTIALLY_FILLED"
        self.order_type = "LIMIT"
        self.quantity_ordered = Decimal("2.0")
        self.limit_price = Decimal("45000.0")
        self.stop_price = None
        self.fills = []


async def demonstrate_basic_usage():
    """Demonstrate basic execution event creation and publishing."""
    # Import the actual classes (these would be real imports in production)
    from gal_friday.models.fill import ExecutionEventPublisher, Fill

    # Create a mock event bus
    event_bus = MockEventBus()

    # Set up the event publisher with dependency injection
    publisher = ExecutionEventPublisher(event_bus)
    Fill.set_event_publisher(publisher)

    # Create a sample fill with comprehensive data
    fill = Fill(
        fill_pk=123,
        fill_id="KRAKEN_FILL_789",
        order_pk=456,
        exchange_order_id="KRAKEN_ORD_123456",
        trading_pair="BTC/USD",
        exchange="KRAKEN",
        side="BUY",
        quantity_filled=Decimal("1.5"),
        fill_price=Decimal("45000.0"),
        commission=Decimal("67.5"),
        commission_asset="USD",
        liquidity_type="TAKER",
        filled_at=datetime.now(UTC),
    )

    # Create a mock order and associate it
    order = MockOrder()
    order.fills = [fill]
    fill.order = order

    summary = fill.get_execution_summary()
    for _key, _value in summary.items():
        pass


    try:
        # Create execution event using the enhanced to_event method
        fill.to_event()


        # Publish the event
        success = await fill.publish_execution_event()

        if success:

            # Show publication statistics
            publisher.get_publication_stats()

        else:
            pass

    except Exception:
        pass


async def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    from gal_friday.models.fill import ExecutionEventPublisher, Fill

    # Create a fill with invalid data to test validation
    invalid_fill = Fill(
        fill_pk=999,
        fill_id="INVALID_FILL",
        order_pk=999,
        trading_pair="",  # Invalid empty trading pair
        exchange="TEST",
        side="BUY",
        quantity_filled=Decimal(0),  # Invalid zero quantity
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
        commission_asset="USD",
        filled_at=datetime.now(UTC),
    )


    with contextlib.suppress(Exception):
        invalid_fill.to_event()

    # Test event bus failure handling

    failing_event_bus = MockEventBus()
    failing_event_bus.should_fail = True

    failing_publisher = ExecutionEventPublisher(failing_event_bus)
    Fill.set_event_publisher(failing_publisher)

    # Create a valid fill for this test
    valid_fill = Fill(
        fill_pk=200,
        fill_id="VALID_FILL",
        order_pk=200,
        trading_pair="ETH/USD",
        exchange="BINANCE",
        side="SELL",
        quantity_filled=Decimal("5.0"),
        fill_price=Decimal("3000.0"),
        commission=Decimal("15.0"),
        commission_asset="USD",
        filled_at=datetime.now(UTC),
    )

    success = await valid_fill.publish_execution_event()

    if not success:
        failing_publisher.get_publication_stats()
    else:
        pass


async def demonstrate_fallback_logic():
    """Demonstrate intelligent fallback logic."""
    from gal_friday.models.fill import Fill

    # Create a fill with minimal data to test fallbacks
    minimal_fill = Fill(
        fill_pk=300,
        fill_id="MINIMAL_FILL",
        order_pk=300,
        trading_pair="DOGE/USD",
        exchange="COINBASE",
        side="BUY",
        quantity_filled=Decimal("1000.0"),
        fill_price=Decimal("0.08"),
        commission=Decimal("0.08"),
        commission_asset="USD",
        filled_at=datetime.now(UTC),
    )

    # No order relationship - test fallbacks
    minimal_fill.order = None
    minimal_fill.exchange_order_id = None


    with contextlib.suppress(Exception):
        minimal_fill.to_event()



async def demonstrate_monitoring():
    """Demonstrate monitoring and statistics capabilities."""
    from gal_friday.models.fill import ExecutionEventPublisher, Fill

    event_bus = MockEventBus()
    publisher = ExecutionEventPublisher(event_bus)
    Fill.set_event_publisher(publisher)

    # Create multiple fills and publish events
    fills = []
    for i in range(5):
        fill = Fill(
            fill_pk=400 + i,
            fill_id=f"BATCH_FILL_{i}",
            order_pk=400 + i,
            trading_pair="BTC/USD",
            exchange="BINANCE",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity_filled=Decimal(f"{0.1 * (i + 1)}"),
            fill_price=Decimal("50000.0"),
            commission=Decimal(f"{5.0 * (i + 1)}"),
            commission_asset="USD",
            filled_at=datetime.now(UTC),
        )
        fills.append(fill)


    successful_publications = 0
    for i, fill in enumerate(fills):
        success = await fill.publish_execution_event()
        if success:
            successful_publications += 1
        else:
            pass

    # Show final statistics
    publisher.get_publication_stats()

    # Show event bus contents
    for i, _event_data in enumerate(event_bus.published_events[:3]):  # Show first 3
        pass


async def main():
    """Run all demonstration examples."""
    await demonstrate_basic_usage()
    await demonstrate_error_handling()
    await demonstrate_fallback_logic()
    await demonstrate_monitoring()



if __name__ == "__main__":
    # Note: In a real application, you would need to ensure proper imports
    # and database connections. This is a demonstration script.
    try:
        asyncio.run(main())
    except ImportError:
        pass
    except Exception:
        pass
