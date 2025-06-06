#!/usr/bin/env python3
"""
Example demonstrating the enterprise-grade execution event creation and publishing system.

This example shows how to:
1. Create Fill instances with proper data
2. Generate ExecutionReportEvents using the new builder
3. Publish events to an event bus
4. Monitor publication statistics
5. Handle errors gracefully
"""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict

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
            'topic': topic,
            'event': event,
            'timestamp': datetime.now(timezone.utc)
        })
        print(f"📤 Published event to topic '{topic}': {event.event_id}")


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
    print("🚀 Starting Enterprise Execution Event Demonstration\n")
    
    # Import the actual classes (these would be real imports in production)
    from gal_friday.models.fill import Fill, ExecutionEventBuilder, ExecutionEventPublisher
    
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
        filled_at=datetime.now(timezone.utc)
    )
    
    # Create a mock order and associate it
    order = MockOrder()
    order.fills = [fill]
    fill.order = order
    
    print("📊 Fill Details:")
    summary = fill.get_execution_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("\n🔧 Creating Execution Event...")
    
    try:
        # Create execution event using the enhanced to_event method
        event = fill.to_event()
        print(f"✅ Event created successfully!")
        print(f"   Event ID: {event.event_id}")
        print(f"   Trading Pair: {event.trading_pair}")
        print(f"   Quantity Filled: {event.quantity_filled}")
        print(f"   Average Fill Price: {event.average_fill_price}")
        print(f"   Order Status: {event.order_status}")
        
        print("\n📤 Publishing Event...")
        
        # Publish the event
        success = await fill.publish_execution_event()
        
        if success:
            print("✅ Event published successfully!")
            
            # Show publication statistics
            stats = publisher.get_publication_stats()
            print(f"\n📈 Publication Statistics:")
            print(f"   Published Events: {stats['published_events']}")
            print(f"   Failed Publications: {stats['failed_publications']}")
            print(f"   Success Rate: {stats['success_rate']:.2%}")
            
        else:
            print("❌ Event publication failed!")
            
    except Exception as e:
        print(f"❌ Error in execution event workflow: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n🛡️ Demonstrating Error Handling\n")
    
    from gal_friday.models.fill import Fill, ExecutionEventPublisher
    
    # Create a fill with invalid data to test validation
    invalid_fill = Fill(
        fill_pk=999,
        fill_id="INVALID_FILL",
        order_pk=999,
        trading_pair="",  # Invalid empty trading pair
        exchange="TEST",
        side="BUY",
        quantity_filled=Decimal("0"),  # Invalid zero quantity
        fill_price=Decimal("100.0"),
        commission=Decimal("1.0"),
        commission_asset="USD",
        filled_at=datetime.now(timezone.utc)
    )
    
    print("🧪 Testing with invalid fill data...")
    
    try:
        event = invalid_fill.to_event()
        print("❌ Unexpected success - validation should have failed!")
    except Exception as e:
        print(f"✅ Validation correctly caught error: {e}")
    
    # Test event bus failure handling
    print("\n🧪 Testing event bus failure handling...")
    
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
        filled_at=datetime.now(timezone.utc)
    )
    
    success = await valid_fill.publish_execution_event()
    
    if not success:
        print("✅ Event bus failure correctly handled!")
        stats = failing_publisher.get_publication_stats()
        print(f"   Failed Publications: {stats['failed_publications']}")
    else:
        print("❌ Expected publication to fail!")


async def demonstrate_fallback_logic():
    """Demonstrate intelligent fallback logic."""
    print("\n🔄 Demonstrating Intelligent Fallback Logic\n")
    
    from gal_friday.models.fill import Fill, ExecutionEventBuilder
    
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
        filled_at=datetime.now(timezone.utc)
    )
    
    # No order relationship - test fallbacks
    minimal_fill.order = None
    minimal_fill.exchange_order_id = None
    
    print("🧪 Testing fallback logic with minimal data...")
    
    try:
        event = minimal_fill.to_event()
        print("✅ Event created successfully with fallbacks!")
        print(f"   Exchange Order ID (fallback): {event.exchange_order_id}")
        print(f"   Order Status (fallback): {event.order_status}")
        print(f"   Order Type (fallback): {event.order_type}")
        print(f"   Signal ID: {event.signal_id or 'None (expected)'}")
        
    except Exception as e:
        print(f"❌ Fallback logic failed: {e}")


async def demonstrate_monitoring():
    """Demonstrate monitoring and statistics capabilities."""
    print("\n📊 Demonstrating Monitoring and Statistics\n")
    
    from gal_friday.models.fill import Fill, ExecutionEventPublisher
    
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
            filled_at=datetime.now(timezone.utc)
        )
        fills.append(fill)
    
    print("🔄 Publishing batch of events...")
    
    successful_publications = 0
    for i, fill in enumerate(fills):
        success = await fill.publish_execution_event()
        if success:
            successful_publications += 1
            print(f"   ✅ Event {i+1}/5 published")
        else:
            print(f"   ❌ Event {i+1}/5 failed")
    
    # Show final statistics
    stats = publisher.get_publication_stats()
    print(f"\n📈 Final Statistics:")
    print(f"   Total Published: {stats['published_events']}")
    print(f"   Total Failed: {stats['failed_publications']}")
    print(f"   Success Rate: {stats['success_rate']:.2%}")
    print(f"   Last Publication: {stats['last_publication_time']}")
    
    # Show event bus contents
    print(f"\n📨 Event Bus Contents:")
    print(f"   Total Events Received: {len(event_bus.published_events)}")
    for i, event_data in enumerate(event_bus.published_events[:3]):  # Show first 3
        print(f"   Event {i+1}: {event_data['event'].trading_pair} - {event_data['event'].quantity_filled}")


async def main():
    """Run all demonstration examples."""
    print("=" * 80)
    print("   ENTERPRISE-GRADE EXECUTION EVENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    await demonstrate_basic_usage()
    await demonstrate_error_handling()
    await demonstrate_fallback_logic()
    await demonstrate_monitoring()
    
    print("\n" + "=" * 80)
    print("   DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\n✨ The enterprise-grade execution event system is ready for production!")
    print("🔧 Key features demonstrated:")
    print("   • Comprehensive event creation with validation")
    print("   • Intelligent fallback logic for missing data")
    print("   • Robust error handling and recovery")
    print("   • Event publishing with monitoring")
    print("   • Production-ready statistics and logging")


if __name__ == "__main__":
    # Note: In a real application, you would need to ensure proper imports
    # and database connections. This is a demonstration script.
    try:
        asyncio.run(main())
    except ImportError as e:
        print(f"⚠️  Import Error: {e}")
        print("💡 This example requires the actual gal_friday modules to be importable.")
        print("   In a real environment, ensure proper Python path and dependencies.")
    except Exception as e:
        print(f"❌ Error running demonstration: {e}")
        print("💡 This is expected in a standalone environment.") 