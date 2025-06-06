"""Example demonstrating usage of enhanced mock classes for testing.

This script shows how to use the enhanced mock implementations to create
comprehensive and realistic test scenarios for the Gal-Friday trading system.
"""

import asyncio
import sys
import threading
import time
from decimal import Decimal
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gal_friday.cli_service_mocks import (
    Console,
    ConfigManager,
    HaltRecoveryManager,
    LoggerService,
    MainAppController,
    MonitoringService,
    PortfolioManager,
    PubSubManager,
    Table,
)


async def demonstrate_enhanced_console():
    """Demonstrate enhanced Console mock features."""
    print("=== Enhanced Console Mock Demo ===")
    
    console = Console(force_terminal=True)
    
    # Basic printing with history tracking
    console.print("Welcome to Gal-Friday CLI", style="bold")
    console.print("System status: RUNNING", style="green")
    console.print("Portfolio value: $105,000", style="blue")
    
    # Retrieve output history for verification
    history = console.get_output_history()
    print(f"Captured {len(history)} console messages:")
    for i, msg in enumerate(history, 1):
        print(f"  {i}. {msg}")
    
    # Clear history for next test
    console.clear_output_history()
    print(f"History cleared. Current history length: {len(console.get_output_history())}")
    print()


async def demonstrate_enhanced_table():
    """Demonstrate enhanced Table mock features."""
    print("=== Enhanced Table Mock Demo ===")
    
    table = Table(title="Portfolio Status")
    
    # Add columns with styling
    table.add_column("Symbol", style="cyan", width=10)
    table.add_column("Position", style="magenta", width=15)
    table.add_column("P&L", style="green", width=12)
    
    # Add data rows
    table.add_row("BTC/USD", "1.0", "+$5,000")
    table.add_row("ETH/USD", "10.0", "+$2,500")
    table.add_row("ADA/USD", "1000.0", "-$500")
    
    # Retrieve structured data
    data = table.get_data()
    print(f"Table title: {data['title']}")
    print(f"Columns: {len(data['columns'])}")
    print(f"Rows: {len(data['rows'])}")
    
    for col in data['columns']:
        print(f"  Column: {col['header']} (style: {col['style']}, width: {col['width']})")
    
    for i, row in enumerate(data['rows']):
        print(f"  Row {i + 1}: {row}")
    print()


async def demonstrate_enhanced_logger():
    """Demonstrate enhanced LoggerService mock features."""
    print("=== Enhanced LoggerService Mock Demo ===")
    
    # Initialize with log level and capture enabled
    config = ConfigManager()
    logger = LoggerService(log_level="DEBUG", capture_logs=True)
    
    # Log messages at different levels
    logger.debug("Debug message with context", source_module="TestModule", context={"key": "debug_value"})
    logger.info("Trading system initialized", source_module="MainApp")
    logger.warning("High volatility detected", source_module="RiskManager", context={"volatility": 0.85})
    logger.error("API connection failed", source_module="ExchangeAPI", context={"endpoint": "/api/v1/trades"})
    
    # Simulate exception logging
    try:
        raise ValueError("Test exception for demonstration")
    except ValueError as e:
        logger.exception("Exception occurred during processing", source_module="TestModule")
    
    # Retrieve and analyze captured logs
    all_logs = logger.get_captured_logs()
    error_logs = logger.get_captured_logs(level="ERROR")
    warning_logs = logger.get_captured_logs(level="WARNING")
    
    print(f"Total logs captured: {len(all_logs)}")
    print(f"Error logs: {len(error_logs)}")
    print(f"Warning logs: {len(warning_logs)}")
    
    # Show detailed log information
    for log in all_logs[-3:]:  # Show last 3 logs
        print(f"  [{log['level']}] {log['message']} - Module: {log['source_module']}")
        if log['context']:
            print(f"    Context: {log['context']}")
    
    # Change log level and test filtering
    logger.set_log_level("WARNING")
    logger.debug("This debug message should be filtered out")
    logger.warning("This warning message should appear")
    
    filtered_logs = logger.get_captured_logs()
    print(f"After setting log level to WARNING: {len(filtered_logs)} total logs")
    print()


async def demonstrate_enhanced_monitoring():
    """Demonstrate enhanced MonitoringService mock features."""
    print("=== Enhanced MonitoringService Mock Demo ===")
    
    # Initialize monitoring service
    monitoring = MonitoringService(initial_halt_state=False)
    
    print(f"Initial halt state: {monitoring.is_halted()}")
    
    # Trigger halt and resume operations
    await monitoring.trigger_halt("High drawdown detected", "RiskManager")
    print(f"After halt: {monitoring.is_halted()}")
    
    await monitoring.trigger_halt("Duplicate halt attempt", "ManualTrigger")  # Should show already halted
    
    await monitoring.trigger_resume("ManualOverride")
    print(f"After resume: {monitoring.is_halted()}")
    
    await monitoring.trigger_resume("DuplicateResume")  # Should show not halted
    
    # Check halt history
    history = monitoring.get_halt_history()
    print(f"Halt/Resume history ({len(history)} events):")
    for event in history:
        print(f"  {event['action'].upper()}: {event.get('reason', 'N/A')} by {event['source']} at {event['timestamp']}")
    
    # Get detailed status
    status = monitoring.get_halt_status()
    print(f"Current status: {status}")
    print()


async def demonstrate_enhanced_portfolio():
    """Demonstrate enhanced PortfolioManager mock features."""
    print("=== Enhanced PortfolioManager Mock Demo ===")
    
    # Initialize with custom state
    portfolio = PortfolioManager(
        initial_state={
            "total_value": Decimal("50000"),
            "cash": Decimal("25000")
        }
    )
    
    print("Initial portfolio state:")
    state = portfolio.get_current_state()
    for key, value in state.items():
        if key != "positions":
            print(f"  {key}: {value}")
    
    # Simulate trading activity
    print("\nSimulating trades...")
    portfolio.simulate_trade_result(Decimal("1500"), "BTC/USD")  # Winning trade
    portfolio.simulate_trade_result(Decimal("-800"), "ETH/USD")   # Losing trade
    portfolio.simulate_trade_result(Decimal("2000"), "ADA/USD")   # Winning trade
    
    # Update portfolio state manually
    portfolio.update_state({
        "cash": Decimal("30000"),
        "total_value": Decimal("52700")
    })
    
    print("Portfolio state after trading:")
    updated_state = portfolio.get_current_state()
    for key, value in updated_state.items():
        if key != "positions":
            print(f"  {key}: {value}")
    
    # Reset to default state
    portfolio.reset_to_default()
    print("\nAfter reset to default:")
    reset_state = portfolio.get_current_state()
    print(f"  Total value: {reset_state['total_value']}")
    print(f"  Total trades: {reset_state['total_trades']}")
    print()


async def demonstrate_enhanced_pubsub():
    """Demonstrate enhanced PubSubManager mock features."""
    print("=== Enhanced PubSubManager Mock Demo ===")
    
    pubsub = PubSubManager()
    
    # Create mock event handlers
    trade_events = []
    halt_events = []
    
    async def trade_handler(event):
        trade_events.append(event)
        print(f"  Trade handler received: {event}")
    
    async def halt_handler(event):
        halt_events.append(event)
        print(f"  Halt handler received: {event}")
    
    # Subscribe to events
    pubsub.subscribe("TradeEvent", trade_handler)
    pubsub.subscribe("HaltEvent", halt_handler)
    pubsub.subscribe("TradeEvent", lambda e: print(f"  Secondary trade handler: {e}"))
    
    # Create mock events
    class MockEvent:
        def __init__(self, event_type, data):
            self.event_type = event_type
            self.data = data
        
        def __str__(self):
            return f"{self.event_type}: {self.data}"
    
    # Publish events
    print("Publishing events...")
    await pubsub.publish(MockEvent("TradeEvent", {"symbol": "BTC/USD", "action": "buy"}))
    await pubsub.publish(MockEvent("HaltEvent", {"reason": "manual", "source": "user"}))
    await pubsub.publish(MockEvent("TradeEvent", {"symbol": "ETH/USD", "action": "sell"}))
    
    # Check event history
    all_events = pubsub.get_published_events()
    trade_only_events = pubsub.get_published_events(event_type="TradeEvent")
    
    print(f"\nEvent summary:")
    print(f"  Total events published: {len(all_events)}")
    print(f"  Trade events: {len(trade_only_events)}")
    print(f"  Events received by trade handler: {len(trade_events)}")
    print(f"  Events received by halt handler: {len(halt_events)}")
    
    # Show subscribers
    trade_subscribers = pubsub.get_subscribers("TradeEvent")
    print(f"  TradeEvent subscribers: {len(trade_subscribers)}")
    print()


async def demonstrate_enhanced_config():
    """Demonstrate enhanced ConfigManager mock features."""
    print("=== Enhanced ConfigManager Mock Demo ===")
    
    # Initialize with custom configuration
    custom_config = {
        "trading": {
            "max_position_size": 0.05,
            "enable_stop_loss": True,
            "risk_level": "medium"
        },
        "api": {
            "timeout": 30,
            "retry_count": 3
        }
    }
    
    config = ConfigManager(config_data=custom_config)
    
    # Test various get methods
    print("Configuration values:")
    print(f"  Max position size: {config.get('trading.max_position_size', 0.02)}")
    print(f"  API timeout: {config.get_int('api.timeout', 10)}")
    print(f"  Stop loss enabled: {config.get_bool('trading.enable_stop_loss', False)}")
    print(f"  Risk level: {config.get('trading.risk_level', 'low')}")
    print(f"  Non-existent key: {config.get('missing.key', 'default_value')}")
    
    # Set new configuration values
    config.set("trading.new_feature", True)
    config.set("api.rate_limit", 100)
    
    # Get all configuration
    all_config = config.get_all()
    print(f"\nComplete configuration structure:")
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")
    
    print_dict(all_config)
    print()


async def demonstrate_enhanced_recovery():
    """Demonstrate enhanced HaltRecoveryManager mock features."""
    print("=== Enhanced HaltRecoveryManager Mock Demo ===")
    
    recovery = HaltRecoveryManager()
    
    # Add recovery items
    recovery.add_recovery_item("check_positions", "Verify all open positions", "high")
    recovery.add_recovery_item("reconcile_balances", "Reconcile account balances", "medium")
    recovery.add_recovery_item("restart_feeds", "Restart market data feeds", "high")
    recovery.add_recovery_item("validate_orders", "Validate pending orders", "low")
    
    # Check recovery status
    status = recovery.get_recovery_status()
    print(f"Recovery status: {status['completed_items']}/{status['total_items']} completed")
    print(f"Pending items: {status['pending_items']}")
    
    print("\nRecovery items:")
    for item in status['recovery_items']:
        print(f"  [{item['priority'].upper()}] {item['id']}: {item['description']}")
    
    # Complete some items
    print("\nCompleting recovery items...")
    success1 = recovery.complete_item("check_positions", "Alice")
    success2 = recovery.complete_item("restart_feeds", "Bob")
    success3 = recovery.complete_item("nonexistent_item", "Charlie")  # Should fail
    
    print(f"  check_positions completed: {success1}")
    print(f"  restart_feeds completed: {success2}")
    print(f"  nonexistent_item completed: {success3}")
    
    # Check final status
    final_status = recovery.get_recovery_status()
    print(f"\nFinal status: {final_status['completed_items']}/{final_status['total_items']} completed")
    
    if final_status['completed_items'] > 0:
        print("Completed items:")
        for item in final_status['completed_items']:
            print(f"  {item['id']} - completed by {item['completed_by']} at {item['completed_at']}")
    print()


async def demonstrate_enhanced_app_controller():
    """Demonstrate enhanced MainAppController mock features."""
    print("=== Enhanced MainAppController Mock Demo ===")
    
    controller = MainAppController()
    
    print(f"Initial running state: {controller.is_running()}")
    
    # Add shutdown callbacks
    shutdown_messages = []
    
    async def cleanup_callback():
        shutdown_messages.append("Database connections closed")
    
    async def notification_callback():
        shutdown_messages.append("Notifications sent")
    
    async def failing_callback():
        raise Exception("Simulated callback failure")
    
    controller.add_shutdown_callback(cleanup_callback)
    controller.add_shutdown_callback(notification_callback)
    controller.add_shutdown_callback(failing_callback)  # This will fail but not break shutdown
    
    # Stop the application
    print("Stopping application...")
    await controller.stop()
    
    print(f"Running state after stop: {controller.is_running()}")
    print(f"Shutdown callbacks executed: {len(shutdown_messages)}")
    for msg in shutdown_messages:
        print(f"  - {msg}")
    
    # Try to stop again (should show already stopped)
    print("\nAttempting to stop again...")
    await controller.stop()
    
    # Check status
    status = controller.get_status()
    print(f"Controller status: {status}")
    print()


async def demonstrate_thread_safety():
    """Demonstrate thread safety of enhanced mocks."""
    print("=== Thread Safety Demo ===")
    
    logger = LoggerService(capture_logs=True)
    monitoring = MonitoringService()
    portfolio = PortfolioManager()
    
    # Create concurrent operations
    async def log_worker(worker_id, count):
        for i in range(count):
            logger.info(f"Worker {worker_id} message {i}", source_module=f"Worker{worker_id}")
            await asyncio.sleep(0.01)  # Small delay to allow interleaving
    
    async def monitoring_worker(worker_id, count):
        for i in range(count):
            if i % 2 == 0:
                await monitoring.trigger_halt(f"Halt from worker {worker_id}-{i}", f"Worker{worker_id}")
            else:
                await monitoring.trigger_resume(f"Worker{worker_id}")
            await asyncio.sleep(0.01)
    
    async def portfolio_worker(worker_id, count):
        for i in range(count):
            profit_loss = Decimal("100") if i % 2 == 0 else Decimal("-50")
            portfolio.simulate_trade_result(profit_loss, f"PAIR{worker_id}")
            await asyncio.sleep(0.01)
    
    # Run workers concurrently
    print("Running concurrent operations...")
    tasks = []
    
    # Create 3 workers for each service
    for i in range(3):
        tasks.append(log_worker(i, 5))
        tasks.append(monitoring_worker(i, 3))
        tasks.append(portfolio_worker(i, 4))
    
    await asyncio.gather(*tasks)
    
    # Check results
    logs = logger.get_captured_logs()
    halt_history = monitoring.get_halt_history()
    portfolio_state = portfolio.get_current_state()
    
    print(f"Results after concurrent operations:")
    print(f"  Total logs: {len(logs)}")
    print(f"  Halt/resume events: {len(halt_history)}")
    print(f"  Total trades: {portfolio_state['total_trades']}")
    print(f"  Final halt state: {monitoring.is_halted()}")
    print()


async def run_comprehensive_demo():
    """Run all demonstrations in sequence."""
    print("🚀 Enhanced Mock Classes Comprehensive Demo\n")
    
    demos = [
        demonstrate_enhanced_console,
        demonstrate_enhanced_table,
        demonstrate_enhanced_logger,
        demonstrate_enhanced_monitoring,
        demonstrate_enhanced_portfolio,
        demonstrate_enhanced_pubsub,
        demonstrate_enhanced_config,
        demonstrate_enhanced_recovery,
        demonstrate_enhanced_app_controller,
        demonstrate_thread_safety,
    ]
    
    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"❌ Error in {demo.__name__}: {e}")
            continue
    
    print("✅ All demonstrations completed successfully!")
    print("\nThe enhanced mock classes provide:")
    print("  • Realistic behavior for testing")
    print("  • Thread-safe operations")
    print("  • Comprehensive state tracking")
    print("  • Configurable test scenarios")
    print("  • Easy verification of interactions")


if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(run_comprehensive_demo()) 