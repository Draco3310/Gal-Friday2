"""Example demonstrating usage of enhanced mock classes for testing.

This script shows how to use the enhanced mock implementations to create
comprehensive and realistic test scenarios for the Gal-Friday trading system.
"""

from decimal import Decimal
from pathlib import Path
import sys

import asyncio

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gal_friday.cli_service_mocks import (
    ConfigManager,
    Console,
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
    console = Console(force_terminal=True)

    # Basic printing with history tracking
    console.print("Welcome to Gal-Friday CLI", style="bold")
    console.print("System status: RUNNING", style="green")
    console.print("Portfolio value: $105,000", style="blue")

    # Retrieve output history for verification
    history = console.get_output_history()
    for _i, _msg in enumerate(history, 1):
        pass

    # Clear history for next test
    console.clear_output_history()


async def demonstrate_enhanced_table():
    """Demonstrate enhanced Table mock features."""
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

    for _col in data["columns"]:
        pass

    for _i, _row in enumerate(data["rows"]):
        pass


async def demonstrate_enhanced_logger():
    """Demonstrate enhanced LoggerService mock features."""
    # Initialize with log level and capture enabled
    ConfigManager()
    logger = LoggerService(log_level="DEBUG", capture_logs=True)

    # Log messages at different levels
    logger.debug("Debug message with context", source_module="TestModule", context={"key": "debug_value"})
    logger.info("Trading system initialized", source_module="MainApp")
    logger.warning("High volatility detected", source_module="RiskManager", context={"volatility": 0.85})
    logger.error("API connection failed", source_module="ExchangeAPI", context={"endpoint": "/api/v1/trades"})

    # Simulate exception logging
    try:
        raise ValueError("Test exception for demonstration")
    except ValueError:
        logger.exception("Exception occurred during processing", source_module="TestModule")

    # Retrieve and analyze captured logs
    all_logs = logger.get_captured_logs()
    logger.get_captured_logs(level="ERROR")
    logger.get_captured_logs(level="WARNING")


    # Show detailed log information
    for log in all_logs[-3:]:  # Show last 3 logs
        if log["context"]:
            pass

    # Change log level and test filtering
    logger.set_log_level("WARNING")
    logger.debug("This debug message should be filtered out")
    logger.warning("This warning message should appear")

    logger.get_captured_logs()


async def demonstrate_enhanced_monitoring():
    """Demonstrate enhanced MonitoringService mock features."""
    # Initialize monitoring service
    monitoring = MonitoringService(initial_halt_state=False)


    # Trigger halt and resume operations
    await monitoring.trigger_halt("High drawdown detected", "RiskManager")

    await monitoring.trigger_halt("Duplicate halt attempt", "ManualTrigger")  # Should show already halted

    await monitoring.trigger_resume("ManualOverride")

    await monitoring.trigger_resume("DuplicateResume")  # Should show not halted

    # Check halt history
    history = monitoring.get_halt_history()
    for _event in history:
        pass

    # Get detailed status
    monitoring.get_halt_status()


async def demonstrate_enhanced_portfolio():
    """Demonstrate enhanced PortfolioManager mock features."""
    # Initialize with custom state
    portfolio = PortfolioManager(
        initial_state={
            "total_value": Decimal(50000),
            "cash": Decimal(25000),
        },
    )

    state = portfolio.get_current_state()
    for key in state:
        if key != "positions":
            pass

    # Simulate trading activity
    portfolio.simulate_trade_result(Decimal(1500), "BTC/USD")  # Winning trade
    portfolio.simulate_trade_result(Decimal(-800), "ETH/USD")   # Losing trade
    portfolio.simulate_trade_result(Decimal(2000), "ADA/USD")   # Winning trade

    # Update portfolio state manually
    portfolio.update_state({
        "cash": Decimal(30000),
        "total_value": Decimal(52700),
    })

    updated_state = portfolio.get_current_state()
    for key in updated_state:
        if key != "positions":
            pass

    # Reset to default state
    portfolio.reset_to_default()
    portfolio.get_current_state()


async def demonstrate_enhanced_pubsub():
    """Demonstrate enhanced PubSubManager mock features."""
    pubsub = PubSubManager()

    # Create mock event handlers
    trade_events = []
    halt_events = []

    async def trade_handler(event):
        trade_events.append(event)

    async def halt_handler(event):
        halt_events.append(event)

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
    await pubsub.publish(MockEvent("TradeEvent", {"symbol": "BTC/USD", "action": "buy"}))
    await pubsub.publish(MockEvent("HaltEvent", {"reason": "manual", "source": "user"}))
    await pubsub.publish(MockEvent("TradeEvent", {"symbol": "ETH/USD", "action": "sell"}))

    # Check event history
    pubsub.get_published_events()
    pubsub.get_published_events(event_type="TradeEvent")


    # Show subscribers
    pubsub.get_subscribers("TradeEvent")


async def demonstrate_enhanced_config():
    """Demonstrate enhanced ConfigManager mock features."""
    # Initialize with custom configuration
    custom_config = {
        "trading": {
            "max_position_size": 0.05,
            "enable_stop_loss": True,
            "risk_level": "medium",
        },
        "api": {
            "timeout": 30,
            "retry_count": 3,
        },
    }

    config = ConfigManager(config_data=custom_config)

    # Test various get methods

    # Set new configuration values
    config.set("trading.new_feature", True)
    config.set("api.rate_limit", 100)

    # Get all configuration
    all_config = config.get_all()
    def print_dict(d, indent=0):
        for value in d.values():
            if isinstance(value, dict):
                print_dict(value, indent + 1)
            else:
                pass

    print_dict(all_config)


async def demonstrate_enhanced_recovery():
    """Demonstrate enhanced HaltRecoveryManager mock features."""
    recovery = HaltRecoveryManager()

    # Add recovery items
    recovery.add_recovery_item("check_positions", "Verify all open positions", "high")
    recovery.add_recovery_item("reconcile_balances", "Reconcile account balances", "medium")
    recovery.add_recovery_item("restart_feeds", "Restart market data feeds", "high")
    recovery.add_recovery_item("validate_orders", "Validate pending orders", "low")

    # Check recovery status
    status = recovery.get_recovery_status()

    for _item in status["recovery_items"]:
        pass

    # Complete some items
    recovery.complete_item("check_positions", "Alice")
    recovery.complete_item("restart_feeds", "Bob")
    recovery.complete_item("nonexistent_item", "Charlie")  # Should fail


    # Check final status
    final_status = recovery.get_recovery_status()

    if final_status["completed_items"] > 0:
        for _item in final_status["completed_items"]:
            pass


async def demonstrate_enhanced_app_controller():
    """Demonstrate enhanced MainAppController mock features."""
    controller = MainAppController()


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
    await controller.stop()

    for _msg in shutdown_messages:
        pass

    # Try to stop again (should show already stopped)
    await controller.stop()

    # Check status
    controller.get_status()


async def demonstrate_thread_safety():
    """Demonstrate thread safety of enhanced mocks."""
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
            profit_loss = Decimal(100) if i % 2 == 0 else Decimal(-50)
            portfolio.simulate_trade_result(profit_loss, f"PAIR{worker_id}")
            await asyncio.sleep(0.01)

    # Run workers concurrently
    tasks = []

    # Create 3 workers for each service
    for i in range(3):
        tasks.append(log_worker(i, 5))
        tasks.append(monitoring_worker(i, 3))
        tasks.append(portfolio_worker(i, 4))

    await asyncio.gather(*tasks)

    # Check results
    logger.get_captured_logs()
    monitoring.get_halt_history()
    portfolio.get_current_state()



async def run_comprehensive_demo():
    """Run all demonstrations in sequence."""
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
        except Exception:
            continue



if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(run_comprehensive_demo())
