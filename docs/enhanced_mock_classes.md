# Enhanced Mock Classes for CLI Service

## Overview

This document describes the enhanced mock classes that have been implemented to replace placeholder mock implementations in the Gal-Friday trading system CLI service. These mocks provide realistic behavior for testing scenarios and are designed to be configurable, thread-safe, and properly typed.

## Key Features

All enhanced mock classes include:

- **Thread-safe operations** with proper locking mechanisms
- **Configurable behavior** for different testing scenarios
- **State tracking** for testing verification
- **Proper type annotations** for better development experience
- **Comprehensive documentation** for each method
- **Realistic behavior** that mirrors production components

## Enhanced Mock Classes

### 1. Enhanced Console Mock (`cli_service_mocks.py`)

**Location**: `gal_friday/cli_service_mocks.py`

**Features**:
- Output history tracking for test verification
- Thread-safe message recording
- Basic styling support
- Configurable terminal behavior

**Key Methods**:
```python
console = Console(force_terminal=True, legacy_windows=False)
console.print("Message", style="bold")
history = console.get_output_history()  # For testing
console.clear_output_history()  # Reset for next test
```

### 2. Enhanced Table Mock (`cli_service_mocks.py`)

**Features**:
- Column and row data tracking
- Thread-safe operations
- Structured data storage for testing

**Key Methods**:
```python
table = Table(title="Test Table")
table.add_column("Header", style="bold", width=20)
table.add_row("cell1", "cell2")
data = table.get_data()  # Get complete table structure
```

### 3. Enhanced MonitoringService Mock

**Available in both files with realistic HALT/RESUME behavior**

**Features**:
- State tracking with timestamps
- History of halt/resume events
- Thread-safe operations
- Configurable initial state

**Key Methods**:
```python
# Initialize with custom state
monitoring = MonitoringService(initial_halt_state=False)

# Standard operations
is_halted = monitoring.is_halted()
await monitoring.trigger_halt("Test reason", "Test source")
await monitoring.trigger_resume("Test source")

# Enhanced features for testing
history = monitoring.get_halt_history()
status = monitoring.get_halt_status()
```

### 4. Enhanced LoggerService Mock

**Features**:
- Configurable log levels
- Log message capture for testing
- Thread-safe logging
- Proper message formatting with context

**Key Methods**:
```python
# Initialize with configuration
logger = LoggerService(log_level="DEBUG", capture_logs=True)

# Standard logging
logger.info("Message", source_module="TestModule", context={"key": "value"})
logger.error("Error", exc_info=exception)

# Enhanced testing features
logs = logger.get_captured_logs(level="ERROR")  # Filter by level
logger.clear_captured_logs()  # Reset for next test
logger.set_log_level("WARNING")  # Change level dynamically
```

### 5. Enhanced PortfolioManager Mock

**Features**:
- Realistic portfolio state with Decimal precision
- Trade simulation capabilities
- State updates for testing scenarios
- Thread-safe operations

**Key Methods**:
```python
# Initialize with custom state
portfolio = PortfolioManager(initial_state={"total_value": Decimal("50000")})

# Standard operations
state = portfolio.get_current_state()

# Enhanced testing features
portfolio.update_state({"cash": Decimal("25000")})
portfolio.simulate_trade_result(Decimal("1000"), "BTC/USD")
portfolio.reset_to_default()
```

### 6. Enhanced ConfigManager Mock

**Features**:
- Realistic configuration structure
- Dot notation key access
- Type-safe value retrieval
- Thread-safe operations

**Key Methods**:
```python
# Initialize with custom config
config = ConfigManager(config_data={"section": {"key": "value"}})

# Standard operations
value = config.get("section.key", "default")
int_value = config.get_int("section.number", 0)
bool_value = config.get_bool("section.flag", False)

# Enhanced testing features
config.set("section.new_key", "new_value")
all_config = config.get_all()
```

### 7. Enhanced PubSubManager Mock

**Features**:
- Event publication and subscription tracking
- Async event handler execution
- Event history for testing verification
- Thread-safe operations

**Key Methods**:
```python
pubsub = PubSubManager()

# Standard operations
await pubsub.publish(event)
pubsub.subscribe("event_type", handler_function)
pubsub.unsubscribe("event_type", handler_function)

# Enhanced testing features
events = pubsub.get_published_events(event_type="TradeEvent")
subscribers = pubsub.get_subscribers("event_type")
pubsub.clear_published_events()
```

### 8. Enhanced MainAppController Mock

**Features**:
- Application lifecycle management
- Shutdown callback execution
- State tracking
- Thread-safe operations

**Key Methods**:
```python
controller = MainAppController()

# Standard operations
await controller.stop()

# Enhanced testing features
is_running = controller.is_running()
controller.add_shutdown_callback(async_callback_function)
status = controller.get_status()
```

### 9. Enhanced HaltRecoveryManager Mock (`cli_service_mocks.py`)

**Features**:
- Recovery workflow simulation
- Item completion tracking
- Recovery status reporting

**Key Methods**:
```python
recovery = HaltRecoveryManager()

# Standard operations
status = recovery.get_recovery_status()
success = recovery.complete_item("item_id", "user_name")

# Enhanced testing features
recovery.add_recovery_item("test_item", "Test description", "high")
recovery.reset_recovery_state()
```

## Usage in Tests

### Basic Test Setup

```python
import asyncio
from gal_friday.cli_service import (
    MockLoggerService,
    MockConfigManager,
    MockMonitoringService
)

async def test_example():
    # Create enhanced mocks
    config = MockConfigManager()
    logger = MockLoggerService(config, None, capture_logs=True)
    monitoring = MockMonitoringService()
    
    # Test operations
    await monitoring.trigger_halt("Test halt", "TestModule")
    assert monitoring.is_halted()
    
    # Verify logged messages
    logs = logger.get_captured_logs(level="INFO")
    assert len(logs) > 0
```

### Advanced Test Scenarios

```python
async def test_portfolio_simulation():
    # Setup with initial state
    portfolio = MockPortfolioManager(
        initial_state={"total_value": Decimal("100000")}
    )
    
    # Simulate trading
    portfolio.simulate_trade_result(Decimal("5000"), "BTC/USD")
    portfolio.simulate_trade_result(Decimal("-2000"), "ETH/USD")
    
    # Verify state changes
    state = portfolio.get_current_state()
    assert state["total_trades"] == 2
    assert state["winning_trades"] == 1
    assert state["losing_trades"] == 1
```

## Testing Benefits

### 1. **Realistic Behavior**
- Mocks behave like production components
- State persistence between operations
- Proper error handling and edge cases

### 2. **Comprehensive Verification**
- Capture and inspect all interactions
- Verify call sequences and parameters
- Check state changes over time

### 3. **Configurable Scenarios**
- Set up different initial states
- Simulate various failure conditions
- Test edge cases and error paths

### 4. **Thread Safety**
- Safe for concurrent testing
- No race conditions in test scenarios
- Proper synchronization mechanisms

## Migration from Old Mocks

### Before (Placeholder Implementation)
```python
class LoggerService:
    def info(self, message: str) -> None:
        rich_print(f"INFO: {message}")
```

### After (Enhanced Implementation)
```python
class LoggerService:
    def __init__(self, *, log_level: str = "INFO", capture_logs: bool = True):
        # Comprehensive initialization with configuration
        
    def info(self, message: str, *args, source_module: str | None = None, 
             context: dict | None = None) -> None:
        self._log("INFO", message, *args, source_module=source_module, context=context)
        
    def get_captured_logs(self, level: str | None = None) -> list[dict]:
        # Enhanced testing capabilities
```

## Best Practices

### 1. **Always Use Enhanced Features in Tests**
```python
# Good - Verify behavior
logs = logger.get_captured_logs(level="ERROR")
assert any("expected error" in log["message"] for log in logs)

# Good - Setup realistic state
portfolio = MockPortfolioManager(
    initial_state={"cash": Decimal("50000")}
)
```

### 2. **Reset State Between Tests**
```python
def setup_test():
    logger.clear_captured_logs()
    portfolio.reset_to_default()
    recovery.reset_recovery_state()
```

### 3. **Use Thread-Safe Operations**
```python
# All enhanced mocks are thread-safe by default
async def concurrent_test():
    tasks = [
        monitoring.trigger_halt("reason1", "source1"),
        monitoring.trigger_halt("reason2", "source2")
    ]
    await asyncio.gather(*tasks)
    # State remains consistent
```

## Conclusion

The enhanced mock classes provide a robust foundation for testing the CLI service and related components. They offer realistic behavior while maintaining the simplicity needed for effective testing. The comprehensive state tracking and configuration options enable thorough testing of complex scenarios while ensuring thread safety and proper error handling. 