# Configurable Reconciliation System

## Overview

The Gal Friday reconciliation service has been enhanced with a configurable, strategy-based architecture that replaces the previous hardcoded "full" reconciliation approach. This new system provides multiple reconciliation strategies optimized for different operational scenarios.

## Key Improvements

### Before (Hardcoded Approach)
- Single "full" reconciliation type hardcoded in the system
- No flexibility for different operational needs
- Limited optimization for specific use cases
- Difficult to adapt to varying performance requirements

### After (Strategy Pattern)
- Multiple configurable reconciliation strategies
- Dynamic strategy selection based on configuration
- Optimized strategies for different scenarios
- Enterprise-grade flexibility and maintainability

## Supported Reconciliation Types

### 1. Full Reconciliation (`full`)
**Use Case:** Comprehensive daily reconciliation
- **Description:** Complete reconciliation of all positions, balances, and trades
- **Performance:** Highest thoroughness, moderate performance impact
- **Auto-Resolution:** Configurable threshold for small discrepancies
- **Best For:** End-of-day reconciliation, compliance reporting

### 2. Incremental Reconciliation (`incremental`)
**Use Case:** Frequent updates without full overhead
- **Description:** Reconciles only changes since last reconciliation
- **Performance:** Fast execution, minimal resource usage
- **Scope:** Configurable lookback period (default: 1 hour)
- **Best For:** Active trading periods, frequent monitoring

### 3. Real-Time Reconciliation (`real_time`)
**Use Case:** High-frequency trading environments
- **Description:** Continuous reconciliation focusing on recent critical data
- **Performance:** Optimized for speed, immediate alerts
- **Features:** No auto-resolution, immediate critical alerts
- **Best For:** Live trading, critical position monitoring

### 4. Emergency Reconciliation (`emergency`)
**Use Case:** Crisis situations requiring immediate attention
- **Description:** Real-time strategy with strictest thresholds
- **Features:** Critical alerting, no auto-resolution, emergency notifications
- **Best For:** System issues, suspected discrepancies, audit requirements

### 5. Position-Only Reconciliation (`position_only`)
**Use Case:** Quick position verification
- **Description:** Reconciles positions only, skips balances and trades
- **Performance:** Fast execution, focused scope
- **Best For:** Position monitoring, quick checks

### 6. Balance-Only Reconciliation (`balance_only`)
**Use Case:** Account balance verification
- **Description:** Reconciles balances only, skips positions and trades
- **Performance:** Fast execution, focused scope
- **Best For:** Account balance monitoring, funding checks

## Configuration

### Basic Configuration
```yaml
reconciliation:
  default_type: "full"  # Default strategy
  interval_minutes: 60  # Periodic reconciliation interval
  auto_resolve_threshold: 0.001  # 0.1% auto-resolution threshold
  enable_alerts: true
```

### Strategy-Specific Configuration
```yaml
reconciliation:
  real_time_cutoff_minutes: 15  # Real-time lookback
  incremental_cutoff_hours: 1   # Incremental lookback
  emergency_alert_threshold: 0.01  # Emergency threshold
```

### Environment-Specific Configuration
```yaml
# Production - Frequent, strict reconciliation
production:
  reconciliation:
    default_type: "full"
    interval_minutes: 30
    auto_resolve_threshold: 0.001

# Development - Lenient, incremental reconciliation
development:
  reconciliation:
    default_type: "incremental"
    interval_minutes: 60
    auto_resolve_threshold: 0.01
```

## API Usage

### Programmatic Strategy Selection
```python
# Use configured default strategy
result = await reconciliation_service.perform_configurable_reconciliation()

# Override with specific strategy
result = await reconciliation_service.perform_configurable_reconciliation(
    ReconciliationType.REAL_TIME
)

# Emergency reconciliation
result = await reconciliation_service.perform_emergency_reconciliation()
```

### Legacy Compatibility
```python
# Legacy method still works (uses new strategy internally)
report = await reconciliation_service.run_reconciliation()

# Legacy method with strategy override
report = await reconciliation_service.run_reconciliation(
    ReconciliationType.INCREMENTAL
)
```

### Monitoring and Status
```python
# Get supported reconciliation types
types = await reconciliation_service.get_supported_reconciliation_types()

# Get current configuration
config = reconciliation_service.get_current_reconciliation_config()

# Get service status with strategy information
status = await reconciliation_service.get_reconciliation_status()
```

## Strategy Selection Guidelines

### High-Frequency Trading
```yaml
reconciliation:
  default_type: "real_time"
  real_time_cutoff_minutes: 5
  interval_minutes: 10
```

### Standard Trading Operations
```yaml
reconciliation:
  default_type: "incremental"
  incremental_cutoff_hours: 2
  interval_minutes: 30
```

### End-of-Day Processing
```yaml
reconciliation:
  default_type: "full"
  interval_minutes: 60
  historical_lookback_hours: 24
```

### Crisis Management
```python
# Trigger emergency reconciliation programmatically
await reconciliation_service.perform_emergency_reconciliation()
```

## Performance Characteristics

| Strategy | Execution Time | Resource Usage | Thoroughness | Use Case |
|----------|---------------|----------------|--------------|----------|
| Full | High | High | Complete | Daily reconciliation |
| Incremental | Medium | Medium | Recent changes | Active trading |
| Real-Time | Low | Low | Critical data | HFT monitoring |
| Emergency | Low | Low | Critical focus | Crisis response |
| Position-Only | Low | Low | Positions only | Quick checks |
| Balance-Only | Low | Low | Balances only | Account verification |

## Error Handling and Fallbacks

### Strategy Validation
- Unsupported strategies fall back to configured default
- Invalid configuration triggers warning and uses FULL strategy
- Missing configuration uses sensible defaults

### Failure Recovery
- Failed strategies are logged with full context
- Consecutive failures trigger critical alerts
- Emergency reconciliation available as fallback

### Backward Compatibility
- Legacy `run_reconciliation()` method preserved
- Legacy `ReconciliationReport` format maintained
- Existing integrations continue to work unchanged

## Monitoring and Alerting

### Strategy-Specific Metrics
- Execution time per strategy type
- Discrepancy patterns by strategy
- Success/failure rates per strategy

### Enhanced Alerts
- Strategy-specific alert thresholds
- Real-time critical discrepancy alerts
- Emergency reconciliation notifications

### Audit Trail
- Complete reconciliation history with strategy used
- Configuration changes tracked
- Performance metrics per strategy

## Migration Guide

### From Hardcoded "Full" Reconciliation

1. **No Code Changes Required** - Existing code continues to work
2. **Optional Configuration** - Add strategy configuration for customization
3. **Gradual Migration** - Switch to new API methods when ready

### Recommended Migration Steps

1. **Phase 1:** Add configuration file with default strategy
2. **Phase 2:** Switch to `perform_configurable_reconciliation()` API
3. **Phase 3:** Implement strategy-specific monitoring
4. **Phase 4:** Optimize strategies based on operational patterns

## Examples

### Example 1: Basic Strategy Configuration
```python
# Simple configuration-based reconciliation
result = await reconciliation_service.perform_configurable_reconciliation()
print(f"Reconciliation completed: {result.status}")
print(f"Strategy used: {result.reconciliation_type.value}")
print(f"Discrepancies found: {result.discrepancies_found}")
```

### Example 2: Dynamic Strategy Selection
```python
# Choose strategy based on trading activity
if is_high_frequency_trading:
    strategy_type = ReconciliationType.REAL_TIME
elif is_end_of_day:
    strategy_type = ReconciliationType.FULL
else:
    strategy_type = ReconciliationType.INCREMENTAL

result = await reconciliation_service.perform_configurable_reconciliation(strategy_type)
```

### Example 3: Crisis Response
```python
# Emergency reconciliation with immediate alerts
try:
    result = await reconciliation_service.perform_emergency_reconciliation()
    if result.discrepancies_found > 0:
        await notify_risk_management(result)
except Exception as e:
    await escalate_to_operations(f"Emergency reconciliation failed: {e}")
```

## Best Practices

1. **Strategy Selection:**
   - Use FULL for comprehensive daily reconciliation
   - Use INCREMENTAL for frequent monitoring during trading hours
   - Use REAL_TIME for high-frequency trading environments
   - Use EMERGENCY for crisis situations only

2. **Configuration Management:**
   - Set environment-specific configurations
   - Monitor performance and adjust thresholds
   - Use version control for configuration changes

3. **Monitoring:**
   - Track reconciliation performance by strategy
   - Set up alerts for strategy failures
   - Monitor auto-resolution rates

4. **Error Handling:**
   - Always handle strategy execution exceptions
   - Implement fallback mechanisms for critical operations
   - Log strategy selection decisions for audit trails

This configurable reconciliation system provides the flexibility and performance optimization required for enterprise-grade trading operations while maintaining full backward compatibility with existing systems. 