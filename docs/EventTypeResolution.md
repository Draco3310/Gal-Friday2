# Resolving EventType Issues

This document explains how to properly address mypy errors related to EventType in the Gal-Friday2 project.

## The Issue

We're seeing mypy errors like:

```
src\gal_friday\event_bus.py:24: error: "type[EventType]" has no attribute "PORTFOLIO_UPDATE"  [attr-defined]
```

This is happening because:

1. We have two definitions of `EventType` in the codebase:
   - `src/gal_friday/core/events.py`: The primary definition 
   - `src/gal_friday/core/placeholder_classes.py`: A secondary definition for avoiding circular imports

2. In `src/gal_friday/event_bus.py`, we're trying to extend the `EventType` enum by adding attributes at runtime:
   ```python
   EventType.PORTFOLIO_UPDATE = EventType.SYSTEM_STATE_CHANGE
   # ...etc
   ```

3. This approach doesn't work with mypy type checking because Enum values are meant to be defined at class definition time, not dynamically added later.

## Solution Approach

Choose one of these approaches to fix the issues:

### Option 1: Properly extend the EventType enum

Instead of dynamically adding attributes, update the `events.py` file to include all needed event types:

```python
class EventType(Enum):
    """Enumeration of possible event types within the system."""

    # Existing types...
    
    # Add these new types
    PORTFOLIO_UPDATE = auto()
    PORTFOLIO_RECONCILIATION = auto()
    PORTFOLIO_DISCREPANCY = auto()
    RISK_LIMIT_ALERT = auto()
    MARKET_DATA_RAW = auto()
    FEATURE_CALCULATED = auto()  # Instead of FEATURES_CALCULATED
```

### Option 2: Using type ignores

If you need to maintain backward compatibility and can't change the enum, add type ignores:

```python
# Add missing event types needed by tests
# These will be available as EventType.PORTFOLIO_UPDATE, etc.
EventType.PORTFOLIO_UPDATE = EventType.SYSTEM_STATE_CHANGE  # type: ignore
EventType.PORTFOLIO_RECONCILIATION = EventType.SYSTEM_STATE_CHANGE  # type: ignore
EventType.PORTFOLIO_DISCREPANCY = EventType.SYSTEM_STATE_CHANGE  # type: ignore
EventType.RISK_LIMIT_ALERT = EventType.POTENTIAL_HALT_TRIGGER  # type: ignore
EventType.MARKET_DATA_RAW = EventType.MARKET_DATA_L2  # type: ignore
EventType.FEATURE_CALCULATED = EventType.FEATURES_CALCULATED  # type: ignore
```

### Option 3: Creating a proxy class

Create a proxy class that wraps the original EventType but allows dynamic attributes:

```python
class EventTypeProxy:
    """Proxy class for EventType to allow dynamic attributes."""
    
    def __init__(self, event_type_cls):
        self._event_type_cls = event_type_cls
        self._extra_mappings = {}
    
    def __getattr__(self, name):
        if name in self._extra_mappings:
            return self._extra_mappings[name]
        return getattr(self._event_type_cls, name)
    
    def add_mapping(self, name, value):
        self._extra_mappings[name] = value

# Create proxy
proxy_event_type = EventTypeProxy(EventType)

# Add mappings
proxy_event_type.add_mapping('PORTFOLIO_UPDATE', EventType.SYSTEM_STATE_CHANGE)
# ...and so on
```

## Recommendation

Option 1 is the most type-safe and maintainable approach. It ensures all event types are properly defined in one place and eliminates runtime modifications of the enum. 