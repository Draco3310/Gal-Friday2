# Complete Validation Logic in Position Manager __post_init__

## Task ID
**POS-MANAGER-001**

## Priority
**High**

## Epic
**Data Validation & Model Integrity**

## Story
As a developer working with position management, I need complete validation logic in the PositionManager's `__post_init__` method so that position data integrity is maintained and invalid states are caught early.

## Problem Statement
Line 160 in `gal_friday/portfolio/position_manager.py` contains a commented pass statement in the `__post_init__` method, indicating incomplete validation logic that needs to be finished or removed as dead code.

## Acceptance Criteria
- [ ] Review the current `__post_init__` method and determine what validation is needed
- [ ] Implement proper validation logic for position manager initialization
- [ ] Validate all required fields and their constraints
- [ ] Add proper error handling with descriptive error messages
- [ ] Remove the commented pass statement
- [ ] Ensure validation covers business logic constraints (e.g., position limits)
- [ ] Add comprehensive unit tests for validation scenarios

## Technical Requirements
- Review line 160 in `gal_friday/portfolio/position_manager.py`
- Implement validation for position manager state consistency
- Validate configuration parameters and dependencies
- Ensure proper error handling with custom exceptions
- Follow Pydantic or dataclass validation patterns
- Add logging for validation failures

## Definition of Done
- [ ] Complete validation logic is implemented in `__post_init__`
- [ ] All required fields and constraints are validated
- [ ] Proper error messages are provided for validation failures
- [ ] Dead code (commented pass) is removed
- [ ] Unit tests cover all validation scenarios
- [ ] Integration tests verify position manager initialization
- [ ] Code review completed and approved

## Dependencies
- Understanding of position management business rules
- Knowledge of required position manager configuration
- Integration requirements with portfolio management system

## Estimated Effort
**Story Points: 5**

## Risk Assessment
**Medium Risk** - Improper validation could lead to runtime errors or invalid position states

## Implementation Notes
```python
# Example validation implementation
def __post_init__(self):
    """Validate position manager configuration and state."""
    if not self.session_maker:
        raise ValueError("session_maker is required for database operations")
    
    if self.max_positions <= 0:
        raise ValueError("max_positions must be positive")
    
    if not self.risk_limits:
        raise ValueError("risk_limits configuration is required")
    
    # Validate risk limit values
    for limit_type, limit_value in self.risk_limits.items():
        if limit_value <= 0:
            raise ValueError(f"Risk limit {limit_type} must be positive")
    
    self.logger.info("Position manager validation completed successfully")
```

## Related Files
- `gal_friday/portfolio/position_manager.py` (line 160)
- Position model and validation rules
- Portfolio management configuration
- Risk management integration 