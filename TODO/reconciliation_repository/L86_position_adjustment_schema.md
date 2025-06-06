# Provide Real Schema for Position Adjustments with Validation

## Task ID
**RECON-REPO-002**

## Priority
**High**

## Epic
**Data Schema & Position Management**

## Story
As a developer working with position reconciliation, I need a real schema for position adjustments with proper validation so that position corrections are processed accurately and safely.

## Problem Statement
Line 86 in `gal_friday/dal/repositories/reconciliation_repository.py` contains example code that needs to be replaced with a real schema for position adjustments, including proper validation to ensure data integrity.

## Acceptance Criteria
- [ ] Define comprehensive schema for position adjustment records
- [ ] Implement validation logic for adjustment data
- [ ] Add business rule validation (e.g., adjustment limits, authorization)
- [ ] Create proper error handling for invalid adjustments
- [ ] Remove example code and replace with production implementation
- [ ] Add audit trail for position adjustments
- [ ] Implement proper authorization checks for adjustments

## Technical Requirements
- Review line 86 in `gal_friday/dal/repositories/reconciliation_repository.py`
- Define Pydantic model for position adjustment schema
- Add validation for adjustment types and amounts
- Implement business rule validation
- Add database constraints for position adjustment table
- Create audit logging for all adjustments
- Follow financial data handling best practices

## Definition of Done
- [ ] Position adjustment schema is fully defined and documented
- [ ] Validation logic covers all adjustment scenarios
- [ ] Business rules are implemented and tested
- [ ] Example code is removed and replaced with production code
- [ ] Audit trail functionality is implemented
- [ ] Authorization checks are in place
- [ ] Unit tests cover all validation and business rules
- [ ] Integration tests verify adjustment processing
- [ ] Code review completed and approved

## Dependencies
- Understanding of position adjustment business requirements
- Knowledge of financial regulations and audit requirements
- Integration with position management and risk systems

## Estimated Effort
**Story Points: 8**

## Risk Assessment
**High Risk** - Position adjustments directly affect financial data and require strict validation

## Implementation Notes
```python
# Example position adjustment schema
from pydantic import BaseModel, validator
from typing import Optional, Dict, Any
from decimal import Decimal
from datetime import datetime
from enum import Enum

class AdjustmentType(str, Enum):
    CORRECTION = "correction"
    REBALANCE = "rebalance"
    SPLIT = "split"
    DIVIDEND = "dividend"
    FEES = "fees"

class PositionAdjustment(BaseModel):
    """Schema for position adjustment records."""
    
    # Required fields
    position_id: str
    adjustment_type: AdjustmentType
    quantity_change: Decimal
    reason: str
    authorized_by: str
    timestamp: datetime
    
    # Optional fields
    price_adjustment: Optional[Decimal] = None
    reference_id: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    
    @validator('quantity_change')
    def validate_quantity_change(cls, v):
        if v == 0:
            raise ValueError("quantity_change cannot be zero")
        return v
    
    @validator('reason')
    def validate_reason(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("reason must be at least 10 characters")
        return v.strip()
```

## Related Files
- `gal_friday/dal/repositories/reconciliation_repository.py` (line 86)
- Position management models and repositories
- Audit logging system
- Authorization and user management systems
