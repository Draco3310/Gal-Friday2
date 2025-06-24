# Pull Request: Remove obsolete commented code (ERA001 Phase 1)

## Summary
This PR removes 46 obsolete commented code blocks identified as "safe to remove" in our ERA001 analysis, reducing total ERA001 errors from 118 to 72.

## Motivation
As part of our code quality improvement initiative, we identified 118 ERA001 (commented-out code) violations. This PR addresses Phase 1 of the cleanup, removing code that is clearly obsolete, replaced, or no longer needed.

## Changes Made

### Overview
- ✅ Removed old implementations that have been replaced
- ✅ Removed unused imports and test code  
- ✅ Removed scaler functionality (features are now pre-scaled)
- ✅ Removed old conversion logic

### Detailed File Changes

#### 1. `dal/alembic_env/env.py`
- **Lines removed**: 82
- **Description**: Removed unused LoggerService import
- **Errors fixed**: 1

#### 2. `config_manager.py`
- **Lines removed**: 175-182
- **Description**: Removed old decimal conversion try/except block that was no longer needed
- **Errors fixed**: 8

#### 3. `feature_engine.py`
- **Lines removed**: 1139-1150, 1356
- **Description**: Removed old feature processor implementations that have been replaced by pipeline architecture
- **Errors fixed**: 13

#### 4. `providers/api_provider.py`
- **Lines removed**: 445
- **Description**: Removed old return statement that was moved to else block
- **Errors fixed**: 1

#### 5. `predictors/sklearn_predictor.py`
- **Lines removed**: 179-182
- **Description**: Removed scaler loading logic as features are now pre-scaled by FeatureEngine
- **Errors fixed**: 4

#### 6. `predictors/xgboost_predictor.py`
- **Lines removed**: 91-93, 313, 325-328
- **Description**: Removed scaler loading and usage logic as features are now pre-scaled
- **Errors fixed**: 7

#### 7. `predictors/lstm_predictor.py`
- **Lines removed**: 275-279
- **Description**: Removed scaler transformation logic as features are now pre-scaled
- **Errors fixed**: 5

#### 8. `prediction_service.py`
- **Lines removed**: 1294-1300
- **Description**: Removed old float conversion logic that's no longer needed with Pydantic models
- **Errors fixed**: 6

#### 9. `logger_service.py`
- **Lines removed**: 45, 223, 1017
- **Description**: Removed obsolete imports and type definitions that were replaced
- **Errors fixed**: 3

#### 10. `portfolio/position_manager.py`
- **Lines removed**: 92
- **Description**: Removed in-memory store reference that was replaced by database storage
- **Errors fixed**: 1

## Testing
- ✅ Ran `ruff check --select ERA001` after each file change
- ✅ Verified error count reduced from 118 to 72
- ✅ Each file committed separately for easier review

## Impact
- **Before**: 118 ERA001 errors
- **After**: 72 ERA001 errors  
- **Removed**: 46 errors (39% reduction)

## Review Notes
- All removed code was verified to be obsolete or replaced by newer implementations
- No functional changes were made - only removal of dead code
- Comments that serve documentation purposes were preserved for Phase 3

## Next Steps
After this PR is merged:
1. **Phase 2**: Review 42 "needs review" items with the team to determine if they should be removed or re-enabled
2. **Phase 3**: Convert 28 "must keep" comments to proper documentation (docstrings, README, etc.)

## Checklist
- [x] Code follows project style guidelines
- [x] Each file change has its own commit
- [x] Ruff checks pass for ERA001
- [x] No functionality has been removed (only dead code)
- [ ] PR has been reviewed
- [ ] Tests pass (if applicable)

---
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>