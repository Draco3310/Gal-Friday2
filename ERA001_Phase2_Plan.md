# ERA001 Phase 2: Team Review Plan

## Current Status (Post-Phase 1)
- **Initial ERA001 errors**: 118
- **Errors removed in Phase 1**: 46
- **Current ERA001 errors**: 72

## Actual Error Distribution by File:
- monitoring_service.py: 17 errors
- feature_engine.py: 12 errors
- portfolio/position_manager.py: 7 errors
- models/order.py: 6 errors
- models/trade.py: 4 errors
- models/signal.py: 4 errors
- models/system_log.py: 3 errors
- execution/websocket_client.py: 3 errors
- model_lifecycle/registry.py: 2 errors
- model_lifecycle/experiment_manager.py: 2 errors
- market_price_service_enhancements.py: 2 errors
- dal/models/experiment.py: 2 errors
- backtesting_engine.py: 2 errors
- strategy_arbitrator.py: 1 error
- models/log.py: 1 error
- feature_engine_enhancements.py: 1 error
- execution_handler_enhancements.py: 1 error
- dal/connection_pool.py: 1 error
- dal/__init__.py: 1 error

## Revised Categories:
- **Phase 2 (Needs Review)**: ~44 errors
- **Phase 3 (Must Keep/Documentation)**: ~28 errors

## Phase 2: Items Requiring Team Review (42 errors)

### 1. Import Workarounds (1 error)
#### dal/__init__.py
- **Line 6**: `# from .influxdb_client import TimeSeriesDB # Commented out to avoid ModuleNotFoundError during alembic autogen`
- **Decision needed**: Should we implement conditional import or find another solution?

### 2. Database Relationships (19 errors)
#### dal/models/experiment.py (2 errors)
- **Lines 67-68**: Relationship definitions for ExperimentAssignment and ExperimentOutcome
- **Decision needed**: Are these models planned for implementation?

#### models/order.py (6 errors)
- Trade relationships commented out (circular import issues)
- **Decision needed**: How to resolve circular imports? Use TYPE_CHECKING?

#### models/trade.py (4 errors)
- Order relationships commented out (circular import issues)
- **Decision needed**: Same circular import resolution needed

#### models/signal.py (4 errors)
- Related model imports commented out
- **Decision needed**: Circular import pattern

#### models/system_log.py (3 errors)
- Event conversion imports commented out
- **Decision needed**: Move to TYPE_CHECKING or separate module?

### 3. Feature Engineering TODOs (8 errors)
#### feature_engine_enhancements.py (6 errors)
- **Line 371**: TODO for dynamic feature config updates
- **Line 385**: TODO for feature importance tracking
- **Lines 476-480**: Placeholder implementation comments
- **Decision needed**: Which features are still needed?

#### market_price_service_enhancements.py (2 errors)
- **Lines 371-372**: ConfigManager.set() method TODO
- **Decision needed**: Is this method planned for implementation?

### 4. Disabled Monitoring Features (20 errors)
#### monitoring_service.py (13 errors)
- **Lines 1046-1053**: Health check implementation (8 errors)
- **Lines 1162-1163**: System resource checks (2 errors)
- **Line 1168**: Position monitoring
- **Line 1173**: Risk alerts
- **Line 1176**: Performance tracking
- **Decision needed**: Which monitoring features should be re-enabled?

#### monitoring/dashboard_backend.py (7 errors)
- **Lines 301-330**: WebSocket implementation
- **Decision needed**: Is WebSocket functionality being replaced or should it be re-enabled?

### 5. Risk Management (5 errors)
#### portfolio_manager.py
- **Lines 589-595**: Risk validation checks
- **Decision needed**: Critical feature - why was it disabled? Should it be re-enabled?

### 6. Market Price Service Features (16 errors)
#### market_price_service.py
- **Lines 1058-1075**: Reconnection logic and rate limiting
- **Decision needed**: Were these causing issues? Are they needed for production?

### 7. Model Lifecycle Validation (5 errors)
#### model_lifecycle/experiment_manager.py
- **Lines 205-210**: Feature validation and engineering
- **Decision needed**: Why were these disabled? Are they needed?

### 8. Optional Validations (1 error)
#### strategy_arbitrator.py
- **Line 1432**: Feature registry validation warning
- **Decision needed**: Should this validation be enabled?

## Review Process

### Step 1: Categorize by Priority
**High Priority** (potential bugs or missing features):
- Risk management validations
- Monitoring health checks
- Market price service reconnection

**Medium Priority** (nice to have):
- Feature registry validation
- Dashboard WebSocket
- Feature importance tracking

**Low Priority** (can defer):
- TODOs for future enhancements
- Optional validations

### Step 2: Decision Matrix
For each item, determine:
1. **Why was it disabled?**
   - Performance issues?
   - Incomplete implementation?
   - External dependency problems?
   - Replaced by better solution?

2. **What's the impact of removal?**
   - Does it break functionality?
   - Does it reduce system reliability?
   - Does it affect monitoring/debugging?

3. **What's needed to re-enable?**
   - Simple uncomment?
   - Need to implement missing pieces?
   - Need to fix underlying issues?

### Step 3: Create Action Items
Based on review, create tickets for:
- **Re-enable**: Features that should be turned back on
- **Implement**: Missing functionality that's needed
- **Remove**: Confirmed obsolete code
- **Document**: Items that need explanation but should stay commented

## Expected Outcomes

### Likely Distribution:
- **Remove** (~15 errors): Confirmed obsolete
- **Re-enable** (~10 errors): Important features that were temporarily disabled
- **Implement** (~12 errors): TODOs for needed functionality
- **Defer** (~5 errors): Low priority items for future

### Deliverables:
1. Decision document with team consensus
2. JIRA/GitHub issues for implementation work
3. Updated codebase with re-enabled features
4. List of items to move to Phase 3 (documentation)

## Questions for Team Review

1. **Risk Management**: Why were risk limit validations disabled? Are we operating without safety checks?

2. **Monitoring**: Which health checks are critical for production?

3. **Reconnection Logic**: Have we had issues with the market price service disconnecting?

4. **ConfigManager.set()**: Is dynamic configuration a required feature?

5. **WebSocket Dashboard**: Is this being replaced by another solution?

6. **Database Models**: Are ExperimentAssignment and ExperimentOutcome on the roadmap?

7. **Feature Engineering**: Which validation steps are necessary vs nice-to-have?

## Next Meeting Agenda
1. Review high-priority items (15 min)
2. Quick decisions on medium priority (10 min)
3. Bulk decision on low priority (5 min)
4. Create action items and assign owners (10 min)
5. Schedule implementation timeline (5 min)