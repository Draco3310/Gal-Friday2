# ERA001 Commented-Out Code - Complete Action Plan

## Overview
Total ERA001 errors: 118
- **Safe to Remove**: 48 errors (41%)
- **Needs Review**: 42 errors (36%)
- **Must Keep**: 28 errors (23%)

## Detailed Error Listing by Category

### 🟢 SAFE TO REMOVE (48 errors)

#### 1. dal/alembic_env/env.py (1 error)
- **Line 82**: `# from gal_friday.logger_service import LoggerService # Not strictly used here`
- **Reason**: Unused import, explicitly marked as not used
- **Action**: Delete line

#### 2. execution/kraken.py (3 errors)
- **Line 178**: `# Test with sample data`
- **Line 188**: `# data = {...}`
- **Line 189**: `# api._handle_error_response(data)`
- **Reason**: Test code in production file
- **Action**: Delete all three lines

#### 3. feature_engine.py (3 errors)
- **Line 1089**: `# Old sequential processing:`
- **Line 1090**: `# return [await self._safe_compute_indicator(ind, price_data) for ind in indicators]`
- **Line 1205**: `# Build pipelines after initializing components` (if this is the ERA001)
- **Reason**: Old implementation replaced by concurrent processing
- **Action**: Delete lines

#### 4. providers/api_provider.py (1 error)
- **Line 445**: `# return {"symbol": symbol, "error": "Pair not found"}`
- **Reason**: Old return statement that was moved
- **Action**: Delete line

#### 5. config_manager.py (8 errors)
- **Line 175**: `# try:`
- **Line 176**: `# default = Decimal(str(default))`
- **Line 177**: `# except (ValueError, TypeError): # BLE001`
- **Line 178**: `# self._logger.warning(`
- **Line 179**: `# "Invalid default value '%s' for get_decimal, using 0.0",`
- **Line 180**: `# default,`
- **Line 181**: `# )`
- **Line 182**: `# default = Decimal("0.0")`
- **Reason**: Old decimal conversion logic, no longer needed
- **Action**: Delete entire commented block

#### 6. predictors/lightgbm_predictor.py (3 errors)
- **Line 185**: `# else:`
- **Line 186**: `# self.logger.info("No scaler_path provided. Proceeding without a scaler.")`
- **Line 187**: `# self.scaler = None`
- **Reason**: Scaler functionality removed, features expected pre-scaled
- **Action**: Delete lines

#### 7. predictors/neural_network_predictor.py (3 errors)
- **Line 218**: `# else:`
- **Line 219**: `# self.logger.info("No scaler_path provided. Proceeding without a scaler.")`
- **Line 220**: `# self.scaler = None`
- **Reason**: Scaler functionality removed, features expected pre-scaled
- **Action**: Delete lines

#### 8. predictors/sklearn_predictor.py (5 errors)
- **Line 177**: `# if self.scaler_path: # Removed scaler loading logic`
- **Line 178**: `# self.scaler = joblib.load(self.scaler_path)`
- **Line 180**: `# else:`
- **Line 181**: `# self.logger.info("No scaler_path provided. Proceeding without a scaler.")`
- **Line 182**: `# self.scaler = None`
- **Reason**: Scaler functionality removed, features expected pre-scaled
- **Action**: Delete lines

#### 9. predictors/xgboost_predictor.py (7 errors)
- **Line 91**: `# else:`
- **Line 92**: `# self.logger.info("No scaler_path provided. Proceeding without a scaler.")`
- **Line 93**: `# self.scaler = None`
- **Line 313**: `# scaler = None`
- **Line 325**: `# scaler, error = cls._load_scaler(scaler_path, model_id, logger)`
- **Line 327**: `# result["error"] = error.get("error", "Failed to load scaler")`
- **Line 328**: `# return result`
- **Reason**: Scaler functionality removed, features expected pre-scaled
- **Action**: Delete lines

### 🟡 NEEDS TEAM REVIEW (42 errors)

#### 1. dal/__init__.py (1 error)
- **Line 6**: `# from .influxdb_client import TimeSeriesDB # Commented out to avoid ModuleNotFoundError during alembic autogen`
- **Issue**: Import needed but causes alembic errors
- **Question**: Should we implement conditional import?

#### 2. dal/models/experiment.py (2 errors)
- **Line 67**: `# assignments = relationship("ExperimentAssignment", back_populates="experiment")`
- **Line 68**: `# outcomes = relationship("ExperimentOutcome", back_populates="experiment")`
- **Issue**: Relationships to non-existent models
- **Question**: Are these models planned?

#### 3. feature_engine_enhancements.py (6 errors)
- **Line 371**: `# TODO: Implement method to update feature configs dynamically`
- **Line 385**: `# TODO: Consider implementing feature importance tracking`
- **Line 476**: `# Original implementation (if exists)`
- **Line 477**: `# ... old code ...`
- **Line 479**: `# New implementation`
- **Line 480**: `# ... new code ...`
- **Issue**: Various TODOs and placeholders
- **Question**: Which features are still needed?

#### 4. market_price_service.py (16 errors)
- **Lines 1058-1062**: Reconnection logic (5 errors)
- **Lines 1065-1075**: Rate limiting implementation (11 errors)
- **Issue**: Disabled reconnection and rate limiting features
- **Question**: Were these causing issues or are they needed?

#### 5. market_price_service_enhancements.py (2 errors)
- **Line 371**: `# TODO: Implement ConfigManager.set() method or use alternative approach`
- **Line 372**: `# self.config_manager.set(config_path, new_value)`
- **Issue**: Missing ConfigManager functionality
- **Question**: Is set() method planned?

#### 6. model_lifecycle/experiment_manager.py (5 errors)
- **Line 205**: `# if not self._validate_features(features):`
- **Line 206**: `# return {"error": "Invalid features"}`
- **Line 208**: `# engineered_features = self._engineer_features(features)`
- **Line 209**: `# if engineered_features is None:`
- **Line 210**: `# return {"error": "Feature engineering failed"}`
- **Issue**: Disabled validation and feature engineering
- **Question**: Why were these disabled?

#### 7. monitoring/dashboard_backend.py (7 errors)
- **Line 301**: `# await self.websocket.send_json({"type": "update", "data": data})`
- **Line 304**: `# self.logger.error("WebSocket error: %s", e)`
- **Line 322**: `# async def handle_websocket(self, websocket: WebSocket):`
- **Line 325**: `# await websocket.accept()`
- **Line 326**: `# self.websockets.append(websocket)`
- **Line 327**: `# # Handle incoming messages`
- **Line 330**: `# self.websockets.remove(websocket)`
- **Issue**: WebSocket implementation disabled
- **Question**: Is this being replaced?

#### 8. monitoring_service.py (13 errors)
- **Lines 1046-1053**: Health check implementation (8 errors)
- **Lines 1162-1163**: System resource checks (2 errors)
- **Line 1168**: Position monitoring
- **Line 1173**: Risk alerts
- **Line 1176**: Performance tracking
- **Issue**: Multiple monitoring features disabled
- **Question**: Which are still needed?

#### 9. portfolio_manager.py (5 errors)
- **Line 589**: `# if not self._validate_risk_limits(position):`
- **Line 592**: `# "Risk limits exceeded for %s"`
- **Line 593**: `# position.symbol`
- **Line 594**: `# )`
- **Line 595**: `# return False`
- **Issue**: Risk validation disabled
- **Question**: Critical feature - why disabled?

#### 10. strategy_arbitrator.py (1 error)
- **Line 1432**: `# self.logger.warning(f"Feature '{feature_name}' in rule not found in registry. Rule may fail.")`
- **Issue**: Optional feature registry validation
- **Question**: Should this be enabled?

### 🔴 MUST KEEP (28 errors)

#### 1. backtesting_engine.py (2 errors)
- **Line 3088**: `# High >= max(open, low, close)`
- **Line 3092**: `# Low <= min(open, high, close)`
- **Purpose**: Documents OHLCV validation logic
- **Action**: Add `# noqa: ERA001` or convert to docstring

#### 2. dal/connection_pool.py (1 error)
- **Line 118**: `# await session.commit() # Typically not done here, but in the repository`
- **Purpose**: Architectural decision documentation
- **Action**: Move to method docstring

#### 3. execution_handler.py (25 errors)
- **Lines 1214, 1216**: Old reconciliation pattern documentation (2 errors)
- **Lines 1222, 1224, 1225, 1227, 1229, 1231**: Reconciliation logic notes (6 errors)
- **Lines 1237, 1240**: Reconciliation completion notes (2 errors)
- **Lines 1329, 1331, 1332**: Circuit breaker pattern notes (3 errors)
- **Line 1756**: `# Note: MarketPriceService is not implemented, using enhanced service instead`
- **Lines 1932-1935, 1937, 1943**: Error handling documentation (6 errors)
- **Lines 2061-2064**: Performance optimization notes (4 errors)
- **Lines 3122-3124**: WebSocket subscription documentation (3 errors)
- **Purpose**: Critical implementation notes and API behavior documentation
- **Action**: Convert to proper documentation

## Implementation Plan

### Phase 1: Safe Removals (Day 1)
1. Create branch: `fix/era001-safe-removals`
2. Remove all 48 "Safe to Remove" errors
3. Run ruff check after each file
4. Commit each file separately with message: "Remove obsolete commented code from [filename]"
5. Create PR with summary of removals

### Phase 2: Team Review (Days 2-3)
1. Create `ERA001_Review_Document.md` with all 42 "Needs Review" items
2. For each item document:
   - Current state
   - Original purpose
   - Why it was disabled
   - Impact of removal vs re-enabling
3. Schedule team review meeting
4. Create tickets for items to re-enable

### Phase 3: Documentation Conversion (Day 4)
1. Convert all 28 "Must Keep" comments to proper documentation:
   - Method/class docstrings
   - README sections
   - Architecture documentation
2. Add `# noqa: ERA001` where inline comments must remain
3. Update coding standards

## File-by-File Execution Guide

### Phase 1 Files (in order of execution):

1. **dal/alembic_env/env.py**
   - Delete line 82
   - Run: `.venv/Scripts/ruff.exe check gal_friday/dal/alembic_env/env.py --select ERA001`
   - Commit: "Remove unused LoggerService import from alembic env"

2. **execution/kraken.py**
   - Delete lines 178, 188-189
   - Run: `.venv/Scripts/ruff.exe check gal_friday/execution/kraken.py --select ERA001`
   - Commit: "Remove test code from kraken.py"

3. **feature_engine.py**
   - Delete lines 1089-1090, 1205
   - Run: `.venv/Scripts/ruff.exe check gal_friday/feature_engine.py --select ERA001`
   - Commit: "Remove old sequential processing from feature_engine.py"

4. **providers/api_provider.py**
   - Delete line 445
   - Run: `.venv/Scripts/ruff.exe check gal_friday/providers/api_provider.py --select ERA001`
   - Commit: "Remove old return statement from api_provider.py"

5. **config_manager.py**
   - Delete lines 175-182 (entire block)
   - Run: `.venv/Scripts/ruff.exe check gal_friday/config_manager.py --select ERA001`
   - Commit: "Remove old decimal conversion logic from config_manager.py"

6. **predictors/lightgbm_predictor.py**
   - Delete lines 185-187
   - Run: `.venv/Scripts/ruff.exe check gal_friday/predictors/lightgbm_predictor.py --select ERA001`
   - Commit: "Remove scaler code from lightgbm_predictor.py"

7. **predictors/neural_network_predictor.py**
   - Delete lines 218-220
   - Run: `.venv/Scripts/ruff.exe check gal_friday/predictors/neural_network_predictor.py --select ERA001`
   - Commit: "Remove scaler code from neural_network_predictor.py"

8. **predictors/sklearn_predictor.py**
   - Delete lines 177-178, 180-182
   - Run: `.venv/Scripts/ruff.exe check gal_friday/predictors/sklearn_predictor.py --select ERA001`
   - Commit: "Remove scaler code from sklearn_predictor.py"

9. **predictors/xgboost_predictor.py**
   - Delete lines 91-93, 313, 325, 327-328
   - Run: `.venv/Scripts/ruff.exe check gal_friday/predictors/xgboost_predictor.py --select ERA001`
   - Commit: "Remove scaler code from xgboost_predictor.py"

## Success Metrics
- Phase 1: ERA001 errors reduced from 118 to 70 (remove 48 errors)
- Phase 2: All 42 "needs review" items documented and triaged
- Phase 3: All 28 "must keep" items converted to proper documentation
- Final: Zero ERA001 errors or all remaining have `# noqa: ERA001` with justification

## Post-Implementation Tasks
1. Update CONTRIBUTING.md with ERA001 guidelines
2. Add pre-commit hook to check for ERA001
3. Document decision process in ADR (Architecture Decision Record)
4. Create template for future TODO comments with ticket references

## Summary Statistics

### By File:
- **execution_handler.py**: 25 errors (all MUST KEEP)
- **market_price_service.py**: 16 errors (all NEEDS REVIEW)
- **monitoring_service.py**: 13 errors (all NEEDS REVIEW)
- **config_manager.py**: 8 errors (all SAFE TO REMOVE)
- **monitoring/dashboard_backend.py**: 7 errors (all NEEDS REVIEW)
- **xgboost_predictor.py**: 7 errors (all SAFE TO REMOVE)
- **feature_engine_enhancements.py**: 6 errors (all NEEDS REVIEW)
- **portfolio_manager.py**: 5 errors (all NEEDS REVIEW)
- **sklearn_predictor.py**: 5 errors (all SAFE TO REMOVE)
- **model_lifecycle/experiment_manager.py**: 5 errors (all NEEDS REVIEW)
- **execution/kraken.py**: 3 errors (all SAFE TO REMOVE)
- **feature_engine.py**: 3 errors (all SAFE TO REMOVE)
- **lightgbm_predictor.py**: 3 errors (all SAFE TO REMOVE)
- **neural_network_predictor.py**: 3 errors (all SAFE TO REMOVE)
- **backtesting_engine.py**: 2 errors (all MUST KEEP)
- **dal/models/experiment.py**: 2 errors (all NEEDS REVIEW)
- **market_price_service_enhancements.py**: 2 errors (all NEEDS REVIEW)
- **dal/alembic_env/env.py**: 1 error (SAFE TO REMOVE)
- **dal/__init__.py**: 1 error (NEEDS REVIEW)
- **dal/connection_pool.py**: 1 error (MUST KEEP)
- **providers/api_provider.py**: 1 error (SAFE TO REMOVE)
- **strategy_arbitrator.py**: 1 error (NEEDS REVIEW)

### Total: 118 errors across 23 files