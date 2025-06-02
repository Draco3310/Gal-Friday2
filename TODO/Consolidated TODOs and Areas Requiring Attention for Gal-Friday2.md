# **Consolidated TODOs and Areas Requiring Attention for Gal-Friday2**

## **Overview**

This document consolidates all identified TODOs, placeholders, "For now" comments, ERA001 tags, and other areas indicating work-in-progress or future implementation requirements from the Gal-Friday2 codebase. The information has been gathered by cross-referencing TODO_SUMMARY_1.md, TODO_SUMMARY_2.md, and TODO_SUMMARY_3.md.

## **Detailed Findings**

### **File: gal_friday/simulated_execution_handler.py** ()

**Line 439-440:** TODO: Implement min/max order size validation when these features are fully implemented (ERA001: intentional placeholder for future implementation)

# For now, we'll skip min/max position validation since it's not yet implemented  
# TODO: Implement min/max order size validation when these features  
# are fully implemented (ERA001: intentional placeholder for future implementation)  
if order.order_type == OrderType.MARKET:  
    # Always accept market orders for MVP

### **File: gal_friday/prediction_service.py**

**Line 974:** TODO: Implement confidence floor in future versions

# Normalize the weights to sum to 1.0  
# TODO: Implement confidence floor in future versions  
# For MVP, we trust all predictors equally within the weighted scheme  
total_weight = sum(active_weights.values())

**Line 1257-1258:** TODO: Consider more graceful handling of in-flight tasks

# Cancel all running prediction tasks  
# TODO: Consider more graceful handling of in-flight tasks  
# For now, active tasks for old/removed models will complete  
# naturally, but their results won't be used

### **File: gal_friday/monitoring_service.py**

**General Note:** Contains several placeholder implementations for testing/development (Lines 46-159: Multiple placeholder classes).

**Line 475:** For future implementation: Create and publish a close position command

# For future implementation: Create and publish a close position command  
# event = ClosePositionCommand(  
#     trading_pair=trading_pair,

**Line 711:** For now, placeholder implementation

# For now, placeholder implementation  
# This is a placeholder - actual implementation would:

**Line 848:** TODO: Add startup time tracking if needed

# Record when the service actually started  
# TODO: Add startup time tracking if needed  
await asyncio.sleep(0.1)  # Allow other tasks to run

**Line 951:** This is a placeholder - actual implementation would:

This is a placeholder - actual implementation would:  
1. Calculate historical volatility from price data  
2. Apply appropriate volatility models (GARCH, etc.)

### **File: gal_friday/main.py**

**Line 46:** ERA001: Removed commented-out import.

# ERA001: Removed commented-out import.

**Line 906-907:** TODO: Refactor to take session_maker if needed (applies to _init_strategy_arbitrator and _init_cli_service)

self._init_cli_service()  
self._init_strategy_arbitrator() # TODO: Refactor to take session_maker if needed  
self._init_cli_service()         # TODO: Refactor to take session_maker if needed

### **File: gal_friday/kraken_historical_data_service.py**

**Line 339:** TODO: Implement fetching trade data from Kraken API

# TODO: Implement fetching trade data from Kraken API  
self.logger.warning(  
    "Trade data fetching not yet implemented",

**Line 504-505:** TODO: Implement actual API call using aiohttp or ccxt

# TODO: Implement actual API call using aiohttp or ccxt  
# This is a placeholder for the actual implementation  
return None

**Line 818:** TODO: Check for gaps within the data range

# TODO: Check for gaps within the data range  
# For now, assuming data is complete if we have expected number of points

### **File: gal_friday/execution_handler.py**

**Line 232-234:** TODO: Add state for managing WebSocket connection if used for MVP

# TODO: Add state for managing WebSocket connection if used for MVP  
# TODO: Add mapping for internal IDs to exchange IDs (cl_ord_id ->  
# exchange_order_id)

**Line 232-234:** TODO: Add mapping for internal IDs to exchange IDs (cl_ord_id -> exchange_order_id)

# TODO: Add state for managing WebSocket connection if used for MVP  
# TODO: Add mapping for internal IDs to exchange IDs (cl_ord_id ->  
# exchange_order_id)

**Line 251 (Source 1) / Lines 255-258 (Source 2, 3):** TODO: Implement Kraken Adapter Pattern

# TODO: Implement Kraken Adapter Pattern  
# For MVP, using direct REST API calls  
# Future: Create abstract adapter interface and Kraken-specific implementation

**Line 301:** TODO: Implement WebSocket connection logic here if used for MVP

# TODO: Implement WebSocket connection logic here if used for MVP  
self._started = True

**Line 359-360:** TODO: Implement WebSocket disconnection logic

# TODO: Implement WebSocket disconnection logic  
# TODO: Implement configurable cancellation of open orders on stop  
self._started = False

**Line 359-360:** TODO: Implement configurable cancellation of open orders on stop

# TODO: Implement WebSocket disconnection logic  
# TODO: Implement configurable cancellation of open orders on stop  
self._started = False

**Line 919 (Source 1) / Line 923 (Source 2, 3):** TODO: Consider using AddOrderBatch if placing SL/TP simultaneously

# TODO: Consider using AddOrderBatch if placing SL/TP simultaneously  
return order_response

**Line 1205:** Assuming single order response for now

# Assuming single order response for now

**Line 1285:** This is a placeholder for future WebSocket implementation.

This is a placeholder for future WebSocket implementation.  
# Placeholder

### **File: gal_friday/dal/alembic_env/env.py**

**Line 67:** This is a bit of a hack; ideally ConfigManager is usable without full app setup

# This is a bit of a hack; ideally ConfigManager is usable without full app setup  
class MinimalLogger:  
    def get_logger(self, name): return self

### **File: gal_friday/feature_engine.py**

**Line 235:** For now, we'll just return and not update if format is wrong.

# For now, we'll just return and not update if format is wrong.  
return

**Line 889:** Placeholder comment

# Placeholder

**Line 1134:** Placeholder Feature Processing Methods / TODO: Implement cloud storage for feature cache (optional)

# --- Placeholder Feature Processing Methods ---

*(Note: Source 1 specifies "Implement cloud storage for feature cache (optional)" for this line)*

### **File: gal_friday/risk_manager.py**

**General Note:** Contains multiple placeholder implementations and mock objects for runtime fallbacks (Lines 63-132: Various placeholder implementations for PortfolioManager, MarketPriceService, and ExchangeInfoService).

**Line 1355:** For now, we store the daily volatility.

# For now, we store the daily volatility.

### **File: gal_friday/logger_service.py**

**Line 217:** For now, assuming Log model maps to 'logs' table.

# For now, assuming Log model maps to 'logs' table.

**Line 352:** For now, treating as non-retryable by this handler's direct logic.

# For now, treating as non-retryable by this handler's direct logic.

### **File: gal_friday/backtesting_engine.py**

**Line 279:** TA-Lib not installed warning (indicates optional dependency). ATR calculation will not work; uses a default fallback value.

log.warning("TA-Lib not installed. ATR calculation will not work.")  
atr_value = Decimal("20.0")  # Default fallback value

### **File: gal_friday/database.py**

**Lines 3-8:** Placeholder function for getting the database connection string. This should be replaced with actual configuration loading.

# Placeholder function for getting the database connection string  
def get_database_url() -> str:  
    """Return the database connection string."""  
    # For now, using a placeholder value.  
    # This should be replaced with actual configuration loading.  
    return "postgresql+asyncpg://user:password@host/dbname_placeholder"

### **File: gal_friday/data_ingestor.py**

**Line 721:** Reusing book data handler for now

await self._handle_book_data(data)  # Reuse book data handler for now

**Line 1400:** Hardcoded exchange for now

exchange="kraken",  # Hardcoded for now

### **File: gal_friday/portfolio_manager.py**

**Line 333:** Order cancellation still handled locally for now

self._handle_order_cancellation(event)  # Still handled locally for now

**Line 1145:** Using absolute threshold for larger balances? For now, use absolute.

# Use a relative threshold for larger balances? For now, use absolute.

### **File: gal_friday/strategy_arbitrator.py**

**Line 86:** For now, continue with MVP using the first strategy.

# For now, continue with MVP using the first strategy.

**Line 475:** Removed commented-out lines for ERA001

# Removed commented-out lines for ERA001

### **File: gal_friday/utils/performance_optimizer.py**

**Line 213:** For now, assume all connections are healthy

# For now, assume all connections are healthy

**Line 252:** For now, return mock analysis

# For now, return mock analysis

### **File: gal_friday/simulated_market_price_service.py**

**Line 110:** For now, we assume direct assignment

# For now, we assume direct assignment or

**Line 1437:** For now, this test relies on the default 0.1% spread which likely won't be zero.

# For now, this test relies on the default 0.1% spread which likely won't be zero.

### **File: gal_friday/portfolio/position_manager.py**

**Line 32:** For now, keeping if PositionManager logic uses them directly.

# For now, keeping if PositionManager logic uses them directly.

**Line 101:** For now, let's assume it just logs the number of active positions.

# For now, let's assume it just logs the number of active positions.

### **File: gal_friday/feature_engine.py**

**Code Quality Improvements Needed:**
- Break down large methods into smaller, focused functions for better maintainability
- Remove all commented-out code before merging to main
- Ensure consistent error handling patterns across all feature calculations
- Add comprehensive docstrings for all public methods and classes
- Consider adding performance monitoring for feature calculation pipelines

**Future Enhancements:**
- Add support for feature importance analysis
- Implement feature selection capabilities
- Add support for custom feature transformations
- Consider adding support for feature store integration
- Add more comprehensive input validation
- Implement feature drift detection
- Add support for feature versioning

**Testing Recommendations:**
- Add more edge case tests for feature calculations
- Add performance benchmarks for feature pipelines
- Test with larger datasets to identify potential bottlenecks
- Add integration tests with real market data
- Test error handling for malformed inputs

### **File: gal_friday/core/feature_registry_client.py**

**Future Improvements:**
- Add support for feature registry versioning
- Implement caching for frequently accessed feature definitions
- Add validation for feature registry schema
- Consider adding support for remote feature registries
- Add metrics for registry access patterns

### **File: tests/unit/test_feature_engine_pipeline_construction.py**

**Testing Recommendations:**
- Add tests for error conditions and edge cases
- Test with different combinations of imputation and scaling strategies
- Add performance tests for pipeline execution
- Test with various input data types and shapes
- Add property-based tests for data transformations

### **File: gal_friday/model_lifecycle/retraining_pipeline.py**

**Line 284:** Simple difference for now - could use KL divergence

# Simple difference for now - could use KL divergence

### **File: gal_friday/monitoring/dashboard_backend.py**

**Line 162:** Placeholder value for "uptime_pct"

"uptime_pct": 99.9,  # Placeholder

**Line 209:** Placeholder for correlation calculation for "correlation_risk"

"correlation_risk": 0,  # Placeholder for correlation calculation

### **File: gal_friday/monitoring/dashboard_service.py**

**Line 57:** Placeholder metrics - in real implementation would connect to portfolio manager

# Placeholder metrics - in real implementation would connect to portfolio manager

### **File: gal_friday/core/events.py**

**Lines 1237-1238:** Cleanup placeholder comment / Removed TODO section

# Cleanup placeholder comment  
# Removed the TODO section as definitions are now added.

### **File: gal_friday/core/placeholder_classes.py**

Lines 56, 160, 164: Placeholder type hints  
(Note: Source 2 referenced these under gal_friday/logger_service.py but Source 3 clarifies the file)

### **File: gal_friday/predictors/xgboost_predictor.py**

**Line 398:** Placeholder comment about configuration source (enhancement)

### **File: gal_friday/model_lifecycle/registry.py**

**Line 390:** Add model versioning metadata (enhancement)

## **Consolidated Summary of Key Areas Requiring Attention**

Based on the detailed findings, the most significant themes for areas requiring attention include:

1. **Kraken API Integration & WebSocket Management:**  
   * Implementing fetching of trade data from Kraken API.  
   * Full implementation of WebSocket connection/disconnection logic.  
   * Adding state for WebSocket management and mapping internal to exchange IDs.  
   * Consideration of Kraken Adapter Pattern for abstraction.  
2. **Order and Position Management:**  
   * Implementing min/max order size validation.  
   * Consideration of AddOrderBatch for simultaneous SL/TP placement.  
   * Publishing close position commands.  
3. **Data Handling and Validation:**  
   * Checking for gaps in historical data ranges.  
   * More graceful handling of in-flight tasks in the prediction service.  
4. **Configuration and Initialization:**  
   * Refactoring main.py components to take session_maker.  
   * Replacing placeholder database connection strings with actual configuration.  
   * Addressing the "hack" related to ConfigManager usability in alembic_env/env.py.  
5. **Feature Enhancements