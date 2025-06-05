# Task: Replace temporary debug log with real metrics computation loop and error handling.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1827`
- **Keyword/Pattern:** `"TODO" message`
- **Current State:** The code contains a temporary debug log message instead of actual metrics computation implementation.

### 2. Problem Statement
The temporary debug log message indicates incomplete implementation of the core metrics computation functionality. This creates a significant gap in the risk management system's ability to calculate and monitor critical risk metrics in real-time. Without proper metrics computation, the system cannot provide accurate risk assessment or trigger appropriate risk controls when needed.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Remove Debug Placeholder:** Replace temporary log message with production metrics computation logic
2. **Implement Computation Loop:** Create robust metrics calculation with proper error handling and recovery
3. **Add Performance Monitoring:** Track computation performance and resource usage
4. **Build Error Recovery:** Implement fallback mechanisms for failed calculations
5. **Create Metrics Validation:** Add consistency checks and data quality validation
6. **Add Logging Strategy:** Replace debug logs with structured, actionable logging

#### b. Pseudocode or Implementation Sketch
```python
import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import traceback

class MetricsComputationEngine:
    """Production-grade metrics computation with error handling"""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.logger = risk_manager.logger
        self.computation_errors = 0
        self.last_successful_computation = None
        self.metrics_cache = {}
        self.computation_stats = {
            'total_computations': 0,
            'successful_computations': 0,
            'failed_computations': 0,
            'average_computation_time': 0.0
        }
    
    async def run_metrics_computation_loop(self) -> None:
        """
        Replace: self.logger.debug("TODO: Calculate periodic risk metrics")
        with production metrics computation loop
        """
        
        computation_start = time.time()
        computation_id = f"comp_{int(time.time())}"
        
        try:
            self.logger.info(f"Starting risk metrics computation {computation_id}")
            
            # Validate prerequisites
            await self._validate_computation_prerequisites()
            
            # Run core metrics calculations
            metrics_results = await self._execute_metrics_calculations()
            
            # Validate computation results
            await self._validate_computation_results(metrics_results)
            
            # Update internal state
            await self._update_metrics_state(metrics_results)
            
            # Record successful computation
            computation_time = time.time() - computation_start
            await self._record_successful_computation(computation_id, computation_time, metrics_results)
            
            self.logger.info(
                f"Completed risk metrics computation {computation_id} in {computation_time:.3f}s: "
                f"drawdown={metrics_results.get('current_drawdown', 0):.4f}, "
                f"var={metrics_results.get('var_95', 0):.4f}, "
                f"positions={metrics_results.get('position_count', 0)}"
            )
            
        except PrerequisiteValidationError as e:
            await self._handle_prerequisite_error(computation_id, e)
        except MetricsCalculationError as e:
            await self._handle_calculation_error(computation_id, e)
        except ResultValidationError as e:
            await self._handle_validation_error(computation_id, e)
        except Exception as e:
            await self._handle_unexpected_error(computation_id, e)
    
    async def _validate_computation_prerequisites(self) -> None:
        """Validate that all required data and services are available"""
        
        # Check portfolio manager availability
        if not self.risk_manager.portfolio_manager:
            raise PrerequisiteValidationError("PortfolioManager not available")
        
        # Check market data service
        if not self.risk_manager.market_data_service:
            raise PrerequisiteValidationError("MarketDataService not available")
        
        # Check minimum data requirements
        positions = await self.risk_manager.portfolio_manager.get_all_positions()
        if positions is None:
            raise PrerequisiteValidationError("Unable to retrieve position data")
        
        # Check account info availability
        account_info = await self.risk_manager.portfolio_manager.get_account_info()
        if not account_info or 'total_equity' not in account_info:
            raise PrerequisiteValidationError("Account information unavailable or incomplete")
    
    async def _execute_metrics_calculations(self) -> Dict[str, Any]:
        """Execute all risk metrics calculations with proper error handling"""
        
        metrics_results = {}
        calculation_errors = []
        
        # Calculate portfolio metrics
        try:
            portfolio_metrics = await self._calculate_portfolio_metrics()
            metrics_results.update(portfolio_metrics)
        except Exception as e:
            calculation_errors.append(f"Portfolio metrics: {e}")
            self.logger.warning(f"Portfolio metrics calculation failed: {e}")
        
        # Calculate risk metrics
        try:
            risk_metrics = await self._calculate_risk_metrics()
            metrics_results.update(risk_metrics)
        except Exception as e:
            calculation_errors.append(f"Risk metrics: {e}")
            self.logger.warning(f"Risk metrics calculation failed: {e}")
        
        # Calculate performance metrics
        try:
            performance_metrics = await self._calculate_performance_metrics()
            metrics_results.update(performance_metrics)
        except Exception as e:
            calculation_errors.append(f"Performance metrics: {e}")
            self.logger.warning(f"Performance metrics calculation failed: {e}")
        
        # Calculate exposure metrics
        try:
            exposure_metrics = await self._calculate_exposure_metrics()
            metrics_results.update(exposure_metrics)
        except Exception as e:
            calculation_errors.append(f"Exposure metrics: {e}")
            self.logger.warning(f"Exposure metrics calculation failed: {e}")
        
        # Check if we have minimum required metrics
        required_metrics = ['current_drawdown', 'total_equity', 'position_count']
        missing_metrics = [m for m in required_metrics if m not in metrics_results]
        
        if missing_metrics:
            raise MetricsCalculationError(
                f"Critical metrics calculation failed: {missing_metrics}. "
                f"Errors: {calculation_errors}"
            )
        
        if calculation_errors:
            self.logger.warning(
                f"Some metrics calculations failed but core metrics available: {calculation_errors}"
            )
        
        return metrics_results
    
    async def _validate_computation_results(self, metrics_results: Dict[str, Any]) -> None:
        """Validate the computed metrics for consistency and reasonableness"""
        
        validation_errors = []
        
        # Validate drawdown metrics
        current_drawdown = metrics_results.get('current_drawdown', 0)
        if current_drawdown < 0 or current_drawdown > 1:
            validation_errors.append(f"Invalid drawdown value: {current_drawdown}")
        
        # Validate equity values
        total_equity = metrics_results.get('total_equity', 0)
        if total_equity < 0:
            validation_errors.append(f"Negative total equity: {total_equity}")
        
        # Validate VaR
        var_95 = metrics_results.get('var_95')
        if var_95 is not None and (var_95 < 0 or var_95 > 1):
            validation_errors.append(f"Invalid VaR value: {var_95}")
        
        # Check for NaN or infinite values
        for key, value in metrics_results.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    validation_errors.append(f"Non-finite value for {key}: {value}")
        
        if validation_errors:
            raise ResultValidationError(f"Metrics validation failed: {validation_errors}")
    
    async def _handle_calculation_error(self, computation_id: str, error: Exception) -> None:
        """Handle metrics calculation errors with fallback logic"""
        
        self.computation_errors += 1
        self.computation_stats['failed_computations'] += 1
        
        # Log detailed error information
        self.logger.error(
            f"Metrics computation {computation_id} failed: {error}",
            extra={
                'computation_id': computation_id,
                'error_type': type(error).__name__,
                'consecutive_errors': self.computation_errors,
                'stack_trace': traceback.format_exc()
            }
        )
        
        # Try fallback computation using cached data
        if self.metrics_cache and self.computation_errors < 3:
            self.logger.info(f"Attempting fallback computation using cached data")
            try:
                fallback_metrics = await self._compute_fallback_metrics()
                await self._update_metrics_state(fallback_metrics)
                self.logger.info(f"Fallback computation successful for {computation_id}")
                return
            except Exception as fallback_error:
                self.logger.error(f"Fallback computation also failed: {fallback_error}")
        
        # Publish computation failure event
        await self._publish_computation_failure_event(computation_id, error)
        
        # Check if we need to halt risk monitoring
        if self.computation_errors >= 5:
            await self._handle_critical_computation_failure()
    
    async def _record_successful_computation(self, computation_id: str, 
                                           computation_time: float, 
                                           metrics_results: Dict[str, Any]) -> None:
        """Record successful computation for monitoring and statistics"""
        
        self.computation_errors = 0  # Reset error counter
        self.last_successful_computation = datetime.now(timezone.utc)
        self.metrics_cache = metrics_results.copy()
        
        # Update statistics
        self.computation_stats['total_computations'] += 1
        self.computation_stats['successful_computations'] += 1
        
        # Update rolling average computation time
        current_avg = self.computation_stats['average_computation_time']
        total_successful = self.computation_stats['successful_computations']
        self.computation_stats['average_computation_time'] = (
            (current_avg * (total_successful - 1) + computation_time) / total_successful
        )
        
        # Publish performance metrics
        await self._publish_computation_performance_metrics(computation_id, computation_time)
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Comprehensive error handling with graceful degradation; fallback mechanisms using cached data; circuit breaker pattern for repeated failures
- **Configuration:** Configurable error thresholds, computation timeouts, and fallback strategies; logging levels for different environments
- **Testing:** Unit tests for each computation component; integration tests for error scenarios; stress tests for high-load conditions; performance benchmarks
- **Dependencies:** Integration with portfolio and market data services; pub/sub system for error notifications; monitoring service for performance tracking

### 4. Acceptance Criteria
- [ ] Temporary debug log message is completely removed and replaced with production computation logic
- [ ] All risk metrics calculations are implemented with proper error handling and validation
- [ ] Computation failures are handled gracefully with fallback mechanisms and appropriate logging
- [ ] Performance metrics track computation time, success rate, and resource usage
- [ ] Error recovery mechanisms prevent single failures from stopping risk monitoring
- [ ] Structured logging provides actionable information for debugging and monitoring
- [ ] Computation results are validated for consistency and mathematical correctness
- [ ] Integration tests verify end-to-end computation flow under normal and error conditions
- [ ] Performance benchmarks show acceptable computation time (<5 seconds for typical portfolios) 