# Task: Allow configurable cancellation of open orders on shutdown with safety checks.

### 1. Context
- **File:** `gal_friday/execution_handler.py`
- **Line:** `550`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO for implementing configurable order cancellation during system shutdown.

### 2. Problem Statement
Without configurable order cancellation on shutdown, the system may leave open orders in the market when shutting down, creating potential exposure and risk. However, automatically cancelling all orders could also be problematic in certain situations. The system needs intelligent, configurable logic to safely handle open orders during shutdown based on order type, market conditions, and risk parameters.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Shutdown Configuration Framework:** Implement configurable shutdown behavior per order type and strategy
2. **Build Safety Check System:** Add comprehensive safety checks before order cancellation
3. **Implement Graceful Cancellation Logic:** Handle different order types with appropriate cancellation strategies
4. **Add Risk Assessment:** Evaluate market conditions and exposure before cancellation decisions
5. **Create Audit Trail:** Log all shutdown decisions for compliance and analysis
6. **Build Emergency Procedures:** Handle scenarios where cancellation fails or is unsafe

#### b. Pseudocode or Implementation Sketch
```python
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import logging

class ShutdownOrderAction(str, Enum):
    CANCEL_ALL = "cancel_all"
    CANCEL_CONDITIONAL = "cancel_conditional"
    LEAVE_OPEN = "leave_open"
    CONVERT_TO_MARKET = "convert_to_market"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"

@dataclass
class ShutdownConfig:
    """Configuration for shutdown order handling"""
    default_action: ShutdownOrderAction
    action_by_order_type: Dict[OrderType, ShutdownOrderAction]
    action_by_strategy: Dict[str, ShutdownOrderAction]
    safety_checks_enabled: bool = True
    max_cancellation_time: int = 30  # seconds
    require_confirmation: bool = False
    preserve_stop_losses: bool = True
    market_hours_only: bool = True

@dataclass
class OrderCancellationResult:
    """Result of order cancellation attempt"""
    order_id: str
    symbol: str
    action_taken: ShutdownOrderAction
    success: bool
    reason: str
    timestamp: datetime
    market_impact_estimate: float

class ConfigurableShutdownHandler:
    """Handles configurable order cancellation during shutdown"""
    
    def __init__(self, config: ShutdownConfig, execution_handler, market_data_service):
        self.config = config
        self.execution_handler = execution_handler
        self.market_data = market_data_service
        self.logger = logging.getLogger(__name__)
        self.shutdown_in_progress = False
        
    async def handle_shutdown_orders(self) -> List[OrderCancellationResult]:
        """
        Handle open orders during shutdown with configurable logic
        Replace TODO with comprehensive shutdown handling
        """
        
        if self.shutdown_in_progress:
            self.logger.warning("Shutdown already in progress, skipping duplicate call")
            return []
        
        self.shutdown_in_progress = True
        shutdown_start = datetime.now(timezone.utc)
        
        try:
            self.logger.info("Starting configurable order shutdown process")
            
            # Get all open orders
            open_orders = await self.execution_handler.get_open_orders()
            
            if not open_orders:
                self.logger.info("No open orders to handle during shutdown")
                return []
            
            self.logger.info(f"Found {len(open_orders)} open orders to evaluate for shutdown")
            
            # Perform safety checks
            safety_check_passed = await self._perform_safety_checks(open_orders)
            
            if not safety_check_passed and self.config.safety_checks_enabled:
                self.logger.error("Safety checks failed, aborting order cancellation")
                return self._create_safety_failure_results(open_orders)
            
            # Process each order according to configuration
            cancellation_results = []
            
            for order in open_orders:
                try:
                    result = await self._process_order_for_shutdown(order)
                    cancellation_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing order {order.get('order_id')}: {e}")
                    cancellation_results.append(
                        self._create_error_result(order, str(e))
                    )
            
            # Log shutdown summary
            await self._log_shutdown_summary(cancellation_results, shutdown_start)
            
            # Wait for cancellations to complete
            await self._wait_for_cancellation_completion(cancellation_results)
            
            return cancellation_results
            
        except Exception as e:
            self.logger.error(f"Critical error in shutdown order handling: {e}")
            return []
        
        finally:
            self.shutdown_in_progress = False
    
    async def _perform_safety_checks(self, open_orders: List[Dict[str, Any]]) -> bool:
        """Perform comprehensive safety checks before order cancellation"""
        
        try:
            # Check market hours
            if self.config.market_hours_only:
                is_market_open = await self._is_market_open()
                if not is_market_open:
                    self.logger.warning("Market is closed, cancellation may be risky")
                    return False
            
            # Check for critical stop losses
            critical_stop_losses = [
                order for order in open_orders 
                if order.get('type') == OrderType.STOP_LOSS.value
                and self._is_critical_stop_loss(order)
            ]
            
            if critical_stop_losses and self.config.preserve_stop_losses:
                self.logger.warning(f"Found {len(critical_stop_losses)} critical stop losses")
                return False
            
            # Check market volatility
            high_volatility_symbols = await self._check_market_volatility(open_orders)
            if high_volatility_symbols:
                self.logger.warning(f"High volatility detected in: {high_volatility_symbols}")
                # Don't fail, but log for awareness
            
            # Check large position sizes
            large_orders = [
                order for order in open_orders
                if abs(order.get('quantity', 0)) > self._get_large_order_threshold(order.get('symbol'))
            ]
            
            if large_orders:
                self.logger.warning(f"Found {len(large_orders)} large orders requiring careful handling")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in safety checks: {e}")
            return False
    
    async def _process_order_for_shutdown(self, order: Dict[str, Any]) -> OrderCancellationResult:
        """Process individual order according to shutdown configuration"""
        
        order_id = order.get('order_id', 'unknown')
        symbol = order.get('symbol', 'unknown')
        order_type = OrderType(order.get('type', 'limit'))
        strategy_id = order.get('strategy_id')
        
        # Determine action based on configuration hierarchy
        action = self._determine_shutdown_action(order_type, strategy_id)
        
        self.logger.info(f"Processing order {order_id} ({symbol}) with action: {action.value}")
        
        try:
            if action == ShutdownOrderAction.CANCEL_ALL:
                success = await self._cancel_order(order)
                reason = "Cancelled as per shutdown configuration"
                
            elif action == ShutdownOrderAction.CANCEL_CONDITIONAL:
                success = await self._conditional_cancel_order(order)
                reason = "Conditionally cancelled based on market conditions"
                
            elif action == ShutdownOrderAction.LEAVE_OPEN:
                success = True
                reason = "Left open as per configuration"
                
            elif action == ShutdownOrderAction.CONVERT_TO_MARKET:
                success = await self._convert_to_market_order(order)
                reason = "Converted to market order for immediate execution"
                
            else:
                success = False
                reason = f"Unknown action: {action}"
            
            # Estimate market impact
            market_impact = await self._estimate_market_impact(order, action)
            
            return OrderCancellationResult(
                order_id=order_id,
                symbol=symbol,
                action_taken=action,
                success=success,
                reason=reason,
                timestamp=datetime.now(timezone.utc),
                market_impact_estimate=market_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error processing order {order_id}: {e}")
            return OrderCancellationResult(
                order_id=order_id,
                symbol=symbol,
                action_taken=action,
                success=False,
                reason=f"Error: {e}",
                timestamp=datetime.now(timezone.utc),
                market_impact_estimate=0.0
            )
    
    def _determine_shutdown_action(self, order_type: OrderType, strategy_id: Optional[str]) -> ShutdownOrderAction:
        """Determine shutdown action based on configuration hierarchy"""
        
        # Strategy-specific configuration takes highest priority
        if strategy_id and strategy_id in self.config.action_by_strategy:
            return self.config.action_by_strategy[strategy_id]
        
        # Order type-specific configuration
        if order_type in self.config.action_by_order_type:
            return self.config.action_by_order_type[order_type]
        
        # Default action
        return self.config.default_action
    
    async def _cancel_order(self, order: Dict[str, Any]) -> bool:
        """Cancel a specific order"""
        
        try:
            order_id = order.get('order_id')
            symbol = order.get('symbol')
            
            success = await self.execution_handler.cancel_order(order_id, symbol)
            
            if success:
                self.logger.info(f"Successfully cancelled order {order_id}")
            else:
                self.logger.warning(f"Failed to cancel order {order_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order.get('order_id')}: {e}")
            return False
    
    async def _conditional_cancel_order(self, order: Dict[str, Any]) -> bool:
        """Conditionally cancel order based on market conditions"""
        
        symbol = order.get('symbol')
        order_type = order.get('type')
        
        # Don't cancel stop losses in volatile markets
        if order_type == OrderType.STOP_LOSS.value:
            volatility = await self._get_symbol_volatility(symbol)
            if volatility > 0.05:  # High volatility threshold
                self.logger.info(f"Preserving stop loss for {symbol} due to high volatility")
                return True  # "Success" by not cancelling
        
        # Don't cancel profitable limit orders near execution
        if order_type == OrderType.LIMIT.value:
            near_execution = await self._is_order_near_execution(order)
            if near_execution:
                self.logger.info(f"Preserving limit order {order.get('order_id')} near execution")
                return True
        
        # Default to cancellation
        return await self._cancel_order(order)
    
    async def _wait_for_cancellation_completion(self, results: List[OrderCancellationResult]) -> None:
        """Wait for cancellation operations to complete"""
        
        cancellation_attempts = [r for r in results if r.action_taken == ShutdownOrderAction.CANCEL_ALL]
        
        if not cancellation_attempts:
            return
        
        max_wait_time = self.config.max_cancellation_time
        start_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Waiting up to {max_wait_time}s for {len(cancellation_attempts)} cancellations to complete")
        
        while (datetime.now(timezone.utc) - start_time).total_seconds() < max_wait_time:
            # Check if any orders are still open
            remaining_orders = await self._get_remaining_open_orders(cancellation_attempts)
            
            if not remaining_orders:
                self.logger.info("All cancellations completed successfully")
                break
            
            self.logger.debug(f"{len(remaining_orders)} orders still pending cancellation")
            await asyncio.sleep(1)
        
        else:
            remaining_orders = await self._get_remaining_open_orders(cancellation_attempts)
            if remaining_orders:
                self.logger.warning(f"Timeout: {len(remaining_orders)} orders still open after {max_wait_time}s")
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of cancellation failures; fallback procedures for stuck orders; timeout handling for slow exchanges
- **Configuration:** Hierarchical configuration system (strategy > order type > default); runtime configuration updates; environment-specific settings
- **Testing:** Unit tests for configuration logic; integration tests with exchange APIs; stress tests for high-volume cancellations
- **Dependencies:** Integration with execution handler; market data service for safety checks; configuration management for dynamic behavior

### 4. Acceptance Criteria
- [ ] Configurable shutdown behavior supports different actions per order type and strategy
- [ ] Safety checks prevent unsafe cancellations during volatile market conditions
- [ ] Critical stop losses are preserved when configured to do so
- [ ] Cancellation timeout mechanism prevents indefinite blocking during shutdown
- [ ] Comprehensive logging provides audit trail of all shutdown decisions
- [ ] Error handling ensures graceful degradation when cancellations fail
- [ ] Performance testing shows shutdown completion within configured time limits
- [ ] Integration tests verify behavior with real exchange APIs
- [ ] TODO placeholder is completely replaced with production-ready implementation 