# Task: Implement unsubscribe logic for prediction handler during shutdown

### 1. Context
- **File:** `gal_friday/model_lifecycle/experiment_manager.py`
- **Line:** `230`
- **Keyword/Pattern:** `pass`
- **Current State:** Empty pass statement where unsubscribe logic should be implemented during shutdown

### 2. Problem Statement
The experiment manager lacks proper unsubscribe logic for prediction handlers during shutdown, potentially causing resource leaks, memory issues, and improper cleanup of event subscriptions. This can lead to orphaned handlers and system instability during shutdown sequences.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Subscription Registry:** Track all active prediction handler subscriptions
2. **Implement Graceful Unsubscribe:** Clean unsubscription process with proper error handling
3. **Add Shutdown Sequencing:** Ordered shutdown process for different handler types
4. **Build Resource Cleanup:** Comprehensive resource cleanup including memory and connections
5. **Create Health Monitoring:** Monitor unsubscribe process and detect failures
6. **Add Recovery Mechanisms:** Handle failed unsubscribe attempts gracefully

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Set, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time
import weakref
from abc import ABC, abstractmethod

class SubscriptionType(str, Enum):
    """Types of subscriptions"""
    PREDICTION_HANDLER = "prediction_handler"
    MODEL_UPDATE = "model_update"
    EXPERIMENT_EVENT = "experiment_event"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_HANDLER = "error_handler"

class HandlerState(str, Enum):
    """States of prediction handlers"""
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class SubscriptionInfo:
    """Information about a subscription"""
    subscription_id: str
    handler_id: str
    subscription_type: SubscriptionType
    topic: str
    callback: Callable[..., Any]
    created_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    state: HandlerState = HandlerState.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnsubscribeResult:
    """Result of unsubscribe operation"""
    subscription_id: str
    success: bool
    error_message: Optional[str] = None
    cleanup_time: float = 0.0
    resources_freed: Dict[str, Any] = field(default_factory=dict)

class PredictionHandlerProtocol(Protocol):
    """Protocol for prediction handlers"""
    
    async def stop(self) -> None:
        """Stop the handler"""
        ...
    
    async def cleanup(self) -> None:
        """Cleanup handler resources"""
        ...
    
    def get_subscription_info(self) -> Dict[str, Any]:
        """Get handler subscription information"""
        ...

class SubscriptionManager:
    """Manages all prediction handler subscriptions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Subscription tracking
        self.subscriptions: Dict[str, SubscriptionInfo] = {}
        self.handlers: Dict[str, PredictionHandlerProtocol] = {}
        self.subscription_lock = asyncio.Lock()
        
        # Shutdown tracking
        self.shutdown_in_progress = False
        self.shutdown_timeout = 30.0  # seconds
        
        # Statistics
        self.unsubscribe_stats = {
            'total_unsubscribes': 0,
            'successful_unsubscribes': 0,
            'failed_unsubscribes': 0,
            'forced_shutdowns': 0,
            'resources_freed': {}
        }
    
    async def register_subscription(self, subscription_info: SubscriptionInfo, 
                                  handler: PredictionHandlerProtocol) -> None:
        """Register a new subscription"""
        
        async with self.subscription_lock:
            self.subscriptions[subscription_info.subscription_id] = subscription_info
            self.handlers[subscription_info.handler_id] = handler
            
            self.logger.info(f"Registered subscription: {subscription_info.subscription_id}")
    
    async def unsubscribe_handler(self, subscription_id: str, force: bool = False) -> UnsubscribeResult:
        """Unsubscribe a specific prediction handler"""
        
        start_time = time.time()
        
        try:
            async with self.subscription_lock:
                if subscription_id not in self.subscriptions:
                    return UnsubscribeResult(
                        subscription_id=subscription_id,
                        success=False,
                        error_message="Subscription not found"
                    )
                
                subscription = self.subscriptions[subscription_id]
                handler = self.handlers.get(subscription.handler_id)
                
                if not handler:
                    # Clean up orphaned subscription
                    del self.subscriptions[subscription_id]
                    return UnsubscribeResult(
                        subscription_id=subscription_id,
                        success=True,
                        cleanup_time=time.time() - start_time
                    )
                
                # Mark as stopping
                subscription.state = HandlerState.STOPPING
                
                try:
                    # Graceful shutdown
                    if not force:
                        await asyncio.wait_for(
                            handler.stop(),
                            timeout=self.shutdown_timeout / 2
                        )
                    
                    # Cleanup resources
                    await asyncio.wait_for(
                        handler.cleanup(),
                        timeout=self.shutdown_timeout / 4
                    )
                    
                    # Remove from tracking
                    del self.subscriptions[subscription_id]
                    del self.handlers[subscription.handler_id]
                    
                    subscription.state = HandlerState.STOPPED
                    
                    result = UnsubscribeResult(
                        subscription_id=subscription_id,
                        success=True,
                        cleanup_time=time.time() - start_time,
                        resources_freed=handler.get_subscription_info() if hasattr(handler, 'get_subscription_info') else {}
                    )
                    
                    self.unsubscribe_stats['successful_unsubscribes'] += 1
                    self.logger.info(f"Successfully unsubscribed handler: {subscription_id}")
                    
                    return result
                    
                except asyncio.TimeoutError:
                    # Force shutdown if timeout
                    subscription.state = HandlerState.ERROR
                    self.unsubscribe_stats['forced_shutdowns'] += 1
                    
                    # Still remove from tracking to prevent leaks
                    if subscription_id in self.subscriptions:
                        del self.subscriptions[subscription_id]
                    if subscription.handler_id in self.handlers:
                        del self.handlers[subscription.handler_id]
                    
                    self.logger.warning(f"Forced shutdown of handler: {subscription_id}")
                    
                    return UnsubscribeResult(
                        subscription_id=subscription_id,
                        success=True,  # Consider forced shutdown as success
                        error_message="Forced shutdown due to timeout",
                        cleanup_time=time.time() - start_time
                    )
                
                except Exception as e:
                    subscription.state = HandlerState.ERROR
                    self.unsubscribe_stats['failed_unsubscribes'] += 1
                    
                    self.logger.error(f"Error unsubscribing handler {subscription_id}: {e}")
                    
                    return UnsubscribeResult(
                        subscription_id=subscription_id,
                        success=False,
                        error_message=str(e),
                        cleanup_time=time.time() - start_time
                    )
        
        except Exception as e:
            self.logger.error(f"Critical error during unsubscribe: {e}")
            return UnsubscribeResult(
                subscription_id=subscription_id,
                success=False,
                error_message=f"Critical error: {e}",
                cleanup_time=time.time() - start_time
            )
        
        finally:
            self.unsubscribe_stats['total_unsubscribes'] += 1

class ExperimentShutdownManager:
    """Manages shutdown process for experiment manager"""
    
    def __init__(self, subscription_manager: SubscriptionManager):
        self.subscription_manager = subscription_manager
        self.logger = logging.getLogger(__name__)
        
        # Shutdown configuration
        self.shutdown_timeout = 60.0  # Total shutdown timeout
        self.batch_size = 5  # Number of handlers to shutdown concurrently
        self.grace_period = 2.0  # Time between shutdown batches
    
    async def shutdown_all_handlers(self) -> Dict[str, UnsubscribeResult]:
        """
        Shutdown all prediction handlers during experiment manager shutdown
        Replace pass statement with comprehensive unsubscribe logic
        """
        
        try:
            self.subscription_manager.shutdown_in_progress = True
            self.logger.info("Starting shutdown of all prediction handlers")
            
            # Get all active subscriptions
            async with self.subscription_manager.subscription_lock:
                active_subscriptions = [
                    sub_id for sub_id, sub_info in self.subscription_manager.subscriptions.items()
                    if sub_info.state == HandlerState.ACTIVE
                ]
            
            if not active_subscriptions:
                self.logger.info("No active handlers to shutdown")
                return {}
            
            self.logger.info(f"Shutting down {len(active_subscriptions)} prediction handlers")
            
            # Shutdown in batches for better resource management
            results = {}
            
            for i in range(0, len(active_subscriptions), self.batch_size):
                batch = active_subscriptions[i:i + self.batch_size]
                
                self.logger.info(f"Shutting down batch {i//self.batch_size + 1}: {len(batch)} handlers")
                
                # Shutdown batch concurrently
                batch_tasks = [
                    self.subscription_manager.unsubscribe_handler(sub_id)
                    for sub_id in batch
                ]
                
                try:
                    batch_results = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=self.shutdown_timeout / len(range(0, len(active_subscriptions), self.batch_size))
                    )
                    
                    # Process batch results
                    for sub_id, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            results[sub_id] = UnsubscribeResult(
                                subscription_id=sub_id,
                                success=False,
                                error_message=str(result)
                            )
                        else:
                            results[sub_id] = result
                
                except asyncio.TimeoutError:
                    # Handle batch timeout - force shutdown remaining handlers
                    self.logger.warning(f"Batch timeout - forcing shutdown of remaining handlers")
                    
                    for sub_id in batch:
                        if sub_id not in results:
                            force_result = await self.subscription_manager.unsubscribe_handler(sub_id, force=True)
                            results[sub_id] = force_result
                
                # Grace period between batches
                if i + self.batch_size < len(active_subscriptions):
                    await asyncio.sleep(self.grace_period)
            
            # Summary logging
            successful = sum(1 for r in results.values() if r.success)
            failed = len(results) - successful
            
            self.logger.info(f"Shutdown complete: {successful} successful, {failed} failed")
            
            # Force cleanup any remaining subscriptions
            await self._cleanup_remaining_subscriptions()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error during shutdown: {e}")
            # Emergency cleanup
            await self._emergency_cleanup()
            raise
        
        finally:
            self.subscription_manager.shutdown_in_progress = False
    
    async def _cleanup_remaining_subscriptions(self) -> None:
        """Cleanup any remaining subscriptions after shutdown"""
        
        async with self.subscription_manager.subscription_lock:
            remaining = list(self.subscription_manager.subscriptions.keys())
            
            if remaining:
                self.logger.warning(f"Force cleaning {len(remaining)} remaining subscriptions")
                
                for sub_id in remaining:
                    try:
                        del self.subscription_manager.subscriptions[sub_id]
                    except KeyError:
                        pass
                
                # Clear handlers
                self.subscription_manager.handlers.clear()
    
    async def _emergency_cleanup(self) -> None:
        """Emergency cleanup in case of critical errors"""
        
        try:
            self.logger.critical("Performing emergency cleanup of all subscriptions")
            
            # Clear all tracking data
            self.subscription_manager.subscriptions.clear()
            self.subscription_manager.handlers.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            self.logger.critical(f"Emergency cleanup failed: {e}")

class PredictionHandlerUnsubscriber:
    """Enhanced unsubscribe logic for experiment manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.subscription_manager = SubscriptionManager()
        self.shutdown_manager = ExperimentShutdownManager(self.subscription_manager)
        
        # Health monitoring
        self.health_check_interval = 30.0
        self.health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize the unsubscriber"""
        self.health_check_task = asyncio.create_task(self._health_monitor())
    
    async def unsubscribe_prediction_handler(self, handler_id: str) -> bool:
        """
        Unsubscribe specific prediction handler
        Implements the logic that was missing at line 230
        """
        
        try:
            # Find subscription by handler ID
            subscription_id = None
            async with self.subscription_manager.subscription_lock:
                for sub_id, sub_info in self.subscription_manager.subscriptions.items():
                    if sub_info.handler_id == handler_id:
                        subscription_id = sub_id
                        break
            
            if not subscription_id:
                self.logger.warning(f"No subscription found for handler: {handler_id}")
                return True  # Consider as success if already cleaned
            
            # Unsubscribe the handler
            result = await self.subscription_manager.unsubscribe_handler(subscription_id)
            
            if result.success:
                self.logger.info(f"Successfully unsubscribed prediction handler: {handler_id}")
            else:
                self.logger.error(f"Failed to unsubscribe handler {handler_id}: {result.error_message}")
            
            return result.success
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing prediction handler {handler_id}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """
        Complete shutdown with comprehensive unsubscribe logic
        Replace pass statement with full implementation
        """
        
        try:
            self.logger.info("Starting experiment manager shutdown")
            
            # Stop health monitoring
            if self.health_check_task and not self.health_check_task.done():
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all handlers
            results = await self.shutdown_manager.shutdown_all_handlers()
            
            # Log final statistics
            stats = self.get_unsubscribe_statistics()
            self.logger.info(f"Shutdown statistics: {stats}")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    async def _health_monitor(self) -> None:
        """Monitor health of subscriptions"""
        
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check for stale subscriptions
                current_time = time.time()
                stale_threshold = 300.0  # 5 minutes
                
                stale_subscriptions = []
                async with self.subscription_manager.subscription_lock:
                    for sub_id, sub_info in self.subscription_manager.subscriptions.items():
                        if current_time - sub_info.last_activity > stale_threshold:
                            stale_subscriptions.append(sub_id)
                
                # Cleanup stale subscriptions
                for sub_id in stale_subscriptions:
                    self.logger.warning(f"Cleaning up stale subscription: {sub_id}")
                    await self.subscription_manager.unsubscribe_handler(sub_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitor: {e}")
    
    def get_unsubscribe_statistics(self) -> Dict[str, Any]:
        """Get comprehensive unsubscribe statistics"""
        
        return {
            **self.subscription_manager.unsubscribe_stats,
            'active_subscriptions': len(self.subscription_manager.subscriptions),
            'active_handlers': len(self.subscription_manager.handlers),
            'shutdown_in_progress': self.subscription_manager.shutdown_in_progress
        }
```

#### c. Key Considerations & Dependencies
- **Resource Management:** Proper cleanup of memory, connections, and event subscriptions
- **Error Handling:** Graceful handling of unsubscribe failures; fallback mechanisms
- **Performance:** Efficient batch processing; timeout management; concurrent operations
- **Monitoring:** Health monitoring; statistics tracking; shutdown process visibility

### 4. Acceptance Criteria
- [ ] Complete unsubscribe logic replaces pass statement at line 230
- [ ] Graceful shutdown process for all prediction handlers
- [ ] Proper resource cleanup including memory and connections
- [ ] Error handling for failed unsubscribe attempts
- [ ] Batch processing for efficient shutdown of multiple handlers
- [ ] Health monitoring for stale subscriptions
- [ ] Statistics tracking for unsubscribe operations
- [ ] Timeout management with fallback to forced shutdown
- [ ] Emergency cleanup procedures for critical errors
- [ ] Comprehensive logging and monitoring of shutdown process
- [ ] Pass statement is completely replaced with production-ready implementation 