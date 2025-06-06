# Task: Implement order state tracking and lifecycle management.

### 1. Context
- **File:** `gal_friday/execution_handler.py`
- **Line:** `551`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing order state tracking and lifecycle management functionality.

### 2. Problem Statement
Without proper order state tracking and lifecycle management, the system cannot maintain accurate records of order progression through various states (submitted, pending, filled, cancelled, rejected), leading to inconsistent portfolio tracking, incorrect risk calculations, and potential trading errors. This prevents proper reconciliation and monitoring of trading activities.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Order State Model:** Comprehensive order state definition with state transitions and validation
2. **Build State Tracking Engine:** Real-time order state tracking with event-driven updates
3. **Implement Lifecycle Management:** Complete order lifecycle from creation to final state with audit trail
4. **Add State Persistence:** Persistent storage of order states with recovery capabilities
5. **Create State Synchronization:** Synchronization between internal state and exchange state
6. **Build Monitoring and Alerts:** Real-time monitoring of order states with anomaly detection

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import asyncio
import logging

class OrderState(str, Enum):
    """Comprehensive order state enumeration"""
    CREATED = "created"
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"

@dataclass
class OrderStateEvent:
    """Order state change event"""
    order_id: str
    from_state: Optional[OrderState]
    to_state: OrderState
    timestamp: datetime
    exchange_id: Optional[str] = None
    fill_quantity: Optional[float] = None
    fill_price: Optional[float] = None
    reason: Optional[str] = None

@dataclass
class OrderLifecycleData:
    """Complete order lifecycle information"""
    order_id: str
    current_state: OrderState
    state_history: List[OrderStateEvent]
    creation_time: datetime
    last_update_time: datetime
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0

class OrderStateTracker:
    """Enterprise-grade order state tracking and lifecycle management"""
    
    def __init__(self, persistence_service, event_publisher, config: Dict[str, Any]):
        self.persistence = persistence_service
        self.event_publisher = event_publisher
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.order_states: Dict[str, OrderLifecycleData] = {}
        
    async def create_order_tracking(self, order_id: str, initial_data: Dict[str, Any]) -> OrderLifecycleData:
        """
        Create order state tracking for new order
        Replace TODO with comprehensive order lifecycle initialization
        """
        
        try:
            self.logger.debug(f"Creating order tracking for {order_id}")
            
            # Create initial lifecycle data
            lifecycle_data = OrderLifecycleData(
                order_id=order_id,
                current_state=OrderState.CREATED,
                state_history=[],
                creation_time=datetime.now(timezone.utc),
                last_update_time=datetime.now(timezone.utc),
                remaining_quantity=initial_data.get('quantity', 0.0)
            )
            
            # Create initial state event
            initial_event = OrderStateEvent(
                order_id=order_id,
                from_state=None,
                to_state=OrderState.CREATED,
                timestamp=lifecycle_data.creation_time
            )
            
            lifecycle_data.state_history.append(initial_event)
            
            # Store in memory and persistence
            self.order_states[order_id] = lifecycle_data
            await self._persist_order_state(lifecycle_data)
            
            # Publish creation event
            await self.event_publisher.publish('order.created', {
                'order_id': order_id,
                'state': OrderState.CREATED.value,
                'timestamp': initial_event.timestamp.isoformat()
            })
            
            self.logger.info(f"Order tracking created for {order_id}")
            return lifecycle_data
            
        except Exception as e:
            self.logger.error(f"Error creating order tracking for {order_id}: {e}")
            raise OrderStateError(f"Failed to create order tracking: {e}")
    
    async def update_order_state(self, order_id: str, new_state: OrderState, **kwargs) -> OrderLifecycleData:
        """Update order state with validation and event publishing"""
        
        try:
            if order_id not in self.order_states:
                # Try to load from persistence
                lifecycle_data = await self._load_order_state(order_id)
                if not lifecycle_data:
                    raise OrderStateError(f"Order {order_id} not found")
                self.order_states[order_id] = lifecycle_data
            else:
                lifecycle_data = self.order_states[order_id]
            
            current_state = lifecycle_data.current_state
            
            # Validate state transition
            if not self._is_valid_transition(current_state, new_state):
                raise OrderStateError(
                    f"Invalid state transition for {order_id}: {current_state.value} -> {new_state.value}"
                )
            
            # Create state event
            state_event = OrderStateEvent(
                order_id=order_id,
                from_state=current_state,
                to_state=new_state,
                timestamp=datetime.now(timezone.utc),
                exchange_id=kwargs.get('exchange_id'),
                fill_quantity=kwargs.get('fill_quantity'),
                fill_price=kwargs.get('fill_price'),
                reason=kwargs.get('reason')
            )
            
            # Update lifecycle data
            lifecycle_data.current_state = new_state
            lifecycle_data.last_update_time = state_event.timestamp
            lifecycle_data.state_history.append(state_event)
            
            # Update fill information if applicable
            if state_event.fill_quantity:
                lifecycle_data.filled_quantity += state_event.fill_quantity
                lifecycle_data.remaining_quantity -= state_event.fill_quantity
            
            # Persist updated state
            await self._persist_order_state(lifecycle_data)
            
            # Publish state change event
            await self.event_publisher.publish('order.state_changed', {
                'order_id': order_id,
                'from_state': current_state.value,
                'to_state': new_state.value,
                'timestamp': state_event.timestamp.isoformat()
            })
            
            self.logger.info(
                f"Order {order_id} state updated: {current_state.value} -> {new_state.value}"
            )
            
            return lifecycle_data
            
        except Exception as e:
            self.logger.error(f"Error updating order state for {order_id}: {e}")
            raise OrderStateError(f"Failed to update order state: {e}")
    
    def _is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """Validate order state transitions"""
        
        # Define valid state transitions
        valid_transitions = {
            OrderState.CREATED: {OrderState.PENDING_SUBMIT, OrderState.FAILED},
            OrderState.PENDING_SUBMIT: {OrderState.SUBMITTED, OrderState.FAILED},
            OrderState.SUBMITTED: {OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.FAILED},
            OrderState.ACKNOWLEDGED: {OrderState.PENDING, OrderState.REJECTED, OrderState.FAILED},
            OrderState.PENDING: {
                OrderState.PARTIALLY_FILLED, OrderState.FILLED, 
                OrderState.PENDING_CANCEL, OrderState.CANCELLED, 
                OrderState.REJECTED, OrderState.EXPIRED, OrderState.FAILED
            },
            OrderState.PARTIALLY_FILLED: {
                OrderState.FILLED, OrderState.PENDING_CANCEL, 
                OrderState.CANCELLED, OrderState.EXPIRED, OrderState.FAILED
            },
            OrderState.PENDING_CANCEL: {OrderState.CANCELLED, OrderState.FILLED, OrderState.FAILED},
            # Terminal states
            OrderState.FILLED: set(),
            OrderState.CANCELLED: set(),
            OrderState.REJECTED: set(),
            OrderState.EXPIRED: set(),
            OrderState.FAILED: set()
        }
        
        return to_state in valid_transitions.get(from_state, set())
    
    async def get_order_state(self, order_id: str) -> Optional[OrderLifecycleData]:
        """Get current order state and lifecycle data"""
        
        if order_id in self.order_states:
            return self.order_states[order_id]
        
        # Try to load from persistence
        lifecycle_data = await self._load_order_state(order_id)
        if lifecycle_data:
            self.order_states[order_id] = lifecycle_data
        
        return lifecycle_data
    
    async def synchronize_with_exchange(self, exchange_service) -> None:
        """Synchronize internal order states with exchange states"""
        
        try:
            self.logger.info("Starting order state synchronization with exchange")
            
            # Get all active orders
            active_orders = []
            for state in [OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, OrderState.PENDING, 
                         OrderState.PARTIALLY_FILLED, OrderState.PENDING_CANCEL]:
                orders = await self.get_orders_by_state(state)
                active_orders.extend(orders)
            
            for order_data in active_orders:
                try:
                    # Query exchange for current order status
                    exchange_status = await exchange_service.get_order_status(
                        order_data.exchange_order_id or order_data.order_id
                    )
                    
                    # Update state if necessary
                    await self._reconcile_order_state(order_data, exchange_status)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to synchronize order {order_data.order_id}: {e}")
            
            self.logger.info(f"Order synchronization complete for {len(active_orders)} orders")
            
        except Exception as e:
            self.logger.error(f"Error during order state synchronization: {e}")
            raise OrderStateError(f"Synchronization failed: {e}")

class OrderStateError(Exception):
    """Exception raised for order state tracking errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of state transition validation failures; recovery from persistence service outages; robust synchronization with exchange APIs
- **Configuration:** Configurable state transition rules; customizable anomaly detection thresholds; environment-specific validation settings
- **Testing:** Unit tests for state transition validation; integration tests with persistence and exchange services; stress tests for high-volume state updates
- **Dependencies:** Persistence service for state storage; event publisher for real-time notifications; exchange API service for synchronization

### 4. Acceptance Criteria
- [ ] Order state model supports all trading order lifecycle states with proper validation
- [ ] State transition validation prevents invalid state changes and maintains data integrity
- [ ] Real-time state tracking updates order states immediately upon receiving exchange confirmations
- [ ] Persistence layer stores complete order lifecycle history with recovery capabilities
- [ ] Synchronization mechanism ensures consistency between internal and exchange order states
- [ ] Event publishing enables real-time monitoring and integration with other system components
- [ ] Performance testing shows state updates complete under 10ms for normal operations
- [ ] Anomaly detection identifies and alerts on unusual order state patterns
- [ ] TODO placeholder is completely replaced with production-ready implementation
