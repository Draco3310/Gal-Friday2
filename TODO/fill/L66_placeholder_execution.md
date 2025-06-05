# Task: Replace placeholder execution event logic with actual event construction

### 1. Context
- **File:** `gal_friday/models/fill.py`
- **Line:** `66`
- **Keyword/Pattern:** `"Placeholder"`
- **Current State:** Placeholder execution event logic that needs actual ExecutionReportEvent implementation

### 2. Problem Statement
The fill model contains placeholder execution event logic that prevents proper event construction and publishing. This limits the system's ability to track and report trade executions, impacting audit trails and real-time monitoring.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Define ExecutionReportEvent Schema:** Complete event structure for execution reporting
2. **Implement Event Construction:** Build actual event creation from fill data
3. **Add Event Validation:** Ensure event data integrity and completeness
4. **Create Event Publishing:** Integrate with event bus for real-time notifications
5. **Build Event Persistence:** Store execution events for audit and analysis
6. **Add Event Monitoring:** Track event creation and delivery

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
import time
import uuid

class ExecutionType(str, Enum):
    """Types of execution events"""
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionSide(str, Enum):
    """Execution sides"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class ExecutionReportEvent:
    """Complete execution report event structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # Order information
    order_id: str = ""
    client_order_id: str = ""
    exchange_order_id: str = ""
    
    # Execution details
    execution_type: ExecutionType = ExecutionType.FILL
    execution_id: str = ""
    side: ExecutionSide = ExecutionSide.BUY
    symbol: str = ""
    
    # Quantities and prices
    quantity: Decimal = Decimal('0')
    price: Decimal = Decimal('0')
    cumulative_quantity: Decimal = Decimal('0')
    remaining_quantity: Decimal = Decimal('0')
    
    # Financial details
    commission: Decimal = Decimal('0')
    commission_currency: str = "USD"
    net_proceeds: Decimal = Decimal('0')
    
    # Market data
    last_price: Optional[Decimal] = None
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    
    # Metadata
    exchange: str = ""
    venue: str = ""
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # Additional information
    text: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

class ExecutionEventBuilder:
    """Builder for creating execution report events"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_execution_event(self, fill_data: 'Fill') -> ExecutionReportEvent:
        """
        Create actual ExecutionReportEvent from fill data
        Replace placeholder logic with complete event construction
        """
        
        try:
            # Extract core execution information
            event = ExecutionReportEvent(
                order_id=fill_data.order_id,
                client_order_id=getattr(fill_data, 'client_order_id', ''),
                exchange_order_id=self._get_exchange_order_id(fill_data),
                execution_type=self._determine_execution_type(fill_data),
                execution_id=fill_data.fill_id,
                side=ExecutionSide(fill_data.side.lower()) if hasattr(fill_data, 'side') else ExecutionSide.BUY,
                symbol=fill_data.symbol
            )
            
            # Set quantities and prices
            event.quantity = fill_data.quantity
            event.price = fill_data.price
            event.cumulative_quantity = self._calculate_cumulative_quantity(fill_data)
            event.remaining_quantity = self._calculate_remaining_quantity(fill_data)
            
            # Calculate financial details
            event.commission = self._calculate_commission(fill_data)
            event.commission_currency = self._get_commission_currency(fill_data)
            event.net_proceeds = self._calculate_net_proceeds(fill_data)
            
            # Add market data if available
            event.last_price = getattr(fill_data, 'last_price', None)
            event.bid_price = getattr(fill_data, 'bid_price', None)
            event.ask_price = getattr(fill_data, 'ask_price', None)
            
            # Set exchange and venue information
            event.exchange = getattr(fill_data, 'exchange', 'unknown')
            event.venue = getattr(fill_data, 'venue', event.exchange)
            
            # Add strategy and signal information
            event.strategy_id = getattr(fill_data, 'strategy_id', None)
            event.signal_id = self._extract_signal_id(fill_data)
            
            # Add metadata and tags
            event.text = f"Fill executed: {event.quantity} @ {event.price}"
            event.tags = self._create_event_tags(fill_data)
            
            # Validate event before returning
            if self._validate_execution_event(event):
                self.logger.info(f"Created execution event: {event.event_id}")
                return event
            else:
                raise ValueError("Invalid execution event data")
                
        except Exception as e:
            self.logger.error(f"Error creating execution event: {e}")
            raise
    
    def _get_exchange_order_id(self, fill_data: 'Fill') -> str:
        """Extract exchange order ID from fill data"""
        
        # Check various possible sources for exchange order ID
        if hasattr(fill_data, 'exchange_order_id') and fill_data.exchange_order_id:
            return str(fill_data.exchange_order_id)
        
        if hasattr(fill_data, 'external_order_id') and fill_data.external_order_id:
            return str(fill_data.external_order_id)
        
        # Fallback to order ID if no exchange ID available
        return fill_data.order_id
    
    def _determine_execution_type(self, fill_data: 'Fill') -> ExecutionType:
        """Determine the type of execution from fill data"""
        
        if hasattr(fill_data, 'execution_type'):
            return ExecutionType(fill_data.execution_type)
        
        # Determine based on fill characteristics
        if hasattr(fill_data, 'is_partial') and fill_data.is_partial:
            return ExecutionType.PARTIAL_FILL
        
        return ExecutionType.FILL
    
    def _calculate_cumulative_quantity(self, fill_data: 'Fill') -> Decimal:
        """Calculate cumulative quantity from fill data"""
        
        if hasattr(fill_data, 'cumulative_quantity'):
            return Decimal(str(fill_data.cumulative_quantity))
        
        # If not available, assume this fill represents cumulative
        return fill_data.quantity
    
    def _calculate_remaining_quantity(self, fill_data: 'Fill') -> Decimal:
        """Calculate remaining quantity from fill data"""
        
        if hasattr(fill_data, 'remaining_quantity'):
            return Decimal(str(fill_data.remaining_quantity))
        
        # Calculate from order quantity if available
        if hasattr(fill_data, 'order_quantity'):
            return Decimal(str(fill_data.order_quantity)) - self._calculate_cumulative_quantity(fill_data)
        
        return Decimal('0')
    
    def _calculate_commission(self, fill_data: 'Fill') -> Decimal:
        """Calculate commission from fill data"""
        
        if hasattr(fill_data, 'commission'):
            return Decimal(str(fill_data.commission))
        
        # Calculate default commission if not provided
        notional = fill_data.quantity * fill_data.price
        default_commission_rate = Decimal('0.001')  # 0.1%
        
        return notional * default_commission_rate
    
    def _get_commission_currency(self, fill_data: 'Fill') -> str:
        """Get commission currency from fill data"""
        
        if hasattr(fill_data, 'commission_currency'):
            return fill_data.commission_currency
        
        # Extract from symbol or default to USD
        if hasattr(fill_data, 'symbol') and '/' in fill_data.symbol:
            return fill_data.symbol.split('/')[1]
        
        return "USD"
    
    def _calculate_net_proceeds(self, fill_data: 'Fill') -> Decimal:
        """Calculate net proceeds after commission"""
        
        gross_proceeds = fill_data.quantity * fill_data.price
        commission = self._calculate_commission(fill_data)
        
        # For sells, subtract commission; for buys, add commission to cost
        if hasattr(fill_data, 'side') and fill_data.side.upper() == 'SELL':
            return gross_proceeds - commission
        else:
            return gross_proceeds + commission
    
    def _extract_signal_id(self, fill_data: 'Fill') -> Optional[str]:
        """Extract signal ID from fill data or related order"""
        
        # Check direct signal ID
        if hasattr(fill_data, 'signal_id') and fill_data.signal_id:
            return str(fill_data.signal_id)
        
        # Check order metadata
        if hasattr(fill_data, 'order_metadata'):
            metadata = fill_data.order_metadata
            if isinstance(metadata, dict) and 'signal_id' in metadata:
                return str(metadata['signal_id'])
        
        # Generate signal ID from order ID if not available
        if hasattr(fill_data, 'order_id'):
            return f"signal_{fill_data.order_id}"
        
        return None
    
    def _create_event_tags(self, fill_data: 'Fill') -> Dict[str, str]:
        """Create event tags from fill data"""
        
        tags = {}
        
        # Add basic tags
        if hasattr(fill_data, 'venue'):
            tags['venue'] = str(fill_data.venue)
        
        if hasattr(fill_data, 'liquidity'):
            tags['liquidity'] = str(fill_data.liquidity)
        
        if hasattr(fill_data, 'order_type'):
            tags['order_type'] = str(fill_data.order_type)
        
        # Add custom tags from metadata
        if hasattr(fill_data, 'metadata') and isinstance(fill_data.metadata, dict):
            for key, value in fill_data.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    tags[f"custom_{key}"] = str(value)
        
        return tags
    
    def _validate_execution_event(self, event: ExecutionReportEvent) -> bool:
        """Validate execution event data"""
        
        # Check required fields
        if not event.order_id:
            self.logger.error("Missing order_id in execution event")
            return False
        
        if not event.symbol:
            self.logger.error("Missing symbol in execution event")
            return False
        
        if event.quantity <= 0:
            self.logger.error("Invalid quantity in execution event")
            return False
        
        if event.price <= 0:
            self.logger.error("Invalid price in execution event")
            return False
        
        # Validate cumulative vs remaining quantities
        if event.cumulative_quantity < event.quantity:
            self.logger.warning("Cumulative quantity less than fill quantity")
        
        return True

class ExecutionEventPublisher:
    """Publisher for execution report events"""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        
        # Event tracking
        self.published_events = 0
        self.failed_publications = 0
    
    async def publish_execution_event(self, event: ExecutionReportEvent) -> bool:
        """Publish execution event to event bus"""
        
        try:
            if self.event_bus:
                await self.event_bus.publish('execution_reports', event)
                self.published_events += 1
                self.logger.info(f"Published execution event: {event.event_id}")
                return True
            else:
                self.logger.warning("No event bus configured for execution events")
                return False
                
        except Exception as e:
            self.failed_publications += 1
            self.logger.error(f"Failed to publish execution event: {e}")
            return False
    
    def get_publication_stats(self) -> Dict[str, int]:
        """Get publication statistics"""
        return {
            'published_events': self.published_events,
            'failed_publications': self.failed_publications
        }

# Enhanced Fill class with execution event creation
class EnhancedFill:
    """Enhanced fill model with execution event creation"""
    
    def __init__(self):
        self.event_builder = ExecutionEventBuilder()
        self.event_publisher = ExecutionEventPublisher()
    
    def create_execution_event(self) -> ExecutionReportEvent:
        """
        Create execution event from fill data
        Replace placeholder execution event logic with actual implementation
        """
        return self.event_builder.create_execution_event(self)
    
    async def publish_execution_event(self) -> bool:
        """Create and publish execution event"""
        
        try:
            event = self.create_execution_event()
            return await self.event_publisher.publish_execution_event(event)
        except Exception as e:
            self.logger.error(f"Error publishing execution event: {e}")
            return False
```

#### c. Key Considerations & Dependencies
- **Event Schema:** Complete ExecutionReportEvent definition with all required fields
- **Data Validation:** Comprehensive validation of event data before creation
- **Event Publishing:** Integration with event bus for real-time notifications
- **Error Handling:** Robust error handling for event creation failures

### 4. Acceptance Criteria
- [ ] ExecutionReportEvent schema defined with complete field structure
- [ ] Actual event construction replaces placeholder logic
- [ ] Comprehensive data extraction from fill objects
- [ ] Event validation ensures data integrity
- [ ] Event publishing integration with message bus
- [ ] Error handling for event creation failures
- [ ] Event statistics and monitoring
- [ ] Placeholder execution event logic completely replaced 