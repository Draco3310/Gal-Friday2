# Task: Implement full order book update processing for real-time market data

### 1. Context
- **File:** `gal_friday/execution/websocket_client.py`
- **Line:** `532-533`
- **Keyword/Pattern:** `"Placeholder"`
- **Current State:** Placeholder implementation that does not process order book updates; critical for real-time trading decisions

### 2. Problem Statement
The WebSocket client currently has placeholder code for order book processing, preventing the system from receiving and processing real-time market depth data. This severely limits the trading system's ability to make informed decisions based on current market liquidity, bid-ask spreads, and order flow dynamics.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Order Book Data Models:** Define comprehensive structures for bid/ask levels and market depth
2. **Implement Message Parsing:** Parse incoming WebSocket depth messages from exchange APIs
3. **Build Order Book State Management:** Maintain accurate order book state with incremental updates
4. **Add Market Data Validation:** Validate incoming data integrity and handle malformed messages
5. **Create Event Broadcasting:** Publish order book updates to downstream consumers
6. **Implement Performance Optimization:** Efficient data structures and update algorithms

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
import json
import asyncio
import logging
from enum import Enum
import time

class OrderBookSide(str, Enum):
    """Order book sides"""
    BID = "bid"
    ASK = "ask"

@dataclass
class OrderBookLevel:
    """Individual price level in order book"""
    price: Decimal
    quantity: Decimal
    timestamp: float
    
    def is_empty(self) -> bool:
        """Check if level should be removed"""
        return self.quantity <= 0

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get highest bid price"""
        return self.bids[0] if self.bids else None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get lowest ask price"""
        return self.asks[0] if self.asks else None
    
    def get_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

@dataclass
class OrderBookUpdate:
    """Incremental order book update"""
    symbol: str
    side: OrderBookSide
    price: Decimal
    quantity: Decimal
    timestamp: float
    sequence: int
    
    def is_removal(self) -> bool:
        """Check if this update removes a level"""
        return self.quantity <= 0

class OrderBookManager:
    """Enterprise-grade order book management with real-time updates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Order book storage
        self.order_books: Dict[str, OrderBookSnapshot] = {}
        
        # Update tracking
        self.last_sequence: Dict[str, int] = {}
        self.update_counts: Dict[str, int] = {}
        
        # Performance metrics
        self.processing_times: List[float] = []
        self.message_queue_size = 0
        
        # Configuration
        self.max_depth = config.get('max_order_book_depth', 100)
        self.validation_enabled = config.get('validate_order_book', True)
        
    async def process_depth_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming WebSocket depth message
        Replace placeholder with comprehensive order book processing
        """
        
        try:
            start_time = time.time()
            
            # Parse message based on exchange format
            parsed_update = self._parse_depth_message(message)
            
            if not parsed_update:
                self.logger.warning(f"Failed to parse depth message: {message}")
                return
            
            # Validate message integrity
            if self.validation_enabled:
                validation_result = self._validate_depth_message(parsed_update)
                if not validation_result.is_valid:
                    self.logger.error(f"Invalid depth message: {validation_result.error}")
                    return
            
            # Check sequence number for gap detection
            await self._check_sequence_gap(parsed_update)
            
            # Apply update to order book
            await self._apply_order_book_update(parsed_update)
            
            # Publish update to subscribers
            await self._publish_order_book_update(parsed_update)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Maintain performance metrics window
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-500:]
            
            self.logger.debug(f"Processed depth update for {parsed_update.symbol} in {processing_time:.4f}s")
            
        except Exception as e:
            self.logger.error(f"Error processing depth message: {e}")
            await self._handle_processing_error(message, e)
    
    def _parse_depth_message(self, message: Dict[str, Any]) -> Optional[OrderBookUpdate]:
        """Parse WebSocket depth message based on exchange format"""
        
        try:
            # Handle different exchange message formats
            if 'data' in message and 'symbol' in message['data']:
                # Kraken format
                return self._parse_kraken_depth(message)
            elif 's' in message and 'b' in message:
                # Binance format
                return self._parse_binance_depth(message)
            else:
                # Generic format
                return self._parse_generic_depth(message)
                
        except Exception as e:
            self.logger.error(f"Error parsing depth message: {e}")
            return None
    
    def _parse_kraken_depth(self, message: Dict[str, Any]) -> Optional[OrderBookUpdate]:
        """Parse Kraken-specific depth message format"""
        
        try:
            data = message['data']
            symbol = data['symbol']
            
            # Process bids and asks
            updates = []
            
            if 'bids' in data:
                for bid in data['bids']:
                    updates.append(OrderBookUpdate(
                        symbol=symbol,
                        side=OrderBookSide.BID,
                        price=Decimal(str(bid[0])),
                        quantity=Decimal(str(bid[1])),
                        timestamp=data.get('timestamp', time.time()),
                        sequence=data.get('sequence', 0)
                    ))
            
            if 'asks' in data:
                for ask in data['asks']:
                    updates.append(OrderBookUpdate(
                        symbol=symbol,
                        side=OrderBookSide.ASK,
                        price=Decimal(str(ask[0])),
                        quantity=Decimal(str(ask[1])),
                        timestamp=data.get('timestamp', time.time()),
                        sequence=data.get('sequence', 0)
                    ))
            
            return updates[0] if updates else None
            
        except Exception as e:
            self.logger.error(f"Error parsing Kraken depth message: {e}")
            return None
    
    async def _apply_order_book_update(self, update: OrderBookUpdate) -> None:
        """Apply incremental update to order book state"""
        
        symbol = update.symbol
        
        # Initialize order book if not exists
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBookSnapshot(symbol=symbol)
        
        order_book = self.order_books[symbol]
        
        # Update sequence number
        order_book.sequence = max(order_book.sequence, update.sequence)
        order_book.timestamp = update.timestamp
        
        # Apply update based on side
        if update.side == OrderBookSide.BID:
            self._update_bid_side(order_book, update)
        else:
            self._update_ask_side(order_book, update)
        
        # Maintain order book depth limits
        self._trim_order_book_depth(order_book)
        
        # Update statistics
        self.update_counts[symbol] = self.update_counts.get(symbol, 0) + 1
    
    def _update_bid_side(self, order_book: OrderBookSnapshot, update: OrderBookUpdate) -> None:
        """Update bid side of order book"""
        
        # Find existing level or insertion point
        insertion_index = 0
        found_existing = False
        
        for i, level in enumerate(order_book.bids):
            if level.price == update.price:
                # Update existing level
                if update.is_removal():
                    # Remove level
                    order_book.bids.pop(i)
                else:
                    # Update quantity
                    level.quantity = update.quantity
                    level.timestamp = update.timestamp
                found_existing = True
                break
            elif level.price < update.price:
                insertion_index = i
                break
            else:
                insertion_index = i + 1
        
        # Insert new level if not updating existing
        if not found_existing and not update.is_removal():
            new_level = OrderBookLevel(
                price=update.price,
                quantity=update.quantity,
                timestamp=update.timestamp
            )
            order_book.bids.insert(insertion_index, new_level)
    
    def _update_ask_side(self, order_book: OrderBookSnapshot, update: OrderBookUpdate) -> None:
        """Update ask side of order book"""
        
        # Find existing level or insertion point
        insertion_index = len(order_book.asks)
        found_existing = False
        
        for i, level in enumerate(order_book.asks):
            if level.price == update.price:
                # Update existing level
                if update.is_removal():
                    # Remove level
                    order_book.asks.pop(i)
                else:
                    # Update quantity
                    level.quantity = update.quantity
                    level.timestamp = update.timestamp
                found_existing = True
                break
            elif level.price > update.price:
                insertion_index = i
                break
        
        # Insert new level if not updating existing
        if not found_existing and not update.is_removal():
            new_level = OrderBookLevel(
                price=update.price,
                quantity=update.quantity,
                timestamp=update.timestamp
            )
            order_book.asks.insert(insertion_index, new_level)
    
    def _trim_order_book_depth(self, order_book: OrderBookSnapshot) -> None:
        """Maintain order book depth within configured limits"""
        
        if len(order_book.bids) > self.max_depth:
            order_book.bids = order_book.bids[:self.max_depth]
        
        if len(order_book.asks) > self.max_depth:
            order_book.asks = order_book.asks[:self.max_depth]
    
    async def _publish_order_book_update(self, update: OrderBookUpdate) -> None:
        """Publish order book update to downstream consumers"""
        
        try:
            # Create update event
            event = {
                'type': 'order_book_update',
                'symbol': update.symbol,
                'side': update.side.value,
                'price': float(update.price),
                'quantity': float(update.quantity),
                'timestamp': update.timestamp,
                'sequence': update.sequence
            }
            
            # Publish to message bus (placeholder for actual implementation)
            await self._publish_event(event)
            
        except Exception as e:
            self.logger.error(f"Error publishing order book update: {e}")
    
    def get_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book snapshot"""
        return self.order_books.get(symbol)
    
    def get_market_depth(self, symbol: str, levels: int = 10) -> Dict[str, Any]:
        """Get market depth for specified symbol"""
        
        order_book = self.get_order_book(symbol)
        if not order_book:
            return {}
        
        return {
            'symbol': symbol,
            'bids': [
                {'price': float(level.price), 'quantity': float(level.quantity)}
                for level in order_book.bids[:levels]
            ],
            'asks': [
                {'price': float(level.price), 'quantity': float(level.quantity)}
                for level in order_book.asks[:levels]
            ],
            'timestamp': order_book.timestamp,
            'sequence': order_book.sequence
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get order book processing performance metrics"""
        
        if not self.processing_times:
            return {}
        
        import statistics
        
        return {
            'avg_processing_time': statistics.mean(self.processing_times),
            'median_processing_time': statistics.median(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'total_updates': sum(self.update_counts.values()),
            'symbols_tracked': len(self.order_books),
            'queue_size': self.message_queue_size
        }

@dataclass
class ValidationResult:
    """Result of message validation"""
    is_valid: bool
    error: Optional[str] = None
```

#### c. Key Considerations & Dependencies
- **Performance:** Efficient data structures for order book maintenance; optimized update algorithms; memory management for large order books
- **Reliability:** Sequence number gap detection; message validation; error recovery mechanisms
- **Configuration:** Configurable order book depth; exchange-specific message formats; validation settings
- **Testing:** Unit tests for order book operations; integration tests with WebSocket feeds; performance benchmarks

### 4. Acceptance Criteria
- [ ] Full order book update processing replaces placeholder implementation
- [ ] Support for multiple exchange message formats (Kraken, Binance, generic)
- [ ] Accurate order book state maintenance with incremental updates
- [ ] Bid and ask level management with proper price-time priority
- [ ] Message validation and error handling for malformed data
- [ ] Sequence number tracking and gap detection
- [ ] Performance optimization for high-frequency updates
- [ ] Event broadcasting to downstream consumers
- [ ] Market depth and spread calculation utilities
- [ ] Comprehensive logging and performance metrics
- [ ] Placeholder code is completely replaced with production implementation 