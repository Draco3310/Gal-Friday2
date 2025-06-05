# Task: Complete order book processing logic including bids/asks handling and error scenarios

### 1. Context
- **File:** `gal_friday/execution/websocket_client.py`
- **Line:** `541`
- **Keyword/Pattern:** `TODO`
- **Current State:** Incomplete order book processing logic that lacks comprehensive bids/asks handling and error scenarios

### 2. Problem Statement
The WebSocket client's order book processing logic is incomplete, missing critical functionality for handling bids/asks operations and error scenarios. This creates reliability issues in market data processing and could lead to trading decisions based on incomplete or corrupted market state.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Complete Bid/Ask Processing Logic:** Implement comprehensive bid and ask level management
2. **Add Error Scenario Handling:** Robust error handling for various failure modes
3. **Implement State Recovery:** Automatic recovery from processing errors and data corruption
4. **Add Data Integrity Checks:** Validation and consistency checks for order book state
5. **Create Monitoring Integration:** Real-time monitoring and alerting for processing issues
6. **Build Fallback Mechanisms:** Graceful degradation when processing fails

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import asyncio
import logging
import time
from collections import deque

class ProcessingError(str, Enum):
    """Types of processing errors"""
    INVALID_PRICE = "invalid_price"
    INVALID_QUANTITY = "invalid_quantity"
    SEQUENCE_GAP = "sequence_gap"
    MALFORMED_MESSAGE = "malformed_message"
    STALE_DATA = "stale_data"
    MEMORY_LIMIT = "memory_limit"
    PROCESSING_TIMEOUT = "processing_timeout"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_type: ProcessingError
    message: str
    timestamp: float
    symbol: Optional[str] = None
    sequence: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingMetrics:
    """Metrics for order book processing performance"""
    messages_processed: int = 0
    errors_encountered: int = 0
    average_processing_time: float = 0.0
    last_update_time: float = 0.0
    sequence_gaps: int = 0
    recovery_attempts: int = 0

class OrderBookProcessor:
    """Complete order book processing with comprehensive error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.processing_active = True
        self.last_sequence: Dict[str, int] = {}
        self.error_counts: Dict[ProcessingError, int] = {}
        
        # Error handling configuration
        self.max_sequence_gap = config.get('max_sequence_gap', 10)
        self.stale_data_threshold = config.get('stale_data_threshold_seconds', 30)
        self.max_processing_time = config.get('max_processing_time_seconds', 1.0)
        self.error_recovery_enabled = config.get('error_recovery_enabled', True)
        
        # Performance tracking
        self.metrics = ProcessingMetrics()
        self.processing_times = deque(maxlen=1000)
        
        # Error recovery
        self.recovery_strategies = {
            ProcessingError.SEQUENCE_GAP: self._recover_from_sequence_gap,
            ProcessingError.STALE_DATA: self._recover_from_stale_data,
            ProcessingError.MALFORMED_MESSAGE: self._recover_from_malformed_message,
            ProcessingError.MEMORY_LIMIT: self._recover_from_memory_limit
        }
    
    async def process_order_book_update(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Complete order book processing logic with comprehensive error handling
        Replace TODO with full implementation including bids/asks and error scenarios
        """
        
        start_time = time.time()
        
        try:
            # Validate input parameters
            if not self._validate_processing_input(symbol, data):
                await self._handle_processing_error(
                    ProcessingError.MALFORMED_MESSAGE,
                    "Invalid input parameters",
                    symbol=symbol,
                    raw_data=data
                )
                return False
            
            # Check if processing is active
            if not self.processing_active:
                self.logger.warning(f"Processing disabled, skipping update for {symbol}")
                return False
            
            # Extract sequence number for gap detection
            sequence = data.get('sequence', 0)
            await self._check_sequence_integrity(symbol, sequence)
            
            # Check for stale data
            timestamp = data.get('timestamp', time.time())
            if await self._is_stale_data(timestamp):
                await self._handle_processing_error(
                    ProcessingError.STALE_DATA,
                    f"Stale data detected: {time.time() - timestamp:.2f}s old",
                    symbol=symbol
                )
                return False
            
            # Process bids and asks with error handling
            success = await self._process_bids_asks_with_error_handling(symbol, data)
            
            if success:
                # Update processing metrics
                processing_time = time.time() - start_time
                await self._update_processing_metrics(processing_time)
                
                # Check processing time threshold
                if processing_time > self.max_processing_time:
                    self.logger.warning(
                        f"Slow processing detected: {processing_time:.4f}s for {symbol}"
                    )
                
                self.logger.debug(f"Successfully processed order book update for {symbol}")
                
            return success
            
        except asyncio.TimeoutError:
            await self._handle_processing_error(
                ProcessingError.PROCESSING_TIMEOUT,
                f"Processing timeout exceeded {self.max_processing_time}s",
                symbol=symbol
            )
            return False
            
        except MemoryError:
            await self._handle_processing_error(
                ProcessingError.MEMORY_LIMIT,
                "Memory limit exceeded during processing",
                symbol=symbol
            )
            return False
            
        except Exception as e:
            await self._handle_processing_error(
                ProcessingError.MALFORMED_MESSAGE,
                f"Unexpected error during processing: {e}",
                symbol=symbol,
                raw_data=data
            )
            return False
    
    async def _process_bids_asks_with_error_handling(self, symbol: str, data: Dict[str, Any]) -> bool:
        """Process bids and asks with comprehensive error handling"""
        
        try:
            bids_success = True
            asks_success = True
            
            # Process bids if present
            if 'bids' in data:
                bids_success = await self._process_bid_levels(symbol, data['bids'])
            
            # Process asks if present
            if 'asks' in data:
                asks_success = await self._process_ask_levels(symbol, data['asks'])
            
            # Validate order book consistency after updates
            if bids_success and asks_success:
                consistency_check = await self._validate_order_book_consistency(symbol)
                if not consistency_check:
                    self.logger.error(f"Order book consistency check failed for {symbol}")
                    await self._trigger_order_book_recovery(symbol)
                    return False
            
            return bids_success and asks_success
            
        except Exception as e:
            self.logger.error(f"Error processing bids/asks for {symbol}: {e}")
            return False
    
    async def _process_bid_levels(self, symbol: str, bids: List[List[Any]]) -> bool:
        """Process bid levels with error handling"""
        
        try:
            processed_count = 0
            
            for bid_data in bids:
                try:
                    # Validate bid data structure
                    if not self._validate_level_data(bid_data):
                        self.logger.warning(f"Invalid bid data for {symbol}: {bid_data}")
                        continue
                    
                    # Extract and validate price/quantity
                    price, quantity = await self._extract_price_quantity(bid_data)
                    
                    if price is None or quantity is None:
                        self.logger.warning(f"Invalid price/quantity in bid for {symbol}")
                        continue
                    
                    # Apply bid update to order book
                    await self._apply_bid_update(symbol, price, quantity)
                    processed_count += 1
                    
                except (ValueError, InvalidOperation) as e:
                    await self._handle_processing_error(
                        ProcessingError.INVALID_PRICE,
                        f"Invalid bid price/quantity for {symbol}: {e}",
                        symbol=symbol
                    )
                    continue
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing bid for {symbol}: {e}")
                    continue
            
            self.logger.debug(f"Processed {processed_count}/{len(bids)} bid levels for {symbol}")
            return processed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing bid levels for {symbol}: {e}")
            return False
    
    async def _process_ask_levels(self, symbol: str, asks: List[List[Any]]) -> bool:
        """Process ask levels with error handling"""
        
        try:
            processed_count = 0
            
            for ask_data in asks:
                try:
                    # Validate ask data structure
                    if not self._validate_level_data(ask_data):
                        self.logger.warning(f"Invalid ask data for {symbol}: {ask_data}")
                        continue
                    
                    # Extract and validate price/quantity
                    price, quantity = await self._extract_price_quantity(ask_data)
                    
                    if price is None or quantity is None:
                        self.logger.warning(f"Invalid price/quantity in ask for {symbol}")
                        continue
                    
                    # Apply ask update to order book
                    await self._apply_ask_update(symbol, price, quantity)
                    processed_count += 1
                    
                except (ValueError, InvalidOperation) as e:
                    await self._handle_processing_error(
                        ProcessingError.INVALID_PRICE,
                        f"Invalid ask price/quantity for {symbol}: {e}",
                        symbol=symbol
                    )
                    continue
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error processing ask for {symbol}: {e}")
                    continue
            
            self.logger.debug(f"Processed {processed_count}/{len(asks)} ask levels for {symbol}")
            return processed_count > 0
            
        except Exception as e:
            self.logger.error(f"Error processing ask levels for {symbol}: {e}")
            return False
    
    async def _extract_price_quantity(self, level_data: List[Any]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Extract and validate price and quantity from level data"""
        
        try:
            if len(level_data) < 2:
                return None, None
            
            # Extract price
            try:
                price = Decimal(str(level_data[0]))
                if price < 0:
                    self.logger.warning(f"Negative price detected: {price}")
                    return None, None
            except (ValueError, InvalidOperation):
                self.logger.warning(f"Invalid price format: {level_data[0]}")
                return None, None
            
            # Extract quantity
            try:
                quantity = Decimal(str(level_data[1]))
                if quantity < 0:
                    self.logger.warning(f"Negative quantity detected: {quantity}")
                    return None, None
            except (ValueError, InvalidOperation):
                self.logger.warning(f"Invalid quantity format: {level_data[1]}")
                return None, None
            
            return price, quantity
            
        except Exception as e:
            self.logger.error(f"Error extracting price/quantity: {e}")
            return None, None
    
    async def _check_sequence_integrity(self, symbol: str, sequence: int) -> None:
        """Check for sequence gaps and handle recovery"""
        
        if symbol in self.last_sequence:
            expected_sequence = self.last_sequence[symbol] + 1
            
            if sequence > expected_sequence:
                gap_size = sequence - expected_sequence
                
                if gap_size > self.max_sequence_gap:
                    await self._handle_processing_error(
                        ProcessingError.SEQUENCE_GAP,
                        f"Large sequence gap detected: {gap_size} messages missing",
                        symbol=symbol,
                        sequence=sequence
                    )
                    
                    if self.error_recovery_enabled:
                        await self._recover_from_sequence_gap(symbol, sequence)
                else:
                    self.logger.warning(f"Small sequence gap for {symbol}: {gap_size} messages")
                    self.metrics.sequence_gaps += 1
        
        self.last_sequence[symbol] = sequence
    
    async def _validate_order_book_consistency(self, symbol: str) -> bool:
        """Validate order book consistency after updates"""
        
        try:
            # Get current order book
            order_book = await self._get_order_book(symbol)
            
            if not order_book:
                return True  # Empty order book is consistent
            
            # Check bid/ask spread sanity
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if best_bid and best_ask:
                if best_bid.price >= best_ask.price:
                    self.logger.error(f"Crossed order book detected for {symbol}: bid={best_bid.price}, ask={best_ask.price}")
                    return False
                
                spread = best_ask.price - best_bid.price
                if spread > best_ask.price * Decimal('0.1'):  # 10% spread threshold
                    self.logger.warning(f"Wide spread detected for {symbol}: {spread}")
            
            # Check for duplicate price levels
            bid_prices = [level.price for level in order_book.bids]
            ask_prices = [level.price for level in order_book.asks]
            
            if len(bid_prices) != len(set(bid_prices)):
                self.logger.error(f"Duplicate bid prices detected for {symbol}")
                return False
            
            if len(ask_prices) != len(set(ask_prices)):
                self.logger.error(f"Duplicate ask prices detected for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order book consistency for {symbol}: {e}")
            return False
    
    async def _handle_processing_error(self, error_type: ProcessingError, message: str, 
                                     symbol: Optional[str] = None, 
                                     sequence: Optional[int] = None,
                                     raw_data: Optional[Dict[str, Any]] = None) -> None:
        """Centralized error handling with recovery strategies"""
        
        # Track error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.metrics.errors_encountered += 1
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type,
            message=message,
            timestamp=time.time(),
            symbol=symbol,
            sequence=sequence,
            raw_data=raw_data
        )
        
        # Log error with appropriate level
        if error_type in [ProcessingError.SEQUENCE_GAP, ProcessingError.STALE_DATA]:
            self.logger.warning(f"Processing warning ({error_type.value}): {message}")
        else:
            self.logger.error(f"Processing error ({error_type.value}): {message}")
        
        # Apply recovery strategy if available and enabled
        if self.error_recovery_enabled and error_type in self.recovery_strategies:
            try:
                await self.recovery_strategies[error_type](error_context)
                self.metrics.recovery_attempts += 1
            except Exception as e:
                self.logger.error(f"Error recovery failed for {error_type.value}: {e}")
        
        # Publish error event for monitoring
        await self._publish_error_event(error_context)
    
    async def _recover_from_sequence_gap(self, error_context: ErrorContext) -> None:
        """Recover from sequence gap by requesting order book snapshot"""
        
        if error_context.symbol:
            self.logger.info(f"Attempting recovery from sequence gap for {error_context.symbol}")
            # Trigger order book snapshot request
            await self._request_order_book_snapshot(error_context.symbol)
    
    async def _recover_from_stale_data(self, error_context: ErrorContext) -> None:
        """Recover from stale data by clearing old state"""
        
        if error_context.symbol:
            self.logger.info(f"Clearing stale order book data for {error_context.symbol}")
            await self._clear_order_book_state(error_context.symbol)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        
        return {
            'metrics': {
                'messages_processed': self.metrics.messages_processed,
                'errors_encountered': self.metrics.errors_encountered,
                'average_processing_time': self.metrics.average_processing_time,
                'sequence_gaps': self.metrics.sequence_gaps,
                'recovery_attempts': self.metrics.recovery_attempts
            },
            'error_counts': dict(self.error_counts),
            'processing_active': self.processing_active,
            'symbols_tracked': len(self.last_sequence)
        }
```

#### c. Key Considerations & Dependencies
- **Error Resilience:** Comprehensive error handling for all processing scenarios; automatic recovery mechanisms; graceful degradation
- **Performance:** Efficient processing algorithms; memory management; processing time monitoring
- **Data Integrity:** Validation of price/quantity data; order book consistency checks; sequence integrity verification
- **Monitoring:** Real-time error tracking; performance metrics; alerting for critical issues

### 4. Acceptance Criteria
- [ ] Complete order book processing logic handles all bid and ask operations
- [ ] Comprehensive error handling covers malformed data, sequence gaps, and stale data scenarios
- [ ] Automatic recovery mechanisms for common error conditions
- [ ] Data integrity validation ensures order book consistency
- [ ] Performance monitoring tracks processing times and error rates
- [ ] Memory management prevents processing from consuming excessive resources
- [ ] Sequence gap detection and recovery maintains data continuity
- [ ] Price and quantity validation prevents invalid data from corrupting order book
- [ ] Error event publishing enables real-time monitoring and alerting
- [ ] Processing statistics provide visibility into system health and performance
- [ ] TODO placeholder is completely replaced with production-ready implementation 