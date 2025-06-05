# Task: Add lot size calculation and conversion logic for different order types.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1917`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing lot size calculation and conversion logic for different order types.

### 2. Problem Statement
Without proper lot size calculation and conversion logic, orders cannot be properly formatted for different exchanges and order types. Each exchange has specific lot size requirements, minimum order sizes, and step sizes that must be respected. Failure to correctly calculate and convert lot sizes leads to order rejections and failed trades.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Exchange Specification Service:** Centralized service for lot size specifications per exchange/symbol
2. **Build Lot Size Calculator:** Core logic for calculating appropriate lot sizes based on order parameters
3. **Implement Multi-Exchange Support:** Handle different lot size conventions across various exchanges
4. **Add Order Type Conversion:** Convert between different order size representations (units, notional, percentage)
5. **Create Validation Framework:** Validate calculated lot sizes against exchange requirements
6. **Build Optimization Logic:** Optimize lot sizes for best execution and minimal market impact

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
import logging

class OrderSizeType(str, Enum):
    UNITS = "units"              # Raw units/shares
    NOTIONAL = "notional"        # Dollar/currency amount
    PERCENTAGE = "percentage"    # Percentage of portfolio
    LOTS = "lots"               # Exchange-specific lots
    CONTRACTS = "contracts"      # Futures/options contracts

@dataclass
class LotSizeResult:
    """Result of lot size calculation"""
    original_size: Union[float, Decimal]
    original_type: OrderSizeType
    calculated_lots: Decimal
    calculated_units: Decimal
    calculated_notional: Decimal
    exchange_valid: bool
    adjustments_made: List[str]
    warnings: List[str]

class LotSizeCalculator:
    """Enterprise-grade lot size calculation and conversion service"""
    
    def __init__(self, exchange_info_service, market_data_service, config: Dict[str, Any]):
        self.exchange_info = exchange_info_service
        self.market_data = market_data_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.lot_specs_cache = {}
    
    async def calculate_and_convert_lot_size(self, 
                                           exchange: str,
                                           symbol: str, 
                                           order_size: Union[float, Decimal],
                                           size_type: OrderSizeType,
                                           current_price: Optional[float] = None) -> LotSizeResult:
        """
        Calculate and convert lot sizes for different order types
        Replace TODO with comprehensive lot size calculation logic
        """
        
        try:
            self.logger.debug(f"Calculating lot size for {exchange}:{symbol}, size={order_size}, type={size_type.value}")
            
            # Get exchange lot specifications
            lot_specs = await self._get_lot_specifications(exchange, symbol)
            
            # Get current market price if not provided
            if current_price is None:
                current_price = await self._get_current_price(symbol)
            
            # Convert input size to units (base calculation)
            units = await self._convert_to_units(order_size, size_type, symbol, current_price)
            
            # Calculate lot size from units
            lots = await self._calculate_lots_from_units(units, lot_specs)
            
            # Validate and adjust lot size
            validated_lots, adjustments, warnings = await self._validate_and_adjust_lots(lots, lot_specs)
            
            # Convert back to all representations
            final_units = validated_lots * lot_specs['lot_size']
            final_notional = final_units * Decimal(str(current_price))
            
            # Check if result is exchange-valid
            exchange_valid = await self._validate_exchange_requirements(validated_lots, final_notional, lot_specs)
            
            return LotSizeResult(
                original_size=order_size,
                original_type=size_type,
                calculated_lots=validated_lots,
                calculated_units=final_units,
                calculated_notional=final_notional,
                exchange_valid=exchange_valid,
                adjustments_made=adjustments,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating lot size for {exchange}:{symbol}: {e}")
            raise LotSizeCalculationError(f"Lot size calculation failed: {e}")
    
    async def _convert_to_units(self, order_size: Union[float, Decimal], size_type: OrderSizeType, 
                              symbol: str, current_price: float) -> Decimal:
        """Convert various order size types to base units"""
        
        order_size = Decimal(str(order_size))
        
        if size_type == OrderSizeType.UNITS:
            return order_size
        elif size_type == OrderSizeType.NOTIONAL:
            return order_size / Decimal(str(current_price))
        elif size_type == OrderSizeType.PERCENTAGE:
            portfolio_value = await self._get_portfolio_value()
            notional_amount = (order_size / 100) * Decimal(str(portfolio_value))
            return notional_amount / Decimal(str(current_price))
        else:
            raise ValueError(f"Unsupported order size type: {size_type}")
    
    async def _calculate_lots_from_units(self, units: Decimal, lot_specs: Dict[str, Any]) -> Decimal:
        """Calculate lot size from units based on exchange specifications"""
        
        lot_size = Decimal(str(lot_specs.get('lot_size', 1)))
        raw_lots = units / lot_size
        return raw_lots.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    
    async def _validate_and_adjust_lots(self, lots: Decimal, lot_specs: Dict[str, Any]) -> Tuple[Decimal, List[str], List[str]]:
        """Validate lot size against exchange requirements"""
        
        min_lots = Decimal(str(lot_specs.get('min_lot_size', 1)))
        max_lots = Decimal(str(lot_specs.get('max_lot_size', 1000000)))
        
        adjustments = []
        warnings = []
        
        if lots < min_lots:
            lots = min_lots
            adjustments.append(f"Increased to minimum lot size: {min_lots}")
        
        if lots > max_lots:
            lots = max_lots
            adjustments.append(f"Reduced to maximum lot size: {max_lots}")
        
        return lots, adjustments, warnings

class LotSizeCalculationError(Exception):
    """Exception raised for lot size calculation errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of exchange info service failures; fallback to default lot specifications; comprehensive error logging
- **Configuration:** Configurable lot size specifications per exchange; override settings for special symbols; batch processing optimization
- **Testing:** Unit tests for various lot size calculations; integration tests with exchange APIs; edge case testing for boundary conditions
- **Dependencies:** Exchange info service for lot specifications; market data service for optimization; caching service for performance

### 4. Acceptance Criteria
- [ ] Lot size calculation accurately converts between different order size types (units, notional, percentage, lots)
- [ ] Exchange-specific lot requirements are properly enforced and validated
- [ ] Intelligent rounding optimization improves order fill rates based on market conditions
- [ ] Batch calculation efficiently handles multiple orders for portfolio operations
- [ ] Validation framework prevents invalid lot sizes from reaching exchange APIs
- [ ] Performance testing shows calculation completion under 5ms per order
- [ ] Integration tests verify accuracy with real exchange specifications
- [ ] Error handling provides clear feedback for lot size adjustment reasons
- [ ] TODO placeholder is completely replaced with production-ready implementation 