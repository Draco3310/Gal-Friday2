# Task: Build initial validation and price rounding logic with exchange precision checks.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1870`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing initial validation and price rounding logic with exchange precision checks.

### 2. Problem Statement
Without proper initial validation and price rounding logic, orders may be submitted with invalid prices that don't conform to exchange specifications, leading to order rejections and failed trades. Each exchange has specific precision requirements for different symbols, and the system must validate and round prices accordingly to ensure successful order execution.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Exchange Info Service Integration:** Connect to exchange APIs to retrieve precision specifications
2. **Build Price Validation Framework:** Implement comprehensive price validation against exchange rules
3. **Implement Smart Rounding Logic:** Create intelligent rounding that minimizes market impact
4. **Add Symbol-Specific Validation:** Handle different precision requirements per trading pair
5. **Create Validation Cache:** Cache exchange specifications for performance optimization
6. **Build Error Handling:** Graceful handling of validation failures with actionable feedback

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, Optional, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from dataclasses import dataclass
from enum import Enum
import logging

class RoundingDirection(str, Enum):
    UP = "up"
    DOWN = "down" 
    NEAREST = "nearest"
    AWAY_FROM_ZERO = "away_from_zero"

@dataclass
class ExchangePrecisionInfo:
    """Exchange precision specifications for a symbol"""
    symbol: str
    price_precision: int  # decimal places
    quantity_precision: int  # decimal places
    min_price: Decimal
    max_price: Decimal
    tick_size: Decimal  # minimum price increment
    min_quantity: Decimal
    max_quantity: Decimal
    step_size: Decimal  # minimum quantity increment
    min_notional: Decimal  # minimum order value

@dataclass
class ValidationResult:
    """Result of price/quantity validation"""
    is_valid: bool
    original_value: Decimal
    rounded_value: Decimal
    adjustments_made: list
    validation_errors: list
    warnings: list

class ExchangeValidationService:
    """Service for validating and rounding prices according to exchange specifications"""
    
    def __init__(self, exchange_info_service, cache_service):
        self.exchange_info = exchange_info_service
        self.cache = cache_service
        self.logger = logging.getLogger(__name__)
        self.precision_cache = {}
    
    async def validate_and_round_order(self, symbol: str, price: float, 
                                     quantity: float, order_type: str) -> Tuple[ValidationResult, ValidationResult]:
        """
        Validate and round order parameters according to exchange specifications
        Replace TODO with comprehensive validation logic
        """
        
        try:
            self.logger.debug(f"Validating order for {symbol}: price={price}, quantity={quantity}, type={order_type}")
            
            # Get exchange precision info
            precision_info = await self._get_precision_info(symbol)
            
            # Validate and round price
            price_result = await self._validate_and_round_price(
                symbol, Decimal(str(price)), precision_info, order_type
            )
            
            # Validate and round quantity
            quantity_result = await self._validate_and_round_quantity(
                symbol, Decimal(str(quantity)), precision_info
            )
            
            # Cross-validate price and quantity together
            await self._cross_validate_order(price_result, quantity_result, precision_info)
            
            self.logger.info(
                f"Order validation complete for {symbol}: "
                f"price {price} -> {price_result.rounded_value}, "
                f"quantity {quantity} -> {quantity_result.rounded_value}"
            )
            
            return price_result, quantity_result
            
        except Exception as e:
            self.logger.error(f"Error validating order for {symbol}: {e}")
            raise ValidationError(f"Order validation failed: {e}")
    
    async def _get_precision_info(self, symbol: str) -> ExchangePrecisionInfo:
        """Get exchange precision information for symbol"""
        
        # Check cache first
        cache_key = f"precision_info_{symbol}"
        cached_info = self.precision_cache.get(cache_key)
        
        if cached_info:
            return cached_info
        
        try:
            # Fetch from exchange info service
            exchange_info = await self.exchange_info.get_symbol_info(symbol)
            
            precision_info = ExchangePrecisionInfo(
                symbol=symbol,
                price_precision=exchange_info.get('price_precision', 8),
                quantity_precision=exchange_info.get('quantity_precision', 8),
                min_price=Decimal(str(exchange_info.get('min_price', '0.00000001'))),
                max_price=Decimal(str(exchange_info.get('max_price', '1000000'))),
                tick_size=Decimal(str(exchange_info.get('tick_size', '0.00000001'))),
                min_quantity=Decimal(str(exchange_info.get('min_quantity', '0.00000001'))),
                max_quantity=Decimal(str(exchange_info.get('max_quantity', '1000000'))),
                step_size=Decimal(str(exchange_info.get('step_size', '0.00000001'))),
                min_notional=Decimal(str(exchange_info.get('min_notional', '10')))
            )
            
            # Cache for 5 minutes
            self.precision_cache[cache_key] = precision_info
            
            return precision_info
            
        except Exception as e:
            self.logger.error(f"Failed to get precision info for {symbol}: {e}")
            # Return default precision info as fallback
            return self._get_default_precision_info(symbol)
    
    async def _validate_and_round_price(self, symbol: str, price: Decimal, 
                                      precision_info: ExchangePrecisionInfo,
                                      order_type: str) -> ValidationResult:
        """Validate and round price according to exchange specifications"""
        
        original_price = price
        adjustments_made = []
        validation_errors = []
        warnings = []
        
        # Check price bounds
        if price < precision_info.min_price:
            validation_errors.append(f"Price {price} below minimum {precision_info.min_price}")
            price = precision_info.min_price
            adjustments_made.append("price_raised_to_minimum")
        
        if price > precision_info.max_price:
            validation_errors.append(f"Price {price} above maximum {precision_info.max_price}")
            price = precision_info.max_price
            adjustments_made.append("price_lowered_to_maximum")
        
        # Round to tick size
        rounded_price = self._round_to_tick_size(price, precision_info.tick_size, order_type)
        
        if rounded_price != original_price:
            adjustments_made.append("price_rounded_to_tick_size")
            self.logger.debug(f"Price rounded from {original_price} to {rounded_price}")
        
        # Apply precision limits
        precision_rounded = rounded_price.quantize(
            Decimal('0.1') ** precision_info.price_precision,
            rounding=ROUND_HALF_UP
        )
        
        if precision_rounded != rounded_price:
            adjustments_made.append("price_rounded_to_precision")
            rounded_price = precision_rounded
        
        # Validate final price
        is_valid = len(validation_errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            original_value=original_price,
            rounded_value=rounded_price,
            adjustments_made=adjustments_made,
            validation_errors=validation_errors,
            warnings=warnings
        )
    
    async def _validate_and_round_quantity(self, symbol: str, quantity: Decimal,
                                         precision_info: ExchangePrecisionInfo) -> ValidationResult:
        """Validate and round quantity according to exchange specifications"""
        
        original_quantity = quantity
        adjustments_made = []
        validation_errors = []
        warnings = []
        
        # Check quantity bounds
        if quantity < precision_info.min_quantity:
            validation_errors.append(f"Quantity {quantity} below minimum {precision_info.min_quantity}")
            quantity = precision_info.min_quantity
            adjustments_made.append("quantity_raised_to_minimum")
        
        if quantity > precision_info.max_quantity:
            validation_errors.append(f"Quantity {quantity} above maximum {precision_info.max_quantity}")
            quantity = precision_info.max_quantity
            adjustments_made.append("quantity_lowered_to_maximum")
        
        # Round to step size
        if precision_info.step_size > 0:
            steps = (quantity / precision_info.step_size).quantize(Decimal('1'), rounding=ROUND_DOWN)
            rounded_quantity = steps * precision_info.step_size
            
            if rounded_quantity != original_quantity:
                adjustments_made.append("quantity_rounded_to_step_size")
                quantity = rounded_quantity
        
        # Apply precision limits
        precision_rounded = quantity.quantize(
            Decimal('0.1') ** precision_info.quantity_precision,
            rounding=ROUND_DOWN  # Always round down for quantity to avoid exceeding limits
        )
        
        if precision_rounded != quantity:
            adjustments_made.append("quantity_rounded_to_precision")
            quantity = precision_rounded
        
        # Final quantity check
        if quantity < precision_info.min_quantity:
            validation_errors.append(f"Final quantity {quantity} below minimum after rounding")
        
        is_valid = len(validation_errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            original_value=original_quantity,
            rounded_value=quantity,
            adjustments_made=adjustments_made,
            validation_errors=validation_errors,
            warnings=warnings
        )
    
    def _round_to_tick_size(self, price: Decimal, tick_size: Decimal, order_type: str) -> Decimal:
        """Round price to nearest valid tick size"""
        
        if tick_size <= 0:
            return price
        
        # Calculate number of ticks
        ticks = price / tick_size
        
        # Different rounding strategies based on order type
        if order_type.lower() in ['buy', 'bid']:
            # For buy orders, round down to get better fill price
            rounded_ticks = ticks.quantize(Decimal('1'), rounding=ROUND_DOWN)
        elif order_type.lower() in ['sell', 'ask']:
            # For sell orders, round up to get better fill price
            rounded_ticks = ticks.quantize(Decimal('1'), rounding=ROUND_UP)
        else:
            # For other orders, round to nearest
            rounded_ticks = ticks.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        
        return rounded_ticks * tick_size
    
    async def _cross_validate_order(self, price_result: ValidationResult, 
                                  quantity_result: ValidationResult,
                                  precision_info: ExchangePrecisionInfo) -> None:
        """Cross-validate price and quantity together"""
        
        # Check minimum notional value
        notional_value = price_result.rounded_value * quantity_result.rounded_value
        
        if notional_value < precision_info.min_notional:
            error_msg = (
                f"Order notional value {notional_value} below minimum {precision_info.min_notional}. "
                f"Either increase quantity or price."
            )
            price_result.validation_errors.append(error_msg)
            quantity_result.validation_errors.append(error_msg)
            price_result.is_valid = False
            quantity_result.is_valid = False
    
    def _get_default_precision_info(self, symbol: str) -> ExchangePrecisionInfo:
        """Get default precision info as fallback"""
        
        return ExchangePrecisionInfo(
            symbol=symbol,
            price_precision=8,
            quantity_precision=8,
            min_price=Decimal('0.00000001'),
            max_price=Decimal('1000000'),
            tick_size=Decimal('0.00000001'),
            min_quantity=Decimal('0.00000001'),
            max_quantity=Decimal('1000000'),
            step_size=Decimal('0.00000001'),
            min_notional=Decimal('10')
        )
    
    async def validate_order_pre_submission(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete order validation before submission
        Main entry point replacing TODO
        """
        
        symbol = order_data.get('symbol')
        price = order_data.get('price', 0)
        quantity = order_data.get('quantity', 0)
        order_type = order_data.get('type', 'limit')
        
        if not symbol:
            raise ValidationError("Symbol is required for order validation")
        
        # Validate and round price and quantity
        price_result, quantity_result = await self.validate_and_round_order(
            symbol, price, quantity, order_type
        )
        
        # Return validated order data
        validated_order = order_data.copy()
        validated_order['price'] = float(price_result.rounded_value)
        validated_order['quantity'] = float(quantity_result.rounded_value)
        
        # Add validation metadata
        validated_order['validation_metadata'] = {
            'price_adjustments': price_result.adjustments_made,
            'quantity_adjustments': quantity_result.adjustments_made,
            'validation_errors': price_result.validation_errors + quantity_result.validation_errors,
            'warnings': price_result.warnings + quantity_result.warnings,
            'is_valid': price_result.is_valid and quantity_result.is_valid
        }
        
        return validated_order

class ValidationError(Exception):
    """Exception raised for order validation errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of exchange info service failures; fallback to default precision when specifications unavailable; comprehensive validation error reporting
- **Configuration:** Configurable rounding strategies per order type; adjustable cache timeouts; symbol-specific override settings
- **Testing:** Unit tests for rounding logic with various tick sizes; integration tests with exchange APIs; edge case testing for boundary values
- **Dependencies:** Exchange info service for precision specifications; caching service for performance; configuration management for rounding strategies

### 4. Acceptance Criteria
- [ ] Price validation enforces exchange-specific tick sizes and precision requirements
- [ ] Quantity validation respects step sizes and minimum/maximum bounds
- [ ] Smart rounding logic optimizes fill prices based on order type (buy/sell)
- [ ] Cross-validation ensures minimum notional value requirements are met
- [ ] Validation cache improves performance for repeated symbol lookups
- [ ] Comprehensive error messages provide actionable feedback for validation failures
- [ ] Fallback precision specifications handle exchange info service failures gracefully
- [ ] Performance testing shows validation completion under 10ms per order
- [ ] Integration tests verify compatibility with exchange API specifications
- [ ] TODO placeholder is completely replaced with production-ready implementation 