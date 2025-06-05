# Task: Add market price–dependent checks (fat‑finger limits, SL/TP distance validation).

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1882`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing market price-dependent safety checks including fat-finger limits and stop-loss/take-profit distance validation.

### 2. Problem Statement
Without market price-dependent safety checks, the system cannot prevent obviously erroneous trades (fat-finger errors) or validate that stop-loss and take-profit levels are set at reasonable distances from current market prices. This creates significant risk exposure from trading errors and poorly configured risk management orders.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Market Price Monitoring Service:** Real-time market price tracking with statistical analysis
2. **Implement Fat-Finger Detection:** Statistical anomaly detection for order prices vs market prices  
3. **Build SL/TP Distance Validation:** Intelligent validation of stop-loss and take-profit placement
4. **Add Dynamic Threshold Management:** Adaptive thresholds based on market volatility and liquidity
5. **Create Market Context Analysis:** Consider market conditions when validating order prices
6. **Build Alert and Override System:** Configurable alerts with manual override capabilities

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timezone, timedelta
import logging

class PriceValidationSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MarketCondition(str, Enum):
    NORMAL = "normal"
    VOLATILE = "volatile"
    LOW_LIQUIDITY = "low_liquidity"
    TRENDING = "trending"
    GAPPING = "gapping"

@dataclass
class MarketPriceContext:
    """Current market price context for validation"""
    symbol: str
    current_price: float
    bid_price: float
    ask_price: float
    spread_pct: float
    volatility_1h: float
    volatility_24h: float
    volume_24h: float
    price_change_24h_pct: float
    market_condition: MarketCondition
    last_trade_time: datetime

@dataclass
class PriceValidationResult:
    """Result of market price validation"""
    is_valid: bool
    severity: PriceValidationSeverity
    validation_checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    suggested_price_range: Optional[Tuple[float, float]]
    market_context: MarketPriceContext

class MarketPriceDependentValidator:
    """Validates orders against current market conditions and price context"""
    
    def __init__(self, market_data_service, config: Dict[str, Any]):
        self.market_data = market_data_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.fat_finger_threshold_pct = config.get('fat_finger_threshold_pct', 10.0)
        self.min_sl_distance_pct = config.get('min_sl_distance_pct', 1.0)
        self.max_sl_distance_pct = config.get('max_sl_distance_pct', 50.0)
        self.min_tp_distance_pct = config.get('min_tp_distance_pct', 0.5)
        self.max_tp_distance_pct = config.get('max_tp_distance_pct', 100.0)
        
    async def validate_order_against_market_price(self, order_data: Dict[str, Any]) -> PriceValidationResult:
        """
        Comprehensive market price validation for orders
        Replace TODO with market price-dependent validation logic
        """
        
        symbol = order_data.get('symbol')
        order_price = order_data.get('price', 0)
        order_type = order_data.get('type', 'limit')
        side = order_data.get('side', 'buy')
        
        try:
            self.logger.debug(f"Validating order against market price for {symbol}")
            
            # Get current market context
            market_context = await self._get_market_price_context(symbol)
            
            # Initialize validation result
            validation_checks = {}
            warnings = []
            errors = []
            
            # Perform fat-finger detection
            fat_finger_check = await self._check_fat_finger_limits(
                order_price, side, market_context
            )
            validation_checks['fat_finger'] = fat_finger_check['passed']
            if not fat_finger_check['passed']:
                errors.extend(fat_finger_check['errors'])
            warnings.extend(fat_finger_check['warnings'])
            
            # Validate stop-loss and take-profit distances
            if order_type.lower() in ['stop_loss', 'stop-loss']:
                sl_check = await self._validate_stop_loss_distance(
                    order_price, side, market_context
                )
                validation_checks['stop_loss_distance'] = sl_check['passed']
                if not sl_check['passed']:
                    errors.extend(sl_check['errors'])
                warnings.extend(sl_check['warnings'])
            
            elif order_type.lower() in ['take_profit', 'take-profit']:
                tp_check = await self._validate_take_profit_distance(
                    order_price, side, market_context
                )
                validation_checks['take_profit_distance'] = tp_check['passed']
                if not tp_check['passed']:
                    errors.extend(tp_check['errors'])
                warnings.extend(tp_check['warnings'])
            
            # Check spread and liquidity impact
            spread_check = await self._check_spread_impact(
                order_price, side, market_context
            )
            validation_checks['spread_impact'] = spread_check['passed']
            if not spread_check['passed']:
                warnings.extend(spread_check['warnings'])
            
            # Market condition validation
            market_condition_check = await self._validate_market_conditions(
                order_price, side, market_context
            )
            validation_checks['market_conditions'] = market_condition_check['passed']
            warnings.extend(market_condition_check['warnings'])
            
            # Determine overall validation result
            is_valid = len(errors) == 0
            severity = self._determine_validation_severity(errors, warnings)
            
            # Generate suggested price range
            suggested_range = self._calculate_suggested_price_range(
                side, market_context
            )
            
            result = PriceValidationResult(
                is_valid=is_valid,
                severity=severity,
                validation_checks=validation_checks,
                warnings=warnings,
                errors=errors,
                suggested_price_range=suggested_range,
                market_context=market_context
            )
            
            self.logger.info(
                f"Market price validation for {symbol}: valid={is_valid}, "
                f"severity={severity.value}, checks_passed={sum(validation_checks.values())}/{len(validation_checks)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in market price validation for {symbol}: {e}")
            raise MarketValidationError(f"Market price validation failed: {e}")
    
    async def _get_market_price_context(self, symbol: str) -> MarketPriceContext:
        """Get comprehensive market price context for validation"""
        
        try:
            # Get current market data
            ticker = await self.market_data.get_ticker(symbol)
            orderbook = await self.market_data.get_orderbook(symbol, depth=5)
            
            current_price = ticker.get('last_price', 0)
            bid_price = ticker.get('bid_price', current_price * 0.999)
            ask_price = ticker.get('ask_price', current_price * 1.001)
            
            # Calculate spread
            spread_pct = ((ask_price - bid_price) / current_price) * 100 if current_price > 0 else 0
            
            # Get volatility metrics
            volatility_1h = await self._calculate_volatility(symbol, hours=1)
            volatility_24h = await self._calculate_volatility(symbol, hours=24)
            
            # Get volume and price change
            volume_24h = ticker.get('volume_24h', 0)
            price_change_24h_pct = ticker.get('price_change_24h_pct', 0)
            
            # Determine market condition
            market_condition = self._assess_market_condition(
                volatility_24h, spread_pct, volume_24h, price_change_24h_pct
            )
            
            return MarketPriceContext(
                symbol=symbol,
                current_price=current_price,
                bid_price=bid_price,
                ask_price=ask_price,
                spread_pct=spread_pct,
                volatility_1h=volatility_1h,
                volatility_24h=volatility_24h,
                volume_24h=volume_24h,
                price_change_24h_pct=price_change_24h_pct,
                market_condition=market_condition,
                last_trade_time=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get market context for {symbol}: {e}")
            raise
    
    async def _check_fat_finger_limits(self, order_price: float, side: str, 
                                     market_context: MarketPriceContext) -> Dict[str, Any]:
        """Check for fat-finger errors based on price deviation from market"""
        
        current_price = market_context.current_price
        price_deviation_pct = abs((order_price - current_price) / current_price) * 100
        
        # Adjust thresholds based on market conditions
        adjusted_threshold = self._adjust_fat_finger_threshold(market_context)
        
        errors = []
        warnings = []
        
        if price_deviation_pct > adjusted_threshold:
            if price_deviation_pct > adjusted_threshold * 2:
                errors.append(
                    f"Potential fat-finger error: order price {order_price} deviates "
                    f"{price_deviation_pct:.1f}% from market price {current_price} "
                    f"(threshold: {adjusted_threshold:.1f}%)"
                )
            else:
                warnings.append(
                    f"Large price deviation: {price_deviation_pct:.1f}% from market price "
                    f"(threshold: {adjusted_threshold:.1f}%)"
                )
        
        # Check if order would cross spread significantly
        if side.lower() == 'buy' and order_price > market_context.ask_price * 1.05:
            warnings.append(
                f"Buy order price {order_price} significantly above ask {market_context.ask_price}"
            )
        elif side.lower() == 'sell' and order_price < market_context.bid_price * 0.95:
            warnings.append(
                f"Sell order price {order_price} significantly below bid {market_context.bid_price}"
            )
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _validate_stop_loss_distance(self, sl_price: float, side: str,
                                         market_context: MarketPriceContext) -> Dict[str, Any]:
        """Validate stop-loss distance from current market price"""
        
        current_price = market_context.current_price
        
        if side.lower() == 'buy':
            # For buy orders, stop loss should be below current price
            sl_distance_pct = ((current_price - sl_price) / current_price) * 100
            expected_direction = "below"
        else:
            # For sell orders, stop loss should be above current price  
            sl_distance_pct = ((sl_price - current_price) / current_price) * 100
            expected_direction = "above"
        
        errors = []
        warnings = []
        
        # Adjust distance limits based on volatility
        adjusted_min_distance = self._adjust_sl_distance_for_volatility(
            self.min_sl_distance_pct, market_context.volatility_24h
        )
        
        if sl_distance_pct < 0:
            errors.append(
                f"Stop-loss price {sl_price} on wrong side of market price {current_price} "
                f"(should be {expected_direction} market price for {side} orders)"
            )
        elif sl_distance_pct < adjusted_min_distance:
            errors.append(
                f"Stop-loss too close to market price: {sl_distance_pct:.2f}% "
                f"(minimum: {adjusted_min_distance:.2f}%)"
            )
        elif sl_distance_pct > self.max_sl_distance_pct:
            warnings.append(
                f"Stop-loss very far from market price: {sl_distance_pct:.2f}% "
                f"(maximum recommended: {self.max_sl_distance_pct:.2f}%)"
            )
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _validate_take_profit_distance(self, tp_price: float, side: str,
                                           market_context: MarketPriceContext) -> Dict[str, Any]:
        """Validate take-profit distance from current market price"""
        
        current_price = market_context.current_price
        
        if side.lower() == 'buy':
            # For buy orders, take profit should be above current price
            tp_distance_pct = ((tp_price - current_price) / current_price) * 100
            expected_direction = "above"
        else:
            # For sell orders, take profit should be below current price
            tp_distance_pct = ((current_price - tp_price) / current_price) * 100
            expected_direction = "below"
        
        errors = []
        warnings = []
        
        if tp_distance_pct < 0:
            errors.append(
                f"Take-profit price {tp_price} on wrong side of market price {current_price} "
                f"(should be {expected_direction} market price for {side} orders)"
            )
        elif tp_distance_pct < self.min_tp_distance_pct:
            warnings.append(
                f"Take-profit very close to market price: {tp_distance_pct:.2f}% "
                f"(minimum recommended: {self.min_tp_distance_pct:.2f}%)"
            )
        elif tp_distance_pct > self.max_tp_distance_pct:
            warnings.append(
                f"Take-profit very far from market price: {tp_distance_pct:.2f}% "
                f"(maximum recommended: {self.max_tp_distance_pct:.2f}%)"
            )
        
        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _adjust_fat_finger_threshold(self, market_context: MarketPriceContext) -> float:
        """Adjust fat-finger threshold based on market conditions"""
        
        base_threshold = self.fat_finger_threshold_pct
        
        # Increase threshold during high volatility
        if market_context.volatility_24h > 0.05:  # >5% daily volatility
            base_threshold *= 1.5
        
        # Increase threshold for low liquidity markets
        if market_context.spread_pct > 1.0:  # >1% spread
            base_threshold *= 1.3
        
        # Increase threshold during gapping conditions
        if abs(market_context.price_change_24h_pct) > 10:
            base_threshold *= 1.2
        
        return min(base_threshold, 50.0)  # Cap at 50%
    
    def _calculate_suggested_price_range(self, side: str, 
                                       market_context: MarketPriceContext) -> Tuple[float, float]:
        """Calculate suggested price range for orders"""
        
        if side.lower() == 'buy':
            # For buy orders, suggest range around bid price
            center_price = market_context.bid_price
            lower_bound = center_price * 0.95
            upper_bound = center_price * 1.02
        else:
            # For sell orders, suggest range around ask price
            center_price = market_context.ask_price
            lower_bound = center_price * 0.98
            upper_bound = center_price * 1.05
        
        return (lower_bound, upper_bound)

class MarketValidationError(Exception):
    """Exception raised for market validation errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of market data service failures; fallback validation when real-time data unavailable; comprehensive error logging with context
- **Configuration:** Configurable thresholds per symbol and market condition; dynamic threshold adjustment parameters; override mechanisms for special situations
- **Testing:** Unit tests with various market conditions; integration tests with real market data; stress tests during volatile market periods; edge case validation
- **Dependencies:** Market data service for real-time prices; statistical analysis libraries; configuration management for dynamic thresholds

### 4. Acceptance Criteria
- [ ] Fat-finger detection prevents orders with excessive price deviations from market price
- [ ] Stop-loss distance validation ensures appropriate risk management placement
- [ ] Take-profit distance validation prevents unrealistic profit targets
- [ ] Dynamic threshold adjustment adapts to market volatility and liquidity conditions
- [ ] Market condition analysis provides context for validation decisions
- [ ] Suggested price ranges help users correct invalid orders
- [ ] Performance testing shows validation completion under 50ms per order
- [ ] Integration tests verify accuracy with real market data across various conditions
- [ ] Configuration allows fine-tuning of validation parameters per symbol
- [ ] TODO placeholder is completely replaced with production-ready implementation 