# Task: Implement position sizing and portfolio limit checks prior to order placement.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1907`
- **Keyword/Pattern:** `TODO` 
- **Current State:** The code contains a TODO placeholder for implementing position sizing logic and portfolio limit checks before order placement.

### 2. Problem Statement
Without proper position sizing and portfolio limit checks, the system cannot ensure that new orders comply with risk management rules, portfolio diversification requirements, and regulatory limits. This creates significant risk exposure through oversized positions, excessive concentration, and potential margin violations.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Position Sizing Engine:** Implement sophisticated position sizing algorithms based on risk budgets
2. **Build Portfolio Limit Framework:** Comprehensive checks for exposure, concentration, and correlation limits
3. **Implement Risk-Based Sizing:** Dynamic position sizing based on volatility, confidence, and market conditions
4. **Add Regulatory Compliance:** Ensure position sizes comply with regulatory requirements and internal policies
5. **Create Real-Time Monitoring:** Continuous monitoring of portfolio limits with proactive alerts
6. **Build Optimization Logic:** Optimize position sizes for risk-adjusted returns

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timezone
import logging

class PositionSizingMethod(str, Enum):
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE_OF_PORTFOLIO = "percentage_of_portfolio"
    RISK_BASED = "risk_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class PositionSizingResult:
    """Result of position sizing calculation"""
    recommended_size: float
    max_allowed_size: float
    sizing_method: PositionSizingMethod
    risk_metrics: Dict[str, float]
    limit_checks: Dict[str, bool]
    warnings: List[str]
    size_adjustments: List[str]

class PositionSizingEngine:
    """Enterprise-grade position sizing and portfolio limit checking"""
    
    def __init__(self, portfolio_manager, market_data_service, config: Dict[str, Any]):
        self.portfolio_manager = portfolio_manager
        self.market_data = market_data_service
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def calculate_position_size_and_check_limits(self, signal: Dict[str, Any]) -> PositionSizingResult:
        """
        Calculate optimal position size and verify portfolio limits
        Replace TODO with comprehensive position sizing logic
        """
        
        symbol = signal.get('symbol')
        strategy_id = signal.get('strategy_id')
        confidence = signal.get('confidence', 0.5)
        target_price = signal.get('target_price')
        
        try:
            self.logger.info(f"Calculating position size for {symbol} signal from {strategy_id}")
            
            # Get current portfolio state
            portfolio_state = await self._get_current_portfolio_state()
            
            # Get market data for the symbol
            market_data = await self._get_symbol_market_data(symbol)
            
            # Calculate base position size using configured method
            sizing_method = self._get_sizing_method(strategy_id)
            base_size = await self._calculate_base_position_size(
                signal, portfolio_state, market_data, sizing_method
            )
            
            # Apply risk-based adjustments
            risk_adjusted_size = await self._apply_risk_adjustments(
                base_size, signal, market_data, portfolio_state
            )
            
            # Check portfolio limits and constraints
            limit_check_result = await self._check_portfolio_limits(
                symbol, risk_adjusted_size, target_price, portfolio_state
            )
            
            # Apply limit constraints
            final_size = await self._apply_limit_constraints(
                risk_adjusted_size, limit_check_result
            )
            
            # Calculate risk metrics for the final position
            risk_metrics = await self._calculate_position_risk_metrics(
                symbol, final_size, target_price, portfolio_state
            )
            
            # Generate sizing result
            result = PositionSizingResult(
                recommended_size=final_size,
                max_allowed_size=limit_check_result['max_allowed_size'],
                sizing_method=sizing_method,
                risk_metrics=risk_metrics,
                limit_checks=limit_check_result['checks'],
                warnings=limit_check_result['warnings'],
                size_adjustments=self._generate_size_adjustment_log(
                    base_size, risk_adjusted_size, final_size
                )
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in position sizing for {symbol}: {e}")
            raise PositionSizingError(f"Position sizing failed: {e}")
    
    async def _calculate_base_position_size(self, signal: Dict[str, Any],
                                          portfolio_state: Dict[str, Any],
                                          market_data: Dict[str, Any],
                                          method: PositionSizingMethod) -> float:
        """Calculate base position size using specified method"""
        
        target_price = signal.get('target_price', market_data['current_price'])
        total_equity = portfolio_state['total_equity']
        
        if method == PositionSizingMethod.RISK_BASED:
            # Risk-based sizing using risk budget
            risk_budget_pct = self.config.get('risk_budget_per_trade', 0.02)  # 2% default
            risk_budget_usd = total_equity * risk_budget_pct
            
            # Estimate potential loss
            stop_loss_price = signal.get('stop_loss_price')
            if stop_loss_price:
                potential_loss_per_unit = abs(target_price - stop_loss_price)
            else:
                volatility = market_data.get('volatility_daily', 0.02)
                potential_loss_per_unit = target_price * volatility * 2
            
            if potential_loss_per_unit > 0:
                return risk_budget_usd / potential_loss_per_unit
        
        elif method == PositionSizingMethod.PERCENTAGE_OF_PORTFOLIO:
            # Fixed percentage of total portfolio
            percentage = self.config.get('position_percentage', 0.05)  # 5% default
            position_value = total_equity * percentage
            return position_value / target_price
        
        # Default fallback
        return (total_equity * 0.05) / target_price
    
    async def _check_portfolio_limits(self, symbol: str, position_size: float,
                                    target_price: float, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check position against all portfolio limits"""
        
        position_value = position_size * target_price
        total_equity = portfolio_state['total_equity']
        current_positions = portfolio_state['positions']
        
        checks = {}
        warnings = []
        max_allowed_size = position_size
        
        # Position size limit (5% of portfolio max)
        max_position_pct = self.config.get('max_position_percentage', 0.05)
        max_position_value = total_equity * max_position_pct
        
        if position_value > max_position_value:
            checks['position_percentage'] = False
            max_allowed_size = min(max_allowed_size, max_position_value / target_price)
            warnings.append(f"Position exceeds {max_position_pct*100}% limit")
        else:
            checks['position_percentage'] = True
        
        # Maximum number of positions
        max_positions = self.config.get('max_open_positions', 20)
        if len(current_positions) >= max_positions and symbol not in current_positions:
            checks['max_positions'] = False
            max_allowed_size = 0
            warnings.append(f"Already at maximum {max_positions} positions")
        else:
            checks['max_positions'] = True
        
        return {
            'checks': checks,
            'warnings': warnings,
            'max_allowed_size': max_allowed_size,
            'all_passed': all(checks.values())
        }

class PositionSizingError(Exception):
    """Exception raised for position sizing errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of portfolio service failures; fallback sizing methods when advanced calculations fail; comprehensive error logging
- **Configuration:** Configurable sizing methods per strategy; dynamic risk budget adjustments; environment-specific limit overrides  
- **Testing:** Unit tests for sizing algorithms; integration tests with portfolio manager; stress tests with various market conditions
- **Dependencies:** Portfolio manager for current state; market data service for volatility and correlations; configuration management for limits

### 4. Acceptance Criteria
- [ ] Position sizing algorithms calculate appropriate sizes based on risk budgets and confidence levels
- [ ] Portfolio limit checks prevent excessive concentration and leverage exposure
- [ ] Risk-based adjustments account for volatility, correlation, and market conditions
- [ ] Real-time limit monitoring prevents limit breaches during volatile markets
- [ ] Performance optimization shows sizing calculations under 100ms per signal  
- [ ] Integration tests verify accurate portfolio state tracking and limit enforcement
- [ ] Configuration allows fine-tuning of sizing parameters per strategy and market condition
- [ ] Comprehensive logging provides audit trail for all sizing decisions
- [ ] TODO placeholder is completely replaced with production-ready implementation
