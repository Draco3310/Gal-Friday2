# Task: Implement signal approval handling, including portfolio checks and event publication.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1859`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing signal approval handling with portfolio checks and event publication.

### 2. Problem Statement
Without proper signal approval handling, approved trading signals cannot be effectively processed, validated against portfolio constraints, or communicated to execution systems. This creates a bottleneck in the trading pipeline and prevents proper coordination between risk management, portfolio management, and execution systems.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Signal Approval Framework:** Implement comprehensive approval workflow with multi-stage validation
2. **Build Portfolio Integration:** Check signal compatibility with current portfolio state and limits
3. **Implement Approval Event System:** Publish approval events for downstream consumption
4. **Add Position Sizing Logic:** Calculate optimal position sizes within risk constraints
5. **Create Approval Audit Trail:** Track all approval decisions for compliance and analysis
6. **Build Approval Monitoring:** Monitor approval rates and processing performance

#### b. Pseudocode or Implementation Sketch
```python
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"

class ApprovalCondition(str, Enum):
    REDUCE_POSITION_SIZE = "reduce_position_size"
    WAIT_FOR_MARKET_OPEN = "wait_for_market_open"
    REQUIRE_CONFIRMATION = "require_confirmation"
    SET_TIGHTER_STOP_LOSS = "set_tighter_stop_loss"

@dataclass
class SignalApprovalEvent:
    """Event published when a signal is approved"""
    signal_id: str
    symbol: str
    strategy_id: str
    approval_status: ApprovalStatus
    approved_position_size: float
    original_position_size: float
    approval_timestamp: datetime
    approval_conditions: List[ApprovalCondition]
    portfolio_impact: Dict[str, float]
    risk_adjustments: Dict[str, Any]
    execution_priority: int  # 1-10, higher is more urgent
    valid_until: datetime

@dataclass
class PortfolioConstraints:
    """Current portfolio constraints for signal approval"""
    max_position_size_per_symbol: Dict[str, float]
    max_total_exposure: float
    current_exposure: float
    available_capital: float
    correlation_limits: Dict[str, float]
    sector_limits: Dict[str, float]
    max_open_positions: int
    current_open_positions: int

class SignalApprovalHandler:
    """Enterprise-grade signal approval handling"""
    
    def __init__(self, config: Dict[str, Any], portfolio_manager, pubsub_manager, audit_service):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.pubsub = pubsub_manager
        self.audit = audit_service
        self.logger = logging.getLogger(__name__)
        
        # Approval statistics
        self.approval_stats = {
            'total_processed': 0,
            'approved': 0,
            'conditionally_approved': 0,
            'rejected': 0,
            'average_processing_time': 0.0
        }
    
    async def process_signal_approval(self, signal: Dict[str, Any]) -> SignalApprovalEvent:
        """
        Main entry point for signal approval handling
        Replace TODO with comprehensive approval logic
        """
        
        processing_start = datetime.now(timezone.utc)
        signal_id = signal.get('signal_id', 'unknown')
        
        try:
            self.logger.info(f"Processing signal approval for {signal_id}")
            
            # Get current portfolio constraints
            portfolio_constraints = await self._get_portfolio_constraints()
            
            # Perform comprehensive approval checks
            approval_result = await self._evaluate_signal_for_approval(signal, portfolio_constraints)
            
            # Calculate optimal position sizing
            approved_size = await self._calculate_approved_position_size(
                signal, approval_result, portfolio_constraints
            )
            
            # Create approval event
            approval_event = self._create_approval_event(
                signal, approval_result, approved_size, processing_start
            )
            
            # Log approval decision
            await self._log_approval_decision(approval_event)
            
            # Publish approval event
            await self._publish_approval_event(approval_event)
            
            # Update approval statistics
            self._update_approval_statistics(approval_event, processing_start)
            
            # Handle post-approval actions
            await self._handle_post_approval_actions(approval_event)
            
            self.logger.info(
                f"Signal {signal_id} approval processed: {approval_event.approval_status.value} "
                f"(size: {approved_size:.4f})"
            )
            
            return approval_event
            
        except Exception as e:
            self.logger.error(f"Error processing signal approval for {signal_id}: {e}")
            # Create error approval event
            return self._create_error_approval_event(signal, str(e))
    
    async def _get_portfolio_constraints(self) -> PortfolioConstraints:
        """Get current portfolio constraints and limits"""
        
        # Get current portfolio state
        portfolio_state = await self.portfolio_manager.get_portfolio_state()
        positions = await self.portfolio_manager.get_all_positions()
        account_info = await self.portfolio_manager.get_account_info()
        
        # Calculate current exposures
        current_exposure = sum(abs(pos.market_value) for pos in positions)
        available_capital = account_info.get('available_capital', 0)
        
        # Get position size limits per symbol
        max_position_sizes = {}
        for symbol in self.config.get('symbols', []):
            max_position_sizes[symbol] = self.config.get(
                f'limits.max_position_size.{symbol}',
                self.config.get('limits.max_position_size_default', 10000)
            )
        
        return PortfolioConstraints(
            max_position_size_per_symbol=max_position_sizes,
            max_total_exposure=self.config.get('limits.max_total_exposure', 100000),
            current_exposure=current_exposure,
            available_capital=available_capital,
            correlation_limits=self.config.get('limits.correlation', {}),
            sector_limits=self.config.get('limits.sector', {}),
            max_open_positions=self.config.get('limits.max_open_positions', 20),
            current_open_positions=len(positions)
        )
    
    async def _evaluate_signal_for_approval(self, signal: Dict[str, Any], 
                                          constraints: PortfolioConstraints) -> Dict[str, Any]:
        """Comprehensive signal evaluation for approval"""
        
        approval_checks = {
            'capital_check': await self._check_capital_availability(signal, constraints),
            'position_limit_check': await self._check_position_limits(signal, constraints),
            'exposure_check': await self._check_exposure_limits(signal, constraints),
            'correlation_check': await self._check_correlation_limits(signal, constraints),
            'sector_check': await self._check_sector_limits(signal, constraints),
            'market_condition_check': await self._check_market_conditions(signal),
            'risk_budget_check': await self._check_risk_budget(signal, constraints)
        }
        
        # Determine overall approval status
        approval_status = self._determine_approval_status(approval_checks)
        
        # Collect any required conditions
        approval_conditions = self._collect_approval_conditions(approval_checks)
        
        return {
            'status': approval_status,
            'conditions': approval_conditions,
            'checks': approval_checks,
            'constraints_used': constraints
        }
    
    async def _calculate_approved_position_size(self, signal: Dict[str, Any],
                                              approval_result: Dict[str, Any],
                                              constraints: PortfolioConstraints) -> float:
        """Calculate optimal approved position size within constraints"""
        
        requested_size = signal.get('position_size', 0)
        symbol = signal.get('symbol')
        
        # Start with requested size
        approved_size = abs(requested_size)
        
        # Apply position size limits
        max_symbol_size = constraints.max_position_size_per_symbol.get(symbol, float('inf'))
        approved_size = min(approved_size, max_symbol_size)
        
        # Apply capital constraints
        signal_price = signal.get('target_price', signal.get('current_price', 0))
        if signal_price > 0:
            max_affordable_size = constraints.available_capital / signal_price * 0.9  # 90% of available
            approved_size = min(approved_size, max_affordable_size)
        
        # Apply exposure constraints
        remaining_exposure_capacity = constraints.max_total_exposure - constraints.current_exposure
        if signal_price > 0:
            max_exposure_size = remaining_exposure_capacity / signal_price * 0.8  # 80% of remaining
            approved_size = min(approved_size, max_exposure_size)
        
        # Apply risk-based adjustments
        confidence = signal.get('confidence', 0.5)
        if confidence < 0.7:
            approved_size *= 0.5  # Reduce size for low confidence signals
        
        # Apply minimum size check
        min_size = self.config.get('limits.min_position_size', 100)
        if approved_size < min_size:
            approved_size = 0  # Below minimum, don't trade
        
        # Preserve original sign
        if requested_size < 0:
            approved_size = -approved_size
        
        return approved_size
    
    def _determine_approval_status(self, approval_checks: Dict[str, Any]) -> ApprovalStatus:
        """Determine overall approval status from individual checks"""
        
        # Check for hard rejections
        for check_name, result in approval_checks.items():
            if result.get('status') == 'rejected':
                return ApprovalStatus.REJECTED
        
        # Check for conditions
        has_conditions = any(
            result.get('conditions') for result in approval_checks.values()
        )
        
        if has_conditions:
            return ApprovalStatus.CONDITIONALLY_APPROVED
        
        # Check for warnings that require review
        has_warnings = any(
            result.get('warning') for result in approval_checks.values()
        )
        
        if has_warnings:
            return ApprovalStatus.PENDING_REVIEW
        
        return ApprovalStatus.APPROVED
    
    def _create_approval_event(self, signal: Dict[str, Any], approval_result: Dict[str, Any],
                              approved_size: float, processing_start: datetime) -> SignalApprovalEvent:
        """Create comprehensive approval event"""
        
        # Calculate portfolio impact
        signal_value = abs(approved_size * signal.get('target_price', 0))
        portfolio_impact = {
            'position_value': signal_value,
            'exposure_change': signal_value,
            'capital_utilization': signal_value / approval_result['constraints_used'].available_capital
        }
        
        return SignalApprovalEvent(
            signal_id=signal.get('signal_id', 'unknown'),
            symbol=signal.get('symbol', 'unknown'),
            strategy_id=signal.get('strategy_id', 'unknown'),
            approval_status=approval_result['status'],
            approved_position_size=approved_size,
            original_position_size=signal.get('position_size', 0),
            approval_timestamp=datetime.now(timezone.utc),
            approval_conditions=approval_result['conditions'],
            portfolio_impact=portfolio_impact,
            risk_adjustments=signal.get('risk_adjustments', {}),
            execution_priority=self._calculate_execution_priority(signal, approval_result),
            valid_until=datetime.now(timezone.utc) + timedelta(hours=1)  # 1 hour validity
        )
    
    async def _publish_approval_event(self, approval_event: SignalApprovalEvent) -> None:
        """Publish approval event to interested subscribers"""
        
        event_data = {
            'type': 'SignalApproved',
            'signal_id': approval_event.signal_id,
            'symbol': approval_event.symbol,
            'strategy_id': approval_event.strategy_id,
            'approval_status': approval_event.approval_status.value,
            'approved_position_size': approval_event.approved_position_size,
            'original_position_size': approval_event.original_position_size,
            'approval_timestamp': approval_event.approval_timestamp.isoformat(),
            'approval_conditions': [condition.value for condition in approval_event.approval_conditions],
            'portfolio_impact': approval_event.portfolio_impact,
            'execution_priority': approval_event.execution_priority,
            'valid_until': approval_event.valid_until.isoformat()
        }
        
        # Publish to general approval topic
        await self.pubsub.publish('signals.approved', event_data)
        
        # Publish to execution service if approved
        if approval_event.approval_status in [ApprovalStatus.APPROVED, ApprovalStatus.CONDITIONALLY_APPROVED]:
            await self.pubsub.publish('execution.signals.approved', event_data)
        
        # Publish to portfolio manager for position tracking
        await self.pubsub.publish('portfolio.signals.approved', event_data)
        
        # Publish to strategy-specific topic
        await self.pubsub.publish(f'signals.approved.{approval_event.strategy_id}', event_data)
    
    async def _check_capital_availability(self, signal: Dict[str, Any], 
                                        constraints: PortfolioConstraints) -> Dict[str, Any]:
        """Check if sufficient capital is available for the signal"""
        
        requested_value = abs(signal.get('position_size', 0) * signal.get('target_price', 0))
        available_capital = constraints.available_capital
        
        if requested_value > available_capital:
            return {
                'status': 'rejected',
                'reason': f'Insufficient capital: need {requested_value}, have {available_capital}',
                'conditions': []
            }
        
        # Warning if using more than 80% of available capital
        if requested_value > available_capital * 0.8:
            return {
                'status': 'warning',
                'reason': f'High capital utilization: {requested_value/available_capital:.1%}',
                'conditions': [ApprovalCondition.REDUCE_POSITION_SIZE]
            }
        
        return {'status': 'passed', 'conditions': []}
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of portfolio service failures; fallback approval logic when constraints cannot be retrieved; comprehensive error logging
- **Configuration:** Configurable approval thresholds and limits; dynamic constraint updates; strategy-specific approval rules
- **Testing:** Unit tests for approval logic; integration tests with portfolio manager; stress tests for high-volume approval processing
- **Dependencies:** Integration with PortfolioManager for constraints; PubSub for event publishing; audit service for approval trail

### 4. Acceptance Criteria
- [ ] Signal approval workflow is implemented with comprehensive portfolio checks
- [ ] Position sizing logic respects all portfolio constraints and risk limits
- [ ] Approval events are published with complete information for downstream services
- [ ] Conditional approvals include specific conditions and requirements
- [ ] Approval statistics are tracked for monitoring and optimization
- [ ] Error handling ensures system stability during portfolio service failures
- [ ] Performance testing shows approval processing under 100ms per signal
- [ ] Integration tests verify end-to-end approval workflow
- [ ] TODO placeholder is completely replaced with production-ready implementation 