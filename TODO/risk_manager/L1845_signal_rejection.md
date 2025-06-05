# Task: Create full signal rejection workflow: validate, log, publish rejection events.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1845`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing a complete signal rejection workflow.

### 2. Problem Statement
Without a proper signal rejection workflow, the system cannot effectively filter out risky or invalid trading signals, potentially leading to harmful trades being executed. The absence of validation, logging, and event publication for rejected signals creates blind spots in risk management and makes it difficult to analyze why signals were rejected or optimize the rejection criteria.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Signal Validation Framework:** Implement comprehensive signal validation with configurable rules
2. **Build Rejection Decision Engine:** Create logic to determine rejection reasons and severity
3. **Implement Event Publishing:** Publish rejection events for monitoring and analysis
4. **Add Audit Trail:** Comprehensive logging of all rejection decisions
5. **Create Feedback Loop:** Allow rejection analysis to improve signal quality
6. **Add Performance Monitoring:** Track rejection rates and patterns

#### b. Pseudocode or Implementation Sketch
```python
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

class RejectionReason(str, Enum):
    INSUFFICIENT_CONFIDENCE = "insufficient_confidence"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    RISK_THRESHOLD_BREACH = "risk_threshold_breach"
    MARKET_CONDITION_INVALID = "market_condition_invalid"
    SIGNAL_QUALITY_POOR = "signal_quality_poor"
    CORRELATION_TOO_HIGH = "correlation_too_high"
    VOLATILITY_TOO_HIGH = "volatility_too_high"
    LIQUIDITY_INSUFFICIENT = "liquidity_insufficient"
    BLACKOUT_PERIOD = "blackout_period"
    TECHNICAL_ERROR = "technical_error"

class RejectionSeverity(str, Enum):
    LOW = "low"       # Signal quality issue
    MEDIUM = "medium" # Risk threshold breach
    HIGH = "high"     # Position limit exceeded
    CRITICAL = "critical" # System safety issue

@dataclass
class SignalRejectionEvent:
    """Event published when a signal is rejected"""
    signal_id: str
    symbol: str
    strategy_id: str
    rejection_reason: RejectionReason
    severity: RejectionSeverity
    rejection_timestamp: datetime
    signal_data: Dict[str, Any]
    risk_metrics: Dict[str, float]
    rejection_details: str
    auto_retry_eligible: bool = False

class SignalRejectionWorkflow:
    """Complete signal rejection workflow implementation"""
    
    def __init__(self, config: Dict[str, Any], pubsub_manager, audit_service):
        self.config = config
        self.pubsub = pubsub_manager
        self.audit = audit_service
        self.logger = logging.getLogger(__name__)
        
        # Rejection statistics
        self.rejection_stats = {
            'total_rejections': 0,
            'rejections_by_reason': {},
            'rejections_by_symbol': {},
            'rejection_rate_trend': []
        }
        
        # Load rejection thresholds from configuration
        self.rejection_thresholds = self._load_rejection_thresholds()
    
    async def evaluate_and_reject_signal(self, signal: Dict[str, Any]) -> Optional[SignalRejectionEvent]:
        """
        Main entry point for signal rejection workflow
        Replace TODO with comprehensive rejection logic
        """
        
        try:
            signal_id = signal.get('signal_id', 'unknown')
            self.logger.debug(f"Evaluating signal {signal_id} for potential rejection")
            
            # Perform comprehensive signal validation
            validation_results = await self._validate_signal_comprehensive(signal)
            
            # Determine if signal should be rejected
            rejection_decision = self._make_rejection_decision(validation_results)
            
            if rejection_decision:
                # Create rejection event
                rejection_event = self._create_rejection_event(signal, rejection_decision)
                
                # Log rejection decision
                await self._log_rejection_decision(rejection_event)
                
                # Publish rejection event
                await self._publish_rejection_event(rejection_event)
                
                # Update rejection statistics
                self._update_rejection_statistics(rejection_event)
                
                # Check if auto-retry should be scheduled
                await self._handle_auto_retry_logic(rejection_event)
                
                self.logger.info(
                    f"Signal {signal_id} rejected: {rejection_decision['reason']} "
                    f"(severity: {rejection_decision['severity']})"
                )
                
                return rejection_event
            
            return None  # Signal not rejected
            
        except Exception as e:
            self.logger.error(f"Error in signal rejection workflow for signal {signal_id}: {e}")
            # Create technical error rejection
            return await self._create_technical_error_rejection(signal, str(e))
    
    async def _validate_signal_comprehensive(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive signal validation across multiple dimensions"""
        
        validation_results = {
            'confidence_check': await self._validate_signal_confidence(signal),
            'position_limit_check': await self._validate_position_limits(signal),
            'risk_threshold_check': await self._validate_risk_thresholds(signal),
            'market_condition_check': await self._validate_market_conditions(signal),
            'correlation_check': await self._validate_correlation_limits(signal),
            'liquidity_check': await self._validate_liquidity_requirements(signal),
            'blackout_check': await self._validate_blackout_periods(signal),
            'technical_check': await self._validate_technical_requirements(signal)
        }
        
        return validation_results
    
    def _make_rejection_decision(self, validation_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze validation results and make rejection decision"""
        
        # Check each validation result in order of severity
        for check_name, result in validation_results.items():
            if not result['passed']:
                return {
                    'reason': result['rejection_reason'],
                    'severity': result['severity'],
                    'details': result['details'],
                    'validation_check': check_name,
                    'auto_retry_eligible': result.get('auto_retry_eligible', False)
                }
        
        return None  # All validations passed
    
    def _create_rejection_event(self, signal: Dict[str, Any], 
                               rejection_decision: Dict[str, Any]) -> SignalRejectionEvent:
        """Create comprehensive rejection event"""
        
        return SignalRejectionEvent(
            signal_id=signal.get('signal_id', 'unknown'),
            symbol=signal.get('symbol', 'unknown'),
            strategy_id=signal.get('strategy_id', 'unknown'),
            rejection_reason=RejectionReason(rejection_decision['reason']),
            severity=RejectionSeverity(rejection_decision['severity']),
            rejection_timestamp=datetime.now(timezone.utc),
            signal_data={
                'confidence': signal.get('confidence'),
                'prediction': signal.get('prediction'),
                'position_size': signal.get('position_size'),
                'target_price': signal.get('target_price')
            },
            risk_metrics=signal.get('risk_metrics', {}),
            rejection_details=rejection_decision['details'],
            auto_retry_eligible=rejection_decision.get('auto_retry_eligible', False)
        )
    
    async def _log_rejection_decision(self, rejection_event: SignalRejectionEvent) -> None:
        """Comprehensive logging of rejection decision"""
        
        # Structured logging for analysis
        log_data = {
            'event_type': 'signal_rejection',
            'signal_id': rejection_event.signal_id,
            'symbol': rejection_event.symbol,
            'strategy_id': rejection_event.strategy_id,
            'rejection_reason': rejection_event.rejection_reason.value,
            'severity': rejection_event.severity.value,
            'timestamp': rejection_event.rejection_timestamp.isoformat(),
            'signal_confidence': rejection_event.signal_data.get('confidence'),
            'auto_retry_eligible': rejection_event.auto_retry_eligible
        }
        
        # Log at appropriate level based on severity
        if rejection_event.severity == RejectionSeverity.CRITICAL:
            self.logger.error(f"CRITICAL signal rejection: {log_data}")
        elif rejection_event.severity == RejectionSeverity.HIGH:
            self.logger.warning(f"HIGH severity signal rejection: {log_data}")
        else:
            self.logger.info(f"Signal rejection: {log_data}")
        
        # Store in audit trail
        await self.audit.record_rejection_event(rejection_event)
    
    async def _publish_rejection_event(self, rejection_event: SignalRejectionEvent) -> None:
        """Publish rejection event to interested subscribers"""
        
        event_data = {
            'type': 'SignalRejected',
            'signal_id': rejection_event.signal_id,
            'symbol': rejection_event.symbol,
            'strategy_id': rejection_event.strategy_id,
            'rejection_reason': rejection_event.rejection_reason.value,
            'severity': rejection_event.severity.value,
            'timestamp': rejection_event.rejection_timestamp.isoformat(),
            'rejection_details': rejection_event.rejection_details,
            'auto_retry_eligible': rejection_event.auto_retry_eligible,
            'risk_metrics': rejection_event.risk_metrics
        }
        
        # Publish to general rejection topic
        await self.pubsub.publish('signals.rejected', event_data)
        
        # Publish to severity-specific topic for urgent attention
        if rejection_event.severity in [RejectionSeverity.HIGH, RejectionSeverity.CRITICAL]:
            await self.pubsub.publish(f'signals.rejected.{rejection_event.severity.value}', event_data)
        
        # Publish to strategy-specific topic for strategy optimization
        await self.pubsub.publish(f'signals.rejected.{rejection_event.strategy_id}', event_data)
    
    async def _validate_signal_confidence(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate signal confidence against thresholds"""
        
        confidence = signal.get('confidence', 0.0)
        min_confidence = self.rejection_thresholds.get('min_confidence', 0.6)
        
        if confidence < min_confidence:
            return {
                'passed': False,
                'rejection_reason': RejectionReason.INSUFFICIENT_CONFIDENCE,
                'severity': RejectionSeverity.LOW,
                'details': f"Confidence {confidence:.3f} below minimum {min_confidence:.3f}",
                'auto_retry_eligible': False
            }
        
        return {'passed': True}
    
    async def _validate_position_limits(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate against position size limits"""
        
        symbol = signal.get('symbol')
        requested_size = signal.get('position_size', 0)
        
        # Check maximum position size per symbol
        max_position = self.rejection_thresholds.get(f'max_position.{symbol}', 
                                                   self.rejection_thresholds.get('max_position_default', 10000))
        
        if abs(requested_size) > max_position:
            return {
                'passed': False,
                'rejection_reason': RejectionReason.POSITION_LIMIT_EXCEEDED,
                'severity': RejectionSeverity.HIGH,
                'details': f"Requested size {requested_size} exceeds limit {max_position}",
                'auto_retry_eligible': False
            }
        
        return {'passed': True}
    
    def _update_rejection_statistics(self, rejection_event: SignalRejectionEvent) -> None:
        """Update rejection statistics for monitoring"""
        
        self.rejection_stats['total_rejections'] += 1
        
        # Update by reason
        reason = rejection_event.rejection_reason.value
        self.rejection_stats['rejections_by_reason'][reason] = \
            self.rejection_stats['rejections_by_reason'].get(reason, 0) + 1
        
        # Update by symbol
        symbol = rejection_event.symbol
        self.rejection_stats['rejections_by_symbol'][symbol] = \
            self.rejection_stats['rejections_by_symbol'].get(symbol, 0) + 1
        
        # Update trend (keep last 100 rejections)
        self.rejection_stats['rejection_rate_trend'].append({
            'timestamp': rejection_event.rejection_timestamp,
            'reason': reason,
            'severity': rejection_event.severity.value
        })
        
        if len(self.rejection_stats['rejection_rate_trend']) > 100:
            self.rejection_stats['rejection_rate_trend'].pop(0)
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Comprehensive error handling in validation logic; graceful degradation when validation services are unavailable; fallback rejection for technical errors
- **Configuration:** Configurable rejection thresholds per symbol, strategy, and risk type; runtime threshold updates; environment-specific rejection criteria
- **Testing:** Unit tests for each validation check; integration tests for complete rejection workflow; stress tests for high-volume signal processing
- **Dependencies:** Integration with PubSub for event publishing; audit service for rejection trail; configuration management for dynamic thresholds

### 4. Acceptance Criteria
- [ ] Complete signal rejection workflow is implemented with all validation checks
- [ ] Rejection events are published to appropriate topics with comprehensive data
- [ ] Audit trail captures all rejection decisions with sufficient detail for analysis
- [ ] Rejection statistics are tracked and available for monitoring and optimization
- [ ] Auto-retry logic handles temporary rejection conditions appropriately
- [ ] Configuration allows dynamic adjustment of rejection thresholds without code changes
- [ ] Performance testing shows acceptable rejection processing latency (<50ms per signal)
- [ ] Integration tests verify end-to-end rejection workflow with real signal data
- [ ] TODO placeholder is completely replaced with production-ready implementation 