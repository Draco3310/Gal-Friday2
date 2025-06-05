# Task: Remove placeholder log message and build logic to parse execution reports, updating loss counters and publishing events.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1816`
- **Keyword/Pattern:** `"TODO" message`
- **Current State:** The code contains a TODO placeholder log message without actual logic to parse execution reports and update loss counters.

### 2. Problem Statement
The placeholder log message indicates unfinished functionality in the execution report processing pipeline. Without proper parsing and loss counter logic, the risk management system cannot fulfill its core responsibility of tracking trading performance and preventing excessive losses. This represents a critical gap in the system's ability to protect against catastrophic trading scenarios.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Remove Placeholder Code:** Replace TODO log message with functional execution report parsing
2. **Implement Report Parser:** Create robust parsing logic for different execution report formats
3. **Build Loss Counter System:** Implement persistent loss tracking with configurable reset conditions
4. **Add Event Publishing:** Integrate with pub/sub system for real-time risk notifications
5. **Create Audit Trail:** Ensure all risk decisions are logged for compliance and debugging
6. **Add Performance Monitoring:** Track processing latency and throughput

#### b. Pseudocode or Implementation Sketch
```python
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timezone
import json
from pydantic import ValidationError

class ExecutionReportProcessor:
    """Handles parsing and processing of execution reports"""
    
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        self.logger = risk_manager.logger
        self.loss_counters: Dict[str, int] = {}  # symbol -> consecutive losses
        self.last_reset = datetime.now(timezone.utc)
        
    async def process_execution_report(self, raw_report: Dict[str, Any]) -> None:
        """
        Parse and process execution report, replacing the TODO placeholder
        """
        try:
            # Replace: self.logger.info("TODO: Process execution report")
            self.logger.info(f"Processing execution report for order {raw_report.get('order_id', 'unknown')}")
            
            # Parse and validate the report
            execution_report = self._parse_execution_report(raw_report)
            if not execution_report:
                return
            
            # Update loss counters based on report
            await self._update_loss_counters(execution_report)
            
            # Check for risk threshold breaches
            risk_events = await self._evaluate_risk_conditions(execution_report)
            
            # Publish risk events if any
            if risk_events:
                await self._publish_risk_events(risk_events, execution_report)
            
            # Update audit trail
            await self._update_audit_trail(execution_report, risk_events)
            
            self.logger.info(
                f"Processed execution report {execution_report.order_id}: "
                f"PnL={execution_report.realized_pnl}, "
                f"Consecutive losses for {execution_report.symbol}="
                f"{self.loss_counters.get(execution_report.symbol, 0)}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process execution report: {e}", exc_info=True)
            # Publish error event for monitoring
            await self._publish_error_event(raw_report, str(e))
    
    def _parse_execution_report(self, raw_report: Dict[str, Any]) -> Optional[ExecutionReport]:
        """Parse raw execution report into validated model"""
        try:
            # Handle different report formats (Kraken, simulated, etc.)
            normalized_report = self._normalize_report_format(raw_report)
            
            # Validate using Pydantic model
            execution_report = ExecutionReport(**normalized_report)
            
            return execution_report
            
        except ValidationError as e:
            self.logger.warning(f"Invalid execution report format: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing execution report: {e}")
            return None
    
    def _normalize_report_format(self, raw_report: Dict[str, Any]) -> Dict[str, Any]:
        """Convert various report formats to standard format"""
        
        # Handle Kraken format
        if 'txid' in raw_report:
            return self._normalize_kraken_format(raw_report)
        
        # Handle simulated format
        elif 'simulation_id' in raw_report:
            return self._normalize_simulated_format(raw_report)
        
        # Assume standard format
        else:
            return raw_report
    
    async def _update_loss_counters(self, report: ExecutionReport) -> None:
        """Update consecutive loss counters per symbol"""
        
        # Only track completed fills
        if report.status != OrderStatus.FILLED:
            return
        
        # Only track trades with realized PnL
        if report.realized_pnl is None:
            return
        
        symbol = report.symbol
        
        if report.realized_pnl < 0:
            # Increment loss counter
            self.loss_counters[symbol] = self.loss_counters.get(symbol, 0) + 1
            self.logger.info(
                f"Loss recorded for {symbol}: consecutive losses now "
                f"{self.loss_counters[symbol]} (PnL: {report.realized_pnl})"
            )
        else:
            # Reset counter on profit
            if symbol in self.loss_counters:
                prev_losses = self.loss_counters[symbol]
                self.loss_counters[symbol] = 0
                self.logger.info(
                    f"Profit recorded for {symbol}: reset consecutive losses "
                    f"from {prev_losses} to 0 (PnL: {report.realized_pnl})"
                )
    
    async def _evaluate_risk_conditions(self, report: ExecutionReport) -> List[str]:
        """Evaluate risk conditions and return list of triggered events"""
        risk_events = []
        
        symbol = report.symbol
        consecutive_losses = self.loss_counters.get(symbol, 0)
        
        # Check consecutive loss limits
        max_losses = self.risk_manager.config.get('risk.max_consecutive_losses', 5)
        if consecutive_losses >= max_losses:
            risk_events.append('consecutive_loss_limit_reached')
        
        # Check daily loss limits
        daily_loss = await self._calculate_daily_loss(symbol)
        max_daily_loss = self.risk_manager.config.get('risk.max_daily_loss', 1000.0)
        if daily_loss >= max_daily_loss:
            risk_events.append('daily_loss_limit_reached')
        
        return risk_events
    
    async def _publish_risk_events(self, risk_events: List[str], report: ExecutionReport) -> None:
        """Publish risk events to interested subscribers"""
        for event_type in risk_events:
            event_data = {
                'type': event_type,
                'symbol': report.symbol,
                'order_id': report.order_id,
                'consecutive_losses': self.loss_counters.get(report.symbol, 0),
                'realized_pnl': report.realized_pnl,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action_required': True
            }
            
            await self.risk_manager.pubsub.publish(f'risk.{event_type}', event_data)
            
            self.logger.warning(f"Risk event published: {event_type} for {report.symbol}")
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of malformed execution reports; continue processing other reports if one fails; comprehensive logging for debugging
- **Configuration:** Configurable loss counter thresholds; different limits per symbol or strategy; runtime configuration updates
- **Testing:** Unit tests for report parsing with various formats; integration tests for event publishing; stress tests with high-volume reports
- **Dependencies:** Integration with pub/sub system for event publishing; database connection for audit trail persistence; configuration service for dynamic thresholds

### 4. Acceptance Criteria
- [ ] TODO placeholder log message is completely removed and replaced with functional code
- [ ] Execution report parsing handles multiple formats (Kraken, simulated, standard) correctly
- [ ] Loss counters are accurately maintained per symbol with proper reset logic
- [ ] Risk events are published when thresholds are breached
- [ ] All execution report processing is logged with appropriate detail levels
- [ ] Error scenarios are handled gracefully without stopping the risk manager
- [ ] Audit trail captures all risk decisions for compliance and debugging
- [ ] Performance metrics show acceptable processing latency (<100ms per report)
- [ ] Integration tests verify end-to-end execution report flow 