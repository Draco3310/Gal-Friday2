# Task: Implement execution report handling for tracking consecutive losses, updating risk metrics accordingly and persisting state.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1814`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder without logic to handle execution reports for tracking consecutive losses and updating risk metrics.

### 2. Problem Statement
Without proper execution report handling, the RiskManager cannot track consecutive losses, monitor drawdown patterns, or maintain accurate risk metrics in real-time. This creates a critical blind spot in the risk management system, potentially allowing positions to accumulate losses beyond acceptable thresholds. The inability to persist risk state also means that risk metrics reset on system restart, losing crucial historical context for risk assessment.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Define ExecutionReport Data Model:** Create Pydantic model for standardized execution report structure
2. **Implement Risk Metrics Tracking:** Create persistent storage for consecutive losses, drawdown, and other risk metrics
3. **Build Event Processing Pipeline:** Integrate execution report processing into the main event loop
4. **Add Risk Threshold Monitoring:** Implement configurable thresholds with automated responses
5. **Create Risk State Persistence:** Ensure risk metrics survive system restarts
6. **Implement Risk Event Broadcasting:** Publish risk metric updates to interested services

#### b. Pseudocode or Implementation Sketch
```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

class OrderStatus(str, Enum):
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class ExecutionReport(BaseModel):
    """Standardized execution report model"""
    order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    filled_quantity: float = Field(ge=0)
    average_price: float = Field(gt=0)
    commission: float = Field(ge=0)
    realized_pnl: Optional[float] = None
    timestamp: datetime
    signal_id: Optional[str] = None
    strategy_id: Optional[str] = None

@dataclass
class RiskMetrics:
    """Current risk state"""
    consecutive_losses: int = 0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    active_positions_count: int = 0
    last_updated: datetime = None

class RiskManager:
    def __init__(self, config: dict, pubsub_manager, persistence_service):
        self.config = config
        self.pubsub = pubsub_manager
        self.persistence = persistence_service
        self.risk_metrics = RiskMetrics()
        self.max_consecutive_losses = config.get('risk.max_consecutive_losses', 5)
        self.max_drawdown_threshold = config.get('risk.max_drawdown_threshold', 0.05)
        
    async def _handle_execution_report(self, report: ExecutionReport) -> None:
        """Process execution report and update risk metrics"""
        try:
            # Validate report
            if report.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                return
            
            # Update risk metrics based on realized PnL
            if report.realized_pnl is not None:
                await self._update_pnl_metrics(report.realized_pnl, report)
                
            # Update position count
            await self._update_position_metrics(report)
            
            # Check risk thresholds
            await self._check_risk_thresholds(report)
            
            # Persist updated metrics
            await self._persist_risk_state()
            
            # Publish risk metrics update event
            await self._publish_risk_metrics_update(report)
            
        except Exception as e:
            self.logger.error(f"Error processing execution report {report.order_id}: {e}")
            # Don't re-raise to avoid blocking other event processing
    
    async def _update_pnl_metrics(self, realized_pnl: float, report: ExecutionReport) -> None:
        """Update PnL-based risk metrics"""
        self.risk_metrics.total_realized_pnl += realized_pnl
        self.risk_metrics.daily_pnl += realized_pnl
        
        # Track consecutive losses
        if realized_pnl < 0:
            self.risk_metrics.consecutive_losses += 1
        else:
            self.risk_metrics.consecutive_losses = 0
        
        # Update drawdown metrics
        if realized_pnl < 0:
            self.risk_metrics.current_drawdown += abs(realized_pnl)
            self.risk_metrics.max_drawdown = max(
                self.risk_metrics.max_drawdown, 
                self.risk_metrics.current_drawdown
            )
        elif realized_pnl > 0:
            # Reduce drawdown with profits
            self.risk_metrics.current_drawdown = max(
                0, self.risk_metrics.current_drawdown - realized_pnl
            )
        
        self.risk_metrics.last_updated = datetime.now(timezone.utc)
    
    async def _check_risk_thresholds(self, report: ExecutionReport) -> None:
        """Check if risk thresholds are breached and take action"""
        actions_taken = []
        
        # Check consecutive losses
        if self.risk_metrics.consecutive_losses >= self.max_consecutive_losses:
            await self._handle_consecutive_loss_breach(report)
            actions_taken.append("consecutive_loss_limit")
        
        # Check drawdown threshold
        if self.risk_metrics.current_drawdown >= self.max_drawdown_threshold:
            await self._handle_drawdown_breach(report)
            actions_taken.append("drawdown_limit")
        
        if actions_taken:
            self.logger.warning(f"Risk thresholds breached: {actions_taken}")
    
    async def _persist_risk_state(self) -> None:
        """Persist current risk metrics to database"""
        await self.persistence.save_risk_metrics(self.risk_metrics)
    
    async def _publish_risk_metrics_update(self, report: ExecutionReport) -> None:
        """Publish risk metrics update event"""
        event = {
            'type': 'RiskMetricsUpdated',
            'metrics': self.risk_metrics.__dict__,
            'trigger_report': {
                'order_id': report.order_id,
                'symbol': report.symbol,
                'realized_pnl': report.realized_pnl
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self.pubsub.publish('risk.metrics.updated', event)
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Validate incoming ExecutionReport against Pydantic model; handle malformed data gracefully; prevent single report failures from blocking other processing
- **Configuration:** Add risk threshold configurations to main config file; support runtime threshold updates; configurable persistence intervals
- **Testing:** Unit tests for risk metrics calculations; integration tests for event flow; stress tests with high-volume execution reports; edge case testing for drawdown scenarios
- **Dependencies:** Requires persistence service for risk state storage; PubSub system for event broadcasting; may need integration with PortfolioManager for position updates

### 4. Acceptance Criteria
- [ ] ExecutionReport Pydantic model is created with proper validation and typing
- [ ] Risk metrics (consecutive losses, drawdown, PnL) are accurately calculated and updated
- [ ] Risk state is persisted to database and survives system restarts
- [ ] Risk threshold breaches trigger appropriate automated responses
- [ ] Risk metrics update events are published to interested subscribers
- [ ] Comprehensive error handling prevents single report failures from affecting system stability
- [ ] All execution report processing is covered by unit and integration tests
- [ ] Risk metrics dashboard displays real-time updates from the risk manager
- [ ] All TODO comments and placeholder code related to execution report handling are removed 