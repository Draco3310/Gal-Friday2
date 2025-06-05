# Task: Implement periodic risk metrics calculation, including drawdown and exposure monitoring.

### 1. Context
- **File:** `gal_friday/risk_manager.py`
- **Line:** `1825`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for periodic risk metrics calculation without implementation of drawdown and exposure monitoring.

### 2. Problem Statement
Without periodic risk metrics calculation, the system lacks continuous monitoring of portfolio health, drawdown patterns, and exposure concentrations. This creates dangerous blind spots where risk can accumulate undetected between trading events. The absence of regular risk assessment means the system cannot proactively identify deteriorating risk conditions or take preventive measures before losses become severe.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Design Metrics Calculation Engine:** Create a comprehensive risk metrics calculation system with configurable intervals
2. **Implement Drawdown Monitoring:** Calculate running, maximum, and time-based drawdown metrics
3. **Build Exposure Analytics:** Monitor position concentrations, sector exposure, and correlation risks
4. **Create Risk Scoring System:** Develop composite risk scores for portfolio health assessment
5. **Add Alert Management:** Implement configurable thresholds with graduated alert levels
6. **Build Historical Tracking:** Maintain risk metrics history for trend analysis and reporting

#### b. Pseudocode or Implementation Sketch
```python
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import deque
import statistics

@dataclass
class RiskMetricsSnapshot:
    """Complete risk metrics at a point in time"""
    timestamp: datetime
    total_equity: float
    unrealized_pnl: float
    realized_pnl_today: float
    max_drawdown: float
    current_drawdown: float
    drawdown_duration: timedelta
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    sharpe_ratio: float
    exposure_by_symbol: Dict[str, float]
    concentration_risk: float
    correlation_risk: float
    leverage_ratio: float
    margin_utilization: float

class PeriodicRiskCalculator:
    """Handles periodic calculation of comprehensive risk metrics"""
    
    def __init__(self, risk_manager, portfolio_manager, market_data_service):
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.market_data = market_data_service
        self.logger = risk_manager.logger
        
        # Configuration
        self.calculation_interval = timedelta(minutes=5)  # Configurable
        self.history_retention = timedelta(days=30)
        
        # Historical data
        self.metrics_history: deque = deque(maxlen=10000)
        self.equity_curve: deque = deque(maxlen=10000)
        self.returns_history: deque = deque(maxlen=1000)
        
        # State tracking
        self.peak_equity = 0.0
        self.last_calculation = None
        self.is_running = False
        
    async def start_periodic_calculation(self) -> None:
        """Start the periodic risk metrics calculation loop"""
        self.is_running = True
        self.logger.info("Starting periodic risk metrics calculation")
        
        while self.is_running:
            try:
                # Calculate current risk metrics
                metrics = await self._calculate_comprehensive_metrics()
                
                # Store metrics
                self._store_metrics(metrics)
                
                # Check for threshold breaches
                alerts = await self._evaluate_risk_thresholds(metrics)
                
                # Publish metrics and alerts
                await self._publish_metrics_update(metrics, alerts)
                
                # Update internal state
                self._update_internal_state(metrics)
                
                self.last_calculation = datetime.now(timezone.utc)
                
                if alerts:
                    self.logger.warning(f"Risk alerts triggered: {[a.type for a in alerts]}")
                
            except Exception as e:
                self.logger.error(f"Error in periodic risk calculation: {e}", exc_info=True)
            
            # Wait for next calculation interval
            await asyncio.sleep(self.calculation_interval.total_seconds())
    
    async def _calculate_comprehensive_metrics(self) -> RiskMetricsSnapshot:
        """Calculate complete set of risk metrics"""
        
        # Get current portfolio state
        positions = await self.portfolio_manager.get_all_positions()
        account_info = await self.portfolio_manager.get_account_info()
        
        # Calculate basic metrics
        total_equity = account_info.get('total_equity', 0.0)
        unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(total_equity)
        
        # Calculate exposure metrics
        exposure_metrics = await self._calculate_exposure_metrics(positions)
        
        # Calculate risk metrics (VaR, etc.)
        risk_metrics = self._calculate_risk_metrics()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        return RiskMetricsSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_equity=total_equity,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_today=await self._get_daily_realized_pnl(),
            max_drawdown=drawdown_metrics['max_drawdown'],
            current_drawdown=drawdown_metrics['current_drawdown'],
            drawdown_duration=drawdown_metrics['drawdown_duration'],
            var_95=risk_metrics['var_95'],
            expected_shortfall=risk_metrics['expected_shortfall'],
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            exposure_by_symbol=exposure_metrics['by_symbol'],
            concentration_risk=exposure_metrics['concentration_risk'],
            correlation_risk=exposure_metrics['correlation_risk'],
            leverage_ratio=account_info.get('leverage_ratio', 1.0),
            margin_utilization=account_info.get('margin_utilization', 0.0)
        )
    
    def _calculate_drawdown_metrics(self, current_equity: float) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics"""
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Current drawdown
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        
        # Calculate maximum drawdown from history
        max_drawdown = 0.0
        if len(self.equity_curve) > 1:
            equity_array = np.array(list(self.equity_curve))
            peak_array = np.maximum.accumulate(equity_array)
            drawdown_array = (peak_array - equity_array) / peak_array
            max_drawdown = np.max(drawdown_array)
        
        # Calculate drawdown duration
        drawdown_start = None
        for i, snapshot in enumerate(reversed(list(self.metrics_history))):
            if snapshot.current_drawdown == 0:
                break
            drawdown_start = snapshot.timestamp
        
        drawdown_duration = timedelta(0)
        if drawdown_start and current_drawdown > 0:
            drawdown_duration = datetime.now(timezone.utc) - drawdown_start
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration
        }
    
    async def _calculate_exposure_metrics(self, positions: List) -> Dict[str, Any]:
        """Calculate portfolio exposure and concentration metrics"""
        
        total_exposure = sum(abs(pos.market_value) for pos in positions)
        
        # Exposure by symbol
        exposure_by_symbol = {
            pos.symbol: abs(pos.market_value) / total_exposure if total_exposure > 0 else 0
            for pos in positions
        }
        
        # Concentration risk (Herfindahl index)
        concentration_risk = sum(weight ** 2 for weight in exposure_by_symbol.values())
        
        # Correlation risk (simplified - could be enhanced with actual correlation matrix)
        correlation_risk = await self._estimate_correlation_risk(positions)
        
        return {
            'by_symbol': exposure_by_symbol,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'total_exposure': total_exposure
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate VaR and other risk metrics"""
        
        if len(self.returns_history) < 30:
            return {'var_95': 0.0, 'expected_shortfall': 0.0}
        
        returns = np.array(list(self.returns_history))
        
        # 95% Value at Risk
        var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        
        # Expected Shortfall (average of returns below VaR)
        tail_returns = returns[returns <= var_95]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
        
        return {
            'var_95': abs(var_95),
            'expected_shortfall': abs(expected_shortfall)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate Sharpe ratio and other performance metrics"""
        
        if len(self.returns_history) < 30:
            return {'sharpe_ratio': 0.0}
        
        returns = np.array(list(self.returns_history))
        
        # Annualized Sharpe ratio (assuming daily returns)
        mean_return = np.mean(returns)
        return_std = np.std(returns)
        
        # Annualize (252 trading days)
        sharpe_ratio = (mean_return * 252) / (return_std * np.sqrt(252)) if return_std > 0 else 0.0
        
        return {'sharpe_ratio': sharpe_ratio}
    
    async def _evaluate_risk_thresholds(self, metrics: RiskMetricsSnapshot) -> List[RiskAlert]:
        """Evaluate metrics against configured thresholds"""
        alerts = []
        
        # Drawdown alerts
        if metrics.current_drawdown > self.risk_manager.config.get('risk.max_drawdown_threshold', 0.1):
            alerts.append(RiskAlert('drawdown_threshold_breach', metrics.current_drawdown))
        
        # Concentration alerts
        if metrics.concentration_risk > self.risk_manager.config.get('risk.max_concentration_risk', 0.5):
            alerts.append(RiskAlert('concentration_risk_high', metrics.concentration_risk))
        
        # VaR alerts
        if metrics.var_95 > self.risk_manager.config.get('risk.max_var_threshold', 0.05):
            alerts.append(RiskAlert('var_threshold_breach', metrics.var_95))
        
        return alerts
    
    async def _publish_metrics_update(self, metrics: RiskMetricsSnapshot, alerts: List[RiskAlert]) -> None:
        """Publish risk metrics update event"""
        event_data = {
            'type': 'PeriodicRiskMetricsUpdate',
            'metrics': {
                'timestamp': metrics.timestamp.isoformat(),
                'total_equity': metrics.total_equity,
                'current_drawdown': metrics.current_drawdown,
                'max_drawdown': metrics.max_drawdown,
                'var_95': metrics.var_95,
                'sharpe_ratio': metrics.sharpe_ratio,
                'concentration_risk': metrics.concentration_risk,
                'leverage_ratio': metrics.leverage_ratio
            },
            'alerts': [{'type': alert.type, 'value': alert.value} for alert in alerts],
            'calculation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self.risk_manager.pubsub.publish('risk.periodic_metrics', event_data)
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of missing market data or portfolio information; fallback calculations for incomplete data; continue operations if single metric calculation fails
- **Configuration:** Configurable calculation intervals, threshold values, and metric retention periods; runtime configuration updates without restart
- **Testing:** Unit tests for individual metric calculations; integration tests for complete calculation cycle; performance tests for large portfolios; historical data validation
- **Dependencies:** Integration with PortfolioManager for position data; MarketDataService for current prices; database for historical metrics storage; pub/sub system for real-time updates

### 4. Acceptance Criteria
- [ ] Periodic risk metrics calculation runs at configured intervals without blocking other operations
- [ ] Comprehensive drawdown metrics including current, maximum, and duration are calculated accurately
- [ ] Portfolio exposure metrics track concentration and correlation risks effectively
- [ ] VaR and Expected Shortfall calculations use appropriate statistical methods
- [ ] Risk threshold breaches trigger immediate alerts through the pub/sub system
- [ ] Historical metrics are stored and accessible for trend analysis and reporting
- [ ] Performance impact is minimal (<1% CPU usage during normal operations)
- [ ] All risk metrics are validated against known benchmarks and edge cases
- [ ] Dashboard displays real-time risk metrics with appropriate refresh rates
- [ ] TODO placeholder is completely removed and replaced with production-ready code 