# Task: Implement comprehensive metrics collection and alerting system.

### 1. Context
- **File:** `gal_friday/monitoring_service.py`
- **Line:** `234`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing comprehensive metrics collection and alerting system.

### 2. Problem Statement
Without proper metrics collection and alerting system, the trading system cannot monitor performance, detect anomalies, or alert operators to critical issues. This prevents proper system observability, performance optimization, and operational reliability.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Metrics Collection Framework:** Comprehensive metrics gathering and aggregation
2. **Build Real-time Monitoring:** Live performance and health monitoring
3. **Implement Alerting System:** Smart alerting with escalation and notification channels
4. **Add Performance Analytics:** Detailed performance analysis and reporting
5. **Create Dashboard Interface:** Real-time dashboards with customizable views
6. **Build Historical Analysis:** Trend analysis and anomaly detection

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime, timezone, timedelta
import json

class MetricType(str, Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    condition: str
    severity: AlertSeverity
    message: str
    threshold: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None

class MetricsCollectionSystem:
    """Enterprise-grade metrics collection and alerting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_buffer: List[Metric] = []
        self.metrics_history: Dict[str, List[Metric]] = {}
        
        # Alerting system
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        # Performance tracking
        self.collection_stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'last_collection_time': None
        }
        
    async def start_monitoring(self) -> None:
        """
        Start comprehensive metrics collection and monitoring
        Replace TODO with full monitoring system
        """
        
        try:
            self.logger.info("Starting metrics collection and monitoring system")
            
            # Initialize alert rules
            await self._load_alert_rules()
            
            # Start collection tasks
            collection_task = asyncio.create_task(self._metrics_collection_loop())
            alerting_task = asyncio.create_task(self._alerting_loop())
            
            # Wait for tasks
            await asyncio.gather(collection_task, alerting_task)
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise MonitoringError(f"Monitoring start failed: {e}")
    
    async def collect_metric(self, name: str, value: float, 
                           labels: Optional[Dict[str, str]] = None,
                           metric_type: MetricType = MetricType.GAUGE) -> None:
        """Collect a single metric"""
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=metric_type
        )
        
        self.metrics_buffer.append(metric)
        self.collection_stats['metrics_collected'] += 1
        
        # Store in history
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        self.metrics_history[name].append(metric)
        
        # Keep only recent history
        max_history = self.config.get('max_history_points', 10000)
        if len(self.metrics_history[name]) > max_history:
            self.metrics_history[name] = self.metrics_history[name][-max_history:]
    
    async def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop"""
        
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_trading_metrics()
                await self._flush_metrics_buffer()
                
                self.collection_stats['last_collection_time'] = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.config.get('collection_interval', 30))
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        
        # CPU and memory metrics
        import psutil
        
        await self.collect_metric('system.cpu_percent', psutil.cpu_percent())
        await self.collect_metric('system.memory_percent', psutil.virtual_memory().percent)
        await self.collect_metric('system.disk_usage', psutil.disk_usage('/').percent)
        
        # Trading system specific metrics
        await self.collect_metric('trading.portfolio_value', 1000000.0)  # Placeholder
        await self.collect_metric('trading.positions_count', 5)  # Placeholder
        await self.collect_metric('trading.cash_balance', 500000.0)  # Placeholder
    
    async def _alerting_loop(self) -> None:
        """Main alerting evaluation loop"""
        
        while True:
            try:
                await self._evaluate_alert_rules()
                await self._process_notifications()
                
                await asyncio.sleep(self.config.get('alert_check_interval', 60))
                
            except Exception as e:
                self.logger.error(f"Error in alerting loop: {e}")
                await asyncio.sleep(10)
    
    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules against current metrics"""
        
        for rule_name, rule_config in self.alert_rules.items():
            try:
                metric_name = rule_config['metric']
                condition = rule_config['condition']
                threshold = rule_config['threshold']
                severity = AlertSeverity(rule_config['severity'])
                
                # Get latest metric value
                if metric_name in self.metrics_history:
                    latest_metric = self.metrics_history[metric_name][-1]
                    current_value = latest_metric.value
                    
                    # Evaluate condition
                    alert_triggered = self._evaluate_condition(current_value, condition, threshold)
                    
                    if alert_triggered and rule_name not in self.active_alerts:
                        # Trigger new alert
                        alert = Alert(
                            name=rule_name,
                            condition=condition,
                            severity=severity,
                            message=rule_config.get('message', f'{metric_name} alert triggered'),
                            threshold=threshold,
                            current_value=current_value,
                            triggered_at=datetime.now(timezone.utc)
                        )
                        
                        self.active_alerts[rule_name] = alert
                        self.collection_stats['alerts_triggered'] += 1
                        
                        self.logger.warning(f"Alert triggered: {rule_name} - {alert.message}")
                        
                    elif not alert_triggered and rule_name in self.active_alerts:
                        # Resolve alert
                        self.active_alerts[rule_name].resolved_at = datetime.now(timezone.utc)
                        
                        self.logger.info(f"Alert resolved: {rule_name}")
                        del self.active_alerts[rule_name]
                        
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule_name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        
        if condition == 'greater_than':
            return value > threshold
        elif condition == 'less_than':
            return value < threshold
        elif condition == 'equals':
            return abs(value - threshold) < 0.001
        else:
            return False
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        
        summary = {
            'collection_stats': self.collection_stats,
            'active_alerts': len(self.active_alerts),
            'metrics_count': len(self.metrics_history),
            'buffer_size': len(self.metrics_buffer)
        }
        
        # Recent metric values
        recent_metrics = {}
        for name, history in self.metrics_history.items():
            if history:
                recent_metrics[name] = {
                    'latest_value': history[-1].value,
                    'timestamp': history[-1].timestamp.isoformat(),
                    'count': len(history)
                }
        
        summary['recent_metrics'] = recent_metrics
        
        return summary

class MonitoringError(Exception):
    """Exception raised for monitoring errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Resilient metric collection; graceful degradation during system issues; comprehensive error logging
- **Configuration:** Configurable collection intervals; customizable alert rules; flexible notification channels
- **Testing:** Unit tests for metric calculations; integration tests with monitoring systems; performance tests for high-frequency collection
- **Dependencies:** System monitoring libraries (psutil); time series databases; notification services; dashboard frameworks

### 4. Acceptance Criteria
- [ ] Metrics collection framework gathers comprehensive system and trading metrics
- [ ] Real-time monitoring provides live visibility into system performance and health
- [ ] Alerting system triggers notifications based on configurable rules and thresholds
- [ ] Performance analytics identify trends, anomalies, and optimization opportunities
- [ ] Dashboard interface provides intuitive real-time monitoring with customizable views
- [ ] Historical analysis supports trend analysis and predictive alerting
- [ ] Notification system supports multiple channels (email, SMS, Slack, etc.)
- [ ] Alert management includes escalation, acknowledgment, and resolution tracking
- [ ] System monitoring covers all critical components with appropriate granularity
- [ ] TODO placeholder is completely replaced with production-ready implementation 