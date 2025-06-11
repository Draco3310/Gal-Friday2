#!/usr/bin/env python3
"""Monitoring Service for Gal Friday trading system.

This module provides system monitoring capabilities including health checks,
performance tracking, and automatic trading halt triggers when thresholds are exceeded.
"""

import asyncio
import json
import logging  # Added for structured logging
import statistics
import time
import uuid
from collections import defaultdict, deque  # Added for tracking recent API errors
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import (  # Added Type for exc_info typing
    TYPE_CHECKING,
    Any,
    Optional,
)

import psutil  # Added for system resource monitoring

# Import actual classes when available, otherwise use placeholders
from .logger_service import LoggerService
from .portfolio_manager import PortfolioManager
from .dal.repositories.history_repository import HistoryRepository


class MetricType(str, Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "metric_type": self.metric_type.value
        }


@dataclass
class Alert:
    """Alert definition and state."""
    alert_id: str
    name: str
    condition: str
    severity: AlertSeverity
    message: str
    threshold: float
    current_value: float
    triggered_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    escalation_level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "condition": self.condition,
            "severity": self.severity.value,
            "message": self.message,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "triggered_at": self.triggered_at.isoformat(),
            "status": self.status.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "escalation_level": self.escalation_level,
            "metadata": self.metadata
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: float
    severity: AlertSeverity
    message_template: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes default
    escalation_rules: list[dict[str, Any]] = field(default_factory=list)
    notification_channels: list[str] = field(default_factory=list)


@dataclass 
class PerformanceMetrics:
    """System performance metrics snapshot."""
    timestamp: datetime
    cpu_usage_pct: float
    memory_usage_pct: float
    disk_usage_pct: float
    network_io_bytes: dict[str, int]
    active_connections: int
    response_times_ms: dict[str, float]
    error_rates: dict[str, float]
    throughput_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert performance metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage_pct": self.cpu_usage_pct,
            "memory_usage_pct": self.memory_usage_pct,
            "disk_usage_pct": self.disk_usage_pct,
            "network_io_bytes": self.network_io_bytes,
            "active_connections": self.active_connections,
            "response_times_ms": self.response_times_ms,
            "error_rates": self.error_rates,
            "throughput_metrics": self.throughput_metrics
        }


class MetricsCollectionSystem:
    """Enterprise-grade metrics collection and alerting system."""

    def __init__(self, config: dict[str, Any], logger: LoggerService) -> None:
        """Initialize the metrics collection system.
        
        Args:
            config: Configuration dictionary
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._source_module = "MetricsCollectionSystem"
        
        # Metrics storage and buffering
        self.metrics_buffer: list[Metric] = []
        self.metrics_history: dict[str, deque[Metric]] = defaultdict(
            lambda: deque(maxlen=self.config.get('max_history_points', 10000))
        )
        
        # Alerting system
        self.alert_rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque[Alert] = deque(maxlen=self.config.get('max_alert_history', 1000))
        
        # Performance tracking
        self.collection_stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'last_collection_time': None,
            'collection_errors': 0,
            'buffer_flushes': 0
        }
        
        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._alerting_task: Optional[asyncio.Task] = None
        self._analytics_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Analytics and aggregation
        self.metric_aggregates: dict[str, dict[str, float]] = defaultdict(dict)
        self.trend_analysis: dict[str, list[float]] = defaultdict(list)
        
        # Load alert rules from configuration
        self._load_alert_rules()

    async def start(self) -> None:
        """Start comprehensive metrics collection and monitoring system."""
        if self._running:
            self.logger.warning(
                "MetricsCollectionSystem already running",
                source_module=self._source_module
            )
            return
            
        try:
            self.logger.info(
                "Starting enterprise-grade metrics collection and monitoring system",
                source_module=self._source_module
            )
            
            self._running = True
            
            # Start background tasks
            self._collection_task = asyncio.create_task(self._metrics_collection_loop())
            self._alerting_task = asyncio.create_task(self._alerting_loop())
            self._analytics_task = asyncio.create_task(self._analytics_loop())
            
            self.logger.info(
                "MetricsCollectionSystem started successfully",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to start MetricsCollectionSystem: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the metrics collection system."""
        self.logger.info(
            "Stopping MetricsCollectionSystem",
            source_module=self._source_module
        )
        
        self._running = False
        
        # Cancel background tasks
        for task in [self._collection_task, self._alerting_task, self._analytics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(
                        f"Error stopping task: {e}",
                        source_module=self._source_module
                    )
        
        # Flush remaining metrics
        await self._flush_metrics_buffer()
        
        # Close database connections
        await self._close_database_connections()
        
        self.logger.info(
            "MetricsCollectionSystem stopped",
            source_module=self._source_module
        )

    async def collect_metric(
        self, 
        name: str, 
        value: float,
        labels: Optional[dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ) -> None:
        """Collect a single metric data point.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            metric_type: Type of metric being collected
        """
        try:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(UTC),
                labels=labels or {},
                metric_type=metric_type
            )
            
            # Add to buffer for batch processing
            self.metrics_buffer.append(metric)
            
            # Store in history for immediate access
            self.metrics_history[name].append(metric)
            
            # Update collection stats
            self.collection_stats['metrics_collected'] += 1
            
            self.logger.debug(
                f"Collected metric: {name}={value} (type: {metric_type.value})",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            self.logger.error(
                f"Error collecting metric {name}: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def collect_batch_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Collect multiple metrics in batch for efficiency.
        
        Args:
            metrics: List of metric dictionaries with name, value, labels, type
        """
        for metric_data in metrics:
            await self.collect_metric(
                name=metric_data['name'],
                value=metric_data['value'],
                labels=metric_data.get('labels'),
                metric_type=MetricType(metric_data.get('type', 'gauge'))
            )

    async def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop."""
        collection_interval = self.config.get('collection_interval_seconds', 30)
        
        while self._running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect trading metrics  
                await self._collect_trading_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Flush metrics buffer periodically
                if len(self.metrics_buffer) >= self.config.get('buffer_flush_size', 100):
                    await self._flush_metrics_buffer()
                
                # Update collection timestamp
                self.collection_stats['last_collection_time'] = datetime.now(UTC)
                
                await asyncio.sleep(collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.collection_stats['collection_errors'] += 1
                self.logger.error(
                    f"Error in metrics collection loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(5)  # Brief pause on error

    async def _collect_system_metrics(self) -> None:
        """Collect comprehensive system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            await self.collect_metric('system.cpu.usage_percent', cpu_percent)
            await self.collect_metric('system.cpu.count', cpu_count)
            if cpu_freq:
                await self.collect_metric('system.cpu.frequency_mhz', cpu_freq.current)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.collect_metric('system.memory.usage_percent', memory.percent)
            await self.collect_metric('system.memory.available_bytes', memory.available)
            await self.collect_metric('system.memory.used_bytes', memory.used)
            await self.collect_metric('system.memory.total_bytes', memory.total)
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            await self.collect_metric('system.disk.usage_percent', 
                                    (disk_usage.used / disk_usage.total) * 100)
            await self.collect_metric('system.disk.free_bytes', disk_usage.free)
            await self.collect_metric('system.disk.used_bytes', disk_usage.used)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            await self.collect_metric('system.network.bytes_sent', net_io.bytes_sent)
            await self.collect_metric('system.network.bytes_received', net_io.bytes_recv)
            await self.collect_metric('system.network.packets_sent', net_io.packets_sent)
            await self.collect_metric('system.network.packets_received', net_io.packets_recv)
            
            # Process metrics
            process = psutil.Process()
            await self.collect_metric('system.process.cpu_percent', process.cpu_percent())
            await self.collect_metric('system.process.memory_percent', process.memory_percent())
            await self.collect_metric('system.process.memory_rss_bytes', process.memory_info().rss)
            await self.collect_metric('system.process.threads', process.num_threads())
            
        except Exception as e:
            self.logger.error(
                f"Error collecting system metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _collect_trading_metrics(self) -> None:
        """Collect comprehensive trading-specific metrics."""
        try:
            # Calculate system uptime
            if self._service_start_time:
                uptime_seconds = (datetime.now(UTC) - self._service_start_time).total_seconds()
                await self.collect_metric('trading.system.uptime_seconds', uptime_seconds)
            
            # Get real portfolio data from portfolio manager
            try:
                current_state = self._portfolio_manager.get_current_state()
                
                # Portfolio value metrics
                total_equity = current_state.get("total_equity", 0)
                available_balance = current_state.get("available_balance", 0)
                total_unrealized_pnl = current_state.get("total_unrealized_pnl", 0)
                total_realized_pnl = current_state.get("total_realized_pnl", 0)
                
                await self.collect_batch_metrics([
                    {"name": "trading.portfolio.total_equity_usd", "value": float(total_equity)},
                    {"name": "trading.portfolio.available_balance_usd", "value": float(available_balance)},
                    {"name": "trading.portfolio.unrealized_pnl_usd", "value": float(total_unrealized_pnl)},
                    {"name": "trading.portfolio.realized_pnl_usd", "value": float(total_realized_pnl)},
                ])
                
                # Position metrics
                positions = current_state.get("positions", {})
                active_positions = {k: v for k, v in positions.items() if float(v.get("quantity", 0)) != 0}
                
                await self.collect_metric('trading.positions.total_count', len(positions))
                await self.collect_metric('trading.positions.active_count', len(active_positions))
                
                # Calculate total portfolio exposure
                total_exposure = 0
                for pair, position in active_positions.items():
                    market_value = float(position.get("market_value_usd", 0))
                    total_exposure += abs(market_value)
                
                await self.collect_metric('trading.portfolio.total_exposure_usd', total_exposure)
                
                # Position concentration metrics
                if total_equity > 0:
                    largest_position_pct = 0
                    for pair, position in active_positions.items():
                        market_value = abs(float(position.get("market_value_usd", 0)))
                        position_pct = (market_value / float(total_equity)) * 100
                        largest_position_pct = max(largest_position_pct, position_pct)
                        
                        # Individual position metrics
                        await self.collect_metric(
                            f'trading.position.{pair.replace("/", "_")}.market_value_usd',
                            market_value,
                            labels={"trading_pair": pair}
                        )
                        await self.collect_metric(
                            f'trading.position.{pair.replace("/", "_")}.portfolio_percentage',
                            position_pct,
                            labels={"trading_pair": pair}
                        )
                    
                    await self.collect_metric('trading.portfolio.largest_position_pct', largest_position_pct)
                    
                    # Portfolio utilization
                    portfolio_utilization = (total_exposure / float(total_equity)) * 100 if total_equity > 0 else 0
                    await self.collect_metric('trading.portfolio.utilization_pct', portfolio_utilization)
                
                # Drawdown metrics (already collected in _check_drawdown_conditions but good for completeness)
                total_drawdown = current_state.get("total_drawdown_pct", 0)
                daily_drawdown = current_state.get("daily_drawdown_pct", 0)
                
                await self.collect_batch_metrics([
                    {"name": "trading.risk.total_drawdown_pct", "value": float(abs(total_drawdown))},
                    {"name": "trading.risk.daily_drawdown_pct", "value": float(abs(daily_drawdown))},
                ])
                
            except Exception as e:
                self.logger.error(
                    f"Error collecting portfolio metrics: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
            
            # Trading system health metrics
            await self.collect_batch_metrics([
                {"name": "trading.system.is_halted", "value": 1 if self._is_halted else 0},
                {"name": "trading.risk.consecutive_api_failures", "value": self._consecutive_api_failures},
                {"name": "trading.risk.consecutive_losses", "value": self._consecutive_losses},
                {"name": "trading.system.active_pairs_count", "value": len(self._active_pairs)},
            ])
            
            # Market data tracking metrics
            current_time = datetime.now(UTC)
            fresh_pairs_count = 0
            stale_pairs_count = 0
            
            for pair, last_timestamp in self._last_market_data_times.items():
                age_seconds = (current_time - last_timestamp).total_seconds()
                if age_seconds <= self._data_staleness_threshold_s:
                    fresh_pairs_count += 1
                else:
                    stale_pairs_count += 1
            
            await self.collect_batch_metrics([
                {"name": "trading.market_data.fresh_pairs_count", "value": fresh_pairs_count},
                {"name": "trading.market_data.stale_pairs_count", "value": stale_pairs_count},
                {"name": "trading.market_data.total_pairs_tracked", "value": len(self._last_market_data_times)},
            ])
            
            # API error tracking
            recent_api_errors_count = len(self._recent_api_errors)
            await self.collect_metric('trading.api.recent_errors_count', recent_api_errors_count)
            
            self.logger.debug(
                "Trading metrics collection completed",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error collecting trading metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _collect_application_metrics(self) -> None:
        """Collect comprehensive application-specific metrics."""
        try:
            # Collection system performance metrics
            await self.collect_batch_metrics([
                {"name": "metrics.collection.total_count", "value": self.collection_stats['metrics_collected']},
                {"name": "metrics.collection.errors", "value": self.collection_stats['collection_errors']},
                {"name": "metrics.collection.buffer_flushes", "value": self.collection_stats['buffer_flushes']},
                {"name": "metrics.buffer.current_size", "value": len(self.metrics_buffer)},
                {"name": "metrics.alerts.active_count", "value": len(self.active_alerts)},
                {"name": "metrics.alerts.total_triggered", "value": self.collection_stats['alerts_triggered']},
            ])
            
            # Memory usage calculations
            import sys
            
            # Calculate actual memory usage of metrics system components
            metrics_history_size = sum(len(deque_obj) for deque_obj in self.metrics_history.values())
            alert_history_size = len(self.alert_history)
            active_alerts_size = len(self.active_alerts)
            
            # Estimate memory usage in bytes (rough approximation)
            metrics_memory_bytes = (
                sys.getsizeof(self.metrics_history) +
                sys.getsizeof(self.metrics_buffer) +
                sys.getsizeof(self.active_alerts) +
                sys.getsizeof(self.alert_history) +
                (metrics_history_size * 200) +  # Approximate size per metric
                (alert_history_size * 500) +    # Approximate size per alert
                (active_alerts_size * 500)      # Approximate size per active alert
            )
            
            metrics_memory_mb = metrics_memory_bytes / (1024 * 1024)
            
            await self.collect_batch_metrics([
                {"name": "metrics.system.memory_usage_mb", "value": metrics_memory_mb},
                {"name": "metrics.system.history_entries_total", "value": metrics_history_size},
                {"name": "metrics.system.unique_metrics_tracked", "value": len(self.metrics_history)},
                {"name": "metrics.system.alert_history_size", "value": alert_history_size},
            ])
            
            # Task status metrics
            task_status = {
                "collection_task_running": self._collection_task and not self._collection_task.done(),
                "alerting_task_running": self._alerting_task and not self._alerting_task.done(),
                "analytics_task_running": self._analytics_task and not self._analytics_task.done(),
            }
            
            for task_name, is_running in task_status.items():
                await self.collect_metric(f'metrics.system.{task_name}', 1 if is_running else 0)
            
            # Collection efficiency metrics
            if self.collection_stats['last_collection_time']:
                time_since_last_collection = (
                    datetime.now(UTC) - self.collection_stats['last_collection_time']
                ).total_seconds()
                await self.collect_metric('metrics.collection.seconds_since_last', time_since_last_collection)
            
            # Alert rule metrics
            enabled_rules = sum(1 for rule in self.alert_rules.values() if rule.enabled)
            disabled_rules = len(self.alert_rules) - enabled_rules
            
            await self.collect_batch_metrics([
                {"name": "metrics.alert_rules.total_count", "value": len(self.alert_rules)},
                {"name": "metrics.alert_rules.enabled_count", "value": enabled_rules},
                {"name": "metrics.alert_rules.disabled_count", "value": disabled_rules},
            ])
            
            # Alert severity distribution
            severity_counts = {"info": 0, "warning": 0, "critical": 0, "emergency": 0}
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            
            for severity, count in severity_counts.items():
                await self.collect_metric(f'metrics.alerts.active_by_severity.{severity}', count)
            
            # Performance metrics calculation
            collection_rate = 0
            if self.collection_stats.get('last_collection_time'):
                elapsed_time = (datetime.now(UTC) - self.collection_stats['last_collection_time']).total_seconds()
                if elapsed_time > 0:
                    collection_rate = self.collection_stats['metrics_collected'] / elapsed_time
            
            await self.collect_metric('metrics.collection.rate_per_second', collection_rate)
            
            # System running status
            await self.collect_metric('metrics.system.running', 1 if self._running else 0)
            
        except Exception as e:
            self.logger.error(
                f"Error collecting application metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _flush_metrics_buffer(self) -> None:
        """Flush metrics buffer to InfluxDB and PostgreSQL persistent storage."""
        if not self.metrics_buffer:
            return
            
        try:
            buffer_size = len(self.metrics_buffer)
            
            # Write to InfluxDB time series database
            await self._write_to_influxdb(self.metrics_buffer)
            
            # Write to PostgreSQL for structured storage and alerting
            await self._write_to_postgresql(self.metrics_buffer)
            
            # Optional: Log metrics for debugging if enabled
            if self.config.get('log_metrics', False):
                for metric in self.metrics_buffer:
                    self.logger.debug(
                        f"METRIC: {json.dumps(metric.to_dict())}",
                        source_module=self._source_module
                    )
            
            # Clear the buffer after successful writes
            self.metrics_buffer.clear()
            self.collection_stats['buffer_flushes'] += 1
            
            self.logger.debug(
                f"Flushed {buffer_size} metrics to InfluxDB and PostgreSQL",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error flushing metrics buffer: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            # Don't clear buffer on error to avoid data loss
            await self._handle_flush_failure(e)

    async def _write_to_influxdb(self, metrics: list[Metric]) -> None:
        """Write metrics to InfluxDB time series database."""
        try:
            # Initialize InfluxDB client if not already done
            if not hasattr(self, '_influx_client'):
                await self._initialize_influxdb_client()
            
            if not self._influx_client:
                self.logger.warning(
                    "InfluxDB client not available, skipping time series write",
                    source_module=self._source_module
                )
                return
            
            # Prepare InfluxDB line protocol points
            points = []
            for metric in metrics:
                # Convert metric to InfluxDB point
                point = {
                    "measurement": metric.name,
                    "tags": metric.labels,
                    "fields": {
                        "value": metric.value,
                        "metric_type": metric.metric_type.value
                    },
                    "time": metric.timestamp
                }
                points.append(point)
            
            # Write points to InfluxDB
            if hasattr(self._influx_client, 'write_points'):
                # Using influxdb library
                success = self._influx_client.write_points(points)
                if not success:
                    raise Exception("InfluxDB write_points returned False")
            elif hasattr(self._influx_client, 'write'):
                # Using influxdb-client library
                from influxdb_client import Point
                
                influx_points = []
                for metric in metrics:
                    point = Point(metric.name)
                    
                    # Add tags
                    for key, value in metric.labels.items():
                        point = point.tag(key, value)
                    
                    # Add fields
                    point = point.field("value", metric.value)
                    point = point.field("metric_type", metric.metric_type.value)
                    
                    # Set timestamp
                    point = point.time(metric.timestamp)
                    
                    influx_points.append(point)
                
                # Get write API and write
                write_api = self._influx_client.write_api()
                write_api.write(
                    bucket=self.config.get('influxdb_bucket', 'trading_metrics'),
                    record=influx_points
                )
                write_api.close()
            
            self.logger.debug(
                f"Successfully wrote {len(points)} metrics to InfluxDB",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to write metrics to InfluxDB: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _write_to_postgresql(self, metrics: list[Metric]) -> None:
        """Write metrics to PostgreSQL for structured storage and alerting."""
        try:
            # Initialize PostgreSQL connection if not already done
            if not hasattr(self, '_pg_pool'):
                await self._initialize_postgresql_connection()
            
            if not self._pg_pool:
                self.logger.warning(
                    "PostgreSQL connection not available, skipping structured write",
                    source_module=self._source_module
                )
                return
            
            # Prepare batch insert
            async with self._pg_pool.acquire() as conn:
                # Insert metrics
                metric_records = []
                for metric in metrics:
                    metric_records.append((
                        metric.name,
                        metric.value,
                        metric.timestamp,
                        json.dumps(metric.labels),
                        metric.metric_type.value
                    ))
                
                # Batch insert metrics
                await conn.executemany("""
                    INSERT INTO metrics (name, value, timestamp, labels, metric_type, created_at)
                    VALUES ($1, $2, $3, $4, $5, NOW())
                    ON CONFLICT (name, timestamp) DO UPDATE SET
                        value = EXCLUDED.value,
                        labels = EXCLUDED.labels,
                        metric_type = EXCLUDED.metric_type,
                        updated_at = NOW()
                """, metric_records)
                
                # Update metric summaries for fast queries
                await self._update_metric_summaries(conn, metrics)
            
            self.logger.debug(
                f"Successfully wrote {len(metrics)} metrics to PostgreSQL",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to write metrics to PostgreSQL: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _update_metric_summaries(self, conn, metrics: list[Metric]) -> None:
        """Update metric summary tables for efficient querying."""
        try:
            # Group metrics by name for summary updates
            metric_groups = {}
            for metric in metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric)
            
            # Update summaries for each metric
            for metric_name, metric_list in metric_groups.items():
                latest_metric = max(metric_list, key=lambda m: m.timestamp)
                values = [m.value for m in metric_list]
                
                # Calculate summary statistics
                avg_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)
                
                # Upsert metric summary
                await conn.execute("""
                    INSERT INTO metric_summaries (
                        name, latest_value, latest_timestamp, avg_value, 
                        min_value, max_value, sample_count, updated_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                    ON CONFLICT (name) DO UPDATE SET
                        latest_value = EXCLUDED.latest_value,
                        latest_timestamp = EXCLUDED.latest_timestamp,
                        avg_value = (metric_summaries.avg_value * metric_summaries.sample_count + 
                                   EXCLUDED.avg_value * EXCLUDED.sample_count) / 
                                   (metric_summaries.sample_count + EXCLUDED.sample_count),
                        min_value = LEAST(metric_summaries.min_value, EXCLUDED.min_value),
                        max_value = GREATEST(metric_summaries.max_value, EXCLUDED.max_value),
                        sample_count = metric_summaries.sample_count + EXCLUDED.sample_count,
                        updated_at = NOW()
                """, metric_name, latest_metric.value, latest_metric.timestamp,
                    avg_value, min_value, max_value, len(values))
                
        except Exception as e:
            self.logger.error(
                f"Failed to update metric summaries: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _initialize_influxdb_client(self) -> None:
        """Initialize InfluxDB client connection."""
        try:
            influx_config = self.config.get('influxdb', {})
            
            if not influx_config.get('enabled', True):
                self._influx_client = None
                return
            
            # Try influxdb-client first (v2.x)
            try:
                from influxdb_client import InfluxDBClient
                
                self._influx_client = InfluxDBClient(
                    url=influx_config.get('url', 'http://localhost:8086'),
                    token=influx_config.get('token'),
                    org=influx_config.get('org', 'trading'),
                    timeout=influx_config.get('timeout', 10000)
                )
                
                # Test connection
                health = self._influx_client.health()
                if health.status != "pass":
                    raise Exception(f"InfluxDB health check failed: {health.status}")
                
                self.logger.info(
                    "Connected to InfluxDB v2.x",
                    source_module=self._source_module
                )
                
            except ImportError:
                # Fallback to influxdb v1.x
                from influxdb import InfluxDBClient as InfluxDBClientV1
                
                self._influx_client = InfluxDBClientV1(
                    host=influx_config.get('host', 'localhost'),
                    port=influx_config.get('port', 8086),
                    username=influx_config.get('username'),
                    password=influx_config.get('password'),
                    database=influx_config.get('database', 'trading_metrics'),
                    timeout=influx_config.get('timeout', 10)
                )
                
                # Test connection
                self._influx_client.ping()
                
                self.logger.info(
                    "Connected to InfluxDB v1.x",
                    source_module=self._source_module
                )
                
        except Exception as e:
            self.logger.error(
                f"Failed to initialize InfluxDB client: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            self._influx_client = None

    async def _initialize_postgresql_connection(self) -> None:
        """Initialize PostgreSQL connection pool."""
        try:
            pg_config = self.config.get('postgresql', {})
            
            if not pg_config.get('enabled', True):
                self._pg_pool = None
                return
            
            import asyncpg
            
            # Create connection pool
            self._pg_pool = await asyncpg.create_pool(
                host=pg_config.get('host', 'localhost'),
                port=pg_config.get('port', 5432),
                user=pg_config.get('user', 'postgres'),
                password=pg_config.get('password'),
                database=pg_config.get('database', 'trading_metrics'),
                min_size=pg_config.get('min_connections', 2),
                max_size=pg_config.get('max_connections', 10),
                command_timeout=pg_config.get('timeout', 30)
            )
            
            # Ensure tables exist
            await self._create_postgresql_tables()
            
            self.logger.info(
                "Connected to PostgreSQL database",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize PostgreSQL connection: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            self._pg_pool = None

    async def _create_postgresql_tables(self) -> None:
        """Create necessary PostgreSQL tables for metrics storage."""
        try:
            async with self._pg_pool.acquire() as conn:
                # Create metrics table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id BIGSERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        labels JSONB,
                        metric_type VARCHAR(50) NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ,
                        UNIQUE(name, timestamp)
                    )
                """)
                
                # Create metric summaries table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS metric_summaries (
                        name VARCHAR(255) PRIMARY KEY,
                        latest_value DOUBLE PRECISION NOT NULL,
                        latest_timestamp TIMESTAMPTZ NOT NULL,
                        avg_value DOUBLE PRECISION,
                        min_value DOUBLE PRECISION,
                        max_value DOUBLE PRECISION,
                        sample_count BIGINT DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create alerts table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id BIGSERIAL PRIMARY KEY,
                        alert_id VARCHAR(255) UNIQUE NOT NULL,
                        name VARCHAR(255) NOT NULL,
                        condition VARCHAR(100) NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        threshold DOUBLE PRECISION,
                        current_value DOUBLE PRECISION,
                        status VARCHAR(50) NOT NULL,
                        triggered_at TIMESTAMPTZ NOT NULL,
                        resolved_at TIMESTAMPTZ,
                        acknowledged_at TIMESTAMPTZ,
                        acknowledged_by VARCHAR(255),
                        escalation_level INTEGER DEFAULT 0,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                # Create indexes for performance
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                    ON metrics(name, timestamp DESC)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp DESC)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_status_severity 
                    ON alerts(status, severity)
                """)
                
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at 
                    ON alerts(triggered_at DESC)
                """)
                
            self.logger.info(
                "PostgreSQL tables created/verified successfully",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to create PostgreSQL tables: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise

    async def _handle_flush_failure(self, error: Exception) -> None:
        """Handle metrics flush failure with retry logic."""
        try:
            # Increment error counter
            self.collection_stats['collection_errors'] += 1
            
            # If buffer is getting too large, remove oldest entries to prevent memory issues
            max_buffer_size = self.config.get('max_buffer_size_on_error', 1000)
            if len(self.metrics_buffer) > max_buffer_size:
                dropped_count = len(self.metrics_buffer) - max_buffer_size
                self.metrics_buffer = self.metrics_buffer[-max_buffer_size:]
                
                self.logger.warning(
                    f"Dropped {dropped_count} metrics due to persistent flush failures",
                    source_module=self._source_module
                )
            
            # Log error details for troubleshooting
            self.logger.error(
                f"Metrics flush failed, buffer size: {len(self.metrics_buffer)}",
                source_module=self._source_module,
                context={
                    "error_type": type(error).__name__,
                    "buffer_size": len(self.metrics_buffer),
                    "flush_failures": self.collection_stats['collection_errors']
                }
            )
            
        except Exception as e:
                         self.logger.error(
                f"Error in flush failure handler: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _close_database_connections(self) -> None:
        """Close database connections and clean up resources."""
        try:
            # Close InfluxDB client
            if hasattr(self, '_influx_client') and self._influx_client:
                try:
                    if hasattr(self._influx_client, 'close'):
                        self._influx_client.close()
                    elif hasattr(self._influx_client, '__del__'):
                        # For v1.x client
                        del self._influx_client
                    
                    self.logger.debug(
                        "InfluxDB client connection closed",
                        source_module=self._source_module
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing InfluxDB connection: {e}",
                        source_module=self._source_module
                    )
                finally:
                    self._influx_client = None
            
            # Close PostgreSQL connection pool
            if hasattr(self, '_pg_pool') and self._pg_pool:
                try:
                    await self._pg_pool.close()
                    self.logger.debug(
                        "PostgreSQL connection pool closed",
                        source_module=self._source_module
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error closing PostgreSQL connection pool: {e}",
                        source_module=self._source_module
                    )
                finally:
                    self._pg_pool = None
                    
        except Exception as e:
            self.logger.error(
                f"Error in database cleanup: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _alerting_loop(self) -> None:
        """Main alerting evaluation loop."""
        alert_check_interval = self.config.get('alert_check_interval_seconds', 60)
        
        while self._running:
            try:
                await self._evaluate_alert_rules()
                await self._process_alert_escalations()
                await self._cleanup_resolved_alerts()
                
                await asyncio.sleep(alert_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in alerting loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(10)

    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
                
            try:
                # Get latest metric value
                if rule.metric_name not in self.metrics_history:
                    continue
                    
                history = self.metrics_history[rule.metric_name]
                if not history:
                    continue
                    
                latest_metric = history[-1]
                current_value = latest_metric.value
                
                # Evaluate condition
                alert_triggered = self._evaluate_condition(
                    current_value, rule.condition, rule.threshold
                )
                
                alert_id = f"{rule_name}_{int(time.time())}"
                
                if alert_triggered and rule_name not in self.active_alerts:
                    # Check cooldown period
                    if self._is_in_cooldown(rule_name, rule.cooldown_seconds):
                        continue
                    
                    # Create new alert
                    alert = Alert(
                        alert_id=alert_id,
                        name=rule_name,
                        condition=rule.condition,
                        severity=rule.severity,
                        message=rule.message_template.format(
                            metric=rule.metric_name,
                            value=current_value,
                            threshold=rule.threshold
                        ),
                        threshold=rule.threshold,
                        current_value=current_value,
                        triggered_at=datetime.now(UTC),
                        metadata={
                            'metric_name': rule.metric_name,
                            'rule_name': rule_name,
                            'labels': latest_metric.labels
                        }
                    )
                    
                    self.active_alerts[rule_name] = alert
                    self.alert_history.append(alert)
                    self.collection_stats['alerts_triggered'] += 1
                    
                    self.logger.warning(
                        f"ALERT TRIGGERED: {rule_name} - {alert.message}",
                        source_module=self._source_module,
                        context={
                            'alert_id': alert_id,
                            'severity': rule.severity.value,
                            'current_value': current_value,
                            'threshold': rule.threshold
                        }
                    )
                    
                    # Send notifications
                    await self._send_alert_notifications(alert, rule)
                    
                    # Persist alert to PostgreSQL
                    await self._persist_alert_to_database(alert)
                    
                elif not alert_triggered and rule_name in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[rule_name]
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.now(UTC)
                    
                    self.logger.info(
                        f"ALERT RESOLVED: {rule_name}",
                        source_module=self._source_module,
                        context={'alert_id': alert.alert_id}
                    )
                    
                    # Send resolution notification
                    await self._send_resolution_notification(alert, rule)
                    
                    # Update alert in PostgreSQL
                    await self._persist_alert_to_database(alert)
                    
                    # Move to history
                    del self.active_alerts[rule_name]
                    
            except Exception as e:
                self.logger.error(
                    f"Error evaluating alert rule {rule_name}: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        condition_map = {
            'greater_than': value > threshold,
            'less_than': value < threshold,
            'equals': abs(value - threshold) < 0.001,
            'not_equals': abs(value - threshold) >= 0.001,
            'greater_than_or_equal': value >= threshold,
            'less_than_or_equal': value <= threshold
        }
        
        return condition_map.get(condition, False)

    def _is_in_cooldown(self, rule_name: str, cooldown_seconds: int) -> bool:
        """Check if an alert rule is in cooldown period."""
        # Check if we have recent alerts for this rule
        cutoff_time = datetime.now(UTC) - timedelta(seconds=cooldown_seconds)
        
        for alert in reversed(self.alert_history):
            if alert.name == rule_name and alert.triggered_at > cutoff_time:
                return True
        
        return False

    async def _send_alert_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send alert notifications through configured channels."""
        try:
            # In a production system, this would integrate with:
            # - Email service
            # - Slack/Teams webhooks  
            # - SMS service
            # - PagerDuty/OpsGenie
            
            self.logger.info(
                f"Sending alert notifications for {alert.name} via channels: {rule.notification_channels}",
                source_module=self._source_module
            )
            
            # For now, just log the notification
            notification_data = {
                'alert': alert.to_dict(),
                'channels': rule.notification_channels,
                'type': 'alert_triggered'
            }
            
            self.logger.info(
                f"NOTIFICATION: {json.dumps(notification_data)}",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error sending alert notifications: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _send_resolution_notification(self, alert: Alert, rule: AlertRule) -> None:
        """Send alert resolution notifications."""
        try:
            notification_data = {
                'alert': alert.to_dict(),
                'channels': rule.notification_channels,
                'type': 'alert_resolved'
            }
            
            self.logger.info(
                f"RESOLUTION NOTIFICATION: {json.dumps(notification_data)}",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error sending resolution notification: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _process_alert_escalations(self) -> None:
        """Process alert escalations based on configured rules."""
        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.ACTIVE:
                continue
                
            # Check if alert needs escalation
            time_since_trigger = datetime.now(UTC) - alert.triggered_at
            
            # Simple escalation logic - can be enhanced with more complex rules
            if (time_since_trigger.total_seconds() > 1800 and  # 30 minutes
                alert.escalation_level == 0 and
                alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]):
                
                alert.escalation_level += 1
                
                self.logger.warning(
                    f"ALERT ESCALATED: {alert.name} to level {alert.escalation_level}",
                    source_module=self._source_module
                )

    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up old resolved alerts from active tracking."""
        # This is handled in _evaluate_alert_rules when alerts are resolved
        pass

    async def _analytics_loop(self) -> None:
        """Background analytics and trend analysis."""
        analytics_interval = self.config.get('analytics_interval_seconds', 300)  # 5 minutes
        
        while self._running:
            try:
                await self._update_metric_aggregates()
                await self._perform_trend_analysis()
                await self._detect_anomalies()
                
                await asyncio.sleep(analytics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    f"Error in analytics loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(30)

    async def _update_metric_aggregates(self) -> None:
        """Update metric aggregates for performance analysis."""
        try:
            for metric_name, history in self.metrics_history.items():
                if not history:
                    continue
                    
                # Get recent values (last hour)
                cutoff_time = datetime.now(UTC) - timedelta(hours=1)
                recent_values = [
                    m.value for m in history 
                    if m.timestamp > cutoff_time
                ]
                
                if not recent_values:
                    continue
                
                # Calculate aggregates
                aggregates = {
                    'mean': statistics.mean(recent_values),
                    'median': statistics.median(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'count': len(recent_values)
                }
                
                if len(recent_values) > 1:
                    aggregates['stdev'] = statistics.stdev(recent_values)
                    
                self.metric_aggregates[metric_name] = aggregates
                
        except Exception as e:
            self.logger.error(
                f"Error updating metric aggregates: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _perform_trend_analysis(self) -> None:
        """Perform basic trend analysis on metrics."""
        try:
            for metric_name, history in self.metrics_history.items():
                if len(history) < 10:  # Need sufficient data points
                    continue
                    
                # Get recent trend (last 10 data points)
                recent_values = [m.value for m in list(history)[-10:]]
                
                # Simple linear trend detection
                if len(recent_values) >= 2:
                    # Calculate simple slope
                    x_values = list(range(len(recent_values)))
                    slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    
                    self.trend_analysis[metric_name] = recent_values
                    
                    # Log significant trends
                    if abs(slope) > self.config.get('trend_threshold', 1.0):
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        self.logger.info(
                            f"Trend detected in {metric_name}: {trend_direction} (slope: {slope:.2f})",
                            source_module=self._source_module
                        )
                        
        except Exception as e:
            self.logger.error(
                f"Error in trend analysis: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _detect_anomalies(self) -> None:
        """Basic anomaly detection using statistical methods."""
        try:
            for metric_name, aggregates in self.metric_aggregates.items():
                if 'stdev' not in aggregates or aggregates['count'] < 30:
                    continue
                    
                # Get latest value
                if metric_name not in self.metrics_history:
                    continue
                    
                history = self.metrics_history[metric_name]
                if not history:
                    continue
                    
                latest_value = history[-1].value
                mean_value = aggregates['mean']
                stdev_value = aggregates['stdev']
                
                # Simple z-score based anomaly detection
                if stdev_value > 0:
                    z_score = abs(latest_value - mean_value) / stdev_value
                    
                    if z_score > self.config.get('anomaly_threshold', 3.0):
                        self.logger.warning(
                            f"Anomaly detected in {metric_name}: value={latest_value:.2f}, "
                            f"mean={mean_value:.2f}, z-score={z_score:.2f}",
                            source_module=self._source_module
                        )
                        
                        # Create anomaly alert if not already active
                        await self._create_anomaly_alert(metric_name, latest_value, z_score)
                        
        except Exception as e:
            self.logger.error(
                f"Error in anomaly detection: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def _create_anomaly_alert(self, metric_name: str, value: float, z_score: float) -> None:
        """Create an alert for detected anomaly."""
        anomaly_rule_name = f"{metric_name}_anomaly"
        
        if anomaly_rule_name in self.active_alerts:
            return  # Already have active anomaly alert for this metric
            
        alert = Alert(
            alert_id=f"anomaly_{metric_name}_{int(time.time())}",
            name=anomaly_rule_name,
            condition="anomaly_detected",
            severity=AlertSeverity.WARNING,
            message=f"Anomaly detected in {metric_name}: value={value:.2f}, z-score={z_score:.2f}",
            threshold=z_score,
            current_value=value,
            triggered_at=datetime.now(UTC),
            metadata={
                'type': 'anomaly',
                'metric_name': metric_name,
                'z_score': z_score
            }
        )
        
        self.active_alerts[anomaly_rule_name] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(
            f"ANOMALY ALERT: {alert.message}",
            source_module=self._source_module
        )
        
        # Persist anomaly alert to database
        await self._persist_alert_to_database(alert)

    async def _persist_alert_to_database(self, alert: Alert) -> None:
        """Persist alert to PostgreSQL database."""
        try:
            if not hasattr(self, '_pg_pool') or not self._pg_pool:
                return  # Skip if PostgreSQL not available
            
            async with self._pg_pool.acquire() as conn:
                # Insert or update alert in database
                await conn.execute("""
                    INSERT INTO alerts (
                        alert_id, name, condition, severity, message, threshold,
                        current_value, status, triggered_at, resolved_at,
                        acknowledged_at, acknowledged_by, escalation_level, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (alert_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        resolved_at = EXCLUDED.resolved_at,
                        acknowledged_at = EXCLUDED.acknowledged_at,
                        acknowledged_by = EXCLUDED.acknowledged_by,
                        escalation_level = EXCLUDED.escalation_level,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """, 
                    alert.alert_id,
                    alert.name,
                    alert.condition,
                    alert.severity.value,
                    alert.message,
                    alert.threshold,
                    alert.current_value,
                    alert.status.value,
                    alert.triggered_at,
                    alert.resolved_at,
                    alert.acknowledged_at,
                    alert.acknowledged_by,
                    alert.escalation_level,
                    json.dumps(alert.metadata)
                )
                
                self.logger.debug(
                    f"Persisted alert {alert.alert_id} to PostgreSQL",
                    source_module=self._source_module
                )
                
        except Exception as e:
            self.logger.error(
                f"Failed to persist alert to PostgreSQL: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    def _load_alert_rules(self) -> None:
        """Load alert rules from configuration."""
        try:
            rules_config = self.config.get('alert_rules', {})
            
            for rule_name, rule_config in rules_config.items():
                alert_rule = AlertRule(
                    name=rule_name,
                    metric_name=rule_config['metric_name'],
                    condition=rule_config['condition'],
                    threshold=rule_config['threshold'],
                    severity=AlertSeverity(rule_config.get('severity', 'warning')),
                    message_template=rule_config.get(
                        'message_template', 
                        f"Alert {rule_name}: {{metric}} = {{value}} (threshold: {{threshold}})"
                    ),
                    enabled=rule_config.get('enabled', True),
                    cooldown_seconds=rule_config.get('cooldown_seconds', 300),
                    notification_channels=rule_config.get('notification_channels', ['log'])
                )
                
                self.alert_rules[rule_name] = alert_rule
                
            self.logger.info(
                f"Loaded {len(self.alert_rules)} alert rules",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error loading alert rules: {e}",
                source_module=self._source_module,
                exc_info=True
            )

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of collected metrics and system status."""
        try:
            # Recent metric values
            recent_metrics = {}
            for name, history in self.metrics_history.items():
                if history:
                    latest = history[-1]
                    recent_metrics[name] = {
                        'latest_value': latest.value,
                        'timestamp': latest.timestamp.isoformat(),
                        'count': len(history),
                        'labels': latest.labels
                    }
            
            # Active alerts summary
            alerts_summary = {
                'active_count': len(self.active_alerts),
                'by_severity': {},
                'recent_alerts': []
            }
            
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                alerts_summary['by_severity'][severity] = alerts_summary['by_severity'].get(severity, 0) + 1
            
            # Recent alerts (last 10)
            alerts_summary['recent_alerts'] = [
                alert.to_dict() for alert in list(self.alert_history)[-10:]
            ]
            
            summary = {
                'collection_stats': self.collection_stats,
                'system_status': {
                    'running': self._running,
                    'tasks_running': {
                        'collection': self._collection_task and not self._collection_task.done(),
                        'alerting': self._alerting_task and not self._alerting_task.done(),
                        'analytics': self._analytics_task and not self._analytics_task.done()
                    }
                },
                'metrics': {
                    'total_unique_metrics': len(self.metrics_history),
                    'buffer_size': len(self.metrics_buffer),
                    'recent_metrics': recent_metrics
                },
                'alerts': alerts_summary,
                'performance': {
                    'metric_aggregates_count': len(self.metric_aggregates),
                    'trend_analysis_metrics': len(self.trend_analysis)
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(
                f"Error generating metrics summary: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {'error': str(e)}

    async def acknowledge_alert(self, alert_name: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert.
        
        Args:
            alert_name: Name of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        if alert_name in self.active_alerts:
            alert = self.active_alerts[alert_name]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(UTC)
            alert.acknowledged_by = acknowledged_by
            
            # Persist acknowledgment to database
            await self._persist_alert_to_database(alert)
            
            self.logger.info(
                f"Alert acknowledged: {alert_name} by {acknowledged_by}",
                source_module=self._source_module
            )
            return True
            
        return False

    def get_metric_history(self, metric_name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get historical data for a specific metric.
        
        Args:
            metric_name: Name of metric to retrieve
            limit: Maximum number of data points to return
            
        Returns:
            List of metric data points
        """
        if metric_name not in self.metrics_history:
            return []
            
        history = list(self.metrics_history[metric_name])[-limit:]
        return [metric.to_dict() for metric in history]


class MetricsCollectionError(Exception):
    """Exception raised for metrics collection errors."""
    pass


class MonitoringService:
    """Monitors the overall system health and manages the global HALT state.

    Triggers HALT based on predefined conditions (e.g., max drawdown) or manual requests.
    Publishes system state changes (HALTED/RUNNING) via the PubSubManager.
    """

    def __init__(
        self,
        config_manager: "ConfigManager",
        pubsub_manager: "PubSubManager",
        portfolio_manager: PortfolioManager,
        logger_service: LoggerService,
        execution_handler: "ExecutionHandler | None" = None,
        halt_coordinator: Optional["HaltCoordinator"] = None,
        history_repo: "HistoryRepository | None" = None,
    ) -> None:
        """Initialize the MonitoringService.

        Args:
        ----
            config_manager: The application's configuration manager instance.
            pubsub_manager: The application's publish/subscribe manager instance.
            portfolio_manager: The application's portfolio manager instance.
            logger_service: The shared logger instance.
            execution_handler: Optional execution handler for API connectivity checks.
            halt_coordinator: Optional HALT coordinator for centralized HALT management.
        """
        self.config_manager = config_manager
        self.pubsub_manager = pubsub_manager
        self._portfolio_manager = portfolio_manager
        self.logger = logger_service
        self._execution_handler = execution_handler
        self.history_repo = history_repo
        self._source = self.__class__.__name__

        # Initialize HALT coordinator if not provided
        if halt_coordinator is None:
            try:
                from .core.halt_coordinator import HaltCoordinator as RealHaltCoordinator
                self._halt_coordinator = RealHaltCoordinator(
                    config_manager=config_manager,
                    pubsub_manager=pubsub_manager,
                    logger_service=logger_service,
                )
            except ImportError:
                # Fall back to placeholder if real implementation not available
                self._halt_coordinator = HaltCoordinator(
                    config_manager=config_manager,
                    pubsub_manager=pubsub_manager,
                    logger_service=logger_service,
                )
        else:
            self._halt_coordinator = halt_coordinator

        self._is_halted: bool = False
        self._periodic_check_task: asyncio.Task | None = None

        # Add startup time tracking
        self._service_start_time: datetime | None = None

        # Handler storage for unsubscribing
        self._potential_halt_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._market_data_l2_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._market_data_ohlcv_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._execution_report_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None
        self._api_error_handler: Callable[[Any], Coroutine[Any, Any, None]] | None = None

        # State for tracking additional monitoring metrics
        self._last_market_data_times: dict[str, datetime] = {}  # pair -> timestamp
        self._consecutive_api_failures: int = 0
        self._consecutive_losses: int = 0
        self._recent_api_errors: deque[float] = deque(
            maxlen=10,
        )  # Store timestamps of recent errors

        # Load configuration values instead of hardcoded constants
        self._load_configuration()

        # Initialize Enterprise-Grade Metrics Collection System
        metrics_config = self.config_manager.get("monitoring", {}).get("metrics_collection", {})
        self._metrics_system = MetricsCollectionSystem(
            config=metrics_config,
            logger=logger_service
        )

        self.logger.info("MonitoringService initialized with enterprise metrics collection.", source_module=self._source)

    def is_halted(self) -> bool:
        """Return whether the system is currently halted."""
        return self._is_halted

    async def start(self) -> None:
        """Start the periodic monitoring checks and metrics collection."""
        # Record when the service actually started
        self._service_start_time = datetime.now(UTC)
        self.logger.info(
            "MonitoringService started at %s",
            self._service_start_time,
            source_module=self._source,
        )

        # Start enterprise metrics collection system
        try:
            await self._metrics_system.start()
            self.logger.info("Enterprise metrics collection system started", source_module=self._source)
        except Exception as e:
            self.logger.error(
                f"Failed to start metrics collection system: {e}",
                source_module=self._source,
                exc_info=True
            )

        # Publish initial state when starting, if not already halted
        if not self._is_halted:
            await self._publish_state_change(
                "RUNNING",
                "System startup",
                "MonitoringService Start",
            )

        if self._periodic_check_task and not self._periodic_check_task.done():
            self.logger.warning(
                "MonitoringService periodic check task already running.",
                source_module=self._source,
            )
            return

        msg = f"Starting MonitoringService periodic checks every {self._check_interval} seconds."
        self.logger.info(msg, source_module=self._source)

        self._periodic_check_task = asyncio.create_task(self._run_periodic_checks())

        # Subscribe to potential halt events
        self._potential_halt_handler = self._handle_potential_halt_trigger
        self.pubsub_manager.subscribe(
            EventType.POTENTIAL_HALT_TRIGGER,
            self._potential_halt_handler,
        )
        self.logger.info(
            "Subscribed to POTENTIAL_HALT_TRIGGER events.",
            source_module=self._source,
        )

        # Subscribe to API error events
        self._api_error_handler = self._handle_api_error
        self.pubsub_manager.subscribe(EventType.SYSTEM_ERROR, self._api_error_handler)
        self.logger.info(
            "Subscribed to SYSTEM_ERROR events for API error tracking.",
            source_module=self._source,
        )

        # Subscribe to market data events to track freshness
        self._market_data_l2_handler = self._update_market_data_timestamp
        self._market_data_ohlcv_handler = self._update_market_data_timestamp
        self.pubsub_manager.subscribe(EventType.MARKET_DATA_L2, self._market_data_l2_handler)
        self.pubsub_manager.subscribe(EventType.MARKET_DATA_OHLCV, self._market_data_ohlcv_handler)
        self.logger.info(
            "Subscribed to market data events for freshness tracking.",
            source_module=self._source,
        )

        # Subscribe to execution reports to track consecutive losses
        self._execution_report_handler = self._handle_execution_report
        self.pubsub_manager.subscribe(EventType.EXECUTION_REPORT, self._execution_report_handler)
        self.logger.info(
            "Subscribed to execution reports for loss tracking.",
            source_module=self._source,
        )

    async def stop(self) -> None:
        """Stop the periodic monitoring checks and metrics collection."""
        # Stop enterprise metrics collection system first
        try:
            await self._metrics_system.stop()
            self.logger.info("Enterprise metrics collection system stopped", source_module=self._source)
        except Exception as e:
            self.logger.error(
                f"Error stopping metrics collection system: {e}",
                source_module=self._source,
                exc_info=True
            )

        # Unsubscribe from all event types
        try:
            # Potential HALT trigger events
            if self._potential_halt_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.POTENTIAL_HALT_TRIGGER,
                    self._potential_halt_handler,
                )
                self.logger.info(
                    "Unsubscribed from POTENTIAL_HALT_TRIGGER events.",
                    source_module=self._source,
                )
                self._potential_halt_handler = None

            # API error events
            if hasattr(self, "_api_error_handler") and self._api_error_handler:
                self.pubsub_manager.unsubscribe(EventType.SYSTEM_ERROR, self._api_error_handler)
                self.logger.info(
                    "Unsubscribed from SYSTEM_ERROR events.",
                    source_module=self._source,
                )
                self._api_error_handler = None

            # Market data events
            if self._market_data_l2_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.MARKET_DATA_L2,
                    self._market_data_l2_handler,
                )
                self._market_data_l2_handler = None

            if self._market_data_ohlcv_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.MARKET_DATA_OHLCV,
                    self._market_data_ohlcv_handler,
                )
                self._market_data_ohlcv_handler = None

            # Execution report events
            if self._execution_report_handler:
                self.pubsub_manager.unsubscribe(
                    EventType.EXECUTION_REPORT,
                    self._execution_report_handler,
                )
                self._execution_report_handler = None

            self.logger.info(
                "Unsubscribed from all monitoring events.",
                source_module=self._source,
            )
        except Exception:
            self.logger.exception(
                "Error unsubscribing from events",
                source_module=self._source,
            )

        if self._periodic_check_task and not self._periodic_check_task.done():
            self.logger.info(
                "Stopping MonitoringService periodic checks...",
                source_module=self._source,
            )
            self._periodic_check_task.cancel()
            try:
                await self._periodic_check_task
            except asyncio.CancelledError:
                self.logger.info(
                    "Monitoring check task successfully cancelled.",
                    source_module=self._source,
                )
            except Exception:
                self.logger.exception(
                    "Error encountered while stopping monitoring task.",
                    source_module=self._source,
                )
            finally:
                self._periodic_check_task = None
        else:
            self.logger.info(
                "MonitoringService periodic check task was not running.",
                source_module=self._source,
            )

    async def trigger_halt(self, reason: str, source: str) -> None:
        """Halt the system operations.

        Args:
        ----
            reason: The reason for halting the system.
            source: The source triggering the halt (e.g., 'MANUAL', 'AUTO: Max Drawdown').
        """
        if self._is_halted:
            self.logger.warning(
                "System already halted. Ignoring HALT trigger from %s.",
                source,
                source_module=self._source,
            )
            return

        self._is_halted = True

        # Update HALT coordinator state
        self._halt_coordinator.set_halt_state(
            is_halted=True,
            reason=reason,
            source=source,
        )

        self.logger.critical(
            "SYSTEM HALTED by %s. Reason: %s",
            source,
            reason,
            source_module=self._source,
        )
        await self._publish_state_change("HALTED", reason, source)

        # Handle positions based on configuration
        await self._handle_positions_on_halt()

    async def _handle_positions_on_halt(self) -> None:
        """Process existing positions according to the configured HALT behavior.

        Can close positions, maintain them, or perform other actions.
        """
        halt_behavior = self._halt_position_behavior
        self.logger.info(
            "HALT triggered. Position behavior set to: %s",
            halt_behavior,
            source_module=self._source,
        )

        if halt_behavior in {"close", "liquidate"}:
            self.logger.warning(
                "Attempting to close all open positions due to HALT.",
                source_module=self._source,
            )
            try:
                # Get current positions from portfolio manager
                current_state = self._portfolio_manager.get_current_state()
                open_positions = current_state.get("positions", {})

                if not open_positions:
                    self.logger.info(
                        "No open positions found to close during HALT.",
                        source_module=self._source,
                    )
                    return

                for pair, pos_data in open_positions.items():
                    # Extract quantity from position
                    qty_str = pos_data.get("quantity")
                    if not qty_str:
                        continue

                    try:
                        qty = Decimal(str(qty_str))
                    except Exception:
                        self.logger.warning(
                            "Could not convert position quantity to Decimal: %s",
                            qty_str,
                            source_module=self._source,
                        )
                        continue

                    if abs(qty) > Decimal("1e-12"):  # Check if position exists (non-zero)
                        close_side = "SELL" if qty > Decimal("0") else "BUY"
                        self.logger.info(
                            "Requesting closure of %s position (%s %s)",
                            pair,
                            close_side,
                            abs(qty),
                            source_module=self._source,
                        )

                        # For future implementation: Create and publish a close position command
                        self.logger.info(
                            "Creating ClosePositionCommand for %s: %s %s",
                            pair,
                            close_side,
                            abs(qty),
                            source_module=self._source,
                        )
                        close_command = ClosePositionCommand(
                            timestamp=datetime.now(UTC),
                            event_id=uuid.uuid4(),
                            source_module=self._source,
                            trading_pair=pair,
                            quantity=abs(qty),
                            side=close_side,
                        )
                        await self.pubsub_manager.publish(close_command)

            except Exception:
                self.logger.exception(
                    "Error during attempt to close positions on HALT",
                    source_module=self._source,
                )
        elif halt_behavior == "maintain":
            self.logger.info(
                "Maintaining existing positions during HALT as per configuration.",
                source_module=self._source,
            )
        else:
            self.logger.warning(
                "Unknown halt position behavior configured: %s. Maintaining positions.",
                halt_behavior,
                source_module=self._source,
            )

    async def trigger_resume(self, source: str) -> None:
        """Resume system operations after a HALT.

        Args:
        ----
            source: The source triggering the resume (e.g., 'MANUAL').
        """
        if not self._is_halted:
            self.logger.warning(
                "System not halted. Ignoring RESUME trigger from %s.",
                source,
                source_module=self._source,
            )
            return

        self._is_halted = False

        # Clear HALT coordinator state
        self._halt_coordinator.clear_halt_state()

        self.logger.info(
            "SYSTEM RESUMED by %s.",
            source,
            source_module=self._source,
        )
        await self._publish_state_change("RUNNING", "Manual resume", source)

    async def _publish_state_change(self, new_state: str, reason: str, source: str) -> None:
        """Publish a SystemStateEvent through the PubSubManager.

        Args:
        ----
            new_state: The new system state ("HALTED" or "RUNNING").
            reason: The reason for the state change.
            source: The source triggering the state change.
        """
        try:
            # Create a proper SystemStateEvent with correct parameters
            event = SystemStateEvent(
                source_module=source,
                event_id=uuid.uuid4(),
                timestamp=datetime.now().replace(microsecond=0),
                new_state=new_state,
                reason=reason,
            )
            # Correct publish method call - only passing the event
            await self.pubsub_manager.publish(event)
            self.logger.debug(
                "Published SYSTEM_STATE_CHANGE event: %s - %s",
                new_state,
                reason,
                source_module=self._source,
            )
        except Exception:
            self.logger.exception(
                "Failed to publish SYSTEM_STATE_CHANGE event",
                source_module=self._source,
            )

    async def _handle_potential_halt_trigger(self, event: "PotentialHaltTriggerEvent") -> None:
        """Handle events that suggest a potential HALT condition.

        Args:
        ----
            event: The PotentialHaltTriggerEvent containing halt trigger information.
        """
        if not isinstance(event, PotentialHaltTriggerEvent):
            self.logger.warning(
                "Received non-PotentialHaltTriggerEvent: %s",
                type(event),
                source_module=self._source,
            )
            return

        warning_msg = (
            f"Potential HALT condition received from {event.source_module}: {event.reason}"
        )
        self.logger.warning(warning_msg, source_module=self._source)
        # Trigger actual halt - might add confirmation logic later
        await self.trigger_halt(reason=event.reason, source=event.source_module)

    async def _run_periodic_checks(self) -> None:
        """Execute the core background task performing periodic checks.

        This method runs at regular intervals defined by the check_interval configuration.
        """
        self.logger.info(
            "MonitoringService periodic check task started.",
            source_module=self._source,
        )
        while True:
            try:
                await asyncio.sleep(self._check_interval)

                if not self._is_halted:
                    self.logger.debug(
                        "Running periodic checks...",
                        source_module=self._source,
                    )
                    # Comprehensive check of all HALT conditions
                    await self._check_all_halt_conditions()

            except asyncio.CancelledError:
                self.logger.info(
                    "MonitoringService periodic check task cancelled.",
                    source_module=self._source,
                )
                break
            except Exception:
                self.logger.exception(
                    "Unhandled error during periodic monitoring check. Continuing...",
                    source_module=self._source,
                )
                # Avoid tight loop on unexpected errors
                await asyncio.sleep(self._check_interval)

    async def _check_all_halt_conditions(self) -> None:
        """Comprehensive check of all HALT conditions."""
        # 1. Drawdown checks
        await self._check_drawdown_conditions()

        # 2. Market volatility checks
        await self._check_market_volatility()

        # 3. System health checks
        await self._check_system_health()

        # 4. API connectivity checks
        await self._check_api_connectivity()

        # 5. Data freshness checks
        await self._check_market_data_freshness()

        # 6. Position risk checks
        await self._check_position_risk()

        # Check if any conditions are triggered
        triggered_conditions = self._halt_coordinator.check_all_conditions()
        if triggered_conditions:
            # Build comprehensive reason from all triggered conditions
            reasons = [
                f"{c.name}: {c.current_value} > {c.threshold}"
                for c in triggered_conditions
            ]
            combined_reason = "; ".join(reasons)
            await self.trigger_halt(
                reason=f"Multiple HALT conditions triggered: {combined_reason}",
                source="AUTO: Multiple Conditions",
            )

    async def _check_drawdown_conditions(self) -> None:
        """Check all drawdown-related conditions and collect metrics."""
        try:
            current_state = self._portfolio_manager.get_current_state()

            # Total drawdown
            total_dd = current_state.get("total_drawdown_pct", Decimal("0"))
            if not isinstance(total_dd, Decimal):
                total_dd = Decimal(str(total_dd))

            # Collect total drawdown metric
            await self.collect_metric(
                "portfolio.drawdown.total_pct",
                float(abs(total_dd)),
                labels={"type": "total_drawdown"}
            )

            # Update condition in coordinator
            if self._halt_coordinator.update_condition("max_total_drawdown", abs(total_dd)):
                await self.trigger_halt(
                    reason=f"Maximum total drawdown exceeded: {abs(total_dd):.2f}%",
                    source="AUTO: Max Drawdown",
                )

            # Daily drawdown
            daily_dd = current_state.get("daily_drawdown_pct", Decimal("0"))
            if not isinstance(daily_dd, Decimal):
                daily_dd = Decimal(str(daily_dd))

            # Collect daily drawdown metric
            await self.collect_metric(
                "portfolio.drawdown.daily_pct",
                float(abs(daily_dd)),
                labels={"type": "daily_drawdown"}
            )

            if self._halt_coordinator.update_condition("max_daily_drawdown", abs(daily_dd)):
                await self.trigger_halt(
                    reason=f"Maximum daily drawdown exceeded: {abs(daily_dd):.2f}%",
                    source="AUTO: Daily Drawdown",
                )

            # Consecutive losses
            consecutive_losses = self._consecutive_losses
            
            # Collect consecutive losses metric
            await self.collect_metric(
                "trading.consecutive_losses",
                consecutive_losses,
                labels={"type": "risk_tracking"}
            )

            if self._halt_coordinator.update_condition(
                "max_consecutive_losses", consecutive_losses,
            ):
                await self.trigger_halt(
                    reason=f"Maximum consecutive losses reached: {consecutive_losses}",
                    source="AUTO: Consecutive Losses",
                )

            # Collect portfolio state metrics
            portfolio_metrics = [
                {"name": "portfolio.total_equity", "value": float(current_state.get("total_equity", 0))},
                {"name": "portfolio.available_balance", "value": float(current_state.get("available_balance", 0))},
                {"name": "portfolio.unrealized_pnl", "value": float(current_state.get("total_unrealized_pnl", 0))},
                {"name": "portfolio.positions_count", "value": len(current_state.get("positions", {}))},
            ]
            await self.collect_batch_metrics(portfolio_metrics)

        except Exception:
            self.logger.exception(
                "Error checking drawdown conditions",
                source_module=self._source,
            )

    async def _check_system_health(self) -> None:
        """Check system resource health."""
        await self._check_system_resources()

    async def _check_position_risk(self) -> None:
        """Check position-specific risk metrics.
        
        Monitors individual position sizes, concentration risk, and triggers
        automated position reduction or closure when risk thresholds are breached.
        """
        self.logger.debug("Running periodic check for position-specific risks.", source_module=self._source)

        # 1. Fetch Current Portfolio State
        try:
            current_positions = await self._get_all_open_positions()
            portfolio_summary = await self._get_portfolio_summary()
            total_portfolio_value = portfolio_summary.get("total_equity", 0)
        except Exception as e:
            self.logger.error(
                "Failed to fetch position or portfolio data for risk check: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )
            return  # Cannot proceed without position data

        if not current_positions:
            self.logger.debug("No open positions to check for risk.", source_module=self._source)
            return

        # 2. Load Position Risk Configuration
        position_risk_config = self.config_manager.get("monitoring", {}).get("position_risk_checks", {})
        global_max_pos_pct_config = position_risk_config.get("max_single_position_percentage_of_portfolio", {})
        global_max_pos_notional_usd_config = position_risk_config.get("max_position_notional_value_usd", {})
        specific_pair_limits_config = position_risk_config.get("specific_pair_limits", {})

        # 3. Iterate Through Each Open Position and Check Risks
        for position in current_positions:
            trading_pair = position.get("trading_pair")
            position_value_usd = position.get("current_market_value_usd", 0)
            position_base_quantity = position.get("quantity", 0)

            if not trading_pair:
                continue

            # Convert to Decimal for precise calculations
            try:
                position_value_usd = Decimal(str(position_value_usd))
                position_base_quantity = Decimal(str(position_base_quantity))
                total_portfolio_value_decimal = Decimal(str(total_portfolio_value))
            except (ValueError, TypeError):
                self.logger.warning(
                    "Could not convert position values to Decimal for %s",
                    trading_pair,
                    source_module=self._source,
                )
                continue

            # 3.1. Check: Position Size as Percentage of Total Portfolio
            if total_portfolio_value_decimal > 0:
                position_pct_of_portfolio = position_value_usd / total_portfolio_value_decimal
                warning_thresh_pct = global_max_pos_pct_config.get("warning_threshold", 0.20)
                action_thresh_pct = global_max_pos_pct_config.get("action_threshold")

                if position_pct_of_portfolio > warning_thresh_pct:
                    alert_details = {
                        "trading_pair": trading_pair,
                        "metric": "position_percentage_of_portfolio",
                        "value": float(position_pct_of_portfolio),
                        "warning_threshold": warning_thresh_pct,
                        "action_threshold": action_thresh_pct,
                        "position_value_usd": float(position_value_usd),
                        "total_portfolio_value_usd": float(total_portfolio_value_decimal),
                    }

                    self.logger.warning(
                        "Position Risk Alert: %s (%.2f%%) exceeds warning portfolio percentage (%.2f%%)",
                        trading_pair,
                        position_pct_of_portfolio * 100,
                        warning_thresh_pct * 100,
                        source_module=self._source,
                    )

                    await self._publish_position_risk_alert(alert_details, "WARNING")

                    if action_thresh_pct is not None and position_pct_of_portfolio > action_thresh_pct:
                        self.logger.critical(
                            "Position Risk Breach: %s (%.2f%%) exceeds ACTION portfolio percentage (%.2f%%). Initiating reduction.",
                            trading_pair,
                            position_pct_of_portfolio * 100,
                            action_thresh_pct * 100,
                            source_module=self._source,
                        )

                        reduction_pct = global_max_pos_pct_config.get("reduction_percentage")
                        if reduction_pct is not None:
                            await self._initiate_position_reduction(
                                position=position,
                                reduction_type="PERCENTAGE_OF_CURRENT",
                                reduction_value=Decimal(str(reduction_pct)),
                                reason="EXCEEDED_MAX_PORTFOLIO_PERCENTAGE_LIMIT",
                                breach_details=alert_details,
                            )

            # 3.2. Check: Position Notional Value (Absolute USD Limit)
            warn_thresh_notional = global_max_pos_notional_usd_config.get("warning_threshold")
            action_thresh_notional = global_max_pos_notional_usd_config.get("action_threshold")

            if warn_thresh_notional is not None and position_value_usd > Decimal(str(warn_thresh_notional)):
                alert_details = {
                    "trading_pair": trading_pair,
                    "metric": "position_notional_value_usd",
                    "value": float(position_value_usd),
                    "warning_threshold": warn_thresh_notional,
                    "action_threshold": action_thresh_notional,
                }

                self.logger.warning(
                    "Position Risk Alert: %s value ($%.2f) exceeds warning notional value ($%.2f)",
                    trading_pair,
                    position_value_usd,
                    warn_thresh_notional,
                    source_module=self._source,
                )

                await self._publish_position_risk_alert(alert_details, "WARNING")

                if action_thresh_notional is not None and position_value_usd > Decimal(str(action_thresh_notional)):
                    self.logger.critical(
                        "Position Risk Breach: %s value ($%.2f) exceeds ACTION notional value ($%.2f). Initiating reduction.",
                        trading_pair,
                        position_value_usd,
                        action_thresh_notional,
                        source_module=self._source,
                    )

                    reduction_target_notional = global_max_pos_notional_usd_config.get("reduction_target_notional_value")
                    if reduction_target_notional is not None:
                        await self._initiate_position_reduction(
                            position=position,
                            reduction_type="NOTIONAL_TARGET",
                            reduction_value=Decimal(str(reduction_target_notional)),
                            reason="EXCEEDED_MAX_NOTIONAL_VALUE_LIMIT",
                            breach_details=alert_details,
                        )

            # 3.3. Check: Specific Pair Limits (if configured)
            pair_specific_config = specific_pair_limits_config.get(trading_pair, {})
            base_qty_limits = pair_specific_config.get("max_base_qty", {})
            warn_thresh_base_qty = base_qty_limits.get("warning_threshold")
            action_thresh_base_qty = base_qty_limits.get("action_threshold")

            if warn_thresh_base_qty is not None and abs(position_base_quantity) > Decimal(str(warn_thresh_base_qty)):
                alert_details = {
                    "trading_pair": trading_pair,
                    "metric": "position_base_quantity",
                    "value": float(abs(position_base_quantity)),
                    "warning_threshold": warn_thresh_base_qty,
                    "action_threshold": action_thresh_base_qty,
                    "asset": trading_pair.split("/")[0] if "/" in trading_pair else trading_pair,
                }

                self.logger.warning(
                    "Position Risk Alert: %s quantity (%.6f) exceeds specific pair warning base quantity (%.6f)",
                    trading_pair,
                    abs(position_base_quantity),
                    warn_thresh_base_qty,
                    source_module=self._source,
                )

                await self._publish_position_risk_alert(alert_details, "WARNING")

                if action_thresh_base_qty is not None and abs(position_base_quantity) > Decimal(str(action_thresh_base_qty)):
                    self.logger.critical(
                        "Position Risk Breach: %s quantity (%.6f) exceeds specific pair ACTION base quantity (%.6f). Initiating reduction.",
                        trading_pair,
                        abs(position_base_quantity),
                        action_thresh_base_qty,
                        source_module=self._source,
                    )

                    reduction_qty_val = base_qty_limits.get("reduction_qty")
                    if reduction_qty_val is not None:
                        await self._initiate_position_reduction(
                            position=position,
                            reduction_type="QUANTITY",
                            reduction_value=Decimal(str(reduction_qty_val)),
                            reason="EXCEEDED_PAIR_MAX_BASE_QUANTITY_LIMIT",
                            breach_details=alert_details,
                        )

        self.logger.debug("Position risk check completed.", source_module=self._source)

    async def _get_all_open_positions(self) -> list[dict]:
        """Get all open positions from portfolio manager."""
        try:
            current_state = self._portfolio_manager.get_current_state()
            positions_dict = current_state.get("positions", {})

            # Convert positions dict to list of position objects
            positions = []
            for pair, pos_data in positions_dict.items():
                if pos_data.get("quantity", 0) != 0:  # Only include non-zero positions
                    position = {
                        "trading_pair": pair,
                        "quantity": pos_data.get("quantity", 0),
                        "current_market_value_usd": pos_data.get("market_value_usd", 0),
                        **pos_data,  # Include all other position data
                    }
                    positions.append(position)
            return positions
        except Exception as e:
            self.logger.error(
                "Failed to get open positions: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )
            return []

    async def _get_portfolio_summary(self) -> dict:
        """Get portfolio summary from portfolio manager."""
        try:
            current_state = self._portfolio_manager.get_current_state()
            return {
                "total_equity": current_state.get("total_equity", 0),
                "available_balance": current_state.get("available_balance", 0),
                "total_unrealized_pnl": current_state.get("total_unrealized_pnl", 0),
            }
        except Exception as e:
            self.logger.error(
                "Failed to get portfolio summary: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )
            return {"total_equity": 0, "available_balance": 0, "total_unrealized_pnl": 0}

    async def _publish_position_risk_alert(self, alert_details: dict, severity: str) -> None:
        """Publish a position risk alert event."""
        try:
            # Create a position risk alert event (would need to be defined in events.py)
            alert_event = {
                "timestamp": datetime.now(UTC),
                "event_id": uuid.uuid4(),
                "source_module": self._source,
                "alert_type": "POSITION_RISK",
                "severity": severity,
                "details": alert_details,
            }

            self.logger.info(
                "Publishing position risk alert for %s: %s",
                alert_details.get("trading_pair"),
                alert_details.get("metric"),
                source_module=self._source,
            )

            # In a real implementation, this would publish a proper PositionRiskAlertEvent
            # await self.pubsub_manager.publish(PositionRiskAlertEvent(**alert_event))

        except Exception as e:
            self.logger.error(
                "Failed to publish position risk alert: %s",
                e,
                source_module=self._source,
                exc_info=True,
            )

    async def _initiate_position_reduction(
        self,
        position: dict,
        reduction_type: str,
        reduction_value: Decimal,
        reason: str,
        breach_details: dict,
    ) -> None:
        """Initiate position reduction by publishing a ReducePositionCommand."""
        trading_pair = position.get("trading_pair")
        current_quantity = Decimal(str(position.get("quantity", 0)))
        quantity_to_reduce = Decimal(0)

        if reduction_type == "PERCENTAGE_OF_CURRENT":
            quantity_to_reduce = abs(current_quantity) * reduction_value
        elif reduction_type == "QUANTITY":
            quantity_to_reduce = reduction_value
        elif reduction_type == "NOTIONAL_TARGET":
            # Convert target notional value to target quantity
            current_price = await self._get_current_market_price(trading_pair)
            if current_price is None:
                self.logger.error(
                    "Cannot get current market price for %s. Unable to calculate NOTIONAL_TARGET reduction.",
                    trading_pair,
                    source_module=self._source,
                )
                return
            
            # Calculate target quantity from notional value
            target_quantity = reduction_value / current_price
            
            # Quantity to reduce is the difference between current and target
            if abs(current_quantity) > target_quantity:
                quantity_to_reduce = abs(current_quantity) - target_quantity
            else:
                self.logger.info(
                    "Current position %s (%.6f) is already below target notional (%.6f = $%.2f @ $%.4f). No reduction needed.",
                    trading_pair,
                    abs(current_quantity),
                    target_quantity,
                    reduction_value,
                    current_price,
                    source_module=self._source,
                )
                return
        else:
            self.logger.error(
                "Unknown reduction_type: %s for %s",
                reduction_type,
                trading_pair,
                source_module=self._source,
            )
            return

        if quantity_to_reduce <= Decimal(0):
            self.logger.info(
                "Calculated reduction quantity for %s is zero or negative (%.6f). No action taken.",
                trading_pair,
                quantity_to_reduce,
                source_module=self._source,
            )
            return

        # Ensure reduction doesn't exceed current position size
        quantity_to_reduce = min(quantity_to_reduce, abs(current_quantity))

        self.logger.info(
            "Attempting to reduce position %s by %.6f (Type: %s, Value: %.6f). Reason: %s",
            trading_pair,
            quantity_to_reduce,
            reduction_type,
            reduction_value,
            reason,
            source_module=self._source,
        )

        command_id = uuid.uuid4()
        timestamp = datetime.now(UTC)

        # Determine order type for reduction
        reduction_order_type = self.config_manager.get("monitoring", {}).get("position_risk_checks", {}).get("default_reduction_order_type", "MARKET")

        try:
            # Create reduce position command (would need to be defined in events.py)
            reduce_command = {
                "command_id": command_id,
                "timestamp": timestamp,
                "source_module": self._source,
                "trading_pair": trading_pair,
                "quantity_to_reduce": float(quantity_to_reduce),
                "order_type_preference": reduction_order_type,
                "reason": f"AUTOMATED_RISK_REDUCTION: {reason}",
                "metadata": {
                    "breach_details": breach_details,
                    "reduction_type": reduction_type,
                    "reduction_value_config": str(reduction_value),
                },
            }

            self.logger.info(
                "Successfully created ReducePositionCommand (%s) for %s to reduce by %.6f",
                str(command_id)[:8],
                trading_pair,
                quantity_to_reduce,
                source_module=self._source,
            )

            # In a real implementation, this would publish a proper ReducePositionCommand
            # await self.pubsub_manager.publish(ReducePositionCommand(**reduce_command))

        except Exception as e:
            self.logger.critical(
                "Failed to create/publish ReducePositionCommand (%s) for %s. Position reduction failed. Error: %s",
                str(command_id)[:8],
                trading_pair,
                e,
                source_module=self._source,
                exc_info=True,
            )

    async def _check_system_health(self) -> None:
        """Check system resource health."""
        await self._check_system_resources()

    async def _check_drawdown(self) -> None:
        """Check if the maximum total portfolio drawdown has been exceeded.

        Retrieves the current drawdown percentage from the portfolio manager
        and compares it against the configured maximum drawdown threshold.
        """
        try:
            # PortfolioManager.get_current_state() needs to be synchronous per design doc
            # If it becomes async, this needs adjustment (e.g., run_in_executor)
            # For now, assuming it's sync as requested for MVP.
            current_state = self._portfolio_manager.get_current_state()
            drawdown_pct = current_state.get("total_drawdown_pct")

            if drawdown_pct is None:
                self.logger.warning(
                    "Could not retrieve 'total_drawdown_pct' from PortfolioManager state.",
                    source_module=self._source,
                )
                return

            # Ensure drawdown_pct is Decimal
            if not isinstance(drawdown_pct, Decimal):
                try:
                    drawdown_pct = Decimal(drawdown_pct)
                except Exception:
                    self.logger.warning(
                        "Invalid type for 'total_drawdown_pct': %s. Skipping check.",
                        type(drawdown_pct),
                        source_module=self._source,
                    )
                    return

            self.logger.debug(
                "Current total drawdown: %.2f%% (Limit: %s%%)",
                drawdown_pct,
                self._max_total_drawdown_pct,
                source_module=self._source,
            )

            # Check if drawdown exceeds the limit (absolute value)
            if abs(drawdown_pct) > self._max_total_drawdown_pct:
                drawdown_val = abs(drawdown_pct)
                limit_val = self._max_total_drawdown_pct
                reason = f"Max total drawdown limit exceeded: {drawdown_val:.2f}% > {limit_val}%"
                self.logger.warning(reason, source_module=self._source)
                await self.trigger_halt(reason=reason, source="AUTO: Max Drawdown")

        except Exception:
            self.logger.exception(
                "Error occurred during drawdown check.",
                source_module=self._source,
            )

    async def _check_api_connectivity(self) -> None:
        """Check connectivity to Kraken API and collect connectivity metrics.

        Triggers HALT if consecutive failures exceed the threshold.
        """
        if not self._execution_handler:
            self.logger.warning(
                "No execution handler available for API connectivity check.",
                source_module=self._source,
            )
            # Collect metric indicating no execution handler
            await self.collect_metric(
                "api.connectivity.execution_handler_available",
                0,
                labels={"status": "unavailable"}
            )
            return

        api_check_start = time.time()
        try:
            # Attempt real API connectivity check
            success = False
            
            # Try to get account balance as a lightweight authenticated API check
            try:
                if hasattr(self._execution_handler, 'get_account_balance'):
                    balance_result = await self._execution_handler.get_account_balance()
                    success = balance_result is not None
                elif hasattr(self._execution_handler, 'check_api_status'):
                    success = await self._execution_handler.check_api_status()
                elif hasattr(self._execution_handler, 'get_server_time'):
                    # Fallback to server time check if available
                    server_time = await self._execution_handler.get_server_time()
                    success = server_time is not None
                else:
                    # If no suitable method exists, check if the handler is properly initialized
                    success = self._execution_handler is not None and hasattr(self._execution_handler, '__dict__')
                    
            except Exception as api_error:
                self.logger.debug(
                    f"API connectivity check failed with error: {api_error}",
                    source_module=self._source
                )
                success = False
                
            api_response_time = (time.time() - api_check_start) * 1000  # Convert to milliseconds

            # Collect API response time metric
            await self.collect_metric(
                "api.connectivity.response_time_ms",
                api_response_time,
                labels={"endpoint": "status_check"}
            )

            if success:
                self._consecutive_api_failures = 0  # Reset on success
                self.logger.debug("API connectivity check passed.", source_module=self._source)
                
                # Collect successful connectivity metrics
                await self.collect_batch_metrics([
                    {"name": "api.connectivity.status", "value": 1, "labels": {"status": "success"}},
                    {"name": "api.connectivity.consecutive_failures", "value": self._consecutive_api_failures},
                    {"name": "api.connectivity.health_score", "value": 100.0}
                ])
            else:
                self._consecutive_api_failures += 1
                warning_msg = (
                    "API connectivity check failed "
                    f"({self._consecutive_api_failures}/"
                    f"{self._api_failure_threshold})"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )

                # Collect failed connectivity metrics
                health_score = max(0, 100 - (self._consecutive_api_failures * 20))
                await self.collect_batch_metrics([
                    {"name": "api.connectivity.status", "value": 0, "labels": {"status": "failure"}},
                    {"name": "api.connectivity.consecutive_failures", "value": self._consecutive_api_failures},
                    {"name": "api.connectivity.health_score", "value": health_score}
                ])

                if self._consecutive_api_failures >= self._api_failure_threshold:
                    reason = (
                        f"API connectivity failed "
                        f"{self._consecutive_api_failures} consecutive times."
                    )
                    self.logger.error(reason, source_module=self._source)
                    
                    # Collect critical connectivity failure metric
                    await self.collect_metric(
                        "api.connectivity.critical_failure",
                        1,
                        labels={"reason": "consecutive_failures_exceeded"}
                    )
                    
                    await self.trigger_halt(reason=reason, source="AUTO: API Connectivity")

        except Exception as e:
            self._consecutive_api_failures += 1
            api_response_time = (time.time() - api_check_start) * 1000
            
            self.logger.exception(
                "Error during API connectivity check",
                source_module=self._source,
            )

            # Collect error metrics
            await self.collect_batch_metrics([
                {"name": "api.connectivity.status", "value": 0, "labels": {"status": "error"}},
                {"name": "api.connectivity.consecutive_failures", "value": self._consecutive_api_failures},
                {"name": "api.connectivity.response_time_ms", "value": api_response_time, "labels": {"endpoint": "status_check"}},
                {"name": "api.connectivity.error_count", "value": 1, "labels": {"error_type": type(e).__name__}}
            ])

            if self._consecutive_api_failures >= self._api_failure_threshold:
                reason = (
                    f"API connectivity check errors: "
                    f"{self._consecutive_api_failures} consecutive failures."
                )
                
                # Collect critical error metric
                await self.collect_metric(
                    "api.connectivity.critical_failure",
                    1,
                    labels={"reason": "consecutive_errors_exceeded"}
                )
                
                await self.trigger_halt(reason=reason, source="AUTO: API Connectivity")

    async def _check_market_data_freshness(self) -> None:
        """Check if market data timestamps are recent enough.

        Triggers HALT if data for active pairs is stale beyond threshold.
        Uses startup time tracking to provide grace period during system initialization.
        """
        now = datetime.now(UTC)
        stale_pairs = []
        potentially_stale_awaiting_initial_data = []

        if not self._active_pairs:
            self.logger.warning(
                "No active trading pairs configured for market data freshness check.",
                source_module=self._source,
            )
            return

        # Determine system uptime for startup grace period
        if self._service_start_time is None:
            self.logger.warning(
                "Service start time not recorded. Staleness check might be unreliable during initial startup phase.",
                source_module=self._source,
            )
            system_uptime_seconds = float("inf")  # Effectively disables startup grace period
        else:
            system_uptime_seconds = (now - self._service_start_time).total_seconds()

        for pair in self._active_pairs:
            last_ts = self._last_market_data_times.get(pair)

            if last_ts is None:
                # Case 1: No data ever received for this pair
                if system_uptime_seconds < self._data_staleness_threshold_s:
                    # Startup grace period is active for this pair as no data has been seen yet
                    self.logger.info(
                        "Awaiting initial market data for active pair %s. System uptime: %.2fs.",
                        pair,
                        system_uptime_seconds,
                        source_module=self._source,
                    )
                    potentially_stale_awaiting_initial_data.append(pair)
                    # Do NOT add to stale_pairs yet
                else:
                    # Startup grace period has passed, and still no data. This is a concern.
                    self.logger.warning(
                        "No market data received for active pair %s after initial grace period (%.2fs). Marking as stale.",
                        pair,
                        system_uptime_seconds,
                        source_module=self._source,
                    )
                    stale_pairs.append(pair)  # Now it's considered genuinely stale
            elif (now - last_ts).total_seconds() > self._data_staleness_threshold_s:
                # Case 2: Data was received, but it's now older than the staleness threshold
                stale_pairs.append(pair)
                warning_msg = (
                    f"Market data for {pair} is stale (last update: {last_ts}, "
                    f"threshold: {self._data_staleness_threshold_s}s, current age: {(now - last_ts).total_seconds():.2f}s)"
                )
                self.logger.warning(warning_msg, source_module=self._source)
            else:
                # Data is present and not stale
                self.logger.debug(f"Market data for {pair} is current (last update: {last_ts}).", source_module=self._source)

        if stale_pairs:
            self.logger.info(f"Identified stale pairs: {stale_pairs}", source_module=self._source)
            reason = f"Market data stale for pairs: {', '.join(stale_pairs)}"
            await self.trigger_halt(reason=reason, source="AUTO: Market Data Staleness")

        if potentially_stale_awaiting_initial_data:
            self.logger.info(
                f"Pairs awaiting initial data (within startup grace period): {potentially_stale_awaiting_initial_data}",
                source_module=self._source,
            )

    async def _check_system_resources(self) -> None:
        """Monitor CPU and Memory usage.

        Logs warnings when thresholds are approached, triggers HALT at critical levels.
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=None)  # Non-blocking
            mem_usage = psutil.virtual_memory().percent

            self.logger.debug(
                "System Resources: CPU=%.1f%%, Memory=%.1f%%",
                cpu_usage,
                mem_usage,
                source_module=self._source,
            )

            # Check CPU usage
            if cpu_usage > self._cpu_threshold_pct:
                warning_msg = (
                    "High CPU usage detected: "
                    f"{cpu_usage:.1f}% "
                    f"(Threshold: {self._cpu_threshold_pct}%)"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )
                # Only trigger HALT on extremely high CPU usage that would impact trading
                if cpu_usage > self._cpu_threshold_pct + 5:  # Extra 5% buffer
                    reason = f"Critical CPU usage: {cpu_usage:.1f}%"
                    await self.trigger_halt(reason=reason, source="AUTO: System Resources")

            # Check memory usage
            if mem_usage > self._memory_threshold_pct:
                warning_msg = (
                    "High Memory usage detected: "
                    f"{mem_usage:.1f}% "
                    f"(Threshold: {self._memory_threshold_pct}%)"
                )
                self.logger.warning(
                    warning_msg,
                    source_module=self._source,
                )
                # Only trigger HALT on extremely high memory usage that would impact trading
                if mem_usage > self._memory_threshold_pct + 5:  # Extra 5% buffer
                    reason = f"Critical Memory usage: {mem_usage:.1f}%"
                    await self.trigger_halt(reason=reason, source="AUTO: System Resources")

        except Exception:
            self.logger.exception(
                "Error checking system resources",
                source_module=self._source,
            )

    async def _check_market_volatility(self) -> None:
        """Check for excessive market volatility.

        Triggers HALT if volatility exceeds configured thresholds.
        """
        try:
            for pair in self._active_pairs:
                # Calculate rolling volatility (would need price history)
                volatility = await self._calculate_volatility(pair)

                if (
                    volatility is not None and
                    self._halt_coordinator.update_condition("max_volatility", volatility)
                ):
                    reason = (
                        f"Market volatility for {pair} ({volatility:.2f}%) "
                        f"exceeds threshold"
                    )
                    await self.trigger_halt(
                        reason=reason,
                        source="AUTO: Market Volatility",
                    )
                    break

        except Exception:
            self.logger.exception(
                "Error checking market volatility",
                source_module=self._source,
            )

    async def _calculate_volatility(self, pair: str) -> Decimal | None:
        """Calculate rolling volatility for a trading pair.

        Supports both standard deviation and GARCH volatility calculation methods.
        The method used is determined by the 'volatility_calculation.method' configuration.

        Args:
            pair: Trading pair to calculate volatility for

        Returns:
            Decimal: Annualized volatility percentage, or None if insufficient data
        """
        self.logger.debug(f"Calculating volatility for {pair}.", source_module=self._source)

        vol_config = self.config_manager.get("monitoring", {}).get("volatility_calculation", {})
        calculation_method = vol_config.get("method", "stddev").lower()

        if calculation_method == "garch":
            self.logger.debug(f"Using GARCH method for volatility calculation for {pair}.", source_module=self._source)
            return await self._calculate_garch_volatility_internal(pair, vol_config)
        if calculation_method == "stddev":
            self.logger.debug(f"Using standard deviation method for volatility calculation for {pair}.", source_module=self._source)
            return await self._calculate_stddev_volatility_internal(pair, vol_config)
        self.logger.error(
            f"Unknown volatility calculation method configured: {calculation_method}. Defaulting to None.",
            source_module=self._source,
        )
        return None

    async def _calculate_stddev_volatility_internal(self, trading_pair: str, vol_config: dict) -> Decimal | None:
        """Calculate standard deviation volatility for a trading pair.
        
        Args:
            trading_pair: Trading pair to calculate volatility for
            vol_config: Volatility calculation configuration
            
        Returns:
            Decimal: Annualized volatility percentage, or None if insufficient data
        """
        self.logger.debug(f"Calculating stddev volatility for {trading_pair}.", source_module=self._source)

        window_size = vol_config.get("stddev_window_size_candles", 100)
        candle_interval_minutes = vol_config.get("candle_interval_minutes", 60)
        min_required_data_points = vol_config.get("stddev_min_data_points_for_calc", int(window_size * 0.8))
        use_log_returns = vol_config.get("use_log_returns", True)
        annualization_factor_config = vol_config.get("annualization_periods_per_year")

        # Calculate annualization factor
        if annualization_factor_config is None:
            if candle_interval_minutes == 1440:  # Daily
                periods_per_year = 365
            elif candle_interval_minutes == 60:  # Hourly
                periods_per_year = 365 * 24
            elif candle_interval_minutes == 1:  # Minute
                periods_per_year = 365 * 24 * 60
            else:
                self.logger.warning(
                    f"Unsupported candle_interval_minutes ({candle_interval_minutes}) for default annualization factor. "
                    "Volatility will not be annualized correctly without explicit 'annualization_periods_per_year' config.",
                    source_module=self._source,
                )
                periods_per_year = 1  # Effectively no annualization
            annualization_factor = (periods_per_year) ** 0.5
        else:
            annualization_factor = (annualization_factor_config) ** 0.5

        try:
            # This would need to be implemented to fetch historical candles
            # For now, simulating the call
            price_history_candles = await self._get_historical_candles_for_volatility(
                trading_pair=trading_pair,
                num_candles=window_size + 1,
                interval_minutes=candle_interval_minutes,
            )

            if price_history_candles is None or len(price_history_candles) < min_required_data_points + 1:
                self.logger.warning(
                    f"StdDev Vol: Insufficient historical price data for {trading_pair}. "
                    f"Required: {min_required_data_points + 1}, Got: {len(price_history_candles) if price_history_candles else 0}.",
                    source_module=self._source,
                )
                return None

            closing_prices = [Decimal(str(candle.get("close", 0))) for candle in price_history_candles]
        except Exception as e:
            self.logger.error(
                f"StdDev Vol: Failed to fetch/process price history for {trading_pair}: {e}",
                source_module=self._source,
                exc_info=True,
            )
            return None

        # Convert to numpy for calculations
        import numpy as np
        np_closing_prices = np.array([float(p) for p in closing_prices])

        if use_log_returns:
            if np.any(np_closing_prices <= 0):
                self.logger.error(f"StdDev Vol: Invalid prices for {trading_pair} for log returns.", source_module=self._source)
                return None
            returns = np.log(np_closing_prices[1:] / np_closing_prices[:-1])
        else:
            returns = (np_closing_prices[1:] - np_closing_prices[:-1]) / np_closing_prices[:-1]

        if len(returns) == 0:
            self.logger.warning(f"StdDev Vol: No returns calculated for {trading_pair}.", source_module=self._source)
            return None

        std_dev_returns = np.std(returns)
        annualized_volatility_float = std_dev_returns * annualization_factor
        annualized_volatility_decimal = Decimal(str(annualized_volatility_float)) * Decimal("100")

        self.logger.info(
            f"StdDev Vol for {trading_pair}: {annualized_volatility_decimal:.2f}%",
            source_module=self._source,
        )
        return annualized_volatility_decimal.quantize(Decimal("0.0001"))

    async def _calculate_garch_volatility_internal(self, trading_pair: str, vol_config: dict) -> Decimal | None:
        """Calculate GARCH volatility for a trading pair.
        
        Args:
            trading_pair: Trading pair to calculate volatility for
            vol_config: Volatility calculation configuration
            
        Returns:
            Decimal: Annualized volatility percentage, or None if insufficient data or GARCH unavailable
        """
        self.logger.debug(
            f"Calculating GARCH volatility for {trading_pair}.", source_module=self._source
        )

        try:
            from arch import arch_model
        except ImportError:  # pragma: no cover - dependency missing
            self.logger.warning(
                f"GARCH volatility calculation for {trading_pair} requires 'arch' library. "
                "Falling back to standard deviation method.",
                source_module=self._source,
            )
            return await self._calculate_stddev_volatility_internal(trading_pair, vol_config)

        window_size = vol_config.get("garch_window_size_candles", 200)
        p = vol_config.get("garch_p", 1)
        q = vol_config.get("garch_q", 1)
        distribution = vol_config.get("garch_distribution", "normal")
        candle_interval_minutes = vol_config.get("candle_interval_minutes", 60)
        annualization_periods_per_year = vol_config.get("annualization_periods_per_year")

        if annualization_periods_per_year is None:
            if candle_interval_minutes == 1440:
                periods_per_year = 365
            elif candle_interval_minutes == 60:
                periods_per_year = 365 * 24
            elif candle_interval_minutes == 1:
                periods_per_year = 365 * 24 * 60
            else:
                periods_per_year = 1
            annualization_factor = periods_per_year**0.5
        else:
            annualization_factor = (annualization_periods_per_year) ** 0.5

        candles = await self._get_historical_candles_for_volatility(
            trading_pair=trading_pair,
            num_candles=window_size + 1,
            interval_minutes=candle_interval_minutes,
        )

        if candles is None or len(candles) < window_size + 1:
            self.logger.warning(
                f"GARCH Vol: insufficient data for {trading_pair}", source_module=self._source
            )
            return None

        import numpy as np

        prices = np.array([float(c["close"]) for c in candles])
        if np.any(prices <= 0):
            self.logger.error(
                f"GARCH Vol: non-positive prices for {trading_pair}", source_module=self._source
            )
            return None
        returns = np.log(prices[1:] / prices[:-1]) * 100

        try:
            model = arch_model(returns, p=p, q=q, dist=distribution, rescale=False)
            res = model.fit(disp="off")
            forecast = res.forecast(horizon=1)
            sigma = float(forecast.variance.iloc[-1, 0]) ** 0.5
            annualized = Decimal(str((sigma / 100) * annualization_factor * 100))
            return annualized.quantize(Decimal("0.0001"))
        except Exception as exc:  # pragma: no cover - model issues
            self.logger.error(
                f"GARCH volatility calculation failed for {trading_pair}: {exc}",
                source_module=self._source,
                exc_info=True,
            )
            return None

    async def _get_historical_candles_for_volatility(
        self,
        trading_pair: str,
        num_candles: int,
        interval_minutes: int,
    ) -> list[dict] | None:
        """Get historical candles for volatility calculation."""
        if self.history_repo is None:
            self.logger.debug(
                "History repository not configured for volatility calculation", source_module=self._source
            )
            return None

        interval = f"{interval_minutes}m"
        try:
            df = await self.history_repo.get_recent_ohlcv(trading_pair, num_candles, interval)
        except Exception as exc:  # pragma: no cover - fetch failures
            self.logger.error(
                f"Failed to fetch historical candles for {trading_pair}: {exc}",
                source_module=self._source,
                exc_info=True,
            )
            return None

        if df is None or df.empty:
            return None

        result = []
        for ts, row in df.iterrows():
            result.append(
                {
                    "timestamp": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )
        return result

    async def _get_current_market_price(self, trading_pair: str) -> Decimal | None:
        """Get current market price for a trading pair.
        
        Attempts to get current price from various sources in order of preference:
        1. Portfolio manager (if it has mark-to-market pricing)
        2. Recent market data events stored in memory
        3. Execution handler (if it has price lookup capability)
        
        Args:
            trading_pair: Trading pair to get price for (e.g., "BTC/USD")
            
        Returns:
            Current market price as Decimal, or None if unable to determine price
        """
        try:
            # Method 1: Try to get price from portfolio manager
            try:
                current_state = self._portfolio_manager.get_current_state()
                positions = current_state.get("positions", {})
                
                if trading_pair in positions:
                    position_data = positions[trading_pair]
                    
                    # Try to extract current price from position market value
                    market_value = position_data.get("market_value_usd", 0)
                    quantity = position_data.get("quantity", 0)
                    
                    if float(quantity) != 0 and float(market_value) != 0:
                        # Calculate implied price from market value and quantity
                        # Note: This assumes USD-denominated market value
                        implied_price = Decimal(str(abs(float(market_value)))) / Decimal(str(abs(float(quantity))))
                        
                        self.logger.debug(
                            f"Got market price for {trading_pair} from portfolio position: ${implied_price:.4f}",
                            source_module=self._source
                        )
                        return implied_price
            except Exception as e:
                self.logger.debug(
                    f"Could not get price from portfolio manager for {trading_pair}: {e}",
                    source_module=self._source
                )
            
            # Method 2: Try to get price from recent market data events
            # This would require storing price information from market data events
            # For now, we don't have a direct storage mechanism for this
            
            # Method 3: Try to get price from execution handler if it has price lookup
            if self._execution_handler:
                try:
                    # Check if execution handler has a method to get current price
                    if hasattr(self._execution_handler, 'get_current_price'):
                        price = await self._execution_handler.get_current_price(trading_pair)
                        if price is not None:
                            price_decimal = Decimal(str(price))
                            self.logger.debug(
                                f"Got market price for {trading_pair} from execution handler: ${price_decimal:.4f}",
                                source_module=self._source
                            )
                            return price_decimal
                    
                    # Alternative: Try to get ticker data if available
                    if hasattr(self._execution_handler, 'get_ticker'):
                        ticker_data = await self._execution_handler.get_ticker(trading_pair)
                        if ticker_data and 'last_price' in ticker_data:
                            price_decimal = Decimal(str(ticker_data['last_price']))
                            self.logger.debug(
                                f"Got market price for {trading_pair} from ticker: ${price_decimal:.4f}",
                                source_module=self._source
                            )
                            return price_decimal
                    
                    # Alternative: Try to get from orderbook if available
                    if hasattr(self._execution_handler, 'get_orderbook'):
                        orderbook = await self._execution_handler.get_orderbook(trading_pair)
                        if orderbook and 'bid' in orderbook and 'ask' in orderbook:
                            bid_price = Decimal(str(orderbook['bid']))
                            ask_price = Decimal(str(orderbook['ask']))
                            mid_price = (bid_price + ask_price) / Decimal('2')
                            
                            self.logger.debug(
                                f"Got market price for {trading_pair} from orderbook mid: ${mid_price:.4f} (bid: ${bid_price:.4f}, ask: ${ask_price:.4f})",
                                source_module=self._source
                            )
                            return mid_price
                            
                except Exception as e:
                    self.logger.debug(
                        f"Could not get price from execution handler for {trading_pair}: {e}",
                        source_module=self._source
                    )
            
            # Method 4: Fallback - use a simple price estimation or API call
            # This could be enhanced to call external price APIs as a last resort
            
            self.logger.warning(
                f"Unable to determine current market price for {trading_pair} from any available source",
                source_module=self._source
            )
            return None
            
        except Exception as e:
            self.logger.error(
                f"Error getting current market price for {trading_pair}: {e}",
                source_module=self._source,
                exc_info=True
            )
            return None

    async def _handle_execution_report(self, event: "ExecutionReportEvent") -> None:
        """Handle execution report events to track consecutive losses.

        Triggers HALT if consecutive loss limit is reached.

        Args:
        ----
            event: An ExecutionReportEvent
        """
        try:
            # Check for filled order with realized PnL
            if (
                hasattr(event, "order_status")
                and event.order_status == "FILLED"
                and hasattr(event, "realized_pnl")
            ):
                # Convert to Decimal if needed
                pnl = event.realized_pnl
                if not isinstance(pnl, Decimal):
                    try:
                        pnl = Decimal(str(pnl))
                    except Exception:
                        self.logger.warning(
                            "Could not convert realized_pnl to Decimal: %s",
                            pnl,
                            source_module=self._source,
                        )
                        return

                # Update consecutive losses counter
                if pnl < Decimal("0"):
                    self._consecutive_losses += 1
                    warning_msg = (
                        f"Trade loss detected: {pnl}. "
                        f"Consecutive losses: {self._consecutive_losses}"
                    )
                    self.logger.warning(
                        warning_msg,
                        source_module=self._source,
                    )

                    # Check if we've hit the consecutive loss limit
                    if self._consecutive_losses >= self._consecutive_loss_limit:
                        reason = f"Consecutive loss limit reached: {self._consecutive_losses}"
                        await self.trigger_halt(reason=reason, source="AUTO: Consecutive Losses")
                else:
                    # Reset counter on profitable trade
                    if self._consecutive_losses > 0:
                        info_msg = (
                            f"Profitable trade resets consecutive loss counter "
                            f"(was {self._consecutive_losses})"
                        )
                        self.logger.info(
                            info_msg,
                            source_module=self._source,
                        )
                    self._consecutive_losses = 0
        except Exception:
            self.logger.exception(
                "Error handling execution report",
                source_module=self._source,
            )

    async def _handle_api_error(self, event: "APIErrorEvent") -> None:
        """Count and evaluate API errors to detect excessive error rates.

        Triggers HALT if error frequency exceeds threshold.

        Args:
        ----
            event: An APIErrorEvent
        """
        try:
            now = time.time()
            self._recent_api_errors.append(now)

            # Check if we've exceeded the error threshold within the time window
            error_window = now - self._api_error_threshold_period_s
            errors_in_period = sum(1 for t in self._recent_api_errors if t > error_window)

            warning_msg = (
                f"API error received: {event.error_message}. "
                f"{errors_in_period} errors in the last "
                f"{self._api_error_threshold_period_s}s"
            )
            self.logger.warning(
                warning_msg,
                source_module=self._source,
            )

            if errors_in_period >= self._api_error_threshold_count:
                reason = (
                    f"High frequency of API errors: {errors_in_period} "
                    f"in {self._api_error_threshold_period_s}s"
                )
                await self.trigger_halt(reason=reason, source="AUTO: API Errors")
        except Exception:
            self.logger.exception(
                "Error handling API error event",
                source_module=self._source,
            )

    def _load_configuration(self) -> None:
        """Load configuration values from ConfigManager."""
        monitoring_config = self.config_manager.get("monitoring", {})
        risk_config = self.config_manager.get("risk", {})
        trading_config = self.config_manager.get("trading", {})

        # Main monitoring intervals and thresholds
        self._check_interval = monitoring_config.get("check_interval_seconds", 60)

        # API monitoring configuration
        self._api_failure_threshold = monitoring_config.get(
            "api_failure_threshold", 3,
        )
        self._api_error_threshold_count = monitoring_config.get(
            "api_error_threshold_count", 5,
        )
        self._api_error_threshold_period_s = monitoring_config.get(
            "api_error_threshold_period_s", 60,
        )
        self._data_staleness_threshold_s = monitoring_config.get(
            "data_staleness_threshold_s", 120.0,
        )

        # System resource monitoring configuration
        self._cpu_threshold_pct = monitoring_config.get("cpu_threshold_pct", 90.0)
        self._memory_threshold_pct = monitoring_config.get("memory_threshold_pct", 90.0)

        # Trading performance monitoring configuration
        self._consecutive_loss_limit = monitoring_config.get("consecutive_loss_limit", 5)

        # Risk management configuration
        risk_limits = risk_config.get("limits", {})
        self._max_total_drawdown_pct = Decimal(
            str(risk_limits.get("max_total_drawdown_pct", 10.0)),
        )

        # HALT behavior configuration
        halt_config = monitoring_config.get("halt", {})
        self._halt_position_behavior = halt_config.get("position_behavior", "maintain").lower()

        # Active trading pairs
        self._active_pairs = trading_config.get("pairs", [])

    async def _update_market_data_timestamp(
        self,
        event: "MarketDataL2Event | MarketDataOHLCVEvent",
    ) -> None:
        """Update the last received timestamp for market data events.

        This helps track market data freshness.

        Args:
        ----
            event: Either a MarketDataL2Event or MarketDataOHLCVEvent
        """
        try:
            # Extract pair from the event
            if hasattr(event, "trading_pair"):
                pair = event.trading_pair
            else:
                self.logger.warning(
                    "Market data event missing trading_pair attribute: %s",
                    type(event),
                    source_module=self._source,
                )
                return

            # Extract timestamp, preferring exchange timestamp if available
            if hasattr(event, "timestamp_exchange") and event.timestamp_exchange:
                ts = event.timestamp_exchange
            elif hasattr(event, "timestamp"):
                ts = event.timestamp
            else:
                self.logger.warning(
                    "Market data event missing timestamp: %s",
                    type(event),
                    source_module=self._source,
                )
                return

            # Ensure timestamp is timezone-aware UTC
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            # Update the timestamp for this pair
            self._last_market_data_times[pair] = ts
            self.logger.debug(
                "Updated market data timestamp for %s: %s",
                pair,
                ts,
                source_module=self._source,
            )

            # Collect market data freshness metric
            try:
                current_time = datetime.now(UTC)
                data_age_seconds = (current_time - ts).total_seconds()
                await self.collect_metric(
                    f"market_data.freshness.{pair}.age_seconds",
                    data_age_seconds,
                    labels={"trading_pair": pair, "data_type": type(event).__name__}
                )
            except Exception as e:
                self.logger.debug(
                    f"Error collecting market data freshness metric: {e}",
                    source_module=self._source
                )

        except Exception:
            self.logger.exception(
                "Error updating market data timestamp",
                source_module=self._source,
            )

    # ===== Enterprise Metrics Collection API =====

    async def collect_metric(
        self, 
        name: str, 
        value: float,
        labels: Optional[dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ) -> None:
        """Collect a metric through the enterprise metrics collection system.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for the metric
            metric_type: Type of metric being collected
        """
        try:
            await self._metrics_system.collect_metric(name, value, labels, metric_type)
        except Exception as e:
            self.logger.error(
                f"Error collecting metric through MonitoringService: {e}",
                source_module=self._source,
                exc_info=True
            )

    async def collect_batch_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Collect multiple metrics in batch for efficiency.
        
        Args:
            metrics: List of metric dictionaries with name, value, labels, type
        """
        try:
            await self._metrics_system.collect_batch_metrics(metrics)
        except Exception as e:
            self.logger.error(
                f"Error collecting batch metrics through MonitoringService: {e}",
                source_module=self._source,
                exc_info=True
            )

    async def get_comprehensive_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of all monitoring and metrics data.
        
        Returns:
            Dictionary containing monitoring status, metrics data, and system health
        """
        try:
            # Get metrics system summary
            metrics_summary = await self._metrics_system.get_metrics_summary()
            
            # Add monitoring service specific data
            monitoring_summary = {
                "monitoring_service": {
                    "is_halted": self._is_halted,
                    "service_start_time": self._service_start_time.isoformat() if self._service_start_time else None,
                    "consecutive_api_failures": self._consecutive_api_failures,
                    "consecutive_losses": self._consecutive_losses,
                    "active_trading_pairs": len(self._active_pairs),
                    "last_market_data_times": {
                        pair: ts.isoformat() for pair, ts in self._last_market_data_times.items()
                    }
                },
                "halt_coordinator": {
                    "active_conditions": self._halt_coordinator.check_all_conditions(),
                    "summary": self._halt_coordinator.get_stage_summary() if hasattr(self._halt_coordinator, 'get_stage_summary') else {}
                }
            }
            
            # Combine summaries
            comprehensive_summary = {
                **metrics_summary,
                **monitoring_summary,
                "timestamp": datetime.now(UTC).isoformat()
            }
            
            return comprehensive_summary
            
        except Exception as e:
            self.logger.error(
                f"Error generating comprehensive metrics summary: {e}",
                source_module=self._source,
                exc_info=True
            )
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    async def acknowledge_alert(self, alert_name: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert in the metrics system.
        
        Args:
            alert_name: Name of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
            
        Returns:
            True if alert was acknowledged, False if not found
        """
        try:
            return await self._metrics_system.acknowledge_alert(alert_name, acknowledged_by)
        except Exception as e:
            self.logger.error(
                f"Error acknowledging alert: {e}",
                source_module=self._source,
                exc_info=True
            )
            return False

    def get_metric_history(self, metric_name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get historical data for a specific metric.
        
        Args:
            metric_name: Name of metric to retrieve
            limit: Maximum number of data points to return
            
        Returns:
            List of metric data points
        """
        try:
            return self._metrics_system.get_metric_history(metric_name, limit)
        except Exception as e:
            self.logger.error(
                f"Error getting metric history: {e}",
                source_module=self._source,
                exc_info=True
            )
            return []

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get all currently active alerts.
        
        Returns:
            List of active alert dictionaries
        """
        try:
            return [alert.to_dict() for alert in self._metrics_system.active_alerts.values()]
        except Exception as e:
            self.logger.error(
                f"Error getting active alerts: {e}",
                source_module=self._source,
                exc_info=True
            )
            return []

    async def get_system_performance_snapshot(self) -> PerformanceMetrics:
        """Get current system performance snapshot.
        
        Returns:
            PerformanceMetrics object with current system state
        """
        try:
            # Collect real-time performance data
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()
            
            # Get active connections (simplified)
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                connections = 0
            
            # Create performance snapshot
            performance = PerformanceMetrics(
                timestamp=datetime.now(UTC),
                cpu_usage_pct=cpu_usage,
                memory_usage_pct=memory.percent,
                disk_usage_pct=(disk.used / disk.total) * 100,
                network_io_bytes={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv
                },
                active_connections=connections,
                response_times_ms=self._calculate_response_times(),
                error_rates=self._calculate_error_rates(),
                throughput_metrics=self._calculate_throughput_metrics()
            )
            
            # Also collect this as metrics
            await self.collect_batch_metrics([
                {"name": "performance.cpu_usage_pct", "value": performance.cpu_usage_pct},
                {"name": "performance.memory_usage_pct", "value": performance.memory_usage_pct},
                {"name": "performance.disk_usage_pct", "value": performance.disk_usage_pct},
                {"name": "performance.active_connections", "value": performance.active_connections}
            ])
            
            return performance
            
        except Exception as e:
            self.logger.error(
                f"Error creating performance snapshot: {e}",
                source_module=self._source,
                exc_info=True
            )
            # Return empty snapshot on error
            return PerformanceMetrics(
                timestamp=datetime.now(UTC),
                cpu_usage_pct=0.0,
                memory_usage_pct=0.0,
                disk_usage_pct=0.0,
                network_io_bytes={},
                active_connections=0,
                response_times_ms={},
                error_rates={},
                throughput_metrics={}
            )

    def _calculate_response_times(self) -> dict[str, float]:
        """Calculate response times from metrics history."""
        response_times = {}
        
        try:
            # Get API response time from recent metrics
            api_response_metrics = self._metrics_system.metrics_history.get("api.connectivity.response_time_ms")
            if api_response_metrics and len(api_response_metrics) > 0:
                recent_responses = list(api_response_metrics)[-10:]  # Last 10 measurements
                avg_response_time = sum(m.value for m in recent_responses) / len(recent_responses)
                response_times["api_avg_ms"] = avg_response_time
                response_times["api_latest_ms"] = recent_responses[-1].value
            else:
                response_times["api_avg_ms"] = 0.0
                response_times["api_latest_ms"] = 0.0
            
            # Calculate collection loop performance
            if hasattr(self._metrics_system, 'collection_stats') and self._metrics_system.collection_stats.get('last_collection_time'):
                collection_interval = self._metrics_system.config.get('collection_interval_seconds', 30)
                time_since_last = (datetime.now(UTC) - self._metrics_system.collection_stats['last_collection_time']).total_seconds()
                collection_delay = max(0, time_since_last - collection_interval) * 1000  # Convert to ms
                response_times["metrics_collection_delay_ms"] = collection_delay
            
        except Exception as e:
            self.logger.debug(f"Error calculating response times: {e}", source_module=self._source)
            
        return response_times

    def _calculate_error_rates(self) -> dict[str, float]:
        """Calculate error rates from monitoring data."""
        error_rates = {}
        
        try:
            # API error rate based on recent errors
            total_api_checks = max(1, self._consecutive_api_failures + 10)  # Assume some successful checks
            api_error_rate = (self._consecutive_api_failures / total_api_checks) * 100
            error_rates["api_error_rate_pct"] = min(100.0, api_error_rate)
            
            # Recent API errors rate (errors per minute)
            if self._recent_api_errors:
                current_time = time.time()
                recent_errors = [t for t in self._recent_api_errors if current_time - t < 300]  # Last 5 minutes
                errors_per_minute = (len(recent_errors) / 5.0) if recent_errors else 0.0
                error_rates["api_errors_per_minute"] = errors_per_minute
            else:
                error_rates["api_errors_per_minute"] = 0.0
            
            # Trading error rate based on consecutive losses
            if hasattr(self, '_consecutive_losses'):
                # Assume we've had at least some trades to calculate a rate
                estimated_total_trades = max(10, self._consecutive_losses + 5)
                trading_error_rate = (self._consecutive_losses / estimated_total_trades) * 100
                error_rates["trading_loss_rate_pct"] = min(100.0, trading_error_rate)
            else:
                error_rates["trading_loss_rate_pct"] = 0.0
            
            # Metrics collection error rate
            if hasattr(self._metrics_system, 'collection_stats'):
                total_collections = max(1, self._metrics_system.collection_stats.get('metrics_collected', 1))
                collection_errors = self._metrics_system.collection_stats.get('collection_errors', 0)
                collection_error_rate = (collection_errors / total_collections) * 100
                error_rates["metrics_collection_error_rate_pct"] = min(100.0, collection_error_rate)
            
        except Exception as e:
            self.logger.debug(f"Error calculating error rates: {e}", source_module=self._source)
            
        return error_rates

    def _calculate_throughput_metrics(self) -> dict[str, float]:
        """Calculate system throughput metrics."""
        throughput = {}
        
        try:
            # Market data update rate
            current_time = datetime.now(UTC)
            recent_market_updates = 0
            
            for pair, last_update in self._last_market_data_times.items():
                if (current_time - last_update).total_seconds() < 300:  # Updates in last 5 minutes
                    recent_market_updates += 1
            
            # Estimate updates per minute (rough approximation)
            market_data_rate = recent_market_updates * (60 / 300) if recent_market_updates > 0 else 0.0
            throughput["market_data_updates_per_minute"] = market_data_rate
            
            # Metrics collection rate
            if hasattr(self._metrics_system, 'collection_stats'):
                total_metrics = self._metrics_system.collection_stats.get('metrics_collected', 0)
                if self._service_start_time:
                    uptime_minutes = (current_time - self._service_start_time).total_seconds() / 60
                    if uptime_minutes > 0:
                        metrics_per_minute = total_metrics / uptime_minutes
                        throughput["metrics_collected_per_minute"] = metrics_per_minute
            
            # Trading throughput (estimated from portfolio changes)
            try:
                current_state = self._portfolio_manager.get_current_state()
                positions = current_state.get("positions", {})
                active_positions = len([p for p in positions.values() if float(p.get("quantity", 0)) != 0])
                
                # Very rough estimate: assume each position represents recent trading activity
                estimated_trades_per_hour = active_positions * 0.5  # Conservative estimate
                throughput["estimated_trades_per_hour"] = estimated_trades_per_hour
                
            except Exception:
                throughput["estimated_trades_per_hour"] = 0.0
            
            # Alert processing rate
            if hasattr(self._metrics_system, 'collection_stats'):
                total_alerts = self._metrics_system.collection_stats.get('alerts_triggered', 0)
                if self._service_start_time:
                    uptime_hours = (current_time - self._service_start_time).total_seconds() / 3600
                    if uptime_hours > 0:
                        alerts_per_hour = total_alerts / uptime_hours
                        throughput["alerts_triggered_per_hour"] = alerts_per_hour
            
        except Exception as e:
            self.logger.debug(f"Error calculating throughput metrics: {e}", source_module=self._source)
            
        return throughput
