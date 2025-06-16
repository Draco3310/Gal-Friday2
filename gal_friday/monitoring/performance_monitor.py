"""Performance monitoring for Gal-Friday.

This module tracks system performance metrics, trading performance,
and provides insights for optimization.
"""

import asyncio
import contextlib
import gc
import statistics
import time
import types
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from psutil._common import sdiskio, snetio

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import Event, EventType
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from typing import Any


class MetricType(Enum):
    """Types of performance metrics."""
    # System metrics
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    EVENT_LOOP_LAG = "event_loop_lag"

    # Application metrics
    EVENT_PROCESSING_TIME = "event_processing_time"
    API_LATENCY = "api_latency"
    DB_QUERY_TIME = "db_query_time"
    MODEL_INFERENCE_TIME = "model_inference_time"

    # Trading metrics
    ORDER_PLACEMENT_TIME = "order_placement_time"
    SIGNAL_GENERATION_TIME = "signal_generation_time"
    MARKET_DATA_LAG = "market_data_lag"
    POSITION_CALCULATION_TIME = "position_calculation_time"


@dataclass
class MetricSample:
    """Single metric sample."""
    timestamp: datetime
    value: float
    tags: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class MetricStats:
    """Aggregated metric statistics."""
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    p95: float
    p99: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> "MetricStats":
        """Calculate statistics from samples."""
        if not samples:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)

        sorted_samples = sorted(samples)
        count = len(samples)

        return cls(
            count=count,
            mean=statistics.mean(samples),
            median=statistics.median(samples),
            std_dev=statistics.stdev(samples) if count > 1 else 0,
            min_value=min(samples),
            max_value=max(samples),
            p95=sorted_samples[int(count * 0.95)] if count > 0 else 0,
            p99=sorted_samples[int(count * 0.99)] if count > 0 else 0)


class MetricCollector:
    """Collects and stores metric samples."""

    def __init__(self, window_size: int = 1000) -> None:
        """Initialize the metric collector.

        Args:
            window_size: Maximum number of samples to keep per metric.
        """
        self.window_size = window_size
        self.metrics: dict[MetricType, deque[MetricSample]] = defaultdict(
            lambda: deque(maxlen=window_size))

    def record(
        self,
        metric_type: MetricType,
        value: float,
        tags: dict[str, Any] | None = None) -> None:
        """Record a metric sample."""
        sample = MetricSample(
            timestamp=datetime.now(UTC),
            value=value,
            tags=tags or {})
        self.metrics[metric_type].append(sample)

    def get_samples(self, metric_type: MetricType,
                   since: datetime | None = None) -> list[MetricSample]:
        """Get metric samples, optionally filtered by time."""
        samples = list[Any](self.metrics[metric_type])

        if since:
            samples = [s for s in samples if s.timestamp >= since]

        return samples

    def get_stats(self, metric_type: MetricType,
                  window_minutes: int = 5) -> MetricStats:
        """Get aggregated statistics for a metric."""
        since = datetime.now(UTC) - timedelta(minutes=window_minutes)
        samples = self.get_samples(metric_type, since)
        values = [s.value for s in samples]

        return MetricStats.from_samples(values)


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricCollector,
        metric_type: MetricType,
        tags: dict[str, Any] | None = None) -> None:
        """Initialize the performance timer.

        Args:
            collector: Metric collector to record timings.
            metric_type: Type[Any] of metric being measured.
            tags: Optional tags to associate with the metric.
        """
        self.collector = collector
        self.metric_type = metric_type
        self.tags = tags or {}
        self.start_time: float | None = None

    def __enter__(self) -> "PerformanceTimer":
        """Start the performance timer.

        Returns:
            The PerformanceTimer instance.
        """
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None) -> None:
        """Stop the timer and record the duration.

        Args:
            exc_type: Exception type if an exception was raised in the context.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if self.start_time:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.collector.record(self.metric_type, duration_ms, self.tags)


class SystemMonitor:
    """Monitors system-level performance metrics."""

    def __init__(self, collector: MetricCollector) -> None:
        """Initialize the system monitor.

        Args:
            collector: Metric collector to record system metrics.
        """
        self.collector = collector
        self._process = psutil.Process()
        self._last_network_io: snetio | None = None
        self._last_disk_io: sdiskio | None = None

    async def collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        # CPU usage
        cpu_percent = self._process.cpu_percent(interval=0.1)
        self.collector.record(MetricType.CPU_USAGE, cpu_percent)

        # Memory usage
        memory_info = self._process.memory_info()
        memory_percent = self._process.memory_percent()
        self.collector.record(
            MetricType.MEMORY_USAGE,
            memory_percent,
            {"rss_mb": memory_info.rss / 1024 / 1024})

        # Disk I/O
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io:
                # Calculate read/write rates in MB/s
                read_rate = (
                    disk_io.read_bytes - self._last_disk_io.read_bytes
                ) / 1024 / 1024
                write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / 1024 / 1024
                self.collector.record(
                    MetricType.DISK_IO,
                    read_rate + write_rate,
                    {"read_mb_s": read_rate, "write_mb_s": write_rate})
            self._last_disk_io = disk_io
        except (OSError, psutil.Error) as e:
            self.collector.record(
                MetricType.DISK_IO,
                0,
                {"error": str(e), "operation": "disk_io_counters"})

        # Network I/O
        try:
            net_io = psutil.net_io_counters()
            if net_io and self._last_network_io:
                # Calculate network send/receive rates in MB/s
                sent_rate = (
                    net_io.bytes_sent - self._last_network_io.bytes_sent
                ) / 1024 / 1024
                recv_rate = (net_io.bytes_recv - self._last_network_io.bytes_recv) / 1024 / 1024
                self.collector.record(
                    MetricType.NETWORK_IO,
                    sent_rate + recv_rate,
                    {"sent_mb_s": sent_rate, "recv_mb_s": recv_rate})
            self._last_network_io = net_io
        except (OSError, psutil.Error) as e:
            self.collector.record(
                MetricType.NETWORK_IO,
                0,
                {"error": str(e), "operation": "net_io_counters"})

    async def measure_event_loop_lag(self) -> None:
        """Measure event loop responsiveness."""
        start = time.perf_counter()
        await asyncio.sleep(0)  # Yield to event loop
        lag_ms = (time.perf_counter() - start) * 1000
        self.collector.record(MetricType.EVENT_LOOP_LAG, lag_ms)


class PerformanceMonitor:
    """Main performance monitoring service."""

    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerService,
        pubsub: PubSubManager) -> None:
        """Initialize the performance monitor.

        Args:
            config: Configuration manager instance.
            logger: Logger service instance.
            pubsub: PubSub manager for event handling.
        """
        self.config = config
        self.logger = logger
        self.pubsub = pubsub
        self._source_module = self.__class__.__name__

        self.collector = MetricCollector(
            window_size=config.get_int("performance.metric_window_size", 10000))
        self.system_monitor = SystemMonitor(self.collector)

        self._monitoring_interval = config.get_float("performance.monitoring_interval", 5.0)
        self._monitoring_task: asyncio.Task[Any] | None = None

        # Performance thresholds
        self.thresholds = {
            MetricType.CPU_USAGE: config.get_float("performance.cpu_threshold", 80.0),
            MetricType.MEMORY_USAGE: config.get_float("performance.memory_threshold", 80.0),
            MetricType.EVENT_LOOP_LAG: config.get_float(
                "performance.event_loop_lag_threshold", 100.0),
            MetricType.API_LATENCY: config.get_float("performance.api_latency_threshold", 1000.0),
        }

        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Setup event handlers for performance tracking."""
        # Track event processing times
        async def track_event_processing(event: Event) -> None:
            processing_time = getattr(event, "_processing_time_ms", None)
            if processing_time:
                self.collector.record(
                    MetricType.EVENT_PROCESSING_TIME,
                    processing_time,
                    {
                        "event_type": (
                            str(event.event_type.value)
                            if hasattr(event, "event_type")
                            else "unknown"
                        ),
                    })

        # Subscribe to all events
        for event_type in EventType:
            self.pubsub.subscribe(event_type, track_event_processing)

    def timer(
        self, metric_type: MetricType, **tags: str | int | float | bool) -> PerformanceTimer:
        """Create a performance timer context manager."""
        return PerformanceTimer(self.collector, metric_type, tags)

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        **tags: str | int | float | bool) -> None:
        """Record a performance metric."""
        self.collector.record(metric_type, value, tags)

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        async def monitor_loop() -> None:
            while True:
                try:
                    # Collect system metrics
                    await self.system_monitor.collect_system_metrics()
                    await self.system_monitor.measure_event_loop_lag()

                    # Check thresholds
                    await self._check_thresholds()

                    # Force garbage collection periodically
                    if datetime.now().minute % 5 == 0:
                        gc.collect()

                    await asyncio.sleep(self._monitoring_interval)

                except asyncio.CancelledError:
                    break
                except Exception:
                    self.logger.exception(
                        "Error in performance monitoring",
                        source_module=self._source_module)
                    await asyncio.sleep(self._monitoring_interval)

        self._monitoring_task = asyncio.create_task(monitor_loop())
        self.logger.info(
            "Started performance monitoring",
            source_module=self._source_module)

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

    async def _check_thresholds(self) -> None:
        """Check if any metrics exceed thresholds."""
        alerts = []

        for metric_type, threshold in self.thresholds.items():
            stats = self.collector.get_stats(metric_type, window_minutes=1)

            if stats.count > 0 and stats.mean > threshold:
                alerts.append({
                    "metric": metric_type.value,
                    "threshold": threshold,
                    "current": stats.mean,
                    "max": stats.max_value,
                })

        if alerts:
            self.logger.warning(
                "Performance thresholds exceeded",
                source_module=self._source_module,
                context={"alerts": alerts})

    def get_performance_report(self, window_minutes: int = 60) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "window_minutes": window_minutes,
            "metrics": {},
        }

        # Collect stats for all metrics
        for metric_type in MetricType:
            stats = self.collector.get_stats(metric_type, window_minutes)
            if stats.count > 0:
                metrics_dict = report["metrics"]
                if not isinstance(metrics_dict, dict):
                    raise TypeError("metrics_dict must be a dictionary")
                metrics_dict[metric_type.value] = {
                    "count": stats.count,
                    "mean": round(stats.mean, 2),
                    "median": round(stats.median, 2),
                    "std_dev": round(stats.std_dev, 2),
                    "min": round(stats.min_value, 2),
                    "max": round(stats.max_value, 2),
                    "p95": round(stats.p95, 2),
                    "p99": round(stats.p99, 2),
                }

        # Add system info
        memory_info = psutil.virtual_memory()
        report["system"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(memory_info.total / 1024 / 1024 / 1024, 2),
            "memory_available_gb": round(memory_info.available / 1024 / 1024 / 1024, 2),
            "memory_percent": memory_info.percent,
        }

        return report

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(datetime.now(UTC).timestamp() * 1000)

        for metric_type in MetricType:
            stats = self.collector.get_stats(metric_type, window_minutes=5)
            if stats.count > 0:
                metric_name = f"galfriday_{metric_type.value}"

                # Export key percentiles
                lines.append(f"{metric_name}_mean {stats.mean} {timestamp}")
                lines.append(f"{metric_name}_p95 {stats.p95} {timestamp}")
                lines.append(f"{metric_name}_p99 {stats.p99} {timestamp}")
                lines.append(f"{metric_name}_max {stats.max_value} {timestamp}")

        return "\n".join(lines)