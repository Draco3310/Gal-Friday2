"""Health check system for Gal-Friday.

This module provides comprehensive health checks including liveness probes,
readiness probes, dependency checks, and component health monitoring.
"""

import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from http import HTTPStatus
from typing import Any

import aiohttp
import asyncpg
import psutil

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    details: dict[str, Any] | None = None
    timestamp: datetime | None = None
    duration_ms: float | None = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: list[HealthCheckResult]
    timestamp: datetime

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """Check if system is ready to serve requests."""
        # System is ready if healthy or degraded (partial functionality)
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "is_healthy": self.is_healthy,
            "is_ready": self.is_ready,
            "timestamp": self.timestamp.isoformat(),
            "checks": [check.to_dict() for check in self.checks],
        }


class HealthChecker:
    """Base class for health check implementations."""

    def __init__(self, name: str, critical: bool = True) -> None:
        """Initialize the health checker.

        Args:
            name: Name of the health check
            critical: Whether this is a critical check (fails system if check fails)
        """
        self.name = name
        self.critical = critical  # If True, failure makes system unhealthy

    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        raise NotImplementedError


class LivenessChecker(HealthChecker):
    """Basic liveness check - verifies process is responsive."""

    def __init__(self) -> None:
        """Initialize the liveness checker."""
        super().__init__("liveness", critical=True)

    async def check(self) -> HealthCheckResult:
        """Check if process is alive and responsive."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Simple check - can we run async operations?
            await asyncio.sleep(0.001)

            duration = (asyncio.get_event_loop().time() - start_time) * 1000

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Process is responsive",
                duration_ms=duration)
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Liveness check failed: {e!s}")


class MemoryChecker(HealthChecker):
    """Check system memory usage."""

    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 90.0) -> None:
        """Initialize the memory checker.

        Args:
            warning_threshold: Memory usage percentage for warning state
            critical_threshold: Memory usage percentage for critical state
        """
        super().__init__("memory", critical=False)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent

            details = {
                "usage_percent": usage_percent,
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024),
            }

            if usage_percent >= self.critical_threshold:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Critical memory usage: {usage_percent:.1f}%",
                    details=details)
            if usage_percent >= self.warning_threshold:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"High memory usage: {usage_percent:.1f}%",
                    details=details)
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"Memory usage normal: {usage_percent:.1f}%",
                details=details)

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory: {e!s}")


class CPUChecker(HealthChecker):
    """Check system CPU usage."""

    def __init__(self, warning_threshold: float = 80.0, critical_threshold: float = 95.0) -> None:
        """Initialize the CPU checker.

        Args:
            warning_threshold: CPU usage percentage for warning state
            critical_threshold: CPU usage percentage for critical state
        """
        super().__init__("cpu", critical=False)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    async def check(self) -> HealthCheckResult:
        """Check CPU usage."""
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = await asyncio.get_event_loop().run_in_executor(
                None, psutil.cpu_percent, 1)

            details = {
                "usage_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
            }

            if cpu_percent >= self.critical_threshold:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Critical CPU usage: {cpu_percent:.1f}%",
                    details=details)
            if cpu_percent >= self.warning_threshold:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    details=details)
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message=f"CPU usage normal: {cpu_percent:.1f}%",
                details=details)

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check CPU: {e!s}")


class DatabaseChecker(HealthChecker):
    """Check database connectivity and performance."""

    def __init__(self, connection_string: str, timeout: float = 5.0) -> None:
        """Initialize the database checker.

        Args:
            connection_string: Database connection string
            timeout: Connection timeout in seconds
        """
        super().__init__("database", critical=True)
        self.connection_string = connection_string
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Check database connection."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Connect to database
            conn = await asyncio.wait_for(
                asyncpg.connect(self.connection_string),
                timeout=self.timeout)

            try:
                # Run simple query
                result = await conn.fetchval("SELECT 1")

                duration = (asyncio.get_event_loop().time() - start_time) * 1000

                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    details={"query_result": result},
                    duration_ms=duration)

            finally:
                await conn.close()

        except TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection timeout ({self.timeout}s)")
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {e!s}")


class ExchangeAPIChecker(HealthChecker):
    """Check exchange API connectivity."""

    def __init__(self, api_url: str, timeout: float = 10.0) -> None:
        """Initialize the exchange API checker.

        Args:
            api_url: Base URL of the exchange API
            timeout: Request timeout in seconds
        """
        super().__init__("exchange_api", critical=True)
        self.api_url = api_url
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Check exchange API connectivity."""
        start_time = asyncio.get_event_loop().time()

        try:
            async with aiohttp.ClientSession() as session:
                # Check public endpoint (no auth required)
                url = f"{self.api_url}/0/public/Time"

                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with session.get(url, timeout=timeout) as response:
                    if response.status == HTTPStatus.OK:
                        data = await response.json()

                        duration = (asyncio.get_event_loop().time() - start_time) * 1000

                        return HealthCheckResult(
                            name=self.name,
                            status=HealthStatus.HEALTHY,
                            message="Exchange API accessible",
                            details={
                                "status_code": response.status,
                                "server_time": data.get("result", {}).get("unixtime"),
                            },
                            duration_ms=duration)
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Exchange API returned status {response.status}")

        except TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Exchange API timeout ({self.timeout}s)")
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Exchange API check failed: {e!s}")


class RedisChecker(HealthChecker):
    """Check Redis connectivity."""

    def __init__(self, redis_url: str, timeout: float = 5.0) -> None:
        """Initialize the Redis checker.

        Args:
            redis_url: Redis connection URL
            timeout: Connection timeout in seconds
        """
        super().__init__("redis", critical=False)
        self.redis_url = redis_url
        self.timeout = timeout

    async def check(self) -> HealthCheckResult:
        """Check Redis connection."""
        try:
            # Try to import aioredis, but make it optional
            try:
                import aioredis
            except ImportError:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message="Redis check skipped: aioredis not installed")

            # Use the correct aioredis API for newer versions
            redis = aioredis.from_url(
                self.redis_url,
                socket_timeout=self.timeout)

            try:
                # Ping Redis
                pong = await redis.ping()

                if pong:
                    return HealthCheckResult(
                        name=self.name,
                        status=HealthStatus.HEALTHY,
                        message="Redis connection successful")
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Redis ping failed")

            finally:
                await redis.close()

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.DEGRADED,
                message=f"Redis connection failed: {e!s}")


class ComponentChecker(HealthChecker):
    """Check individual system component health."""

    def __init__(
        self,
        name: str,
        check_func: Callable[..., Coroutine[Any, Any, bool]],
        critical: bool = True) -> None:
        """Initialize the component checker.

        Args:
            name: Name of the component
            check_func: Async function that returns True if healthy
            critical: Whether failure makes system unhealthy
        """
        super().__init__(name, critical)
        self.check_func = check_func

    async def check(self) -> HealthCheckResult:
        """Check component health."""
        try:
            is_healthy = await self.check_func()

            if is_healthy:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message=f"{self.name} is healthy")
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} is unhealthy")

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"{self.name} check failed: {e!s}")


class HealthCheckService:
    """Main health check service coordinator."""

    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize the health check service.

        Args:
            config: Configuration manager instance
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        self.checkers: list[HealthChecker] = []
        self._last_check_result: SystemHealth | None = None
        self._check_interval = config.get_int("health.check_interval_seconds", 30)
        self._periodic_check_task: asyncio.Task[Any] | None = None

        self._initialize_checkers()

    def _initialize_checkers(self) -> None:
        """Initialize health checkers based on configuration."""
        # Always include liveness check
        self.checkers.append(LivenessChecker())

        # System resource checks
        if self.config.get("health.check_memory", True):
            self.checkers.append(MemoryChecker(
                warning_threshold=self.config.get_float("health.memory_warning_threshold", 80.0),
                critical_threshold=self.config.get_float("health.memory_critical_threshold", 90.0)))

        if self.config.get("health.check_cpu", True):
            self.checkers.append(CPUChecker(
                warning_threshold=self.config.get_float("health.cpu_warning_threshold", 80.0),
                critical_threshold=self.config.get_float("health.cpu_critical_threshold", 95.0)))

        # Database check
        db_connection = self.config.get("database.connection_string")
        if db_connection:
            self.checkers.append(DatabaseChecker(db_connection))

        # Exchange API check
        exchange_url = self.config.get("exchange.api_url")
        if exchange_url:
            self.checkers.append(ExchangeAPIChecker(exchange_url))

        # Redis check
        redis_url = self.config.get("redis.url")
        if redis_url:
            self.checkers.append(RedisChecker(redis_url))

        self.logger.info(
            f"Initialized {len(self.checkers)} health checkers",
            source_module=self._source_module)

    def add_component_check(self, name: str, check_func: Callable[[], Coroutine[Any, Any, bool]],
                           critical: bool = True) -> None:
        """Add a custom component health check.

        Args:
            name: Component name
            check_func: Async function that returns True if healthy
            critical: Whether failure makes system unhealthy
        """
        self.checkers.append(ComponentChecker(name, check_func, critical))

    async def check_health(self) -> SystemHealth:
        """Perform all health checks and return system health."""
        results: list[HealthCheckResult | BaseException] = []

        # Run all checks concurrently
        check_tasks = [checker.check() for checker in self.checkers]
        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Process results
        valid_results: list[HealthCheckResult] = []
        overall_status = HealthStatus.HEALTHY

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Check failed with exception
                valid_results.append(HealthCheckResult(
                    name=self.checkers[i].name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {result!s}"))
                if self.checkers[i].critical:
                    overall_status = HealthStatus.UNHEALTHY
            elif isinstance(result, HealthCheckResult):
                valid_results.append(result)

                # Update overall status
                if result.status == HealthStatus.UNHEALTHY and self.checkers[i].critical:
                    overall_status = HealthStatus.UNHEALTHY
                elif (result.status == HealthStatus.DEGRADED and
                      overall_status == HealthStatus.HEALTHY):
                    overall_status = HealthStatus.DEGRADED

        system_health = SystemHealth(
            status=overall_status,
            checks=valid_results,
            timestamp=datetime.now(UTC))

        self._last_check_result = system_health

        # Log if unhealthy
        if not system_health.is_healthy:
            self.logger.warning(
                f"System health check failed: {overall_status.value}",
                source_module=self._source_module,
                context={"failed_checks": [
                    c.name for c in valid_results
                    if c.status == HealthStatus.UNHEALTHY
                ]})

        return system_health

    async def get_liveness(self) -> dict[str, Any]:
        """Get liveness probe result.

        Returns simple healthy/unhealthy status for Kubernetes liveness probe.
        """
        liveness_checker = LivenessChecker()
        result = await liveness_checker.check()

        return {
            "status": "ok" if result.status == HealthStatus.HEALTHY else "error",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_readiness(self) -> dict[str, Any]:
        """Get readiness probe result.

        Returns whether system is ready to handle requests.
        """
        if self._last_check_result and (
            datetime.now(UTC) - self._last_check_result.timestamp < timedelta(seconds=60)
        ):
            # Use cached result if recent
            health = self._last_check_result
        else:
            # Perform new check
            health = await self.check_health()

        return {
            "ready": health.is_ready,
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
        }

    async def start_periodic_checks(self) -> None:
        """Start periodic health checks."""
        async def run_checks() -> None:
            while True:
                try:
                    await self.check_health()
                    await asyncio.sleep(self._check_interval)
                except asyncio.CancelledError:
                    break
                except Exception:
                    self.logger.exception(
                        "Error in periodic health check",
                        source_module=self._source_module)
                    await asyncio.sleep(self._check_interval)

        self._periodic_check_task = asyncio.create_task(run_checks())
        self.logger.info(
            f"Started periodic health checks every {self._check_interval}s",
            source_module=self._source_module)

    async def stop_periodic_checks(self) -> None:
        """Stop periodic health checks."""
        if self._periodic_check_task:
            self._periodic_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._periodic_check_task
            self._periodic_check_task = None