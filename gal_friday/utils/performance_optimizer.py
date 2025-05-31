"""Performance optimization utilities for Gal-Friday."""

import asyncio
import contextlib
import functools
import gc
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast
from typing import TypeVar as TypeVarT

import psutil

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService

T = TypeVar("T")


class LRUCache(Generic[T]):
    """Thread-safe LRU cache implementation."""

    def __init__(self, maxsize: int = 128) -> None:
        """Initialize the LRU cache.

        Args:
            maxsize: Maximum number of items to store in the cache.
        """
        self.cache: OrderedDict[str, T] = OrderedDict()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        """Get item from cache."""
        async with self._lock:
            if key in self.cache:
                # Move to end to mark as recently used
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    async def set(self, key: str, value: T) -> None:
        """Set item in cache."""
        async with self._lock:
            if key in self.cache:
                # Update existing
                self.cache.move_to_end(key)
            # Add new
            elif len(self.cache) >= self.maxsize:
                # Remove oldest
                self.cache.popitem(last=False)

            self.cache[key] = value

    async def clear(self) -> None:
        """Clear cache."""
        async with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class ConnectionPool:
    """Generic connection pool with health checking."""

    def __init__(
        self,
        create_conn: Callable[[], Any],
        logger_service: LoggerService, # Added logger_service
        max_connections: int = 10,
        min_connections: int = 2,
        health_check_interval: int = 30,
    ) -> None:
        """Initialize the connection pool.

        Args:
            create_conn: Factory function to create new connections.
            logger_service: Instance of LoggerService.
            max_connections: Maximum number of connections in the pool.
            min_connections: Minimum number of connections to maintain.
            health_check_interval: Interval in seconds between health checks.
        """
        self.create_conn = create_conn
        self.logger = logger_service # Initialize self.logger
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.health_check_interval = health_check_interval

        self._available: asyncio.Queue[Any] = asyncio.Queue(maxsize=max_connections)
        self._in_use: weakref.WeakSet[Any] = weakref.WeakSet()
        self._created = 0
        self._lock = asyncio.Lock()
        self._health_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the connection pool."""
        # Create minimum connections
        for _ in range(self.min_connections):
            conn = await self.create_conn()
            await self._available.put(conn)
            self._created += 1

        # Start health check task
        self._health_task = asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        """Stop the connection pool."""
        if self._health_task:
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task

        # Close all connections
        while not self._available.empty():
            conn = await self._available.get()
            await self._close_conn(conn)

    async def acquire(self) -> object:
        """Acquire a connection from the pool."""
        try:
            # Try to get available connection
            conn = self._available.get_nowait()
        except asyncio.QueueEmpty:
            # Create new connection if under limit
            async with self._lock:
                if self._created < self.max_connections:
                    conn = await self.create_conn()
                    self._created += 1
                else:
                    # Wait for available connection
                    conn = await self._available.get()

        # Track in-use connections
        self._in_use.add(conn)
        return conn

    async def release(self, conn: object) -> None:
        """Release connection back to pool."""
        if conn in self._in_use:
            self._in_use.discard(conn)

        # Check if connection is still healthy
        if await self._is_healthy(conn):
            await self._available.put(conn)
        else:
            # Replace with new connection
            await self._close_conn(conn)
            async with self._lock:
                if self._created > 0:
                    self._created -= 1
                    if self._created < self.min_connections:
                        new_conn = await self.create_conn()
                        await self._available.put(new_conn)
                        self._created += 1

    async def _health_check_loop(self) -> None:
        """Periodically check connection health."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check available connections
                temp_conns = []
                while not self._available.empty():
                    try:
                        conn = self._available.get_nowait()
                        if await self._is_healthy(conn):
                            temp_conns.append(conn)
                        else:
                            await self._close_conn(conn)
                            async with self._lock:
                                self._created -= 1
                    except asyncio.QueueEmpty:
                        break

                # Put healthy connections back
                for conn in temp_conns:
                    await self._available.put(conn)

                # Ensure minimum connections
                async with self._lock:
                    while self._created < self.min_connections:
                        conn = await self.create_conn()
                        await self._available.put(conn)
                        self._created += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(
                    f"Health check error: {e}",
                )

    async def _is_healthy(self, conn: object) -> bool:
        """Check if connection is healthy."""
        # Override in subclass
        # For now, assume all connections are healthy
        return cast("bool", True)

    async def _close_conn(self, conn: object) -> None:
        """Close a connection."""
        # Override in subclass

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_created": self._created,
            "available": self._available.qsize(),
            "in_use": len(self._in_use),
            "max_connections": self.max_connections,
            "min_connections": self.min_connections,
        }


class QueryOptimizer:
    """SQL query optimization utilities."""

    def __init__(self, logger: LoggerService) -> None:
        """Initialize the QueryOptimizer.

        Args:
            logger: Logger service instance for logging messages.
        """
        self.logger = logger
        self._source_module = self.__class__.__name__
        self.slow_query_threshold = 1.0  # seconds
        self.query_stats: dict[str, dict[str, Any]] = {}

    async def analyze_query(
        self, query: str, params: tuple[Any, ...] | None = None,
    ) -> dict[str, Any]:
        """Analyze query performance."""
        start_time = time.time()

        # This would connect to database and run EXPLAIN
        # For now, return mock analysis
        suggestions: list[str] = []
        analysis = {
            "query": query,
            "estimated_cost": 100.0,
            "estimated_rows": 1000,
            "index_scan": True,
            "suggestions": suggestions,
        }

        # Check for common issues
        if "SELECT *" in query.upper():
            suggestions.append("Avoid SELECT *, specify needed columns")

        if "NOT IN" in query.upper():
            suggestions.append("Consider using NOT EXISTS instead of NOT IN")

        max_joins = 3  # Maximum recommended number of joins before suggesting denormalization
        if query.upper().count("JOIN") > max_joins:
            suggestions.append("Many JOINs detected, consider denormalization")

        execution_time = time.time() - start_time

        # Track query statistics
        query_key = self._normalize_query(query)
        if query_key not in self.query_stats:
            self.query_stats[query_key] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0,
                "max_time": 0,
            }

        stats = self.query_stats[query_key]
        stats["count"] += 1
        stats["total_time"] += execution_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["max_time"] = max(stats["max_time"], execution_time)

        # Log slow queries
        if execution_time > self.slow_query_threshold:
            self.logger.warning(
                f"Slow query detected: {execution_time:.2f}s",
                source_module=self._source_module,
                context={"query": query_key},
            )

        return analysis

    def _normalize_query(self, query: str) -> str:
        """Normalize query for statistics tracking."""
        # Remove extra whitespace
        normalized = " ".join(query.split())

        # Remove parameter values for grouping
        # This is simplified - real implementation would parse SQL
        import re
        normalized = re.sub(r"=\s*\$\d+", "= ?", normalized)
        normalized = re.sub(r"=\s*\d+", "= ?", normalized)
        return re.sub(r"=\s*'[^']+'", "= ?", normalized)

    def get_slow_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get slowest queries."""
        sorted_queries = sorted(
            self.query_stats.items(),
            key=lambda x: x[1]["avg_time"],
            reverse=True,
        )

        return [
            {
                "query": query,
                **stats,
            }
            for query, stats in sorted_queries[:limit]
        ]


class MemoryOptimizer:
    """Memory optimization and monitoring."""

    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize the MemoryOptimizer.

        Args:
            config: Configuration manager instance.
            logger: Logger service instance for logging messages.
        """
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Thresholds
        self.memory_limit_mb = config.get_int("performance.memory_limit_mb", 4096)
        self.gc_threshold_mb = config.get_int("performance.gc_threshold_mb", 1024)

        # State
        self._last_gc_time = time.time()
        self._gc_interval = 300  # 5 minutes

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    async def optimize_memory(self) -> dict[str, Any]:
        """Perform memory optimization."""
        usage = self.get_memory_usage()

        # Check if we need to free memory
        if usage["rss_mb"] > self.gc_threshold_mb:
            self.logger.info(
                f"Memory usage high ({usage['rss_mb']:.1f}MB), running garbage collection",
                source_module=self._source_module,
            )

            # Force garbage collection
            gc.collect()

            # Get new usage
            new_usage = self.get_memory_usage()
            freed_mb = usage["rss_mb"] - new_usage["rss_mb"]

            self.logger.info(
                f"Garbage collection freed {freed_mb:.1f}MB",
                source_module=self._source_module,
            )

            self._last_gc_time = time.time()

        return usage

    def should_run_gc(self) -> bool:
        """Check if garbage collection should run."""
        # Time-based check
        if time.time() - self._last_gc_time > self._gc_interval:
            return True

        # Memory-based check
        usage = self.get_memory_usage()
        rss_mb = cast("float", usage["rss_mb"])
        return rss_mb > self.gc_threshold_mb


class PerformanceOptimizer:
    """Main performance optimization coordinator."""

    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize the PerformanceOptimizer.

        Args:
            config: Configuration manager instance.
            logger: Logger service instance for logging messages.
        """
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Components
        self.query_optimizer = QueryOptimizer(logger)
        self.memory_optimizer = MemoryOptimizer(config, logger)

        # Caches
        self.model_cache = LRUCache[Any](maxsize=50)
        self.prediction_cache = LRUCache[Any](maxsize=1000)
        self.feature_cache = LRUCache[Any](maxsize=500)

        # Monitoring
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitor_interval = config.get_int("performance.monitor_interval", 60)

    async def start(self) -> None:
        """Start performance optimizer."""
        self.logger.info(
            "Starting performance optimizer",
            source_module=self._source_module,
        )

        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_performance())

    async def stop(self) -> None:
        """Stop performance optimizer."""
        if self._monitor_task:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task

        # Clear caches
        await self.model_cache.clear()
        await self.prediction_cache.clear()
        await self.feature_cache.clear()

    async def _monitor_performance(self) -> None:
        """Monitor and optimize performance."""
        while True:
            try:
                await asyncio.sleep(self._monitor_interval)

                # Check memory usage
                memory_usage = await self.memory_optimizer.optimize_memory()

                # Log cache statistics
                cache_stats = {
                    "model_cache": self.model_cache.get_stats(),
                    "prediction_cache": self.prediction_cache.get_stats(),
                    "feature_cache": self.feature_cache.get_stats(),
                }

                self.logger.info(
                    "Performance metrics",
                    source_module=self._source_module,
                    context={
                        "memory": memory_usage,
                        "caches": cache_stats,
                    },
                )

                # Check for issues
                if memory_usage["rss_mb"] > self.memory_optimizer.memory_limit_mb:
                    self.logger.error(
                        "Memory limit exceeded: "
                        f"{memory_usage['rss_mb']:.1f}MB > "
                        f"{self.memory_optimizer.memory_limit_mb}MB",
                        source_module=self._source_module,
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in performance monitoring",
                    source_module=self._source_module,
                )

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "memory": self.memory_optimizer.get_memory_usage(),
            "caches": {
                "model": self.model_cache.get_stats(),
                "prediction": self.prediction_cache.get_stats(),
                "feature": self.feature_cache.get_stats(),
            },
            "slow_queries": self.query_optimizer.get_slow_queries(),
            "optimizations": {
                "last_gc": time.time() - self.memory_optimizer._last_gc_time,
                "gc_threshold_mb": self.memory_optimizer.gc_threshold_mb,
                "memory_limit_mb": self.memory_optimizer.memory_limit_mb,
            },
        }


# Decorators for optimization

F = TypeVarT("F", bound=Callable[..., Any])

def cached(cache_name: str = "default", ttl: int = 300) -> Callable[[F], F]:
    """Decorator for caching function results."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(
            self: "PerformanceOptimizer", *args: object, **kwargs: object,
        ) -> object:
            # Generate cache key
            cache_key = f"{func.__name__}:{args!s}:{kwargs!s}"

            # Get cache
            cache = getattr(self, f"{cache_name}_cache", None)
            if not cache:
                # No cache, call function
                return await func(self, *args, **kwargs)

            # Check cache
            result = await cache.get(cache_key)
            if result is not None:
                return result

            # Call function and cache result
            result = await func(self, *args, **kwargs)
            await cache.set(cache_key, result)

            return result

        # TODO: Investigate proper generic typing for this decorator
        return wrapper # type: ignore[return-value]
    return decorator


def rate_limited(calls: int = 10, period: int = 60) -> Callable[[F], F]:
    """Decorator for rate limiting function calls."""
    def decorator(func: F) -> F:
        call_times: list[float] = []

        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> object:
            now = time.time()

            # Remove old calls
            nonlocal call_times
            call_times = [t for t in call_times if now - t < period]

            # Check rate limit
            if len(call_times) >= calls:
                wait_time = period - (now - call_times[0])
                raise Exception(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")

            # Record call
            call_times.append(now)

            # Execute function
            return await func(*args, **kwargs)

        # TODO: Investigate proper generic typing for this decorator
        return wrapper # type: ignore[return-value]
    return decorator


def timed(name: str | None = None) -> Callable[[F], F]:
    """Decorator for timing function execution."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(
            self: "PerformanceOptimizer", *args: object, **kwargs: object,
        ) -> object:
            start_time = time.time()

            try:
                result = await func(self, *args, **kwargs)
                execution_time = time.time() - start_time

                # Log timing
                if hasattr(self, "logger"):
                    self.logger.debug(
                        f"{name or func.__name__} completed in {execution_time:.3f}s",
                        source_module=self.__class__.__name__,
                    )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Log error with timing
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"{name or func.__name__} failed after {execution_time:.3f}s: {e!s}",
                        source_module=self.__class__.__name__,
                    )

                raise

        # TODO: Investigate proper generic typing for this decorator
        return wrapper # type: ignore[return-value]
    return decorator
