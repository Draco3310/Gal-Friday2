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
        """Check if connection is healthy.
        
        This method should be overridden in subclasses to implement
        connection-specific health checks. For database connections,
        this typically involves executing a simple query.
        """
        # Default implementation for generic connections
        if hasattr(conn, 'is_closed'):
            # For asyncpg-style connections
            return not conn.is_closed()
        elif hasattr(conn, 'closed'):
            # For aiopg/psycopg-style connections
            return conn.closed == 0
        elif hasattr(conn, 'ping'):
            # For connections with ping method
            try:
                await conn.ping()
                return True
            except Exception as e:
                self.logger.debug(
                    f"Connection ping failed: {e}",
                    source_module=self._source_module,
                )
                return False
        else:
            # Unknown connection type - log warning once
            if not hasattr(self, '_health_check_warned'):
                self.logger.warning(
                    "Unable to determine health check method for connection type: %s. "
                    "Consider implementing a specific health check.",
                    type(conn).__name__,
                    source_module=self._source_module,
                )
                self._health_check_warned = True
            return True  # Assume healthy if we can't check

    async def _close_conn(self, conn: object) -> None:
        """Close a connection.
        
        This method should be overridden in subclasses to implement
        connection-specific cleanup.
        """
        try:
            if hasattr(conn, 'close'):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()
            elif hasattr(conn, 'terminate'):
                # For connections that use terminate instead of close
                if asyncio.iscoroutinefunction(conn.terminate):
                    await conn.terminate()
                else:
                    conn.terminate()
            else:
                self.logger.warning(
                    "Unable to close connection of type: %s",
                    type(conn).__name__,
                    source_module=self._source_module,
                )
        except Exception as e:
            self.logger.warning(
                f"Error closing connection: {e}",
                source_module=self._source_module,
            )

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
        """Analyze query performance and provide optimization suggestions."""
        start_time = time.time()
        
        # Normalize query for analysis
        normalized_query = query.strip().upper()
        suggestions: list[str] = []
        
        # Analyze query structure
        query_type = "UNKNOWN"
        if normalized_query.startswith("SELECT"):
            query_type = "SELECT"
        elif normalized_query.startswith("INSERT"):
            query_type = "INSERT"
        elif normalized_query.startswith("UPDATE"):
            query_type = "UPDATE"
        elif normalized_query.startswith("DELETE"):
            query_type = "DELETE"
        
        # Estimate complexity based on query structure
        estimated_cost = self._estimate_query_cost(normalized_query)
        estimated_rows = self._estimate_result_rows(normalized_query)
        uses_index = self._check_index_usage(normalized_query)
        
        # Common performance issues and suggestions
        if "SELECT *" in normalized_query:
            suggestions.append("Avoid SELECT *, specify needed columns to reduce data transfer")
            estimated_cost *= 1.5  # Penalize for SELECT *
        
        if "NOT IN" in normalized_query:
            suggestions.append("Consider using NOT EXISTS instead of NOT IN for better null handling")
            estimated_cost *= 1.2
        
        if " OR " in normalized_query and normalized_query.count(" OR ") > 2:
            suggestions.append("Multiple OR conditions can prevent index usage, consider UNION")
            uses_index = False
            estimated_cost *= 1.3
        
        if "LIKE '%%" in normalized_query or "LIKE '%" in normalized_query:
            suggestions.append("Leading wildcard in LIKE prevents index usage")
            uses_index = False
            estimated_cost *= 2.0
        
        # Join analysis
        join_count = normalized_query.count(" JOIN ")
        if join_count > 3:
            suggestions.append(f"{join_count} JOINs detected, consider denormalization or materialized views")
            estimated_cost *= (1.1 ** join_count)
        
        # Subquery analysis
        if "SELECT" in normalized_query[7:]:  # Skip the first SELECT
            subquery_count = normalized_query.count("SELECT") - 1
            if subquery_count > 0:
                suggestions.append(f"{subquery_count} subqueries detected, consider using JOINs or CTEs")
                estimated_cost *= (1.2 ** subquery_count)
        
        # Check for missing WHERE clause in UPDATE/DELETE
        if query_type in ["UPDATE", "DELETE"] and " WHERE " not in normalized_query:
            suggestions.append(f"WARNING: {query_type} without WHERE clause affects all rows!")
            estimated_rows = 999999  # Indicate large impact
        
        # Function usage that prevents index
        problematic_functions = ["UPPER(", "LOWER(", "COALESCE(", "CAST(", "CONVERT("]
        for func in problematic_functions:
            if func in normalized_query and " WHERE " in normalized_query:
                # Check if function is used in WHERE clause
                where_clause = normalized_query.split(" WHERE ")[1].split(" ORDER BY ")[0]
                if func in where_clause:
                    suggestions.append(f"{func.rstrip('(')} in WHERE clause may prevent index usage")
                    uses_index = False
                    estimated_cost *= 1.3
        
        execution_time = time.time() - start_time

        analysis = {
            "query": query,
            "query_type": query_type,
            "estimated_cost": round(estimated_cost, 2),
            "estimated_rows": estimated_rows,
            "index_scan": uses_index,
            "join_count": join_count,
            "analysis_time_ms": round(execution_time * 1000, 2),
            "suggestions": suggestions,
        }

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

        # Log slow query analysis
        if execution_time > self.slow_query_threshold:
            self.logger.warning(
                f"Slow query analysis: {execution_time:.2f}s",
                source_module=self._source_module,
                context={"query": query_key, "suggestions": suggestions},
            )

        return analysis

    def _estimate_query_cost(self, normalized_query: str) -> float:
        """Estimate relative cost of query execution."""
        base_cost = 10.0
        
        # Table scan indicators
        if " WHERE " not in normalized_query:
            base_cost *= 10.0  # No WHERE clause likely means full table scan
        
        # Aggregation functions
        for agg_func in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]:
            if agg_func in normalized_query:
                base_cost *= 1.5
        
        # GROUP BY and ORDER BY
        if " GROUP BY " in normalized_query:
            base_cost *= 2.0
        if " ORDER BY " in normalized_query:
            base_cost *= 1.5
        
        # DISTINCT
        if " DISTINCT " in normalized_query:
            base_cost *= 1.8
        
        return base_cost

    def _estimate_result_rows(self, normalized_query: str) -> int:
        """Estimate number of result rows based on query structure."""
        if " LIMIT 1" in normalized_query:
            return 1
        elif " LIMIT " in normalized_query:
            # Extract limit value
            try:
                limit_part = normalized_query.split(" LIMIT ")[1].split()[0]
                return int(limit_part)
            except (IndexError, ValueError):
                pass
        
        # Aggregate without GROUP BY usually returns 1 row
        if any(f in normalized_query for f in ["COUNT(", "SUM(", "AVG("]) and " GROUP BY " not in normalized_query:
            return 1
        
        # Default estimates based on query type
        if " WHERE " in normalized_query:
            if "=" in normalized_query:
                return 10  # Equality condition
            else:
                return 100  # Range condition
        else:
            return 1000  # No WHERE clause

    def _check_index_usage(self, normalized_query: str) -> bool:
        """Check if query is likely to use indexes."""
        # Simple heuristics for index usage
        if " WHERE " not in normalized_query:
            return False
        
        where_clause = normalized_query.split(" WHERE ")[1].split(" ORDER BY ")[0].split(" GROUP BY ")[0]
        
        # Good index indicators
        if "=" in where_clause and "LIKE" not in where_clause:
            return True
        if " BETWEEN " in where_clause:
            return True
        if " IN (" in where_clause and "NOT IN" not in where_clause:
            return True
        
        return False

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


class DatabaseConnectionPool(ConnectionPool):
    """Specialized connection pool for database connections with proper health checks."""
    
    def __init__(
        self,
        create_conn: Callable[[], Any],
        logger_service: LoggerService,
        max_connections: int = 10,
        min_connections: int = 2,
        health_check_interval: int = 30,
        health_check_query: str = "SELECT 1",
    ) -> None:
        """Initialize database connection pool with health check query.
        
        Args:
            create_conn: Async function to create a new connection
            logger_service: Logger service instance
            max_connections: Maximum number of connections
            min_connections: Minimum number of connections to maintain
            health_check_interval: Seconds between health checks
            health_check_query: SQL query to use for health checks
        """
        super().__init__(
            create_conn, 
            logger_service, 
            max_connections, 
            min_connections, 
            health_check_interval
        )
        self._health_check_query = health_check_query
        self._source_module = self.__class__.__name__
    
    async def _is_healthy(self, conn: object) -> bool:
        """Check if database connection is healthy by executing a test query."""
        try:
            # First check if connection is closed
            if hasattr(conn, 'is_closed') and conn.is_closed():
                return False
            elif hasattr(conn, 'closed') and conn.closed != 0:
                return False
            
            # Execute health check query
            if hasattr(conn, 'execute'):
                # For asyncpg-style connections
                await conn.execute(self._health_check_query)
                return True
            elif hasattr(conn, 'fetch') or hasattr(conn, 'fetchval'):
                # For other async database connections
                await conn.fetchval(self._health_check_query)
                return True
            elif hasattr(conn, 'cursor'):
                # For aiopg-style connections
                async with conn.cursor() as cursor:
                    await cursor.execute(self._health_check_query)
                    await cursor.fetchone()
                return True
            else:
                # Fall back to parent implementation
                return await super()._is_healthy(conn)
                
        except Exception as e:
            self.logger.debug(
                f"Database health check failed: {e}",
                source_module=self._source_module,
                context={"query": self._health_check_query}
            )
            return False
