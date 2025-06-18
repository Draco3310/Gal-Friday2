"""Performance optimization utilities for Gal-Friday."""

import asyncio
import contextlib
import functools
import gc
import hashlib
import inspect
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from typing import (
    Any, Generic, TypeVar, cast, Dict, Optional, Union, Tuple, Protocol, 
    runtime_checkable, overload, Type, ParamSpec, Concatenate
)
from typing import TypeVar as TypeVarT

import psutil

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService

# Type variables for comprehensive generic support
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)  # Covariant type variable for protocols
K = TypeVar("K")  # Cache key type
V = TypeVar("V")  # Cache value type
P = ParamSpec("P")  # Parameter specification for function signatures
AsyncP = ParamSpec("AsyncP")  # Parameter specification for async functions
AsyncT = TypeVar("AsyncT")  # Return type for async functions
AsyncT_co = TypeVar("AsyncT_co", covariant=True)  # Covariant for async protocols


# Protocols for type safety
@runtime_checkable
class CacheableFunction(Protocol[P, T_co]):
    """Protocol for cacheable functions."""
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co: ...


@runtime_checkable
class AsyncCacheableFunction(Protocol[AsyncP, AsyncT_co]):
    """Protocol for async cacheable functions."""
    def __call__(self, *args: AsyncP.args, **kwargs: AsyncP.kwargs) -> Awaitable[AsyncT_co]: ...


@dataclass
class CacheEntry(Generic[V]):
    """Type-safe cache entry with metadata."""
    value: V
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access information."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheConfig:
    """Type-safe cache configuration."""
    max_size: Optional[int] = 128
    ttl: Optional[float] = None
    typed_keys: bool = True
    thread_safe: bool = True
    eviction_policy: str = "lru"
    key_serializer: Optional[Callable[..., str]] = None


class CacheKeyGenerator(Generic[K]):
    """Type-safe cache key generator."""
    
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        
    def generate_key(
        self, 
        func: Callable[..., Any], 
        args: Tuple[Any, ...], 
        kwargs: Dict[str, Any]
    ) -> K:
        """Generate type-safe cache key."""
        
        if self.config.key_serializer:
            return cast(K, self.config.key_serializer((func.__name__, args, kwargs)))
        
        # Default key generation with type information
        key_parts = [func.__module__ or "", func.__qualname__]
        
        # Add arguments to key
        for arg in args:
            if self.config.typed_keys:
                key_parts.append(f"{type(arg).__name__}:{repr(arg)}")
            else:
                key_parts.append(repr(arg))
        
        # Add keyword arguments to key
        for k, v in sorted(kwargs.items()):
            if self.config.typed_keys:
                key_parts.append(f"{k}={type(v).__name__}:{repr(v)}")
            else:
                key_parts.append(f"{k}={repr(v)}")
        
        # Create hash of key parts
        key_string = "|".join(key_parts)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return cast(K, key_hash)


class TypeSafeCache(Generic[K, V]):
    """Type-safe cache implementation with comprehensive features."""
    
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock() if config.thread_safe else None
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: K) -> Optional[V]:
        """Get value from cache with type safety."""
        
        with self._lock if self._lock else contextlib.nullcontext():
            if key not in self._cache:
                self.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self.misses += 1
                return None
            
            # Update access information
            entry.touch()
            
            # Move to end for LRU
            if self.config.eviction_policy == "lru":
                self._cache.move_to_end(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: K, value: V) -> None:
        """Put value in cache with type safety."""
        
        with self._lock if self._lock else contextlib.nullcontext():
            # Check if eviction is needed
            if (self.config.max_size is not None and 
                len(self._cache) >= self.config.max_size and 
                key not in self._cache):
                self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=self.config.ttl
            )
            
            self._cache[key] = entry
    
    def _evict_entries(self) -> None:
        """Evict entries based on policy."""
        
        if not self._cache:
            return
        
        if self.config.eviction_policy == "lru":
            # Remove least recently used
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        elif self.config.eviction_policy == "lfu":
            # Remove least frequently used
            lfu_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            del self._cache[lfu_key]
        
        self.evictions += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock if self._lock else contextlib.nullcontext():
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._cache),
            "max_size": self.config.max_size
        }


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
        health_check_interval: int = 30) -> None:
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
        self._source_module = self.__class__.__name__
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
                    f"Health check error: {e}")

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
            return bool(conn.closed == 0)
        elif hasattr(conn, 'ping'):
            # For connections with ping method
            try:
                await conn.ping()
                return True
            except Exception as e:
                self.logger.debug(
                    f"Connection ping failed: {e}",
                    source_module=self._source_module)
                return False
        else:
            # Unknown connection type - log warning once
            if not hasattr(self, '_health_check_warned'):
                self.logger.warning(
                    "Unable to determine health check method for connection type: %s. "
                    "Consider implementing a specific health check.",
                    type(conn).__name__,
                    source_module=self._source_module)
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
                    source_module=self._source_module)
        except Exception as e:
            self.logger.warning(
                f"Error closing connection: {e}",
                source_module=self._source_module)

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
        self, query: str, params: tuple[Any, ...] | None = None) -> dict[str, Any]:
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
                context={"query": query_key, "suggestions": suggestions})

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
            reverse=True)

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
                source_module=self._source_module)

            # Force garbage collection
            gc.collect()

            # Get new usage
            new_usage = self.get_memory_usage()
            freed_mb = usage["rss_mb"] - new_usage["rss_mb"]

            self.logger.info(
                f"Garbage collection freed {freed_mb:.1f}MB",
                source_module=self._source_module)

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
            source_module=self._source_module)

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
                    })

                # Check for issues
                if memory_usage["rss_mb"] > self.memory_optimizer.memory_limit_mb:
                    self.logger.error(
                        "Memory limit exceeded: "
                        f"{memory_usage['rss_mb']:.1f}MB > "
                        f"{self.memory_optimizer.memory_limit_mb}MB",
                        source_module=self._source_module)

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in performance monitoring",
                    source_module=self._source_module)

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


# Type-safe decorators for performance optimization

class TypedCacheDecorator(Generic[P, T]):
    """Type-safe caching decorator that preserves function signatures."""
    
    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self.cache: TypeSafeCache[str, T] = TypeSafeCache(self.config)
        self.key_generator: CacheKeyGenerator[str] = CacheKeyGenerator(self.config)
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Create type-safe caching decorator with proper generic typing."""
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Generate cache key
            cache_key = self.key_generator.generate_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods with proper typing
        wrapper.cache_clear = self.cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = self.cache.stats  # type: ignore[attr-defined]
        
        return wrapper


class AsyncTypedCacheDecorator(Generic[AsyncP, AsyncT]):
    """Type-safe async caching decorator."""
    
    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self.cache: TypeSafeCache[str, AsyncT] = TypeSafeCache(self.config)
        self.key_generator: CacheKeyGenerator[str] = CacheKeyGenerator(self.config)
    
    def __call__(self, func: Callable[AsyncP, Awaitable[AsyncT]]) -> Callable[AsyncP, Awaitable[AsyncT]]:
        """Create type-safe async caching decorator."""
        
        @functools.wraps(func)
        async def async_wrapper(*args: AsyncP.args, **kwargs: AsyncP.kwargs) -> AsyncT:
            cache_key = self.key_generator.generate_key(func, args, kwargs)
            
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = await func(*args, **kwargs)
            self.cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods
        async_wrapper.cache_clear = self.cache.clear  # type: ignore[attr-defined]
        async_wrapper.cache_stats = self.cache.stats  # type: ignore[attr-defined]
        
        return async_wrapper


class TypedRateLimitDecorator(Generic[P, T]):
    """Type-safe rate limiting decorator."""
    
    def __init__(self, calls: int = 10, period: int = 60) -> None:
        self.calls = calls
        self.period = period
        self.call_times: list[float] = []
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Create type-safe rate limiting decorator."""
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            now = time.time()
            
            with self._lock:
                # Remove old calls
                self.call_times = [t for t in self.call_times if now - t < self.period]
                
                # Check rate limit
                if len(self.call_times) >= self.calls:
                    wait_time = self.period - (now - self.call_times[0])
                    raise RuntimeError(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")
                
                # Record call
                self.call_times.append(now)
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper


class AsyncTypedRateLimitDecorator(Generic[AsyncP, AsyncT]):
    """Type-safe async rate limiting decorator."""
    
    def __init__(self, calls: int = 10, period: int = 60) -> None:
        self.calls = calls
        self.period = period
        self.call_times: list[float] = []
        self._lock = asyncio.Lock()
    
    def __call__(self, func: Callable[AsyncP, Awaitable[AsyncT]]) -> Callable[AsyncP, Awaitable[AsyncT]]:
        """Create type-safe async rate limiting decorator."""
        
        @functools.wraps(func)
        async def async_wrapper(*args: AsyncP.args, **kwargs: AsyncP.kwargs) -> AsyncT:
            now = time.time()
            
            async with self._lock:
                # Remove old calls
                self.call_times = [t for t in self.call_times if now - t < self.period]
                
                # Check rate limit
                if len(self.call_times) >= self.calls:
                    wait_time = self.period - (now - self.call_times[0])
                    raise RuntimeError(f"Rate limit exceeded. Try again in {wait_time:.1f} seconds")
                
                # Record call
                self.call_times.append(now)
            
            # Execute function
            return await func(*args, **kwargs)
        
        return async_wrapper


class TypedTimingDecorator(Generic[P, T]):
    """Type-safe timing decorator with logging."""
    
    def __init__(self, name: Optional[str] = None, logger: Optional[LoggerService] = None) -> None:
        self.name = name
        self.logger = logger
    
    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        """Create type-safe timing decorator."""
        
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            function_name = self.name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log timing if logger available
                if self.logger:
                    self.logger.debug(
                        f"{function_name} completed in {execution_time:.3f}s",
                        source_module="TypedTimingDecorator")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log error with timing
                if self.logger:
                    self.logger.error(
                        f"{function_name} failed after {execution_time:.3f}s: {e!s}",
                        source_module="TypedTimingDecorator")
                
                raise
        
        return wrapper


class AsyncTypedTimingDecorator(Generic[AsyncP, AsyncT]):
    """Type-safe async timing decorator with logging."""
    
    def __init__(self, name: Optional[str] = None, logger: Optional[LoggerService] = None) -> None:
        self.name = name
        self.logger = logger
    
    def __call__(self, func: Callable[AsyncP, Awaitable[AsyncT]]) -> Callable[AsyncP, Awaitable[AsyncT]]:
        """Create type-safe async timing decorator."""
        
        @functools.wraps(func)
        async def async_wrapper(*args: AsyncP.args, **kwargs: AsyncP.kwargs) -> AsyncT:
            start_time = time.time()
            function_name = self.name or func.__name__
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log timing if logger available
                if self.logger:
                    self.logger.debug(
                        f"{function_name} completed in {execution_time:.3f}s",
                        source_module="AsyncTypedTimingDecorator")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log error with timing
                if self.logger:
                    self.logger.error(
                        f"{function_name} failed after {execution_time:.3f}s: {e!s}",
                        source_module="AsyncTypedTimingDecorator")
                
                raise
        
        return async_wrapper


# Public API with overloads for comprehensive type safety

@overload
def cache() -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(*, max_size: Optional[int] = ...) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(*, ttl: Optional[float] = ...) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(*, max_size: Optional[int] = ..., ttl: Optional[float] = ...) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

@overload
def cache(config: CacheConfig) -> Callable[[Callable[P, T]], Callable[P, T]]: ...

def cache(
    config: Optional[CacheConfig] = None,
    *,
    max_size: Optional[int] = None,
    ttl: Optional[float] = None,
    typed_keys: bool = True,
    thread_safe: bool = True,
    eviction_policy: str = "lru"
) -> Any:
    """
    Type-safe caching decorator with comprehensive generic typing.
    
    """
    
    # Create configuration if not provided
    if config is None:
        config = CacheConfig(
            max_size=max_size,
            ttl=ttl,
            typed_keys=typed_keys,
            thread_safe=thread_safe,
            eviction_policy=eviction_policy
        )
    
    return TypedCacheDecorator(config)


def async_cache(
    config: Optional[CacheConfig] = None,
    *,
    max_size: Optional[int] = None,
    ttl: Optional[float] = None,
    typed_keys: bool = True,
    thread_safe: bool = True,
    eviction_policy: str = "lru"
) -> Any:
    """Type-safe async caching decorator with comprehensive generic typing."""
    
    if config is None:
        config = CacheConfig(
            max_size=max_size,
            ttl=ttl,
            typed_keys=typed_keys,
            thread_safe=thread_safe,
            eviction_policy=eviction_policy
        )
    
    return AsyncTypedCacheDecorator(config)


def rate_limited(calls: int = 10, period: int = 60) -> Any:
    """Type-safe rate limiting decorator with proper generic typing."""
    return TypedRateLimitDecorator(calls, period)


def async_rate_limited(calls: int = 10, period: int = 60) -> Any:
    """Type-safe async rate limiting decorator with proper generic typing."""
    return AsyncTypedRateLimitDecorator(calls, period)


def timed(
    name: Optional[str] = None, 
    logger: Optional[LoggerService] = None
) -> Any:
    """Type-safe timing decorator with proper generic typing."""
    return TypedTimingDecorator(name, logger)


def async_timed(
    name: Optional[str] = None, 
    logger: Optional[LoggerService] = None
) -> Any:
    """Type-safe async timing decorator with proper generic typing."""
    return AsyncTypedTimingDecorator(name, logger)


# Legacy compatibility decorators (deprecated but maintained for backward compatibility)
F = TypeVarT("F", bound=Callable[..., Any])

def cached(cache_name: str = "default", ttl: int = 300) -> Callable[[F], F]:
    """
    Legacy caching decorator for backward compatibility.
    
    DEPRECATED: Use @cache() decorator with proper generic typing instead.
    This decorator is maintained for backward compatibility but lacks proper type safety.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(
            self: "PerformanceOptimizer", *args: object, **kwargs: object) -> object:
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

        return wrapper # type: ignore[return-value]
    return decorator


# Method caching support with proper typing
class MethodCacheDescriptor(Generic[T]):
    """Type-safe method caching descriptor for class methods."""
    
    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config or CacheConfig()
        self.caches: weakref.WeakKeyDictionary[Any, TypeSafeCache[str, T]] = weakref.WeakKeyDictionary()
        self.key_generator = CacheKeyGenerator[str](self.config)
        self.original_method: Optional[Callable[..., T]] = None
    
    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
    
    def __call__(self, method: Callable[..., T]) -> "MethodCacheDescriptor[T]":
        """Used as a decorator to wrap the original method."""
        self.original_method = method
        functools.update_wrapper(self, method)
        return self
    
    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Callable[..., T]:
        """Get the cached method for the instance."""
        if instance is None:
            return self  # type: ignore
        
        if self.original_method is None:
            raise RuntimeError("Method cache descriptor not properly initialized")
        
        # Get or create cache for this instance
        if instance not in self.caches:
            self.caches[instance] = TypeSafeCache[str, T](self.config)
        
        instance_cache = self.caches[instance]
        original_method = self.original_method
        
        @functools.wraps(original_method)
        def cached_method(*args: Any, **kwargs: Any) -> T:
            cache_key = self.key_generator.generate_key(original_method, args, kwargs)
            
            cached_result = instance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = original_method(instance, *args, **kwargs)
            instance_cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods
        cached_method.cache_clear = instance_cache.clear  # type: ignore[attr-defined]
        cached_method.cache_stats = instance_cache.stats  # type: ignore[attr-defined]
        
        return cached_method


def cached_method(config: Optional[CacheConfig] = None) -> MethodCacheDescriptor[T]:
    """
    Decorator for creating cached methods with proper typing.
    
    Example:
        class DataProcessor:
            @cached_method(CacheConfig(max_size=50, ttl=300))
            def expensive_computation(self, data: List[int]) -> Dict[str, int]:
                return {"sum": sum(data), "count": len(data)}
    """
    return MethodCacheDescriptor[T](config)


# Example usage and integration helpers
class CacheRegistry:
    """Registry for managing multiple caches with different configurations."""
    
    def __init__(self) -> None:
        self._caches: Dict[str, TypeSafeCache[Any, Any]] = {}
        self._configs: Dict[str, CacheConfig] = {}
    
    def register_cache(self, name: str, config: CacheConfig) -> None:
        """Register a new cache with the given configuration."""
        self._caches[name] = TypeSafeCache(config)
        self._configs[name] = config
    
    def get_cache(self, name: str) -> Optional[TypeSafeCache[Any, Any]]:
        """Get a registered cache by name."""
        return self._caches.get(name)
    
    def clear_all(self) -> None:
        """Clear all registered caches."""
        for cache in self._caches.values():
            cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all registered caches."""
        return {name: cache.stats() for name, cache in self._caches.items()}


# Global cache registry for application-wide cache management
_global_cache_registry = CacheRegistry()


def register_global_cache(name: str, config: CacheConfig) -> None:
    """Register a cache in the global registry."""
    _global_cache_registry.register_cache(name, config)


def get_global_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all global caches."""
    return _global_cache_registry.get_all_stats()


def clear_all_global_caches() -> None:
    """Clear all global caches."""
    _global_cache_registry.clear_all()


# Performance monitoring integration
class CachePerformanceMonitor:
    """Monitor cache performance and provide optimization suggestions."""
    
    def __init__(self, logger: LoggerService) -> None:
        self.logger = logger
        self._source_module = self.__class__.__name__
    
    def analyze_cache_performance(self, cache: TypeSafeCache[Any, Any], cache_name: str) -> Dict[str, Any]:
        """Analyze cache performance and provide recommendations."""
        stats = cache.stats()
        
        analysis = {
            "cache_name": cache_name,
            **stats,
            "recommendations": []
        }
        
        # Analyze hit rate
        hit_rate = stats["hit_rate_percent"]
        if hit_rate < 50:
            analysis["recommendations"].append(
                f"Low hit rate ({hit_rate:.1f}%). Consider increasing cache size or reviewing cache strategy."
            )
        elif hit_rate > 95:
            analysis["recommendations"].append(
                f"Very high hit rate ({hit_rate:.1f}%). Cache is performing excellently."
            )
        
        # Analyze cache utilization
        if stats["max_size"] and stats["cache_size"] < stats["max_size"] * 0.5:
            analysis["recommendations"].append(
                f"Cache is underutilized ({stats['cache_size']}/{stats['max_size']}). "
                "Consider reducing max_size to save memory."
            )
        
        # Analyze eviction rate
        total_operations = stats["hits"] + stats["misses"]
        if total_operations > 0:
            eviction_rate = (stats["evictions"] / total_operations) * 100
            if eviction_rate > 10:
                analysis["recommendations"].append(
                    f"High eviction rate ({eviction_rate:.1f}%). Consider increasing cache size."
                )
        
        return analysis


class DatabaseConnectionPool(ConnectionPool):
    """Specialized connection pool for database connections with proper health checks."""
    
    def __init__(
        self,
        create_conn: Callable[[], Any],
        logger_service: LoggerService,
        max_connections: int = 10,
        min_connections: int = 2,
        health_check_interval: int = 30,
        health_check_query: str = "SELECT 1") -> None:
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
                await conn.fetchval(self._health_check_query)  # type: ignore[attr-defined]
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