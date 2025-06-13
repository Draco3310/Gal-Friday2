import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dal.connection_pool import DatabaseConnectionPool
from ..dal.influxdb_client import TimeSeriesDB
from ..dal.models import Order, Position
from ..dal.repositories.history_repository import HistoryRepository
from ..simulated_market_price_service import (
    DataRequest,
    HistoricalDataPoint,
    HistoricalDataProvider,
)


class DatabaseDataProvider(HistoricalDataProvider):
    """Production-ready provider for fetching historical data from databases."""

    def __init__(self, config: Dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Database connections
        self._connection_pool: Optional[DatabaseConnectionPool] = None
        self._ts_db: Optional[TimeSeriesDB] = None
        self._history_repo: Optional[HistoryRepository] = None
        
        # Configuration
        self._query_timeout = config.get("database", {}).get("query_timeout", 30)
        self._max_batch_size = config.get("database", {}).get("max_batch_size", 10000)
        self._enable_query_cache = config.get("database", {}).get("enable_cache", True)
        
        # Performance tracking
        self._query_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_query_time": 0.0
        }
        
        # Symbol validation cache
        self._valid_symbols_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 3600  # 1 hour
        
        # Query result cache
        self._query_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_size = 100
        
        # Initialized flag
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connections and repositories."""
        if self._initialized:
            return
            
        try:
            # Initialize connection pool if not provided
            if not self._connection_pool:
                from ..config_manager import ConfigManager
                from ..logger_service import LoggerService
                
                config_manager = ConfigManager(self.config)
                logger_service = LoggerService(config_manager, source_module=self._source_module)
                
                self._connection_pool = DatabaseConnectionPool(config_manager, logger_service)
                await self._connection_pool.initialize()
            
            # Initialize InfluxDB client if not provided
            if not self._ts_db:
                from ..config_manager import ConfigManager
                from ..logger_service import LoggerService
                
                config_manager = ConfigManager(self.config)
                logger_service = LoggerService(config_manager, source_module=self._source_module)
                
                self._ts_db = TimeSeriesDB(config_manager, logger_service)
            
            # Create history repository
            from ..logger_service import LoggerService
            logger_service = LoggerService(ConfigManager(self.config), source_module=self._source_module)
            self._history_repo = HistoryRepository(self._ts_db, logger_service)
            
            self._initialized = True
            
            self.logger.info(
                "DatabaseDataProvider initialized successfully",
                extra={"source_module": self._source_module}
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to initialize DatabaseDataProvider: {e}",
                extra={"source_module": self._source_module},
                exc_info=True
            )
            raise

    async def cleanup(self) -> None:
        """Clean up database connections."""
        try:
            if self._connection_pool:
                await self._connection_pool.close()
                self._connection_pool = None
            
            if self._ts_db:
                self._ts_db.close()
                self._ts_db = None
            
            self._history_repo = None
            self._initialized = False
                
        except Exception as e:
            self.logger.error(
                f"Error during cleanup: {e}",
                extra={"source_module": self._source_module}
            )

    async def fetch_data(self, request: DataRequest) -> List[HistoricalDataPoint]:
        """Fetch historical data from appropriate database based on data type."""
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Validate request
        if not await self.validate_symbol(request.symbol):
            raise ValueError(f"Invalid symbol: {request.symbol}")
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if self._enable_query_cache and cache_key in self._query_cache:
            cached_data = self._query_cache[cache_key]
            if self._is_cache_valid(cached_data):
                self._query_metrics["cache_hits"] += 1
                return cached_data["data"]
        
        # Start performance timer
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Determine data source based on frequency
            if self._is_high_frequency_data(request.frequency):
                # Use InfluxDB for high-frequency time-series data
                data_points = await self._fetch_from_influxdb(request)
            else:
                # Use PostgreSQL for lower frequency or aggregated data
                data_points = await self._fetch_from_postgresql(request)
            
            # Record metrics
            query_time = asyncio.get_event_loop().time() - start_time
            self._update_query_metrics(query_time, len(data_points))
            
            # Cache the results
            if self._enable_query_cache:
                self._cache_query_result(cache_key, data_points)
            
            self.logger.debug(
                f"Fetched {len(data_points)} data points in {query_time:.3f}s",
                extra={
                    "source_module": self._source_module,
                    "symbol": request.symbol,
                    "start_date": request.start_date.isoformat(),
                    "end_date": request.end_date.isoformat(),
                    "frequency": request.frequency
                }
            )
            
            return data_points
            
        except Exception as e:
            self.logger.error(
                f"Failed to fetch data: {e}",
                extra={
                    "source_module": self._source_module,
                    "request": str(request)
                },
                exc_info=True
            )
            raise

    async def _fetch_from_influxdb(self, request: DataRequest) -> List[HistoricalDataPoint]:
        """Fetch time-series data from InfluxDB."""
        if not self._history_repo:
            raise RuntimeError("HistoryRepository not initialized")
        
        # Convert frequency to interval format expected by InfluxDB
        interval = self._convert_frequency_to_interval(request.frequency)
        
        # Calculate number of candles needed
        candles_needed = self._calculate_candles_needed(
            request.start_date, 
            request.end_date, 
            interval
        )
        
        # Fetch data using history repository
        df = await self._history_repo.get_recent_ohlcv(
            trading_pair=request.symbol,
            limit=candles_needed,
            interval=interval
        )
        
        if df is None or df.empty:
            return []
        
        # Convert DataFrame to HistoricalDataPoint objects
        data_points = []
        for timestamp, row in df.iterrows():
            # Filter by exact date range
            if timestamp < request.start_date or timestamp > request.end_date:
                continue
                
            data_point = HistoricalDataPoint(
                timestamp=timestamp.to_pydatetime(),
                symbol=request.symbol,
                open=float(row.get("open", 0)),
                high=float(row.get("high", 0)),
                low=float(row.get("low", 0)),
                close=float(row.get("close", 0)),
                volume=float(row.get("volume", 0)),
                metadata={
                    "source": "influxdb",
                    "interval": interval
                }
            )
            
            if not hasattr(request, 'validate_data') or request.validate_data:
                if self._validate_ohlcv(data_point):
                    data_points.append(data_point)
            else:
                data_points.append(data_point)
        
        return data_points

    async def _fetch_from_postgresql(self, request: DataRequest) -> List[HistoricalDataPoint]:
        """Fetch aggregated or processed data from PostgreSQL."""
        async with self._get_db_session() as session:
            # Build query based on available data
            # This is a simplified example - in production, you might query
            # from aggregated tables or views
            query = select(Order).where(
                and_(
                    Order.trading_pair == request.symbol,
                    Order.created_at >= request.start_date,
                    Order.created_at <= request.end_date,
                    Order.status == 'FILLED'
                )
            ).order_by(Order.created_at)
            
            # Execute with timeout
            result = await asyncio.wait_for(
                session.execute(query),
                timeout=self._query_timeout
            )
            
            # Group orders by time bucket to create OHLCV data
            orders = result.scalars().all()
            if not orders:
                return []
            
            # Convert orders to OHLCV format
            data_points = self._aggregate_orders_to_ohlcv(orders, request)
            
            return data_points

    @asynccontextmanager
    async def _get_db_session(self):
        """Get database session from connection pool."""
        if not self._connection_pool:
            raise RuntimeError("Connection pool not initialized")
            
        async with self._connection_pool.acquire() as session:
            yield session

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists in database with caching."""
        # Check cache first
        if symbol in self._valid_symbols_cache:
            cache_entry = self._valid_symbols_cache[symbol]
            if self._is_cache_entry_valid(cache_entry):
                return cache_entry["valid"]
        
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        try:
            # Query database for symbol validity
            async with self._get_db_session() as session:
                # Check if symbol has any recent trades or orders
                query = select(func.count(Order.id)).where(
                    Order.trading_pair == symbol
                ).limit(1)
                
                result = await session.execute(query)
                count = result.scalar()
                exists = count > 0
                
                # Cache result
                self._valid_symbols_cache[symbol] = {
                    "valid": exists,
                    "timestamp": datetime.now(UTC)
                }
                
                return exists
                
        except Exception as e:
            self.logger.warning(
                f"Failed to validate symbol {symbol}: {e}",
                extra={"source_module": self._source_module}
            )
            # Default to true on error to avoid blocking
            return True

    async def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get additional metadata about a symbol from database."""
        metadata = {
            "symbol": symbol,
            "min_price": None,
            "max_price": None,
            "avg_spread": None,
            "total_volume": None,
            "last_update": None
        }
        
        try:
            async with self._get_db_session() as session:
                # Get aggregated statistics from orders
                thirty_days_ago = datetime.now(UTC) - timedelta(days=30)
                
                query = select(
                    func.min(Order.price).label("min_price"),
                    func.max(Order.price).label("max_price"),
                    func.sum(Order.volume).label("total_volume"),
                    func.max(Order.created_at).label("last_update")
                ).where(
                    and_(
                        Order.trading_pair == symbol,
                        Order.created_at >= thirty_days_ago,
                        Order.status == 'FILLED'
                    )
                )
                
                result = await session.execute(query)
                row = result.first()
                
                if row:
                    metadata.update({
                        "min_price": float(row.min_price) if row.min_price else None,
                        "max_price": float(row.max_price) if row.max_price else None,
                        "total_volume": float(row.total_volume) if row.total_volume else None,
                        "last_update": row.last_update.isoformat() if row.last_update else None
                    })
                    
        except Exception as e:
            self.logger.error(
                f"Failed to get symbol metadata: {e}",
                extra={"source_module": self._source_module},
                exc_info=True
            )
        
        return metadata

    def _is_high_frequency_data(self, frequency: str) -> bool:
        """Determine if frequency requires high-frequency data source."""
        # Consider anything under 5 minutes as high frequency
        if frequency.endswith('s'):  # seconds
            return True
        if frequency.endswith('m'):  # minutes
            try:
                minutes = int(frequency[:-1])
                return minutes < 5
            except ValueError:
                pass
        return False

    def _convert_frequency_to_interval(self, frequency: str) -> str:
        """Convert frequency string to InfluxDB interval format."""
        # Map common frequencies to InfluxDB intervals
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '60m',
            '4h': '240m',
            '1d': '1440m'
        }
        return mapping.get(frequency, frequency)

    def _calculate_candles_needed(self, start_date: datetime, end_date: datetime, interval: str) -> int:
        """Calculate number of candles needed for the date range."""
        # Parse interval to minutes
        try:
            interval_minutes = int(interval.rstrip('m'))
        except ValueError:
            interval_minutes = 60  # Default to 1 hour
        
        # Calculate time difference
        time_diff = end_date - start_date
        total_minutes = time_diff.total_seconds() / 60
        
        # Calculate candles needed with buffer
        candles = int(total_minutes / interval_minutes) + 10
        
        # Cap at max batch size
        return min(candles, self._max_batch_size)

    def _validate_ohlcv(self, dp: HistoricalDataPoint) -> bool:
        """Validate OHLCV data integrity."""
        # Basic OHLCV validation
        if dp.high < dp.low:
            return False
        
        if dp.open <= 0 or dp.close <= 0:
            return False
        
        if dp.volume < 0:
            return False
        
        if dp.high < max(dp.open, dp.close):
            return False
        
        if dp.low > min(dp.open, dp.close):
            return False
        
        return True

    def _aggregate_orders_to_ohlcv(self, orders: List[Order], request: DataRequest) -> List[HistoricalDataPoint]:
        """Aggregate orders into OHLCV format."""
        # Group orders by time bucket
        bucket_size = self._parse_frequency_to_timedelta(request.frequency)
        buckets = defaultdict(list)
        
        for order in orders:
            bucket_time = self._round_to_bucket(order.created_at, bucket_size)
            buckets[bucket_time].append(order)
        
        # Convert each bucket to OHLCV
        data_points = []
        for bucket_time, bucket_orders in sorted(buckets.items()):
            if not bucket_orders:
                continue
            
            # Sort orders by time within bucket
            bucket_orders.sort(key=lambda x: x.created_at)
            
            # Calculate OHLCV
            prices = [float(o.price) for o in bucket_orders if o.price]
            volumes = [float(o.volume) for o in bucket_orders if o.volume]
            
            if not prices:
                continue
            
            data_point = HistoricalDataPoint(
                timestamp=bucket_time,
                symbol=request.symbol,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(volumes),
                metadata={
                    "source": "postgresql",
                    "trade_count": len(bucket_orders)
                }
            )
            
            data_points.append(data_point)
        
        return data_points

    def _parse_frequency_to_timedelta(self, frequency: str) -> timedelta:
        """Parse frequency string to timedelta."""
        if frequency.endswith('s'):
            return timedelta(seconds=int(frequency[:-1]))
        elif frequency.endswith('m'):
            return timedelta(minutes=int(frequency[:-1]))
        elif frequency.endswith('h'):
            return timedelta(hours=int(frequency[:-1]))
        elif frequency.endswith('d'):
            return timedelta(days=int(frequency[:-1]))
        else:
            return timedelta(minutes=5)  # Default

    def _round_to_bucket(self, dt: datetime, bucket_size: timedelta) -> datetime:
        """Round datetime to nearest bucket."""
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        delta = dt - epoch
        bucket_seconds = bucket_size.total_seconds()
        rounded_seconds = (delta.total_seconds() // bucket_seconds) * bucket_seconds
        return epoch + timedelta(seconds=rounded_seconds)

    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request."""
        return f"{request.symbol}_{request.start_date.isoformat()}_{request.end_date.isoformat()}_{request.frequency}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False
        
        age = (datetime.now(UTC) - cache_entry["timestamp"]).total_seconds()
        return age < self._cache_ttl

    def _is_cache_entry_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if symbol cache entry is still valid."""
        if "timestamp" not in cache_entry:
            return False
        
        age = (datetime.now(UTC) - cache_entry["timestamp"]).total_seconds()
        return age < self._cache_ttl

    def _cache_query_result(self, cache_key: str, data_points: List[HistoricalDataPoint]) -> None:
        """Cache query results with LRU eviction."""
        # Implement simple LRU by removing oldest entries
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k].get("timestamp", datetime.min))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = {
            "data": data_points,
            "timestamp": datetime.now(UTC)
        }

    def _update_query_metrics(self, query_time: float, result_count: int) -> None:
        """Update internal query performance metrics."""
        self._query_metrics["total_queries"] += 1
        
        # Update rolling average query time
        current_avg = self._query_metrics["average_query_time"]
        total_queries = self._query_metrics["total_queries"]
        
        self._query_metrics["average_query_time"] = (
            (current_avg * (total_queries - 1) + query_time) / total_queries
        )
        
        # Log slow queries
        if query_time > 5.0:  # 5 second threshold
            self.logger.warning(
                f"Slow query detected: {query_time:.2f}s for {result_count} results",
                extra={"source_module": self._source_module}
            )

    async def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about database provider health."""
        diagnostics = {
            "provider": "DatabaseDataProvider",
            "status": "healthy",
            "initialized": self._initialized,
            "metrics": self._query_metrics.copy(),
            "cache_stats": {
                "query_cache_size": len(self._query_cache),
                "symbol_cache_size": len(self._valid_symbols_cache),
                "cache_enabled": self._enable_query_cache
            },
            "connections": {
                "postgresql": "unknown",
                "influxdb": "unknown"
            }
        }
        
        # Check PostgreSQL connection
        if self._connection_pool and self._initialized:
            try:
                async with self._get_db_session() as session:
                    await session.execute(select(1))
                    diagnostics["connections"]["postgresql"] = "healthy"
            except Exception as e:
                diagnostics["connections"]["postgresql"] = f"error: {str(e)}"
                diagnostics["status"] = "degraded"
        
        # Check InfluxDB connection
        if self._ts_db and self._initialized:
            try:
                # Perform health check query
                await self._ts_db.query("buckets()")
                diagnostics["connections"]["influxdb"] = "healthy"
            except Exception as e:
                diagnostics["connections"]["influxdb"] = f"error: {str(e)}"
                diagnostics["status"] = "degraded"
        
        return diagnostics
