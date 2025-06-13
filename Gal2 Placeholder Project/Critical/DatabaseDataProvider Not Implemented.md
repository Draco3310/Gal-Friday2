# DatabaseDataProvider Implementation Design

**File**: `/gal_friday/providers/database_provider.py`
- **Line 20**: `self.logger.info("DatabaseDataProvider not implemented")`
- **Issue**: Returns empty list instead of fetching actual data from database
- **Impact**: No database connectivity for historical data

## Overview
The DatabaseDataProvider is responsible for fetching historical market data from the project's dual-database architecture (PostgreSQL for relational data and InfluxDB for time-series data). This implementation will provide high-performance data retrieval with proper caching, query optimization, and connection pooling.

## Architecture Design

### 1. Core Components

```
DatabaseDataProvider
├── Connection Management
│   ├── PostgreSQL connection pool
│   ├── InfluxDB client management
│   └── Connection health monitoring
├── Query Builder
│   ├── Dynamic query construction
│   ├── Query parameter validation
│   └── SQL injection prevention
├── Data Access Layer Integration
│   ├── HistoryRepository utilization
│   ├── Repository pattern consistency
│   └── Transaction management
├── Performance Optimization
│   ├── Query optimization
│   ├── Result set pagination
│   └── Connection pool tuning
├── Data Transformation
│   ├── Database to domain model mapping
│   ├── Time zone handling
│   └── Decimal precision management
└── Error Handling
    ├── Connection failures
    ├── Query timeouts
    └── Data integrity issues
```

### 2. Key Features

1. **Dual Database Support**: Seamless integration with PostgreSQL and InfluxDB
2. **Connection Pooling**: Efficient resource management with async connection pools
3. **Query Optimization**: Smart query building with indexes and partitioning awareness
4. **Data Consistency**: ACID compliance for critical operations
5. **Performance**: Sub-second response times for typical queries
6. **Monitoring**: Database health metrics and query performance tracking

### 3. Integration Points

- Inherits from `HistoricalDataProvider` abstract base class
- Uses existing DAL repositories (HistoryRepository for time-series data)
- Integrates with ConnectionPool for PostgreSQL connections
- Uses TimeSeriesDB client for InfluxDB operations
- Leverages LoggerService for structured logging
- Reports metrics to MonitoringService

## Implementation Plan

### Phase 1: Core Structure and Dependencies

```python
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional
from decimal import Decimal
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import pandas as pd

from ..simulated_market_price_service import (
    DataRequest,
    HistoricalDataPoint,
    HistoricalDataProvider,
)
from ..dal.connection_pool import ConnectionPool
from ..dal.influxdb_client import TimeSeriesDB
from ..dal.repositories.history_repository import HistoryRepository
from ..dal.models import Order, Position, RiskMetrics
from ..logger_service import LoggerService


class DatabaseDataProvider(HistoricalDataProvider):
    """Production-ready provider for fetching historical data from databases."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: logging.Logger,
        connection_pool: Optional[ConnectionPool] = None,
        ts_db: Optional[TimeSeriesDB] = None
    ) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Database connections
        self._connection_pool = connection_pool
        self._ts_db = ts_db
        self._history_repo: Optional[HistoryRepository] = None
        
        # Configuration
        self._query_timeout = config.get("database.query_timeout", 30)
        self._max_batch_size = config.get("database.max_batch_size", 10000)
        self._enable_query_cache = config.get("database.enable_cache", True)
        
        # Performance tracking
        self._query_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_query_time": 0.0
        }
        
        # Symbol validation cache
        self._valid_symbols_cache: Dict[str, bool] = {}
        self._cache_ttl = 3600  # 1 hour
```

### Phase 2: Database Integration

```python
async def initialize(self) -> None:
    """Initialize database connections and repositories."""
    try:
        # Initialize connection pool if not provided
        if not self._connection_pool:
            self._connection_pool = ConnectionPool(
                connection_string=self.config["database.postgresql_url"],
                min_connections=self.config.get("database.min_connections", 5),
                max_connections=self.config.get("database.max_connections", 20)
            )
            await self._connection_pool.initialize()
        
        # Initialize InfluxDB client if not provided
        if not self._ts_db:
            self._ts_db = TimeSeriesDB(
                url=self.config["database.influxdb_url"],
                token=self.config["database.influxdb_token"],
                org=self.config["database.influxdb_org"],
                bucket=self.config["database.influxdb_bucket"]
            )
            await self._ts_db.initialize()
        
        # Create history repository
        self._history_repo = HistoryRepository(self._ts_db, self.logger)
        
        self.logger.info(
            "DatabaseDataProvider initialized successfully",
            source_module=self._source_module
        )
        
    except Exception as e:
        self.logger.error(
            f"Failed to initialize DatabaseDataProvider: {e}",
            source_module=self._source_module,
            exc_info=True
        )
        raise

async def cleanup(self) -> None:
    """Clean up database connections."""
    try:
        if self._connection_pool:
            await self._connection_pool.close()
        
        if self._ts_db:
            await self._ts_db.close()
            
    except Exception as e:
        self.logger.error(
            f"Error during cleanup: {e}",
            source_module=self._source_module
        )
```

### Phase 3: Data Fetching Implementation

```python
async def fetch_data(self, request: DataRequest) -> List[HistoricalDataPoint]:
    """Fetch historical data from appropriate database based on data type."""
    
    # Validate request
    if not await self.validate_symbol(request.symbol):
        raise ValueError(f"Invalid symbol: {request.symbol}")
    
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
        
        self.logger.debug(
            f"Fetched {len(data_points)} data points in {query_time:.3f}s",
            source_module=self._source_module,
            extra={
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
            source_module=self._source_module,
            exc_info=True,
            extra={"request": request}
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
        
        if request.validate_data and self._validate_ohlcv(data_point):
            data_points.append(data_point)
        elif not request.validate_data:
            data_points.append(data_point)
    
    return data_points

async def _fetch_from_postgresql(self, request: DataRequest) -> List[HistoricalDataPoint]:
    """Fetch aggregated or processed data from PostgreSQL."""
    
    async with self._get_db_session() as session:
        # Build optimized query based on request parameters
        query = self._build_historical_query(request)
        
        # Execute with timeout
        result = await asyncio.wait_for(
            session.execute(query),
            timeout=self._query_timeout
        )
        
        # Process results
        data_points = []
        for row in result:
            data_point = self._map_row_to_data_point(row, request.symbol)
            if data_point:
                data_points.append(data_point)
        
        return data_points

@asynccontextmanager
async def _get_db_session(self):
    """Get database session from connection pool."""
    if not self._connection_pool:
        raise RuntimeError("Connection pool not initialized")
        
    async with self._connection_pool.get_session() as session:
        yield session
```

### Phase 4: Query Optimization and Caching

```python
def _build_historical_query(self, request: DataRequest):
    """Build optimized SQL query for historical data."""
    
    # Base query structure - example using aggregated trade data
    # In practice, this would query appropriate tables based on data requirements
    
    query = f"""
        WITH time_buckets AS (
            SELECT 
                time_bucket('{request.frequency}', created_at) AS bucket_time,
                symbol,
                FIRST(price, created_at) AS open,
                MAX(price) AS high,
                MIN(price) AS low,
                LAST(price, created_at) AS close,
                SUM(volume) AS volume,
                COUNT(*) AS trade_count
            FROM trades
            WHERE 
                symbol = :symbol
                AND created_at >= :start_date
                AND created_at <= :end_date
            GROUP BY bucket_time, symbol
        )
        SELECT * FROM time_buckets
        ORDER BY bucket_time ASC
        LIMIT :limit
    """
    
    return query

async def _implement_query_cache(self, cache_key: str, query_func, ttl: int = 300):
    """Implement query result caching with TTL."""
    
    if self._enable_query_cache:
        # Check cache first
        cached_result = await self._cache.get(cache_key)
        if cached_result:
            self._query_metrics["cache_hits"] += 1
            return cached_result
    
    # Execute query
    result = await query_func()
    
    # Cache result
    if self._enable_query_cache and result:
        await self._cache.set(cache_key, result, ttl=ttl)
    
    return result
```

### Phase 5: Symbol Validation and Metadata

```python
async def validate_symbol(self, symbol: str) -> bool:
    """Validate if symbol exists in database with caching."""
    
    # Check cache first
    if symbol in self._valid_symbols_cache:
        cache_entry = self._valid_symbols_cache[symbol]
        if self._is_cache_valid(cache_entry):
            return cache_entry["valid"]
    
    try:
        # Query database for symbol validity
        async with self._get_db_session() as session:
            # Check if symbol has any recent trades or orders
            query = select(Order.id).where(
                Order.trading_pair == symbol
            ).limit(1)
            
            result = await session.execute(query)
            exists = result.scalar() is not None
            
            # Cache result
            self._valid_symbols_cache[symbol] = {
                "valid": exists,
                "timestamp": datetime.now(UTC)
            }
            
            return exists
            
    except Exception as e:
        self.logger.warning(
            f"Failed to validate symbol {symbol}: {e}",
            source_module=self._source_module
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
        "total_volume": None
    }
    
    try:
        async with self._get_db_session() as session:
            # Get aggregated statistics
            query = f"""
                SELECT 
                    MIN(price) as min_price,
                    MAX(price) as max_price,
                    AVG(ask_price - bid_price) as avg_spread,
                    SUM(volume) as total_volume
                FROM market_data
                WHERE symbol = :symbol
                AND created_at >= NOW() - INTERVAL '30 days'
            """
            
            result = await session.execute(query, {"symbol": symbol})
            row = result.fetchone()
            
            if row:
                metadata.update({
                    "min_price": float(row.min_price) if row.min_price else None,
                    "max_price": float(row.max_price) if row.max_price else None,
                    "avg_spread": float(row.avg_spread) if row.avg_spread else None,
                    "total_volume": float(row.total_volume) if row.total_volume else None
                })
                
    except Exception as e:
        self.logger.error(
            f"Failed to get symbol metadata: {e}",
            source_module=self._source_module
        )
    
    return metadata
```

### Phase 6: Performance and Monitoring

```python
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
            source_module=self._source_module
        )

async def get_diagnostics(self) -> Dict[str, Any]:
    """Get diagnostic information about database provider health."""
    
    diagnostics = {
        "provider": "DatabaseDataProvider",
        "status": "healthy",
        "metrics": self._query_metrics.copy(),
        "connections": {
            "postgresql": "unknown",
            "influxdb": "unknown"
        }
    }
    
    # Check PostgreSQL connection
    try:
        async with self._get_db_session() as session:
            await session.execute("SELECT 1")
            diagnostics["connections"]["postgresql"] = "healthy"
    except Exception as e:
        diagnostics["connections"]["postgresql"] = f"error: {str(e)}"
        diagnostics["status"] = "degraded"
    
    # Check InfluxDB connection
    try:
        if self._ts_db:
            # Perform health check query
            await self._ts_db.ping()
            diagnostics["connections"]["influxdb"] = "healthy"
    except Exception as e:
        diagnostics["connections"]["influxdb"] = f"error: {str(e)}"
        diagnostics["status"] = "degraded"
    
    return diagnostics
```

## Testing Strategy

1. **Unit Tests**
   - Mock database connections
   - Test query building logic
   - Validate data transformation
   - Test error handling scenarios

2. **Integration Tests**
   - Test with real database connections
   - Verify query performance
   - Test connection pool behavior
   - Validate transaction handling

3. **Performance Tests**
   - Benchmark query response times
   - Test concurrent query handling
   - Validate connection pool scaling
   - Memory usage profiling

4. **Data Validation Tests**
   - Test OHLCV data integrity
   - Verify timezone handling
   - Test decimal precision
   - Validate edge cases

## Monitoring & Observability

1. **Metrics to Track**
   - Query execution time (p50, p95, p99)
   - Connection pool utilization
   - Cache hit/miss ratios
   - Error rates by query type
   - Data quality scores

2. **Alerts**
   - Connection pool exhaustion
   - Query timeout rates > 1%
   - Database connection failures
   - Slow query warnings
   - Data validation failures

## Security Considerations

1. **SQL Injection Prevention**
   - Use parameterized queries
   - Validate all input parameters
   - Escape special characters
   - Use SQLAlchemy ORM where possible

2. **Access Control**
   - Implement row-level security
   - Use read-only database users
   - Audit query access patterns
   - Encrypt sensitive data

3. **Connection Security**
   - Use SSL/TLS for all connections
   - Rotate database credentials
   - Implement connection timeouts
   - Monitor for suspicious queries

## Performance Optimization

1. **Query Optimization**
   - Use appropriate indexes
   - Partition large tables
   - Implement query result caching
   - Use database-specific optimizations

2. **Connection Management**
   - Tune connection pool size
   - Implement connection health checks
   - Use persistent connections
   - Monitor connection lifecycle

3. **Data Transfer**
   - Implement pagination for large results
   - Use compression where appropriate
   - Stream large result sets
   - Optimize data serialization

## Future Enhancements

1. **Advanced Caching**
   - Implement Redis caching layer
   - Smart cache invalidation
   - Predictive cache warming
   - Distributed cache support

2. **Query Intelligence**
   - Query plan analysis
   - Automatic index recommendations
   - Query rewriting optimization
   - Cost-based query routing

3. **Multi-Database Support**
   - Support for additional databases
   - Cross-database joins
   - Database federation
   - Read replica routing