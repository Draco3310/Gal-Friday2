# High Priority Solution: Complete Database Integrations

## Overview
This document provides the complete implementation plan for integrating PostgreSQL and InfluxDB throughout the Gal-Friday system. Currently, many components use in-memory storage which leads to data loss on restart and limits scalability.

## Current State Issues

1. **In-Memory Storage Dependencies**
   - `PortfolioManager`: Positions stored in memory dictionaries
   - `ExecutionHandler`: Order tracking in memory
   - `MonitoringService`: Metrics only in memory buffers
   - `RiskManager`: Position limits tracked in memory

2. **Missing Persistence**
   - No trade history storage
   - No model prediction logging
   - No system state persistence
   - No audit trail for risk decisions

3. **Database Schema Gaps**
   - Schema exists but not utilized
   - No connection pooling
   - No transaction management
   - No migration system

## Solution Architecture

### 1. Data Access Layer (DAL) Implementation

#### 1.1 Base Repository Pattern
```python
# gal_friday/dal/base.py
"""Base repository pattern for data access."""

import asyncio
import asyncpg
from typing import Dict, List, Optional, Any, TypeVar, Generic
from datetime import datetime, UTC
import json
from decimal import Decimal
from abc import ABC, abstractmethod

from gal_friday.logger_service import LoggerService

T = TypeVar('T')

class BaseEntity(ABC):
    """Base class for all database entities."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary for database storage."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEntity':
        """Create entity from database record."""
        pass


class BaseRepository(Generic[T], ABC):
    """Base repository with common database operations."""
    
    def __init__(self, db_pool: asyncpg.Pool, logger: LoggerService, table_name: str):
        self.db_pool = db_pool
        self.logger = logger
        self.table_name = table_name
        self._source_module = self.__class__.__name__
        
    async def insert(self, entity: T) -> str:
        """Insert entity and return ID."""
        data = entity.to_dict()
        
        # Generate column names and placeholders
        columns = list(data.keys())
        placeholders = [f'${i+1}' for i in range(len(columns))]
        
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING id
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval(query, *data.values())
                return str(result)
        except Exception:
            self.logger.exception(
                f"Error inserting into {self.table_name}",
                source_module=self._source_module
            )
            raise
            
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update entity by ID."""
        # Generate SET clause
        set_clauses = [f"{col} = ${i+2}" for i, col in enumerate(updates.keys())]
        
        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(query, id, *updates.values())
                return result.split()[-1] != '0'
        except Exception:
            self.logger.exception(
                f"Error updating {self.table_name}",
                source_module=self._source_module
            )
            raise
            
    async def find_by_id(self, id: str) -> Optional[T]:
        """Find entity by ID."""
        query = f"SELECT * FROM {self.table_name} WHERE id = $1"
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, id)
                if row:
                    return self._row_to_entity(dict(row))
                return None
        except Exception:
            self.logger.exception(
                f"Error finding by id in {self.table_name}",
                source_module=self._source_module
            )
            raise
            
    async def find_many(self, 
                       filters: Optional[Dict[str, Any]] = None,
                       order_by: Optional[str] = None,
                       limit: int = 100,
                       offset: int = 0) -> List[T]:
        """Find multiple entities with filtering."""
        query_parts = [f"SELECT * FROM {self.table_name}"]
        params = []
        param_count = 0
        
        # Add WHERE clause if filters provided
        if filters:
            where_clauses = []
            for col, value in filters.items():
                param_count += 1
                where_clauses.append(f"{col} = ${param_count}")
                params.append(value)
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
            
        # Add ORDER BY
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")
            
        # Add pagination
        param_count += 1
        query_parts.append(f"LIMIT ${param_count}")
        params.append(limit)
        
        param_count += 1
        query_parts.append(f"OFFSET ${param_count}")
        params.append(offset)
        
        query = ' '.join(query_parts)
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [self._row_to_entity(dict(row)) for row in rows]
        except Exception:
            self.logger.exception(
                f"Error finding many in {self.table_name}",
                source_module=self._source_module
            )
            raise
            
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        query = f"DELETE FROM {self.table_name} WHERE id = $1"
        
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(query, id)
                return result.split()[-1] != '0'
        except Exception:
            self.logger.exception(
                f"Error deleting from {self.table_name}",
                source_module=self._source_module
            )
            raise
            
    async def execute_transaction(self, operations):
        """Execute multiple operations in a transaction."""
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                for operation in operations:
                    await operation(conn)
                    
    @abstractmethod
    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """Convert database row to entity."""
        pass
```

#### 1.2 Connection Pool Management
```python
# gal_friday/dal/connection_pool.py
"""Database connection pool management."""

import asyncpg
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class DatabaseConnectionPool:
    """Manages database connection pools."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize connection pools."""
        async with self._pool_lock:
            if self.postgres_pool is None:
                try:
                    self.postgres_pool = await asyncpg.create_pool(
                        self.config.get("database.connection_string"),
                        min_size=self.config.get_int("database.pool.min_size", 10),
                        max_size=self.config.get_int("database.pool.max_size", 20),
                        max_inactive_connection_lifetime=300,
                        command_timeout=10
                    )
                    
                    self.logger.info(
                        "Database connection pool initialized",
                        source_module=self._source_module
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to initialize database pool",
                        source_module=self._source_module
                    )
                    raise
                    
    async def close(self):
        """Close all connection pools."""
        if self.postgres_pool:
            await self.postgres_pool.close()
            self.postgres_pool = None
            
    @asynccontextmanager
    async def acquire(self):
        """Acquire a database connection."""
        if not self.postgres_pool:
            await self.initialize()
            
        async with self.postgres_pool.acquire() as conn:
            yield conn
            
    async def execute_query(self, query: str, *args):
        """Execute a query and return results."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
            
    async def execute_command(self, command: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE)."""
        async with self.acquire() as conn:
            return await conn.execute(command, *args)
```

### 2. Entity Models and Repositories

#### 2.1 Order Entity and Repository
```python
# gal_friday/dal/entities/order.py
"""Order entity model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
import uuid

from gal_friday.dal.base import BaseEntity


@dataclass
class OrderEntity(BaseEntity):
    """Database entity for orders."""
    
    order_id: str
    signal_id: str
    trading_pair: str
    exchange: str
    side: str  # BUY/SELL
    order_type: str  # MARKET/LIMIT
    quantity: Decimal
    limit_price: Optional[Decimal]
    status: str
    exchange_order_id: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to database record."""
        return {
            "id": self.order_id,
            "signal_id": self.signal_id,
            "trading_pair": self.trading_pair,
            "exchange": self.exchange,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": float(self.quantity),
            "limit_price": float(self.limit_price) if self.limit_price else None,
            "status": self.status,
            "exchange_order_id": self.exchange_order_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "filled_quantity": float(self.filled_quantity),
            "average_fill_price": float(self.average_fill_price) if self.average_fill_price else None,
            "commission": float(self.commission) if self.commission else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderEntity':
        """Create from database record."""
        return cls(
            order_id=str(data["id"]),
            signal_id=str(data["signal_id"]),
            trading_pair=data["trading_pair"],
            exchange=data["exchange"],
            side=data["side"],
            order_type=data["order_type"],
            quantity=Decimal(str(data["quantity"])),
            limit_price=Decimal(str(data["limit_price"])) if data["limit_price"] else None,
            status=data["status"],
            exchange_order_id=data["exchange_order_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            filled_quantity=Decimal(str(data["filled_quantity"])),
            average_fill_price=Decimal(str(data["average_fill_price"])) if data["average_fill_price"] else None,
            commission=Decimal(str(data["commission"])) if data["commission"] else None
        )


# gal_friday/dal/repositories/order_repository.py
"""Order repository implementation."""

from typing import List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.entities.order import OrderEntity


class OrderRepository(BaseRepository[OrderEntity]):
    """Repository for order data persistence."""
    
    def __init__(self, db_pool, logger):
        super().__init__(db_pool, logger, "orders")
        
    def _row_to_entity(self, row: Dict[str, Any]) -> OrderEntity:
        """Convert database row to order entity."""
        return OrderEntity.from_dict(row)
        
    async def get_active_orders(self) -> List[OrderEntity]:
        """Get all active orders."""
        return await self.find_many(
            filters={"status": "ACTIVE"},
            order_by="created_at DESC"
        )
        
    async def get_orders_by_signal(self, signal_id: str) -> List[OrderEntity]:
        """Get all orders for a signal."""
        return await self.find_many(
            filters={"signal_id": signal_id},
            order_by="created_at ASC"
        )
        
    async def update_order_status(self, 
                                 order_id: str,
                                 status: str,
                                 filled_quantity: Optional[Decimal] = None,
                                 average_fill_price: Optional[Decimal] = None) -> bool:
        """Update order execution status."""
        updates = {"status": status}
        
        if filled_quantity is not None:
            updates["filled_quantity"] = float(filled_quantity)
            
        if average_fill_price is not None:
            updates["average_fill_price"] = float(average_fill_price)
            
        return await self.update(order_id, updates)
        
    async def get_recent_orders(self, hours: int = 24) -> List[OrderEntity]:
        """Get orders from the last N hours."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        query = """
            SELECT * FROM orders
            WHERE created_at > $1
            ORDER BY created_at DESC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, cutoff)
            return [self._row_to_entity(dict(row)) for row in rows]
```

#### 2.2 Position Entity and Repository
```python
# gal_friday/dal/entities/position.py
"""Position entity model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any

from gal_friday.dal.base import BaseEntity


@dataclass
class PositionEntity(BaseEntity):
    """Database entity for positions."""
    
    position_id: str
    trading_pair: str
    side: str  # LONG/SHORT
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    opened_at: datetime
    closed_at: Optional[datetime]
    is_active: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to database record."""
        return {
            "id": self.position_id,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "current_price": float(self.current_price),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "is_active": self.is_active
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionEntity':
        """Create from database record."""
        return cls(
            position_id=str(data["id"]),
            trading_pair=data["trading_pair"],
            side=data["side"],
            quantity=Decimal(str(data["quantity"])),
            entry_price=Decimal(str(data["entry_price"])),
            current_price=Decimal(str(data["current_price"])),
            realized_pnl=Decimal(str(data["realized_pnl"])),
            unrealized_pnl=Decimal(str(data["unrealized_pnl"])),
            opened_at=data["opened_at"],
            closed_at=data["closed_at"],
            is_active=data["is_active"]
        )


# gal_friday/dal/repositories/position_repository.py
"""Position repository implementation."""

from typing import List, Optional, Dict, Any
from decimal import Decimal

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.entities.position import PositionEntity


class PositionRepository(BaseRepository[PositionEntity]):
    """Repository for position data persistence."""
    
    def __init__(self, db_pool, logger):
        super().__init__(db_pool, logger, "positions")
        
    def _row_to_entity(self, row: Dict[str, Any]) -> PositionEntity:
        """Convert database row to position entity."""
        return PositionEntity.from_dict(row)
        
    async def get_active_positions(self) -> List[PositionEntity]:
        """Get all active positions."""
        return await self.find_many(
            filters={"is_active": True},
            order_by="opened_at DESC"
        )
        
    async def get_position_by_pair(self, trading_pair: str) -> Optional[PositionEntity]:
        """Get active position for a trading pair."""
        positions = await self.find_many(
            filters={"trading_pair": trading_pair, "is_active": True},
            limit=1
        )
        return positions[0] if positions else None
        
    async def update_position_price(self, 
                                   position_id: str,
                                   current_price: Decimal,
                                   unrealized_pnl: Decimal) -> bool:
        """Update position with current market price."""
        return await self.update(position_id, {
            "current_price": float(current_price),
            "unrealized_pnl": float(unrealized_pnl)
        })
        
    async def close_position(self,
                           position_id: str,
                           realized_pnl: Decimal) -> bool:
        """Mark position as closed."""
        return await self.update(position_id, {
            "is_active": False,
            "closed_at": datetime.now(UTC),
            "realized_pnl": float(realized_pnl),
            "unrealized_pnl": 0
        })
        
    async def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        query = """
            SELECT 
                COUNT(*) FILTER (WHERE is_active = true) as active_positions,
                COUNT(*) FILTER (WHERE is_active = false) as closed_positions,
                SUM(realized_pnl) as total_realized_pnl,
                SUM(unrealized_pnl) FILTER (WHERE is_active = true) as total_unrealized_pnl
            FROM positions
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query)
            return dict(row)
```

### 3. Time-Series Data with InfluxDB

#### 3.1 InfluxDB Client Wrapper
```python
# gal_friday/dal/influxdb_client.py
"""InfluxDB client for time-series data."""

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class TimeSeriesDB:
    """InfluxDB client wrapper for time-series data."""
    
    def __init__(self, config: ConfigManager, logger: LoggerService):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Initialize InfluxDB client
        self.client = InfluxDBClient(
            url=config.get("influxdb.url"),
            token=config.get("influxdb.token"),
            org=config.get("influxdb.org")
        )
        
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = config.get("influxdb.bucket", "market-data")
        
    async def write_market_data(self, 
                               trading_pair: str,
                               exchange: str,
                               data_type: str,  # 'ohlcv', 'tick', 'orderbook'
                               data: Dict[str, Any]):
        """Write market data point."""
        try:
            point = Point("market_data") \
                .tag("trading_pair", trading_pair) \
                .tag("exchange", exchange) \
                .tag("data_type", data_type)
                
            # Add fields based on data type
            if data_type == "ohlcv":
                point.field("open", float(data["open"])) \
                     .field("high", float(data["high"])) \
                     .field("low", float(data["low"])) \
                     .field("close", float(data["close"])) \
                     .field("volume", float(data["volume"]))
                     
            elif data_type == "tick":
                point.field("price", float(data["price"])) \
                     .field("volume", float(data["volume"])) \
                     .field("side", data["side"])
                     
            elif data_type == "orderbook":
                point.field("bid", float(data["bid"])) \
                     .field("ask", float(data["ask"])) \
                     .field("bid_volume", float(data["bid_volume"])) \
                     .field("ask_volume", float(data["ask_volume"])) \
                     .field("spread", float(data["spread"]))
                     
            # Set timestamp
            if "timestamp" in data:
                point.time(data["timestamp"])
                
            self.write_api.write(bucket=self.bucket, record=point)
            
        except Exception:
            self.logger.exception(
                "Error writing market data to InfluxDB",
                source_module=self._source_module
            )
            
    async def write_metrics(self, 
                          metric_name: str,
                          value: float,
                          tags: Dict[str, str] = None):
        """Write system metrics."""
        try:
            point = Point("system_metrics") \
                .field(metric_name, value)
                
            if tags:
                for key, val in tags.items():
                    point.tag(key, val)
                    
            self.write_api.write(bucket=self.bucket, record=point)
            
        except Exception:
            self.logger.exception(
                "Error writing metrics to InfluxDB",
                source_module=self._source_module
            )
            
    async def query_ohlcv(self,
                         trading_pair: str,
                         timeframe: str,
                         start_time: datetime,
                         end_time: datetime) -> List[Dict[str, Any]]:
        """Query OHLCV data for a time range."""
        query = f'''
            from(bucket: "{self.bucket}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> filter(fn: (r) => r["_measurement"] == "market_data")
            |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
            |> filter(fn: (r) => r["data_type"] == "ohlcv")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> window(every: {timeframe})
            |> sort(columns: ["_time"])
        '''
        
        try:
            result = self.query_api.query(query=query)
            
            candles = []
            for table in result:
                for record in table.records:
                    candles.append({
                        "timestamp": record.get_time(),
                        "open": record.values.get("open"),
                        "high": record.values.get("high"),
                        "low": record.values.get("low"),
                        "close": record.values.get("close"),
                        "volume": record.values.get("volume")
                    })
                    
            return candles
            
        except Exception:
            self.logger.exception(
                "Error querying OHLCV data from InfluxDB",
                source_module=self._source_module
            )
            return []
            
    def close(self):
        """Close InfluxDB client."""
        self.client.close()
```

### 4. Migration System

#### 4.1 Migration Manager
```python
# gal_friday/dal/migrations/migration_manager.py
"""Database migration management system."""

import os
import asyncpg
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from gal_friday.logger_service import LoggerService


class MigrationManager:
    """Manages database schema migrations."""
    
    def __init__(self, db_pool: asyncpg.Pool, logger: LoggerService):
        self.db_pool = db_pool
        self.logger = logger
        self._source_module = self.__class__.__name__
        self.migrations_dir = Path(__file__).parent / "scripts"
        
    async def initialize(self):
        """Create migrations table if not exists."""
        query = """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        async with self.db_pool.acquire() as conn:
            await conn.execute(query)
            
    async def get_current_version(self) -> int:
        """Get current schema version."""
        query = "SELECT MAX(version) FROM schema_migrations"
        
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval(query)
            return result or 0
            
    async def get_pending_migrations(self) -> List[tuple]:
        """Get list of pending migrations."""
        current_version = await self.get_current_version()
        
        migrations = []
        for file in sorted(self.migrations_dir.glob("*.sql")):
            # File format: 001_create_tables.sql
            version = int(file.stem.split("_")[0])
            if version > current_version:
                migrations.append((version, file))
                
        return migrations
        
    async def run_migration(self, version: int, migration_file: Path):
        """Execute a single migration."""
        self.logger.info(
            f"Running migration {version}: {migration_file.name}",
            source_module=self._source_module
        )
        
        # Read migration SQL
        sql = migration_file.read_text()
        
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Execute migration
                await conn.execute(sql)
                
                # Record migration
                await conn.execute(
                    "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
                    version,
                    migration_file.stem
                )
                
        self.logger.info(
            f"Migration {version} completed successfully",
            source_module=self._source_module
        )
        
    async def run_all_migrations(self):
        """Run all pending migrations."""
        await self.initialize()
        
        pending = await self.get_pending_migrations()
        if not pending:
            self.logger.info(
                "No pending migrations",
                source_module=self._source_module
            )
            return
            
        for version, file in pending:
            await self.run_migration(version, file)
            
        self.logger.info(
            f"Completed {len(pending)} migrations",
            source_module=self._source_module
        )
        
    async def rollback_migration(self, target_version: int):
        """Rollback to specific version."""
        current_version = await self.get_current_version()
        
        if target_version >= current_version:
            self.logger.warning(
                f"Target version {target_version} is not less than current {current_version}",
                source_module=self._source_module
            )
            return
            
        # Find rollback scripts
        for version in range(current_version, target_version, -1):
            rollback_file = self.migrations_dir / f"{version:03d}_rollback.sql"
            if rollback_file.exists():
                await self._execute_rollback(version, rollback_file)
            else:
                self.logger.error(
                    f"Rollback script not found for version {version}",
                    source_module=self._source_module
                )
                raise FileNotFoundError(f"Missing rollback script for version {version}")
                
    async def _execute_rollback(self, version: int, rollback_file: Path):
        """Execute rollback script."""
        sql = rollback_file.read_text()
        
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE version = $1",
                    version
                )
```

#### 4.2 Migration Scripts
```sql
-- gal_friday/dal/migrations/scripts/001_initial_schema.sql
-- Initial database schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    limit_price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL,
    exchange_order_id VARCHAR(100),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    average_fill_price DECIMAL(20, 8),
    commission DECIMAL(20, 8),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    
    INDEX idx_orders_signal_id (signal_id),
    INDEX idx_orders_status (status),
    INDEX idx_orders_created_at (created_at)
);

-- Positions table
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trading_pair VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8) NOT NULL,
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    unrealized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    
    INDEX idx_positions_pair (trading_pair),
    INDEX idx_positions_active (is_active),
    INDEX idx_positions_opened_at (opened_at)
);

-- Trade signals table
CREATE TABLE IF NOT EXISTS trade_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trading_pair VARCHAR(20) NOT NULL,
    strategy_id VARCHAR(50) NOT NULL,
    side VARCHAR(10) NOT NULL,
    entry_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    confidence DECIMAL(5, 4),
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP,
    
    INDEX idx_signals_status (status),
    INDEX idx_signals_created_at (created_at)
);
```

### 5. Service Integration Updates

#### 5.1 Updated Portfolio Manager with Database
```python
# Updates to gal_friday/portfolio_manager.py

class PortfolioManager:
    """Portfolio management with database persistence."""
    
    def __init__(self, config, pubsub, logger, db_pool):
        # ... existing init code ...
        self.position_repo = PositionRepository(db_pool, logger)
        
    async def update_position(self, execution_report: ExecutionReportEvent):
        """Update position with database persistence."""
        # ... existing position calculation logic ...
        
        # Persist to database
        if position_exists:
            await self.position_repo.update_position_price(
                position_id=position.position_id,
                current_price=execution_report.average_fill_price,
                unrealized_pnl=unrealized_pnl
            )
        else:
            new_position = PositionEntity(
                position_id=str(uuid.uuid4()),
                trading_pair=execution_report.trading_pair,
                side="LONG" if execution_report.side == "BUY" else "SHORT",
                quantity=execution_report.quantity_filled,
                entry_price=execution_report.average_fill_price,
                current_price=execution_report.average_fill_price,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                opened_at=datetime.now(UTC),
                closed_at=None,
                is_active=True
            )
            await self.position_repo.insert(new_position)
            
    async def load_positions_from_db(self):
        """Load active positions from database on startup."""
        positions = await self.position_repo.get_active_positions()
        
        for position in positions:
            self.positions[position.trading_pair] = {
                "quantity": position.quantity,
                "side": position.side,
                "entry_price": position.entry_price,
                "position_id": position.position_id
            }
```

## Implementation Steps

### Phase 1: Core DAL Implementation (3 days)
1. Implement base repository pattern
2. Create connection pool management
3. Set up migration system
4. Create initial migration scripts

### Phase 2: Entity Repositories (3 days)
1. Implement Order repository
2. Implement Position repository
3. Implement Signal repository
4. Add transaction support

### Phase 3: Time-Series Integration (2 days)
1. Set up InfluxDB client
2. Implement market data persistence
3. Add metrics collection
4. Create data retention policies

### Phase 4: Service Integration (4 days)
1. Update PortfolioManager with persistence
2. Update ExecutionHandler with persistence
3. Update MonitoringService with persistence
4. Update RiskManager with persistence

### Phase 5: Testing and Migration (2 days)
1. Create integration tests
2. Test transaction rollback
3. Performance testing
4. Data migration from in-memory

## Success Criteria

1. **Zero Data Loss**: System restart preserves all state
2. **Performance**: < 10ms for simple queries, < 50ms for complex queries
3. **Reliability**: 99.99% uptime for database operations
4. **Scalability**: Support for 1M+ records per table
5. **Auditability**: Complete audit trail for all changes

## Monitoring and Maintenance

1. **Query Performance**: Monitor slow queries
2. **Connection Pool**: Track pool utilization
3. **Storage Growth**: Monitor disk usage
4. **Backup Schedule**: Daily automated backups
5. **Migration Alerts**: Notify on migration failures 