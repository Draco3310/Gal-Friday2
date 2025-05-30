"""Order repository implementation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import asyncpg

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.entities.order import OrderEntity

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class OrderRepository(BaseRepository[OrderEntity]):
    """Repository for order data persistence."""

    def __init__(self, db_pool: asyncpg.Pool, logger: "LoggerService") -> None:
        """Initialize the order repository.

        Args:
            db_pool: Async database connection pool
            logger: Logger service instance
        """
        super().__init__(db_pool, logger, "orders")

    def _row_to_entity(self, row: dict[str, Any]) -> OrderEntity:
        """Convert database row to order entity."""
        return OrderEntity.from_dict(row)

    async def get_active_orders(self) -> list[OrderEntity]:
        """Get all active orders."""
        return await self.find_many(
            filters={"status": "ACTIVE"},
            order_by="created_at DESC",
        )

    async def get_orders_by_signal(self, signal_id: str) -> list[OrderEntity]:
        """Get all orders for a signal."""
        return await self.find_many(
            filters={"signal_id": signal_id},
            order_by="created_at ASC",
        )

    async def update_order_status(self,
                                 order_id: str,
                                 status: str,
                                 filled_quantity: Decimal | None = None,
                                 average_fill_price: Decimal | None = None) -> bool:
        """Update order execution status."""
        updates: dict[str, Any] = {"status": status}

        if filled_quantity is not None:
            updates["filled_quantity"] = float(filled_quantity)

        if average_fill_price is not None:
            updates["average_fill_price"] = float(average_fill_price)

        return await self.update(order_id, updates)

    async def get_recent_orders(self, hours: int = 24) -> list[OrderEntity]:
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

    async def find_by_exchange_id(self, exchange_order_id: str) -> OrderEntity | None:
        """Find order by exchange order ID."""
        orders = await self.find_many(
            filters={"exchange_order_id": exchange_order_id},
            limit=1,
        )
        return orders[0] if orders else None
