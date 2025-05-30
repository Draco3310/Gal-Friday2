"""Order repository implementation using SQLAlchemy."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.order import Order

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class OrderRepository(BaseRepository[Order]):
    """Repository for order data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker, logger: "LoggerService"
    ) -> None:
        """Initialize the order repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, Order, logger)

    async def get_active_orders(self) -> Sequence[Order]:
        """Get all active orders."""
        # Assuming "ACTIVE" is a valid status string.
        # BaseRepository's find_all handles Sequence[T] return.
        return await self.find_all(
            filters={"status": "ACTIVE"}, order_by="created_at DESC"
        )

    async def get_orders_by_signal(self, signal_id: str) -> Sequence[Order]:
        """Get all orders for a signal."""
        # Assuming signal_id in Order model is a string or compatible type.
        return await self.find_all(
            filters={"signal_id": signal_id}, order_by="created_at ASC"
        )

    async def update_order_status(
        self,
        order_id: str, # Assuming order_id is the primary key (e.g. UUID as string)
        status: str,
        filled_quantity: Decimal | None = None,
        average_fill_price: Decimal | None = None,
    ) -> Order | None:
        """Update order execution status. Returns the updated order or None if not found."""
        updates: dict[str, Any] = {"status": status}

        if filled_quantity is not None:
            updates["filled_quantity"] = filled_quantity # Keep as Decimal

        if average_fill_price is not None:
            updates["average_fill_price"] = average_fill_price # Keep as Decimal
        
        # self.update returns T | None, which is Order | None here
        return await self.update(order_id, updates)

    async def get_recent_orders(self, hours: int = 24) -> Sequence[Order]:
        """Get orders from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(Order)
                    .where(Order.created_at > cutoff)
                    .order_by(Order.created_at.desc())
                )
                result = await session.execute(stmt)
                orders = result.scalars().all()
                self.logger.debug(
                    f"Retrieved {len(orders)} recent orders from last {hours} hours.",
                    source_module=self._source_module
                )
                return orders
        except Exception as e:
            self.logger.exception(
                f"Error retrieving recent orders: {e}",
                source_module=self._source_module
            )
            raise # Or return empty list: return []

    async def find_by_exchange_id(self, exchange_order_id: str) -> Order | None:
        """Find order by exchange order ID."""
        # find_all returns a Sequence[Order]
        orders = await self.find_all(
            filters={"exchange_order_id": exchange_order_id}, limit=1
        )
        if orders:
            self.logger.debug(
                f"Found order with exchange_order_id {exchange_order_id}",
                source_module=self._source_module
            )
            return orders[0]
        
        self.logger.debug(
            f"No order found with exchange_order_id {exchange_order_id}",
            source_module=self._source_module
        )
        return None
