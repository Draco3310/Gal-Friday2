"""Order repository implementation using SQLAlchemy."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.order import Order
from typing import Any

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class OrderRepository(BaseRepository[Order]):
    """Repository for order data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService") -> None:
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
            filters={"status": "ACTIVE"}, order_by="created_at DESC")

    async def get_orders_by_signal(self, signal_id: str) -> Sequence[Order]:
        """Get all orders for a signal."""
        # Assuming signal_id in Order model is a string or compatible type.
        return await self.find_all(
            filters={"signal_id": signal_id}, order_by="created_at ASC")

    async def update_order_status(
        self,
        order_id: str, # Assuming order_id is the primary key (e.g. UUID as string)
        status: str,
        filled_quantity: Decimal | None = None,
        average_fill_price: Decimal | None = None) -> Order | None:
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
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
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
                    source_module=self._source_module)
                return orders
        except Exception as e:
            self.logger.exception(
                f"Error retrieving recent orders: {e}",
                source_module=self._source_module)
            raise # Or return empty list[Any]: return []

    async def find_by_exchange_id(self, exchange_order_id: str) -> Order | None:
        """Find order by exchange order ID."""
        # find_all returns a Sequence[Order]
        orders = await self.find_all(
            filters={"exchange_order_id": exchange_order_id}, limit=1)
        if orders:
            self.logger.debug(
                f"Found order with exchange_order_id {exchange_order_id}",
                source_module=self._source_module)
            return orders[0]

        self.logger.debug(
            f"No order found with exchange_order_id {exchange_order_id}",
            source_module=self._source_module)
        return None

    # NEW: Position-Order relationship methods
    
    async def get_orders_by_position(self, position_id: str | UUID) -> Sequence[Order]:
        """Get all orders that contributed to a specific position.
        
        Args:
            position_id: The UUID of the position (as string or UUID)
            
        Returns:
            Sequence[Any] of orders linked to the position, ordered by creation time
        """
        try:
            position_id_str = str(position_id)
            orders = await self.find_all(
                filters={"position_id": position_id_str}, 
                order_by="created_at ASC"
            )
            
            self.logger.debug(
                f"Found {len(orders)} orders linked to position {position_id_str}",
                source_module=self._source_module)
            return orders
            
        except Exception as e:
            self.logger.exception(
                f"Error retrieving orders for position {position_id}: {e}",
                source_module=self._source_module)
            return []

    async def get_orders_with_position(self) -> Sequence[Order]:
        """Get all orders that are linked to positions with relationship data loaded.
        
        Returns:
            Sequence[Any] of orders with position relationship loaded
        """
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(Order)
                    .options(selectinload(Order.position))
                    .where(Order.position_id.isnot(None))
                    .order_by(Order.created_at.desc())
                )
                result = await session.execute(stmt)
                orders = result.scalars().all()
                
                self.logger.debug(
                    f"Retrieved {len(orders)} orders with position relationships",
                    source_module=self._source_module)
                return orders
                
        except Exception as e:
            self.logger.exception(
                f"Error retrieving orders with positions: {e}",
                source_module=self._source_module)
            return []

    async def get_unlinked_filled_orders(self, hours: int = 24) -> Sequence[Order]:
        """Get filled orders that are not linked to any position.
        
        This is useful for data quality monitoring to identify orders that
        should be linked to positions but aren't.
        
        Args:
            hours: Number of hours to look back for recent orders
            
        Returns:
            Sequence[Any] of filled orders without position links
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(Order)
                    .where(
                        Order.status.in_(["FILLED", "PARTIALLY_FILLED"]),
                        Order.position_id.is_(None),
                        Order.created_at > cutoff
                    )
                    .order_by(Order.created_at.desc())
                )
                result = await session.execute(stmt)
                orders = result.scalars().all()
                
                self.logger.debug(
                    f"Found {len(orders)} unlinked filled orders from last {hours} hours",
                    source_module=self._source_module)
                return orders
                
        except Exception as e:
            self.logger.exception(
                f"Error retrieving unlinked filled orders: {e}",
                source_module=self._source_module)
            return []

    async def link_order_to_position(self, order_id: str | UUID, position_id: str | UUID) -> Order | None:
        """Link an order to a position.
        
        Args:
            order_id: The UUID of the order
            position_id: The UUID of the position
            
        Returns:
            Updated order instance or None if order not found
        """
        try:
            order_id_str = str(order_id)
            position_id_str = str(position_id)
            
            updated_order = await self.update(
                order_id_str, 
                {"position_id": position_id_str}
            )
            
            if updated_order:
                self.logger.info(
                    f"Successfully linked order {order_id_str} to position {position_id_str}",
                    source_module=self._source_module)
            else:
                self.logger.warning(
                    f"Order {order_id_str} not found for position linking",
                    source_module=self._source_module)
                
            return updated_order
            
        except Exception as e:
            self.logger.error(
                f"Failed to link order {order_id} to position {position_id}: {e}",
                source_module=self._source_module)
            return None

    async def unlink_order_from_position(self, order_id: str | UUID) -> Order | None:
        """Remove position link from an order.
        
        Args:
            order_id: The UUID of the order
            
        Returns:
            Updated order instance or None if order not found
        """
        try:
            order_id_str = str(order_id)
            
            updated_order = await self.update(
                order_id_str, 
                {"position_id": None}
            )
            
            if updated_order:
                self.logger.info(
                    f"Successfully unlinked order {order_id_str} from position",
                    source_module=self._source_module)
            else:
                self.logger.warning(
                    f"Order {order_id_str} not found for position unlinking",
                    source_module=self._source_module)
                
            return updated_order
            
        except Exception as e:
            self.logger.error(
                f"Failed to unlink order {order_id} from position: {e}",
                source_module=self._source_module)
            return None