"""Fill repository implementation using SQLAlchemy."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.models.fill import Fill
from typing import Any

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class FillRepository(BaseRepository):
    """Repository for fill data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService") -> None:
        """Initialize the fill repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, Fill, logger)

    async def get_fills_by_trading_pair(
        self,
        trading_pair: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
        offset: int = 0) -> Sequence[Fill]:
        """Get fills for a specific trading pair with optional date filtering.

        Args:
            trading_pair: Trading pair to filter by (e.g., 'BTC/USD')
            start_date: Optional start date filter (inclusive)
            end_date: Optional end date filter (inclusive)
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination

        Returns:
            Sequence[Any] of Fill objects matching the criteria
        """
        try:
            async with self.session_maker() as session:
                stmt = select(Fill).where(Fill.trading_pair == trading_pair)

                # Add date filters if provided
                if start_date:
                    stmt = stmt.where(Fill.filled_at >= start_date)
                if end_date:
                    stmt = stmt.where(Fill.filled_at <= end_date)

                # Add ordering, pagination
                stmt = stmt.order_by(Fill.filled_at.desc())
                
                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset is not None:
                    stmt = stmt.offset(offset)

                result = await session.execute(stmt)
                fills = result.scalars().all()
                
                self.logger.debug(
                    f"Retrieved {len(fills)} fills for {trading_pair} "
                    f"(limit: {limit}, offset: {offset})",
                    source_module=self._source_module)
                return fills

        except Exception as e:
            self.logger.exception(
                f"Error retrieving fills for {trading_pair}: {e}",
                source_module=self._source_module)
            raise

    async def get_fills_by_strategy(
        self,
        strategy_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000,
        offset: int = 0) -> Sequence[Fill]:
        """Get fills for a specific strategy.

        Note: This requires joining with Order and potentially TradeSignal tables
        since strategy_id is not directly on Fill model.

        Args:
            strategy_id: Strategy ID to filter by
            start_date: Optional start date filter (inclusive)
            end_date: Optional end date filter (inclusive)
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination

        Returns:
            Sequence[Any] of Fill objects for the strategy
        """
        try:
            # Import here to avoid circular imports
            from gal_friday.dal.models.order import Order
            from gal_friday.dal.models.trade_signal import TradeSignal

            async with self.session_maker() as session:
                # Join Fill -> Order -> TradeSignal to filter by strategy
                stmt = (
                    select(Fill)
                    .join(Order, Fill.order_pk == Order.id)
                    .join(TradeSignal, Order.signal_id == TradeSignal.id)
                    .where(TradeSignal.strategy_id == strategy_id)
                )

                # Add date filters if provided
                if start_date:
                    stmt = stmt.where(Fill.filled_at >= start_date)
                if end_date:
                    stmt = stmt.where(Fill.filled_at <= end_date)

                # Add ordering, pagination
                stmt = stmt.order_by(Fill.filled_at.desc())
                
                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset is not None:
                    stmt = stmt.offset(offset)

                result = await session.execute(stmt)
                fills = result.scalars().all()
                
                self.logger.debug(
                    f"Retrieved {len(fills)} fills for strategy {strategy_id} "
                    f"(limit: {limit}, offset: {offset})",
                    source_module=self._source_module)
                return fills

        except Exception as e:
            self.logger.exception(
                f"Error retrieving fills for strategy {strategy_id}: {e}",
                source_module=self._source_module)
            raise

    async def get_recent_fills(
        self,
        hours: int = 24,
        limit: int = 1000,
        offset: int = 0) -> Sequence[Fill]:
        """Get recent fills within the specified time window.

        Args:
            hours: Number of hours to look back
            limit: Maximum number of results to return
            offset: Number of results to skip for pagination

        Returns:
            Sequence[Any] of recent Fill objects
        """
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(Fill)
                    .where(Fill.filled_at >= cutoff)
                    .order_by(Fill.filled_at.desc())
                )
                
                if limit is not None:
                    stmt = stmt.limit(limit)
                if offset is not None:
                    stmt = stmt.offset(offset)

                result = await session.execute(stmt)
                fills = result.scalars().all()
                
                self.logger.debug(
                    f"Retrieved {len(fills)} recent fills from last {hours} hours "
                    f"(limit: {limit}, offset: {offset})",
                    source_module=self._source_module)
                return fills

        except Exception as e:
            self.logger.exception(
                f"Error retrieving recent fills: {e}",
                source_module=self._source_module)
            raise

    async def get_fills_count_by_trading_pair(
        self,
        trading_pair: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None) -> int:
        """Get count of fills for pagination purposes.

        Args:
            trading_pair: Trading pair to filter by
            start_date: Optional start date filter (inclusive)
            end_date: Optional end date filter (inclusive)

        Returns:
            Total count of fills matching the criteria
        """
        try:
            from sqlalchemy import func

            async with self.session_maker() as session:
                stmt = select(func.count(Fill.fill_pk)).where(Fill.trading_pair == trading_pair)

                # Add date filters if provided
                if start_date:
                    stmt = stmt.where(Fill.filled_at >= start_date)
                if end_date:
                    stmt = stmt.where(Fill.filled_at <= end_date)

                result = await session.execute(stmt)
                count = result.scalar() or 0
                
                self.logger.debug(
                    f"Found {count} total fills for {trading_pair}",
                    source_module=self._source_module)
                return count

        except Exception as e:
            self.logger.exception(
                f"Error counting fills for {trading_pair}: {e}",
                source_module=self._source_module)
            raise

    async def get_fills_by_order_id(self, order_pk: int) -> Sequence[Fill]:
        """Get all fills for a specific order.

        Args:
            order_pk: Order primary key to filter by

        Returns:
            Sequence[Any] of Fill objects for the order
        """
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(Fill)
                    .where(Fill.order_pk == order_pk)
                    .order_by(Fill.filled_at.asc())
                )

                result = await session.execute(stmt)
                fills = result.scalars().all()
                
                self.logger.debug(
                    f"Retrieved {len(fills)} fills for order {order_pk}",
                    source_module=self._source_module)
                return fills

        except Exception as e:
            self.logger.exception(
                f"Error retrieving fills for order {order_pk}: {e}",
                source_module=self._source_module)
            raise 