"""Position repository implementation."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import asyncpg

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.entities.position import PositionEntity

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class PositionRepository(BaseRepository[PositionEntity]):
    """Repository for position data persistence."""

    def __init__(self, db_pool: asyncpg.Pool, logger: "LoggerService") -> None:
        """Initialize the position repository.

        Args:
            db_pool: Async database connection pool
            logger: Logger service instance
        """
        super().__init__(db_pool, logger, "positions")

    def _row_to_entity(self, row: dict[str, Any]) -> PositionEntity:
        """Convert database row to position entity."""
        return PositionEntity.from_dict(row)

    async def get_active_positions(self) -> list[PositionEntity]:
        """Get all active positions."""
        return await self.find_many(
            filters={"is_active": True},
            order_by="opened_at DESC",
        )

    async def get_position_by_pair(self, trading_pair: str) -> PositionEntity | None:
        """Get active position for a trading pair."""
        positions = await self.find_many(
            filters={"trading_pair": trading_pair, "is_active": True},
            limit=1,
        )
        return positions[0] if positions else None

    async def update_position_price(self,
                                   position_id: str,
                                   current_price: Decimal,
                                   unrealized_pnl: Decimal) -> bool:
        """Update position with current market price."""
        return await self.update(position_id, {
            "current_price": float(current_price),
            "unrealized_pnl": float(unrealized_pnl),
        })

    async def close_position(self,
                           position_id: str,
                           realized_pnl: Decimal) -> bool:
        """Mark position as closed."""
        return await self.update(position_id, {
            "is_active": False,
            "closed_at": datetime.now(UTC),
            "realized_pnl": float(realized_pnl),
            "unrealized_pnl": 0,
        })

    async def get_position_summary(self) -> dict[str, Any]:
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
