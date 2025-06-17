"""Position repository implementation using SQLAlchemy."""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import Numeric, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.position import Position
from typing import Any

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class PositionRepository(BaseRepository[Position]):
    """Repository for position data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService") -> None:
        """Initialize the position repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, Position, logger)

    async def get_active_positions(self) -> Sequence[Position]:
        """Get all active positions."""
        return await self.find_all(
            filters={"is_active": True}, order_by="opened_at DESC")

    async def get_position_by_pair(self, trading_pair: str) -> Position | None:
        """Get active position for a trading pair."""
        positions = await self.find_all(
            filters={"trading_pair": trading_pair, "is_active": True}, limit=1)
        return positions[0] if positions else None

    async def update_position_price(
        self, position_id: str, current_price: Decimal, unrealized_pnl: Decimal) -> Position | None:
        """Update position with current market price. Returns the updated position or None."""
        return await self.update(
            position_id,
            {
                "current_price": current_price, # Keep as Decimal
                "unrealized_pnl": unrealized_pnl, # Keep as Decimal
            })

    async def close_position(
        self, position_id: str, realized_pnl: Decimal) -> Position | None:
        """Mark position as closed. Returns the updated position or None."""
        return await self.update(
            position_id,
            {
                "is_active": False,
                "closed_at": datetime.now(UTC),
                "realized_pnl": realized_pnl, # Keep as Decimal
                "unrealized_pnl": Decimal("0"), # Ensure it's a Decimal
            })

    async def get_position_summary(self) -> dict[str, Any]:
        """Get summary of all positions using SQLAlchemy."""
        try:
            async with self.session_maker() as session:
                stmt = select(
                    func.count(Position.id).filter(Position.is_active == True).label("active_positions"),
                    func.count(Position.id).filter(Position.is_active == False).label("closed_positions"),
                    func.sum(cast(Position.realized_pnl, Numeric)).label("total_realized_pnl"),
                    func.sum(cast(Position.unrealized_pnl, Numeric)).filter(Position.is_active == True).label("total_unrealized_pnl"))
                result = await session.execute(stmt)
                summary = result.one_or_none() # Using one_or_none() as SUM can return None if no rows

                if summary:
                    self.logger.debug("Retrieved position summary.", source_module=self._source_module)
                    # Convert Row to dict[str, Any], handling None for sums if necessary
                    return {
                        "active_positions": summary.active_positions or 0,
                        "closed_positions": summary.closed_positions or 0,
                        "total_realized_pnl": summary.total_realized_pnl or Decimal("0"),
                        "total_unrealized_pnl": summary.total_unrealized_pnl or Decimal("0"),
                    }
                # Should not happen with COUNT/SUM over a table unless it's empty and SUM returns NULL
                self.logger.warning("Position summary query returned no rows.", source_module=self._source_module)
                return {
                    "active_positions": 0,
                    "closed_positions": 0,
                    "total_realized_pnl": Decimal("0"),
                    "total_unrealized_pnl": Decimal("0"),
                }

        except Exception as e:
            self.logger.exception(
                f"Error retrieving position summary: {e}",
                source_module=self._source_module)
            raise