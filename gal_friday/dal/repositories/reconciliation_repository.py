"""Repository for reconciliation data persistence using SQLAlchemy."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.position_adjustment import PositionAdjustment
from gal_friday.dal.models.reconciliation_event import ReconciliationEvent

# ReconciliationReport and ReconciliationStatus would now likely be service-layer
# or domain models, not directly handled by repo.


if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ReconciliationRepository(BaseRepository[ReconciliationEvent]):
    """Repository for ReconciliationEvent data persistence using SQLAlchemy."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService",
    ) -> None:
        """Initialize the reconciliation repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, ReconciliationEvent, logger)

    async def save_reconciliation_event(
        self, event_data: dict[str, Any],
    ) -> ReconciliationEvent:
        """Saves a reconciliation event.

        `event_data` should contain fields for ReconciliationEvent model.
        Example: reconciliation_id (UUID), timestamp (datetime), reconciliation_type (str),
                 status (str), discrepancies_found (int), auto_corrected (int),
                 manual_review_required (int), report (dict), duration_seconds (Decimal).
        """
        if "reconciliation_id" not in event_data:  # Ensure ID is present if not auto-gen by DB
            event_data["reconciliation_id"] = event_data.get("reconciliation_id", uuid.uuid4())
        if "timestamp" in event_data and isinstance(event_data["timestamp"], datetime):
            if event_data["timestamp"].tzinfo is None:
                 event_data["timestamp"] = event_data["timestamp"].replace(tzinfo=UTC)

        return await self.create(event_data)

    async def get_reconciliation_event(
        self, reconciliation_id: uuid.UUID,
    ) -> ReconciliationEvent | None:
        """Get a specific reconciliation event by its ID."""
        return await self.get_by_id(reconciliation_id)

    async def get_recent_reconciliation_events(
        self, days: int = 7, status: str | None = None,
    ) -> Sequence[ReconciliationEvent]:
        """Get reconciliation events from the last N days, optionally filtered by status."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        # F841: filters: dict[str, Any] = {} # Unused
        # Assuming ReconciliationEvent model has a 'timestamp' field
        # This requires a custom query as BaseRepository.find_all
        # doesn't support date range directly.
        # However, if we only filter by status, find_all could be used. For now, custom:

        async with self.session_maker() as session:
            stmt = select(ReconciliationEvent).where(ReconciliationEvent.timestamp > cutoff_date)
            if status:
                stmt = stmt.where(ReconciliationEvent.status == status)
            stmt = stmt.order_by(ReconciliationEvent.timestamp.desc())

            result = await session.execute(stmt)
            events = result.scalars().all()
            self.logger.debug(
                f"Found {len(events)} reconciliation events from last {days} days.",
                source_module=self._source_module
            )
            return events

    async def save_position_adjustment(
        self, adjustment_data: dict[str, Any],
    ) -> PositionAdjustment:
        """Saves a position adjustment.

        `adjustment_data` should contain fields for PositionAdjustment model.
        Example: reconciliation_id (UUID), trading_pair (str), adjustment_type (str),
                 old_value (Decimal), new_value (Decimal), reason (str).
        """
        # PositionAdjustment has an auto-generating adjustment_id by default in its model
        # Ensure reconciliation_id (FK) is provided
        if "reconciliation_id" not in adjustment_data:
            self.logger.error(
                "Cannot save PositionAdjustment without reconciliation_id.",
                source_module=self._source_module
            )
            raise ValueError("reconciliation_id is required to save a PositionAdjustment.")

        # Create PositionAdjustment instance
        # For direct save, we'd need a session here.
        # Assuming it's created and committed like self.create
        async with self.session_maker() as session:
            instance = PositionAdjustment(**adjustment_data)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            self.logger.debug(
                f"Saved new PositionAdjustment with ID {instance.adjustment_id}",
                source_module=self._source_module
            )
            return instance


    async def get_adjustments_for_event(
        self, reconciliation_id: uuid.UUID,
    ) -> Sequence[PositionAdjustment]:
        """Get all position adjustments for a specific reconciliation event."""
        # This could use BaseRepository[PositionAdjustment].find_all if we had one,
        # or a direct query as done here.
        async with self.session_maker() as session:
            stmt = (
                select(PositionAdjustment)
                .where(PositionAdjustment.reconciliation_id == reconciliation_id)
                .order_by(PositionAdjustment.adjusted_at.desc())  # Assuming 'adjusted_at'
            )
            result = await session.execute(stmt)
            adjustments = result.scalars().all()
            self.logger.debug(
                f"Found {len(adjustments)} adjustments for event {reconciliation_id}",
                source_module=self._source_module
            )
            return adjustments

    async def get_adjustment_history(
        self, trading_pair: str | None = None, days: int = 30,
    ) -> Sequence[PositionAdjustment]:
        """Get history of position adjustments, optionally filtered by trading_pair."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        async with self.session_maker() as session:
            stmt = select(PositionAdjustment).where(PositionAdjustment.adjusted_at > cutoff_date)
            if trading_pair:
                stmt = stmt.where(PositionAdjustment.trading_pair == trading_pair)
            stmt = stmt.order_by(PositionAdjustment.adjusted_at.desc())

            result = await session.execute(stmt)
            adjustments = result.scalars().all()
            self.logger.debug(
                f"Retrieved adjustment history for last {days} days.",
                source_module=self._source_module
            )
            return adjustments

# _parse_report is removed as the repository now deals with SQLAlchemy models directly.
# Transformation to domain/service layer objects (like ReconciliationReport)
# would happen in a service layer.
