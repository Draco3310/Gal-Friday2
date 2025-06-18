"""Repository for audit trail persistence."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.audit_entry import AuditEntry

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class AuditRepository(BaseRepository[AuditEntry]):
    """Repository for managing audit trail data."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService") -> None:
        """Initialize the audit repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, AuditEntry, logger)

    async def create_audit_entry(self, audit_data: dict[str, Any]) -> AuditEntry:
        """Create a new audit entry.
        
        Args:
            audit_data: Dictionary containing audit entry data
            
        Returns:
            Created AuditEntry instance
        """
        try:
            async with self.session_maker() as session:
                # Create new audit entry
                audit_entry = AuditEntry(
                    id=audit_data.get("id"),
                    timestamp=datetime.fromisoformat(audit_data.get("timestamp", datetime.now(UTC).isoformat())),
                    order_id=audit_data.get("order_id"),
                    symbol=audit_data.get("symbol"),
                    side=audit_data.get("side"),
                    status=audit_data.get("status"),
                    filled_quantity=audit_data.get("filled_quantity"),
                    average_price=audit_data.get("average_price"),
                    commission=audit_data.get("commission"),
                    realized_pnl=audit_data.get("realized_pnl"),
                    consecutive_losses=audit_data.get("consecutive_losses"),
                    risk_events=audit_data.get("risk_events", []),
                    risk_metrics_snapshot=audit_data.get("risk_metrics_snapshot", {}),
                    service=audit_data.get("service", "risk_manager"),
                    environment=audit_data.get("environment", "production"),
                    risk_manager_version=audit_data.get("risk_manager_version", "1.0.0"),
                    audit_schema_version=audit_data.get("audit_schema_version", "1.0"),
                    instance_id=audit_data.get("instance_id", "unknown"),
                )
                
                session.add(audit_entry)
                await session.commit()
                await session.refresh(audit_entry)
                
                return audit_entry
                
        except Exception as e:
            self.logger.exception(
                f"Error creating audit entry: {e}",
                source_module=self._source_module)
            raise

    async def get_audit_entries_by_order(self, order_id: str) -> Sequence[AuditEntry]:
        """Get all audit entries for a specific order.
        
        Args:
            order_id: The order ID to search for
            
        Returns:
            List of AuditEntry instances
        """
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(AuditEntry)
                    .where(AuditEntry.order_id == order_id)
                    .order_by(AuditEntry.timestamp.desc())
                )
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            self.logger.exception(
                f"Error fetching audit entries for order {order_id}: {e}",
                source_module=self._source_module)
            raise

    async def get_audit_entries_by_symbol(
        self, 
        symbol: str, 
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> Sequence[AuditEntry]:
        """Get audit entries for a specific symbol within a time range.
        
        Args:
            symbol: The trading symbol
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            
        Returns:
            List of AuditEntry instances
        """
        try:
            async with self.session_maker() as session:
                stmt = select(AuditEntry).where(AuditEntry.symbol == symbol)
                
                if start_time:
                    stmt = stmt.where(AuditEntry.timestamp >= start_time)
                if end_time:
                    stmt = stmt.where(AuditEntry.timestamp <= end_time)
                    
                stmt = stmt.order_by(AuditEntry.timestamp.desc())
                
                result = await session.execute(stmt)
                return result.scalars().all()
        except Exception as e:
            self.logger.exception(
                f"Error fetching audit entries for symbol {symbol}: {e}",
                source_module=self._source_module)
            raise

    async def get_risk_event_entries(
        self,
        risk_event_type: str | None = None,
        limit: int = 100
    ) -> Sequence[AuditEntry]:
        """Get audit entries that contain risk events.
        
        Args:
            risk_event_type: Specific risk event type to filter (optional)
            limit: Maximum number of entries to return
            
        Returns:
            List of AuditEntry instances with risk events
        """
        try:
            async with self.session_maker() as session:
                # Note: This query assumes risk_events is stored as JSONB in PostgreSQL
                # For other databases, the query might need adjustment
                stmt = (
                    select(AuditEntry)
                    .where(AuditEntry.risk_events != [])
                    .order_by(AuditEntry.timestamp.desc())
                    .limit(limit)
                )
                
                result = await session.execute(stmt)
                entries = result.scalars().all()
                
                # Filter by specific risk event type if provided
                if risk_event_type:
                    filtered_entries = []
                    for entry in entries:
                        if risk_event_type in entry.risk_events:
                            filtered_entries.append(entry)
                    return filtered_entries
                    
                return entries
                
        except Exception as e:
            self.logger.exception(
                f"Error fetching risk event entries: {e}",
                source_module=self._source_module)
            raise

    async def health_check(self) -> bool:
        """Check if the repository is healthy and can connect to database.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            async with self.session_maker() as session:
                # Simple query to test connection
                stmt = select(1)
                await session.execute(stmt)
                return True
        except Exception as e:
            self.logger.error(
                f"Audit repository health check failed: {e}",
                source_module=self._source_module)
            return False

    async def cleanup_old_entries(self, days_to_keep: int = 90) -> int:
        """Clean up audit entries older than specified days.
        
        Args:
            days_to_keep: Number of days of entries to keep
            
        Returns:
            Number of entries deleted
        """
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)
            
            async with self.session_maker() as session:
                stmt = (
                    select(AuditEntry)
                    .where(AuditEntry.timestamp < cutoff_date)
                )
                result = await session.execute(stmt)
                old_entries = result.scalars().all()
                
                count = len(old_entries)
                for entry in old_entries:
                    await session.delete(entry)
                    
                await session.commit()
                
                self.logger.info(
                    f"Deleted {count} audit entries older than {days_to_keep} days",
                    source_module=self._source_module)
                
                return count
                
        except Exception as e:
            self.logger.exception(
                f"Error cleaning up old audit entries: {e}",
                source_module=self._source_module)
            raise