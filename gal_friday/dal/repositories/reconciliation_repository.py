"""Repository for reconciliation data persistence."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import asyncpg

from gal_friday.dal.base import BaseRepository
from gal_friday.portfolio.reconciliation_service import ReconciliationReport, ReconciliationStatus

# Use TYPE_CHECKING to import LoggerService for type annotations only
if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class ReconciliationRepository(BaseRepository):
    """Repository for reconciliation events and reports."""

    def __init__(self, db_pool: asyncpg.Pool, logger: "LoggerService") -> None:
        """Initialize the reconciliation repository.

        Args:
            db_pool: Database connection pool
            logger: Logger service instance
        """
        super().__init__(db_pool, logger, "reconciliation_events")

    async def save_report(self, report: ReconciliationReport) -> None:
        """Save reconciliation report."""
        query = """
            INSERT INTO reconciliation_events (
                reconciliation_id, timestamp, reconciliation_type,
                status, discrepancies_found, auto_corrected,
                manual_review_required, report, duration_seconds
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """

        await self.db_pool.execute(
            query,
            report.reconciliation_id,
            report.timestamp,
            "full",  # vs "positions_only", "balances_only"
            report.status.value,
            report.total_discrepancies,
            len(report.auto_corrections),
            len(report.manual_review_required),
            json.dumps(report.to_dict()),
            report.duration_seconds,
        )

    async def save_adjustment(self, reconciliation_id: str, adjustment: dict[str, Any]) -> None:
        """Save individual adjustment record."""
        query = """
            INSERT INTO position_adjustments (
                reconciliation_id, trading_pair, adjustment_type,
                old_value, new_value, reason
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """

        await self.db_pool.execute(
            query,
            reconciliation_id,
            adjustment.get("pair") or adjustment.get("currency"),
            adjustment["type"],
            Decimal(adjustment.get("old_quantity") or adjustment.get("old_balance", "0")),
            Decimal(adjustment.get("new_quantity") or adjustment.get("new_balance", "0")),
            adjustment["reason"],
        )

    async def get_latest_report(self) -> ReconciliationReport | None:
        """Get the most recent reconciliation report."""
        query = """
            SELECT * FROM reconciliation_events
            ORDER BY timestamp DESC
            LIMIT 1
        """

        row = await self.db_pool.fetchrow(query)
        if row:
            return self._parse_report(row)
        return None

    async def get_reports_with_discrepancies(self,
                                           days: int = 7) -> list[ReconciliationReport]:
        """Get reports with discrepancies in the last N days."""
        query = """
            SELECT * FROM reconciliation_events
            WHERE timestamp > $1
              AND discrepancies_found > 0
            ORDER BY timestamp DESC
        """

        cutoff = datetime.now(UTC) - timedelta(days=days)
        rows = await self.db_pool.fetch(query, cutoff)

        return [self._parse_report(row) for row in rows]

    async def get_adjustment_history(self,
                                   trading_pair: str | None = None,
                                   days: int = 30) -> list[dict[str, Any]]:
        """Get history of position adjustments."""
        query = """
            SELECT pa.*, re.timestamp as reconciliation_time
            FROM position_adjustments pa
            JOIN reconciliation_events re ON pa.reconciliation_id = re.reconciliation_id
            WHERE re.timestamp > $1
        """

        params: list[Any] = [datetime.now(UTC) - timedelta(days=days)]

        if trading_pair:
            query += " AND pa.trading_pair = $2"
            params.append(trading_pair)

        query += " ORDER BY pa.adjusted_at DESC"

        rows = await self.db_pool.fetch(query, *params)

        return [dict(row) for row in rows]

    def _parse_report(self, row: asyncpg.Record) -> ReconciliationReport:
        """Parse database row into ReconciliationReport."""
        report_data = json.loads(row["report"])

        # Reconstruct ReconciliationReport from stored data
        # This is a simplified version - full implementation would
        # properly deserialize all nested objects
        report = ReconciliationReport(
            reconciliation_id=row["reconciliation_id"],
            timestamp=row["timestamp"],
            status=ReconciliationStatus(row["status"]),
        )

        # Populate from report_data
        report.positions_checked = report_data.get("positions_checked", 0)
        report.balances_checked = report_data.get("balances_checked", 0)
        report.orders_checked = report_data.get("orders_checked", 0)

        return report
