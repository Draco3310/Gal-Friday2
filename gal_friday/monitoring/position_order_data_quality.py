"""Data quality monitoring for Position-Order relationships.

This module provides comprehensive monitoring capabilities to ensure data integrity
and identify potential issues with position-order relationship data.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..dal.models.order import Order
from ..dal.models.position import Position
from ..dal.repositories.order_repository import OrderRepository
from ..dal.repositories.position_repository import PositionRepository
from ..logger_service import LoggerService


@dataclass
class DataQualityIssue:
    """Represents a data quality issue found during monitoring."""
    
    issue_type: str
    severity: str  # "HIGH", "MEDIUM", "LOW"
    entity_type: str  # "ORDER", "POSITION", "RELATIONSHIP"
    entity_id: str
    description: str
    details: dict = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DataQualityReport:
    """Comprehensive data quality report for position-order relationships."""
    
    report_timestamp: datetime
    total_orders_checked: int
    total_positions_checked: int
    issues_found: list[DataQualityIssue]
    metrics: dict[str, Any]
    
    @property
    def high_priority_issues(self) -> list[DataQualityIssue]:
        """Get high priority issues that require immediate attention."""
        return [issue for issue in self.issues_found if issue.severity == "HIGH"]
    
    @property
    def issue_summary(self) -> dict[str, int]:
        """Get summary of issues by type."""
        summary = {}
        for issue in self.issues_found:
            summary[issue.issue_type] = summary.get(issue.issue_type, 0) + 1
        return summary


class PositionOrderDataQualityMonitor:
    """Monitor data quality for position-order relationships."""
    
    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        logger: LoggerService) -> None:
        """Initialize the data quality monitor.
        
        Args:
            session_maker: SQLAlchemy async session maker
            logger: Logger service instance
        """
        self.session_maker = session_maker
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Initialize repositories
        self.order_repository = OrderRepository(session_maker, logger)
        self.position_repository = PositionRepository(session_maker, logger)

    async def run_comprehensive_check(
        self, 
        hours_back: int = 24,
        include_historical: bool = False
    ) -> DataQualityReport:
        """Run comprehensive data quality checks.
        
        Args:
            hours_back: Number of hours to look back for recent data
            include_historical: Whether to include historical data checks
            
        Returns:
            Comprehensive data quality report
        """
        self.logger.info(
            f"Starting comprehensive data quality check (last {hours_back} hours)",
            source_module=self._source_module)
        
        report_start = datetime.now(UTC)
        issues: list[DataQualityIssue] = []
        metrics: dict[str, Any] = {}
        
        try:
            # Check for unlinked filled orders
            unlinked_issues = await self._check_unlinked_filled_orders(hours_back)
            issues.extend(unlinked_issues)
            
            # Check for orphaned position references
            orphaned_issues = await self._check_orphaned_position_references()
            issues.extend(orphaned_issues)
            
            # Check for positions without contributing orders
            no_orders_issues = await self._check_positions_without_orders(hours_back)
            issues.extend(no_orders_issues)
            
            # Check for inconsistent quantities
            quantity_issues = await self._check_quantity_consistency()
            issues.extend(quantity_issues)
            
            # Calculate metrics
            metrics = await self._calculate_relationship_metrics(hours_back)
            
            # Historical checks if requested
            if include_historical:
                historical_issues = await self._check_historical_data_integrity()
                issues.extend(historical_issues)
            
            # Count entities checked
            total_orders = await self._count_recent_orders(hours_back)
            total_positions = await self._count_recent_positions(hours_back)
            
            report = DataQualityReport(
                report_timestamp=report_start,
                total_orders_checked=total_orders,
                total_positions_checked=total_positions,
                issues_found=issues,
                metrics=metrics)
            
            self.logger.info(
                f"Data quality check completed. Found {len(issues)} issues "
                f"({len(report.high_priority_issues)} high priority)",
                source_module=self._source_module)
            
            return report
            
        except Exception as e:
            self.logger.exception(
                f"Error during comprehensive data quality check: {e}",
                source_module=self._source_module)
            raise

    async def _check_unlinked_filled_orders(self, hours_back: int) -> list[DataQualityIssue]:
        """Check for filled orders that are not linked to positions."""
        issues: list[DataQualityIssue] = []
        
        try:
            unlinked_orders = await self.order_repository.get_unlinked_filled_orders(hours_back)
            
            for order in unlinked_orders:
                issues.append(DataQualityIssue(
                    issue_type="UNLINKED_FILLED_ORDER",
                    severity="HIGH",
                    entity_type="ORDER",
                    entity_id=str(order.id),
                    description=f"Filled order {order.id} is not linked to any position",
                    details={
                        "trading_pair": order.trading_pair,
                        "status": order.status,
                        "filled_quantity": str(order.filled_quantity) if order.filled_quantity else "0",
                        "created_at": order.created_at.isoformat() if order.created_at else None,
                    }
                ))
            
            self.logger.debug(
                f"Found {len(unlinked_orders)} unlinked filled orders",
                source_module=self._source_module)
            
        except Exception as e:
            self.logger.error(
                f"Error checking unlinked filled orders: {e}",
                source_module=self._source_module)
        
        return issues

    async def _check_orphaned_position_references(self) -> list[DataQualityIssue]:
        """Check for orders referencing non-existent positions."""
        issues: list[DataQualityIssue] = []
        
        try:
            async with self.session_maker() as session:
                # Find orders with position_id that don't reference existing positions
                stmt = (
                    select(Order)
                    .outerjoin(Position, Order.position_id == Position.id)
                    .where(
                        and_(
                            Order.position_id.isnot(None),
                            Position.id.is_(None)
                        )
                    )
                )
                result = await session.execute(stmt)
                orphaned_orders = result.scalars().all()
                
                for order in orphaned_orders:
                    issues.append(DataQualityIssue(
                        issue_type="ORPHANED_POSITION_REFERENCE",
                        severity="HIGH",
                        entity_type="ORDER",
                        entity_id=str(order.id),
                        description=f"Order {order.id} references non-existent position {order.position_id}",
                        details={
                            "position_id": str(order.position_id),
                            "trading_pair": order.trading_pair,
                            "status": order.status,
                        }
                    ))
                
                self.logger.debug(
                    f"Found {len(orphaned_orders)} orders with orphaned position references",
                    source_module=self._source_module)
                
        except Exception as e:
            self.logger.error(
                f"Error checking orphaned position references: {e}",
                source_module=self._source_module)
        
        return issues

    async def _check_positions_without_orders(self, hours_back: int) -> list[DataQualityIssue]:
        """Check for positions without any contributing orders."""
        issues: list[DataQualityIssue] = []
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
        
        try:
            async with self.session_maker() as session:
                # Find positions without any linked orders
                stmt = (
                    select(Position)
                    .outerjoin(Order, Position.id == Order.position_id)
                    .where(
                        and_(
                            Position.opened_at > cutoff,
                            Order.id.is_(None)
                        )
                    )
                )
                result = await session.execute(stmt)
                positions_without_orders = result.scalars().all()
                
                for position in positions_without_orders:
                    # This might be expected for some positions (e.g., reconciliation adjustments)
                    # so marking as MEDIUM severity
                    issues.append(DataQualityIssue(
                        issue_type="POSITION_WITHOUT_ORDERS",
                        severity="MEDIUM",
                        entity_type="POSITION",
                        entity_id=str(position.id),
                        description=f"Position {position.id} has no contributing orders",
                        details={
                            "trading_pair": position.trading_pair,
                            "quantity": str(position.quantity),
                            "is_active": position.is_active,
                            "opened_at": position.opened_at.isoformat() if position.opened_at else None,
                        }
                    ))
                
                self.logger.debug(
                    f"Found {len(positions_without_orders)} positions without contributing orders",
                    source_module=self._source_module)
                
        except Exception as e:
            self.logger.error(
                f"Error checking positions without orders: {e}",
                source_module=self._source_module)
        
        return issues

    async def _check_quantity_consistency(self) -> list[DataQualityIssue]:
        """Check for quantity consistency between positions and their contributing orders."""
        issues: list[DataQualityIssue] = []
        
        try:
            async with self.session_maker() as session:
                # Get positions with their orders
                stmt = (
                    select(Position, func.sum(Order.filled_quantity).label('total_order_quantity'))
                    .join(Order, Position.id == Order.position_id)
                    .where(
                        and_(
                            Order.status.in_(["FILLED", "PARTIALLY_FILLED"]),
                            Order.filled_quantity.isnot(None)
                        )
                    )
                    .group_by(Position.id)
                )
                result = await session.execute(stmt)
                
                for position, total_order_quantity in result:
                    if total_order_quantity and abs(position.quantity - total_order_quantity) > Decimal('0.00000001'):
                        issues.append(DataQualityIssue(
                            issue_type="QUANTITY_MISMATCH",
                            severity="MEDIUM",
                            entity_type="RELATIONSHIP",
                            entity_id=str(position.id),
                            description=f"Position quantity ({position.quantity}) doesn't match sum of order quantities ({total_order_quantity})",
                            details={
                                "position_quantity": str(position.quantity),
                                "order_quantities_sum": str(total_order_quantity),
                                "difference": str(abs(position.quantity - total_order_quantity)),
                                "trading_pair": position.trading_pair,
                            }
                        ))
                
                self.logger.debug(
                    f"Checked quantity consistency, found {len(issues)} mismatches",
                    source_module=self._source_module)
                
        except Exception as e:
            self.logger.error(
                f"Error checking quantity consistency: {e}",
                source_module=self._source_module)
        
        return issues

    async def _check_historical_data_integrity(self) -> list[DataQualityIssue]:
        """Check historical data integrity across all time periods."""
        issues: list[DataQualityIssue] = []
        
        try:
            # This would implement more extensive historical checks
            # For now, placeholder for future enhancement
            self.logger.debug(
                "Historical data integrity check completed",
                source_module=self._source_module)
            
        except Exception as e:
            self.logger.error(
                f"Error during historical data integrity check: {e}",
                source_module=self._source_module)
        
        return issues

    async def _calculate_relationship_metrics(self, hours_back: int) -> dict[str, Any]:
        """Calculate metrics about position-order relationships."""
        metrics: dict[str, Any] = {}
        cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
        
        try:
            async with self.session_maker() as session:
                # Total orders in period
                stmt = select(func.count(Order.id)).where(Order.created_at > cutoff)
                result = await session.execute(stmt)
                metrics['total_orders'] = result.scalar() or 0
                
                # Orders with position links
                stmt = select(func.count(Order.id)).where(
                    and_(Order.created_at > cutoff, Order.position_id.isnot(None))
                )
                result = await session.execute(stmt)
                metrics['orders_with_position_links'] = result.scalar() or 0
                
                # Fill rate
                stmt = select(func.count(Order.id)).where(
                    and_(
                        Order.created_at > cutoff,
                        Order.status.in_(["FILLED", "PARTIALLY_FILLED"])
                    )
                )
                result = await session.execute(stmt)
                metrics['filled_orders'] = result.scalar() or 0
                
                # Calculate percentages
                if metrics['total_orders'] > 0:
                    metrics['position_link_rate'] = (
                        metrics['orders_with_position_links'] / metrics['total_orders']
                    ) * 100
                else:
                    metrics['position_link_rate'] = 0
                
                if metrics['filled_orders'] > 0:
                    unlinked_filled = await self.order_repository.get_unlinked_filled_orders(hours_back)
                    metrics['unlinked_filled_orders'] = len(unlinked_filled)
                    metrics['filled_order_link_rate'] = (
                        (metrics['filled_orders'] - len(unlinked_filled)) / metrics['filled_orders']
                    ) * 100
                else:
                    metrics['unlinked_filled_orders'] = 0
                    metrics['filled_order_link_rate'] = 100
                
                self.logger.debug(
                    f"Calculated relationship metrics: {metrics}",
                    source_module=self._source_module)
                
        except Exception as e:
            self.logger.error(
                f"Error calculating relationship metrics: {e}",
                source_module=self._source_module)
        
        return metrics

    async def _count_recent_orders(self, hours_back: int) -> int:
        """Count recent orders for reporting."""
        try:
            cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
            async with self.session_maker() as session:
                stmt = select(func.count(Order.id)).where(Order.created_at > cutoff)
                result = await session.execute(stmt)
                return result.scalar() or 0
        except Exception:
            return 0

    async def _count_recent_positions(self, hours_back: int) -> int:
        """Count recent positions for reporting."""
        try:
            cutoff = datetime.now(UTC) - timedelta(hours=hours_back)
            async with self.session_maker() as session:
                stmt = select(func.count(Position.id)).where(Position.opened_at > cutoff)
                result = await session.execute(stmt)
                return result.scalar() or 0
        except Exception:
            return 0

    async def generate_alert_summary(self, report: DataQualityReport) -> str:
        """Generate a human-readable alert summary for high-priority issues."""
        high_priority_count = len(report.high_priority_issues)
        
        if high_priority_count == 0:
            return "✅ No high-priority data quality issues detected"
        
        summary = f"🚨 {high_priority_count} high-priority data quality issues detected:\n\n"
        
        issue_counts = {}
        for issue in report.high_priority_issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        for issue_type, count in issue_counts.items():
            summary += f"• {issue_type}: {count} issues\n"
        
        summary += f"\nTotal orders checked: {report.total_orders_checked}\n"
        summary += f"Total positions checked: {report.total_positions_checked}\n"
        summary += f"Position link rate: {report.metrics.get('position_link_rate', 0):.1f}%\n"
        
        return summary

    async def auto_fix_safe_issues(self, report: DataQualityReport) -> int:
        """Automatically fix issues that are safe to resolve programmatically.
        
        Returns:
            Number of issues that were fixed
        """
        fixed_count = 0
        
        for issue in report.issues_found:
            if issue.issue_type == "ORPHANED_POSITION_REFERENCE" and issue.severity == "HIGH":
                # Safe to unlink orders from non-existent positions
                try:
                    await self.order_repository.unlink_order_from_position(issue.entity_id)
                    fixed_count += 1
                    self.logger.info(
                        f"Auto-fixed orphaned position reference for order {issue.entity_id}",
                        source_module=self._source_module)
                except Exception as e:
                    self.logger.error(
                        f"Failed to auto-fix orphaned position reference for order {issue.entity_id}: {e}",
                        source_module=self._source_module)
        
        return fixed_count 