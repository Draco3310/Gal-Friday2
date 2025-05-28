"""Portfolio reconciliation service for position and balance verification."""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from gal_friday.config_manager import ConfigManager
from gal_friday.dal.repositories.position_repository import PositionRepository
from gal_friday.dal.repositories.reconciliation_repository import ReconciliationRepository
from gal_friday.execution_handler import ExecutionHandler
from gal_friday.logger_service import LoggerService
from gal_friday.monitoring.alerting_system import Alert, AlertingSystem, AlertSeverity
from gal_friday.portfolio_manager import PortfolioManager


class DiscrepancyType(Enum):
    """Types of reconciliation discrepancies."""
    POSITION_MISSING_INTERNAL = "position_missing_internal"
    POSITION_MISSING_EXCHANGE = "position_missing_exchange"
    QUANTITY_MISMATCH = "quantity_mismatch"
    BALANCE_MISMATCH = "balance_mismatch"
    COST_BASIS_MISMATCH = "cost_basis_mismatch"
    ORDER_NOT_TRACKED = "order_not_tracked"


class ReconciliationStatus(Enum):
    """Reconciliation process status."""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    IN_PROGRESS = "in_progress"


@dataclass
class PositionDiscrepancy:
    """Details of a position discrepancy."""
    trading_pair: str
    discrepancy_type: DiscrepancyType
    internal_value: Any | None = None
    exchange_value: Any | None = None
    difference: Decimal | None = None
    severity: str = "medium"  # low, medium, high, critical

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trading_pair": self.trading_pair,
            "type": self.discrepancy_type.value,
            "internal": str(self.internal_value) if self.internal_value else None,
            "exchange": str(self.exchange_value) if self.exchange_value else None,
            "difference": str(self.difference) if self.difference else None,
            "severity": self.severity,
        }


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""
    reconciliation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: ReconciliationStatus = ReconciliationStatus.IN_PROGRESS

    # Position reconciliation
    positions_checked: int = 0
    position_discrepancies: list[PositionDiscrepancy] = field(default_factory=list)

    # Balance reconciliation
    balances_checked: int = 0
    balance_discrepancies: list[dict[str, Any]] = field(default_factory=list)

    # Order reconciliation
    orders_checked: int = 0
    untracked_orders: list[str] = field(default_factory=list)

    # Adjustments
    auto_corrections: list[dict[str, Any]] = field(default_factory=list)
    manual_review_required: list[dict[str, Any]] = field(default_factory=list)

    # Metrics
    duration_seconds: float | None = None
    error_messages: list[str] = field(default_factory=list)

    @property
    def total_discrepancies(self) -> int:
        """Total number of discrepancies found."""
        return (len(self.position_discrepancies) +
                len(self.balance_discrepancies) +
                len(self.untracked_orders))

    @property
    def requires_manual_review(self) -> bool:
        """Check if manual review is needed."""
        return len(self.manual_review_required) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "positions_checked": self.positions_checked,
            "position_discrepancies": [d.to_dict() for d in self.position_discrepancies],
            "balances_checked": self.balances_checked,
            "balance_discrepancies": self.balance_discrepancies,
            "orders_checked": self.orders_checked,
            "untracked_orders": self.untracked_orders,
            "auto_corrections": self.auto_corrections,
            "manual_review_required": self.manual_review_required,
            "duration_seconds": self.duration_seconds,
            "error_messages": self.error_messages,
            "total_discrepancies": self.total_discrepancies,
        }


class ReconciliationService:
    """Automated portfolio reconciliation with exchange."""

    # Constants for failure thresholds
    MAX_CONSECUTIVE_FAILURES = 3

    # Severity thresholds as percentage of total
    CRITICAL_THRESHOLD = 10  # 10% or higher
    HIGH_THRESHOLD = 5      # 5-10%
    MEDIUM_THRESHOLD = 1    # 1-5%

    # USD-based severity thresholds
    USD_CRITICAL = 1000
    USD_HIGH = 100
    USD_MEDIUM = 10

    def __init__(self,
                 config: ConfigManager,
                 portfolio_manager: PortfolioManager,
                 execution_handler: ExecutionHandler,
                 position_repo: PositionRepository,
                 reconciliation_repo: ReconciliationRepository,
                 alerting: AlertingSystem,
                 logger: LoggerService) -> None:
        """Initialize the reconciliation service.

        Args:
            config: Application configuration manager
            portfolio_manager: Portfolio manager instance
            execution_handler: Exchange execution handler
            position_repo: Position repository for data access
            reconciliation_repo: Reconciliation repository for storing reports
            alerting: Alerting system for notifications
            logger: Logger instance for logging
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.execution_handler = execution_handler
        self.position_repo = position_repo
        self.reconciliation_repo = reconciliation_repo
        self.alerting = alerting
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Configuration
        self.reconciliation_interval = config.get_int(
            "reconciliation.interval_minutes",
            60,
        )
        self.auto_correct_threshold = Decimal(
            str(config.get_float("reconciliation.auto_correct_threshold", 0.01)),
        )
        self.critical_threshold = Decimal(
            str(config.get_float("reconciliation.critical_threshold", 0.10)),
        )

        # State
        self._reconciliation_task: asyncio.Task | None = None
        self._last_reconciliation: datetime | None = None
        self._consecutive_failures = 0

    async def start(self) -> None:
        """Start reconciliation service."""
        self.logger.info(
            "Starting reconciliation service",
            source_module=self._source_module,
        )

        # Run initial reconciliation
        await self.run_reconciliation()

        # Start periodic reconciliation
        self._reconciliation_task = asyncio.create_task(self._periodic_reconciliation())

    async def stop(self) -> None:
        """Stop reconciliation service."""
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconciliation_task

    async def _periodic_reconciliation(self) -> None:
        """Run reconciliation periodically."""
        while True:
            try:
                await asyncio.sleep(self.reconciliation_interval * 60)
                await self.run_reconciliation()

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception(
                    "Error in periodic reconciliation",
                    source_module=self._source_module,
                )
                self._consecutive_failures += 1

                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    await self._send_critical_alert(
                        "Reconciliation service failing repeatedly",
                    )

    async def run_reconciliation(self) -> ReconciliationReport:
        """Run complete reconciliation process."""
        start_time = datetime.now(UTC)
        report = ReconciliationReport()

        try:
            self.logger.info(
                "Starting reconciliation process",
                source_module=self._source_module,
            )

            # 1. Reconcile positions
            await self._reconcile_positions(report)

            # 2. Reconcile balances
            await self._reconcile_balances(report)

            # 3. Reconcile recent orders
            await self._reconcile_orders(report)

            # 4. Apply auto-corrections if enabled
            if self.config.get_bool("reconciliation.auto_correct", True):
                await self._apply_auto_corrections(report)

            # 5. Determine final status
            if report.total_discrepancies == 0:
                report.status = ReconciliationStatus.SUCCESS
            elif report.error_messages:
                report.status = ReconciliationStatus.FAILED
            else:
                report.status = ReconciliationStatus.PARTIAL

            # Calculate duration
            report.duration_seconds = (datetime.now(UTC) - start_time).total_seconds()

            # 6. Save report
            await self.reconciliation_repo.save_report(report)

            # 7. Send alerts if needed
            await self._send_reconciliation_alerts(report)

            self._last_reconciliation = datetime.now(UTC)
            self._consecutive_failures = 0

            self.logger.info(
                f"Reconciliation completed: {report.status.value}",
                source_module=self._source_module,
                context={
                    "duration": report.duration_seconds,
                    "discrepancies": report.total_discrepancies,
                },
            )

            return report

        except Exception as e:
            report.status = ReconciliationStatus.FAILED
            report.error_messages.append(str(e))
            report.duration_seconds = (datetime.now(UTC) - start_time).total_seconds()

            self.logger.exception(
                "Reconciliation failed",
                source_module=self._source_module,
            )

            await self.reconciliation_repo.save_report(report)
            await self._send_critical_alert(f"Reconciliation failed: {e!s}")

            return report

    async def _reconcile_positions(self, report: ReconciliationReport) -> None:
        """Reconcile positions with exchange."""
        try:
            # Get positions from both sources
            internal_positions = await self.portfolio_manager.get_all_positions()  # type: ignore[attr-defined]
            exchange_positions = await self.execution_handler.get_exchange_positions()  # type: ignore[attr-defined]

            report.positions_checked = len(internal_positions) + len(exchange_positions)

            # Create position maps
            internal_map = {pos["trading_pair"]: pos for pos in internal_positions}
            exchange_map = {pos["symbol"]: pos for pos in exchange_positions}

            # Check all trading pairs
            all_pairs = set(internal_map.keys()) | set(exchange_map.keys())

            for pair in all_pairs:
                internal_pos = internal_map.get(pair)
                exchange_pos = exchange_map.get(pair)

                # Check for missing positions
                if internal_pos and not exchange_pos:
                    discrepancy = PositionDiscrepancy(
                        trading_pair=pair,
                        discrepancy_type=DiscrepancyType.POSITION_MISSING_EXCHANGE,
                        internal_value=internal_pos["quantity"],
                        severity="critical",
                    )
                    report.position_discrepancies.append(discrepancy)
                    report.manual_review_required.append({
                        "type": "position",
                        "pair": pair,
                        "issue": "Position exists internally but not on exchange",
                    })

                elif exchange_pos and not internal_pos:
                    discrepancy = PositionDiscrepancy(
                        trading_pair=pair,
                        discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
                        exchange_value=exchange_pos["quantity"],
                        severity="high",
                    )
                    report.position_discrepancies.append(discrepancy)

                    # Auto-correct: Add missing position
                    if self._can_auto_correct(Decimal(str(exchange_pos["quantity"]))):
                        await self._add_missing_position(pair, exchange_pos, report)
                    else:
                        report.manual_review_required.append({
                            "type": "position",
                            "pair": pair,
                            "issue": "Position exists on exchange but not tracked internally",
                        })

                elif internal_pos and exchange_pos:
                    # Check quantity match
                    internal_qty = Decimal(str(internal_pos["quantity"]))
                    exchange_qty = Decimal(str(exchange_pos["quantity"]))
                    qty_diff = abs(internal_qty - exchange_qty)

                    if qty_diff > Decimal("0.00000001"):  # Tolerance for rounding
                        severity = self._determine_severity(qty_diff, internal_qty)

                        discrepancy = PositionDiscrepancy(
                            trading_pair=pair,
                            discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                            internal_value=internal_qty,
                            exchange_value=exchange_qty,
                            difference=qty_diff,
                            severity=severity,
                        )
                        report.position_discrepancies.append(discrepancy)

                        # Auto-correct small differences
                        if self._can_auto_correct(qty_diff):
                            await self._adjust_position_quantity(
                                pair, internal_qty, exchange_qty, report,
                            )
                        else:
                            report.manual_review_required.append({
                                "type": "quantity",
                                "pair": pair,
                                "internal": str(internal_qty),
                                "exchange": str(exchange_qty),
                                "difference": str(qty_diff),
                            })

        except Exception as e:
            report.error_messages.append(f"Position reconciliation error: {e!s}")
            raise

    async def _reconcile_balances(self, report: ReconciliationReport) -> None:
        """Reconcile account balances."""
        try:
            # Get balances from both sources
            internal_balances = await self.portfolio_manager.get_balances()  # type: ignore[attr-defined]
            exchange_balances = await self.execution_handler.get_exchange_balances()  # type: ignore[attr-defined]

            report.balances_checked = len(exchange_balances)

            # Check each currency
            for currency, exchange_balance in exchange_balances.items():
                internal_balance = internal_balances.get(currency, Decimal("0"))

                balance_diff = abs(internal_balance - exchange_balance)

                if balance_diff > Decimal("0.00000001"):
                    discrepancy = {
                        "currency": currency,
                        "internal": str(internal_balance),
                        "exchange": str(exchange_balance),
                        "difference": str(balance_diff),
                        "severity": self._determine_balance_severity(
                            balance_diff, currency,
                        ),
                    }
                    report.balance_discrepancies.append(discrepancy)

                    # Auto-correct small differences
                    if self._can_auto_correct_balance(balance_diff, currency):
                        await self._adjust_balance(
                            currency, internal_balance, exchange_balance, report,
                        )
                    else:
                        report.manual_review_required.append({
                            "type": "balance",
                            "currency": currency,
                            "issue": f"Balance mismatch: {balance_diff}",
                        })

        except Exception as e:
            report.error_messages.append(f"Balance reconciliation error: {e!s}")
            raise

    async def _reconcile_orders(self, report: ReconciliationReport) -> None:
        """Reconcile recent orders."""
        try:
            # Get recent orders from exchange (last 24 hours)
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            exchange_orders = await self.execution_handler.get_recent_orders(cutoff)  # type: ignore[attr-defined]

            report.orders_checked = len(exchange_orders)

            # Check if all orders are tracked
            for order in exchange_orders:
                order_id = order["order_id"]

                # Check if order exists in our system
                tracked_order = await self.execution_handler.get_order_by_exchange_id(  # type: ignore[attr-defined]
                    order_id,
                )

                if not tracked_order:
                    report.untracked_orders.append(order_id)

                    # Determine if this affects positions
                    if order["status"] == "filled":
                        report.manual_review_required.append({
                            "type": "order",
                            "order_id": order_id,
                            "pair": order["pair"],
                            "side": order["side"],
                            "quantity": str(order["quantity"]),
                            "issue": "Filled order not tracked in system",
                        })

        except Exception as e:
            report.error_messages.append(f"Order reconciliation error: {e!s}")
            raise

    def _can_auto_correct(self, difference: Decimal) -> bool:
        """Check if difference is small enough for auto-correction."""
        return difference <= self.auto_correct_threshold

    def _can_auto_correct_balance(self, difference: Decimal, currency: str) -> bool:
        """Check if balance difference can be auto-corrected."""
        # More conservative for balance corrections
        threshold = self.auto_correct_threshold / 10
        return difference <= threshold

    def _determine_severity(self, difference: Decimal, total: Decimal) -> str:
        """Determine discrepancy severity.

        Args:
            difference: Absolute difference between values
            total: Total value for percentage calculation

        Returns:
            str: Severity level ("low", "medium", "high", or "critical")
        """
        if total == 0:
            return "critical"

        percentage = (difference / total) * 100

        if percentage >= self.CRITICAL_THRESHOLD:
            return "critical"
        if percentage >= self.HIGH_THRESHOLD:
            return "high"
        if percentage >= self.MEDIUM_THRESHOLD:
            return "medium"
        return "low"

    def _determine_balance_severity(self, difference: Decimal, currency: str) -> str:
        """Determine balance discrepancy severity.

        Args:
            difference: Absolute difference in balance
            currency: Currency code (e.g., 'USD')

        Returns:
            str: Severity level ("low", "medium", "high", or "critical")
        """
        if currency == "USD":
            if difference >= self.USD_CRITICAL:
                return "critical"
            if difference >= self.USD_HIGH:
                return "high"
            if difference >= self.USD_MEDIUM:
                return "medium"
            return "low"
        # For crypto, use percentage-based or medium by default
        return "medium"  # Would need current prices for accurate assessment

    async def _apply_auto_corrections(self, report: ReconciliationReport) -> None:
        """Apply automatic corrections for small discrepancies."""
        self.logger.info(
            f"Applying {len(report.auto_corrections)} auto-corrections",
            source_module=self._source_module,
        )

        for correction in report.auto_corrections:
            try:
                if correction["type"] == "position_quantity":
                    await self.portfolio_manager.adjust_position(  # type: ignore[attr-defined]
                        correction["pair"],
                        correction["new_quantity"],
                        reason="Reconciliation auto-correction",
                    )
                elif correction["type"] == "balance":
                    await self.portfolio_manager.adjust_balance(  # type: ignore[attr-defined]
                        correction["currency"],
                        correction["new_balance"],
                        reason="Reconciliation auto-correction",
                    )

                # Record adjustment in database
                await self.reconciliation_repo.save_adjustment(
                    report.reconciliation_id,
                    correction,
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to apply auto-correction: {e}",
                    source_module=self._source_module,
                    context={"correction": correction},
                )
                report.error_messages.append(
                    f"Auto-correction failed: {correction['type']} - {e!s}",
                )

    async def _add_missing_position(self,
                                   pair: str,
                                   exchange_pos: dict,
                                   report: ReconciliationReport) -> None:
        """Add position that exists on exchange but not internally."""
        correction = {
            "type": "add_position",
            "pair": pair,
            "quantity": exchange_pos["quantity"],
            "entry_price": exchange_pos.get("average_price", 0),
            "reason": "Position found on exchange during reconciliation",
        }

        report.auto_corrections.append(correction)

        # Will be applied in _apply_auto_corrections

    async def _adjust_position_quantity(self,
                                      pair: str,
                                      internal_qty: Decimal,
                                      exchange_qty: Decimal,
                                      report: ReconciliationReport) -> None:
        """Adjust position quantity to match exchange."""
        correction = {
            "type": "position_quantity",
            "pair": pair,
            "old_quantity": str(internal_qty),
            "new_quantity": str(exchange_qty),
            "difference": str(exchange_qty - internal_qty),
            "reason": "Quantity adjustment to match exchange",
        }

        report.auto_corrections.append(correction)

    async def _adjust_balance(self,
                            currency: str,
                            internal_balance: Decimal,
                            exchange_balance: Decimal,
                            report: ReconciliationReport) -> None:
        """Adjust balance to match exchange."""
        correction = {
            "type": "balance",
            "currency": currency,
            "old_balance": str(internal_balance),
            "new_balance": str(exchange_balance),
            "difference": str(exchange_balance - internal_balance),
            "reason": "Balance adjustment to match exchange",
        }

        report.auto_corrections.append(correction)

    async def _send_reconciliation_alerts(self, report: ReconciliationReport) -> None:
        """Send alerts based on reconciliation results."""
        if report.status == ReconciliationStatus.SUCCESS:
            # No alerts for successful reconciliation
            return

        # Determine alert severity
        has_critical_discrepancies = any(
            d.severity == "critical" for d in report.position_discrepancies
        )
        if report.status == ReconciliationStatus.FAILED or has_critical_discrepancies:
            severity = AlertSeverity.CRITICAL
        elif report.requires_manual_review:
            severity = AlertSeverity.ERROR
        else:
            severity = AlertSeverity.WARNING

        # Create alert
        alert = Alert(
            alert_id=f"recon_{report.reconciliation_id}",
            title=f"Reconciliation {report.status.value.upper()}",
            message=self._format_alert_message(report),
            severity=severity,
            source=self._source_module,
            tags={
                "type": "reconciliation",
                "discrepancies": report.total_discrepancies,
                "manual_review": report.requires_manual_review,
            },
        )

        await self.alerting.send_alert(alert)

    def _format_alert_message(self, report: ReconciliationReport) -> str:
        """Format reconciliation alert message."""
        lines = [
            f"Reconciliation completed with {report.total_discrepancies} discrepancies:",
            f"- Position discrepancies: {len(report.position_discrepancies)}",
            f"- Balance discrepancies: {len(report.balance_discrepancies)}",
            f"- Untracked orders: {len(report.untracked_orders)}",
            f"- Auto-corrections applied: {len(report.auto_corrections)}",
            f"- Manual review required: {len(report.manual_review_required)}",
        ]

        if report.error_messages:
            lines.append(f"\nErrors encountered: {len(report.error_messages)}")

        return "\n".join(lines)

    async def _send_critical_alert(self, message: str) -> None:
        """Send critical alert."""
        alert = Alert(
            alert_id=f"recon_critical_{uuid.uuid4()}",
            title="Reconciliation Critical Error",
            message=message,
            severity=AlertSeverity.CRITICAL,
            source=self._source_module,
            tags={"type": "reconciliation_error"},
        )

        await self.alerting.send_alert(alert)

    async def get_reconciliation_status(self) -> dict[str, Any]:
        """Get current reconciliation status."""
        last_report = None
        if self._last_reconciliation:
            last_report = await self.reconciliation_repo.get_latest_report()

        last_run = (
            self._last_reconciliation.isoformat()
            if self._last_reconciliation
            else None
        )
        return {
            "last_run": last_run,
            "next_run": (
                self._last_reconciliation + timedelta(minutes=self.reconciliation_interval)
            ).isoformat() if self._last_reconciliation else None,
            "consecutive_failures": self._consecutive_failures,
            "last_report": last_report.to_dict() if last_report else None,
        }
