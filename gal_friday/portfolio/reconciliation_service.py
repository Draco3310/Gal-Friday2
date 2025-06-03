"""Portfolio reconciliation service for position and balance verification."""

import asyncio
import contextlib
import uuid
from collections.abc import Sequence  # Added Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.config_manager import ConfigManager
from gal_friday.dal.models.position import Position as PositionModel
from gal_friday.dal.repositories.order_repository import OrderRepository

# Import new repositories and models
from gal_friday.dal.repositories.position_repository import PositionRepository
from gal_friday.dal.repositories.reconciliation_repository import ReconciliationRepository
from gal_friday.execution_handler import ExecutionHandler  # Keep as is
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
                 config_manager: ConfigManager, # Renamed for clarity
                 portfolio_manager: PortfolioManager, # Stays as is, its internal methods will be refactored
                 execution_handler: ExecutionHandler, # Stays as is
                 alerting_system: AlertingSystem, # Renamed for clarity
                 logger_service: LoggerService, # Renamed for clarity
                 session_maker: async_sessionmaker[AsyncSession]) -> None:
        """Initialize the reconciliation service.

        Args:
            config_manager: Application configuration manager.
            portfolio_manager: Portfolio manager instance.
            execution_handler: Exchange execution handler.
            alerting_system: Alerting system for notifications.
            logger_service: Logger instance for logging.
            session_maker: SQLAlchemy async_sessionmaker for database sessions.
        """
        self.config = config_manager
        self.portfolio_manager = portfolio_manager
        self.execution_handler = execution_handler
        self.alerting = alerting_system
        self.logger = logger_service
        self._source_module = self.__class__.__name__

        # Instantiate repositories
        self.session_maker = session_maker
        self.position_repository = PositionRepository(session_maker, logger_service)
        self.order_repository = OrderRepository(session_maker, logger_service)
        self.reconciliation_repository = ReconciliationRepository(session_maker, logger_service)

        # Configuration
        self.reconciliation_interval = self.config.get_int(
            "reconciliation.interval_minutes",
            60,
        )
        self.auto_correct_threshold = Decimal(
            str(self.config.get_float("reconciliation.auto_correct_threshold", 0.01)),
        )
        self.critical_threshold = Decimal(
            str(self.config.get_float("reconciliation.critical_threshold", 0.10)),
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
            if self.config.get_bool("reconciliation.auto_correct", default=True):
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

            # 6. Save report and adjustments
            await self._save_reconciliation_event_and_adjustments(report)

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
            # Attempt to save the failed report
            await self._save_reconciliation_event_and_adjustments(report) # Adjustments might be empty
            await self._send_critical_alert(f"Reconciliation failed: {e!s}")

            return report

    async def _reconcile_positions(self, report: ReconciliationReport) -> None:
        """Reconcile positions with exchange."""
        try:
            # Get positions from both sources
            # Assuming portfolio_manager.get_all_db_positions() is refactored to return List[PositionModel]
            internal_positions_models: Sequence[PositionModel] = await self.portfolio_manager.get_all_db_positions() # type: ignore
            exchange_positions_data = await self.execution_handler.get_exchange_positions() # type: ignore

            report.positions_checked = len(internal_positions_models) + len(exchange_positions_data)

            # Create position maps
            internal_map = {pos.trading_pair: pos for pos in internal_positions_models}
            exchange_map = {pos_data["symbol"]: pos_data for pos_data in exchange_positions_data}

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
                        internal_value=internal_pos.quantity, # Access model attribute
                        severity="critical",
                    )
                    report.position_discrepancies.append(discrepancy)
                    report.manual_review_required.append({
                        "type": "position", "pair": pair,
                        "issue": "Position exists internally but not on exchange",
                        "internal_qty": str(internal_pos.quantity),
                    })

                elif exchange_pos and not internal_pos:
                    exchange_pos_data = exchange_pos
                    exchange_qty = Decimal(str(exchange_pos_data.get("quantity", 0)))
                    discrepancy = PositionDiscrepancy(
                        trading_pair=pair,
                        discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
                        exchange_value=exchange_qty,
                        severity="high",
                    )
                    report.position_discrepancies.append(discrepancy)

                    if self._can_auto_correct(exchange_qty): # Check if the exchange quantity itself is small enough
                        await self._add_missing_db_position(pair, exchange_pos_data, report)
                    else:
                        report.manual_review_required.append({
                            "type": "position", "pair": pair,
                            "issue": "Position exists on exchange but not tracked internally",
                            "exchange_qty": str(exchange_qty),
                        })

                elif internal_pos and exchange_pos:
                    internal_qty = internal_pos.quantity
                    exchange_pos_data = exchange_pos
                    exchange_qty = Decimal(str(exchange_pos_data.get("quantity", 0)))
                    qty_diff = abs(internal_qty - exchange_qty)

                    if qty_diff > Decimal("0.00000001"):  # Tolerance
                        severity = self._determine_severity(qty_diff, internal_qty)
                        discrepancy = PositionDiscrepancy(
                            trading_pair=pair, discrepancy_type=DiscrepancyType.QUANTITY_MISMATCH,
                            internal_value=internal_qty, exchange_value=exchange_qty,
                            difference=qty_diff, severity=severity,
                        )
                        report.position_discrepancies.append(discrepancy)

                        if self._can_auto_correct(qty_diff):
                            await self._adjust_db_position_quantity(internal_pos, exchange_qty, report)
                        else:
                            report.manual_review_required.append({
                                "type": "quantity", "pair": pair,
                                "internal": str(internal_qty), "exchange": str(exchange_qty),
                                "difference": str(qty_diff),
                            })
        except Exception as e:
            self.logger.exception("Error during position reconciliation", source_module=self._source_module)
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
            for ex_order_data in exchange_orders:
                exchange_order_id = ex_order_data["order_id"] # Assuming key is "order_id"

                # Check if order exists in our system using OrderRepository
                tracked_order_model = await self.order_repository.find_by_exchange_id(exchange_order_id)

                if not tracked_order_model:
                    report.untracked_orders.append(exchange_order_id)

                    # Determine if this affects positions
                    if ex_order_data.get("status") == "filled": # Use .get for safety
                        report.manual_review_required.append({
                            "type": "order", "order_id": exchange_order_id,
                            "pair": ex_order_data.get("pair", "UNKNOWN"),
                            "side": ex_order_data.get("side", "UNKNOWN"),
                            "quantity": str(ex_order_data.get("quantity", 0)),
                            "issue": "Filled order from exchange not tracked in internal system",
                        })
        except Exception as e:
            self.logger.exception("Error during order reconciliation", source_module=self._source_module)
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

                # Recording of adjustments will be handled by _save_reconciliation_event_and_adjustments
            except Exception as e:
                self.logger.error(
                    f"Failed to apply auto-correction: {e}",
                    source_module=self._source_module,
                    context={"correction": correction},
                )
                report.error_messages.append(
                    f"Auto-correction failed: {correction['type']} - {e!s}",
                )

    async def _add_missing_db_position(self, pair: str, exchange_pos_data: dict, report: ReconciliationReport) -> None:
        """Marks for auto-correction: Add position that exists on exchange but not internally.
        Actual DB write happens via portfolio_manager or directly if this service owns position creation logic.
        For now, this method prepares the 'correction' dict for the report.
        """
        # This method now just prepares the correction for the report.
        # The actual DB creation should be handled by PortfolioManager or a dedicated method
        # that uses self.position_repository.create()
        # For this refactor, we assume `portfolio_manager.create_new_position_from_exchange_data` exists
        # or this is noted for `_apply_auto_corrections`.

        correction_data = {
            "type": "add_position", "pair": pair,
            "quantity": str(exchange_pos_data.get("quantity", 0)), # Ensure string for JSON
            "entry_price": str(exchange_pos_data.get("average_price", 0)), # Ensure string
            "reason": "Position found on exchange, not in internal records. Auto-created.",
            # Fields needed for PositionAdjustmentModel
            "adjustment_type": DiscrepancyType.POSITION_MISSING_INTERNAL.value,
            "old_value": None,
            "new_value": Decimal(str(exchange_pos_data.get("quantity", 0))),
        }
        report.auto_corrections.append(correction_data)
        self.logger.info(f"Marked missing position for {pair} for auto-creation.", source_module=self._source_module)


    async def _adjust_db_position_quantity(
        self, internal_pos_model: PositionModel, exchange_qty: Decimal, report: ReconciliationReport,
    ) -> None:
        """Marks for auto-correction: Adjust internal position quantity to match exchange."""
        # This method now just prepares the correction for the report.
        # Actual DB update via portfolio_manager.adjust_position_quantity(...)

        correction_data = {
            "type": "position_quantity", "pair": internal_pos_model.trading_pair,
            "old_value": internal_pos_model.quantity, # Keep as Decimal for PositionAdjustmentModel
            "new_value": exchange_qty,               # Keep as Decimal
            "reason": "Quantity adjustment to match exchange data.",
            "adjustment_type": DiscrepancyType.QUANTITY_MISMATCH.value,
        }
        report.auto_corrections.append(correction_data)
        self.logger.info(f"Marked position {internal_pos_model.trading_pair} for quantity adjustment.", source_module=self._source_module)

    async def _adjust_balance( # This method's DB interaction is via PortfolioManager
        self, currency: str, internal_balance: Decimal, exchange_balance: Decimal, report: ReconciliationReport,
    ) -> None:
        """Marks for auto-correction: Adjust internal balance to match exchange."""
        # This method now just prepares the correction for the report.
        # Actual DB update via portfolio_manager.adjust_balance(...)
        correction_data = {
            "type": "balance_adjustment", "currency": currency, # 'type' changed for clarity
            "old_value": internal_balance, # Keep as Decimal
            "new_value": exchange_balance, # Keep as Decimal
            "reason": "Balance adjustment to match exchange data.",
            "adjustment_type": DiscrepancyType.BALANCE_MISMATCH.value, # Use enum value
            "trading_pair": currency, # For PositionAdjustmentModel, pair can be currency
        }
        report.auto_corrections.append(correction_data)
        self.logger.info(f"Marked balance for {currency} for adjustment.", source_module=self._source_module)

    async def _save_reconciliation_event_and_adjustments(self, report: ReconciliationReport) -> None:
        """Saves the main reconciliation event and all its adjustments to the database."""
        try:
            event_data = {
                "reconciliation_id": uuid.UUID(report.reconciliation_id),
                "timestamp": report.timestamp,
                "reconciliation_type": "full", # Example, could be more dynamic
                "status": report.status.value,
                "discrepancies_found": report.total_discrepancies,
                "auto_corrected": len(report.auto_corrections),
                "manual_review_required": len(report.manual_review_required),
                "report": report.to_dict(), # Full report as JSON
                "duration_seconds": Decimal(str(report.duration_seconds)) if report.duration_seconds is not None else None,
            }
            created_event = await self.reconciliation_repository.save_reconciliation_event(event_data)
            self.logger.info(f"Saved reconciliation event {created_event.reconciliation_id}", source_module=self._source_module)

            # Save all adjustments (auto-corrections and those needing manual review if they are stored)
            # For now, only saving auto_corrections as explicit adjustments.
            for adj_data in report.auto_corrections:
                # Ensure adj_data matches PositionAdjustmentModel fields
                adjustment_to_save = {
                    "reconciliation_id": created_event.reconciliation_id,
                    "trading_pair": adj_data.get("pair") or adj_data.get("currency", "UNKNOWN_PAIR"),
                    "adjustment_type": adj_data.get("adjustment_type", "UNKNOWN_ADJUSTMENT"),
                    "old_value": adj_data.get("old_value"), # Should be Decimal or None
                    "new_value": adj_data.get("new_value"), # Should be Decimal or None
                    "reason": adj_data.get("reason", ""),
                    # adjusted_at is defaulted by DB
                }
                # Convert numeric strings to Decimal if they came from report.to_dict()
                for key in ["old_value", "new_value"]:
                    if isinstance(adjustment_to_save[key], str):
                        adjustment_to_save[key] = Decimal(adjustment_to_save[key])

                await self.reconciliation_repository.save_position_adjustment(adjustment_to_save)
            self.logger.info(f"Saved {len(report.auto_corrections)} adjustments for event {created_event.reconciliation_id}", source_module=self._source_module)

        except Exception as e:
            self.logger.exception(f"Error saving reconciliation report/adjustments for event {report.reconciliation_id}: {e}", source_module=self._source_module)
            # Decide if this should re-raise or just log

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
        if self._last_reconciliation: # Fetch the ReconciliationEventModel
            # Assuming ReconciliationRepository has a method like get_latest_event()
            # This needs to be adapted if get_latest_report returns the Pydantic model.
            # For now, let's assume it can fetch the model or we adapt.
            # This part shows a slight mismatch if get_latest_report still returns Pydantic model.
            # Ideally, repo methods return SQLAlchemy models.
            # Get the most recent reconciliation event (limit to 1 day for efficiency)
            recent_events = await self.reconciliation_repository.get_recent_reconciliation_events(days=1, status=None)
            latest_event_model = recent_events[0] if recent_events else None
            if latest_event_model:
                # Convert model to dict for status, or use a Pydantic model constructed from it
                last_report_data = latest_event_model.report # The JSONB field
            else:
                last_report_data = None
        else:
            last_report_data = None

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
