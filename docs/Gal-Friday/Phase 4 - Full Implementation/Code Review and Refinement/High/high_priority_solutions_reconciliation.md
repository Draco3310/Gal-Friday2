# High Priority Solution: Portfolio Reconciliation Implementation

## Overview
This document provides the complete implementation plan for portfolio reconciliation in the Gal-Friday system. Portfolio reconciliation ensures that internal position tracking matches the actual positions held at the exchange, preventing trading errors due to position mismatches.

## Current State Problems

1. **Single Source of Truth**
   - Portfolio Manager only tracks positions internally
   - No validation against exchange data
   - Manual position updates from execution reports
   - Risk of drift between internal and actual positions

2. **Missing Verification**
   - No periodic sync with exchange
   - No detection of discrepancies
   - No handling of missed execution reports
   - No recovery from system restarts

3. **Operational Risks**
   - Incorrect position sizes could violate risk limits
   - Wrong P&L calculations
   - Potential for accidental over-leveraging
   - No audit trail for position adjustments

## Solution Architecture

### 1. Reconciliation Service Core

#### 1.1 Reconciliation Service Implementation
```python
# gal_friday/portfolio/reconciliation_service.py
"""Portfolio reconciliation service for position and balance verification."""

import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.core.pubsub import PubSubManager
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.execution_handler import ExecutionHandler
from gal_friday.dal.repositories.position_repository import PositionRepository
from gal_friday.dal.repositories.reconciliation_repository import ReconciliationRepository
from gal_friday.monitoring.alerting_system import AlertingSystem, Alert, AlertSeverity


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
    internal_value: Optional[Any] = None
    exchange_value: Optional[Any] = None
    difference: Optional[Decimal] = None
    severity: str = "medium"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trading_pair": self.trading_pair,
            "type": self.discrepancy_type.value,
            "internal": str(self.internal_value) if self.internal_value else None,
            "exchange": str(self.exchange_value) if self.exchange_value else None,
            "difference": str(self.difference) if self.difference else None,
            "severity": self.severity
        }


@dataclass
class ReconciliationReport:
    """Complete reconciliation report."""
    reconciliation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: ReconciliationStatus = ReconciliationStatus.IN_PROGRESS
    
    # Position reconciliation
    positions_checked: int = 0
    position_discrepancies: List[PositionDiscrepancy] = field(default_factory=list)
    
    # Balance reconciliation
    balances_checked: int = 0
    balance_discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Order reconciliation
    orders_checked: int = 0
    untracked_orders: List[str] = field(default_factory=list)
    
    # Adjustments
    auto_corrections: List[Dict[str, Any]] = field(default_factory=list)
    manual_review_required: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metrics
    duration_seconds: Optional[float] = None
    error_messages: List[str] = field(default_factory=list)
    
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
        
    def to_dict(self) -> Dict[str, Any]:
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
            "total_discrepancies": self.total_discrepancies
        }


class ReconciliationService:
    """Automated portfolio reconciliation with exchange."""
    
    def __init__(self,
                 config: ConfigManager,
                 portfolio_manager: PortfolioManager,
                 execution_handler: ExecutionHandler,
                 position_repo: PositionRepository,
                 reconciliation_repo: ReconciliationRepository,
                 alerting: AlertingSystem,
                 logger: LoggerService):
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.execution_handler = execution_handler
        self.position_repo = position_repo
        self.reconciliation_repo = reconciliation_repo
        self.alerting = alerting
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Configuration
        self.reconciliation_interval = config.get_int("reconciliation.interval_minutes", 60)
        self.auto_correct_threshold = Decimal(str(config.get_float("reconciliation.auto_correct_threshold", 0.01)))
        self.critical_threshold = Decimal(str(config.get_float("reconciliation.critical_threshold", 0.10)))
        
        # State
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._last_reconciliation: Optional[datetime] = None
        self._consecutive_failures = 0
        
    async def start(self):
        """Start reconciliation service."""
        self.logger.info(
            "Starting reconciliation service",
            source_module=self._source_module
        )
        
        # Run initial reconciliation
        await self.run_reconciliation()
        
        # Start periodic reconciliation
        self._reconciliation_task = asyncio.create_task(self._periodic_reconciliation())
        
    async def stop(self):
        """Stop reconciliation service."""
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
                
    async def _periodic_reconciliation(self):
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
                    source_module=self._source_module
                )
                self._consecutive_failures += 1
                
                if self._consecutive_failures >= 3:
                    await self._send_critical_alert(
                        "Reconciliation service failing repeatedly"
                    )
                    
    async def run_reconciliation(self) -> ReconciliationReport:
        """Run complete reconciliation process."""
        start_time = datetime.now(UTC)
        report = ReconciliationReport()
        
        try:
            self.logger.info(
                "Starting reconciliation process",
                source_module=self._source_module
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
                    "discrepancies": report.total_discrepancies
                }
            )
            
            return report
            
        except Exception as e:
            report.status = ReconciliationStatus.FAILED
            report.error_messages.append(str(e))
            report.duration_seconds = (datetime.now(UTC) - start_time).total_seconds()
            
            self.logger.exception(
                "Reconciliation failed",
                source_module=self._source_module
            )
            
            await self.reconciliation_repo.save_report(report)
            await self._send_critical_alert(f"Reconciliation failed: {str(e)}")
            
            return report
            
    async def _reconcile_positions(self, report: ReconciliationReport):
        """Reconcile positions with exchange."""
        try:
            # Get positions from both sources
            internal_positions = await self.portfolio_manager.get_all_positions()
            exchange_positions = await self.execution_handler.get_exchange_positions()
            
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
                        severity="critical"
                    )
                    report.position_discrepancies.append(discrepancy)
                    report.manual_review_required.append({
                        "type": "position",
                        "pair": pair,
                        "issue": "Position exists internally but not on exchange"
                    })
                    
                elif exchange_pos and not internal_pos:
                    discrepancy = PositionDiscrepancy(
                        trading_pair=pair,
                        discrepancy_type=DiscrepancyType.POSITION_MISSING_INTERNAL,
                        exchange_value=exchange_pos["quantity"],
                        severity="high"
                    )
                    report.position_discrepancies.append(discrepancy)
                    
                    # Auto-correct: Add missing position
                    if self._can_auto_correct(Decimal(str(exchange_pos["quantity"]))):
                        await self._add_missing_position(pair, exchange_pos, report)
                    else:
                        report.manual_review_required.append({
                            "type": "position",
                            "pair": pair,
                            "issue": "Position exists on exchange but not tracked internally"
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
                            severity=severity
                        )
                        report.position_discrepancies.append(discrepancy)
                        
                        # Auto-correct small differences
                        if self._can_auto_correct(qty_diff):
                            await self._adjust_position_quantity(
                                pair, internal_qty, exchange_qty, report
                            )
                        else:
                            report.manual_review_required.append({
                                "type": "quantity",
                                "pair": pair,
                                "internal": str(internal_qty),
                                "exchange": str(exchange_qty),
                                "difference": str(qty_diff)
                            })
                            
        except Exception as e:
            report.error_messages.append(f"Position reconciliation error: {str(e)}")
            raise
            
    async def _reconcile_balances(self, report: ReconciliationReport):
        """Reconcile account balances."""
        try:
            # Get balances from both sources
            internal_balances = await self.portfolio_manager.get_balances()
            exchange_balances = await self.execution_handler.get_exchange_balances()
            
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
                            balance_diff, currency
                        )
                    }
                    report.balance_discrepancies.append(discrepancy)
                    
                    # Auto-correct small differences
                    if self._can_auto_correct_balance(balance_diff, currency):
                        await self._adjust_balance(
                            currency, internal_balance, exchange_balance, report
                        )
                    else:
                        report.manual_review_required.append({
                            "type": "balance",
                            "currency": currency,
                            "issue": f"Balance mismatch: {balance_diff}"
                        })
                        
        except Exception as e:
            report.error_messages.append(f"Balance reconciliation error: {str(e)}")
            raise
            
    async def _reconcile_orders(self, report: ReconciliationReport):
        """Reconcile recent orders."""
        try:
            # Get recent orders from exchange (last 24 hours)
            cutoff = datetime.now(UTC) - timedelta(hours=24)
            exchange_orders = await self.execution_handler.get_recent_orders(cutoff)
            
            report.orders_checked = len(exchange_orders)
            
            # Check if all orders are tracked
            for order in exchange_orders:
                order_id = order["order_id"]
                
                # Check if order exists in our system
                tracked_order = await self.execution_handler.get_order_by_exchange_id(
                    order_id
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
                            "issue": "Filled order not tracked in system"
                        })
                        
        except Exception as e:
            report.error_messages.append(f"Order reconciliation error: {str(e)}")
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
        """Determine discrepancy severity."""
        if total == 0:
            return "critical"
            
        percentage = (difference / total) * 100
        
        if percentage >= 10:
            return "critical"
        elif percentage >= 5:
            return "high"
        elif percentage >= 1:
            return "medium"
        else:
            return "low"
            
    def _determine_balance_severity(self, difference: Decimal, currency: str) -> str:
        """Determine balance discrepancy severity."""
        # USD-based thresholds
        if currency == "USD":
            if difference >= 1000:
                return "critical"
            elif difference >= 100:
                return "high"
            elif difference >= 10:
                return "medium"
            else:
                return "low"
        else:
            # For crypto, use percentage-based
            return "medium"  # Would need current prices for accurate assessment
            
    async def _apply_auto_corrections(self, report: ReconciliationReport):
        """Apply automatic corrections for small discrepancies."""
        self.logger.info(
            f"Applying {len(report.auto_corrections)} auto-corrections",
            source_module=self._source_module
        )
        
        for correction in report.auto_corrections:
            try:
                if correction["type"] == "position_quantity":
                    await self.portfolio_manager.adjust_position(
                        correction["pair"],
                        correction["new_quantity"],
                        reason="Reconciliation auto-correction"
                    )
                elif correction["type"] == "balance":
                    await self.portfolio_manager.adjust_balance(
                        correction["currency"],
                        correction["new_balance"],
                        reason="Reconciliation auto-correction"
                    )
                    
                # Record adjustment in database
                await self.reconciliation_repo.save_adjustment(
                    report.reconciliation_id,
                    correction
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to apply auto-correction: {e}",
                    source_module=self._source_module,
                    context={"correction": correction}
                )
                report.error_messages.append(
                    f"Auto-correction failed: {correction['type']} - {str(e)}"
                )
                
    async def _add_missing_position(self, 
                                   pair: str, 
                                   exchange_pos: Dict,
                                   report: ReconciliationReport):
        """Add position that exists on exchange but not internally."""
        correction = {
            "type": "add_position",
            "pair": pair,
            "quantity": exchange_pos["quantity"],
            "entry_price": exchange_pos.get("average_price", 0),
            "reason": "Position found on exchange during reconciliation"
        }
        
        report.auto_corrections.append(correction)
        
        # Will be applied in _apply_auto_corrections
        
    async def _adjust_position_quantity(self,
                                      pair: str,
                                      internal_qty: Decimal,
                                      exchange_qty: Decimal,
                                      report: ReconciliationReport):
        """Adjust position quantity to match exchange."""
        correction = {
            "type": "position_quantity",
            "pair": pair,
            "old_quantity": str(internal_qty),
            "new_quantity": str(exchange_qty),
            "difference": str(exchange_qty - internal_qty),
            "reason": "Quantity adjustment to match exchange"
        }
        
        report.auto_corrections.append(correction)
        
    async def _adjust_balance(self,
                            currency: str,
                            internal_balance: Decimal,
                            exchange_balance: Decimal,
                            report: ReconciliationReport):
        """Adjust balance to match exchange."""
        correction = {
            "type": "balance",
            "currency": currency,
            "old_balance": str(internal_balance),
            "new_balance": str(exchange_balance),
            "difference": str(exchange_balance - internal_balance),
            "reason": "Balance adjustment to match exchange"
        }
        
        report.auto_corrections.append(correction)
        
    async def _send_reconciliation_alerts(self, report: ReconciliationReport):
        """Send alerts based on reconciliation results."""
        if report.status == ReconciliationStatus.SUCCESS:
            # No alerts for successful reconciliation
            return
            
        # Determine alert severity
        if report.status == ReconciliationStatus.FAILED:
            severity = AlertSeverity.CRITICAL
        elif any(d.severity == "critical" for d in report.position_discrepancies):
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
                "manual_review": report.requires_manual_review
            }
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
            f"- Manual review required: {len(report.manual_review_required)}"
        ]
        
        if report.error_messages:
            lines.append(f"\nErrors encountered: {len(report.error_messages)}")
            
        return "\n".join(lines)
        
    async def _send_critical_alert(self, message: str):
        """Send critical alert."""
        alert = Alert(
            alert_id=f"recon_critical_{uuid.uuid4()}",
            title="Reconciliation Critical Error",
            message=message,
            severity=AlertSeverity.CRITICAL,
            source=self._source_module,
            tags={"type": "reconciliation_error"}
        )
        
        await self.alerting.send_alert(alert)
        
    async def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get current reconciliation status."""
        last_report = None
        if self._last_reconciliation:
            last_report = await self.reconciliation_repo.get_latest_report()
            
        return {
            "last_run": self._last_reconciliation.isoformat() if self._last_reconciliation else None,
            "next_run": (
                self._last_reconciliation + timedelta(minutes=self.reconciliation_interval)
            ).isoformat() if self._last_reconciliation else None,
            "consecutive_failures": self._consecutive_failures,
            "last_report": last_report.to_dict() if last_report else None
        }
```

### 2. Database Support for Reconciliation

#### 2.1 Reconciliation Repository
```python
# gal_friday/dal/repositories/reconciliation_repository.py
"""Repository for reconciliation data persistence."""

import asyncpg
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from gal_friday.dal.base import BaseRepository
from gal_friday.portfolio.reconciliation_service import ReconciliationReport


class ReconciliationRepository(BaseRepository):
    """Repository for reconciliation events and reports."""
    
    def __init__(self, db_pool: asyncpg.Pool, logger):
        super().__init__(db_pool, logger, "reconciliation_events")
        
    async def save_report(self, report: ReconciliationReport):
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
            report.duration_seconds
        )
        
    async def save_adjustment(self, reconciliation_id: str, adjustment: Dict[str, Any]):
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
            adjustment["reason"]
        )
        
    async def get_latest_report(self) -> Optional[ReconciliationReport]:
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
                                           days: int = 7) -> List[ReconciliationReport]:
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
                                   trading_pair: Optional[str] = None,
                                   days: int = 30) -> List[Dict[str, Any]]:
        """Get history of position adjustments."""
        query = """
            SELECT pa.*, re.timestamp as reconciliation_time
            FROM position_adjustments pa
            JOIN reconciliation_events re ON pa.reconciliation_id = re.reconciliation_id
            WHERE re.timestamp > $1
        """
        
        params = [datetime.now(UTC) - timedelta(days=days)]
        
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
            status=ReconciliationStatus(row["status"])
        )
        
        # Populate from report_data
        report.positions_checked = report_data.get("positions_checked", 0)
        report.balances_checked = report_data.get("balances_checked", 0)
        report.orders_checked = report_data.get("orders_checked", 0)
        
        return report
```

### 3. Enhanced Execution Handler Integration

#### 3.1 Exchange Position and Balance Queries
```python
# Updates to gal_friday/execution_handler.py

class ExecutionHandler:
    """Enhanced execution handler with reconciliation support."""
    
    async def get_exchange_positions(self) -> List[Dict[str, Any]]:
        """Get current positions from exchange."""
        try:
            # Query Kraken for open positions
            response = await self._make_private_request(
                "/0/private/OpenPositions",
                {}
            )
            
            if response.get("error"):
                raise Exception(f"Failed to get positions: {response['error']}")
                
            positions = []
            for pos_id, pos_data in response.get("result", {}).items():
                positions.append({
                    "position_id": pos_id,
                    "symbol": self._normalize_pair(pos_data["pair"]),
                    "quantity": Decimal(pos_data["vol"]),
                    "side": pos_data["type"],
                    "average_price": Decimal(pos_data["cost"]) / Decimal(pos_data["vol"]),
                    "unrealized_pnl": Decimal(pos_data.get("net", "0")),
                    "margin_used": Decimal(pos_data.get("margin", "0"))
                })
                
            return positions
            
        except Exception:
            self.logger.exception(
                "Error fetching exchange positions",
                source_module=self._source_module
            )
            raise
            
    async def get_exchange_balances(self) -> Dict[str, Decimal]:
        """Get current balances from exchange."""
        try:
            response = await self._make_private_request(
                "/0/private/Balance",
                {}
            )
            
            if response.get("error"):
                raise Exception(f"Failed to get balances: {response['error']}")
                
            balances = {}
            for asset, balance in response.get("result", {}).items():
                # Normalize asset names
                normalized_asset = self._normalize_asset(asset)
                balances[normalized_asset] = Decimal(balance)
                
            return balances
            
        except Exception:
            self.logger.exception(
                "Error fetching exchange balances",
                source_module=self._source_module
            )
            raise
            
    async def get_recent_orders(self, since: datetime) -> List[Dict[str, Any]]:
        """Get recent orders from exchange."""
        try:
            # Convert datetime to unix timestamp
            since_timestamp = int(since.timestamp())
            
            response = await self._make_private_request(
                "/0/private/ClosedOrders",
                {"start": since_timestamp}
            )
            
            if response.get("error"):
                raise Exception(f"Failed to get orders: {response['error']}")
                
            orders = []
            for order_id, order_data in response.get("result", {}).get("closed", {}).items():
                orders.append({
                    "order_id": order_id,
                    "pair": self._normalize_pair(order_data["descr"]["pair"]),
                    "side": order_data["descr"]["type"],
                    "quantity": Decimal(order_data["vol"]),
                    "executed_quantity": Decimal(order_data["vol_exec"]),
                    "price": Decimal(order_data.get("price", "0")),
                    "status": order_data["status"],
                    "created_at": datetime.fromtimestamp(order_data["opentm"], UTC),
                    "closed_at": datetime.fromtimestamp(order_data["closetm"], UTC) 
                        if order_data.get("closetm") else None
                })
                
            return orders
            
        except Exception:
            self.logger.exception(
                "Error fetching recent orders",
                source_module=self._source_module
            )
            raise
            
    async def get_order_by_exchange_id(self, exchange_order_id: str) -> Optional[Dict]:
        """Check if we're tracking an order by exchange ID."""
        # Query from order repository
        return await self.order_repo.find_by_exchange_id(exchange_order_id)
        
    def _normalize_pair(self, kraken_pair: str) -> str:
        """Normalize Kraken pair format to standard format."""
        # Kraken uses formats like "XXRPZUSD" or "XRPUSD"
        # Normalize to "XRP/USD"
        replacements = {
            "XXRP": "XRP",
            "XDGE": "DOGE",
            "ZUSD": "USD",
            "XXBT": "BTC"
        }
        
        pair = kraken_pair
        for old, new in replacements.items():
            pair = pair.replace(old, new)
            
        # Add slash if not present
        if "/" not in pair:
            # Find currency boundary (assumes 3-4 char currencies)
            for i in range(3, 5):
                if pair[i:] in ["USD", "EUR", "BTC", "ETH"]:
                    pair = f"{pair[:i]}/{pair[i:]}"
                    break
                    
        return pair
        
    def _normalize_asset(self, kraken_asset: str) -> str:
        """Normalize Kraken asset codes."""
        asset_map = {
            "ZUSD": "USD",
            "XXRP": "XRP",
            "XDGE": "DOGE",
            "XXBT": "BTC",
            "XETH": "ETH"
        }
        
        return asset_map.get(kraken_asset, kraken_asset)
```

### 4. Reconciliation Dashboard Integration

#### 4.1 Dashboard API Endpoints
```python
# Add to gal_friday/monitoring/dashboard_backend.py

@app.get("/api/reconciliation/status")
async def get_reconciliation_status(recon_service: ReconciliationService = Depends(get_recon_service)):
    """Get current reconciliation status."""
    return await recon_service.get_reconciliation_status()


@app.get("/api/reconciliation/reports")
async def get_reconciliation_reports(
    days: int = 7,
    only_discrepancies: bool = False,
    recon_repo: ReconciliationRepository = Depends(get_recon_repo)
):
    """Get recent reconciliation reports."""
    if only_discrepancies:
        reports = await recon_repo.get_reports_with_discrepancies(days)
    else:
        reports = await recon_repo.get_recent_reports(days)
        
    return {
        "reports": [r.to_dict() for r in reports],
        "count": len(reports)
    }


@app.get("/api/reconciliation/adjustments")
async def get_adjustment_history(
    trading_pair: Optional[str] = None,
    days: int = 30,
    recon_repo: ReconciliationRepository = Depends(get_recon_repo)
):
    """Get history of reconciliation adjustments."""
    adjustments = await recon_repo.get_adjustment_history(trading_pair, days)
    
    return {
        "adjustments": adjustments,
        "count": len(adjustments)
    }


@app.post("/api/reconciliation/run")
async def trigger_reconciliation(
    recon_service: ReconciliationService = Depends(get_recon_service)
):
    """Manually trigger reconciliation."""
    report = await recon_service.run_reconciliation()
    
    return {
        "status": "completed",
        "report": report.to_dict()
    }


@app.post("/api/reconciliation/approve-adjustment")
async def approve_manual_adjustment(
    adjustment_id: str,
    approved: bool,
    notes: Optional[str] = None,
    recon_service: ReconciliationService = Depends(get_recon_service)
):
    """Approve or reject a manual adjustment."""
    # This would be implemented to handle manual review items
    return {
        "status": "processed",
        "adjustment_id": adjustment_id,
        "approved": approved
    }
```

### 5. Testing Framework

#### 5.1 Reconciliation Tests
```python
# tests/test_reconciliation.py
"""Tests for portfolio reconciliation."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, UTC
import uuid

from gal_friday.portfolio.reconciliation_service import (
    ReconciliationService,
    ReconciliationReport,
    PositionDiscrepancy,
    DiscrepancyType
)


class TestReconciliationService:
    """Test reconciliation service functionality."""
    
    @pytest.mark.asyncio
    async def test_position_reconciliation_exact_match(self, reconciliation_service):
        """Test reconciliation when positions match exactly."""
        # Set up matching positions
        internal_positions = [
            {"trading_pair": "XRP/USD", "quantity": Decimal("1000")}
        ]
        exchange_positions = [
            {"symbol": "XRP/USD", "quantity": Decimal("1000")}
        ]
        
        # Mock the data sources
        reconciliation_service.portfolio_manager.get_all_positions = AsyncMock(
            return_value=internal_positions
        )
        reconciliation_service.execution_handler.get_exchange_positions = AsyncMock(
            return_value=exchange_positions
        )
        
        # Run reconciliation
        report = await reconciliation_service.run_reconciliation()
        
        # Verify no discrepancies
        assert report.status == ReconciliationStatus.SUCCESS
        assert len(report.position_discrepancies) == 0
        assert report.total_discrepancies == 0
        
    @pytest.mark.asyncio
    async def test_position_quantity_mismatch(self, reconciliation_service):
        """Test detection of position quantity mismatch."""
        # Set up mismatched positions
        internal_positions = [
            {"trading_pair": "XRP/USD", "quantity": Decimal("1000")}
        ]
        exchange_positions = [
            {"symbol": "XRP/USD", "quantity": Decimal("950")}
        ]
        
        # Mock the data sources
        reconciliation_service.portfolio_manager.get_all_positions = AsyncMock(
            return_value=internal_positions
        )
        reconciliation_service.execution_handler.get_exchange_positions = AsyncMock(
            return_value=exchange_positions
        )
        
        # Run reconciliation
        report = await reconciliation_service.run_reconciliation()
        
        # Verify discrepancy detected
        assert len(report.position_discrepancies) == 1
        
        discrepancy = report.position_discrepancies[0]
        assert discrepancy.discrepancy_type == DiscrepancyType.QUANTITY_MISMATCH
        assert discrepancy.trading_pair == "XRP/USD"
        assert discrepancy.internal_value == Decimal("1000")
        assert discrepancy.exchange_value == Decimal("950")
        assert discrepancy.difference == Decimal("50")
        
    @pytest.mark.asyncio
    async def test_missing_position_detection(self, reconciliation_service):
        """Test detection of missing positions."""
        # Position exists internally but not on exchange
        internal_positions = [
            {"trading_pair": "XRP/USD", "quantity": Decimal("1000")}
        ]
        exchange_positions = []
        
        # Mock the data sources
        reconciliation_service.portfolio_manager.get_all_positions = AsyncMock(
            return_value=internal_positions
        )
        reconciliation_service.execution_handler.get_exchange_positions = AsyncMock(
            return_value=exchange_positions
        )
        
        # Run reconciliation
        report = await reconciliation_service.run_reconciliation()
        
        # Verify critical discrepancy
        assert len(report.position_discrepancies) == 1
        
        discrepancy = report.position_discrepancies[0]
        assert discrepancy.discrepancy_type == DiscrepancyType.POSITION_MISSING_EXCHANGE
        assert discrepancy.severity == "critical"
        assert len(report.manual_review_required) == 1
        
    @pytest.mark.asyncio
    async def test_auto_correction(self, reconciliation_service):
        """Test automatic correction of small discrepancies."""
        # Small quantity mismatch within auto-correct threshold
        internal_positions = [
            {"trading_pair": "XRP/USD", "quantity": Decimal("1000.005")}
        ]
        exchange_positions = [
            {"symbol": "XRP/USD", "quantity": Decimal("1000.000")}
        ]
        
        # Mock the data sources
        reconciliation_service.portfolio_manager.get_all_positions = AsyncMock(
            return_value=internal_positions
        )
        reconciliation_service.execution_handler.get_exchange_positions = AsyncMock(
            return_value=exchange_positions
        )
        reconciliation_service.portfolio_manager.adjust_position = AsyncMock()
        
        # Set auto-correct threshold
        reconciliation_service.auto_correct_threshold = Decimal("0.01")
        
        # Run reconciliation
        report = await reconciliation_service.run_reconciliation()
        
        # Verify auto-correction applied
        assert len(report.auto_corrections) == 1
        
        correction = report.auto_corrections[0]
        assert correction["type"] == "position_quantity"
        assert correction["pair"] == "XRP/USD"
        assert Decimal(correction["new_quantity"]) == Decimal("1000.000")
        
        # Verify adjustment was called
        reconciliation_service.portfolio_manager.adjust_position.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_balance_reconciliation(self, reconciliation_service):
        """Test balance reconciliation."""
        # Set up balance mismatch
        internal_balances = {
            "USD": Decimal("10000.00"),
            "XRP": Decimal("5000.00")
        }
        exchange_balances = {
            "USD": Decimal("9999.50"),
            "XRP": Decimal("5000.00")
        }
        
        # Mock the data sources
        reconciliation_service.portfolio_manager.get_balances = AsyncMock(
            return_value=internal_balances
        )
        reconciliation_service.execution_handler.get_exchange_balances = AsyncMock(
            return_value=exchange_balances
        )
        
        # Run reconciliation
        report = await reconciliation_service.run_reconciliation()
        
        # Verify balance discrepancy detected
        assert len(report.balance_discrepancies) == 1
        
        discrepancy = report.balance_discrepancies[0]
        assert discrepancy["currency"] == "USD"
        assert Decimal(discrepancy["difference"]) == Decimal("0.50")
        
    @pytest.mark.asyncio
    async def test_untracked_order_detection(self, reconciliation_service):
        """Test detection of untracked orders."""
        # Order exists on exchange but not in our system
        exchange_orders = [{
            "order_id": "UNTRACKED-123",
            "pair": "XRP/USD",
            "side": "buy",
            "quantity": Decimal("100"),
            "status": "filled"
        }]
        
        # Mock the data sources
        reconciliation_service.execution_handler.get_recent_orders = AsyncMock(
            return_value=exchange_orders
        )
        reconciliation_service.execution_handler.get_order_by_exchange_id = AsyncMock(
            return_value=None  # Not found
        )
        
        # Run reconciliation
        report = await reconciliation_service.run_reconciliation()
        
        # Verify untracked order detected
        assert len(report.untracked_orders) == 1
        assert report.untracked_orders[0] == "UNTRACKED-123"
        assert len(report.manual_review_required) == 1
```

## Implementation Steps

### Phase 1: Core Reconciliation Service (3 days)
1. Implement ReconciliationService class
2. Create discrepancy detection logic
3. Build reconciliation report structure
4. Add database persistence

### Phase 2: Exchange Integration (2 days)
1. Enhance ExecutionHandler with position/balance queries
2. Add order history retrieval
3. Implement data normalization
4. Handle exchange-specific formats

### Phase 3: Auto-Correction Logic (2 days)
1. Define auto-correction thresholds
2. Implement position adjustments
3. Add balance adjustments
4. Create audit trail

### Phase 4: Dashboard and Alerts (2 days)
1. Add reconciliation API endpoints
2. Create reconciliation UI components
3. Implement alert rules
4. Add manual review interface

### Phase 5: Testing and Validation (1 day)
1. Unit tests for all components
2. Integration tests with mock exchange
3. Performance testing
4. Manual validation procedures

## Success Criteria

1. **Accuracy**: 100% detection of position/balance discrepancies
2. **Automation**: 95% of small discrepancies auto-corrected
3. **Frequency**: Hourly reconciliation without performance impact
4. **Audit Trail**: Complete history of all adjustments
5. **Recovery Time**: < 5 minutes to detect and correct discrepancies

## Monitoring and Maintenance

1. **Reconciliation Metrics**:
   - Success rate
   - Discrepancy frequency by type
   - Auto-correction success rate
   - Manual review queue size

2. **Performance Metrics**:
   - Reconciliation duration
   - API call count
   - Database query time

3. **Business Metrics**:
   - Position accuracy rate
   - Balance accuracy rate
   - Time to resolution

4. **Alerts**:
   - Failed reconciliation
   - Critical discrepancies
   - Manual review backlog
   - Repeated failures 