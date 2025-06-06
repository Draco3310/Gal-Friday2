"""Portfolio reconciliation service for position and balance verification."""

import asyncio
import contextlib
import uuid
from abc import ABC, abstractmethod
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


class ReconciliationType(str, Enum):
    """Supported reconciliation types for dynamic strategy selection."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DELTA = "delta"
    POSITION_ONLY = "position_only"
    BALANCE_ONLY = "balance_only"
    TRADE_ONLY = "trade_only"
    REAL_TIME = "real_time"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"


@dataclass
class ReconciliationConfig:
    """Configuration for reconciliation strategies."""
    reconciliation_type: ReconciliationType
    max_discrepancy_threshold: float = 0.01
    auto_resolve_threshold: float = 0.001
    include_pending_trades: bool = True
    historical_lookback_hours: int = 24
    enable_alerts: bool = True
    batch_size: int = 1000
    timeout_seconds: int = 300
    retry_attempts: int = 3
    # Additional strategy-specific configurations
    real_time_cutoff_minutes: int = 15
    incremental_cutoff_hours: int = 1
    emergency_alert_threshold: float = 0.10


@dataclass
class ReconciliationResult:
    """Enhanced result of reconciliation process with strategy-specific details."""
    reconciliation_id: str
    reconciliation_type: ReconciliationType
    status: str  # 'completed', 'failed', 'partial'
    start_time: datetime
    end_time: datetime
    total_records_processed: int
    discrepancies_found: int
    auto_resolved_count: int
    manual_resolution_required: int
    summary: dict[str, Any]
    errors: list[str]
    strategy_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage and reporting."""
        return {
            "reconciliation_id": self.reconciliation_id,
            "reconciliation_type": self.reconciliation_type.value,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "total_records_processed": self.total_records_processed,
            "discrepancies_found": self.discrepancies_found,
            "auto_resolved_count": self.auto_resolved_count,
            "manual_resolution_required": self.manual_resolution_required,
            "summary": self.summary,
            "errors": self.errors,
            "strategy_metrics": self.strategy_metrics,
        }


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
    reconciliation_type: ReconciliationType = ReconciliationType.FULL

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
            "reconciliation_type": self.reconciliation_type.value,
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


class BaseReconciliationStrategy(ABC):
    """Abstract base class for reconciliation strategies."""
    
    def __init__(self, 
                 config: ReconciliationConfig, 
                 reconciliation_service: 'ReconciliationService') -> None:
        """Initialize the strategy with configuration and service dependencies.
        
        Args:
            config: ReconciliationConfiguration for this strategy
            reconciliation_service: ReconciliationService instance for accessing shared methods
        """
        self.config = config
        self.service = reconciliation_service
        self.logger = reconciliation_service.logger
        
    @abstractmethod
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute the reconciliation strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the strategy name."""
        pass
    
    async def _validate_prerequisites(self) -> bool:
        """Validate that prerequisites for reconciliation are met."""
        try:
            # Check if the execution handler is available
            if not hasattr(self.service.execution_handler, 'get_exchange_positions'):
                self.logger.error("Exchange service methods not available", 
                                source_module=self.service._source_module)
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Prerequisites validation failed: {e}", 
                            source_module=self.service._source_module)
            return False


class FullReconciliationStrategy(BaseReconciliationStrategy):
    """Complete reconciliation of all positions, balances, and trades."""
    
    def get_strategy_name(self) -> str:
        return "Full Reconciliation"
    
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute full reconciliation using existing service methods."""
        
        reconciliation_id = f"full_{int(datetime.now().timestamp())}"
        start_time = datetime.now(UTC)
        
        try:
            self.logger.info(f"Starting full reconciliation {reconciliation_id}", 
                           source_module=self.service._source_module)
            
            # Validate prerequisites
            if not await self._validate_prerequisites():
                raise ValueError("Prerequisites not met for full reconciliation")
            
            # Use existing service method to create and populate report
            report = ReconciliationReport(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.FULL
            )
            
            # Execute all reconciliation steps using existing service methods
            await self.service._reconcile_positions(report)
            await self.service._reconcile_balances(report)
            await self.service._reconcile_orders(report)
            
            # Apply auto-corrections if enabled
            if self.config.auto_resolve_threshold > 0:
                await self.service._apply_auto_corrections(report)
            
            # Create result with strategy-specific metrics
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.FULL,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(UTC),
                total_records_processed=report.positions_checked + report.balances_checked + report.orders_checked,
                discrepancies_found=report.total_discrepancies,
                auto_resolved_count=len(report.auto_corrections),
                manual_resolution_required=len(report.manual_review_required),
                summary={
                    "positions_checked": report.positions_checked,
                    "balances_checked": report.balances_checked,
                    "orders_checked": report.orders_checked,
                    "position_discrepancies": len(report.position_discrepancies),
                    "balance_discrepancies": len(report.balance_discrepancies),
                    "untracked_orders": len(report.untracked_orders),
                },
                errors=report.error_messages,
                strategy_metrics={
                    "comprehensive_check": True,
                    "include_historical": True,
                    "lookback_hours": self.config.historical_lookback_hours
                }
            )
            
            self.logger.info(f"Full reconciliation {reconciliation_id} completed successfully", 
                           source_module=self.service._source_module)
            return result
            
        except Exception as e:
            self.logger.error(f"Full reconciliation {reconciliation_id} failed: {e}", 
                            source_module=self.service._source_module)
            return ReconciliationResult(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.FULL,
                status='failed',
                start_time=start_time,
                end_time=datetime.now(UTC),
                total_records_processed=0,
                discrepancies_found=0,
                auto_resolved_count=0,
                manual_resolution_required=0,
                summary={},
                errors=[str(e)]
            )


class IncrementalReconciliationStrategy(BaseReconciliationStrategy):
    """Reconcile only changes since last reconciliation."""
    
    def get_strategy_name(self) -> str:
        return "Incremental Reconciliation"
    
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute incremental reconciliation."""
        
        reconciliation_id = f"incremental_{int(datetime.now().timestamp())}"
        start_time = datetime.now(UTC)
        
        try:
            self.logger.info(f"Starting incremental reconciliation {reconciliation_id}", 
                           source_module=self.service._source_module)
            
            # Get cutoff time for incremental reconciliation
            cutoff_time = start_time - timedelta(hours=self.config.incremental_cutoff_hours)
            
            # Create report for incremental reconciliation
            report = ReconciliationReport(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.INCREMENTAL
            )
            
            # For incremental, focus primarily on recent changes
            # We'll use existing methods but with more focused scope
            await self.service._reconcile_positions(report)  # This will get all positions
            
            # Filter recent orders only
            await self._reconcile_recent_orders_only(report, cutoff_time)
            
            # Apply auto-corrections for small discrepancies
            if self.config.auto_resolve_threshold > 0:
                await self.service._apply_auto_corrections(report)
            
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.INCREMENTAL,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(UTC),
                total_records_processed=report.positions_checked + report.orders_checked,
                discrepancies_found=report.total_discrepancies,
                auto_resolved_count=len(report.auto_corrections),
                manual_resolution_required=len(report.manual_review_required),
                summary={
                    "cutoff_time": cutoff_time.isoformat(),
                    "lookback_hours": self.config.incremental_cutoff_hours,
                    "positions_checked": report.positions_checked,
                    "recent_orders_checked": report.orders_checked,
                },
                errors=report.error_messages,
                strategy_metrics={
                    "incremental": True,
                    "cutoff_hours": self.config.incremental_cutoff_hours,
                    "focused_scope": True
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Incremental reconciliation failed: {e}", 
                            source_module=self.service._source_module)
            return self._create_error_result(reconciliation_id, start_time, e, ReconciliationType.INCREMENTAL)
    
    async def _reconcile_recent_orders_only(self, report: ReconciliationReport, cutoff_time: datetime) -> None:
        """Reconcile only recent orders since cutoff time."""
        try:
            # Get recent orders from exchange
            exchange_orders = await self.service.execution_handler.get_recent_orders(cutoff_time)  # type: ignore[attr-defined]
            
            report.orders_checked = len(exchange_orders)
            
            # Check if all orders are tracked
            for ex_order_data in exchange_orders:
                exchange_order_id = ex_order_data["order_id"]
                
                # Check if order exists in our system
                tracked_order_model = await self.service.order_repository.find_by_exchange_id(exchange_order_id)
                
                if not tracked_order_model:
                    report.untracked_orders.append(exchange_order_id)
                    
                    if ex_order_data.get("status") == "filled":
                        report.manual_review_required.append({
                            "type": "order", "order_id": exchange_order_id,
                            "pair": ex_order_data.get("pair", "UNKNOWN"),
                            "side": ex_order_data.get("side", "UNKNOWN"),
                            "quantity": str(ex_order_data.get("quantity", 0)),
                            "issue": "Recent filled order not tracked in internal system",
                        })
        except Exception as e:
            self.logger.exception("Error during recent order reconciliation", 
                                source_module=self.service._source_module)
            report.error_messages.append(f"Recent order reconciliation error: {e!s}")
            raise

    def _create_error_result(self, reconciliation_id: str, start_time: datetime, 
                           error: Exception, recon_type: ReconciliationType) -> ReconciliationResult:
        """Create error result for failed reconciliation."""
        return ReconciliationResult(
            reconciliation_id=reconciliation_id,
            reconciliation_type=recon_type,
            status='failed',
            start_time=start_time,
            end_time=datetime.now(UTC),
            total_records_processed=0,
            discrepancies_found=0,
            auto_resolved_count=0,
            manual_resolution_required=0,
            summary={},
            errors=[str(error)]
        )


class RealTimeReconciliationStrategy(BaseReconciliationStrategy):
    """Continuous real-time reconciliation for critical operations."""
    
    def get_strategy_name(self) -> str:
        return "Real-Time Reconciliation"
    
    async def execute_reconciliation(self) -> ReconciliationResult:
        """Execute real-time reconciliation focusing on recent critical data."""
        
        reconciliation_id = f"realtime_{int(datetime.now().timestamp())}"
        start_time = datetime.now(UTC)
        
        try:
            self.logger.info(f"Starting real-time reconciliation {reconciliation_id}", 
                           source_module=self.service._source_module)
            
            # Real-time reconciliation focuses on very recent data
            cutoff_time = start_time - timedelta(minutes=self.config.real_time_cutoff_minutes)
            
            # Create focused report for real-time processing
            report = ReconciliationReport(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.REAL_TIME
            )
            
            # Focus on positions and very recent orders only
            await self.service._reconcile_positions(report)
            await self._reconcile_recent_orders_only(report, cutoff_time)
            
            # For real-time, identify critical discrepancies immediately
            critical_discrepancies = [
                d for d in report.position_discrepancies 
                if d.severity == "critical"
            ]
            
            # Send immediate alerts for critical issues
            if critical_discrepancies:
                await self._send_immediate_alerts(critical_discrepancies)
            
            result = ReconciliationResult(
                reconciliation_id=reconciliation_id,
                reconciliation_type=ReconciliationType.REAL_TIME,
                status='completed',
                start_time=start_time,
                end_time=datetime.now(UTC),
                total_records_processed=report.positions_checked + report.orders_checked,
                discrepancies_found=report.total_discrepancies,
                auto_resolved_count=0,  # Real-time doesn't auto-resolve
                manual_resolution_required=report.total_discrepancies,
                summary={
                    "cutoff_minutes": self.config.real_time_cutoff_minutes,
                    "critical_issues_found": len(critical_discrepancies),
                    "immediate_alerts_sent": len(critical_discrepancies) > 0,
                },
                errors=report.error_messages,
                strategy_metrics={
                    "real_time": True,
                    "cutoff_minutes": self.config.real_time_cutoff_minutes,
                    "critical_focus": True,
                    "no_auto_resolution": True
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Real-time reconciliation failed: {e}", 
                            source_module=self.service._source_module)
            return self._create_error_result(reconciliation_id, start_time, e, ReconciliationType.REAL_TIME)
    
    async def _reconcile_recent_orders_only(self, report: ReconciliationReport, cutoff_time: datetime) -> None:
        """Reconcile only very recent orders for real-time processing."""
        try:
            exchange_orders = await self.service.execution_handler.get_recent_orders(cutoff_time)  # type: ignore[attr-defined]
            
            report.orders_checked = len(exchange_orders)
            
            # For real-time, focus on filled orders that may affect positions
            for ex_order_data in exchange_orders:
                if ex_order_data.get("status") == "filled":
                    exchange_order_id = ex_order_data["order_id"]
                    tracked_order_model = await self.service.order_repository.find_by_exchange_id(exchange_order_id)
                    
                    if not tracked_order_model:
                        report.untracked_orders.append(exchange_order_id)
                        # Mark as requiring immediate attention
                        report.manual_review_required.append({
                            "type": "critical_order", 
                            "order_id": exchange_order_id,
                            "pair": ex_order_data.get("pair", "UNKNOWN"),
                            "side": ex_order_data.get("side", "UNKNOWN"),
                            "quantity": str(ex_order_data.get("quantity", 0)),
                            "issue": "Recent filled order not tracked - immediate review required",
                            "severity": "critical",
                            "timestamp": ex_order_data.get("timestamp", ""),
                        })
        except Exception as e:
            self.logger.exception("Error during real-time order reconciliation", 
                                source_module=self.service._source_module)
            report.error_messages.append(f"Real-time order reconciliation error: {e!s}")
            raise
    
    async def _send_immediate_alerts(self, critical_discrepancies: list[PositionDiscrepancy]) -> None:
        """Send immediate alerts for critical discrepancies."""
        for discrepancy in critical_discrepancies:
            alert = Alert(
                alert_id=f"realtime_critical_{uuid.uuid4()}",
                title=f"CRITICAL: Real-Time Reconciliation Alert",
                message=f"Critical discrepancy detected in {discrepancy.trading_pair}: "
                       f"{discrepancy.discrepancy_type.value}",
                severity=AlertSeverity.CRITICAL,
                source=self.service._source_module,
                tags={
                    "type": "real_time_reconciliation",
                    "trading_pair": discrepancy.trading_pair,
                    "discrepancy_type": discrepancy.discrepancy_type.value,
                    "immediate_action_required": True,
                },
            )
            await self.service.alerting.send_alert(alert)

    def _create_error_result(self, reconciliation_id: str, start_time: datetime, 
                           error: Exception, recon_type: ReconciliationType) -> ReconciliationResult:
        """Create error result for failed reconciliation."""
        return ReconciliationResult(
            reconciliation_id=reconciliation_id,
            reconciliation_type=recon_type,
            status='failed',
            start_time=start_time,
            end_time=datetime.now(UTC),
            total_records_processed=0,
            discrepancies_found=0,
            auto_resolved_count=0,
            manual_resolution_required=0,
            summary={},
            errors=[str(error)]
        )


class ReconciliationStrategyFactory:
    """Factory for creating reconciliation strategies based on configuration."""
    
    _strategies = {
        ReconciliationType.FULL: FullReconciliationStrategy,
        ReconciliationType.INCREMENTAL: IncrementalReconciliationStrategy,
        ReconciliationType.REAL_TIME: RealTimeReconciliationStrategy,
        # Additional strategies can be added here as they are implemented
    }
    
    @classmethod
    def create_strategy(cls, 
                       config: ReconciliationConfig, 
                       reconciliation_service: 'ReconciliationService') -> BaseReconciliationStrategy:
        """Create reconciliation strategy based on configuration.
        
        Args:
            config: ReconciliationConfiguration specifying the strategy type and parameters
            reconciliation_service: ReconciliationService instance for accessing shared methods
            
        Returns:
            BaseReconciliationStrategy: Configured strategy instance
            
        Raises:
            ValueError: If reconciliation type is not supported
        """
        strategy_class = cls._strategies.get(config.reconciliation_type)
        if not strategy_class:
            available_types = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported reconciliation type: {config.reconciliation_type}. "
                f"Available types: {[t.value for t in available_types]}"
            )
        
        return strategy_class(config, reconciliation_service)
    
    @classmethod
    def get_supported_types(cls) -> list[ReconciliationType]:
        """Get list of supported reconciliation types."""
        return list(cls._strategies.keys())
    
    @classmethod
    def is_supported(cls, reconciliation_type: ReconciliationType) -> bool:
        """Check if a reconciliation type is supported."""
        return reconciliation_type in cls._strategies


class ReconciliationService:
    """Automated portfolio reconciliation with exchange using configurable strategies."""

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
        self._current_reconciliation_type: ReconciliationType | None = None

    def _load_reconciliation_config(self, reconciliation_type: ReconciliationType | None = None) -> ReconciliationConfig:
        """Load reconciliation configuration from config manager.
        
        Args:
            reconciliation_type: Optional override for reconciliation type
            
        Returns:
            ReconciliationConfig: Configured reconciliation parameters
        """
        # Use provided type or get from configuration
        if not reconciliation_type:
            config_type_str = self.config.get('reconciliation.default_type', 'full')
            try:
                reconciliation_type = ReconciliationType(config_type_str)
            except ValueError:
                self.logger.warning(
                    f"Invalid reconciliation type '{config_type_str}' in config, defaulting to FULL",
                    source_module=self._source_module
                )
                reconciliation_type = ReconciliationType.FULL
        
        # Load strategy-specific configuration
        config_dict = self.config.get('reconciliation', {})
        
        return ReconciliationConfig(
            reconciliation_type=reconciliation_type,
            max_discrepancy_threshold=config_dict.get('max_discrepancy_threshold', 0.01),
            auto_resolve_threshold=config_dict.get('auto_resolve_threshold', 0.001),
            include_pending_trades=config_dict.get('include_pending_trades', True),
            historical_lookback_hours=config_dict.get('historical_lookback_hours', 24),
            enable_alerts=config_dict.get('enable_alerts', True),
            batch_size=config_dict.get('batch_size', 1000),
            timeout_seconds=config_dict.get('timeout_seconds', 300),
            retry_attempts=config_dict.get('retry_attempts', 3),
            real_time_cutoff_minutes=config_dict.get('real_time_cutoff_minutes', 15),
            incremental_cutoff_hours=config_dict.get('incremental_cutoff_hours', 1),
            emergency_alert_threshold=config_dict.get('emergency_alert_threshold', 0.10)
        )

    async def perform_configurable_reconciliation(self, reconciliation_type: ReconciliationType | None = None) -> ReconciliationResult:
        """
        Perform reconciliation with configurable type using strategy pattern.
        This replaces the hardcoded "full" reconciliation approach.
        
        Args:
            reconciliation_type: Optional reconciliation type override
            
        Returns:
            ReconciliationResult: Detailed results with strategy-specific metrics
        """
        try:
            # Get reconciliation configuration
            reconciliation_config = self._load_reconciliation_config(reconciliation_type)
            
            # Store current reconciliation type for tracking
            self._current_reconciliation_type = reconciliation_config.reconciliation_type
            
            self.logger.info(
                f"Starting {reconciliation_config.reconciliation_type.value} reconciliation using strategy pattern",
                source_module=self._source_module,
                context={
                    "strategy": reconciliation_config.reconciliation_type.value,
                    "auto_resolve_threshold": reconciliation_config.auto_resolve_threshold,
                    "lookback_hours": reconciliation_config.historical_lookback_hours
                }
            )
            
            # Validate that the reconciliation type is supported
            if not ReconciliationStrategyFactory.is_supported(reconciliation_config.reconciliation_type):
                available_types = [t.value for t in ReconciliationStrategyFactory.get_supported_types()]
                raise ValueError(
                    f"Reconciliation type '{reconciliation_config.reconciliation_type.value}' not supported. "
                    f"Available: {available_types}"
                )
            
            # Create appropriate strategy
            strategy = ReconciliationStrategyFactory.create_strategy(
                reconciliation_config,
                self
            )
            
            self.logger.info(
                f"Created {strategy.get_strategy_name()} strategy",
                source_module=self._source_module
            )
            
            # Execute reconciliation using strategy
            result = await strategy.execute_reconciliation()
            
            # Log results with strategy-specific context
            self._log_reconciliation_result(result)
            
            # Store results for auditing with dynamic reconciliation type
            await self._store_reconciliation_result(result)
            
            # Update service state
            self._last_reconciliation = datetime.now(UTC)
            self._consecutive_failures = 0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Configurable reconciliation failed: {e}", 
                            source_module=self._source_module)
            self._consecutive_failures += 1
            raise

    def _log_reconciliation_result(self, result: ReconciliationResult) -> None:
        """Log reconciliation result with strategy-specific details."""
        duration = (result.end_time - result.start_time).total_seconds()
        
        context = {
            "reconciliation_type": result.reconciliation_type.value,
            "status": result.status,
            "duration_seconds": duration,
            "discrepancies_found": result.discrepancies_found,
            "auto_resolved": result.auto_resolved_count,
            "manual_review_required": result.manual_resolution_required,
            "total_records": result.total_records_processed,
            "strategy_metrics": result.strategy_metrics
        }
        
        if result.status == 'completed':
            self.logger.info(
                f"{result.reconciliation_type.value.title()} reconciliation completed successfully",
                source_module=self._source_module,
                context=context
            )
        else:
            self.logger.error(
                f"{result.reconciliation_type.value.title()} reconciliation failed",
                source_module=self._source_module,
                context=context
            )

    async def _store_reconciliation_result(self, result: ReconciliationResult) -> None:
        """Store reconciliation result with dynamic reconciliation type."""
        try:
            event_data = {
                "reconciliation_id": uuid.UUID(result.reconciliation_id),
                "timestamp": result.start_time,
                "reconciliation_type": result.reconciliation_type.value,  # Dynamic type from strategy
                "status": result.status,
                "discrepancies_found": result.discrepancies_found,
                "auto_corrected": result.auto_resolved_count,
                "manual_review_required": result.manual_resolution_required,
                "report": result.to_dict(),  # Full result as JSON
                "duration_seconds": Decimal(str((result.end_time - result.start_time).total_seconds())),
            }
            created_event = await self.reconciliation_repository.save_reconciliation_event(event_data)
            self.logger.info(
                f"Saved {result.reconciliation_type.value} reconciliation event {created_event.reconciliation_id}",
                source_module=self._source_module
            )

        except Exception as e:
            self.logger.exception(
                f"Error storing reconciliation result for {result.reconciliation_id}: {e}",
                source_module=self._source_module
            )

    async def start(self) -> None:
        """Start reconciliation service."""
        self.logger.info(
            "Starting reconciliation service",
            source_module=self._source_module,
        )

        # Run initial reconciliation using configurable strategy
        await self.perform_configurable_reconciliation()

        # Start periodic reconciliation
        self._reconciliation_task = asyncio.create_task(self._periodic_reconciliation())

    async def stop(self) -> None:
        """Stop reconciliation service."""
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconciliation_task

    async def _periodic_reconciliation(self) -> None:
        """Run reconciliation periodically using configurable strategy."""
        while True:
            try:
                await asyncio.sleep(self.reconciliation_interval * 60)
                await self.perform_configurable_reconciliation()

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

    async def run_reconciliation(self, reconciliation_type: ReconciliationType | None = None) -> ReconciliationReport:
        """
        Run complete reconciliation process with configurable strategy.
        
        Maintained for backward compatibility while using new strategy pattern internally.
        
        Args:
            reconciliation_type: Optional reconciliation type override
            
        Returns:
            ReconciliationReport: Legacy format report for backward compatibility
        """
        try:
            self.logger.info(
                "Starting reconciliation process via legacy interface",
                source_module=self._source_module,
                context={"using_strategy_pattern": True}
            )

            # Use the new configurable reconciliation approach
            result = await self.perform_configurable_reconciliation(reconciliation_type)
            
            # Convert ReconciliationResult to legacy ReconciliationReport format for backward compatibility
            legacy_report = self._convert_result_to_legacy_report(result)
            
            # Send alerts using legacy method for compatibility
            await self._send_reconciliation_alerts(legacy_report)
            
            return legacy_report

        except Exception as e:
            # Create a failed legacy report for backward compatibility
            failed_report = ReconciliationReport()
            failed_report.status = ReconciliationStatus.FAILED
            failed_report.error_messages.append(str(e))
            failed_report.duration_seconds = 0
            failed_report.reconciliation_type = reconciliation_type or ReconciliationType.FULL

            self.logger.exception(
                "Legacy reconciliation interface failed",
                source_module=self._source_module,
            )
            
            await self._send_critical_alert(f"Reconciliation failed: {e!s}")
            return failed_report

    def _convert_result_to_legacy_report(self, result: ReconciliationResult) -> ReconciliationReport:
        """Convert ReconciliationResult to legacy ReconciliationReport format."""
        # Map new status to legacy status
        status_mapping = {
            'completed': ReconciliationStatus.SUCCESS,
            'failed': ReconciliationStatus.FAILED,
            'partial': ReconciliationStatus.PARTIAL
        }
        
        legacy_report = ReconciliationReport()
        legacy_report.reconciliation_id = result.reconciliation_id
        legacy_report.timestamp = result.start_time
        legacy_report.status = status_mapping.get(result.status, ReconciliationStatus.FAILED)
        legacy_report.reconciliation_type = result.reconciliation_type
        legacy_report.duration_seconds = (result.end_time - result.start_time).total_seconds()
        legacy_report.error_messages = result.errors

        # Populate summary data from result summary
        summary = result.summary
        legacy_report.positions_checked = summary.get('positions_checked', 0)
        legacy_report.balances_checked = summary.get('balances_checked', 0)
        legacy_report.orders_checked = summary.get('orders_checked', 0)
        
        # Create empty collections for discrepancies (strategy handles the actual logic)
        legacy_report.position_discrepancies = []
        legacy_report.balance_discrepancies = []
        legacy_report.untracked_orders = []
        legacy_report.auto_corrections = []
        legacy_report.manual_review_required = []
        
        return legacy_report

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
                "reconciliation_type": (self._current_reconciliation_type.value 
                                      if self._current_reconciliation_type 
                                      else report.reconciliation_type.value),  # Dynamic type from strategy
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
            "last_report": last_report_data if last_report_data else None,
            "current_reconciliation_type": (self._current_reconciliation_type.value 
                                          if self._current_reconciliation_type 
                                          else None),
            "supported_reconciliation_types": [t.value for t in ReconciliationStrategyFactory.get_supported_types()],
        }

    async def get_supported_reconciliation_types(self) -> list[str]:
        """Get list of supported reconciliation types.
        
        Returns:
            list[str]: List of supported reconciliation type names
        """
        return [t.value for t in ReconciliationStrategyFactory.get_supported_types()]

    def get_current_reconciliation_config(self) -> dict[str, Any]:
        """Get current reconciliation configuration for monitoring.
        
        Returns:
            dict[str, Any]: Current reconciliation configuration
        """
        try:
            config = self._load_reconciliation_config()
            return {
                "reconciliation_type": config.reconciliation_type.value,
                "max_discrepancy_threshold": config.max_discrepancy_threshold,
                "auto_resolve_threshold": config.auto_resolve_threshold,
                "include_pending_trades": config.include_pending_trades,
                "historical_lookback_hours": config.historical_lookback_hours,
                "enable_alerts": config.enable_alerts,
                "batch_size": config.batch_size,
                "timeout_seconds": config.timeout_seconds,
                "retry_attempts": config.retry_attempts,
                "real_time_cutoff_minutes": config.real_time_cutoff_minutes,
                "incremental_cutoff_hours": config.incremental_cutoff_hours,
                "emergency_alert_threshold": config.emergency_alert_threshold,
            }
        except Exception as e:
            self.logger.error(f"Error getting reconciliation config: {e}", 
                            source_module=self._source_module)
            return {"error": str(e)}

    async def perform_emergency_reconciliation(self) -> ReconciliationResult:
        """Perform emergency reconciliation with critical alerting.
        
        Returns:
            ReconciliationResult: Emergency reconciliation results
        """
        self.logger.warning(
            "Emergency reconciliation requested",
            source_module=self._source_module
        )
        
        # Use real-time strategy for emergency situations with stricter thresholds
        emergency_config = self._load_reconciliation_config(ReconciliationType.REAL_TIME)
        emergency_config.emergency_alert_threshold = 0.01  # Very low threshold
        emergency_config.auto_resolve_threshold = 0.0  # No auto-resolution in emergency
        
        try:
            strategy = ReconciliationStrategyFactory.create_strategy(emergency_config, self)
            result = await strategy.execute_reconciliation()
            
            # Send emergency alert regardless of outcome
            await self._send_emergency_alert(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Emergency reconciliation failed: {e}", 
                            source_module=self._source_module)
            await self._send_critical_alert(f"Emergency reconciliation failed: {e!s}")
            raise

    async def _send_emergency_alert(self, result: ReconciliationResult) -> None:
        """Send emergency alert for reconciliation results."""
        alert = Alert(
            alert_id=f"emergency_recon_{result.reconciliation_id}",
            title="EMERGENCY RECONCILIATION COMPLETED",
            message=f"Emergency reconciliation completed with {result.discrepancies_found} discrepancies. "
                   f"Status: {result.status}. Manual review required: {result.manual_resolution_required}",
            severity=AlertSeverity.CRITICAL,
            source=self._source_module,
            tags={
                "type": "emergency_reconciliation",
                "reconciliation_id": result.reconciliation_id,
                "discrepancies": result.discrepancies_found,
                "status": result.status,
            },
        )
        await self.alerting.send_alert(alert)
