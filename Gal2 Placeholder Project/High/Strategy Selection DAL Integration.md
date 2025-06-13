# Strategy Selection DAL Integration Implementation Design

**File**: `/gal_friday/strategy_selection.py`
- **Line 493**: `# Placeholder for now - should integrate with DAL`
- **Line 502**: `# Placeholder for now - should integrate with DAL`
- **Line 512**: `# Placeholder for now - should integrate with DAL`
- **Impact**: Strategy management not persisted

## Overview
The Strategy Selection system currently lacks proper DAL integration for persisting strategy performance metrics, configurations, and selection decisions. This design implements a comprehensive data persistence layer that enables strategy performance tracking, historical analysis, and audit trails for regulatory compliance.

## Architecture Design

### 1. Current DAL Integration Gaps

```
Missing DAL Components:
├── Strategy Models
│   ├── No StrategyConfiguration model
│   ├── No StrategyPerformance model
│   └── No StrategySelection model
├── Repository Layer
│   ├── No StrategyRepository
│   ├── No PerformanceMetricsRepository
│   └── No SelectionHistoryRepository
├── Data Persistence
│   ├── Strategy metrics not stored
│   ├── Selection decisions not logged
│   └── No performance history
└── Query Capabilities
    ├── No strategy comparison queries
    ├── No performance trending
    └── No audit trail access
```

### 2. Production DAL Integration Strategy

```
Enhanced Strategy DAL System:
├── Database Models
│   ├── StrategyConfig (strategy definitions)
│   ├── StrategyPerformanceSnapshot (metrics)
│   ├── StrategySelectionEvent (decisions)
│   └── StrategyBacktest (validation results)
├── Repository Layer
│   ├── Async CRUD operations
│   ├── Complex query support
│   ├── Performance optimization
│   └── Connection pooling
├── Data Services
│   ├── Performance aggregation
│   ├── Historical analysis
│   ├── Comparison utilities
│   └── Export capabilities
└── Integration Layer
    ├── Event-driven updates
    ├── Cache management
    ├── Transaction handling
    └── Error recovery
```

## Implementation Plan

### Phase 1: Database Models

```python
from datetime import datetime, UTC
from decimal import Decimal
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

from sqlalchemy import (
    DateTime, Numeric, String, Text, Boolean, Integer, 
    Index, ForeignKey, JSON, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .models_base import Base


class StrategyConfig(Base):
    """Strategy configuration and metadata model."""
    
    __tablename__ = "strategy_configs"
    
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    strategy_id: Mapped[str] = mapped_column(
        String(100), 
        nullable=False, 
        unique=True, 
        index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Strategy Classification
    strategy_type: Mapped[str] = mapped_column(String(50), nullable=False)
    trading_pairs: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    timeframes: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    
    # Configuration Parameters
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    risk_parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Operational Settings
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_paper_trading: Mapped[bool] = mapped_column(Boolean, default=True)
    max_position_size: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    max_daily_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now(),
        onupdate=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    
    # Relationships
    performance_snapshots = relationship(
        "StrategyPerformanceSnapshot", 
        back_populates="strategy_config",
        cascade="all, delete-orphan"
    )
    selection_events = relationship(
        "StrategySelectionEvent", 
        back_populates="strategy_config"
    )
    backtest_results = relationship(
        "StrategyBacktestResult", 
        back_populates="strategy_config"
    )
    
    __table_args__ = (
        Index("idx_strategy_configs_type_active", "strategy_type", "is_active"),
        Index("idx_strategy_configs_updated_at", "updated_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "trading_pairs": self.trading_pairs,
            "timeframes": self.timeframes,
            "parameters": self.parameters,
            "risk_parameters": self.risk_parameters,
            "is_active": self.is_active,
            "is_paper_trading": self.is_paper_trading,
            "max_position_size": float(self.max_position_size),
            "max_daily_trades": self.max_daily_trades,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version
        }


class StrategyPerformanceSnapshot(Base):
    """Strategy performance metrics snapshot model."""
    
    __tablename__ = "strategy_performance_snapshots"
    
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    strategy_config_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("strategy_configs.id"),
        nullable=False,
        index=True
    )
    
    # Snapshot Metadata
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False,
        index=True
    )
    evaluation_period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False
    )
    evaluation_period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False
    )
    
    # Financial Performance Metrics
    total_return: Mapped[Decimal] = mapped_column(Numeric(15, 8), nullable=False)
    annualized_return: Mapped[Decimal] = mapped_column(Numeric(15, 8), nullable=False)
    sharpe_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=True)
    sortino_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=True)
    calmar_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=True)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    current_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    
    # Trading Metrics
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    win_rate: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    profit_factor: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=True)
    average_win: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=True)
    average_loss: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=True)
    largest_win: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=True)
    largest_loss: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=True)
    
    # Risk Metrics
    volatility: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    downside_deviation: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=True)
    var_95: Mapped[Decimal] = mapped_column(Numeric(15, 8), nullable=True)
    cvar_95: Mapped[Decimal] = mapped_column(Numeric(15, 8), nullable=True)
    max_drawdown_duration_days: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=True)
    
    # Execution Metrics
    average_slippage_bps: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=True)
    fill_rate: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    average_latency_ms: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=True)
    api_error_rate: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    
    # Operational Metrics
    cpu_usage_avg: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=True)
    memory_usage_avg_mb: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=True)
    signal_generation_rate: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=True)
    halt_frequency: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    
    # Additional Metrics (JSON for flexibility)
    extended_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    
    # Relationships
    strategy_config = relationship("StrategyConfig", back_populates="performance_snapshots")
    
    __table_args__ = (
        Index("idx_perf_snapshots_strategy_date", "strategy_config_id", "snapshot_date"),
        Index("idx_perf_snapshots_sharpe", "sharpe_ratio"),
        Index("idx_perf_snapshots_return", "total_return"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "strategy_config_id": str(self.strategy_config_id),
            "snapshot_date": self.snapshot_date.isoformat(),
            "evaluation_period_start": self.evaluation_period_start.isoformat(),
            "evaluation_period_end": self.evaluation_period_end.isoformat(),
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else None,
            "sortino_ratio": float(self.sortino_ratio) if self.sortino_ratio else None,
            "calmar_ratio": float(self.calmar_ratio) if self.calmar_ratio else None,
            "max_drawdown": float(self.max_drawdown),
            "current_drawdown": float(self.current_drawdown),
            "total_trades": self.total_trades,
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor) if self.profit_factor else None,
            "volatility": float(self.volatility),
            "fill_rate": float(self.fill_rate),
            "api_error_rate": float(self.api_error_rate),
            "halt_frequency": float(self.halt_frequency),
            "extended_metrics": self.extended_metrics,
            "created_at": self.created_at.isoformat()
        }


class StrategySelectionEvent(Base):
    """Strategy selection decision and reasoning model."""
    
    __tablename__ = "strategy_selection_events"
    
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Selection Context
    selection_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False,
        index=True
    )
    selection_type: Mapped[str] = mapped_column(
        String(50), 
        nullable=False,
        index=True
    )  # "automatic", "manual", "emergency", "scheduled"
    
    # Selected Strategy
    selected_strategy_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("strategy_configs.id"),
        nullable=False,
        index=True
    )
    previous_strategy_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("strategy_configs.id"),
        nullable=True
    )
    
    # Selection Reasoning
    selection_criteria: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_strategies: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    selection_scores: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    market_conditions: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    
    # Transition Details
    transition_phase: Mapped[str] = mapped_column(String(50), nullable=False)
    transition_completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=True
    )
    allocation_percentage: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), 
        nullable=False
    )
    
    # Metadata
    triggered_by: Mapped[str] = mapped_column(String(100), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    risk_override: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    strategy_config = relationship(
        "StrategyConfig", 
        foreign_keys=[selected_strategy_id],
        back_populates="selection_events"
    )
    
    __table_args__ = (
        Index("idx_selection_events_timestamp", "selection_timestamp"),
        Index("idx_selection_events_type", "selection_type"),
        Index("idx_selection_events_strategy", "selected_strategy_id"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "selection_timestamp": self.selection_timestamp.isoformat(),
            "selection_type": self.selection_type,
            "selected_strategy_id": str(self.selected_strategy_id),
            "previous_strategy_id": str(self.previous_strategy_id) if self.previous_strategy_id else None,
            "selection_criteria": self.selection_criteria,
            "candidate_strategies": self.candidate_strategies,
            "selection_scores": self.selection_scores,
            "market_conditions": self.market_conditions,
            "transition_phase": self.transition_phase,
            "transition_completed_at": self.transition_completed_at.isoformat() if self.transition_completed_at else None,
            "allocation_percentage": float(self.allocation_percentage),
            "triggered_by": self.triggered_by,
            "reason": self.reason,
            "risk_override": self.risk_override
        }


class StrategyBacktestResult(Base):
    """Strategy backtest validation results model."""
    
    __tablename__ = "strategy_backtest_results"
    
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    strategy_config_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        ForeignKey("strategy_configs.id"),
        nullable=False,
        index=True
    )
    
    # Backtest Parameters
    backtest_start_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False
    )
    backtest_end_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False
    )
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    
    # Results Summary
    final_value: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    total_return: Mapped[Decimal] = mapped_column(Numeric(15, 8), nullable=False)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    sharpe_ratio: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=True)
    total_trades: Mapped[int] = mapped_column(Integer, nullable=False)
    win_rate: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    
    # Detailed Results
    detailed_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    trade_history: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    # Validation Status
    validation_status: Mapped[str] = mapped_column(
        String(50), 
        nullable=False,
        index=True
    )  # "passed", "failed", "warning"
    validation_notes: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Relationships
    strategy_config = relationship("StrategyConfig", back_populates="backtest_results")
    
    __table_args__ = (
        Index("idx_backtest_results_strategy_date", "strategy_config_id", "created_at"),
        Index("idx_backtest_results_status", "validation_status"),
    )
```

### Phase 2: Repository Layer

```python
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, asc, func
from sqlalchemy.orm import selectinload

from ..base import BaseRepository
from .models import (
    StrategyConfig, 
    StrategyPerformanceSnapshot, 
    StrategySelectionEvent,
    StrategyBacktestResult
)


class StrategyRepository(BaseRepository[StrategyConfig]):
    """Repository for strategy configuration management."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, StrategyConfig)
    
    async def get_by_strategy_id(self, strategy_id: str) -> Optional[StrategyConfig]:
        """Get strategy by strategy_id."""
        stmt = select(StrategyConfig).where(StrategyConfig.strategy_id == strategy_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_active_strategies(
        self, 
        strategy_type: Optional[str] = None,
        trading_pairs: Optional[List[str]] = None
    ) -> List[StrategyConfig]:
        """Get active strategies with optional filtering."""
        stmt = select(StrategyConfig).where(StrategyConfig.is_active == True)
        
        if strategy_type:
            stmt = stmt.where(StrategyConfig.strategy_type == strategy_type)
        
        if trading_pairs:
            # Check if any of the specified trading pairs are in the strategy's trading_pairs JSON
            for pair in trading_pairs:
                stmt = stmt.where(StrategyConfig.trading_pairs.contains([pair]))
        
        stmt = stmt.order_by(StrategyConfig.updated_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def create_strategy_version(
        self, 
        base_strategy: StrategyConfig,
        updates: Dict[str, Any],
        created_by: str
    ) -> StrategyConfig:
        """Create new version of existing strategy."""
        new_strategy = StrategyConfig(
            strategy_id=base_strategy.strategy_id,
            name=updates.get("name", base_strategy.name),
            description=updates.get("description", base_strategy.description),
            strategy_type=base_strategy.strategy_type,
            trading_pairs=updates.get("trading_pairs", base_strategy.trading_pairs),
            timeframes=updates.get("timeframes", base_strategy.timeframes),
            parameters=updates.get("parameters", base_strategy.parameters),
            risk_parameters=updates.get("risk_parameters", base_strategy.risk_parameters),
            is_active=updates.get("is_active", True),
            is_paper_trading=updates.get("is_paper_trading", base_strategy.is_paper_trading),
            max_position_size=updates.get("max_position_size", base_strategy.max_position_size),
            max_daily_trades=updates.get("max_daily_trades", base_strategy.max_daily_trades),
            created_by=created_by,
            version=base_strategy.version + 1
        )
        
        # Deactivate previous version
        base_strategy.is_active = False
        
        self.session.add(new_strategy)
        await self.session.commit()
        return new_strategy
    
    async def search_strategies(
        self,
        search_term: str,
        limit: int = 50
    ) -> List[StrategyConfig]:
        """Search strategies by name or description."""
        stmt = (
            select(StrategyConfig)
            .where(
                or_(
                    StrategyConfig.name.ilike(f"%{search_term}%"),
                    StrategyConfig.description.ilike(f"%{search_term}%"),
                    StrategyConfig.strategy_id.ilike(f"%{search_term}%")
                )
            )
            .order_by(StrategyConfig.updated_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


class PerformanceMetricsRepository(BaseRepository[StrategyPerformanceSnapshot]):
    """Repository for strategy performance metrics."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, StrategyPerformanceSnapshot)
    
    async def get_latest_performance(
        self, 
        strategy_config_id: UUID
    ) -> Optional[StrategyPerformanceSnapshot]:
        """Get latest performance snapshot for strategy."""
        stmt = (
            select(StrategyPerformanceSnapshot)
            .where(StrategyPerformanceSnapshot.strategy_config_id == strategy_config_id)
            .order_by(StrategyPerformanceSnapshot.snapshot_date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_performance_history(
        self,
        strategy_config_id: UUID,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[StrategyPerformanceSnapshot]:
        """Get performance history for strategy."""
        stmt = (
            select(StrategyPerformanceSnapshot)
            .where(StrategyPerformanceSnapshot.strategy_config_id == strategy_config_id)
        )
        
        if start_date:
            stmt = stmt.where(StrategyPerformanceSnapshot.snapshot_date >= start_date)
        if end_date:
            stmt = stmt.where(StrategyPerformanceSnapshot.snapshot_date <= end_date)
        
        stmt = (
            stmt.order_by(StrategyPerformanceSnapshot.snapshot_date.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_top_performers(
        self,
        metric: str = "sharpe_ratio",
        period_days: int = 30,
        limit: int = 10
    ) -> List[StrategyPerformanceSnapshot]:
        """Get top performing strategies by specified metric."""
        cutoff_date = datetime.now(UTC) - timedelta(days=period_days)
        
        # Map metric names to model attributes
        metric_map = {
            "sharpe_ratio": StrategyPerformanceSnapshot.sharpe_ratio,
            "total_return": StrategyPerformanceSnapshot.total_return,
            "sortino_ratio": StrategyPerformanceSnapshot.sortino_ratio,
            "calmar_ratio": StrategyPerformanceSnapshot.calmar_ratio,
            "win_rate": StrategyPerformanceSnapshot.win_rate
        }
        
        if metric not in metric_map:
            raise ValueError(f"Unknown metric: {metric}")
        
        stmt = (
            select(StrategyPerformanceSnapshot)
            .where(StrategyPerformanceSnapshot.snapshot_date >= cutoff_date)
            .order_by(metric_map[metric].desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_performance_comparison(
        self,
        strategy_ids: List[UUID],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[UUID, List[StrategyPerformanceSnapshot]]:
        """Compare performance across multiple strategies."""
        stmt = (
            select(StrategyPerformanceSnapshot)
            .where(
                and_(
                    StrategyPerformanceSnapshot.strategy_config_id.in_(strategy_ids),
                    StrategyPerformanceSnapshot.snapshot_date >= start_date,
                    StrategyPerformanceSnapshot.snapshot_date <= end_date
                )
            )
            .order_by(
                StrategyPerformanceSnapshot.strategy_config_id,
                StrategyPerformanceSnapshot.snapshot_date
            )
        )
        
        result = await self.session.execute(stmt)
        snapshots = result.scalars().all()
        
        # Group by strategy_config_id
        comparison = {}
        for snapshot in snapshots:
            if snapshot.strategy_config_id not in comparison:
                comparison[snapshot.strategy_config_id] = []
            comparison[snapshot.strategy_config_id].append(snapshot)
        
        return comparison


class SelectionHistoryRepository(BaseRepository[StrategySelectionEvent]):
    """Repository for strategy selection history."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, StrategySelectionEvent)
    
    async def get_current_selection(self) -> Optional[StrategySelectionEvent]:
        """Get current active strategy selection."""
        stmt = (
            select(StrategySelectionEvent)
            .where(StrategySelectionEvent.transition_completed_at.is_not(None))
            .order_by(StrategySelectionEvent.selection_timestamp.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_selection_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        selection_type: Optional[str] = None,
        limit: int = 100
    ) -> List[StrategySelectionEvent]:
        """Get strategy selection history with filters."""
        stmt = select(StrategySelectionEvent)
        
        conditions = []
        if start_date:
            conditions.append(StrategySelectionEvent.selection_timestamp >= start_date)
        if end_date:
            conditions.append(StrategySelectionEvent.selection_timestamp <= end_date)
        if selection_type:
            conditions.append(StrategySelectionEvent.selection_type == selection_type)
        
        if conditions:
            stmt = stmt.where(and_(*conditions))
        
        stmt = (
            stmt.order_by(StrategySelectionEvent.selection_timestamp.desc())
            .limit(limit)
            .options(selectinload(StrategySelectionEvent.strategy_config))
        )
        
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
    
    async def get_strategy_usage_stats(
        self,
        period_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get strategy usage statistics."""
        cutoff_date = datetime.now(UTC) - timedelta(days=period_days)
        
        stmt = (
            select(
                StrategySelectionEvent.selected_strategy_id,
                func.count().label("selection_count"),
                func.avg(StrategySelectionEvent.allocation_percentage).label("avg_allocation"),
                func.max(StrategySelectionEvent.selection_timestamp).label("last_selected")
            )
            .where(StrategySelectionEvent.selection_timestamp >= cutoff_date)
            .group_by(StrategySelectionEvent.selected_strategy_id)
            .order_by(desc("selection_count"))
        )
        
        result = await self.session.execute(stmt)
        return [
            {
                "strategy_id": row.selected_strategy_id,
                "selection_count": row.selection_count,
                "avg_allocation": float(row.avg_allocation),
                "last_selected": row.last_selected.isoformat()
            }
            for row in result
        ]


class StrategyBacktestRepository(BaseRepository[StrategyBacktestResult]):
    """Repository for strategy backtest results."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, StrategyBacktestResult)
    
    async def get_latest_backtest(
        self, 
        strategy_config_id: UUID
    ) -> Optional[StrategyBacktestResult]:
        """Get latest backtest result for strategy."""
        stmt = (
            select(StrategyBacktestResult)
            .where(StrategyBacktestResult.strategy_config_id == strategy_config_id)
            .order_by(StrategyBacktestResult.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_validation_summary(
        self,
        validation_status: Optional[str] = None,
        days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get backtest validation summary."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
        
        stmt = select(StrategyBacktestResult).where(
            StrategyBacktestResult.created_at >= cutoff_date
        )
        
        if validation_status:
            stmt = stmt.where(StrategyBacktestResult.validation_status == validation_status)
        
        result = await self.session.execute(stmt)
        backtests = result.scalars().all()
        
        return [
            {
                "id": str(backtest.id),
                "strategy_config_id": str(backtest.strategy_config_id),
                "validation_status": backtest.validation_status,
                "total_return": float(backtest.total_return),
                "sharpe_ratio": float(backtest.sharpe_ratio) if backtest.sharpe_ratio else None,
                "max_drawdown": float(backtest.max_drawdown),
                "created_at": backtest.created_at.isoformat()
            }
            for backtest in backtests
        ]
```

### Phase 3: Enhanced Strategy Selection Service

```python
class EnhancedStrategySelectionService:
    """Production-ready strategy selection service with full DAL integration."""
    
    def __init__(
        self,
        strategy_repo: StrategyRepository,
        performance_repo: PerformanceMetricsRepository,
        selection_repo: SelectionHistoryRepository,
        backtest_repo: StrategyBacktestRepository,
        logger: LoggerService
    ):
        self.strategy_repo = strategy_repo
        self.performance_repo = performance_repo
        self.selection_repo = selection_repo
        self.backtest_repo = backtest_repo
        self.logger = logger
        self._source_module = self.__class__.__name__
    
    async def persist_performance_snapshot(
        self,
        strategy_id: str,
        metrics: StrategyPerformanceMetrics
    ) -> StrategyPerformanceSnapshot:
        """Persist strategy performance metrics to database."""
        try:
            # Get strategy config
            strategy_config = await self.strategy_repo.get_by_strategy_id(strategy_id)
            if not strategy_config:
                raise ValueError(f"Strategy not found: {strategy_id}")
            
            # Create performance snapshot
            snapshot = StrategyPerformanceSnapshot(
                strategy_config_id=strategy_config.id,
                snapshot_date=metrics.evaluation_timestamp,
                evaluation_period_start=metrics.evaluation_timestamp - timedelta(days=30),
                evaluation_period_end=metrics.evaluation_timestamp,
                total_return=Decimal(str(metrics.total_return)),
                annualized_return=Decimal(str(metrics.annualized_return)),
                sharpe_ratio=Decimal(str(metrics.sharpe_ratio)) if metrics.sharpe_ratio else None,
                sortino_ratio=Decimal(str(metrics.sortino_ratio)) if metrics.sortino_ratio else None,
                calmar_ratio=Decimal(str(metrics.calmar_ratio)) if metrics.calmar_ratio else None,
                max_drawdown=Decimal(str(metrics.max_drawdown)),
                current_drawdown=Decimal(str(metrics.current_drawdown)),
                total_trades=metrics.total_trades,
                win_rate=Decimal(str(metrics.win_rate)),
                profit_factor=Decimal(str(metrics.profit_factor)) if metrics.profit_factor else None,
                average_win=metrics.average_win,
                average_loss=metrics.average_loss,
                largest_win=metrics.largest_win,
                largest_loss=metrics.largest_loss,
                volatility=Decimal(str(metrics.volatility)),
                downside_deviation=Decimal(str(metrics.downside_deviation)) if metrics.downside_deviation else None,
                var_95=Decimal(str(metrics.var_95)) if metrics.var_95 else None,
                cvar_95=Decimal(str(metrics.cvar_95)) if metrics.cvar_95 else None,
                max_drawdown_duration_days=Decimal(str(metrics.max_drawdown_duration_days)) if metrics.max_drawdown_duration_days else None,
                average_slippage_bps=Decimal(str(metrics.average_slippage_bps)) if metrics.average_slippage_bps else None,
                fill_rate=Decimal(str(metrics.fill_rate)),
                average_latency_ms=Decimal(str(metrics.average_latency_ms)) if metrics.average_latency_ms else None,
                api_error_rate=Decimal(str(metrics.api_error_rate)),
                cpu_usage_avg=Decimal(str(metrics.cpu_usage_avg)) if metrics.cpu_usage_avg else None,
                memory_usage_avg_mb=Decimal(str(metrics.memory_usage_avg_mb)) if metrics.memory_usage_avg_mb else None,
                signal_generation_rate=Decimal(str(metrics.signal_generation_rate)) if metrics.signal_generation_rate else None,
                halt_frequency=Decimal(str(metrics.halt_frequency))
            )
            
            created_snapshot = await self.performance_repo.create(snapshot)
            
            self.logger.info(
                f"Persisted performance snapshot for strategy {strategy_id}",
                source_module=self._source_module
            )
            
            return created_snapshot
            
        except Exception as e:
            self.logger.error(
                f"Failed to persist performance snapshot: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def log_selection_decision(
        self,
        selected_strategy_id: str,
        previous_strategy_id: Optional[str],
        selection_criteria: Dict[str, Any],
        candidate_strategies: List[Dict[str, Any]],
        selection_scores: Dict[str, Any],
        market_conditions: Dict[str, Any],
        triggered_by: str,
        reason: Optional[str] = None
    ) -> StrategySelectionEvent:
        """Log strategy selection decision to database."""
        try:
            # Get strategy configs
            selected_config = await self.strategy_repo.get_by_strategy_id(selected_strategy_id)
            if not selected_config:
                raise ValueError(f"Selected strategy not found: {selected_strategy_id}")
            
            previous_config_id = None
            if previous_strategy_id:
                previous_config = await self.strategy_repo.get_by_strategy_id(previous_strategy_id)
                if previous_config:
                    previous_config_id = previous_config.id
            
            # Create selection event
            selection_event = StrategySelectionEvent(
                selection_timestamp=datetime.now(UTC),
                selection_type="automatic",  # Can be parameterized
                selected_strategy_id=selected_config.id,
                previous_strategy_id=previous_config_id,
                selection_criteria=selection_criteria,
                candidate_strategies=candidate_strategies,
                selection_scores=selection_scores,
                market_conditions=market_conditions,
                transition_phase="shadow_mode",  # Initial phase
                allocation_percentage=Decimal("0"),  # Start with 0, increase during transition
                triggered_by=triggered_by,
                reason=reason,
                risk_override=False
            )
            
            created_event = await self.selection_repo.create(selection_event)
            
            self.logger.info(
                f"Logged strategy selection: {selected_strategy_id}",
                source_module=self._source_module,
                extra={
                    "selection_id": str(created_event.id),
                    "triggered_by": triggered_by
                }
            )
            
            return created_event
            
        except Exception as e:
            self.logger.error(
                f"Failed to log selection decision: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def get_strategy_analytics(
        self,
        strategy_id: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for a strategy."""
        try:
            strategy_config = await self.strategy_repo.get_by_strategy_id(strategy_id)
            if not strategy_config:
                raise ValueError(f"Strategy not found: {strategy_id}")
            
            start_date = datetime.now(UTC) - timedelta(days=days_back)
            end_date = datetime.now(UTC)
            
            # Get performance history
            performance_history = await self.performance_repo.get_performance_history(
                strategy_config.id, start_date, end_date
            )
            
            # Get latest performance
            latest_performance = await self.performance_repo.get_latest_performance(
                strategy_config.id
            )
            
            # Get selection history
            selection_history = await self.selection_repo.get_selection_history(
                start_date, end_date
            )
            strategy_selections = [
                event for event in selection_history 
                if event.selected_strategy_id == strategy_config.id
            ]
            
            # Get latest backtest
            latest_backtest = await self.backtest_repo.get_latest_backtest(
                strategy_config.id
            )
            
            return {
                "strategy_config": strategy_config.to_dict(),
                "latest_performance": latest_performance.to_dict() if latest_performance else None,
                "performance_history": [p.to_dict() for p in performance_history],
                "selection_count": len(strategy_selections),
                "last_selected": strategy_selections[0].selection_timestamp.isoformat() if strategy_selections else None,
                "latest_backtest": latest_backtest.to_dict() if latest_backtest else None,
                "analytics_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": days_back
                }
            }
            
        except Exception as e:
            self.logger.error(
                f"Failed to get strategy analytics: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def export_strategy_data(
        self,
        strategy_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Export comprehensive strategy data for analysis."""
        try:
            # Get strategies
            if strategy_ids:
                strategies = []
                for strategy_id in strategy_ids:
                    config = await self.strategy_repo.get_by_strategy_id(strategy_id)
                    if config:
                        strategies.append(config)
            else:
                strategies = await self.strategy_repo.get_active_strategies()
            
            export_data = {
                "export_timestamp": datetime.now(UTC).isoformat(),
                "strategies": [],
                "performance_data": [],
                "selection_history": [],
                "backtest_results": []
            }
            
            for strategy in strategies:
                # Strategy config
                export_data["strategies"].append(strategy.to_dict())
                
                # Performance data
                performance_history = await self.performance_repo.get_performance_history(
                    strategy.id, start_date, end_date
                )
                export_data["performance_data"].extend([p.to_dict() for p in performance_history])
                
                # Backtest results
                latest_backtest = await self.backtest_repo.get_latest_backtest(strategy.id)
                if latest_backtest:
                    export_data["backtest_results"].append(latest_backtest.to_dict())
            
            # Selection history
            selection_history = await self.selection_repo.get_selection_history(
                start_date, end_date
            )
            export_data["selection_history"] = [s.to_dict() for s in selection_history]
            
            return export_data
            
        except Exception as e:
            self.logger.error(
                f"Failed to export strategy data: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
```

## Testing Strategy

1. **Unit Tests**
   - Repository CRUD operations
   - Model validation and constraints
   - Service method functionality
   - Error handling scenarios

2. **Integration Tests**
   - Database transactions
   - Repository relationships
   - Service-to-repository interactions
   - Performance under load

3. **Data Integrity Tests**
   - Foreign key constraints
   - JSON field validation
   - Concurrent access scenarios
   - Migration compatibility

## Monitoring & Observability

1. **Database Metrics**
   - Query performance
   - Connection pool utilization
   - Transaction success rates
   - Index effectiveness

2. **Data Quality Metrics**
   - Record consistency
   - Missing data detection
   - Outlier identification
   - Historical trend validation

## Security Considerations

1. **Data Protection**
   - Encrypt sensitive strategy parameters
   - Audit trail for all modifications
   - Role-based access controls
   - Data retention policies

2. **Query Security**
   - Parameterized queries
   - Input validation
   - SQL injection prevention
   - Rate limiting

## Future Enhancements

1. **Advanced Analytics**
   - Machine learning insights
   - Predictive performance modeling
   - Risk attribution analysis
   - Market regime detection

2. **Operational Features**
   - Real-time dashboards
   - Automated reporting
   - Alert systems
   - A/B testing framework