"""Repository implementations for strategy selection and performance tracking.

This module provides repository classes for managing strategy configurations,
performance metrics, selection events, and backtest results.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, UTC
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, asc, func
from sqlalchemy.orm import selectinload

from ..base import BaseRepository
from ..models.strategy_models import (
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