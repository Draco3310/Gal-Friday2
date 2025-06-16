"""Database models for strategy selection and performance tracking.

This module defines SQLAlchemy models for persisting strategy configurations,
performance metrics, selection decisions, and backtest results.
"""

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

from .base import Base
from typing import Any


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
        Index("idx_strategy_configs_updated_at", "updated_at"))
    
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
        Index("idx_perf_snapshots_return", "total_return"))
    
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
        Index("idx_selection_events_strategy", "selected_strategy_id"))
    
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
        Index("idx_backtest_results_status", "validation_status"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "strategy_config_id": str(self.strategy_config_id),
            "backtest_start_date": self.backtest_start_date.isoformat(),
            "backtest_end_date": self.backtest_end_date.isoformat(),
            "initial_capital": float(self.initial_capital),
            "final_value": float(self.final_value),
            "total_return": float(self.total_return),
            "max_drawdown": float(self.max_drawdown),
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else None,
            "total_trades": self.total_trades,
            "win_rate": float(self.win_rate),
            "detailed_metrics": self.detailed_metrics,
            "trade_history": self.trade_history,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by
        }