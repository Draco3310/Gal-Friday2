"""SQLAlchemy model for the 'risk_metrics' table."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID as PythonUUID

from sqlalchemy import DateTime, Index, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from .models_base import Base


class RiskMetrics(Base):
    """Represents the current risk state and metrics."""

    __tablename__ = "risk_metrics"

    id: Mapped[PythonUUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.uuid_generate_v4())
    
    # Risk counters
    consecutive_losses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    consecutive_wins: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Drawdown metrics
    current_drawdown: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    max_drawdown: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    current_drawdown_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=0)
    max_drawdown_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False, default=0)
    
    # P&L metrics
    daily_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    weekly_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    monthly_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    total_realized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    
    # Position metrics
    active_positions_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_exposure: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False, default=0)
    
    # Risk limit tracking
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False, default="NORMAL")  # NORMAL, ELEVATED, CRITICAL
    risk_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, default=0)  # 0-100
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp()
    )
    last_updated: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), 
        onupdate=func.current_timestamp(), index=True
    )
    
    # Period tracking
    daily_reset_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    weekly_reset_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    monthly_reset_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_risk_metrics_updated", "last_updated"),
        Index("idx_risk_metrics_risk_level", "risk_level"))

    def __repr__(self) -> str:
        return (
            f"<RiskMetrics(id={self.id}, consecutive_losses={self.consecutive_losses}, "
            f"current_drawdown_pct={self.current_drawdown_pct}, risk_level='{self.risk_level}')>"
        ) 