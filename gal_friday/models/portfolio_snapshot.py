from datetime import datetime

from sqlalchemy import Column, Integer, Numeric, Float, DateTime, JSON
from sqlalchemy.sql import func

from .base import Base


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    snapshot_pk = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_timestamp = Column(DateTime(timezone=True), unique=True, nullable=False, server_default=func.now(), index=True)
    total_equity = Column(Numeric(18, 8), nullable=False)
    available_balance = Column(Numeric(18, 8), nullable=False)
    total_exposure_pct = Column(Float, nullable=False)
    daily_drawdown_pct = Column(Float, nullable=False)
    weekly_drawdown_pct = Column(Float, nullable=False)
    total_drawdown_pct = Column(Float, nullable=False)
    positions = Column(JSON, nullable=False)  # JSON object detailing current positions

    def __repr__(self):
        return (
            f"<PortfolioSnapshot(snapshot_pk={self.snapshot_pk}, "
            f"snapshot_timestamp='{self.snapshot_timestamp}', total_equity={self.total_equity})>"
        )
