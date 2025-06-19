from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, Integer, Numeric
from sqlalchemy.sql import func

from .base import Base


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    snapshot_pk = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_timestamp = Column(
        DateTime(timezone=True), unique=True, nullable=False,
        server_default=func.now(), index=True,
    )
    total_equity = Column(Numeric(18, 8), nullable=False)
    available_balance = Column(Numeric(18, 8), nullable=False)
    total_exposure_pct = Column(Float, nullable=False)
    daily_drawdown_pct = Column(Float, nullable=False)
    weekly_drawdown_pct = Column(Float, nullable=False)
    total_drawdown_pct = Column(Float, nullable=False)
    positions = Column(JSON, nullable=False)  # JSON object detailing current positions

    def __repr__(self) -> str: # Added -> str
        return (
            f"<PortfolioSnapshot(snapshot_pk={self.snapshot_pk}, "
            f"snapshot_timestamp='{self.snapshot_timestamp}', total_equity={self.total_equity})>"
        )

    def to_dict(self) -> dict[str, Any]: # Added to_dict method with type hints
        """Converts the PortfolioSnapshot object to a dictionary."""
        return {
            "snapshot_pk": self.snapshot_pk,
            "snapshot_timestamp": self.snapshot_timestamp.isoformat() if self.snapshot_timestamp else None,
            "total_equity": float(self.total_equity) if self.total_equity is not None else None,
            "available_balance": float(self.available_balance) if self.available_balance is not None else None,
            "total_exposure_pct": self.total_exposure_pct,
            "daily_drawdown_pct": self.daily_drawdown_pct,
            "weekly_drawdown_pct": self.weekly_drawdown_pct,
            "total_drawdown_pct": self.total_drawdown_pct,
            "positions": self.positions, # Assuming positions is already a dict[str, Any] or JSON-serializable
        }
