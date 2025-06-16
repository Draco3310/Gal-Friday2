"""Repository for risk metrics persistence."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from gal_friday.dal.models import RiskMetrics
from gal_friday.dal.repositories.base import BaseRepository


class RiskMetricsRepository(BaseRepository):
    """Repository for managing risk metrics data."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the RiskMetricsRepository."""
        super().__init__(RiskMetrics, session)
        self.logger = logging.getLogger(__name__)

    async def get_current_metrics(self) -> Optional[RiskMetrics]:
        """Get the most recent risk metrics record.
        
        Returns:
            The most recent RiskMetrics record or None if no records exist
        """
        try:
            stmt = (
                select(RiskMetrics)
                .order_by(RiskMetrics.last_updated.desc())
                .limit(1)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error fetching current risk metrics: {e}")
            raise

    async def update_metrics(self, metrics_data: Dict[str, Any]) -> RiskMetrics:
        """Update risk metrics or create new record if none exists.
        
        Args:
            metrics_data: Dictionary containing metric values to update
            
        Returns:
            Updated or newly created RiskMetrics record
        """
        try:
            # Get current metrics or create new
            current_metrics = await self.get_current_metrics()
            
            if current_metrics is None:
                # Create new metrics record
                current_metrics = RiskMetrics()
                self.session.add(current_metrics)
            
            # Update fields
            for field, value in metrics_data.items():
                if hasattr(current_metrics, field):
                    setattr(current_metrics, field, value)
            
            # Update timestamp
            current_metrics.last_updated = datetime.now(timezone.utc)
            
            await self.session.commit()
            await self.session.refresh(current_metrics)
            
            return current_metrics
            
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error updating risk metrics: {e}")
            raise

    async def reset_daily_metrics(self) -> None:
        """Reset daily metrics at the start of a new trading day."""
        try:
            current_metrics = await self.get_current_metrics()
            if current_metrics:
                current_metrics.daily_pnl = Decimal("0")
                current_metrics.daily_reset_at = datetime.now(timezone.utc)
                await self.session.commit()
                self.logger.info("Daily risk metrics reset completed")
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error resetting daily metrics: {e}")
            raise

    async def reset_weekly_metrics(self) -> None:
        """Reset weekly metrics at the start of a new trading week."""
        try:
            current_metrics = await self.get_current_metrics()
            if current_metrics:
                current_metrics.weekly_pnl = Decimal("0")
                current_metrics.weekly_reset_at = datetime.now(timezone.utc)
                await self.session.commit()
                self.logger.info("Weekly risk metrics reset completed")
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error resetting weekly metrics: {e}")
            raise

    async def reset_monthly_metrics(self) -> None:
        """Reset monthly metrics at the start of a new trading month."""
        try:
            current_metrics = await self.get_current_metrics()
            if current_metrics:
                current_metrics.monthly_pnl = Decimal("0")
                current_metrics.monthly_reset_at = datetime.now(timezone.utc)
                await self.session.commit()
                self.logger.info("Monthly risk metrics reset completed")
        except Exception as e:
            await self.session.rollback()
            self.logger.error(f"Error resetting monthly metrics: {e}")
            raise

    async def calculate_risk_score(self, metrics: RiskMetrics) -> Decimal:
        """Calculate composite risk score based on current metrics.
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            Risk score between 0-100 (higher = more risk)
        """
        risk_score = Decimal("0")
        
        # Consecutive losses component (0-30 points)
        if metrics.consecutive_losses > 0:
            loss_score = min(metrics.consecutive_losses * 5, 30)
            risk_score += Decimal(str(loss_score))
        
        # Drawdown component (0-40 points)
        if metrics.current_drawdown_pct > 0:
            drawdown_score = min(float(metrics.current_drawdown_pct) * 2, 40)
            risk_score += Decimal(str(drawdown_score))
        
        # Exposure component (0-30 points)
        # Assuming total_exposure is a percentage of capital
        if metrics.total_exposure > Decimal("100"):
            exposure_score = min((float(metrics.total_exposure) - 100) / 5, 30)
            risk_score += Decimal(str(exposure_score))
        
        return min(risk_score, Decimal("100"))

    async def update_risk_level(self, metrics: RiskMetrics) -> str:
        """Update risk level based on current metrics.
        
        Args:
            metrics: Current risk metrics
            
        Returns:
            Updated risk level (NORMAL, ELEVATED, CRITICAL)
        """
        risk_score = await self.calculate_risk_score(metrics)
        
        if risk_score >= Decimal("70"):
            risk_level = "CRITICAL"
        elif risk_score >= Decimal("40"):
            risk_level = "ELEVATED"
        else:
            risk_level = "NORMAL"
        
        metrics.risk_score = risk_score
        metrics.risk_level = risk_level
        
        return risk_level 