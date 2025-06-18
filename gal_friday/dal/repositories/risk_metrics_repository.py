"""Repository for risk metrics persistence."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gal_friday.dal.base import BaseRepository
from gal_friday.dal.models.risk_metrics import RiskMetrics

if TYPE_CHECKING:
    from gal_friday.logger_service import LoggerService


class RiskMetricsRepository(BaseRepository[RiskMetrics]):
    """Repository for managing risk metrics data."""

    def __init__(
        self, session_maker: async_sessionmaker[AsyncSession], logger: "LoggerService") -> None:
        """Initialize the risk metrics repository.

        Args:
            session_maker: SQLAlchemy async_sessionmaker for creating sessions.
            logger: Logger service instance.
        """
        super().__init__(session_maker, RiskMetrics, logger)

    async def get_current_metrics(self) -> RiskMetrics | None:
        """Get the most recent risk metrics record.

        Returns:
            The most recent RiskMetrics record or None if no records exist
        """
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(RiskMetrics)
                    .order_by(RiskMetrics.last_updated.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()
        except Exception:
            self.logger.exception(
                "Error fetching current risk metrics: ",
                source_module=self._source_module)
            raise

    async def update_metrics(self, metrics_data: dict[str, Any]) -> RiskMetrics:
        """Update risk metrics or create new record if none exists.

        Args:
            metrics_data: Dictionary containing metric values to update

        Returns:
            Updated or newly created RiskMetrics record
        """
        try:
            async with self.session_maker() as session:
                # Get current metrics or create new
                stmt = (
                    select(RiskMetrics)
                    .order_by(RiskMetrics.last_updated.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                current_metrics = result.scalar_one_or_none()

                if current_metrics is None:
                    # Create new metrics record
                    current_metrics = RiskMetrics()
                    session.add(current_metrics)

                # Update fields
                for field, value in metrics_data.items():
                    if hasattr(current_metrics, field):
                        setattr(current_metrics, field, value)

                # Update timestamp
                current_metrics.last_updated = datetime.now(UTC)

                await session.commit()
                await session.refresh(current_metrics)

                return current_metrics

        except Exception:
            self.logger.exception(
                "Error updating risk metrics: ",
                source_module=self._source_module)
            raise

    async def reset_daily_metrics(self) -> None:
        """Reset daily metrics at the start of a new trading day."""
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(RiskMetrics)
                    .order_by(RiskMetrics.last_updated.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                current_metrics = result.scalar_one_or_none()

                if current_metrics:
                    current_metrics.daily_pnl = Decimal(0)
                    current_metrics.daily_reset_at = datetime.now(UTC)
                    await session.commit()
                    self.logger.info(
                        "Daily risk metrics reset completed",
                        source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error resetting daily metrics: ",
                source_module=self._source_module)
            raise

    async def reset_weekly_metrics(self) -> None:
        """Reset weekly metrics at the start of a new trading week."""
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(RiskMetrics)
                    .order_by(RiskMetrics.last_updated.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                current_metrics = result.scalar_one_or_none()

                if current_metrics:
                    current_metrics.weekly_pnl = Decimal(0)
                    current_metrics.weekly_reset_at = datetime.now(UTC)
                    await session.commit()
                    self.logger.info(
                        "Weekly risk metrics reset completed",
                        source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error resetting weekly metrics: ",
                source_module=self._source_module)
            raise

    async def reset_monthly_metrics(self) -> None:
        """Reset monthly metrics at the start of a new trading month."""
        try:
            async with self.session_maker() as session:
                stmt = (
                    select(RiskMetrics)
                    .order_by(RiskMetrics.last_updated.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                current_metrics = result.scalar_one_or_none()

                if current_metrics:
                    current_metrics.monthly_pnl = Decimal(0)
                    current_metrics.monthly_reset_at = datetime.now(UTC)
                    await session.commit()
                    self.logger.info(
                        "Monthly risk metrics reset completed",
                        source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error resetting monthly metrics: ",
                source_module=self._source_module)
            raise

    async def calculate_risk_score(self, metrics: RiskMetrics) -> Decimal:
        """Calculate composite risk score based on current metrics.

        Args:
            metrics: Current risk metrics

        Returns:
            Risk score between 0-100 (higher = more risk)
        """
        risk_score = Decimal(0)

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
        if metrics.total_exposure > Decimal(100):
            exposure_score = min((float(metrics.total_exposure) - 100) / 5, 30)
            risk_score += Decimal(str(exposure_score))

        return min(risk_score, Decimal(100))

    async def update_risk_level(self, metrics: RiskMetrics) -> str:
        """Update risk level based on current metrics.

        Args:
            metrics: Current risk metrics

        Returns:
            Updated risk level (NORMAL, ELEVATED, CRITICAL)
        """
        risk_score = await self.calculate_risk_score(metrics)

        if risk_score >= Decimal(70):
            risk_level = "CRITICAL"
        elif risk_score >= Decimal(40):
            risk_level = "ELEVATED"
        else:
            risk_level = "NORMAL"

        metrics.risk_score = risk_score
        metrics.risk_level = risk_level

        return risk_level
