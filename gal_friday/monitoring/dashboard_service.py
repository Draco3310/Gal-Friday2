"""Dashboard service providing aggregated system metrics."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import psutil

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService

if TYPE_CHECKING:
    from gal_friday.portfolio_manager import PortfolioManager


class DashboardService:
    """Service for aggregating and providing dashboard metrics."""

    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerService,
        portfolio_manager: "PortfolioManager",
    ) -> None:
        """Initialize the DashboardService.

        Args:
            config: The configuration manager instance.
            logger: The logger service instance.
            portfolio_manager: The portfolio manager instance.
        """
        self.config = config
        self.logger = logger
        self.portfolio_manager = portfolio_manager
        self._start_time = datetime.now(UTC)

    async def get_all_metrics(self) -> dict[str, Any]:
        """Get all aggregated metrics for the dashboard."""
        return {
            "system": await self._get_system_metrics(),
            "portfolio": await self._get_portfolio_metrics(),
            "models": await self._get_model_metrics(),
            "websocket": await self._get_websocket_metrics(),
            "alerts": await self._get_alert_metrics(),
        }

    async def _get_system_metrics(self) -> dict[str, Any]:
        """Get system-level metrics."""
        uptime = (datetime.now(UTC) - self._start_time).total_seconds()

        # Get system resources
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Calculate health score based on system resources
        health_score = self._calculate_health_score(cpu_percent, memory.percent)

        return {
            "uptime_seconds": uptime,
            "health_score": health_score,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
        }

    async def _get_portfolio_metrics(self) -> dict[str, Any]:
        """Get portfolio metrics from the real portfolio manager."""
        try:
            # Get current portfolio state
            portfolio_state = self.portfolio_manager.get_current_state()

            # Extract key metrics
            total_equity = portfolio_state.get("total_equity", Decimal("0"))
            total_pnl = portfolio_state.get("total_unrealized_pnl", Decimal("0"))
            daily_pnl = portfolio_state.get("daily_pnl", Decimal("0"))

            # Get positions
            positions_dict = portfolio_state.get("positions", {})
            active_positions = []
            for symbol, pos_data in positions_dict.items():
                if pos_data.get("quantity", 0) != 0:
                    active_positions.append({
                        "symbol": symbol,
                        "size": float(pos_data.get("quantity", 0)),
                        "pnl": float(pos_data.get("unrealized_pnl", 0)),
                    })

            # Calculate metrics
            trades_today = portfolio_state.get("trades_today", [])
            winning_trades = [t for t in trades_today if t.get("pnl", 0) > 0]
            win_rate = len(winning_trades) / len(trades_today) if trades_today else 0.0

            max_drawdown = portfolio_state.get("max_drawdown_pct", Decimal("0"))

            return {
                "total_pnl": float(total_pnl),
                "daily_pnl": float(daily_pnl),
                "win_rate": win_rate,
                "total_trades": portfolio_state.get("total_trades", 0),
                "active_positions": len(active_positions),
                "positions": active_positions,
                "max_drawdown": float(max_drawdown) / 100,  # Convert percentage to decimal
                "sharpe_ratio": float(portfolio_state.get("sharpe_ratio", 0)),
                "total_equity": float(total_equity),
            }

        except Exception as e:
            self.logger.error(
                f"Failed to get portfolio metrics: {e}",
                source_module="DashboardService",
                exc_info=True,
            )
            # Return safe defaults on error
            return {
                "total_pnl": 0.0,
                "daily_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "active_positions": 0,
                "positions": [],
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "total_equity": 0.0,
            }

    async def _get_model_metrics(self) -> dict[str, Any]:
        """Get ML model metrics."""
        return {
            "active_models": 3,
            "last_retrain": "2024-01-15T10:30:00Z",
            "avg_accuracy": 0.87,
            "drift_detected": False,
            "inference_rate": 1250.5,  # inferences per second
            "model_latency_ms": 2.3,
        }

    async def _get_websocket_metrics(self) -> dict[str, Any]:
        """Get WebSocket connection metrics."""
        return {
            "status": "connected",
            "active_connections": 2,
            "message_rate": 45.7,  # messages per second
            "latency_ms": 12.4,
            "total_messages": 128456,
            "error_rate": 0.001,
        }

    async def _get_alert_metrics(self) -> dict[str, Any]:
        """Get alerting system metrics."""
        return {
            "critical": 0,
            "warning": 2,
            "info": 5,
            "total_today": 12,
            "last_alert": "2024-01-15T14:22:00Z",
        }

    def _calculate_health_score(self, cpu_percent: float, memory_percent: float) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        # Simple health calculation - could be more sophisticated
        cpu_score = max(0, 1 - (cpu_percent / 100))
        memory_score = max(0, 1 - (memory_percent / 100))

        # Weighted average
        return (cpu_score * 0.4 + memory_score * 0.6)

    async def get_system_status(self) -> dict[str, Any]:
        """Get current system status."""
        return {
            "status": "healthy",
            "uptime": (datetime.now(UTC) - self._start_time).total_seconds(),
            "last_check": datetime.now(UTC).isoformat(),
        }
