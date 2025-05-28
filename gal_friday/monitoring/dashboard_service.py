"""Dashboard service providing aggregated system metrics."""

from datetime import UTC, datetime
from typing import Any

import psutil

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService


class DashboardService:
    """Service for aggregating and providing dashboard metrics."""

    def __init__(self, config: ConfigManager, logger: LoggerService) -> None:
        """Initialize the DashboardService.

        Args:
            config: The configuration manager instance.
            logger: The logger service instance.
        """
        self.config = config
        self.logger = logger
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
        """Get portfolio metrics."""
        # Placeholder metrics - in real implementation would connect to portfolio manager
        return {
            "total_pnl": 1250.75,
            "daily_pnl": 45.20,
            "win_rate": 0.68,
            "total_trades": 127,
            "active_positions": 3,
            "positions": [
                {"symbol": "XRP/USD", "size": 1000, "pnl": 23.45},
                {"symbol": "BTC/USD", "size": 0.1, "pnl": -5.67},
                {"symbol": "ETH/USD", "size": 2.5, "pnl": 12.89},
            ],
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.85,
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
