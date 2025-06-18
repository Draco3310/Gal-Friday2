#!/usr/bin/env python3
"""Quick test to verify dashboard functionality."""

from pathlib import Path
import sys
from typing import Any

import asyncio

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.monitoring.dashboard_service import DashboardService, RealTimeDashboard
from gal_friday.portfolio_manager import PortfolioManager


async def quick_test() -> None:
    """Quick test of dashboard functionality."""
    # Initialize services
    config = ConfigManager()
    # Create a basic logger for PubSubManager
    import logging
    basic_logger = logging.getLogger(__name__)
    pubsub = PubSubManager(basic_logger, config)
    logger = LoggerService(config, pubsub)

    # Create mock portfolio manager that satisfies the PortfolioManager interface
    class MockPortfolioManager(PortfolioManager):
        def __init__(self) -> None:
            # Initialize with minimal required arguments
            pass

        def get_current_state(self) -> dict[str, Any]:
            return {
                "total_equity": 1000000.0,
                "total_unrealized_pnl": 15000.0,
                "daily_pnl": 2500.0,
                "positions": {
                    "AAPL": {"quantity": 100, "unrealized_pnl": 1500.0},
                    "GOOGL": {"quantity": 50, "unrealized_pnl": 2500.0},
                    "MSFT": {"quantity": 75, "unrealized_pnl": -500.0},
                },
                "trades_today": [
                    {"pnl": 250.0}, {"pnl": -100.0}, {"pnl": 150.0},
                ],
                "total_trades": 156,
                "max_drawdown_pct": -8.5,
                "sharpe_ratio": 1.34,
            }

    portfolio_manager = MockPortfolioManager()
    dashboard_service = DashboardService(config, logger, portfolio_manager)

    # Test basic dashboard service
    await dashboard_service.get_all_metrics()

    # Test real-time dashboard
    dashboard_config = {"test_mode": True}
    real_time_dashboard = RealTimeDashboard(dashboard_service, dashboard_config)

    # Start dashboard
    await real_time_dashboard.start_dashboard()

    # Let it collect some data
    await asyncio.sleep(2)

    # Check collected data
    if real_time_dashboard.current_data:

        for _widget_type, _data in real_time_dashboard.current_data.items():
            pass

    # Stop dashboard
    await real_time_dashboard.stop_dashboard()




if __name__ == "__main__":
    asyncio.run(quick_test())
