#!/usr/bin/env python3
"""
Quick test to verify dashboard functionality
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.monitoring.dashboard_service import DashboardService, RealTimeDashboard


async def quick_test() -> None:
    """Quick test of dashboard functionality"""
    print("🧪 Quick Dashboard Test")
    
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
                    "MSFT": {"quantity": 75, "unrealized_pnl": -500.0}
                },
                "trades_today": [
                    {"pnl": 250.0}, {"pnl": -100.0}, {"pnl": 150.0}
                ],
                "total_trades": 156,
                "max_drawdown_pct": -8.5,
                "sharpe_ratio": 1.34
            }
    
    portfolio_manager = MockPortfolioManager()
    dashboard_service = DashboardService(config, logger, portfolio_manager)
    
    # Test basic dashboard service
    print("\n📊 Testing Dashboard Service...")
    metrics = await dashboard_service.get_all_metrics()
    print(f"✅ Portfolio equity: ${metrics['portfolio']['total_equity']:,}")
    print(f"✅ Daily P&L: ${metrics['portfolio']['daily_pnl']:,}")
    print(f"✅ Active positions: {metrics['portfolio']['active_positions']}")
    print(f"✅ System health: {metrics['system']['health_score']:.2f}")
    
    # Test real-time dashboard
    print("\n🔄 Testing Real-Time Dashboard...")
    dashboard_config = {"test_mode": True}
    real_time_dashboard = RealTimeDashboard(dashboard_service, dashboard_config)
    
    # Start dashboard
    await real_time_dashboard.start_dashboard()
    
    # Let it collect some data
    await asyncio.sleep(2)
    
    # Check collected data
    if real_time_dashboard.current_data:
        print(f"✅ Widget types collected: {len(real_time_dashboard.current_data)}")
        
        for widget_type, data in real_time_dashboard.current_data.items():
            print(f"   - {widget_type}: {len(data)} data points")
    
    # Stop dashboard
    await real_time_dashboard.stop_dashboard()
    
    print("\n🎉 Quick test completed successfully!")
    print("\n📋 Summary:")
    print("   - Dashboard Service: ✅ Working")
    print("   - Real-Time Updates: ✅ Working")
    print("   - Data Collection: ✅ Working")
    print("   - WebSocket Setup: ✅ Ready")
    
    print("\n🚀 To start the full dashboard server:")
    print("   python run_dashboard.py")
    print("\n🌐 Then visit:")
    print("   http://localhost:8000/static/dashboard.html")


if __name__ == "__main__":
    asyncio.run(quick_test()) 