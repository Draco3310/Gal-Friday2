#!/usr/bin/env python3
"""
Script to run the Gal Friday Real-Time Trading Dashboard
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi.staticfiles import StaticFiles

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gal_friday.cli_service_mocks import ConfigManager, LoggerService
from gal_friday.monitoring.dashboard_service import DashboardService, RealTimeDashboard
from typing import Any


async def create_dashboard_app() -> RealTimeDashboard:
    """Create and configure the dashboard application"""
    
    # Initialize configuration
    config = ConfigManager()
    
    # Initialize logger
    logger = LoggerService()
    
    # Create a mock portfolio manager for demonstration
    # In production, this would be the actual portfolio manager
    class MockPortfolioManager:
        def get_current_state(self):
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
    
    # Initialize services
    portfolio_manager = MockPortfolioManager()
    dashboard_service = DashboardService(config, logger, portfolio_manager)
    
    # Dashboard configuration
    dashboard_config = {
        "update_intervals": {
            "portfolio": 5,
            "metrics": 10,
            "system": 15,
            "alerts": 30
        },
        "max_connections": 100,
        "heartbeat_interval": 30
    }
    
    # Create real-time dashboard
    real_time_dashboard = RealTimeDashboard(dashboard_service, dashboard_config)
    
    # Mount static files to serve the HTML dashboard
    static_dir = Path(__file__).parent
    real_time_dashboard.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Add route to serve the main dashboard page
    @real_time_dashboard.app.get("/dashboard")
    async def serve_dashboard():
        """Serve the main dashboard HTML page"""
        dashboard_path = static_dir / "dashboard.html"
        with open(dashboard_path, 'r') as f:
            return Response(content=f.read(), media_type="text/html")
    
    # Import Response here to avoid circular import
    from fastapi import Response
    
    return real_time_dashboard


async def main():
    """Main function to run the dashboard"""
    
    print("🚀 Starting Gal Friday Real-Time Trading Dashboard...")
    
    try:
        # Create dashboard app
        dashboard = await create_dashboard_app()
        
        # Start the dashboard background tasks
        await dashboard.start_dashboard()
        
        # Configuration for the server
        host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
        port = int(os.getenv("DASHBOARD_PORT", "8000"))
        
        print(f"📊 Dashboard will be available at:")
        print(f"   - Main API: http://{host}:{port}")
        print(f"   - Dashboard UI: http://{host}:{port}/static/dashboard.html")
        print(f"   - WebSocket: ws://{host}:{port}/ws")
        print(f"   - Health Check: http://{host}:{port}/health")
        
        # Run the server
        config = uvicorn.Config(
            dashboard.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard shutdown requested...")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        raise
    finally:
        if 'dashboard' in locals():
            await dashboard.stop_dashboard()
        print("✅ Dashboard stopped successfully")


if __name__ == "__main__":
    # Run the dashboard
    asyncio.run(main()) 