#!/usr/bin/env python3
"""Test script for the Real-Time Trading Dashboard."""

import json
import logging
from pathlib import Path
import sys

import aiohttp
import asyncio
import websockets

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Any

from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.monitoring.dashboard_service import DashboardService, RealTimeDashboard
from gal_friday.portfolio_manager import PortfolioManager


async def test_dashboard_service() -> bool:
    """Test the basic dashboard service functionality."""
    try:
        # Initialize configuration and logger
        config = ConfigManager()
        # Create a basic logger for PubSubManager
        basic_logger = logging.getLogger(__name__)
        pubsub = PubSubManager(basic_logger, config)
        logger = LoggerService(config, pubsub)

        # Create mock portfolio manager
        class MockPortfolioManager(PortfolioManager):
            def __init__(self) -> None:
                pass

            def get_current_state(self) -> dict[str, Any]:
                return {
                    "total_equity": 500000.0,
                    "total_unrealized_pnl": 7500.0,
                    "daily_pnl": 1250.0,
                    "positions": {
                        "AAPL": {"quantity": 50, "unrealized_pnl": 750.0},
                        "GOOGL": {"quantity": 25, "unrealized_pnl": 1250.0},
                    },
                    "trades_today": [{"pnl": 100.0}, {"pnl": -50.0}],
                    "total_trades": 78,
                    "max_drawdown_pct": -5.2,
                    "sharpe_ratio": 1.15,
                }

        portfolio_manager = MockPortfolioManager()
        dashboard_service = DashboardService(config, logger, portfolio_manager)

        # Test getting all metrics
        await dashboard_service.get_all_metrics()


        return True

    except Exception:
        return False


async def test_real_time_dashboard() -> bool:
    """Test the real-time dashboard functionality."""
    try:
        # Initialize services
        config = ConfigManager()
        # Create a basic logger for PubSubManager
        basic_logger = logging.getLogger(__name__)
        pubsub = PubSubManager(basic_logger, config)
        logger = LoggerService(config, pubsub)

        class MockPortfolioManager(PortfolioManager):
            def __init__(self) -> None:
                pass

            def get_current_state(self) -> dict[str, Any]:
                return {
                    "total_equity": 750000.0,
                    "total_unrealized_pnl": 12500.0,
                    "daily_pnl": 1875.0,
                    "positions": {
                        "AAPL": {"quantity": 75, "unrealized_pnl": 1125.0},
                        "GOOGL": {"quantity": 35, "unrealized_pnl": 1750.0},
                        "MSFT": {"quantity": 60, "unrealized_pnl": -375.0},
                    },
                    "trades_today": [
                        {"pnl": 150.0}, {"pnl": -75.0}, {"pnl": 200.0},
                    ],
                    "total_trades": 134,
                    "max_drawdown_pct": -6.8,
                    "sharpe_ratio": 1.28,
                }

        portfolio_manager = MockPortfolioManager()
        dashboard_service = DashboardService(config, logger, portfolio_manager)

        dashboard_config = {"test_mode": True}
        real_time_dashboard = RealTimeDashboard(dashboard_service, dashboard_config)

        # Test starting the dashboard
        await real_time_dashboard.start_dashboard()

        # Let it run for a few seconds to collect data
        await asyncio.sleep(3)

        # Check if data was collected
        if real_time_dashboard.current_data:

            # Check portfolio data
            if "portfolio_value" in real_time_dashboard.current_data:
                real_time_dashboard.current_data["portfolio_value"]

            # Check trading metrics
            if "trading_metrics" in real_time_dashboard.current_data:
                real_time_dashboard.current_data["trading_metrics"]

        # Stop the dashboard
        await real_time_dashboard.stop_dashboard()

        return True

    except Exception:
        return False


async def test_websocket_connection(port: int = 8000) -> bool:
    """Test WebSocket connection to a running dashboard."""
    try:
        uri = f"ws://localhost:{port}/ws"

        async with websockets.connect(uri) as websocket:

            # Send a ping
            ping_message = json.dumps({"type": "ping"})
            await websocket.send(ping_message)

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)

            if response_data.get("type") == "initial_data" or response_data.get("type") == "pong":
                pass

            return True

    except TimeoutError:
        return False
    except Exception:
        return False


async def test_http_endpoints(port: int = 8000) -> bool:
    """Test HTTP endpoints of a running dashboard."""
    try:
        base_url = f"http://localhost:{port}"

        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    await response.json()

            # Test main endpoint
            async with session.get(f"{base_url}/") as response:
                if response.status == 200:
                    await response.json()

            # Test widgets endpoint
            async with session.get(f"{base_url}/api/widgets") as response:
                if response.status == 200:
                    await response.json()

        return True

    except Exception:
        return False


async def run_all_tests() -> bool:
    """Run all dashboard tests."""
    tests = [
        ("Dashboard Service", test_dashboard_service()),
        ("Real-Time Dashboard", test_real_time_dashboard()),
    ]

    results = []
    for test_name, test_coro in tests:

        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception:
            results.append((test_name, False))

    # Print summary

    passed = 0
    for test_name, result in results:
        if result:
            passed += 1


    if passed == len(results):
        pass
    else:
        pass

    return passed == len(results)


async def main() -> None:
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Test against running server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

        websocket_result = await test_websocket_connection(port)
        http_result = await test_http_endpoints(port)

        if websocket_result and http_result:
            pass
        else:
            pass
    else:
        # Run unit tests
        await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
