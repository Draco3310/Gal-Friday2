#!/usr/bin/env python3
"""
Test script for the Real-Time Trading Dashboard
"""

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import websockets

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gal_friday.cli_service_mocks import ConfigManager, LoggerService
from gal_friday.monitoring.dashboard_service import DashboardService, RealTimeDashboard


async def test_dashboard_service():
    """Test the basic dashboard service functionality"""
    print("🧪 Testing Dashboard Service...")
    
    try:
        # Initialize configuration and logger
        config = ConfigManager()
        logger = LoggerService()
        
        # Create mock portfolio manager
        class MockPortfolioManager:
            def get_current_state(self):
                return {
                    "total_equity": 500000.0,
                    "total_unrealized_pnl": 7500.0,
                    "daily_pnl": 1250.0,
                    "positions": {
                        "AAPL": {"quantity": 50, "unrealized_pnl": 750.0},
                        "GOOGL": {"quantity": 25, "unrealized_pnl": 1250.0}
                    },
                    "trades_today": [{"pnl": 100.0}, {"pnl": -50.0}],
                    "total_trades": 78,
                    "max_drawdown_pct": -5.2,
                    "sharpe_ratio": 1.15
                }
        
        portfolio_manager = MockPortfolioManager()
        dashboard_service = DashboardService(config, logger, portfolio_manager)
        
        # Test getting all metrics
        metrics = await dashboard_service.get_all_metrics()
        
        print("✅ Basic dashboard service test passed")
        print(f"   - System metrics: {len(metrics['system'])} items")
        print(f"   - Portfolio metrics: {len(metrics['portfolio'])} items")
        print(f"   - Portfolio equity: ${metrics['portfolio']['total_equity']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard service test failed: {e}")
        return False


async def test_real_time_dashboard():
    """Test the real-time dashboard functionality"""
    print("🧪 Testing Real-Time Dashboard...")
    
    try:
        # Initialize services
        config = ConfigManager()
        logger = LoggerService()
        
        class MockPortfolioManager:
            def get_current_state(self):
                return {
                    "total_equity": 750000.0,
                    "total_unrealized_pnl": 12500.0,
                    "daily_pnl": 1875.0,
                    "positions": {
                        "AAPL": {"quantity": 75, "unrealized_pnl": 1125.0},
                        "GOOGL": {"quantity": 35, "unrealized_pnl": 1750.0},
                        "MSFT": {"quantity": 60, "unrealized_pnl": -375.0}
                    },
                    "trades_today": [
                        {"pnl": 150.0}, {"pnl": -75.0}, {"pnl": 200.0}
                    ],
                    "total_trades": 134,
                    "max_drawdown_pct": -6.8,
                    "sharpe_ratio": 1.28
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
            print("✅ Real-time dashboard test passed")
            print(f"   - Widget types: {list(real_time_dashboard.current_data.keys())}")
            
            # Check portfolio data
            if 'portfolio_value' in real_time_dashboard.current_data:
                portfolio_data = real_time_dashboard.current_data['portfolio_value']
                print(f"   - Portfolio equity: ${portfolio_data.get('total_equity', 0):,}")
                print(f"   - Daily P&L: ${portfolio_data.get('daily_pnl', 0):,}")
            
            # Check trading metrics
            if 'trading_metrics' in real_time_dashboard.current_data:
                metrics_data = real_time_dashboard.current_data['trading_metrics']
                print(f"   - Total trades: {metrics_data.get('total_trades', 0)}")
                print(f"   - Win rate: {metrics_data.get('win_rate', 0):.1f}%")
        
        # Stop the dashboard
        await real_time_dashboard.stop_dashboard()
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time dashboard test failed: {e}")
        return False


async def test_websocket_connection(port=8000):
    """Test WebSocket connection to a running dashboard"""
    print("🧪 Testing WebSocket Connection...")
    
    try:
        uri = f"ws://localhost:{port}/ws"
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established")
            
            # Send a ping
            ping_message = json.dumps({"type": "ping"})
            await websocket.send(ping_message)
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            if response_data.get("type") == "initial_data":
                print("✅ Received initial data")
                print(f"   - Data keys: {list(response_data.get('data', {}).keys())}")
            elif response_data.get("type") == "pong":
                print("✅ Ping-pong successful")
            
            return True
            
    except asyncio.TimeoutError:
        print("❌ WebSocket test timed out")
        return False
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False


async def test_http_endpoints(port=8000):
    """Test HTTP endpoints of a running dashboard"""
    print("🧪 Testing HTTP Endpoints...")
    
    try:
        base_url = f"http://localhost:{port}"
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print("✅ Health endpoint working")
                    print(f"   - Status: {health_data.get('status')}")
                
            # Test main endpoint
            async with session.get(f"{base_url}/") as response:
                if response.status == 200:
                    main_data = await response.json()
                    print("✅ Main endpoint working")
                    print(f"   - Message: {main_data.get('message')}")
                
            # Test widgets endpoint
            async with session.get(f"{base_url}/api/widgets") as response:
                if response.status == 200:
                    widgets_data = await response.json()
                    print("✅ Widgets endpoint working")
                    print(f"   - Connections: {widgets_data.get('connections', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ HTTP endpoints test failed: {e}")
        return False


async def run_all_tests():
    """Run all dashboard tests"""
    print("🚀 Starting Dashboard Tests...\n")
    
    tests = [
        ("Dashboard Service", test_dashboard_service()),
        ("Real-Time Dashboard", test_real_time_dashboard()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Dashboard is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(results)


async def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Test against running server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        print(f"Testing against running server on port {port}...")
        
        websocket_result = await test_websocket_connection(port)
        http_result = await test_http_endpoints(port)
        
        if websocket_result and http_result:
            print("🎉 Server tests passed!")
        else:
            print("⚠️  Server tests failed!")
    else:
        # Run unit tests
        await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main()) 