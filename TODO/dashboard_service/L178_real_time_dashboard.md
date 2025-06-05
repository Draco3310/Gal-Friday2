# Task: Implement real-time dashboard with live trading metrics and portfolio visualization.

### 1. Context
- **File:** `gal_friday/dashboard_service.py`
- **Line:** `178`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing real-time dashboard with live trading metrics and portfolio visualization.

### 2. Problem Statement
Without a real-time dashboard, operators and traders cannot monitor system performance, portfolio status, or trading activities in real-time. This prevents effective system monitoring, decision-making, and operational oversight of the trading system.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Dashboard Framework:** Web-based real-time dashboard infrastructure
2. **Build Live Data Streaming:** WebSocket-based real-time data updates
3. **Implement Portfolio Visualization:** Interactive charts and portfolio metrics
4. **Add Trading Metrics Display:** Real-time trading performance and statistics
5. **Create Alert Management:** Visual alert system with notification handling
6. **Build Customizable Views:** User-configurable dashboard layouts and widgets

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, WebSocket

class WidgetType(str, Enum):
    """Dashboard widget types"""
    PORTFOLIO_VALUE = "portfolio_value"
    POSITION_TABLE = "position_table"
    PRICE_CHART = "price_chart"
    TRADING_METRICS = "trading_metrics"
    ALERT_PANEL = "alert_panel"

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)

class RealTimeDashboard:
    """Enterprise-grade real-time trading dashboard"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # FastAPI app for dashboard
        self.app = FastAPI(title="Gal Friday Trading Dashboard")
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Dashboard state
        self.current_data: Dict[str, Any] = {}
        
        # Data update tasks
        self.update_tasks: List[asyncio.Task] = []
        
        self._setup_routes()
    
    async def start_dashboard(self) -> None:
        """
        Start real-time dashboard service
        Replace TODO with comprehensive dashboard system
        """
        
        try:
            self.logger.info("Starting real-time trading dashboard")
            
            # Start data update tasks
            await self._start_data_updates()
            
            self.logger.info("Dashboard service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            raise DashboardError(f"Dashboard start failed: {e}")
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes for dashboard"""
        
        @self.app.get("/")
        async def dashboard_home():
            """Main dashboard page"""
            return {"message": "Gal Friday Trading Dashboard"}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket)
        
        @self.app.get("/api/data/{widget_type}")
        async def get_widget_data(widget_type: str):
            """Get current data for specific widget type"""
            return self.current_data.get(widget_type, {})
    
    async def _handle_websocket_connection(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection for real-time updates"""
        
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            # Send initial data
            await self._send_initial_data(websocket)
            
            # Keep connection alive
            while True:
                try:
                    message = await websocket.receive_text()
                    await self._handle_client_message(websocket, message)
                except Exception as e:
                    self.logger.warning(f"WebSocket message error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _send_initial_data(self, websocket: WebSocket) -> None:
        """Send initial dashboard data to new connection"""
        
        initial_data = {
            "type": "initial_data",
            "data": self.current_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send_text(json.dumps(initial_data))
    
    async def _start_data_updates(self) -> None:
        """Start background tasks for data updates"""
        
        # Portfolio data update task
        portfolio_task = asyncio.create_task(self._update_portfolio_data())
        self.update_tasks.append(portfolio_task)
        
        # Trading metrics update task
        metrics_task = asyncio.create_task(self._update_trading_metrics())
        self.update_tasks.append(metrics_task)
    
    async def _update_portfolio_data(self) -> None:
        """Update portfolio data periodically"""
        
        while True:
            try:
                # Get current portfolio data (placeholder)
                portfolio_data = {
                    "total_value": 1000000.0,
                    "cash_balance": 250000.0,
                    "positions_value": 750000.0,
                    "daily_pnl": 15000.0,
                    "positions": [
                        {"symbol": "AAPL", "quantity": 100, "value": 15000.0},
                        {"symbol": "GOOGL", "quantity": 50, "value": 125000.0}
                    ]
                }
                
                self.current_data[WidgetType.PORTFOLIO_VALUE] = portfolio_data
                
                # Broadcast to all connected clients
                await self._broadcast_update(WidgetType.PORTFOLIO_VALUE, portfolio_data)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating portfolio data: {e}")
                await asyncio.sleep(10)
    
    async def _update_trading_metrics(self) -> None:
        """Update trading metrics periodically"""
        
        while True:
            try:
                # Get current trading metrics (placeholder)
                metrics_data = {
                    "total_trades": 156,
                    "winning_trades": 89,
                    "win_rate": 57.1,
                    "sharpe_ratio": 1.23,
                    "max_drawdown": -5.2
                }
                
                self.current_data[WidgetType.TRADING_METRICS] = metrics_data
                
                # Broadcast to all connected clients
                await self._broadcast_update(WidgetType.TRADING_METRICS, metrics_data)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating trading metrics: {e}")
                await asyncio.sleep(15)
    
    async def _broadcast_update(self, widget_type: WidgetType, data: Dict[str, Any]) -> None:
        """Broadcast data update to all connected clients"""
        
        if not self.active_connections:
            return
        
        message = {
            "type": "data_update",
            "widget_type": widget_type.value,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        message_json = json.dumps(message)
        
        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                self.logger.warning(f"Failed to send update to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)

class DashboardError(Exception):
    """Exception raised for dashboard errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Robust WebSocket connection management; graceful degradation during data outages; comprehensive error logging
- **Configuration:** Configurable widget types; customizable refresh intervals; flexible layout management
- **Testing:** Unit tests for data updates; integration tests with WebSocket connections; performance tests for multiple clients
- **Dependencies:** FastAPI for web framework; WebSocket support; JavaScript charting libraries

### 4. Acceptance Criteria
- [ ] Real-time dashboard provides live trading metrics and portfolio visualization
- [ ] WebSocket-based data streaming delivers updates without page refresh
- [ ] Portfolio visualization shows current positions, P&L, and performance metrics
- [ ] Trading metrics display includes win rate, Sharpe ratio, and drawdown analysis
- [ ] Alert management system provides visual notifications and alert history
- [ ] Customizable dashboard layouts allow users to configure widget placement
- [ ] Performance optimization handles multiple concurrent dashboard users
- [ ] Responsive design works across desktop and mobile devices
- [ ] Data security ensures proper authentication and authorization
- [ ] TODO placeholder is completely replaced with production-ready implementation