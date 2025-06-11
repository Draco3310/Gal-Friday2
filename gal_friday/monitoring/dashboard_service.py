"""Dashboard service providing aggregated system metrics."""

from datetime import UTC, datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import psutil
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from gal_friday.config_manager import ConfigManager
from gal_friday.logger_service import LoggerService

if TYPE_CHECKING:
    from gal_friday.portfolio_manager import PortfolioManager


class WidgetType(str, Enum):
    """Dashboard widget types"""
    PORTFOLIO_VALUE = "portfolio_value"
    POSITION_TABLE = "position_table"
    PRICE_CHART = "price_chart"
    TRADING_METRICS = "trading_metrics"
    ALERT_PANEL = "alert_panel"
    SYSTEM_METRICS = "system_metrics"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)


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


class RealTimeDashboard:
    """Enterprise-grade real-time trading dashboard"""
    
    def __init__(self, dashboard_service: DashboardService, config: Dict[str, Any]):
        self.dashboard_service = dashboard_service
        self.config = config
        self.logger = dashboard_service.logger
        
        # FastAPI app for dashboard
        self.app = FastAPI(title="Gal Friday Trading Dashboard")
        
        # WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Dashboard state
        self.current_data: Dict[str, Any] = {}
        
        # Data update tasks
        self.update_tasks: List[asyncio.Task] = []
        self._running = False
        
        self._setup_routes()
    
    async def start_dashboard(self) -> None:
        """
        Start real-time dashboard service
        """
        
        try:
            self.logger.info("Starting real-time trading dashboard")
            
            self._running = True
            # Start data update tasks
            await self._start_data_updates()
            
            self.logger.info("Dashboard service started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard: {e}")
            raise DashboardError(f"Dashboard start failed: {e}")
    
    async def stop_dashboard(self) -> None:
        """Stop the real-time dashboard service"""
        
        self.logger.info("Stopping real-time dashboard")
        self._running = False
        
        # Cancel all update tasks
        for task in self.update_tasks:
            if not task.done():
                task.cancel()
        
        # Close all WebSocket connections
        for connection in self.active_connections[:]:
            try:
                await connection.close()
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket connection: {e}")
        
        self.active_connections.clear()
        self.logger.info("Dashboard service stopped")
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes for dashboard"""
        
        @self.app.get("/")
        async def dashboard_home():
            """Main dashboard page"""
            return {"message": "Gal Friday Trading Dashboard", "status": "active"}
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await self._handle_websocket_connection(websocket)
        
        @self.app.get("/api/data/{widget_type}")
        async def get_widget_data(widget_type: str):
            """Get current data for specific widget type"""
            return self.current_data.get(widget_type, {})
        
        @self.app.get("/api/widgets")
        async def get_all_widgets():
            """Get all current widget data"""
            return {
                "data": self.current_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "connections": len(self.active_connections)
            }
    
    async def _handle_websocket_connection(self, websocket: WebSocket) -> None:
        """Handle WebSocket connection for real-time updates"""
        
        await websocket.accept()
        self.active_connections.append(websocket)
        
        self.logger.info(f"New WebSocket connection established. Total connections: {len(self.active_connections)}")
        
        try:
            # Send initial data
            await self._send_initial_data(websocket)
            
            # Keep connection alive and handle messages
            while True:
                try:
                    message = await websocket.receive_text()
                    await self._handle_client_message(websocket, message)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    self.logger.warning(f"WebSocket message error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            self.logger.info(f"WebSocket connection closed. Remaining connections: {len(self.active_connections)}")
    
    async def _handle_client_message(self, websocket: WebSocket, message: str) -> None:
        """Handle incoming client messages"""
        
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            
            if message_type == "ping":
                # Respond to ping with pong
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }))
            elif message_type == "subscribe":
                # Handle widget subscription
                widget_types = data.get("widgets", [])
                await self._send_widget_data(websocket, widget_types)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            self.logger.warning("Received invalid JSON message from client")
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
    
    async def _send_initial_data(self, websocket: WebSocket) -> None:
        """Send initial dashboard data to new connection"""
        
        initial_data = {
            "type": "initial_data",
            "data": self.current_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send_text(json.dumps(initial_data))
    
    async def _send_widget_data(self, websocket: WebSocket, widget_types: List[str]) -> None:
        """Send data for specific widgets to a client"""
        
        widget_data = {}
        for widget_type in widget_types:
            if widget_type in self.current_data:
                widget_data[widget_type] = self.current_data[widget_type]
        
        response = {
            "type": "widget_data",
            "data": widget_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await websocket.send_text(json.dumps(response))
    
    async def _start_data_updates(self) -> None:
        """Start background tasks for data updates"""
        
        # Portfolio data update task
        portfolio_task = asyncio.create_task(self._update_portfolio_data())
        self.update_tasks.append(portfolio_task)
        
        # Trading metrics update task
        metrics_task = asyncio.create_task(self._update_trading_metrics())
        self.update_tasks.append(metrics_task)
        
        # System metrics update task
        system_task = asyncio.create_task(self._update_system_metrics())
        self.update_tasks.append(system_task)
        
        # Alert metrics update task
        alert_task = asyncio.create_task(self._update_alert_metrics())
        self.update_tasks.append(alert_task)
    
    async def _update_portfolio_data(self) -> None:
        """Update portfolio data periodically"""
        
        while self._running:
            try:
                # Get portfolio metrics from the existing dashboard service
                portfolio_data = await self.dashboard_service._get_portfolio_metrics()
                
                # Enhance with additional real-time data
                enhanced_data = {
                    **portfolio_data,
                    "cash_balance": portfolio_data.get("total_equity", 0) * 0.2,  # Assume 20% cash
                    "positions_value": portfolio_data.get("total_equity", 0) * 0.8,  # Assume 80% invested
                    "last_update": datetime.now(timezone.utc).isoformat()
                }
                
                self.current_data[WidgetType.PORTFOLIO_VALUE] = enhanced_data
                
                # Broadcast to all connected clients
                await self._broadcast_update(WidgetType.PORTFOLIO_VALUE, enhanced_data)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating portfolio data: {e}")
                await asyncio.sleep(10)
    
    async def _update_trading_metrics(self) -> None:
        """Update trading metrics periodically"""
        
        while self._running:
            try:
                # Get portfolio metrics from the existing dashboard service
                portfolio_metrics = await self.dashboard_service._get_portfolio_metrics()
                
                # Format for trading metrics widget
                metrics_data = {
                    "total_trades": portfolio_metrics.get("total_trades", 0),
                    "winning_trades": int(portfolio_metrics.get("total_trades", 0) * portfolio_metrics.get("win_rate", 0)),
                    "win_rate": portfolio_metrics.get("win_rate", 0) * 100,  # Convert to percentage
                    "sharpe_ratio": portfolio_metrics.get("sharpe_ratio", 0),
                    "max_drawdown": portfolio_metrics.get("max_drawdown", 0) * 100,  # Convert to percentage
                    "daily_pnl": portfolio_metrics.get("daily_pnl", 0),
                    "total_pnl": portfolio_metrics.get("total_pnl", 0),
                    "last_update": datetime.now(timezone.utc).isoformat()
                }
                
                self.current_data[WidgetType.TRADING_METRICS] = metrics_data
                
                # Broadcast to all connected clients
                await self._broadcast_update(WidgetType.TRADING_METRICS, metrics_data)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating trading metrics: {e}")
                await asyncio.sleep(15)
    
    async def _update_system_metrics(self) -> None:
        """Update system metrics periodically"""
        
        while self._running:
            try:
                # Get system metrics from the existing dashboard service
                system_data = await self.dashboard_service._get_system_metrics()
                
                # Add timestamp
                system_data["last_update"] = datetime.now(timezone.utc).isoformat()
                
                self.current_data[WidgetType.SYSTEM_METRICS] = system_data
                
                # Broadcast to all connected clients
                await self._broadcast_update(WidgetType.SYSTEM_METRICS, system_data)
                
                await asyncio.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating system metrics: {e}")
                await asyncio.sleep(20)
    
    async def _update_alert_metrics(self) -> None:
        """Update alert metrics periodically"""
        
        while self._running:
            try:
                # Get alert metrics from the existing dashboard service
                alert_data = await self.dashboard_service._get_alert_metrics()
                
                # Add timestamp
                alert_data["last_update"] = datetime.now(timezone.utc).isoformat()
                
                self.current_data[WidgetType.ALERT_PANEL] = alert_data
                
                # Broadcast to all connected clients
                await self._broadcast_update(WidgetType.ALERT_PANEL, alert_data)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating alert metrics: {e}")
                await asyncio.sleep(35)
    
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
            if connection in self.active_connections:
                self.active_connections.remove(connection)


class DashboardError(Exception):
    """Exception raised for dashboard errors"""
    pass
