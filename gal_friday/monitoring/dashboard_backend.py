"""FastAPI backend for the Gal-Friday monitoring dashboard.

This module provides the REST API and WebSocket endpoints for the
real-time monitoring dashboard.
"""

from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
from typing import Any

import aioredis
import asyncio
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import (
    EventType,
    ExecutionReportEvent,
    MarketDataL2Event,
    SystemStateEvent,
    TradeSignalProposedEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.monitoring.live_data_collector import LiveDataCollector
from gal_friday.monitoring_service import MonitoringService
from gal_friday.portfolio_manager import PortfolioManager


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        """Initialize the ConnectionManager with empty connection sets.

        Initializes the WebSocket connection tracking structures.
        """
        self.active_connections: set[WebSocket] = set()
        self.connection_subscriptions: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        self.connection_subscriptions.pop(websocket, None)

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        """Send message to specific connection."""
        try:
            await websocket.send_text(message)
        except Exception:
            # Connection might be closed
            self.disconnect(websocket)

    async def broadcast(self, message: str, channel: str | None = None) -> None:
        """Broadcast message to all connections or specific channel."""
        disconnected = []

        for connection in self.active_connections:
            # Check if connection is subscribed to channel
            if channel and channel not in self.connection_subscriptions.get(connection, set()):
                continue

            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

    def subscribe(self, websocket: WebSocket, channel: str) -> None:
        """Subscribe connection to a channel."""
        if websocket in self.connection_subscriptions:
            self.connection_subscriptions[websocket].add(channel)

    def unsubscribe(self, websocket: WebSocket, channel: str) -> None:
        """Unsubscribe connection from a channel."""
        if websocket in self.connection_subscriptions:
            self.connection_subscriptions[websocket].discard(channel)


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self, redis_client: aioredis.Redis | None = None) -> None:
        """Initialize the MetricsCollector with optional Redis client.

        Args:
            redis_client: Optional Redis client for persistent metric storage
        """
        self.redis = redis_client
        self.metrics_buffer: deque[dict[str, Any]] = deque(maxlen=1000)
        self.aggregated_metrics: dict[str, dict[str, float | int | str]] = {}

    async def record_metric(
        self,
        metric_type: str,
        value: float | str | dict[str, float | int | str],
        tags: dict[str, str] | None = None) -> None:
        """Record a metric value."""
        timestamp = datetime.now(UTC)
        metric = {
            "type": metric_type,
            "value": value,
            "timestamp": timestamp.isoformat(),
            "tags": tags or {},
        }

        # Add to buffer
        self.metrics_buffer.append(metric)

        # Store in Redis if available
        if self.redis:
            key = f"metric:{metric_type}:{timestamp.timestamp()}"
            await self.redis.setex(key, 3600, json.dumps(metric, default=str))

    async def get_recent_metrics(self, metric_type: str, minutes: int = 5) -> list[dict[str, Any]]:
        """Get recent metrics of specified type."""
        cutoff = datetime.now(UTC) - timedelta(minutes=minutes)

        return [
            m for m in self.metrics_buffer
            if m["type"] == metric_type and
            datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

    async def calculate_aggregates(self) -> None:
        """Calculate aggregated metrics."""
        # System health
        self.aggregated_metrics["system_health"] = await self._calculate_system_health()

        # Calculate and store trading performance
        trading_perf = await self._calculate_trading_performance()
        self.aggregated_metrics["trading_performance"] = trading_perf

        # Risk metrics
        self.aggregated_metrics["risk_metrics"] = await self._calculate_risk_metrics()

    async def _calculate_system_health(self) -> dict[str, Any]:
        """Calculate system health metrics."""
        api_errors = await self.get_recent_metrics("api_error", 5)
        latency_metrics = await self.get_recent_metrics("latency", 5)

        return {
            "api_error_rate": len(api_errors) / 5,  # Errors per minute
            "avg_latency": (
                sum(m["value"] for m in latency_metrics) / len(latency_metrics)
                if latency_metrics
                else 0
            ),
            "uptime_pct": self.calculate_uptime(),
        }

    async def _calculate_trading_performance(self) -> dict[str, Any]:
        """Calculate trading performance metrics."""
        trades = await self.get_recent_metrics("trade_complete", 60)

        if not trades:
            return {
                "trades_count": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "total_pnl": 0,
            }

        winning_trades = [t for t in trades if t["value"].get("pnl", 0) > 0]
        total_pnl = sum(Decimal(str(t["value"].get("pnl", 0))) for t in trades)

        return {
            "trades_count": len(trades),
            "win_rate": len(winning_trades) / len(trades) * 100,
            "avg_profit": float(total_pnl / len(trades)),
            "total_pnl": float(total_pnl),
        }

    async def _calculate_risk_metrics(self) -> dict[str, Any]:
        """Calculate risk metrics."""
        positions = await self.get_recent_metrics("position_update", 1)

        if not positions:
            return {
                "total_exposure": 0,
                "max_position_size": 0,
                "correlation_risk": 0,
            }

        latest_positions = positions[-1]["value"] if positions else {}

        return {
            "total_exposure": sum(
                abs(float(p.get("value", 0)))
                for p in latest_positions.values()
            ),
            "max_position_size": max(
                abs(float(p.get("value", 0)))
                for p in latest_positions.values()
            ) if latest_positions else 0,
            "correlation_risk": self.calculate_correlation_risk(latest_positions),
        }

    def calculate_uptime(self) -> float:
        """Calculate system uptime percentage.

        Returns:
            float: Uptime percentage (0-100)
        """
        # This is a simplified calculation
        # In production, this would track actual downtime events
        total_downtime_minutes = len([m for m in self.metrics_buffer
                                     if m["type"] == "system_down"])

        # Calculate uptime based on the last 24 hours
        total_minutes = 24 * 60
        uptime_minutes = total_minutes - total_downtime_minutes

        return round((uptime_minutes / total_minutes) * 100, 2)

    def calculate_correlation_risk(self, positions: dict[str, Any]) -> float:
        """Calculate correlation risk for current positions.

        Args:
            positions: Dictionary of current positions

        Returns:
            float: Correlation risk score (0-100)
        """
        if not positions or len(positions) < 2:
            return 0.0

        # Simplified correlation risk calculation
        # In production, this would use actual price correlation data
        # For now, calculate based on position concentration
        position_values = [abs(float(p.get("value", 0))) for p in positions.values()]
        total_value = sum(position_values)

        if total_value == 0:
            return 0.0

        # Calculate concentration using Herfindahl index
        concentration_index = sum((v/total_value)**2 for v in position_values)

        # Convert to risk score (0-100)
        # Higher concentration = higher risk
        correlation_risk = round(concentration_index * 100, 2)

        return min(correlation_risk, 100.0)  # Cap at 100


# Global instances
manager = ConnectionManager()
metrics_collector = MetricsCollector()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    # Startup
    logger = app.state.logger
    logger.info("Starting dashboard backend...", source_module="dashboard")

    # Initialize Redis connection if configured
    redis_url = app.state.config.get("redis.url")
    if redis_url:
        try:
            app.state.redis = aioredis.from_url(redis_url)
            metrics_collector.redis = app.state.redis
            logger.info("Connected to Redis")
        except Exception:
            logger.exception("Failed to connect to Redis: ")

    # Start background tasks
    app.state.background_tasks = set()

    # Metrics aggregation task
    async def aggregate_metrics() -> None:
        while True:
            await metrics_collector.calculate_aggregates()
            await asyncio.sleep(5)  # Update every 5 seconds

    task = asyncio.create_task(aggregate_metrics())
    app.state.background_tasks.add(task)

    yield

    # Shutdown
    logger.info("Shutting down dashboard backend...")

    # Cancel background tasks
    for task in app.state.background_tasks:
        task.cancel()

    # Close Redis connection
    if hasattr(app.state, "redis"):
        await app.state.redis.close()


# Create FastAPI app
app = FastAPI(
    title="Gal-Friday Monitoring Dashboard",
    version="1.0.0",
    lifespan=lifespan)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])


# Dependency injection
def get_config() -> ConfigManager:
    """Get configuration manager."""
    return app.state.config  # type: ignore[no-any-return]


def get_monitoring_service() -> MonitoringService:
    """Get monitoring service."""
    return app.state.monitoring_service  # type: ignore[no-any-return]


def get_portfolio_manager() -> PortfolioManager:
    """Get portfolio manager."""
    return app.state.portfolio_manager  # type: ignore[no-any-return]


# REST API Endpoints

@app.get("/api/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/api/system/status")
async def get_system_status() -> dict[str, Any]:
    """Get current system status.

    Returns:
        dict[str, Any]: System status including halt state and details
    """
    monitoring = get_monitoring_service()
    halt_status = (
        monitoring._halt_coordinator.get_halt_status()
        if hasattr(monitoring, "_halt_coordinator")
        else {}
    )

    return {
        "is_halted": monitoring.is_halted(),
        "halt_details": halt_status,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/api/system/halt")
async def trigger_halt(reason: str) -> dict[str, str]:
    """Manually trigger system HALT.

    Args:
        reason: Reason for halting the system

    Returns:
        dict[str, Any]: Status and reason for the halt
    """
    monitoring = get_monitoring_service()
    await monitoring.trigger_halt(reason, "Dashboard API")
    return {"status": "halted", "reason": reason}


@app.post("/api/system/resume")
async def trigger_resume() -> dict[str, str]:
    """Resume system from HALT.

    Returns:
        dict[str, Any]: Status and message confirming the resume
    """
    monitoring = get_monitoring_service()
    await monitoring.trigger_resume("Dashboard API")
    return {"status": "RESUMED", "message": "System resumed from HALT"}


@app.get("/api/portfolio/state")
async def get_portfolio_state(
    portfolio: PortfolioManager = Depends(get_portfolio_manager),  # noqa: B008
) -> dict[str, Any]:
    """Get current portfolio state."""
    state = portfolio.get_current_state()

    # Convert Decimal to float for JSON serialization
    return {
        "total_equity": float(state.get("total_equity", 0)),
        "cash_balance": float(state.get("cash_balance", 0)),
        "total_drawdown_pct": float(state.get("total_drawdown_pct", 0)),
        "daily_drawdown_pct": float(state.get("daily_drawdown_pct", 0)),
        "positions": {
            pair: {
                "quantity": float(pos.get("quantity", 0)),
                "value": float(pos.get("value", 0)),
                "unrealized_pnl": float(pos.get("unrealized_pnl", 0)),
                "side": pos.get("side"),
            }
            for pair, pos in state.get("positions", {}).items()
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/metrics/recent")
async def get_recent_metrics(metric_type: str, minutes: int = 5) -> dict[str, Any]:
    """Get recent metrics of specified type."""
    metrics = await metrics_collector.get_recent_metrics(metric_type, minutes)
    return {"metrics": metrics, "count": len(metrics)}


@app.get("/api/metrics/aggregated")
async def get_aggregated_metrics() -> dict[str, Any]:
    """Get aggregated system metrics."""
    return metrics_collector.aggregated_metrics


@app.get("/api/orders/active")
async def get_active_orders() -> dict[str, Any]:
    """Get list[Any] of active orders from execution handler."""
    try:
        data_collector = app.state.live_data_collector
        if data_collector:
            orders = await data_collector.get_active_orders()
            return {
                "orders": orders,
                "count": len(orders),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        # Return empty result if data collector not available
        return {
            "orders": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Data collector not initialized",
        }

    except Exception:
        logger = app.state.logger
        if logger:
            logger.exception("Failed to get active orders: ", source_module="dashboard")

        # Return empty result on error instead of mock data
        return {
            "orders": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve active orders",
        }


@app.get("/api/trades/history")
async def get_trade_history(limit: int = 50) -> dict[str, Any]:
    """Get recent trade history."""
    trades = await metrics_collector.get_recent_metrics("trade_complete", 1440)  # 24 hours

    # Sort by timestamp and limit
    trades.sort(key=lambda x: x["timestamp"], reverse=True)

    return {
        "trades": trades[:limit],
        "total": len(trades),
    }


@app.get("/api/trades/live")
async def get_live_trades(limit: int = 50) -> dict[str, Any]:
    """Get live trade history from execution handler."""
    try:
        data_collector = app.state.live_data_collector
        if data_collector:
            trades = await data_collector.get_recent_trades(limit)
            return {
                "trades": trades,
                "count": len(trades),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "trades": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Data collector not initialized",
        }

    except Exception:
        logger = app.state.logger
        if logger:
            logger.exception("Failed to get live trades: ", source_module="dashboard")

        return {
            "trades": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve live trades",
        }


@app.get("/api/portfolio/live")
async def get_live_portfolio() -> dict[str, Any]:
    """Get live portfolio metrics."""
    try:
        data_collector = app.state.live_data_collector
        if data_collector:
            portfolio_metrics = await data_collector.get_portfolio_metrics()
            return {
                "portfolio": portfolio_metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "portfolio": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Data collector not initialized",
        }

    except Exception:
        logger = app.state.logger
        if logger:
            logger.exception("Failed to get portfolio metrics: ", source_module="dashboard")

        return {
            "portfolio": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve portfolio metrics",
        }


@app.get("/api/strategy/live")
async def get_live_strategy_metrics() -> dict[str, Any]:
    """Get live strategy performance metrics."""
    try:
        data_collector = app.state.live_data_collector
        if data_collector:
            strategy_metrics = await data_collector.get_strategy_metrics()
            return {
                "strategy": strategy_metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "strategy": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Data collector not initialized",
        }

    except Exception:
        logger = app.state.logger
        if logger:
            logger.exception("Failed to get strategy metrics: ", source_module="dashboard")

        return {
            "strategy": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve strategy metrics",
        }


@app.get("/api/risk/live")
async def get_live_risk_metrics() -> dict[str, Any]:
    """Get live risk management metrics."""
    try:
        data_collector = app.state.live_data_collector
        if data_collector:
            risk_metrics = await data_collector.get_risk_metrics()
            return {
                "risk": risk_metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "risk": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Data collector not initialized",
        }

    except Exception:
        logger = app.state.logger
        if logger:
            logger.exception("Failed to get risk metrics: ", source_module="dashboard")

        return {
            "risk": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve risk metrics",
        }


@app.get("/api/system/health")
async def get_system_health() -> dict[str, Any]:
    """Get comprehensive system health metrics."""
    try:
        data_collector = app.state.live_data_collector
        if data_collector:
            health_metrics = await data_collector.get_system_health()
            return {
                "health": health_metrics,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        return {
            "health": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Data collector not initialized",
        }

    except Exception:
        logger = app.state.logger
        if logger:
            logger.exception("Failed to get system health: ", source_module="dashboard")

        return {
            "health": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve system health",
        }


# WebSocket endpoints

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now(UTC).isoformat(),
        })

        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            if data.get("action") == "subscribe":
                channel = data.get("channel")
                if channel:
                    manager.subscribe(websocket, channel)
                    await websocket.send_json({
                        "type": "subscription",
                        "channel": channel,
                        "status": "subscribed",
                    })

            elif data.get("action") == "unsubscribe":
                channel = data.get("channel")
                if channel:
                    manager.unsubscribe(websocket, channel)
                    await websocket.send_json({
                        "type": "subscription",
                        "channel": channel,
                        "status": "unsubscribed",
                    })

            elif data.get("action") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(UTC).isoformat(),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception:
        logger = app.state.logger
        logger.exception("WebSocket error: ", source_module="websocket")
        manager.disconnect(websocket)


class EventBroadcaster:
    """Broadcasts system events to WebSocket clients."""

    def __init__(self, pubsub: PubSubManager, connection_manager: ConnectionManager) -> None:
        """Initialize the EventBroadcaster with PubSub and ConnectionManager.

        Args:
            pubsub: PubSubManager instance for event subscription
            connection_manager: ConnectionManager for WebSocket connections
        """
        self.pubsub = pubsub
        self.manager = connection_manager

    async def start(self) -> None:
        """Subscribe to events and broadcast to clients."""
        # Subscribe to all relevant event types
        event_handlers = {
            EventType.MARKET_DATA_L2: self._handle_market_data,
            EventType.TRADE_SIGNAL_PROPOSED: self._handle_signal,
            EventType.TRADE_SIGNAL_APPROVED: self._handle_signal,
            EventType.EXECUTION_REPORT: self._handle_execution,
            EventType.SYSTEM_STATE_CHANGE: self._handle_system_state,
        }

        for event_type, handler in event_handlers.items():
            self.pubsub.subscribe(event_type, handler)  # type: ignore[arg-type]

    async def _handle_market_data(self, event: MarketDataL2Event) -> None:
        """Broadcast market data updates."""
        bid_price = float(event.bids[0][0]) if event.bids else None
        ask_price = float(event.asks[0][0]) if event.asks else None
        spread = (
            (ask_price - bid_price)
            if bid_price is not None and ask_price is not None
            else None
        )

        message = {
            "type": "market_data",
            "pair": event.trading_pair,
            "bid": bid_price,
            "ask": ask_price,
            "spread": spread,
            "timestamp": event.timestamp.isoformat(),
        }

        await self.manager.broadcast(json.dumps(message), "market_data")

    async def _handle_signal(self, event: TradeSignalProposedEvent) -> None:
        """Broadcast trade signal updates."""
        message = {
            "type": "trade_signal",
            "signal_id": str(event.signal_id),
            "pair": event.trading_pair,
            "side": event.side,
            "status": "proposed" if isinstance(event, TradeSignalProposedEvent) else "approved",
            "timestamp": event.timestamp.isoformat(),
        }

        await self.manager.broadcast(json.dumps(message), "signals")

    async def _handle_execution(self, event: ExecutionReportEvent) -> None:
        """Broadcast execution updates."""
        message = {
            "type": "execution_report",
            "order_id": event.exchange_order_id,
            "pair": event.trading_pair,
            "status": event.order_status,
            "filled": float(event.quantity_filled),
            "total": float(event.quantity_ordered),
            "timestamp": event.timestamp.isoformat(),
        }

        await self.manager.broadcast(json.dumps(message), "executions")

        # Also record as metric
        if event.order_status == "CLOSED":
            await metrics_collector.record_metric(
                "trade_complete",
                {
                    "pair": event.trading_pair,
                    "side": event.side,
                    "quantity": float(event.quantity_filled),
                    "price": float(event.average_fill_price) if event.average_fill_price else 0,
                    "pnl": 0,  # Would be calculated from position
                })

    async def _handle_system_state(self, event: SystemStateEvent) -> None:
        """Broadcast system state changes."""
        message = {
            "type": "system_state",
            "state": event.new_state,
            "reason": event.reason,
            "timestamp": event.timestamp.isoformat(),
        }

        await self.manager.broadcast(json.dumps(message), "system")

        # Record as metric
        await metrics_collector.record_metric(
            "system_state_change",
            {
                "state": event.new_state,
                "reason": event.reason,
            })


# Initialize the dashboard with system components
def initialize_dashboard(
    config: ConfigManager,
    pubsub: PubSubManager,
    monitoring: MonitoringService,
    portfolio: PortfolioManager,
    logger: LoggerService,
    execution_handler: Any = None,
    strategy_selection: Any = None,
    risk_manager: Any = None) -> FastAPI:
    """Initialize dashboard with system components."""
    app.state.config = config
    app.state.pubsub = pubsub
    app.state.monitoring_service = monitoring
    app.state.portfolio_manager = portfolio
    app.state.logger = logger

    # Initialize live data collector if components are available
    if execution_handler or portfolio or strategy_selection or risk_manager:
        app.state.live_data_collector = LiveDataCollector(
            logger=logger,
            pubsub_manager=pubsub,
            execution_handler=execution_handler,
            portfolio_manager=portfolio,
            strategy_selection=strategy_selection,
            risk_manager=risk_manager,
            monitoring_service=monitoring,
        )

        # Start data collection
        asyncio.create_task(app.state.live_data_collector.start_data_collection())
    else:
        app.state.live_data_collector = None

    # Create event broadcaster
    broadcaster = EventBroadcaster(pubsub, manager)
    broadcast_task = asyncio.create_task(broadcaster.start())
    app.state.background_tasks.add(broadcast_task)

    return app
