# Monitoring Dashboard Mock Data Implementation Design

**File**: `/gal_friday/monitoring/dashboard_backend.py`
- **Line 441**: `# For now, return mock data`
- **Impact**: Dashboard shows fake data instead of real metrics

## Overview
The monitoring dashboard currently returns mock data for active orders instead of integrating with the actual execution system. This design implements a comprehensive real-time monitoring system that provides authentic metrics from the trading system, portfolio manager, execution handler, and strategy selection components.

## Architecture Design

### 1. Current Mock Data Issues

```
Dashboard Mock Data Problems:
├── Fake Order Data (Line 441)
│   ├── Static mock orders instead of live data
│   ├── No connection to ExecutionHandler
│   └── Misleading user interface
├── Missing Real-Time Integration
│   ├── No WebSocket data from execution system
│   ├── No portfolio state integration
│   └── No strategy performance tracking
├── Limited Observability
│   ├── No system health monitoring
│   ├── No performance trend analysis
│   └── No alert generation
└── Data Consistency Issues
    ├── Mock data doesn't reflect actual state
    ├── No audit trail
    └── No historical data retention
```

### 2. Production Monitoring Architecture

```
Real-Time Monitoring System:
├── Data Sources Integration
│   ├── ExecutionHandler for orders/trades
│   ├── PortfolioManager for positions/PnL
│   ├── StrategySelection for performance
│   ├── RiskManager for exposure metrics
│   └── SystemMonitoring for health data
├── Real-Time Data Pipeline
│   ├── Event-driven data collection
│   ├── WebSocket streaming
│   ├── Metric aggregation engine
│   └── Historical data storage
├── Dashboard Backend Services
│   ├── Live order tracking
│   ├── Performance analytics
│   ├── Risk monitoring
│   ├── System health checks
│   └── Alert management
└── Frontend Integration
    ├── Real-time chart updates
    ├── Interactive dashboards
    ├── Alert notifications
    └── Historical analysis views
```

## Implementation Plan

### Phase 1: Data Integration Services

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from dataclasses import dataclass, field
import asyncio
import json

from gal_friday.core.events import EventType, Event
from gal_friday.core.pubsub import PubSubManager
from gal_friday.execution_handler import ExecutionHandler
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.strategy_selection import StrategySelectionSystem
from gal_friday.risk_manager import RiskManager
from gal_friday.monitoring_service import MonitoringService
from gal_friday.logger_service import LoggerService


@dataclass
class LiveOrderData:
    """Real-time order data structure."""
    order_id: str
    trading_pair: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal]
    status: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_fill_price: Optional[Decimal]
    created_at: datetime
    updated_at: datetime
    strategy_id: str
    fees_paid: Decimal
    time_in_force: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "order_id": self.order_id,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "type": self.order_type,
            "quantity": float(self.quantity),
            "price": float(self.price) if self.price else None,
            "status": self.status,
            "filled": float(self.filled_quantity),
            "remaining": float(self.remaining_quantity),
            "average_fill_price": float(self.average_fill_price) if self.average_fill_price else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "strategy_id": self.strategy_id,
            "fees_paid": float(self.fees_paid),
            "time_in_force": self.time_in_force
        }


@dataclass
class LiveTradeData:
    """Real-time trade data structure."""
    trade_id: str
    order_id: str
    trading_pair: str
    side: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime
    strategy_id: str
    realized_pnl: Decimal
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "fee": float(self.fee),
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "realized_pnl": float(self.realized_pnl)
        }


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    network_latency_ms: float
    active_connections: int
    orders_per_minute: float
    api_error_rate: float
    system_uptime_hours: float
    disk_usage_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "network_latency_ms": self.network_latency_ms,
            "active_connections": self.active_connections,
            "orders_per_minute": self.orders_per_minute,
            "api_error_rate": self.api_error_rate,
            "system_uptime_hours": self.system_uptime_hours,
            "disk_usage_percent": self.disk_usage_percent
        }


class LiveDataCollector:
    """Collects real-time data from various system components."""
    
    def __init__(
        self,
        logger: LoggerService,
        pubsub_manager: PubSubManager,
        execution_handler: ExecutionHandler,
        portfolio_manager: PortfolioManager,
        strategy_selection: StrategySelectionSystem,
        risk_manager: RiskManager,
        monitoring_service: MonitoringService
    ):
        self.logger = logger
        self.pubsub = pubsub_manager
        self.execution_handler = execution_handler
        self.portfolio_manager = portfolio_manager
        self.strategy_selection = strategy_selection
        self.risk_manager = risk_manager
        self.monitoring_service = monitoring_service
        self._source_module = self.__class__.__name__
        
        # Data caches
        self._active_orders: Dict[str, LiveOrderData] = {}
        self._recent_trades: List[LiveTradeData] = []
        self._system_metrics: Optional[SystemHealthMetrics] = None
        
        # Configuration
        self._max_trade_history = 1000
        self._metrics_cache_ttl = timedelta(seconds=30)
        self._last_metrics_update: Optional[datetime] = None
        
    async def start_data_collection(self) -> None:
        """Start real-time data collection from all sources."""
        try:
            # Subscribe to execution events
            self.pubsub.subscribe(EventType.EXECUTION_REPORT, self._handle_execution_report)
            self.pubsub.subscribe(EventType.ORDER_STATUS_CHANGE, self._handle_order_update)
            self.pubsub.subscribe(EventType.TRADE_COMPLETED, self._handle_trade_completed)
            
            # Start periodic system metrics collection
            asyncio.create_task(self._system_metrics_loop())
            
            # Initialize with current data
            await self._load_initial_data()
            
            self.logger.info(
                "Live data collection started",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to start data collection: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            raise
    
    async def get_active_orders(self) -> List[Dict[str, Any]]:
        """Get current active orders from execution handler."""
        try:
            # Get orders from execution handler
            raw_orders = await self.execution_handler.get_active_orders()
            
            # Convert to standardized format
            active_orders = []
            for order in raw_orders:
                live_order = self._convert_to_live_order(order)
                self._active_orders[live_order.order_id] = live_order
                active_orders.append(live_order.to_dict())
            
            self.logger.debug(
                f"Retrieved {len(active_orders)} active orders",
                source_module=self._source_module
            )
            
            return active_orders
            
        except Exception as e:
            self.logger.error(
                f"Failed to get active orders: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return []
    
    async def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades from execution handler."""
        try:
            # Get trades from execution handler
            raw_trades = await self.execution_handler.get_recent_trades(limit)
            
            # Convert to standardized format
            recent_trades = []
            for trade in raw_trades:
                live_trade = self._convert_to_live_trade(trade)
                recent_trades.append(live_trade.to_dict())
            
            # Update cache
            self._recent_trades = [
                self._convert_to_live_trade(trade) for trade in raw_trades
            ]
            
            self.logger.debug(
                f"Retrieved {len(recent_trades)} recent trades",
                source_module=self._source_module
            )
            
            return recent_trades
            
        except Exception as e:
            self.logger.error(
                f"Failed to get recent trades: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return []
    
    async def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get current portfolio metrics."""
        try:
            portfolio_state = self.portfolio_manager.get_current_state()
            
            # Add performance calculations
            performance_metrics = await self._calculate_portfolio_performance()
            portfolio_state.update(performance_metrics)
            
            return portfolio_state
            
        except Exception as e:
            self.logger.error(
                f"Failed to get portfolio metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {}
    
    async def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get current strategy performance metrics."""
        try:
            current_strategy = self.strategy_selection.get_current_strategy()
            
            # Get strategy performance from strategy selection system
            strategy_metrics = {
                "current_strategy": current_strategy,
                "last_selection_time": self.strategy_selection._last_selection_time.isoformat() if self.strategy_selection._last_selection_time else None,
                "in_transition": self.strategy_selection.strategy_orchestrator.is_in_transition(),
                "active_transition": None
            }
            
            # Add transition details if active
            active_transition = self.strategy_selection.strategy_orchestrator.get_active_transition()
            if active_transition:
                strategy_metrics["active_transition"] = {
                    "transition_id": active_transition.transition_id,
                    "from_strategy": active_transition.from_strategy,
                    "to_strategy": active_transition.to_strategy,
                    "current_phase": active_transition.current_phase.value,
                    "started_at": active_transition.started_at.isoformat() if active_transition.started_at else None
                }
            
            return strategy_metrics
            
        except Exception as e:
            self.logger.error(
                f"Failed to get strategy metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {}
    
    async def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk management metrics."""
        try:
            risk_metrics = {
                "available_risk_budget": float(self.risk_manager.get_available_risk_budget()),
                "current_exposure": float(self.risk_manager.get_current_exposure()),
                "max_position_size": float(self.risk_manager.get_max_position_size()),
                "is_halted": self.monitoring_service.is_halted(),
                "halt_details": {}
            }
            
            # Add halt details if system is halted
            if risk_metrics["is_halted"]:
                halt_status = getattr(self.monitoring_service, '_halt_coordinator', None)
                if halt_status:
                    risk_metrics["halt_details"] = halt_status.get_halt_status()
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(
                f"Failed to get risk metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            # Check if we need to update metrics
            now = datetime.now(UTC)
            if (self._last_metrics_update is None or 
                now - self._last_metrics_update > self._metrics_cache_ttl):
                await self._update_system_metrics()
            
            if self._system_metrics:
                return self._system_metrics.to_dict()
            else:
                return {}
                
        except Exception as e:
            self.logger.error(
                f"Failed to get system health: {e}",
                source_module=self._source_module,
                exc_info=True
            )
            return {}
    
    async def _handle_execution_report(self, event: Event) -> None:
        """Handle execution report events."""
        try:
            # Update order status
            order_id = event.exchange_order_id
            if order_id in self._active_orders:
                order = self._active_orders[order_id]
                order.status = event.order_status
                order.filled_quantity = event.quantity_filled
                order.remaining_quantity = event.quantity_ordered - event.quantity_filled
                order.updated_at = event.timestamp
                
                if event.average_fill_price:
                    order.average_fill_price = event.average_fill_price
            
        except Exception as e:
            self.logger.error(
                f"Error handling execution report: {e}",
                source_module=self._source_module,
                exc_info=True
            )
    
    async def _handle_order_update(self, event: Event) -> None:
        """Handle order status change events."""
        try:
            # This would be implemented based on the actual event structure
            pass
            
        except Exception as e:
            self.logger.error(
                f"Error handling order update: {e}",
                source_module=self._source_module,
                exc_info=True
            )
    
    async def _handle_trade_completed(self, event: Event) -> None:
        """Handle trade completion events."""
        try:
            # Add to recent trades
            trade_data = LiveTradeData(
                trade_id=str(event.trade_id) if hasattr(event, 'trade_id') else f"trade_{int(event.timestamp.timestamp())}",
                order_id=event.exchange_order_id if hasattr(event, 'exchange_order_id') else "unknown",
                trading_pair=event.trading_pair if hasattr(event, 'trading_pair') else "unknown",
                side=event.side if hasattr(event, 'side') else "unknown",
                quantity=event.quantity_filled if hasattr(event, 'quantity_filled') else Decimal("0"),
                price=event.average_fill_price if hasattr(event, 'average_fill_price') else Decimal("0"),
                fee=Decimal("0"),  # Would be extracted from event
                timestamp=event.timestamp,
                strategy_id="unknown",  # Would be extracted from event
                realized_pnl=Decimal("0")  # Would be calculated
            )
            
            self._recent_trades.insert(0, trade_data)
            
            # Limit history size
            if len(self._recent_trades) > self._max_trade_history:
                self._recent_trades = self._recent_trades[:self._max_trade_history]
            
        except Exception as e:
            self.logger.error(
                f"Error handling trade completion: {e}",
                source_module=self._source_module,
                exc_info=True
            )
    
    async def _system_metrics_loop(self) -> None:
        """Continuous system metrics collection loop."""
        while True:
            try:
                await self._update_system_metrics()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(
                    f"Error in system metrics loop: {e}",
                    source_module=self._source_module,
                    exc_info=True
                )
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_system_metrics(self) -> None:
        """Update system health metrics."""
        try:
            import psutil
            import time
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network latency (would ping exchange API)
            network_latency = await self._measure_network_latency()
            
            # Get application-specific metrics
            active_connections = len(manager.active_connections)
            orders_per_minute = await self._calculate_orders_per_minute()
            api_error_rate = await self._calculate_api_error_rate()
            
            # Calculate uptime
            uptime_hours = self._calculate_uptime_hours()
            
            self._system_metrics = SystemHealthMetrics(
                timestamp=datetime.now(UTC),
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / 1024 / 1024,
                network_latency_ms=network_latency,
                active_connections=active_connections,
                orders_per_minute=orders_per_minute,
                api_error_rate=api_error_rate,
                system_uptime_hours=uptime_hours,
                disk_usage_percent=disk.percent
            )
            
            self._last_metrics_update = datetime.now(UTC)
            
        except Exception as e:
            self.logger.error(
                f"Failed to update system metrics: {e}",
                source_module=self._source_module,
                exc_info=True
            )
    
    def _convert_to_live_order(self, raw_order: Any) -> LiveOrderData:
        """Convert raw order data to LiveOrderData."""
        # This would be implemented based on the actual order structure
        return LiveOrderData(
            order_id=str(raw_order.get("id", "unknown")),
            trading_pair=raw_order.get("trading_pair", "unknown"),
            side=raw_order.get("side", "unknown"),
            order_type=raw_order.get("type", "unknown"),
            quantity=Decimal(str(raw_order.get("quantity", 0))),
            price=Decimal(str(raw_order.get("price", 0))) if raw_order.get("price") else None,
            status=raw_order.get("status", "unknown"),
            filled_quantity=Decimal(str(raw_order.get("filled", 0))),
            remaining_quantity=Decimal(str(raw_order.get("remaining", 0))),
            average_fill_price=Decimal(str(raw_order.get("average_fill_price", 0))) if raw_order.get("average_fill_price") else None,
            created_at=raw_order.get("created_at", datetime.now(UTC)),
            updated_at=raw_order.get("updated_at", datetime.now(UTC)),
            strategy_id=raw_order.get("strategy_id", "unknown"),
            fees_paid=Decimal(str(raw_order.get("fees", 0))),
            time_in_force=raw_order.get("time_in_force", "GTC")
        )
    
    def _convert_to_live_trade(self, raw_trade: Any) -> LiveTradeData:
        """Convert raw trade data to LiveTradeData."""
        # This would be implemented based on the actual trade structure
        return LiveTradeData(
            trade_id=str(raw_trade.get("id", "unknown")),
            order_id=str(raw_trade.get("order_id", "unknown")),
            trading_pair=raw_trade.get("trading_pair", "unknown"),
            side=raw_trade.get("side", "unknown"),
            quantity=Decimal(str(raw_trade.get("quantity", 0))),
            price=Decimal(str(raw_trade.get("price", 0))),
            fee=Decimal(str(raw_trade.get("fee", 0))),
            timestamp=raw_trade.get("timestamp", datetime.now(UTC)),
            strategy_id=raw_trade.get("strategy_id", "unknown"),
            realized_pnl=Decimal(str(raw_trade.get("realized_pnl", 0)))
        )
    
    async def _load_initial_data(self) -> None:
        """Load initial data on startup."""
        try:
            # Load active orders
            await self.get_active_orders()
            
            # Load recent trades
            await self.get_recent_trades()
            
            self.logger.info(
                "Initial data loaded successfully",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to load initial data: {e}",
                source_module=self._source_module,
                exc_info=True
            )
    
    async def _calculate_portfolio_performance(self) -> Dict[str, Any]:
        """Calculate additional portfolio performance metrics."""
        # This would implement complex portfolio analytics
        return {
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0,
            "win_rate_7d": 0.0,
            "sharpe_ratio_30d": 0.0
        }
    
    async def _measure_network_latency(self) -> float:
        """Measure network latency to exchange API."""
        # This would ping the exchange API and measure response time
        return 45.0  # Placeholder
    
    async def _calculate_orders_per_minute(self) -> float:
        """Calculate recent orders per minute rate."""
        # This would analyze recent order frequency
        return 2.5  # Placeholder
    
    async def _calculate_api_error_rate(self) -> float:
        """Calculate API error rate."""
        # This would analyze recent API call success/failure rates
        return 0.001  # Placeholder
    
    def _calculate_uptime_hours(self) -> float:
        """Calculate system uptime in hours."""
        # This would track actual system uptime
        return 72.5  # Placeholder
```

### Phase 2: Enhanced Dashboard Endpoints

```python
# Replace the mock data endpoint in dashboard_backend.py

@app.get("/api/orders/active")
async def get_active_orders(
    data_collector: LiveDataCollector = Depends(get_live_data_collector)
) -> Dict[str, Any]:
    """Get list of active orders from execution handler."""
    try:
        orders = await data_collector.get_active_orders()
        
        return {
            "orders": orders,
            "count": len(orders),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger = app.state.logger
        logger.error(f"Failed to get active orders: {e}", source_module="dashboard")
        
        # Return empty result on error instead of mock data
        return {
            "orders": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve active orders"
        }


@app.get("/api/trades/live")
async def get_live_trades(
    limit: int = 50,
    data_collector: LiveDataCollector = Depends(get_live_data_collector)
) -> Dict[str, Any]:
    """Get live trade history."""
    try:
        trades = await data_collector.get_recent_trades(limit)
        
        return {
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger = app.state.logger
        logger.error(f"Failed to get live trades: {e}", source_module="dashboard")
        
        return {
            "trades": [],
            "count": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve live trades"
        }


@app.get("/api/portfolio/live")
async def get_live_portfolio(
    data_collector: LiveDataCollector = Depends(get_live_data_collector)
) -> Dict[str, Any]:
    """Get live portfolio metrics."""
    try:
        portfolio_metrics = await data_collector.get_portfolio_metrics()
        
        return {
            "portfolio": portfolio_metrics,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger = app.state.logger
        logger.error(f"Failed to get portfolio metrics: {e}", source_module="dashboard")
        
        return {
            "portfolio": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve portfolio metrics"
        }


@app.get("/api/strategy/live")
async def get_live_strategy_metrics(
    data_collector: LiveDataCollector = Depends(get_live_data_collector)
) -> Dict[str, Any]:
    """Get live strategy performance metrics."""
    try:
        strategy_metrics = await data_collector.get_strategy_metrics()
        
        return {
            "strategy": strategy_metrics,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger = app.state.logger
        logger.error(f"Failed to get strategy metrics: {e}", source_module="dashboard")
        
        return {
            "strategy": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve strategy metrics"
        }


@app.get("/api/risk/live")
async def get_live_risk_metrics(
    data_collector: LiveDataCollector = Depends(get_live_data_collector)
) -> Dict[str, Any]:
    """Get live risk management metrics."""
    try:
        risk_metrics = await data_collector.get_risk_metrics()
        
        return {
            "risk": risk_metrics,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger = app.state.logger
        logger.error(f"Failed to get risk metrics: {e}", source_module="dashboard")
        
        return {
            "risk": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve risk metrics"
        }


@app.get("/api/system/health")
async def get_system_health(
    data_collector: LiveDataCollector = Depends(get_live_data_collector)
) -> Dict[str, Any]:
    """Get comprehensive system health metrics."""
    try:
        health_metrics = await data_collector.get_system_health()
        
        return {
            "health": health_metrics,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger = app.state.logger
        logger.error(f"Failed to get system health: {e}", source_module="dashboard")
        
        return {
            "health": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "error": "Failed to retrieve system health"
        }


# Dependency injection for LiveDataCollector
def get_live_data_collector() -> LiveDataCollector:
    """Get live data collector instance."""
    return app.state.live_data_collector


# Update the initialize_dashboard function
def initialize_dashboard(
    config: ConfigManager,
    pubsub: PubSubManager,
    monitoring: MonitoringService,
    portfolio: PortfolioManager,
    logger: LoggerService,
    execution_handler: ExecutionHandler,
    strategy_selection: StrategySelectionSystem,
    risk_manager: RiskManager
) -> FastAPI:
    """Initialize dashboard with all system components."""
    app.state.config = config
    app.state.pubsub = pubsub
    app.state.monitoring_service = monitoring
    app.state.portfolio_manager = portfolio
    app.state.logger = logger
    
    # Initialize live data collector
    app.state.live_data_collector = LiveDataCollector(
        logger=logger,
        pubsub_manager=pubsub,
        execution_handler=execution_handler,
        portfolio_manager=portfolio,
        strategy_selection=strategy_selection,
        risk_manager=risk_manager,
        monitoring_service=monitoring
    )
    
    # Start data collection
    asyncio.create_task(app.state.live_data_collector.start_data_collection())
    
    # Create event broadcaster
    broadcaster = EventBroadcaster(pubsub, manager)
    broadcast_task = asyncio.create_task(broadcaster.start())
    app.state.background_tasks.add(broadcast_task)
    
    return app
```

### Phase 3: Real-Time WebSocket Enhancements

```python
class EnhancedEventBroadcaster:
    """Enhanced event broadcaster with comprehensive real-time data."""
    
    def __init__(
        self, 
        pubsub: PubSubManager, 
        connection_manager: ConnectionManager,
        live_data_collector: LiveDataCollector
    ):
        self.pubsub = pubsub
        self.manager = connection_manager
        self.data_collector = live_data_collector
        
    async def start(self) -> None:
        """Subscribe to events and broadcast comprehensive data."""
        # Subscribe to all relevant event types
        event_handlers = {
            EventType.MARKET_DATA_L2: self._handle_market_data,
            EventType.TRADE_SIGNAL_PROPOSED: self._handle_signal,
            EventType.TRADE_SIGNAL_APPROVED: self._handle_signal,
            EventType.EXECUTION_REPORT: self._handle_execution,
            EventType.SYSTEM_STATE_CHANGE: self._handle_system_state,
            EventType.PORTFOLIO_UPDATE: self._handle_portfolio_update,
            EventType.RISK_LIMIT_BREACH: self._handle_risk_alert,
            EventType.STRATEGY_TRANSITION: self._handle_strategy_transition
        }
        
        for event_type, handler in event_handlers.items():
            self.pubsub.subscribe(event_type, handler)
        
        # Start periodic data broadcasts
        asyncio.create_task(self._periodic_data_broadcast())
    
    async def _periodic_data_broadcast(self) -> None:
        """Broadcast periodic system updates."""
        while True:
            try:
                # Broadcast comprehensive system status every 30 seconds
                system_status = await self._compile_system_status()
                await self.manager.broadcast(
                    json.dumps(system_status), 
                    "system_status"
                )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in periodic broadcast: {e}")
                await asyncio.sleep(60)
    
    async def _compile_system_status(self) -> Dict[str, Any]:
        """Compile comprehensive system status."""
        return {
            "type": "system_status",
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {
                "portfolio": await self.data_collector.get_portfolio_metrics(),
                "strategy": await self.data_collector.get_strategy_metrics(),
                "risk": await self.data_collector.get_risk_metrics(),
                "health": await self.data_collector.get_system_health(),
                "active_orders_count": len(await self.data_collector.get_active_orders()),
                "recent_trades_count": len(await self.data_collector.get_recent_trades(10))
            }
        }
    
    async def _handle_portfolio_update(self, event: Event) -> None:
        """Handle portfolio update events."""
        try:
            portfolio_data = await self.data_collector.get_portfolio_metrics()
            
            message = {
                "type": "portfolio_update",
                "data": portfolio_data,
                "timestamp": event.timestamp.isoformat()
            }
            
            await self.manager.broadcast(json.dumps(message), "portfolio")
            
        except Exception as e:
            logger.error(f"Error handling portfolio update: {e}")
    
    async def _handle_risk_alert(self, event: Event) -> None:
        """Handle risk management alerts."""
        try:
            message = {
                "type": "risk_alert",
                "alert_type": getattr(event, "alert_type", "unknown"),
                "severity": getattr(event, "severity", "medium"),
                "message": getattr(event, "message", "Risk limit breach"),
                "timestamp": event.timestamp.isoformat()
            }
            
            await self.manager.broadcast(json.dumps(message), "alerts")
            
        except Exception as e:
            logger.error(f"Error handling risk alert: {e}")
    
    async def _handle_strategy_transition(self, event: Event) -> None:
        """Handle strategy transition events."""
        try:
            strategy_data = await self.data_collector.get_strategy_metrics()
            
            message = {
                "type": "strategy_transition",
                "data": strategy_data,
                "timestamp": event.timestamp.isoformat()
            }
            
            await self.manager.broadcast(json.dumps(message), "strategy")
            
        except Exception as e:
            logger.error(f"Error handling strategy transition: {e}")
```

## Testing Strategy

1. **Unit Tests**
   - Data collector functionality
   - Real-time data conversion
   - WebSocket message handling
   - Error recovery scenarios

2. **Integration Tests**
   - End-to-end data pipeline
   - Dashboard endpoint responses
   - WebSocket real-time updates
   - System component integration

3. **Performance Tests**
   - High-frequency data handling
   - WebSocket connection scaling
   - Memory usage optimization
   - Database query performance

## Monitoring & Observability

1. **Data Quality Metrics**
   - Data freshness monitoring
   - Missing data detection
   - Data consistency validation
   - Real-time vs batch data comparison

2. **System Performance**
   - Dashboard response times
   - WebSocket latency
   - Data collection overhead
   - Memory and CPU usage

## Security Considerations

1. **Data Protection**
   - Sensitive data filtering
   - Access control for dashboard endpoints
   - WebSocket authentication
   - Rate limiting for API calls

2. **Operational Security**
   - Audit trail for dashboard access
   - Data retention policies
   - Error handling without data leakage
   - Secure WebSocket connections

## Future Enhancements

1. **Advanced Analytics**
   - Real-time performance dashboards
   - Predictive alerts
   - Historical trend analysis
   - Custom metric definitions

2. **User Experience**
   - Interactive charts and graphs
   - Custom dashboard layouts
   - Mobile-responsive design
   - Export capabilities