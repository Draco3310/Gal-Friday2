"""Production-ready components for backtesting engine."""

from collections import defaultdict, deque
import contextlib
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
import heapq
import logging
from typing import Any, cast

import asyncio

from .config_manager import ConfigManager
from .core.events import Event, EventType
from .core.pubsub import EventHandler
from .dal.models.position import Position
from .models.order import Order


@dataclass
class SimulatedTimeEvent:
    """Event wrapper for time-based simulation."""
    timestamp: datetime
    original_event: Event
    priority: int = 0

    def __lt__(self, other: "SimulatedTimeEvent") -> bool:
        """Compare events for priority queue ordering."""
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.priority < other.priority


class BacktestPubSubManager:
    """Production-ready PubSub manager for backtesting with time simulation."""

    def __init__(
        self,
        logger: logging.Logger,
        config_manager: ConfigManager,
        simulation_start: datetime,
        simulation_end: datetime,
    ) -> None:
        self.logger = logger
        self.config = config_manager
        self._source_module = self.__class__.__name__

        # Event subscriptions
        self._subscribers: dict[EventType, list[EventHandler[Any]]] = defaultdict(list)

        # Time simulation
        self._simulation_start = simulation_start
        self._simulation_end = simulation_end
        self._current_time = simulation_start
        self._time_acceleration = config_manager.get_float("backtesting.time_acceleration", 1.0)

        # Event queue with time ordering
        self._event_queue: list[SimulatedTimeEvent] = []
        self._processed_events: int = 0
        self._dropped_events: int = 0

        # Performance metrics
        self._event_processing_times: deque[float] = deque(maxlen=1000)
        self._max_queue_size = config_manager.get_int("backtesting.max_queue_size", 100000)

        # State tracking
        self._is_running = False
        self._processing_task: asyncio.Task[Any] | None = None

        self.logger.info(
            "BacktestPubSubManager initialized for period %s to %s",
            simulation_start.isoformat(),
            simulation_end.isoformat(),
            extra={"source_module": self._source_module},
        )

    async def start(self) -> None:
        """Start the backtesting event processing loop."""
        if self._is_running:
            self.logger.warning("BacktestPubSubManager already running", extra={"source_module": self._source_module})
            return

        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_events())

        self.logger.info("BacktestPubSubManager started", extra={"source_module": self._source_module})

    async def stop_consuming(self) -> None:
        """Stop event processing and cleanup."""
        self._is_running = False

        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        # Log final statistics
        self.logger.info(
            "BacktestPubSubManager stopped. Processed: %d, Dropped: %d, Queue size: %d",
            self._processed_events,
            self._dropped_events,
            len(self._event_queue),
            extra={"source_module": self._source_module},
        )

    def subscribe(self, event_type: EventType, handler: EventHandler[Any]) -> None:
        """Subscribe to specific event types."""
        self._subscribers[event_type].append(handler)
        self.logger.debug(
            "Subscribed handler for event type: %s (total handlers: %d)",
            event_type.name,
            len(self._subscribers[event_type]),
            extra={"source_module": self._source_module},
        )

    def unsubscribe(self, event_type: EventType, handler: EventHandler[Any]) -> None:
        """Unsubscribe from specific event types."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            self.logger.debug(
                "Unsubscribed handler for event type: %s",
                event_type.name,
                extra={"source_module": self._source_module},
            )

    async def publish(self, event: Event, scheduled_time: datetime | None = None) -> None:
        """Publish event with optional time scheduling for backtesting."""
        event_time = scheduled_time or self._current_time

        # Validate event time is within simulation bounds
        if event_time < self._simulation_start or event_time > self._simulation_end:
            self.logger.warning(
                "Event time %s outside simulation bounds, dropping event",
                event_time.isoformat(),
                extra={"source_module": self._source_module},
            )
            self._dropped_events += 1
            return

        # Check queue size limits
        if len(self._event_queue) >= self._max_queue_size:
            # Drop oldest events to make room
            self._event_queue = self._event_queue[-self._max_queue_size//2:]
            self._dropped_events += len(self._event_queue) - len(self._event_queue)
            self.logger.warning("Event queue full, dropped old events", extra={"source_module": self._source_module})

        # Create simulated time event
        sim_event = SimulatedTimeEvent(
            timestamp=event_time,
            original_event=event,
            priority=self._get_event_priority(event),
        )

        # Add to priority queue
        heapq.heappush(self._event_queue, sim_event)

        self.logger.debug(
            "Published event %s scheduled for %s (queue size: %d)",
            event.__class__.__name__,
            event_time.isoformat(),
            len(self._event_queue),
            extra={"source_module": self._source_module},
        )

    async def _process_events(self) -> None:
        """Main event processing loop for backtesting."""
        while self._is_running:
            try:
                # Check if we have events to process
                if not self._event_queue:
                    await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                    continue

                # Get next event from queue
                sim_event = heapq.heappop(self._event_queue)

                # Update simulation time
                self._current_time = sim_event.timestamp

                # Process the event
                await self._handle_event(sim_event.original_event)
                self._processed_events += 1

                # Apply time acceleration delay
                if self._time_acceleration > 1.0:
                    delay = 0.001 / self._time_acceleration
                    await asyncio.sleep(delay)

            except Exception as e:
                self.logger.error(
                    "Error in event processing loop: %s",
                    str(e),
                    exc_info=True,
                    extra={"source_module": self._source_module},
                )
                await asyncio.sleep(0.1)

    async def _handle_event(self, event: Event) -> None:
        """Handle individual event by calling all subscribers."""
        event_type = getattr(event, "event_type", None)
        if not event_type:
            self.logger.warning("Event missing event_type: %s", event, extra={"source_module": self._source_module})
            return

        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return

        # Track processing time
        start_time = asyncio.get_event_loop().time()

        # Execute all handlers concurrently
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._execute_handler(handler, event))
            tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Record performance metrics
        processing_time = asyncio.get_event_loop().time() - start_time
        self._event_processing_times.append(processing_time)

    async def _execute_handler(self, handler: EventHandler[Any], event: Event) -> None:
        """Execute a single event handler with error handling."""
        try:
            # Apply timeout to handler execution
            timeout = self.config.get_float("backtesting.handler_timeout", 5.0)
            await asyncio.wait_for(handler(event), timeout=timeout)

        except TimeoutError:
            self.logger.exception(
                "Handler timeout for event %s",
                event.__class__.__name__,
                extra={"source_module": self._source_module},
            )
        except Exception as e:
            self.logger.error(
                "Handler error for event %s: %s",
                event.__class__.__name__,
                str(e),
                exc_info=True,
                extra={"source_module": self._source_module},
            )

    def _get_event_priority(self, event: Event) -> int:
        """Determine event priority for processing order."""
        # Market data events have highest priority
        if hasattr(event, "event_type"):
            if event.event_type.name.startswith("MARKET"):
                return 1
            if event.event_type.name.startswith("ORDER"):
                return 2
            if event.event_type.name.startswith("RISK"):
                return 3
        return 10  # Default priority

    def get_simulation_time(self) -> datetime:
        """Get current simulation time."""
        return self._current_time

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about the PubSub manager."""
        avg_processing_time = 0.0
        if self._event_processing_times:
            avg_processing_time = sum(self._event_processing_times) / len(self._event_processing_times)

        return {
            "processed_events": self._processed_events,
            "dropped_events": self._dropped_events,
            "queue_size": len(self._event_queue),
            "avg_processing_time_ms": avg_processing_time * 1000,
            "current_simulation_time": self._current_time.isoformat(),
            "subscribers_count": sum(len(handlers) for handlers in self._subscribers.values()),
            "is_running": self._is_running,
        }


@dataclass
class BacktestRiskLimits:
    """Risk limits configuration for backtesting."""
    max_position_size: Decimal
    max_daily_loss: Decimal
    max_drawdown: Decimal
    max_concentration: Decimal  # % of portfolio in single position
    max_leverage: Decimal
    position_limit_per_symbol: int


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    approved: bool
    reason: str | None = None


class BacktestRiskManager:
    """Production-ready risk manager for backtesting scenarios."""

    def __init__(
        self,
        logger: logging.Logger,
        config_manager: ConfigManager,
        initial_capital: Decimal,
    ) -> None:
        self.logger = logger
        self.config = config_manager
        self.initial_capital = initial_capital
        self._source_module = self.__class__.__name__

        # Risk limits
        self.risk_limits = BacktestRiskLimits(
            max_position_size=config_manager.get_decimal("risk.max_position_size", Decimal(10000)),
            max_daily_loss=config_manager.get_decimal("risk.max_daily_loss", Decimal(1000)),
            max_drawdown=config_manager.get_decimal("risk.max_drawdown", Decimal("0.20")),  # 20%
            max_concentration=config_manager.get_decimal("risk.max_concentration", Decimal("0.10")),  # 10%
            max_leverage=config_manager.get_decimal("risk.max_leverage", Decimal("2.0")),
            position_limit_per_symbol=config_manager.get_int("risk.position_limit_per_symbol", 3),
        )

        # Portfolio tracking
        self._current_positions: dict[str, list[Position]] = {}
        self._daily_pnl: Decimal = Decimal(0)
        self._total_pnl: Decimal = Decimal(0)
        self._peak_value: Decimal = initial_capital
        self._current_value: Decimal = initial_capital

        # Risk metrics
        self._risk_violations: list[dict[str, Any]] = []
        self._rejected_orders: int = 0
        self._total_risk_checks: int = 0

        self.logger.info(
            "BacktestRiskManager initialized with capital: %s",
            initial_capital,
            extra={"source_module": self._source_module},
        )

    async def validate_order(self, order: Order) -> RiskCheckResult:
        """Validate order against risk limits."""
        self._total_risk_checks += 1

        # Check position size limits
        if order.quantity * order.limit_price > self.risk_limits.max_position_size:  # type: ignore[attr-defined]
            violation = {
                "type": "position_size_exceeded",
                "order_id": order.id,  # type: ignore[attr-defined]
                "amount": order.quantity * order.limit_price,  # type: ignore[attr-defined]
                "limit": self.risk_limits.max_position_size,
            }
            self._risk_violations.append(violation)
            self._rejected_orders += 1

            return RiskCheckResult(
                approved=False,
                reason=f"Position size {order.quantity * order.limit_price} exceeds limit {self.risk_limits.max_position_size}",  # type: ignore[attr-defined]
            )

        # Check concentration limits
        symbol_exposure = self._calculate_symbol_exposure(cast("str", order.trading_pair))
        new_exposure = symbol_exposure + (order.quantity * order.limit_price)  # type: ignore[attr-defined]
        concentration = new_exposure / self._current_value

        if concentration > self.risk_limits.max_concentration:
            violation = {
                "type": "concentration_exceeded",
                "symbol": order.trading_pair,
                "concentration": float(concentration),
                "limit": float(self.risk_limits.max_concentration),
            }
            self._risk_violations.append(violation)
            self._rejected_orders += 1

            return RiskCheckResult(
                approved=False,
                reason=f"Concentration {concentration:.2%} exceeds limit {self.risk_limits.max_concentration:.2%}",
            )

        # Check position count limits
        position_count = len(self._current_positions.get(cast("str", order.trading_pair), []))
        if position_count >= self.risk_limits.position_limit_per_symbol:
            violation = {
                "type": "position_count_exceeded",
                "symbol": order.trading_pair,
                "count": position_count,
                "limit": self.risk_limits.position_limit_per_symbol,
            }
            self._risk_violations.append(violation)
            self._rejected_orders += 1

            return RiskCheckResult(
                approved=False,
                reason=f"Position count {position_count} exceeds limit {self.risk_limits.position_limit_per_symbol}",
            )

        # Check daily loss limits
        if self._daily_pnl < -self.risk_limits.max_daily_loss:
            violation = {
                "type": "daily_loss_exceeded",
                "daily_pnl": float(self._daily_pnl),
                "limit": float(self.risk_limits.max_daily_loss),
            }
            self._risk_violations.append(violation)
            self._rejected_orders += 1

            return RiskCheckResult(
                approved=False,
                reason=f"Daily loss {self._daily_pnl} exceeds limit {self.risk_limits.max_daily_loss}",
            )

        # Check drawdown limits
        drawdown = (self._peak_value - self._current_value) / self._peak_value
        if drawdown > self.risk_limits.max_drawdown:
            violation = {
                "type": "drawdown_exceeded",
                "drawdown": float(drawdown),
                "limit": float(self.risk_limits.max_drawdown),
            }
            self._risk_violations.append(violation)
            self._rejected_orders += 1

            return RiskCheckResult(
                approved=False,
                reason=f"Drawdown {drawdown:.2%} exceeds limit {self.risk_limits.max_drawdown:.2%}",
            )

        return RiskCheckResult(approved=True)

    def update_portfolio_value(self, new_value: Decimal) -> None:
        """Update current portfolio value and related metrics."""
        self._current_value = new_value

        # Update peak value
        self._peak_value = max(self._peak_value, new_value)

        # Update P&L
        self._total_pnl = new_value - self.initial_capital

    def add_position(self, position: Position) -> None:
        """Add position to tracking."""
        symbol = position.trading_pair
        if symbol not in self._current_positions:
            self._current_positions[symbol] = []
        self._current_positions[symbol].append(position)

    def remove_position(self, position_id: str) -> None:
        """Remove position from tracking."""
        for symbol, positions in self._current_positions.items():
            self._current_positions[symbol] = [
                p for p in positions if p.id != position_id
            ]

    def _calculate_symbol_exposure(self, symbol: str) -> Decimal:
        """Calculate current exposure to a specific symbol."""
        positions = self._current_positions.get(symbol, [])
        total_exposure = Decimal(0)

        for position in positions:
            total_exposure += position.quantity * position.entry_price

        return total_exposure

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get comprehensive risk metrics."""
        current_drawdown = Decimal(0)
        if self._peak_value > 0:
            current_drawdown = (self._peak_value - self._current_value) / self._peak_value

        return {
            "current_value": float(self._current_value),
            "total_pnl": float(self._total_pnl),
            "daily_pnl": float(self._daily_pnl),
            "peak_value": float(self._peak_value),
            "current_drawdown": float(current_drawdown),
            "max_drawdown_limit": float(self.risk_limits.max_drawdown),
            "risk_violations": len(self._risk_violations),
            "rejected_orders": self._rejected_orders,
            "total_risk_checks": self._total_risk_checks,
            "approval_rate": (self._total_risk_checks - self._rejected_orders) / max(1, self._total_risk_checks),
        }


@dataclass
class BacktestSymbolInfo:
    """Symbol information for backtesting."""
    symbol: str
    base_asset: str
    quote_asset: str
    min_quantity: Decimal
    max_quantity: Decimal
    tick_size: Decimal
    lot_size: Decimal
    is_active: bool = True


class BacktestExchangeInfoService:
    """Production-ready exchange info service for backtesting."""

    def __init__(self, logger: logging.Logger, config_manager: ConfigManager) -> None:
        """Initialize the instance."""
        self.logger = logger
        self.config = config_manager
        self._source_module = self.__class__.__name__

        # Symbol information
        self._symbols: dict[str, BacktestSymbolInfo] = {}
        self._initialize_default_symbols()

        # Market hours (simplified - assuming 24/7 crypto)
        self._market_open_time = time(0, 0)  # Midnight
        self._market_close_time = time(23, 59)  # End of day

        # Trading status
        self._is_market_open = True
        self._maintenance_mode = False

        self.logger.info(
            "BacktestExchangeInfoService initialized with %d symbols",
            len(self._symbols),
            extra={"source_module": self._source_module},
        )

    def _initialize_default_symbols(self) -> None:
        """Initialize default symbols for backtesting."""
        default_symbols = [
            ("XRP/USD", "XRP", "USD"),
            ("DOGE/USD", "DOGE", "USD"),
            ("BTC/USD", "BTC", "USD"),
            ("ETH/USD", "ETH", "USD"),
        ]

        for symbol, base, quote in default_symbols:
            self._symbols[symbol] = BacktestSymbolInfo(
                symbol=symbol,
                base_asset=base,
                quote_asset=quote,
                min_quantity=Decimal("0.01"),
                max_quantity=Decimal(1000000),
                tick_size=Decimal("0.0001"),
                lot_size=Decimal("0.01"),
            )

    def get_symbol_info(self, symbol: str) -> BacktestSymbolInfo | None:
        """Get symbol information."""
        return self._symbols.get(symbol)

    def is_symbol_valid(self, symbol: str) -> bool:
        """Check if symbol is valid for trading."""
        symbol_info = self._symbols.get(symbol)
        return symbol_info is not None and symbol_info.is_active

    def get_all_symbols(self) -> list[str]:
        """Get list[Any] of all available symbols."""
        return [info.symbol for info in self._symbols.values() if info.is_active]

    def is_market_open(self, current_time: datetime | None = None) -> bool:
        """Check if market is open (simplified for 24/7 crypto)."""
        if self._maintenance_mode:
            return False
        return self._is_market_open

    def add_symbol(self, symbol_info: BacktestSymbolInfo) -> None:
        """Add new symbol for backtesting."""
        self._symbols[symbol_info.symbol] = symbol_info
        self.logger.info("Added symbol: %s", symbol_info.symbol, extra={"source_module": self._source_module})

    def set_maintenance_mode(self, enabled: bool) -> None:
        """Set maintenance mode status."""
        self._maintenance_mode = enabled
        self.logger.info("Maintenance mode: %s", "enabled" if enabled else "disabled", extra={"source_module": self._source_module})

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information."""
        return {
            "total_symbols": len(self._symbols),
            "active_symbols": sum(1 for info in self._symbols.values() if info.is_active),
            "market_open": self.is_market_open(),
            "maintenance_mode": self._maintenance_mode,
        }


class BacktestComponentFactory:
    """Factory for creating backtesting components."""

    @staticmethod
    def create_pubsub_manager(
        logger: logging.Logger,
        config: ConfigManager,
        simulation_start: datetime,
        simulation_end: datetime,
    ) -> BacktestPubSubManager:
        """Create backtesting PubSub manager."""
        return BacktestPubSubManager(logger, config, simulation_start, simulation_end)

    @staticmethod
    def create_risk_manager(
        logger: logging.Logger,
        config: ConfigManager,
        initial_capital: Decimal,
    ) -> BacktestRiskManager:
        """Create backtesting risk manager."""
        return BacktestRiskManager(logger, config, initial_capital)

    @staticmethod
    def create_exchange_info_service(
        logger: logging.Logger,
        config: ConfigManager,
    ) -> BacktestExchangeInfoService:
        """Create backtesting exchange info service."""
        return BacktestExchangeInfoService(logger, config)
