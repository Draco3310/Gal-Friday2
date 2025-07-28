"""Execution handler for managing order placement and lifecycle with Kraken exchange."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
import secrets
import time
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import aiohttp
import asyncio

# Removed incorrect import: from gal_friday.core.errors import ExecutionHandlerAuthenticationError
from gal_friday.core.events import (
    ClosePositionCommand,
    EventType,
    ExecutionReportEvent,
    TradeSignalApprovedEvent,
)
from gal_friday.dal.connection_pool import DatabaseConnectionPool
from gal_friday.dal.repositories.order_repository import OrderRepository
from gal_friday.execution.adapters import (
    BatchOrderRequest,
    ExecutionAdapter,
    KrakenExecutionAdapter,
    OrderRequest,
)
from gal_friday.execution.execution_config import ExecutionHandlerConfig
from gal_friday.execution.order_execution_strategies import OrderExecutionContext
from gal_friday.execution.order_persistence_service import OrderPersistenceService
from gal_friday.execution.websocket_client import KrakenWebSocketClient
from gal_friday.execution_handler_enhancements import (
    EnhancedMarketDataService,
)
from gal_friday.utils.kraken_api import generate_kraken_signature

from .interfaces.service_protocol import ServiceProtocol

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from gal_friday.config_manager import ConfigManager
    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.logger_service import LoggerService

# These types are not yet implemented, use Any for now
MonitoringService = Any
EventStore = Any

# Remove hardcoded URL - will be loaded from configuration


@dataclass
class ContingentOrderParamsRequest:
    """Parameters for preparing a contingent order (SL/TP).

    Attributes:
        pair_name: Kraken pair name (e.g., XXBTZUSD)
        order_side: "buy" or "sell"
        contingent_order_type: e.g., "stop-loss", "take-profit"
        trigger_price: The price at which the order triggers
        volume: The volume of the order
        pair_details: Exchange-provided info for the pair
        originating_signal_id: The ID of the originating signal
        log_marker: "SL" or "TP" for logging
        limit_price: For stop-loss-limit / take-profit-limit (optional)
    """

    pair_name: str
    order_side: str
    contingent_order_type: str
    trigger_price: Decimal
    volume: Decimal
    pair_details: dict[str, Any] | None
    originating_signal_id: UUID
    log_marker: str
    limit_price: Decimal | None = None


@dataclass
class OrderStatusReportParameters:
    """Parameters for handling and reporting order status.

    Attributes:
        exchange_order_id: The ID of the order on the exchange
        client_order_id: The client-side ID of the order
        signal_id: The ID of the signal that triggered the order (optional)
        order_data: The data of the order
        current_status: The current status of the order
        current_filled_qty: The currently filled quantity of the order
        avg_fill_price: The average fill price of the order (optional)
        commission: The commission of the order (optional)
    """

    exchange_order_id: str
    client_order_id: str
    signal_id: UUID | None
    order_data: dict[str, Any]
    current_status: str
    current_filled_qty: Decimal
    avg_fill_price: Decimal | None
    commission: Decimal | None


# Enterprise-Grade Enumerations and Data Classes
class OrderState(str, Enum):
    """Comprehensive order state enumeration."""

    CREATED = "created"
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    PENDING_CANCEL = "pending_cancel"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderPriority(str, Enum):
    """Order processing priority levels."""

    CRITICAL = "critical"  # Emergency liquidations, stop losses
    HIGH = "high"  # Market orders, time-sensitive trades
    NORMAL = "normal"  # Standard limit orders
    LOW = "low"  # Background rebalancing


class ShutdownOrderAction(str, Enum):
    """Shutdown order handling actions."""

    CANCEL_ALL = "cancel_all"
    CANCEL_CONDITIONAL = "cancel_conditional"
    LEAVE_OPEN = "leave_open"
    CONVERT_TO_MARKET = "convert_to_market"


@dataclass
class OrderStateEvent:
    """Order state change event."""

    order_id: str
    from_state: OrderState | None
    to_state: OrderState
    timestamp: datetime
    exchange_id: str | None = None
    fill_quantity: float | None = None
    fill_price: float | None = None
    reason: str | None = None


@dataclass
class OrderLifecycleData:
    """Complete order lifecycle information."""

    order_id: str
    current_state: OrderState
    state_history: list[OrderStateEvent]
    creation_time: datetime
    last_update_time: datetime
    exchange_order_id: str | None = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0


@dataclass
class ShutdownConfig:
    """Configuration for shutdown order handling."""

    default_action: ShutdownOrderAction
    action_by_order_type: dict[str, ShutdownOrderAction]
    action_by_strategy: dict[str, ShutdownOrderAction]
    safety_checks_enabled: bool = True
    max_cancellation_time: int = 30  # seconds
    require_confirmation: bool = False
    preserve_stop_losses: bool = True
    market_hours_only: bool = True


@dataclass
class OrderCancellationResult:
    """Result of order cancellation attempt."""

    order_id: str
    symbol: str
    action_taken: ShutdownOrderAction
    success: bool
    reason: str
    timestamp: datetime
    market_impact_estimate: float


@dataclass
class AsyncOrderRequest:
    """Asynchronous order request with metadata."""

    order_request: dict[str, Any]
    priority: OrderPriority
    callback: Callable[..., Any] | None = None
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BatchOperation:
    """Batch operation definition."""

    operation_id: str
    operation_type: str
    orders: list[AsyncOrderRequest]
    progress_callback: Callable[..., Any] | None = None
    max_concurrent: int = 10
    timeout_seconds: float = 300.0


# Enterprise Exception Classes
class OrderStateError(Exception):
    """Order state tracking errors."""


class BatchProcessingError(Exception):
    """Batch processing errors."""


class OrderProcessingError(Exception):
    """Order processing errors."""


class RateLimitTracker:
    """Tracks and enforces API rate limits to prevent exceeding exchange limits.

    Attributes:
        config: The configuration manager
        logger: The logger service
        private_calls_per_second: The number of private calls allowed per second
        public_calls_per_second: The number of public calls allowed per second
        window_size: The size of the rate limit window in seconds
        _private_call_timestamps: The timestamps of recent private calls
        _public_call_timestamps: The timestamps of recent public calls
    """

    def __init__(self, config: ConfigManager, logger_service: LoggerService) -> None:
        """Initialize the rate limit tracker with configuration and logger."""
        self.config = config
        self.logger: LoggerService = logger_service

        # Configure rate limits based on tier/API key level
        # These should come from configuration
        self.private_calls_per_second = self.config.get_int("exchange.rate_limit.private_calls_per_second", 1)
        self.public_calls_per_second = self.config.get_int("exchange.rate_limit.public_calls_per_second", 1)

        # Tracking timestamps of recent calls
        self._private_call_timestamps: list[float] = []
        self._public_call_timestamps: list[float] = []

        # Window size in seconds for tracking
        self.window_size = 1.0  # 1 second window

        self._source_module = self.__class__.__name__

    async def wait_for_private_capacity(self) -> None:
        """Wait until there's capacity to make a private API call.

        Uses self-regulating approach by pruning old timestamps and waiting if needed.
        """
        while True:
            current_time = time.time()

            # Prune timestamps older than the window
            self._private_call_timestamps = [
                ts for ts in self._private_call_timestamps if current_time - ts < self.window_size
            ]

            # Check if we're below the limit
            if len(self._private_call_timestamps) < self.private_calls_per_second:
                # We have capacity, add current time and proceed
                self._private_call_timestamps.append(current_time)
                return

            # No capacity, wait a bit and try again
            sleep_time = 0.05  # 50ms
            await asyncio.sleep(sleep_time)

    async def wait_for_public_capacity(self) -> None:
        """Wait until there's capacity to make a public API call.

        Similar to private capacity but uses public limits.
        """
        while True:
            current_time = time.time()

            # Prune timestamps older than the window
            self._public_call_timestamps = [
                ts for ts in self._public_call_timestamps if current_time - ts < self.window_size
            ]

            # Check if we're below the limit
            if len(self._public_call_timestamps) < self.public_calls_per_second:
                # We have capacity, add current time and proceed
                self._public_call_timestamps.append(current_time)
                return

            # No capacity, wait a bit and try again
            sleep_time = 0.05  # 50ms
            await asyncio.sleep(sleep_time)

    def reset(self) -> None:
        """Reset all tracking."""
        self._private_call_timestamps = []
        self._public_call_timestamps = []


# Order State Tracking System
class OrderStateTracker:
    """Enterprise-grade order state tracking and lifecycle management."""

    def __init__(self, persistence_service: Any, event_publisher: Any, config: dict[str, Any]) -> None:
        """Initialize the instance."""
        self.persistence = persistence_service
        self.event_publisher = event_publisher
        self.config = config
        self.logger: Any = None  # Will be set by the execution handler (LoggerService)
        self.order_states: dict[str, OrderLifecycleData] = {}

    def _raise_order_not_found(self, order_id: str) -> None:
        """Raise OrderStateError for order not found."""
        raise OrderStateError(f"Order {order_id} not found")

    def _raise_invalid_transition(self, order_id: str, current_state: OrderState, new_state: OrderState) -> None:
        """Raise OrderStateError for invalid state transition."""
        raise OrderStateError(
            f"Invalid state transition for {order_id}: {current_state.value} -> {new_state.value}",
        )

    def _raise_order_state_null_after_loading(self, order_id: str) -> None:
        """Raise OrderStateError for null state after loading."""
        raise OrderStateError(f"Order {order_id} state unexpectedly None after loading")

    def _raise_order_state_unexpectedly_null(self, order_id: str) -> None:
        """Raise OrderStateError for unexpectedly null state."""
        raise OrderStateError(f"Order {order_id} state is unexpectedly None")

    def _raise_lifecycle_data_unexpectedly_null(self, order_id: str) -> None:
        """Raise OrderStateError for unexpectedly null lifecycle data."""
        raise OrderStateError(f"Order {order_id} lifecycle data is unexpectedly None")

    async def create_order_tracking(
        self,
        order_id: str,
        initial_data: dict[str, Any],
    ) -> OrderLifecycleData:
        """Create order state tracking for new order."""
        try:
            if self.logger:
                self.logger.debug(f"Creating order tracking for {order_id}")

            lifecycle_data = OrderLifecycleData(
                order_id=order_id,
                current_state=OrderState.CREATED,
                state_history=[],
                creation_time=datetime.now(UTC),
                last_update_time=datetime.now(UTC),
                remaining_quantity=initial_data.get("quantity", 0.0),
            )

            initial_event = OrderStateEvent(
                order_id=order_id,
                from_state=None,
                to_state=OrderState.CREATED,
                timestamp=lifecycle_data.creation_time,
            )

            lifecycle_data.state_history.append(initial_event)
            self.order_states[order_id] = lifecycle_data

            # Persist and publish
            await self._persist_order_state(lifecycle_data)
            await self._publish_state_event("order.created", initial_event)

            if self.logger:
                self.logger.info(f"Order tracking created for {order_id}")

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error creating order tracking for {order_id}: ")
            raise OrderStateError(f"Failed to create order tracking: {e}") from e
        else:
            return lifecycle_data

    async def update_order_state(
        self,
        order_id: str,
        new_state: OrderState,
        **kwargs: Any,
    ) -> OrderLifecycleData:
        """Update order state with validation and event publishing."""
        try:
            # Load or get order state
            lifecycle_data = await self._load_or_get_order_state(order_id)
            current_state = lifecycle_data.current_state

            # Validate state transition
            self._validate_state_transition(order_id, current_state, new_state)

            # Create state event
            state_event = self._create_state_event(order_id, current_state, new_state, **kwargs)

            # Update lifecycle data
            self._update_lifecycle_data(lifecycle_data, new_state, state_event)

            # Persist and publish
            await self._persist_and_publish_state(lifecycle_data, state_event, current_state, new_state)

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error updating order state for {order_id}: ")
            raise OrderStateError(f"Failed to update order state: {e}") from e
        else:
            return lifecycle_data

    def _is_valid_transition(self, from_state: OrderState, to_state: OrderState) -> bool:
        """Validate order state transitions."""
        valid_transitions = {
            OrderState.CREATED: {OrderState.PENDING_SUBMIT, OrderState.FAILED},
            OrderState.PENDING_SUBMIT: {OrderState.SUBMITTED, OrderState.FAILED},
            OrderState.SUBMITTED: {
                OrderState.ACKNOWLEDGED,
                OrderState.REJECTED,
                OrderState.FAILED,
            },
            OrderState.ACKNOWLEDGED: {OrderState.PENDING, OrderState.REJECTED, OrderState.FAILED},
            OrderState.PENDING: {
                OrderState.PARTIALLY_FILLED,
                OrderState.FILLED,
                OrderState.PENDING_CANCEL,
                OrderState.CANCELLED,
                OrderState.REJECTED,
                OrderState.EXPIRED,
                OrderState.FAILED,
            },
            OrderState.PARTIALLY_FILLED: {
                OrderState.FILLED,
                OrderState.PENDING_CANCEL,
                OrderState.CANCELLED,
                OrderState.EXPIRED,
                OrderState.FAILED,
            },
            OrderState.PENDING_CANCEL: {
                OrderState.CANCELLED,
                OrderState.FILLED,
                OrderState.FAILED,
            },
            # Terminal states
            OrderState.FILLED: set[Any](),
            OrderState.CANCELLED: set[Any](),
            OrderState.REJECTED: set[Any](),
            OrderState.EXPIRED: set[Any](),
            OrderState.FAILED: set[Any](),
        }

        return to_state in valid_transitions.get(from_state, set())

    async def get_order_state(self, order_id: str) -> OrderLifecycleData | None:
        """Get current order state and lifecycle data."""
        if order_id in self.order_states:
            return self.order_states[order_id]

        # Try to load from persistence
        lifecycle_data = await self._load_order_state(order_id)
        if lifecycle_data:
            self.order_states[order_id] = lifecycle_data

        return lifecycle_data

    async def persist_order_state(self, lifecycle_data: OrderLifecycleData) -> None:
        """Persist order state to storage.

        Args:
            lifecycle_data: Order lifecycle data to persist
        """
        await self._persist_order_state(lifecycle_data)

    async def load_order_state(self, order_id: str) -> OrderLifecycleData | None:
        """Load order state from persistence.

        Args:
            order_id: Order ID to load

        Returns:
            Order lifecycle data if found, None otherwise
        """
        return await self._load_order_state(order_id)

    async def _persist_order_state(self, lifecycle_data: OrderLifecycleData) -> None:
        """Persist order state to storage using enterprise-grade DAL."""
        try:
            if self.persistence:
                # Create order state record for database
                order_state_data = {
                    "order_id": lifecycle_data.order_id,
                    "current_state": lifecycle_data.current_state.value,
                    "exchange_order_id": lifecycle_data.exchange_order_id,
                    "filled_quantity": lifecycle_data.filled_quantity,
                    "remaining_quantity": lifecycle_data.remaining_quantity,
                    "creation_time": lifecycle_data.creation_time,
                    "last_update_time": lifecycle_data.last_update_time,
                    "state_history": [
                        {
                            "from_state": event.from_state.value if event.from_state else None,
                            "to_state": event.to_state.value,
                            "timestamp": event.timestamp.isoformat(),
                            "exchange_id": event.exchange_id,
                            "fill_quantity": event.fill_quantity,
                            "fill_price": event.fill_price,
                            "reason": event.reason,
                        }
                        for event in lifecycle_data.state_history
                    ],
                }

                await self.persistence.save_order_state(lifecycle_data.order_id, order_state_data)

                if self.logger:
                    self.logger.debug(
                        f"Persisted order state for order {lifecycle_data.order_id} "
                        f"with state {lifecycle_data.current_state.value}",
                        source_module=self.__class__.__name__,
                    )
        except Exception:
            if self.logger:
                self.logger.exception(
                    f"Failed to persist order state for order {lifecycle_data.order_id}: ",
                    source_module=self.__class__.__name__,
                )

    async def _load_order_state(self, order_id: str) -> OrderLifecycleData | None:
        """Load order state from persistence using enterprise-grade DAL."""
        try:
            if self.persistence:
                order_state_data = await self.persistence.load_order_state(order_id)

                if order_state_data:
                    # Reconstruct state history
                    state_history = []
                    for event_data in order_state_data.get("state_history", []):
                        from_state = OrderState(event_data["from_state"]) if event_data["from_state"] else None
                        to_state = OrderState(event_data["to_state"])
                        timestamp = datetime.fromisoformat(event_data["timestamp"])

                        state_event = OrderStateEvent(
                            order_id=order_id,
                            from_state=from_state,
                            to_state=to_state,
                            timestamp=timestamp,
                            exchange_id=event_data.get("exchange_id"),
                            fill_quantity=event_data.get("fill_quantity"),
                            fill_price=event_data.get("fill_price"),
                            reason=event_data.get("reason"),
                        )
                        state_history.append(state_event)

                    # Reconstruct OrderLifecycleData
                    lifecycle_data = OrderLifecycleData(
                        order_id=order_id,
                        current_state=OrderState(order_state_data["current_state"]),
                        state_history=state_history,
                        creation_time=order_state_data["creation_time"],
                        last_update_time=order_state_data["last_update_time"],
                        exchange_order_id=order_state_data.get("exchange_order_id"),
                        filled_quantity=float(order_state_data.get("filled_quantity", 0.0)),
                        remaining_quantity=float(order_state_data.get("remaining_quantity", 0.0)),
                    )

                    if self.logger:
                        self.logger.debug(
                            f"Loaded order state for order {order_id} with state {lifecycle_data.current_state.value}",
                            source_module=self.__class__.__name__,
                        )

                    return lifecycle_data

        except Exception:
            if self.logger:
                self.logger.exception(
                    f"Failed to load order state for order {order_id}: ",
                    source_module=self.__class__.__name__,
                )
            return None
        else:
            return None

    async def _publish_state_event(self, event_type: str, event_data: OrderStateEvent) -> None:
        """Publish state change event."""
        try:
            if self.event_publisher:
                await self.event_publisher.publish(
                    event_type,
                    {
                        "order_id": event_data.order_id,
                        "from_state": event_data.from_state.value if event_data.from_state else None,
                        "to_state": event_data.to_state.value,
                        "timestamp": event_data.timestamp.isoformat(),
                    },
                )
        except Exception:
            if self.logger:
                self.logger.exception("Failed to publish state event: ")

    async def _load_or_get_order_state(self, order_id: str) -> OrderLifecycleData:
        """Load or get order state with validation."""
        if order_id not in self.order_states:
            lifecycle_data = await self._load_order_state(order_id)
            if not lifecycle_data:
                self._raise_order_not_found(order_id)
            # MyPy knows lifecycle_data is not None after the check above
            if lifecycle_data is None:
                raise RuntimeError("Order lifecycle data unexpectedly None")
            self.order_states[order_id] = lifecycle_data
        else:
            lifecycle_data = self.order_states[order_id]
            # Defensive check - this should never be None due to the logic above
            if lifecycle_data is None:
                raise RuntimeError("Order lifecycle data unexpectedly None")
        return lifecycle_data

    def _validate_state_transition(self, order_id: str, current_state: OrderState, new_state: OrderState) -> None:
        """Validate that the state transition is allowed."""
        if not self._is_valid_transition(current_state, new_state):
            self._raise_invalid_transition(order_id, current_state, new_state)

    def _create_state_event(
        self,
        order_id: str,
        current_state: OrderState,
        new_state: OrderState,
        **kwargs: Any,
    ) -> OrderStateEvent:
        """Create order state event with provided data."""
        return OrderStateEvent(
            order_id=order_id,
            from_state=current_state,
            to_state=new_state,
            timestamp=datetime.now(UTC),
            exchange_id=kwargs.get("exchange_id"),
            fill_quantity=kwargs.get("fill_quantity"),
            fill_price=kwargs.get("fill_price"),
            reason=kwargs.get("reason"),
        )

    def _update_lifecycle_data(
        self,
        lifecycle_data: OrderLifecycleData,
        new_state: OrderState,
        state_event: OrderStateEvent,
    ) -> None:
        """Update lifecycle data with new state and fill information."""
        # Defensive check
        if lifecycle_data is None:
            raise RuntimeError("Order lifecycle data unexpectedly None")

        lifecycle_data.current_state = new_state
        lifecycle_data.last_update_time = state_event.timestamp
        lifecycle_data.state_history.append(state_event)

        # Update fill information if applicable
        if state_event.fill_quantity:
            lifecycle_data.filled_quantity += state_event.fill_quantity
            lifecycle_data.remaining_quantity -= state_event.fill_quantity

    async def _persist_and_publish_state(
        self,
        lifecycle_data: OrderLifecycleData,
        state_event: OrderStateEvent,
        current_state: OrderState,
        new_state: OrderState,
    ) -> None:
        """Persist lifecycle data and publish state change event."""
        await self._persist_order_state(lifecycle_data)
        await self._publish_state_event("order.state_changed", state_event)

        if self.logger:
            self.logger.info(
                f"Order {state_event.order_id} state updated: {current_state.value} -> {new_state.value}",
            )


# Configurable Shutdown Handler
class ConfigurableShutdownHandler:
    """Handles configurable order cancellation during shutdown."""

    def __init__(self, config: ShutdownConfig, execution_handler: Any, market_data_service: Any = None) -> None:
        """Initialize the instance."""
        self.config = config
        self.execution_handler = execution_handler
        self.market_data = market_data_service
        self.logger = execution_handler.logger
        self.shutdown_in_progress = False

    async def handle_shutdown_orders(self) -> list[OrderCancellationResult]:
        """Handle open orders during shutdown with configurable logic."""
        if self.shutdown_in_progress:
            self.logger.warning("Shutdown already in progress, skipping duplicate call")
            return []

        self.shutdown_in_progress = True
        shutdown_start = datetime.now(UTC)

        try:
            self.logger.info("Starting configurable order shutdown process")

            # Get all open orders from execution handler
            open_orders = []
            if hasattr(self.execution_handler, "get_open_orders"):
                order_map_data = self.execution_handler.get_open_orders()
                open_orders.extend(
                    [
                        {
                            "order_id": order_info["order_id"],
                            "client_order_id": order_info["client_order_id"],
                            "symbol": "UNKNOWN",  # Would need to enhance to track symbols
                            "order_type": "UNKNOWN",
                        }
                        for order_info in order_map_data
                    ],
                )

            if not open_orders:
                self.logger.info("No open orders to handle during shutdown")
                return []

            self.logger.info(f"Found {len(open_orders)} open orders to evaluate for shutdown")

            # Perform safety checks
            safety_check_passed = await self._perform_safety_checks(open_orders)

            if not safety_check_passed and self.config.safety_checks_enabled:
                self.logger.exception("Safety checks failed, aborting order cancellation")
                return self._create_safety_failure_results(open_orders)

            # Process each order according to configuration
            cancellation_results = []

            for order in open_orders:
                try:
                    result = await self._process_order_for_shutdown(order)
                    cancellation_results.append(result)

                except Exception as e:
                    self.logger.exception(f"Error processing order {order.get('order_id')}: ")
                    cancellation_results.append(self._create_error_result(order, str(e)))

            # Log shutdown summary
            await self._log_shutdown_summary(cancellation_results, shutdown_start)

            # Wait for cancellations to complete
            await self._wait_for_cancellation_completion(cancellation_results)

        except Exception:
            self.logger.exception("Critical error in shutdown order handling: ")
            return []
        else:
            return cancellation_results

        finally:
            self.shutdown_in_progress = False

    async def _perform_safety_checks(self, open_orders: list[dict[str, Any]]) -> bool:
        """Perform comprehensive safety checks before order cancellation."""
        try:
            # Check for critical stop losses if preservation is enabled
            if self.config.preserve_stop_losses:
                stop_loss_orders = [
                    order for order in open_orders if order.get("order_type", "").lower() in ["stop-loss", "stop_loss"]
                ]
                if stop_loss_orders:
                    self.logger.warning(
                        f"Found {len(stop_loss_orders)} stop loss orders, preserving as per config",
                    )
                    return False

        except Exception:
            self.logger.exception("Error in safety checks: ")
            return False
        else:
            return True

    def _create_safety_failure_results(
        self,
        open_orders: list[dict[str, Any]],
    ) -> list[OrderCancellationResult]:
        """Create results for safety check failures."""
        return [
            OrderCancellationResult(
                order_id=order.get("order_id", "unknown"),
                symbol=order.get("symbol", "unknown"),
                action_taken=ShutdownOrderAction.LEAVE_OPEN,
                success=False,
                reason="Safety checks failed",
                timestamp=datetime.now(UTC),
                market_impact_estimate=0.0,
            )
            for order in open_orders
        ]

    def _create_error_result(
        self,
        order: dict[str, Any],
        error_msg: str,
    ) -> OrderCancellationResult:
        """Create error result for order processing failure."""
        return OrderCancellationResult(
            order_id=order.get("order_id", "unknown"),
            symbol=order.get("symbol", "unknown"),
            action_taken=ShutdownOrderAction.LEAVE_OPEN,
            success=False,
            reason=f"Processing error: {error_msg}",
            timestamp=datetime.now(UTC),
            market_impact_estimate=0.0,
        )

    async def _process_order_for_shutdown(self, order: dict[str, Any]) -> OrderCancellationResult:
        """Process individual order according to shutdown configuration."""
        order_id = order.get("order_id", "unknown")
        symbol = order.get("symbol", "unknown")
        order_type = order.get("order_type", "unknown")

        # Determine action based on configuration
        action = self._determine_shutdown_action(order_type, None)

        success = False
        reason = "Unknown error"

        try:
            if action == ShutdownOrderAction.CANCEL_ALL:
                # Attempt to cancel the order
                success = await self.execution_handler.cancel_order(order_id)
                reason = "Cancelled as per shutdown configuration"

            elif action == ShutdownOrderAction.CANCEL_CONDITIONAL:
                # Enhanced conditional cancellation with market condition checks
                should_cancel, cancel_reason = await self._evaluate_conditional_cancellation(order)

                if should_cancel:
                    success = await self.execution_handler.cancel_order(order_id)
                    reason = f"Conditionally cancelled: {cancel_reason}"
                else:
                    success = True  # Successfully decided to leave open
                    reason = f"Left open due to market conditions: {cancel_reason}"

            elif action == ShutdownOrderAction.LEAVE_OPEN:
                success = True
                reason = "Left open as per configuration"

            elif action == ShutdownOrderAction.CONVERT_TO_MARKET:
                # Enhanced convert to market with proper order replacement
                success, reason = await self._convert_order_to_market(order)
                if not success:
                    # Fallback to cancellation if conversion fails
                    cancel_success = await self.execution_handler.cancel_order(order_id)
                    reason = f"Conversion failed, cancelled instead: {reason}"
                    success = cancel_success

            # No else clause needed - all enum values are covered

            return OrderCancellationResult(
                order_id=order_id,
                symbol=symbol,
                action_taken=action,
                success=success,
                reason=reason,
                timestamp=datetime.now(UTC),
                market_impact_estimate=0.0,
            )

        except Exception as e:
            self.logger.exception(f"Error processing order {order_id}: ")
            return OrderCancellationResult(
                order_id=order_id,
                symbol=symbol,
                action_taken=action,
                success=False,
                reason=f"Error: {e}",
                timestamp=datetime.now(UTC),
                market_impact_estimate=0.0,
            )

    def _determine_shutdown_action(
        self,
        order_type: str,
        strategy_id: str | None,
    ) -> ShutdownOrderAction:
        """Determine shutdown action based on configuration hierarchy."""
        # Strategy-specific configuration takes highest priority
        if strategy_id and strategy_id in self.config.action_by_strategy:
            return self.config.action_by_strategy[strategy_id]

        # Order type-specific configuration
        if order_type in self.config.action_by_order_type:
            return self.config.action_by_order_type[order_type]

        # Default action
        return self.config.default_action

    async def _log_shutdown_summary(
        self,
        results: list[OrderCancellationResult],
        start_time: datetime,
    ) -> None:
        """Log summary of shutdown operations."""
        duration = (datetime.now(UTC) - start_time).total_seconds()
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        self.logger.info(
            f"Shutdown completed in {duration:.2f}s. Success: {successful}, Failed: {failed}",
        )

    async def _wait_for_cancellation_completion(
        self,
        results: list[OrderCancellationResult],
    ) -> None:
        """Wait for cancellation operations to complete."""
        cancellation_attempts = [
            r
            for r in results
            if r.action_taken in [ShutdownOrderAction.CANCEL_ALL, ShutdownOrderAction.CANCEL_CONDITIONAL]
        ]

        if not cancellation_attempts:
            return

        max_wait_time = self.config.max_cancellation_time

        self.logger.info(
            f"Waiting up to {max_wait_time}s for {len(cancellation_attempts)} cancellations to complete",
        )

        # Enhanced to actually check order status
        try:
            await self._verify_cancellation_completion(cancellation_attempts, max_wait_time)
        except Exception:
            self.logger.exception("Error verifying cancellation completion: ")
            # Fallback to simple wait
            await asyncio.sleep(min(max_wait_time, 5))

    async def _evaluate_conditional_cancellation(self, order: dict[str, Any]) -> tuple[bool, str]:
        """Evaluate whether an order should be conditionally cancelled based on market conditions."""
        try:
            order_id = order.get("order_id", "unknown")
            symbol = order.get("symbol", "unknown")
            order_type = order.get("order_type", "unknown")

            # Check if market data service is available
            if not self.market_data:
                return True, "No market data available, defaulting to cancel"

            # Get current market conditions
            market_conditions = await self._get_market_conditions(symbol)

            # Evaluate based on order type using dispatch
            evaluation_result = await self._evaluate_order_cancellation_by_type(
                order_id,
                symbol,
                order_type,
                market_conditions,
            )

        except Exception as e:
            self.logger.exception(
                f"Error evaluating conditional cancellation for order {order.get('order_id')}: ",
            )
            # Conservative approach: don't cancel if we can't evaluate properly
            return False, f"Evaluation error: {e}"
        else:
            return evaluation_result

    async def _get_market_conditions(self, symbol: str) -> dict[str, Any]:
        """Get current market conditions for a symbol."""
        return {
            "current_price": await self.market_data.get_current_price(symbol),
            "is_market_open": await self.market_data.is_market_open(symbol),
            "volatility": await self.market_data.get_volatility(symbol),
        }

    async def _evaluate_order_cancellation_by_type(
        self,
        order_id: str,
        symbol: str,
        order_type: str,
        market_conditions: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate cancellation based on order type and market conditions."""
        # Market closed protection
        if not market_conditions["is_market_open"]:
            self.logger.info(f"Market closed for {symbol}, preserving order {order_id}")
            return False, "Market closed - preserving order"

        order_type_lower = order_type.lower()

        # Order type specific evaluations
        evaluators = {
            "limit": self._evaluate_limit_order_cancellation,
            "stop-loss": self._evaluate_stop_loss_cancellation,
            "stop_loss": self._evaluate_stop_loss_cancellation,
            "take-profit": self._evaluate_take_profit_cancellation,
            "take_profit": self._evaluate_take_profit_cancellation,
        }

        evaluator = evaluators.get(order_type_lower)
        if evaluator:
            return await evaluator(order_id, symbol, market_conditions)

        # Default behavior for other order types
        return True, "Standard conditional cancellation"

    async def _evaluate_limit_order_cancellation(
        self,
        order_id: str,
        symbol: str,
        market_conditions: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate limit order cancellation based on volatility."""
        volatility = market_conditions.get("volatility")
        if volatility and volatility > 0.05:  # 5% volatility threshold
            self.logger.info(
                f"High volatility ({volatility:.2%}) for {symbol}, preserving limit order {order_id}",
            )
            return False, f"High volatility ({volatility:.2%}) - preserving limit order"
        return True, "Standard conditional cancellation"

    async def _evaluate_stop_loss_cancellation(
        self,
        order_id: str,
        symbol: str,
        market_conditions: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate stop-loss order cancellation - never cancel for risk management."""
        self.logger.info(f"Stop-loss order {order_id} preserved for risk management")
        return False, "Stop-loss order preserved for risk management"

    async def _evaluate_take_profit_cancellation(
        self,
        order_id: str,
        symbol: str,
        market_conditions: dict[str, Any],
    ) -> tuple[bool, str]:
        """Evaluate take-profit order cancellation based on current price."""
        current_price = market_conditions.get("current_price")
        if current_price is not None:
            # Enhanced take-profit evaluation with price comparison
            should_cancel, reason = await self._evaluate_take_profit_order(
                order_id,
                current_price,
                symbol,
            )
            return should_cancel, reason
        return True, "No current price available"

    async def _verify_cancellation_completion(
        self,
        cancellation_results: list[OrderCancellationResult],
        max_wait_time: int,
    ) -> None:
        """Verify that cancellation requests have been processed."""
        start_time = time.time()
        check_interval = 1.0  # Check every second

        pending_cancellations = [r for r in cancellation_results if r.success]

        while pending_cancellations and (time.time() - start_time) < max_wait_time:
            verified_cancellations = []

            for result in pending_cancellations:
                try:
                    # Check order status via execution handler
                    order_details = await self.execution_handler.query_order_details(
                        result.order_id,
                    )

                    if order_details:
                        status = order_details.get("status", "").lower()
                        if status in ["canceled", "cancelled", "closed"]:
                            verified_cancellations.append(result)
                            self.logger.debug(f"Verified cancellation of order {result.order_id}")

                except Exception as e:
                    self.logger.debug(f"Could not verify cancellation of {result.order_id}: {e}")

            # Remove verified cancellations from pending list[Any]
            for verified in verified_cancellations:
                if verified in pending_cancellations:
                    pending_cancellations.remove(verified)

            if pending_cancellations:
                await asyncio.sleep(check_interval)

        if pending_cancellations:
            self.logger.warning(
                f"{len(pending_cancellations)} cancellations still pending after {max_wait_time}s",
            )

    async def _convert_order_to_market(self, order: dict[str, Any]) -> tuple[bool, str]:
        """Convert a limit order to market order for immediate execution."""
        try:
            # Validate order type
            validation_result = self._validate_order_for_conversion(order)
            if not validation_result[0]:
                return validation_result

            order_id = order.get("order_id", "unknown")
            symbol = order.get("symbol", "unknown")

            # Get and validate original order details
            order_info = await self._get_order_conversion_info(order_id, symbol)
            if not order_info:
                return False, "Could not retrieve or validate order information"

            # Check market conditions
            market_check = await self._check_market_conditions_for_conversion(symbol, order_info)
            if not market_check[0]:
                return market_check

            # Execute the conversion
            return await self._execute_order_conversion(order_id, order_info)

        except Exception as e:
            self.logger.exception(f"Error converting order {order.get('order_id')} to market: ")
            return False, f"Conversion error: {e}"

    def _validate_order_for_conversion(self, order: dict[str, Any]) -> tuple[bool, str]:
        """Validate if order can be converted to market order."""
        order_type = order.get("order_type", "unknown")
        if order_type.lower() != "limit":
            return False, f"Cannot convert {order_type} order to market"
        return True, ""

    async def _get_order_conversion_info(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        """Get and validate order information for conversion."""
        original_order = await self.execution_handler.query_order_details(order_id)
        if not original_order:
            return None

        # Extract order information
        order_info = {
            "pair": original_order.get("descr", {}).get("pair", symbol),
            "side": original_order.get("descr", {}).get("type", "unknown"),
            "volume": original_order.get("vol", "0"),
        }

        # Validate all required fields are present
        if not all(order_info.values()):
            return None

        return order_info

    async def _check_market_conditions_for_conversion(
        self,
        symbol: str,
        order_info: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check market conditions before conversion."""
        if not self.market_data:
            return True, ""  # No market data service, proceed anyway

        # Check if market is open
        is_market_open = await self.market_data.is_market_open(symbol)
        if not is_market_open:
            return False, "Market closed - cannot execute market order"

        # Check market impact for large orders
        await self._log_market_impact_warning(symbol, order_info)

        return True, ""

    async def _log_market_impact_warning(self, symbol: str, order_info: dict[str, Any]) -> None:
        """Log warning for large market orders."""
        current_price = await self.market_data.get_current_price(symbol)
        if current_price:
            volume_float = float(order_info["volume"])
            order_value = volume_float * current_price
            if order_value > 10000:  # $10k threshold
                self.logger.warning(
                    f"Large market order conversion: {volume_float} {order_info['pair']} worth ~${order_value:.0f}",
                )

    async def _execute_order_conversion(self, order_id: str, order_info: dict[str, Any]) -> tuple[bool, str]:
        """Execute the order conversion (cancel limit, place market)."""
        # Step 1: Cancel the original limit order
        cancel_success = await self.execution_handler.cancel_order(order_id)
        if not cancel_success:
            return False, "Failed to cancel original order"

        # Step 2: Prepare and place market order
        market_order_params = {
            "pair": order_info["pair"],
            "type": order_info["side"],
            "ordertype": "market",
            "volume": order_info["volume"],
            "userref": f"convert_{int(time.time())}",
            "validate": "false",
        }

        self.logger.info(f"Converting limit order {order_id} to market order for immediate execution")

        result = await self.execution_handler.make_private_request_with_retry(
            "/0/private/AddOrder",
            market_order_params,
        )

        return self._process_market_order_result(result, order_id)

    def _process_market_order_result(self, result: dict[str, Any] | None, original_order_id: str) -> tuple[bool, str]:
        """Process the result of market order placement."""
        if result and not result.get("error"):
            kraken_result = result.get("result", {})
            new_txids = kraken_result.get("txid", [])

            if new_txids:
                new_order_id = new_txids[0] if isinstance(new_txids, list) else new_txids
                self.logger.info(
                    f"Successfully converted order {original_order_id} to market order {new_order_id}",
                )
                return True, f"Converted to market order {new_order_id}"
            return False, "Market order placement returned no order ID"

        error_msg = result.get("error", ["Unknown error"]) if result else ["API request failed"]
        return False, f"Market order placement failed: {error_msg}"

    async def _evaluate_take_profit_order(
        self,
        order_id: str,
        current_price: float,
        symbol: str,
    ) -> tuple[bool, str]:
        """Evaluate take-profit order based on current market price and order details."""
        try:
            # Get order details with early return
            order_details = await self.execution_handler.query_order_details(order_id)
            if not order_details:
                return True, "Could not retrieve order details, defaulting to cancel"

            # Extract order price with early return
            order_price = self._extract_order_price(order_details)
            if order_price is None:
                self.logger.warning(f"Could not determine price for take-profit order {order_id}")
                return False, "Unknown order price - preserving order for safety"

            # Get order side
            order_side = order_details.get("descr", {}).get("type", "").lower()

            # Use dispatch table for side-specific evaluation
            side_evaluators = {
                "sell": self._evaluate_sell_take_profit,
                "buy": self._evaluate_buy_take_profit,
            }

            evaluator = side_evaluators.get(order_side)
            if evaluator:
                result = evaluator(order_id, current_price, order_price)
                if result[0] is not None:  # Definitive decision made
                    return (result[0], result[1])

            # Check time-based factors
            time_result = await self._evaluate_order_age(order_details, current_price, order_price)
            if time_result[0] is not None:
                return (time_result[0], time_result[1])

            # Default evaluation based on price difference
            return self._evaluate_price_difference(current_price, order_price)

        except Exception as e:
            self.logger.exception(f"Error evaluating take-profit order {order_id}: ")
            # Conservative approach: don't cancel if we can't evaluate properly
            return False, f"Evaluation error: {e}"

    def _extract_order_price(self, order_details: dict[str, Any]) -> float | None:
        """Extract order price from order details."""
        # Try to get price from order description
        descr = order_details.get("descr", {})
        if "price" in descr:
            with contextlib.suppress(ValueError, TypeError):
                return float(descr["price"])

        # Try to get price from order data
        if "price" in order_details:
            with contextlib.suppress(ValueError, TypeError):
                return float(order_details["price"])

        return None

    def _evaluate_sell_take_profit(
        self,
        order_id: str,
        current_price: float,
        order_price: float,
    ) -> tuple[bool | None, str]:
        """Evaluate sell-side take-profit order."""
        # Within 5% of target - preserve
        if current_price >= order_price * 0.95:
            self.logger.info(
                f"Take-profit sell order {order_id} close to target: "
                f"current ${current_price:.2f} vs target ${order_price:.2f}",
            )
            return False, f"Close to target price (${order_price:.2f}) - preserving order"

        # More than 15% below target - cancel
        if current_price <= order_price * 0.85:
            self.logger.info(
                f"Take-profit sell order {order_id} far from target: "
                f"current ${current_price:.2f} vs target ${order_price:.2f}",
            )
            return True, f"Far from target price (${order_price:.2f}) - cancelling order"

        # No definitive decision
        return None, ""

    def _evaluate_buy_take_profit(
        self,
        order_id: str,
        current_price: float,
        order_price: float,
    ) -> tuple[bool | None, str]:
        """Evaluate buy-side take-profit order."""
        # Within 5% of target - preserve
        if current_price <= order_price * 1.05:
            self.logger.info(
                f"Take-profit buy order {order_id} close to target: "
                f"current ${current_price:.2f} vs target ${order_price:.2f}",
            )
            return False, f"Close to target price (${order_price:.2f}) - preserving order"

        # More than 15% above target - cancel
        if current_price >= order_price * 1.15:
            self.logger.info(
                f"Take-profit buy order {order_id} far from target: "
                f"current ${current_price:.2f} vs target ${order_price:.2f}",
            )
            return True, f"Far from target price (${order_price:.2f}) - cancelling order"

        # No definitive decision
        return None, ""

    async def _evaluate_order_age(
        self,
        order_details: dict[str, Any],
        current_price: float,
        order_price: float,
    ) -> tuple[bool | None, str]:
        """Evaluate order based on age and price difference."""
        order_age = await self._get_order_age_hours(order_details)
        if order_age and order_age > 24:  # Order older than 24 hours
            price_diff_pct = abs(current_price - order_price) / current_price
            if price_diff_pct > 0.10:  # 10% threshold
                return True, "Stale order (>24h) far from market - cancelling"
            return False, "Keeping aged order as still reasonable"

        # No age-based decision
        return None, ""

    def _evaluate_price_difference(
        self,
        current_price: float,
        order_price: float,
    ) -> tuple[bool, str]:
        """Default evaluation based on price difference."""
        price_diff_pct = abs(current_price - order_price) / current_price

        if price_diff_pct > 0.10:  # 10% threshold
            return True, f"Take-profit order too far from market ({price_diff_pct:.1%}) - cancelling"

        return False, f"Take-profit order reasonably positioned ({price_diff_pct:.1%}) - preserving"

    async def _get_order_age_hours(self, order_details: dict[str, Any]) -> float | None:
        """Calculate order age in hours."""
        try:
            # Try to get order creation time
            open_time = order_details.get("opentm")
            if open_time:
                # Kraken timestamps are in seconds
                order_time = datetime.fromtimestamp(float(open_time), tz=UTC)
                current_time = datetime.now(UTC)
                age_delta = current_time - order_time
                return age_delta.total_seconds() / 3600  # Return hours

        except Exception as e:
            self.logger.debug(f"Could not calculate order age: {e}")
            return None
        else:
            return None


# Async Order Processor
class AsyncOrderProcessor:
    """Enterprise-grade asynchronous order processing engine."""

    def __init__(self, exchange_adapter: Any, config: dict[str, Any]) -> None:
        """Initialize the instance."""
        self.exchange_adapter = exchange_adapter
        self.config = config
        self.logger = exchange_adapter.logger if hasattr(exchange_adapter, "logger") else None

        # Processing queues by priority
        self.order_queues: dict[OrderPriority, asyncio.Queue[Any]] = {
            OrderPriority.CRITICAL: asyncio.Queue(maxsize=1000),
            OrderPriority.HIGH: asyncio.Queue(maxsize=5000),
            OrderPriority.NORMAL: asyncio.Queue(maxsize=10000),
            OrderPriority.LOW: asyncio.Queue(maxsize=20000),
        }

        # Active processing tasks
        self.processing_tasks: list[asyncio.Task[Any]] = []
        self.batch_operations: dict[str, BatchOperation] = {}

        # Concurrency control
        self.max_concurrent_orders = config.get("max_concurrent_orders", 50)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_orders)
        self.rate_limiter = asyncio.Semaphore(config.get("max_requests_per_second", 10))

        # Background task tracking
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self._running = False

    async def start_processing(self) -> None:
        """Start asynchronous order processing."""
        if self._running:
            if self.logger:
                self.logger.warning("Order processor already running")
            return

        self._running = True
        if self.logger:
            self.logger.info("Starting asynchronous order processor")

        # Start processing workers for each priority level
        for priority in OrderPriority:
            for worker_id in range(self._get_worker_count(priority)):
                task = asyncio.create_task(self._process_orders_worker(priority, worker_id))
                self.processing_tasks.append(task)

        if self.logger:
            self.logger.info(f"Started {len(self.processing_tasks)} processing tasks")

    async def submit_order_async(
        self,
        order_request: dict[str, Any],
        priority: OrderPriority = OrderPriority.NORMAL,
        callback: Callable[..., Any] | None = None,
    ) -> str:
        """Submit order for asynchronous processing."""
        async_request = AsyncOrderRequest(
            order_request=order_request,
            priority=priority,
            callback=callback,
            metadata={"submission_time": time.time()},
        )

        try:
            await self.order_queues[priority].put(async_request)
            if self.logger:
                self.logger.debug(f"Order queued with {priority.value} priority")
            return f"async_order_{int(time.time() * 1000000)}"

        except asyncio.QueueFull as e:
            if self.logger:
                self.logger.exception("Order queue full for priority")
            raise OrderProcessingError(f"Queue full for priority {priority.value}") from e

    async def stop_processing(self) -> None:
        """Stop asynchronous order processing."""
        self._running = False

        # Cancel all processing tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.processing_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.processing_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except TimeoutError:
                if self.logger:
                    self.logger.warning("Some processing tasks did not stop gracefully")

        self.processing_tasks.clear()
        if self.logger:
            self.logger.info("Async order processor stopped")

    def _get_worker_count(self, priority: OrderPriority) -> int:
        """Get worker count for priority level."""
        worker_counts = {
            OrderPriority.CRITICAL: 5,
            OrderPriority.HIGH: 10,
            OrderPriority.NORMAL: 15,
            OrderPriority.LOW: 5,
        }
        return worker_counts.get(priority, 5)

    async def _process_orders_worker(self, priority: OrderPriority, worker_id: int) -> None:
        """Worker task for processing orders from priority queue."""
        if self.logger:
            self.logger.debug(f"Started {priority.value} priority worker {worker_id}")

        while self._running:
            try:
                # Get order from queue with timeout
                order_request = await asyncio.wait_for(
                    self.order_queues[priority].get(),
                    timeout=1.0,
                )

                # Process the order
                await self._execute_order_request(order_request)

            except TimeoutError:  # noqa: S112
                # Expected timeout when queue is empty, continue checking other queues
                continue
            except Exception:
                if self.logger:
                    self.logger.exception(f"Worker {priority.value}-{worker_id} error: ")

    async def _execute_order_request(self, request: AsyncOrderRequest) -> dict[str, Any]:
        """Execute individual order request using real exchange adapter."""
        async with self.semaphore, self.rate_limiter:
            try:
                execution_start = time.time()

                # Use strategy pattern for order execution
                execution_context = OrderExecutionContext(self.exchange_adapter, self.logger)
                result = await execution_context.execute_order(request, execution_start)

                # Handle callbacks
                await self._handle_order_callback(request, result)

                # Update processing metrics
                self._update_processing_metrics(request, result)

            except Exception as e:
                return await self._handle_execution_error(request, e)
            else:
                return result

    async def _handle_order_callback(self, request: AsyncOrderRequest, result: dict[str, Any]) -> None:
        """Handle order execution callback if provided."""
        if request.callback:
            try:
                await request.callback(result)
            except Exception:
                if self.logger:
                    self.logger.exception("Order callback failed:")

    async def _handle_execution_error(self, request: AsyncOrderRequest, error: Exception) -> dict[str, Any]:
        """Handle order execution errors with retry logic."""
        if self.logger:
            self.logger.exception(
                f"Order execution failed for {request.priority.value} priority order: ",
            )

        # Check if retry is possible
        if request.retry_count < request.max_retries:
            request.retry_count += 1
            retry_delay = self._calculate_retry_delay(request.retry_count)

            if self.logger:
                self.logger.warning(
                    f"Retrying order execution in {retry_delay:.1f}s "
                    f"(attempt {request.retry_count}/{request.max_retries})",
                )

            # Schedule retry
            retry_task = asyncio.create_task(self._schedule_retry(request, retry_delay))
            self._background_tasks.add(retry_task)
            retry_task.add_done_callback(self._background_tasks.discard)
        elif self.logger:
            self.logger.exception(
                f"Order execution failed permanently after {request.max_retries} retries",
            )

        # Return error result
        return {
            "success": False,
            "error": str(error),
            "retry_count": request.retry_count,
            "max_retries_exceeded": request.retry_count >= request.max_retries,
        }

    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay for retries."""
        base_delay = self.config.get("error_retry_delay_seconds", 1.0)
        return float(min(float(base_delay) * (2 ** (retry_count - 1)), 30.0))  # Max 30 second delay

    async def _schedule_retry(self, request: AsyncOrderRequest, delay: float) -> None:
        """Schedule order retry with delay."""
        await asyncio.sleep(delay)
        try:
            await self.order_queues[request.priority].put(request)
        except Exception:
            if self.logger:
                self.logger.exception("Failed to reschedule order retry: ")

    def _update_processing_metrics(
        self,
        request: AsyncOrderRequest,
        result: dict[str, Any],
    ) -> None:
        """Update processing metrics for monitoring."""
        try:
            # Calculate processing duration
            processing_duration = time.time() - request.metadata.get(
                "submission_time",
                time.time(),
            )

            # Update metrics (would integrate with monitoring service)
            metrics = {
                "priority": request.priority.value,
                "success": result.get("success", False),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "processing_duration_s": processing_duration,
                "retry_count": request.retry_count,
                "queue_wait_time_s": processing_duration - (result.get("execution_time_ms", 0) / 1000),
            }

            if self.logger:
                self.logger.debug(f"Order processing metrics: {metrics}")

        except Exception as e:
            if self.logger:
                self.logger.debug(f"Failed to update processing metrics: {e}")


class ExecutionHandler(ServiceProtocol):
    """Handle interaction with the exchange API (Kraken) to place, manage, and monitor orders.

    Processes approved trade signals, translates them to exchange-specific parameters,
    places orders, and monitors their execution.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        monitoring_service: MonitoringService,
        logger_service: LoggerService,
        event_store: EventStore | None = None,
    ) -> None:
        """Initialize the execution handler with required services and configuration."""
        # Core service dependencies
        self.logger = logger_service
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.monitoring = monitoring_service
        self.event_store = event_store

        # Load configuration
        self.handler_config = ExecutionHandlerConfig.from_config_manager(config_manager)

        # Set up direct attribute access for frequently used config values
        self.api_key = self.handler_config.api_key
        self.api_secret = self.handler_config.api_secret
        self.api_base_url = self.handler_config.api_url
        self._use_websocket_for_orders = self.handler_config.websocket.use_for_order_updates

        # Validate API credentials
        self._validate_api_credentials()

        # Initialize core components
        self._session: aiohttp.ClientSession | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._initialize_order_mappings()

        # Initialize subsystems
        self._adapter = self._initialize_execution_adapter()
        self._initialize_websocket_components()
        self.rate_limiter = self._initialize_rate_limiter()
        self.persistence = self._initialize_persistence_service()
        self.order_state_tracker = self._initialize_order_state_tracker()
        self.market_data_service = self._initialize_market_data_service()
        self.shutdown_handler = self._initialize_shutdown_handler()
        self.async_processor = self._initialize_async_processor()

        # Log initialization summary
        self._log_initialization_summary()

    def _validate_api_credentials(self) -> None:
        """Validate that API credentials are configured."""
        if not self.handler_config.api_key or not self.handler_config.api_secret:
            self.logger.critical(
                "Kraken API Key or Secret Key not configured. ExecutionHandler cannot function.",
                source_module=self.__class__.__name__,
            )

    def _initialize_order_mappings(self) -> None:
        """Initialize order ID mapping structures."""
        # Maps internal client order IDs to Kraken exchange order IDs
        self._internal_to_exchange_order_id: dict[str, str] = {}
        # Maps Kraken exchange order IDs back to internal client order IDs
        self._exchange_to_internal_order_id: dict[str, str] = {}
        # Track orders awaiting confirmation or updates (includes full order context)
        self._pending_orders_by_cl_ord_id: dict[str, TradeSignalApprovedEvent] = {}
        # Legacy mapping for backward compatibility during transition
        self._order_map: dict[str, str] = {}  # cl_ord_id -> txid
        # Internal pair -> Kraken details
        self._pair_info: dict[str, dict[str, Any]] = {}
        # Handler references
        self._trade_signal_handler: None | (Callable[[TradeSignalApprovedEvent], Coroutine[Any, Any, None]]) = None
        self._close_position_handler: None | (Callable[[ClosePositionCommand], Coroutine[Any, Any, None]]) = None
        # Store active monitoring tasks
        self._order_monitoring_tasks: dict[str, asyncio.Task[Any]] = {}  # txid -> Task[Any]
        # Track signals that have had SL/TP orders placed
        self._placed_sl_tp_signals: set[UUID] = set()

    def _initialize_execution_adapter(self) -> ExecutionAdapter | None:
        """Initialize the appropriate exchange adapter based on configuration."""
        if self.handler_config.exchange.lower() == "kraken":
            adapter = KrakenExecutionAdapter(
                config=self.config,
                logger=self.logger,
            )
            self.logger.info(
                "Initialized Kraken execution adapter",
                source_module=self.__class__.__name__,
            )
            return adapter
        self.logger.error(
            "Unsupported exchange: %s. Currently only 'kraken' is supported.",
            self.handler_config.exchange,
            source_module=self.__class__.__name__,
        )
        return None

    def _initialize_websocket_components(self) -> None:
        """Initialize WebSocket-related components if enabled."""
        self.websocket_client: KrakenWebSocketClient | None = None

        if self.handler_config.websocket.use_for_order_updates:
            # WebSocket connection state and configuration
            self._websocket_connection_state = "DISCONNECTED"  # DISCONNECTED, CONNECTING, CONNECTED, AUTHENTICATED
            self._websocket_auth_token: str | None = None
            self._websocket_connection_task: asyncio.Task[Any] | None = None
            self._subscribed_channels: set[str] = set()
            self._max_reconnect_attempts = self.handler_config.websocket.max_reconnect_attempts
            self._reconnect_delay_seconds = self.handler_config.websocket.reconnect_delay_seconds
            self._current_reconnect_attempts = 0
            self._is_running = False  # Service lifecycle tracking for reconnection logic

            # Initialize the WebSocket client
            self.websocket_client = KrakenWebSocketClient(
                config=self.config,
                pubsub=self.pubsub,
                logger=self.logger,
            )
        else:
            self._websocket_connection_state = "DISABLED"
            self._websocket_connection_task = None

    def _initialize_rate_limiter(self) -> RateLimitTracker:
        """Initialize rate limiting component."""
        return RateLimitTracker(self.config, self.logger)

    def _initialize_order_state_tracker(self) -> OrderStateTracker:
        """Initialize order state tracking component."""
        tracker = OrderStateTracker(
            persistence_service=self.persistence,
            event_publisher=self.pubsub,
            config={
                "persistence_enabled": self.handler_config.order_state.persistence_enabled,
                "cache_size": self.handler_config.order_state.cache_size,
                "batch_persistence": self.handler_config.order_state.batch_persistence,
                "persistence_interval_seconds": self.handler_config.order_state.persistence_interval_seconds,
            },
        )
        tracker.logger = self.logger
        return tracker

    def _initialize_shutdown_handler(self) -> ConfigurableShutdownHandler:
        """Initialize configurable shutdown handler."""
        shutdown_config = ShutdownConfig(
            default_action=ShutdownOrderAction(self.handler_config.shutdown.default_action),
            action_by_order_type={
                "stop-loss": ShutdownOrderAction(self.handler_config.shutdown.stop_loss_action),
                "take-profit": ShutdownOrderAction(self.handler_config.shutdown.take_profit_action),
                "limit": ShutdownOrderAction(self.handler_config.shutdown.limit_action),
                "market": ShutdownOrderAction(self.handler_config.shutdown.market_action),
            },
            action_by_strategy={},
            safety_checks_enabled=self.handler_config.shutdown.safety_checks_enabled,
            max_cancellation_time=self.handler_config.shutdown.max_cancellation_time,
            preserve_stop_losses=self.handler_config.shutdown.preserve_stop_losses,
            market_hours_only=self.handler_config.shutdown.market_hours_only,
        )

        return ConfigurableShutdownHandler(
            config=shutdown_config,
            execution_handler=self,
            market_data_service=self.market_data_service,
        )

    def _initialize_async_processor(self) -> AsyncOrderProcessor:
        """Initialize asynchronous order processor."""
        async_config = {
            "max_concurrent_orders": self.handler_config.async_processing.max_concurrent_orders,
            "max_requests_per_second": self.handler_config.async_processing.max_requests_per_second,
            "enable_priority_queues": self.handler_config.async_processing.enable_priority_queues,
            "error_retry_max_attempts": self.handler_config.async_processing.error_retry_max_attempts,
            "error_retry_delay_seconds": self.handler_config.async_processing.error_retry_delay_seconds,
            "health_check_interval": self.handler_config.async_processing.health_check_interval,
        }

        return AsyncOrderProcessor(
            exchange_adapter=self._adapter,
            config=async_config,
        )

    def _log_initialization_summary(self) -> None:
        """Log summary of enabled features."""
        features_enabled = []
        if self.persistence:
            features_enabled.append("database persistence")
        if self.market_data_service:
            features_enabled.append("market data integration")
        if self.handler_config.websocket.use_for_order_updates:
            features_enabled.append("websocket order updates")
        features_enabled.extend(
            [
                "order state tracking",
                "configurable shutdown",
                "async processing",
                "batch orders",
                "enterprise monitoring",
            ],
        )

        self.logger.info(
            f"ExecutionHandler initialized successfully with {len(features_enabled)} "
            f"enterprise features: {', '.join(features_enabled)}",
            source_module=self.__class__.__name__,
        )

    async def initialize(self) -> None:
        """Async initialization hook for compatibility with ServiceProtocol."""
        # No asynchronous setup currently required
        return

    def _initialize_persistence_service(self) -> Any:
        """Initialize persistence service for order state tracking."""
        try:
            # Create database components
            db_pool, order_repository = self._create_database_components()

            # Create and return persistence service
            persistence_service = self._create_order_persistence_service(order_repository)

            self.logger.info("Persistence service initialized successfully", source_module=self.__class__.__name__)

        except Exception:
            self.logger.exception(
                "Failed to initialize persistence service: . Operating without persistence.",
                source_module=self.__class__.__name__,
            )
            return None
        else:
            return persistence_service

    def _create_database_components(self) -> tuple[Any, Any]:
        """Create database pool and order repository."""
        # Create connection pool
        db_pool = DatabaseConnectionPool(self.config, self.logger)
        session_maker = db_pool.get_session_maker()
        if not session_maker:
            raise RuntimeError("Failed to get session maker from database pool")

        # Create order repository
        order_repository = OrderRepository(session_maker, self.logger)
        return db_pool, order_repository

    def _create_order_persistence_service(self, order_repository: Any) -> Any:
        """Create order persistence service wrapper."""
        return OrderPersistenceService(order_repository, self.logger)

    def _initialize_market_data_service(self) -> Any:
        """Initialize market data service for intelligent shutdown decisions."""
        try:
            # Import market data components
            # Note: MarketPriceService is not implemented, using enhanced service instead

            market_data_service = EnhancedMarketDataService(self.config, self.logger)

            self.logger.info(
                "Market data service initialized for shutdown decisions",
                source_module=self.__class__.__name__,
            )

        except Exception:
            self.logger.exception(
                "Failed to initialize market data service: . Operating with basic shutdown logic.",
                source_module=self.__class__.__name__,
            )
            return None
        else:
            return market_data_service

    async def _load_persisted_order_states(self) -> None:
        """Load any persisted order states during startup."""
        try:
            if self.order_state_tracker and self.persistence:
                # Get list[Any] of active orders from persistence
                active_orders = await self.persistence.get_active_orders()

                for order_data in active_orders:
                    order_id = order_data.get("order_id")
                    if order_id:
                        lifecycle_data = await self.order_state_tracker.load_order_state(order_id)
                        if lifecycle_data:
                            self.order_state_tracker.order_states[order_id] = lifecycle_data

                self.logger.info(
                    f"Loaded {len(active_orders)} persisted order states",
                    source_module=self.__class__.__name__,
                )

        except Exception:
            self.logger.exception(
                "Error loading persisted order states: ",
                source_module=self.__class__.__name__,
            )

    async def _persist_all_pending_order_states(self) -> None:
        """Persist all pending order states during shutdown."""
        try:
            if self.order_state_tracker:
                pending_count = 0
                for lifecycle_data in self.order_state_tracker.order_states.values():
                    if lifecycle_data.current_state in [
                        OrderState.PENDING,
                        OrderState.SUBMITTED,
                        OrderState.PARTIALLY_FILLED,
                    ]:
                        await self.order_state_tracker.persist_order_state(lifecycle_data)
                        pending_count += 1

                self.logger.info(
                    f"Persisted {pending_count} pending order states during shutdown",
                    source_module=self.__class__.__name__,
                )

        except Exception:
            self.logger.exception("Error persisting pending order states: ", source_module=self.__class__.__name__)

    async def start(self) -> None:
        """Initialize API client session, load exchange info, and subscribe to events."""
        self.logger.info("Starting ExecutionHandler...", source_module=self.__class__.__name__)

        # Initialize the execution adapter
        if self._adapter:
            try:
                await self._adapter.initialize()
                self.logger.info("Execution adapter initialized successfully", source_module=self.__class__.__name__)
            except Exception:
                self.logger.exception("Failed to initialize execution adapter", source_module=self.__class__.__name__)
                raise
        else:
            self.logger.error(
                "No execution adapter configured. ExecutionHandler cannot function.",
                source_module=self.__class__.__name__,
            )
            raise RuntimeError("No execution adapter configured")

        # Keep legacy session for non-adapter operations (to be refactored later)
        self._session = aiohttp.ClientSession()

        # Load exchange info for legacy code paths
        await self._load_exchange_info()

        # Check if info loading failed significantly
        if not self._pair_info:
            self.logger.error(
                ("Failed to load essential exchange pair info. ExecutionHandler will not function correctly."),
                source_module=self.__class__.__name__,
            )

        # Store the handler for unsubscribing
        self._trade_signal_handler = self.handle_trade_signal_approved
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler)

        # Subscribe to close position commands for emergency HALT
        self._close_position_handler = self.handle_close_position_command
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, self._close_position_handler)

        self.logger.info(
            "ExecutionHandler started. Subscribed to TRADE_SIGNAL_APPROVED and close position commands.",
            source_module=self.__class__.__name__,
        )

        # Enterprise WebSocket Connection Logic
        if self._use_websocket_for_orders and self.websocket_client:
            self._is_running = True
            self.logger.info(
                "Starting WebSocket connection for order updates...",
                source_module=self.__class__.__name__,
            )
            try:
                await self._connect_websocket()
            except Exception:
                self.logger.exception(
                    "Failed to establish WebSocket connection during startup. Will attempt reconnection.",
                    source_module=self.__class__.__name__,
                )
                # Don't fail startup due to WebSocket issues - will attempt reconnection
        else:
            self.logger.info(
                "WebSocket disabled for order updates. Using polling-based monitoring.",
                source_module=self.__class__.__name__,
            )

        # --- Start Enterprise Components ---
        # Start asynchronous order processor
        try:
            await self.async_processor.start_processing()
            self.logger.info("Asynchronous order processor started successfully", source_module=self.__class__.__name__)
        except Exception:
            self.logger.exception("Failed to start async processor: ", source_module=self.__class__.__name__)

        # Load any persisted order states
        await self._load_persisted_order_states()

        self.logger.info("All enterprise components started successfully", source_module=self.__class__.__name__)

    async def stop(self) -> None:
        """Close API client session and potentially cancel orders."""
        self.logger.info("Stopping ExecutionHandler...", source_module=self.__class__.__name__)

        # Phase 1: Unsubscribe from event handlers
        await self._unsubscribe_event_handlers()

        # Phase 2: Process shutdown orders
        await self._process_shutdown_orders()

        # Phase 3: Cancel monitoring tasks
        await self._cancel_monitoring_tasks()

        # Phase 4: Close network sessions
        await self._close_network_sessions()

        # Phase 5: Stop async components
        await self._stop_async_components()

        # Phase 6: Persist final state
        await self._persist_final_state()

        # Phase 7: Clean up adapters
        await self._cleanup_adapters()

        # Phase 8: Disconnect WebSocket if needed
        await self._shutdown_websocket()

        self.logger.info("ExecutionHandler stopped.", source_module=self.__class__.__name__)

    async def _unsubscribe_event_handlers(self) -> None:
        """Unsubscribe from all event handlers."""
        if self._trade_signal_handler is not None:
            try:
                self.pubsub.unsubscribe(EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler)
                self.logger.info("Unsubscribed from TRADE_SIGNAL_APPROVED.")
                self._trade_signal_handler = None
            except Exception:
                self.logger.exception("Error unsubscribing from trade signal handler")

        if self._close_position_handler is not None:
            try:
                self.pubsub.unsubscribe(EventType.TRADE_SIGNAL_APPROVED, self._close_position_handler)
                self.logger.info("Unsubscribed from close position commands.")
                self._close_position_handler = None
            except Exception:
                self.logger.exception("Error unsubscribing from close position handler")

    async def _process_shutdown_orders(self) -> None:
        """Process orders during shutdown using configurable handler."""
        try:
            cancellation_results = await self.shutdown_handler.handle_shutdown_orders()

            if cancellation_results:
                successful = sum(1 for r in cancellation_results if r.success)
                total = len(cancellation_results)
                self.logger.info(
                    f"Configurable shutdown completed: {successful}/{total} orders processed successfully",
                    source_module=self.__class__.__name__,
                )
            else:
                self.logger.info("No orders required shutdown processing", source_module=self.__class__.__name__)
        except Exception:
            self.logger.exception("Error during configurable shutdown: ", source_module=self.__class__.__name__)

    async def _cancel_monitoring_tasks(self) -> None:
        """Cancel all ongoing order monitoring tasks."""
        try:
            for task_id, task in list(self._order_monitoring_tasks.items()):
                if not task.done():
                    task.cancel()
                    self.logger.info(f"Cancelled monitoring task for order {task_id}")
            self._order_monitoring_tasks.clear()
            self.logger.info("All order monitoring tasks cancelled.")
        except Exception:
            self.logger.exception("Error cancelling monitoring tasks", source_module=self.__class__.__name__)

    async def _close_network_sessions(self) -> None:
        """Close all network sessions."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("AIOHTTP session closed.", source_module=self.__class__.__name__)

    async def _stop_async_components(self) -> None:
        """Stop asynchronous processing components."""
        try:
            if hasattr(self, "async_processor") and self.async_processor:
                await self.async_processor.stop_processing()
                self.logger.info("Async order processor stopped successfully", source_module=self.__class__.__name__)
        except Exception:
            self.logger.exception("Error stopping async processor: ", source_module=self.__class__.__name__)

    async def _persist_final_state(self) -> None:
        """Persist any remaining order states before shutdown."""
        try:
            await self._persist_all_pending_order_states()
        except Exception:
            self.logger.exception("Error persisting final order states: ", source_module=self.__class__.__name__)

    async def _cleanup_adapters(self) -> None:
        """Clean up execution adapters."""
        if self._adapter:
            try:
                await self._adapter.cleanup()
                self.logger.info("Execution adapter cleaned up successfully", source_module=self.__class__.__name__)
            except Exception:
                self.logger.exception("Error cleaning up execution adapter", source_module=self.__class__.__name__)

    async def _shutdown_websocket(self) -> None:
        """Shutdown WebSocket connection if enabled."""
        if self.handler_config.websocket.use_for_order_updates and self.websocket_client:
            self._is_running = False  # Signal that we're intentionally stopping
            self.logger.info("Disconnecting WebSocket connection...", source_module=self.__class__.__name__)
            try:
                await self._disconnect_websocket()
            except Exception:
                self.logger.exception("Error during WebSocket disconnection", source_module=self.__class__.__name__)

    async def _make_public_request_with_retry(self, url: str, max_retries: int = 3) -> dict[str, Any] | None:
        """Make a public request with retry logic for transient errors."""
        base_delay = self.config.get_float("exchange.retry_base_delay_s", 1.0)

        for attempt in range(max_retries + 1):
            try:
                # Wait for rate limit capacity before making the request
                await self.rate_limiter.wait_for_public_capacity()

                if not self._session:
                    self.logger.error(
                        "Cannot make public request: AIOHTTP session is not available.",
                        source_module=self.__class__.__name__,
                    )
                    return None

                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response.raise_for_status()
                    data: dict[str, Any] = await response.json()

                    if data.get("error"):
                        error_str = str(data["error"])
                        if self._is_retryable_error(error_str) and attempt < max_retries:
                            delay = min(base_delay * (2**attempt), 30.0)
                            jitter = secrets.SystemRandom().uniform(0, delay * 0.1)
                            total_delay = delay + jitter
                            self.logger.warning(
                                "Retryable API error for %s: %s. Retrying in %.2fs (Attempt %d/%d)",
                                url,
                                error_str,
                                total_delay,
                                attempt + 1,
                                max_retries + 1,
                                source_module=self.__class__.__name__,
                            )
                            await asyncio.sleep(total_delay)
                            continue

                        self.logger.exception(
                            "Error in public API response: %s",
                            error_str,
                            source_module=self.__class__.__name__,
                        )
                        return None

                    return data

            except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError, TimeoutError) as e:
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 30.0)
                    jitter = secrets.SystemRandom().uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        ("Error during public request to %s: %s. Retrying in %.2fs (Attempt %d/%d)"),
                        url,
                        e,
                        total_delay,
                        attempt + 1,
                        max_retries + 1,
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    continue
                self.logger.exception(
                    "Failed to make public request to %s after %d attempts. Last error recorded.",
                    url,
                    max_retries + 1,
                    source_module=self.__class__.__name__,
                )
                return None
            except Exception:
                self.logger.exception(
                    "Unexpected error during public request to %s. Error: %s",
                    url,
                    source_module=self.__class__.__name__,
                )
                return None

        last_error_message = "Unknown error"
        if "last_exception" in locals() and locals()["last_exception"]:
            last_error_message = str(locals()["last_exception"])
        self.logger.error(
            ("Failed to make public request to %s after %d attempts. Last error: %s"),
            url,
            max_retries + 1,
            last_error_message,
            source_module=self.__class__.__name__,
        )
        return None

    async def _load_exchange_info(self) -> None:
        """Fetch and store tradable asset pair information from Kraken."""
        uri_path = "/0/public/AssetPairs"
        url = self.api_base_url + uri_path
        self.logger.info("Loading exchange asset pair info from %s...", url, source_module=self.__class__.__name__)

        if not self._validate_session():
            return

        try:
            # Use the new method with retry and rate limiting for public requests
            data = await self._make_public_request_with_retry(url)
            if not data:
                self.logger.error("Failed to fetch asset pairs data.", source_module=self.__class__.__name__)
                return

            result = data.get("result", {})
            if not result:
                self.logger.error("AssetPairs result is empty.", source_module=self.__class__.__name__)
                return

            await self._process_asset_pairs(result)

        except Exception:  # Catch-all for unexpected errors
            self.logger.exception("Unexpected error loading exchange info.", source_module=self.__class__.__name__)

    def _validate_session(self) -> bool:
        """Validate that the AIOHTTP session is available."""
        if not self._session or self._session.closed:
            self.logger.error(
                "Cannot load exchange info: AIOHTTP session is not available.",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    async def _process_asset_pairs(self, result: dict[str, Any]) -> None:
        """Process and store asset pairs data."""
        loaded_count = 0
        internal_pairs = self.config.get_list("trading.pairs", [])

        if not internal_pairs:
            self.logger.warning(
                ("No trading pairs defined in config [trading.pairs]. Cannot map exchange info."),
                source_module=self.__class__.__name__,
            )
            return

        kraken_pair_map = {v.get("altname", k): k for k, v in result.items()}

        for internal_pair_name in internal_pairs:
            if self._process_single_pair(internal_pair_name, kraken_pair_map, result):
                loaded_count += 1

        self._log_loading_results(loaded_count, len(internal_pairs))

    def _process_single_pair(
        self,
        internal_pair_name: str,
        kraken_pair_map: dict[str, Any],
        result: dict[str, Any],
    ) -> bool:
        """Process a single trading pair and store its information."""
        kraken_altname = internal_pair_name.replace("/", "")
        kraken_key = kraken_pair_map.get(kraken_altname)

        if not kraken_key or kraken_key not in result:
            self.logger.warning(
                ("Could not find matching AssetPairs info for configured pair: %s"),
                internal_pair_name,
                source_module=self.__class__.__name__,
            )
            return False

        pair_data = result[kraken_key]
        self._pair_info[internal_pair_name] = {
            "kraken_pair_key": kraken_key,
            "altname": pair_data.get("altname"),
            "wsname": pair_data.get("wsname"),
            "base": pair_data.get("base"),
            "quote": pair_data.get("quote"),
            "pair_decimals": pair_data.get("pair_decimals"),
            "cost_decimals": pair_data.get("cost_decimals"),
            "lot_decimals": pair_data.get("lot_decimals"),
            "ordermin": pair_data.get("ordermin"),
            "costmin": pair_data.get("costmin"),
            "tick_size": pair_data.get("tick_size"),
            "status": pair_data.get("status"),
        }
        self.logger.debug("Loaded info for %s", internal_pair_name, source_module=self.__class__.__name__)
        return True

    def _log_loading_results(self, loaded_count: int, total_pairs: int) -> None:
        """Log the results of loading asset pairs."""
        self.logger.info(
            ("Successfully loaded info for %s asset pairs out of %s configured."),
            loaded_count,
            total_pairs,
            source_module=self.__class__.__name__,
        )

        if loaded_count < total_pairs:
            self.logger.warning(
                (
                    "Mismatch between configured pairs and loaded exchange info. "
                    "Some configured pairs may not be tradeable."
                ),
                source_module=self.__class__.__name__,
            )

    def _generate_kraken_signature(self, uri_path: str, data: dict[str, Any], nonce: int) -> str:
        """Generate the API-Sign header required by Kraken private endpoints."""
        if self.api_secret is None:
            raise ValueError("API secret must be configured")
        return generate_kraken_signature(uri_path, data, nonce, self.api_secret)

    def _format_decimal(self, value: Decimal, precision: int) -> str:
        """Format a Decimal value to a string with a specific precision."""
        # Use quantization to set the number of decimal places
        # Ensure it rounds correctly, default rounding is ROUND_HALF_EVEN
        quantizer = Decimal("1e-" + str(precision))
        return str(value.quantize(quantizer))

    async def _make_private_request(self, uri_path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make an authenticated request to a private Kraken REST endpoint."""
        response_data: dict[str, Any]

        if not self._session or self._session.closed:
            self.logger.error(
                "AIOHTTP session is not available for private request.",
                source_module=self.__class__.__name__,
            )
            response_data = {"error": ["EGeneral:InternalError - HTTP session closed"]}
            return response_data

        # Generate nonce and signature
        nonce = int(time.time() * 1000)  # Kraken uses milliseconds nonce
        request_data = data.copy()  # Avoid modifying the original dict[str, Any]
        request_data["nonce"] = nonce
        api_sign: str

        try:
            api_sign = self._generate_kraken_signature(uri_path, request_data, nonce)
        except ValueError:  # Raised by _generate_kraken_signature if API secret is invalid
            # Error already logged by _generate_kraken_signature
            response_data = {"error": ["EGeneral:InternalError - Signature generation failed"]}
            return response_data

        if self.api_key is None:
            raise ValueError("API key must be configured")
        headers = {
            "API-Key": self.api_key,
            "API-Sign": api_sign,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        url = self.api_base_url + uri_path
        timeout = aiohttp.ClientTimeout(total=self.config.get("exchange.request_timeout_seconds", 10))

        try:
            self.logger.debug(
                "Sending private request to %s with data: %s",
                url,
                request_data,
                source_module=self.__class__.__name__,
            )
            async with self._session.post(url, headers=headers, data=request_data, timeout=timeout) as response:
                response.raise_for_status()  # Raise exception for bad status codes (4xx, 5xx)
                result: dict[str, Any] = await response.json()
                self.logger.debug("Received response from %s: %s", url, result, source_module=self.__class__.__name__)
                # Check for API-level errors within the JSON response
                if result.get("error"):
                    self.logger.error(
                        "Kraken API error for %s: %s",
                        uri_path,
                        result["error"],
                        source_module=self.__class__.__name__,
                    )
                response_data = result  # Store result, whether success or API error

        except aiohttp.ClientResponseError as e:
            error_body = await response.text()  # response object should be available here
            self.logger.exception(
                ("HTTP Error: %s %s for %s. Body: %s"),
                e.status,
                e.message,
                e.request_info.url,
                error_body[:500],
                source_module=self.__class__.__name__,
            )
            response_data = {"error": [f"EGeneral:HTTPError - {e.status}: {e.message}"]}
        except aiohttp.ClientConnectionError as e:
            self.logger.exception("Connection Error to %s: %s", url, source_module=self.__class__.__name__)
            response_data = {"error": [f"EGeneral:ConnectionError - {e!s}"]}
        except TimeoutError:
            self.logger.exception("Request Timeout for %s: %s", url, source_module=self.__class__.__name__)
            response_data = {"error": ["EGeneral:Timeout"]}
        except Exception:  # Catch-all for unexpected errors
            self.logger.exception(
                "Unexpected error during private API request to %s: %s",
                url,
                source_module=self.__class__.__name__,
            )
            response_data = {"error": ["EGeneral:Unexpected - Unknown error during request"]}

        return response_data

    def _is_retryable_error(self, error_str: str) -> bool:
        """Check if a Kraken error string indicates a potentially transient issue."""
        # Use enhanced error classifier if available
        if hasattr(self, "_error_classifier"):
            error_instance = self._error_classifier.classify_error(error_str)
            return bool(self._error_classifier.should_retry(error_instance))

        # Fallback to basic error codes
        retryable_codes = [
            "EGeneral:Temporary",
            "EService:Unavailable",
            "EService:Busy",
            "EGeneral:Timeout",
            "EGeneral:ConnectionError",
            "EAPI:Rate limit exceeded",
            # Add more specific Kraken codes if identified
        ]
        # Check if any retryable code is found in the error string
        return any(code in error_str for code in retryable_codes)

    async def _make_private_request_with_retry(
        self,
        uri_path: str,
        data: dict[str, Any],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Make a private request with retry logic for transient errors."""
        base_delay = self.config.get_float("exchange.retry_base_delay_s", 1.0)
        final_result: dict[str, Any] | None = None
        last_error_info_str: str = "No specific error was recorded."

        for attempt in range(max_retries + 1):
            try:
                await self.rate_limiter.wait_for_private_capacity()
                current_result = await self._make_private_request(uri_path, data)

                if not current_result.get("error"):
                    # Successful API call
                    self.logger.debug(
                        "API call to %s successful in attempt %d.",
                        uri_path,
                        attempt + 1,
                        source_module=self.__class__.__name__,
                    )
                    final_result = current_result
                    break  # Exit loop on success

                # API call returned an error
                error_str = str(current_result.get("error", "Unknown API error"))
                last_error_info_str = f"APIError: {error_str}"

                if self._is_retryable_error(error_str) and attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 30.0)  # Cap delay at 30s
                    jitter = secrets.SystemRandom().uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        "Retryable API error for %s: %s. Retrying in %.2fs (Attempt %d/%d)",
                        uri_path,
                        error_str,
                        total_delay,
                        attempt + 1,
                        max_retries + 1,
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    # Loop continues to next attempt
                else:
                    # Permanent API error or max retries for this API error
                    self.logger.error(
                        "Permanent API error for %s or max retries for API error: %s",
                        uri_path,
                        error_str,
                        source_module=self.__class__.__name__,
                    )
                    final_result = current_result  # Store the error dict[str, Any]
                    break  # Exit loop

            except Exception as e:
                last_error_info_str = f"Exception: {type(e).__name__} - {e!s}"
                self.logger.exception(
                    "Exception during API request to %s (Attempt %d/%d): %s",
                    uri_path,
                    attempt + 1,
                    max_retries + 1,
                    source_module=self.__class__.__name__,
                )

                if isinstance(e, aiohttp.ClientConnectionError | asyncio.TimeoutError) and attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 30.0)
                    jitter = secrets.SystemRandom().uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        "Network error for %s (%s). Retrying in %.2fs (Attempt %d/%d): %s",
                        uri_path,
                        type(e).__name__,
                        total_delay,
                        attempt + 1,
                        max_retries + 1,
                        e,
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    # Loop continues to next attempt
                else:
                    # Non-retryable exception or max retries for this exception type
                    self.logger.exception(
                        "Non-retryable exception or max retries for network error. URI: %s. Error: %s",
                        uri_path,
                        source_module=self.__class__.__name__,
                    )
                    final_result = {
                        "error": [
                            f"EGeneral:RequestException - {last_error_info_str}",
                        ],
                    }
                    break  # Exit loop

        # After the loop, evaluate final_result
        if final_result and not final_result.get("error"):
            # Success was achieved and loop was broken
            return final_result

        if final_result and final_result.get("error"):
            # A permanent error (API or exception) was captured, and loop was broken
            return final_result

        # If final_result is None, it means loop completed all attempts due to retryable errors
        self.logger.error(
            "API request to %s failed after all %d attempts. Last known error: %s",
            uri_path,
            max_retries + 1,
            last_error_info_str,
            source_module=self.__class__.__name__,
        )
        return {
            "error": [
                f"EGeneral:MaxRetriesExceeded - Last known error: {last_error_info_str}",
            ],
        }

    async def handle_trade_signal_approved(self, event: TradeSignalApprovedEvent) -> None:
        """Process an approved trade signal event.

        Check HALT status, translate signal to API parameters, place order, and handle response.
        """
        self.logger.info("Received approved trade signal: %s", event.signal_id, source_module=self.__class__.__name__)

        # 1. Check HALT status FIRST
        if self.monitoring.is_halted():
            error_msg = "Execution blocked: System HALTED"
            self.logger.critical(
                "%s. Discarding approved signal: %s (%s %s %s)",
                error_msg,
                event.signal_id,
                event.trading_pair,
                event.side,
                event.quantity,
                source_module=self.__class__.__name__,
            )
            # Publish a REJECTED execution report for tracking
            # Assuming self._publish_error_execution_report exists and takes
            # optional cl_ord_id
            task = asyncio.create_task(
                self._publish_error_execution_report(
                    event=event,  # Pass event as a keyword argument
                    error_message=error_msg,  # Pass error_message as a keyword argument
                    cl_ord_id=f"internal_{event.signal_id}_halted",
                ),
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return  # Stop processing this signal

        # 2. Translate the signal to API parameters
        kraken_params = self._translate_signal_to_kraken_params(event)

        # 3. Handle translation failure
        if not kraken_params:
            self.logger.error(
                "Failed to translate signal %s. Order not placed.",
                event.signal_id,
                source_module=self.__class__.__name__,
            )
            # Publish an error report to indicate failure before sending
            await self._publish_error_execution_report(
                event=event,  # Pass event as a keyword argument
                error_message="Signal translation failed",
                # Pass error_message as a keyword argument
                cl_ord_id=None,  # Client order ID
            )
            return

        # 4. Generate Client Order ID and add to params
        # Using timestamp and signal prefix for basic uniqueness
        cl_ord_id = (
            # Microseconds
            f"gf-{str(event.signal_id)[:8]}-{int(time.time() * 1000000)}"
        )
        kraken_params["cl_ord_id"] = cl_ord_id

        # 5. Make the API request to place the order
        self.logger.info(
            "Placing order for signal %s with cl_ord_id %s",
            event.signal_id,
            cl_ord_id,
            source_module=self.__class__.__name__,
        )
        # Determine if we should use batch order placement
        has_sl = event.sl_price is not None
        has_tp = event.tp_price is not None
        use_batch = self.config.get("execution_handler.batch_orders.enabled", True) and (has_sl or has_tp)

        if use_batch and (event.sl_price or event.tp_price):
            # Use AddOrderBatch for simultaneous SL/TP placement
            result = await self._place_batch_order_with_sl_tp(kraken_params, event)
        else:
            # Use single order placement
            uri_path = "/0/private/AddOrder"
            result = await self._make_private_request_with_retry(uri_path, kraken_params)

        # 6. Handle the response from the AddOrder call
        await self._handle_add_order_response(result, event, cl_ord_id)

    async def _place_batch_order_with_sl_tp(
        self,
        base_params: dict[str, Any],
        event: TradeSignalApprovedEvent,
    ) -> dict[str, Any]:
        """Place batch order with stop-loss and take-profit using Kraken's AddOrderBatch endpoint."""
        try:
            # Build batch orders
            batch_orders = await self._build_batch_orders(base_params, event)

            # Prepare and execute batch request
            batch_params = self._prepare_batch_params(batch_orders)

            self.logger.info(
                f"Placing batch order with {len(batch_orders)} orders for signal {event.signal_id}",
                source_module=self.__class__.__name__,
            )

            # Execute batch order
            result = await self._execute_batch_order(batch_params)

            # Process and format result
            if self._is_batch_order_successful(result):
                return self._format_batch_order_result(result, batch_orders)

        except Exception:
            self.logger.exception(
                f"Error placing batch order for signal {event.signal_id}: ",
                source_module=self.__class__.__name__,
            )
            # Fallback to single order placement
            return await self._fallback_to_single_order(base_params)
        else:
            return result

    async def _build_batch_orders(
        self,
        base_params: dict[str, Any],
        event: TradeSignalApprovedEvent,
    ) -> list[dict[str, Any]]:
        """Build the batch orders list including primary and conditional orders."""
        batch_orders = []

        # Primary order
        primary_order = base_params.copy()
        primary_order["userref"] = int(time.time() * 1000)  # Unique reference
        batch_orders.append(primary_order)

        # Add stop-loss order if specified
        if event.sl_price:
            sl_order = await self._prepare_sl_order(base_params, event)
            if sl_order:
                batch_orders.append(sl_order)

        # Add take-profit order if specified
        if event.tp_price:
            tp_order = await self._prepare_tp_order(base_params, event)
            if tp_order:
                batch_orders.append(tp_order)

        return batch_orders

    def _prepare_batch_params(self, batch_orders: list[dict[str, Any]]) -> dict[str, Any]:
        """Prepare batch request parameters."""
        return {
            "orders": batch_orders,
            "deadline": str(
                int(time.time()) + self.config.get_int("execution_handler.batch_orders.deadline_seconds", 300),
            ),
            "validate": "false",  # Set to "true" for validation only
        }

    async def _execute_batch_order(self, batch_params: dict[str, Any]) -> dict[str, Any]:
        """Execute the batch order request."""
        uri_path = "/0/private/AddOrderBatch"
        return await self._make_private_request_with_retry(uri_path, batch_params)

    def _is_batch_order_successful(self, result: dict[str, Any]) -> bool:
        """Check if batch order was successful."""
        return bool(result and not result.get("error") and "orders" in result.get("result", {}))

    def _format_batch_order_result(
        self,
        result: dict[str, Any],
        batch_orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Format batch order result for consistent handling."""
        batch_result = result.get("result", {})
        self.logger.info(
            f"Batch order placement successful: {len(batch_result['orders'])} orders placed",
            source_module=self.__class__.__name__,
        )

        # Combine order results for consistent handling
        combined_txids = []
        for order_result in batch_result["orders"]:
            if order_result.get("txid"):
                combined_txids.extend(order_result["txid"])

        # Return in format compatible with single order response
        return {
            "result": {
                "txid": combined_txids,
                "descr": {"order": f"Batch order with {len(batch_orders)} orders"},
            },
        }

    async def _fallback_to_single_order(self, base_params: dict[str, Any]) -> dict[str, Any]:
        """Fallback to single order placement if batch fails."""
        uri_path = "/0/private/AddOrder"
        return await self._make_private_request_with_retry(uri_path, base_params)

    async def _prepare_sl_order(
        self,
        base_params: dict[str, Any],
        event: TradeSignalApprovedEvent,
    ) -> dict[str, Any] | None:
        """Prepare stop-loss order parameters for batch placement."""
        try:
            pair_info = self._pair_info.get(event.trading_pair)
            if not pair_info:
                return None

            # Determine SL order side (opposite of main order)
            sl_side = "sell" if event.side.lower() == "buy" else "buy"

            # Create stop-loss order parameters
            return {
                "pair": base_params["pair"],
                "type": sl_side,
                "ordertype": "stop-loss",
                "volume": base_params["volume"],
                "price": self._format_decimal(event.sl_price, pair_info.get("pair_decimals", 8)),
                "userref": int(time.time() * 1000) + 1,  # Unique reference
                "timeinforce": "GTC",
            }

        except Exception:
            self.logger.exception(
                "Error preparing stop-loss order: ",
                source_module=self.__class__.__name__,
            )
            return None

    async def _prepare_tp_order(
        self,
        base_params: dict[str, Any],
        event: TradeSignalApprovedEvent,
    ) -> dict[str, Any] | None:
        """Prepare take-profit order parameters for batch placement."""
        try:
            pair_info = self._pair_info.get(event.trading_pair)
            if not pair_info:
                return None

            # Determine TP order side (opposite of main order)
            tp_side = "sell" if event.side.lower() == "buy" else "buy"

            # Create take-profit order parameters
            return {
                "pair": base_params["pair"],
                "type": tp_side,
                "ordertype": "take-profit",
                "volume": base_params["volume"],
                "price": self._format_decimal(event.tp_price, pair_info.get("pair_decimals", 8)),
                "userref": int(time.time() * 1000) + 2,  # Unique reference
                "timeinforce": "GTC",
            }

        except Exception:
            self.logger.exception(
                "Error preparing take-profit order: ",
                source_module=self.__class__.__name__,
            )
            return None

    def _translate_signal_to_kraken_params(self, event: TradeSignalApprovedEvent) -> dict[str, Any] | None:
        """Translate internal signal format to Kraken API parameters.

        Includes validation of the parameters against exchange requirements.
        """
        params = {}
        internal_pair = event.trading_pair

        # 1. Get and validate pair info
        pair_info = self._get_and_validate_pair_info(internal_pair, event.signal_id)
        if not pair_info:
            return None

        # 2. Get and validate pair name
        kraken_pair_name = self._get_and_validate_pair_name(internal_pair, pair_info, event.signal_id)
        if not kraken_pair_name:
            return None
        params["pair"] = kraken_pair_name

        # 3. Validate and set order side
        if not self._validate_and_set_order_side(params, event):
            return None

        # 4. Validate and format volume
        if not self._validate_and_format_volume(params, event, pair_info):
            return None

        # 5. Map and validate order type
        if not self._map_and_validate_order_type(params, event, pair_info):
            return None

        # 6. Handle SL/TP warnings
        self._handle_sl_tp_warnings(event)

        self.logger.debug(
            "Translated signal %s to Kraken params: %s",
            event.signal_id,
            params,
            source_module=self.__class__.__name__,
        )
        return params

    def _get_and_validate_pair_info(self, internal_pair: str, signal_id: UUID) -> dict[str, Any] | None:
        """Get and validate trading pair information."""
        # Convert UUID to string for logging
        signal_id_str = str(signal_id)

        pair_info = self._pair_info.get(internal_pair)
        if not pair_info:
            self.logger.error(
                "No exchange info found for pair %s. Cannot translate signal %s.",
                internal_pair,
                signal_id_str,
                source_module=self.__class__.__name__,
            )
            return None

        if pair_info.get("status") != "online":
            self.logger.error(
                "Pair %s is not online (status: %s). Cannot place order for signal %s.",
                internal_pair,
                pair_info.get("status"),
                signal_id_str,
                source_module=self.__class__.__name__,
            )
            return None

        return pair_info

    def _get_and_validate_pair_name(self, internal_pair: str, pair_info: dict[str, Any], signal_id: UUID) -> str | None:
        """Get and validate the Kraken pair name."""
        # Convert UUID to string for logging
        signal_id_str = str(signal_id)

        kraken_pair_name = cast("str | None", pair_info.get("altname"))
        if not kraken_pair_name:
            self.logger.error(
                "Missing Kraken altname for pair %s in loaded info for signal %s.",
                internal_pair,
                signal_id_str,
                source_module=self.__class__.__name__,
            )
            return None
        return kraken_pair_name

    def _validate_and_set_order_side(self, params: dict[str, Any], event: TradeSignalApprovedEvent) -> bool:
        """Validate and set the order side parameter."""
        order_side = event.side.lower()
        if order_side not in ["buy", "sell"]:
            self.logger.error(
                "Invalid order side '%s' in signal %s.",
                event.side,
                event.signal_id,
                source_module=self.__class__.__name__,
            )
            return False
        params["type"] = order_side
        return True

    def _validate_and_format_volume(
        self,
        params: dict[str, Any],
        event: TradeSignalApprovedEvent,
        pair_info: dict[str, Any],
    ) -> bool:
        """Validate and format the order volume."""
        lot_decimals = pair_info.get("lot_decimals")
        ordermin_str = pair_info.get("ordermin")
        if lot_decimals is None or ordermin_str is None:
            self.logger.error(
                "Missing lot_decimals or ordermin for pair %s. Cannot validate/format volume.",
                event.trading_pair,
                source_module=self.__class__.__name__,
            )
            return False

        try:
            ordermin = Decimal(ordermin_str)
            if event.quantity < ordermin:
                self.logger.error(
                    "Order quantity %s is below minimum %s for pair %s. Signal %s.",
                    event.quantity,
                    ordermin,
                    event.trading_pair,
                    event.signal_id,
                    source_module=self.__class__.__name__,
                )
                return False
            params["volume"] = self._format_decimal(event.quantity, lot_decimals)
        except (TypeError, ValueError):
            self.logger.exception(
                "Error processing volume/ordermin for pair %s: %s",
                event.trading_pair,
                source_module=self.__class__.__name__,
            )
            return False
        else:
            return True

    def _map_and_validate_order_type(
        self,
        params: dict[str, Any],
        event: TradeSignalApprovedEvent,
        pair_info: dict[str, Any],
    ) -> bool:
        """Map and validate the order type, setting price for limit orders."""
        order_type = event.order_type.lower()
        pair_decimals = pair_info.get("pair_decimals")

        if pair_decimals is None:
            self.logger.error(
                "Missing pair_decimals for pair %s. Cannot format price.",
                event.trading_pair,
                source_module=self.__class__.__name__,
            )
            return False

        if order_type == "limit":
            return self._handle_limit_order(params, event, pair_decimals)
        if order_type == "market":
            params["ordertype"] = "market"
            return True
        self.logger.error(
            "Unsupported order type '%s' for Kraken translation. Signal %s.",
            event.order_type,
            event.signal_id,
            source_module=self.__class__.__name__,
        )
        return False

    def _handle_limit_order(self, params: dict[str, Any], event: TradeSignalApprovedEvent, pair_decimals: int) -> bool:
        """Handle limit order specific parameters and validation."""
        params["ordertype"] = "limit"
        if event.limit_price is None:
            self.logger.error(
                "Limit price is required for limit order. Signal %s.",
                event.signal_id,
                source_module=self.__class__.__name__,
            )
            return False
        try:
            params["price"] = self._format_decimal(event.limit_price, pair_decimals)
        except (TypeError, ValueError):
            self.logger.exception(
                "Error processing limit price for pair %s: %s",
                event.trading_pair,
                source_module=self.__class__.__name__,
            )
            return False
        else:
            return True

    def _handle_sl_tp_warnings(self, event: TradeSignalApprovedEvent) -> None:
        """Handle warnings for stop-loss and take-profit parameters."""
        if event.sl_price or event.tp_price:
            self.logger.warning(
                "SL/TP prices in signal %s; handling deferred.",
                event.signal_id,
                source_module=self.__class__.__name__,
            )

    async def _handle_add_order_response(
        self,
        result: dict[str, Any],
        originating_event: TradeSignalApprovedEvent,
        cl_ord_id: str,
    ) -> None:
        """Process the response from the AddOrder API call and publish initial status.

        Checks for errors, stores order mapping, publishes execution report, and starts monitoring.
        """
        if not result:
            # Should not happen if _make_private_request works correctly, but
            # check anyway
            self.logger.error(
                "Received empty response for AddOrder call related to signal %s",
                originating_event.signal_id,
                source_module=self.__class__.__name__,
            )
            await self._publish_error_execution_report(originating_event, "Empty API response", cl_ord_id)
            return

        # Check for API-level errors first
        if result.get("error"):
            # Kraken errors are usually a list[Any] of strings
            error_msg = str(result["error"])
            self.logger.error(
                "AddOrder API call failed for signal %s (cl_ord_id: %s): %s",
                originating_event.signal_id,
                cl_ord_id,
                error_msg,
                source_module=self.__class__.__name__,
            )
            # Publish REJECTED/ERROR status
            await self._publish_error_execution_report(originating_event, error_msg, cl_ord_id)
            return

        # Process successful response
        try:
            kraken_result_data = result.get("result", {})
            txids = kraken_result_data.get("txid")
            descr = kraken_result_data.get("descr", {}).get("order", "N/A")

            if txids and isinstance(txids, list):
                # Handle multiple order IDs (e.g., for conditional orders or order chains)
                if len(txids) > 1:
                    self.logger.info(
                        "Received multiple order IDs (%d) for signal %s",
                        len(txids),
                        originating_event.signal_id,
                        source_module=self.__class__.__name__,
                    )

                # Process all returned order IDs
                for idx, kraken_order_id in enumerate(txids):
                    # For multiple orders, append index to client order ID to maintain uniqueness
                    indexed_cl_ord_id = cl_ord_id if idx == 0 else f"{cl_ord_id}-{idx}"

                    self.logger.info(
                        "Order %d/%d via API for signal %s: cl_ord_id=%s, TXID=%s, Descr=%s",
                        idx + 1,
                        len(txids),
                        originating_event.signal_id,
                        indexed_cl_ord_id,
                        kraken_order_id,
                        descr,
                        source_module=self.__class__.__name__,
                    )

                    # Store the mapping for future reference (e.g., cancellation, status checks)
                    self._order_map[indexed_cl_ord_id] = kraken_order_id

                    # Enterprise Order ID Mapping[Any, Any] - Update bidirectional mappings
                    self._update_order_id_mapping(indexed_cl_ord_id, kraken_order_id)

                    # Add to pending orders tracking for WebSocket correlation
                    self._pending_orders_by_cl_ord_id[indexed_cl_ord_id] = originating_event

                    # Publish initial "NEW" execution report for each order
                    report = ExecutionReportEvent(
                        source_module=self.__class__.__name__,
                        event_id=UUID(
                            int=int(time.time() * 1000000) + idx,
                        ),  # Ensure unique event IDs
                        timestamp=datetime.now(UTC),
                        signal_id=originating_event.signal_id,
                        exchange_order_id=kraken_order_id,
                        client_order_id=indexed_cl_ord_id,
                        trading_pair=originating_event.trading_pair,
                        exchange=self.config.get("exchange.name", "kraken"),
                        order_status="NEW",
                        order_type=originating_event.order_type,
                        side=originating_event.side,
                        quantity_ordered=originating_event.quantity,
                        quantity_filled=Decimal(0),
                        limit_price=originating_event.limit_price,
                        average_fill_price=None,
                        commission=None,
                        commission_asset=None,
                        timestamp_exchange=None,
                        error_message=None,
                    )

                    # Using asyncio.create_task for fire-and-forget publishing
                    task = asyncio.create_task(self.pubsub.publish(report))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

                    self.logger.debug(
                        "Published NEW ExecutionReport for %s / %s",
                        indexed_cl_ord_id,
                        kraken_order_id,
                        source_module=self.__class__.__name__,
                    )

                    # Start monitoring each order status
                    self._start_order_monitoring(
                        indexed_cl_ord_id,
                        kraken_order_id,
                        originating_event,
                    )
            else:
                # This case indicates success HTTP status but unexpected result
                # format
                error_msg = "AddOrder response missing or invalid 'txid' field."
                self.logger.error(
                    "%s cl_ord_id: %s. Response: %s",
                    error_msg,
                    cl_ord_id,
                    result,
                    source_module=self.__class__.__name__,
                )
                await self._publish_error_execution_report(originating_event, error_msg, cl_ord_id)

        except Exception:  # Catch potential errors during response parsing
            self.logger.exception(
                "Error processing successful AddOrder response for signal %s (cl_ord_id: %s): %s",
                originating_event.signal_id,
                cl_ord_id,
                source_module=self.__class__.__name__,
            )
            await self._publish_error_execution_report(
                originating_event,
                "Internal error processing response after AddOrder",
                cl_ord_id,
            )

    async def _connect_websocket(self) -> None:
        """Connect to the exchange WebSocket API with enterprise-grade connection management.

        Establishes WebSocket connection with authentication, subscription management,
        and robust error handling with automatic reconnection capabilities.
        """
        if not self.websocket_client:
            self.logger.error(
                "WebSocket client not initialized. Cannot establish connection.",
                source_module=self.__class__.__name__,
            )
            return

        if self._websocket_connection_state in ["CONNECTED", "AUTHENTICATED"]:
            self.logger.info(
                "WebSocket already connected/authenticated. Skipping connection attempt.",
                source_module=self.__class__.__name__,
            )
            return

        try:
            self._websocket_connection_state = "CONNECTING"
            self.logger.info("Attempting to establish WebSocket connection...", source_module=self.__class__.__name__)

            # Cancel any existing connection task
            if self._websocket_connection_task and not self._websocket_connection_task.done():
                self._websocket_connection_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._websocket_connection_task

            # Establish connection using the WebSocket client
            await self.websocket_client.connect()

            # Update state tracking
            self._websocket_connection_state = "AUTHENTICATED"  # KrakenWebSocketClient handles auth internally
            self._current_reconnect_attempts = 0

            # Subscribe to private channels for order updates
            await self._subscribe_to_order_channels()

            self.logger.info(
                "WebSocket connection established and authenticated successfully",
                source_module=self.__class__.__name__,
            )

        except Exception:
            self.logger.exception("WebSocket connection attempt failed", source_module=self.__class__.__name__)
            self._websocket_connection_state = "ERROR"

            # Attempt reconnection if we're still running
            if self._is_running:
                await self._handle_websocket_reconnect()

    async def _subscribe_to_order_channels(self) -> None:
        """Subscribe to private WebSocket channels for order updates.

        Subscribes to channels needed for order lifecycle management:
        - ownTrades: For execution reports and fill notifications
        - openOrders: For order status updates
        """
        if not self.websocket_client or self._websocket_connection_state != "AUTHENTICATED":
            self.logger.warning(
                "Cannot subscribe to order channels - WebSocket not authenticated",
                source_module=self.__class__.__name__,
            )
            return

        try:
            # Get active trading pairs from configuration
            active_pairs = self.config.get_list("trading.pairs", [])
            if not active_pairs:
                self.logger.warning(
                    "No trading pairs configured. WebSocket subscriptions may be limited.",
                    source_module=self.__class__.__name__,
                )
                # Use a default set or subscribe to all available pairs
                active_pairs = ["XBT/USD", "ETH/USD"]  # Default pairs for order monitoring

            # Note: Private channels in Kraken WebSocket are typically subscribed to globally
            # The actual subscription is handled internally by KrakenWebSocketClient
            # which subscribes to ownTrades and openOrders channels automatically

            self._subscribed_channels.update(["ownTrades", "openOrders"])

            self.logger.info(
                "Successfully subscribed to order update channels: %s",
                ", ".join(self._subscribed_channels),
                source_module=self.__class__.__name__,
            )

        except Exception:
            self.logger.exception("Failed to subscribe to order channels", source_module=self.__class__.__name__)

    async def _handle_websocket_message(self, message: dict[str, Any]) -> None:
        """Process a message received from the exchange WebSocket with enterprise-grade handling.

        Handles different message types including order updates, fill notifications,
        and maintains bidirectional order ID mapping between internal and exchange IDs.

        Args:
            message: Raw message from WebSocket containing order updates or other data
        """
        try:
            self.logger.debug("Received WebSocket message: %s", message, source_module=self.__class__.__name__)

            # Extract message type and relevant data
            message_type = self._parse_message_type(message)

            if message_type == "ORDER_UPDATE":
                await self._process_order_update_message(message)
            elif message_type == "FILL_NOTIFICATION":
                await self._process_fill_notification_message(message)
            elif message_type == "AUTH_RESPONSE":
                await self._process_auth_response_message(message)
            elif message_type == "SUBSCRIPTION_ACK":
                await self._process_subscription_ack_message(message)
            elif message_type == "HEARTBEAT":
                # Heartbeat messages don't require special processing
                pass
            else:
                self.logger.debug(
                    "Received unhandled WebSocket message type: %s",
                    message_type,
                    source_module=self.__class__.__name__,
                )

        except Exception:
            self.logger.exception(
                "Error processing WebSocket message. Message: %s",
                message,
                source_module=self.__class__.__name__,
            )

    def _parse_message_type(self, message: dict[str, Any] | list[Any]) -> str:
        """Parse WebSocket message to determine its type.

        Args:
            message: Raw WebSocket message

        Returns:
            String identifier for message type
        """
        if isinstance(message, dict):
            return self._parse_dict_message(message)
        if isinstance(message, list) and len(message) >= 3:
            return self._parse_list_message(message)
        return "UNKNOWN"

    def _parse_dict_message(self, message: dict[str, Any]) -> str:
        """Parse dictionary-format WebSocket message."""
        # Check for event-based messages
        if "event" in message:
            return self._parse_event_message(message["event"])

        # Check for channel-based messages
        channel = message.get("channel") or message.get("channelName")
        if channel:
            return self._parse_channel_message(channel)

        return "UNKNOWN"

    def _parse_event_message(self, event: str | None) -> str:
        """Parse event-based message types."""
        event_type_map = {
            "subscriptionStatus": "SUBSCRIPTION_ACK",
            "systemStatus": "HEARTBEAT",
        }

        if event in event_type_map:
            return event_type_map[event]
        if event:
            return f"EVENT_{event.upper()}"
        return "EVENT_UNKNOWN"

    def _parse_channel_message(self, channel: str) -> str:
        """Parse channel-based message types."""
        channel_type_map = {
            "ownTrades": "FILL_NOTIFICATION",
            "openOrders": "ORDER_UPDATE",
        }
        return channel_type_map.get(channel, "UNKNOWN")

    def _parse_list_message(self, message: list[Any]) -> str:
        """Parse list-format WebSocket message (typical for Kraken)."""
        if len(message) > 2:
            channel_name = message[2]
            return self._parse_channel_message(str(channel_name))
        return "UNKNOWN"

    async def _process_order_update_message(self, message: dict[str, Any]) -> None:
        """Process order status update messages from WebSocket.

        Updates order ID mappings and publishes execution reports for status changes.
        """
        try:
            # Extract order data from message (format varies by exchange)
            order_data = self._extract_order_data_from_message(message)
            if not order_data:
                return

            exchange_order_id = order_data.get("orderid") or order_data.get("txid")
            client_order_id = order_data.get("userref") or order_data.get("clientOrderId")

            # Try to resolve client order ID from our mapping if not in message
            if not client_order_id and exchange_order_id:
                client_order_id = self._exchange_to_internal_order_id.get(exchange_order_id)

            # Update bidirectional mapping if we have both IDs
            if exchange_order_id and client_order_id:
                self._update_order_id_mapping(client_order_id, exchange_order_id)

            if client_order_id:
                # Create and publish execution report
                await self._create_execution_report_from_websocket(
                    order_data=order_data,
                    client_order_id=client_order_id,
                    exchange_order_id=exchange_order_id,
                    message_type="ORDER_UPDATE",
                )

                # Clean up completed orders from pending tracking
                order_status = order_data.get("status", "").lower()
                if order_status in ["closed", "canceled", "cancelled", "expired", "rejected"]:
                    self._pending_orders_by_cl_ord_id.pop(client_order_id, None)

            else:
                self.logger.warning(
                    "Received order update for unknown order - no client_order_id mapping: %s",
                    order_data,
                    source_module=self.__class__.__name__,
                )

        except Exception:
            self.logger.exception("Error processing order update message", source_module=self.__class__.__name__)

    async def _process_fill_notification_message(self, message: dict[str, Any]) -> None:
        """Process trade fill notifications from WebSocket.

        Handles execution reports for partial and full fills.
        """
        try:
            # Extract trade data from message
            trade_data = self._extract_trade_data_from_message(message)
            if not trade_data:
                return

            for trade_info in trade_data:
                exchange_order_id = trade_info.get("ordertxid")
                if not exchange_order_id:
                    continue
                client_order_id = self._exchange_to_internal_order_id.get(exchange_order_id)

                if client_order_id:
                    await self._create_execution_report_from_websocket(
                        order_data=trade_info,
                        client_order_id=client_order_id,
                        exchange_order_id=exchange_order_id,
                        message_type="FILL_NOTIFICATION",
                    )
                else:
                    self.logger.warning(
                        "Received fill notification for unmapped order: %s",
                        exchange_order_id,
                        source_module=self.__class__.__name__,
                    )

        except Exception:
            self.logger.exception("Error processing fill notification", source_module=self.__class__.__name__)

    async def _process_auth_response_message(self, message: dict[str, Any]) -> None:
        """Process authentication response from WebSocket."""
        try:
            status = message.get("status")
            if status == "ok":
                self._websocket_connection_state = "AUTHENTICATED"
                self.logger.info("WebSocket authentication successful", source_module=self.__class__.__name__)
                await self._subscribe_to_order_channels()
            else:
                self.logger.error(
                    "WebSocket authentication failed: %s",
                    message.get("errorMessage", "Unknown error"),
                    source_module=self.__class__.__name__,
                )
                self._websocket_connection_state = "ERROR"

        except Exception:
            self.logger.exception("Error processing auth response", source_module=self.__class__.__name__)

    async def _process_subscription_ack_message(self, message: dict[str, Any]) -> None:
        """Process subscription acknowledgment from WebSocket."""
        try:
            status = message.get("status")
            channel = message.get("channelName", "unknown")

            if status == "subscribed":
                self._subscribed_channels.add(channel)
                self.logger.info(
                    "Successfully subscribed to WebSocket channel: %s",
                    channel,
                    source_module=self.__class__.__name__,
                )
            elif status == "error":
                self.logger.error(
                    "Failed to subscribe to WebSocket channel %s: %s",
                    channel,
                    message.get("errorMessage", "Unknown error"),
                    source_module=self.__class__.__name__,
                )

        except Exception:
            self.logger.exception("Error processing subscription ack", source_module=self.__class__.__name__)

    def _extract_order_data_from_message(self, message: dict[str, Any] | list[Any]) -> dict[str, Any] | None:
        """Extract order data from WebSocket message.

        Handles different message formats from Kraken WebSocket.
        """
        try:
            # Handle different message formats
            if isinstance(message, list) and len(message) >= 2:
                # List format: [channel_id, data, channel_name, ...]
                data = message[1]
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if isinstance(first_item, dict):
                        return cast("dict[str, Any]", first_item)
                if isinstance(data, dict):
                    return cast("dict[str, Any]", data)
            elif isinstance(message, dict):
                # Dict format with direct order data
                result = message.get("data", message)
                if isinstance(result, dict):
                    return cast("dict[str, Any]", result)
                return None

        except Exception:
            self.logger.exception("Error extracting order data from message", source_module=self.__class__.__name__)
            return None
        else:
            return None

    def _extract_trade_data_from_message(self, message: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
        """Extract trade data from WebSocket message."""
        try:
            if isinstance(message, list) and len(message) >= 2:
                data = message[1]
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
            elif isinstance(message, dict):
                trades = message.get("data", [])
                if isinstance(trades, list):
                    return trades
                if isinstance(trades, dict):
                    return [trades]

        except Exception:
            self.logger.exception("Error extracting trade data from message", source_module=self.__class__.__name__)
            return []
        else:
            return []

    def _update_order_id_mapping(self, client_order_id: str, exchange_order_id: str) -> None:
        """Update bidirectional order ID mapping.

        Maintains mappings between internal client order IDs and exchange order IDs.
        """
        self._internal_to_exchange_order_id[client_order_id] = exchange_order_id
        self._exchange_to_internal_order_id[exchange_order_id] = client_order_id

        # Also update legacy mapping for backward compatibility
        self._order_map[client_order_id] = exchange_order_id

        self.logger.debug(
            "Updated order ID mapping: %s -> %s",
            client_order_id,
            exchange_order_id,
            source_module=self.__class__.__name__,
        )

    async def _create_execution_report_from_websocket(
        self,
        order_data: dict[str, Any],
        client_order_id: str,
        exchange_order_id: str | None,
        message_type: str,
    ) -> None:
        """Create and publish ExecutionReportEvent from WebSocket data.

        Converts WebSocket order/trade data into standardized execution reports.
        """
        try:
            # Extract signal ID from pending orders if available
            signal_id = None
            originating_event = self._pending_orders_by_cl_ord_id.get(client_order_id)
            if originating_event:
                signal_id = originating_event.signal_id

            # Parse order details from WebSocket data
            order_status = self._map_exchange_status_to_internal(
                order_data.get("status", "unknown"),
            )

            # Extract quantities and prices
            quantity_ordered = Decimal(order_data.get("vol", "0"))
            quantity_filled = Decimal(order_data.get("vol_exec", "0"))

            # Extract prices
            limit_price = None
            avg_fill_price = None

            if "price" in order_data:
                price_str = order_data["price"]
                if price_str and price_str != "0":
                    if message_type == "FILL_NOTIFICATION":
                        avg_fill_price = Decimal(price_str)
                    else:
                        limit_price = Decimal(price_str)

            # Extract trading pair
            trading_pair = "UNKNOWN"
            if "pair" in order_data:
                kraken_pair = order_data["pair"]
                internal_pair = self._map_kraken_pair_to_internal(kraken_pair)
                trading_pair = internal_pair or kraken_pair

            # Extract side and order type
            descr = order_data.get("descr", {})
            side = descr.get("type", "unknown").upper()
            order_type = descr.get("ordertype", "unknown").upper()

            # Extract commission
            commission = None
            if order_data.get("fee"):
                commission = Decimal(order_data["fee"])

            # Create execution report
            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=UUID(int=int(time.time() * 1000000)),
                timestamp=datetime.now(UTC),
                signal_id=signal_id,
                exchange_order_id=exchange_order_id or "UNKNOWN",
                client_order_id=client_order_id,
                trading_pair=trading_pair,
                exchange=self.config.get("exchange.name", "kraken"),
                order_status=order_status,
                order_type=order_type,
                side=side,
                quantity_ordered=quantity_ordered,
                quantity_filled=quantity_filled,
                limit_price=limit_price,
                average_fill_price=avg_fill_price,
                commission=commission,
                commission_asset=self._get_quote_currency(trading_pair),
                timestamp_exchange=self._parse_exchange_timestamp(order_data),
                error_message=order_data.get("reason") if order_status in ["REJECTED", "CANCELLED"] else None,
            )

            # Publish the execution report
            publish_task = asyncio.create_task(self.pubsub.publish(report))
            self._background_tasks.add(publish_task)
            publish_task.add_done_callback(self._background_tasks.discard)

            self.logger.info(
                "Published WebSocket execution report: %s %s for %s",
                order_status,
                message_type,
                client_order_id,
                source_module=self.__class__.__name__,
            )

        except Exception:
            self.logger.exception(
                "Error creating execution report from WebSocket data",
                source_module=self.__class__.__name__,
            )

    def _map_exchange_status_to_internal(self, exchange_status: str) -> str:
        """Map exchange-specific order status to internal status.

        Args:
            exchange_status: Status from exchange (e.g., Kraken)

        Returns:
            Standardized internal status
        """
        status_map = {
            "pending": "NEW",
            "open": "OPEN",
            "closed": "FILLED",
            "canceled": "CANCELLED",
            "cancelled": "CANCELLED",
            "expired": "EXPIRED",
            "rejected": "REJECTED",
        }

        return status_map.get(exchange_status.lower(), exchange_status.upper())

    def _parse_exchange_timestamp(self, order_data: dict[str, Any]) -> datetime | None:
        """Parse exchange timestamp from order data."""
        try:
            timestamp_fields = ["opentm", "closetm", "timestamp", "time"]
            for field in timestamp_fields:
                if order_data.get(field):
                    timestamp_value = order_data[field]
                    if isinstance(timestamp_value, int | float):
                        return datetime.fromtimestamp(timestamp_value, tz=UTC)
                    if isinstance(timestamp_value, str):
                        try:
                            return datetime.fromtimestamp(float(timestamp_value), tz=UTC)
                        except ValueError:
                            # Log and continue to next field
                            self.logger.debug(
                                "Failed to parse timestamp from field %s with value %s",
                                field,
                                timestamp_value,
                                source_module=self.__class__.__name__,
                            )
                            continue
        except Exception:
            return None
        else:
            return None

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an open order on the exchange."""
        self.logger.info("Attempting to cancel order %s", exchange_order_id, source_module=self.__class__.__name__)
        uri_path = "/0/private/CancelOrder"
        params = {"txid": exchange_order_id}

        result = await self._make_private_request_with_retry(uri_path, params)

        if not result or result.get("error"):
            error_val = "Unknown cancel error"
            if result:
                error_val = result.get("error", "Unknown cancel error")
            error_detail = str(error_val)
            self.logger.error(
                "Failed to cancel order %s: %s",
                exchange_order_id,
                error_detail,
                source_module=self.__class__.__name__,
            )
            return False

        # Check response - successful cancellation might have count > 0
        count = result.get("result", {}).get("count", 0)
        if count > 0:
            self.logger.info(
                "Successfully initiated cancellation for order %s. Count: %s",
                exchange_order_id,
                count,
                source_module=self.__class__.__name__,
            )
            # Note: The status monitor will pick up the 'canceled' status and publish a report
            return True
        # Order might have already been closed/canceled
        self.logger.warning(
            "Cancel req for %s (count 0): order may be in terminal state.",
            exchange_order_id,
            source_module=self.__class__.__name__,
        )
        return False

    def _start_order_monitoring(
        self,
        cl_ord_id: str,
        kraken_order_id: str,
        originating_event: TradeSignalApprovedEvent,
    ) -> None:
        """Start monitoring tasks for a newly placed order."""
        # Start status monitoring
        monitor_task = asyncio.create_task(
            self._monitor_order_status(kraken_order_id, cl_ord_id, originating_event.signal_id),
        )
        # Store task reference for cancellation on stop
        self._order_monitoring_tasks[kraken_order_id] = monitor_task

        # For limit orders, also set up timeout monitoring
        if originating_event.order_type.upper() == "LIMIT":
            timeout_s = self.config.get_float("order.limit_order_timeout_s", 300.0)  # 5 mins default
            if timeout_s > 0:
                self.logger.info(
                    "Scheduling timeout check for limit order %s in %ss.",
                    kraken_order_id,
                    timeout_s,
                    source_module=self.__class__.__name__,
                )
                limit_order_timeout_task = asyncio.create_task(
                    self._monitor_limit_order_timeout(kraken_order_id, timeout_s),
                )
                self._background_tasks.add(limit_order_timeout_task)
                limit_order_timeout_task.add_done_callback(self._background_tasks.discard)

    async def _query_order_details(self, exchange_order_id: str) -> dict[str, Any] | None:
        """Query the exchange for order details with retry logic."""
        uri_path = "/0/private/QueryOrders"
        params = {"txid": exchange_order_id, "trades": "true"}  # Include trade info
        query_result = await self._make_private_request_with_retry(uri_path, params)

        if not query_result or query_result.get("error"):
            error_val = "Unknown query error"
            if query_result:
                error_val = query_result.get("error", "Unknown query error")
            error_str = str(error_val)
            self.logger.error(
                "Error querying order %s: %s",
                exchange_order_id,
                error_str,
                source_module=self.__class__.__name__,
            )
            if "EOrder:Unknown order" in error_str:
                self.logger.error(
                    "Order %s not found. Stopping monitoring for this reason.",
                    exchange_order_id,
                    source_module=self.__class__.__name__,
                )
            return None

        result_field = query_result.get("result")
        if not isinstance(result_field, dict):
            self.logger.error(
                "QueryOrders response for %s missing 'result' dict[str, Any] or is wrong type: %s",
                exchange_order_id,
                result_field,
                source_module=self.__class__.__name__,
            )
            return None

        order_data_any = result_field.get(exchange_order_id)
        if order_data_any is None:
            self.logger.warning(
                "Order %s not found in QueryOrders result's main dict[str, Any]. Retrying.",
                exchange_order_id,
                source_module=self.__class__.__name__,
            )
            return None

        if not isinstance(order_data_any, dict):
            self.logger.error(
                "Order data for %s is not a dict[str, Any]: %s",
                exchange_order_id,
                order_data_any,
                source_module=self.__class__.__name__,
            )
            return None

        return order_data_any

    async def _parse_order_data(
        self,
        order_data: dict[str, Any],
        exchange_order_id: str,
    ) -> tuple[str, Decimal, Decimal | None, Decimal | None] | None:
        """Parse relevant fields from the raw order data from the exchange."""
        try:
            current_status = order_data.get("status")
            if not isinstance(current_status, str):
                self.logger.error("Order %s has invalid or missing status: %s", exchange_order_id, current_status)
                return None

            current_filled_qty_str = order_data.get("vol_exec", "0")
            avg_fill_price_str = order_data.get("price")  # Average price for filled portion
            fee_str = order_data.get("fee")

            current_filled_qty = Decimal(current_filled_qty_str)
            avg_fill_price = Decimal(avg_fill_price_str) if avg_fill_price_str else None
            commission = Decimal(fee_str) if fee_str else None
        except Exception:  # Catches potential Decimal conversion errors or others
            self.logger.exception(
                "Error parsing numeric data for order %s. Data: %s. Error: %s",
                exchange_order_id,
                order_data,
                source_module=self.__class__.__name__,
            )
            return None
        else:
            return current_status, current_filled_qty, avg_fill_price, commission

    async def _handle_order_status_change(self, params: OrderStatusReportParameters) -> None:
        """Publish an execution report when order status or fill quantity changes."""
        self.logger.info(
            "Status change for %s: Status='%s', Filled=%s. Publishing report.",
            params.exchange_order_id,
            params.current_status,
            params.current_filled_qty,
            source_module=self.__class__.__name__,
        )
        await self._publish_status_execution_report(params)

    async def _handle_sl_tp_for_closed_order(
        self,
        exchange_order_id: str,
        client_order_id: str,
        signal_id: UUID | None,
        current_filled_qty: Decimal,
    ) -> None:
        """Handle SL/TP order placement if an entry order is fully filled."""
        if signal_id is None:
            return

        # Check if this is an entry order (not an SL/TP order itself)
        is_entry_order = not (client_order_id.startswith(("gf-sl-", "gf-tp-")))

        if is_entry_order and not await self._has_sl_tp_been_placed(signal_id):
            try:
                original_event = await self._get_originating_signal_event(signal_id)
                if original_event and (original_event.sl_price or original_event.tp_price):
                    self.logger.info(
                        "Order %s fully filled. Triggering SL/TP placement for signal %s.",
                        exchange_order_id,
                        signal_id,
                        source_module=self.__class__.__name__,
                    )
                    sl_tp_handling_task = asyncio.create_task(
                        self._handle_sl_tp_orders(original_event, exchange_order_id, current_filled_qty),
                    )
                    self._background_tasks.add(sl_tp_handling_task)
                    sl_tp_handling_task.add_done_callback(self._background_tasks.discard)
                else:
                    self.logger.info(
                        "Order %s fully filled, but no SL/TP prices found for signal %s.",
                        exchange_order_id,
                        signal_id,
                        source_module=self.__class__.__name__,
                    )
                    # Still mark as processed to avoid repeated checks
                    await self._mark_sl_tp_as_placed(signal_id)
            except Exception:
                self.logger.exception(
                    "Error in SL/TP handling for %s: %s",
                    exchange_order_id,
                    source_module=self.__class__.__name__,
                )

    async def _monitor_order_status(self, exchange_order_id: str, client_order_id: str, signal_id: UUID | None) -> None:
        """Monitor the status of a specific order via polling.

        Periodically check order status, publish update, and handle SL/TP order for filled order.
        """
        self._source_module = self.__class__.__name__  # Ensure source_module is set
        self.logger.info(
            "Starting status monitoring for order %s (cl=%s)",
            exchange_order_id,
            client_order_id,
            source_module=self._source_module,
        )

        poll_interval = self.config.get_float("order.status_poll_interval_s", 5.0)
        max_poll_duration = self.config.get_float("order.max_poll_duration_s", 3600.0)  # 1 hour default
        start_time = time.time()
        last_known_status: str | None = "NEW"
        last_known_filled_qty: Decimal = Decimal(0)

        while time.time() - start_time < max_poll_duration:
            await asyncio.sleep(poll_interval)

            order_details_result = await self._query_order_details(exchange_order_id)

            if order_details_result is None:  # Error querying, stop monitoring
                break

            order_data = order_details_result
            parsed_data = await self._parse_order_data(order_data, exchange_order_id)
            if not parsed_data:
                continue  # Error parsing, skip this update

            current_status, current_filled_qty, avg_fill_price, commission = parsed_data

            status_changed = current_status != last_known_status
            fill_increased = current_filled_qty > last_known_filled_qty

            if status_changed or fill_increased:
                status_change_params = OrderStatusReportParameters(
                    exchange_order_id=exchange_order_id,
                    client_order_id=client_order_id,
                    signal_id=signal_id,
                    order_data=order_data,
                    current_status=current_status,
                    current_filled_qty=current_filled_qty,
                    avg_fill_price=avg_fill_price,
                    commission=commission,
                )
                await self._handle_order_status_change(status_change_params)
                last_known_status = current_status
                last_known_filled_qty = current_filled_qty

                if current_status in ["closed", "canceled", "expired"]:
                    await self._handle_sl_tp_for_closed_order(
                        exchange_order_id,
                        client_order_id,
                        signal_id,
                        current_filled_qty,
                    )

            if current_status in ["closed", "canceled", "expired"]:
                self.logger.info(
                    "Order %s reached terminal state '%s'. Stopping monitoring.",
                    exchange_order_id,
                    current_status,
                    source_module=self._source_module,
                )
                break
        else:  # Loop finished due to timeout
            self.logger.warning(
                "Stopped monitoring order %s after timeout (%ss). Last status: %s",
                exchange_order_id,
                max_poll_duration,
                last_known_status,
                source_module=self._source_module,
            )

        self._order_monitoring_tasks.pop(exchange_order_id, None)

    async def _get_originating_signal_event(self, signal_id: UUID | None) -> TradeSignalApprovedEvent | None:
        """Retrieve the original signal event that led to an order.

        Uses the event store to fetch historical events.
        """
        if not self.event_store:
            self.logger.warning(
                "EventStore not available. Cannot retrieve originating signal event for %s.",
                signal_id,
                source_module=self.__class__.__name__,
            )
            return None
        if not signal_id:
            return None

        try:
            # Get the approved signal event
            events = await self.event_store.get_events_by_correlation(
                correlation_id=signal_id,
                event_types=[TradeSignalApprovedEvent],
            )

            if events:
                # Return the most recent approved signal
                return events[-1]  # type: ignore[no-any-return]

            self.logger.warning(
                f"No approved signal event found for signal_id: {signal_id}",
                source_module=self.__class__.__name__,
            )

        except Exception:
            self.logger.exception(
                f"Error retrieving signal event for {signal_id}",
                source_module=self.__class__.__name__,
            )
            return None
        else:
            return None

    async def _publish_error_execution_report(
        self,
        event: TradeSignalApprovedEvent,
        error_message: str,
        cl_ord_id: str | None,
        exchange_order_id: str | None = None,
    ) -> None:
        """Publish an ExecutionReportEvent for a failed/rejected order."""
        effective_exchange_order_id = exchange_order_id if exchange_order_id is not None else "NO_EXCHANGE_ID"
        effective_cl_ord_id = cl_ord_id if cl_ord_id else f"internal_{event.signal_id}_error"
        report = ExecutionReportEvent(
            source_module=self.__class__.__name__,
            event_id=UUID(int=int(time.time() * 1000000)),
            timestamp=datetime.now(UTC),
            signal_id=event.signal_id,
            exchange_order_id=effective_exchange_order_id,
            client_order_id=effective_cl_ord_id,
            trading_pair=event.trading_pair,
            exchange=self.config.get("exchange.name", "kraken"),
            order_status="REJECTED",  # Or "ERROR" depending on context
            order_type=event.order_type,
            side=event.side,
            quantity_ordered=event.quantity,
            quantity_filled=Decimal(0),
            limit_price=event.limit_price,
            average_fill_price=None,
            commission=None,
            commission_asset=None,
            timestamp_exchange=None,
            error_message=error_message,
        )
        publish_task = asyncio.create_task(self.pubsub.publish(report))
        self._background_tasks.add(publish_task)
        publish_task.add_done_callback(self._background_tasks.discard)
        self.logger.debug(
            "Published REJECTED/ERROR ExecutionReport for signal %s, cl_ord_id: %s",
            event.signal_id,
            cl_ord_id,
            source_module=self.__class__.__name__,
        )

    async def _publish_status_execution_report(self, params: OrderStatusReportParameters) -> None:
        """Publish ExecutionReportEvent based on polled status."""
        try:
            # Extract necessary fields from order_data (Kraken specific)
            descr = params.order_data.get("descr", {})
            order_type_str = descr.get("ordertype")
            side_str = descr.get("type")
            pair = descr.get("pair")  # Kraken pair name

            # Map pair back to internal name
            internal_pair_nullable = self._map_kraken_pair_to_internal(pair) if pair else "UNKNOWN"
            internal_pair = internal_pair_nullable if internal_pair_nullable is not None else "UNKNOWN"

            raw_vol = params.order_data.get("vol")  # Use a different variable name to avoid F841 if it's an issue
            quantity_ordered_val = Decimal(raw_vol) if raw_vol else Decimal(0)
            limit_price_str_val = descr.get("price")  # Price for limit orders
            limit_price_val = Decimal(limit_price_str_val) if limit_price_str_val else None

            # Determine commission asset (e.g., quote currency of the pair)
            commission_asset = None
            if params.commission:
                commission_asset = self._get_quote_currency(internal_pair)

            exchange_timestamp_val = None
            opentm = params.order_data.get("opentm")
            if opentm:
                exchange_timestamp_val = datetime.fromtimestamp(opentm, tz=UTC)

            # Include reason if status is 'canceled' or 'expired'
            error_message_val = params.order_data.get("reason")

            report_exchange_id = params.exchange_order_id
            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=UUID(int=int(time.time() * 1000000)),  # Generate a proper UUID
                timestamp=datetime.now(UTC),
                signal_id=params.signal_id,
                exchange_order_id=report_exchange_id,
                client_order_id=params.client_order_id,
                trading_pair=internal_pair,
                exchange=self.config.get("exchange.name", "kraken"),
                order_status=params.current_status.upper(),  # Standardize status
                order_type=order_type_str.upper() if order_type_str else "UNKNOWN",
                side=side_str.upper() if side_str else "UNKNOWN",
                quantity_ordered=quantity_ordered_val,
                quantity_filled=params.current_filled_qty,
                limit_price=limit_price_val,
                average_fill_price=params.avg_fill_price,
                commission=params.commission,
                commission_asset=commission_asset,
                timestamp_exchange=exchange_timestamp_val,
                error_message=error_message_val,
            )
            # Using asyncio.create_task for fire-and-forget publishing
            publish_status_task = asyncio.create_task(self.pubsub.publish(report))
            self._background_tasks.add(publish_status_task)
            publish_status_task.add_done_callback(self._background_tasks.discard)
            self.logger.debug(
                "Published %s ExecutionReport for %s / %s",
                params.current_status.upper(),
                params.client_order_id,
                params.exchange_order_id,
                source_module=self.__class__.__name__,
            )
        except Exception:
            self.logger.exception(
                "Error publishing execution report for order %s (cl_ord_id: %s): %s",
                params.exchange_order_id,
                params.client_order_id,
                source_module=self.__class__.__name__,
            )

    def _map_kraken_pair_to_internal(self, kraken_pair: str) -> str | None:
        """Map Kraken pair name (e.g., XXBTZUSD) back to internal name (e.g., BTC/USD)."""
        for internal_name, info in self._pair_info.items():
            if (
                info.get("altname") == kraken_pair
                or info.get("wsname") == kraken_pair
                or info.get("kraken_pair_key") == kraken_pair
            ):
                return internal_name

        self.logger.warning(
            "Could not map Kraken pair '%s' back to internal name.",
            kraken_pair,
            source_module=self.__class__.__name__,
        )
        return None

    def _get_quote_currency(self, internal_pair: str) -> str | None:
        """Get the quote currency for an internal pair name."""
        info = self._pair_info.get(internal_pair)
        return cast("str | None", info.get("quote")) if info else None  # Added cast

    async def _has_sl_tp_been_placed(self, signal_id: UUID | None) -> bool:
        """Check if SL/TP orders have already been placed for a signal."""
        if signal_id is None:
            return False
        return signal_id in self._placed_sl_tp_signals

    async def _mark_sl_tp_as_placed(self, signal_id: UUID | None) -> None:
        """Mark that SL/TP orders have been placed for a signal."""
        if signal_id is not None:
            self._placed_sl_tp_signals.add(signal_id)

    async def _handle_sl_tp_orders(
        self,
        originating_event: TradeSignalApprovedEvent,
        filled_order_id: str,
        filled_quantity: Decimal,
    ) -> None:
        """Place SL and/or TP orders contingent on the filled entry order.

        Creates stop-loss and take-profit orders based on the original signal parameters.
        Uses batch order placement when both SL and TP are needed for efficiency.
        """
        self.logger.info(
            "Handling SL/TP placement for filled order %s (Signal: %s)",
            filled_order_id,
            originating_event.signal_id,
            source_module=self.__class__.__name__,
        )

        kraken_pair_name = self._get_kraken_pair_name(originating_event.trading_pair)
        if not kraken_pair_name:
            return  # Error logged in helper

        # Determine side for SL/TP (opposite of entry)
        exit_side = "sell" if originating_event.side.upper() == "BUY" else "buy"
        current_pair_info = self._pair_info.get(originating_event.trading_pair)

        # --- Enterprise Implementation: Use Adapter Pattern for Batch Orders ---
        if self._adapter and originating_event.sl_price and originating_event.tp_price:
            # Both SL and TP are present - use batch placement for efficiency
            self.logger.info(
                "Using batch order placement for both SL and TP orders",
                source_module=self.__class__.__name__,
            )

            batch_request = BatchOrderRequest(orders=[], validate_only=False)

            # Prepare Stop Loss Order
            sl_order = OrderRequest(
                trading_pair=originating_event.trading_pair,
                side=exit_side.upper(),
                order_type="stop-loss",
                quantity=filled_quantity,
                stop_price=originating_event.sl_price,
                client_order_id=f"gf-sl-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}",
                metadata={
                    "reduce_only": "true",
                    "originating_signal": str(originating_event.signal_id),
                },
            )
            batch_request.orders.append(sl_order)

            # Prepare Take Profit Order
            tp_order = OrderRequest(
                trading_pair=originating_event.trading_pair,
                side=exit_side.upper(),
                order_type="take-profit",
                quantity=filled_quantity,
                stop_price=originating_event.tp_price,
                client_order_id=f"gf-tp-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}",
                metadata={
                    "reduce_only": "true",
                    "originating_signal": str(originating_event.signal_id),
                },
            )
            batch_request.orders.append(tp_order)

            try:
                # Place both orders together
                batch_response = await self._adapter.place_batch_orders([sl_order, tp_order])

                # Check if any orders were successful
                successful_orders = [r for r in batch_response if r.get("success", False)]
                if successful_orders:
                    # Process each order result
                    for i, order_result in enumerate(batch_response):
                        if order_result.get("success") and order_result.get("exchange_order_ids"):
                            order_type = "SL" if i == 0 else "TP"
                            exchange_order_id = order_result["exchange_order_ids"][0]
                            client_order_id = order_result.get("client_order_id")

                            self.logger.info(
                                "%s order placed successfully: cl_ord_id=%s, exchange_id=%s",
                                order_type,
                                client_order_id,
                                exchange_order_id,
                                source_module=self.__class__.__name__,
                            )

                            # Update order mappings
                            if client_order_id:
                                self._order_map[client_order_id] = exchange_order_id
                                self._update_order_id_mapping(client_order_id, exchange_order_id)
                                self._pending_orders_by_cl_ord_id[client_order_id] = originating_event

                            # Publish execution report
                            if client_order_id:
                                await self._publish_initial_execution_report(
                                    originating_event=originating_event,
                                    client_order_id=client_order_id,
                                    exchange_order_id=exchange_order_id,
                                    order_type=batch_request.orders[i].order_type,
                                )

                                # Start monitoring
                                self._start_order_monitoring(client_order_id, exchange_order_id, originating_event)
                        else:
                            order_type = "SL" if i == 0 else "TP"
                            self.logger.error(
                                "%s order placement failed: %s",
                                order_type,
                                order_result.get("error_message", "Unknown error"),
                                source_module=self.__class__.__name__,
                            )
                else:
                    self.logger.error(
                        "Batch order placement failed: %s",
                        "Batch order placement failed",
                        source_module=self.__class__.__name__,
                    )
                    # Fall back to individual placement
                    await self._place_sl_tp_individually(
                        originating_event,
                        kraken_pair_name,
                        exit_side,
                        filled_quantity,
                        current_pair_info,
                    )

            except Exception:
                self.logger.exception(
                    "Exception during batch order placement. Falling back to individual placement.",
                    source_module=self.__class__.__name__,
                )
                # Fall back to individual placement
                await self._place_sl_tp_individually(
                    originating_event,
                    kraken_pair_name,
                    exit_side,
                    filled_quantity,
                    current_pair_info,
                )
        else:
            # Either adapter not available or only one of SL/TP is set
            # Use individual placement
            await self._place_sl_tp_individually(
                originating_event,
                kraken_pair_name,
                exit_side,
                filled_quantity,
                current_pair_info,
            )

        # Mark SL/TP as placed for this signal
        await self._mark_sl_tp_as_placed(originating_event.signal_id)

    async def _place_sl_tp_individually(
        self,
        originating_event: TradeSignalApprovedEvent,
        kraken_pair_name: str,
        exit_side: str,
        filled_quantity: Decimal,
        current_pair_info: dict[str, Any] | None,
    ) -> None:
        """Place SL and TP orders individually using legacy method.

        This is the fallback method when batch placement is not available
        or when only one of SL/TP is needed.
        """
        # Place Stop Loss Order
        if originating_event.sl_price:
            sl_request_params = ContingentOrderParamsRequest(
                pair_name=kraken_pair_name,
                order_side=exit_side,
                contingent_order_type="stop-loss",
                trigger_price=originating_event.sl_price,
                volume=filled_quantity,
                pair_details=current_pair_info,
                originating_signal_id=originating_event.signal_id,
                log_marker="SL",
            )
            sl_params = self._prepare_contingent_order_params(sl_request_params)

            if sl_params:
                sl_cl_ord_id = f"gf-sl-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                sl_params["cl_ord_id"] = sl_cl_ord_id
                sl_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    "Placing SL order for signal %s with cl_ord_id %s",
                    originating_event.signal_id,
                    sl_cl_ord_id,
                    source_module=self.__class__.__name__,
                )

                sl_result = await self._make_private_request_with_retry("/0/private/AddOrder", sl_params)
                # Handle SL order placement response (publish report, start monitoring)
                await self._handle_add_order_response(sl_result, originating_event, sl_cl_ord_id)

        # Place Take Profit Order
        if originating_event.tp_price:
            tp_request_params = ContingentOrderParamsRequest(
                pair_name=kraken_pair_name,
                order_side=exit_side,
                contingent_order_type="take-profit",
                trigger_price=originating_event.tp_price,
                volume=filled_quantity,
                pair_details=current_pair_info,
                originating_signal_id=originating_event.signal_id,
                log_marker="TP",
            )
            tp_params = self._prepare_contingent_order_params(tp_request_params)

            if tp_params:
                tp_cl_ord_id = f"gf-tp-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                tp_params["cl_ord_id"] = tp_cl_ord_id
                tp_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    "Placing TP order for signal %s with cl_ord_id %s",
                    originating_event.signal_id,
                    tp_cl_ord_id,
                    source_module=self.__class__.__name__,
                )

                tp_result = await self._make_private_request_with_retry("/0/private/AddOrder", tp_params)
                # Handle TP order placement response (publish report, start monitoring)
                await self._handle_add_order_response(tp_result, originating_event, tp_cl_ord_id)

    async def _monitor_limit_order_timeout(self, exchange_order_id: str, timeout_seconds: float) -> None:
        """Check if a limit order is filled after a timeout and cancel if not."""
        await asyncio.sleep(timeout_seconds)
        self.logger.info(
            "Timeout reached for limit order %s. Checking status.",
            exchange_order_id,
            source_module=self.__class__.__name__,
        )

        uri_path = "/0/private/QueryOrders"
        params = {"txid": exchange_order_id}
        query_result = await self._make_private_request_with_retry(uri_path, params)

        if not query_result or query_result.get("error"):
            error_val = query_result.get("error", "Unknown query error")
            error_str = str(error_val)
            self.logger.error(
                "Error querying order %s: %s",
                exchange_order_id,
                error_str,
                source_module=self.__class__.__name__,
            )
            if "EOrder:Unknown order" in error_str:
                self.logger.error(
                    "Order %s not found. Stopping monitoring for this reason.",
                    exchange_order_id,
                    source_module=self.__class__.__name__,
                )
            return  # Cannot determine status, don't cancel arbitrarily

        order_data = query_result.get("result", {}).get(exchange_order_id)
        if not order_data:
            log_msg = f"Order {exchange_order_id} not found during timeout check"
            log_msg += " (already closed/canceled?)."
            self.logger.warning(log_msg, source_module=self.__class__.__name__)
            return  # Order likely already closed or canceled

        status = order_data.get("status")
        if status in ["open", "pending"]:
            log_msg = f"Limit order {exchange_order_id} still '{status}'"
            log_msg += f" after {timeout_seconds}s timeout. Attempting cancellation."
            self.logger.warning(log_msg, source_module=self.__class__.__name__)
            # Call cancel_order method
            cancel_success = await self.cancel_order(exchange_order_id)
            if not cancel_success:
                self.logger.error(
                    "Failed to cancel timed-out limit order %s.",
                    exchange_order_id,
                    source_module=self.__class__.__name__,
                )
            # The cancel_order method should publish the CANCELED report
        else:
            log_msg = f"Limit order {exchange_order_id} already in terminal state '{status}'"
            log_msg += " during timeout check."
            self.logger.info(log_msg, source_module=self.__class__.__name__)

    def _prepare_contingent_order_params(self, request: ContingentOrderParamsRequest) -> dict[str, Any] | None:
        """Prepare parameters for SL/TP orders, including validation."""
        params = {
            "pair": request.pair_name,
            "type": request.order_side,
            "ordertype": request.contingent_order_type,
        }

        if not request.pair_details:
            log_msg = f"Missing pair_details for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.error(log_msg, source_module=self.__class__.__name__)
            return None

        # Validate and format volume
        lot_decimals = request.pair_details.get("lot_decimals")
        if lot_decimals is None:  # Basic check
            log_msg = f"Missing lot_decimals for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.error(log_msg, source_module=self.__class__.__name__)
            return None

        try:
            params["volume"] = self._format_decimal(request.volume, lot_decimals)
        except Exception:
            log_msg = f"Error formatting volume for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.exception(log_msg, source_module=self.__class__.__name__)
            return None

        # Validate and format price(s)
        pair_decimals = request.pair_details.get("pair_decimals")
        if pair_decimals is None:
            log_msg = f"Missing pair_decimals for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.error(log_msg, source_module=self.__class__.__name__)
            return None

        try:
            params["price"] = self._format_decimal(request.trigger_price, pair_decimals)
            if request.limit_price is not None:
                params["price2"] = self._format_decimal(request.limit_price, pair_decimals)
        except Exception:
            log_msg = f"Error formatting price for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.exception(log_msg, source_module=self.__class__.__name__)
            return None

        # Add other necessary parameters (e.g., timeinforce if needed)
        return params

    def _get_kraken_pair_name(self, internal_pair: str) -> str | None:
        """Get the Kraken pair name from stored info."""
        info = self._pair_info.get(internal_pair)
        name = info.get("altname") if info else None
        if not name:
            self.logger.error(
                "Could not find Kraken pair name for internal pair '%s'",
                internal_pair,
                source_module=self.__class__.__name__,
            )
        return name

    async def handle_close_position_command(self, event: ClosePositionCommand) -> None:
        """Handle emergency position closure during HALT.

        Args:
            event: ClosePositionCommand containing position details to close
        """
        try:
            self.logger.warning(
                f"Processing emergency position closure for {event.trading_pair}",
                source_module=self.__class__.__name__,
                context={
                    "quantity": str(event.quantity),
                    "side": event.side,
                    "reason": "HALT triggered",
                },
            )

            # Get Kraken pair name
            pair_name = self._get_kraken_pair_name(event.trading_pair)
            if not pair_name:
                self.logger.critical(
                    f"Cannot close position: Unknown pair {event.trading_pair}",
                    source_module=self.__class__.__name__,
                )
                return

            # Create market order for immediate execution
            order_params = {
                "pair": pair_name,
                "type": event.side.lower(),
                "ordertype": "market",
                "volume": str(event.quantity),
                "validate": False,  # Skip validation for emergency orders
            }

            # Add emergency flag for special handling
            order_params["userref"] = "HALT_CLOSE"

            # Execute order with priority handling
            result = await self._place_order_with_priority(order_params)

            if result and not result.get("error"):
                self.logger.info(
                    "Emergency close order placed successfully",
                    source_module=self.__class__.__name__,
                    context={"order_id": result.get("result", {}).get("txid")},
                )
            else:
                self.logger.critical(
                    "Failed to place emergency close order",
                    source_module=self.__class__.__name__,
                    context={"error": result.get("error") if result else "No response"},
                )

        except Exception:
            self.logger.critical(
                "Critical error during emergency position closure",
                source_module=self.__class__.__name__,
            )

    async def _place_order_with_priority(self, params: dict[str, Any]) -> dict[str, Any]:
        """Place order with priority handling for emergencies.

        Args:
            params: Order parameters

        Returns:
            API response dict[str, Any]
        """
        # Bypass normal rate limiting for emergency orders
        if params.get("userref") == "HALT_CLOSE":
            # Direct placement without rate limit wait
            return await self._make_private_request("/0/private/AddOrder", params)
        # Normal rate-limited placement
        await self.rate_limiter.wait_for_private_capacity()
        return await self._make_private_request("/0/private/AddOrder", params)

    async def _disconnect_websocket(self) -> None:
        """Disconnect WebSocket connection with graceful cleanup.

        Properly disconnects WebSocket connection, cancels tasks, and cleans up state.
        """
        if not self.websocket_client or self._websocket_connection_state in [
            "DISCONNECTED",
            "DISABLED",
        ]:
            return

        try:
            self.logger.info("Initiating WebSocket disconnection...", source_module=self.__class__.__name__)

            # Cancel connection task if running
            if self._websocket_connection_task and not self._websocket_connection_task.done():
                self._websocket_connection_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._websocket_connection_task

            # Disconnect the WebSocket client
            await self.websocket_client.disconnect()

            # Clean up state
            self._websocket_connection_state = "DISCONNECTED"
            self._subscribed_channels.clear()
            self._websocket_auth_token = None
            self._current_reconnect_attempts = 0

            self.logger.info("WebSocket disconnection completed", source_module=self.__class__.__name__)

        except Exception:
            self.logger.exception("Error during WebSocket disconnection", source_module=self.__class__.__name__)

    async def _handle_websocket_reconnect(self) -> None:
        """Handle WebSocket reconnection with exponential backoff.

        Implements enterprise-grade reconnection logic with configurable retry limits
        and exponential backoff to prevent overwhelming the exchange.
        """
        if not self._is_running:
            self.logger.info(
                "Service is shutting down. Skipping WebSocket reconnection.",
                source_module=self.__class__.__name__,
            )
            return

        if self._current_reconnect_attempts >= self._max_reconnect_attempts:
            self.logger.error(
                "Max WebSocket reconnect attempts (%d) reached. Giving up.",
                self._max_reconnect_attempts,
                source_module=self.__class__.__name__,
            )
            self._websocket_connection_state = "ERROR"

            # Optionally publish system alert for monitoring
            # Future[Any] enhancement: Integrate with monitoring service alerts
            return

        self._current_reconnect_attempts += 1

        # Calculate delay with exponential backoff and jitter
        base_delay = self._reconnect_delay_seconds
        exponential_delay = base_delay * (2 ** (self._current_reconnect_attempts - 1))
        max_delay = 300  # Cap at 5 minutes
        delay = min(exponential_delay, max_delay)

        # Add jitter to prevent thundering herd
        jitter = secrets.SystemRandom().uniform(0, delay * 0.1)
        total_delay = delay + jitter

        self.logger.info(
            "Attempting WebSocket reconnection %d/%d in %.2f seconds...",
            self._current_reconnect_attempts,
            self._max_reconnect_attempts,
            total_delay,
            source_module=self.__class__.__name__,
        )

        await asyncio.sleep(total_delay)

        if self._is_running:  # Check again after delay
            await self._connect_websocket()

    def get_exchange_order_id(self, client_order_id: str) -> str | None:
        """Get the exchange order ID for a given client order ID.

        Args:
            client_order_id: Internal client order ID

        Returns:
            Exchange order ID if found, None otherwise
        """
        return self._internal_to_exchange_order_id.get(client_order_id)

    def get_client_order_id(self, exchange_order_id: str) -> str | None:
        """Get the client order ID for a given exchange order ID.

        Args:
            exchange_order_id: Exchange order ID

        Returns:
            Client order ID if found, None otherwise
        """
        return self._exchange_to_internal_order_id.get(exchange_order_id)

    def is_websocket_connected(self) -> bool:
        """Check if WebSocket is connected and authenticated.

        Returns:
            True if WebSocket is connected and ready for use
        """
        return (
            self._use_websocket_for_orders
            and self._websocket_connection_state == "AUTHENTICATED"
            and self.websocket_client is not None
        )

    def get_open_orders(self) -> list[dict[str, str]]:
        """Get list of currently tracked open orders.

        Returns:
            List of dictionaries with order_id and client_order_id
        """
        orders = []
        for client_id, exchange_id in self._order_map.items():
            orders.append(
                {
                    "order_id": exchange_id,
                    "client_order_id": client_id,
                },
            )
        return orders

    async def query_order_details(self, exchange_order_id: str) -> dict[str, Any] | None:
        """Query the exchange for order details.

        Args:
            exchange_order_id: The exchange order ID to query

        Returns:
            Order details dictionary if found, None otherwise
        """
        return await self._query_order_details(exchange_order_id)

    async def make_private_request_with_retry(self, uri_path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Make a private API request with retry logic.

        Args:
            uri_path: The API endpoint path
            params: Request parameters

        Returns:
            API response dictionary if successful, None otherwise
        """
        return await self._make_private_request_with_retry(uri_path, params)

    async def _publish_initial_execution_report(
        self,
        originating_event: TradeSignalApprovedEvent,
        client_order_id: str,
        exchange_order_id: str,
        order_type: str,
    ) -> None:
        """Publish initial NEW execution report for an order.

        Helper method to avoid code duplication when publishing initial reports
        for both regular and batch-placed orders.
        """
        report = ExecutionReportEvent(
            source_module=self.__class__.__name__,
            event_id=UUID(int=int(time.time() * 1000000)),
            timestamp=datetime.now(UTC),
            signal_id=originating_event.signal_id,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            trading_pair=originating_event.trading_pair,
            exchange=self.config.get("exchange.name", "kraken"),
            order_status="NEW",
            order_type=order_type,
            side=originating_event.side,
            quantity_ordered=originating_event.quantity,
            quantity_filled=Decimal(0),
            limit_price=originating_event.limit_price,
            average_fill_price=None,
            commission=None,
            commission_asset=None,
            timestamp_exchange=None,
            error_message=None,
        )

        # Using asyncio.create_task for fire-and-forget publishing
        task = asyncio.create_task(self.pubsub.publish(report))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        self.logger.debug(
            "Published NEW ExecutionReport for %s / %s",
            client_order_id,
            exchange_order_id,
            source_module=self.__class__.__name__,
        )
