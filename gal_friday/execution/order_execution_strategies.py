"""Order execution strategies for different execution methods."""

from abc import ABC, abstractmethod
import time
from typing import TYPE_CHECKING, Any, cast

import asyncio

if TYPE_CHECKING:
    from gal_friday.execution_handler import AsyncOrderRequest


class OrderExecutionStrategy(ABC):
    """Abstract base class for order execution strategies."""

    def __init__(self, logger: Any = None) -> None:
        """Initialize with optional logger."""
        self.logger = logger

    @abstractmethod
    async def execute(self, request: "AsyncOrderRequest", execution_start: float) -> dict[str, Any]:
        """Execute the order request and return result."""

    def _add_execution_metadata(
        self,
        result: dict[str, Any],
        request: "AsyncOrderRequest",
        execution_start: float,
    ) -> dict[str, Any]:
        """Add common execution metadata to result."""
        result.update(
            {
                "async_execution": True,
                "execution_time_ms": (time.time() - execution_start) * 1000,
                "priority": request.priority.value,
                "retry_count": request.retry_count,
                "worker_processed": True,
            },
        )
        return result


class ExchangeAdapterStrategy(OrderExecutionStrategy):
    """Execute orders using the exchange adapter's place_order method."""

    def __init__(self, exchange_adapter: Any, logger: Any = None) -> None:
        """Initialize with exchange adapter."""
        super().__init__(logger)
        self.exchange_adapter = exchange_adapter

    async def execute(self, request: "AsyncOrderRequest", execution_start: float) -> dict[str, Any]:
        """Execute order using exchange adapter."""
        result = cast("dict[str, Any]", await self.exchange_adapter.place_order(request.order_request))
        result = self._add_execution_metadata(result, request, execution_start)

        if self.logger:
            self.logger.info(
                f"Async order executed via {self.exchange_adapter.__class__.__name__}: "
                f"{result.get('order_id', 'unknown')} in {result['execution_time_ms']:.1f}ms",
            )

        return result


class DirectAPIStrategy(OrderExecutionStrategy):
    """Execute orders using direct API calls."""

    def __init__(self, exchange_adapter: Any, logger: Any = None) -> None:
        """Initialize with exchange adapter for API access."""
        super().__init__(logger)
        self.exchange_adapter = exchange_adapter

    async def execute(self, request: "AsyncOrderRequest", execution_start: float) -> dict[str, Any]:
        """Execute order using direct API call."""
        order_params = request.order_request
        api_result = await self.exchange_adapter.make_private_request_with_retry(
            "/0/private/AddOrder",
            order_params,
        )

        # Process API response
        if api_result and not api_result.get("error"):
            kraken_result = api_result.get("result", {})
            txids = kraken_result.get("txid", [])
            order_id = self._extract_order_id(txids)

            result = {
                "success": True,
                "order_id": order_id,
                "exchange_response": api_result,
            }
        else:
            error_msg = api_result.get("error", ["Unknown error"]) if api_result else ["API request failed"]
            result = {
                "success": False,
                "error": error_msg,
            }

        return self._add_execution_metadata(result, request, execution_start)

    def _extract_order_id(self, txids: Any) -> str:
        """Extract order ID from transaction IDs."""
        if txids and isinstance(txids, list):
            return str(txids[0])
        if txids:
            return str(txids)
        return f"async_order_{int(time.time() * 1000000)}"


class SimulationStrategy(OrderExecutionStrategy):
    """Execute orders in simulation mode for testing."""

    async def execute(self, request: "AsyncOrderRequest", execution_start: float) -> dict[str, Any]:
        """Execute simulated order."""
        # Simulate network latency
        await asyncio.sleep(0.1)

        order_id = f"sim_async_{int(time.time() * 1000000)}"
        result = {
            "success": True,
            "order_id": order_id,
            "simulated": True,
            "order_params": request.order_request,
        }

        result = self._add_execution_metadata(result, request, execution_start)

        if self.logger:
            self.logger.debug(
                f"Simulated async order execution: {order_id} (no exchange adapter available)",
            )

        return result


class OrderExecutionContext:
    """Context for selecting and executing appropriate strategy."""

    def __init__(self, exchange_adapter: Any = None, logger: Any = None) -> None:
        """Initialize with optional exchange adapter and logger."""
        self.exchange_adapter = exchange_adapter
        self.logger = logger
        self._strategy = self._select_strategy()

    def _select_strategy(self) -> OrderExecutionStrategy:
        """Select appropriate execution strategy based on available components."""
        if self.exchange_adapter and hasattr(self.exchange_adapter, "place_order"):
            return ExchangeAdapterStrategy(self.exchange_adapter, self.logger)
        if self.exchange_adapter and hasattr(self.exchange_adapter, "make_private_request_with_retry"):
            return DirectAPIStrategy(self.exchange_adapter, self.logger)
        return SimulationStrategy(self.logger)

    async def execute_order(self, request: "AsyncOrderRequest", execution_start: float) -> dict[str, Any]:
        """Execute order using selected strategy."""
        return await self._strategy.execute(request, execution_start)
