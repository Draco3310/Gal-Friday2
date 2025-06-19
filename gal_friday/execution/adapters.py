"""Execution adapters for different exchanges.

This module implements the adapter pattern for abstracting exchange-specific
execution logic, providing a clean interface for order management across
different trading venues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal
import time
from typing import TYPE_CHECKING, Any

import aiohttp
import asyncio

from gal_friday.exceptions import ExecutionError as ExchangeError
from gal_friday.utils.kraken_api import generate_kraken_signature

if TYPE_CHECKING:
    from gal_friday.config_manager import ConfigManager
    from gal_friday.logger_service import LoggerService


@dataclass
class OrderRequest:
    """Standardized order request across all exchanges."""
    trading_pair: str
    side: str  # "BUY" or "SELL"
    order_type: str  # "MARKET", "LIMIT", etc.
    quantity: Decimal
    price: Decimal | None = None
    client_order_id: str | None = None
    time_in_force: str | None = None
    stop_price: Decimal | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class OrderResponse:
    """Standardized order response from exchange."""
    success: bool
    exchange_order_ids: list[str]
    client_order_id: str | None
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None

@dataclass
class BatchOrderRequest:
    """Request for placing multiple orders simultaneously."""
    orders: list[OrderRequest]
    validate_only: bool = False

@dataclass
class BatchOrderResponse:
    """Response from batch order placement."""
    success: bool
    order_results: list[OrderResponse]
    error_message: str | None = None

class ExecutionAdapter(ABC):
    """Abstract base class for exchange execution adapters.

    Defines the interface that all exchange adapters must implement,
    abstracting away exchange-specific details for order management.
    """

    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerService) -> None:
        """Initialize the execution adapter.

        Args:
            config: Configuration manager instance
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._session: aiohttp.ClientSession | None = None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the adapter (load exchange info, setup session, etc.)."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources (close sessions, etc.)."""
        ...

    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a single order on the exchange.

        Args:
            order_request: Standardized order request

        Returns:
            Standardized order response
        """
        ...

    @abstractmethod
    async def place_batch_orders(self, orders: list[OrderRequest]) -> list[dict[str, Any]]:
        """Place multiple orders simultaneously (if supported).

        Args:
            orders: Orders to submit

        Returns:
            List of order response dictionaries
        """
        ...

    @abstractmethod
    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an order on the exchange.

        Args:
            exchange_order_id: Exchange-specific order ID

        Returns:
            True if cancellation was successful
        """
        ...

    @abstractmethod
    async def get_order_status(self, exchange_order_id: str) -> dict[str, Any] | None:
        """Get the current status of an order.

        Args:
            exchange_order_id: Exchange-specific order ID

        Returns:
            Order status information or None if not found
        """
        ...

    @abstractmethod
    async def get_account_balances(self) -> dict[str, Decimal]:
        """Get account balances from the exchange.

        Returns:
            Dictionary of currency to available balance
        """
        ...

    @abstractmethod
    async def get_open_positions(self) -> dict[str, Any]:
        """Get open positions from the exchange.

        Returns:
            Dictionary of position information
        """
        ...

    @abstractmethod
    def get_exchange_name(self) -> str:
        """Get the name identifier for this exchange."""
        ...

class KrakenExecutionAdapter(ExecutionAdapter):
    """Kraken-specific execution adapter implementation."""

    def __init__(
        self,
        config: ConfigManager,
        logger: LoggerService) -> None:
        """Initialize the Kraken adapter."""
        super().__init__(config, logger)

        self.api_key = self.config.get("kraken.api_key", default=None)
        self.api_secret = self.config.get("kraken.secret_key", default=None)
        self.api_base_url = self.config.get("exchange.api_url", "https://api.kraken.com")

        # Exchange info storage
        self._pair_info: dict[str, dict[str, Any]] = {}

        # External Kraken API wrapper (if provided)
        self.kraken_api: Any | None = None

        # Rate limiting
        self._last_api_call_time = 0.0
        self._api_call_delay = 1.0  # Minimum delay between API calls

        # Optional enhanced components
        self._error_classifier: Any | None = None
        self._batch_processor: Any | None = None

        # Initialize enhanced components
        try:
            from gal_friday.execution_handler_enhancements import (
                KrakenErrorClassifier,
                OptimizedBatchProcessor,
            )
            self._error_classifier = KrakenErrorClassifier(logger)
            self._batch_processor = OptimizedBatchProcessor(self, logger, config)
        except ImportError:
            self.logger.warning(
                "Enhanced execution components not available, using basic functionality",
                source_module=self.__class__.__name__,
            )
            self._error_classifier = None
            self._batch_processor = None

    async def initialize(self) -> None:
        """Initialize the Kraken adapter."""
        if not self.api_key or not self.api_secret:
            raise ValueError("Kraken API credentials not configured")

        self._session = aiohttp.ClientSession()
        await self._load_exchange_info()

        self.logger.info(
            "Kraken execution adapter initialized",
            source_module=self.__class__.__name__)

    async def cleanup(self) -> None:
        """Clean up Kraken adapter resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info(
                "Kraken adapter session closed",
                source_module=self.__class__.__name__)

    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place a single order on Kraken."""
        kraken_params = self._translate_order_request_to_kraken(order_request)
        if not kraken_params:
            return OrderResponse(
                success=False,
                exchange_order_ids=[],
                client_order_id=order_request.client_order_id,
                error_message="Failed to translate order request to Kraken format")

        try:
            result = await self._make_private_request("/0/private/AddOrder", kraken_params)
            return self._parse_add_order_response(result, order_request.client_order_id)
        except Exception as e:
            self.logger.exception(
                "Error placing order on Kraken: %s",
                str(e),
                source_module=self.__class__.__name__)
            return OrderResponse(
                success=False,
                exchange_order_ids=[],
                client_order_id=order_request.client_order_id,
                error_message=f"Exception during order placement: {e!s}")

    async def place_batch_orders(self, orders: list[OrderRequest]) -> list[dict[str, Any]]:
        """Place multiple orders on Kraken using AddOrderBatch."""
        responses: list[dict[str, Any]] = [
            {
                "success": False,
                "exchange_order_ids": [],
                "client_order_id": str(order.client_order_id),
                "error_message": "Failed to translate order request to Kraken format",
                "raw_response": None,
            }
            for order in orders
        ]

        payloads: list[dict[str, Any]] = []
        pending_indices: list[int] = []

        for idx, order in enumerate(orders):
            params = self._create_order_payload(order)
            if params:
                payloads.append(params)
                pending_indices.append(idx)

        if not payloads:
            return responses

        try:
            if self.kraken_api is None:
                raise ExchangeError("Kraken API not initialized")
            batch_results = await self.kraken_api.add_order_batch(payloads)
        except Exception as e:
            self.logger.exception(
                "Error placing batch orders on Kraken: %s",
                str(e),
                source_module=self.__class__.__name__)
            raise ExchangeError(f"add_order_batch failed: {e!s}") from e

        if not isinstance(batch_results, list):
            raise ExchangeError("Unexpected response format from add_order_batch")

        for order_idx, result in zip(pending_indices, batch_results, strict=False):
            parsed = self._parse_add_order_response(
                result,
                str(orders[order_idx].client_order_id))
            responses[order_idx] = asdict(parsed)

        return responses

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an order on Kraken."""
        try:
            params = {"txid": exchange_order_id}
            result = await self._make_private_request("/0/private/CancelOrder", params)

            if result.get("error"):
                self.logger.error(
                    "Kraken cancel order error: %s",
                    result["error"],
                    source_module=self.__class__.__name__)
                return False

            count = result.get("result", {}).get("count", 0)
            return bool(count > 0)

        except Exception as e:
            self.logger.exception(
                "Exception cancelling order %s: %s",
                exchange_order_id,
                str(e),
                source_module=self.__class__.__name__)
            return False

    async def get_order_status(self, exchange_order_id: str) -> dict[str, Any] | None:
        """Get order status from Kraken."""
        try:
            params = {"txid": exchange_order_id, "trades": "true"}
            result = await self._make_private_request("/0/private/QueryOrders", params)

            if result.get("error"):
                self.logger.error(
                    "Kraken query order error: %s",
                    result["error"],
                    source_module=self.__class__.__name__)
                return None

            orders = result.get("result", {})
            order_data = orders.get(exchange_order_id)
            return order_data if order_data is not None else None

        except Exception as e:
            self.logger.exception(
                "Exception querying order %s: %s",
                exchange_order_id,
                str(e),
                source_module=self.__class__.__name__)
            return None

    async def get_account_balances(self) -> dict[str, Decimal]:
        """Get account balances from Kraken."""
        try:
            result = await self._make_private_request("/0/private/Balance", {})

            if result.get("error"):
                self.logger.error(
                    "Kraken balance query error: %s",
                    result["error"],
                    source_module=self.__class__.__name__)
                return {}

            balances = result.get("result", {})
            return {
                currency: Decimal(str(balance))
                for currency, balance in balances.items()
                if Decimal(str(balance)) > 0
            }

        except Exception as e:
            self.logger.exception(
                "Exception getting account balances: %s",
                str(e),
                source_module=self.__class__.__name__)
            return {}

    async def get_open_positions(self) -> dict[str, Any]:
        """Get open positions from Kraken."""
        try:
            result = await self._make_private_request("/0/private/OpenPositions", {})

            if result.get("error"):
                self.logger.error(
                    "Kraken open positions query error: %s",
                    result["error"],
                    source_module=self.__class__.__name__)
                return {}

            positions = result.get("result", {})
            return dict(positions)

        except Exception as e:
            self.logger.exception(
                "Exception getting open positions: %s",
                str(e),
                source_module=self.__class__.__name__)
            return {}

    def get_exchange_name(self) -> str:
        """Get the exchange name."""
        return "kraken"

    # Private helper methods

    async def _load_exchange_info(self) -> None:
        """Load exchange pair information from Kraken."""
        try:
            url = f"{self.api_base_url}/0/public/AssetPairs"

            if self._session is None:
                raise ExchangeError("Session not initialized")
            async with self._session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

            if data.get("error"):
                self.logger.error(
                    "Error loading Kraken asset pairs: %s",
                    data["error"],
                    source_module=self.__class__.__name__)
                return

            result = data.get("result", {})

            # Process and store pair info
            internal_pairs = self.config.get_list("trading.pairs", [])
            kraken_pair_map = {v.get("altname", k): k for k, v in result.items()}

            for internal_pair in internal_pairs:
                kraken_altname = internal_pair.replace("/", "")
                kraken_key = kraken_pair_map.get(kraken_altname)

                if kraken_key and kraken_key in result:
                    pair_data = result[kraken_key]
                    self._pair_info[internal_pair] = {
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

            self.logger.info(
                "Loaded info for %d trading pairs",
                len(self._pair_info),
                source_module=self.__class__.__name__)

        except Exception as e:
            self.logger.exception(
                "Exception loading exchange info: %s",
                str(e),
                source_module=self.__class__.__name__)

    def _translate_order_request_to_kraken(self, order_request: OrderRequest) -> dict[str, Any] | None:
        """Translate standardized order request to Kraken format."""
        pair_info = self._pair_info.get(order_request.trading_pair)
        if not pair_info:
            self.logger.error(
                "No pair info for %s",
                order_request.trading_pair,
                source_module=self.__class__.__name__)
            return None

        kraken_pair = pair_info.get("altname")
        if not kraken_pair:
            self.logger.error(
                "No Kraken pair name for %s",
                order_request.trading_pair,
                source_module=self.__class__.__name__)
            return None

        params = {
            "pair": kraken_pair,
            "type": order_request.side.lower(),
            "ordertype": order_request.order_type.lower(),
            "volume": self._format_decimal(order_request.quantity, pair_info.get("lot_decimals", 8)),
        }

        if order_request.client_order_id:
            params["cl_ord_id"] = order_request.client_order_id

        if order_request.price and order_request.order_type.upper() == "LIMIT":
            params["price"] = self._format_decimal(order_request.price, pair_info.get("pair_decimals", 4))

        if order_request.stop_price:
            params["price"] = self._format_decimal(order_request.stop_price, pair_info.get("pair_decimals", 4))

        if order_request.time_in_force:
            params["timeinforce"] = order_request.time_in_force

        return params

    def _parse_add_order_response(self, result: dict[str, Any], client_order_id: str | None) -> OrderResponse:
        """Parse Kraken AddOrder response."""
        if result.get("error"):
            return OrderResponse(
                success=False,
                exchange_order_ids=[],
                client_order_id=client_order_id,
                error_message=str(result["error"]),
                raw_response=result)

        kraken_result = result.get("result", {})
        txids = kraken_result.get("txid", [])

        if not txids:
            return OrderResponse(
                success=False,
                exchange_order_ids=[],
                client_order_id=client_order_id,
                error_message="No transaction IDs returned",
                raw_response=result)

        return OrderResponse(
            success=True,
            exchange_order_ids=txids,
            client_order_id=client_order_id,
            raw_response=result)

    async def _make_private_request(self, uri_path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated request to Kraken API."""
        if not self._session:
            raise RuntimeError("Session not initialized")

        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call_time
        if time_since_last_call < self._api_call_delay:
            await asyncio.sleep(self._api_call_delay - time_since_last_call)

        # Generate nonce and signature
        nonce = int(time.time() * 1000)
        request_data = data.copy()
        request_data["nonce"] = nonce

        api_sign = generate_kraken_signature(uri_path, request_data, nonce, self.api_secret)

        headers = {
            "API-Key": self.api_key,
            "API-Sign": api_sign,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }

        url = self.api_base_url + uri_path

        try:
            async with self._session.post(url, headers=headers, data=request_data) as response:
                response.raise_for_status()
                result = await response.json()
                self._last_api_call_time = time.time()
                return dict(result)

        except Exception as e:
            self.logger.exception(
                "Error in private API request to %s: %s",
                uri_path,
                str(e),
                source_module=self.__class__.__name__)
            return {"error": [f"Request failed: {e!s}"]}

    def _format_decimal(self, value: Decimal, precision: int) -> str:
        """Format decimal value to string with specified precision."""
        quantizer = Decimal("1e-" + str(precision))
        return str(value.quantize(quantizer))

    def _create_order_payload(self, order: OrderRequest) -> dict[str, Any] | None:
        """Create Kraken API payload from an OrderRequest."""
        return self._translate_order_request_to_kraken(order)

    def _are_all_contingent_orders(self, orders: list[OrderRequest]) -> bool:
        """Check if all orders are contingent (SL/TP) orders."""
        contingent_types = {"stop-loss", "take-profit", "stop-loss-limit", "take-profit-limit"}
        return all(order.order_type.lower() in contingent_types for order in orders)

    async def _place_contingent_batch(self, batch_request: BatchOrderRequest) -> BatchOrderResponse:
        """Place contingent orders using batch placement logic."""
        # For SL/TP orders, we can potentially use better logic
        return await self._place_orders_individually(batch_request)

    async def _place_orders_individually(self, batch_request: BatchOrderRequest) -> BatchOrderResponse:
        """Place orders individually when batch placement is not available."""
        # Try to use OptimizedBatchProcessor if available
        if hasattr(self, "_batch_processor") and self._batch_processor is not None:
            from gal_friday.execution_handler_enhancements import BatchStrategy

            # Use smart routing strategy for optimal performance
            batch_result = await self._batch_processor.process_batch_orders(
                batch_request.orders,
                BatchStrategy.SMART_ROUTING,
            )

            # Convert to BatchOrderResponse format
            order_results = []
            if hasattr(batch_result, "successful_orders") and hasattr(batch_result, "failed_orders"):
                for order_result in batch_result.successful_orders + batch_result.failed_orders:
                    order_results.append(OrderResponse(
                        success=order_result.get("success", False),
                        exchange_order_ids=order_result.get("exchange_order_ids", []),
                        client_order_id=order_result.get("client_order_id"),
                        error_message=order_result.get("error"),
                    ))

                success_rate = getattr(batch_result, "success_rate", 0.0)
                return BatchOrderResponse(
                    success=success_rate >= 0.5,  # Consider batch successful if at least 50% succeed
                    order_results=order_results,
                    error_message=(
                        f"Batch execution completed with {success_rate:.1%} success rate"
                        if success_rate < 1.0 else None
                    ),
                )

        # Fallback to original individual placement
        results = []
        overall_success = True

        for order in batch_request.orders:
            result = await self.place_order(order)
            results.append(result)
            if not result.success:
                overall_success = False

        return BatchOrderResponse(
            success=overall_success,
            order_results=results,
            error_message="Some orders failed" if not overall_success else None)
