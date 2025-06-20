"""Kraken exchange execution handler implementation.

This module provides a Kraken-specific implementation of the ExecutionHandlerInterface,
handling authentication, API communication, and order processing for the Kraken
cryptocurrency exchange.
"""

from datetime import UTC, datetime
from decimal import Decimal
import time
from typing import Any
import uuid

from gal_friday.config_manager import ConfigManager
from gal_friday.core.asset_registry import AssetSpecification, ExchangeSpecification

# Import events for HALT triggers
from gal_friday.core.events import PotentialHaltTriggerEvent
from gal_friday.core.pubsub import PubSubManager

# Import custom exceptions
from gal_friday.exceptions import (
    ExecutionHandlerAuthenticationError,
    ExecutionHandlerCriticalError,
    ExecutionHandlerNetworkError,
)

# Import the proper interface
from gal_friday.interfaces.execution_handler_interface import (
    ExecutionHandlerInterface,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    PositionInfo,
    TimeInForce,
)
from gal_friday.logger_service import LoggerService
from gal_friday.monitoring_service import MonitoringService
from gal_friday.utils.kraken_api import generate_kraken_signature


class KrakenExecutionError(ExecutionHandlerCriticalError):
    """Base exception for Kraken execution errors."""


class UnknownKrakenActionError(KrakenExecutionError):
    """Raised when an unknown Kraken action is encountered.

    Args:
        action: The unknown action that was encountered.
    """
    def __init__(self, action: str) -> None:
        """Initialize the UnknownKrakenActionError.

        Args:
            action: The unknown action that was encountered.
        """
        super().__init__(f"Unknown execution action: {action}")


class KrakenCredentialsMissingError(ExecutionHandlerAuthenticationError):
    """Raised when Kraken API credentials are not configured."""

    def __init__(self) -> None:
        """Initialize the KrakenCredentialsMissingError."""
        super().__init__("API key and secret are required for private Kraken API requests")


class KrakenApiSecretMissingError(ExecutionHandlerAuthenticationError):
    """Raised when Kraken API secret is not set."""

    def __init__(self) -> None:
        """Initialize the KrakenApiSecretMissingError."""
        super().__init__("API secret is not set.")


class KrakenExecutionHandler(ExecutionHandlerInterface):
    """Kraken-specific implementation of the ExecutionHandlerInterface.

    This class handles order execution on the Kraken cryptocurrency exchange by:
    1. Translating internal order formats to Kraken API parameters
    2. Handling Kraken-specific authentication and API endpoints
    3. Processing Kraken API responses into standardized formats
    4. Properly handling errors and triggering HALT conditions when appropriate
    """

    def __init__(
        self,
        exchange_spec: ExchangeSpecification,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        monitoring_service: MonitoringService,
        logger_service: LoggerService,
        **kwargs: Any,
    ) -> None:
        """Initialize the Kraken-specific execution handler.

        Args:
        ----
            exchange_spec: Exchange specification for Kraken
            config_manager: Configuration manager instance
            pubsub_manager: PubSub manager for event handling
            monitoring_service: Monitoring service for tracking orders and performance
            logger_service: Logger service for logging
            **kwargs: Additional keyword arguments
        """
        super().__init__(exchange_spec, **kwargs)

        self.config = config_manager
        self.pubsub = pubsub_manager
        self.monitoring = monitoring_service
        self.logger = logger_service

        # Initialize Kraken-specific attributes
        self._api_url = self.config.get("kraken.api_url", "https://api.kraken.com")
        self._api_key = self.config.get("kraken.api_key")
        self._api_secret = self.config.get("kraken.api_secret")
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        self._connected = False

        # Rate limiting and connection settings
        self._request_timeout = self.config.get_float("kraken.request_timeout_s", 10.0)
        self._max_retries = self.config.get_int("kraken.max_retries", 3)

        # Trading fee configuration
        self._default_maker_fee = self.config.get_decimal(
            "kraken.default_maker_fee_pct", Decimal("0.0016"))
        self._default_taker_fee = self.config.get_decimal(
            "kraken.default_taker_fee_pct", Decimal("0.0026"))

        # Validate API credentials
        if not self._api_key or not self._api_secret:
            self.logger.warning(
                "Kraken API credentials not configured. Live trading will not be available.",
                source_module=self.__class__.__name__)

        self.logger.info(
            "KrakenExecutionHandler initialized.",
            source_module=self.__class__.__name__)

    # Core trading operations - implementing ExecutionHandlerInterface

    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit an order to the Kraken exchange.

        Args:
            order_request: Universal order request structure
        Returns:
            Universal order response structure
        """
        try:
            self.logger.info(
                f"Submitting {order_request.side} {order_request.order_type.name} order for {order_request.symbol}",
                source_module=self.__class__.__name__)

            # Validate order request
            validation_errors = self.validate_order_request(order_request)
            if validation_errors:
                error_msg = f"Order validation failed: {', '.join(validation_errors)}"
                self.logger.error(error_msg, source_module=self.__class__.__name__)
                return OrderResponse(
                    success=False,
                    error_code="VALIDATION_ERROR",
                    error_message=error_msg)

            # Convert to Kraken format
            kraken_params = self._convert_order_request_to_kraken(order_request)

            # Submit to Kraken API
            uri_path = "/0/private/AddOrder"
            result = await self._make_private_request_with_retry(uri_path, kraken_params)

            return self._parse_add_order_response(result, order_request)

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"order_request": str(order_request)})
            return OrderResponse(
                success=False,
                error_code="INTERNAL_ERROR",
                error_message=f"Failed to submit order: {e!s}")

    async def cancel_order(
        self, exchange_order_id: str, symbol: str | None = None) -> OrderResponse:
        """Cancel an existing order.

        Args:
            exchange_order_id: Exchange-specific order identifier
            symbol: Trading symbol (optional for Kraken)

        Returns:
            Order response indicating cancellation status
        """
        try:
            self.logger.info(
                f"Cancelling order: {exchange_order_id}",
                source_module=self.__class__.__name__)

            uri_path = "/0/private/CancelOrder"
            params = {"txid": exchange_order_id}

            result = await self._make_private_request_with_retry(uri_path, params)

            if result.get("error"):
                return OrderResponse(
                    success=False,
                    exchange_order_id=exchange_order_id,
                    error_code="CANCEL_FAILED",
                    error_message=str(result["error"]))

            # Check cancellation count
            count = result.get("result", {}).get("count", 0)
            if count > 0:
                return OrderResponse(
                    success=True,
                    exchange_order_id=exchange_order_id,
                    status=OrderStatus.CANCELLED)
            return OrderResponse(
                success=False,
                exchange_order_id=exchange_order_id,
                error_code="NOT_CANCELLED",
                error_message="Order may already be in terminal state")

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"exchange_order_id": exchange_order_id})
            return OrderResponse(
                success=False,
                exchange_order_id=exchange_order_id,
                error_code="INTERNAL_ERROR",
                error_message=f"Failed to cancel order: {e!s}")

    async def modify_order(
        self, exchange_order_id: str, modifications: dict[str, Any]) -> OrderResponse:
        """Modify an existing order (if supported by exchange).

        Args:
            exchange_order_id: Exchange-specific order identifier
            modifications: Dictionary of fields to modify

        Returns:
            Order response with updated order details
        """
        try:
            self.logger.info(
                f"Modifying order {exchange_order_id}: {modifications}",
                source_module=self.__class__.__name__)

            # Build modification parameters
            kraken_params = {"txid": exchange_order_id}

            # Map common modification fields
            if "quantity" in modifications:
                kraken_params["volume"] = str(modifications["quantity"])
            if "price" in modifications:
                kraken_params["price"] = str(modifications["price"])
            if "stop_price" in modifications:
                kraken_params["price2"] = str(modifications["stop_price"])

            uri_path = "/0/private/EditOrder"
            result = await self._make_private_request_with_retry(uri_path, kraken_params)

            if result.get("error"):
                return OrderResponse(
                    success=False,
                    exchange_order_id=exchange_order_id,
                    error_code="MODIFY_FAILED",
                    error_message=str(result["error"]))

            return OrderResponse(
                success=True,
                exchange_order_id=exchange_order_id,
                status=OrderStatus.OPEN)

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"exchange_order_id": exchange_order_id})
            return OrderResponse(
                success=False,
                exchange_order_id=exchange_order_id,
                error_code="INTERNAL_ERROR",
                error_message=f"Failed to modify order: {e!s}")

    async def get_order_status(
        self, exchange_order_id: str) -> OrderResponse:
        """Get current status of an order.

        Args:
            exchange_order_id: Exchange-specific order identifier

        Returns:
            Current order status and details
        """
        try:
            uri_path = "/0/private/QueryOrders"
            params = {"txid": exchange_order_id}

            result = await self._make_private_request_with_retry(uri_path, params)

            if result.get("error"):
                return OrderResponse(
                    success=False,
                    exchange_order_id=exchange_order_id,
                    error_code="QUERY_FAILED",
                    error_message=str(result["error"]))

            order_data = result.get("result", {}).get(exchange_order_id)
            if not order_data:
                return OrderResponse(
                    success=False,
                    exchange_order_id=exchange_order_id,
                    error_code="ORDER_NOT_FOUND",
                    error_message="Order not found")

            return self._parse_order_data_to_response(order_data, exchange_order_id)

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"exchange_order_id": exchange_order_id})
            return OrderResponse(
                success=False,
                exchange_order_id=exchange_order_id,
                error_code="INTERNAL_ERROR",
                error_message=f"Failed to query order: {e!s}")

    # Portfolio and position management

    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get current account balances.

        Returns:
            Dictionary mapping currency/asset to available balance
        """
        try:
            uri_path = "/0/private/Balance"
            result = await self._make_private_request_with_retry(uri_path, {})

            if result.get("error"):
                self.logger.error(
                    f"Failed to get account balance: {result['error']}",
                    source_module=self.__class__.__name__)
                return {}

            balances = {}
            balance_data = result.get("result", {})
            for currency, balance_str in balance_data.items():
                try:
                    balances[currency] = Decimal(balance_str)
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Invalid balance value for {currency}: {balance_str}",
                        source_module=self.__class__.__name__)

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"operation": "get_account_balance"})
            self.logger.exception(
                f"Failed to get account balance: {e!s}",
                source_module=self.__class__.__name__)
            return {}
        else:
            return balances

    async def get_positions(
        self, symbol: str | None = None) -> list[PositionInfo]:
        """Get current positions.

        Args:
            symbol: Optional filter for specific symbol

        Returns:
            List of current positions
        """
        try:
            uri_path = "/0/private/OpenPositions"
            params = {}
            if symbol:
                params["pair"] = self.normalize_symbol_for_exchange(symbol)

            result = await self._make_private_request_with_retry(uri_path, params)

            if result.get("error"):
                self.logger.error(
                    f"Failed to get positions: {result['error']}",
                    source_module=self.__class__.__name__)
                return []

            positions = []
            position_data = result.get("result", {})

            for pos_id, pos_info in position_data.items():
                try:
                    position = self._parse_position_data(pos_id, pos_info)
                    if position:
                        positions.append(position)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse position {pos_id}: {e!s}",
                        source_module=self.__class__.__name__)

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"operation": "get_positions"})
            self.logger.exception(
                f"Failed to get positions: {e!s}",
                source_module=self.__class__.__name__)
            return []
        else:
            return positions

    # Market data and exchange info

    async def get_supported_assets(self) -> list[AssetSpecification]:
        """Get list[Any] of assets supported by this exchange.

        Returns:
            List of asset specifications
        """
        try:
            uri_path = "/0/public/AssetPairs"
            url = self._api_url + uri_path

            # Use public request method
            result = await self._make_public_request_with_retry(url)

            if not result or result.get("error"):
                self.logger.error(
                    f"Failed to get asset pairs: {result.get('error') if result else 'No response'}",
                    source_module=self.__class__.__name__)
                return []

            assets = []
            pair_data = result.get("result", {})

            for pair_name, pair_info in pair_data.items():
                try:
                    asset_spec = self._parse_asset_pair_to_spec(pair_name, pair_info)
                    if asset_spec:
                        assets.append(asset_spec)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse asset pair {pair_name}: {e!s}",
                        source_module=self.__class__.__name__)

        except Exception as e:
            await self._trigger_halt_if_needed(e, {"operation": "get_supported_assets"})
            self.logger.exception(
                f"Failed to get supported assets: {e!s}",
                source_module=self.__class__.__name__)
            return []
        else:
            return assets

    async def get_trading_fees(
        self, symbol: str) -> dict[str, Decimal]:
        """Get trading fees for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with maker/taker fees
        """
        try:
            uri_path = "/0/private/TradeVolume"
            params = {"pair": self.normalize_symbol_for_exchange(symbol)}

            result = await self._make_private_request_with_retry(uri_path, params)

            if result.get("error"):
                self.logger.warning(
                    f"Failed to get trading fees for {symbol}: {result['error']}",
                    source_module=self.__class__.__name__)
                # Return default fees
                return {
                    "maker_fee": self._default_maker_fee,
                    "taker_fee": self._default_taker_fee,
                }

            # Parse fee information from response
            fees_data = result.get("result", {}).get("fees", {})
            symbol_fees = fees_data.get(self.normalize_symbol_for_exchange(symbol), {})

            return {
                "maker_fee": Decimal(str(symbol_fees.get("fee", self._default_maker_fee))),
                "taker_fee": Decimal(str(symbol_fees.get("fee", self._default_taker_fee))),
            }

        except Exception as e:
            await self._trigger_halt_if_needed(
                e, {"operation": "get_trading_fees", "symbol": symbol})
            self.logger.exception(
                f"Failed to get trading fees for {symbol}: {e!s}",
                source_module=self.__class__.__name__)
            return {
                "maker_fee": self._default_maker_fee,
                "taker_fee": self._default_taker_fee,
            }

    # Connection and lifecycle management

    async def connect(self) -> bool:
        """Establish connection to exchange.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test connection with server time request
            uri_path = "/0/public/Time"
            url = self._api_url + uri_path

            result = await self._make_public_request_with_retry(url)

            if result and not result.get("error"):
                self._connected = True
                self.logger.info(
                    "Successfully connected to Kraken exchange",
                    source_module=self.__class__.__name__)
                return True
            else:
                self._connected = False
                self.logger.error(
                    f"Failed to connect to Kraken: {result.get('error') if result else 'No response'}",
                    source_module=self.__class__.__name__)
                return False

        except Exception as e:
            self._connected = False
            self.logger.exception(
                f"Failed to connect to Kraken: {e!s}",
                source_module=self.__class__.__name__)
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from exchange."""
        self._connected = False
        self.logger.info(
            "Disconnected from Kraken exchange",
            source_module=self.__class__.__name__)

    async def is_connected(self) -> bool:
        """Check if connected to exchange.

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    # Helper methods for data conversion and parsing

    def normalize_symbol_for_exchange(
        self, universal_symbol: str) -> str:
        """Convert universal symbol to exchange-specific format.

        Args:
            universal_symbol: Universal symbol format (e.g., "XRP/USD")

        Returns:
            Exchange-specific symbol format
        """
        # Kraken typically removes the slash and may have specific mappings
        kraken_symbol = universal_symbol.replace("/", "")

        # Handle specific Kraken symbol mappings
        symbol_mappings = {
            "BTCUSD": "XBTUSD",
            "XRPUSD": "XXRPZUSD",
            # Add more mappings as needed
        }

        return symbol_mappings.get(kraken_symbol, kraken_symbol)

    def _convert_order_request_to_kraken(self, order_request: OrderRequest) -> dict[str, Any]:
        """Convert universal order request to Kraken API parameters."""
        kraken_params = {
            "nonce": int(time.time() * 1000),
            "pair": self.normalize_symbol_for_exchange(order_request.symbol),
            "type": order_request.side.lower(),
            "volume": str(order_request.quantity),
        }

        # Map order type
        if order_request.order_type == OrderType.MARKET:
            kraken_params["ordertype"] = "market"
        elif order_request.order_type == OrderType.LIMIT:
            kraken_params["ordertype"] = "limit"
            if order_request.limit_price:
                kraken_params["price"] = str(order_request.limit_price)
        elif order_request.order_type == OrderType.STOP:
            kraken_params["ordertype"] = "stop-loss"
            if order_request.stop_price:
                kraken_params["price"] = str(order_request.stop_price)
        elif order_request.order_type == OrderType.STOP_LIMIT:
            kraken_params["ordertype"] = "stop-loss-limit"
            if order_request.stop_price:
                kraken_params["price"] = str(order_request.stop_price)
            if order_request.limit_price:
                kraken_params["price2"] = str(order_request.limit_price)

        # Map time in force
        if order_request.time_in_force == TimeInForce.IOC:
            kraken_params["oflags"] = "ioc"
        elif order_request.time_in_force == TimeInForce.FOK:
            kraken_params["oflags"] = "fok"

        # Add client order ID if provided
        if order_request.client_order_id:
            kraken_params["cl_ord_id"] = order_request.client_order_id

        return kraken_params

    def _parse_add_order_response(
        self, result: dict[str, Any], order_request: OrderRequest) -> OrderResponse:
        """Parse Kraken AddOrder response into OrderResponse."""
        if result.get("error"):
            return OrderResponse(
                success=False,
                client_order_id=order_request.client_order_id,
                error_code="KRAKEN_ERROR",
                error_message=str(result["error"]))

        order_result = result.get("result", {})
        txids = order_result.get("txid", [])

        if not txids:
            return OrderResponse(
                success=False,
                client_order_id=order_request.client_order_id,
                error_code="NO_ORDER_ID",
                error_message="No order ID returned from Kraken")

        return OrderResponse(
            success=True,
            exchange_order_id=txids[0],
            client_order_id=order_request.client_order_id,
            status=OrderStatus.PENDING,
            created_at=datetime.now(UTC))

    def _parse_order_data_to_response(
        self, order_data: dict[str, Any], exchange_order_id: str) -> OrderResponse:
        """Parse Kraken order data into OrderResponse."""
        try:
            # Map Kraken status to our status enum
            kraken_status = order_data.get("status", "unknown")
            status_mapping = {
                "pending": OrderStatus.PENDING,
                "open": OrderStatus.OPEN,
                "closed": OrderStatus.FILLED,
                "canceled": OrderStatus.CANCELLED,
                "expired": OrderStatus.EXPIRED,
            }
            status = status_mapping.get(kraken_status, OrderStatus.PENDING)

            # Extract order details
            vol_exec = order_data.get("vol_exec", "0")
            vol = order_data.get("vol", "0")
            price = order_data.get("price")
            fee = order_data.get("fee", "0")

            return OrderResponse(
                success=True,
                exchange_order_id=exchange_order_id,
                status=status,
                filled_quantity=Decimal(vol_exec),
                remaining_quantity=Decimal(vol) - Decimal(vol_exec),
                average_fill_price=Decimal(price) if price else None,
                commission=Decimal(fee),
                raw_response=order_data)

        except Exception as e:
            return OrderResponse(
                success=False,
                exchange_order_id=exchange_order_id,
                error_code="PARSE_ERROR",
                error_message=f"Failed to parse order data: {e!s}")

    def _parse_position_data(self, pos_id: str, pos_info: dict[str, Any]) -> PositionInfo | None:
        """Parse Kraken position data into PositionInfo."""
        try:
            # Extract position details
            pair = pos_info.get("pair", "")
            side = "LONG" if float(pos_info.get("vol", 0)) > 0 else "SHORT"
            quantity = abs(Decimal(pos_info.get("vol", "0")))
            avg_price = (
                Decimal(pos_info.get("cost", "0")) / quantity
                if quantity > 0
                else Decimal(0)
            )

            return PositionInfo(
                symbol=pair,
                exchange_id=self.exchange_id,
                side=side,
                quantity=quantity,
                average_entry_price=avg_price,
                unrealized_pnl=Decimal(pos_info.get("net", "0")),
                raw_data=pos_info)

        except Exception as e:
            self.logger.warning(
                f"Failed to parse position data for {pos_id}: {e!s}",
                source_module=self.__class__.__name__)
            return None

    def _parse_asset_pair_to_spec(
        self, pair_name: str, pair_info: dict[str, Any]) -> AssetSpecification | None:
        """Parse Kraken asset pair into AssetSpecification."""
        try:
            from gal_friday.core.asset_registry import AssetType

            return AssetSpecification(
                symbol=pair_info.get("altname", pair_name),
                asset_type=AssetType.CRYPTO,  # Kraken is primarily crypto
                base_asset=pair_info.get("base", ""),
                quote_asset=pair_info.get("quote", ""),
                min_order_size=Decimal(pair_info.get("ordermin", "0")),
                tick_size=Decimal(pair_info.get("tick_size", "0.01")),
                lot_size=Decimal(1),  # Default for Kraken
                exchange_symbol=pair_info.get("altname", pair_name),
                exchange_metadata=pair_info)

        except Exception as e:
            self.logger.warning(
                f"Failed to parse asset pair {pair_name}: {e!s}",
                source_module=self.__class__.__name__)
            return None

    async def _trigger_halt_if_needed(
        self, error: Exception, context: dict[str, Any]) -> None:
        """Trigger system HALT if error conditions warrant it.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        self._consecutive_errors += 1

        # Trigger HALT for critical errors or too many consecutive errors
        should_halt = (
            isinstance(
                error,
                ExecutionHandlerCriticalError | ExecutionHandlerAuthenticationError) or
            self._consecutive_errors >= self._max_consecutive_errors
        )

        if should_halt:
            halt_event = PotentialHaltTriggerEvent(
                source_module=self.__class__.__name__,
                event_id=uuid.uuid4(),
                timestamp=datetime.now(UTC),
                reason=f"Critical execution handler error: {error!s}")
            await self.pubsub.publish(halt_event)
            self.logger.critical(
                "Triggered potential system HALT due to execution handler errors",
                source_module=self.__class__.__name__,
                context={
                    "error_type": type(error).__name__,
                    "consecutive_errors": self._consecutive_errors,
                    "exchange": "kraken",
                    **context,
                })

    async def _make_private_request_with_retry(
        self, uri_path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make authenticated request to Kraken with retry logic."""
        from typing import cast

        import aiohttp
        import asyncio

        if not self._api_key or not self._api_secret:
            raise KrakenCredentialsMissingError

        for attempt in range(self._max_retries + 1):
            try:
                # Generate authentication headers
                headers = self._generate_auth_headers(uri_path, params)
                url = self._api_url + uri_path

                async with (
                    aiohttp.ClientSession() as session,
                    session.post(
                        url,
                        headers=headers,
                        data=params,
                        timeout=aiohttp.ClientTimeout(total=self._request_timeout)) as response):
                    response.raise_for_status()
                    result = cast("dict[str, Any]", await response.json())

                    # Reset error counter on success
                    if not result.get("error"):
                        self._consecutive_errors = 0

                    return result

            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Request failed (attempt {attempt + 1}/"
                        f"{self._max_retries + 1}), retrying in {wait_time}s: {e!s}",
                        source_module=self.__class__.__name__)
                    await asyncio.sleep(wait_time)
                    continue

                raise ExecutionHandlerNetworkError(
                    f"Request failed after {self._max_retries + 1} attempts: {e!s}") from e

        # Should not reach here, but just in case
        return {"error": ["Request failed"]}

    async def _make_public_request_with_retry(self, url: str) -> dict[str, Any] | None:
        """Make public request to Kraken with retry logic."""
        from typing import cast

        import aiohttp
        import asyncio

        for attempt in range(self._max_retries + 1):
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self._request_timeout)) as response):
                    response.raise_for_status()
                    return cast("dict[str, Any]", await response.json())

            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Public request failed (attempt {attempt + 1}/"
                        f"{self._max_retries + 1}), retrying in {wait_time}s: {e!s}",
                        source_module=self.__class__.__name__)
                    await asyncio.sleep(wait_time)
                    continue

                self.logger.exception(
                    f"Public request failed after {self._max_retries + 1} attempts: {e!s}",
                    source_module=self.__class__.__name__)
                return None

        return None

    def _generate_auth_headers(
        self, uri_path: str, request_data: dict[str, Any]) -> dict[str, str]:
        """Generate Kraken-specific authentication headers."""
        if not self._api_key or not self._api_secret:
            raise KrakenCredentialsMissingError

        # Generate signature
        nonce = int(request_data["nonce"])
        signature = generate_kraken_signature(uri_path, request_data, nonce, self._api_secret)

        return {
            "API-Key": self._api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }

    # ... (continue with remaining methods from original implementation)
    # Additional helper methods would be implemented here
