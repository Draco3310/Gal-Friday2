"""Kraken exchange execution handler implementation.

This module provides a Kraken-specific implementation of the ExecutionHandler interface,
handling authentication, API communication, and order processing for the Kraken
cryptocurrency exchange.
"""

import base64
import hashlib
import hmac
import time
from typing import Any  # Removed Optional, Tuple, Dict
import urllib.parse

from ..config_manager import ConfigManager
from ..core.pubsub import PubSubManager
from ..execution_handler import ExecutionHandler
from ..logger_service import LoggerService
from ..monitoring_service import MonitoringService


class KrakenExecutionError(Exception):
    """Base exception for Kraken execution errors."""

class UnknownKrakenActionError(KrakenExecutionError):
    """Raised when an unknown Kraken action is encountered."""

    def __init__(self, action: str) -> None:
        super().__init__(f"Unknown execution action: {action}")

class KrakenCredentialsMissingError(KrakenExecutionError):
    """Raised when Kraken API credentials are not configured."""

    def __init__(self) -> None:
        super().__init__("API key and secret are required for private Kraken API requests")

class KrakenApiSecretMissingError(KrakenExecutionError):
    """Raised when Kraken API secret is not set."""

    def __init__(self) -> None:
        super().__init__("API secret is not set.")


class KrakenExecutionHandler(ExecutionHandler):
    """
    Kraken-specific implementation of the ExecutionHandler.

    This class handles order execution on the Kraken cryptocurrency exchange by:
    1. Translating internal order formats to Kraken API parameters
    2. Handling Kraken-specific authentication and API endpoints
    3. Processing Kraken API responses into standardized formats

    The implementation follows a clean separation of concerns pattern:
    - Market price functionality has been moved to a dedicated KrakenMarketPriceService
    - All Kraken-specific logic is contained within this class
    - Base class functionality is extended/overridden as needed

    Usage:
        handler = KrakenExecutionHandler(
            config_manager=config_manager,
            pubsub_manager=pubsub_manager,
            monitoring_service=monitoring_service,
            logger_service=logger_service
        )

        # Place an order
        result = await handler.place_order({
            "trading_pair": "XBT/USD",
            "order_type": "buy",
            "order_subtype": "limit",
            "quantity": "0.001",
            "price": "50000.0"
        })
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        monitoring_service: MonitoringService,  # Now required
        logger_service: LoggerService,
    ) -> None:
        """
        Initialize the Kraken-specific execution handler.

        Args
        ----
            config_manager: Configuration manager instance
            pubsub_manager: PubSub manager for event handling
            monitoring_service: Monitoring service for tracking orders and performance
            logger_service: Logger service for logging
        """
        # Pass all required dependencies to the base class constructor
        super().__init__(
            config_manager=config_manager,
            pubsub_manager=pubsub_manager,
            monitoring_service=monitoring_service,
            logger_service=logger_service,
        )

        # Initialize Kraken-specific attributes
        self._api_url = self.config.get("kraken.api_url", "https://api.kraken.com")
        self._api_key = self.config.get("kraken.api_key")
        self._api_secret = self.config.get("kraken.api_secret")

        # Validate API credentials
        if not self._api_key or not self._api_secret:
            self.logger.warning(
                "Kraken API credentials not configured. Live trading will not be available."
            )

        self.logger.info(
            "KrakenExecutionHandler initialized."
        )

    def _get_api_endpoint(self, action: str) -> str:
        """
        Get the Kraken API endpoint path for a given action.

        Args
        ----
            action: The action to perform (add_order, cancel_order, etc.)

        Returns
        -------
            The API endpoint path

        Raises
        ------
            ValueError: If the action is unknown
        """
        endpoints = {
            "add_order": "/0/private/AddOrder",
            "cancel_order": "/0/private/CancelOrder",
            "query_orders": "/0/private/QueryOrders",
            "open_orders": "/0/private/OpenOrders",
            "closed_orders": "/0/private/ClosedOrders",
        }

        path = endpoints.get(action)
        if not path:
            raise UnknownKrakenActionError(action)
        return path

    def _prepare_add_order_params(self, internal_data: dict[str, Any]) -> dict[str, Any]:
        """Prepare parameters for the 'add_order' action."""
        kraken_params = {}
        # Map internal trading pair to Kraken pair format
        if "trading_pair" in internal_data:
            pair = internal_data["trading_pair"]
            # Replace / with nothing for Kraken format (e.g., XBT/USD -> XBTUSD)
            kraken_params["pair"] = pair.replace("/", "")

        # Map order type
        if "order_type" in internal_data:
            order_type = internal_data["order_type"].lower()
            kraken_params["type"] = "buy" if order_type == "buy" else "sell"

        # Map order subtype (market, limit, etc.)
        if "order_subtype" in internal_data:
            subtype = internal_data["order_subtype"].lower()
            kraken_params["ordertype"] = subtype

            # For limit orders, include price
            if subtype == "limit" and "price" in internal_data:
                kraken_params["price"] = str(internal_data["price"])

        # Include volume (quantity)
        if "quantity" in internal_data:
            kraken_params["volume"] = str(internal_data["quantity"])

        # Optional parameters
        if "leverage" in internal_data:
            kraken_params["leverage"] = str(internal_data["leverage"])

        if "time_in_force" in internal_data:
            kraken_params["oflags"] = internal_data["time_in_force"]
        return kraken_params

    def _prepare_request_data(self, internal_data: dict[str, Any], action: str) -> dict[str, Any]:
        """
        Translate internal order details to Kraken API parameters.

        Args
        ----
            internal_data: Internal order parameters
            action: The action being performed

        Returns
        -------
            Kraken-specific parameters
        """
        kraken_params: dict[str, Any] = {"nonce": int(time.time() * 1000)}

        if action == "add_order":
            kraken_params.update(self._prepare_add_order_params(internal_data))
        elif action == "cancel_order" and "order_id" in internal_data:
            kraken_params["txid"] = internal_data["order_id"]
        elif action == "query_orders" and "order_ids" in internal_data:
            kraken_params["txid"] = ",".join(internal_data["order_ids"])

        return kraken_params

    def _generate_auth_headers(
        self, uri_path: str, request_data: dict[str, Any]
    ) -> dict[str, str]:
        """
        Generate Kraken-specific authentication headers.

        Args
        ----
            uri_path: API endpoint path
            request_data: Request parameters (the dictionary, not urlencoded string)

        Returns
        -------
            Authentication headers
        """
        if not self._api_key or not self._api_secret:
            raise KrakenCredentialsMissingError

        # Nonce is already in request_data and should be an int
        nonce = int(request_data["nonce"])

        # Signature is generated based on the request_data dictionary
        signature = self._generate_kraken_signature(uri_path, request_data, nonce)

        return {
            "API-Key": self._api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }

    def _parse_response(self, response_data: dict[str, Any], action: str) -> dict[str, Any]:
        """
        Parse Kraken's response for a given action into a standard format.

        Args
        ----
            response_data: Raw response from Kraken API
            action: The action that was performed

        Returns
        -------
            Standardized response dictionary
        """
        parsed: dict[str, Any] = {"success": False, "data": None, "error": None}

        # Check for error field (Kraken returns errors as a list)
        if response_data.get("error") and len(response_data["error"]) > 0:
            parsed["error"] = str(response_data["error"])
            self.logger.error(
                "Kraken API error for %s: %s",
                action,
                parsed["error"]
            )
        # Check for result field
        elif "result" in response_data:
            parsed["success"] = True
            parsed["data"] = response_data["result"]

            # Log specific details based on action type
            if action == "add_order" and "txid" in response_data["result"]:
                self.logger.info(
                    "Order placed successfully: %s",
                    response_data["result"]["txid"]
                )
            elif action == "cancel_order":
                self.logger.info(
                    "Order cancelled: %s",
                    response_data["result"]
                )
        else:
            parsed["error"] = "Unknown response format from Kraken API"
            self.logger.error(
                "Unknown response format from Kraken API for %s",
                action
            )

        return parsed

    def _generate_kraken_signature(self, uri_path: str, data: dict[str, Any], nonce: int) -> str:
        """
        Generate the API-Sign header required by Kraken private endpoints.

        This method now matches the superclass signature.

        Args
        ----
            uri_path: API endpoint path
            data: Request parameters as a dictionary
            nonce: Unique nonce value

        Returns
        -------
            Base64-encoded signature
        """
        if self._api_secret is None:
            # Should have been caught by _api_key/_api_secret check in _generate_auth_headers
            # but as a safeguard for direct calls or typing.
            raise KrakenApiSecretMissingError

        post_data_str = urllib.parse.urlencode(data)  # Urlencode the data here

        # Decode API secret from base64
        secret = base64.b64decode(self._api_secret)

        # Create signature input
        # Ensure nonce is a string when concatenating for sha256 input
        signature_message = (
            uri_path.encode() + hashlib.sha256((str(nonce) + post_data_str).encode()).digest()
        )

        # Create HMAC
        signature = hmac.new(secret, signature_message, hashlib.sha512)

        # Return base64-encoded signature
        return base64.b64encode(signature.digest()).decode()

    async def place_order(self, order_details: dict[str, Any]) -> dict[str, Any]:
        """
        Place an order on the Kraken exchange.

        Args
        ----
            order_details: Dictionary containing order details

        Returns
        -------
            Response containing order status
        """
        # Add Kraken-specific validation
        if "trading_pair" not in order_details:
            return {"success": False, "data": None, "error": "Trading pair is required"}

        if "order_type" not in order_details:
            return {"success": False, "data": None, "error": "Order type is required"}

        if "quantity" not in order_details:
            return {"success": False, "data": None, "error": "Quantity is required"}

        log_msg = "Placing {} order for {}".format(
            order_details.get("order_type"),
            order_details.get("trading_pair")
        )
        self.logger.info(log_msg, source_module=self.__class__.__name__)

        action = "add_order"
        uri_path = self._get_api_endpoint(action)
        # _prepare_request_data adds nonce
        kraken_params = self._prepare_request_data(order_details, action)

        # _make_private_request_with_retry is inherited from ExecutionHandler
        # It calls _generate_auth_headers which now calls the corrected _generate_kraken_signature
        api_result = await self._make_private_request_with_retry(uri_path, kraken_params)
        return self._parse_response(api_result, action)

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on the Kraken exchange.

        This method now implements the cancellation fully and returns bool.

        Args
        ----
            order_id: The ID of the order to cancel

        Returns
        -------
            True if cancellation was successful (or acknowledged by Kraken), False otherwise.
        """
        self.logger.info("Cancelling order: %s", order_id)

        action = "cancel_order"
        uri_path = self._get_api_endpoint(action)
        internal_details = {"order_id": order_id}
        # _prepare_request_data adds nonce
        kraken_params = self._prepare_request_data(internal_details, action)

        api_result = await self._make_private_request_with_retry(uri_path, kraken_params)
        parsed_response = self._parse_response(api_result, action)

        # Return True if the 'success' field in the parsed response is True
        # and there are no errors, or if Kraken indicates success in its specific way.
        # For Kraken, a successful cancel_order usually means
        # the 'count' of cancelled orders is > 0.
        if parsed_response.get("success") and parsed_response.get("data") is not None:
            # Kraken's CancelOrder response has a 'count' field in 'data'
            # if data is a dict and 'count' is present and > 0,
            # consider it a success.
            data_val = parsed_response["data"]
            if isinstance(data_val, dict) and data_val.get("count", 0) > 0:
                return True
            # If 'count' is not present or 0, but success is true and no error,
            # it might mean the order was already cancelled or didn't exist.
            # Depending on strictness, this could be True or False.
            # For now, let's assume if Kraken says success and gives data,
            # it's handled.
            # However, the prompt implies a boolean success/failure.
            # A more robust check might be needed
            # based on Kraken's exact API response for all cases.
            # If 'error' is present, it's definitely False.
            if parsed_response.get("error"):
                return False
            # If success is true, no error, but also no positive confirmation like 'count'.
            # This case is ambiguous. Let's default to False if positive confirmation is missing.
            # Re-evaluating: if success is True and no error, it should be True.
            # The presence of 'data' with 'count' > 0 is a stronger confirmation.
            return not parsed_response.get("error")  # Treat as successful if no error.

        return False  # Default to False if not explicitly successful

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        """
        Get the current status of an order on the Kraken exchange.

        This method now implements the status retrieval fully.

        Args
        ----
            order_id: The ID of the order to check

        Returns
        -------
            Response containing order status
        """
        # Construct log message to handle potentially long order_id
        log_message_part1 = "Checking status of order: "
        log_message = f"{log_message_part1}{order_id}"
        self.logger.info(log_message, source_module=self.__class__.__name__)

        action = "query_orders"
        uri_path = self._get_api_endpoint(action)

        # Prepare parameters for querying orders.
        # The '_prepare_request_data' method expects "order_ids" as a list.
        internal_details = {"order_ids": [order_id]}
        kraken_params = self._prepare_request_data(internal_details, action)

        # Add 'trades=true' to include trade information with the order status.
        # Kraken API expects this parameter as a string ("true" or "false").
        kraken_params["trades"] = "true"

        api_result = await self._make_private_request_with_retry(uri_path, kraken_params)
        return self._parse_response(api_result, action)
