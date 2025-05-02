# Execution Handler Module

import aiohttp
import asyncio
import base64
import binascii  # Add missing import for binascii
import hashlib
import hmac
import time
import urllib.parse
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple, cast, Callable, Coroutine  # Added Callable, Coroutine
from uuid import UUID
from datetime import datetime  # Missing import for datetime

from gal_friday.core.events import (
    EventType,
    TradeSignalApprovedEvent,
    ExecutionReportEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.config_manager import ConfigManager
from gal_friday.monitoring_service import MonitoringService
from gal_friday.logger_service import LoggerService

print("Execution Handler Loaded")

KRAKEN_API_URL = "https://api.kraken.com"


class ExecutionHandler:
    """
    Handles interaction with the exchange API (Kraken) to place, manage,
    and monitor orders based on approved trade signals.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        monitoring_service: MonitoringService,
        logger_service: LoggerService,
    ):
        self.logger = logger_service
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.monitoring = monitoring_service

        self.api_key = self.config.get("kraken.api_key", default=None)
        self.api_secret = self.config.get("kraken.secret_key", default=None)
        self.api_base_url = self.config.get("exchange.api_url", KRAKEN_API_URL)

        if not self.api_key or not self.api_secret:
            self.logger.critical(
                (
                    "Kraken API Key or Secret Key not configured. "
                    "ExecutionHandler cannot function."
                ),
                source_module=self.__class__.__name__,
            )
            # Consider raising an exception here to prevent startup without keys
            # raise ValueError("API keys missing")

        self._session: Optional[aiohttp.ClientSession] = None
        # TODO: Add state for managing WebSocket connection if used
        # TODO: Add mapping for internal IDs to exchange IDs (cl_ord_id -> txid)
        self._order_map: Dict[str, str] = {}  # cl_ord_id -> txid
        self._pair_info: Dict[str, Dict[str, Any]] = {}  # Internal pair -> Kraken details
        # Add type hint for the handler attribute
        self._trade_signal_handler: Optional[Callable[[TradeSignalApprovedEvent], Coroutine[Any, Any, None]]] = None

        self.logger.info(
            "ExecutionHandler initialized.",
            source_module=self.__class__.__name__,
        )

    async def start(self) -> None:
        """
        Initializes API client session, loads exchange info, and subscribes to events.
        """
        self.logger.info(
            "Starting ExecutionHandler...",
            source_module=self.__class__.__name__,
        )
        self._session = aiohttp.ClientSession()
        await self._load_exchange_info()
        # Check if info loading failed significantly
        if not self._pair_info:
            self.logger.error(
                (
                    "Failed to load essential exchange pair info. "
                    "ExecutionHandler will not function correctly."
                ),
                source_module=self.__class__.__name__,
            )
            # Consider preventing full startup? For now, just log error.
            # await self.stop() # Option: Stop immediately
            # return

        # Store the handler for unsubscribing
        self._trade_signal_handler = self.handle_trade_signal_approved
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler)
        self.logger.info(
            "ExecutionHandler started. Subscribed to TRADE_SIGNAL_APPROVED.",
            source_module=self.__class__.__name__,
        )
        # TODO: Implement WebSocket connection logic here if used for MVP

    async def stop(self) -> None:
        """Closes API client session and potentially cancels orders."""
        self.logger.info(
            "Stopping ExecutionHandler...",
            source_module=self.__class__.__name__,
        )

        # Unsubscribe first
        # Check if the handler is not None instead of truthiness
        if self._trade_signal_handler is not None:
            try:
                self.pubsub.unsubscribe(EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler)
                self.logger.info("Unsubscribed from TRADE_SIGNAL_APPROVED.")
                self._trade_signal_handler = None
            except Exception as e:
                self.logger.error(f"Error unsubscribing: {e}", exc_info=True)

        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info(
                "AIOHTTP session closed.",
                source_module=self.__class__.__name__,
            )

        self.logger.info(
            "ExecutionHandler stopped.",
            source_module=self.__class__.__name__,
        )
        # TODO: Implement WebSocket disconnection logic
        # TODO: Implement configurable cancellation of open orders on stop

    async def _load_exchange_info(self) -> None:
        """Fetches and stores tradable asset pair information from Kraken."""
        uri_path = "/0/public/AssetPairs"
        url = self.api_base_url + uri_path
        self.logger.info(
            f"Loading exchange asset pair info from {url}...",
            source_module=self.__class__.__name__,
        )

        if not self._validate_session():
            return

        try:
            result = await self._fetch_asset_pairs(url)
            if not result:
                return

            await self._process_asset_pairs(result)

        except aiohttp.ClientResponseError as e:
            self.logger.error(
                f"HTTP Error fetching AssetPairs: {e.status} {e.message}",
                source_module=self.__class__.__name__,
            )
        except aiohttp.ClientConnectionError as e:
            self.logger.error(
                f"Connection Error fetching AssetPairs: {e}",
                source_module=self.__class__.__name__,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                f"Timeout fetching AssetPairs from {url}",
                source_module=self.__class__.__name__,
            )
        except Exception:  # Catch-all for unexpected errors
            self.logger.error(
                "Unexpected error loading exchange info.",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    def _validate_session(self) -> bool:
        """Validates that the AIOHTTP session is available."""
        if not self._session or self._session.closed:
            self.logger.error(
                "Cannot load exchange info: AIOHTTP session is not available.",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    async def _fetch_asset_pairs(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetches asset pairs data from the exchange API."""
        if not self._session:
            return None
            
        async with self._session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            response.raise_for_status()
            data: Dict[str, Any] = await response.json()

            if data.get("error"):
                self.logger.error(
                    f"Error fetching AssetPairs: {data['error']}",
                    source_module=self.__class__.__name__,
                )
                return None

            result = data.get("result", {})
            if not result:
                self.logger.error(
                    "AssetPairs result is empty.",
                    source_module=self.__class__.__name__,
                )
                return None

            return cast(Dict[str, Any], result)

    async def _process_asset_pairs(self, result: dict) -> None:
        """Processes and stores asset pairs data."""
        loaded_count = 0
        internal_pairs = self.config.get_list("trading.pairs", [])

        if not internal_pairs:
            self.logger.warning(
                (
                    "No trading pairs defined in config [trading.pairs]. "
                    "Cannot map exchange info."
                ),
                source_module=self.__class__.__name__,
            )
            return

        kraken_pair_map = {v.get("altname", k): k for k, v in result.items()}

        for internal_pair_name in internal_pairs:
            if self._process_single_pair(
                internal_pair_name, kraken_pair_map, result
            ):
                loaded_count += 1

        self._log_loading_results(loaded_count, len(internal_pairs))

    def _process_single_pair(
        self,
        internal_pair_name: str,
        kraken_pair_map: dict,
        result: dict,
    ) -> bool:
        """Processes a single trading pair and stores its information."""
        kraken_altname = internal_pair_name.replace("/", "")
        kraken_key = kraken_pair_map.get(kraken_altname)

        if not kraken_key or kraken_key not in result:
            self.logger.warning(
                (
                    f"Could not find matching AssetPairs info for "
                    f"configured pair: {internal_pair_name}"
                ),
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
        self.logger.debug(
            f"Loaded info for {internal_pair_name}",
            source_module=self.__class__.__name__,
        )
        return True

    def _log_loading_results(self, loaded_count: int, total_pairs: int) -> None:
        """Logs the results of loading asset pairs."""
        self.logger.info(
            (
                f"Successfully loaded info for {loaded_count} asset pairs "
                f"out of {total_pairs} configured."
            ),
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

    def _generate_kraken_signature(
        self,
        uri_path: str,
        data: Dict[str, Any],
        nonce: int,
    ) -> str:
        """Generates the API-Sign header required by Kraken private endpoints."""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(nonce) + postdata).encode()
        message = uri_path.encode() + hashlib.sha256(encoded).digest()

        try:
            secret_decoded = base64.b64decode(self.api_secret)
        except binascii.Error as e:
            self.logger.error(
                f"Invalid base64 API secret: {e}",
                source_module=self.__class__.__name__,
            )
            raise ValueError("Invalid API Secret format") from e

        mac = hmac.new(secret_decoded, message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _format_decimal(self, value: Decimal, precision: int) -> str:
        """Formats a Decimal value to a string with a specific precision."""
        # Use quantization to set the number of decimal places
        # Ensure it rounds correctly, default rounding is ROUND_HALF_EVEN
        quantizer = Decimal("1e-" + str(precision))
        return str(value.quantize(quantizer))

    async def _make_private_request(
        self,
        uri_path: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Makes an authenticated request to a private Kraken REST endpoint."""
        if not self._session or self._session.closed:
            self.logger.error(
                "AIOHTTP session is not available for private request.",
                source_module=self.__class__.__name__,
            )
            return {"error": ["EGeneral:InternalError - HTTP session closed"]}

        # Generate nonce and signature
        nonce = int(time.time() * 1000)  # Kraken uses milliseconds nonce
        request_data = data.copy()  # Avoid modifying the original dict
        request_data["nonce"] = nonce

        try:
            api_sign = self._generate_kraken_signature(uri_path, request_data, nonce)
        except ValueError as e:
            # Handle invalid API secret format from signature generation
            self.logger.error(
                f"Failed to generate signature: {e}",
                source_module=self.__class__.__name__,
            )
            return {"error": [f"EGeneral:InternalError - {e}"]}

        headers = {
            "API-Key": self.api_key,
            "API-Sign": api_sign,
            # Important for Kraken's API requirements
            "Content-Type": (
                "application/x-www-form-urlencoded; charset=utf-8"
            ),
        }
        url = self.api_base_url + uri_path

        # Timeout configuration (example: 10 seconds total)
        timeout = aiohttp.ClientTimeout(
            total=self.config.get("exchange.request_timeout_seconds", 10)
        )

        try:
            self.logger.debug(
                f"Sending private request to {url} with data: {request_data}",
                source_module=self.__class__.__name__,
            )
            async with self._session.post(
                url, headers=headers, data=request_data, timeout=timeout
            ) as response:

                # Log rate limit headers if present (useful for debugging)
                # remaining = response.headers.get('RateLimit-Remaining')
                # limit = response.headers.get('RateLimit-Limit')
                # if remaining and limit:
                #     self.logger.debug(
                #         f"Rate Limit: {remaining}/{limit} remaining.",
                #         source_module=self.__class__.__name__,
                #     )

                # Check HTTP status first
                response.raise_for_status()  # Raise exception for bad status codes (4xx, 5xx)

                # Process successful response
                result: Dict[str, Any] = await response.json()
                self.logger.debug(
                    f"Received response from {url}: {result}",
                    source_module=self.__class__.__name__,
                )

                # Check for API-level errors within the JSON response
                if result.get("error"):
                    # Don't log full data dict here, might contain sensitive info inadvertently
                    self.logger.error(
                        f"Kraken API error for {uri_path}: {result['error']}",
                        source_module=self.__class__.__name__,
                    )
                    # Return the full error structure as Kraken provides it

                return result

        except aiohttp.ClientResponseError as e:
            # Handle HTTP errors (400 Bad Request, 401 Unauthorized, etc.)
            error_body = await response.text()
            self.logger.error(
                (
                    f"HTTP Error: {e.status} {e.message} "
                    f"for {e.request_info.url}. "
                    f"Body: {error_body[:500]}"
                ),
                source_module=self.__class__.__name__,
            )
            # Standardize error format slightly for internal handling
            return {"error": [f"EGeneral:HTTPError - {e.status}: {e.message}"]}
        except aiohttp.ClientConnectionError as e:
            self.logger.error(
                f"Connection Error to {url}: {e}", source_module=self.__class__.__name__
            )
            # TODO: Implement retry logic here for connection errors?
            return {"error": [f"EGeneral:ConnectionError - {e}"]}
        except asyncio.TimeoutError:
            self.logger.error(
                f"Request Timeout for {url}",
                source_module=self.__class__.__name__,
            )
            return {"error": ["EGeneral:Timeout"]}
        except Exception as e:
            # Catch-all for unexpected errors during the request/response handling
            self.logger.error(
                f"Unexpected error during private API request to {url}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            return {"error": [f"EGeneral:Unexpected - {e}"]}

    async def handle_trade_signal_approved(self, event: TradeSignalApprovedEvent) -> None:
        """
        Processes an approved trade signal event: checks HALT, translates, places order,
        handles response.
        """
        self.logger.info(
            f"Received approved trade signal: {event.signal_id}",
            source_module=self.__class__.__name__,
        )

        # 1. Check HALT status FIRST
        if self.monitoring.is_halted():
            error_msg = "Execution blocked: System HALTED"
            self.logger.critical(
                (
                    f"{error_msg}. Discarding approved signal: {event.signal_id} "
                    f"({event.trading_pair} {event.side} {event.quantity})"
                ),
                source_module=self.__class__.__name__,
            )
            # Publish a REJECTED execution report for tracking
            # Assuming self._publish_error_execution_report exists and takes optional cl_ord_id
            asyncio.create_task(
                self._publish_error_execution_report(
                    event, error_msg, cl_ord_id=f"internal_{event.signal_id}_halted"
                )
            )
            return  # Stop processing this signal

        # 2. Translate the signal to API parameters
        kraken_params = self._translate_signal_to_kraken_params(event)

        # 3. Handle translation failure
        if not kraken_params:
            self.logger.error(
                f"Failed to translate signal {event.signal_id}. Order not placed.",
                source_module=self.__class__.__name__,
            )
            # Publish an error report to indicate failure before sending
            await self._publish_error_execution_report(event, "Signal translation failed", None)
            return

        # 4. Generate Client Order ID and add to params
        # Using timestamp and signal prefix for basic uniqueness
        cl_ord_id = (
            f"gf-{str(event.signal_id)[:8]}-{int(time.time() * 1000000)}"  # Microseconds
        )
        kraken_params["cl_ord_id"] = cl_ord_id
        # Optional: Add userref if needed
        # kraken_params['userref'] = ...

        # 5. Make the API request to place the order
        self.logger.info(
            f"Placing order for signal {event.signal_id} with cl_ord_id {cl_ord_id}",
            source_module=self.__class__.__name__,
        )
        uri_path = "/0/private/AddOrder"  # For single order placement
        # TODO: Consider using AddOrderBatch if placing SL/TP simultaneously later
        result = await self._make_private_request(uri_path, kraken_params)

        # 6. Handle the response from the AddOrder call
        await self._handle_add_order_response(result, event, cl_ord_id)

    def _translate_signal_to_kraken_params(
        self,
        event: TradeSignalApprovedEvent,
    ) -> Optional[Dict[str, Any]]:
        """
        Translates internal signal format to Kraken API parameters,
        including validation.
        """
        params = {}
        internal_pair = event.trading_pair

        # 1. Get and validate pair info
        pair_info = self._get_and_validate_pair_info(internal_pair, event.signal_id)
        if not pair_info:
            return None

        # 2. Get and validate pair name
        kraken_pair_name = self._get_and_validate_pair_name(
            internal_pair, pair_info, event.signal_id
        )
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
            f"Translated signal {event.signal_id} to Kraken params: {params}",
            source_module=self.__class__.__name__,
        )
        return params

    def _get_and_validate_pair_info(
        self,
        internal_pair: str,
        signal_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """Gets and validates trading pair information."""
        # Convert UUID to string for logging
        signal_id_str = str(signal_id)
        
        pair_info = self._pair_info.get(internal_pair)
        if not pair_info:
            self.logger.error(
                (
                    f"No exchange info found for pair {internal_pair}. "
                    f"Cannot translate signal {signal_id_str}."
                ),
                source_module=self.__class__.__name__,
            )
            return None

        if pair_info.get("status") != "online":
            self.logger.error(
                (
                    f"Pair {internal_pair} is not online "
                    f"(status: {pair_info.get('status')}). "
                    f"Cannot place order for signal {signal_id_str}."
                ),
                source_module=self.__class__.__name__,
            )
            return None

        return pair_info

    def _get_and_validate_pair_name(
        self,
        internal_pair: str,
        pair_info: Dict[str, Any],
        signal_id: UUID,
    ) -> Optional[str]:
        """Gets and validates the Kraken pair name."""
        # Convert UUID to string for logging
        signal_id_str = str(signal_id)
        
        kraken_pair_name = cast(Optional[str], pair_info.get("altname"))
        if not kraken_pair_name:
            self.logger.error(
                (
                    f"Missing Kraken altname for pair {internal_pair} "
                    f"in loaded info for signal {signal_id_str}."
                ),
                source_module=self.__class__.__name__,
            )
            return None
        return kraken_pair_name

    def _validate_and_set_order_side(
        self,
        params: Dict[str, Any],
        event: TradeSignalApprovedEvent,
    ) -> bool:
        """Validates and sets the order side parameter."""
        order_side = event.side.lower()
        if order_side not in ["buy", "sell"]:
            self.logger.error(
                f"Invalid order side '{event.side}' in signal {event.signal_id}.",
                source_module=self.__class__.__name__,
            )
            return False
        params["type"] = order_side
        return True

    def _validate_and_format_volume(
        self,
        params: Dict[str, Any],
        event: TradeSignalApprovedEvent,
        pair_info: Dict[str, Any],
    ) -> bool:
        """Validates and formats the order volume."""
        lot_decimals = pair_info.get("lot_decimals")
        ordermin_str = pair_info.get("ordermin")
        if lot_decimals is None or ordermin_str is None:
            self.logger.error(
                (
                    f"Missing lot_decimals or ordermin for pair {event.trading_pair}. "
                    "Cannot validate/format volume."
                ),
                source_module=self.__class__.__name__,
            )
            return False

        try:
            ordermin = Decimal(ordermin_str)
            if event.quantity < ordermin:
                self.logger.error(
                    (
                        f"Order quantity {event.quantity} is below minimum {ordermin} "
                        f"for pair {event.trading_pair}. Signal {event.signal_id}."
                    ),
                    source_module=self.__class__.__name__,
                )
                return False
            params["volume"] = self._format_decimal(event.quantity, lot_decimals)
            return True
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"Error processing volume/ordermin for pair {event.trading_pair}: {e}",
                source_module=self.__class__.__name__,
            )
            return False

    def _map_and_validate_order_type(
        self,
        params: Dict[str, Any],
        event: TradeSignalApprovedEvent,
        pair_info: Dict[str, Any],
    ) -> bool:
        """Maps and validates the order type, setting price for limit orders."""
        order_type = event.order_type.lower()
        pair_decimals = pair_info.get("pair_decimals")

        if pair_decimals is None:
            self.logger.error(
                (
                    f"Missing pair_decimals for pair {event.trading_pair}. "
                    "Cannot format price."
                ),
                source_module=self.__class__.__name__,
            )
            return False

        if order_type == "limit":
            return self._handle_limit_order(params, event, pair_decimals)
        elif order_type == "market":
            params["ordertype"] = "market"
            return True
        else:
            self.logger.error(
                (
                    f"Unsupported order type '{event.order_type}' for Kraken "
                    f"translation. Signal {event.signal_id}."
                ),
                source_module=self.__class__.__name__,
            )
            return False

    def _handle_limit_order(
        self,
        params: Dict[str, Any],
        event: TradeSignalApprovedEvent,
        pair_decimals: int,
    ) -> bool:
        """Handles limit order specific parameters and validation."""
        params["ordertype"] = "limit"
        if event.limit_price is None:
            self.logger.error(
                (
                    f"Limit price is required for limit order. "
                    f"Signal {event.signal_id}."
                ),
                source_module=self.__class__.__name__,
            )
            return False
        try:
            params["price"] = self._format_decimal(event.limit_price, pair_decimals)
            return True
        except (TypeError, ValueError) as e:
            self.logger.error(
                f"Error processing limit price for pair {event.trading_pair}: {e}",
                source_module=self.__class__.__name__,
            )
            return False

    def _handle_sl_tp_warnings(self, event: TradeSignalApprovedEvent) -> None:
        """Handles warnings for stop-loss and take-profit parameters."""
        if event.sl_price or event.tp_price:
            self.logger.warning(
                (
                    f"SL/TP prices found in signal {event.signal_id}, "
                    "but handling is deferred in MVP ExecutionHandler."
                ),
                source_module=self.__class__.__name__,
            )

    async def _handle_add_order_response(
        self,
        result: Dict[str, Any],
        originating_event: TradeSignalApprovedEvent,
        cl_ord_id: str,
    ) -> None:
        """Processes the response from the AddOrder API call and publishes initial status."""
        if not result:
            # Should not happen if _make_private_request works correctly, but check anyway
            self.logger.error(
                (
                    f"Received empty response for AddOrder call related to "
                    f"signal {originating_event.signal_id}"
                ),
                source_module=self.__class__.__name__,
            )
            await self._publish_error_execution_report(
                originating_event, "Empty API response", cl_ord_id
            )
            return

        # Check for API-level errors first
        if result.get("error"):
            error_msg = str(result["error"])  # Kraken errors are usually a list of strings
            self.logger.error(
                (
                    f"AddOrder API call failed for signal {originating_event.signal_id} "
                    f"(cl_ord_id: {cl_ord_id}): {error_msg}"
                ),
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
                kraken_order_id = txids[0]  # Assuming single order response for now
                self.logger.info(
                    (
                        f"Order placed successfully via API for signal "
                        f"{originating_event.signal_id}. cl_ord_id: {cl_ord_id}, "
                        f"Kraken TXID: {kraken_order_id}, Description: {descr}"
                    ),
                    source_module=self.__class__.__name__,
                )

                # Store the mapping for future reference (e.g., cancellation, status checks)
                self._order_map[cl_ord_id] = kraken_order_id

                # Publish initial "NEW" execution report
                report = ExecutionReportEvent(
                    source_module=self.__class__.__name__,
                    event_id=UUID(int=0),  # Generate a proper UUID
                    timestamp=datetime.utcnow(),
                    signal_id=originating_event.signal_id,
                    exchange_order_id=kraken_order_id,
                    client_order_id=cl_ord_id,
                    trading_pair=originating_event.trading_pair,
                    exchange=self.config.get("exchange.name", "kraken"),
                    order_status="NEW", 
                    order_type=originating_event.order_type,
                    side=originating_event.side,
                    # Ensure quantity/price types match Event definition (Decimal)
                    quantity_ordered=originating_event.quantity, 
                    quantity_filled=Decimal(0),
                    limit_price=originating_event.limit_price,
                    average_fill_price=None,
                    commission=None,
                    commission_asset=None,
                    timestamp_exchange=None, # API response might contain a timestamp?
                    error_message=None,
                )
                # Using asyncio.create_task for fire-and-forget publishing
                asyncio.create_task(self.pubsub.publish(report))
                self.logger.debug(
                    f"Published NEW ExecutionReport for {cl_ord_id} / {kraken_order_id}",
                    source_module=self.__class__.__name__,
                )

            else:
                # This case indicates success HTTP status but unexpected result format
                error_msg = "AddOrder response missing or invalid 'txid' field."
                self.logger.error(
                    f"{error_msg} cl_ord_id: {cl_ord_id}. Response: {result}",
                    source_module=self.__class__.__name__,
                )
                await self._publish_error_execution_report(originating_event, error_msg, cl_ord_id)

        except Exception as e:
            # Catch potential errors during response parsing
            self.logger.error(
                f"Error processing successful AddOrder response for signal "
                f"{originating_event.signal_id} (cl_ord_id: {cl_ord_id})",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            await self._publish_error_execution_report(
                originating_event, f"Internal error processing response: {e}", cl_ord_id
            )

    async def _publish_error_execution_report(
        self,
        originating_event: TradeSignalApprovedEvent,
        error_msg: str,
        cl_ord_id: Optional[str] = None,
    ) -> None:
        """Helper method to publish an ExecutionReportEvent indicating a failure."""
        self.logger.info(
            (
                f"Publishing error execution report for signal {originating_event.signal_id} "
                f"(cl_ord_id: {cl_ord_id}): {error_msg}"
            ),
            source_module=self.__class__.__name__,
        )
        try:
            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=UUID(int=0),  # Generate a proper UUID
                timestamp=datetime.utcnow(),
                signal_id=originating_event.signal_id,
                exchange_order_id="none" if not cl_ord_id else cl_ord_id,  # Use non-None value
                client_order_id=cl_ord_id,
                trading_pair=originating_event.trading_pair,
                exchange="kraken",  # Assuming Kraken
                order_status="REJECTED",  # Or maybe "ERROR"
                order_type=originating_event.order_type,
                side=originating_event.side,
                quantity_ordered=originating_event.quantity,
                # Filled quantity, price, commission are 0/None for rejection
                quantity_filled=Decimal(0),
                average_fill_price=None,
                limit_price=originating_event.limit_price,
                stop_price=None,  # Need to decide how SL/TP prices are handled in signals/reports
                commission=None,
                commission_asset=None,
                timestamp_exchange=None,
                error_message=error_msg,
            )
            # Publish the ExecutionReportEvent using the refactored PubSubManager
            # Use create_task for fire-and-forget
            asyncio.create_task(self.pubsub.publish(report)) 
            self.logger.critical(
                f"Published ERROR ExecutionReport. Reason: {error_msg}",
                source_module=self.__class__.__name__,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to publish ERROR execution report: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    async def _connect_websocket(self) -> None:
        pass  # Placeholder

    async def _handle_websocket_message(self, message: Dict[str, Any]) -> None:
        pass  # Placeholder

    async def cancel_order(self, order_id: str, is_cl_ord_id: bool = False) -> None:
        pass  # Placeholder
