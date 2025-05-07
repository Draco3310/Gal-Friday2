# Execution Handler Module

import aiohttp
import asyncio
import base64
import binascii  # Add missing import for binascii
import hashlib
import hmac
import time
import urllib.parse
import random  # Add import for random (needed for jitter)
from decimal import Decimal

# Added Callable, Coroutine
from typing import Dict, Any, Optional, cast, Callable, Coroutine, List, Set, Tuple
from uuid import UUID
from datetime import datetime, timezone  # Modified import

from gal_friday.core.events import (
    EventType,
    TradeSignalApprovedEvent,
    ExecutionReportEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.config_manager import ConfigManager
from gal_friday.monitoring_service import MonitoringService
from gal_friday.logger_service import LoggerService

# TODO: Replace debug print with proper logging
print("Execution Handler Loaded")

KRAKEN_API_URL = "https://api.kraken.com"


class RateLimitTracker:
    """
    Tracks and enforces API rate limits to prevent exceeding exchange limits.
    """

    def __init__(
        self, config: ConfigManager, logger_service: LoggerService
    ):  # Added logger_service
        self.config = config
        self.logger: LoggerService = logger_service  # Added type hint and assigned logger_service

        # Configure rate limits based on tier/API key level
        # These should come from configuration
        self.private_calls_per_second = self.config.get_int(
            "exchange.rate_limit.private_calls_per_second", 1
        )
        self.public_calls_per_second = self.config.get_int(
            "exchange.rate_limit.public_calls_per_second", 1
        )

        # Tracking timestamps of recent calls
        self._private_call_timestamps: List[float] = []
        self._public_call_timestamps: List[float] = []

        # Window size in seconds for tracking
        self.window_size = 1.0  # 1 second window

        self._source_module = self.__class__.__name__

    async def wait_for_private_capacity(self) -> None:
        """
        Waits until there's capacity to make a private API call.
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
        """
        Waits until there's capacity to make a public API call.
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
        """Resets all tracking."""
        self._private_call_timestamps = []
        self._public_call_timestamps = []


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
        # TODO: Add mapping for internal IDs to exchange IDs (cl_ord_id ->
        # txid)
        self._order_map: Dict[str, str] = {}  # cl_ord_id -> txid
        # Internal pair -> Kraken details
        self._pair_info: Dict[str, Dict[str, Any]] = {}
        # Add type hint for the handler attribute
        self._trade_signal_handler: Optional[
            Callable[[TradeSignalApprovedEvent], Coroutine[Any, Any, None]]
        ] = None

        # Store active monitoring tasks
        self._order_monitoring_tasks: Dict[str, asyncio.Task] = {}  # txid -> Task

        # Track signals that have had SL/TP orders placed
        self._placed_sl_tp_signals: Set[UUID] = set()

        # Initialize rate limiter
        self.rate_limiter = RateLimitTracker(self.config, self.logger)  # Pass logger

        # TODO: Implement Kraken Adapter Pattern
        # The current implementation directly interacts with Kraken API.
        # Future refactoring should create a BaseExecutionAdapter abstract class
        # defining methods like place_order, cancel_order, query_order_status,
        # get_exchange_info. Then implement a KrakenExecutionAdapter class
        # inheriting from the base, moving all Kraken-specific logic (URL paths,
        # authentication, parameter translation, response parsing) into it.
        # The ExecutionHandler would then use this adapter.

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
                self.pubsub.unsubscribe(
                    EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler
                )
                self.logger.info("Unsubscribed from TRADE_SIGNAL_APPROVED.")
                self._trade_signal_handler = None
            except Exception as e:
                self.logger.error(f"Error unsubscribing: {e}", exc_info=True)

        # Cancel ongoing monitoring tasks
        try:
            for task_id, task in list(self._order_monitoring_tasks.items()):
                if not task.done():
                    task.cancel()
                    self.logger.info(f"Cancelled monitoring task for order {task_id}")
            self._order_monitoring_tasks.clear()
            self.logger.info("All order monitoring tasks cancelled.")
        except Exception as e:
            self.logger.error(
                f"Error cancelling monitoring tasks: {e}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )

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

    async def _make_public_request_with_retry(
        self, url: str, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Makes a public request with retry logic for transient errors."""
        base_delay = self.config.get_float("exchange.retry_base_delay_s", 1.0)
        # last_exception = None # F841: local variable
        # 'last_exception' is assigned to but never used

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

                async with self._session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    response.raise_for_status()
                    data: Dict[str, Any] = await response.json()

                    if data.get("error"):
                        error_str = str(data["error"])
                        if self._is_retryable_error(error_str) and attempt < max_retries:
                            delay = min(base_delay * (2**attempt), 30.0)
                            jitter = random.uniform(0, delay * 0.1)
                            total_delay = delay + jitter
                            self.logger.warning(
                                (
                                    f"Retryable API error for {url}: {error_str}. "
                                    f"Retrying in {total_delay:.2f}s "
                                    f"(Attempt {attempt + 1}/{max_retries + 1})"
                                ),
                                source_module=self.__class__.__name__,
                            )
                            await asyncio.sleep(total_delay)
                            continue

                        self.logger.error(
                            f"Error in public API response: {error_str}",
                            source_module=self.__class__.__name__,
                        )
                        return None

                    return data

            except (
                aiohttp.ClientResponseError,
                aiohttp.ClientConnectionError,
                asyncio.TimeoutError,
            ) as e:
                # last_exception = e
                if attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 30.0)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        (
                            f"Error during public request to {url}: {e}. "
                            f"Retrying in {total_delay:.2f}s "
                            f"(Attempt {attempt + 1}/{max_retries + 1})"
                        ),
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    continue
                else:
                    self.logger.error(
                        f"Failed to make public request to {url}"
                        f"after {max_retries + 1} attempts."
                        f"Last error: {e}",
                        source_module=self.__class__.__name__,
                    )
                    return None
            except Exception as e:
                self.logger.error(
                    f"Unexpected error during public request to {url}: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )
                return None

        last_error_message = "Unknown error"
        if "last_exception" in locals() and locals()["last_exception"]:
            last_error_message = str(locals()["last_exception"])
        self.logger.error(
            (
                f"Failed to make public request to {url} "
                f"after {max_retries + 1} attempts. "
                f"Last error: {last_error_message}"
            ),
            source_module=self.__class__.__name__,
        )
        return None

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
            # Use the new method with retry and rate limiting for public requests
            data = await self._make_public_request_with_retry(url)
            if not data:
                self.logger.error(
                    "Failed to fetch asset pairs data.",
                    source_module=self.__class__.__name__,
                )
                return

            result = data.get("result", {})
            if not result:
                self.logger.error(
                    "AssetPairs result is empty.",
                    source_module=self.__class__.__name__,
                )
                return

            await self._process_asset_pairs(result)

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
            if self._process_single_pair(internal_pair_name, kraken_pair_map, result):
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
            "Content-Type": ("application/x-www-form-urlencoded; charset=utf-8"),
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
                    # Don't log full data dict here, might contain sensitive
                    # info inadvertently
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
            # Catch-all for unexpected errors during the request/response
            # handling
            self.logger.error(
                f"Unexpected error during private API request to {url}",
                source_module=self.__class__.__name__,
                exc_info=True,
            )
            return {"error": [f"EGeneral:Unexpected - {e}"]}

    def _is_retryable_error(self, error_str: str) -> bool:
        """Check if a Kraken error string indicates a potentially transient issue."""
        # Add known transient error codes/messages from Kraken docs
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
        self, uri_path: str, data: Dict[str, Any], max_retries: int = 3
    ) -> Dict[str, Any]:
        """Makes a private request with retry logic for transient errors."""
        base_delay = self.config.get_float("exchange.retry_base_delay_s", 1.0)
        # last_exception = None # F841: local variable 'last_exception'
        # is assigned to but never used
        last_result = None

        for attempt in range(max_retries + 1):
            try:
                # Wait for rate limit capacity before making the request
                await self.rate_limiter.wait_for_private_capacity()

                result = await self._make_private_request(uri_path, data)
                last_result = result

                # Check for API-level errors within the result
                if result.get("error"):
                    error_str = str(result["error"])
                    if self._is_retryable_error(error_str) and attempt < max_retries:
                        delay = min(base_delay * (2**attempt), 30.0)  # Cap delay at 30s
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter
                        self.logger.warning(
                            (
                                f"Retryable API error for {uri_path}: {error_str}. "
                                f"Retrying in {total_delay:.2f}s "
                                f"(Attempt {attempt + 1}/{max_retries + 1})"
                            ),
                            source_module=self.__class__.__name__,
                        )
                        await asyncio.sleep(total_delay)
                        continue  # Go to next attempt
                    else:
                        # Permanent error or max retries reached
                        return result
                else:
                    # Successful API call (no 'error' field or empty error list)
                    return result

            except Exception as e:
                # Catch exceptions from _make_private_request itself
                self.logger.error(
                    f"Unexpected exception during retry loop for {uri_path}: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )
                # last_exception = e

                # Determine if the exception is retryable
                if (
                    isinstance(e, (aiohttp.ClientConnectionError, asyncio.TimeoutError))
                    and attempt < max_retries
                ):
                    delay = min(base_delay * (2**attempt), 30.0)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        (
                            f"Network error for {uri_path}: {e}. "
                            f"Retrying in {total_delay:.2f}s "
                            f"(Attempt {attempt + 1}/{max_retries + 1})"
                        ),
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    continue
                else:
                    break  # Non-retryable exception

        # If loop finishes, all retries failed
        self.logger.error(
            (f"API request to {uri_path} failed after {max_retries + 1} " f"attempts."),
            source_module=self.__class__.__name__,
        )

        # Return the last known error result or a generic max retries error
        if last_result:
            return last_result

        # If we have an exception but no result, create a generic error response
        # if isinstance(last_exception, Exception):
        #     return {"error": [f"EGeneral:UnexpectedRetryFailure - {last_exception}"]}

        # Fallback generic error
        return {"error": ["EGeneral:MaxRetriesExceeded"]}

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
            # Assuming self._publish_error_execution_report exists and takes
            # optional cl_ord_id
            asyncio.create_task(
                self._publish_error_execution_report(
                    event=event,  # Pass event as a keyword argument
                    error_message=error_msg,  # Pass error_message as a keyword argument
                    cl_ord_id=f"internal_{event.signal_id}_halted",
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
            await self._publish_error_execution_report(
                event=event,  # Pass event as a keyword argument
                error_message="Signal translation failed",
                # Pass error_message as a keyword argument
                cl_ord_id=None  # Client order ID
            )
            return

        # 4. Generate Client Order ID and add to params
        # Using timestamp and signal prefix for basic uniqueness
        cl_ord_id = (
            # Microseconds
            f"gf-{str(event.signal_id)[:8]}-{int(time.time() * 1000000)}"
        )
        kraken_params["cl_ord_id"] = cl_ord_id
        # Optional: Add userref if needed
        # kraken_params['userref'] = ...

        # 5. Make the API request to place the order
        self.logger.info(
            f"Placing order for signal {event.signal_id} " f"with cl_ord_id {cl_ord_id}",
            source_module=self.__class__.__name__,
        )
        uri_path = "/0/private/AddOrder"  # For single order placement
        # TODO: Consider using AddOrderBatch if placing SL/TP simultaneously
        # later

        # Updated to use retry logic
        result = await self._make_private_request_with_retry(uri_path, kraken_params)

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
                (f"Missing pair_decimals for pair {event.trading_pair}. Cannot format price."),
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
                (f"Limit price is required for limit order. " f"Signal {event.signal_id}."),
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
            # Should not happen if _make_private_request works correctly, but
            # check anyway
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
            # Kraken errors are usually a list of strings
            error_msg = str(result["error"])
            self.logger.error(
                (
                    f"AddOrder API call failed for signal {originating_event.signal_id} "
                    f"(cl_ord_id: {cl_ord_id}): {error_msg}"
                ),
                source_module=self.__class__.__name__,
            )
            # Publish REJECTED/ERROR status
            await self._publish_error_execution_report(
                originating_event, error_msg, cl_ord_id
            )
            return

        # Process successful response
        try:
            kraken_result_data = result.get("result", {})
            txids = kraken_result_data.get("txid")
            descr = kraken_result_data.get("descr", {}).get("order", "N/A")

            if txids and isinstance(txids, list):
                # Assuming single order response for now
                kraken_order_id = txids[0]
                self.logger.info(
                    (
                        f"Order placed successfully via API for signal "
                        f"{originating_event.signal_id}. cl_ord_id: {cl_ord_id}, "
                        f"Kraken TXID: {kraken_order_id}, Description: {descr}"
                    ),
                    source_module=self.__class__.__name__,
                )

                # Store the mapping for future reference (e.g., cancellation,
                # status checks)
                self._order_map[cl_ord_id] = kraken_order_id

                # Publish initial "NEW" execution report
                report = ExecutionReportEvent(
                    source_module=self.__class__.__name__,
                    event_id=UUID(int=int(time.time() * 1000000)),  # Generate a proper UUID
                    timestamp=datetime.utcnow(),
                    signal_id=originating_event.signal_id,
                    exchange_order_id=kraken_order_id,
                    client_order_id=cl_ord_id,
                    trading_pair=originating_event.trading_pair,
                    exchange=self.config.get("exchange.name", "kraken"),
                    order_status="NEW",
                    order_type=originating_event.order_type,
                    side=originating_event.side,
                    # Ensure quantity/price types match Event definition
                    # (Decimal)
                    quantity_ordered=originating_event.quantity,
                    quantity_filled=Decimal(0),
                    limit_price=originating_event.limit_price,
                    average_fill_price=None,
                    commission=None,
                    commission_asset=None,
                    timestamp_exchange=None,  # API response might contain a timestamp?
                    error_message=None,
                )
                # Using asyncio.create_task for fire-and-forget publishing
                asyncio.create_task(self.pubsub.publish(report))
                self.logger.debug(
                    f"Published NEW ExecutionReport for {cl_ord_id} / {kraken_order_id}",
                    source_module=self.__class__.__name__,
                )

                # Start monitoring the order status
                self._start_order_monitoring(cl_ord_id, kraken_order_id, originating_event)

            else:
                # This case indicates success HTTP status but unexpected result
                # format
                error_msg = "AddOrder response missing or invalid 'txid' field."
                self.logger.error(
                    f"{error_msg} cl_ord_id: {cl_ord_id}. Response: {result}",
                    source_module=self.__class__.__name__,
                )
                await self._publish_error_execution_report(
                    originating_event, error_msg, cl_ord_id
                )

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

    async def _connect_websocket(self) -> None:
        pass  # Placeholder

    async def _handle_websocket_message(self, message: Dict[str, Any]) -> None:
        pass  # Placeholder

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancels an open order on the exchange."""
        self.logger.info(
            f"Attempting to cancel order {exchange_order_id}",
            source_module=self.__class__.__name__,
        )
        uri_path = "/0/private/CancelOrder"
        params = {"txid": exchange_order_id}

        result = await self._make_private_request_with_retry(uri_path, params)

        if not result or result.get("error"):
            self.logger.error(
                f"Failed to cancel order {exchange_order_id}: "
                f"{result.get('error', 'Unknown cancel error')}",
                source_module=self.__class__.__name__,
            )
            return False

        # Check response - successful cancellation might have count > 0
        count = result.get("result", {}).get("count", 0)
        if count > 0:
            self.logger.info(
                (
                    f"Successfully initiated cancellation for order {exchange_order_id}. "
                    f"Count: {count}"
                ),
                source_module=self.__class__.__name__,
            )
            # Note: The status monitor will pick up the 'canceled' status and publish a report
            return True
        else:
            # Order might have already been closed/canceled
            self.logger.warning(
                f"Cancellation request for {exchange_order_id} returned count 0. "
                f"Order might already be in terminal state.",
                source_module=self.__class__.__name__,
            )
            return False

    def _start_order_monitoring(
        self, cl_ord_id: str, kraken_order_id: str, originating_event: TradeSignalApprovedEvent
    ) -> None:
        """Starts monitoring tasks for a newly placed order."""
        # Start status monitoring
        monitor_task = asyncio.create_task(
            self._monitor_order_status(kraken_order_id, cl_ord_id, originating_event.signal_id)
        )
        # Store task reference for cancellation on stop
        self._order_monitoring_tasks[kraken_order_id] = monitor_task

        # For limit orders, also set up timeout monitoring
        if originating_event.order_type.upper() == "LIMIT":
            timeout_s = self.config.get_float(
                "order.limit_order_timeout_s", 300.0
            )  # 5 mins default
            if timeout_s > 0:
                self.logger.info(
                    f"Scheduling timeout check for limit order {kraken_order_id} in {timeout_s}s.",
                    source_module=self.__class__.__name__,
                )
                asyncio.create_task(
                    self._monitor_limit_order_timeout(kraken_order_id, cl_ord_id, timeout_s)
                )

    async def _query_order_details(self, exchange_order_id: str) -> Optional[Dict[str, Any]]:
        """Queries the exchange for order details with retry logic."""
        uri_path = "/0/private/QueryOrders"
        params = {"txid": exchange_order_id, "trades": "true"}  # Include trade info
        query_result = await self._make_private_request_with_retry(uri_path, params)

        if not query_result or query_result.get("error"):
            error_str = str(query_result.get("error", "Unknown query error"))
            self.logger.error(
                f"Error querying order {exchange_order_id}: {error_str}",
                source_module=self.__class__.__name__,
            )
            if "EOrder:Unknown order" in error_str:
                self.logger.error(
                    f"Order {exchange_order_id} not found. Stopping monitoring for this reason.",
                    source_module=self.__class__.__name__,
                )
            return None

        result_field = query_result.get("result")
        if not isinstance(result_field, dict):
            self.logger.error(
                (f"QueryOrders response for {exchange_order_id} missing 'result' dict "
                 f"or is wrong type: {result_field}"),
                source_module=self.__class__.__name__,
            )
            return None

        order_data_any = result_field.get(exchange_order_id)
        if order_data_any is None:
            self.logger.warning(
                (f"Order {exchange_order_id} not found in QueryOrders result's main dict. "
                 f"Retrying."),
                source_module=self.__class__.__name__,
            )
            return None

        if not isinstance(order_data_any, dict):
            self.logger.error(
                f"Order data for {exchange_order_id} is not a dict: {order_data_any}",
                source_module=self.__class__.__name__,
            )
            return None

        return order_data_any  # cast(Dict[str, Any], order_data_any)

    async def _parse_order_data(
        self, order_data: Dict[str, Any], exchange_order_id: str
    ) -> Optional[Tuple[str, Decimal, Optional[Decimal], Optional[Decimal]]]:
        """Parses relevant fields from the raw order data from the exchange."""
        try:
            current_status = order_data.get("status")
            if not isinstance(current_status, str):
                self.logger.error(
                    f"Order {exchange_order_id} has invalid or missing status: {current_status}"
                )
                return None

            current_filled_qty_str = order_data.get("vol_exec", "0")
            avg_fill_price_str = order_data.get("price")  # Average price for filled portion
            fee_str = order_data.get("fee")

            current_filled_qty = Decimal(current_filled_qty_str)
            avg_fill_price = Decimal(avg_fill_price_str) if avg_fill_price_str else None
            commission = Decimal(fee_str) if fee_str else None
            return current_status, current_filled_qty, avg_fill_price, commission
        except Exception as e:
            self.logger.error(
                (
                    f"Error parsing numeric data for order {exchange_order_id}: {e}. "
                    f"Data: {order_data}"
                ),
                source_module=self.__class__.__name__,
            )
            return None

    async def _handle_order_status_change(
        self,
        exchange_order_id: str,
        client_order_id: str,
        signal_id: Optional[UUID],
        order_data: Dict[str, Any],
        current_status: str,
        current_filled_qty: Decimal,
        avg_fill_price: Optional[Decimal],
        commission: Optional[Decimal],
    ) -> None:
        """Publishes an execution report when order status or fill quantity changes."""
        self.logger.info(
            f"Status change for {exchange_order_id}: Status='{current_status}', "
            f"Filled={current_filled_qty}. Publishing report.",
            source_module=self.__class__.__name__,
        )
        await self._publish_status_execution_report(
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            signal_id=signal_id,
            order_data=order_data,  # Pass raw data for easier field access
            current_status=current_status,
            current_filled_qty=current_filled_qty,
            avg_fill_price=avg_fill_price,
            commission=commission,
        )

    async def _handle_sl_tp_for_closed_order(
        self,
        exchange_order_id: str,
        client_order_id: str,
        signal_id: Optional[UUID],
        current_filled_qty: Decimal,
    ) -> None:
        """Handles SL/TP order placement if an entry order is fully filled."""
        if signal_id is None:
            return

        # Check if this is an entry order (not an SL/TP order itself)
        is_entry_order = not (
            client_order_id.startswith("gf-sl-") or client_order_id.startswith("gf-tp-")
        )

        if is_entry_order and not await self._has_sl_tp_been_placed(signal_id):
            try:
                original_event = await self._get_originating_signal_event(signal_id)
                if original_event and (original_event.sl_price or original_event.tp_price):
                    self.logger.info(
                        f"Order {exchange_order_id} fully filled. "
                        f"Triggering SL/TP placement for signal {signal_id}.",
                        source_module=self.__class__.__name__,
                    )
                    asyncio.create_task(
                        self._handle_sl_tp_orders(
                            original_event, exchange_order_id, current_filled_qty
                        )
                    )
                else:
                    self.logger.info(
                        f"Order {exchange_order_id} fully filled, "
                        f"but no SL/TP prices found for signal {signal_id}.",
                        source_module=self.__class__.__name__,
                    )
                    # Still mark as processed to avoid repeated checks
                    await self._mark_sl_tp_as_placed(signal_id)
            except Exception as e:
                self.logger.error(
                    f"Error in SL/TP handling for {exchange_order_id}: {e}",
                    source_module=self.__class__.__name__,
                    exc_info=True,
                )

    async def _monitor_order_status(
        self, exchange_order_id: str, client_order_id: str, signal_id: Optional[UUID]
    ) -> None:
        """Monitors the status of a specific order via polling."""
        self._source_module = self.__class__.__name__  # Ensure source_module is set
        self.logger.info(
            f"Starting status monitoring for order {exchange_order_id} (cl={client_order_id})",
            source_module=self._source_module,
        )

        poll_interval = self.config.get_float("order.status_poll_interval_s", 5.0)
        max_poll_duration = self.config.get_float(
            "order.max_poll_duration_s", 3600.0
        )  # 1 hour default
        start_time = time.time()
        last_known_status: Optional[str] = "NEW"
        last_known_filled_qty: Decimal = Decimal(0)

        while time.time() - start_time < max_poll_duration:
            await asyncio.sleep(poll_interval)

            order_details_result = await self._query_order_details(exchange_order_id)

            if order_details_result is None:  # Fatal error querying, stop monitoring
                break
            if order_details_result is False:  # Non-fatal error, continue polling
                continue

            order_data = order_details_result
            parsed_data = await self._parse_order_data(order_data, exchange_order_id)
            if not parsed_data:
                continue  # Error parsing, skip this update

            current_status, current_filled_qty, avg_fill_price, commission = parsed_data

            status_changed = current_status != last_known_status
            fill_increased = current_filled_qty > last_known_filled_qty

            if status_changed or fill_increased:
                await self._handle_order_status_change(
                    exchange_order_id,
                    client_order_id,
                    signal_id,
                    order_data,
                    current_status,
                    current_filled_qty,
                    avg_fill_price,
                    commission,
                )
                last_known_status = current_status
                last_known_filled_qty = current_filled_qty

                if current_status in ["closed", "canceled", "expired"]:
                    await self._handle_sl_tp_for_closed_order(
                        exchange_order_id, client_order_id, signal_id, current_filled_qty
                    )

            if current_status in ["closed", "canceled", "expired"]:
                self.logger.info(
                    f"Order {exchange_order_id} reached terminal state '{current_status}'. "
                    f"Stopping monitoring.",
                    source_module=self._source_module,
                )
                break
        else:  # Loop finished due to timeout
            self.logger.warning(
                (
                    f"Stopped monitoring order {exchange_order_id} after timeout "
                    f"({max_poll_duration}s). Last status: {last_known_status}"
                ),
                source_module=self._source_module,
            )

        self._order_monitoring_tasks.pop(exchange_order_id, None)

    async def _get_originating_signal_event(
        self, signal_id: Optional[UUID]
    ) -> Optional[TradeSignalApprovedEvent]:
        """
        Retrieves the original signal event that led to an order.
        In a full implementation, this would fetch from an event store or cache.
        For now, this is a placeholder that returns None.
        """
        # TODO: Implement event retrieval from cache or storage
        # For MVP, we might need to store original events in memory
        self.logger.warning(
            (
                f"_get_originating_signal_event not fully implemented. "
                f"Unable to retrieve event for signal {signal_id}"
            ),
            source_module=self.__class__.__name__,
        )
        return None

    async def _publish_error_execution_report(
        self,
        event: TradeSignalApprovedEvent,
        error_message: str,
        cl_ord_id: Optional[str],
        exchange_order_id: Optional[str] = None,
    ) -> None:
        """Helper to publish an ExecutionReportEvent for a failed/rejected order."""
        effective_exchange_order_id = (
            exchange_order_id if exchange_order_id is not None else "NO_EXCHANGE_ID"
        )
        effective_cl_ord_id = cl_ord_id if cl_ord_id else f"internal_{event.signal_id}_error"
        report = ExecutionReportEvent(
            source_module=self.__class__.__name__,
            event_id=UUID(int=int(time.time() * 1000000)),
            timestamp=datetime.utcnow(),
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
        asyncio.create_task(self.pubsub.publish(report))
        self.logger.debug(
            (f"Published REJECTED/ERROR ExecutionReport "
             f"for signal {event.signal_id}, cl_ord_id: {cl_ord_id}"),
            source_module=self.__class__.__name__,
        )

    async def _publish_status_execution_report(
        self,
        exchange_order_id: str,
        client_order_id: str,
        signal_id: Optional[UUID],
        order_data: Dict,  # In Python 3.8, Dict is from typing. In 3.9+ can use dict
        current_status: str,
        current_filled_qty: Decimal,
        avg_fill_price: Optional[Decimal],
        commission: Optional[Decimal],
    ) -> None:
        """Helper to publish ExecutionReportEvent based on polled status."""
        try:
            # Extract necessary fields from order_data (Kraken specific)
            descr = order_data.get("descr", {})
            order_type_str = descr.get("ordertype")
            side_str = descr.get("type")
            pair = descr.get("pair")  # Kraken pair name

            # Map pair back to internal name
            internal_pair_nullable = self._map_kraken_pair_to_internal(pair) if pair else "UNKNOWN"
            internal_pair = (
                internal_pair_nullable if internal_pair_nullable is not None else "UNKNOWN"
            )

            raw_vol = order_data.get(
                "vol"
            )  # Use a different variable name to avoid F841 if it's an issue
            quantity_ordered_val = Decimal(raw_vol) if raw_vol else Decimal(0)
            limit_price_str_val = descr.get("price")  # Price for limit orders
            limit_price_val = Decimal(limit_price_str_val) if limit_price_str_val else None

            # Determine commission asset (e.g., quote currency of the pair)
            commission_asset = self._get_quote_currency(internal_pair) if commission else None

            exchange_timestamp_val = (
                datetime.fromtimestamp(order_data.get("opentm", time.time()), tz=timezone.utc)
                if order_data.get("opentm")
                else None
            )
            # Include reason if status is 'canceled' or 'expired'
            error_message_val = order_data.get("reason")

            if exchange_order_id is not None:
                report_exchange_id = exchange_order_id
            else:
                report_exchange_id = "NO_EXCHANGE_ID"
            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=UUID(int=int(time.time() * 1000000)),  # Generate a proper UUID
                timestamp=datetime.utcnow(),
                signal_id=signal_id,
                exchange_order_id=report_exchange_id,
                client_order_id=client_order_id,
                trading_pair=internal_pair,
                exchange=self.config.get("exchange.name", "kraken"),
                order_status=current_status.upper(),  # Standardize status
                order_type=order_type_str.upper() if order_type_str else "UNKNOWN",
                side=side_str.upper() if side_str else "UNKNOWN",
                quantity_ordered=quantity_ordered_val,
                quantity_filled=current_filled_qty,
                limit_price=limit_price_val,
                average_fill_price=avg_fill_price,
                commission=commission,
                commission_asset=commission_asset,
                timestamp_exchange=exchange_timestamp_val,
                error_message=error_message_val,
            )
            # Using asyncio.create_task for fire-and-forget publishing
            asyncio.create_task(self.pubsub.publish(report))
            self.logger.debug(
                (
                    f"Published {current_status.upper()} ExecutionReport "
                    f"for {client_order_id} / {exchange_order_id}"
                ),
                source_module=self.__class__.__name__,
            )
        except Exception as e:
            self.logger.error(
                (
                    f"Error publishing execution report for order {exchange_order_id}"
                    f" (cl_ord_id: {client_order_id}): {e}"
                ),
                source_module=self.__class__.__name__,
                exc_info=True,
            )

    def _map_kraken_pair_to_internal(self, kraken_pair: str) -> Optional[str]:
        """Maps Kraken pair name (e.g., XXBTZUSD) back to internal name (e.g., BTC/USD)."""
        for internal_name, info in self._pair_info.items():
            if (
                info.get("altname") == kraken_pair
                or info.get("wsname") == kraken_pair
                or info.get("kraken_pair_key") == kraken_pair
            ):
                return internal_name

        self.logger.warning(
            f"Could not map Kraken pair '{kraken_pair}' back to internal name.",
            source_module=self.__class__.__name__,
        )
        return None

    def _get_quote_currency(self, internal_pair: str) -> Optional[str]:
        """Gets the quote currency for an internal pair name."""
        info = self._pair_info.get(internal_pair)
        return cast(Optional[str], info.get("quote")) if info else None  # Added cast

    async def _has_sl_tp_been_placed(self, signal_id: Optional[UUID]) -> bool:
        """Checks if SL/TP orders have already been placed for a signal."""
        if signal_id is None:
            return False
        return signal_id in self._placed_sl_tp_signals

    async def _mark_sl_tp_as_placed(self, signal_id: Optional[UUID]) -> None:
        """Marks that SL/TP orders have been placed for a signal."""
        if signal_id is not None:
            self._placed_sl_tp_signals.add(signal_id)

    async def _handle_sl_tp_orders(
        self,
        originating_event: TradeSignalApprovedEvent,
        filled_order_id: str,
        filled_quantity: Decimal,
    ) -> None:
        """Places SL and/or TP orders contingent on the filled entry order."""
        self.logger.info(
            f"Handling SL/TP placement for filled order {filled_order_id} "
            f"(Signal: {originating_event.signal_id})",
            source_module=self.__class__.__name__,
        )

        kraken_pair_name = self._get_kraken_pair_name(originating_event.trading_pair)
        if not kraken_pair_name:
            return  # Error logged in helper

        # Determine side for SL/TP (opposite of entry)
        exit_side = "sell" if originating_event.side.upper() == "BUY" else "buy"

        # Place Stop Loss Order
        if originating_event.sl_price:
            sl_params = self._prepare_contingent_order_params(
                pair=kraken_pair_name,
                side=exit_side,
                order_type="stop-loss",  # Could be stop-loss-limit with price2
                price=originating_event.sl_price,  # Stop price
                volume=filled_quantity,
                pair_info=self._pair_info.get(originating_event.trading_pair),
                signal_id=originating_event.signal_id,
                contingent_type="SL",
            )

            if sl_params:
                sl_cl_ord_id = (
                    f"gf-sl-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                )
                sl_params["cl_ord_id"] = sl_cl_ord_id
                sl_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    f"Placing SL order for signal {originating_event.signal_id} "
                    f"with cl_ord_id {sl_cl_ord_id}",
                    source_module=self.__class__.__name__,
                )

                sl_result = await self._make_private_request_with_retry(
                    "/0/private/AddOrder", sl_params
                )
                # Handle SL order placement response (publish report, start monitoring)
                await self._handle_add_order_response(sl_result, originating_event, sl_cl_ord_id)

        # Place Take Profit Order
        if originating_event.tp_price:
            tp_params = self._prepare_contingent_order_params(
                pair=kraken_pair_name,
                side=exit_side,
                order_type="take-profit",  # Could be take-profit-limit with price2
                price=originating_event.tp_price,  # Take profit price
                volume=filled_quantity,
                pair_info=self._pair_info.get(originating_event.trading_pair),
                signal_id=originating_event.signal_id,
                contingent_type="TP",
            )

            if tp_params:
                tp_cl_ord_id = (
                    f"gf-tp-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                )
                tp_params["cl_ord_id"] = tp_cl_ord_id
                tp_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    f"Placing TP order for signal {originating_event.signal_id} "
                    f"with cl_ord_id {tp_cl_ord_id}",
                    source_module=self.__class__.__name__,
                )

                tp_result = await self._make_private_request_with_retry(
                    "/0/private/AddOrder", tp_params
                )
                # Handle TP order placement response (publish report, start monitoring)
                await self._handle_add_order_response(tp_result, originating_event, tp_cl_ord_id)

        # Mark SL/TP as placed for this signal
        await self._mark_sl_tp_as_placed(originating_event.signal_id)

    def _prepare_contingent_order_params(
        self,
        pair: str,
        side: str,
        order_type: str,
        price: Decimal,
        volume: Decimal,
        pair_info: Optional[Dict],
        signal_id: UUID,
        contingent_type: str,
        price2: Optional[Decimal] = None,
    ) -> Optional[Dict[str, Any]]:
        """Helper to prepare parameters for SL/TP orders, including validation."""
        params = {"pair": pair, "type": side, "ordertype": order_type}

        if not pair_info:
            self.logger.error(
                f"Missing pair_info for contingent order {contingent_type} (Signal: {signal_id})",
                source_module=self.__class__.__name__,
            )
            return None

        # Validate and format volume
        lot_decimals = pair_info.get("lot_decimals")
        if lot_decimals is None:  # Basic check
            self.logger.error(
                (
                    f"Missing lot_decimals for contingent order {contingent_type} "
                    f"(Signal: {signal_id})"
                ),
                source_module=self.__class__.__name__,
            )
            return None

        try:
            params["volume"] = self._format_decimal(volume, lot_decimals)
        except Exception as e:
            self.logger.error(
                (
                    f"Error formatting volume for contingent order {contingent_type} "
                    f"(Signal: {signal_id}): "
                    f"{e}"
                ),
                source_module=self.__class__.__name__,
            )
            return None

        # Validate and format price(s)
        pair_decimals = pair_info.get("pair_decimals")
        if pair_decimals is None:
            self.logger.error(
                (
                    f"Missing pair_decimals for contingent order {contingent_type} "
                    f"(Signal: {signal_id})"
                ),
                source_module=self.__class__.__name__,
            )
            return None

        try:
            params["price"] = self._format_decimal(price, pair_decimals)
            if price2 is not None:
                params["price2"] = self._format_decimal(price2, pair_decimals)
        except Exception as e:
            self.logger.error(
                (
                    f"Error formatting price for contingent order {contingent_type} "
                    f"(Signal: {signal_id}): "
                    f"{e}"
                ),
                source_module=self.__class__.__name__,
            )
            return None

        # Add other necessary parameters (e.g., timeinforce if needed)
        return params

    def _get_kraken_pair_name(self, internal_pair: str) -> Optional[str]:
        """Helper to get the Kraken pair name from stored info."""
        info = self._pair_info.get(internal_pair)
        name = info.get("altname") if info else None
        if not name:
            self.logger.error(
                f"Could not find Kraken pair name for internal pair '{internal_pair}'",
                source_module=self.__class__.__name__,
            )
        return name

    async def _monitor_limit_order_timeout(
        self, exchange_order_id: str, client_order_id: str, timeout_seconds: float
    ) -> None:
        """Checks if a limit order is filled after a timeout and cancels if not."""
        await asyncio.sleep(timeout_seconds)
        self.logger.info(
            f"Timeout reached for limit order {exchange_order_id}. Checking status.",
            source_module=self.__class__.__name__,
        )

        uri_path = "/0/private/QueryOrders"
        params = {"txid": exchange_order_id}
        query_result = await self._make_private_request_with_retry(uri_path, params)

        if not query_result or query_result.get("error"):
            self.logger.error(
                f"Error querying order {exchange_order_id} for timeout check: "
                f"{query_result.get('error', 'Unknown query error')}",
                source_module=self.__class__.__name__,
            )
            return  # Cannot determine status, don't cancel arbitrarily

        order_data = query_result.get("result", {}).get(exchange_order_id)
        if not order_data:
            self.logger.warning(
                f"Order {exchange_order_id} not found during timeout check "
                f"(already closed/canceled?).",
                source_module=self.__class__.__name__,
            )
            return  # Order likely already closed or canceled

        status = order_data.get("status")
        if status in ["open", "pending"]:
            self.logger.warning(
                f"Limit order {exchange_order_id} still '{status}' after "
                f"{timeout_seconds}s timeout. Attempting cancellation.",
                source_module=self.__class__.__name__,
            )
            # Call cancel_order method
            cancel_success = await self.cancel_order(exchange_order_id)
            if not cancel_success:
                self.logger.error(
                    f"Failed to cancel timed-out limit order {exchange_order_id}.",
                    source_module=self.__class__.__name__,
                )
            # The cancel_order method should publish the CANCELED report
        else:
            self.logger.info(
                f"Limit order {exchange_order_id} already in terminal state "
                f"'{status}' during timeout check.",
                source_module=self.__class__.__name__,
            )
