"""Exchange execution handler for interacting with the Kraken API to execute trades."""

# Execution Handler Module

import asyncio
import base64
import binascii  # Add missing import for binascii
from collections.abc import Coroutine
from dataclasses import dataclass  # Add dataclass import
from datetime import datetime, timezone  # Modified import
from decimal import Decimal
import hashlib
import hmac
import random  # Add import for random (needed for jitter)
import time

# Added Callable, Coroutine
from typing import Any, Optional, cast

from collections.abc import Callable
import urllib.parse
from uuid import UUID

import aiohttp

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import EventType, ExecutionReportEvent, TradeSignalApprovedEvent
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.monitoring_service import MonitoringService

# TODO: Replace debug print with proper logging
print("Execution Handler Loaded")

KRAKEN_API_URL = "https://api.kraken.com"


class InvalidAPICredentialFormatError(ValueError):
    """Raised when an API credential is not in the expected format."""

    def __init__(self, message: str = "API secret must be base64 encoded.", *args: object) -> None:
        super().__init__(message, *args)



@dataclass
class ContingentOrderParamsRequest:
    """Parameters for preparing a contingent order (SL/TP)."""

    pair_name: str  # Kraken pair name (e.g., XXBTZUSD)
    order_side: str  # "buy" or "sell"
    contingent_order_type: str  # e.g., "stop-loss", "take-profit"
    trigger_price: Decimal  # The price at which the order triggers
    volume: Decimal
    pair_details: dict | None  # Exchange-provided info for the pair
    originating_signal_id: UUID
    log_marker: str  # "SL" or "TP" for logging
    limit_price: Decimal | None = None  # For stop-loss-limit / take-profit-limit


@dataclass
class OrderStatusReportParameters:
    """Parameters for handling and reporting order status."""

    exchange_order_id: str
    client_order_id: str
    signal_id: UUID | None
    order_data: dict[str, Any]
    current_status: str
    current_filled_qty: Decimal
    avg_fill_price: Decimal | None
    commission: Decimal | None


class RateLimitTracker:
    """Tracks and enforces API rate limits to prevent exceeding exchange limits."""

    def __init__(
        self, config: ConfigManager, logger_service: LoggerService
    ) -> None:
        """Initialize the rate limit tracker with configuration and logger."""
        self.config = config
        self.logger: LoggerService = logger_service

        # Configure rate limits based on tier/API key level
        # These should come from configuration
        self.private_calls_per_second = self.config.get_int(
            "exchange.rate_limit.private_calls_per_second", 1
        )
        self.public_calls_per_second = self.config.get_int(
            "exchange.rate_limit.public_calls_per_second", 1
        )

        # Tracking timestamps of recent calls
        self._private_call_timestamps: list[float] = []
        self._public_call_timestamps: list[float] = []

        # Window size in seconds for tracking
        self.window_size = 1.0  # 1 second window

        self._source_module = self.__class__.__name__

    async def wait_for_private_capacity(self) -> None:
        """
        Wait until there's capacity to make a private API call.

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
        Wait until there's capacity to make a public API call.

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


class ExecutionHandler:
    """
    Handle interaction with the exchange API (Kraken) to place, manage, and monitor orders.

    Processes approved trade signals, translates them to exchange-specific parameters,
    places orders, and monitors their execution.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        pubsub_manager: PubSubManager,
        monitoring_service: MonitoringService,
        logger_service: LoggerService,
    ) -> None:
        """Initialize the execution handler with required services and configuration."""
        self.logger = logger_service
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.monitoring = monitoring_service

        self.api_key = self.config.get("kraken.api_key", default=None)
        self.api_secret = self.config.get("kraken.secret_key", default=None)
        self.api_base_url = self.config.get("exchange.api_url", KRAKEN_API_URL)

        if not self.api_key or not self.api_secret:
            self.logger.critical(
                ("Kraken API Key or Secret Key not configured. ExecutionHandler cannot function."),
                source_module=self.__class__.__name__,
            )

        self._session: aiohttp.ClientSession | None = None
        # TODO: Add state for managing WebSocket connection if used
        # TODO: Add mapping for internal IDs to exchange IDs (cl_ord_id ->
        # txid)
        self._order_map: dict[str, str] = {}  # cl_ord_id -> txid
        # Internal pair -> Kraken details
        self._pair_info: dict[str, dict[str, Any]] = {}
        # Add type hint for the handler attribute
        self._trade_signal_handler: None | (
            Callable[[TradeSignalApprovedEvent], Coroutine[Any, Any, None]]
        ) = None

        # Store active monitoring tasks
        self._order_monitoring_tasks: dict[str, asyncio.Task] = {}  # txid -> Task

        # Track signals that have had SL/TP orders placed
        self._placed_sl_tp_signals: set[UUID] = set()

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

        self._background_tasks: set[asyncio.Task] = set()

        self.logger.info(
            "ExecutionHandler initialized.",
            source_module=self.__class__.__name__,
        )

    async def start(self) -> None:
        """Initialize API client session, load exchange info, and subscribe to events."""
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

        # Store the handler for unsubscribing
        self._trade_signal_handler = self.handle_trade_signal_approved
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler)
        self.logger.info(
            "ExecutionHandler started. Subscribed to TRADE_SIGNAL_APPROVED.",
            source_module=self.__class__.__name__,
        )
        # TODO: Implement WebSocket connection logic here if used for MVP

    async def stop(self) -> None:
        """Close API client session and potentially cancel orders."""
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
            except Exception:
                self.logger.exception("Error unsubscribing")

        # Cancel ongoing monitoring tasks
        try:
            for task_id, task in list(self._order_monitoring_tasks.items()):
                if not task.done():
                    task.cancel()
                    self.logger.info("Cancelled monitoring task for order %s", task_id)
            self._order_monitoring_tasks.clear()
            self.logger.info("All order monitoring tasks cancelled.")
        except Exception:
            self.logger.exception(
                "Error cancelling monitoring tasks",
                source_module=self.__class__.__name__,
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
    ) -> dict[str, Any] | None:
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

                async with self._session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    response.raise_for_status()
                    data: dict[str, Any] = await response.json()

                    if data.get("error"):
                        error_str = str(data["error"])
                        if self._is_retryable_error(error_str) and attempt < max_retries:
                            delay = min(base_delay * (2**attempt), 30.0)
                            jitter = random.uniform(0, delay * 0.1)
                            total_delay = delay + jitter
                            self.logger.warning(
                                "Retryable API error for %s: %s. "
                                "Retrying in %.2fs (Attempt %d/%d)",
                                url, error_str, total_delay, attempt + 1, max_retries + 1,
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
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        (
                            "Error during public request to %s: %s. "
                            "Retrying in %.2fs "
                            "(Attempt %d/%d)"
                        ),
                        url, e, total_delay, attempt + 1, max_retries + 1,
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    continue
                self.logger.exception(
                    "Failed to make public request to %s after %d attempts. Last error recorded.",
                    url, max_retries + 1,
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
            (
                "Failed to make public request to %s "
                "after %d attempts. "
                "Last error: %s"
            ),
            url, max_retries + 1, last_error_message,
            source_module=self.__class__.__name__,
        )
        return None

    async def _load_exchange_info(self) -> None:
        """Fetch and store tradable asset pair information from Kraken."""
        uri_path = "/0/public/AssetPairs"
        url = self.api_base_url + uri_path
        self.logger.info(
            "Loading exchange asset pair info from %s...",
            url,
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
            self.logger.exception(
                "Unexpected error loading exchange info.",
                source_module=self.__class__.__name__,
            )

    def _validate_session(self) -> bool:
        """Validate that the AIOHTTP session is available."""
        if not self._session or self._session.closed:
            self.logger.error(
                "Cannot load exchange info: AIOHTTP session is not available.",
                source_module=self.__class__.__name__,
            )
            return False
        return True

    async def _process_asset_pairs(self, result: dict) -> None:
        """Process and store asset pairs data."""
        loaded_count = 0
        internal_pairs = self.config.get_list("trading.pairs", [])

        if not internal_pairs:
            self.logger.warning(
                (
                    "No trading pairs defined in config [trading.pairs]. Cannot map exchange info."
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
        """Process a single trading pair and store its information."""
        kraken_altname = internal_pair_name.replace("/", "")
        kraken_key = kraken_pair_map.get(kraken_altname)

        if not kraken_key or kraken_key not in result:
            self.logger.warning(
                (
                    "Could not find matching AssetPairs info for "
                    "configured pair: %s"
                ),
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
        self.logger.debug(
            "Loaded info for %s",
            internal_pair_name,
            source_module=self.__class__.__name__,
        )
        return True

    def _log_loading_results(self, loaded_count: int, total_pairs: int) -> None:
        """Log the results of loading asset pairs."""
        self.logger.info(
            (
                "Successfully loaded info for %s asset pairs "
                "out of %s configured."
            ),
            loaded_count, total_pairs,
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
        data: dict[str, Any],
        nonce: int,
    ) -> str:
        """Generate the API-Sign header required by Kraken private endpoints."""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(nonce) + postdata).encode()
        message = uri_path.encode() + hashlib.sha256(encoded).digest()

        try:
            secret_decoded = base64.b64decode(self.api_secret)
        except binascii.Error as e:
            self.logger.exception(
                "Invalid base64 API secret",
                source_module=self.__class__.__name__,
            )
            raise InvalidAPICredentialFormatError from e

        mac = hmac.new(secret_decoded, message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def _format_decimal(self, value: Decimal, precision: int) -> str:
        """Format a Decimal value to a string with a specific precision."""
        # Use quantization to set the number of decimal places
        # Ensure it rounds correctly, default rounding is ROUND_HALF_EVEN
        quantizer = Decimal("1e-" + str(precision))
        return str(value.quantize(quantizer))

    async def _make_private_request(
        self,
        uri_path: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
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
        request_data = data.copy()  # Avoid modifying the original dict
        request_data["nonce"] = nonce
        api_sign: str

        try:
            api_sign = self._generate_kraken_signature(uri_path, request_data, nonce)
        except ValueError: # Raised by _generate_kraken_signature if API secret is invalid
            # Error already logged by _generate_kraken_signature
            response_data = {"error": ["EGeneral:InternalError - Signature generation failed"]}
            return response_data

        headers = {
            "API-Key": self.api_key,
            "API-Sign": api_sign,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        }
        url = self.api_base_url + uri_path
        timeout = aiohttp.ClientTimeout(
            total=self.config.get("exchange.request_timeout_seconds", 10)
        )

        try:
            self.logger.debug(
                "Sending private request to %s with data: %s",
                url, request_data,
                source_module=self.__class__.__name__,
            )
            async with self._session.post(
                url, headers=headers, data=request_data, timeout=timeout
            ) as response:
                response.raise_for_status()  # Raise exception for bad status codes (4xx, 5xx)
                result: dict[str, Any] = await response.json()
                self.logger.debug(
                    "Received response from %s: %s",
                    url, result,
                    source_module=self.__class__.__name__,
                )
                # Check for API-level errors within the JSON response
                if result.get("error"):
                    self.logger.error(
                        "Kraken API error for %s: %s",
                        uri_path, result["error"],
                        source_module=self.__class__.__name__,
                    )
                response_data = result # Store result, whether success or API error

        except aiohttp.ClientResponseError as e:
            error_body = await response.text() # response object should be available here
            self.logger.exception(
                (
                    "HTTP Error: %s %s for %s. Body: %s"
                ),
                e.status, e.message, e.request_info.url, error_body[:500],
                source_module=self.__class__.__name__,
            )
            response_data = {"error": [f"EGeneral:HTTPError - {e.status}: {e.message}"]}
        except aiohttp.ClientConnectionError as e:
            self.logger.exception(
                "Connection Error to %s: %s",
                url,
                source_module=self.__class__.__name__
            )
            response_data = {"error": [f"EGeneral:ConnectionError - {e!s}"]}
        except TimeoutError:
            self.logger.exception(
                "Request Timeout for %s: %s",
                url,
                source_module=self.__class__.__name__,
            )
            response_data = {"error": ["EGeneral:Timeout"]}
        except Exception: # Catch-all for unexpected errors
            self.logger.exception(
                "Unexpected error during private API request to %s: %s",
                url,
                source_module=self.__class__.__name__,
            )
            response_data = {"error": ["EGeneral:Unexpected - Unknown error during request"]}

        return response_data

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
        self, uri_path: str, data: dict[str, Any], max_retries: int = 3
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
                        uri_path, attempt + 1,
                        source_module=self.__class__.__name__
                    )
                    final_result = current_result
                    break  # Exit loop on success

                # API call returned an error
                error_str = str(current_result.get("error", "Unknown API error"))
                last_error_info_str = f"APIError: {error_str}"

                if self._is_retryable_error(error_str) and attempt < max_retries:
                    delay = min(base_delay * (2**attempt), 30.0)  # Cap delay at 30s
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        "Retryable API error for %s: %s. Retrying in %.2fs (Attempt %d/%d)",
                        uri_path, error_str, total_delay, attempt + 1, max_retries + 1,
                        source_module=self.__class__.__name__,
                    )
                    await asyncio.sleep(total_delay)
                    # Loop continues to next attempt
                else:
                    # Permanent API error or max retries for this API error
                    self.logger.error(
                        "Permanent API error for %s or max retries for API error: %s",
                        uri_path, error_str,
                        source_module=self.__class__.__name__
                    )
                    final_result = current_result  # Store the error dict
                    break  # Exit loop

            except Exception as e:
                last_error_info_str = f"Exception: {type(e).__name__} - {e!s}"
                self.logger.exception(
                    "Exception during API request to %s (Attempt %d/%d): %s",
                    uri_path, attempt + 1, max_retries + 1,
                    source_module=self.__class__.__name__
                )

                if (
                    isinstance(e, (aiohttp.ClientConnectionError, asyncio.TimeoutError))
                    and attempt < max_retries
                ):
                    delay = min(base_delay * (2**attempt), 30.0)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    self.logger.warning(
                        "Network error for %s (%s). Retrying in %.2fs (Attempt %d/%d): %s",
                        uri_path, type(e).__name__, total_delay, attempt + 1, max_retries + 1, e,
                        source_module=self.__class__.__name__
                    )
                    await asyncio.sleep(total_delay)
                    # Loop continues to next attempt
                else:
                    # Non-retryable exception or max retries for this exception type
                    self.logger.exception(
                        "Non-retryable exception or max retries for network error. "
                        "URI: %s. Error: %s",
                        uri_path,
                        source_module=self.__class__.__name__
                    )
                    final_result = {
                        "error": [
                            f"EGeneral:RequestException - {last_error_info_str}"
                        ]
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
            uri_path, max_retries + 1, last_error_info_str,
            source_module=self.__class__.__name__,
        )
        return {
            "error": [
                f"EGeneral:MaxRetriesExceeded - Last known error: {last_error_info_str}"
            ]
        }

    async def handle_trade_signal_approved(self, event: TradeSignalApprovedEvent) -> None:
        """
        Process an approved trade signal event.

        Check HALT status, translate signal to API parameters, place order, and handle response.
        """
        self.logger.info(
            "Received approved trade signal: %s",
            event.signal_id,
            source_module=self.__class__.__name__,
        )

        # 1. Check HALT status FIRST
        if self.monitoring.is_halted():
            error_msg = "Execution blocked: System HALTED"
            self.logger.critical(
                "%s. Discarding approved signal: %s (%s %s %s)",
                error_msg, event.signal_id, event.trading_pair, event.side, event.quantity,
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
                )
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
            event.signal_id, cl_ord_id,
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
    ) -> dict[str, Any] | None:
        """
        Translate internal signal format to Kraken API parameters.

        Includes validation of the parameters against exchange requirements.
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
            "Translated signal %s to Kraken params: %s",
            event.signal_id, params,
            source_module=self.__class__.__name__,
        )
        return params

    def _get_and_validate_pair_info(
        self,
        internal_pair: str,
        signal_id: UUID,
    ) -> dict[str, Any] | None:
        """Get and validate trading pair information."""
        # Convert UUID to string for logging
        signal_id_str = str(signal_id)

        pair_info = self._pair_info.get(internal_pair)
        if not pair_info:
            self.logger.error(
                "No exchange info found for pair %s. Cannot translate signal %s.",
                internal_pair, signal_id_str,
                source_module=self.__class__.__name__,
            )
            return None

        if pair_info.get("status") != "online":
            self.logger.error(
                "Pair %s is not online (status: %s). Cannot place order for signal %s.",
                internal_pair, pair_info.get("status"), signal_id_str,
                source_module=self.__class__.__name__,
            )
            return None

        return pair_info

    def _get_and_validate_pair_name(
        self,
        internal_pair: str,
        pair_info: dict[str, Any],
        signal_id: UUID,
    ) -> str | None:
        """Get and validate the Kraken pair name."""
        # Convert UUID to string for logging
        signal_id_str = str(signal_id)

        kraken_pair_name = cast(Optional[str], pair_info.get("altname"))
        if not kraken_pair_name:
            self.logger.error(
                "Missing Kraken altname for pair %s in loaded info for signal %s.",
                internal_pair, signal_id_str,
                source_module=self.__class__.__name__,
            )
            return None
        return kraken_pair_name

    def _validate_and_set_order_side(
        self,
        params: dict[str, Any],
        event: TradeSignalApprovedEvent,
    ) -> bool:
        """Validate and set the order side parameter."""
        order_side = event.side.lower()
        if order_side not in ["buy", "sell"]:
            self.logger.error(
                "Invalid order side '%s' in signal %s.",
                event.side, event.signal_id,
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
                    event.quantity, ordermin, event.trading_pair, event.signal_id,
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
            event.order_type, event.signal_id,
            source_module=self.__class__.__name__,
        )
        return False

    def _handle_limit_order(
        self,
        params: dict[str, Any],
        event: TradeSignalApprovedEvent,
        pair_decimals: int,
    ) -> bool:
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
                "SL/TP prices in signal %s; handling deferred in MVP Handler.",
                event.signal_id,
                source_module=self.__class__.__name__,
            )

    async def _handle_add_order_response(
        self,
        result: dict[str, Any],
        originating_event: TradeSignalApprovedEvent,
        cl_ord_id: str,
    ) -> None:
        """
        Process the response from the AddOrder API call and publish initial status.

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
            await self._publish_error_execution_report(
                originating_event, "Empty API response", cl_ord_id
            )
            return

        # Check for API-level errors first
        if result.get("error"):
            # Kraken errors are usually a list of strings
            error_msg = str(result["error"])
            self.logger.error(
                "AddOrder API call failed for signal %s (cl_ord_id: %s): %s",
                originating_event.signal_id, cl_ord_id, error_msg,
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
                # Assuming single order response for now
                kraken_order_id = txids[0]
                self.logger.info(
                    "Order via API for signal %s: cl_ord_id=%s, TXID=%s, Descr=%s",
                    originating_event.signal_id, cl_ord_id, kraken_order_id, descr,
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
                task = asyncio.create_task(self.pubsub.publish(report))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                self.logger.debug(
                    "Published NEW ExecutionReport for %s / %s",
                    cl_ord_id, kraken_order_id,
                    source_module=self.__class__.__name__,
                )

                # Start monitoring the order status
                self._start_order_monitoring(cl_ord_id, kraken_order_id, originating_event)

            else:
                # This case indicates success HTTP status but unexpected result
                # format
                error_msg = "AddOrder response missing or invalid 'txid' field."
                self.logger.error(
                    "%s cl_ord_id: %s. Response: %s",
                    error_msg, cl_ord_id, result,
                    source_module=self.__class__.__name__,
                )
                await self._publish_error_execution_report(originating_event, error_msg, cl_ord_id)

        except Exception:  # Catch potential errors during response parsing
            self.logger.exception(
                "Error processing successful AddOrder response for signal %s (cl_ord_id: %s): %s",
                originating_event.signal_id, cl_ord_id,
                source_module=self.__class__.__name__,
            )
            await self._publish_error_execution_report(
                originating_event, "Internal error processing response after AddOrder", cl_ord_id
            )

    async def _connect_websocket(self) -> None:
        """
        Connect to the exchange WebSocket API.

        This is a placeholder for future WebSocket implementation.
        """
        # Placeholder

    async def _handle_websocket_message(self, message: dict[str, Any]) -> None:
        """
        Process a message received from the exchange WebSocket.

        This is a placeholder for future WebSocket implementation.
        """
        # Placeholder

    async def cancel_order(self, exchange_order_id: str) -> bool:
        """Cancel an open order on the exchange."""
        self.logger.info(
            "Attempting to cancel order %s",
            exchange_order_id,
            source_module=self.__class__.__name__,
        )
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
                exchange_order_id, error_detail,
                source_module=self.__class__.__name__,
            )
            return False

        # Check response - successful cancellation might have count > 0
        count = result.get("result", {}).get("count", 0)
        if count > 0:
            self.logger.info(
                "Successfully initiated cancellation for order %s. Count: %s",
                exchange_order_id, count,
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
        self, cl_ord_id: str, kraken_order_id: str, originating_event: TradeSignalApprovedEvent
    ) -> None:
        """Start monitoring tasks for a newly placed order."""
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
                    "Scheduling timeout check for limit order %s in %ss.",
                    kraken_order_id, timeout_s,
                    source_module=self.__class__.__name__,
                )
                limit_order_timeout_task = asyncio.create_task(
                    self._monitor_limit_order_timeout(kraken_order_id, timeout_s)
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
                exchange_order_id, error_str,
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
                "QueryOrders response for %s missing 'result' dict or is wrong type: %s",
                exchange_order_id, result_field,
                source_module=self.__class__.__name__,
            )
            return None

        order_data_any = result_field.get(exchange_order_id)
        if order_data_any is None:
            self.logger.warning(
                "Order %s not found in QueryOrders result's main dict. Retrying.",
                exchange_order_id,
                source_module=self.__class__.__name__,
            )
            return None

        if not isinstance(order_data_any, dict):
            self.logger.error(
                "Order data for %s is not a dict: %s",
                exchange_order_id, order_data_any,
                source_module=self.__class__.__name__,
            )
            return None

        return order_data_any

    async def _parse_order_data(
        self, order_data: dict[str, Any], exchange_order_id: str
    ) -> tuple[str, Decimal, Decimal | None, Decimal | None] | None:
        """Parse relevant fields from the raw order data from the exchange."""
        try:
            current_status = order_data.get("status")
            if not isinstance(current_status, str):
                self.logger.error(
                    "Order %s has invalid or missing status: %s",
                    exchange_order_id, current_status
                )
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
                exchange_order_id, order_data,
                source_module=self.__class__.__name__,
            )
            return None
        else:
            return current_status, current_filled_qty, avg_fill_price, commission

    async def _handle_order_status_change(
        self,
        params: OrderStatusReportParameters,
    ) -> None:
        """Publish an execution report when order status or fill quantity changes."""
        self.logger.info(
            "Status change for %s: Status='%s', Filled=%s. Publishing report.",
            params.exchange_order_id, params.current_status, params.current_filled_qty,
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
        is_entry_order = not (
            client_order_id.startswith(("gf-sl-", "gf-tp-"))
        )

        if is_entry_order and not await self._has_sl_tp_been_placed(signal_id):
            try:
                original_event = await self._get_originating_signal_event(signal_id)
                if original_event and (original_event.sl_price or original_event.tp_price):
                    self.logger.info(
                        "Order %s fully filled. Triggering SL/TP placement for signal %s.",
                        exchange_order_id, signal_id,
                        source_module=self.__class__.__name__,
                    )
                    sl_tp_handling_task = asyncio.create_task(
                        self._handle_sl_tp_orders(
                            original_event, exchange_order_id, current_filled_qty
                        )
                    )
                    self._background_tasks.add(sl_tp_handling_task)
                    sl_tp_handling_task.add_done_callback(self._background_tasks.discard)
                else:
                    self.logger.info(
                        "Order %s fully filled, but no SL/TP prices found for signal %s.",
                        exchange_order_id, signal_id,
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

    async def _monitor_order_status(
        self, exchange_order_id: str, client_order_id: str, signal_id: UUID | None
    ) -> None:
        """
        Monitor the status of a specific order via polling.

        Periodically check order status, publish update, and handle SL/TP order for filled order.
        """
        self._source_module = self.__class__.__name__  # Ensure source_module is set
        self.logger.info(
            "Starting status monitoring for order %s (cl=%s)",
            exchange_order_id, client_order_id,
            source_module=self._source_module,
        )

        poll_interval = self.config.get_float("order.status_poll_interval_s", 5.0)
        max_poll_duration = self.config.get_float(
            "order.max_poll_duration_s", 3600.0
        )  # 1 hour default
        start_time = time.time()
        last_known_status: str | None = "NEW"
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
                        exchange_order_id, client_order_id, signal_id, current_filled_qty
                    )

            if current_status in ["closed", "canceled", "expired"]:
                self.logger.info(
                    "Order %s reached terminal state '%s'. Stopping monitoring.",
                    exchange_order_id, current_status,
                    source_module=self._source_module,
                )
                break
        else:  # Loop finished due to timeout
            self.logger.warning(
                "Stopped monitoring order %s after timeout (%ss). Last status: %s",
                exchange_order_id, max_poll_duration, last_known_status,
                source_module=self._source_module,
            )

        self._order_monitoring_tasks.pop(exchange_order_id, None)

    async def _get_originating_signal_event(
        self, signal_id: UUID | None
    ) -> TradeSignalApprovedEvent | None:
        """
        Retrieve the original signal event that led to an order.

        In a full implementation, this would fetch from an event store or cache.
        For now, this is a placeholder that returns None.
        """
        # TODO: Implement event retrieval from cache or storage
        # For MVP, we might need to store original events in memory
        self.logger.warning(
            "_get_originating_signal_event not fully implemented. "
            "Unable to retrieve event for signal %s",
            signal_id,
            source_module=self.__class__.__name__,
        )
        return None

    async def _publish_error_execution_report(
        self,
        event: TradeSignalApprovedEvent,
        error_message: str,
        cl_ord_id: str | None,
        exchange_order_id: str | None = None,
    ) -> None:
        """Publish an ExecutionReportEvent for a failed/rejected order."""
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
        publish_task = asyncio.create_task(self.pubsub.publish(report))
        self._background_tasks.add(publish_task)
        publish_task.add_done_callback(self._background_tasks.discard)
        self.logger.debug(
            "Published REJECTED/ERROR ExecutionReport for signal %s, cl_ord_id: %s",
            event.signal_id, cl_ord_id,
            source_module=self.__class__.__name__,
        )

    async def _publish_status_execution_report(
        self,
        params: OrderStatusReportParameters,
    ) -> None:
        """Publish ExecutionReportEvent based on polled status."""
        try:
            # Extract necessary fields from order_data (Kraken specific)
            descr = params.order_data.get("descr", {})
            order_type_str = descr.get("ordertype")
            side_str = descr.get("type")
            pair = descr.get("pair")  # Kraken pair name

            # Map pair back to internal name
            internal_pair_nullable = self._map_kraken_pair_to_internal(pair) if pair else "UNKNOWN"
            internal_pair = (
                internal_pair_nullable if internal_pair_nullable is not None else "UNKNOWN"
            )

            raw_vol = params.order_data.get(
                "vol"
            )  # Use a different variable name to avoid F841 if it's an issue
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
                exchange_timestamp_val = datetime.fromtimestamp(
                    opentm, tz=timezone.utc
                )

            # Include reason if status is 'canceled' or 'expired'
            error_message_val = params.order_data.get("reason")

            if params.exchange_order_id is not None:
                report_exchange_id = params.exchange_order_id
            else:
                report_exchange_id = "NO_EXCHANGE_ID"
            report = ExecutionReportEvent(
                source_module=self.__class__.__name__,
                event_id=UUID(int=int(time.time() * 1000000)),  # Generate a proper UUID
                timestamp=datetime.utcnow(),
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
                params.current_status.upper(), params.client_order_id, params.exchange_order_id,
                source_module=self.__class__.__name__,
            )
        except Exception:
            self.logger.exception(
                "Error publishing execution report for order %s (cl_ord_id: %s): %s",
                params.exchange_order_id, params.client_order_id,
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
        return cast(Optional[str], info.get("quote")) if info else None  # Added cast

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
        """
        Place SL and/or TP orders contingent on the filled entry order.

        Creates stop-loss and take-profit orders based on the original signal parameters.
        """
        self.logger.info(
            "Handling SL/TP placement for filled order %s (Signal: %s)",
            filled_order_id, originating_event.signal_id,
            source_module=self.__class__.__name__,
        )

        kraken_pair_name = self._get_kraken_pair_name(originating_event.trading_pair)
        if not kraken_pair_name:
            return  # Error logged in helper

        # Determine side for SL/TP (opposite of entry)
        exit_side = "sell" if originating_event.side.upper() == "BUY" else "buy"
        current_pair_info = self._pair_info.get(originating_event.trading_pair)

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
                sl_cl_ord_id = (
                    f"gf-sl-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                )
                sl_params["cl_ord_id"] = sl_cl_ord_id
                sl_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    "Placing SL order for signal %s with cl_ord_id %s",
                    originating_event.signal_id, sl_cl_ord_id,
                    source_module=self.__class__.__name__,
                )

                sl_result = await self._make_private_request_with_retry(
                    "/0/private/AddOrder", sl_params
                )
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
                tp_cl_ord_id = (
                    f"gf-tp-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                )
                tp_params["cl_ord_id"] = tp_cl_ord_id
                tp_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    "Placing TP order for signal %s with cl_ord_id %s",
                    originating_event.signal_id, tp_cl_ord_id,
                    source_module=self.__class__.__name__,
                )

                tp_result = await self._make_private_request_with_retry(
                    "/0/private/AddOrder", tp_params
                )
                # Handle TP order placement response (publish report, start monitoring)
                await self._handle_add_order_response(tp_result, originating_event, tp_cl_ord_id)

        # Mark SL/TP as placed for this signal
        await self._mark_sl_tp_as_placed(originating_event.signal_id)

    async def _monitor_limit_order_timeout(
        self, exchange_order_id: str, timeout_seconds: float
    ) -> None:
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
                exchange_order_id, error_str,
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
            self.logger.warning(
                log_msg,
                source_module=self.__class__.__name__,
            )
            return  # Order likely already closed or canceled

        status = order_data.get("status")
        if status in ["open", "pending"]:
            log_msg = f"Limit order {exchange_order_id} still '{status}'"
            log_msg += f" after {timeout_seconds}s timeout. Attempting cancellation."
            self.logger.warning(
                log_msg,
                source_module=self.__class__.__name__,
            )
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
            self.logger.info(
                log_msg,
                source_module=self.__class__.__name__,
            )

    def _prepare_contingent_order_params(
        self,
        request: ContingentOrderParamsRequest,
    ) -> dict[str, Any] | None:
        """Prepare parameters for SL/TP orders, including validation."""
        params = {
            "pair": request.pair_name,
            "type": request.order_side,
            "ordertype": request.contingent_order_type,
        }

        if not request.pair_details:
            log_msg = f"Missing pair_details for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.error(
                log_msg,
                source_module=self.__class__.__name__,
            )
            return None

        # Validate and format volume
        lot_decimals = request.pair_details.get("lot_decimals")
        if lot_decimals is None:  # Basic check
            log_msg = f"Missing lot_decimals for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.error(
                log_msg,
                source_module=self.__class__.__name__,
            )
            return None

        try:
            params["volume"] = self._format_decimal(request.volume, lot_decimals)
        except Exception:
            log_msg = f"Error formatting volume for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.exception(
                log_msg,
                source_module=self.__class__.__name__,
            )
            return None

        # Validate and format price(s)
        pair_decimals = request.pair_details.get("pair_decimals")
        if pair_decimals is None:
            log_msg = f"Missing pair_decimals for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.error(
                log_msg,
                source_module=self.__class__.__name__,
            )
            return None

        try:
            params["price"] = self._format_decimal(request.trigger_price, pair_decimals)
            if request.limit_price is not None:
                params["price2"] = self._format_decimal(request.limit_price, pair_decimals)
        except Exception:
            log_msg = f"Error formatting price for contingent order {request.log_marker}"
            log_msg += f" (Signal: {request.originating_signal_id})"
            self.logger.exception(
                log_msg,
                source_module=self.__class__.__name__,
            )
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
