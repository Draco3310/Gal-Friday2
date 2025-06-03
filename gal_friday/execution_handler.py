"""Execution handler for managing order placement and lifecycle with Kraken exchange."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import aiohttp

from gal_friday.config_manager import ConfigManager

# Removed incorrect import: from gal_friday.core.errors import ExecutionHandlerAuthenticationError
from gal_friday.core.events import (
    ClosePositionCommand,
    EventType,
    ExecutionReportEvent,
    TradeSignalApprovedEvent,
)
from gal_friday.exceptions import (
    ExecutionHandlerAuthenticationError,
)
from gal_friday.execution.adapters import (
    BatchOrderRequest,
    ExecutionAdapter,
    KrakenExecutionAdapter,
    OrderRequest,
)
from gal_friday.execution.websocket_client import KrakenWebSocketClient
from gal_friday.utils.kraken_api import generate_kraken_signature

if TYPE_CHECKING:
    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.logger_service import LoggerService

# Remove hardcoded URL - will be loaded from configuration


class ExecutionHandlerAuthenticationError(ValueError):
    """Raised when an API credential is not in the expected format or other auth issue.

    Args:
        message: Custom error message. Defaults to API secret format error.
        *args: Additional arguments to pass to the parent class.
    """
    def __init__(self, message: str = "API secret must be base64 encoded.", *args: object) -> None:
        """Initialize the error with a custom message.

        Args:
            message: Custom error message. Defaults to API secret format error.
            *args: Additional arguments to pass to the parent class.
        """
        super().__init__(message, *args)


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
    pair_details: dict | None
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

    def __init__(
        self,
        config: ConfigManager,
        logger_service: LoggerService,
    ) -> None:
        """Initialize the rate limit tracker with configuration and logger."""
        self.config = config
        self.logger: LoggerService = logger_service

        # Configure rate limits based on tier/API key level
        # These should come from configuration
        self.private_calls_per_second = self.config.get_int(
            "exchange.rate_limit.private_calls_per_second",
            1,
        )
        self.public_calls_per_second = self.config.get_int(
            "exchange.rate_limit.public_calls_per_second",
            1,
        )

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


class ExecutionHandler:
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
        self.logger = logger_service
        self.config = config_manager
        self.pubsub = pubsub_manager
        self.monitoring = monitoring_service
        self.event_store = event_store

        self.api_key = self.config.get("kraken.api_key", default=None)
        self.api_secret = self.config.get("kraken.secret_key", default=None)
        self.api_base_url = self.config.get("exchange.api_url", "https://api.kraken.com")

        if not self.api_key or not self.api_secret:
            self.logger.critical(
                ("Kraken API Key or Secret Key not configured. ExecutionHandler cannot function."),
                source_module=self.__class__.__name__,
            )

        self._session: aiohttp.ClientSession | None = None

        # --- Initialize Execution Adapter Pattern (Enterprise Architecture) ---
        # Configure which exchange adapter to use
        self._exchange_name = self.config.get("execution_handler.exchange", "kraken")
        self._adapter: ExecutionAdapter | None = None

        # Initialize the appropriate adapter based on configuration
        if self._exchange_name.lower() == "kraken":
            self._adapter = KrakenExecutionAdapter(
                config=self.config,
                logger=self.logger,
            )
            self.logger.info(
                "Initialized Kraken execution adapter",
                source_module=self.__class__.__name__,
            )
        else:
            # Future: Add support for other exchanges here
            self.logger.error(
                "Unsupported exchange: %s. Currently only 'kraken' is supported.",
                self._exchange_name,
                source_module=self.__class__.__name__,
            )

        # --- Enhanced WebSocket State Management (Production Ready) ---
        self._websocket_config = self.config.get("execution_handler.websocket", {})
        self._use_websocket_for_orders = self._websocket_config.get("use_for_order_updates", False)

        if self._use_websocket_for_orders:
            # WebSocket connection state and configuration
            self._websocket_connection_state = "DISCONNECTED"  # DISCONNECTED, CONNECTING, CONNECTED, AUTHENTICATED
            self._websocket_auth_token: str | None = None
            self._websocket_connection_task: asyncio.Task | None = None
            self._subscribed_channels: set[str] = set()
            self._max_reconnect_attempts = self._websocket_config.get("max_reconnect_attempts", 5)
            self._reconnect_delay_seconds = self._websocket_config.get("reconnect_delay_seconds", 5)
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
            self.websocket_client = None

        # --- Enhanced Order ID Mapping (Bidirectional & Comprehensive) ---
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
        # Add type hint for the handler attribute
        self._trade_signal_handler: None | (
            Callable[[TradeSignalApprovedEvent], Coroutine[Any, Any, None]]
        ) = None
        self._close_position_handler: None | (
            Callable[[ClosePositionCommand], Coroutine[Any, Any, None]]
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

        # Initialize the execution adapter
        if self._adapter:
            try:
                await self._adapter.initialize()
                self.logger.info(
                    "Execution adapter initialized successfully",
                    source_module=self.__class__.__name__,
                )
            except Exception:
                self.logger.exception(
                    "Failed to initialize execution adapter",
                    source_module=self.__class__.__name__,
                )
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
                (
                    "Failed to load essential exchange pair info. "
                    "ExecutionHandler will not function correctly."
                ),
                source_module=self.__class__.__name__,
            )

        # Store the handler for unsubscribing
        self._trade_signal_handler = self.handle_trade_signal_approved
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, self._trade_signal_handler)

        # Subscribe to close position commands for emergency HALT
        self._close_position_handler = self.handle_close_position_command
        self.pubsub.subscribe(EventType.TRADE_SIGNAL_APPROVED, self._close_position_handler)

        self.logger.info(
            "ExecutionHandler started. Subscribed to TRADE_SIGNAL_APPROVED and close position commands.",  # noqa: E501
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
                    EventType.TRADE_SIGNAL_APPROVED,
                    self._trade_signal_handler,
                )
                self.logger.info("Unsubscribed from TRADE_SIGNAL_APPROVED.")
                self._trade_signal_handler = None
            except Exception:
                self.logger.exception("Error unsubscribing")

        if self._close_position_handler is not None:
            try:
                self.pubsub.unsubscribe(
                    EventType.TRADE_SIGNAL_APPROVED,
                    self._close_position_handler,
                )
                self.logger.info("Unsubscribed from close position commands.")
                self._close_position_handler = None
            except Exception:
                self.logger.exception("Error unsubscribing from close position handler")

        # --- Configurable Cancellation of Open Orders (Enterprise Feature) ---
        cancel_on_stop = self.config.get_bool("execution_handler.cancel_orders_on_stop", False)
        cancel_timeout = self.config.get_float("execution_handler.cancel_orders_timeout_seconds", 10.0)

        if cancel_on_stop and self._order_map:
            self.logger.warning(
                "Cancelling %d open orders on shutdown (configured behavior)",
                len(self._order_map),
                source_module=self.__class__.__name__,
            )

            # Collect all active order IDs
            active_orders = list(self._order_map.values())

            # Create cancellation tasks with timeout
            cancellation_tasks = []
            for exchange_order_id in active_orders:
                task = asyncio.create_task(self.cancel_order(exchange_order_id))
                cancellation_tasks.append(task)

            if cancellation_tasks:
                try:
                    # Wait for all cancellations with timeout
                    results = await asyncio.wait_for(
                        asyncio.gather(*cancellation_tasks, return_exceptions=True),
                        timeout=cancel_timeout,
                    )

                    successful = sum(1 for r in results if isinstance(r, bool) and r)
                    failed = len(results) - successful

                    self.logger.info(
                        "Order cancellation complete. Success: %d, Failed: %d",
                        successful,
                        failed,
                        source_module=self.__class__.__name__,
                    )
                except TimeoutError:
                    self.logger.error(
                        "Order cancellation timed out after %.1f seconds",
                        cancel_timeout,
                        source_module=self.__class__.__name__,
                    )
                    # Cancel the cancellation tasks
                    for task in cancellation_tasks:
                        if not task.done():
                            task.cancel()
        elif cancel_on_stop:
            self.logger.info(
                "No open orders to cancel on shutdown",
                source_module=self.__class__.__name__,
            )

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

        # Clean up the execution adapter
        if self._adapter:
            try:
                await self._adapter.cleanup()
                self.logger.info(
                    "Execution adapter cleaned up successfully",
                    source_module=self.__class__.__name__,
                )
            except Exception:
                self.logger.exception(
                    "Error cleaning up execution adapter",
                    source_module=self.__class__.__name__,
                )

        self.logger.info(
            "ExecutionHandler stopped.",
            source_module=self.__class__.__name__,
        )

        # Enterprise WebSocket Disconnection Logic
        if self._use_websocket_for_orders and self.websocket_client:
            self._is_running = False  # Signal that we're intentionally stopping
            self.logger.info(
                "Disconnecting WebSocket connection...",
                source_module=self.__class__.__name__,
            )
            try:
                await self._disconnect_websocket()
            except Exception:
                self.logger.exception(
                    "Error during WebSocket disconnection",
                    source_module=self.__class__.__name__,
                )

        # TODO: Implement configurable cancellation of open orders on stop
        # Future enhancement: Add configuration option to auto-cancel open orders during shutdown
        # This would include safety checks to prevent accidental cancellations

    async def _make_public_request_with_retry(
        self,
        url: str,
        max_retries: int = 3,
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
                            jitter = secrets.SystemRandom().uniform(0, delay * 0.1)
                            total_delay = delay + jitter
                            self.logger.warning(
                                "Retryable API error for %s: %s. "
                                "Retrying in %.2fs (Attempt %d/%d)",
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
                        (
                            "Error during public request to %s: %s. "
                            "Retrying in %.2fs "
                            "(Attempt %d/%d)"
                        ),
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
            ("Failed to make public request to %s " "after %d attempts. " "Last error: %s"),
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
        kraken_pair_map: dict,
        result: dict,
    ) -> bool:
        """Process a single trading pair and store its information."""
        kraken_altname = internal_pair_name.replace("/", "")
        kraken_key = kraken_pair_map.get(kraken_altname)

        if not kraken_key or kraken_key not in result:
            self.logger.warning(
                ("Could not find matching AssetPairs info for " "configured pair: %s"),
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
            ("Successfully loaded info for %s asset pairs " "out of %s configured."),
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

    def _generate_kraken_signature(
        self,
        uri_path: str,
        data: dict[str, Any],
        nonce: int,
    ) -> str:
        """Generate the API-Sign header required by Kraken private endpoints."""
        return generate_kraken_signature(uri_path, data, nonce, self.api_secret)

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
        except ValueError:  # Raised by _generate_kraken_signature if API secret is invalid
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
            total=self.config.get("exchange.request_timeout_seconds", 10),
        )

        try:
            self.logger.debug(
                "Sending private request to %s with data: %s",
                url,
                request_data,
                source_module=self.__class__.__name__,
            )
            async with self._session.post(
                url,
                headers=headers,
                data=request_data,
                timeout=timeout,
            ) as response:
                response.raise_for_status()  # Raise exception for bad status codes (4xx, 5xx)
                result: dict[str, Any] = await response.json()
                self.logger.debug(
                    "Received response from %s: %s",
                    url,
                    result,
                    source_module=self.__class__.__name__,
                )
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
            self.logger.exception(
                "Connection Error to %s: %s",
                url,
                source_module=self.__class__.__name__,
            )
            response_data = {"error": [f"EGeneral:ConnectionError - {e!s}"]}
        except TimeoutError:
            self.logger.exception(
                "Request Timeout for %s: %s",
                url,
                source_module=self.__class__.__name__,
            )
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
                    final_result = current_result  # Store the error dict
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

                if (
                    isinstance(e, aiohttp.ClientConnectionError | asyncio.TimeoutError)
                    and attempt < max_retries
                ):
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
                        "Non-retryable exception or max retries for network error. "
                        "URI: %s. Error: %s",
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
        kraken_pair_name = self._get_and_validate_pair_name(
            internal_pair,
            pair_info,
            event.signal_id,
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
            event.signal_id,
            params,
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

    def _get_and_validate_pair_name(
        self,
        internal_pair: str,
        pair_info: dict[str, Any],
        signal_id: UUID,
    ) -> str | None:
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
            await self._publish_error_execution_report(
                originating_event,
                "Empty API response",
                cl_ord_id,
            )
            return

        # Check for API-level errors first
        if result.get("error"):
            # Kraken errors are usually a list of strings
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

                    # Enterprise Order ID Mapping - Update bidirectional mappings
                    self._update_order_id_mapping(indexed_cl_ord_id, kraken_order_id)

                    # Add to pending orders tracking for WebSocket correlation
                    self._pending_orders_by_cl_ord_id[indexed_cl_ord_id] = originating_event

                    # Publish initial "NEW" execution report for each order
                    report = ExecutionReportEvent(
                        source_module=self.__class__.__name__,
                        event_id=UUID(int=int(time.time() * 1000000) + idx),  # Ensure unique event IDs
                        timestamp=datetime.utcnow(),
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
                    self._start_order_monitoring(indexed_cl_ord_id, kraken_order_id, originating_event)
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
            self.logger.info(
                "Attempting to establish WebSocket connection...",
                source_module=self.__class__.__name__,
            )

            # Cancel any existing connection task
            if self._websocket_connection_task and not self._websocket_connection_task.done():
                self._websocket_connection_task.cancel()
                try:
                    await self._websocket_connection_task
                except asyncio.CancelledError:
                    pass

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

        except Exception as e:
            self.logger.error(
                "WebSocket connection attempt failed: %s",
                str(e),
                source_module=self.__class__.__name__,
            )
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

        except Exception as e:
            self.logger.error(
                "Failed to subscribe to order channels: %s",
                str(e),
                source_module=self.__class__.__name__,
            )

    async def _handle_websocket_message(self, message: dict[str, Any]) -> None:
        """Process a message received from the exchange WebSocket with enterprise-grade handling.

        Handles different message types including order updates, fill notifications,
        and maintains bidirectional order ID mapping between internal and exchange IDs.
        
        Args:
            message: Raw message from WebSocket containing order updates or other data
        """
        try:
            self.logger.debug(
                "Received WebSocket message: %s",
                message,
                source_module=self.__class__.__name__,
            )

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

        except Exception as e:
            self.logger.exception(
                "Error processing WebSocket message: %s. Message: %s",
                str(e),
                message,
                source_module=self.__class__.__name__,
            )

    def _parse_message_type(self, message: dict[str, Any]) -> str:
        """Parse WebSocket message to determine its type.
        
        Args:
            message: Raw WebSocket message
            
        Returns:
            String identifier for message type
        """
        # Handle event-based messages
        if isinstance(message, dict):
            if "event" in message:
                event = message.get("event")
                if event == "subscriptionStatus":
                    return "SUBSCRIPTION_ACK"
                if event == "systemStatus":
                    return "HEARTBEAT"
                return f"EVENT_{event.upper()}"
            if "channel" in message or "channelName" in message:
                channel = message.get("channel") or message.get("channelName")
                if channel in ["ownTrades"]:
                    return "FILL_NOTIFICATION"
                if channel in ["openOrders"]:
                    return "ORDER_UPDATE"

        # Handle list-based message format (typical for Kraken)
        elif isinstance(message, list) and len(message) >= 3:
            channel_name = message[2] if len(message) > 2 else None
            if channel_name == "ownTrades":
                return "FILL_NOTIFICATION"
            if channel_name == "openOrders":
                return "ORDER_UPDATE"

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

        except Exception as e:
            self.logger.exception(
                "Error processing order update message: %s",
                str(e),
                source_module=self.__class__.__name__,
            )

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

        except Exception as e:
            self.logger.exception(
                "Error processing fill notification: %s",
                str(e),
                source_module=self.__class__.__name__,
            )

    async def _process_auth_response_message(self, message: dict[str, Any]) -> None:
        """Process authentication response from WebSocket."""
        try:
            status = message.get("status")
            if status == "ok":
                self._websocket_connection_state = "AUTHENTICATED"
                self.logger.info(
                    "WebSocket authentication successful",
                    source_module=self.__class__.__name__,
                )
                await self._subscribe_to_order_channels()
            else:
                self.logger.error(
                    "WebSocket authentication failed: %s",
                    message.get("errorMessage", "Unknown error"),
                    source_module=self.__class__.__name__,
                )
                self._websocket_connection_state = "ERROR"

        except Exception as e:
            self.logger.exception(
                "Error processing auth response: %s",
                str(e),
                source_module=self.__class__.__name__,
            )

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

        except Exception as e:
            self.logger.exception(
                "Error processing subscription ack: %s",
                str(e),
                source_module=self.__class__.__name__,
            )

    def _extract_order_data_from_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Extract order data from WebSocket message.
        
        Handles different message formats from Kraken WebSocket.
        """
        try:
            # Handle different message formats
            if isinstance(message, list) and len(message) >= 2:
                # List format: [channel_id, data, channel_name, ...]
                data = message[1]
                if isinstance(data, list) and len(data) > 0:
                    return data[0]  # First order in the list
                if isinstance(data, dict):
                    return data
            elif isinstance(message, dict):
                # Dict format with direct order data
                return message.get("data", message)

            return None

        except Exception as e:
            self.logger.error(
                "Error extracting order data from message: %s",
                str(e),
                source_module=self.__class__.__name__,
            )
            return None

    def _extract_trade_data_from_message(self, message: dict[str, Any]) -> list[dict[str, Any]]:
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

            return []

        except Exception as e:
            self.logger.error(
                "Error extracting trade data from message: %s",
                str(e),
                source_module=self.__class__.__name__,
            )
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
            order_status = self._map_exchange_status_to_internal(order_data.get("status", "unknown"))

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
                timestamp=datetime.utcnow(),
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

        except Exception as e:
            self.logger.exception(
                "Error creating execution report from WebSocket data: %s",
                str(e),
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
                    if isinstance(timestamp_value, (int, float)):
                        return datetime.fromtimestamp(timestamp_value, tz=UTC)
                    if isinstance(timestamp_value, str):
                        try:
                            return datetime.fromtimestamp(float(timestamp_value), tz=UTC)
                        except ValueError:
                            continue
            return None
        except Exception:
            return None

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
            timeout_s = self.config.get_float(
                "order.limit_order_timeout_s",
                300.0,
            )  # 5 mins default
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
                "QueryOrders response for %s missing 'result' dict or is wrong type: %s",
                exchange_order_id,
                result_field,
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
                self.logger.error(
                    "Order %s has invalid or missing status: %s",
                    exchange_order_id,
                    current_status,
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
                exchange_order_id,
                order_data,
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
                        self._handle_sl_tp_orders(
                            original_event,
                            exchange_order_id,
                            current_filled_qty,
                        ),
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

    async def _monitor_order_status(
        self,
        exchange_order_id: str,
        client_order_id: str,
        signal_id: UUID | None,
    ) -> None:
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
        max_poll_duration = self.config.get_float(
            "order.max_poll_duration_s",
            3600.0,
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

    async def _get_originating_signal_event(
        self,
        signal_id: UUID | None,
    ) -> TradeSignalApprovedEvent | None:
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
                return events[-1]  # type: ignore

            self.logger.warning(
                f"No approved signal event found for signal_id: {signal_id}",
                source_module=self.__class__.__name__,
            )
            return None

        except Exception:
            self.logger.exception(
                f"Error retrieving signal event for {signal_id}",
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
            event.signal_id,
            cl_ord_id,
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
                "vol",
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
                    opentm,
                    tz=UTC,
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
                metadata={"reduce_only": "true", "originating_signal": str(originating_event.signal_id)},
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
                metadata={"reduce_only": "true", "originating_signal": str(originating_event.signal_id)},
            )
            batch_request.orders.append(tp_order)

            try:
                # Place both orders together
                batch_response = await self._adapter.place_batch_orders(batch_request)

                if batch_response.success:
                    # Process each order result
                    for i, order_result in enumerate(batch_response.order_results):
                        if order_result.success and order_result.exchange_order_ids:
                            order_type = "SL" if i == 0 else "TP"
                            exchange_order_id = order_result.exchange_order_ids[0]
                            client_order_id = order_result.client_order_id

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
                            await self._publish_initial_execution_report(
                                originating_event=originating_event,
                                client_order_id=client_order_id,
                                exchange_order_id=exchange_order_id,
                                order_type=batch_request.orders[i].order_type,
                            )

                            # Start monitoring
                            self._start_order_monitoring(
                                client_order_id,
                                exchange_order_id,
                                originating_event,
                            )
                        else:
                            order_type = "SL" if i == 0 else "TP"
                            self.logger.error(
                                "%s order placement failed: %s",
                                order_type,
                                order_result.error_message,
                                source_module=self.__class__.__name__,
                            )
                else:
                    self.logger.error(
                        "Batch order placement failed: %s",
                        batch_response.error_message,
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
                sl_cl_ord_id = (
                    f"gf-sl-{str(originating_event.signal_id)[:8]}-{int(time.time() * 1000000)}"
                )
                sl_params["cl_ord_id"] = sl_cl_ord_id
                sl_params["reduce_only"] = "true"  # Good practice for exits

                self.logger.info(
                    "Placing SL order for signal %s with cl_ord_id %s",
                    originating_event.signal_id,
                    sl_cl_ord_id,
                    source_module=self.__class__.__name__,
                )

                sl_result = await self._make_private_request_with_retry(
                    "/0/private/AddOrder",
                    sl_params,
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
                    originating_event.signal_id,
                    tp_cl_ord_id,
                    source_module=self.__class__.__name__,
                )

                tp_result = await self._make_private_request_with_retry(
                    "/0/private/AddOrder",
                    tp_params,
                )
                # Handle TP order placement response (publish report, start monitoring)
                await self._handle_add_order_response(tp_result, originating_event, tp_cl_ord_id)

    async def _monitor_limit_order_timeout(
        self,
        exchange_order_id: str,
        timeout_seconds: float,
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
                exc_info=True,
            )

    async def _place_order_with_priority(self, params: dict) -> dict:
        """Place order with priority handling for emergencies.

        Args:
            params: Order parameters

        Returns:
            API response dict
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
        if not self.websocket_client or self._websocket_connection_state in ["DISCONNECTED", "DISABLED"]:
            return

        try:
            self.logger.info(
                "Initiating WebSocket disconnection...",
                source_module=self.__class__.__name__,
            )

            # Cancel connection task if running
            if self._websocket_connection_task and not self._websocket_connection_task.done():
                self._websocket_connection_task.cancel()
                try:
                    await self._websocket_connection_task
                except asyncio.CancelledError:
                    pass

            # Disconnect the WebSocket client
            await self.websocket_client.disconnect()

            # Clean up state
            self._websocket_connection_state = "DISCONNECTED"
            self._subscribed_channels.clear()
            self._websocket_auth_token = None
            self._current_reconnect_attempts = 0

            self.logger.info(
                "WebSocket disconnection completed",
                source_module=self.__class__.__name__,
            )

        except Exception as e:
            self.logger.exception(
                "Error during WebSocket disconnection: %s",
                str(e),
                source_module=self.__class__.__name__,
            )

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
            # Future enhancement: Integrate with monitoring service alerts
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
            timestamp=datetime.utcnow(),
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
