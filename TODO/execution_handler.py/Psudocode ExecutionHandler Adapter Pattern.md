# File: gal_friday/interfaces/execution_adapter_interface.py (New File)
# Defines the abstract interface for an execution adapter.

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable, List, Tuple # For Python 3.8+
from datetime import datetime
from decimal import Decimal # Ensure Decimal is imported

# Assuming Order and ExecutionReportEvent are defined elsewhere (e.g., core.models, core.events)
# from gal_friday.models.order import Order # Conceptual
# from gal_friday.core.events import ExecutionReportEvent # Conceptual

# Placeholder for actual Order and ExecutionReportEvent for pseudocode context
class SLTPOptions(Protocol): # Placeholder for SL/TP parameters
    stop_loss_price: Optional[Decimal]
    take_profit_price: Optional[Decimal]
    # Potentially other SL/TP related params like trigger method, order type for SL/TP

class Order: # Placeholder - Enhanced to include SL/TP info
    def __init__(self, client_order_id: str, symbol: str, side: str, type: str, quantity: Decimal, price: Optional[Decimal] = None, sltp_options: Optional[SLTPOptions] = None):
        self.client_order_id = client_order_id
        self.symbol = symbol
        self.side = side # Added for use in KrakenExecutionAdapter
        self.type = type # Added for use in KrakenExecutionAdapter
        self.quantity = quantity # Added for use in KrakenExecutionAdapter
        self.price = price # Added for use in KrakenExecutionAdapter
        self.sltp_options = sltp_options # e.g., {"stop_loss_price": Decimal("100"), "take_profit_price": Decimal("120")}
        # ... other attributes ...

class ExecutionReportEvent: # Placeholder
    def __init__(self, internal_order_id: str, exchange_order_id: Optional[str], status: str, **kwargs): # Added **kwargs
        self.internal_order_id = internal_order_id
        self.exchange_order_id = exchange_order_id # Added
        self.status = status # Added
        # ... other attributes ...

@runtime_checkable
class BaseExecutionAdapter(Protocol):
    """
    Abstract interface for an exchange execution adapter.
    Defines the contract for interacting with an exchange's execution functionalities.
    """

    async def connect(self) -> bool:
        """Connect any necessary services, like private WebSockets for order updates."""
        ...

    async def disconnect(self) -> None:
        """Disconnect any services, like private WebSockets."""
        ...

    async def submit_order(self, order: Order) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
        """
        Submits an order to the exchange, potentially including linked SL/TP orders.
        Returns the internal client order ID of the primary order, 
        and a list of tuples: (order_role: str, exchange_order_id: Optional[str]) 
        for primary, SL, TP orders.
        Example roles: "PRIMARY", "STOP_LOSS", "TAKE_PROFIT".
        Exchange order IDs might be populated later via WebSocket updates for some orders in a batch.
        """
        ...

    async def cancel_order(self, internal_cl_ord_id: str, exchange_order_id: Optional[str] = None) -> bool:
        """
        Cancels an order on the exchange.
        Can use either internal client order ID or exchange order ID.
        Returns True if cancellation request was successfully sent, False otherwise.
        Actual cancellation confirmation comes via order updates.
        """
        ...

    async def get_order_status(self, internal_cl_ord_id: str, exchange_order_id: Optional[str] = None) -> Optional[dict]: # dict represents parsed status
        """
        Queries the status of a specific order.
        Returns a dictionary with order status details or None if not found/error.
        """
        ...

    async def get_open_orders(self, trading_pair: Optional[str] = None) -> list[dict]: # list of dicts representing open orders
        """
        Fetches all open orders, optionally filtered by trading pair.
        """
        ...

    async def get_account_balance(self, asset: Optional[str] = None) -> dict[str, Decimal]: # asset symbol -> balance
        """
        Fetches account balances, optionally for a specific asset.
        """
        ...
    
    def get_exchange_order_id(self, internal_cl_ord_id: str) -> Optional[str]:
        """Gets the mapped exchange order ID for a given internal client order ID (typically the primary)."""
        ...

# --- End of execution_adapter_interface.py ---
```

```python
# File: gal_friday/execution/kraken_execution_adapter.py (New or Refactored from parts of existing ExecutionHandler)
# Implements the BaseExecutionAdapter for the Kraken exchange.

from decimal import Decimal
from typing import Any, Optional, Callable, Coroutine, List, Tuple # For Python 3.8+

# from ..interfaces.execution_adapter_interface import BaseExecutionAdapter, Order, SLTPOptions, ExecutionReportEvent # Adjust import
# from ..core.events import OrderStatus # Conceptual
# from ..execution.websocket_client import KrakenWebSocketClient # Or a more generic one
# from ..config_manager import ConfigManager
# from ..logger_service import LoggerService
# from ..core.pubsub import PubSubManager
# from ..utils.kraken_api import KrakenRESTAPIClient # Conceptual REST client

# Placeholder for actual OrderStatus for pseudocode context
class OrderStatus: # Placeholder
    NEW = "NEW"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    # ... other statuses

class KrakenExecutionAdapter(BaseExecutionAdapter): # Implements the Protocol

    def __init__(self, 
                 config_manager: ConfigManager, 
                 pubsub_manager: PubSubManager, 
                 logger_service: LoggerService,
                 # exchange_spec: Any # Potentially needed for some exchange info
                ):
        self.config_manager = config_manager
        self.pubsub_manager = pubsub_manager
        self.logger = logger_service.get_logger(self.__class__.__name__)
        self._source_module = self.__class__.__name__

        self._kraken_rest_api = instantiate_kraken_rest_client(config_manager, logger_service) # Handles REST calls
        
        # --- WebSocket State and ID Mapping (Moved here from ExecutionHandler) ---
        self._websocket_config = self.config_manager.get("execution_handler.websocket", {}) # Or adapter-specific config
        self._use_websocket_for_orders = self._websocket_config.get("use_for_order_updates", True) # Default to True for adapter

        if self._use_websocket_for_orders:
            self.websocket_client = instantiate_websocket_client( # e.g., KrakenWebSocketClient for private feeds
                ws_url=self._websocket_config.get("private_url"),
                api_key=self.config_manager.get_secret("kraken_api_key"),
                api_secret=self.config_manager.get_secret("kraken_api_secret"),
                on_message_callback=self._handle_websocket_message,
                on_open_callback=self._on_websocket_open,
                on_close_callback=self._on_websocket_close,
                on_error_callback=self._on_websocket_error,
                logger=self.logger
            )
            self._websocket_connection_state = "DISCONNECTED"
            self._websocket_connection_task = None
            self._subscribed_channels = set()
            self._max_reconnect_attempts = self._websocket_config.get("max_reconnect_attempts", 5)
            self._reconnect_delay_seconds = self._websocket_config.get("reconnect_delay_seconds", 5)
            self._current_reconnect_attempts = 0
        else:
            self.websocket_client = None
            self._websocket_connection_state = "DISABLED"

        self._internal_to_exchange_order_id: dict[str, str] = {} # Maps primary internal ID to primary exchange ID
        self._exchange_to_internal_order_id: dict[str, str] = {} # Maps primary exchange ID to primary internal ID
        # For batch orders, one internal_cl_ord_id (for the strategy order) might map to multiple exchange IDs (primary, SL, TP)
        # We might need a more complex structure if we need to track SL/TP internal IDs separately.
        # For now, assume internal_cl_ord_id is for the primary conceptual order.
        self._pending_orders_by_cl_ord_id: dict[str, Any] = {} 

        self.logger.info("KrakenExecutionAdapter initialized.")

    async def connect(self) -> bool:
        # ... (Same as previous version) ...
        pass

    async def disconnect(self) -> None:
        # ... (Same as previous version) ...
        pass

    async def submit_order(self, order: Order) -> Tuple[str, List[Tuple[str, Optional[str]]]]:
        self.logger.info(f"Submitting order via KrakenAdapter: {order.client_order_id}")
        internal_cl_ord_id = order.client_order_id
        self._pending_orders_by_cl_ord_id[internal_cl_ord_id] = order 
        
        exchange_order_results: List[Tuple[str, Optional[str]]] = [] # (role, exchange_id)

        if order.sltp_options and (order.sltp_options.stop_loss_price or order.sltp_options.take_profit_price):
            # --- Use AddOrderBatch if SL/TP is present ---
            # This assumes Kraken's Batch Add Order endpoint is available
            self.logger.info(f"SL/TP options present for order {internal_cl_ord_id}. Attempting batch submission.")
            batch_params = self._convert_order_to_kraken_batch_params(order) # New helper method
            
            response_data = await self._kraken_rest_api.add_order_batch(orders=batch_params) # Conceptual REST call

            if response_data and not response_data.get("error"):
                # Kraken's AddOrderBatch response needs careful parsing.
                # It might return a list of results, each with 'txid' or 'error'.
                # Example conceptual parsing:
                results = response_data.get("result", [])
                if isinstance(results, list): # Assuming it returns a list for batch
                    for i, res_item in enumerate(results):
                        role = batch_params[i].get("role", "UNKNOWN") # Assuming role was added in params
                        userref = batch_params[i].get("userref") # Assuming userref was part of each order in batch

                        if res_item and not res_item.get("error"):
                            exchange_id = res_item.get("txid") # Kraken often uses 'txid' for order ID
                            if isinstance(exchange_id, list): exchange_id = exchange_id[0] # if txid is a list

                            if exchange_id:
                                exchange_order_results.append((role, exchange_id))
                                # Map IDs, potentially linking SL/TP exchange IDs back to the primary internal ID
                                # or to their own derived internal IDs if generated.
                                if role == "PRIMARY" and userref == internal_cl_ord_id:
                                    self._internal_to_exchange_order_id[internal_cl_ord_id] = exchange_id
                                    self._exchange_to_internal_order_id[exchange_id] = internal_cl_ord_id
                                self.logger.info(f"Batch order item {role} (userref: {userref}) submitted, exchange ID: {exchange_id}")
                                ack_report = self._create_execution_report_for_ack(userref or f"{internal_cl_ord_id}_{role}", exchange_id, order, role) # Pass role
                                await self.pubsub_manager.publish(ack_report)
                            else:
                                exchange_order_results.append((role, None))
                                self.logger.error(f"Batch order item {role} (userref: {userref}) submitted but no txid: {res_item}")
                        else:
                            exchange_order_results.append((role, None))
                            item_error = res_item.get("error", "Unknown error in batch item")
                            self.logger.error(f"Error in batch order item {role} (userref: {userref}): {item_error}")
                            reject_report = self._create_execution_report_for_rejection(userref or f"{internal_cl_ord_id}_{role}", order, str(item_error), role) # Pass role
                            await self.pubsub_manager.publish(reject_report)
                else: # Non-list result or unexpected format
                     self.logger.error(f"Unexpected AddOrderBatch response format for {internal_cl_ord_id}: {response_data}")
                     # Mark all as failed for simplicity
                     exchange_order_results.append(("PRIMARY", None))
                     if order.sltp_options.stop_loss_price: exchange_order_results.append(("STOP_LOSS", None))
                     if order.sltp_options.take_profit_price: exchange_order_results.append(("TAKE_PROFIT", None))

            else: # Error at batch level
                error_msg = response_data.get("error", ["Unknown batch error"]) if response_data else ["No response for batch"]
                self.logger.error(f"Failed to submit order batch for {internal_cl_ord_id} to Kraken: {error_msg}")
                self._pending_orders_by_cl_ord_id.pop(internal_cl_ord_id, None)
                reject_report = self._create_execution_report_for_rejection(internal_cl_ord_id, order, str(error_msg), "PRIMARY_BATCH_FAIL")
                await self.pubsub_manager.publish(reject_report)
                exchange_order_results.append(("PRIMARY", None)) # Indicate failure
                if order.sltp_options.stop_loss_price: exchange_order_results.append(("STOP_LOSS", None))
                if order.sltp_options.take_profit_price: exchange_order_results.append(("TAKE_PROFIT", None))
        else:
            # --- Use single AddOrder if no SL/TP ---
            kraken_params = self._convert_order_to_kraken_params(order) # Existing helper
            response_data = await self._kraken_rest_api.add_order(**kraken_params) 

            if response_data and not response_data.get("error"):
                exchange_order_ids = response_data.get("result", {}).get("txid", [])
                if exchange_order_ids:
                    exchange_order_id = exchange_order_ids[0] 
                    self._internal_to_exchange_order_id[internal_cl_ord_id] = exchange_order_id
                    self._exchange_to_internal_order_id[exchange_order_id] = internal_cl_ord_id
                    self.logger.info(f"Order {internal_cl_ord_id} submitted to Kraken, exchange ID(s): {exchange_order_ids}")
                    exchange_order_results.append(("PRIMARY", exchange_order_id))
                    ack_report = self._create_execution_report_for_ack(internal_cl_ord_id, exchange_order_id, order, "PRIMARY")
                    await self.pubsub_manager.publish(ack_report)
                else:
                    self.logger.error(f"Kraken order submission for {internal_cl_ord_id} successful but no txid returned: {response_data}")
                    exchange_order_results.append(("PRIMARY", None))
            else:
                error_msg = response_data.get("error", ["Unknown error"]) if response_data else ["No response"]
                self.logger.error(f"Failed to submit order {internal_cl_ord_id} to Kraken: {error_msg}")
                self._pending_orders_by_cl_ord_id.pop(internal_cl_ord_id, None)
                reject_report = self._create_execution_report_for_rejection(internal_cl_ord_id, order, str(error_msg), "PRIMARY")
                await self.pubsub_manager.publish(reject_report)
                exchange_order_results.append(("PRIMARY", None))
        
        # If primary failed but was part of a batch attempt, it might already be in exchange_order_results
        # Ensure primary is always present in the result list for consistency, even if None
        if not any(role == "PRIMARY" for role, ex_id in exchange_order_results):
            primary_ex_id = self._internal_to_exchange_order_id.get(internal_cl_ord_id)
            exchange_order_results.insert(0, ("PRIMARY", primary_ex_id))
            
        return internal_cl_ord_id, exchange_order_results

    # ... (cancel_order, get_order_status, etc. remain similar to previous version) ...
    # ... (WebSocket internal methods remain similar) ...

    # --- New Helper for Batch Order Params ---
    def _convert_order_to_kraken_batch_params(self, order: Order) -> List[dict]:
        batch_orders = []
        
        # 1. Primary Order
        primary_params = self._convert_order_to_kraken_params(order) # Reuse existing helper
        primary_params["role"] = "PRIMARY" # Add role for parsing response
        # Kraken AddOrderBatch might require 'orders' to be a list of dictionaries.
        # Each dictionary is one order. Ensure userref is unique if needed for linking.
        batch_orders.append(primary_params)

        # 2. Stop-Loss Order (if specified)
        if order.sltp_options and order.sltp_options.stop_loss_price:
            sl_userref = f"{order.client_order_id}_SL" # Generate unique userref for SL
            sl_params = {
                "pair": self._get_kraken_pair_name(order.symbol),
                "type": "sell" if order.side.lower() == "buy" else "buy", # Opposite side for SL
                "ordertype": "stop-loss", # Or "stop-loss-limit" if price2 is used
                "price": str(order.sltp_options.stop_loss_price), # This is the trigger price for stop-loss
                # "price2": Optional limit price for stop-loss-limit
                "volume": str(order.quantity), # Full quantity
                "userref": sl_userref,
                "role": "STOP_LOSS",
                # Potentially "reduce_only": True if applicable
                # Kraken specific: "close[ordertype]", "close[price]", "close[price2]" might be used for conditional close.
                # Or it might be a separate order type like 'stop-loss'. Refer to Kraken API for linked SL/TP.
                # Assuming simple stop-loss for now.
            }
            # If Kraken uses specific conditional close parameters on the primary order instead of separate orders in a batch:
            # primary_params["stopprice"] = str(order.sltp_options.stop_loss_price)
            # primary_params["ordertype"] = "stop-loss-limit" # or similar for linked orders
            # else, add as a separate order in the batch:
            batch_orders.append(sl_params)

        # 3. Take-Profit Order (if specified)
        if order.sltp_options and order.sltp_options.take_profit_price:
            tp_userref = f"{order.client_order_id}_TP"
            tp_params = {
                "pair": self._get_kraken_pair_name(order.symbol),
                "type": "sell" if order.side.lower() == "buy" else "buy", # Opposite side
                "ordertype": "limit", # TP is typically a limit order
                "price": str(order.sltp_options.take_profit_price),
                "volume": str(order.quantity),
                "userref": tp_userref,
                "role": "TAKE_PROFIT",
                # Potentially "reduce_only": True
                # If Kraken uses specific conditional close parameters on the primary order:
                # primary_params["limitprice"] = str(order.sltp_options.take_profit_price)
                # (This part is highly dependent on Kraken's exact mechanism for OCO or linked SL/TP with primary)
                # else, add as a separate order:
            }
            batch_orders.append(tp_params)
        
        # If only primary and no SL/TP, this method shouldn't have been called,
        # but if it was, batch_orders would contain only the primary.
        # The calling `submit_order` logic decides whether to use batch or single.
        return batch_orders


    def _convert_order_to_kraken_params(self, order: Order) -> dict:
        # ... (same as previous version, ensure client_order_id is mapped to userref) ...
        # Ensure userref is correctly formatted (e.g., Kraken might expect an integer for userref).
        # For simplicity, keeping as string and assuming API client handles conversion or API accepts string.
        # Max 32-bit unsigned integer for userref, typically. For client_order_id (UUID string),
        # a hash or a sequence number might be needed if the exchange expects an int userref.
        # For now, passing order.client_order_id directly, assuming it's handled or Kraken allows string userref.
        # Let's refine this to ensure client_order_id is handled as userref properly.
        # Kraken 'userref' is a 32-bit integer. We need a way to map our string client_order_id.
        # This could be done by maintaining a local map or hashing, or using a portion if it fits.
        # For simplicity in pseudocode, we'll assume a string userref is acceptable or mapped.
        # A better approach: generate an int userref, map it to internal client_order_id.
        
        internal_cl_ord_id_int = self._get_or_create_int_userref(order.client_order_id)

        params = {
            "pair": self._get_kraken_pair_name(order.symbol),
            "type": order.side.lower(), 
            "ordertype": order.type.lower(), 
            "volume": str(order.quantity),
            "userref": str(internal_cl_ord_id_int), # Pass as string, Kraken API expects int but usually handles string representation
        }
        if order.type.lower() == "limit" and order.price is not None:
            params["price"] = str(order.price)
        return params

    def _get_or_create_int_userref(self, client_order_id_str: str) -> int:
        # Placeholder: In a real system, map UUID string to a unique 32-bit integer for userref.
        # This could involve a sequence generator + lookup table, or hashing (with collision risk).
        # For now, just hash and take a portion, acknowledging this isn't robust.
        # return abs(hash(client_order_id_str)) % (2**31 -1) # Example, not production ready
        # This mapping needs careful design. For pseudocode, we'll assume it's handled.
        # For this iteration, let's assume the full client_order_id (string) is attempted if the API allows,
        # or that a more robust mapping to an int userref would be implemented.
        # The `_convert_order_to_kraken_params` had `order.client_order_id[:32]`. If string is allowed, that's fine.
        # If int is strictly required, then a mapping service is needed.
        # For now, let's assume a string userref is acceptable by the conceptual _kraken_rest_api.add_order
        # or that the params dictionary above will be processed further before API call.
        # The key is that `userref` needs to be set for each order in the batch if we want to correlate them.
        
        # For the purpose of this pseudocode, we'll assume the _convert_order_to_kraken_params
        # correctly sets a 'userref' that can be correlated.
        # If generating separate userrefs for SL/TP, they should be derived from the primary client_order_id.
        return abs(hash(client_order_id_str)) % (2**31 -1) # Example only

    def _create_execution_report_for_ack(self, internal_id: str, exchange_id: Optional[str], order_details: Any, role: str) -> ExecutionReportEvent: # Added role
        # return ExecutionReportEvent(internal_order_id=internal_id, exchange_order_id=exchange_id, status=OrderStatus.NEW, role=role, ...)
        pass 

    def _create_execution_report_for_rejection(self, internal_id: str, order_details: Any, reason: str, role: str) -> ExecutionReportEvent: # Added role
        # return ExecutionReportEvent(internal_order_id=internal_id, status=OrderStatus.REJECTED, reject_reason=reason, role=role, ...)
        pass 

# --- End of kraken_execution_adapter.py ---
```

```python
# File: gal_friday/execution_handler.py (Refactored to use an adapter)

# ... (imports) ...

class ExecutionHandler: # This class becomes more generic
    # ... (__init__, start, stop remain similar) ...

    async def submit_order_request(self, order: Order) -> Tuple[str, List[Tuple[str, Optional[str]]]]: 
        self.logger.info(f"ExecutionHandler received order request: {order.client_order_id}")
        
        # 1. Pre-trade checks ...
        #    IF NOT self.risk_manager.validate_order_pre_submission(order):
        #        ...
        #        reject_report = self._create_execution_report_for_local_rejection(order.client_order_id, "Risk check failed", "PRIMARY")
        #        await self.pubsub_manager.publish(reject_report)
        #        # Return format indicates failure for all potential parts (Primary, SL, TP)
        #        results = [("PRIMARY", None)]
        #        if order.sltp_options:
        #            if order.sltp_options.stop_loss_price: results.append(("STOP_LOSS", None))
        #            if order.sltp_options.take_profit_price: results.append(("TAKE_PROFIT", None))
        #        return order.client_order_id, results


        # 2. Delegate to the adapter
        try:
            internal_id, exchange_results = await self.adapter.submit_order(order)
            self.logger.info(f"Order {internal_id} submission attempt via adapter. Results: {exchange_results}")
            return internal_id, exchange_results
        except Exception as e: 
            self.logger.error(f"Error submitting order {order.client_order_id} via adapter: {e}", exc_info=True)
            reject_report = self._create_execution_report_for_local_rejection(order.client_order_id, f"Adapter submission error: {e}", "PRIMARY")
            await self.pubsub_manager.publish(reject_report)
            results = [("PRIMARY", None)]
            if order.sltp_options:
                if order.sltp_options.stop_loss_price: results.append(("STOP_LOSS", None))
                if order.sltp_options.take_profit_price: results.append(("TAKE_PROFIT", None))
            return order.client_order_id, results

    # ... (cancel_order_request remains similar) ...
    
    # DEF _create_execution_report_for_local_rejection(self, internal_id: str, reason: str, role: str) -> ExecutionReportEvent:
    #     # Helper to create a rejection event for orders failing local checks
    #     pass # Placeholder

# --- End of refactored execution_handler.py ---
