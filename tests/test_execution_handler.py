"""Tests for execution handler functionality.

This module tests the execution handler with a comprehensive mock
Kraken API, including order lifecycle, error scenarios, and edge cases.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from gal_friday.core.events import (
    EventType,
    ExecutionReportEvent,
    TradeSignalApprovedEvent,
)
from gal_friday.execution_handler import (
    ExecutionHandler,
)


class MockKrakenAPI:
    """Comprehensive mock of Kraken API for testing."""

    def __init__(self):
        self.orders = {}
        self.order_counter = 1000
        self.api_call_count = 0
        self.rate_limit_enabled = True
        self.error_responses = {}
        self.latency_ms = 50

        # Market state
        self.current_prices = {
            "XRPUSD": Decimal("0.5000"),
            "DOGEUSD": Decimal("0.0800"),
        }

        # Account state
        self.balances = {
            "USD": Decimal("100000"),
            "XRP": Decimal("0"),
            "DOGE": Decimal("0"),
        }

    async def add_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mock order placement."""
        await asyncio.sleep(self.latency_ms / 1000)  # Simulate network latency
        self.api_call_count += 1

        # Check for configured errors
        if "add_order" in self.error_responses:
            return self.error_responses["add_order"]

        # Validate parameters
        if not all(k in params for k in ["pair", "type", "ordertype", "volume"]):
            return {"error": ["EGeneral:Invalid arguments"]}

        # Generate order ID
        order_id = f"O{self.order_counter}"
        self.order_counter += 1

        # Create order record
        self.orders[order_id] = {
            "status": "open",
            "opentm": datetime.now(UTC).timestamp(),
            "descr": {
                "pair": params["pair"],
                "type": params["type"],
                "ordertype": params["ordertype"],
                "price": params.get("price", "0"),
                "order": f"{params['type']} {params['volume']} {params['pair']} @ {params.get('price', 'market')}",
            },
            "vol": params["volume"],
            "vol_exec": "0",
            "cost": "0",
            "fee": "0",
            "price": "0",
            "userref": params.get("userref"),
            "trades": [],
        }

        return {
            "error": [],
            "result": {
                "descr": {"order": self.orders[order_id]["descr"]["order"]},
                "txid": [order_id],
            },
        }

    async def query_orders(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mock order status query."""
        await asyncio.sleep(self.latency_ms / 1000)
        self.api_call_count += 1

        if "query_orders" in self.error_responses:
            return self.error_responses["query_orders"]

        txid = params.get("txid")
        if not txid or txid not in self.orders:
            return {"error": ["EOrder:Unknown order"]}

        # Simulate order lifecycle
        order = self.orders[txid]
        age = (datetime.now(UTC).timestamp() - order["opentm"])

        # Auto-fill market orders quickly
        if order["descr"]["ordertype"] == "market" and age > 0.5:
            self._fill_order(txid)

        # Partially fill limit orders over time
        elif order["descr"]["ordertype"] == "limit" and order["status"] == "open":
            if age > 2:
                self._partially_fill_order(txid, Decimal("0.3"))
            if age > 5:
                self._partially_fill_order(txid, Decimal("0.7"))
            if age > 10:
                self._fill_order(txid)

        return {
            "error": [],
            "result": {txid: order},
        }

    async def cancel_order(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mock order cancellation."""
        await asyncio.sleep(self.latency_ms / 1000)
        self.api_call_count += 1

        if "cancel_order" in self.error_responses:
            return self.error_responses["cancel_order"]

        txid = params.get("txid")
        if not txid or txid not in self.orders:
            return {"error": ["EOrder:Unknown order"]}

        order = self.orders[txid]
        if order["status"] in ["closed", "canceled"]:
            return {"error": ["EOrder:Order already closed"]}

        order["status"] = "canceled"
        order["reason"] = "User requested"

        return {
            "error": [],
            "result": {"count": 1},
        }

    async def get_asset_pairs(self) -> dict[str, Any]:
        """Mock asset pairs info."""
        await asyncio.sleep(self.latency_ms / 1000)

        return {
            "error": [],
            "result": {
                "XXRPZUSD": {
                    "altname": "XRPUSD",
                    "wsname": "XRP/USD",
                    "base": "XXRP",
                    "quote": "ZUSD",
                    "pair_decimals": 4,
                    "lot_decimals": 0,
                    "ordermin": "10",
                    "costmin": "0.5",
                    "tick_size": "0.0001",
                    "status": "online",
                },
                "XDGEZUSD": {
                    "altname": "DOGEUSD",
                    "wsname": "DOGE/USD",
                    "base": "XDGE",
                    "quote": "ZUSD",
                    "pair_decimals": 5,
                    "lot_decimals": 0,
                    "ordermin": "50",
                    "costmin": "0.5",
                    "tick_size": "0.00001",
                    "status": "online",
                },
            },
        }

    def _fill_order(self, txid: str):
        """Simulate order fill."""
        order = self.orders[txid]
        if order["status"] != "open":
            return

        order["status"] = "closed"
        order["vol_exec"] = order["vol"]

        # Calculate fill price
        pair = order["descr"]["pair"]
        if pair in self.current_prices:
            base_price = self.current_prices[pair]
            # Add some slippage for market orders
            if order["descr"]["ordertype"] == "market":
                slippage = Decimal("0.0001") if order["descr"]["type"] == "buy" else Decimal("-0.0001")
                order["price"] = str(base_price + slippage)
            else:
                order["price"] = order["descr"]["price"]

        # Calculate cost and fees
        volume = Decimal(order["vol"])
        price = Decimal(order["price"])
        order["cost"] = str(volume * price)
        order["fee"] = str(Decimal(order["cost"]) * Decimal("0.0016"))  # 0.16% fee

        # Add trade record
        order["trades"].append(f"T{self.order_counter}")
        self.order_counter += 1

    def _partially_fill_order(self, txid: str, fill_ratio: Decimal):
        """Simulate partial order fill."""
        order = self.orders[txid]
        if order["status"] != "open":
            return

        total_vol = Decimal(order["vol"])
        current_filled = Decimal(order["vol_exec"])
        target_filled = total_vol * fill_ratio

        if target_filled > current_filled:
            order["vol_exec"] = str(target_filled)
            # Update price as weighted average
            # For simplicity, use order price
            order["price"] = order["descr"]["price"]


class TestExecutionHandler:
    """Test suite for ExecutionHandler functionality."""

    @pytest.fixture
    async def mock_kraken_api(self):
        """Create mock Kraken API instance."""
        return MockKrakenAPI()

    @pytest.fixture
    async def execution_handler(self, mock_config_manager, pubsub_manager,
                                mock_logger, mock_kraken_api):
        """Create ExecutionHandler with mocked dependencies."""
        # Configure for testing
        mock_config_manager.config["kraken"] = {
            "api_key": "test_key",
            "secret_key": "dGVzdF9zZWNyZXQ=",  # base64 encoded
        }

        # Create monitoring service mock
        monitoring = AsyncMock()
        monitoring.is_halted.return_value = False

        handler = ExecutionHandler(
            mock_config_manager,
            pubsub_manager,
            monitoring,
            mock_logger,
        )

        # Patch HTTP session to use mock API
        handler._session = AsyncMock()

        # Mock API responses
        async def mock_get(url, **kwargs):
            response = AsyncMock()
            if "AssetPairs" in url:
                response.json.return_value = await mock_kraken_api.get_asset_pairs()
            response.raise_for_status = Mock()
            return response

        async def mock_post(url, **kwargs):
            response = AsyncMock()
            data = kwargs.get("data", {})

            if "AddOrder" in url:
                response.json.return_value = await mock_kraken_api.add_order(data)
            elif "QueryOrders" in url:
                response.json.return_value = await mock_kraken_api.query_orders(data)
            elif "CancelOrder" in url:
                response.json.return_value = await mock_kraken_api.cancel_order(data)

            response.raise_for_status = Mock()
            return response

        handler._session.get = mock_get
        handler._session.post = mock_post
        handler._session.closed = False

        # Store mock API reference
        handler._mock_api = mock_kraken_api

        await handler.start()
        yield handler
        await handler.stop()


class TestOrderLifecycle:
    """Test complete order lifecycle scenarios."""

    @pytest.mark.asyncio
    async def test_market_order_full_lifecycle(self, execution_handler, pubsub_manager):
        """Test market order from signal to fill."""
        execution_reports = []

        async def capture_reports(event):
            if isinstance(event, ExecutionReportEvent):
                execution_reports.append(event)

        pubsub_manager.subscribe(EventType.EXECUTION_REPORT, capture_reports)

        # Create approved signal
        signal = TradeSignalApprovedEvent(
            source_module="RiskManager",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=uuid.uuid4(),
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            quantity=Decimal("1000"),
            order_type="MARKET",
            limit_price=None,
            sl_price=Decimal("0.4900"),
            tp_price=Decimal("0.5200"),
        )

        # Process signal
        await execution_handler.handle_trade_signal_approved(signal)

        # Wait for order placement and initial report
        await asyncio.sleep(0.1)

        # Should have NEW status report
        assert len(execution_reports) >= 1
        assert execution_reports[0].order_status == "NEW"
        assert execution_reports[0].quantity_ordered == Decimal("1000")

        # Wait for order to fill (market orders fill quickly in mock)
        await asyncio.sleep(1)

        # Should have FILLED status report
        filled_reports = [r for r in execution_reports if r.order_status == "CLOSED"]
        assert len(filled_reports) > 0
        assert filled_reports[0].quantity_filled == Decimal("1000")
        assert filled_reports[0].average_fill_price is not None
        assert filled_reports[0].commission is not None

    @pytest.mark.asyncio
    async def test_limit_order_partial_fills(self, execution_handler, pubsub_manager):
        """Test limit order with partial fills."""
        execution_reports = []

        async def capture_reports(event):
            if isinstance(event, ExecutionReportEvent):
                execution_reports.append(event)

        pubsub_manager.subscribe(EventType.EXECUTION_REPORT, capture_reports)

        # Create limit order signal
        signal = TradeSignalApprovedEvent(
            source_module="RiskManager",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=uuid.uuid4(),
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            quantity=Decimal("1000"),
            order_type="LIMIT",
            limit_price=Decimal("0.4999"),
            sl_price=None,
            tp_price=None,
        )

        await execution_handler.handle_trade_signal_approved(signal)

        # Wait for partial fills
        await asyncio.sleep(3)

        # Check for partial fill reports
        partial_reports = [r for r in execution_reports if r.order_status == "OPEN" and r.quantity_filled > 0]
        assert len(partial_reports) > 0
        assert partial_reports[0].quantity_filled < partial_reports[0].quantity_ordered

    @pytest.mark.asyncio
    async def test_stop_loss_placement_after_fill(self, execution_handler, pubsub_manager):
        """Test SL order placement after entry order fills."""
        sl_orders_placed = []

        async def capture_sl_orders(event):
            if isinstance(event, ExecutionReportEvent) and "sl" in event.client_order_id:
                sl_orders_placed.append(event)

        pubsub_manager.subscribe(EventType.EXECUTION_REPORT, capture_sl_orders)

        # Create signal with SL
        signal = TradeSignalApprovedEvent(
            source_module="RiskManager",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            signal_id=uuid.uuid4(),
            trading_pair="XRP/USD",
            exchange="kraken",
            side="BUY",
            quantity=Decimal("1000"),
            order_type="MARKET",
            limit_price=None,
            sl_price=Decimal("0.4900"),
            tp_price=None,
        )

        # Mock the signal retrieval
        async def mock_get_signal(signal_id):
            return signal if signal_id == signal.signal_id else None

        execution_handler._get_originating_signal_event = mock_get_signal

        await execution_handler.handle_trade_signal_approved(signal)

        # Wait for entry fill and SL placement
        await asyncio.sleep(2)

        # Should have SL order placed
        assert len(sl_orders_placed) > 0
        sl_order = sl_orders_placed[0]
        assert sl_order.side == "SELL"  # Opposite of entry
        assert sl_order.quantity_ordered == Decimal("1000")


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_api_error_handling(self, execution_handler, mock_kraken_api, pubsub_manager):
        """Test handling of API errors."""
        error_reports = []

        async def capture_errors(event):
            if isinstance(event, ExecutionReportEvent) and event.error_message:
                error_reports.append(event)

        pubsub_manager.subscribe(EventType.EXECUTION_REPORT, capture_errors)

        # Configure API to return error
        mock_kraken_api.error_responses["add_order"] = {
            "error": ["EOrder:Insufficient funds"],
        }

        signal = TradeSignalApprovedEvent.create_test_signal()
        await execution_handler.handle_trade_signal_approved(signal)

        await asyncio.sleep(0.1)

        assert len(error_reports) > 0
        assert "Insufficient funds" in error_reports[0].error_message
        assert error_reports[0].order_status == "REJECTED"

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, execution_handler):
        """Test rate limit enforcement."""
        # Send multiple orders rapidly
        signals = []
        for i in range(5):
            signal = TradeSignalApprovedEvent.create_test_signal(
                pair="XRP/USD",
                quantity=Decimal("100"),
            )
            signals.append(signal)

        # Process all signals concurrently
        tasks = [execution_handler.handle_trade_signal_approved(s) for s in signals]

        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Should take at least 4 seconds due to rate limiting (1 call/second)
        duration = end_time - start_time
        assert duration >= 3  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_halt_state_rejection(self, execution_handler, pubsub_manager):
        """Test order rejection when system is halted."""
        # Set system to halted state
        execution_handler.monitoring.is_halted.return_value = True

        rejected_reports = []

        async def capture_rejected(event):
            if isinstance(event, ExecutionReportEvent) and event.order_status == "REJECTED":
                rejected_reports.append(event)

        pubsub_manager.subscribe(EventType.EXECUTION_REPORT, capture_rejected)

        signal = TradeSignalApprovedEvent.create_test_signal()
        await execution_handler.handle_trade_signal_approved(signal)

        await asyncio.sleep(0.1)

        assert len(rejected_reports) > 0
        assert "HALTED" in rejected_reports[0].error_message

    @pytest.mark.asyncio
    async def test_order_timeout_cancellation(self, execution_handler, mock_config_manager):
        """Test automatic cancellation of timed-out limit orders."""
        # Set short timeout for testing
        mock_config_manager.config["order"] = {"limit_order_timeout_s": 0.5}

        signal = TradeSignalApprovedEvent.create_test_signal(
            order_type="LIMIT",
            limit_price=Decimal("0.4800"),  # Won't fill at this price
        )

        await execution_handler.handle_trade_signal_approved(signal)

        # Wait for timeout
        await asyncio.sleep(1)

        # Check if order was cancelled
        mock_api = execution_handler._mock_api
        cancelled_orders = [o for o in mock_api.orders.values() if o["status"] == "canceled"]
        assert len(cancelled_orders) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimum_order_size_validation(self, execution_handler):
        """Test rejection of orders below minimum size."""
        signal = TradeSignalApprovedEvent.create_test_signal(
            quantity=Decimal("5"),  # Below minimum of 10 for XRP
        )

        await execution_handler.handle_trade_signal_approved(signal)

        # Order should be rejected during validation
        # Check logs for error
        assert any("below minimum" in str(log["message"])
                  for log in execution_handler.logger.messages
                  if log["level"] == "ERROR")

    @pytest.mark.asyncio
    async def test_decimal_precision_handling(self, execution_handler):
        """Test proper decimal precision in order parameters."""
        signal = TradeSignalApprovedEvent.create_test_signal(
            quantity=Decimal("1234.56789"),
            limit_price=Decimal("0.123456789"),
        )

        # Capture the actual API call
        original_post = execution_handler._session.post
        captured_params = {}

        async def capture_post(url, **kwargs):
            captured_params.update(kwargs.get("data", {}))
            return await original_post(url, **kwargs)

        execution_handler._session.post = capture_post

        await execution_handler.handle_trade_signal_approved(signal)
        await asyncio.sleep(0.1)

        # Check precision formatting
        assert captured_params.get("volume") == "1235"  # 0 decimals for XRP
        assert captured_params.get("price") == "0.1235"  # 4 decimals for XRP/USD

    @pytest.mark.asyncio
    async def test_simultaneous_sl_tp_orders(self, execution_handler):
        """Test placing both SL and TP orders after fill."""
        orders_placed = []

        async def capture_orders(event):
            if isinstance(event, ExecutionReportEvent):
                orders_placed.append(event)

        execution_handler.pubsub.subscribe(EventType.EXECUTION_REPORT, capture_orders)

        signal = TradeSignalApprovedEvent.create_test_signal(
            order_type="MARKET",
            sl_price=Decimal("0.4900"),
            tp_price=Decimal("0.5200"),
        )

        # Mock signal retrieval
        async def mock_get_signal(signal_id):
            return signal if signal_id == signal.signal_id else None

        execution_handler._get_originating_signal_event = mock_get_signal

        await execution_handler.handle_trade_signal_approved(signal)

        # Wait for entry fill and contingent orders
        await asyncio.sleep(2)

        # Should have entry + SL + TP orders
        sl_orders = [o for o in orders_placed if "sl" in o.client_order_id]
        tp_orders = [o for o in orders_placed if "tp" in o.client_order_id]

        assert len(sl_orders) > 0
        assert len(tp_orders) > 0
        assert sl_orders[0].side != signal.side
        assert tp_orders[0].side != signal.side


# Helper method for creating test signals
def create_test_signal(**kwargs):
    """Create test trade signal with defaults."""
    defaults = {
        "source_module": "TestModule",
        "event_id": uuid.uuid4(),
        "timestamp": datetime.now(UTC),
        "signal_id": uuid.uuid4(),
        "trading_pair": kwargs.get("pair", "XRP/USD"),
        "exchange": "kraken",
        "side": "BUY",
        "quantity": kwargs.get("quantity", Decimal("1000")),
        "order_type": kwargs.get("order_type", "MARKET"),
        "limit_price": kwargs.get("limit_price"),
        "sl_price": kwargs.get("sl_price"),
        "tp_price": kwargs.get("tp_price"),
    }

    return TradeSignalApprovedEvent(**defaults)

# Monkey patch for testing
TradeSignalApprovedEvent.create_test_signal = staticmethod(create_test_signal)
