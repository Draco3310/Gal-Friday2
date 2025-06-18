"""
Comprehensive tests for the enterprise-grade execution event creation in Fill model.

Tests cover:
- ExecutionEventBuilder functionality
- Fill.to_event() method
- Event validation and error handling
- Event publishing
- Fallback logic and edge cases
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from gal_friday.core.events import ExecutionReportEvent
from gal_friday.models.fill import ExecutionEventBuilder, ExecutionEventPublisher, Fill


class TestExecutionEventBuilder:
    """Test cases for ExecutionEventBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = ExecutionEventBuilder()

        # Create mock order
        self.mock_order = Mock()
        self.mock_order.exchange_order_id = "EX123456"
        self.mock_order.client_order_id = uuid.uuid4()
        self.mock_order.signal_id = uuid.uuid4()
        self.mock_order.status = "FILLED"
        self.mock_order.order_type = "LIMIT"
        self.mock_order.quantity_ordered = Decimal("100.0")
        self.mock_order.limit_price = Decimal("50000.0")
        self.mock_order.stop_price = None
        self.mock_order.fills = []

        # Create mock fill
        self.mock_fill = Mock(spec=Fill)
        self.mock_fill.fill_pk = 123
        self.mock_fill.fill_id = "FILL123"
        self.mock_fill.order_pk = 456
        self.mock_fill.exchange_order_id = "EX123456"
        self.mock_fill.trading_pair = "BTC/USD"
        self.mock_fill.exchange = "KRAKEN"
        self.mock_fill.side = "BUY"
        self.mock_fill.quantity_filled = Decimal("50.0")
        self.mock_fill.fill_price = Decimal("50000.0")
        self.mock_fill.commission = Decimal("25.0")
        self.mock_fill.commission_asset = "USD"
        self.mock_fill.liquidity_type = "MAKER"
        self.mock_fill.filled_at = datetime.now(UTC)
        self.mock_fill.order = self.mock_order
        self.mock_fill.__class__.__name__ = "Fill"

    def test_create_execution_event_success(self):
        """Test successful creation of execution event."""
        event = self.builder.create_execution_event(self.mock_fill)

        assert isinstance(event, ExecutionReportEvent)
        assert event.exchange_order_id == "EX123456"
        assert event.trading_pair == "BTC/USD"
        assert event.exchange == "KRAKEN"
        assert event.side == "BUY"
        assert event.quantity_filled == Decimal("50.0")
        assert event.average_fill_price == Decimal("50000.0")
        assert event.commission == Decimal("25.0")
        assert event.commission_asset == "USD"

    def test_validate_fill_data_success(self):
        """Test successful fill data validation."""
        # Should not raise any exception
        self.builder._validate_fill_data(self.mock_fill)

    def test_validate_fill_data_missing_trading_pair(self):
        """Test validation failure for missing trading pair."""
        self.mock_fill.trading_pair = None

        with pytest.raises(ValueError, match="Fill must have a trading pair"):
            self.builder._validate_fill_data(self.mock_fill)

    def test_validate_fill_data_invalid_quantity(self):
        """Test validation failure for invalid quantity."""
        self.mock_fill.quantity_filled = Decimal(0)

        with pytest.raises(ValueError, match="Fill quantity must be positive"):
            self.builder._validate_fill_data(self.mock_fill)

    def test_validate_fill_data_invalid_price(self):
        """Test validation failure for invalid price."""
        self.mock_fill.fill_price = Decimal(-100)

        with pytest.raises(ValueError, match="Fill price must be positive"):
            self.builder._validate_fill_data(self.mock_fill)

    def test_get_exchange_order_id_from_fill(self):
        """Test getting exchange order ID from fill."""
        result = self.builder._get_exchange_order_id(self.mock_fill)
        assert result == "EX123456"

    def test_get_exchange_order_id_from_order(self):
        """Test getting exchange order ID from related order."""
        self.mock_fill.exchange_order_id = None
        result = self.builder._get_exchange_order_id(self.mock_fill)
        assert result == "EX123456"

    def test_get_exchange_order_id_fallback_to_client(self):
        """Test fallback to client order ID."""
        self.mock_fill.exchange_order_id = None
        self.mock_order.exchange_order_id = None

        result = self.builder._get_exchange_order_id(self.mock_fill)
        assert result.startswith("client_")

    def test_get_exchange_order_id_final_fallback(self):
        """Test final fallback to fill PK."""
        self.mock_fill.exchange_order_id = None
        self.mock_fill.order = None

        result = self.builder._get_exchange_order_id(self.mock_fill)
        assert result == "fill_123"

    def test_determine_order_status_from_order(self):
        """Test determining order status from order."""
        result = self.builder._determine_order_status(self.mock_fill)
        assert result == "FILLED"

    def test_determine_order_status_no_order(self):
        """Test determining order status without order context."""
        self.mock_fill.order = None
        result = self.builder._determine_order_status(self.mock_fill)
        assert result == "FILLED"

    def test_get_order_type_from_order(self):
        """Test getting order type from related order."""
        result = self.builder._get_order_type(self.mock_fill)
        assert result == "LIMIT"

    def test_get_order_type_from_liquidity(self):
        """Test inferring order type from liquidity type."""
        self.mock_fill.order = None
        result = self.builder._get_order_type(self.mock_fill)
        assert result == "LIMIT"  # MAKER -> LIMIT

    def test_get_order_type_fallback(self):
        """Test order type fallback."""
        self.mock_fill.order = None
        self.mock_fill.liquidity_type = None
        result = self.builder._get_order_type(self.mock_fill)
        assert result == "MARKET"

    def test_get_signal_id_success(self):
        """Test getting signal ID from order."""
        result = self.builder._get_signal_id(self.mock_fill)
        assert result == self.mock_order.signal_id

    def test_get_signal_id_no_order(self):
        """Test getting signal ID without order."""
        self.mock_fill.order = None
        result = self.builder._get_signal_id(self.mock_fill)
        assert result is None

    def test_get_limit_price_for_limit_order(self):
        """Test getting limit price for limit order."""
        result = self.builder._get_limit_price(self.mock_fill)
        assert result == Decimal("50000.0")

    def test_get_limit_price_for_market_order(self):
        """Test getting limit price for market order."""
        self.mock_order.order_type = "MARKET"
        result = self.builder._get_limit_price(self.mock_fill)
        assert result is None


class TestExecutionEventPublisher:
    """Test cases for ExecutionEventPublisher class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_event_bus = AsyncMock()
        self.publisher = ExecutionEventPublisher(self.mock_event_bus)

        self.mock_event = Mock(spec=ExecutionReportEvent)
        self.mock_event.event_id = uuid.uuid4()
        self.mock_event.exchange_order_id = "EX123456"

    @pytest.mark.asyncio
    async def test_publish_execution_event_success(self):
        """Test successful event publishing."""
        result = await self.publisher.publish_execution_event(self.mock_event)

        assert result is True
        assert self.publisher.published_events == 1
        assert self.publisher.failed_publications == 0
        self.mock_event_bus.publish.assert_called_once_with("execution_reports", self.mock_event)

    @pytest.mark.asyncio
    async def test_publish_execution_event_no_bus(self):
        """Test publishing without event bus."""
        publisher = ExecutionEventPublisher(None)
        result = await publisher.publish_execution_event(self.mock_event)

        assert result is False

    @pytest.mark.asyncio
    async def test_publish_execution_event_failure(self):
        """Test handling of publishing failure."""
        self.mock_event_bus.publish.side_effect = Exception("Connection failed")

        result = await self.publisher.publish_execution_event(self.mock_event)

        assert result is False
        assert self.publisher.published_events == 0
        assert self.publisher.failed_publications == 1

    def test_get_publication_stats(self):
        """Test getting publication statistics."""
        self.publisher.published_events = 10
        self.publisher.failed_publications = 2

        stats = self.publisher.get_publication_stats()

        assert stats["published_events"] == 10
        assert stats["failed_publications"] == 2
        assert stats["success_rate"] == 10/12  # 83.33%


class TestFillEnhancements:
    """Test cases for enhanced Fill class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a real Fill instance for testing
        self.fill = Fill(
            fill_pk=123,
            fill_id="FILL123",
            order_pk=456,
            exchange_order_id="EX123456",
            trading_pair="BTC/USD",
            exchange="KRAKEN",
            side="BUY",
            quantity_filled=Decimal("50.0"),
            fill_price=Decimal("50000.0"),
            commission=Decimal("25.0"),
            commission_asset="USD",
            liquidity_type="MAKER",
            filled_at=datetime.now(UTC),
        )

        # Create mock order
        self.mock_order = Mock()
        self.mock_order.exchange_order_id = "EX123456"
        self.mock_order.client_order_id = uuid.uuid4()
        self.mock_order.signal_id = uuid.uuid4()
        self.mock_order.status = "FILLED"
        self.mock_order.order_type = "LIMIT"
        self.mock_order.quantity_ordered = Decimal("100.0")
        self.mock_order.limit_price = Decimal("50000.0")
        self.mock_order.stop_price = None
        self.mock_order.fills = [self.fill]

        self.fill.order = self.mock_order

    def test_to_event_success(self):
        """Test successful event creation from fill."""
        event = self.fill.to_event()

        assert isinstance(event, ExecutionReportEvent)
        assert event.trading_pair == "BTC/USD"
        assert event.quantity_filled == Decimal("50.0")
        assert event.commission == Decimal("25.0")

    def test_to_event_with_custom_builder(self):
        """Test event creation with custom builder."""
        custom_builder = Mock(spec=ExecutionEventBuilder)
        mock_event = Mock(spec=ExecutionReportEvent)
        custom_builder.create_execution_event.return_value = mock_event

        Fill.set_event_builder(custom_builder)

        result = self.fill.to_event()

        assert result == mock_event
        custom_builder.create_execution_event.assert_called_once_with(self.fill)

        # Clean up
        Fill.set_event_builder(None)

    @pytest.mark.asyncio
    async def test_publish_execution_event_success(self):
        """Test successful event publishing from fill."""
        mock_publisher = Mock(spec=ExecutionEventPublisher)
        mock_publisher.publish_execution_event = AsyncMock(return_value=True)

        Fill.set_event_publisher(mock_publisher)

        result = await self.fill.publish_execution_event()

        assert result is True

        # Clean up
        Fill.set_event_publisher(None)

    @pytest.mark.asyncio
    async def test_publish_execution_event_failure(self):
        """Test handling of event publishing failure."""
        mock_publisher = Mock(spec=ExecutionEventPublisher)
        mock_publisher.publish_execution_event = AsyncMock(side_effect=Exception("Publishing failed"))

        Fill.set_event_publisher(mock_publisher)

        result = await self.fill.publish_execution_event()

        assert result is False

        # Clean up
        Fill.set_event_publisher(None)

    def test_get_execution_summary(self):
        """Test getting execution summary."""
        summary = self.fill.get_execution_summary()

        assert summary["fill_id"] == "FILL123"
        assert summary["trading_pair"] == "BTC/USD"
        assert summary["quantity_filled"] == 50.0
        assert summary["fill_price"] == 50000.0
        assert summary["gross_value"] == 2500000.0  # 50 * 50000
        assert summary["net_value"] == 2499975.0   # gross - commission

    def test_dependency_injection_methods(self):
        """Test dependency injection methods."""
        builder = ExecutionEventBuilder()
        publisher = ExecutionEventPublisher()

        Fill.set_event_builder(builder)
        Fill.set_event_publisher(publisher)

        assert Fill._event_builder == builder
        assert Fill._event_publisher == publisher


class TestIntegration:
    """Integration tests for the complete execution event workflow."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_event_bus = AsyncMock()

        # Create complete fill with order
        self.fill = Fill(
            fill_pk=123,
            fill_id="FILL123",
            order_pk=456,
            exchange_order_id="EX123456",
            trading_pair="BTC/USD",
            exchange="KRAKEN",
            side="BUY",
            quantity_filled=Decimal("1.5"),
            fill_price=Decimal("45000.0"),
            commission=Decimal("67.5"),
            commission_asset="USD",
            liquidity_type="TAKER",
            filled_at=datetime.now(UTC),
        )

        self.mock_order = Mock()
        self.mock_order.exchange_order_id = "EX123456"
        self.mock_order.client_order_id = uuid.uuid4()
        self.mock_order.signal_id = uuid.uuid4()
        self.mock_order.status = "PARTIALLY_FILLED"
        self.mock_order.order_type = "MARKET"
        self.mock_order.quantity_ordered = Decimal("2.0")
        self.mock_order.limit_price = None
        self.mock_order.stop_price = None
        self.mock_order.fills = [self.fill]

        self.fill.order = self.mock_order

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow from fill to published event."""
        # Set up publisher with event bus
        publisher = ExecutionEventPublisher(self.mock_event_bus)
        Fill.set_event_publisher(publisher)

        # Create and publish event
        result = await self.fill.publish_execution_event()

        # Verify success
        assert result is True
        assert publisher.published_events == 1

        # Verify event bus was called
        self.mock_event_bus.publish.assert_called_once()
        call_args = self.mock_event_bus.publish.call_args
        assert call_args[0][0] == "execution_reports"

        published_event = call_args[0][1]
        assert isinstance(published_event, ExecutionReportEvent)
        assert published_event.trading_pair == "BTC/USD"
        assert published_event.quantity_filled == Decimal("1.5")
        assert published_event.order_status == "PARTIALLY_FILLED"

        # Clean up
        Fill.set_event_publisher(None)


# Error cases and edge cases
class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_create_event_with_invalid_fill(self):
        """Test error handling with invalid fill data."""
        invalid_fill = Fill(
            fill_pk=1,
            trading_pair="",  # Invalid empty trading pair
            exchange="KRAKEN",
            side="BUY",
            quantity_filled=Decimal(0),  # Invalid zero quantity
            fill_price=Decimal("50000.0"),
            commission=Decimal("25.0"),
            commission_asset="USD",
            filled_at=datetime.now(UTC),
        )

        with pytest.raises(RuntimeError, match="Execution event creation failed"):
            invalid_fill.to_event()

    def test_builder_handles_missing_order_gracefully(self):
        """Test that builder handles missing order data gracefully."""
        fill = Fill(
            fill_pk=123,
            fill_id="FILL123",
            trading_pair="ETH/USD",
            exchange="BINANCE",
            side="SELL",
            quantity_filled=Decimal("10.0"),
            fill_price=Decimal("3000.0"),
            commission=Decimal("30.0"),
            commission_asset="USD",
            filled_at=datetime.now(UTC),
        )
        # No order relationship
        fill.order = None

        # Should still create event successfully with fallbacks
        event = fill.to_event()
        assert isinstance(event, ExecutionReportEvent)
        assert event.trading_pair == "ETH/USD"
        assert event.exchange_order_id == "fill_123"  # Fallback ID
