"""Tests for order cancellation handling in PortfolioManager."""
# ruff: noqa: D101, D102, D103, D107

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from gal_friday.core.events import (
    EventType,
    ExecutionReportEvent,
    ExecutionReportParams,
    OrderCancellationEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.interfaces.market_price_service_interface import MarketPriceService
from gal_friday.portfolio_manager import PortfolioManager


class DummyMarketPriceService(MarketPriceService):
    async def start(self) -> None:  # pragma: no cover - not used
        pass

    async def stop(self) -> None:  # pragma: no cover - not used
        pass

    async def get_latest_price(self, trading_pair: str) -> Decimal | None:
        return Decimal("1")

    async def get_bid_ask_spread(self, trading_pair: str):
        return Decimal("1"), Decimal("1.1")

    async def get_price_timestamp(self, trading_pair: str):
        return datetime.now(UTC)

    async def is_price_fresh(self, trading_pair: str, max_age_seconds: float = 60.0):
        return True

    async def convert_amount(
        self,
        from_amount: Decimal,
        from_currency: str,
        to_currency: str,
    ) -> Decimal:
        return from_amount

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        timeframe: str,
        since: datetime,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
        return []


class MockPubSub:
    def __init__(self) -> None:
        self.published: list = []

    def subscribe(self, event_type: EventType, handler) -> None:
        pass

    async def publish(self, event) -> None:
        self.published.append(event)


@pytest.mark.asyncio
async def test_order_cancellation_publishes_event(
    mock_config_manager,
    db_session_maker,
    mock_logger,
) -> None:
    with patch.object(PortfolioManager, "_initialize_state", new=AsyncMock()):
        mock_pubsub = MockPubSub()
        manager = PortfolioManager(
            config_manager=mock_config_manager,
            pubsub_manager=cast(PubSubManager, mock_pubsub),
            market_price_service=DummyMarketPriceService(),
            logger_service=mock_logger,
            session_maker=db_session_maker,
        )
        await manager._initialization_task

    params = ExecutionReportParams(
        source_module="test",
        exchange_order_id="ORD1",
        trading_pair="XRP/USD",
        exchange="TEST",
        order_status="CANCELED",
        order_type="LIMIT",
        side="BUY",
        quantity_ordered=Decimal("1"),
    )
    report = ExecutionReportEvent.create(params)

    await manager._handle_execution_report(report)

    assert len(mock_pubsub.published) == 1
    event = mock_pubsub.published[0]
    assert isinstance(event, OrderCancellationEvent)
    assert event.exchange_order_id == "ORD1"
    assert event.event_type == EventType.ORDER_CANCELLATION
