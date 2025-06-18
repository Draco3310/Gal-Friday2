"""Tests for reporting trade outcomes."""

from decimal import Decimal
from typing import TYPE_CHECKING, cast
import uuid

import asyncio
import pytest

from gal_friday.core.events import TradeOutcomeEvent
from gal_friday.core.pubsub import EventType, PubSubManager
from gal_friday.strategy_arbitrator import StrategyArbitrator

if TYPE_CHECKING:
    from gal_friday.core.feature_registry_client import FeatureRegistryClient
    from gal_friday.market_price_service import MarketPriceService
    from gal_friday.strategy_selection import StrategySelectionSystem


class DummyFeatureRegistry:
    """Minimal feature registry stub used for testing."""

    def is_loaded(self) -> bool:
        """Indicate the registry is loaded."""
        return True

    def get_feature_definition(self, _feature: str) -> dict:
        """Return an empty feature definition."""
        return {}


class DummyMarketPriceService:
    """Stub market price service."""

    async def get_latest_price(self, _symbol: str) -> Decimal:
        """Return a static price."""
        return Decimal("1.0")


@pytest.mark.asyncio
async def test_report_trade_outcome_publishes_event(
    pubsub_manager: PubSubManager,
    mock_logger,
) -> None:
    """Verify that trade outcomes emit events and timeseries logs."""
    config = {
        "strategy_arbitrator": {
            "strategies": [
                {
                    "id": "test_strategy",
                    "buy_threshold": 0.6,
                    "sell_threshold": 0.4,
                    "entry_type": "MARKET",
                },
            ],
        },
    }

    arbitrator = StrategyArbitrator(
        config,
        pubsub_manager,
        mock_logger,
        cast("MarketPriceService", DummyMarketPriceService()),
        cast("FeatureRegistryClient", DummyFeatureRegistry()),
    )

    arbitrator._strategy_selection_enabled = True
    arbitrator.strategy_selection_system = cast("StrategySelectionSystem", object())

    received: list[TradeOutcomeEvent] = []

    async def capture(event: TradeOutcomeEvent) -> None:
        received.append(event)

    pubsub_manager.subscribe(EventType.TRADE_OUTCOME_REPORTED, capture)

    signal_id = uuid.uuid4()
    await arbitrator.report_trade_outcome(
        str(signal_id),
        "win",
        Decimal("5.0"),
        "tp_hit",
    )

    await asyncio.sleep(0.1)

    assert len(received) == 1
    event = received[0]
    assert event.signal_id == signal_id
    assert event.strategy_id == "test_strategy"
    assert event.outcome == "win"
    assert event.pnl == Decimal("5.0")
    assert event.exit_reason == "tp_hit"

    ts_logs = [m for m in mock_logger.messages if m["level"] == "TS"]
    assert ts_logs
