from datetime import UTC, datetime
from decimal import Decimal

import pytest

from gal_friday.core.events import MarketDataOHLCVEvent
from gal_friday.monitoring_service import MonitoringService


@pytest.mark.asyncio
async def test_market_price_from_event(mock_config_manager, pubsub_manager, mock_portfolio_manager, mock_logger):
    monitoring = MonitoringService(
        mock_config_manager,
        pubsub_manager,
        mock_portfolio_manager,
        mock_logger,
    )

    event = MarketDataOHLCVEvent(
        trading_pair="XRP/USD",
        exchange="kraken",
        interval="1m",
        timestamp_bar_start=datetime.now(UTC),
        open="0.5",
        high="0.6",
        low="0.4",
        close="0.55",
        volume="100",
    )
    await monitoring._update_market_data_timestamp(event)
    price = await monitoring._get_current_market_price("XRP/USD")
    assert price == Decimal("0.55")
