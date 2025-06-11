import asyncio
from datetime import datetime, timedelta, UTC
import heapq
import pytest

from gal_friday.simulated_market_price_service import (
    RealTimeSimulationEngine,
    HistoricalDataPoint,
    DataRequest,
)

class DummyLoader:
    async def load_historical_data(self, request: DataRequest):
        start = request.start_date
        return [
            HistoricalDataPoint(start, request.symbol, 1, 1, 1, 1, 1),
            HistoricalDataPoint(start + timedelta(minutes=1), request.symbol, 2, 2, 2, 2, 2),
        ]

@pytest.mark.asyncio
async def test_simulation_engine_populates_events_with_historical_data():
    start = datetime(2021, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    config = {
        'symbols': ['BTC/USD'],
        'start_time': start,
        'end_time': end,
        'frequency': '1m',
    }
    engine = RealTimeSimulationEngine(config, DummyLoader())
    await engine._load_simulation_data()

    assert len(engine.event_queue) == 2
    first_event = heapq.heappop(engine.event_queue)
    assert isinstance(first_event.data['price_data'], dict)
    assert first_event.timestamp == start
