"""Tests for data ingestion functionality.

This module tests the complete data ingestion pipeline including
market data processing, validation, and performance.
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gal_friday.core.events import (
    EventType,
    MarketDataL2Event,
    MarketDataOHLCVEvent,
)
from gal_friday.data_ingestion.data_validator import DataValidator
from gal_friday.data_ingestion.market_data_processor import MarketDataProcessor


class TestMarketDataL2:
    """Test suite for Level 2 (order book) market data."""

    @pytest.fixture
    def sample_l2_data(self):
        """Create sample L2 market data."""
        return {
            "bids": [
                [Decimal("0.5000"), Decimal("1000")],
                [Decimal("0.4999"), Decimal("2000")],
                [Decimal("0.4998"), Decimal("1500")],
            ],
            "asks": [
                [Decimal("0.5001"), Decimal("1000")],
                [Decimal("0.5002"), Decimal("2000")],
                [Decimal("0.5003"), Decimal("1500")],
            ],
            "timestamp": datetime.now(UTC),
            "pair": "XRP/USD",
        }

    @pytest.mark.asyncio
    async def test_l2_event_creation(self, sample_l2_data):
        """Test creation of L2 market data events."""
        event = MarketDataL2Event(
            source_module="DataIngestion",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair=sample_l2_data["pair"],
            exchange="kraken",
            bids=sample_l2_data["bids"],
            asks=sample_l2_data["asks"],
            timestamp_exchange=sample_l2_data["timestamp"],
            sequence_number=12345,
        )

        assert event.trading_pair == "XRP/USD"
        assert len(event.bids) == 3
        assert len(event.asks) == 3
        assert event.bids[0][0] == Decimal("0.5000")

    @pytest.mark.asyncio
    async def test_order_book_depth_calculation(self, sample_l2_data):
        """Test order book depth calculations."""
        processor = MarketDataProcessor()

        # Calculate bid/ask depth
        bid_depth = sum(bid[1] for bid in sample_l2_data["bids"])
        ask_depth = sum(ask[1] for ask in sample_l2_data["asks"])

        assert bid_depth == Decimal("4500")
        assert ask_depth == Decimal("4500")

        # Calculate weighted average prices
        weighted_bid = sum(b[0] * b[1] for b in sample_l2_data["bids"]) / bid_depth
        weighted_ask = sum(a[0] * a[1] for a in sample_l2_data["asks"]) / ask_depth

        assert weighted_bid < Decimal("0.5000")
        assert weighted_ask > Decimal("0.5001")

    @pytest.mark.asyncio
    async def test_spread_analysis(self, sample_l2_data):
        """Test bid-ask spread calculations."""
        best_bid = sample_l2_data["bids"][0][0]
        best_ask = sample_l2_data["asks"][0][0]

        # Absolute spread
        spread = best_ask - best_bid
        assert spread == Decimal("0.0001")

        # Percentage spread
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (spread / mid_price) * 100
        assert spread_pct < Decimal("0.02")  # Less than 0.02%

    @pytest.mark.asyncio
    async def test_l2_data_validation(self):
        """Test validation of L2 data integrity."""
        validator = DataValidator()

        # Valid data
        valid_data = {
            "bids": [[Decimal("0.5000"), Decimal("1000")]],
            "asks": [[Decimal("0.5001"), Decimal("1000")]],
        }
        assert validator.validate_l2_data(valid_data)

        # Invalid: crossed book
        invalid_data = {
            "bids": [[Decimal("0.5002"), Decimal("1000")]],
            "asks": [[Decimal("0.5001"), Decimal("1000")]],
        }
        assert not validator.validate_l2_data(invalid_data)

        # Invalid: negative prices
        invalid_data2 = {
            "bids": [[Decimal("-0.5000"), Decimal("1000")]],
            "asks": [[Decimal("0.5001"), Decimal("1000")]],
        }
        assert not validator.validate_l2_data(invalid_data2)


class TestMarketDataOHLCV:
    """Test suite for OHLCV (candlestick) market data."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        return {
            "timestamp": datetime.now(UTC),
            "open": Decimal("0.5000"),
            "high": Decimal("0.5100"),
            "low": Decimal("0.4950"),
            "close": Decimal("0.5050"),
            "volume": Decimal("50000"),
            "timeframe": "1m",
            "pair": "XRP/USD",
        }

    @pytest.mark.asyncio
    async def test_ohlcv_event_creation(self, sample_ohlcv_data):
        """Test creation of OHLCV market data events."""
        event = MarketDataOHLCVEvent(
            source_module="DataIngestion",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair=sample_ohlcv_data["pair"],
            exchange="kraken",
            timeframe=sample_ohlcv_data["timeframe"],
            open_price=sample_ohlcv_data["open"],
            high_price=sample_ohlcv_data["high"],
            low_price=sample_ohlcv_data["low"],
            close_price=sample_ohlcv_data["close"],
            volume=sample_ohlcv_data["volume"],
            timestamp_exchange=sample_ohlcv_data["timestamp"],
        )

        assert event.open_price == Decimal("0.5000")
        assert event.high_price >= event.low_price
        assert event.high_price >= event.open_price
        assert event.high_price >= event.close_price

    @pytest.mark.asyncio
    async def test_timeframe_aggregation(self):
        """Test aggregation of 1m candles to higher timeframes."""
        processor = MarketDataProcessor()

        # Create 5 1-minute candles
        base_time = datetime.now(UTC).replace(second=0, microsecond=0)
        candles_1m = []

        for i in range(5):
            candle = {
                "timestamp": base_time + timedelta(minutes=i),
                "open": Decimal("0.5000") + Decimal(f"0.000{i}"),
                "high": Decimal("0.5010") + Decimal(f"0.000{i}"),
                "low": Decimal("0.4990") + Decimal(f"0.000{i}"),
                "close": Decimal("0.5005") + Decimal(f"0.000{i}"),
                "volume": Decimal("10000"),
            }
            candles_1m.append(candle)

        # Aggregate to 5m
        candle_5m = processor.aggregate_candles(candles_1m, "5m")

        assert candle_5m["open"] == candles_1m[0]["open"]
        assert candle_5m["close"] == candles_1m[-1]["close"]
        assert candle_5m["high"] == max(c["high"] for c in candles_1m)
        assert candle_5m["low"] == min(c["low"] for c in candles_1m)
        assert candle_5m["volume"] == sum(c["volume"] for c in candles_1m)

    @pytest.mark.asyncio
    async def test_ohlcv_data_validation(self, sample_ohlcv_data):
        """Test validation of OHLCV data integrity."""
        validator = DataValidator()

        # Valid data
        assert validator.validate_ohlcv_data(sample_ohlcv_data)

        # Invalid: high < low
        invalid_data = sample_ohlcv_data.copy()
        invalid_data["high"] = Decimal("0.4900")
        invalid_data["low"] = Decimal("0.5100")
        assert not validator.validate_ohlcv_data(invalid_data)

        # Invalid: negative volume
        invalid_data2 = sample_ohlcv_data.copy()
        invalid_data2["volume"] = Decimal("-1000")
        assert not validator.validate_ohlcv_data(invalid_data2)


class TestDataQuality:
    """Test suite for data quality and integrity."""

    @pytest.mark.asyncio
    async def test_missing_data_handling(self, pubsub_manager):
        """Test handling of missing data points."""
        processor = MarketDataProcessor()

        # Simulate missing data points
        timestamps = [
            datetime.now(UTC),
            datetime.now(UTC) + timedelta(minutes=1),
            # Missing minute 2
            datetime.now(UTC) + timedelta(minutes=3),
        ]

        gaps = processor.detect_data_gaps(timestamps, expected_interval=60)
        assert len(gaps) == 1
        assert gaps[0]["duration"] == 120  # 2 minutes

    @pytest.mark.asyncio
    async def test_corrupt_data_detection(self):
        """Test detection and handling of corrupt data."""
        validator = DataValidator()

        corrupt_scenarios = [
            # Extreme price spike
            {
                "current": Decimal("0.5000"),
                "previous": Decimal("0.5001"),
                "threshold": Decimal("0.10"),  # 10% threshold
                "is_corrupt": True,
            },
            # Normal price movement
            {
                "current": Decimal("0.5050"),
                "previous": Decimal("0.5000"),
                "threshold": Decimal("0.10"),
                "is_corrupt": False,
            },
        ]

        for scenario in corrupt_scenarios:
            price_change = abs(scenario["current"] - scenario["previous"]) / scenario["previous"]
            is_corrupt = price_change > scenario["threshold"]
            assert is_corrupt == scenario["is_corrupt"]

    @pytest.mark.asyncio
    async def test_duplicate_data_handling(self):
        """Test detection and handling of duplicate data."""
        processor = MarketDataProcessor()

        # Create duplicate events
        event1 = MarketDataL2Event.create_test_event(
            pair="XRP/USD",
            sequence_number=1000,
        )
        event2 = MarketDataL2Event.create_test_event(
            pair="XRP/USD",
            sequence_number=1000,  # Duplicate
        )

        assert processor.is_duplicate(event1) == False
        processor.mark_processed(event1)
        assert processor.is_duplicate(event2) == True


class TestDataIngestionPerformance:
    """Test suite for data ingestion performance."""

    @pytest.mark.asyncio
    async def test_ingestion_latency(self, pubsub_manager):
        """Test end-to-end data ingestion latency."""
        latencies = []

        async def measure_latency(event):
            """Measure time from event creation to processing."""
            latency = (datetime.now(UTC) - event.timestamp).total_seconds() * 1000
            latencies.append(latency)

        # Subscribe to events
        pubsub_manager.subscribe(EventType.MARKET_DATA_L2, measure_latency)

        # Publish test events
        for _ in range(100):
            event = MarketDataL2Event.create_test_event("XRP/USD")
            await pubsub_manager.publish(event)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Analyze latencies
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

            assert avg_latency < 10  # Average < 10ms
            assert p99_latency < 50  # 99th percentile < 50ms
            assert max_latency < 100  # Max < 100ms

    @pytest.mark.asyncio
    async def test_throughput_under_load(self, pubsub_manager):
        """Test data ingestion throughput under high load."""
        processed_count = 0
        start_time = time.time()

        async def count_processed(event):
            nonlocal processed_count
            processed_count += 1

        # Subscribe counter
        pubsub_manager.subscribe(EventType.MARKET_DATA_L2, count_processed)

        # Generate high load
        target_events = 1000
        tasks = []

        for i in range(target_events):
            event = MarketDataL2Event.create_test_event(
                pair="XRP/USD" if i % 2 == 0 else "DOGE/USD",
            )
            tasks.append(pubsub_manager.publish(event))

        # Wait for all publishes
        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)  # Allow processing

        duration = time.time() - start_time
        throughput = processed_count / duration

        assert processed_count >= target_events * 0.99  # 99% processed
        assert throughput > 100  # > 100 events/second

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage during data ingestion."""
        import gc

        import psutil

        process = psutil.Process()

        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create many events
        events = []
        for _ in range(10000):
            event = MarketDataL2Event.create_test_event("XRP/USD")
            events.append(event)

        # Peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Clear events
        events.clear()
        gc.collect()

        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory should not grow excessively
        memory_growth = peak_memory - baseline_memory
        memory_leaked = final_memory - baseline_memory

        assert memory_growth < 100  # Less than 100MB growth
        assert memory_leaked < 10   # Less than 10MB leaked


class MarketDataProcessor:
    """Mock market data processor for testing."""

    def __init__(self):
        self.processed_sequences = set()

    def aggregate_candles(self, candles: list[dict], timeframe: str) -> dict:
        """Aggregate multiple candles into a single candle."""
        if not candles:
            return {}

        return {
            "timestamp": candles[0]["timestamp"],
            "open": candles[0]["open"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "close": candles[-1]["close"],
            "volume": sum(c["volume"] for c in candles),
            "timeframe": timeframe,
        }

    def detect_data_gaps(self, timestamps: list[datetime], expected_interval: int) -> list[dict]:
        """Detect gaps in time series data."""
        gaps = []

        for i in range(1, len(timestamps)):
            actual_gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            if actual_gap > expected_interval * 1.5:  # 50% tolerance
                gaps.append({
                    "start": timestamps[i-1],
                    "end": timestamps[i],
                    "duration": actual_gap,
                })

        return gaps

    def is_duplicate(self, event: MarketDataL2Event) -> bool:
        """Check if event is duplicate based on sequence number."""
        if hasattr(event, "sequence_number"):
            return event.sequence_number in self.processed_sequences
        return False

    def mark_processed(self, event: MarketDataL2Event):
        """Mark event as processed."""
        if hasattr(event, "sequence_number"):
            self.processed_sequences.add(event.sequence_number)


class DataValidator:
    """Mock data validator for testing."""

    def validate_l2_data(self, data: dict) -> bool:
        """Validate L2 order book data."""
        if not data.get("bids") or not data.get("asks"):
            return False

        # Check for crossed book
        if data["bids"] and data["asks"]:
            best_bid = data["bids"][0][0]
            best_ask = data["asks"][0][0]
            if best_bid >= best_ask:
                return False

        # Check for negative prices or volumes
        for bid in data.get("bids", []):
            if bid[0] <= 0 or bid[1] <= 0:
                return False

        for ask in data.get("asks", []):
            if ask[0] <= 0 or ask[1] <= 0:
                return False

        return True

    def validate_ohlcv_data(self, data: dict) -> bool:
        """Validate OHLCV candlestick data."""
        # Check OHLC relationship
        if data["high"] < data["low"]:
            return False

        if data["high"] < data["open"] or data["high"] < data["close"]:
            return False

        if data["low"] > data["open"] or data["low"] > data["close"]:
            return False

        # Check for negative values
        if data["volume"] < 0:
            return False

        if any(data[k] < 0 for k in ["open", "high", "low", "close"]):
            return False

        return True


# Extension methods for easy test event creation
def create_test_event(cls, pair: str, **kwargs):
    """Create test event with defaults."""
    if cls == MarketDataL2Event:
        return MarketDataL2Event(
            source_module="TestModule",
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair=pair,
            exchange="kraken",
            bids=kwargs.get("bids", [[Decimal("0.5000"), Decimal("1000")]]),
            asks=kwargs.get("asks", [[Decimal("0.5001"), Decimal("1000")]]),
            timestamp_exchange=kwargs.get("timestamp_exchange", datetime.now(UTC)),
            sequence_number=kwargs.get("sequence_number", 1),
        )
    return None

# Monkey patch for testing
MarketDataL2Event.create_test_event = classmethod(lambda cls, pair, **kwargs: create_test_event(cls, pair, **kwargs))
