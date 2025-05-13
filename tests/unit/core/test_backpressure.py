"""Unit tests for the backpressure handling in the PubSub system."""

import datetime
import uuid
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import EventType, MarketDataL2Event, PredictionEvent, SystemStateEvent
from gal_friday.core.pubsub import BackpressureStrategy, PubSubManager, SimpleThresholdBackpressure


class CustomBackpressureStrategy(BackpressureStrategy):
    """Custom backpressure strategy for testing."""

    def __init__(self, allowed_event_types=None):
        """Initialize the custom backpressure strategy.

        Args
        ----
            allowed_event_types: List of event types that should be accepted
        """
        self.allowed_event_types = allowed_event_types or []
        self.called_with = []

    async def should_accept(self, event_type: EventType, queue_size: int) -> bool:
        """Record calls and accept only allowed event types."""
        self.called_with.append((event_type, queue_size))
        return event_type in self.allowed_event_types


class TestBackpressureHandling:
    """Test suite for backpressure handling in PubSubManager."""

    def __init__(self) -> None:
        """Initialize the test suite with an empty events list."""
        self.events: List[Any] = []

    @pytest.fixture
    def config_manager(self):
        """Fixture for a config manager with test values."""
        # Create a mock logger
        logger = MagicMock()

        # Patch the ConfigManager.get method to return our test values
        config_mgr = ConfigManager(logger_service=logger)

        # Mock the get methods to return our test values
        config_mgr.get_bool = MagicMock(return_value=True)
        config_mgr.get_int = MagicMock(
            side_effect=lambda key, default=0: {
                "pubsub.backpressure.critical_threshold": 100,
                "pubsub.backpressure.high_threshold": 75,
                "pubsub.backpressure.medium_threshold": 50,
                "pubsub.backpressure.low_threshold": 25,
                "pubsub.backpressure.default_threshold": 30,
                "pubsub.metrics_log_interval_s": 0,
            }.get(key, default)
        )
        config_mgr.get_float = MagicMock(return_value=0.5)

        return config_mgr

    @pytest.fixture
    def logger_mock(self):
        """Mock logger for testing."""
        logger = MagicMock()
        return logger

    @pytest.fixture
    async def pubsub_manager(self, logger_mock, config_manager):
        """Create and start a PubSubManager instance for testing."""
        manager = PubSubManager(logger_mock, config_manager)

        # Set up the backpressure strategy directly
        strategy = SimpleThresholdBackpressure(config_manager)
        strategy.thresholds = {
            9: 100,  # SYSTEM_STATE_CHANGE (9)
            8: 75,  # EXECUTION_REPORT (8)
            6: 75,  # TRADE_SIGNAL_APPROVED (6)
            5: 50,  # TRADE_SIGNAL_PROPOSED (5)
            4: 50,  # PREDICTION_GENERATED (4)
            3: 25,  # FEATURES_CALCULATED (3)
            2: 25,  # MARKET_DATA_OHLCV (2)
            1: 25,  # MARKET_DATA_L2 (1)
        }
        strategy.default_threshold = 30
        manager._backpressure_strategy = strategy

        # Ensure backpressure is enabled
        manager._enable_backpressure = True

        await manager.start()
        yield manager
        await manager.stop()

    @pytest.fixture
    def critical_event(self):
        """Create a critical system state event for testing."""
        return SystemStateEvent(
            source_module="test",
            event_id=uuid.uuid4(),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            new_state="HALTED",
            reason="Test reason",
        )

    @pytest.fixture
    def low_priority_event(self):
        """Create a low priority market data event for testing."""
        return MarketDataL2Event(
            source_module="test",
            event_id=uuid.uuid4(),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            trading_pair="BTC/USD",
            exchange="test_exchange",
            bids=[("30000", "1.0")],
            asks=[("30100", "1.0")],
            is_snapshot=True,
        )

    @pytest.mark.asyncio
    async def test_simple_threshold_backpressure(self, config_manager):
        """Test that SimpleThresholdBackpressure correctly applies thresholds."""
        # Create a strategy with predefined thresholds without relying on ConfigManager
        strategy = SimpleThresholdBackpressure(config_manager)

        # Directly set the thresholds dictionary instead of using config_manager's methods
        # Thresholds based on actual EventType.value values
        strategy.thresholds = {
            9: 100,  # SYSTEM_STATE_CHANGE (9)
            8: 75,  # EXECUTION_REPORT (8)
            6: 75,  # TRADE_SIGNAL_APPROVED (6)
            5: 50,  # TRADE_SIGNAL_PROPOSED (5)
            4: 50,  # PREDICTION_GENERATED (4)
            3: 25,  # FEATURES_CALCULATED (3)
            2: 25,  # MARKET_DATA_OHLCV (2)
            1: 25,  # MARKET_DATA_L2 (1)
        }
        strategy.default_threshold = 30

        # Critical event (SYSTEM_STATE_CHANGE) should be accepted even with high queue size
        assert await strategy.should_accept(EventType.SYSTEM_STATE_CHANGE, 90) is True
        assert await strategy.should_accept(EventType.SYSTEM_STATE_CHANGE, 110) is False

        # Medium priority event (PREDICTION_GENERATED)
        assert await strategy.should_accept(EventType.PREDICTION_GENERATED, 40) is True
        assert await strategy.should_accept(EventType.PREDICTION_GENERATED, 60) is False

        # Low priority event (MARKET_DATA_L2)
        assert await strategy.should_accept(EventType.MARKET_DATA_L2, 20) is True
        assert await strategy.should_accept(EventType.MARKET_DATA_L2, 30) is False

    @pytest.mark.asyncio
    async def test_backpressure_enabled_rejects_events(self, pubsub_manager, low_priority_event):
        """Test that events are rejected when queue size exceeds threshold."""
        # Patch the queue size to simulate a full queue
        with patch.object(pubsub_manager._event_queue, "qsize", return_value=50):
            # This low priority event should be rejected (threshold is 25)
            result = await pubsub_manager.publish(low_priority_event)

            assert result is False
            assert pubsub_manager._events_rejected_count == 1

    @pytest.mark.asyncio
    async def test_backpressure_disabled_accepts_all_events(
        self, pubsub_manager, low_priority_event, config_manager
    ):
        """Test that events are accepted when backpressure is disabled."""
        # Disable backpressure
        with patch.object(config_manager, "get_bool", return_value=False):
            pubsub_manager._enable_backpressure = False

            # Patch the queue size to simulate a full queue
            with patch.object(pubsub_manager._event_queue, "qsize", return_value=100):
                # Even with a full queue, event should be accepted when backpressure is disabled
                result = await pubsub_manager.publish(low_priority_event)

                assert result is True
                assert pubsub_manager._events_rejected_count == 0

    @pytest.mark.asyncio
    async def test_priority_based_acceptance(
        self, pubsub_manager, critical_event, low_priority_event
    ):
        """Test that high priority events are accepted even with moderate queue size."""
        # Patch the queue size to simulate a moderately full queue (30)
        # This is below critical threshold (100) but above low priority threshold (25)
        with patch.object(pubsub_manager._event_queue, "qsize", return_value=30):
            # Critical event should be accepted
            critical_result = await pubsub_manager.publish(critical_event)

            # Low priority event should be rejected
            low_priority_result = await pubsub_manager.publish(low_priority_event)

            assert critical_result is True
            assert low_priority_result is False
            assert pubsub_manager._events_rejected_count == 1

    @pytest.mark.asyncio
    async def test_custom_backpressure_strategy(self, logger_mock, config_manager):
        """Test that a custom backpressure strategy can be used."""
        # Create custom strategy that only accepts SYSTEM_STATE_CHANGE events
        custom_strategy = CustomBackpressureStrategy(
            allowed_event_types=[EventType.SYSTEM_STATE_CHANGE]
        )

        # Create manager with custom strategy
        manager = PubSubManager(logger_mock, config_manager)
        manager._backpressure_strategy = custom_strategy

        # Ensure backpressure is enabled
        manager._enable_backpressure = True

        await manager.start()

        try:
            # Create test events
            critical_event = SystemStateEvent(
                source_module="test",
                event_id=uuid.uuid4(),
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                new_state="HALTED",
                reason="Test reason",
            )

            prediction_event = PredictionEvent(
                source_module="test",
                event_id=uuid.uuid4(),
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                trading_pair="BTC/USD",
                exchange="test_exchange",
                timestamp_prediction_for=datetime.datetime.now(datetime.timezone.utc),
                model_id="test_model",
                prediction_target="price_direction",
                prediction_value=0.75,
            )

            # Publish events
            critical_result = await manager.publish(critical_event)
            prediction_result = await manager.publish(prediction_event)

            # Verify results
            assert critical_result is True
            assert prediction_result is False

            # Verify strategy was called with correct parameters
            assert len(custom_strategy.called_with) == 2
            assert custom_strategy.called_with[0][0] == EventType.SYSTEM_STATE_CHANGE
            assert custom_strategy.called_with[1][0] == EventType.PREDICTION_GENERATED

        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_dynamic_flow_control(self, pubsub_manager):
        """Test that dynamic flow control is applied based on processing times."""
        event_type = EventType.PREDICTION_GENERATED

        # Add slow processing times for PREDICTION_GENERATED events
        pubsub_manager._last_processing_times[event_type] = [
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]  # > 0.5s threshold

        # Patch sleep to verify it's called with expected delay
        with patch("asyncio.sleep") as mock_sleep:
            # Apply flow control
            await pubsub_manager._apply_dynamic_flow_control(event_type)

            # Verify sleep was called with a delay
            mock_sleep.assert_called_once()
            delay = mock_sleep.call_args[0][0]
            assert delay > 0  # Dynamic delay was applied
