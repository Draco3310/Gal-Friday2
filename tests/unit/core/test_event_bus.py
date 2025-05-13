"""Tests for the event_bus module."""

from unittest.mock import MagicMock

from gal_friday.core.events import Event
from gal_friday.event_bus import EventBus


class TestEvent(Event):
    """Test event class."""

    def __init__(self, value):
        """Initialize a test event.

        Args:
            value: The test value to store in the event
        """
        self.value = value
        super().__init__()


def test_event_bus_initialization():
    """Test that the EventBus initializes correctly."""
    event_bus = EventBus()
    assert event_bus is not None
    assert len(event_bus._subscribers) == 0


def test_event_bus_subscribe_and_publish():
    """Test subscribing to events and publishing events."""
    event_bus = EventBus()

    # Create a mock subscriber
    mock_subscriber = MagicMock()

    # Subscribe to TestEvent
    event_bus.subscribe(TestEvent, mock_subscriber)

    # Verify subscriber was added
    assert len(event_bus._subscribers.get(TestEvent, [])) == 1

    # Publish an event
    test_event = TestEvent("test_value")
    event_bus.publish(test_event)

    # Verify subscriber was called with the event
    mock_subscriber.assert_called_once_with(test_event)


def test_event_bus_unsubscribe():
    """Test unsubscribing from events."""
    event_bus = EventBus()

    # Create a mock subscriber
    mock_subscriber = MagicMock()

    # Subscribe to TestEvent
    event_bus.subscribe(TestEvent, mock_subscriber)

    # Verify subscriber was added
    assert len(event_bus._subscribers.get(TestEvent, [])) == 1

    # Unsubscribe
    event_bus.unsubscribe(TestEvent, mock_subscriber)

    # Verify subscriber was removed
    assert len(event_bus._subscribers.get(TestEvent, [])) == 0

    # Publish an event
    test_event = TestEvent("test_value")
    event_bus.publish(test_event)

    # Verify subscriber was not called
    mock_subscriber.assert_not_called()


def test_event_bus_multiple_subscribers():
    """Test multiple subscribers for the same event type."""
    event_bus = EventBus()

    # Create mock subscribers
    mock_subscriber1 = MagicMock()
    mock_subscriber2 = MagicMock()

    # Subscribe both to TestEvent
    event_bus.subscribe(TestEvent, mock_subscriber1)
    event_bus.subscribe(TestEvent, mock_subscriber2)

    # Verify subscribers were added
    assert len(event_bus._subscribers.get(TestEvent, [])) == 2

    # Publish an event
    test_event = TestEvent("test_value")
    event_bus.publish(test_event)

    # Verify both subscribers were called with the event
    mock_subscriber1.assert_called_once_with(test_event)
    mock_subscriber2.assert_called_once_with(test_event)
