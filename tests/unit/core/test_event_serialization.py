"""Unit tests for event serialization and deserialization functionality."""

import datetime
import json
import uuid
from decimal import Decimal

import pytest

from gal_friday.core.events import (
    Event,
    EventType,
    FeatureEvent,
    SystemStateEvent,
    TradeSignalApprovedEvent,
)


def test_event_serialization_base_event():
    """Test serialization and deserialization for simple event types."""
    # Create a basic system state event
    event_id = uuid.uuid4()
    now = datetime.datetime.now(datetime.timezone.utc)

    original_event = SystemStateEvent(
        source_module="test_module",
        event_id=event_id,
        timestamp=now,
        new_state="RUNNING",
        reason="System starting up",
    )

    # Serialize the event
    json_str = original_event.to_json()

    # Verify JSON string contains expected values
    assert json_str is not None
    assert isinstance(json_str, str)
    assert "SystemStateEvent" in json_str
    assert "RUNNING" in json_str
    assert "System starting up" in json_str
    assert str(event_id) in json_str

    # Deserialize the event
    deserialized_event = Event.from_json(json_str)

    # Verify deserialized event is correct
    assert isinstance(deserialized_event, SystemStateEvent)
    assert deserialized_event.event_id == event_id
    assert deserialized_event.timestamp == now
    assert deserialized_event.source_module == "test_module"
    assert deserialized_event.new_state == "RUNNING"
    assert deserialized_event.reason == "System starting up"
    assert deserialized_event.event_type == EventType.SYSTEM_STATE_CHANGE


def test_event_serialization_decimal_fields():
    """Test serialization and deserialization of events with Decimal fields."""
    # Create a trade signal approved event with Decimal fields
    event_id = uuid.uuid4()
    signal_id = uuid.uuid4()
    now = datetime.datetime.now(datetime.timezone.utc)

    original_event = TradeSignalApprovedEvent(
        source_module="test_module",
        event_id=event_id,
        timestamp=now,
        signal_id=signal_id,
        trading_pair="BTC/USD",
        exchange="kraken",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("0.123456"),
        sl_price=Decimal("30000.50"),
        tp_price=Decimal("35000.25"),
        risk_parameters={"max_position_size": 0.1, "max_risk_per_trade": 0.02},
        limit_price=Decimal("32500.75"),
    )

    # Serialize the event
    json_str = original_event.to_json()

    # Verify JSON contains Decimal values encoded properly
    assert "0.123456" in json_str
    assert "30000.50" in json_str
    assert "35000.25" in json_str
    assert "32500.75" in json_str

    # Deserialize the event
    deserialized_event = Event.from_json(json_str)

    # Verify deserialized event is correct with proper Decimal values
    assert isinstance(deserialized_event, TradeSignalApprovedEvent)
    assert deserialized_event.quantity == Decimal("0.123456")
    assert deserialized_event.sl_price == Decimal("30000.50")
    assert deserialized_event.tp_price == Decimal("35000.25")
    assert deserialized_event.limit_price == Decimal("32500.75")


def test_event_serialization_complex_nested_structures():
    """Test serialization and deserialization of events with nested structures."""
    event_id = uuid.uuid4()
    timestamp = datetime.datetime.now(datetime.timezone.utc)

    original_event = FeatureEvent(
        source_module="test_module",
        event_id=event_id,
        timestamp=timestamp,
        trading_pair="ETH/USD",
        exchange="kraken",
        timestamp_features_for=timestamp,
        features={
            "rsi_14": 67.5,
            "macd": {"macd_line": 25.5, "signal_line": 20.2, "histogram": 5.3},
            "timestamps": [
                timestamp - datetime.timedelta(minutes=5),
                timestamp - datetime.timedelta(minutes=4),
                timestamp - datetime.timedelta(minutes=3),
            ],
            "related_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
            "price_data": {
                "open": Decimal("3200.50"),
                "high": Decimal("3250.25"),
                "low": Decimal("3190.75"),
                "close": Decimal("3240.00"),
            },
        },
    )

    # Serialize the event
    json_str = original_event.to_json()

    # Deserialize the event
    deserialized_event = Event.from_json(json_str)

    # Verify deserialized event is correct
    assert isinstance(deserialized_event, FeatureEvent)
    assert deserialized_event.trading_pair == "ETH/USD"
    assert deserialized_event.exchange == "kraken"

    # Check nested structures
    features = deserialized_event.features
    assert features["rsi_14"] == 67.5
    assert features["macd"]["macd_line"] == 25.5
    assert features["macd"]["signal_line"] == 20.2
    assert features["macd"]["histogram"] == 5.3

    # Check deserialized timestamps in list
    assert isinstance(features["timestamps"][0], datetime.datetime)
    assert len(features["timestamps"]) == 3

    # Check Decimal values in nested dict
    assert isinstance(features["price_data"]["open"], Decimal)
    assert features["price_data"]["open"] == Decimal("3200.50")


def test_event_serialization_error_handling():
    """Test error handling during serialization and deserialization."""
    # Test with invalid JSON string
    with pytest.raises(json.JSONDecodeError):
        Event.from_json("{invalid json")

    # Test with missing __event_type__
    with pytest.raises(ValueError, match="missing __event_type__"):
        Event.from_json('{"field": "value"}')

    # Test with unknown event type
    with pytest.raises(KeyError, match="Unknown event type"):
        Event.from_json('{"__event_type__": "NonExistentEvent"}')

    # Test with missing required fields
    json_str = '{"__event_type__": "SystemStateEvent", "new_state": "RUNNING"}'
    with pytest.raises(ValueError, match="Failed to reconstruct event"):
        Event.from_json(json_str)
