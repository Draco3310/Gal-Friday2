"""Integration tests for FeatureEngine."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import asyncio
import numpy as np  # For np.nan
import pandas as pd  # For type hints and creating Series inputs if needed by pipelines
import pytest

from gal_friday.core.events import EventType
from gal_friday.feature_engine import FeatureEngine

# --- Mocks and Helpers ---

class MockPubSubManager:
    def __init__(self):
        self.subscriptions = {}
        self.published_events = []

    def subscribe(self, event_type: EventType, handler):
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        self.subscriptions[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler):
        if event_type in self.subscriptions:
            self.subscriptions[event_type].remove(handler)

    async def publish(self, event: dict):
        self.published_events.append(event)
        event_type_str = event.get("event_type")
        if event_type_str:
            # Attempt to convert string to EventType enum member if necessary
            try:
                event_type_enum = EventType[event_type_str]
                if event_type_enum in self.subscriptions:
                    for handler in self.subscriptions[event_type_enum]:
                        # Check if handler is async, FeatureEngine methods are async
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event) # Should not happen for FeatureEngine
            except KeyError:
                # Handle cases where event_type_str is not a valid EventType member name
                pass # Or log a warning

    def get_last_published_event(self):
        return self.published_events[-1] if self.published_events else None

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def mock_pubsub():
    return MockPubSubManager()

@pytest.fixture
def sample_feature_config() -> dict:
    """Provides a sample feature configuration for the FeatureEngine."""
    return {
        "rsi_14": {
            "calculator_type": "rsi", "input_type": "close_series", "category": "TECHNICAL",
            "parameters": {"period": 14},
            "imputation": {"strategy": "constant", "fill_value": 50.0}, # Output imputation
            "scaling": {"method": "minmax", "feature_range": (0, 100)},
        },
        "macd_default": {
            "calculator_type": "macd", "input_type": "close_series", "category": "TECHNICAL",
            # Uses default params: 12, 26, 9
            "imputation": None, # Default (fillna(0) for MACD)
            "scaling": "passthrough",
        },
        "l2_spread_test": {
            "calculator_type": "l2_spread", "input_type": "l2_book_series", "category": "L2_ORDER_BOOK",
            "imputation": {"strategy": "mean"}, # Fill NaNs with mean of spread values
            "scaling": None, # Default (StandardScaler)
        },
        "vol_delta_60s": {
            "calculator_type": "volume_delta", "input_type": "trades_and_bar_starts", "category": "TRADE_DATA",
            "parameters": {"bar_interval_seconds": 60},
            "imputation": "passthrough", # Test passthrough, expect 0.0 for no trades or actual delta
        },
    }

@pytest.fixture
async def initialized_feature_engine(sample_feature_config, mock_pubsub, mock_logger):
    """Creates and starts a FeatureEngine instance."""
    config = {
        "exchange_name": "test_exchange",
        "feature_engine": {"trade_history_maxlen": 200}, # Increased for more trade history
        "features": sample_feature_config,
    }
    engine = FeatureEngine(config=config, pubsub_manager=mock_pubsub, logger_service=mock_logger)
    await engine.start() # Subscribes to events if FeatureEngine does that on start
    yield engine
    await engine.stop() # Cleans up subscriptions

def create_ohlcv_event(trading_pair: str, timestamp: datetime, o: float, h: float, l: float, c: float, v: float) -> dict:
    return {
        "event_type": EventType.MARKET_DATA_OHLCV.name, # Ensure name for string comparison
        "payload": {
            "trading_pair": trading_pair,
            "exchange": "test_exchange",
            "timestamp_bar_start": timestamp.isoformat().replace("+00:00", "Z"),
            "open": str(o), "high": str(h), "low": str(l), "close": str(c), "volume": str(v),
        },
    }

def create_l2_event(trading_pair: str, timestamp: datetime, bids: list, asks: list) -> dict:
    return {
        "event_type": EventType.MARKET_DATA_L2.name,
        "payload": {
            "trading_pair": trading_pair,
            "exchange": "test_exchange",
            "timestamp_exchange": timestamp.isoformat().replace("+00:00", "Z"),
            "bids": [[str(p), str(v)] for p, v in bids],
            "asks": [[str(p), str(v)] for p, v in asks],
        },
    }

def create_trade_event(trading_pair: str, timestamp: datetime, price: Decimal, volume: Decimal, side: str) -> dict:
    return {
        "event_type": EventType.MARKET_DATA_TRADE.name,
        "payload": {
            "trading_pair": trading_pair,
            "exchange": "test_exchange",
            "timestamp_exchange": timestamp.isoformat().replace("+00:00", "Z"),
            "price": str(price),
            "volume": str(volume),
            "side": side,
        },
    }

@pytest.mark.asyncio
async def test_end_to_end_feature_calculation_single_bar(initialized_feature_engine, mock_pubsub):
    engine = initialized_feature_engine
    trading_pair = "BTC/USD"

    # --- Setup Data History ---
    # Populate some history for OHLCV (at least > min_history_required)
    base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
    for i in range(30): # Create 30 bars of history
        ts = base_time - timedelta(minutes=(30-i))
        # Use direct call to _handle_ohlcv_update to bypass pubsub for setup data
        engine._handle_ohlcv_update(trading_pair, create_ohlcv_event(
            trading_pair, ts, 100+i, 102+i, 99+i, 101+i, 10+i,
        )["payload"])

    # L2 data (send one snapshot, will be used as "latest" for the target bar)
    l2_ts = base_time + timedelta(minutes=1, seconds=50) # Just before OHLCV bar close
    l2_event = create_l2_event(trading_pair, l2_ts,
                               bids=[[Decimal("130.00"), Decimal("1.0")], [Decimal("129.95"), Decimal("2.0")]],
                               asks=[[Decimal("130.05"), Decimal("0.5")], [Decimal("130.10"), Decimal("1.5")]])
    engine._handle_l2_update(trading_pair, l2_event["payload"])


    # Trade data
    trade_ts1 = base_time + timedelta(minutes=1, seconds=10) # Within the target OHLCV bar
    trade_ts2 = base_time + timedelta(minutes=1, seconds=20)
    trade1 = create_trade_event(trading_pair, trade_ts1, Decimal("130.02"), Decimal("0.5"), "buy")
    trade2 = create_trade_event(trading_pair, trade_ts2, Decimal("130.04"), Decimal("0.3"), "sell")
    # Use direct call to _handle_trade_event (it's async)
    await engine._handle_trade_event(trade1)
    await engine._handle_trade_event(trade2)

    # --- Trigger Feature Calculation ---
    # This is the OHLCV bar that should trigger feature calculation for t=base_time + 1 minute
    target_bar_ts = base_time + timedelta(minutes=1)
    ohlcv_event = create_ohlcv_event(trading_pair, target_bar_ts, 130, 130.5, 129.5, 130.2, 2.0)

    # Ensure pubsub is clear before the main event
    mock_pubsub.published_events.clear()

    await engine.process_market_data(ohlcv_event) # This should trigger _calculate_and_publish_features

    # --- Assertions ---
    assert len(mock_pubsub.published_events) == 1
    published_event = mock_pubsub.get_last_published_event()

    assert published_event is not None
    assert published_event["event_type"] == EventType.FEATURES_CALCULATED.name
    payload = published_event["payload"]
    assert payload["trading_pair"] == trading_pair
    assert payload["timestamp_features_for"] == target_bar_ts.isoformat().replace("+00:00", "Z")

    features = payload["features"]

    # Check presence of all configured features (names are based on pipeline_name.replace('_pipeline',''))
    assert "rsi_14" in features # From rsi_14_custom_pipeline
    assert "macd_default_MACD_12_26_9" in features # From macd_default_pipeline + column name
    assert "macd_default_MACDH_12_26_9" in features
    assert "macd_default_MACDS_12_26_9" in features
    assert "l2_spread_test_abs_spread" in features # From l2_spread_test_pipeline + column name
    assert "l2_spread_test_pct_spread" in features
    assert "vol_delta_60s" in features # From vol_delta_60s_pipeline (series name)

    # Spot check some values (approximate, as exact values depend on full history and TA lib)
    # RSI (scaled 0-100)
    rsi_val = float(features["rsi_14"])
    assert 0 <= rsi_val <= 100

    # MACD (passthrough scaling, so raw values from ta lib, then fillna(0))
    # If the history is somewhat trending, MACD values should not be extreme NaNs after fillna(0)
    assert pd.notna(float(features["macd_default_MACD_12_26_9"]))

    # L2 Spread (abs_spread for the single L2 book provided)
    # Bids: [[D("130.00"), D("1.0")], [D("129.95"), D("2.0")]]
    # Asks: [[D("130.05"), D("0.5")], [D("130.10"), D("1.5")]]
    # Best bid = 130.00, Best ask = 130.05. Spread = 0.05
    expected_abs_spread = 0.05
    assert np.isclose(float(features["l2_spread_test_abs_spread"]), expected_abs_spread)

    # Volume Delta (buy 0.5, sell 0.3 for the bar) -> Delta = 0.2
    expected_vol_delta = 0.2
    assert np.isclose(float(features["vol_delta_60s"]), expected_vol_delta)


@pytest.mark.asyncio
async def test_feature_calculation_with_passthrough_imputation(initialized_feature_engine, mock_pubsub):
    engine = initialized_feature_engine # Uses sample_feature_config
    trading_pair = "ETH/USD"

    # Configure one feature specifically for passthrough imputation (e.g., a new RSI)
    # This requires modifying the engine's config or creating a new engine for this test.
    # For simplicity, we'll rely on 'vol_delta_60s' which has 'passthrough' imputation.
    # If it results in NaN (e.g. no trades), it should be filtered out before formatting.
    # If it's 0.0 (no trades), it should be "0.00000000".

    base_time = datetime(2023, 1, 1, 10, 0, 0, tzinfo=UTC)
    for i in range(30):
        ts = base_time - timedelta(minutes=(30-i))
        engine._handle_ohlcv_update(trading_pair, create_ohlcv_event(
            trading_pair, ts, 10+i, 12+i, 9+i, 11+i, 5+i,
        )["payload"])

    # No trades for this bar for vol_delta
    target_bar_ts = base_time + timedelta(minutes=1)
    ohlcv_event = create_ohlcv_event(trading_pair, target_bar_ts, 130, 130.5, 129.5, 130.2, 2.0)

    mock_pubsub.published_events.clear()
    await engine.process_market_data(ohlcv_event)

    published_event = mock_pubsub.get_last_published_event()
    assert published_event is not None
    features = published_event["payload"]["features"]

    # Volume Delta with passthrough imputation and no trades for this bar should result in 0.0
    # (as per _pipeline_compute_volume_delta's logic for empty relevant_trades)
    # and then formatted to "0.00000000"
    assert "vol_delta_60s" in features
    assert features["vol_delta_60s"] == "0.00000000"

    # Test a case where a feature might produce NaN and passthrough allows it
    # We need a feature that can produce NaN from calculation (e.g. RSI with insufficient data)
    # and is configured with passthrough. Let's assume 'macd_default' scaling is passthrough.
    # If input data for MACD is too short (not the case here due to 30 bars), it would be NaN.
    # Here, 'macd_default' has default imputation (fillna(0)), so its values won't be NaN string.
    # This test mostly confirms that the passthrough for vol_delta works as expected (0.0 not NaN).
    # A more targeted test for NaN passthrough would involve a feature producing NaN
    # and configured with "imputation": "passthrough".
    # For example, if l2_spread_test had passthrough and no l2 book was set, it would be NaN.
    # The current l2_spread_test uses mean imputation, so it won't show raw NaNs.

    # Let's simulate no L2 book for l2_spread_test, which has mean imputation.
    # It should be imputed, not missing.
    engine.l2_books.pop(trading_pair, None) # Remove L2 book
    mock_pubsub.published_events.clear()
    await engine.process_market_data(ohlcv_event) # Re-process
    published_event_no_l2 = mock_pubsub.get_last_published_event()
    features_no_l2 = published_event_no_l2["payload"]["features"]

    # l2_spread_test has mean imputation. Since there's no data to compute a mean from (single event processing),
    # its fillna(df.mean()) will result in NaNs for the means themselves, so the output is NaN.
    # This would then be filtered out by `if pd.notna(val)` before formatting.
    assert "l2_spread_test_abs_spread" not in features_no_l2 # Because it became NaN and was filtered
    assert "l2_spread_test_pct_spread" not in features_no_l2
