"""Core test fixtures for the Gal-Friday trading system.

This module defines the complete test fixtures used for testing, including
market data events, trading signals, and mock services.
"""

from datetime import UTC, datetime
from decimal import Decimal
import uuid

import asyncio
import pytest

from gal_friday.core.pubsub import PubSubManager


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "trading": {
            "pairs": ["XRP/USD", "DOGE/USD"],
            "mode": "paper",
        },
        "risk": {
            "limits": {
                "max_total_drawdown_pct": 15.0,
                "max_daily_drawdown_pct": 2.0,
                "max_consecutive_losses": 5,
            },
            "sizing": {
                "risk_per_trade_pct": 0.5,
            },
        },
        "monitoring": {
            "check_interval_seconds": 1,  # Fast for tests
            "max_data_staleness_seconds": 5,
            "max_api_errors_per_minute": 10,
            "max_volatility_threshold": 5.0,
            "halt": {
                "position_behavior": "close",
            },
        },
        "exchange": {
            "name": "kraken",
            "api_url": "https://api.kraken.com",
            "rate_limit": {
                "private_calls_per_second": 1,
                "public_calls_per_second": 1,
            },
        },
    }


@pytest.fixture
async def pubsub_manager():
    """Create a test PubSubManager instance."""
    pubsub = PubSubManager()
    await pubsub.start()
    yield pubsub
    await pubsub.stop()


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    class MockLogger:
        def __init__(self):
            self.messages = []

        def log(self, level, message, **kwargs):
            self.messages.append({
                "level": level,
                "message": message,
                "kwargs": kwargs,
            })

        def info(self, message, **kwargs):
            self.log("INFO", message, **kwargs)

        def warning(self, message, **kwargs):
            self.log("WARNING", message, **kwargs)

        def error(self, message, **kwargs):
            self.log("ERROR", message, **kwargs)

        def critical(self, message, **kwargs):
            self.log("CRITICAL", message, **kwargs)

        def debug(self, message, **kwargs):
            self.log("DEBUG", message, **kwargs)

        def exception(self, message, **kwargs):
            self.log("ERROR", message, **kwargs)

        async def log_timeseries(
            self,
            measurement: str,
            tags: dict[str, str],
            fields: dict[str, object],
            timestamp: datetime | None = None,
        ) -> None:
            self.log(
                "TS",
                measurement,
                tags=tags,
                fields=fields,
                timestamp=timestamp,
            )

    return MockLogger()


@pytest.fixture
def sample_market_data():
    """Provide sample market data for tests."""
    return {
        "XRP/USD": {
            "bids": [
                (Decimal("0.5000"), Decimal(1000)),
                (Decimal("0.4999"), Decimal(2000)),
                (Decimal("0.4998"), Decimal(1500)),
            ],
            "asks": [
                (Decimal("0.5001"), Decimal(1000)),
                (Decimal("0.5002"), Decimal(2000)),
                (Decimal("0.5003"), Decimal(1500)),
            ],
            "timestamp": datetime.now(UTC),
        },
        "DOGE/USD": {
            "bids": [
                (Decimal("0.0800"), Decimal(10000)),
                (Decimal("0.0799"), Decimal(20000)),
                (Decimal("0.0798"), Decimal(15000)),
            ],
            "asks": [
                (Decimal("0.0801"), Decimal(10000)),
                (Decimal("0.0802"), Decimal(20000)),
                (Decimal("0.0803"), Decimal(15000)),
            ],
            "timestamp": datetime.now(UTC),
        },
    }


@pytest.fixture
def mock_config_manager(test_config):
    """Create a mock ConfigManager for testing."""
    class MockConfigManager:
        def __init__(self, config):
            self.config = config

        def get(self, key, default=None):
            """Get configuration value by dot-separated key."""
            parts = key.split(".")
            value = self.config
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            return value

        def get_int(self, key, default=None):
            """Get integer configuration value."""
            value = self.get(key, default)
            return int(value) if value is not None else default

        def get_float(self, key, default=None):
            """Get float configuration value."""
            value = self.get(key, default)
            return float(value) if value is not None else default

        def get_decimal(self, key, default=None):
            """Get decimal configuration value."""
            value = self.get(key, default)
            return Decimal(str(value)) if value is not None else (Decimal(default) if default else None)

        def get_list(self, key, default=None):
            """Get list configuration value."""
            value = self.get(key, default)
            return value if isinstance(value, list) else (default or [])

    return MockConfigManager(test_config)


@pytest.fixture
def mock_portfolio_state():
    """Provide mock portfolio state for testing."""
    return {
        "total_drawdown_pct": Decimal("1.5"),
        "daily_drawdown_pct": Decimal("0.5"),
        "current_equity": Decimal(98500),
        "initial_equity": Decimal(100000),
        "positions": {
            "XRP/USD": {
                "quantity": Decimal(1000),
                "side": "BUY",
                "entry_price": Decimal("0.5000"),
                "current_price": Decimal("0.5050"),
                "unrealized_pnl": Decimal("50.00"),
            },
        },
    }


@pytest.fixture
def sample_trade_signal():
    """Provide sample trade signal for testing."""
    from gal_friday.core.events import TradeSignalProposedEvent

    return TradeSignalProposedEvent(
        source_module="TestModule",
        event_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        signal_id=uuid.uuid4(),
        trading_pair="XRP/USD",
        exchange="kraken",
        side="BUY",
        entry_type="LIMIT",
        proposed_entry_price=Decimal("0.5000"),
        proposed_sl_price=Decimal("0.4900"),
        proposed_tp_price=Decimal("0.5200"),
        strategy_id="test_strategy",
    )


@pytest.fixture
def mock_exchange_api():
    """Create a mock exchange API for testing."""
    class MockKrakenAPI:
        def __init__(self):
            self.orders = {}
            self.balances = {
                "USD": Decimal(100000),
                "XRP": Decimal(0),
                "DOGE": Decimal(0),
            }
            self.order_counter = 0

        async def add_order(self, params):
            """Mock order placement."""
            order_id = f"TEST-{self.order_counter}"
            self.order_counter += 1

            self.orders[order_id] = {
                "status": "open",
                "pair": params["pair"],
                "type": params["type"],
                "ordertype": params["ordertype"],
                "volume": params["volume"],
                "price": params.get("price"),
                "timestamp": datetime.now(UTC),
            }

            return {"error": [], "result": {"txid": [order_id]}}

        async def query_orders(self, txid):
            """Mock order query."""
            if txid in self.orders:
                return {"error": [], "result": {txid: self.orders[txid]}}
            return {"error": ["Order not found"]}

        async def cancel_order(self, txid):
            """Mock order cancellation."""
            if txid in self.orders:
                self.orders[txid]["status"] = "canceled"
                return {"error": [], "result": {"count": 1}}
            return {"error": ["Order not found"]}

        async def get_asset_pairs(self):
            """Mock asset pairs info."""
            return {
                "error": [],
                "result": {
                    "XXRPZUSD": {
                        "altname": "XRPUSD",
                        "wsname": "XRP/USD",
                        "base": "XXRP",
                        "quote": "ZUSD",
                        "pair_decimals": 4,
                        "lot_decimals": 0,
                        "ordermin": "10",
                    },
                },
            }

    return MockKrakenAPI()


@pytest.fixture
def mock_prediction_model():
    """Create a mock ML model for testing."""
    class MockPredictionModel:
        def __init__(self, default_prediction=0.5):
            self.default_prediction = default_prediction
            self.prediction_count = 0

        def predict(self, features):
            """Generate mock predictions."""
            self.prediction_count += 1

            # Add some variation based on features
            base = self.default_prediction
            if "rsi" in features and features["rsi"] < 30:
                base += 0.1  # Oversold, likely to go up
            elif "rsi" in features and features["rsi"] > 70:
                base -= 0.1  # Overbought, likely to go down

            return min(max(base, 0.0), 1.0)

    return MockPredictionModel()


@pytest.fixture
def integrated_system(mock_config_manager, pubsub_manager, mock_logger):
    """Create an integrated test system with all components."""
    class IntegratedTestSystem:
        def __init__(self, config, pubsub, logger):
            self.config = config
            self.pubsub = pubsub
            self.logger = logger
            self.generated_signals = []
            self.approved_signals = []
            self.placed_orders = []

        async def inject_market_data(self, data):
            """Inject market data into the system."""
            # Implementation would publish market data events

        async def inject_prediction(self, predictions):
            """Inject prediction data into the system."""
            # Implementation would publish prediction events

        def get_generated_signals(self):
            """Get list of generated trade signals."""
            return self.generated_signals

        def get_approved_signals(self):
            """Get list of approved trade signals."""
            return self.approved_signals

        def get_placed_orders(self):
            """Get list of placed orders."""
            return self.placed_orders

    return IntegratedTestSystem(mock_config_manager, pubsub_manager, mock_logger)
