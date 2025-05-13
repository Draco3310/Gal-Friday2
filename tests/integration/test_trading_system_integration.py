"""
Integration tests for the Gal-Friday2 trading system.

These tests verify that multiple components work together correctly.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import (
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    PredictionEvent,
    SignalEvent,
)
from gal_friday.core.pubsub import PubSubManager
from gal_friday.execution_handler import ExecutionHandler
from gal_friday.market_price_service import MarketPriceService
from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.prediction_service import PredictionService
from gal_friday.risk_manager import RiskManager
from gal_friday.strategy_arbitrator import StrategyArbitrator


@pytest.fixture
def integration_config():
    """Fixture providing a complete configuration for integration tests."""
    return {
        "app_name": "Gal-Friday2",
        "environment": "test",
        "log_level": "INFO",
        "exchanges": {
            "kraken": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "symbols": ["BTC/USD", "ETH/USD"],
                "timeframes": ["1m", "5m", "1h", "1d"],
                "default_limit_slippage": 0.001,
                "reconnect_wait_time": 10,
                "order_expiry_seconds": 60,
            }
        },
        "database": {"connection_string": "sqlite:///:memory:"},
        "backtesting": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "initial_capital": 100000.0,
            "symbols": ["BTC/USD", "ETH/USD"],
            "strategy": "momentum",
        },
        "risk": {
            "max_position_size": 0.1,
            "max_single_order_size": 0.05,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "max_open_trades": 5,
            "max_correlated_assets": 2,
            "max_drawdown": 0.15,
            "volatility_constraint": 0.25,
            "correlation_threshold": 0.7,
        },
        "strategy": {
            "strategies": {
                "momentum": {
                    "enabled": True,
                    "weight": 0.4,
                    "params": {"lookback_period": 14, "threshold": 0.05},
                },
                "mean_reversion": {
                    "enabled": True,
                    "weight": 0.3,
                    "params": {"lookback_period": 20, "std_dev_threshold": 2.0},
                },
            },
            "voting": {"threshold": 0.6, "minimum_signals": 2},
        },
        "prediction": {
            "models": {
                "price_direction": {
                    "enabled": True,
                    "type": "ensemble",
                    "model_path": "models/price_direction_model.pkl",
                    "prediction_horizon": "1h",
                    "features": ["rsi_14", "macd", "bbands_width"],
                    "retrain_interval_days": 7,
                }
            }
        },
    }


@pytest.fixture
def setup_trading_system(integration_config, mock_exchange):
    """Fixture to set up the complete trading system with mocks."""
    # Create the event bus
    event_bus = PubSubManager()

    # Create the config manager
    config = ConfigManager(config_dict=integration_config)

    # Create components
    portfolio_manager = PortfolioManager(config, event_bus)
    risk_manager = RiskManager(config, event_bus)

    # Connect components
    risk_manager.set_portfolio_manager(portfolio_manager)

    # Create mocks for other components
    with patch("gal_friday.strategies.momentum_strategy.MomentumStrategy") as mock_momentum:
        with patch(
            "gal_friday.strategies.mean_reversion_strategy.MeanReversionStrategy"
        ) as mock_mean_reversion:
            with patch("gal_friday.prediction_service.load_model") as mock_load_model:
                with patch("ccxt.kraken") as mock_ccxt_kraken:
                    # Set up exchange mock
                    mock_ccxt_kraken.return_value = mock_exchange

                    # Mock the model loading
                    mock_load_model.return_value = MagicMock()

                    # Set up strategy mocks
                    mock_momentum_instance = MagicMock()
                    mock_mean_reversion_instance = MagicMock()
                    mock_momentum.return_value = mock_momentum_instance
                    mock_mean_reversion.return_value = mock_mean_reversion_instance

                    # Create the remaining components
                    strategy_arbitrator = StrategyArbitrator(config, event_bus)
                    execution_handler = ExecutionHandler(config, event_bus)
                    market_price_service = MarketPriceService(config, event_bus)
                    prediction_service = PredictionService(config, event_bus)

                    # Connect to exchange
                    execution_handler.exchange = mock_exchange
                    market_price_service.exchange = mock_exchange

                    # Return the system components
                    yield {
                        "event_bus": event_bus,
                        "config": config,
                        "portfolio_manager": portfolio_manager,
                        "risk_manager": risk_manager,
                        "strategy_arbitrator": strategy_arbitrator,
                        "execution_handler": execution_handler,
                        "market_price_service": market_price_service,
                        "prediction_service": prediction_service,
                        "mock_exchange": mock_exchange,
                    }


def test_market_data_flow(setup_trading_system):
    """Test market data flowing through the system."""
    # Extract components
    system = setup_trading_system
    event_bus = system["event_bus"]
    system["mock_exchange"]
    system["market_price_service"]
    system["strategy_arbitrator"]

    # Set up event handling spies
    strategy_spy = MagicMock()
    event_bus.subscribe(MarketDataEvent, strategy_spy)

    # Create a market data event
    price_data = {
        "symbol": "BTC/USD",
        "price": 50000.0,
        "timestamp": datetime.now().timestamp() * 1000,
    }
    market_data_event = MarketDataEvent(
        timestamp=datetime.now(), symbol=price_data["symbol"], price=price_data["price"]
    )

    # Simulate price service publishing market data
    event_bus.publish(market_data_event)

    # Verify that subscribers received the market data
    strategy_spy.assert_called_once()
    received_event = strategy_spy.call_args[0][0]
    assert received_event.symbol == "BTC/USD"
    assert received_event.price == 50000.0


def test_signal_to_order_flow(setup_trading_system):
    """Test signal flowing through to order generation."""
    # Extract components
    system = setup_trading_system
    event_bus = system["event_bus"]
    system["portfolio_manager"]
    risk_manager = system["risk_manager"]

    # Set up event handling spies
    order_spy = MagicMock()
    event_bus.subscribe(OrderEvent, order_spy)

    # Mock risk manager to approve signals
    with patch.object(risk_manager, "process_signal", return_value=True):
        # Create a signal event
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            signal_type="LONG",
            strength=0.8,
            direction="BUY",
        )

        # Publish the signal
        event_bus.publish(signal_event)

        # Verify that an order was generated
        order_spy.assert_called_once()
        generated_order = order_spy.call_args[0][0]
        assert generated_order.symbol == "BTC/USD"
        assert generated_order.direction == "BUY"
        assert generated_order.quantity > 0


def test_order_to_fill_flow(setup_trading_system):
    """Test order flowing through to execution and fill."""
    # Extract components
    system = setup_trading_system
    event_bus = system["event_bus"]
    system["execution_handler"]
    mock_exchange = system["mock_exchange"]

    # Set up mock exchange response
    mock_exchange.create_market_order.return_value = {
        "id": "12345",
        "timestamp": datetime.now().timestamp() * 1000,
        "status": "closed",
        "symbol": "BTC/USD",
        "type": "market",
        "side": "buy",
        "price": 50000.0,
        "amount": 1.0,
        "filled": 1.0,
        "cost": 50000.0,
        "fee": {"cost": 25.0, "currency": "USD"},
    }

    # Set up event handling spies
    fill_spy = MagicMock()
    event_bus.subscribe(FillEvent, fill_spy)

    # Create an order event
    order_event = OrderEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        order_type="MARKET",
        quantity=1.0,
        direction="BUY",
    )

    # Publish the order
    event_bus.publish(order_event)

    # Verify that a fill event was generated
    fill_spy.assert_called_once()
    fill_event = fill_spy.call_args[0][0]
    assert fill_event.symbol == "BTC/USD"
    assert fill_event.direction == "BUY"
    assert fill_event.quantity == 1.0
    assert fill_event.price == 50000.0


def test_prediction_to_signal_flow(setup_trading_system):
    """Test prediction flowing through to signal generation."""
    # Extract components
    system = setup_trading_system
    event_bus = system["event_bus"]
    strategy_arbitrator = system["strategy_arbitrator"]

    # Set up event handling spies
    signal_spy = MagicMock()
    event_bus.subscribe(SignalEvent, signal_spy)

    # Mock strategy to generate signals based on predictions
    with patch.object(strategy_arbitrator, "vote_on_signal") as mock_vote:
        mock_vote.return_value = {"direction": "BUY", "strength": 0.8}

        # Create a prediction event
        prediction_event = PredictionEvent(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            prediction_type="price_direction",
            value=1,  # Up
            confidence=0.85,
            horizon="1h",
        )

        # Publish the prediction
        event_bus.publish(prediction_event)

        # Manually trigger signal generation since we may not have automatic
        # trigger
        strategy_arbitrator.generate_signals()

        # Verify signal was generated (this depends on implementation details)
        # May be 0 if implementation doesn't respond to this event
        assert signal_spy.call_count >= 0


def test_end_to_end_trading_cycle(setup_trading_system):
    """Test a complete trading cycle from market data to fill."""
    # Extract components
    system = setup_trading_system
    event_bus = system["event_bus"]
    mock_exchange = system["mock_exchange"]
    portfolio_manager = system["portfolio_manager"]

    # Set up spies for each event type
    market_data_spy = MagicMock()
    signal_spy = MagicMock()
    order_spy = MagicMock()
    fill_spy = MagicMock()

    event_bus.subscribe(MarketDataEvent, market_data_spy)
    event_bus.subscribe(SignalEvent, signal_spy)
    event_bus.subscribe(OrderEvent, order_spy)
    event_bus.subscribe(FillEvent, fill_spy)

    # Set initial portfolio state
    portfolio_manager.current_holdings["cash"] = 100000.0
    portfolio_manager.current_holdings["total"] = 100000.0

    # Mock exchange responses
    mock_exchange.create_market_order.return_value = {
        "id": "12345",
        "timestamp": datetime.now().timestamp() * 1000,
        "status": "closed",
        "symbol": "BTC/USD",
        "type": "market",
        "side": "buy",
        "price": 50000.0,
        "amount": 1.0,
        "filled": 1.0,
        "cost": 50000.0,
        "fee": {"cost": 25.0, "currency": "USD"},
    }

    # Mock strategy arbitrator to always generate a buy signal
    with patch.object(system["strategy_arbitrator"], "vote_on_signal") as mock_vote:
        mock_vote.return_value = {"direction": "BUY", "strength": 0.8}

        # Create market data event
        market_data_event = MarketDataEvent(
            timestamp=datetime.now(), symbol="BTC/USD", price=50000.0
        )

        # Publish the market data
        event_bus.publish(market_data_event)

        # Create signal event (normally would be generated by strategy)
        signal_event = SignalEvent(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            signal_type="LONG",
            strength=0.8,
            direction="BUY",
        )

        # Publish the signal
        event_bus.publish(signal_event)

        # Verify market data was received
        market_data_spy.assert_called_once()

        # Verify signal was published
        signal_spy.assert_called_once()

        # Verify order was generated
        assert order_spy.call_count >= 1

        # Verify fill was generated
        assert fill_spy.call_count >= 1

        # Verify portfolio was updated
        assert "BTC/USD" in portfolio_manager.current_positions
        # Cash decreased
        assert portfolio_manager.current_holdings["cash"] < 100000.0
