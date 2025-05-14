"""Tests for the strategy_arbitrator module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.event_bus import MarketDataEvent, SignalEvent
from gal_friday.strategy_arbitrator import StrategyArbitrator


@pytest.fixture
def strategy_config():
    """Fixture providing strategy configuration."""
    return {
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
            "breakout": {
                "enabled": True,
                "weight": 0.3,
                "params": {"lookback_period": 30, "volume_factor": 1.5},
            },
        },
        "voting": {
            "threshold": 0.6,  # 60% confidence threshold
            "minimum_signals": 2,  # At least 2 strategies must agree
        },
    }


def test_strategy_arbitrator_initialization(strategy_config, event_bus):
    """Test that the StrategyArbitrator initializes correctly."""
    config = ConfigManager(config_dict={"strategy": strategy_config})
    arbitrator = StrategyArbitrator(config, event_bus)

    assert arbitrator is not None
    assert len(arbitrator.strategies) == 3  # Three strategies
    assert arbitrator.voting_threshold == strategy_config["voting"]["threshold"]
    assert arbitrator.minimum_signals == strategy_config["voting"]["minimum_signals"]


@patch("gal_friday.strategies.momentum_strategy.MomentumStrategy")
@patch("gal_friday.strategies.mean_reversion_strategy.MeanReversionStrategy")
@patch("gal_friday.strategies.breakout_strategy.BreakoutStrategy")
def test_strategy_arbitrator_strategy_loading(
    mock_breakout, mock_mean_reversion, mock_momentum, strategy_config, event_bus
):
    """Test loading strategies in the arbitrator."""
    # Set up mocks for strategies
    mock_momentum_instance = MagicMock()
    mock_mean_reversion_instance = MagicMock()
    mock_breakout_instance = MagicMock()

    mock_momentum.return_value = mock_momentum_instance
    mock_mean_reversion.return_value = mock_mean_reversion_instance
    mock_breakout.return_value = mock_breakout_instance

    # Create config
    config = ConfigManager(config_dict={"strategy": strategy_config})

    # Initialize arbitrator with strategy mocks
    with patch("gal_friday.strategy_arbitrator.MomentumStrategy", mock_momentum):
        with patch("gal_friday.strategy_arbitrator.MeanReversionStrategy", mock_mean_reversion):
            with patch("gal_friday.strategy_arbitrator.BreakoutStrategy", mock_breakout):
                arbitrator = StrategyArbitrator(config, event_bus)

    # Verify strategies were loaded with correct weights
    assert mock_momentum.call_count == 1
    assert mock_mean_reversion.call_count == 1
    assert mock_breakout.call_count == 1

    # Verify weights
    assert arbitrator.strategy_weights["momentum"] == 0.4
    assert arbitrator.strategy_weights["mean_reversion"] == 0.3
    assert arbitrator.strategy_weights["breakout"] == 0.3


@patch("gal_friday.strategies.momentum_strategy.MomentumStrategy")
@patch("gal_friday.strategies.mean_reversion_strategy.MeanReversionStrategy")
@patch("gal_friday.strategies.breakout_strategy.BreakoutStrategy")
def test_strategy_arbitrator_handle_market_data(
    mock_breakout, mock_mean_reversion, mock_momentum, strategy_config, event_bus
):
    """Test handling market data events in the arbitrator."""
    # Set up mocks for strategies
    mock_momentum_instance = MagicMock()
    mock_mean_reversion_instance = MagicMock()
    mock_breakout_instance = MagicMock()

    mock_momentum.return_value = mock_momentum_instance
    mock_mean_reversion.return_value = mock_mean_reversion_instance
    mock_breakout.return_value = mock_breakout_instance

    # Create config
    config = ConfigManager(config_dict={"strategy": strategy_config})

    # Create mock event_bus to capture published events
    mock_event_bus = MagicMock()

    # Initialize arbitrator with strategy mocks
    with patch("gal_friday.strategy_arbitrator.MomentumStrategy", mock_momentum):
        with patch("gal_friday.strategy_arbitrator.MeanReversionStrategy", mock_mean_reversion):
            with patch("gal_friday.strategy_arbitrator.BreakoutStrategy", mock_breakout):
                arbitrator = StrategyArbitrator(config, mock_event_bus)

    # Create a market data event
    market_data = MarketDataEvent(timestamp=datetime.now(), symbol="BTC/USD", price=50000.0)

    # Process the market data event
    arbitrator.handle_market_data(market_data)

    # Verify each strategy received the market data
    mock_momentum_instance.on_market_data.assert_called_once_with(market_data)
    mock_mean_reversion_instance.on_market_data.assert_called_once_with(market_data)
    mock_breakout_instance.on_market_data.assert_called_once_with(market_data)


@patch("gal_friday.strategies.momentum_strategy.MomentumStrategy")
@patch("gal_friday.strategies.mean_reversion_strategy.MeanReversionStrategy")
@patch("gal_friday.strategies.breakout_strategy.BreakoutStrategy")
def test_strategy_arbitrator_vote_on_signals(
    mock_breakout, mock_mean_reversion, mock_momentum, strategy_config, event_bus
):
    """Test voting mechanism for signals in the arbitrator."""
    # Set up mocks for strategies
    mock_momentum_instance = MagicMock()
    mock_mean_reversion_instance = MagicMock()
    mock_breakout_instance = MagicMock()

    mock_momentum.return_value = mock_momentum_instance
    mock_mean_reversion.return_value = mock_mean_reversion_instance
    mock_breakout.return_value = mock_breakout_instance

    # Create config
    config = ConfigManager(config_dict={"strategy": strategy_config})

    # Create mock event_bus to capture published events
    mock_event_bus = MagicMock()

    # Initialize arbitrator with strategy mocks
    with patch("gal_friday.strategy_arbitrator.MomentumStrategy", mock_momentum):
        with patch("gal_friday.strategy_arbitrator.MeanReversionStrategy", mock_mean_reversion):
            with patch("gal_friday.strategy_arbitrator.BreakoutStrategy", mock_breakout):
                arbitrator = StrategyArbitrator(config, mock_event_bus)

    # Set up strategy signals
    # Scenario 1: Strong agreement (all strategies agree)
    mock_momentum_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.8}
    }
    mock_mean_reversion_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.7}
    }
    mock_breakout_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.9}
    }

    # Request a vote for BTC/USD
    result = arbitrator.vote_on_signal("BTC/USD")

    # Verify a signal was generated with high strength
    assert result is not None
    assert result["direction"] == "BUY"
    # Should be a weighted average of the signals
    assert result["strength"] > 0.7

    # Scenario 2: Mixed signals (no clear consensus)
    mock_momentum_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.6}
    }
    mock_mean_reversion_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "SELL", "strength": 0.7}
    }
    mock_breakout_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.5}
    }

    # Request a vote for BTC/USD
    result = arbitrator.vote_on_signal("BTC/USD")

    # Verify no signal is generated due to lack of consensus
    assert result is None or result["strength"] < arbitrator.voting_threshold

    # Scenario 3: Agreement but below threshold
    mock_momentum_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.4}
    }
    mock_mean_reversion_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.3}
    }
    mock_breakout_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.5}
    }

    # Request a vote for BTC/USD
    result = arbitrator.vote_on_signal("BTC/USD")

    # Verify no signal is generated due to low strength
    assert result is None or result["strength"] < arbitrator.voting_threshold


@patch("gal_friday.strategies.momentum_strategy.MomentumStrategy")
@patch("gal_friday.strategies.mean_reversion_strategy.MeanReversionStrategy")
@patch("gal_friday.strategies.breakout_strategy.BreakoutStrategy")
def test_strategy_arbitrator_generate_signals(
    mock_breakout, mock_mean_reversion, mock_momentum, strategy_config, event_bus
):
    """Test signal generation in the arbitrator."""
    # Set up mocks for strategies
    mock_momentum_instance = MagicMock()
    mock_mean_reversion_instance = MagicMock()
    mock_breakout_instance = MagicMock()

    mock_momentum.return_value = mock_momentum_instance
    mock_mean_reversion.return_value = mock_mean_reversion_instance
    mock_breakout.return_value = mock_breakout_instance

    # Create config
    config = ConfigManager(config_dict={"strategy": strategy_config})

    # Create mock event_bus to capture published events
    mock_event_bus = MagicMock()

    # Initialize arbitrator with strategy mocks
    with patch("gal_friday.strategy_arbitrator.MomentumStrategy", mock_momentum):
        with patch("gal_friday.strategy_arbitrator.MeanReversionStrategy", mock_mean_reversion):
            with patch("gal_friday.strategy_arbitrator.BreakoutStrategy", mock_breakout):
                arbitrator = StrategyArbitrator(config, mock_event_bus)

    # Set up strategy signals for multiple symbols
    mock_momentum_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.8},
        "ETH/USD": {"direction": "SELL", "strength": 0.7},
        "SOL/USD": {"direction": "BUY", "strength": 0.3},
    }
    mock_mean_reversion_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.7},
        "ETH/USD": {"direction": "SELL", "strength": 0.6},
        "SOL/USD": {"direction": "SELL", "strength": 0.8},
    }
    mock_breakout_instance.get_current_signal.return_value = {
        "BTC/USD": {"direction": "BUY", "strength": 0.9},
        "ETH/USD": {"direction": "BUY", "strength": 0.9},
        "SOL/USD": {"direction": "BUY", "strength": 0.6},
    }

    # Set up tradable symbols
    arbitrator.symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

    # Call generate_signals
    arbitrator.generate_signals()

    # Verify event_bus.publish was called for the right signals
    # We expect:
    # - BTC/USD: Strong BUY signal published (all agree)
    # - ETH/USD: No signal published (disagreement)
    # - SOL/USD: No signal published (mixed/weak signals)

    # Count how many times publish was called
    assert mock_event_bus.publish.call_count >= 1

    # Extract the published events
    published_events = [call.args[0] for call in mock_event_bus.publish.call_args_list]

    # Check that at least one BTC/USD BUY signal was published
    btc_buy_signals = [
        event
        for event in published_events
        if isinstance(event, SignalEvent)
        and event.symbol == "BTC/USD"
        and event.direction == "BUY"
    ]

    assert len(btc_buy_signals) >= 1
