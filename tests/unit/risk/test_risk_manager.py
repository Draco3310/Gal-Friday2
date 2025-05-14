"""Tests for the risk_manager module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from gal_friday.config_manager import ConfigManager
from gal_friday.event_bus import OrderEvent, SignalEvent
from gal_friday.risk_manager import RiskManager


@pytest.fixture
def risk_config():
    """Fixture providing risk management configuration."""
    return {
        "max_position_size": 0.1,  # 10% of portfolio
        "max_single_order_size": 0.05,  # 5% of portfolio
        "stop_loss_pct": 0.05,  # 5% stop loss
        "take_profit_pct": 0.1,  # 10% take profit
        "max_open_trades": 5,
        "max_correlated_assets": 2,
        "max_drawdown": 0.15,  # 15% maximum drawdown
        "volatility_constraint": 0.25,  # 25% annualized volatility limit
        "correlation_threshold": 0.7,  # 70% correlation threshold
    }


def test_risk_manager_initialization(risk_config, event_bus):
    """Test that the RiskManager initializes correctly."""
    config = ConfigManager(config_dict={"risk": risk_config})
    risk_manager = RiskManager(config, event_bus)

    assert risk_manager is not None
    assert risk_manager.max_position_size == risk_config["max_position_size"]
    assert risk_manager.max_single_order_size == risk_config["max_single_order_size"]
    assert risk_manager.stop_loss_pct == risk_config["stop_loss_pct"]
    assert risk_manager.take_profit_pct == risk_config["take_profit_pct"]
    assert risk_manager.max_open_trades == risk_config["max_open_trades"]


def test_risk_manager_order_validation(risk_config, event_bus):
    """Test order validation by the risk manager."""
    config = ConfigManager(config_dict={"risk": risk_config})

    # Create a mock event_bus to capture published events
    mock_event_bus = MagicMock()

    # Create risk manager with mock event bus
    risk_manager = RiskManager(config, mock_event_bus)

    # Mock portfolio manager for checking positions
    mock_portfolio_manager = MagicMock()
    mock_portfolio_manager.get_risk_exposure.return_value = {
        "BTC/USD": 0.05,  # 5% of portfolio in BTC
        "ETH/USD": 0.03,  # 3% of portfolio in ETH
    }
    mock_portfolio_manager.current_holdings = {"total": 100000.0}
    risk_manager.set_portfolio_manager(mock_portfolio_manager)

    # Create a valid order (within limits)
    valid_order = OrderEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        order_type="MARKET",
        quantity=0.04,  # 4% of portfolio, below single order size limit
        direction="BUY",
    )

    # Validate the order
    result = risk_manager.validate_order(valid_order)
    assert result is True

    # Create an order that exceeds single order size limit
    large_order = OrderEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        order_type="MARKET",
        quantity=0.06,  # 6% of portfolio, above 5% single order limit
        direction="BUY",
    )

    # Validate the large order
    result = risk_manager.validate_order(large_order)
    assert result is False

    # Create an order that would exceed max position size when combined with
    # existing position
    position_limit_order = OrderEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        order_type="MARKET",
        quantity=0.06,
        # Would make BTC position 11% (5% + 6%), above 10% limit
        direction="BUY",
    )

    # Validate the position limit order
    result = risk_manager.validate_order(position_limit_order)
    assert result is False


def test_risk_manager_signal_filtering(risk_config, event_bus):
    """Test signal filtering by the risk manager."""
    config = ConfigManager(config_dict={"risk": risk_config})

    # Create a mock event_bus to capture published events
    mock_event_bus = MagicMock()

    # Create risk manager with mock event bus
    risk_manager = RiskManager(config, mock_event_bus)

    # Mock portfolio manager for checking positions
    mock_portfolio_manager = MagicMock()
    mock_portfolio_manager.get_risk_exposure.return_value = {
        "BTC/USD": 0.05,  # 5% of portfolio in BTC
        "ETH/USD": 0.03,  # 3% of portfolio in ETH
    }
    mock_portfolio_manager.current_holdings = {"total": 100000.0}
    mock_portfolio_manager.get_open_trade_count.return_value = 3  # 3 open trades
    risk_manager.set_portfolio_manager(mock_portfolio_manager)

    # Create a signal for a new asset
    new_asset_signal = SignalEvent(
        timestamp=datetime.now(),
        symbol="SOL/USD",
        signal_type="LONG",
        strength=0.8,  # Strong signal
        direction="BUY",
    )

    # Process the signal
    result = risk_manager.process_signal(new_asset_signal)
    assert result is True  # Should pass

    # Simulate max open trades reached
    mock_portfolio_manager.get_open_trade_count.return_value = 5  # Max open trades

    # Create another signal for a new asset
    another_new_signal = SignalEvent(
        timestamp=datetime.now(),
        symbol="DOT/USD",
        signal_type="LONG",
        strength=0.7,
        direction="BUY",
    )

    # Process the signal
    result = risk_manager.process_signal(another_new_signal)
    assert result is False  # Should fail due to max open trades

    # Reset open trade count
    mock_portfolio_manager.get_open_trade_count.return_value = 3

    # Create a signal for a highly correlated asset (assume BTC and ETH are correlated)
    # Mock the correlation matrix
    with patch.object(risk_manager, "get_correlation_matrix") as mock_corr:
        mock_corr.return_value = {
            ("BTC/USD", "ETH/USD"): 0.85,  # High correlation
            ("BTC/USD", "SOL/USD"): 0.45,
            ("ETH/USD", "SOL/USD"): 0.40,
        }

        # Signal for ETH when BTC already has a position
        correlated_signal = SignalEvent(
            timestamp=datetime.now(),
            symbol="ETH/USD",
            signal_type="LONG",
            strength=0.9,
            direction="BUY",
        )

        # Process the signal - this could pass or fail depending on implementation details
        # Let's assume it should fail due to correlation constraints
        result = risk_manager.process_signal(correlated_signal)
        assert result is False  # Should fail due to correlation constraints


def test_risk_manager_position_sizing(risk_config, event_bus):
    """Test position sizing by the risk manager."""
    config = ConfigManager(config_dict={"risk": risk_config})

    # Create risk manager
    risk_manager = RiskManager(config, event_bus)

    # Mock portfolio manager
    mock_portfolio_manager = MagicMock()
    mock_portfolio_manager.current_holdings = {"total": 100000.0}
    risk_manager.set_portfolio_manager(mock_portfolio_manager)

    # Test position sizing for a buy signal
    signal = SignalEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        signal_type="LONG",
        strength=0.8,
        direction="BUY",
    )

    # Get position size
    position_size = risk_manager.calculate_position_size(signal, price=50000.0)

    # Check if position size is within limits
    assert position_size > 0
    # Max position value should be 10% of portfolio
    max_position_value = 100000.0 * 0.1
    assert position_size * 50000.0 <= max_position_value

    # Test scaling position size based on signal strength
    weak_signal = SignalEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        signal_type="LONG",
        strength=0.3,  # Weak signal
        direction="BUY",
    )

    # Get position size for weak signal
    weak_position_size = risk_manager.calculate_position_size(weak_signal, price=50000.0)

    # Check if position size is reduced for weaker signal
    assert weak_position_size < position_size


def test_risk_manager_drawdown_protection(risk_config, event_bus):
    """Test drawdown protection in the risk manager."""
    config = ConfigManager(config_dict={"risk": risk_config})

    # Create risk manager
    risk_manager = RiskManager(config, event_bus)

    # Mock portfolio manager with no drawdown
    mock_portfolio_manager = MagicMock()
    mock_portfolio_manager.get_current_drawdown.return_value = 0.05  # 5% drawdown
    risk_manager.set_portfolio_manager(mock_portfolio_manager)

    # Create a signal
    signal = SignalEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        signal_type="LONG",
        strength=0.8,
        direction="BUY",
    )

    # Process the signal with low drawdown
    result = risk_manager.process_signal(signal)
    assert result is True  # Should pass with low drawdown

    # Simulate high drawdown
    # 20% drawdown, exceeds 15% limit
    mock_portfolio_manager.get_current_drawdown.return_value = 0.20

    # Process the signal with high drawdown
    result = risk_manager.process_signal(signal)
    assert result is False  # Should fail due to excessive drawdown
