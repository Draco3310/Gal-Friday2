"""
Tests for the portfolio_manager module.
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from gal_friday.portfolio_manager import PortfolioManager
from gal_friday.event_bus import EventBus
from gal_friday.core.events import OrderEvent, FillEvent, MarketDataEvent


def test_portfolio_manager_initialization(config_manager, event_bus):
    """Test that the PortfolioManager initializes correctly."""
    portfolio = PortfolioManager(config_manager, event_bus)
    
    assert portfolio is not None
    assert portfolio.current_positions == {}
    assert portfolio.current_holdings["cash"] == config_manager.get("backtesting.initial_capital")
    assert portfolio.current_holdings["total"] == config_manager.get("backtesting.initial_capital")


def test_portfolio_manager_update_from_fill(config_manager, event_bus):
    """Test updating portfolio from a fill event."""
    portfolio = PortfolioManager(config_manager, event_bus)
    
    # Initial state
    assert portfolio.current_holdings["cash"] == 100000.0
    assert portfolio.current_holdings["total"] == 100000.0
    assert "BTC/USD" not in portfolio.current_positions
    
    # Create a fill event for buying BTC
    fill_event = FillEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        exchange="kraken",
        quantity=2.0,
        price=50000.0,
        commission=25.0,
        direction="BUY"
    )
    
    # Process the fill event
    portfolio._update_from_fill(fill_event)
    
    # Check updated positions
    assert "BTC/USD" in portfolio.current_positions
    assert portfolio.current_positions["BTC/USD"] == 2.0
    
    # Check updated holdings
    # Cash should decrease by price * quantity + commission
    expected_cash = 100000.0 - (50000.0 * 2.0) - 25.0
    assert portfolio.current_holdings["cash"] == expected_cash
    
    # Total should remain the same (minus commission) as we now hold equivalent BTC value
    expected_total = 100000.0 - 25.0
    assert abs(portfolio.current_holdings["total"] - expected_total) < 0.01
    
    # Create a fill event for selling part of BTC
    fill_event = FillEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        exchange="kraken",
        quantity=1.0,
        price=52000.0,  # Price went up
        commission=25.0,
        direction="SELL"
    )
    
    # Process the fill event
    portfolio._update_from_fill(fill_event)
    
    # Check updated positions
    assert portfolio.current_positions["BTC/USD"] == 1.0
    
    # Check updated holdings
    # Cash should increase by sell price * quantity - commission
    expected_cash += (52000.0 * 1.0) - 25.0
    assert portfolio.current_holdings["cash"] == expected_cash
    
    # Total should reflect profit from price increase minus commission
    # 1 BTC at 52000 + cash - commissions
    expected_total = (1.0 * 52000.0) + expected_cash
    assert abs(portfolio.current_holdings["total"] - expected_total) < 0.01


def test_portfolio_manager_generate_order_from_signal(config_manager, event_bus):
    """Test generating orders from signals."""
    # Create a mock event_bus to capture published events
    mock_event_bus = MagicMock()
    
    # Create portfolio with mock event bus
    portfolio = PortfolioManager(config_manager, mock_event_bus)
    
    # Set initial state
    portfolio.current_holdings["cash"] = 100000.0
    portfolio.current_holdings["total"] = 100000.0
    
    # Create a signal event (import is mocked since we don't want to depend on actual implementation)
    with patch('gal_friday.core.events.SignalEvent') as MockSignalEvent:
        signal_event = MockSignalEvent.return_value
        signal_event.symbol = "BTC/USD"
        signal_event.direction = "BUY"
        signal_event.suggested_quantity = None  # Let portfolio calculate quantity
        
        # Generate order from signal
        portfolio.generate_order_from_signal(signal_event)
        
        # Verify order was published
        assert mock_event_bus.publish.call_count == 1
        published_event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(published_event, OrderEvent)
        assert published_event.symbol == "BTC/USD"
        assert published_event.direction == "BUY"
        assert published_event.quantity > 0


def test_portfolio_manager_update_value(config_manager, event_bus):
    """Test updating portfolio value based on market prices."""
    portfolio = PortfolioManager(config_manager, event_bus)
    
    # Set initial state with positions
    portfolio.current_positions = {
        "BTC/USD": 2.0,
        "ETH/USD": 10.0
    }
    portfolio.current_holdings = {
        "cash": 20000.0,
        "total": 100000.0  # Will be recalculated
    }
    
    # Create market price updates
    btc_price_event = MarketDataEvent(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        price=55000.0
    )
    
    eth_price_event = MarketDataEvent(
        timestamp=datetime.now(),
        symbol="ETH/USD",
        price=3500.0
    )
    
    # Update portfolio values
    portfolio.update_value(btc_price_event)
    portfolio.update_value(eth_price_event)
    
    # Calculate expected total
    expected_total = 20000.0 + (2.0 * 55000.0) + (10.0 * 3500.0)
    
    # Check updated total value
    assert abs(portfolio.current_holdings["total"] - expected_total) < 0.01


def test_portfolio_manager_risk_exposure(config_manager, event_bus):
    """Test calculating risk exposure of portfolio."""
    portfolio = PortfolioManager(config_manager, event_bus)
    
    # Set initial state with positions
    portfolio.current_positions = {
        "BTC/USD": 2.0,
        "ETH/USD": 10.0
    }
    
    # Set current market prices
    portfolio.current_prices = {
        "BTC/USD": 50000.0,
        "ETH/USD": 3000.0
    }
    
    portfolio.current_holdings = {
        "cash": 20000.0,
        "BTC/USD": 2.0 * 50000.0,  # position value
        "ETH/USD": 10.0 * 3000.0,  # position value
        "total": 20000.0 + (2.0 * 50000.0) + (10.0 * 3000.0)
    }
    
    # Calculate exposure
    exposure = portfolio.get_risk_exposure()
    
    # Expected exposure percentages
    total_value = portfolio.current_holdings["total"]
    expected_btc_exposure = (2.0 * 50000.0) / total_value
    expected_eth_exposure = (10.0 * 3000.0) / total_value
    
    # Check calculated exposure
    assert "BTC/USD" in exposure
    assert "ETH/USD" in exposure
    assert abs(exposure["BTC/USD"] - expected_btc_exposure) < 0.01
    assert abs(exposure["ETH/USD"] - expected_eth_exposure) < 0.01
    assert abs(sum(exposure.values()) - (1.0 - 20000.0/total_value)) < 0.01  # Total exposure (excluding cash)