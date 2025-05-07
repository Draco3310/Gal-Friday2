"""
Tests for the backtesting_engine module.
"""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime
from gal_friday.backtesting_engine import BacktestingEngine
from gal_friday.config_manager import ConfigManager
from gal_friday.event_bus import EventBus


@pytest.fixture
def backtest_config():
    """Fixture for backtest configuration."""
    return {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "initial_capital": 100000.0,
        "symbols": ["BTC/USD", "ETH/USD"],
        "strategy": "momentum",
        "strategy_params": {
            "lookback_period": 14,
            "threshold": 0.05
        }
    }


@pytest.fixture
def mock_historical_data():
    """Fixture providing mock historical price data."""
    # Create sample price data for BTC/USD
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="1H")
    btc_prices = pd.DataFrame({
        'open': range(100, 100 + len(dates)),
        'high': range(105, 105 + len(dates)),
        'low': range(95, 95 + len(dates)),
        'close': range(102, 102 + len(dates)),
        'volume': [1000] * len(dates)
    }, index=dates)

    # Create sample price data for ETH/USD
    eth_prices = pd.DataFrame({
        'open': range(50, 50 + len(dates)),
        'high': range(55, 55 + len(dates)),
        'low': range(45, 45 + len(dates)),
        'close': range(52, 52 + len(dates)),
        'volume': [2000] * len(dates)
    }, index=dates)

    return {
        "BTC/USD": btc_prices,
        "ETH/USD": eth_prices
    }


def test_backtesting_engine_initialization(backtest_config):
    """Test that the BacktestingEngine initializes correctly."""
    config = ConfigManager(config_dict={"backtesting": backtest_config})
    event_bus = EventBus()

    engine = BacktestingEngine(config, event_bus)

    assert engine is not None
    assert engine.initial_capital == backtest_config["initial_capital"]
    assert engine.start_date == datetime.strptime(
        backtest_config["start_date"], "%Y-%m-%d")
    assert engine.end_date == datetime.strptime(
        backtest_config["end_date"], "%Y-%m-%d")
    assert engine.symbols == backtest_config["symbols"]


@patch("gal_friday.historical_data_service.HistoricalDataService")
def test_backtesting_engine_load_data(
        mock_data_service,
        backtest_config,
        mock_historical_data):
    """Test loading historical data in the backtesting engine."""
    # Set up mocks
    mock_data_service_instance = mock_data_service.return_value
    mock_data_service_instance.get_historical_data.side_effect = lambda symbol, start_date, end_date, timeframe: mock_historical_data[
        symbol]

    # Initialize engine
    config = ConfigManager(config_dict={"backtesting": backtest_config})
    event_bus = EventBus()

    with patch("gal_friday.backtesting_engine.HistoricalDataService", mock_data_service):
        engine = BacktestingEngine(config, event_bus)
        engine.load_data()

    # Verify data was loaded
    assert len(engine.data) == len(backtest_config["symbols"])
    for symbol in backtest_config["symbols"]:
        assert symbol in engine.data
        assert not engine.data[symbol].empty


@patch("gal_friday.historical_data_service.HistoricalDataService")
@patch("gal_friday.portfolio_manager.PortfolioManager")
@patch("gal_friday.risk_manager.RiskManager")
@patch("gal_friday.strategy_arbitrator.StrategyArbitrator")
@patch("gal_friday.simulated_execution_handler.SimulatedExecutionHandler")
def test_backtesting_engine_run(
    mock_execution, mock_strategy, mock_risk, mock_portfolio,
    mock_data_service, backtest_config, mock_historical_data
):
    """Test running a backtest."""
    # Set up mocks
    mock_data_service_instance = mock_data_service.return_value
    mock_data_service_instance.get_historical_data.side_effect = lambda symbol, start_date, end_date, timeframe: mock_historical_data[
        symbol]

    # Initialize engine
    config = ConfigManager(config_dict={"backtesting": backtest_config})
    event_bus = EventBus()

    with patch("gal_friday.backtesting_engine.HistoricalDataService", mock_data_service):
        with patch("gal_friday.backtesting_engine.PortfolioManager", mock_portfolio):
            with patch("gal_friday.backtesting_engine.RiskManager", mock_risk):
                with patch("gal_friday.backtesting_engine.StrategyArbitrator", mock_strategy):
                    with patch("gal_friday.backtesting_engine.SimulatedExecutionHandler", mock_execution):
                        engine = BacktestingEngine(config, event_bus)
                        engine.load_data()
                        results = engine.run()

    # Verify components were used
    mock_portfolio.assert_called_once()
    mock_risk.assert_called_once()
    mock_strategy.assert_called_once()
    mock_execution.assert_called_once()

    # Basic check on results
    assert results is not None
