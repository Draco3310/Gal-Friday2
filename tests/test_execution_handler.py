"""
Tests for the execution_handler module.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from gal_friday.execution_handler import ExecutionHandler
from gal_friday.event_bus import EventBus
from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import OrderEvent, FillEvent


@pytest.fixture
def execution_config():
    """Fixture providing execution configuration."""
    return {
        "exchanges": {
            "kraken": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "symbols": ["BTC/USD", "ETH/USD"],
                "default_limit_slippage": 0.001,  # 0.1% slippage for limit orders
                "reconnect_wait_time": 10,        # 10 seconds wait time for reconnect
                "order_expiry_seconds": 60        # 60 seconds before unfilled orders expire
            }
        }
    }


def test_execution_handler_initialization(execution_config, event_bus):
    """Test that the ExecutionHandler initializes correctly."""
    config = ConfigManager(config_dict=execution_config)
    execution_handler = ExecutionHandler(config, event_bus)
    
    assert execution_handler is not None
    assert execution_handler.exchange_name == "kraken"
    assert execution_handler.api_key == "test_key"
    assert execution_handler.api_secret == "test_secret"
    assert execution_handler.symbols == ["BTC/USD", "ETH/USD"]


@patch("ccxt.kraken")
def test_execution_handler_connect(mock_ccxt_kraken, execution_config, event_bus):
    """Test connecting to the exchange."""
    # Set up mock
    mock_exchange = MagicMock()
    mock_ccxt_kraken.return_value = mock_exchange
    
    # Initialize handler
    config = ConfigManager(config_dict=execution_config)
    
    with patch("gal_friday.execution_handler.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        execution_handler = ExecutionHandler(config, event_bus)
        execution_handler.connect()
    
    # Verify exchange connection
    mock_ccxt_kraken.assert_called_once_with({
        'apiKey': 'test_key',
        'secret': 'test_secret',
    })
    assert execution_handler.exchange == mock_exchange
    mock_exchange.load_markets.assert_called_once()


@patch("ccxt.kraken")
def test_execution_handler_execute_market_order(mock_ccxt_kraken, execution_config, event_bus):
    """Test executing a market order."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_exchange.create_market_order.return_value = {
        'id': '12345',
        'timestamp': datetime.now().timestamp() * 1000,
        'status': 'closed',
        'symbol': 'BTC/USD',
        'type': 'market',
        'side': 'buy',
        'price': 50000.0,
        'amount': 1.0,
        'filled': 1.0,
        'cost': 50000.0,
        'fee': {'cost': 25.0, 'currency': 'USD'}
    }
    mock_ccxt_kraken.return_value = mock_exchange
    
    # Set up mock event bus to capture published events
    mock_event_bus = MagicMock()
    
    # Initialize handler
    config = ConfigManager(config_dict=execution_config)
    
    with patch("gal_friday.execution_handler.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        execution_handler = ExecutionHandler(config, mock_event_bus)
        execution_handler.connect()
        
        # Create a market order event
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            order_type="MARKET",
            quantity=1.0,
            direction="BUY"
        )
        
        # Execute the order
        execution_handler.execute_order(order_event)
    
    # Verify order was placed
    mock_exchange.create_market_order.assert_called_once_with(
        'BTC/USD', 'buy', 1.0
    )
    
    # Verify fill event was published
    assert mock_event_bus.publish.call_count == 1
    published_event = mock_event_bus.publish.call_args[0][0]
    assert isinstance(published_event, FillEvent)
    assert published_event.symbol == "BTC/USD"
    assert published_event.direction == "BUY"
    assert published_event.quantity == 1.0
    assert published_event.price == 50000.0
    assert published_event.commission == 25.0


@patch("ccxt.kraken")
def test_execution_handler_execute_limit_order(mock_ccxt_kraken, execution_config, event_bus):
    """Test executing a limit order."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_exchange.fetch_ticker.return_value = {
        'symbol': 'BTC/USD',
        'bid': 49900.0,
        'ask': 50100.0,
        'last': 50000.0
    }
    mock_exchange.create_limit_order.return_value = {
        'id': '12345',
        'timestamp': datetime.now().timestamp() * 1000,
        'status': 'open',  # Limit orders start as open
        'symbol': 'BTC/USD',
        'type': 'limit',
        'side': 'buy',
        'price': 50000.0,
        'amount': 1.0,
        'filled': 0.0,
        'cost': 0.0
    }
    mock_ccxt_kraken.return_value = mock_exchange
    
    # Set up mock event bus to capture published events
    mock_event_bus = MagicMock()
    
    # Initialize handler
    config = ConfigManager(config_dict=execution_config)
    
    with patch("gal_friday.execution_handler.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        execution_handler = ExecutionHandler(config, mock_event_bus)
        execution_handler.connect()
        
        # Create a limit order event
        order_event = OrderEvent(
            timestamp=datetime.now(),
            symbol="BTC/USD",
            order_type="LIMIT",
            quantity=1.0,
            direction="BUY",
            limit_price=50000.0  # Specified limit price
        )
        
        # Execute the order
        execution_handler.execute_order(order_event)
    
    # Verify order was placed
    mock_exchange.create_limit_order.assert_called_once_with(
        'BTC/USD', 'buy', 1.0, 50000.0
    )
    
    # Verify no fill event published yet (since limit order is still open)
    assert mock_event_bus.publish.call_count == 0
    
    # Now simulate the order being filled
    mock_exchange.fetch_order.return_value = {
        'id': '12345',
        'timestamp': datetime.now().timestamp() * 1000,
        'status': 'closed',  # Now it's filled
        'symbol': 'BTC/USD',
        'type': 'limit',
        'side': 'buy',
        'price': 50000.0,
        'amount': 1.0,
        'filled': 1.0,
        'cost': 50000.0,
        'fee': {'cost': 25.0, 'currency': 'USD'}
    }
    
    # Check order status (this would normally be called by a periodic check)
    execution_handler.check_order_status('12345')
    
    # Verify fill event was published
    assert mock_event_bus.publish.call_count == 1
    published_event = mock_event_bus.publish.call_args[0][0]
    assert isinstance(published_event, FillEvent)
    assert published_event.symbol == "BTC/USD"
    assert published_event.direction == "BUY"
    assert published_event.quantity == 1.0


@patch("ccxt.kraken")
def test_execution_handler_cancel_order(mock_ccxt_kraken, execution_config, event_bus):
    """Test cancelling an order."""
    # Set up mock exchange
    mock_exchange = MagicMock()
    mock_exchange.cancel_order.return_value = {
        'id': '12345',
        'status': 'canceled'
    }
    mock_ccxt_kraken.return_value = mock_exchange
    
    # Initialize handler
    config = ConfigManager(config_dict=execution_config)
    
    with patch("gal_friday.execution_handler.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        execution_handler = ExecutionHandler(config, event_bus)
        execution_handler.connect()
        
        # Track open order
        execution_handler.open_orders['12345'] = {
            'symbol': 'BTC/USD',
            'order_type': 'LIMIT',
            'quantity': 1.0,
            'direction': 'BUY',
            'timestamp': datetime.now()
        }
        
        # Cancel the order
        execution_handler.cancel_order('12345')
    
    # Verify the order was cancelled
    mock_exchange.cancel_order.assert_called_once_with('12345', 'BTC/USD')
    assert '12345' not in execution_handler.open_orders


@patch("ccxt.kraken")
def test_execution_handler_handle_exchange_error(mock_ccxt_kraken, execution_config, event_bus):
    """Test handling exchange errors."""
    # Set up mock exchange with error
    mock_exchange = MagicMock()
    mock_exchange.create_market_order.side_effect = Exception("API connection error")
    mock_ccxt_kraken.return_value = mock_exchange
    
    # Set up mock logger
    mock_logger = MagicMock()
    
    # Initialize handler
    config = ConfigManager(config_dict=execution_config)
    
    with patch("gal_friday.execution_handler.ccxt") as mock_ccxt:
        mock_ccxt.kraken = mock_ccxt_kraken
        with patch("gal_friday.execution_handler.logger", mock_logger):
            execution_handler = ExecutionHandler(config, event_bus)
            execution_handler.connect()
            
            # Create a market order event
            order_event = OrderEvent(
                timestamp=datetime.now(),
                symbol="BTC/USD",
                order_type="MARKET",
                quantity=1.0,
                direction="BUY"
            )
            
            # Execute the order (should handle the error)
            execution_handler.execute_order(order_event)
    
    # Verify error was logged
    assert mock_logger.error.call_count == 1
    
    # Verify retry mechanism was triggered (if implemented)
    # This would depend on your implementation details