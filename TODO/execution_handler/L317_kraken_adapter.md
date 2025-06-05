# Task: Refactor using a Kraken adapter implementing a BaseExecutionAdapter interface for exchange abstraction.

### 1. Context
- **File:** `gal_friday/execution_handler.py`
- **Line:** `317`
- **Keyword/Pattern:** `TODO`
- **Current State:** The execution handler is tightly coupled to Kraken-specific implementation without proper abstraction for supporting multiple exchanges.

### 2. Problem Statement
The current execution handler lacks proper abstraction for exchange operations, making it difficult to add support for additional exchanges or modify exchange-specific behavior. This tight coupling creates technical debt and reduces the system's flexibility to adapt to different trading venues or exchange API changes. The monolithic approach also makes testing and maintenance more complex.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Exchange Adapter Interface:** Define abstract base class for exchange operations
2. **Implement Kraken Adapter:** Extract Kraken-specific logic into dedicated adapter
3. **Refactor Execution Handler:** Modify to use adapter pattern for exchange operations
4. **Add Exchange Factory:** Create factory pattern for dynamic adapter selection
5. **Implement Configuration Management:** Support exchange selection via configuration
6. **Create Comprehensive Testing:** Ensure all adapters maintain consistent behavior

#### b. Pseudocode or Implementation Sketch
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from enum import Enum

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"

class OrderRequest(BaseModel):
    """Standardized order request across all exchanges"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None

class OrderResponse(BaseModel):
    """Standardized order response across all exchanges"""
    exchange_order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    status: str
    filled_quantity: float
    remaining_quantity: float
    average_price: Optional[float]
    timestamp: datetime
    commission: Optional[float] = None

class BaseExecutionAdapter(ABC):
    """Abstract base class for exchange execution adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.exchange_name = self._get_exchange_name()
    
    @abstractmethod
    def _get_exchange_name(self) -> str:
        """Return the name of the exchange"""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to exchange"""
        pass
    
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place an order on the exchange"""
        pass
    
    @abstractmethod
    async def cancel_order(self, exchange_order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, exchange_order_id: str, symbol: str) -> OrderResponse:
        """Get current status of an order"""
        pass
    
    @abstractmethod
    async def get_account_balance(self) -> Dict[str, float]:
        """Get current account balances"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    def validate_order_request(self, order_request: OrderRequest) -> None:
        """Validate order request against exchange requirements"""
        pass

class KrakenExecutionAdapter(BaseExecutionAdapter):
    """Kraken-specific execution adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kraken_client = None
        self.symbol_mapping = self._load_symbol_mapping()
        self.precision_mapping = self._load_precision_mapping()
        
    def _get_exchange_name(self) -> str:
        return "Kraken"
    
    async def connect(self) -> bool:
        """Establish connection to Kraken API"""
        try:
            import krakenex
            self.kraken_client = krakenex.API(
                key=self.config['api_key'],
                secret=self.config['api_secret']
            )
            
            # Test connection with account info request
            result = await self._make_authenticated_request('Balance', {})
            if result.get('error'):
                self.logger.error(f"Kraken connection failed: {result['error']}")
                return False
            
            self.logger.info("Successfully connected to Kraken")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Kraken: {e}")
            return False
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place order on Kraken exchange"""
        
        # Validate order request
        self.validate_order_request(order_request)
        
        # Convert to Kraken format
        kraken_request = self._convert_to_kraken_format(order_request)
        
        try:
            # Place order via Kraken API
            result = await self._make_authenticated_request('AddOrder', kraken_request)
            
            if result.get('error'):
                raise ExecutionError(f"Kraken order failed: {result['error']}")
            
            # Convert response to standard format
            return self._convert_from_kraken_format(result['result'], order_request)
            
        except Exception as e:
            self.logger.error(f"Error placing Kraken order: {e}")
            raise ExecutionError(f"Failed to place order: {e}")
    
    def _convert_to_kraken_format(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Convert standard order request to Kraken API format"""
        
        kraken_symbol = self.symbol_mapping.get(order_request.symbol, order_request.symbol)
        
        kraken_request = {
            'pair': kraken_symbol,
            'type': order_request.side.value,
            'ordertype': self._convert_order_type(order_request.order_type),
            'volume': str(order_request.quantity),
        }
        
        # Add price for limit orders
        if order_request.order_type == OrderType.LIMIT and order_request.price:
            kraken_request['price'] = str(order_request.price)
        
        # Add stop price for stop orders
        if order_request.order_type in [OrderType.STOP_LOSS] and order_request.stop_price:
            kraken_request['price'] = str(order_request.stop_price)
        
        # Add client order ID if provided
        if order_request.client_order_id:
            kraken_request['userref'] = order_request.client_order_id
        
        return kraken_request
    
    def validate_order_request(self, order_request: OrderRequest) -> None:
        """Validate order request against Kraken requirements"""
        
        # Check symbol support
        if order_request.symbol not in self.symbol_mapping:
            raise ValidationError(f"Symbol {order_request.symbol} not supported on Kraken")
        
        # Check minimum quantity
        min_quantity = self.precision_mapping.get(order_request.symbol, {}).get('min_quantity', 0)
        if order_request.quantity < min_quantity:
            raise ValidationError(f"Quantity {order_request.quantity} below minimum {min_quantity}")
        
        # Check price precision
        if order_request.price:
            price_precision = self.precision_mapping.get(order_request.symbol, {}).get('price_precision', 8)
            if len(str(order_request.price).split('.')[-1]) > price_precision:
                raise ValidationError(f"Price precision exceeds {price_precision} decimal places")

class ExecutionAdapterFactory:
    """Factory for creating exchange adapters"""
    
    _adapters = {
        'kraken': KrakenExecutionAdapter,
        # Future adapters can be added here
        # 'binance': BinanceExecutionAdapter,
        # 'coinbase': CoinbaseExecutionAdapter,
    }
    
    @classmethod
    def create_adapter(cls, exchange_name: str, config: Dict[str, Any]) -> BaseExecutionAdapter:
        """Create adapter for specified exchange"""
        
        adapter_class = cls._adapters.get(exchange_name.lower())
        if not adapter_class:
            available = list(cls._adapters.keys())
            raise ValueError(f"Exchange '{exchange_name}' not supported. Available: {available}")
        
        return adapter_class(config)
    
    @classmethod
    def list_supported_exchanges(cls) -> List[str]:
        """Return list of supported exchanges"""
        return list(cls._adapters.keys())

class RefactoredExecutionHandler:
    """Refactored execution handler using adapter pattern"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create adapter based on configuration
        exchange_name = config.get('exchange', 'kraken')
        exchange_config = config.get('exchange_config', {})
        
        self.adapter = ExecutionAdapterFactory.create_adapter(exchange_name, exchange_config)
        self.is_connected = False
    
    async def initialize(self) -> None:
        """Initialize execution handler and connect to exchange"""
        try:
            self.is_connected = await self.adapter.connect()
            if not self.is_connected:
                raise ConnectionError(f"Failed to connect to {self.adapter.exchange_name}")
            
            self.logger.info(f"Execution handler initialized with {self.adapter.exchange_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution handler: {e}")
            raise
    
    async def place_order(self, order_request: OrderRequest) -> OrderResponse:
        """Place order using configured exchange adapter"""
        if not self.is_connected:
            raise ConnectionError("Execution handler not connected")
        
        return await self.adapter.place_order(order_request)
    
    async def cancel_order(self, exchange_order_id: str, symbol: str) -> bool:
        """Cancel order using configured exchange adapter"""
        if not self.is_connected:
            raise ConnectionError("Execution handler not connected")
        
        return await self.adapter.cancel_order(exchange_order_id, symbol)
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Standardized error handling across all adapters; exchange-specific error code mapping; connection failure recovery
- **Configuration:** Exchange selection via configuration; adapter-specific configuration sections; runtime adapter switching capability
- **Testing:** Mock adapters for unit testing; integration tests with each exchange; behavior consistency tests across adapters
- **Dependencies:** Exchange-specific client libraries; configuration management system; logging framework; async/await support

### 4. Acceptance Criteria
- [ ] BaseExecutionAdapter interface is implemented with all required abstract methods
- [ ] KrakenExecutionAdapter implements all interface methods with proper Kraken API integration
- [ ] ExecutionAdapterFactory supports dynamic adapter creation based on configuration
- [ ] All order operations are abstracted through standardized request/response models
- [ ] Exchange-specific validation and error handling are encapsulated within adapters
- [ ] Configuration supports multiple exchange setups and runtime selection
- [ ] Comprehensive test suite covers all adapters and interface compliance
- [ ] Documentation explains how to add new exchange adapters
- [ ] Performance metrics show no degradation compared to original implementation
- [ ] All TODO comments related to exchange abstraction are removed 