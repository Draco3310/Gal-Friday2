"""Enhanced execution handler interface supporting multi-exchange and multi-asset trading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Protocol, TypedDict, Unpack

from gal_friday.core.asset_registry import AssetSpecification, ExchangeSpecification


class OrderType(Enum):
    """Universal order types across all exchanges."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()
    BRACKET = auto()          # For futures/options
    ICEBERG = auto()          # Large order splitting
    FILL_OR_KILL = auto()     # FOK
    IMMEDIATE_OR_CANCEL = auto()  # IOC


class OrderStatus(Enum):
    """Universal order status across all exchanges."""
    PENDING = auto()          # Order submitted but not confirmed
    OPEN = auto()            # Order active in market
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class TimeInForce(Enum):
    """Time in force options."""
    GTC = auto()  # Good Till Cancelled
    GTD = auto()  # Good Till Date
    DAY = auto()  # Day order
    IOC = auto()  # Immediate or Cancel
    FOK = auto()  # Fill or Kill


@dataclass(frozen=True)
class OrderRequest:
    """Universal order request structure."""
    # Core order details
    symbol: str
    exchange_id: str
    order_type: OrderType
    side: str  # "BUY" or "SELL"
    quantity: Decimal

    # Price details (depending on order type)
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None

    # Order behavior
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False    # For derivatives
    post_only: bool = False      # Maker-only orders

    # Advanced order features
    iceberg_quantity: Decimal | None = None  # For iceberg orders

    # Bracket order details (for sophisticated exchanges)
    take_profit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None

    # Risk management
    max_show_quantity: Decimal | None = None
    min_quantity: Decimal | None = None

    # Client tracking
    client_order_id: str | None = None
    strategy_id: str | None = None
    signal_id: str | None = None

    # Metadata
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize metadata dictionary if not provided."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


@dataclass(frozen=True)
class OrderResponse:
    """Universal order response structure."""
    success: bool
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    status: OrderStatus | None = None

    # Execution details
    filled_quantity: Decimal = Decimal(0)
    remaining_quantity: Decimal | None = None
    average_fill_price: Decimal | None = None

    # Fees and costs
    commission: Decimal | None = None
    commission_currency: str | None = None

    # Timestamps
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Error handling
    error_code: str | None = None
    error_message: str | None = None

    # Exchange-specific data
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class PositionInfo:
    """Universal position information structure."""
    symbol: str
    exchange_id: str
    side: str  # "LONG" or "SHORT"
    quantity: Decimal
    average_entry_price: Decimal

    # Current valuation
    current_price: Decimal | None = None
    unrealized_pnl: Decimal | None = None

    # For margin/derivatives
    margin_used: Decimal | None = None
    maintenance_margin: Decimal | None = None

    # Timestamps
    opened_at: datetime | None = None
    updated_at: datetime | None = None

    # Exchange-specific data
    raw_data: dict[str, Any] | None = None


class ExecutionHandlerKwargs(TypedDict, total=False):
    """Configuration parameters for ExecutionHandler initialization.

    All fields are optional and provide flexibility for different
    exchange implementations and deployment environments.
    """

    # Authentication credentials
    api_key: str
    """API key for exchange authentication."""

    secret_key: str
    """Secret key for exchange authentication."""

    passphrase: str
    """Passphrase for exchange authentication (required by some exchanges)."""

    # Environment configuration
    sandbox_mode: bool
    """Enable sandbox/testnet mode for testing."""

    api_base_url: str
    """Custom API base URL (overrides default exchange endpoints)."""

    websocket_url: str
    """Custom WebSocket URL for real-time data feeds."""

    # Connection settings
    connection_timeout: float
    """Connection timeout in seconds (default: 30.0)."""

    read_timeout: float
    """Read timeout in seconds (default: 60.0)."""

    max_retries: int
    """Maximum number of retry attempts for failed requests (default: 3)."""

    retry_delay: float
    """Delay between retry attempts in seconds (default: 1.0)."""

    # Rate limiting
    max_requests_per_second: int
    """Maximum requests per second to avoid exchange rate limits."""

    burst_limit: int
    """Maximum burst requests allowed before rate limiting kicks in."""

    # Risk management
    max_order_value: Decimal
    """Maximum order value limit for risk management."""

    daily_loss_limit: Decimal
    """Daily loss limit for automatic trading halt."""

    position_size_limit: Decimal
    """Maximum position size allowed per symbol."""

    # Logging and monitoring
    log_level: str
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    enable_order_logging: bool
    """Enable detailed logging of all order operations."""

    enable_metrics: bool
    """Enable performance metrics collection."""

    # Configuration overrides
    config_override: dict[str, Any]
    """Dictionary of configuration overrides for specific exchange settings."""

    # Advanced features
    enable_websocket: bool
    """Enable WebSocket connections for real-time updates."""

    enable_auto_reconnect: bool
    """Enable automatic reconnection on connection loss."""

    heartbeat_interval: float
    """Heartbeat interval in seconds for connection health checks."""

    # Portfolio management
    enable_position_tracking: bool
    """Enable automatic position tracking and reconciliation."""

    enable_balance_caching: bool
    """Enable balance caching to reduce API calls."""

    cache_ttl: int
    """Cache time-to-live in seconds for cached data."""

    # Order management
    default_time_in_force: TimeInForce
    """Default time-in-force for orders when not specified."""

    enable_order_validation: bool
    """Enable client-side order validation before submission."""

    enable_duplicate_detection: bool
    """Enable duplicate order detection and prevention."""

    # Testing and development
    dry_run_mode: bool
    """Enable dry run mode for testing without actual order execution."""

    mock_responses: bool
    """Use mock responses for testing (bypasses actual exchange calls)."""

    # Performance optimization
    connection_pool_size: int
    """Size of HTTP connection pool for better performance."""

    enable_compression: bool
    """Enable request/response compression."""

    # Compliance and auditing
    enable_audit_trail: bool
    """Enable comprehensive audit trail logging."""

    compliance_mode: str
    """Compliance mode setting (STRICT, NORMAL, RELAXED)."""

class ExecutionHandlerInterface(ABC):
    """Enhanced interface for execution handlers supporting multiple exchanges and asset types."""

    def __init__(
        self,
        exchange_spec: ExchangeSpecification,
        **kwargs: Unpack[ExecutionHandlerKwargs]) -> None:
        """Initialize with exchange specification and configuration."""
        self.exchange_spec = exchange_spec
        self.exchange_id = exchange_spec.exchange_id

    # Core trading operations
    @abstractmethod
    async def submit_order(self, order_request: OrderRequest) -> OrderResponse:
        """Submit an order to the exchange.

        Args:
            order_request: Universal order request structure
        Returns:
            Universal order response structure
        Raises:
            OrderValidationError: If order parameters are invalid
            ExchangeConnectionError: If exchange is unavailable
            InsufficientFundsError: If insufficient balance for order
        """

    @abstractmethod
    async def cancel_order(self, exchange_order_id: str,
                          symbol: str | None = None) -> OrderResponse:
        """Cancel an existing order.

        Args:
            exchange_order_id: Exchange-specific order identifier
            symbol: Trading symbol (required by some exchanges)

        Returns:
            Order response indicating cancellation status
        """

    @abstractmethod
    async def modify_order(self, exchange_order_id: str,
                          modifications: dict[str, Any]) -> OrderResponse:
        """Modify an existing order (if supported by exchange).

        Args:
            exchange_order_id: Exchange-specific order identifier
            modifications: Dictionary of fields to modify (price, quantity, etc.)

        Returns:
            Order response with updated order details
        """

    @abstractmethod
    async def get_order_status(self, exchange_order_id: str) -> OrderResponse:
        """Get current status of an order.

        Args:
            exchange_order_id: Exchange-specific order identifier
        Returns:
            Current order status and details
        """

    # Portfolio and position management
    @abstractmethod
    async def get_account_balance(self) -> dict[str, Decimal]:
        """Get current account balances.

        Returns:
            Dictionary mapping currency/asset to available balance
        """

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[PositionInfo]:
        """Get current positions.

        Args:
            symbol: Optional filter for specific symbol
        Returns:
            List of current positions
        """

    # Market data and exchange info
    @abstractmethod
    async def get_supported_assets(self) -> list[AssetSpecification]:
        """Get list[Any] of assets supported by this exchange.

        Returns:
            List of asset specifications
        """

    @abstractmethod
    async def get_trading_fees(self, symbol: str) -> dict[str, Decimal]:
        """Get trading fees for a symbol.

        Args:
            symbol: Trading symbol
        Returns:
            Dictionary with maker/taker fees
        """

    # Connection and lifecycle management
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange.

        Returns:
            True if connection successful, False otherwise
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from exchange."""

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if connected to exchange.

        Returns:
            True if connected, False otherwise
        """

    # Validation and capability checking
    def validate_order_request(self, order_request: OrderRequest) -> list[str]:
        """Validate order request against exchange capabilities.

        Args:
            order_request: Order request to validate
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check if order type is supported
        if (order_request.order_type == OrderType.BRACKET and
                not self.exchange_spec.supports_bracket_orders):
            errors.append("Bracket orders not supported by this exchange")

        if (order_request.order_type == OrderType.STOP and
                not self.exchange_spec.supports_stop_orders):
            errors.append("Stop orders not supported by this exchange")

        # Check price requirements
        if (order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and
                order_request.limit_price is None):
            errors.append("Limit price required for limit orders")

        if (order_request.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and
                order_request.stop_price is None):
            errors.append("Stop price required for stop orders")

        return errors

    def supports_order_type(self, order_type: OrderType) -> bool:
        """Check if exchange supports a specific order type.

        Args:
            order_type: Order type to check
        Returns:
            True if supported, False otherwise
        """
        if order_type == OrderType.MARKET:
            return self.exchange_spec.supports_market_orders
        if order_type == OrderType.LIMIT:
            return self.exchange_spec.supports_limit_orders
        if order_type == OrderType.STOP:
            return self.exchange_spec.supports_stop_orders
        if order_type == OrderType.BRACKET:
            return self.exchange_spec.supports_bracket_orders
        return False

    # Asset-specific utilities
    def normalize_symbol_for_exchange(self, universal_symbol: str) -> str:
        """Convert universal symbol to exchange-specific format.

        Args:
            universal_symbol: Universal symbol format (e.g., "XRP/USD")

        Returns:
            Exchange-specific symbol format
        """
        # Default implementation - override in specific handlers
        return universal_symbol.replace("/", "")

    def get_exchange_capabilities(self) -> dict[str, Any]:
        """Get exchange capabilities summary.

        Returns:
            Dictionary of exchange capabilities
        """
        return {
            "exchange_id": self.exchange_id,
            "supports_limit_orders": self.exchange_spec.supports_limit_orders,
            "supports_market_orders": self.exchange_spec.supports_market_orders,
            "supports_stop_orders": self.exchange_spec.supports_stop_orders,
            "supports_bracket_orders": self.exchange_spec.supports_bracket_orders,
            "supports_margin": self.exchange_spec.supports_margin,
            "max_orders_per_second": self.exchange_spec.max_orders_per_second,
            "typical_latency_ms": self.exchange_spec.typical_latency_ms,
        }


# Protocol for execution handler factory
class ExecutionHandlerFactory(Protocol):
    """Protocol for creating execution handlers for different exchanges."""

    def create_handler(
        self,
        exchange_id: str,
        **kwargs: Unpack[ExecutionHandlerKwargs]) -> ExecutionHandlerInterface:
        """Create an execution handler for the specified exchange.

        Args:
            exchange_id: Exchange identifier
            **kwargs: Additional configuration parameters
        Returns:
            Execution handler instance for the exchange
        """
        ...
