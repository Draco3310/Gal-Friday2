"""Configuration dataclasses for ExecutionHandler to reduce initialization complexity."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WebSocketConfig:
    """WebSocket-specific configuration."""

    use_for_order_updates: bool = False
    max_reconnect_attempts: int = 5
    reconnect_delay_seconds: int = 5
    heartbeat_interval: int = 30
    subscription_channels: list[str] = field(default_factory=lambda: ["openOrders", "ownTrades"])


@dataclass
class OrderStateConfig:
    """Order state tracking configuration."""

    persistence_enabled: bool = True
    cache_size: int = 10000
    batch_persistence: bool = True
    persistence_interval_seconds: int = 30


@dataclass
class ShutdownConfig:
    """Shutdown behavior configuration."""

    default_action: str = "cancel_all"
    stop_loss_action: str = "leave_open"
    take_profit_action: str = "cancel_all"
    limit_action: str = "cancel_conditional"
    market_action: str = "leave_open"
    safety_checks_enabled: bool = True
    max_cancellation_time: int = 30
    preserve_stop_losses: bool = True
    market_hours_only: bool = False


@dataclass
class AsyncProcessingConfig:
    """Asynchronous order processing configuration."""

    max_concurrent_orders: int = 50
    max_requests_per_second: int = 10
    enable_priority_queues: bool = True
    error_retry_max_attempts: int = 3
    error_retry_delay_seconds: float = 1.0
    health_check_interval: int = 60
    queue_sizes: dict[str, int] = field(
        default_factory=lambda: {
            "critical": 1000,
            "high": 5000,
            "normal": 10000,
            "low": 20000,
        },
    )


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    private_calls_per_second: int = 1
    public_calls_per_second: int = 1
    window_size: float = 1.0
    burst_capacity: int = 5


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "gal_friday"
    user: str = "gal_friday"
    password: str = ""
    pool_size: int = 20
    max_overflow: int = 30


@dataclass
class ExecutionHandlerConfig:
    """Complete configuration for ExecutionHandler."""

    # Exchange configuration
    exchange: str = "kraken"
    api_url: str = "https://api.kraken.com"
    api_key: str | None = None
    api_secret: str | None = None

    # Sub-configurations
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    order_state: OrderStateConfig = field(default_factory=OrderStateConfig)
    shutdown: ShutdownConfig = field(default_factory=ShutdownConfig)
    async_processing: AsyncProcessingConfig = field(default_factory=AsyncProcessingConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Feature flags
    enable_persistence: bool = True
    enable_market_data: bool = True
    enable_websocket: bool = False
    enable_async_processing: bool = True

    @classmethod
    def from_config_manager(cls, config_manager: Any) -> "ExecutionHandlerConfig":
        """Create ExecutionHandlerConfig from ConfigManager instance."""
        return cls(
            # Exchange configuration
            exchange=config_manager.get("execution_handler.exchange", "kraken"),
            api_url=config_manager.get("exchange.api_url", "https://api.kraken.com"),
            api_key=config_manager.get("kraken.api_key", default=None),
            api_secret=config_manager.get("kraken.secret_key", default=None),
            # WebSocket configuration
            websocket=WebSocketConfig(
                use_for_order_updates=config_manager.get("execution_handler.websocket.use_for_order_updates", False),
                max_reconnect_attempts=config_manager.get("execution_handler.websocket.max_reconnect_attempts", 5),
                reconnect_delay_seconds=config_manager.get("execution_handler.websocket.reconnect_delay_seconds", 5),
            ),
            # Order state configuration
            order_state=OrderStateConfig(
                persistence_enabled=config_manager.get("execution_handler.order_state.persistence_enabled", True),
                cache_size=config_manager.get_int("execution_handler.order_state.cache_size", 10000),
                batch_persistence=config_manager.get("execution_handler.order_state.batch_persistence", True),
                persistence_interval_seconds=config_manager.get_int(
                    "execution_handler.order_state.persistence_interval_seconds",
                    30,
                ),
            ),
            # Shutdown configuration
            shutdown=ShutdownConfig(
                default_action=config_manager.get("execution_handler.shutdown.default_action", "cancel_all"),
                stop_loss_action=config_manager.get("execution_handler.shutdown.stop_loss_action", "leave_open"),
                take_profit_action=config_manager.get("execution_handler.shutdown.take_profit_action", "cancel_all"),
                limit_action=config_manager.get("execution_handler.shutdown.limit_action", "cancel_conditional"),
                market_action=config_manager.get("execution_handler.shutdown.market_action", "leave_open"),
                safety_checks_enabled=config_manager.get("execution_handler.shutdown.safety_checks_enabled", True),
                max_cancellation_time=config_manager.get_int("execution_handler.shutdown.max_cancellation_time", 30),
                preserve_stop_losses=config_manager.get("execution_handler.shutdown.preserve_stop_losses", True),
                market_hours_only=config_manager.get("execution_handler.shutdown.market_hours_only", False),
            ),
            # Async processing configuration
            async_processing=AsyncProcessingConfig(
                max_concurrent_orders=config_manager.get_int("execution_handler.async.max_concurrent_orders", 50),
                max_requests_per_second=config_manager.get_int("execution_handler.async.max_requests_per_second", 10),
                enable_priority_queues=config_manager.get("execution_handler.async.enable_priority_queues", True),
                error_retry_max_attempts=config_manager.get_int("execution_handler.async.error_retry_max_attempts", 3),
                error_retry_delay_seconds=config_manager.get_float(
                    "execution_handler.async.error_retry_delay_seconds",
                    1.0,
                ),
                health_check_interval=config_manager.get_int("execution_handler.async.health_check_interval", 60),
            ),
            # Rate limit configuration
            rate_limit=RateLimitConfig(
                private_calls_per_second=config_manager.get_int("exchange.rate_limit.private_calls_per_second", 1),
                public_calls_per_second=config_manager.get_int("exchange.rate_limit.public_calls_per_second", 1),
            ),
            # Database configuration
            database=DatabaseConfig(
                host=config_manager.get("database.host", "localhost"),
                port=config_manager.get_int("database.port", 5432),
                database=config_manager.get("database.name", "gal_friday"),
                user=config_manager.get("database.user", "gal_friday"),
                password=config_manager.get("database.password", ""),
                pool_size=config_manager.get_int("database.pool_size", 20),
                max_overflow=config_manager.get_int("database.max_overflow", 30),
            ),
        )
