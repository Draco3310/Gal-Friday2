"""WebSocket market data ingestion."""


from gal_friday.config_manager import ConfigManager
from gal_friday.core.pubsub import PubSubManager
from gal_friday.execution.websocket_client import KrakenWebSocketClient
from gal_friday.logger_service import LoggerService


class WebSocketMarketDataService:
    """Market data ingestion via WebSocket."""

    def __init__(self,
                 config: ConfigManager,
                 pubsub: PubSubManager,
                 logger: LoggerService) -> None:
        """Initialize the WebSocket market data service.

        Args:
            config: Configuration manager instance
            pubsub: Pub/Sub manager for event distribution
            logger: Logger service for logging messages
        """
        self.config = config
        self.pubsub = pubsub
        self.logger = logger
        self._source_module = self.__class__.__name__

        # WebSocket client
        self.ws_client = KrakenWebSocketClient(config, pubsub, logger)

        # Subscriptions
        self.pairs: set[str] = set(config.get_list("trading.pairs", ["XRP/USD", "DOGE/USD"]))
        self.channels = ["book", "ticker", "trade", "ohlc"]

    async def start(self) -> None:
        """Start market data service."""
        self.logger.info(
            "Starting WebSocket market data service",
            source_module=self._source_module,
        )

        # Connect WebSocket
        await self.ws_client.connect()

        # Subscribe to market data
        await self.ws_client.subscribe_market_data(
            list(self.pairs),
            self.channels,
        )

    async def stop(self) -> None:
        """Stop market data service."""
        await self.ws_client.disconnect()

    async def add_pair(self, pair: str) -> None:
        """Add a trading pair subscription."""
        if pair not in self.pairs:
            self.pairs.add(pair)
            await self.ws_client.subscribe_market_data([pair], self.channels)

    async def remove_pair(self, pair: str) -> None:
        """Remove a trading pair subscription."""
        if pair in self.pairs:
            self.pairs.remove(pair)
            # Note: Kraken doesn't support unsubscribe, would need to reconnect
