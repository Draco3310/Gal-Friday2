import ccxt.async_support as ccxt
from typing import Dict, Any, List, Optional
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class ExchangeManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange_id = config.get('exchange_id', 'kraken')
        self.exchange_class = getattr(ccxt, self.exchange_id)

        exchange_config = {
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'enableRateLimit': True,
        }

        if config.get('sandbox_mode', False):
            exchange_config['sandbox'] = True

        self.exchange = self.exchange_class(exchange_config)
        logger.info(f"Initialized exchange: {self.exchange_id}")

    async def close(self):
        await self.exchange.close()

    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """
        Fetches OHLCV data and returns a DataFrame.
        """
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            raise

    async def fetch_balance(self) -> Dict[str, float]:
        """
        Fetches free balance.
        """
        try:
            balance = await self.exchange.fetch_balance()
            return balance.get('free', {})
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise

    async def create_order(self, symbol: str, type: str, side: str, amount: float, price: Optional[float] = None):
        """
        Creates an order.
        """
        try:
            order = await self.exchange.create_order(symbol, type, side, amount, price)
            logger.info(f"Created {side} order for {symbol}: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Error creating order for {symbol}: {e}")
            raise

    async def get_ticker(self, symbol: str):
        return await self.exchange.fetch_ticker(symbol)
