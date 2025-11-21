import asyncio
import logging
import sys
from src.config_loader import ConfigLoader
from src.core.logger import setup_logger
from src.core.database import DatabaseHandler
from src.core.exchange import ExchangeManager
from src.core.plugin_loader import PluginLoader
from src.core.execution import PositionManager

async def main():
    # 1. Setup
    logger = setup_logger()
    logger.info("Starting Gal-Friday V2...")

    # 2. Config
    try:
        config = ConfigLoader.load_config("config.yaml")
    except Exception as e:
        logger.critical(f"Failed to load config: {e}")
        sys.exit(1)

    # 3. Core Components
    db = DatabaseHandler(config['database']['path'])
    exchange = ExchangeManager(config['exchange'])
    position_manager = PositionManager(db, exchange)

    # 4. Strategy Loading
    strat_conf = config['strategy']
    try:
        StrategyClass = PluginLoader.load_strategy(strat_conf['module'], strat_conf['class'])
        strategy = StrategyClass(strat_conf) # Initialize with strategy config
        logger.info(f"Strategy {strat_conf['class']} initialized.")
    except Exception as e:
        logger.critical(f"Failed to load strategy: {e}")
        await exchange.close()
        sys.exit(1)

    # 5. Main Loop
    symbol = strat_conf['symbol']
    timeframe = config['trading']['timeframe']
    interval = config['trading'].get('loop_interval_seconds', 60)

    logger.info(f"Entering main loop for {symbol} ({timeframe})...")

    try:
        while True:
            try:
                # Fetch Data
                logger.info("Fetching market data...")
                df = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)

                # Analyze
                logger.info("Analyzing market data...")
                signal = await strategy.analyze(df)

                # Execute
                await position_manager.process_signal(signal)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            logger.info(f"Sleeping for {interval} seconds...")
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        await exchange.close()
        logger.info("Exchange connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
