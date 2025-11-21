import asyncio
import logging
import sys
from src.config_loader import ConfigLoader
from src.core.logger import setup_logger
from src.core.database import DatabaseHandler
from src.core.exchange import ExchangeManager
from src.core.plugin_loader import PluginLoader
from src.core.execution import PositionManager
from src.core.feature_engine import FeatureEngine

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
    feature_engine = FeatureEngine()

    # 4. Plugin Loading
    # Strategy
    strat_conf = config['strategy']
    try:
        StrategyClass = PluginLoader.load_strategy(strat_conf['module'], strat_conf['class'])
        strategy = StrategyClass(strat_conf) # Initialize with strategy config
        logger.info(f"Strategy {strat_conf['class']} initialized.")
    except Exception as e:
        logger.critical(f"Failed to load strategy: {e}")
        await exchange.close()
        sys.exit(1)

    # Predictor
    pred_conf = config.get('predictor', {})
    predictor = None
    if pred_conf.get('enabled', False):
        try:
            PredictorClass = PluginLoader.load_predictor(pred_conf['module'], pred_conf['class'])
            predictor = PredictorClass(pred_conf)
            logger.info(f"Predictor {pred_conf['class']} initialized.")
        except Exception as e:
            logger.error(f"Failed to load predictor: {e}. Continuing without ML.")

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
                # Fetch enough data for lookback (indicators + model)
                df = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)

                if df.empty:
                    logger.warning("No data received.")
                    await asyncio.sleep(10)
                    continue

                # Feature Engineering
                logger.info("Computing features...")
                enriched_df = feature_engine.compute_features(df)

                if enriched_df.empty:
                    logger.warning("Not enough data for features.")
                    await asyncio.sleep(10)
                    continue

                # Prediction
                ml_score = None
                if predictor:
                    try:
                        logger.info("Generating ML prediction...")
                        ml_score = predictor.predict(enriched_df)
                        logger.info(f"ML Score: {ml_score}")
                    except Exception as e:
                        logger.error(f"Prediction error: {e}")

                # Analyze
                logger.info("Analyzing market data...")
                # Donchian strategy expects enriched data now (it calculates its own indicators in V1,
                # but V2 strategy might assume indicators are present or calculate them.
                # My DonchianStrategy implementation calculates them inside analyze().
                # So passing enriched_df is fine, it will re-calculate or overwrite.
                # Ideally strategy should use pre-calc features if available.
                # But current DonchianStrategy implementation uses pandas-ta on the df passed.
                # Since compute_features already added columns, pandas-ta might re-add or use existing?
                # pandas-ta usually appends. If columns exist, it might duplicate or overwrite.
                # For Donchian, it needs DCL/DCU. FeatureEngine calculates RSI/MACD/BB/ATR.
                # So Strategy will calculate Donchian itself.

                signal = await strategy.analyze(enriched_df, ml_score=ml_score)

                # Execute
                await position_manager.process_signal(signal)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                logger.debug(traceback.format_exc())

            logger.info(f"Sleeping for {interval} seconds...")
            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        await exchange.close()
        logger.info("Exchange connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
