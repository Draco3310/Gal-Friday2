from datetime import UTC, datetime
from decimal import Decimal
import logging  # <-- Import logging
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch
import uuid

import yaml

from gal_friday.backtesting_engine import BacktestingEngine
from gal_friday.config_manager import ConfigManager
from gal_friday.core.events import EventType, PredictionEvent
from gal_friday.core.feature_registry_client import FeatureRegistryClient  # Added import
from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.market_price_service import MarketPriceService  # StrategyArbitrator needs it
from gal_friday.strategy_arbitrator import StrategyArbitrator

# A directory for temporary test files (like dummy data and configs)
TEST_TEMP_DIR = Path(__file__).parent / "test_temp_output"
TEST_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Dummy Feature Registry YAML content
SAMPLE_FEATURE_REGISTRY_YAML = """
features:
  rsi_14_default:
    calculator_type: "rsi"
    input_type: "ohlcv"
    parameters:
      period: 14
    imputation: "skip" # or some default
    scaling: "min_max" # 0-100 for RSI
    output_properties:
      primary_output: "RSI_14" # Example if pandas_ta outputs RSI_14
      is_multidimensional: false
    version: 1

  macd_default:
    calculator_type: "macd"
    input_type: "ohlcv"
    parameters:
      fast: 12
      slow: 26
      signal: 9
    imputation: "skip"
    scaling: "none"
    output_properties:
      primary_output: "MACD_12_26_9" # pandas_ta default for MACD line
      is_multidimensional: true
    version: 1
"""

# Dummy Historical Data CSV
SAMPLE_HISTORICAL_DATA_CSV = """timestamp,open,high,low,close,volume,pair
2023-01-01T00:00:00Z,100,105,99,102,1000,BTC/USD
2023-01-01T00:01:00Z,102,106,101,103,1200,BTC/USD
2023-01-01T00:02:00Z,103,107,102,104,1100,BTC/USD
2023-01-01T00:03:00Z,104,108,103,105,1300,BTC/USD
2023-01-01T00:04:00Z,105,109,104,106,1400,BTC/USD
"""


class MockPredictionService:
    def __init__(self, config: dict, pubsub_manager: PubSubManager, logger_service: LoggerService):
        self.config = config
        self.pubsub = pubsub_manager
        self.logger = logger_service
        self.source_module = self.__class__.__name__
        self._is_running = False
        self.received_features_count = 0
        self.published_predictions_count = 0

    async def start(self):
        self._is_running = True
        await self.pubsub.subscribe(EventType.FEATURES_CALCULATED, self.handle_feature_event)
        self.logger.info(f"{self.source_module} started and subscribed to FEATURES_CALCULATED.")

    async def stop(self):
        self._is_running = False
        self.logger.info(f"{self.source_module} stopped.")

    async def handle_feature_event(self, event_dict: dict):  # PubSub delivers dicts
        if not self._is_running:
            return

        self.logger.debug(f"{self.source_module} received feature event dict: {event_dict.get('event_type')}")

        # Reconstruct the FeatureEvent if necessary, or just grab the payload
        # For this mock, we'll assume the payload is what we need
        if event_dict.get("event_type") != EventType.FEATURES_CALCULATED.name:
            self.logger.warning(f"MockPredictionService received non-feature event: {event_dict.get('event_type')}")
            return

        payload = event_dict.get("payload")
        if not payload or not isinstance(payload, dict):
            self.logger.error("Feature event payload missing or invalid.")
            return

        features = payload.get("features")  # This should be dict[str, float]
        timestamp_features_for_str = payload.get("timestamp_features_for")
        trading_pair = payload.get("trading_pair")
        exchange = payload.get("exchange")

        if not features or not trading_pair or not timestamp_features_for_str:
            self.logger.error("Essential data missing in feature event payload for prediction generation.")
            return

        self.received_features_count += 1

        # Create a dummy prediction
        # In a real service, this would involve model inference
        dummy_prediction_value = 0.75  # Example: 75% chance of price increase

        try:
            timestamp_prediction_for = datetime.fromisoformat(timestamp_features_for_str)
        except ValueError:
            self.logger.exception(f"Could not parse timestamp_features_for: {timestamp_features_for_str}")
            return

        prediction_event = PredictionEvent(
            source_module=self.source_module,
            event_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            trading_pair=trading_pair,
            exchange=exchange,  # Use exchange from FeatureEvent
            timestamp_prediction_for=timestamp_prediction_for,
            model_id="mock_model_v1",
            prediction_target="price_up_prob",
            prediction_value=dummy_prediction_value,
            confidence=0.9,
            associated_features={"triggering_features": features},  # Key part for StrategyArbitrator
        )

        await self.pubsub.publish(prediction_event)
        self.published_predictions_count += 1
        self.logger.info(
            f"{self.source_module} published PredictionEvent for {trading_pair} at {timestamp_prediction_for}",
        )


class TestBacktestingEngineFeatureFlow(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Create dummy feature registry file
        cls.feature_registry_path = TEST_TEMP_DIR / "feature_registry.yaml"
        with cls.feature_registry_path.open("w") as f:
            f.write(SAMPLE_FEATURE_REGISTRY_YAML)

        # Create dummy historical data file
        cls.historical_data_path = TEST_TEMP_DIR / "historical_data.csv"
        with cls.historical_data_path.open("w") as f:
            f.write(SAMPLE_HISTORICAL_DATA_CSV)

    @classmethod
    def tearDownClass(cls):
        # Clean up dummy files
        if cls.feature_registry_path.exists():
            cls.feature_registry_path.unlink()
        if cls.historical_data_path.exists():
            cls.historical_data_path.unlink()
        # Could remove TEST_TEMP_DIR if empty and desired

    async def test_features_flow_to_strategy_arbitrator(self):  # noqa: PLR0915 - test-complexity: comprehensive integration test requires extensive setup and validation
        # 1. Setup Configuration
        app_config = {
            "app_name": "GalFridayTestBacktestFeatureFlow",
            "logging": {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {"simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
                "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple", "level": "DEBUG"}},
                "root": {"handlers": ["console"], "level": "INFO"},
                "loggers": {"gal_friday": {"level": "DEBUG", "propagate": False, "handlers": ["console"]}},
            },
            "feature_engine": {
                "feature_registry_path": str(self.feature_registry_path),
                "active_feature_rules": [  # Activate features from the registry
                    {"feature_key": "rsi_14_default", "active": True},
                    {"feature_key": "macd_default", "active": True},
                ],
                "output_ohlcv_path": str(TEST_TEMP_DIR / "feature_engine_ohlcv_output.parquet"),
                "output_trades_path": str(TEST_TEMP_DIR / "feature_engine_trades_output.parquet"),
            },
            "prediction_service": {  # Config for MockPredictionService
                "model_paths": {"mock_model_v1": "dummy_path"},
            },
            "strategy_arbitrator": {
                "strategies": [
                    {
                        "id": "test_strategy_rsi_confirm",
                        "buy_threshold": 0.7,  # Based on MockPredictionService's dummy_prediction_value
                        "sell_threshold": 0.3,
                        "entry_type": "MARKET",
                        "sl_pct": 0.01,  # 1%
                        "tp_pct": 0.02,  # 2%
                        "prediction_interpretation": "prob_up",
                        "confirmation_rules": [
                            {"feature": "rsi_14_default", "condition": "lt", "threshold": 70},
                            {
                                "feature": "macd_default_MACD_12_26_9",
                                "condition": "gt",
                                "threshold": -10,
                            },  # Ensure it's a float
                        ],
                    },
                ],
            },
            "backtest": {
                "data_path": str(self.historical_data_path),
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-01-01T00:04:00Z",
                "trading_pairs": ["BTC/USD"],
                "initial_capital": 10000.0,
                "ohlcv_interval": "1min",  # Matches data
                "output_dir": str(TEST_TEMP_DIR / "backtest_results"),
            },
            "exchange_name": "simulated_test_exchange",
            "exchange_info_service": {  # Needed for SimulatedMarketPriceService
                "exchange_name": "simulated_test_exchange",
                "trading_pairs_config": {
                    "BTC/USD": {
                        "price_precision": 2,
                        "quantity_precision": 6,
                        "min_quantity": 0.00001,
                        "maker_fee_pct": 0.001,  # 0.1%
                        "taker_fee_pct": 0.002,  # 0.2%
                    },
                },
            },
        }
        # Create a temporary app config file for ConfigManager
        temp_app_config_path = TEST_TEMP_DIR / "temp_app_config.yaml"
        with temp_app_config_path.open("w") as f:
            yaml.dump(app_config, f)

        config_manager = ConfigManager(config_path=str(temp_app_config_path))

        # 2. Initialize services that BacktestingEngine doesn't create internally for _execute_simulation
        # First PubSubManager, then LoggerService that might use it (or at least needs config_manager)
        # LoggerService needs pubsub_manager for event handling, and config_manager for its own config.
        # PubSubManager needs a logger_service. This is a bit of a startup order puzzle.
        # Let's assume PubSubManager can be initialized with a placeholder/basic logger first,
        # or LoggerService is initialized first and PubSubManager gets the instance.
        # Given LoggerService takes pubsub_manager in __init__, PubSub must be first.

        # Simplest for test: Create a basic logger for PubSubManager if strict typing demands it,
        # then create the full logger_service, then ensure PubSubManager uses the full one if needed.
        # However, PubSubManager's __init__ takes logger_service.
        # So, LoggerService must be created first.
        # And LoggerService takes config_manager and pubsub_manager.

        # Let's look at BacktestingEngine._initialize_services:
        # 1. self.logger_service = LoggerService(config=log_config, is_backtest=True) # This was the old way
        # 2. self.pubsub_manager = PubSubManager(logger_service=self.logger_service)
        # This order worked because LoggerService didn't used to take pubsub_manager.

        # Current LoggerService signature: LoggerService(config_manager, pubsub_manager, db_session_maker=None)
        # Current PubSubManager signature: PubSubManager(logger_service)

        # Revised initialization to break cycle and match signatures:
        # 1. ConfigManager (already done)
        # 2. Standard logger for PubSubManager
        test_pubsub_logger = logging.getLogger("TestPubSubLogger")
        # 3. PubSubManager
        pubsub_manager = PubSubManager(logger=test_pubsub_logger, config_manager=config_manager)
        # 4. LoggerService (full service)
        logger_service = LoggerService(config_manager=config_manager, pubsub_manager=pubsub_manager)

        # Now, services that depend on logger_service (like the one used by BacktestingEngine internally)
        # or pubsub_manager can use these fully initialized instances.

        # Feature Registry Client for StrategyArbitrator
        feature_registry_client = FeatureRegistryClient(config_path=str(self.feature_registry_path))

        # ExchangeInfoService and SimulatedMarketPriceService for StrategyArbitrator
        # For the backtest, BacktestingEngine will manage its own data provider.
        # This is a bit tricky because MarketPriceService is usually live.
        # For the test, we can instantiate SimulatedMarketPriceService for StrategyArbitrator.
        # It might need to be updated with prices by the BacktestingEngine's loop, or
        # StrategyArbitrator simply uses the latest price available from its own instance.
        # Let's make it simple: SA gets its own SimulatedMarketPriceService that can be fed.

        # The BacktestingEngine will create its own HistoricalDataProvider.
        # The SimulatedMarketPriceService for StrategyArbitrator needs a way to get current prices.
        # For simplicity in this test, we'll mock get_latest_price on the MarketPriceService
        # instance that StrategyArbitrator uses.
        mock_market_price_service = AsyncMock(spec=MarketPriceService)
        mock_market_price_service.get_latest_price = AsyncMock(return_value=Decimal("105.0"))  # Dummy price

        # MockPredictionService
        mock_prediction_service = MockPredictionService(
            config=config_manager.get("prediction_service", {}),
            pubsub_manager=pubsub_manager,
            logger_service=logger_service,
        )

        # StrategyArbitrator
        strategy_arbitrator = StrategyArbitrator(
            config=app_config,  # Pass the full app_config dict
            pubsub_manager=pubsub_manager,
            logger_service=logger_service,
            market_price_service=mock_market_price_service,  # Use the mocked one
            feature_registry_client=feature_registry_client,  # Pass the client
        )

        # BacktestingEngine
        # It will create its own FeatureEngine.
        # It needs a dict of other services to run in its simulation loop.
        # Crucially, these services must share the same pubsub_manager and logger_service.

        backtest_services_for_simulation = {
            "logger_service": logger_service,  # For consistency, though BE might manage its own primarily
            "pubsub_manager": pubsub_manager,  # Crucial: must be the same instance
            # "historical_data_provider": created by BE
            # "market_price_service": created by BE (SimulatedMarketPriceService)
            # "portfolio_manager": created by BE (SimulatedPortfolioManager)
            # "execution_handler": created by BE (SimulatedExecutionHandler)
            # "feature_engine": created by BE
            "prediction_service": mock_prediction_service,  # Our mock
            "strategy_arbitrator": strategy_arbitrator,  # The one we configured
            # "risk_manager": Not strictly needed for this feature flow test
        }

        # 3. Patch StrategyArbitrator._validate_confirmation_rule
        # We use `gal_friday.strategy_arbitrator.StrategyArbitrator` because that's where the class is defined.
        with patch(
            "gal_friday.strategy_arbitrator.StrategyArbitrator._validate_confirmation_rule",
            wraps=strategy_arbitrator._validate_confirmation_rule,
        ) as mock_validate_rule:
            # 4. Initialize and Run BacktestingEngine
            # BacktestingEngine itself is not async in its constructor or run_backtest method
            # but _execute_simulation is.
            backtesting_engine = BacktestingEngine(
                config=config_manager,  # Pass the ConfigManager instance
                data_dir=str(TEST_TEMP_DIR),  # Not directly used if data_path in config is absolute
            )

            # Load data into the engine (mimicking part of run_backtest setup)
            # This is normally handled by run_backtest, but we need to ensure _data is populated
            # for _execute_simulation if called directly.
            # Let's call run_backtest as it sets up more things.

            # The run_backtest method is synchronous but calls async _execute_simulation.
            # We need to run it in a way that the asyncio event loop is managed.
            # Since the test method is async, we can await _execute_simulation directly
            # after appropriate setup.

            # run_backtest prepares data and then calls _execute_simulation
            # For this test, we'll manually call the parts of run_backtest needed
            # or simplify. Let's try to use `run_backtest` if possible,
            # but it might be too encompassing.

            # Simplified approach: Call _initialize_services, then _execute_simulation
            # This requires self._data to be populated.

            raw_data = backtesting_engine._load_raw_data(str(self.historical_data_path))
            assert raw_data is not None, "Failed to load raw data for backtest."

            backtest_run_config = backtesting_engine._get_backtest_config()
            assert backtesting_engine._validate_config(backtest_run_config), "Backtest config validation failed."

            cleaned_data = backtesting_engine._clean_and_validate_data(
                raw_data,
                backtest_run_config["start_date"],
                backtest_run_config["end_date"],
            )
            assert cleaned_data is not None, "Data cleaning failed."

            processed_data = backtesting_engine._process_pairs_data(cleaned_data)
            assert processed_data is not None, "Data processing failed."
            backtesting_engine._data = processed_data  # Set the data for the engine

            # Now, call the core simulation part
            await backtesting_engine._execute_simulation(
                services=backtest_services_for_simulation,
                run_config=backtest_run_config,
            )

            # 5. Assertions
            assert mock_validate_rule.called, "StrategyArbitrator._validate_confirmation_rule was not called."

            assert mock_prediction_service.received_features_count > 0, (
                "MockPredictionService did not receive features."
            )
            assert mock_prediction_service.published_predictions_count > 0, (
                "MockPredictionService did not publish predictions."
            )

            # Inspect calls to _validate_confirmation_rule
            # Each call to _validate_confirmation_rule has args: (self, rule, features, trading_pair, primary_side)
            # We are interested in the `features` argument (positional index 2 in args, or keyword 'features')
            found_rsi = False
            found_macd = False
            for call_args in mock_validate_rule.call_args_list:
                args, kwargs = call_args
                # More robust: get by name from signature if possible, or rely on known signature
                # For simplicity, assuming 'features' is the 3rd positional argument (index 2)
                # or a keyword argument. The 'wraps' makes it a bit complex.
                # Let's inspect by looking for the features dict passed to the *original* method.
                # The `call_args` for a wrapped method might be tricky.

                # Let's verify the content of features passed to the rule validation.
                # The wrapped function's call args might be just `(rule, features, trading_pair, primary_side)`
                # if the `self` is bound.

                # If mock_validate_rule is a MagicMock wrapping the original:
                # The arguments captured are those passed to the mock itself.
                # So, call_args[0] are positional, call_args[1] are kwargs.
                # The signature is (self, rule, features, trading_pair, primary_side)
                # When called as `strategy_arbitrator_instance._validate_confirmation_rule(...)`,
                # `self` is not part of `*args` in `call_args`.

                # For the wrapped method:
                # The first argument to the mock_validate_rule will be the `rule`

                received_rule_features = call_args.args[1]  # This should be the `features` dict
                assert isinstance(received_rule_features, dict), (
                    "Features argument to _validate_confirmation_rule was not a dict."
                )

                if "rsi_14_default" in received_rule_features:
                    found_rsi = True
                    assert isinstance(received_rule_features["rsi_14_default"], float), "RSI feature is not a float."
                if "macd_default_MACD_12_26_9" in received_rule_features:
                    found_macd = True
                    assert isinstance(received_rule_features["macd_default_MACD_12_26_9"], float), (
                        "MACD feature is not a float."
                    )

            assert found_rsi, "RSI feature was not found in features passed to _validate_confirmation_rule."
            assert found_macd, "MACD feature was not found in features passed to _validate_confirmation_rule."

            # Additional check: FeatureEngine's own capture within BacktestingEngine
            # This ensures FeatureEngine produced something sensible.
            # backtesting_engine.current_features would hold the *last* set of features.
            assert backtesting_engine.current_features is not None, "BacktestingEngine did not capture any features."
            if backtesting_engine.current_features:  # Check if not None
                assert "rsi_14_default" in backtesting_engine.current_features
                assert isinstance(backtesting_engine.current_features["rsi_14_default"], float)
                assert "macd_default_MACD_12_26_9" in backtesting_engine.current_features
                assert isinstance(backtesting_engine.current_features["macd_default_MACD_12_26_9"], float)
                assert "macd_default_MACDh_12_26_9" in backtesting_engine.current_features
                assert isinstance(backtesting_engine.current_features["macd_default_MACDh_12_26_9"], float)
                assert "macd_default_MACDs_12_26_9" in backtesting_engine.current_features
                assert isinstance(backtesting_engine.current_features["macd_default_MACDs_12_26_9"], float)

                # Basic sanity check for RSI value (MinMax scaled 0-100)
                rsi_val = backtesting_engine.current_features["rsi_14_default"]
                assert 0.0 <= rsi_val <= 100.0, f"RSI value {rsi_val} out of expected 0-100 range."


if __name__ == "__main__":
    unittest.main()
