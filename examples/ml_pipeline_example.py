#!/usr/bin/env python3
"""ML Pipeline Example Script.

This script demonstrates how to use the comprehensive ML pipeline for price prediction
with feature engineering, model training, and performance monitoring.

Example Usage:
    python examples/ml_pipeline_example.py --symbol BTC/USD --action train
    python examples/ml_pipeline_example.py --symbol BTC/USD --action predict
    python examples/ml_pipeline_example.py --action status
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

import asyncio
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gal_friday.core.pubsub import PubSubManager
from gal_friday.logger_service import LoggerService
from gal_friday.ml_prediction_pipeline import ModelType
from gal_friday.prediction_service import PredictionService


class MLPipelineExample:
    """Example class demonstrating ML pipeline usage."""

    def __init__(self):
        """Initialize the ML pipeline example."""
        self.logger = LoggerService()
        self.pubsub = PubSubManager()
        self.executor = ProcessPoolExecutor(max_workers=4)

        # Example configuration
        self.config = {
            "prediction_service": {
                "models": [],  # We'll use ML pipeline instead
                "ml_pipeline": {
                    "model_storage_path": "./models/examples",
                    "feature_engineering": {
                        "nan_threshold": 0.5,
                        "include_time_features": True,
                        "cyclical_encoding": True,
                    },
                    "model_training": {
                        "default_cv_folds": 3,  # Reduced for example
                        "default_performance_threshold": 0.0,
                    },
                },
            },
        }

        # Initialize prediction service
        self.prediction_service = PredictionService(
            config=self.config,
            pubsub_manager=self.pubsub,
            process_pool_executor=self.executor,
            logger_service=self.logger,
        )

    def generate_sample_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Generate sample OHLCV data for demonstration.

        Args:
            symbol: Trading symbol
            days: Number of days of data to generate

        Returns:
            DataFrame with sample market data
        """
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="1H")

        # Generate realistic price data with trend and volatility
        np.random.seed(42)  # For reproducible results

        base_price = 50000 if "BTC" in symbol else 3000
        trend = 0.001 * np.arange(len(dates))  # Slight upward trend
        noise = np.random.normal(0, 0.02, len(dates))  # 2% volatility

        # Generate price series
        log_returns = trend + noise
        prices = base_price * np.exp(np.cumsum(log_returns))

        # Generate OHLCV data
        data = []
        for _i, (date, close) in enumerate(zip(dates, prices, strict=False)):
            # Generate realistic OHLC based on close
            volatility = abs(np.random.normal(0, 0.01))
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            open_price = close + np.random.normal(0, close * 0.005)

            # Generate volume (inverse correlation with price volatility)
            base_volume = 1000000
            volume = base_volume * (1 + np.random.normal(0, 0.3))

            data.append({
                "timestamp": date,
                "open": max(open_price, 0),
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": max(close, 0),
                "volume": max(volume, 0),
            })

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)


        return df

    async def train_model_example(self, symbol: str, model_type: str = "random_forest") -> None:
        """Demonstrate model training with the ML pipeline.

        Args:
            symbol: Trading symbol to train for
            model_type: Type of model to train
        """
        try:
            # Generate sample training data
            training_data = self.generate_sample_data(symbol, days=60)

            # Convert model type string to enum
            model_type_enum = ModelType(model_type.lower())

            # Define hyperparameters based on model type
            hyperparameters = {}
            if model_type_enum == ModelType.RANDOM_FOREST:
                hyperparameters = {
                    "n_estimators": 50,  # Reduced for example
                    "max_depth": 10,
                    "random_state": 42,
                }
            elif model_type_enum == ModelType.XGBOOST:
                hyperparameters = {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "random_state": 42,
                }


            # Train the model
            start_time = datetime.now()
            await self.prediction_service.train_model(
                symbol=symbol,
                training_data=training_data,
                model_type=model_type_enum,
                hyperparameters=hyperparameters,
            )
            (datetime.now() - start_time).total_seconds()


            # Get model performance metrics
            performance = self.prediction_service.get_model_performance_metrics(symbol)
            if performance:
                performance.get("performance_metrics", {})

                performance.get("operational_metrics", {})

        except Exception:
            raise

    async def predict_example(self, symbol: str) -> None:
        """Demonstrate advanced prediction with the ML pipeline.

        Args:
            symbol: Trading symbol to predict for
        """
        try:
            # Generate sample feature data (in practice, this would come from your feature engine)
            features = {
                "sma_20": 51000.0,
                "sma_50": 49500.0,
                "rsi": 65.5,
                "macd": 150.0,
                "macd_signal": 120.0,
                "volatility_20": 0.025,
                "bb_position": 0.7,
                "volume_ratio_20": 1.2,
                "price_change_1": 0.015,
                "high_low_ratio": 1.02,
                "vwap": 50800.0,
            }

            for _key, _value in list(features.items())[:5]:  # Show first 5
                pass

            # Make prediction
            start_time = datetime.now()
            prediction_result = await self.prediction_service.predict_with_ml_pipeline(
                symbol=symbol,
                features=features,
                prediction_horizon=60,  # 1 hour ahead
                confidence_level=0.95,
            )
            (datetime.now() - start_time).total_seconds()


            ci_lower, ci_upper = prediction_result["confidence_interval"]

            # Show feature importance
            feature_importance = prediction_result.get("feature_importance", {})
            if feature_importance:
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                for _feature, _importance in sorted_features:
                    pass

            # Show accuracy metrics if available
            accuracy_metrics = prediction_result.get("accuracy_metrics")
            if accuracy_metrics:
                pass

        except Exception:
            raise

    async def show_pipeline_status(self):
        """Demonstrate pipeline status monitoring."""
        try:
            status = self.prediction_service.get_ml_pipeline_status()


            prediction_stats = status.get("prediction_stats", {})

            last_training = prediction_stats.get("last_training_time")
            last_prediction = prediction_stats.get("last_prediction_time")

            if last_training:
                pass
            if last_prediction:
                pass

        except Exception:
            raise

    async def retrain_example(self, symbol: str) -> None:
        """Demonstrate automatic model retraining."""
        try:
            # Generate fresh training data
            current_data = self.generate_sample_data(symbol, days=45)

            # Check if retraining is needed
            was_retrained = await self.prediction_service.retrain_model_if_needed(
                symbol=symbol,
                current_data=current_data,
                performance_threshold=0.8,  # High threshold to trigger retraining
            )

            if was_retrained:
                pass
            else:
                pass

        except Exception:
            raise

    async def run_complete_example(self, symbol: str) -> None:
        """Run a complete end-to-end example."""
        try:
            # Step 1: Train a model
            await self.train_model_example(symbol, "random_forest")

            # Step 2: Make predictions
            await self.predict_example(symbol)

            # Step 3: Show pipeline status
            await self.show_pipeline_status()

            # Step 4: Demonstrate retraining check
            await self.retrain_example(symbol)


        except Exception:
            raise
        finally:
            # Cleanup
            self.executor.shutdown(wait=True)


async def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="ML Pipeline Example")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading symbol")
    parser.add_argument("--action", choices=["train", "predict", "status", "retrain", "complete"],
                       default="complete", help="Action to perform")
    parser.add_argument("--model-type", choices=["random_forest", "xgboost", "linear_regression"],
                       default="random_forest", help="Model type for training")

    args = parser.parse_args()

    example = MLPipelineExample()

    try:
        if args.action == "train":
            await example.train_model_example(args.symbol, args.model_type)
        elif args.action == "predict":
            await example.predict_example(args.symbol)
        elif args.action == "status":
            await example.show_pipeline_status()
        elif args.action == "retrain":
            await example.retrain_example(args.symbol)
        elif args.action == "complete":
            await example.run_complete_example(args.symbol)

    except KeyboardInterrupt:
        pass
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    # Create models directory if it doesn't exist
    Path("./models/examples").mkdir(parents=True, exist_ok=True)

    # Run the example
    asyncio.run(main())
