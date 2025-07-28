"""Example demonstrating the comprehensive backtesting framework.

This example shows how to use the enhanced BacktestingEngine with:
- Multiple execution modes (vectorized vs event-driven)
- Performance analytics
- Benchmarking
- Strategy comparison
- Parameter optimization
"""

import datetime as dt
import logging
from pathlib import Path

import asyncio
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import the comprehensive backtesting components
from gal_friday.backtesting_engine import (
    BacktestConfig,
    BacktestError,
    BacktestingEngine,
    BacktestMode,
)


class SimpleStrategy:
    """Example strategy for demonstration purposes."""

    def __init__(self, name: str = "SimpleStrategy", sma_period: int = 20, rsi_period: int = 14) -> None:
        self.name = name
        self.sma_period = sma_period
        self.rsi_period = rsi_period
        self.position = 0

    def generate_signals(self, timestamp, prices, positions, cash):
        """Generate trading signals based on simple moving average crossover."""
        signals = []

        # Simplified signal generation (would need more sophisticated logic in practice)
        for symbol, price in prices.items():
            if price > 100:  # Simple buy condition
                if positions.get(symbol, 0) == 0 and cash > price * 100:
                    signals.append({
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 100,
                    })
            elif price < 95 and positions.get(symbol, 0) > 0:  # Simple sell condition
                signals.append({
                    "symbol": symbol,
                        "action": "sell",
                        "quantity": positions[symbol],
                    })

        return signals


class TrendFollowingStrategy:
    """Example trend following strategy."""

    def __init__(self, name: str = "TrendFollowing", fast_period: int = 10, slow_period: int = 30) -> None:
        self.name = name
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, timestamp, prices, positions, cash):
        """Generate signals based on moving average crossover."""
        signals = []

        # Simplified trend following logic
        for symbol, price in prices.items():
            # This would calculate actual moving averages in a real implementation
            if price > 105:  # Uptrend condition
                if positions.get(symbol, 0) == 0 and cash > price * 50:
                    signals.append({
                        "symbol": symbol,
                        "action": "buy",
                        "quantity": 50,
                    })
            elif price < 90 and positions.get(symbol, 0) > 0:  # Downtrend condition
                signals.append({
                    "symbol": symbol,
                        "action": "sell",
                        "quantity": positions[symbol],
                    })

        return signals


def create_sample_data():
    """Create sample OHLCV data for demonstration."""
    # Generate sample data for BTCUSD and ETHUSD
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    # Create a random number generator
    rng = np.random.default_rng(42)

    data = []

    # BTC data
    btc_price = 40000
    for date in dates:
        # Random walk with trend
        btc_price += rng.normal(0, 500)
        btc_price = max(btc_price, 20000)  # Floor price

        data.append({
            "symbol": "BTCUSD",
            "timestamp": date,
            "open": btc_price + rng.normal(0, 100),
            "high": btc_price + abs(rng.normal(200, 100)),
            "low": btc_price - abs(rng.normal(200, 100)),
            "close": btc_price,
            "volume": rng.uniform(1000, 10000),
        })

    # ETH data
    eth_price = 2500
    for date in dates:
        eth_price += rng.normal(0, 50)
        eth_price = max(eth_price, 1000)  # Floor price

        data.append({
            "symbol": "ETHUSD",
            "timestamp": date,
            "open": eth_price + rng.normal(0, 20),
            "high": eth_price + abs(rng.normal(50, 20)),
            "low": eth_price - abs(rng.normal(50, 20)),
            "close": eth_price,
            "volume": rng.uniform(5000, 50000),
        })

    return pd.DataFrame(data)


class MockConfigManager:
    """Mock configuration manager for the example."""

    def __init__(self):
        self.config = {}

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def get_all(self):
        return self.config

    def __contains__(self, key: str) -> bool:
        return key in self.config


async def setup_and_load_data(engine: BacktestingEngine) -> str:
    """Initialize engine and load sample data."""
    # Create and load sample data
    sample_data = create_sample_data()

    # Save to CSV and load it (demonstrating CSV loading capability)
    csv_path = "sample_crypto_data.csv"
    sample_data.to_csv(csv_path, index=False)
    engine.load_data_from_csv(csv_path, symbol_column="symbol")

    # Display data summary
    data_summary = engine.get_data_summary()
    for _symbol, _details in data_summary["symbol_details"].items():
        pass

    return csv_path


async def run_single_strategy_backtest(engine: BacktestingEngine) -> None:
    """Run single strategy backtest."""
    backtest_config = BacktestConfig(
        start_date=dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
        end_date=dt.datetime(2023, 6, 30, tzinfo=dt.UTC),
        initial_capital=100000.0,
        symbols=["BTCUSD", "ETHUSD"],
        mode=BacktestMode.VECTORIZED,  # Start with vectorized for speed
        commission_rate=0.001,
        slippage_rate=0.0005,
        benchmark_symbol="BTCUSD",  # Use BTC as benchmark
        output_dir="backtest_results",
    )

    strategy = SimpleStrategy(name="SimpleMA", sma_period=20)

    try:
        results = await engine.run_backtest(backtest_config, strategy)

        enhanced_metrics = results.get("enhanced_metrics")
        if enhanced_metrics:
            pass

        # Generate and display report
        engine.generate_report(results, "backtest_report.txt")

    except BacktestError:
        return


async def compare_execution_modes(engine: BacktestingEngine, strategy: SimpleStrategy) -> None:
    """Compare different execution modes."""
    event_config = BacktestConfig(
        start_date=dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
        end_date=dt.datetime(2023, 3, 31, tzinfo=dt.UTC),  # Shorter period for demo
        initial_capital=100000.0,
        symbols=["BTCUSD"],
        mode=BacktestMode.EVENT_DRIVEN,
        commission_rate=0.001,
    )

    try:
        event_results = await engine.run_backtest(event_config, strategy)
        event_metrics = event_results.get("enhanced_metrics")
        if event_metrics:
            pass
    except Exception as e:
        logger.exception("Error occurred: %s", e)


def run_strategy_comparison(engine: BacktestingEngine) -> None:
    """Run strategy comparison."""
    strategies = [
        SimpleStrategy(name="Simple_MA20", sma_period=20),
        SimpleStrategy(name="Simple_MA50", sma_period=50),
        TrendFollowingStrategy(name="Trend_10_30", fast_period=10, slow_period=30),
        TrendFollowingStrategy(name="Trend_5_15", fast_period=5, slow_period=15),
    ]

    comparison_config = BacktestConfig(
        start_date=dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
        end_date=dt.datetime(2023, 4, 30, tzinfo=dt.UTC),
        initial_capital=100000.0,
        symbols=["BTCUSD"],
        mode=BacktestMode.VECTORIZED,
        commission_rate=0.001,
    )

    comparison_results = engine.run_strategy_comparison(strategies, comparison_config)

    summary = comparison_results.get("comparison_summary", {})
    if summary:
        for result in summary.values():
            if result.get("strategy"):
                pass


def run_parameter_optimization(engine: BacktestingEngine) -> None:
    """Run parameter optimization."""
    parameter_grid = {
        "sma_period": [10, 20, 30, 50],
        "rsi_period": [10, 14, 20],
    }

    optimization_config = BacktestConfig(
        start_date=dt.datetime(2023, 1, 1, tzinfo=dt.UTC),
        end_date=dt.datetime(2023, 3, 31, tzinfo=dt.UTC),
        initial_capital=50000.0,
        symbols=["BTCUSD"],
        mode=BacktestMode.VECTORIZED,
        commission_rate=0.001,
    )

    try:
        optimization_results = engine.optimize_parameters(
            SimpleStrategy,
            parameter_grid,
            optimization_config,
            optimization_metric="sharpe_ratio",
        )

        best_result = optimization_results.get("best_result")
        if best_result:
            pass

    except Exception as e:
        logger.exception("Error occurred: %s", e)


async def main():
    """Main example function demonstrating the comprehensive backtesting framework."""
    # 1. Initialize the backtesting engine
    config_manager = MockConfigManager()
    engine = BacktestingEngine(config=config_manager)

    # 2. Setup and load data
    csv_path = await setup_and_load_data(engine)

    # 3. Run single strategy backtest
    await run_single_strategy_backtest(engine)

    # 4. Compare execution modes
    strategy = SimpleStrategy(name="SimpleMA", sma_period=20)
    await compare_execution_modes(engine, strategy)

    # 5. Run strategy comparison
    run_strategy_comparison(engine)

    # 6. Run parameter optimization
    run_parameter_optimization(engine)

    # 7. Clean up
    engine.clear_data()

    # Clean up sample files
    try:
        Path(csv_path).unlink()
        Path("backtest_report.txt").unlink(missing_ok=True)
    except Exception as e:
        logger.exception("Error cleaning up files: %s", e)



if __name__ == "__main__":
    asyncio.run(main())
