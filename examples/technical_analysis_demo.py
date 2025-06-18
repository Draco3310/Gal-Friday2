"""Demonstration of the new technical analysis module.

This script shows how to use the different implementations and features
of the technical analysis module.
"""

import contextlib

import numpy as np
import pandas as pd

from gal_friday.technical_analysis import (
    StubImplementation,
    atr,
    bbands,
    create_technical_analysis_service,
    ema,
    macd,
    # Backward compatibility imports
    rsi,
    sma,
)


def generate_sample_data(n_points=100):
    """Generate realistic sample price data."""
    np.random.seed(42)

    # Generate realistic price movement with trend and volatility
    dates = pd.date_range("2024-01-01", periods=n_points, freq="h")

    # Start at 100 and add random walk with slight upward trend
    prices = [100.0]
    for _ in range(n_points - 1):
        change = np.random.normal(0.001, 0.02)  # 0.1% average return, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    close_prices = np.array(prices)

    # Generate OHLC data
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, n_points)))

    return {
        "dates": dates,
        "close": close_prices,
        "high": high_prices,
        "low": low_prices,
    }


def demo_backward_compatibility():
    """Demonstrate backward compatibility with talib_stubs interface."""
    data = generate_sample_data(50)

    # Use the module-level functions (backward compatible)
    rsi(data["close"], timeperiod=14)

    upper, middle, lower = bbands(data["close"], timeperiod=20)

    ema(data["close"], timeperiod=12)
    sma(data["close"], timeperiod=12)

    macd_line, signal_line, histogram = macd(data["close"])

    atr(data["high"], data["low"], data["close"], timeperiod=14)


def demo_dependency_injection():
    """Demonstrate dependency injection pattern."""
    data = generate_sample_data(100)

    # Create stub service for testing
    stub_service = create_technical_analysis_service({
        "use_production_indicators": False,
    })

    stub_service.rsi(data["close"])

    # Create production service
    try:
        prod_service = create_technical_analysis_service({
            "use_production_indicators": True,
            "indicator_library": "pandas_ta",
        })

        prod_service.rsi(data["close"])

        # Compare implementations
        upper_stub, middle_stub, lower_stub = stub_service.bbands(data["close"])
        upper_prod, middle_prod, lower_prod = prod_service.bbands(data["close"])


    except ImportError:
        pass


def demo_edge_cases():
    """Demonstrate edge case handling."""
    # Create service
    service = StubImplementation()

    # Test with empty array
    service.rsi(np.array([]))

    # Test with single value
    service.rsi(np.array([100.0]))

    # Test with NaN values
    nan_data = np.array([100, 101, np.nan, 103, 104])
    with contextlib.suppress(Exception):
        service.sma(nan_data, timeperiod=3)

    # Test with period longer than data
    short_data = np.array([100, 101, 102])
    service.sma(short_data, timeperiod=10)


def demo_configuration_options():
    """Demonstrate different configuration options."""
    configs = [
        {
            "name": "Default (Stub)",
            "config": {},
        },
        {
            "name": "Production with pandas-ta",
            "config": {
                "use_production_indicators": True,
                "indicator_library": "pandas_ta",
            },
        },
        {
            "name": "Production with TA-Lib (falls back if not installed)",
            "config": {
                "use_production_indicators": True,
                "indicator_library": "talib",
            },
        },
    ]

    data = generate_sample_data(30)

    for cfg in configs:
        try:
            service = create_technical_analysis_service(cfg["config"])
            service.rsi(data["close"], timeperiod=14)
        except Exception:
            pass


def main():
    """Run all demonstrations."""
    demo_backward_compatibility()
    demo_dependency_injection()
    demo_edge_cases()
    demo_configuration_options()



if __name__ == "__main__":
    main()
