"""Demonstration of the new technical analysis module.

This script shows how to use the different implementations and features
of the technical analysis module.
"""

import numpy as np
import pandas as pd
from gal_friday.technical_analysis import (
    create_technical_analysis_service,
    StubImplementation,
    # Backward compatibility imports
    rsi, bbands, ema, sma, macd, atr
)


def generate_sample_data(n_points=100):
    """Generate realistic sample price data."""
    np.random.seed(42)
    
    # Generate realistic price movement with trend and volatility
    dates = pd.date_range('2024-01-01', periods=n_points, freq='h')
    
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
        'dates': dates,
        'close': close_prices,
        'high': high_prices,
        'low': low_prices,
    }


def demo_backward_compatibility():
    """Demonstrate backward compatibility with talib_stubs interface."""
    print("=== Backward Compatibility Demo ===\n")
    
    data = generate_sample_data(50)
    
    # Use the module-level functions (backward compatible)
    rsi_result = rsi(data['close'], timeperiod=14)
    print(f"RSI (last 5 values): {rsi_result[-5:]}")
    
    upper, middle, lower = bbands(data['close'], timeperiod=20)
    print(f"Bollinger Bands (last value):")
    print(f"  Upper: {upper[-1]:.2f}")
    print(f"  Middle: {middle[-1]:.2f}")
    print(f"  Lower: {lower[-1]:.2f}")
    
    ema_result = ema(data['close'], timeperiod=12)
    sma_result = sma(data['close'], timeperiod=12)
    print(f"\nEMA vs SMA (last value):")
    print(f"  EMA: {ema_result[-1]:.2f}")
    print(f"  SMA: {sma_result[-1]:.2f}")
    
    macd_line, signal_line, histogram = macd(data['close'])
    print(f"\nMACD (last value):")
    print(f"  MACD Line: {macd_line[-1]:.4f}")
    print(f"  Signal Line: {signal_line[-1]:.4f}")
    print(f"  Histogram: {histogram[-1]:.4f}")
    
    atr_result = atr(data['high'], data['low'], data['close'], timeperiod=14)
    print(f"\nATR (last 5 values): {atr_result[-5:]}")


def demo_dependency_injection():
    """Demonstrate dependency injection pattern."""
    print("\n\n=== Dependency Injection Demo ===\n")
    
    data = generate_sample_data(100)
    
    # Create stub service for testing
    print("1. Using Stub Implementation (for testing):")
    stub_service = create_technical_analysis_service({
        'use_production_indicators': False
    })
    
    stub_rsi = stub_service.rsi(data['close'])
    print(f"   Stub RSI (all values should be 50.0): {stub_rsi[:5]}")
    
    # Create production service
    print("\n2. Using Production Implementation (pandas-ta):")
    try:
        prod_service = create_technical_analysis_service({
            'use_production_indicators': True,
            'indicator_library': 'pandas_ta'
        })
        
        prod_rsi = prod_service.rsi(data['close'])
        print(f"   Production RSI (varied values): {prod_rsi[-5:]}")
        
        # Compare implementations
        print("\n3. Comparing Stub vs Production:")
        upper_stub, middle_stub, lower_stub = stub_service.bbands(data['close'])
        upper_prod, middle_prod, lower_prod = prod_service.bbands(data['close'])
        
        print(f"   Stub BB width: {(upper_stub[-1] - lower_stub[-1]):.4f}")
        print(f"   Prod BB width: {(upper_prod[-1] - lower_prod[-1]):.4f}")
        
    except ImportError:
        print("   pandas-ta not installed - production mode not available")
        print("   Install with: pip install pandas-ta")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n\n=== Edge Case Handling Demo ===\n")
    
    # Create service
    service = StubImplementation()
    
    # Test with empty array
    print("1. Empty array:")
    empty_result = service.rsi(np.array([]))
    print(f"   Result length: {len(empty_result)} (should be 0)")
    
    # Test with single value
    print("\n2. Single value array:")
    single_result = service.rsi(np.array([100.0]))
    print(f"   Result: {single_result} (length: {len(single_result)})")
    
    # Test with NaN values
    print("\n3. Array with NaN values:")
    nan_data = np.array([100, 101, np.nan, 103, 104])
    try:
        nan_result = service.sma(nan_data, timeperiod=3)
        print(f"   SMA handled NaN gracefully: {nan_result}")
    except Exception as e:
        print(f"   Error handling NaN: {e}")
    
    # Test with period longer than data
    print("\n4. Period longer than data:")
    short_data = np.array([100, 101, 102])
    long_period_result = service.sma(short_data, timeperiod=10)
    print(f"   SMA with period 10 on 3 data points: {long_period_result}")


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n\n=== Configuration Options Demo ===\n")
    
    configs = [
        {
            'name': 'Default (Stub)',
            'config': {}
        },
        {
            'name': 'Production with pandas-ta',
            'config': {
                'use_production_indicators': True,
                'indicator_library': 'pandas_ta'
            }
        },
        {
            'name': 'Production with TA-Lib (falls back if not installed)',
            'config': {
                'use_production_indicators': True,
                'indicator_library': 'talib'
            }
        }
    ]
    
    data = generate_sample_data(30)
    
    for cfg in configs:
        print(f"\n{cfg['name']}:")
        try:
            service = create_technical_analysis_service(cfg['config'])
            rsi_result = service.rsi(data['close'], timeperiod=14)
            print(f"  Service type: {type(service).__name__}")
            print(f"  RSI sample: {rsi_result[-3:]}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Run all demonstrations."""
    print("Technical Analysis Module Demonstration")
    print("=" * 50)
    
    demo_backward_compatibility()
    demo_dependency_injection()
    demo_edge_cases()
    demo_configuration_options()
    
    print("\n\nDemo completed successfully!")
    print("\nNext steps:")
    print("1. Run tests: pytest tests/unit/test_technical_analysis.py -v")
    print("2. Update any imports from talib_stubs to technical_analysis")
    print("3. Configure your application to use production indicators")
    print("4. Remove talib_stubs.py when migration is complete")


if __name__ == "__main__":
    main() 