# Comprehensive Backtesting Framework

The enhanced BacktestingEngine provides a comprehensive framework for backtesting trading strategies with advanced performance analytics, multiple execution modes, benchmarking, and strategy optimization capabilities.

## Overview

The comprehensive backtesting framework builds on the existing BacktestingEngine infrastructure while adding:

- **Multiple Execution Modes**: Vectorized and event-driven backtesting
- **Enhanced Performance Analytics**: Comprehensive metrics including Sharpe ratio, Calmar ratio, drawdown analysis
- **Benchmarking System**: Compare strategies against market indices 
- **Strategy Comparison**: Test multiple strategies simultaneously
- **Parameter Optimization**: Grid search optimization for strategy parameters
- **Professional Reporting**: Detailed reports with performance summaries
- **Data Management**: Easy data loading and management utilities

## Quick Start

```python
import asyncio
import datetime as dt
from gal_friday.backtesting_engine import (
    BacktestingEngine, 
    BacktestConfig, 
    BacktestMode
)

# Initialize engine
config_manager = MockConfigManager()  # Your config manager
engine = BacktestingEngine(config=config_manager)

# Load data
engine.load_data_from_csv("historical_data.csv", symbol_column="symbol")

# Configure backtest
config = BacktestConfig(
    start_date=dt.datetime(2023, 1, 1),
    end_date=dt.datetime(2023, 12, 31),
    initial_capital=100000.0,
    symbols=['BTCUSD', 'ETHUSD'],
    mode=BacktestMode.VECTORIZED,
    benchmark_symbol='BTCUSD'
)

# Run backtest
results = await engine.run_backtest(config, your_strategy)

# Generate report
report = engine.generate_report(results, "backtest_report.txt")
```

## Core Components

### BacktestConfig

Configuration class for backtest parameters:

```python
@dataclass
class BacktestConfig:
    start_date: dt.datetime          # Backtest start date
    end_date: dt.datetime            # Backtest end date  
    initial_capital: float           # Starting capital
    symbols: list[str]               # Trading symbols
    mode: BacktestMode              # Execution mode
    commission_rate: float = 0.001   # Commission rate
    slippage_rate: float = 0.0005   # Slippage rate
    benchmark_symbol: str | None     # Benchmark for comparison
    output_dir: str = "results"      # Output directory
```

### BacktestMode

Execution modes available:

- `BacktestMode.VECTORIZED`: Fast vectorized backtesting for strategy development
- `BacktestMode.EVENT_DRIVEN`: Realistic event-driven simulation with full service integration

### PerformanceMetrics

Enhanced performance metrics:

```python
@dataclass
class PerformanceMetrics:
    total_return: float         # Total return
    annualized_return: float    # Annualized return
    volatility: float          # Return volatility
    sharpe_ratio: float        # Sharpe ratio
    sortino_ratio: float       # Sortino ratio
    calmar_ratio: float        # Calmar ratio
    max_drawdown: float        # Maximum drawdown
    total_trades: int          # Number of trades
    win_rate: float           # Win rate percentage
    profit_factor: float      # Profit factor
    benchmark_return: float   # Benchmark return (if applicable)
    alpha: float             # Alpha vs benchmark
    beta: float              # Beta vs benchmark
```

## Execution Modes

### Vectorized Mode

Fast backtesting suitable for strategy development and parameter optimization:

```python
config = BacktestConfig(
    # ... other parameters
    mode=BacktestMode.VECTORIZED
)

results = await engine.run_backtest(config, strategy)
```

**Advantages:**
- Very fast execution
- Good for parameter optimization
- Simplified execution model

**Use cases:**
- Strategy development
- Parameter optimization
- Quick performance estimates

### Event-Driven Mode

Realistic backtesting with full service integration:

```python
config = BacktestConfig(
    # ... other parameters  
    mode=BacktestMode.EVENT_DRIVEN
)

# Provide pre-initialized services
services = {
    "portfolio_manager": your_portfolio_manager,
    "execution_handler": your_execution_handler,
    # ... other services
}

results = await engine.run_backtest(config, strategy, services)
```

**Advantages:**
- Realistic simulation
- Full service integration
- Event-driven architecture

**Use cases:**
- Final strategy validation
- Production-like testing
- Complex strategy interactions

## Strategy Interface

Strategies should implement a signal generation interface:

```python
class YourStrategy:
    def __init__(self, name: str, **parameters):
        self.name = name
        # Initialize with parameters
    
    def generate_signals(self, timestamp, prices, positions, cash):
        """
        Generate trading signals.
        
        Args:
            timestamp: Current timestamp
            prices: Dict of {symbol: current_price}
            positions: Dict of {symbol: position_size} 
            cash: Available cash
            
        Returns:
            List of signal dictionaries:
            [{"symbol": "BTCUSD", "action": "buy", "quantity": 100}, ...]
        """
        signals = []
        # Your signal logic here
        return signals
```

## Benchmarking

Compare strategy performance against benchmarks:

```python
config = BacktestConfig(
    # ... other parameters
    benchmark_symbol='BTCUSD'  # Use BTC as benchmark
)

results = await engine.run_backtest(config, strategy)

# Benchmark analysis included in results
benchmark_analysis = results["benchmark_analysis"]
alpha = benchmark_analysis["alpha_pct"]  # Strategy alpha
```

## Strategy Comparison

Compare multiple strategies:

```python
strategies = [
    Strategy1(name="Strategy1", param1=10),
    Strategy2(name="Strategy2", param2=20),
    Strategy3(name="Strategy3", param3=30),
]

comparison_results = engine.run_strategy_comparison(strategies, config)

# Get best performing strategies
summary = comparison_results["comparison_summary"]
best_return = summary["best_total_return"]["strategy"]
best_sharpe = summary["best_sharpe_ratio"]["strategy"]
```

## Parameter Optimization

Optimize strategy parameters using grid search:

```python
parameter_grid = {
    'ma_fast': [5, 10, 15, 20],
    'ma_slow': [20, 30, 40, 50],
    'rsi_period': [10, 14, 20]
}

optimization_results = engine.optimize_parameters(
    StrategyClass,
    parameter_grid, 
    config,
    optimization_metric="sharpe_ratio"
)

best_params = optimization_results["best_result"]["parameters"]
best_sharpe = optimization_results["best_result"]["metric_value"]
```

Available optimization metrics:
- `total_return`
- `sharpe_ratio`
- `sortino_ratio`
- `calmar_ratio`
- `profit_factor`

## Data Management

### Loading Data

Load historical data from CSV:

```python
# Load multi-symbol data
engine.load_data_from_csv("data.csv", symbol_column="symbol")

# Load single symbol data  
engine.load_data_from_csv("btc_data.csv")  # Uses "DEFAULT_SYMBOL"
```

Required CSV format:
```csv
symbol,timestamp,open,high,low,close,volume
BTCUSD,2023-01-01,40000,41000,39500,40500,1000
ETHUSD,2023-01-01,2500,2550,2480,2520,5000
```

### Data Summary

Get information about loaded data:

```python
summary = engine.get_data_summary()
print(f"Loaded {summary['symbol_count']} symbols")

for symbol, details in summary['symbol_details'].items():
    print(f"{symbol}: {details['data_points']} points")
    print(f"  Period: {details['start_date']} to {details['end_date']}")
```

## Reporting

### Generate Reports

Create comprehensive backtest reports:

```python
# Generate and save report
report_content = engine.generate_report(results, "backtest_report.txt")

# Or just get report content
report_content = engine.generate_report(results)
print(report_content)
```

Example report output:
```
================================================================================
COMPREHENSIVE BACKTESTING REPORT
================================================================================

CONFIGURATION:
  Period: 2023-01-01 to 2023-12-31
  Initial Capital: $100,000.00
  Symbols: ['BTCUSD', 'ETHUSD']
  Mode: vectorized
  Commission Rate: 0.0010

PERFORMANCE METRICS:
  Total Return: 15.23%
  Annualized Return: 14.87%
  Volatility: 22.45%
  Sharpe Ratio: 0.66
  Sortino Ratio: 0.89
  Calmar Ratio: 1.12
  Max Drawdown: 13.25%
  Win Rate: 58.3%
  Total Trades: 24
  Profit Factor: 1.45

BENCHMARK COMPARISON:
  Benchmark Symbol: BTCUSD
  Benchmark Return: 8.45%
  Alpha: 6.78%

EXECUTION DETAILS:
  Execution Time: 2.34 seconds
  Framework Version: comprehensive_v1.0
```

## Integration with Existing Code

The comprehensive framework is designed to work with existing BacktestingEngine infrastructure:

### Compatibility

- **Existing performance calculations** are preserved and enhanced
- **Current service architecture** is maintained
- **Event-driven simulation** uses existing `_execute_simulation()` method
- **Configuration system** remains compatible

### Migration

To migrate from basic to comprehensive backtesting:

1. **Update imports**:
```python
from gal_friday.backtesting_engine import (
    BacktestingEngine, 
    BacktestConfig, 
    BacktestMode
)
```

2. **Use new configuration**:
```python
# Old way
run_config = {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31", 
    "initial_capital": 100000,
    "trading_pairs": ["BTCUSD"]
}

# New way  
config = BacktestConfig(
    start_date=dt.datetime(2023, 1, 1),
    end_date=dt.datetime(2023, 12, 31),
    initial_capital=100000.0,
    symbols=["BTCUSD"]
)
```

3. **Use public API**:
```python
# Old way (internal methods)
await engine._execute_simulation(services, run_config)

# New way (public API)
results = await engine.run_backtest(config, strategy, services)
```

## Best Practices

### Performance

- Use **vectorized mode** for parameter optimization and strategy development
- Use **event-driven mode** for final validation and production testing
- **Limit data ranges** for optimization to reasonable periods
- **Profile memory usage** with large datasets

### Strategy Development

- **Start simple** with basic strategies before adding complexity
- **Validate signals** with smaller datasets first
- **Use benchmarking** to ensure strategies add value
- **Consider transaction costs** in strategy design

### Testing

- **Test both execution modes** to ensure consistency
- **Validate against known results** when possible
- **Use realistic commission/slippage** rates
- **Test edge cases** (market gaps, low volume periods)

## Error Handling

The framework provides comprehensive error handling:

```python
from gal_friday.backtesting_engine import BacktestError

try:
    results = await engine.run_backtest(config, strategy)
except BacktestError as e:
    print(f"Backtest failed: {e}")
    # Handle error appropriately
```

Common error scenarios:
- **Missing data** for specified symbols/periods
- **Invalid strategy** signal generation
- **Configuration errors** (invalid dates, parameters)
- **Service initialization** failures

## Examples

See `examples/comprehensive_backtest_example.py` for a complete working example demonstrating all framework features.

## API Reference

### BacktestingEngine Methods

#### Core Methods
- `run_backtest(config, strategy, services)` - Run comprehensive backtest
- `run_strategy_comparison(strategies, config)` - Compare multiple strategies  
- `optimize_parameters(strategy_class, parameter_grid, config)` - Parameter optimization

#### Data Management
- `load_data_from_csv(file_path, symbol_column)` - Load historical data
- `get_data_summary()` - Get data information
- `clear_data()` - Clear loaded data

#### Reporting
- `generate_report(results, output_path)` - Generate comprehensive reports

#### Internal Methods (Advanced)
- `_run_vectorized_backtest()` - Vectorized execution
- `_add_benchmark_analysis()` - Benchmark comparison
- `_calculate_enhanced_metrics()` - Enhanced metrics calculation

For more details, see the source code documentation and examples. 