# Test Code in Production Files Implementation Design

**File**: `/gal_friday/simulated_market_price_service.py`
- **Lines 2733-2995**: Extensive example usage code mixed with production implementation
- **Line 2993**: `main_logger = logging.getLogger("SimulatedMarketPriceServiceExample")`
- **Issue**: Example/test code embedded in production module affects maintainability and deployment size

**File**: `/gal_friday/data_ingestor.py`
- **Line 1695**: `# Example Usage (for testing purposes - requires libraries installed)`
- **Line 1191**: `# This is a simplified implementation` for checksum calculation
- **Issue**: Test/example code mixed with production logic creates deployment bloat

**Impact**: Production modules contain test code increasing deployment size, potential security exposure, and maintenance complexity

## Overview
Production modules contain embedded example/test code that should be separated into dedicated test files or example directories. This design implements clean separation of concerns with proper modularization while maintaining functionality for development and testing workflows.

## Architecture Design

### 1. Current Implementation Issues

```
Test Code in Production Problems:
├── SimulatedMarketPriceService Module
│   ├── 260+ lines of example code in production file
│   ├── Test data generation mixed with core logic
│   ├── Example async functions in production namespace
│   └── Development dependencies in production imports
├── DataIngestor Module
│   ├── Example usage function in production file
│   ├── Test configuration generation code
│   ├── Mock services embedded in production
│   └── Development logging in production module
└── Deployment Impact
    ├── Increased bundle size with unused code
    ├── Potential security exposure of test data
    ├── Maintenance complexity from mixed concerns
    └── Unclear separation between test and production code
```

### 2. Production Code Separation Architecture

```
Clean Module Architecture:
├── Production Modules (Core Business Logic Only)
│   ├── SimulatedMarketPriceService
│   │   ├── Core service implementation
│   │   ├── Production methods only
│   │   ├── Minimal imports for runtime
│   │   └── Clean API surface
│   ├── DataIngestor
│   │   ├── Core ingestion logic
│   │   ├── Production error handling
│   │   ├── Runtime-only dependencies
│   │   └── Clean separation of concerns
│   └── Configuration Management
│       ├── Production config validation
│       ├── Runtime environment detection
│       └── Clean deployment artifacts
├── Example/Demo Modules (Separate Files)
│   ├── examples/simulated_market_demo.py
│   │   ├── Complete usage demonstrations
│   │   ├── Sample data generation
│   │   ├── Interactive examples
│   │   └── Development-only imports
│   ├── examples/data_ingestor_demo.py
│   │   ├── Comprehensive usage examples
│   │   ├── Mock service configurations
│   │   ├── Test data generation
│   │   └── Example workflows
│   └── Example Configuration
│       ├── Environment-specific examples
│       ├── Sample data sets
│       └── Development utilities
├── Test Modules (Dedicated Test Files)
│   ├── tests/integration/test_simulated_market.py
│   │   ├── Comprehensive integration tests
│   │   ├── Mock data factories
│   │   ├── Test utilities
│   │   └── Assertion frameworks
│   ├── tests/unit/test_data_ingestor.py
│   │   ├── Unit test coverage
│   │   ├── Mock configurations
│   │   ├── Error scenario testing
│   │   └── Performance benchmarks
│   └── Test Infrastructure
│       ├── Shared test utilities
│       ├── Mock service factories
│       └── Test data management
└── Documentation & Utilities
    ├── README examples with code references
    ├── Interactive notebooks for demonstrations
    ├── API documentation with usage examples
    └── Development setup guides
```

### 3. Key Features

1. **Clean Separation**: Complete isolation of production, example, and test code
2. **Maintainability**: Clear module boundaries and responsibilities
3. **Deployment Efficiency**: Minimal production bundle size
4. **Developer Experience**: Rich examples and documentation without production bloat
5. **Security**: No test data or development utilities in production deployment
6. **Modularity**: Reusable example and test components

## Implementation Plan

### Phase 1: Extract SimulatedMarketPriceService Examples

**File**: `/gal_friday/simulated_market_price_service.py`
**Target Lines**: Lines 2733-2995 - Remove embedded example code

**Action**: Clean production module by removing example code:

```python
# Remove from production file (lines 2733-2995):
# - async def _setup_service_and_data()
# - async def _test_price_interpolation()
# - async def _test_price_simulation()
# - async def _test_price_trend_analysis()
# - async def _test_volatility_calculation()
# - async def _main_example()
# - if __name__ == "__main__": asyncio.run(_main_example())

# Keep only production class and methods:
class SimulatedMarketPriceService:
    """Production-ready simulated market price service."""
    
    def __init__(self, config: MarketSimulationConfig, logger_service: LoggerService):
        """Initialize with production configuration only."""
        self.config = config
        self.logger = logger_service
        # Production initialization only
    
    # ... all production methods remain unchanged ...
```

**New File**: `/examples/simulated_market_demo.py`

```python
"""
Comprehensive demonstration of SimulatedMarketPriceService capabilities.

This module provides interactive examples and demonstrations for the
SimulatedMarketPriceService. Use this for learning, testing, and
development workflows.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pandas as pd

from gal_friday.simulated_market_price_service import (
    SimulatedMarketPriceService,
    MarketSimulationConfig,
    PriceSimulationResult,
    TrendAnalysisResult,
    VolatilityAnalysisResult
)
from gal_friday.logger_service import LoggerService


class SimulatedMarketDemoRunner:
    """Comprehensive demo runner for market simulation examples."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.service: SimulatedMarketPriceService | None = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for demo environment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("SimulatedMarketDemo")
    
    async def setup_demo_environment(self) -> Tuple[SimulatedMarketPriceService, datetime]:
        """Set up comprehensive demo environment with sample data."""
        # Create rich sample historical data
        sample_data = self._generate_comprehensive_sample_data()
        
        # Configure demo service
        config = MarketSimulationConfig(
            default_volatility=Decimal("0.02"),
            trend_strength=Decimal("0.15"),
            noise_factor=Decimal("0.008"),
            price_precision=8,
            enable_gap_detection=True,
            enable_trend_analysis=True
        )
        
        # Initialize service with mock logger
        mock_logger = MockLoggerService()
        self.service = SimulatedMarketPriceService(config, mock_logger)
        
        # Load sample data
        await self.service.load_historical_data("DEMO/USDT", sample_data)
        
        base_timestamp = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        self.logger.info("Demo environment initialized successfully")
        return self.service, base_timestamp
    
    def _generate_comprehensive_sample_data(self) -> pd.DataFrame:
        """Generate rich sample data for comprehensive demonstrations."""
        timestamps = pd.date_range(
            start="2023-01-01 00:00:00",
            end="2023-01-01 23:59:00",
            freq="1min",
            tz="UTC"
        )
        
        # Generate realistic price data with trends and volatility
        base_price = Decimal("100.0")
        prices = []
        
        for i, ts in enumerate(timestamps):
            # Add trend component
            trend = Decimal(str(0.001 * i))
            
            # Add volatility component
            volatility = Decimal(str(0.5 * math.sin(i * 0.01)))
            
            # Add noise
            noise = Decimal(str(random.uniform(-0.2, 0.2)))
            
            price = base_price + trend + volatility + noise
            prices.append(max(price, Decimal("1.0")))  # Ensure positive prices
        
        return pd.DataFrame({
            "open": [p * Decimal("0.999") for p in prices],
            "high": [p * Decimal("1.002") for p in prices],
            "low": [p * Decimal("0.998") for p in prices],
            "close": prices,
            "volume": [Decimal(str(random.uniform(1000, 10000))) for _ in prices]
        }, index=timestamps)
    
    async def demonstrate_price_interpolation(self) -> None:
        """Demonstrate advanced price interpolation capabilities."""
        self.logger.info("=== Price Interpolation Demonstration ===")
        
        if not self.service:
            await self.setup_demo_environment()
        
        # Test various interpolation scenarios
        test_scenarios = [
            {
                "name": "Standard Interpolation",
                "timestamp": datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
                "description": "Normal price interpolation between data points"
            },
            {
                "name": "Edge Case - Start of Data",
                "timestamp": datetime(2023, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
                "description": "Interpolation near data boundaries"
            },
            {
                "name": "High Volatility Period",
                "timestamp": datetime(2023, 1, 1, 18, 45, 0, tzinfo=timezone.utc),
                "description": "Interpolation during high volatility"
            }
        ]
        
        for scenario in test_scenarios:
            self.logger.info(f"\n--- {scenario['name']} ---")
            self.logger.info(f"Description: {scenario['description']}")
            
            try:
                result = await self.service.get_interpolated_price(
                    "DEMO/USDT", 
                    scenario["timestamp"]
                )
                
                self.logger.info(f"Interpolated Price: {result.price}")
                self.logger.info(f"Confidence: {result.confidence}")
                self.logger.info(f"Data Quality: {result.data_quality}")
                
                if result.metadata:
                    self.logger.info(f"Metadata: {result.metadata}")
                    
            except Exception as e:
                self.logger.error(f"Interpolation failed: {e}")
    
    async def demonstrate_price_simulation(self) -> None:
        """Demonstrate comprehensive price simulation features."""
        self.logger.info("\n=== Price Simulation Demonstration ===")
        
        if not self.service:
            await self.setup_demo_environment()
        
        simulation_scenarios = [
            {
                "name": "Short-term Simulation (1 hour)",
                "duration": timedelta(hours=1),
                "steps": 60,
                "description": "High-frequency short-term price simulation"
            },
            {
                "name": "Medium-term Simulation (6 hours)",
                "duration": timedelta(hours=6),
                "steps": 360,
                "description": "Medium-term trend following simulation"
            },
            {
                "name": "Long-term Simulation (24 hours)",
                "duration": timedelta(hours=24),
                "steps": 1440,
                "description": "Long-term price evolution simulation"
            }
        ]
        
        base_timestamp = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        
        for scenario in simulation_scenarios:
            self.logger.info(f"\n--- {scenario['name']} ---")
            self.logger.info(f"Description: {scenario['description']}")
            
            try:
                result = await self.service.simulate_price_path(
                    trading_pair="DEMO/USDT",
                    start_timestamp=base_timestamp,
                    duration=scenario["duration"],
                    steps=scenario["steps"]
                )
                
                self.logger.info(f"Simulation Points: {len(result.price_path)}")
                self.logger.info(f"Start Price: {result.price_path[0].price}")
                self.logger.info(f"End Price: {result.price_path[-1].price}")
                
                # Calculate simulation statistics
                prices = [point.price for point in result.price_path]
                price_change = ((prices[-1] - prices[0]) / prices[0]) * 100
                max_price = max(prices)
                min_price = min(prices)
                volatility = self._calculate_volatility(prices)
                
                self.logger.info(f"Price Change: {price_change:.2f}%")
                self.logger.info(f"Price Range: {min_price} - {max_price}")
                self.logger.info(f"Realized Volatility: {volatility:.4f}")
                
            except Exception as e:
                self.logger.error(f"Simulation failed: {e}")
    
    async def demonstrate_trend_analysis(self) -> None:
        """Demonstrate advanced trend analysis capabilities."""
        self.logger.info("\n=== Trend Analysis Demonstration ===")
        
        if not self.service:
            await self.setup_demo_environment()
        
        analysis_periods = [
            {"hours": 1, "name": "Short-term Trend"},
            {"hours": 6, "name": "Medium-term Trend"},
            {"hours": 12, "name": "Long-term Trend"}
        ]
        
        analysis_timestamp = datetime(2023, 1, 1, 18, 0, 0, tzinfo=timezone.utc)
        
        for period in analysis_periods:
            self.logger.info(f"\n--- {period['name']} Analysis ---")
            
            try:
                result = await self.service.analyze_price_trend(
                    trading_pair="DEMO/USDT",
                    timestamp=analysis_timestamp,
                    lookback_hours=period["hours"]
                )
                
                self.logger.info(f"Trend Direction: {result.trend_direction}")
                self.logger.info(f"Trend Strength: {result.trend_strength}")
                self.logger.info(f"Confidence: {result.confidence}")
                self.logger.info(f"Price Change: {result.price_change_percent:.2f}%")
                
                if result.support_levels:
                    self.logger.info(f"Support Levels: {result.support_levels}")
                if result.resistance_levels:
                    self.logger.info(f"Resistance Levels: {result.resistance_levels}")
                    
            except Exception as e:
                self.logger.error(f"Trend analysis failed: {e}")
    
    async def demonstrate_volatility_analysis(self) -> None:
        """Demonstrate comprehensive volatility analysis features."""
        self.logger.info("\n=== Volatility Analysis Demonstration ===")
        
        if not self.service:
            await self.setup_demo_environment()
        
        analysis_timestamp = datetime(2023, 1, 1, 20, 0, 0, tzinfo=timezone.utc)
        
        try:
            result = await self.service.analyze_volatility(
                trading_pair="DEMO/USDT",
                timestamp=analysis_timestamp,
                lookback_hours=6
            )
            
            self.logger.info(f"Current Volatility: {result.current_volatility}")
            self.logger.info(f"Historical Average: {result.historical_average}")
            self.logger.info(f"Volatility Percentile: {result.volatility_percentile}")
            self.logger.info(f"Volatility Regime: {result.volatility_regime}")
            
            if result.volatility_bands:
                self.logger.info(f"Volatility Bands: {result.volatility_bands}")
            
            if result.risk_metrics:
                self.logger.info(f"Risk Metrics: {result.risk_metrics}")
                
        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {e}")
    
    def _calculate_volatility(self, prices: List[Decimal]) -> float:
        """Calculate simple volatility for demonstration."""
        if len(prices) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(prices)):
            ret = float((prices[i] - prices[i-1]) / prices[i-1])
            returns.append(ret)
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)
    
    async def run_comprehensive_demo(self) -> None:
        """Run complete demonstration of all features."""
        self.logger.info("Starting Comprehensive SimulatedMarketPriceService Demo")
        self.logger.info("=" * 60)
        
        try:
            await self.setup_demo_environment()
            await self.demonstrate_price_interpolation()
            await self.demonstrate_price_simulation()
            await self.demonstrate_trend_analysis()
            await self.demonstrate_volatility_analysis()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Demo completed successfully!")
            self.logger.info("Check the logs above for detailed results.")
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise


class MockLoggerService:
    """Mock logger service for demonstration purposes."""
    
    def info(self, message: str, **kwargs) -> None:
        logging.getLogger("MockLogger").info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        logging.getLogger("MockLogger").warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        logging.getLogger("MockLogger").error(message)


async def main() -> None:
    """Entry point for interactive demonstration."""
    demo_runner = SimulatedMarketDemoRunner()
    await demo_runner.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 2: Extract DataIngestor Examples

**File**: `/gal_friday/data_ingestor.py`
**Target Line**: Line 1695 - Remove embedded example code

**Action**: Clean production module:

```python
# Remove from production file (lines 1695+):
# - async def main()
# - def _create_test_config()
# - def _setup_logging()
# - async def _run_test()
# - if __name__ == "__main__": asyncio.run(main())

# Keep only production class and methods
class DataIngestor:
    """Production-ready data ingestion service."""
    
    def __init__(self, config: DataIngestorConfig, event_bus: asyncio.Queue, logger_service: LoggerService):
        """Initialize with production configuration only."""
        # Production initialization only
    
    # ... all production methods remain unchanged ...
```

**New File**: `/examples/data_ingestor_demo.py`

```python
"""
Comprehensive demonstration of DataIngestor capabilities.

This module provides examples, test scenarios, and interactive
demonstrations for the DataIngestor service.
"""

import asyncio
import logging
from typing import Any, Dict, List
from datetime import datetime, timezone
from decimal import Decimal

from gal_friday.data_ingestor import DataIngestor, DataIngestorConfig
from gal_friday.logger_service import LoggerService


class DataIngestorDemoRunner:
    """Comprehensive demo runner for data ingestion examples."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.ingestor: DataIngestor | None = None
        self.event_bus: asyncio.Queue[Any] = asyncio.Queue()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging for demo environment."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('data_ingestor_demo.log')
            ]
        )
        return logging.getLogger("DataIngestorDemo")
    
    def create_comprehensive_demo_config(self) -> DataIngestorConfig:
        """Create comprehensive configuration for demonstration."""
        return DataIngestorConfig(
            # Exchange configurations for demo
            exchange_configs={
                "binance": {
                    "api_key": "demo_api_key",
                    "api_secret": "demo_api_secret",
                    "base_url": "https://api.binance.com",
                    "rate_limit": 1200,
                    "timeout": 30
                },
                "coinbase": {
                    "api_key": "demo_coinbase_key",
                    "api_secret": "demo_coinbase_secret",
                    "base_url": "https://api.exchange.coinbase.com",
                    "rate_limit": 300,
                    "timeout": 30
                }
            },
            
            # Data source configurations
            data_sources=[
                {
                    "name": "market_data",
                    "type": "real_time",
                    "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT"],
                    "intervals": ["1m", "5m", "1h"],
                    "enabled": True
                },
                {
                    "name": "order_book",
                    "type": "snapshot",
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "depth": 20,
                    "enabled": True
                },
                {
                    "name": "trade_stream",
                    "type": "real_time",
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "enabled": True
                }
            ],
            
            # Processing configurations
            processing_config={
                "batch_size": 100,
                "max_queue_size": 10000,
                "processing_timeout": 30,
                "error_retry_limit": 3,
                "enable_data_validation": True,
                "enable_duplicate_detection": True
            },
            
            # Storage configurations
            storage_config={
                "primary_store": "postgresql",
                "backup_store": "file_system",
                "compression_enabled": True,
                "encryption_enabled": False,  # Demo only
                "retention_days": 30
            }
        )
    
    async def setup_demo_environment(self) -> None:
        """Set up comprehensive demo environment."""
        config = self.create_comprehensive_demo_config()
        mock_logger = MockLoggerService()
        
        self.ingestor = DataIngestor(config, self.event_bus, mock_logger)
        
        self.logger.info("Demo environment initialized successfully")
        self.logger.info(f"Event bus created: {type(self.event_bus)}")
        self.logger.info(f"Configuration loaded: {len(config.data_sources)} data sources")
    
    async def demonstrate_real_time_ingestion(self) -> None:
        """Demonstrate real-time data ingestion capabilities."""
        self.logger.info("\n=== Real-Time Data Ingestion Demo ===")
        
        if not self.ingestor:
            await self.setup_demo_environment()
        
        # Simulate real-time market data
        sample_market_data = [
            {
                "symbol": "BTC/USDT",
                "timestamp": datetime.now(timezone.utc),
                "open": Decimal("45000.00"),
                "high": Decimal("45500.00"),
                "low": Decimal("44800.00"),
                "close": Decimal("45200.00"),
                "volume": Decimal("123.45"),
                "source": "binance"
            },
            {
                "symbol": "ETH/USDT",
                "timestamp": datetime.now(timezone.utc),
                "open": Decimal("3200.00"),
                "high": Decimal("3250.00"),
                "low": Decimal("3180.00"),
                "close": Decimal("3220.00"),
                "volume": Decimal("456.78"),
                "source": "binance"
            }
        ]
        
        try:
            # Process sample data
            for data_point in sample_market_data:
                await self.ingestor.process_market_data(data_point)
                self.logger.info(f"Processed: {data_point['symbol']} - {data_point['close']}")
            
            # Check event bus for processed events
            processed_events = []
            while not self.event_bus.empty():
                try:
                    event = self.event_bus.get_nowait()
                    processed_events.append(event)
                except asyncio.QueueEmpty:
                    break
            
            self.logger.info(f"Generated {len(processed_events)} events from ingestion")
            
        except Exception as e:
            self.logger.error(f"Real-time ingestion demo failed: {e}")
    
    async def demonstrate_batch_processing(self) -> None:
        """Demonstrate batch data processing capabilities."""
        self.logger.info("\n=== Batch Processing Demo ===")
        
        if not self.ingestor:
            await self.setup_demo_environment()
        
        # Simulate historical batch data
        batch_data = []
        base_timestamp = datetime.now(timezone.utc)
        
        for i in range(50):  # 50 data points
            timestamp = base_timestamp.replace(minute=i)
            batch_data.append({
                "symbol": "BTC/USDT",
                "timestamp": timestamp,
                "open": Decimal("45000.00") + Decimal(str(i * 10)),
                "high": Decimal("45500.00") + Decimal(str(i * 10)),
                "low": Decimal("44800.00") + Decimal(str(i * 10)),
                "close": Decimal("45200.00") + Decimal(str(i * 10)),
                "volume": Decimal("100.00") + Decimal(str(i)),
                "source": "historical"
            })
        
        try:
            start_time = datetime.now()
            await self.ingestor.process_batch_data(batch_data)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Processed {len(batch_data)} records in {processing_time:.2f} seconds")
            self.logger.info(f"Throughput: {len(batch_data) / processing_time:.2f} records/second")
            
        except Exception as e:
            self.logger.error(f"Batch processing demo failed: {e}")
    
    async def demonstrate_error_handling(self) -> None:
        """Demonstrate comprehensive error handling."""
        self.logger.info("\n=== Error Handling Demo ===")
        
        if not self.ingestor:
            await self.setup_demo_environment()
        
        # Test various error scenarios
        error_scenarios = [
            {
                "name": "Invalid Data Format",
                "data": {"invalid": "data_structure"},
                "expected_error": "Data validation error"
            },
            {
                "name": "Missing Required Fields",
                "data": {"symbol": "BTC/USDT"},  # Missing other required fields
                "expected_error": "Missing required fields"
            },
            {
                "name": "Invalid Price Data",
                "data": {
                    "symbol": "BTC/USDT",
                    "timestamp": datetime.now(timezone.utc),
                    "open": "invalid_price",
                    "close": Decimal("45000.00")
                },
                "expected_error": "Price conversion error"
            }
        ]
        
        for scenario in error_scenarios:
            self.logger.info(f"\n--- Testing: {scenario['name']} ---")
            
            try:
                await self.ingestor.process_market_data(scenario["data"])
                self.logger.warning(f"Expected error but processing succeeded!")
                
            except Exception as e:
                self.logger.info(f"Caught expected error: {type(e).__name__}: {e}")
                self.logger.info(f"Error handling working correctly")
    
    async def demonstrate_performance_monitoring(self) -> None:
        """Demonstrate performance monitoring and metrics."""
        self.logger.info("\n=== Performance Monitoring Demo ===")
        
        if not self.ingestor:
            await self.setup_demo_environment()
        
        # Generate load test data
        load_test_data = []
        for i in range(1000):  # 1000 data points for load testing
            load_test_data.append({
                "symbol": f"SYMBOL_{i % 10}/USDT",
                "timestamp": datetime.now(timezone.utc),
                "open": Decimal("100.00"),
                "high": Decimal("105.00"),
                "low": Decimal("95.00"),
                "close": Decimal("102.00"),
                "volume": Decimal("50.00"),
                "source": "load_test"
            })
        
        try:
            # Monitor performance during load test
            start_time = datetime.now()
            memory_start = self._get_memory_usage()
            
            await self.ingestor.process_batch_data(load_test_data)
            
            end_time = datetime.now()
            memory_end = self._get_memory_usage()
            
            processing_time = (end_time - start_time).total_seconds()
            memory_delta = memory_end - memory_start
            
            self.logger.info(f"Load Test Results:")
            self.logger.info(f"  Records Processed: {len(load_test_data)}")
            self.logger.info(f"  Total Time: {processing_time:.2f} seconds")
            self.logger.info(f"  Throughput: {len(load_test_data) / processing_time:.2f} records/sec")
            self.logger.info(f"  Memory Delta: {memory_delta:.2f} MB")
            self.logger.info(f"  Avg Time per Record: {(processing_time * 1000) / len(load_test_data):.2f} ms")
            
        except Exception as e:
            self.logger.error(f"Performance monitoring demo failed: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified)."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    async def run_comprehensive_demo(self) -> None:
        """Run complete demonstration of all features."""
        self.logger.info("Starting Comprehensive DataIngestor Demo")
        self.logger.info("=" * 50)
        
        try:
            await self.setup_demo_environment()
            await self.demonstrate_real_time_ingestion()
            await self.demonstrate_batch_processing()
            await self.demonstrate_error_handling()
            await self.demonstrate_performance_monitoring()
            
            self.logger.info("\n" + "=" * 50)
            self.logger.info("Demo completed successfully!")
            self.logger.info("Check 'data_ingestor_demo.log' for detailed results.")
            
        except KeyboardInterrupt:
            self.logger.info("Demo interrupted by user")
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            raise


class MockLoggerService:
    """Mock logger service for demonstration purposes."""
    
    def info(self, message: str, **kwargs) -> None:
        logging.getLogger("MockLogger").info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        logging.getLogger("MockLogger").warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        logging.getLogger("MockLogger").error(message)


async def main() -> None:
    """Entry point for interactive demonstration."""
    demo_runner = DataIngestorDemoRunner()
    await demo_runner.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 3: Create Comprehensive Test Suite

**New File**: `/tests/integration/test_simulated_market_comprehensive.py`

```python
"""
Comprehensive integration tests for SimulatedMarketPriceService.

This module provides complete test coverage including edge cases,
performance tests, and integration scenarios.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import pandas as pd

from gal_friday.simulated_market_price_service import SimulatedMarketPriceService, MarketSimulationConfig
from gal_friday.logger_service import LoggerService
from tests.fixtures.mock_logger import MockLoggerService
from tests.fixtures.sample_data import MarketDataFactory


class TestSimulatedMarketPriceServiceIntegration:
    """Comprehensive integration test suite."""
    
    @pytest.fixture
    async def service_with_data(self):
        """Set up service with comprehensive test data."""
        config = MarketSimulationConfig(
            default_volatility=Decimal("0.02"),
            trend_strength=Decimal("0.15"),
            noise_factor=Decimal("0.01")
        )
        
        logger = MockLoggerService()
        service = SimulatedMarketPriceService(config, logger)
        
        # Load comprehensive test data
        test_data = MarketDataFactory.create_comprehensive_dataset(
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            duration_hours=24,
            frequency_minutes=1
        )
        
        await service.load_historical_data("TEST/USDT", test_data)
        return service
    
    @pytest.mark.asyncio
    async def test_price_interpolation_comprehensive(self, service_with_data):
        """Test price interpolation across various scenarios."""
        service = service_with_data
        
        # Test scenarios
        test_cases = [
            {
                "name": "Standard interpolation",
                "timestamp": datetime(2023, 1, 1, 12, 30, 0, tzinfo=timezone.utc),
                "expected_confidence": 0.8
            },
            {
                "name": "Edge case - boundary",
                "timestamp": datetime(2023, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
                "expected_confidence": 0.6
            },
            {
                "name": "High frequency data",
                "timestamp": datetime(2023, 1, 1, 18, 45, 15, tzinfo=timezone.utc),
                "expected_confidence": 0.9
            }
        ]
        
        for case in test_cases:
            result = await service.get_interpolated_price("TEST/USDT", case["timestamp"])
            
            assert result.price > 0
            assert result.confidence >= case["expected_confidence"]
            assert result.data_quality in ["high", "medium", "low"]
            assert isinstance(result.metadata, dict)
    
    @pytest.mark.asyncio
    async def test_price_simulation_comprehensive(self, service_with_data):
        """Test price simulation with various parameters."""
        service = service_with_data
        
        simulation_scenarios = [
            {"duration": timedelta(hours=1), "steps": 60, "name": "Short-term"},
            {"duration": timedelta(hours=6), "steps": 360, "name": "Medium-term"},
            {"duration": timedelta(hours=24), "steps": 1440, "name": "Long-term"}
        ]
        
        base_timestamp = datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        
        for scenario in simulation_scenarios:
            result = await service.simulate_price_path(
                trading_pair="TEST/USDT",
                start_timestamp=base_timestamp,
                duration=scenario["duration"],
                steps=scenario["steps"]
            )
            
            assert len(result.price_path) == scenario["steps"]
            assert all(point.price > 0 for point in result.price_path)
            assert result.simulation_metadata["scenario_name"] == scenario["name"]
            
            # Verify timestamp progression
            timestamps = [point.timestamp for point in result.price_path]
            for i in range(1, len(timestamps)):
                assert timestamps[i] > timestamps[i-1]
    
    @pytest.mark.asyncio
    async def test_trend_analysis_comprehensive(self, service_with_data):
        """Test comprehensive trend analysis functionality."""
        service = service_with_data
        
        analysis_timestamp = datetime(2023, 1, 1, 18, 0, 0, tzinfo=timezone.utc)
        
        for lookback_hours in [1, 6, 12, 24]:
            result = await service.analyze_price_trend(
                trading_pair="TEST/USDT",
                timestamp=analysis_timestamp,
                lookback_hours=lookback_hours
            )
            
            assert result.trend_direction in ["up", "down", "sideways"]
            assert 0 <= result.trend_strength <= 1
            assert 0 <= result.confidence <= 1
            assert isinstance(result.price_change_percent, float)
            
            if result.support_levels:
                assert all(level > 0 for level in result.support_levels)
            if result.resistance_levels:
                assert all(level > 0 for level in result.resistance_levels)
    
    @pytest.mark.asyncio
    async def test_volatility_analysis_comprehensive(self, service_with_data):
        """Test comprehensive volatility analysis."""
        service = service_with_data
        
        analysis_timestamp = datetime(2023, 1, 1, 20, 0, 0, tzinfo=timezone.utc)
        
        result = await service.analyze_volatility(
            trading_pair="TEST/USDT",
            timestamp=analysis_timestamp,
            lookback_hours=6
        )
        
        assert result.current_volatility >= 0
        assert result.historical_average >= 0
        assert 0 <= result.volatility_percentile <= 100
        assert result.volatility_regime in ["low", "medium", "high", "extreme"]
        
        if result.volatility_bands:
            assert "upper" in result.volatility_bands
            assert "lower" in result.volatility_bands
        
        if result.risk_metrics:
            assert isinstance(result.risk_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling_comprehensive(self, service_with_data):
        """Test comprehensive error handling scenarios."""
        service = service_with_data
        
        # Test invalid trading pair
        with pytest.raises(ValueError, match="Trading pair not found"):
            await service.get_interpolated_price("INVALID/PAIR", datetime.now(timezone.utc))
        
        # Test timestamp outside data range
        future_timestamp = datetime(2030, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Timestamp outside available data range"):
            await service.get_interpolated_price("TEST/USDT", future_timestamp)
        
        # Test invalid simulation parameters
        with pytest.raises(ValueError, match="Invalid simulation parameters"):
            await service.simulate_price_path(
                trading_pair="TEST/USDT",
                start_timestamp=datetime.now(timezone.utc),
                duration=timedelta(hours=-1),  # Negative duration
                steps=100
            )
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, service_with_data):
        """Test performance benchmarks for all operations."""
        service = service_with_data
        
        import time
        
        # Benchmark interpolation
        start_time = time.time()
        for i in range(100):
            timestamp = datetime(2023, 1, 1, 12, i % 60, 0, tzinfo=timezone.utc)
            await service.get_interpolated_price("TEST/USDT", timestamp)
        interpolation_time = time.time() - start_time
        
        # Benchmark simulation
        start_time = time.time()
        await service.simulate_price_path(
            trading_pair="TEST/USDT",
            start_timestamp=datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            duration=timedelta(hours=6),
            steps=360
        )
        simulation_time = time.time() - start_time
        
        # Performance assertions
        assert interpolation_time < 10.0  # 100 interpolations under 10 seconds
        assert simulation_time < 5.0  # 360-step simulation under 5 seconds
        
        print(f"Performance Results:")
        print(f"  Interpolation: {interpolation_time:.2f}s for 100 operations")
        print(f"  Simulation: {simulation_time:.2f}s for 360 steps")
```

### Phase 4: Enhanced Documentation Integration

**New File**: `/docs/examples/README.md`

```markdown
# Examples and Demonstrations

This directory contains comprehensive examples and interactive demonstrations
for all Gal-Friday2 components.

## Directory Structure

```
examples/
├── simulated_market_demo.py      # Market simulation demonstrations
├── data_ingestor_demo.py         # Data ingestion examples
├── trading_strategy_examples/    # Strategy implementation examples
├── portfolio_management_demo.py  # Portfolio management demonstrations
└── integration_examples/         # End-to-end integration examples
```

## Running Examples

### SimulatedMarketPriceService Demo

```bash
# Run comprehensive market simulation demo
python examples/simulated_market_demo.py

# Run specific demo components
python -c "
from examples.simulated_market_demo import SimulatedMarketDemoRunner
import asyncio

async def run_interpolation_demo():
    runner = SimulatedMarketDemoRunner()
    await runner.demonstrate_price_interpolation()

asyncio.run(run_interpolation_demo())
"
```

### DataIngestor Demo

```bash
# Run comprehensive data ingestion demo
python examples/data_ingestor_demo.py

# Run performance monitoring demo only
python -c "
from examples.data_ingestor_demo import DataIngestorDemoRunner
import asyncio

async def run_performance_demo():
    runner = DataIngestorDemoRunner()
    await runner.demonstrate_performance_monitoring()

asyncio.run(run_performance_demo())
"
```

## Example Configuration

All examples use safe demo configurations that:
- Don't require real API keys
- Use mock services where appropriate
- Generate realistic sample data
- Include comprehensive error handling
- Provide detailed logging

## Interactive Development

Examples are designed for interactive development:
- Rich logging output for understanding behavior
- Modular functions for testing specific features
- Comprehensive error handling with helpful messages
- Performance monitoring and benchmarks
```

## Testing Strategy

1. **Module Separation Tests**
   - Verify production modules contain no example code
   - Test that examples run independently
   - Validate import dependencies

2. **Functionality Tests**
   - Ensure examples demonstrate all features
   - Test error scenarios in examples
   - Verify mock services work correctly

3. **Performance Tests**
   - Benchmark example execution times
   - Monitor memory usage during demos
   - Test with various data sizes

4. **Documentation Tests**
   - Verify all examples are documented
   - Test code snippets in README files
   - Validate example configurations

## Deployment Strategy

1. **Production Builds**
   - Exclude examples/ directory from production
   - Strip test code from production modules
   - Minimize production bundle size

2. **Development Builds**
   - Include full examples and documentation
   - Enable development logging
   - Include interactive tools

3. **Security Considerations**
   - No real credentials in example code
   - Safe demo configurations only
   - Clear separation of demo vs production code