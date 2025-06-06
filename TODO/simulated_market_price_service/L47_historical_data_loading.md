# Task: Implement historical data loading and caching system with multiple data sources.

### 1. Context
- **File:** `gal_friday/simulated_market_price_service.py`
- **Line:** `47`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing historical data loading and caching system with support for multiple data sources.

### 2. Problem Statement
Without proper historical data loading and caching system, the simulated market price service cannot efficiently provide historical price data for backtesting, strategy development, and analysis. This prevents accurate simulation of market conditions and creates performance bottlenecks when repeatedly accessing the same historical data sets.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Data Source Abstraction:** Unified interface for multiple historical data providers
2. **Build Intelligent Caching System:** Multi-layer caching with memory, disk, and database tiers
3. **Implement Data Validation:** Comprehensive validation of historical data quality and completeness
4. **Add Data Preprocessing:** Normalization, resampling, and gap-filling capabilities
5. **Create Performance Optimization:** Efficient data loading with lazy loading and compression
6. **Build Monitoring System:** Real-time monitoring of data loading performance and cache efficiency

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
from datetime import datetime, timezone, timedelta
import pandas as pd

class DataSource(str, Enum):
    """Supported historical data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    BINANCE = "binance"
    KRAKEN = "kraken"
    LOCAL_FILES = "local_files"
    DATABASE = "database"

@dataclass
class DataRequest:
    """Historical data request specification"""
    symbol: str
    start_date: datetime
    end_date: datetime
    frequency: str
    data_source: Optional[DataSource] = None
    include_volume: bool = True
    validate_data: bool = True
    cache_result: bool = True

@dataclass
class HistoricalDataPoint:
    """Single historical data point"""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class HistoricalDataProvider(ABC):
    """Abstract base class for historical data providers"""
    
    @abstractmethod
    async def fetch_data(self, request: DataRequest) -> List[HistoricalDataPoint]:
        """Fetch historical data for the given request"""
        pass
    
    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by this provider"""
        pass

class CacheLayer(ABC):
    """Abstract base class for cache layers"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[List[HistoricalDataPoint]]:
        """Get data from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, data: List[HistoricalDataPoint], ttl: int = 3600) -> None:
        """Store data in cache with TTL"""
        pass

class HistoricalDataLoader:
    """Enterprise-grade historical data loading and caching system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize data providers
        self.providers: Dict[DataSource, HistoricalDataProvider] = {}
        self._initialize_providers()
        
        # Initialize cache layers
        self.memory_cache = MemoryCache(config.get('memory_cache_size', 1000))
        self.disk_cache = DiskCache(config.get('disk_cache_path', './cache'))
        
        # Performance tracking
        self.cache_stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'provider_requests': 0,
            'total_requests': 0
        }
    
    async def load_historical_data(self, request: DataRequest) -> List[HistoricalDataPoint]:
        """
        Load historical data with intelligent caching
        Replace TODO with comprehensive data loading system
        """
        
        try:
            self.logger.info(
                f"Loading historical data for {request.symbol}: "
                f"{request.start_date} to {request.end_date}"
            )
            
            self.cache_stats['total_requests'] += 1
            
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Try memory cache first
            data = await self.memory_cache.get(cache_key)
            if data:
                self.cache_stats['memory_hits'] += 1
                self.logger.debug("Data found in memory cache")
                return data
            
            # Try disk cache
            data = await self.disk_cache.get(cache_key)
            if data:
                self.cache_stats['disk_hits'] += 1
                self.logger.debug("Data found in disk cache")
                
                # Store in memory cache for faster future access
                if request.cache_result:
                    await self.memory_cache.set(cache_key, data)
                
                return data
            
            # Load from data provider
            data = await self._load_from_provider(request)
            
            # Validate data quality if requested
            if request.validate_data:
                quality_score = await self._validate_data_quality(data, request)
                self.logger.info(f"Data quality score: {quality_score:.2f}")
                
                if quality_score < 0.8:
                    self.logger.warning(f"Low data quality detected for {request.symbol}")
            
            # Cache the results
            if request.cache_result and data:
                await self.memory_cache.set(cache_key, data)
                await self.disk_cache.set(cache_key, data)
            
            self.logger.info(f"Loaded {len(data)} data points for {request.symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading historical data for {request.symbol}: {e}")
            raise DataLoadingError(f"Failed to load historical data: {e}")
    
    async def _load_from_provider(self, request: DataRequest) -> List[HistoricalDataPoint]:
        """Load data from appropriate provider"""
        
        self.cache_stats['provider_requests'] += 1
        
        # Determine data source
        data_source = request.data_source
        if not data_source:
            data_source = await self._select_best_provider(request)
        
        if data_source not in self.providers:
            raise ValueError(f"Data source {data_source.value} not available")
        
        provider = self.providers[data_source]
        
        # Validate symbol support
        if not await provider.validate_symbol(request.symbol):
            raise ValueError(f"Symbol {request.symbol} not supported by {data_source.value}")
        
        # Fetch data
        data = await provider.fetch_data(request)
        
        self.logger.debug(f"Fetched {len(data)} points from {data_source.value}")
        return data
    
    async def _select_best_provider(self, request: DataRequest) -> DataSource:
        """Select best data provider for the request"""
        
        # Provider selection logic based on symbol type and requirements
        symbol_lower = request.symbol.lower()
        
        if any(crypto in symbol_lower for crypto in ['btc', 'eth', 'usdt', 'bnb']):
            # Cryptocurrency symbols - prefer crypto exchanges
            if DataSource.BINANCE in self.providers:
                return DataSource.BINANCE
            elif DataSource.KRAKEN in self.providers:
                return DataSource.KRAKEN
        
        # Traditional assets - prefer traditional data providers
        if DataSource.ALPHA_VANTAGE in self.providers:
            return DataSource.ALPHA_VANTAGE
        elif DataSource.YAHOO_FINANCE in self.providers:
            return DataSource.YAHOO_FINANCE
        
        # Fallback to first available provider
        if self.providers:
            return next(iter(self.providers.keys()))
        
        raise ValueError("No data providers available")
    
    async def _validate_data_quality(self, data: List[HistoricalDataPoint], 
                                   request: DataRequest) -> float:
        """Validate data quality and completeness"""
        
        if not data:
            return 0.0
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'timestamp': point.timestamp,
                'open': point.open,
                'high': point.high,
                'low': point.low,
                'close': point.close,
                'volume': point.volume
            }
            for point in data
        ])
        
        # Check for duplicates
        duplicate_count = df.duplicated(subset=['timestamp']).sum()
        
        # Check for missing data points
        expected_points = self._calculate_expected_points(request)
        missing_points = max(0, expected_points - len(data))
        
        # Calculate quality score
        data_completeness = len(data) / expected_points if expected_points > 0 else 1.0
        duplicate_ratio = duplicate_count / len(data) if len(data) > 0 else 0
        
        quality_score = data_completeness * (1 - duplicate_ratio)
        
        return min(1.0, max(0.0, quality_score))
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate unique cache key for request"""
        
        key_parts = [
            request.symbol,
            request.start_date.isoformat(),
            request.end_date.isoformat(),
            request.frequency,
            str(request.data_source.value if request.data_source else 'auto'),
            str(request.include_volume)
        ]
        
        return "_".join(key_parts)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        total_requests = self.cache_stats['total_requests']
        cache_hit_rate = 0.0
        
        if total_requests > 0:
            total_hits = self.cache_stats['memory_hits'] + self.cache_stats['disk_hits']
            cache_hit_rate = total_hits / total_requests
        
        return {
            **self.cache_stats,
            'cache_hit_rate': cache_hit_rate,
            'memory_hit_rate': self.cache_stats['memory_hits'] / total_requests if total_requests > 0 else 0,
            'disk_hit_rate': self.cache_stats['disk_hits'] / total_requests if total_requests > 0 else 0
        }

class DataLoadingError(Exception):
    """Exception raised for data loading errors"""
    pass
