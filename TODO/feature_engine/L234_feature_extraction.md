# Task: Implement advanced feature extraction with technical indicators and market microstructure.

### 1. Context
- **File:** `gal_friday/feature_engine.py`
- **Line:** `234`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing advanced feature extraction with technical indicators and market microstructure.

### 2. Problem Statement
Without advanced feature extraction capabilities, the trading system cannot generate sophisticated features from market data for ML models and trading strategies. This limits the system's ability to capture complex market patterns, technical indicators, and microstructure signals that are crucial for effective trading decisions.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Feature Extraction Framework:** Comprehensive feature engineering pipeline
2. **Build Technical Indicators Library:** Extensive collection of technical analysis indicators
3. **Implement Market Microstructure Features:** Order book, volume, and tick-level features
4. **Add Statistical Features:** Rolling statistics, volatility measures, and correlation features
5. **Create Feature Selection Engine:** Automated feature selection and importance ranking
6. **Build Feature Storage System:** Efficient storage and retrieval of computed features

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import talib
from scipy import stats
import logging
import time

class FeatureCategory(str, Enum):
    """Categories of features"""
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    MICROSTRUCTURE = "microstructure"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    CUSTOM = "custom"

@dataclass
class FeatureSpec:
    """Feature specification and metadata"""
    name: str
    category: FeatureCategory
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    lookback_period: int = 1
    is_target: bool = False

@dataclass
class FeatureResult:
    """Result of feature extraction"""
    features: pd.DataFrame
    feature_specs: List[FeatureSpec]
    extraction_time: float
    quality_metrics: Dict[str, float]

class AdvancedFeatureEngine:
    """Enterprise-grade feature extraction with technical indicators and market microstructure"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature registry
        self.feature_registry: Dict[str, FeatureSpec] = {}
        self.feature_functions: Dict[str, Callable] = {}
        
        # Feature cache
        self.feature_cache: Dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self.extraction_stats = {
            'features_extracted': 0,
            'extraction_time_total': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self._initialize_feature_library()
    
    async def extract_features(self, data: pd.DataFrame, 
                             feature_specs: List[FeatureSpec]) -> FeatureResult:
        """
        Extract advanced features with technical indicators and market microstructure
        Replace TODO with comprehensive feature extraction system
        """
        
        try:
            start_time = time.time()
            self.logger.info(f"Extracting {len(feature_specs)} features from {len(data)} data points")
            
            # Validate input data
            self._validate_input_data(data)
            
            # Initialize feature DataFrame
            features_df = pd.DataFrame(index=data.index)
            
            # Extract features by category for optimal performance
            for category in FeatureCategory:
                category_specs = [spec for spec in feature_specs if spec.category == category]
                if category_specs:
                    category_features = await self._extract_category_features(data, category_specs)
                    features_df = pd.concat([features_df, category_features], axis=1)
            
            # Calculate feature quality metrics
            quality_metrics = self._calculate_feature_quality(features_df)
            
            # Clean and validate features
            features_df = self._clean_features(features_df)
            
            extraction_time = time.time() - start_time
            self.extraction_stats['features_extracted'] += len(feature_specs)
            self.extraction_stats['extraction_time_total'] += extraction_time
            
            result = FeatureResult(
                features=features_df,
                feature_specs=feature_specs,
                extraction_time=extraction_time,
                quality_metrics=quality_metrics
            )
            
            self.logger.info(f"Feature extraction completed in {extraction_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            raise FeatureExtractionError(f"Feature extraction failed: {e}")
    
    def _extract_technical_feature(self, data: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
        """Extract technical indicator features"""
        
        feature_name = spec.name.lower()
        params = spec.parameters
        
        if feature_name.startswith('sma'):
            period = params.get('period', 20)
            return data['close'].rolling(window=period).mean()
        
        elif feature_name.startswith('ema'):
            period = params.get('period', 20)
            return data['close'].ewm(span=period).mean()
        
        elif feature_name.startswith('rsi'):
            period = params.get('period', 14)
            return pd.Series(talib.RSI(data['close'].values, timeperiod=period), index=data.index)
        
        elif feature_name.startswith('macd'):
            fast = params.get('fast_period', 12)
            slow = params.get('slow_period', 26)
            signal = params.get('signal_period', 9)
            macd, signal_line, histogram = talib.MACD(
                data['close'].values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            if 'signal' in feature_name:
                return pd.Series(signal_line, index=data.index)
            elif 'histogram' in feature_name:
                return pd.Series(histogram, index=data.index)
            else:
                return pd.Series(macd, index=data.index)
        
        elif feature_name.startswith('bb'):  # Bollinger Bands
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2)
            upper, middle, lower = talib.BBANDS(
                data['close'].values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            if 'upper' in feature_name:
                return pd.Series(upper, index=data.index)
            elif 'lower' in feature_name:
                return pd.Series(lower, index=data.index)
            else:
                return pd.Series(middle, index=data.index)
        
        elif feature_name.startswith('atr'):
            period = params.get('period', 14)
            return pd.Series(talib.ATR(data['high'].values, data['low'].values, data['close'].values, timeperiod=period), index=data.index)
        
        else:
            raise ValueError(f"Unknown technical feature: {feature_name}")
    
    def _extract_microstructure_feature(self, data: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
        """Extract market microstructure features"""
        
        feature_name = spec.name.lower()
        params = spec.parameters
        
        if feature_name.startswith('vwap'):
            window = params.get('window', 20)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            return (typical_price * data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()
        
        elif feature_name.startswith('volume_ratio'):
            window = params.get('window', 20)
            volume_ma = data['volume'].rolling(window=window).mean()
            return data['volume'] / volume_ma
        
        elif feature_name.startswith('money_flow'):
            window = params.get('window', 14)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=window).sum()
            negative_mf = negative_flow.rolling(window=window).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            return mfi
        
        else:
            raise ValueError(f"Unknown microstructure feature: {feature_name}")
    
    def register_custom_feature(self, name: str, feature_func: Callable, spec: FeatureSpec) -> None:
        """Register a custom feature function"""
        
        self.feature_functions[name] = feature_func
        self.feature_registry[name] = spec
        
        self.logger.info(f"Registered custom feature: {name}")

class FeatureExtractionError(Exception):
    """Exception raised for feature extraction errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Robust handling of missing data; graceful degradation for failed feature calculations; comprehensive error logging
- **Configuration:** Configurable feature parameters; customizable quality thresholds; flexible feature selection criteria
- **Testing:** Unit tests for individual features; integration tests with ML pipelines; performance tests for large datasets
- **Dependencies:** TA-Lib for technical indicators; pandas and NumPy for data manipulation; scikit-learn for feature selection

### 4. Acceptance Criteria
- [ ] Advanced feature extraction supports comprehensive technical indicators and market microstructure features
- [ ] Technical indicators library includes moving averages, oscillators, volatility measures, and momentum indicators
- [ ] Market microstructure features capture order book dynamics, volume patterns, and price-volume relationships
- [ ] Statistical features provide rolling statistics, correlation measures, and distribution characteristics
- [ ] Feature selection engine automatically identifies important features and removes redundant ones
- [ ] Feature quality assessment validates completeness, variance, and correlation properties
- [ ] Custom feature registration allows extensible feature engineering capabilities
- [ ] Performance optimization handles large datasets efficiently with caching and parallel processing
- [ ] Feature storage system provides efficient retrieval and versioning of computed features
- [ ] TODO placeholder is completely replaced with production-ready implementation 