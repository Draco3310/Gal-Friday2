# TA-Lib Integration Implementation Design

**File**: `/gal_friday/backtesting_engine.py`
- **Lines 173-187**: Minimal placeholder for TA-Lib functionality
- **Issue**: ATR calculation returns hardcoded value of 20.0
- **Impact**: Technical indicators not properly calculated

## Overview
The current TA-Lib integration in BacktestingEngine uses a minimal placeholder that returns hardcoded ATR values. This design implements a production-ready technical analysis system with multiple calculation backends, proper error handling, performance optimization, and comprehensive indicator support.

## Architecture Design

### 1. Current Problems

```
Current TA-Lib Issues:
├── Hardcoded Values
│   ├── ATR always returns 20.0
│   ├── No real calculations
│   └── No data validation
├── Limited Functionality
│   ├── Only ATR placeholder
│   ├── No other indicators
│   └── No customization
├── No Error Handling
│   ├── No input validation
│   ├── No graceful degradation
│   └── No fallback options
└── Performance Issues
    ├── No caching
    ├── No optimization
    └── No vectorization
```

### 2. Production Implementation Strategy

```
Enterprise TA-Lib System:
├── Multi-Backend Support
│   ├── TA-Lib native library
│   ├── Pandas-TA implementation
│   ├── Custom NumPy calculations
│   └── Fallback implementations
├── Comprehensive Indicators
│   ├── Trend indicators (SMA, EMA, MACD)
│   ├── Volatility indicators (ATR, Bollinger Bands)
│   ├── Momentum indicators (RSI, Stochastic)
│   └── Volume indicators (OBV, VWAP)
├── Performance Optimization
│   ├── Vectorized calculations
│   ├── Result caching
│   ├── Lazy evaluation
│   └── Memory management
└── Quality Assurance
    ├── Input validation
    ├── Error handling
    ├── Result verification
    └── Performance monitoring
```

## Implementation Plan

### Phase 1: Enhanced Technical Analysis Framework

```python
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import time
from functools import lru_cache, wraps
import hashlib

# Optional imports with graceful fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    talib = None

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None


class IndicatorType(Enum):
    """Enumeration of available technical indicators."""
    ATR = "atr"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bbands"
    SMA = "sma"
    EMA = "ema"
    STOCH = "stoch"
    OBV = "obv"
    VWAP = "vwap"
    WILLIAMS_R = "williams_r"
    CCI = "cci"
    MFI = "mfi"


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    indicator_type: IndicatorType
    parameters: Dict[str, Any] = field(default_factory=dict)
    cache_enabled: bool = True
    validation_enabled: bool = True
    fallback_value: Optional[float] = None


@dataclass
class IndicatorResult:
    """Result container for technical indicator calculations."""
    indicator_type: IndicatorType
    values: Union[np.ndarray, Tuple[np.ndarray, ...]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    calculation_time_ms: float = 0.0
    cache_hit: bool = False
    backend_used: str = "unknown"


class TechnicalAnalysisBackend(ABC):
    """Abstract base class for technical analysis backends."""
    
    @abstractmethod
    def supports_indicator(self, indicator_type: IndicatorType) -> bool:
        """Check if backend supports the given indicator."""
        pass
    
    @abstractmethod
    def calculate(
        self,
        indicator_config: IndicatorConfig,
        data: Dict[str, np.ndarray]
    ) -> IndicatorResult:
        """Calculate technical indicator."""
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Get backend identifier."""
        pass


class TALibBackend(TechnicalAnalysisBackend):
    """Production TA-Lib backend implementation."""
    
    def __init__(self):
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib not installed. Install with: conda install -c conda-forge ta-lib")
        self.logger = logging.getLogger(__name__)
        
        # Supported indicators mapping
        self._indicator_map = {
            IndicatorType.ATR: self._calculate_atr,
            IndicatorType.RSI: self._calculate_rsi,
            IndicatorType.MACD: self._calculate_macd,
            IndicatorType.BBANDS: self._calculate_bbands,
            IndicatorType.SMA: self._calculate_sma,
            IndicatorType.EMA: self._calculate_ema,
            IndicatorType.STOCH: self._calculate_stoch,
            IndicatorType.WILLIAMS_R: self._calculate_williams_r,
            IndicatorType.CCI: self._calculate_cci,
            IndicatorType.MFI: self._calculate_mfi,
        }
    
    def supports_indicator(self, indicator_type: IndicatorType) -> bool:
        """Check if TA-Lib supports the indicator."""
        return indicator_type in self._indicator_map
    
    def calculate(
        self,
        indicator_config: IndicatorConfig,
        data: Dict[str, np.ndarray]
    ) -> IndicatorResult:
        """Calculate indicator using TA-Lib."""
        start_time = time.perf_counter()
        
        try:
            calculator = self._indicator_map[indicator_config.indicator_type]
            values = calculator(data, indicator_config.parameters)
            
            calculation_time = (time.perf_counter() - start_time) * 1000
            
            return IndicatorResult(
                indicator_type=indicator_config.indicator_type,
                values=values,
                metadata={
                    "data_length": len(data.get("close", [])),
                    "parameters": indicator_config.parameters
                },
                calculation_time_ms=calculation_time,
                backend_used="talib"
            )
            
        except Exception as e:
            self.logger.error(f"TA-Lib calculation failed for {indicator_config.indicator_type}: {e}")
            raise
    
    def _calculate_atr(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate ATR using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        return talib.ATR(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            timeperiod=timeperiod
        )
    
    def _calculate_rsi(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate RSI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        return talib.RSI(data["close"].astype(np.float64), timeperiod=timeperiod)
    
    def _calculate_macd(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using TA-Lib."""
        fastperiod = params.get("fastperiod", 12)
        slowperiod = params.get("slowperiod", 26)
        signalperiod = params.get("signalperiod", 9)
        
        return talib.MACD(
            data["close"].astype(np.float64),
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
    
    def _calculate_bbands(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using TA-Lib."""
        timeperiod = params.get("timeperiod", 20)
        nbdevup = params.get("nbdevup", 2.0)
        nbdevdn = params.get("nbdevdn", 2.0)
        matype = params.get("matype", 0)
        
        return talib.BBANDS(
            data["close"].astype(np.float64),
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype
        )
    
    def _calculate_sma(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate SMA using TA-Lib."""
        timeperiod = params.get("timeperiod", 20)
        return talib.SMA(data["close"].astype(np.float64), timeperiod=timeperiod)
    
    def _calculate_ema(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate EMA using TA-Lib."""
        timeperiod = params.get("timeperiod", 20)
        return talib.EMA(data["close"].astype(np.float64), timeperiod=timeperiod)
    
    def _calculate_stoch(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic using TA-Lib."""
        fastk_period = params.get("fastk_period", 14)
        slowk_period = params.get("slowk_period", 3)
        slowd_period = params.get("slowd_period", 3)
        
        return talib.STOCH(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period
        )
    
    def _calculate_williams_r(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate Williams %R using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        return talib.WILLR(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            timeperiod=timeperiod
        )
    
    def _calculate_cci(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate CCI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        return talib.CCI(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            timeperiod=timeperiod
        )
    
    def _calculate_mfi(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate MFI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        return talib.MFI(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            data["volume"].astype(np.float64),
            timeperiod=timeperiod
        )
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "talib"


class PandasTABackend(TechnicalAnalysisBackend):
    """Pandas-TA backend implementation."""
    
    def __init__(self):
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta not installed. Install with: pip install pandas-ta")
        self.logger = logging.getLogger(__name__)
        
        self._indicator_map = {
            IndicatorType.ATR: self._calculate_atr,
            IndicatorType.RSI: self._calculate_rsi,
            IndicatorType.MACD: self._calculate_macd,
            IndicatorType.BBANDS: self._calculate_bbands,
            IndicatorType.SMA: self._calculate_sma,
            IndicatorType.EMA: self._calculate_ema,
            IndicatorType.VWAP: self._calculate_vwap,
        }
    
    def supports_indicator(self, indicator_type: IndicatorType) -> bool:
        """Check if pandas-ta supports the indicator."""
        return indicator_type in self._indicator_map
    
    def calculate(
        self,
        indicator_config: IndicatorConfig,
        data: Dict[str, np.ndarray]
    ) -> IndicatorResult:
        """Calculate indicator using pandas-ta."""
        start_time = time.perf_counter()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            calculator = self._indicator_map[indicator_config.indicator_type]
            values = calculator(df, indicator_config.parameters)
            
            calculation_time = (time.perf_counter() - start_time) * 1000
            
            return IndicatorResult(
                indicator_type=indicator_config.indicator_type,
                values=values,
                metadata={
                    "data_length": len(df),
                    "parameters": indicator_config.parameters
                },
                calculation_time_ms=calculation_time,
                backend_used="pandas_ta"
            )
            
        except Exception as e:
            self.logger.error(f"Pandas-TA calculation failed for {indicator_config.indicator_type}: {e}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Calculate ATR using pandas-ta."""
        length = params.get("timeperiod", 14)
        result = ta.atr(df["high"], df["low"], df["close"], length=length)
        return result.values
    
    def _calculate_rsi(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Calculate RSI using pandas-ta."""
        length = params.get("timeperiod", 14)
        result = ta.rsi(df["close"], length=length)
        return result.values
    
    def _calculate_macd(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using pandas-ta."""
        fast = params.get("fastperiod", 12)
        slow = params.get("slowperiod", 26)
        signal = params.get("signalperiod", 9)
        
        result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        return result[f"MACD_{fast}_{slow}_{signal}"].values, result[f"MACDs_{fast}_{slow}_{signal}"].values, result[f"MACDh_{fast}_{slow}_{signal}"].values
    
    def _calculate_bbands(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using pandas-ta."""
        length = params.get("timeperiod", 20)
        std = params.get("nbdevup", 2.0)
        
        result = ta.bbands(df["close"], length=length, std=std)
        return result[f"BBU_{length}_{std}"].values, result[f"BBM_{length}_{std}"].values, result[f"BBL_{length}_{std}"].values
    
    def _calculate_sma(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Calculate SMA using pandas-ta."""
        length = params.get("timeperiod", 20)
        result = ta.sma(df["close"], length=length)
        return result.values
    
    def _calculate_ema(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Calculate EMA using pandas-ta."""
        length = params.get("timeperiod", 20)
        result = ta.ema(df["close"], length=length)
        return result.values
    
    def _calculate_vwap(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Calculate VWAP using pandas-ta."""
        result = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        return result.values
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "pandas_ta"


class CustomBackend(TechnicalAnalysisBackend):
    """Custom implementation backend with NumPy calculations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._indicator_map = {
            IndicatorType.ATR: self._calculate_atr,
            IndicatorType.RSI: self._calculate_rsi,
            IndicatorType.SMA: self._calculate_sma,
            IndicatorType.EMA: self._calculate_ema,
            IndicatorType.VWAP: self._calculate_vwap,
        }
    
    def supports_indicator(self, indicator_type: IndicatorType) -> bool:
        """Check if custom backend supports the indicator."""
        return indicator_type in self._indicator_map
    
    def calculate(
        self,
        indicator_config: IndicatorConfig,
        data: Dict[str, np.ndarray]
    ) -> IndicatorResult:
        """Calculate indicator using custom NumPy implementation."""
        start_time = time.perf_counter()
        
        try:
            calculator = self._indicator_map[indicator_config.indicator_type]
            values = calculator(data, indicator_config.parameters)
            
            calculation_time = (time.perf_counter() - start_time) * 1000
            
            return IndicatorResult(
                indicator_type=indicator_config.indicator_type,
                values=values,
                metadata={
                    "data_length": len(data.get("close", [])),
                    "parameters": indicator_config.parameters
                },
                calculation_time_ms=calculation_time,
                backend_used="custom"
            )
            
        except Exception as e:
            self.logger.error(f"Custom calculation failed for {indicator_config.indicator_type}: {e}")
            raise
    
    def _calculate_atr(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate ATR using custom NumPy implementation."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        timeperiod = params.get("timeperiod", 14)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Calculate ATR using SMA
        atr = np.full_like(tr, np.nan)
        atr[timeperiod-1:] = np.convolve(tr, np.ones(timeperiod)/timeperiod, mode='valid')
        
        return atr
    
    def _calculate_rsi(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate RSI using custom NumPy implementation."""
        close = data["close"]
        timeperiod = params.get("timeperiod", 14)
        
        # Calculate price changes
        delta = np.diff(close)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses
        avg_gain = np.convolve(gains, np.ones(timeperiod)/timeperiod, mode='valid')
        avg_loss = np.convolve(losses, np.ones(timeperiod)/timeperiod, mode='valid')
        
        # Calculate RSI
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with NaN for proper alignment
        result = np.full_like(close, np.nan)
        result[timeperiod:] = rsi
        
        return result
    
    def _calculate_sma(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate SMA using custom NumPy implementation."""
        close = data["close"]
        timeperiod = params.get("timeperiod", 20)
        
        sma = np.full_like(close, np.nan)
        sma[timeperiod-1:] = np.convolve(close, np.ones(timeperiod)/timeperiod, mode='valid')
        
        return sma
    
    def _calculate_ema(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate EMA using custom NumPy implementation."""
        close = data["close"]
        timeperiod = params.get("timeperiod", 20)
        
        alpha = 2.0 / (timeperiod + 1)
        ema = np.full_like(close, np.nan)
        ema[0] = close[0]
        
        for i in range(1, len(close)):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_vwap(self, data: Dict[str, np.ndarray], params: Dict[str, Any]) -> np.ndarray:
        """Calculate VWAP using custom NumPy implementation."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]
        
        # Typical price
        typical_price = (high + low + close) / 3
        
        # VWAP calculation
        cumulative_volume = np.cumsum(volume)
        cumulative_price_volume = np.cumsum(typical_price * volume)
        
        vwap = cumulative_price_volume / cumulative_volume
        
        return vwap
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "custom"
```

### Phase 2: Production Technical Analysis Manager

```python
class TechnicalAnalysisManager:
    """Production-ready technical analysis manager with caching and fallbacks."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize backends in order of preference
        self.backends: List[TechnicalAnalysisBackend] = []
        self._initialize_backends()
        
        # Caching configuration
        self.cache_enabled = config.get("technical_analysis.cache_enabled", True)
        self.cache_size = config.get("technical_analysis.cache_size", 1000)
        self.cache: Dict[str, IndicatorResult] = {}
        
        # Performance tracking
        self.calculation_stats = {
            "total_calculations": 0,
            "cache_hits": 0,
            "backend_usage": {},
            "average_calculation_time": 0.0
        }
        
        self.logger.info(f"TechnicalAnalysisManager initialized with {len(self.backends)} backends")
    
    def _initialize_backends(self) -> None:
        """Initialize available backends in order of preference."""
        # Try TA-Lib first (most comprehensive)
        try:
            backend = TALibBackend()
            self.backends.append(backend)
            self.logger.info("TA-Lib backend initialized")
        except ImportError:
            self.logger.warning("TA-Lib backend not available")
        
        # Try pandas-ta second
        try:
            backend = PandasTABackend()
            self.backends.append(backend)
            self.logger.info("Pandas-TA backend initialized")
        except ImportError:
            self.logger.warning("Pandas-TA backend not available")
        
        # Always have custom backend as fallback
        backend = CustomBackend()
        self.backends.append(backend)
        self.logger.info("Custom backend initialized")
        
        if not self.backends:
            raise RuntimeError("No technical analysis backends available")
    
    def calculate_indicator(
        self,
        indicator_config: IndicatorConfig,
        data: Dict[str, np.ndarray]
    ) -> IndicatorResult:
        """Calculate technical indicator with caching and fallbacks."""
        
        # Validate input data
        self._validate_input_data(data, indicator_config)
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._generate_cache_key(indicator_config, data)
            if cache_key in self.cache:
                result = self.cache[cache_key]
                result.cache_hit = True
                self.calculation_stats["cache_hits"] += 1
                return result
        
        # Try backends in order of preference
        last_error = None
        for backend in self.backends:
            if not backend.supports_indicator(indicator_config.indicator_type):
                continue
                
            try:
                result = backend.calculate(indicator_config, data)
                
                # Cache the result
                if self.cache_enabled:
                    self._cache_result(cache_key, result)
                
                # Update statistics
                self._update_statistics(backend.get_backend_name(), result.calculation_time_ms)
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Backend {backend.get_backend_name()} failed for {indicator_config.indicator_type}: {e}"
                )
                continue
        
        # All backends failed
        if indicator_config.fallback_value is not None:
            self.logger.warning(f"Using fallback value for {indicator_config.indicator_type}")
            return self._create_fallback_result(indicator_config, data)
        
        raise RuntimeError(f"All backends failed for {indicator_config.indicator_type}: {last_error}")
    
    def _validate_input_data(self, data: Dict[str, np.ndarray], config: IndicatorConfig) -> None:
        """Validate input data for indicator calculation."""
        required_fields = self._get_required_fields(config.indicator_type)
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field '{field}' for {config.indicator_type}")
            
            if not isinstance(data[field], np.ndarray):
                raise TypeError(f"Field '{field}' must be numpy array")
            
            if len(data[field]) == 0:
                raise ValueError(f"Field '{field}' cannot be empty")
            
            # Check for NaN or infinite values
            if not np.isfinite(data[field]).all():
                raise ValueError(f"Field '{field}' contains NaN or infinite values")
    
    def _get_required_fields(self, indicator_type: IndicatorType) -> List[str]:
        """Get required data fields for indicator type."""
        field_map = {
            IndicatorType.ATR: ["high", "low", "close"],
            IndicatorType.RSI: ["close"],
            IndicatorType.MACD: ["close"],
            IndicatorType.BBANDS: ["close"],
            IndicatorType.SMA: ["close"],
            IndicatorType.EMA: ["close"],
            IndicatorType.STOCH: ["high", "low", "close"],
            IndicatorType.VWAP: ["high", "low", "close", "volume"],
            IndicatorType.MFI: ["high", "low", "close", "volume"],
        }
        return field_map.get(indicator_type, ["close"])
    
    def _generate_cache_key(self, config: IndicatorConfig, data: Dict[str, np.ndarray]) -> str:
        """Generate cache key for indicator calculation."""
        # Create hash from data and parameters
        data_str = ""
        for key in sorted(data.keys()):
            data_str += f"{key}:{hash(data[key].tobytes())}"
        
        params_str = str(sorted(config.parameters.items()))
        key_str = f"{config.indicator_type.value}:{data_str}:{params_str}"
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_result(self, cache_key: str, result: IndicatorResult) -> None:
        """Cache calculation result with LRU eviction."""
        if len(self.cache) >= self.cache_size:
            # Simple LRU: remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
    
    def _create_fallback_result(self, config: IndicatorConfig, data: Dict[str, np.ndarray]) -> IndicatorResult:
        """Create fallback result when all backends fail."""
        data_length = len(data.get("close", []))
        fallback_array = np.full(data_length, config.fallback_value)
        
        return IndicatorResult(
            indicator_type=config.indicator_type,
            values=fallback_array,
            metadata={"fallback": True, "data_length": data_length},
            calculation_time_ms=0.0,
            backend_used="fallback"
        )
    
    def _update_statistics(self, backend_name: str, calculation_time: float) -> None:
        """Update calculation statistics."""
        self.calculation_stats["total_calculations"] += 1
        
        if backend_name not in self.calculation_stats["backend_usage"]:
            self.calculation_stats["backend_usage"][backend_name] = 0
        self.calculation_stats["backend_usage"][backend_name] += 1
        
        # Update average calculation time
        total = self.calculation_stats["total_calculations"]
        current_avg = self.calculation_stats["average_calculation_time"]
        self.calculation_stats["average_calculation_time"] = (
            (current_avg * (total - 1) + calculation_time) / total
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the technical analysis manager."""
        cache_hit_rate = 0.0
        if self.calculation_stats["total_calculations"] > 0:
            cache_hit_rate = self.calculation_stats["cache_hits"] / self.calculation_stats["total_calculations"]
        
        return {
            "available_backends": [backend.get_backend_name() for backend in self.backends],
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
            "cache_hit_rate": cache_hit_rate,
            "calculation_stats": self.calculation_stats.copy(),
            "supported_indicators": [indicator.value for indicator in IndicatorType]
        }


# Enhanced BacktestingEngine integration
class EnhancedBacktestingEngineTA:
    """Enhanced backtesting engine with production-ready TA integration."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize technical analysis manager
        self.ta_manager = TechnicalAnalysisManager(config, logger)
        
        # Pre-configured indicators for common use cases
        self.default_indicators = {
            "atr_14": IndicatorConfig(
                indicator_type=IndicatorType.ATR,
                parameters={"timeperiod": 14},
                fallback_value=20.0
            ),
            "rsi_14": IndicatorConfig(
                indicator_type=IndicatorType.RSI,
                parameters={"timeperiod": 14},
                fallback_value=50.0
            ),
            "sma_20": IndicatorConfig(
                indicator_type=IndicatorType.SMA,
                parameters={"timeperiod": 20}
            ),
            "ema_12": IndicatorConfig(
                indicator_type=IndicatorType.EMA,
                parameters={"timeperiod": 12}
            ),
            "bbands_20": IndicatorConfig(
                indicator_type=IndicatorType.BBANDS,
                parameters={"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0}
            )
        }
    
    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        length: int = 14
    ) -> pd.Series:
        """Calculate ATR with production-ready implementation."""
        
        data = {
            "high": high.values,
            "low": low.values,
            "close": close.values
        }
        
        config = IndicatorConfig(
            indicator_type=IndicatorType.ATR,
            parameters={"timeperiod": length},
            fallback_value=20.0
        )
        
        try:
            result = self.ta_manager.calculate_indicator(config, data)
            return pd.Series(result.values, index=high.index)
            
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {e}")
            # Return fallback values
            return pd.Series([20.0] * len(high), index=high.index)
    
    def calculate_multiple_indicators(
        self,
        data: Dict[str, pd.Series],
        indicator_names: List[str]
    ) -> Dict[str, pd.Series]:
        """Calculate multiple indicators efficiently."""
        
        results = {}
        
        # Convert pandas Series to numpy arrays
        np_data = {key: series.values for key, series in data.items()}
        index = data["close"].index
        
        for indicator_name in indicator_names:
            if indicator_name not in self.default_indicators:
                self.logger.warning(f"Unknown indicator: {indicator_name}")
                continue
            
            try:
                config = self.default_indicators[indicator_name]
                result = self.ta_manager.calculate_indicator(config, np_data)
                
                if isinstance(result.values, tuple):
                    # Multiple outputs (e.g., MACD, Bollinger Bands)
                    for i, values in enumerate(result.values):
                        key = f"{indicator_name}_{i}"
                        results[key] = pd.Series(values, index=index)
                else:
                    # Single output
                    results[indicator_name] = pd.Series(result.values, index=index)
                    
            except Exception as e:
                self.logger.error(f"Failed to calculate {indicator_name}: {e}")
                continue
        
        return results
    
    def get_ta_diagnostics(self) -> Dict[str, Any]:
        """Get technical analysis diagnostics."""
        return self.ta_manager.get_diagnostics()
```

## Testing Strategy

1. **Unit Tests**
   - Test each backend independently
   - Validate indicator calculations against known values
   - Test error handling and fallbacks
   - Verify caching functionality

2. **Integration Tests**
   - Test with real market data
   - Verify backend fallback mechanisms
   - Test performance with large datasets
   - Validate numerical accuracy

3. **Performance Tests**
   - Benchmark calculation speeds
   - Test memory usage patterns
   - Validate caching effectiveness
   - Test concurrent calculations

4. **Accuracy Tests**
   - Compare results with established libraries
   - Test edge cases and boundary conditions
   - Verify mathematical correctness
   - Test with synthetic data

## Monitoring & Observability

1. **Performance Metrics**
   - Calculation latency (p50, p95, p99)
   - Cache hit rates
   - Backend usage distribution
   - Memory consumption

2. **Quality Metrics**
   - Calculation accuracy scores
   - Error rates by indicator
   - Fallback usage frequency
   - Data validation failures

3. **Operational Metrics**
   - Backend availability
   - System resource usage
   - Calculation throughput
   - Error recovery success rates

## Security Considerations

1. **Input Validation**
   - Sanitize all numerical inputs
   - Validate array dimensions
   - Check for malicious data patterns
   - Implement rate limiting

2. **Resource Protection**
   - Limit calculation complexity
   - Prevent memory exhaustion
   - Timeout long calculations
   - Monitor CPU usage

## Future Enhancements

1. **Advanced Features**
   - GPU acceleration for calculations
   - Distributed calculation support
   - Real-time streaming indicators
   - Custom indicator development framework

2. **Additional Indicators**
   - Ichimoku Cloud
   - Fibonacci retracements
   - Elliott Wave patterns
   - Custom proprietary indicators

3. **Performance Optimization**
   - Parallel calculation engine
   - Advanced caching strategies
   - Memory-mapped data structures
   - JIT compilation support