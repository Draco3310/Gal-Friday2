"""Enhanced technical analysis implementation with multiple backends and production features."""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

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
    parameters: Dict[str, Any] = field(default_factory=dict[str, Any])
    cache_enabled: bool = True
    validation_enabled: bool = True
    fallback_value: Optional[float] = None


@dataclass
class IndicatorResult:
    """Result container for technical indicator calculations."""
    indicator_type: IndicatorType
    values: Union[np.ndarray[Any, Any], Tuple[np.ndarray[Any, Any], ...]]
    metadata: Dict[str, Any] = field(default_factory=dict[str, Any])
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
        data: Dict[str, np.ndarray[Any, Any]]
    ) -> IndicatorResult:
        """Calculate technical indicator."""
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Get backend identifier."""
        pass


class TALibBackend(TechnicalAnalysisBackend):
    """Production TA-Lib backend implementation."""
    
    def __init__(self) -> None:
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib not installed. Install with: conda install -c conda-forge ta-lib")
        self.logger = logging.getLogger(__name__)
        self._source_module = self.__class__.__name__
        
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
        data: Dict[str, np.ndarray[Any, Any]]
    ) -> IndicatorResult:
        """Calculate indicator using TA-Lib."""
        start_time = time.perf_counter()
        
        try:
            calculator = self._indicator_map[indicator_config.indicator_type]
            values = calculator(data, indicator_config.parameters)
            
            calculation_time = (time.perf_counter() - start_time) * 1000
            
            return IndicatorResult(
                indicator_type=indicator_config.indicator_type,
                values=np.array(values) if not isinstance(values, np.ndarray) else values,
                metadata={
                    "data_length": len(data.get("close", [])),
                    "parameters": indicator_config.parameters
                },
                calculation_time_ms=calculation_time,
                backend_used="talib"
            )
            
        except Exception as e:
            self.logger.error(
                f"TA-Lib calculation failed for {indicator_config.indicator_type}: {e}",
                extra={"source_module": self._source_module}
            )
            raise
    
    def _calculate_atr(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate ATR using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        timeperiod = params.get("timeperiod", 14)
        return cast(np.ndarray[Any, Any], talib.ATR(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            timeperiod=timeperiod
        ))
    
    def _calculate_rsi(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate RSI using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        timeperiod = params.get("timeperiod", 14)
        return cast(np.ndarray[Any, Any], talib.RSI(data["close"].astype(np.float64), timeperiod=timeperiod))
    
    def _calculate_macd(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate MACD using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        fastperiod = params.get("fastperiod", 12)
        slowperiod = params.get("slowperiod", 26)
        signalperiod = params.get("signalperiod", 9)
        
        return cast(Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]], talib.MACD(
            data["close"].astype(np.float64),
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        ))
    
    def _calculate_bbands(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Bollinger Bands using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        timeperiod = params.get("timeperiod", 20)
        nbdevup = params.get("nbdevup", 2.0)
        nbdevdn = params.get("nbdevdn", 2.0)
        matype = params.get("matype", 0)
        
        return cast(Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]], talib.BBANDS(
            data["close"].astype(np.float64),
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype
        ))
    
    def _calculate_sma(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate SMA using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        timeperiod = params.get("timeperiod", 20)
        return cast(np.ndarray[Any, Any], talib.SMA(data["close"].astype(np.float64), timeperiod=timeperiod))
    
    def _calculate_ema(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate EMA using TA-Lib."""
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        timeperiod = params.get("timeperiod", 20)
        return cast(np.ndarray[Any, Any], talib.EMA(data["close"].astype(np.float64), timeperiod=timeperiod))
    
    def _calculate_stoch(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Stochastic using TA-Lib."""
        fastk_period = params.get("fastk_period", 14)
        slowk_period = params.get("slowk_period", 3)
        slowd_period = params.get("slowd_period", 3)
        
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        return cast(Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]], talib.STOCH(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period
        ))
    
    def _calculate_williams_r(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate Williams %R using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        return cast(np.ndarray[Any, Any], talib.WILLR(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            timeperiod=timeperiod
        ))
    
    def _calculate_cci(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate CCI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        return cast(np.ndarray[Any, Any], talib.CCI(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            timeperiod=timeperiod
        ))
    
    def _calculate_mfi(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate MFI using TA-Lib."""
        timeperiod = params.get("timeperiod", 14)
        if talib is None:
            raise RuntimeError("TA-Lib not available")
        return cast(np.ndarray[Any, Any], talib.MFI(
            data["high"].astype(np.float64),
            data["low"].astype(np.float64),
            data["close"].astype(np.float64),
            data["volume"].astype(np.float64),
            timeperiod=timeperiod
        ))
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "talib"


class PandasTABackend(TechnicalAnalysisBackend):
    """Pandas-TA backend implementation."""
    
    def __init__(self) -> None:
        if not PANDAS_TA_AVAILABLE:
            raise ImportError("pandas-ta not installed. Install with: pip install pandas-ta")
        self.logger = logging.getLogger(__name__)
        self._source_module = self.__class__.__name__
        
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
        data: Dict[str, np.ndarray[Any, Any]]
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
                values=np.array(values) if not isinstance(values, np.ndarray) else values,
                metadata={
                    "data_length": len(df),
                    "parameters": indicator_config.parameters
                },
                calculation_time_ms=calculation_time,
                backend_used="pandas_ta"
            )
            
        except Exception as e:
            self.logger.error(
                f"Pandas-TA calculation failed for {indicator_config.indicator_type}: {e}",
                extra={"source_module": self._source_module}
            )
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate ATR using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        length = params.get("timeperiod", 14)
        result = ta.atr(df["high"], df["low"], df["close"], length=length)
        return cast(np.ndarray[Any, Any], result.values)
    
    def _calculate_rsi(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate RSI using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        length = params.get("timeperiod", 14)
        result = ta.rsi(df["close"], length=length)
        return cast(np.ndarray[Any, Any], result.values)
    
    def _calculate_macd(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate MACD using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        fast = params.get("fastperiod", 12)
        slow = params.get("slowperiod", 26)
        signal = params.get("signalperiod", 9)
        
        result = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
        macd_col = f"MACD_{fast}_{slow}_{signal}"
        signal_col = f"MACDs_{fast}_{slow}_{signal}"
        hist_col = f"MACDh_{fast}_{slow}_{signal}"
        
        return (
            cast(np.ndarray[Any, Any], result[macd_col].values),
            cast(np.ndarray[Any, Any], result[signal_col].values),
            cast(np.ndarray[Any, Any], result[hist_col].values)
        )
    
    def _calculate_bbands(self, df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Calculate Bollinger Bands using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        length = params.get("timeperiod", 20)
        std = params.get("nbdevup", 2.0)
        
        result = ta.bbands(df["close"], length=length, std=std)
        upper_col = f"BBU_{length}_{std}"
        middle_col = f"BBM_{length}_{std}"
        lower_col = f"BBL_{length}_{std}"
        
        return (
            cast(np.ndarray[Any, Any], result[upper_col].values),
            cast(np.ndarray[Any, Any], result[middle_col].values),
            cast(np.ndarray[Any, Any], result[lower_col].values)
        )
    
    def _calculate_sma(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate SMA using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        length = params.get("timeperiod", 20)
        result = ta.sma(df["close"], length=length)
        return cast(np.ndarray[Any, Any], result.values)
    
    def _calculate_ema(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate EMA using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        length = params.get("timeperiod", 20)
        result = ta.ema(df["close"], length=length)
        return cast(np.ndarray[Any, Any], result.values)
    
    def _calculate_vwap(self, df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate VWAP using pandas-ta."""
        if ta is None:
            raise RuntimeError("pandas-ta not available")
        result = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        return cast(np.ndarray[Any, Any], result.values)
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "pandas_ta"


class CustomBackend(TechnicalAnalysisBackend):
    """Custom implementation backend with NumPy calculations."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._source_module = self.__class__.__name__
        
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
        data: Dict[str, np.ndarray[Any, Any]]
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
            self.logger.error(
                f"Custom calculation failed for {indicator_config.indicator_type}: {e}",
                extra={"source_module": self._source_module}
            )
            raise
    
    def _calculate_atr(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
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
        
        # Calculate ATR using EMA
        atr = np.full_like(tr, np.nan)
        atr[timeperiod-1] = np.mean(tr[:timeperiod])
        
        # EMA calculation
        alpha = 1.0 / timeperiod
        for i in range(timeperiod, len(tr)):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        return np.asarray(atr, dtype=np.float64)
    
    def _calculate_rsi(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate RSI using custom NumPy implementation."""
        close = data["close"]
        timeperiod = params.get("timeperiod", 14)
        
        # Calculate price changes
        delta = np.diff(close)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses using EMA
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial average
        avg_gain[timeperiod] = np.mean(gains[:timeperiod])
        avg_loss[timeperiod] = np.mean(losses[:timeperiod])
        
        # EMA calculation
        alpha = 1.0 / timeperiod
        for i in range(timeperiod + 1, len(close)):
            avg_gain[i] = alpha * gains[i-1] + (1 - alpha) * avg_gain[i-1]
            avg_loss[i] = alpha * losses[i-1] + (1 - alpha) * avg_loss[i-1]
        
        # Calculate RSI
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        rsi[:timeperiod] = np.nan
        
        return np.asarray(rsi, dtype=np.float64)
    
    def _calculate_sma(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate SMA using custom NumPy implementation."""
        close = data["close"]
        timeperiod = params.get("timeperiod", 20)
        
        sma = np.full_like(close, np.nan)
        sma[timeperiod-1:] = np.convolve(close, np.ones(timeperiod)/timeperiod, mode='valid')
        
        return sma
    
    def _calculate_ema(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Calculate EMA using custom NumPy implementation."""
        close = data["close"]
        timeperiod = params.get("timeperiod", 20)
        
        alpha = 2.0 / (timeperiod + 1)
        ema = np.full_like(close, np.nan)
        ema[timeperiod-1] = np.mean(close[:timeperiod])
        
        for i in range(timeperiod, len(close)):
            ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_vwap(self, data: Dict[str, np.ndarray[Any, Any]], params: Dict[str, Any]) -> np.ndarray[Any, Any]:
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
        
        vwap = cumulative_price_volume / (cumulative_volume + 1e-10)
        
        return np.asarray(vwap, dtype=np.float64)
    
    def get_backend_name(self) -> str:
        """Get backend name."""
        return "custom"


class TechnicalAnalysisManager:
    """Production-ready technical analysis manager with caching and fallbacks."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Initialize backends in order of preference
        self.backends: List[TechnicalAnalysisBackend] = []
        self._initialize_backends()
        
        # Caching configuration
        self.cache_enabled = config.get("technical_analysis", {}).get("cache_enabled", True)
        self.cache_size = config.get("technical_analysis", {}).get("cache_size", 1000)
        self.cache: Dict[str, IndicatorResult] = {}
        
        # Performance tracking
        self.calculation_stats: Dict[str, Union[int, float, Dict[str, int]]] = {
            "total_calculations": 0,
            "cache_hits": 0,
            "backend_usage": {},
            "average_calculation_time": 0.0
        }
        
        self.logger.info(
            f"TechnicalAnalysisManager initialized with {len(self.backends)} backends",
            extra={"source_module": self._source_module}
        )
    
    def _initialize_backends(self) -> None:
        """Initialize available backends in order of preference."""
        # Try TA-Lib first (most comprehensive)
        try:
            talib_backend: TechnicalAnalysisBackend = TALibBackend()
            self.backends.append(talib_backend)
            self.logger.info("TA-Lib backend initialized", extra={"source_module": self._source_module})
        except ImportError:
            self.logger.warning("TA-Lib backend not available", extra={"source_module": self._source_module})
        
        # Try pandas-ta second
        try:
            pandas_backend: TechnicalAnalysisBackend = PandasTABackend()
            self.backends.append(pandas_backend)
            self.logger.info("Pandas-TA backend initialized", extra={"source_module": self._source_module})
        except ImportError:
            self.logger.warning("Pandas-TA backend not available", extra={"source_module": self._source_module})
        
        # Always have custom backend as fallback
        custom_backend: TechnicalAnalysisBackend = CustomBackend()
        self.backends.append(custom_backend)
        self.logger.info("Custom backend initialized", extra={"source_module": self._source_module})
        
        if not self.backends:
            raise RuntimeError("No technical analysis backends available")
    
    def calculate_indicator(
        self,
        indicator_config: IndicatorConfig,
        data: Dict[str, np.ndarray[Any, Any]]
    ) -> IndicatorResult:
        """Calculate technical indicator with caching and fallbacks."""
        
        # Validate input data
        self._validate_input_data(data, indicator_config)
        
        # Check cache first
        cache_key = None
        if self.cache_enabled:
            cache_key = self._generate_cache_key(indicator_config, data)
            if cache_key in self.cache:
                result = self.cache[cache_key]
                result.cache_hit = True
                cache_hits = cast(int, self.calculation_stats["cache_hits"])
                self.calculation_stats["cache_hits"] = cache_hits + 1
                return result
        
        # Try backends in order of preference
        last_error = None
        for backend in self.backends:
            if not backend.supports_indicator(indicator_config.indicator_type):
                continue
                
            try:
                result = backend.calculate(indicator_config, data)
                
                # Cache the result
                if self.cache_enabled and cache_key:
                    self._cache_result(cache_key, result)
                
                # Update statistics
                self._update_statistics(backend.get_backend_name(), result.calculation_time_ms)
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Backend {backend.get_backend_name()} failed for {indicator_config.indicator_type}: {e}",
                    extra={"source_module": self._source_module}
                )
                continue
        
        # All backends failed
        if indicator_config.fallback_value is not None:
            self.logger.warning(
                f"Using fallback value for {indicator_config.indicator_type}",
                extra={"source_module": self._source_module}
            )
            return self._create_fallback_result(indicator_config, data)
        
        raise RuntimeError(f"All backends failed for {indicator_config.indicator_type}: {last_error}")
    
    def _validate_input_data(self, data: Dict[str, np.ndarray[Any, Any]], config: IndicatorConfig) -> None:
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
            IndicatorType.WILLIAMS_R: ["high", "low", "close"],
            IndicatorType.CCI: ["high", "low", "close"],
            IndicatorType.OBV: ["close", "volume"],
        }
        return field_map.get(indicator_type, ["close"])
    
    def _generate_cache_key(self, config: IndicatorConfig, data: Dict[str, np.ndarray[Any, Any]]) -> str:
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
    
    def _create_fallback_result(self, config: IndicatorConfig, data: Dict[str, np.ndarray[Any, Any]]) -> IndicatorResult:
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
        total_calc = cast(int, self.calculation_stats["total_calculations"])
        self.calculation_stats["total_calculations"] = total_calc + 1
        
        backend_usage = cast(Dict[str, int], self.calculation_stats["backend_usage"])
        if backend_name not in backend_usage:
            backend_usage[backend_name] = 0
        backend_usage[backend_name] += 1
        
        # Update average calculation time
        total = cast(int, self.calculation_stats["total_calculations"])
        current_avg = cast(float, self.calculation_stats["average_calculation_time"])
        self.calculation_stats["average_calculation_time"] = (
            (current_avg * (total - 1) + calculation_time) / total
        )
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the technical analysis manager."""
        cache_hit_rate = 0.0
        total_calculations = cast(int, self.calculation_stats["total_calculations"])
        if total_calculations > 0:
            cache_hits = cast(int, self.calculation_stats["cache_hits"])
            cache_hit_rate = cache_hits / total_calculations
        
        return {
            "available_backends": [backend.get_backend_name() for backend in self.backends],
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
            "cache_hit_rate": cache_hit_rate,
            "calculation_stats": self.calculation_stats.copy(),
            "supported_indicators": [indicator.value for indicator in IndicatorType]
        }