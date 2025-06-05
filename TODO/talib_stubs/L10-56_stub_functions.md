# Task: Each function returns zeros; implement proper calculations using `talib` or another technical analysis library, ensuring error handling and vectorized operations.

### 1. Context
- **File:** `gal_friday/talib_stubs.py`
- **Line:** `10-56`
- **Keyword/Pattern:** `"stub functions"`
- **Current State:** All technical analysis functions (SMA, EMA, RSI, MACD, etc.) return arrays of zeros instead of performing actual calculations.

### 2. Problem Statement
The current stub functions return meaningless zero values for all technical indicators, completely undermining the validity of any trading strategy or predictive model that relies on these indicators. This creates a false sense of functionality during development while producing nonsensical results that would lead to catastrophic trading decisions in production. The lack of proper vectorized operations also means poor performance when processing large datasets.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Audit All Stub Functions:** Catalog all 40+ technical analysis functions currently stubbed
2. **Implement Core Indicators:** Start with most critical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
3. **Add Vectorized Operations:** Ensure all calculations use NumPy vectorization for performance
4. **Implement Error Handling:** Add robust input validation and NaN handling
5. **Create Comprehensive Tests:** Validate against known good implementations and edge cases
6. **Add Performance Optimization:** Implement caching and memory-efficient calculations

#### b. Pseudocode or Implementation Sketch
```python
import numpy as np
from typing import Tuple, Optional
import logging

class ProductionTechnicalAnalysis:
    """Production-grade technical analysis calculations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _validate_input(self, data: np.ndarray, min_length: int = 1) -> np.ndarray:
        """Validate and clean input data"""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        
        if len(data) < min_length:
            raise ValueError(f"Insufficient data: need at least {min_length} points")
        
        # Handle NaN values
        if np.isnan(data).any():
            self.logger.warning("NaN values detected in input data")
            # Forward fill NaN values
            mask = ~np.isnan(data)
            data[~mask] = np.interp(np.where(~mask)[0], np.where(mask)[0], data[mask])
        
        return data
    
    def SMA(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Simple Moving Average with proper validation and performance"""
        close = self._validate_input(close, timeperiod)
        
        # Use pandas rolling for efficiency, but implement pure NumPy version
        weights = np.ones(timeperiod) / timeperiod
        sma = np.convolve(close, weights, mode='valid')
        
        # Pad with NaN to match input length
        result = np.full(len(close), np.nan)
        result[timeperiod-1:] = sma
        
        return result
    
    def EMA(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        """Exponential Moving Average"""
        close = self._validate_input(close, timeperiod)
        
        alpha = 2.0 / (timeperiod + 1.0)
        result = np.empty_like(close)
        
        # Initialize first value
        result[0] = close[0]
        
        # Vectorized EMA calculation
        for i in range(1, len(close)):
            result[i] = alpha * close[i] + (1 - alpha) * result[i-1]
        
        return result
    
    def RSI(self, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        close = self._validate_input(close, timeperiod + 1)
        
        # Calculate price changes
        delta = np.diff(close)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate average gains and losses
        avg_gains = np.convolve(gains, np.ones(timeperiod)/timeperiod, mode='valid')
        avg_losses = np.convolve(losses, np.ones(timeperiod)/timeperiod, mode='valid')
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad result
        result = np.full(len(close), np.nan)
        result[timeperiod:] = rsi
        
        return result
    
    def MACD(self, close: np.ndarray, fastperiod: int = 12, 
             slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD calculation returning (macd, signal, histogram)"""
        close = self._validate_input(close, slowperiod)
        
        # Calculate EMAs
        ema_fast = self.EMA(close, fastperiod)
        ema_slow = self.EMA(close, slowperiod)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        signal_line = self.EMA(macd_line[~np.isnan(macd_line)], signalperiod)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Input validation for array types, lengths, and NaN values; graceful handling of insufficient data
- **Configuration:** Configurable default periods for all indicators; option to use different calculation methods
- **Testing:** Unit tests against known reference implementations (talib, pandas-ta); performance benchmarks; edge case testing
- **Dependencies:** NumPy for vectorized operations; optional pandas for advanced rolling operations; comparison with talib for validation

### 4. Acceptance Criteria
- [ ] All 40+ technical analysis functions implement mathematically correct calculations
- [ ] Input validation handles arrays, lists, NaN values, and insufficient data gracefully
- [ ] All calculations are vectorized using NumPy for optimal performance
- [ ] Results match reference implementations (talib) within acceptable tolerance (1e-8)
- [ ] Comprehensive test suite covers normal cases, edge cases, and performance benchmarks
- [ ] Memory usage is optimized for large datasets (1M+ data points)
- [ ] All functions include proper docstrings with parameter descriptions and return types
- [ ] Error logging provides actionable information for debugging data quality issues 