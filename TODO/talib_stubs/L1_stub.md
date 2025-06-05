# Task: Replace the stub implementation with actual `talib` functions or integrate a dependency injection approach so real technical indicators can be used in production.

### 1. Context
- **File:** `gal_friday/talib_stubs.py`
- **Line:** `1`
- **Keyword/Pattern:** `"Stub"`
- **Current State:** The code contains a stub implementation for the talib library that returns dummy values instead of real technical analysis calculations.

### 2. Problem Statement
The current stub implementation prevents the system from performing actual technical analysis calculations, which are critical for trading strategy evaluation and signal generation. This creates a significant gap between development/testing environments and production systems, potentially leading to incorrect trading decisions and financial losses. Without real technical indicators, the entire predictive modeling and strategy execution pipeline operates on meaningless data.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Evaluate Integration Options:** Assess whether to use the actual `talib` library, an alternative like `pandas-ta`, or create a custom technical analysis service
2. **Implement Dependency Injection Pattern:** Create an abstract interface for technical analysis functions to allow swapping implementations
3. **Add Configuration Management:** Enable switching between stub and production implementations via configuration
4. **Implement Production Functions:** Replace all stub functions with actual calculations
5. **Add Comprehensive Testing:** Create unit tests comparing stub vs real implementations and performance benchmarks

#### b. Pseudocode or Implementation Sketch
```python
from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np

class TechnicalAnalysisInterface(ABC):
    """Abstract interface for technical analysis calculations"""
    
    @abstractmethod
    def SMA(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        pass
    
    @abstractmethod
    def EMA(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        pass
    
    @abstractmethod
    def RSI(self, close: np.ndarray, timeperiod: int = 14) -> np.ndarray:
        pass

class TALibImplementation(TechnicalAnalysisInterface):
    """Production implementation using talib"""
    
    def __init__(self):
        try:
            import talib
            self.talib = talib
        except ImportError:
            raise ImportError("talib library not installed. Run: pip install TA-Lib")
    
    def SMA(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        return self.talib.SMA(close, timeperiod=timeperiod)

class StubImplementation(TechnicalAnalysisInterface):
    """Stub implementation for testing"""
    
    def SMA(self, close: np.ndarray, timeperiod: int = 30) -> np.ndarray:
        # Return rolling mean for more realistic stub behavior
        return np.convolve(close, np.ones(timeperiod)/timeperiod, mode='valid')

# Factory function
def create_technical_analysis_service(config: dict) -> TechnicalAnalysisInterface:
    if config.get('use_production_indicators', False):
        return TALibImplementation()
    return StubImplementation()
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful fallback when talib installation fails; validation of input arrays for NaN values and proper dtypes
- **Configuration:** Add `technical_analysis.use_production_indicators` boolean flag to main configuration
- **Testing:** Comprehensive unit tests comparing stub vs production outputs; performance benchmarks; integration tests with real market data
- **Dependencies:** Conditional dependency on `TA-Lib` library; may require system-level installation on some platforms; alternative fallback to `pandas-ta` or custom implementations

### 4. Acceptance Criteria
- [ ] Abstract interface `TechnicalAnalysisInterface` is created with all required technical analysis methods
- [ ] Production implementation using `talib` library is fully functional with proper error handling
- [ ] Configuration flag allows switching between stub and production implementations
- [ ] All technical analysis functions return mathematically correct values
- [ ] Comprehensive test suite covers both implementations with real market data validation
- [ ] Documentation explains installation requirements and configuration options
- [ ] Performance benchmarks demonstrate acceptable calculation speed for production use 