# Task: Implement interpolation of missing data rows according to selected method.

### 1. Context
- **File:** `gal_friday/data_ingestion/gap_detector.py`
- **Line:** `216`
- **Keyword/Pattern:** `pass`
- **Current State:** The code contains a pass statement where interpolation logic should be implemented for missing data rows.

### 2. Problem Statement
Without proper interpolation of missing data, the system cannot provide continuous data feeds for trading algorithms, which can lead to incorrect technical analysis calculations, failed predictions, and potential trading errors. Missing data gaps create inconsistencies in historical analysis and real-time decision making, potentially causing the system to miss trading opportunities or make decisions based on incomplete information.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Design Interpolation Framework:** Create flexible interpolation system supporting multiple methods
2. **Implement Core Interpolation Methods:** Linear, cubic spline, forward-fill, backward-fill, and time-weighted
3. **Add Data Quality Assessment:** Evaluate gap characteristics before choosing interpolation method
4. **Create Validation System:** Ensure interpolated data maintains statistical properties
5. **Build Configuration Management:** Allow method selection per symbol and data type
6. **Add Monitoring and Alerting:** Track interpolation frequency and quality metrics

#### b. Pseudocode or Implementation Sketch
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

class InterpolationMethod(str, Enum):
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    TIME_WEIGHTED = "time_weighted"
    POLYNOMIAL = "polynomial"
    SEASONAL = "seasonal"
    NONE = "none"  # Skip interpolation

@dataclass
class GapInfo:
    """Information about a detected data gap"""
    start_time: datetime
    end_time: datetime
    duration: timedelta
    symbol: str
    data_type: str  # 'ohlcv', 'trades', 'orderbook'
    gap_size: int  # number of missing data points
    preceding_data_quality: float  # quality score of data before gap
    following_data_quality: float  # quality score of data after gap

@dataclass
class InterpolationConfig:
    """Configuration for interpolation methods"""
    default_method: InterpolationMethod = InterpolationMethod.LINEAR
    max_gap_duration: timedelta = timedelta(minutes=30)
    min_surrounding_data_points: int = 5
    quality_threshold: float = 0.8
    method_overrides: Dict[str, InterpolationMethod] = None  # per symbol/type
    validation_enabled: bool = True
    
class DataInterpolator:
    """Production-grade data interpolation system"""
    
    def __init__(self, config: InterpolationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.interpolation_stats = {
            'total_gaps_processed': 0,
            'successful_interpolations': 0,
            'failed_interpolations': 0,
            'methods_used': {},
            'average_gap_duration': timedelta(0)
        }
        
    def interpolate_missing_data(self, gap_info: GapInfo, 
                               preceding_data: pd.DataFrame,
                               following_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Replace: pass
        with comprehensive interpolation logic
        """
        
        try:
            self.logger.info(
                f"Interpolating {gap_info.gap_size} missing data points for "
                f"{gap_info.symbol} ({gap_info.data_type}) gap from "
                f"{gap_info.start_time} to {gap_info.end_time}"
            )
            
            # Validate interpolation feasibility
            if not self._validate_interpolation_feasibility(gap_info, preceding_data, following_data):
                self.logger.warning(f"Gap interpolation not feasible for {gap_info.symbol}")
                return None
            
            # Select interpolation method
            method = self._select_interpolation_method(gap_info, preceding_data, following_data)
            
            # Perform interpolation
            interpolated_data = self._perform_interpolation(
                method, gap_info, preceding_data, following_data
            )
            
            # Validate interpolated results
            if self.config.validation_enabled:
                if not self._validate_interpolated_data(interpolated_data, preceding_data, following_data):
                    self.logger.error(f"Interpolated data failed validation for {gap_info.symbol}")
                    return None
            
            # Update statistics
            self._update_interpolation_stats(gap_info, method, success=True)
            
            # Add metadata to interpolated data
            interpolated_data = self._add_interpolation_metadata(interpolated_data, method, gap_info)
            
            self.logger.info(
                f"Successfully interpolated {len(interpolated_data)} data points "
                f"using {method.value} method for {gap_info.symbol}"
            )
            
            return interpolated_data
            
        except Exception as e:
            self.logger.error(f"Error interpolating data for {gap_info.symbol}: {e}", exc_info=True)
            self._update_interpolation_stats(gap_info, None, success=False)
            return None
    
    def _validate_interpolation_feasibility(self, gap_info: GapInfo,
                                          preceding_data: pd.DataFrame,
                                          following_data: pd.DataFrame) -> bool:
        """Validate that interpolation is feasible and advisable"""
        
        # Check gap duration limits
        if gap_info.duration > self.config.max_gap_duration:
            self.logger.warning(
                f"Gap duration {gap_info.duration} exceeds maximum "
                f"{self.config.max_gap_duration} for {gap_info.symbol}"
            )
            return False
        
        # Check surrounding data availability
        if (len(preceding_data) < self.config.min_surrounding_data_points or
            len(following_data) < self.config.min_surrounding_data_points):
            self.logger.warning(
                f"Insufficient surrounding data for interpolation: "
                f"preceding={len(preceding_data)}, following={len(following_data)}"
            )
            return False
        
        # Check data quality
        if (gap_info.preceding_data_quality < self.config.quality_threshold or
            gap_info.following_data_quality < self.config.quality_threshold):
            self.logger.warning(
                f"Surrounding data quality too low for reliable interpolation: "
                f"preceding={gap_info.preceding_data_quality}, "
                f"following={gap_info.following_data_quality}"
            )
            return False
        
        return True
    
    def _select_interpolation_method(self, gap_info: GapInfo,
                                   preceding_data: pd.DataFrame,
                                   following_data: pd.DataFrame) -> InterpolationMethod:
        """Select optimal interpolation method based on gap characteristics"""
        
        # Check for method overrides
        override_key = f"{gap_info.symbol}_{gap_info.data_type}"
        if (self.config.method_overrides and 
            override_key in self.config.method_overrides):
            return self.config.method_overrides[override_key]
        
        # Select method based on gap characteristics
        if gap_info.duration <= timedelta(minutes=5):
            # Short gaps - use linear interpolation
            return InterpolationMethod.LINEAR
        
        elif gap_info.duration <= timedelta(minutes=15):
            # Medium gaps - use cubic spline for smoothness
            return InterpolationMethod.CUBIC_SPLINE
        
        elif gap_info.data_type == 'ohlcv':
            # OHLCV data - use time-weighted method
            return InterpolationMethod.TIME_WEIGHTED
        
        else:
            # Default to forward fill for other cases
            return InterpolationMethod.FORWARD_FILL
    
    def _perform_interpolation(self, method: InterpolationMethod,
                             gap_info: GapInfo,
                             preceding_data: pd.DataFrame,
                             following_data: pd.DataFrame) -> pd.DataFrame:
        """Execute the selected interpolation method"""
        
        # Create time index for missing data points
        time_index = self._create_interpolation_time_index(gap_info)
        
        # Combine surrounding data for interpolation
        combined_data = pd.concat([preceding_data, following_data]).sort_index()
        
        if method == InterpolationMethod.LINEAR:
            return self._linear_interpolation(combined_data, time_index)
        
        elif method == InterpolationMethod.CUBIC_SPLINE:
            return self._cubic_spline_interpolation(combined_data, time_index)
        
        elif method == InterpolationMethod.FORWARD_FILL:
            return self._forward_fill_interpolation(preceding_data, time_index)
        
        elif method == InterpolationMethod.BACKWARD_FILL:
            return self._backward_fill_interpolation(following_data, time_index)
        
        elif method == InterpolationMethod.TIME_WEIGHTED:
            return self._time_weighted_interpolation(combined_data, time_index, gap_info)
        
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
    
    def _linear_interpolation(self, data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform linear interpolation"""
        
        # Reindex data to include missing time points
        extended_data = data.reindex(data.index.union(time_index))
        
        # Interpolate missing values
        interpolated = extended_data.interpolate(method='linear')
        
        # Return only the interpolated points
        return interpolated.loc[time_index]
    
    def _cubic_spline_interpolation(self, data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform cubic spline interpolation for smoother results"""
        
        # Convert timestamps to numeric for spline fitting
        data_numeric = data.copy()
        data_numeric.index = data_numeric.index.astype(np.int64) // 10**9  # Convert to seconds
        
        # Create extended time index
        extended_index = np.union1d(data_numeric.index.values, 
                                   time_index.astype(np.int64) // 10**9)
        
        interpolated_data = []
        
        for column in data.columns:
            # Fit cubic spline
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(data_numeric.index.values, data_numeric[column].values)
            
            # Interpolate at new time points
            interpolated_values = cs(time_index.astype(np.int64) // 10**9)
            interpolated_data.append(interpolated_values)
        
        # Create result DataFrame
        result = pd.DataFrame(
            np.column_stack(interpolated_data),
            index=time_index,
            columns=data.columns
        )
        
        return result
    
    def _time_weighted_interpolation(self, data: pd.DataFrame, 
                                   time_index: pd.DatetimeIndex,
                                   gap_info: GapInfo) -> pd.DataFrame:
        """Perform time-weighted interpolation considering temporal distance"""
        
        # Get boundary values
        last_before = data[data.index < gap_info.start_time].iloc[-1]
        first_after = data[data.index > gap_info.end_time].iloc[0]
        
        # Calculate time weights
        total_duration = (gap_info.end_time - gap_info.start_time).total_seconds()
        
        interpolated_data = []
        for timestamp in time_index:
            # Time from start of gap
            elapsed = (timestamp - gap_info.start_time).total_seconds()
            weight = elapsed / total_duration  # 0 to 1
            
            # Weighted interpolation
            interpolated_row = last_before * (1 - weight) + first_after * weight
            interpolated_data.append(interpolated_row)
        
        return pd.DataFrame(interpolated_data, index=time_index)
    
    def _validate_interpolated_data(self, interpolated_data: pd.DataFrame,
                                  preceding_data: pd.DataFrame,
                                  following_data: pd.DataFrame) -> bool:
        """Validate interpolated data for reasonableness"""
        
        try:
            # Check for NaN values
            if interpolated_data.isnull().any().any():
                self.logger.error("Interpolated data contains NaN values")
                return False
            
            # Check for reasonable value ranges
            combined_data = pd.concat([preceding_data, following_data])
            data_min = combined_data.min()
            data_max = combined_data.max()
            
            # Allow 10% buffer beyond observed range
            buffer = (data_max - data_min) * 0.1
            lower_bound = data_min - buffer
            upper_bound = data_max + buffer
            
            for column in interpolated_data.columns:
                if (interpolated_data[column] < lower_bound[column]).any():
                    self.logger.warning(f"Interpolated {column} values below reasonable range")
                    return False
                if (interpolated_data[column] > upper_bound[column]).any():
                    self.logger.warning(f"Interpolated {column} values above reasonable range")
                    return False
            
            # Check for smoothness (no sudden jumps)
            for column in interpolated_data.columns:
                if len(interpolated_data) > 1:
                    diffs = interpolated_data[column].diff().abs()
                    max_diff = diffs.max()
                    typical_diff = combined_data[column].diff().abs().quantile(0.95)
                    
                    if max_diff > typical_diff * 3:  # Allow 3x typical movement
                        self.logger.warning(f"Interpolated {column} shows unusual jumps")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating interpolated data: {e}")
            return False
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of interpolation failures; fallback to simpler methods; comprehensive validation of results
- **Configuration:** Flexible method selection per symbol and data type; configurable quality thresholds and limits; runtime parameter updates
- **Testing:** Unit tests for each interpolation method; integration tests with real gap scenarios; performance benchmarks; accuracy validation against known data
- **Dependencies:** NumPy and pandas for data manipulation; SciPy for advanced interpolation methods; statistical libraries for validation

### 4. Acceptance Criteria
- [ ] Pass statement is replaced with comprehensive interpolation implementation
- [ ] Multiple interpolation methods (linear, cubic spline, time-weighted) are implemented and tested
- [ ] Method selection algorithm chooses optimal approach based on gap characteristics
- [ ] Data validation ensures interpolated values are reasonable and smooth
- [ ] Configuration system allows method selection per symbol and data type
- [ ] Performance metrics track interpolation success rates and method usage
- [ ] Error handling prevents interpolation failures from stopping data ingestion
- [ ] Integration tests verify interpolation accuracy with real market data gaps
- [ ] Documentation explains interpolation methods and configuration options 