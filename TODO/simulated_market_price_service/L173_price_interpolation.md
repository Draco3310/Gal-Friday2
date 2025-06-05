# Task: Implement price interpolation and missing data handling algorithms.

### 1. Context
- **File:** `gal_friday/simulated_market_price_service.py`
- **Line:** `173`
- **Keyword/Pattern:** `TODO`
- **Current State:** The code contains a TODO placeholder for implementing price interpolation and missing data handling algorithms.

### 2. Problem Statement
Without proper price interpolation and missing data handling, the simulated market price service cannot provide continuous price data during gaps in historical datasets. This creates unrealistic trading scenarios during backtesting and prevents accurate simulation of market conditions when data is incomplete or missing.

### 3. Proposed Solution (Enterprise-Grade)

#### a. High-Level Plan
1. **Create Interpolation Engine:** Multiple interpolation algorithms for different market scenarios
2. **Build Gap Detection System:** Intelligent detection of data gaps and missing periods
3. **Implement Smart Fill Strategies:** Context-aware filling strategies based on market conditions
4. **Add Volatility Preservation:** Maintain realistic volatility characteristics during interpolation
5. **Create Quality Assessment:** Validation of interpolated data quality and impact analysis
6. **Build Configuration System:** Configurable interpolation methods per symbol and timeframe

#### b. Pseudocode or Implementation Sketch
```python
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging

class InterpolationMethod(str, Enum):
    """Available interpolation methods"""
    LINEAR = "linear"
    SPLINE = "spline"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"

@dataclass
class DataGap:
    """Information about a data gap"""
    start_time: datetime
    end_time: datetime
    gap_type: str
    duration_minutes: int
    before_price: Optional[float] = None
    after_price: Optional[float] = None

@dataclass
class InterpolationResult:
    """Result of price interpolation"""
    interpolated_data: List[dict]
    quality_score: float
    method_used: InterpolationMethod
    gaps_filled: List[DataGap]
    warnings: List[str]

class PriceInterpolator:
    """Enterprise-grade price interpolation and missing data handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Interpolation methods registry
        self.interpolation_methods = {
            InterpolationMethod.LINEAR: self._linear_interpolation,
            InterpolationMethod.SPLINE: self._spline_interpolation,
            InterpolationMethod.VOLATILITY_ADJUSTED: self._volatility_adjusted_interpolation,
            InterpolationMethod.FORWARD_FILL: self._forward_fill,
            InterpolationMethod.BACKWARD_FILL: self._backward_fill
        }
    
    async def interpolate_missing_data(self, data: List[dict], symbol: str, 
                                     frequency: str) -> InterpolationResult:
        """
        Interpolate missing data points using intelligent algorithms
        Replace TODO with comprehensive interpolation system
        """
        
        try:
            self.logger.info(f"Starting interpolation for {symbol} with {len(data)} data points")
            
            if not data:
                return InterpolationResult(
                    interpolated_data=[],
                    quality_score=0.0,
                    method_used=InterpolationMethod.LINEAR,
                    gaps_filled=[],
                    warnings=["No data provided for interpolation"]
                )
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data)
            
            # Detect gaps in the data
            gaps = await self._detect_gaps(df, symbol, frequency)
            
            if not gaps:
                self.logger.debug("No gaps detected in data")
                return InterpolationResult(
                    interpolated_data=data,
                    quality_score=1.0,
                    method_used=InterpolationMethod.LINEAR,
                    gaps_filled=[],
                    warnings=[]
                )
            
            self.logger.info(f"Detected {len(gaps)} gaps to interpolate")
            
            # Choose interpolation method
            method = self._select_interpolation_method(gaps, symbol)
            
            # Perform interpolation
            interpolated_df = await self._perform_interpolation(df, gaps, method)
            
            # Convert back to list of dictionaries
            interpolated_data = interpolated_df.to_dict('records')
            
            # Calculate quality score
            quality_score = self._calculate_interpolation_quality(df, interpolated_df, gaps)
            
            result = InterpolationResult(
                interpolated_data=interpolated_data,
                quality_score=quality_score,
                method_used=method,
                gaps_filled=gaps,
                warnings=[]
            )
            
            self.logger.info(
                f"Interpolation complete: {len(interpolated_data)} points, "
                f"quality={quality_score:.2f}, filled {len(gaps)} gaps"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during interpolation for {symbol}: {e}")
            raise InterpolationError(f"Interpolation failed: {e}")
    
    async def _detect_gaps(self, df: pd.DataFrame, symbol: str, frequency: str) -> List[DataGap]:
        """Detect gaps in price data"""
        
        gaps = []
        
        if len(df) < 2:
            return gaps
        
        # Calculate expected time interval
        interval_minutes = self._frequency_to_minutes(frequency)
        expected_interval = timedelta(minutes=interval_minutes)
        
        # Check for gaps between consecutive data points
        for i in range(1, len(df)):
            prev_time = pd.to_datetime(df.iloc[i-1]['timestamp'])
            curr_time = pd.to_datetime(df.iloc[i]['timestamp'])
            
            time_diff = curr_time - prev_time
            
            # If gap is larger than expected interval
            if time_diff > expected_interval * 1.5:  # Allow 50% tolerance
                gap = DataGap(
                    start_time=prev_time,
                    end_time=curr_time,
                    gap_type="data_gap",
                    duration_minutes=int(time_diff.total_seconds() / 60),
                    before_price=df.iloc[i-1]['close'],
                    after_price=df.iloc[i]['open']
                )
                
                gaps.append(gap)
        
        return gaps
    
    def _select_interpolation_method(self, gaps: List[DataGap], symbol: str) -> InterpolationMethod:
        """Select best interpolation method based on gap characteristics"""
        
        if not gaps:
            return InterpolationMethod.LINEAR
        
        # For small gaps, use linear interpolation
        avg_gap_duration = sum(gap.duration_minutes for gap in gaps) / len(gaps)
        
        if avg_gap_duration < 60:  # Less than 1 hour
            return InterpolationMethod.LINEAR
        elif avg_gap_duration < 240:  # Less than 4 hours
            return InterpolationMethod.VOLATILITY_ADJUSTED
        else:
            return InterpolationMethod.FORWARD_FILL
    
    async def _perform_interpolation(self, df: pd.DataFrame, gaps: List[DataGap], 
                                   method: InterpolationMethod) -> pd.DataFrame:
        """Perform actual interpolation to fill gaps"""
        
        interpolated_df = df.copy()
        
        for gap in gaps:
            try:
                # Find the indices around the gap
                before_idx = df[pd.to_datetime(df['timestamp']) <= gap.start_time].index[-1]
                after_idx = df[pd.to_datetime(df['timestamp']) >= gap.end_time].index[0]
                
                # Generate timestamps for the gap
                gap_timestamps = self._generate_gap_timestamps(gap.start_time, gap.end_time)
                
                # Interpolate prices using selected method
                interpolation_func = self.interpolation_methods[method]
                interpolated_points = await interpolation_func(df, before_idx, after_idx, gap_timestamps, gap)
                
                # Insert interpolated points
                for point in interpolated_points:
                    interpolated_df = pd.concat([interpolated_df, pd.DataFrame([point])], ignore_index=True)
                
            except Exception as e:
                self.logger.warning(f"Failed to interpolate gap {gap.start_time} to {gap.end_time}: {e}")
        
        # Sort by timestamp
        interpolated_df = interpolated_df.sort_values('timestamp').reset_index(drop=True)
        
        return interpolated_df
    
    async def _linear_interpolation(self, df: pd.DataFrame, before_idx: int, after_idx: int,
                                  timestamps: List[datetime], gap: DataGap) -> List[Dict[str, Any]]:
        """Linear interpolation between two points"""
        
        before_point = df.iloc[before_idx]
        after_point = df.iloc[after_idx]
        
        interpolated_points = []
        
        for i, timestamp in enumerate(timestamps):
            # Calculate interpolation factor
            total_duration = (gap.end_time - gap.start_time).total_seconds()
            current_duration = (timestamp - gap.start_time).total_seconds()
            factor = current_duration / total_duration if total_duration > 0 else 0
            
            # Linear interpolation for price
            interpolated_price = before_point['close'] + factor * (after_point['open'] - before_point['close'])
            
            point = {
                'timestamp': timestamp,
                'open': interpolated_price,
                'high': interpolated_price,
                'low': interpolated_price,
                'close': interpolated_price,
                'volume': self._interpolate_volume(before_point, after_point, factor),
                'interpolated': True
            }
            
            interpolated_points.append(point)
        
        return interpolated_points
    
    def _interpolate_volume(self, before_point: pd.Series, after_point: pd.Series, factor: float) -> float:
        """Interpolate volume with some randomness"""
        
        base_volume = before_point['volume'] + factor * (after_point['volume'] - before_point['volume'])
        
        # Add some randomness (±20%)
        random_factor = np.random.uniform(0.8, 1.2)
        
        return max(0, base_volume * random_factor)
    
    def _calculate_interpolation_quality(self, original_df: pd.DataFrame, 
                                       interpolated_df: pd.DataFrame, 
                                       gaps: List[DataGap]) -> float:
        """Calculate quality score for interpolation"""
        
        if not gaps:
            return 1.0
        
        # Simple quality metric based on price continuity
        interpolated_count = len(interpolated_df) - len(original_df)
        gap_coverage = interpolated_count / sum(gap.duration_minutes for gap in gaps) if gaps else 1.0
        
        return min(1.0, gap_coverage)

class InterpolationError(Exception):
    """Exception raised for interpolation errors"""
    pass
```

#### c. Key Considerations & Dependencies
- **Error Handling:** Graceful handling of interpolation failures; fallback methods when primary interpolation fails
- **Configuration:** Configurable interpolation methods per symbol; adjustable quality thresholds
- **Testing:** Unit tests for interpolation algorithms; validation against known data patterns; performance tests for large datasets
- **Dependencies:** NumPy and pandas for mathematical operations; statistical analysis libraries

### 4. Acceptance Criteria
- [ ] Multiple interpolation algorithms handle different types of data gaps appropriately
- [ ] Gap detection accurately identifies missing data periods and classifies gap types
- [ ] Volatility preservation maintains realistic price movement characteristics during interpolation
- [ ] Quality assessment provides meaningful metrics for interpolation accuracy and reliability
- [ ] Configuration system allows fine-tuning of interpolation behavior per symbol and timeframe
- [ ] Performance optimization handles large datasets efficiently without memory issues
- [ ] Validation framework ensures interpolated data maintains statistical properties of original data
- [ ] Error recovery provides fallback interpolation methods when primary method fails
- [ ] TODO placeholder is completely replaced with production-ready implementation 