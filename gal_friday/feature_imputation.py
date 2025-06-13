"""Advanced imputation system for Gal-Friday cryptocurrency trading data.

This module provides enterprise-grade missing data imputation specifically designed
for 24/7 cryptocurrency markets, focusing on predictive accuracy and performance.
"""

import asyncio
import time
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Type
from collections import defaultdict, deque

# Scikit-learn imports
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import interpolate
from scipy.stats import pearsonr

# Gal-Friday imports
from .logger_service import LoggerService


# === Configuration and Data Models ===

class ImputationMethod(str, Enum):
    """Available imputation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATION = "linear_interpolation"
    SPLINE_INTERPOLATION = "spline_interpolation"
    KNN = "knn"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    VWAP_WEIGHTED = "vwap_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    MARKET_SESSION_AWARE = "market_session_aware"


class DataType(str, Enum):
    """Types of financial data for context-aware imputation."""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    INDICATOR = "indicator"
    RATIO = "ratio"
    SENTIMENT = "sentiment"


class ImputationQuality(str, Enum):
    """Quality levels for imputation strategies."""
    FAST = "fast"          # Simple, performant methods
    BALANCED = "balanced"  # Good accuracy/performance tradeoff
    ACCURATE = "accurate"  # Highest accuracy, more computation


@dataclass
class ImputationConfig:
    """Configuration for imputation strategies."""
    feature_key: str
    data_type: DataType
    primary_method: ImputationMethod
    fallback_method: ImputationMethod
    quality_level: ImputationQuality = ImputationQuality.BALANCED
    
    # Method-specific parameters
    max_gap_minutes: int = 15
    knn_neighbors: int = 5
    model_lookback_periods: int = 100
    confidence_threshold: float = 0.8
    
    # Performance constraints
    max_computation_time_ms: float = 100.0
    cache_results: bool = True
    
    # Validation parameters
    enable_validation: bool = True
    validation_sample_size: int = 1000
    
    # Context parameters
    consider_market_session: bool = True
    consider_volatility_regime: bool = True
    use_cross_asset_correlation: bool = False


@dataclass
class ImputationResult:
    """Result of an imputation operation."""
    imputed_values: Union[np.ndarray, pd.Series]
    method_used: ImputationMethod
    confidence_score: float
    computation_time_ms: float
    missing_count: int
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImputationMetrics:
    """Performance metrics for imputation strategies."""
    method: ImputationMethod
    feature_key: str
    accuracy_score: float  # R² or similar
    mean_absolute_error: float
    mean_squared_error: float
    computation_time_ms: float
    cache_hit_rate: float
    samples_evaluated: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


# === Abstract Base Classes ===

class ImputationStrategy(ABC):
    """Abstract base class for all imputation strategies."""
    
    def __init__(self, config: ImputationConfig, logger: LoggerService):
        self.config = config
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Performance tracking
        self._computation_times = deque(maxlen=100)
        self._accuracy_scores = deque(maxlen=50)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
    @abstractmethod
    async def impute(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series,
        context: Optional[Dict[str, Any]] = None
    ) -> ImputationResult:
        """Impute missing values in the data series.
        
        Args:
            data: Time series data with missing values (NaN)
            missing_mask: Boolean mask indicating missing values
            context: Additional context (market data, correlations, etc.)
            
        Returns:
            ImputationResult with imputed values and metadata
        """
        pass
        
    @abstractmethod
    def validate_parameters(self) -> bool:
        """Validate strategy-specific parameters."""
        pass
        
    def get_cache_key(self, data: pd.Series, missing_mask: pd.Series) -> str:
        """Generate cache key for imputation results."""
        data_hash = hash(tuple(data.dropna().tail(10)))
        mask_hash = hash(tuple(missing_mask.astype(int)))
        return f"{self.config.feature_key}_{data_hash}_{mask_hash}"
        
    def should_use_cache(self, cache_key: str) -> bool:
        """Determine if cached result should be used."""
        if not self.config.cache_results:
            return False
        return cache_key in self._cache
        
    def update_performance_metrics(
        self, 
        computation_time: float, 
        accuracy_score: float
    ) -> None:
        """Update internal performance tracking."""
        self._computation_times.append(computation_time)
        self._accuracy_scores.append(accuracy_score)
        
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of strategy performance."""
        return {
            "avg_computation_time_ms": np.mean(self._computation_times) if self._computation_times else 0,
            "avg_accuracy_score": np.mean(self._accuracy_scores) if self._accuracy_scores else 0,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "total_computations": len(self._computation_times)
        }


# === Concrete Strategy Implementations ===

class SimpleImputationStrategy(ImputationStrategy):
    """Fast, simple imputation methods optimized for real-time trading."""
    
    async def impute(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series,
        context: Optional[Dict[str, Any]] = None
    ) -> ImputationResult:
        """Perform simple imputation using statistical methods."""
        start_time = time.time()
        cache_key = self.get_cache_key(data, missing_mask)
        
        if self.should_use_cache(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]
            
        self._cache_misses += 1
        
        try:
            imputed_data = data.copy()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                return ImputationResult(
                    imputed_values=imputed_data,
                    method_used=self.config.primary_method,
                    confidence_score=1.0,
                    computation_time_ms=0,
                    missing_count=0
                )
            
            # Apply the configured simple method
            if self.config.primary_method == ImputationMethod.MEAN:
                fill_value = data.mean()
                imputed_data[missing_mask] = fill_value
                confidence = 0.6  # Moderate confidence for mean
                
            elif self.config.primary_method == ImputationMethod.MEDIAN:
                fill_value = data.median()
                imputed_data[missing_mask] = fill_value
                confidence = 0.65  # Slightly better than mean for outliers
                
            elif self.config.primary_method == ImputationMethod.FORWARD_FILL:
                imputed_data = data.fillna(method='ffill', limit=self.config.max_gap_minutes)
                # Calculate confidence based on gap size
                remaining_missing = imputed_data.isna().sum()
                confidence = max(0.8 - (remaining_missing / missing_count * 0.3), 0.3)
                
            elif self.config.primary_method == ImputationMethod.BACKWARD_FILL:
                imputed_data = data.fillna(method='bfill', limit=self.config.max_gap_minutes)
                remaining_missing = imputed_data.isna().sum()
                confidence = max(0.75 - (remaining_missing / missing_count * 0.3), 0.3)
                
            else:
                # Default to forward fill
                imputed_data = data.fillna(method='ffill')
                confidence = 0.7
                
            # Handle any remaining missing values with fallback
            if imputed_data.isna().any():
                imputed_data = self._apply_fallback(imputed_data, data)
                confidence *= 0.8  # Reduce confidence when fallback is used
                
            computation_time = (time.time() - start_time) * 1000
            
            result = ImputationResult(
                imputed_values=imputed_data,
                method_used=self.config.primary_method,
                confidence_score=confidence,
                computation_time_ms=computation_time,
                missing_count=missing_count,
                strategy_metadata={
                    "fill_strategy": self.config.primary_method.value,
                    "fallback_used": imputed_data.isna().sum() == 0
                }
            )
            
            # Cache result
            if self.config.cache_results:
                self._cache[cache_key] = result
                
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error in simple imputation for {self.config.feature_key}: {e}",
                source_module=self._source_module
            )
            raise
            
    def _apply_fallback(self, imputed_data: pd.Series, original_data: pd.Series) -> pd.Series:
        """Apply fallback method for remaining missing values."""
        if self.config.fallback_method == ImputationMethod.MEAN:
            imputed_data = imputed_data.fillna(original_data.mean())
        elif self.config.fallback_method == ImputationMethod.MEDIAN:
            imputed_data = imputed_data.fillna(original_data.median())
        else:
            # Default fallback to forward fill then mean
            imputed_data = imputed_data.fillna(method='ffill').fillna(original_data.mean())
            
        return imputed_data
        
    def validate_parameters(self) -> bool:
        """Validate simple imputation parameters."""
        return (
            self.config.max_gap_minutes > 0 and
            self.config.max_gap_minutes <= 1440  # Max 24 hours
        )


class TimeSeriesImputationStrategy(ImputationStrategy):
    """Advanced time series imputation using interpolation methods."""
    
    async def impute(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series,
        context: Optional[Dict[str, Any]] = None
    ) -> ImputationResult:
        """Perform time series interpolation."""
        start_time = time.time()
        cache_key = self.get_cache_key(data, missing_mask)
        
        if self.should_use_cache(cache_key):
            self._cache_hits += 1
            return self._cache[cache_key]
            
        self._cache_misses += 1
        
        try:
            imputed_data = data.copy()
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                return ImputationResult(
                    imputed_values=imputed_data,
                    method_used=self.config.primary_method,
                    confidence_score=1.0,
                    computation_time_ms=0,
                    missing_count=0
                )
            
            # Apply interpolation method
            if self.config.primary_method == ImputationMethod.LINEAR_INTERPOLATION:
                imputed_data = data.interpolate(method='linear', limit=self.config.max_gap_minutes)
                confidence = self._calculate_interpolation_confidence(data, missing_mask, 'linear')
                
            elif self.config.primary_method == ImputationMethod.SPLINE_INTERPOLATION:
                # Use spline interpolation for smoother curves
                imputed_data = data.interpolate(
                    method='spline', 
                    order=min(3, len(data.dropna()) - 1),
                    limit=self.config.max_gap_minutes
                )
                confidence = self._calculate_interpolation_confidence(data, missing_mask, 'spline')
                
            else:
                # Default to linear
                imputed_data = data.interpolate(method='linear')
                confidence = 0.75
                
            # Handle any remaining missing values
            if imputed_data.isna().any():
                imputed_data = imputed_data.fillna(method='ffill').fillna(data.mean())
                confidence *= 0.7
                
            computation_time = (time.time() - start_time) * 1000
            
            result = ImputationResult(
                imputed_values=imputed_data,
                method_used=self.config.primary_method,
                confidence_score=confidence,
                computation_time_ms=computation_time,
                missing_count=missing_count,
                strategy_metadata={
                    "interpolation_method": self.config.primary_method.value,
                    "max_gap_handled": self.config.max_gap_minutes
                }
            )
            
            if self.config.cache_results:
                self._cache[cache_key] = result
                
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error in time series imputation for {self.config.feature_key}: {e}",
                source_module=self._source_module
            )
            raise
            
    def _calculate_interpolation_confidence(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series, 
        method: str
    ) -> float:
        """Calculate confidence based on interpolation quality."""
        base_confidence = 0.8 if method == 'linear' else 0.85
        
        # Reduce confidence for large gaps
        consecutive_missing = self._get_max_consecutive_missing(missing_mask)
        gap_penalty = min(consecutive_missing / self.config.max_gap_minutes * 0.3, 0.4)
        
        # Reduce confidence based on data volatility
        volatility = data.std() / data.mean() if data.mean() != 0 else 0
        volatility_penalty = min(volatility * 0.1, 0.2)
        
        return max(base_confidence - gap_penalty - volatility_penalty, 0.3)
        
    def _get_max_consecutive_missing(self, missing_mask: pd.Series) -> int:
        """Get maximum consecutive missing values."""
        consecutive_counts = []
        current_count = 0
        
        for is_missing in missing_mask:
            if is_missing:
                current_count += 1
            else:
                if current_count > 0:
                    consecutive_counts.append(current_count)
                current_count = 0
                
        if current_count > 0:
            consecutive_counts.append(current_count)
            
        return max(consecutive_counts) if consecutive_counts else 0
        
    def validate_parameters(self) -> bool:
        """Validate time series imputation parameters."""
        return (
            self.config.max_gap_minutes > 0 and
            self.config.max_gap_minutes <= 720  # Max 12 hours for interpolation
        )


class KNNImputationStrategy(ImputationStrategy):
    """K-Nearest Neighbors imputation for complex missing patterns."""
    
    def __init__(self, config: ImputationConfig, logger: LoggerService):
        super().__init__(config, logger)
        self._knn_imputer = None
        self._feature_scaler = StandardScaler()
        
    async def impute(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series,
        context: Optional[Dict[str, Any]] = None
    ) -> ImputationResult:
        """Perform KNN-based imputation."""
        start_time = time.time()
        
        try:
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                return ImputationResult(
                    imputed_values=data,
                    method_used=ImputationMethod.KNN,
                    confidence_score=1.0,
                    computation_time_ms=0,
                    missing_count=0
                )
            
            # Prepare feature matrix for KNN
            feature_matrix = self._prepare_feature_matrix(data, context)
            
            if feature_matrix is None or feature_matrix.shape[1] < 2:
                # Fall back to simple method if insufficient features
                simple_strategy = SimpleImputationStrategy(self.config, self.logger)
                return await simple_strategy.impute(data, missing_mask, context)
            
            # Initialize KNN imputer if needed
            if self._knn_imputer is None:
                self._knn_imputer = KNNImputer(
                    n_neighbors=min(self.config.knn_neighbors, len(data) // 4),
                    weights='distance'
                )
            
            # Perform imputation
            imputed_matrix = self._knn_imputer.fit_transform(feature_matrix)
            imputed_values = pd.Series(
                imputed_matrix[:, 0], 
                index=data.index, 
                name=data.name
            )
            
            # Calculate confidence based on neighbor quality
            confidence = self._calculate_knn_confidence(data, missing_mask, feature_matrix)
            
            computation_time = (time.time() - start_time) * 1000
            
            result = ImputationResult(
                imputed_values=imputed_values,
                method_used=ImputationMethod.KNN,
                confidence_score=confidence,
                computation_time_ms=computation_time,
                missing_count=missing_count,
                strategy_metadata={
                    "n_neighbors": self.config.knn_neighbors,
                    "features_used": feature_matrix.shape[1],
                    "neighbor_quality": confidence
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error in KNN imputation for {self.config.feature_key}: {e}",
                source_module=self._source_module
            )
            # Fall back to simple method
            simple_strategy = SimpleImputationStrategy(self.config, self.logger)
            return await simple_strategy.impute(data, missing_mask, context)
            
    def _prepare_feature_matrix(
        self, 
        data: pd.Series, 
        context: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Prepare feature matrix for KNN imputation."""
        features = [data.values.reshape(-1, 1)]
        
        if context:
            # Add lagged features
            for lag in [1, 2, 3, 5, 10]:
                if len(data) > lag:
                    lagged = data.shift(lag).values.reshape(-1, 1)
                    features.append(lagged)
            
            # Add rolling statistics
            for window in [5, 10, 20]:
                if len(data) > window:
                    rolling_mean = data.rolling(window).mean().values.reshape(-1, 1)
                    rolling_std = data.rolling(window).std().values.reshape(-1, 1)
                    features.extend([rolling_mean, rolling_std])
            
            # Add related market data if available
            if 'related_series' in context:
                for series_name, series_data in context['related_series'].items():
                    if len(series_data) == len(data):
                        features.append(series_data.values.reshape(-1, 1))
        
        if len(features) > 1:
            feature_matrix = np.hstack(features)
            return feature_matrix
        else:
            return None
            
    def _calculate_knn_confidence(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series, 
        feature_matrix: np.ndarray
    ) -> float:
        """Calculate confidence based on KNN neighbor quality."""
        base_confidence = 0.75
        
        # Penalty for high missing ratio
        missing_ratio = missing_mask.sum() / len(missing_mask)
        missing_penalty = missing_ratio * 0.3
        
        # Bonus for more features
        feature_bonus = min((feature_matrix.shape[1] - 1) * 0.05, 0.15)
        
        return max(base_confidence - missing_penalty + feature_bonus, 0.4)
        
    def validate_parameters(self) -> bool:
        """Validate KNN imputation parameters."""
        return (
            self.config.knn_neighbors > 0 and
            self.config.knn_neighbors <= 50 and
            self.config.model_lookback_periods >= 20
        )


class CryptoFinancialImputationStrategy(ImputationStrategy):
    """Cryptocurrency-specific imputation using financial domain knowledge."""
    
    async def impute(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series,
        context: Optional[Dict[str, Any]] = None
    ) -> ImputationResult:
        """Perform crypto-specific imputation."""
        start_time = time.time()
        
        try:
            missing_count = missing_mask.sum()
            
            if missing_count == 0:
                return ImputationResult(
                    imputed_values=data,
                    method_used=self.config.primary_method,
                    confidence_score=1.0,
                    computation_time_ms=0,
                    missing_count=0
                )
            
            imputed_data = data.copy()
            
            if self.config.primary_method == ImputationMethod.VWAP_WEIGHTED:
                imputed_data = await self._vwap_weighted_imputation(data, missing_mask, context)
                confidence = 0.85
                
            elif self.config.primary_method == ImputationMethod.VOLATILITY_ADJUSTED:
                imputed_data = await self._volatility_adjusted_imputation(data, missing_mask, context)
                confidence = 0.80
                
            elif self.config.primary_method == ImputationMethod.MARKET_SESSION_AWARE:
                imputed_data = await self._market_session_aware_imputation(data, missing_mask, context)
                confidence = 0.75
                
            else:
                # Default to forward fill for crypto
                imputed_data = data.fillna(method='ffill', limit=self.config.max_gap_minutes)
                confidence = 0.70
                
            # Handle any remaining missing values
            if imputed_data.isna().any():
                imputed_data = imputed_data.fillna(method='ffill').fillna(data.mean())
                confidence *= 0.8
                
            computation_time = (time.time() - start_time) * 1000
            
            result = ImputationResult(
                imputed_values=imputed_data,
                method_used=self.config.primary_method,
                confidence_score=confidence,
                computation_time_ms=computation_time,
                missing_count=missing_count,
                strategy_metadata={
                    "crypto_method": self.config.primary_method.value,
                    "data_type": self.config.data_type.value
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error in crypto financial imputation for {self.config.feature_key}: {e}",
                source_module=self._source_module
            )
            raise
            
    async def _vwap_weighted_imputation(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series, 
        context: Optional[Dict[str, Any]]
    ) -> pd.Series:
        """Impute using volume-weighted average price logic."""
        imputed_data = data.copy()
        
        if context and 'volume' in context:
            volume = context['volume']
            
            # Calculate VWAP for recent periods
            recent_periods = min(20, len(data))
            recent_data = data.tail(recent_periods)
            recent_volume = volume.tail(recent_periods)
            
            # Volume-weighted average
            vwap = (recent_data * recent_volume).sum() / recent_volume.sum()
            
            # Use VWAP for imputation with some noise
            for idx in data.index[missing_mask]:
                # Add small random variation based on recent volatility
                volatility = recent_data.std()
                noise = np.random.normal(0, volatility * 0.1)
                imputed_data[idx] = vwap + noise
        else:
            # Fallback to weighted average of recent values
            imputed_data = data.fillna(method='ffill')
            
        return imputed_data
        
    async def _volatility_adjusted_imputation(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series, 
        context: Optional[Dict[str, Any]]
    ) -> pd.Series:
        """Impute considering market volatility regime."""
        imputed_data = data.copy()
        
        # Calculate recent volatility
        volatility = data.rolling(20).std().iloc[-1] if len(data) >= 20 else data.std()
        
        # Adjust imputation based on volatility regime
        if volatility > data.std() * 1.5:  # High volatility
            # Use more conservative forward fill
            imputed_data = data.fillna(method='ffill', limit=5)
        else:  # Normal/low volatility
            # Can use interpolation more aggressively
            imputed_data = data.interpolate(method='linear', limit=self.config.max_gap_minutes)
            
        return imputed_data
        
    async def _market_session_aware_imputation(
        self, 
        data: pd.Series, 
        missing_mask: pd.Series, 
        context: Optional[Dict[str, Any]]
    ) -> pd.Series:
        """Impute considering different market sessions with advanced patterns."""
        try:
            # Import enhanced temporal pattern engine if available
            from .feature_engine_enhancements import AdvancedTemporalPatternEngine
            
            # Use advanced temporal pattern analysis
            pattern_engine = AdvancedTemporalPatternEngine(self.logger)
            
            # Create DataFrame for pattern analysis
            df = pd.DataFrame({'value': data})
            temporal_patterns = pattern_engine.extract_temporal_patterns(df, 'value')
            
            imputed_data = data.copy()
            
            # Use market session effects if available
            if 'market_sessions' in temporal_patterns and hasattr(data.index, 'hour'):
                session_effects = temporal_patterns['market_sessions']
                
                for idx in data.index[missing_mask]:
                    hour = idx.hour
                    
                    # Find which session this hour belongs to
                    session_value = None
                    for session_name, session_info in session_effects.items():
                        if 'hours' in session_info:
                            # Parse hours from string like "22:00-10:00 UTC"
                            hours_str = session_info['hours'].replace(' UTC', '')
                            start_str, end_str = hours_str.split('-')
                            start_hour = int(start_str.split(':')[0])
                            end_hour = int(end_str.split(':')[0])
                            
                            # Check if hour is in session (handle crossing midnight)
                            if start_hour > end_hour:
                                if hour >= start_hour or hour < end_hour:
                                    session_value = session_info.get('mean', data.mean())
                                    break
                            else:
                                if start_hour <= hour < end_hour:
                                    session_value = session_info.get('mean', data.mean())
                                    break
                    
                    if session_value is not None:
                        # Add trend adjustment
                        recent_data = data.iloc[max(0, data.index.get_loc(idx) - 5):data.index.get_loc(idx)]
                        if len(recent_data.dropna()) > 1:
                            trend = recent_data.dropna().diff().mean()
                            imputed_data[idx] = session_value + trend * 0.3
                        else:
                            imputed_data[idx] = session_value
                    else:
                        # Fallback to hourly patterns
                        if 'time_of_day' in temporal_patterns:
                            hourly_stats = temporal_patterns['time_of_day'].get('hourly_statistics', {})
                            if 'mean' in hourly_stats and hour in hourly_stats['mean']:
                                imputed_data[idx] = hourly_stats['mean'][hour]
                
            elif hasattr(data.index, 'hour'):
                # Fallback to simple hourly patterns
                hourly_patterns = data.groupby(data.index.hour).mean()
                
                for idx in data.index[missing_mask]:
                    hour = idx.hour
                    if hour in hourly_patterns:
                        base_value = hourly_patterns[hour]
                        recent_trend = data.iloc[-5:].mean() - data.iloc[-10:-5].mean()
                        imputed_data[idx] = base_value + recent_trend * 0.5
            else:
                # Final fallback
                imputed_data = data.fillna(method='ffill')
                
            return imputed_data
            
        except ImportError:
            # Fallback to simple implementation
            imputed_data = data.copy()
            
            if hasattr(data.index, 'hour'):
                hourly_patterns = data.groupby(data.index.hour).mean()
                
                for idx in data.index[missing_mask]:
                    hour = idx.hour
                    if hour in hourly_patterns:
                        # Use hourly average with some adjustment
                        base_value = hourly_patterns[hour]
                        recent_trend = data.iloc[-5:].mean() - data.iloc[-10:-5].mean()
                        imputed_data[idx] = base_value + recent_trend * 0.5
            else:
                # Fallback
                imputed_data = data.fillna(method='ffill')
                
            return imputed_data
        
    def validate_parameters(self) -> bool:
        """Validate crypto financial imputation parameters."""
        return True  # Basic validation, could be extended


# === Management and Orchestration ===

class ImputationManager:
    """Central manager for imputation strategies and execution."""
    
    def __init__(self, logger: LoggerService, config_path: Optional[str] = None):
        self.logger = logger
        self._source_module = self.__class__.__name__
        
        # Strategy registry
        self._strategies: Dict[str, Type[ImputationStrategy]] = {
            'simple': SimpleImputationStrategy,
            'timeseries': TimeSeriesImputationStrategy,
            'knn': KNNImputationStrategy,
            'crypto_financial': CryptoFinancialImputationStrategy
        }
        
        # Configuration storage
        self._feature_configs: Dict[str, ImputationConfig] = {}
        self._default_config = ImputationConfig(
            feature_key="default",
            data_type=DataType.PRICE,
            primary_method=ImputationMethod.FORWARD_FILL,
            fallback_method=ImputationMethod.MEAN,
            quality_level=ImputationQuality.BALANCED
        )
        
        # Performance tracking
        self._performance_metrics: Dict[str, List[ImputationMetrics]] = defaultdict(list)
        self._strategy_instances: Dict[str, ImputationStrategy] = {}
        
        # Load configuration
        if config_path:
            self.load_configuration(config_path)
        else:
            self._load_default_configurations()
            
    def register_strategy(self, name: str, strategy_class: Type[ImputationStrategy]) -> None:
        """Register a new imputation strategy."""
        self._strategies[name] = strategy_class
        self.logger.info(
            f"Registered imputation strategy: {name}",
            source_module=self._source_module
        )
        
    def configure_feature(self, feature_key: str, config: ImputationConfig) -> None:
        """Configure imputation for a specific feature."""
        self._feature_configs[feature_key] = config
        self.logger.debug(
            f"Configured imputation for feature {feature_key}: {config.primary_method.value}",
            source_module=self._source_module
        )
        
    async def impute_feature(
        self, 
        feature_key: str, 
        data: pd.Series, 
        context: Optional[Dict[str, Any]] = None
    ) -> ImputationResult:
        """Impute missing values for a specific feature."""
        try:
            # Get configuration
            config = self._feature_configs.get(feature_key, self._default_config)
            
            # Create missing mask
            missing_mask = data.isna()
            
            if not missing_mask.any():
                return ImputationResult(
                    imputed_values=data,
                    method_used=config.primary_method,
                    confidence_score=1.0,
                    computation_time_ms=0,
                    missing_count=0
                )
            
            # Select strategy
            strategy = await self._get_strategy_instance(feature_key, config)
            
            # Perform imputation
            result = await strategy.impute(data, missing_mask, context)
            
            # Track performance
            await self._track_performance(feature_key, result)
            
            self.logger.debug(
                f"Imputed {result.missing_count} values for {feature_key} "
                f"using {result.method_used.value} (confidence: {result.confidence_score:.2f})",
                source_module=self._source_module
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error imputing feature {feature_key}: {e}",
                source_module=self._source_module
            )
            raise
            
    async def _get_strategy_instance(
        self, 
        feature_key: str, 
        config: ImputationConfig
    ) -> ImputationStrategy:
        """Get or create strategy instance for feature."""
        strategy_key = f"{feature_key}_{config.primary_method.value}"
        
        if strategy_key not in self._strategy_instances:
            # Determine strategy type based on method
            if config.primary_method in [ImputationMethod.MEAN, ImputationMethod.MEDIAN, 
                                       ImputationMethod.FORWARD_FILL, ImputationMethod.BACKWARD_FILL]:
                strategy_class = self._strategies['simple']
            elif config.primary_method in [ImputationMethod.LINEAR_INTERPOLATION, 
                                         ImputationMethod.SPLINE_INTERPOLATION]:
                strategy_class = self._strategies['timeseries']
            elif config.primary_method == ImputationMethod.KNN:
                strategy_class = self._strategies['knn']
            elif config.primary_method in [ImputationMethod.VWAP_WEIGHTED,
                                         ImputationMethod.VOLATILITY_ADJUSTED,
                                         ImputationMethod.MARKET_SESSION_AWARE]:
                strategy_class = self._strategies['crypto_financial']
            else:
                strategy_class = self._strategies['simple']  # Default
            
            # Create strategy instance
            self._strategy_instances[strategy_key] = strategy_class(config, self.logger)
            
        return self._strategy_instances[strategy_key]
        
    async def _track_performance(self, feature_key: str, result: ImputationResult) -> None:
        """Track imputation performance metrics."""
        metrics = ImputationMetrics(
            method=result.method_used,
            feature_key=feature_key,
            accuracy_score=result.confidence_score,  # Using confidence as proxy
            mean_absolute_error=0.0,  # Would need validation data
            mean_squared_error=0.0,   # Would need validation data  
            computation_time_ms=result.computation_time_ms,
            cache_hit_rate=0.0,  # Would get from strategy
            samples_evaluated=result.missing_count
        )
        
        self._performance_metrics[feature_key].append(metrics)
        
        # Keep only recent metrics
        if len(self._performance_metrics[feature_key]) > 100:
            self._performance_metrics[feature_key] = self._performance_metrics[feature_key][-100:]
            
    def get_performance_summary(self, feature_key: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary for features."""
        if feature_key:
            metrics = self._performance_metrics.get(feature_key, [])
            if not metrics:
                return {}
                
            return {
                "feature": feature_key,
                "total_imputations": len(metrics),
                "avg_computation_time_ms": np.mean([m.computation_time_ms for m in metrics]),
                "avg_confidence": np.mean([m.accuracy_score for m in metrics]),
                "methods_used": list(set(m.method.value for m in metrics)),
                "last_updated": max(m.timestamp for m in metrics) if metrics else None
            }
        else:
            # Summary for all features
            summary = {}
            for fkey in self._performance_metrics:
                summary[fkey] = self.get_performance_summary(fkey)
            return summary
            
    def load_configuration(self, config_path: str) -> None:
        """Load imputation configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            for feature_config in config_data.get('features', []):
                config = ImputationConfig(
                    feature_key=feature_config['feature_key'],
                    data_type=DataType(feature_config.get('data_type', 'price')),
                    primary_method=ImputationMethod(feature_config['primary_method']),
                    fallback_method=ImputationMethod(feature_config.get('fallback_method', 'mean')),
                    quality_level=ImputationQuality(feature_config.get('quality_level', 'balanced')),
                    max_gap_minutes=feature_config.get('max_gap_minutes', 15),
                    knn_neighbors=feature_config.get('knn_neighbors', 5),
                    confidence_threshold=feature_config.get('confidence_threshold', 0.8)
                )
                self._feature_configs[config.feature_key] = config
                
            self.logger.info(
                f"Loaded imputation configuration for {len(self._feature_configs)} features",
                source_module=self._source_module
            )
            
        except Exception as e:
            self.logger.error(
                f"Error loading imputation configuration from {config_path}: {e}",
                source_module=self._source_module
            )
            self._load_default_configurations()
            
    def _load_default_configurations(self) -> None:
        """Load default configurations for common feature types."""
        # Price features
        price_config = ImputationConfig(
            feature_key="price_default",
            data_type=DataType.PRICE,
            primary_method=ImputationMethod.FORWARD_FILL,
            fallback_method=ImputationMethod.LINEAR_INTERPOLATION,
            max_gap_minutes=10
        )
        
        # Volume features  
        volume_config = ImputationConfig(
            feature_key="volume_default", 
            data_type=DataType.VOLUME,
            primary_method=ImputationMethod.FORWARD_FILL,
            fallback_method=ImputationMethod.MEAN,
            max_gap_minutes=5
        )
        
        # Volatility features
        volatility_config = ImputationConfig(
            feature_key="volatility_default",
            data_type=DataType.VOLATILITY,
            primary_method=ImputationMethod.LINEAR_INTERPOLATION,
            fallback_method=ImputationMethod.MEDIAN,
            max_gap_minutes=15
        )
        
        # Technical indicators
        indicator_config = ImputationConfig(
            feature_key="indicator_default",
            data_type=DataType.INDICATOR,
            primary_method=ImputationMethod.LINEAR_INTERPOLATION,
            fallback_method=ImputationMethod.FORWARD_FILL,
            max_gap_minutes=20
        )
        
        self._feature_configs.update({
            "price_default": price_config,
            "volume_default": volume_config, 
            "volatility_default": volatility_config,
            "indicator_default": indicator_config
        })


# === Validation and Testing Framework ===

class ImputationValidator:
    """Validates and benchmarks imputation strategies."""
    
    def __init__(self, logger: LoggerService):
        self.logger = logger
        self._source_module = self.__class__.__name__
        
    async def validate_strategy(
        self, 
        strategy: ImputationStrategy,
        test_data: pd.Series,
        missing_patterns: List[pd.Series],
        ground_truth: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Validate a strategy against test data with known missing patterns."""
        results = {}
        
        for i, missing_mask in enumerate(missing_patterns):
            # Create test data with artificial missing values
            test_series = test_data.copy()
            test_series[missing_mask] = np.nan
            
            try:
                # Perform imputation
                result = await strategy.impute(test_series, missing_mask)
                
                # Calculate metrics if ground truth available
                if ground_truth is not None:
                    original_values = ground_truth[missing_mask]
                    imputed_values = result.imputed_values[missing_mask]
                    
                    mae = mean_absolute_error(original_values, imputed_values)
                    mse = mean_squared_error(original_values, imputed_values)
                    
                    # Correlation with original
                    corr, _ = pearsonr(original_values, imputed_values)
                    
                    results[f"pattern_{i}_mae"] = mae
                    results[f"pattern_{i}_mse"] = mse
                    results[f"pattern_{i}_correlation"] = corr
                    results[f"pattern_{i}_confidence"] = result.confidence_score
                    results[f"pattern_{i}_time_ms"] = result.computation_time_ms
                    
            except Exception as e:
                self.logger.error(
                    f"Error validating strategy for pattern {i}: {e}",
                    source_module=self._source_module
                )
                results[f"pattern_{i}_error"] = True
                
        return results
        
    def generate_missing_patterns(
        self, 
        data_length: int, 
        pattern_types: List[str] = None
    ) -> List[pd.Series]:
        """Generate various missing data patterns for testing."""
        if pattern_types is None:
            pattern_types = ['random', 'consecutive', 'periodic', 'burst']
            
        patterns = []
        
        for pattern_type in pattern_types:
            if pattern_type == 'random':
                # Random 10% missing
                mask = np.random.choice([True, False], size=data_length, p=[0.1, 0.9])
                
            elif pattern_type == 'consecutive':
                # Consecutive missing blocks
                mask = np.zeros(data_length, dtype=bool)
                start_idx = np.random.randint(0, data_length - 20)
                mask[start_idx:start_idx + 15] = True
                
            elif pattern_type == 'periodic':
                # Periodic missing (every 10th value)
                mask = np.zeros(data_length, dtype=bool)
                mask[::10] = True
                
            elif pattern_type == 'burst':
                # Multiple small bursts
                mask = np.zeros(data_length, dtype=bool)
                for _ in range(5):
                    start_idx = np.random.randint(0, data_length - 5)
                    mask[start_idx:start_idx + 3] = True
                    
            patterns.append(pd.Series(mask))
            
        return patterns
        
    async def benchmark_strategies(
        self,
        strategies: List[ImputationStrategy],
        test_data: pd.Series,
        num_trials: int = 10
    ) -> pd.DataFrame:
        """Benchmark multiple strategies against each other."""
        results = []
        
        missing_patterns = self.generate_missing_patterns(len(test_data))
        
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__
            
            for trial in range(num_trials):
                validation_results = await self.validate_strategy(
                    strategy, test_data, missing_patterns, test_data
                )
                
                # Aggregate results
                avg_mae = np.mean([v for k, v in validation_results.items() if k.endswith('_mae')])
                avg_mse = np.mean([v for k, v in validation_results.items() if k.endswith('_mse')])
                avg_corr = np.mean([v for k, v in validation_results.items() if k.endswith('_correlation')])
                avg_time = np.mean([v for k, v in validation_results.items() if k.endswith('_time_ms')])
                
                results.append({
                    'strategy': strategy_name,
                    'trial': trial,
                    'avg_mae': avg_mae,
                    'avg_mse': avg_mse,
                    'avg_correlation': avg_corr,
                    'avg_time_ms': avg_time
                })
                
        return pd.DataFrame(results)


# === Integration Helper ===

def create_imputation_manager(
    logger: LoggerService, 
    config: Optional[Dict[str, Any]] = None
) -> ImputationManager:
    """Factory function to create configured ImputationManager."""
    manager = ImputationManager(logger)
    
    if config:
        # Configure features based on provided config
        for feature_key, feature_config in config.items():
            imputation_config = ImputationConfig(
                feature_key=feature_key,
                data_type=DataType(feature_config.get('data_type', 'price')),
                primary_method=ImputationMethod(feature_config.get('primary_method', 'forward_fill')),
                fallback_method=ImputationMethod(feature_config.get('fallback_method', 'mean')),
                quality_level=ImputationQuality(feature_config.get('quality_level', 'balanced')),
                max_gap_minutes=feature_config.get('max_gap_minutes', 15),
                knn_neighbors=feature_config.get('knn_neighbors', 5)
            )
            manager.configure_feature(feature_key, imputation_config)
    
    return manager 