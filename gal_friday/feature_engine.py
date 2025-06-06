"""Feature engineering for Gal-Friday using a configurable Scikit-learn pipeline approach.

This module provides the `FeatureEngine` class, which is responsible for:
1.  Loading canonical feature definitions from a YAML-based Feature Registry
    (e.g., `config/feature_registry.yaml`).
2.  Processing application-level configuration (from `config.yaml`) to activate
    and override these registry definitions. This allows for customized feature sets
    per bot instance.
3.  Managing historical market data, including OHLCV bars, L2 order book snapshots,
    and raw trade data.
4.  Constructing Scikit-learn pipelines for each activated and configured feature.
    These pipelines handle:
    a.  Input data selection and preparation.
    b.  Core feature calculation, often leveraging `pandas-ta` or custom static methods
        (e.g., `_pipeline_compute_rsi`). These calculator methods are designed for
        robustness, including "Zero NaN" output policies where appropriate (e.g.,
        filling RSI NaNs with 50, or MACD NaNs with 0).
    c.  Optional final output imputation (as a fallback, if a calculator still
        produces NaNs despite its internal handling).
    d.  Centralized output scaling (e.g., StandardScaler, MinMaxScaler) applied
        universally based on feature configuration, ensuring features are ready for
        model consumption.
5.  Executing these pipelines when new market data arrives (typically triggered by
    the close of a new OHLCV bar).
6.  Validating the dictionary of calculated numerical features against a Pydantic model
    (`PublishedFeaturesV1`) to ensure data integrity and contract adherence.
7.  Publishing the validated, numerical features as a dictionary payload within a
    `FEATURES_CALCULATED` event on the PubSub system.

Predictors downstream expect features to be pre-scaled and numerical, as handled by this engine.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field  # Added for InternalFeatureSpec
from datetime import datetime
from decimal import Decimal
from enum import Enum  # Added for _LocalFeatureCategory
from pathlib import Path  # Added for feature registry
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import yaml  # Added for feature registry
from sklearn.base import clone  # For cloning pipelines to modify params at runtime
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

from gal_friday.core.events import EventType
from gal_friday.core.feature_models import PublishedFeaturesV1  # Added for Pydantic model

# Attempt to import FeatureCategory, handle potential circularity if it arises
try:
    from gal_friday.interfaces.feature_engine_interface import FeatureCategory
except ImportError:
    # Define local fallback enum
    class _LocalFeatureCategory(Enum):
        TECHNICAL = "TECHNICAL"
        L2_ORDER_BOOK = "L2_ORDER_BOOK"
        TRADE_DATA = "TRADE_DATA"
        SENTIMENT = "SENTIMENT"
        CUSTOM = "CUSTOM"
        UNKNOWN = "UNKNOWN"

    FeatureCategory = _LocalFeatureCategory  # type: ignore[assignment,misc]


if TYPE_CHECKING:
    from collections.abc import Callable

    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.interfaces.historical_data_service_interface import HistoricalDataService
    from gal_friday.logger_service import LoggerService

# Define InternalFeatureSpec
@dataclass
class InternalFeatureSpec:
    key: str  # Unique key for the feature. Used for activation via app config and as a base for published feature names.
    calculator_type: str # Defines the core calculation logic (e.g., "rsi", "macd"). Maps to a `_pipeline_compute_{calculator_type}` method.
    input_type: str # Specifies the type of input data required by the calculator (e.g., 'close_series', 'ohlcv_df', 'l2_book_series').
    category: FeatureCategory = FeatureCategory.TECHNICAL # Categorizes the feature (e.g., TECHNICAL, L2_ORDER_BOOK, TRADE_DATA).
    parameters: dict[str, Any] = field(default_factory=dict) # Dictionary of parameters passed to the feature calculator function.
    imputation: dict[str, Any] | str | None = None # Configuration for the output imputation step in the pipeline (e.g., `{"strategy": "constant", "fill_value": 0.0}`). Applied as a final fallback.
    scaling: dict[str, Any] | str | None = None    # Configuration for the output scaling step (e.g., `{"method": "standard"}`). Applied by FeatureEngine.
    description: str = "" # Human-readable description of the feature and its configuration.
    version: str | None = None # Version string for the feature definition, loaded from the registry.
    output_properties: dict[str, Any] = field(default_factory=dict) # Dictionary describing expected output characteristics (e.g., `{"value_type": "float", "range": [0, 1]}`).
    
    # Enhanced fields for comprehensive output handling and multiple outputs
    output_specs: list["OutputSpec"] = field(default_factory=list) # Detailed specifications for each output
    output_naming_pattern: str | None = None # Pattern for naming outputs (e.g., '{feature_name}_{output_name}')
    dependencies: list[str] = field(default_factory=list) # Other features this depends on
    required_lookback_periods: int = 1 # Minimum data points required
    author: str | None = None # Feature author/creator
    created_at: str | None = None # Creation timestamp
    tags: list[str] = field(default_factory=list) # Feature tags for organization
    cache_enabled: bool = True # Whether to cache feature results
    cache_ttl_minutes: int | None = None # Cache time-to-live in minutes
    computation_priority: int = 5 # Computation priority (1-10)
    
    @property
    def output_names(self) -> list[str]:
        """Get list of all output names based on specs and naming pattern."""
        if not self.output_specs:
            return [self.key]  # Default to feature key if no specs
        
        if self.output_naming_pattern:
            return [
                self.output_naming_pattern.format(
                    feature_name=self.key,
                    output_name=spec.name
                )
                for spec in self.output_specs
            ]
        return [spec.name for spec in self.output_specs]
    
    @property
    def expected_output_count(self) -> int:
        """Get expected number of outputs."""
        return len(self.output_specs) if self.output_specs else 1


class OutputType(Enum):
    """Types of feature outputs supported."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"
    STRING = "string"


@dataclass
class OutputSpec:
    """Specification for a single feature output."""
    name: str
    output_type: OutputType = OutputType.NUMERIC
    description: str | None = None
    validation_range: tuple[float, float] | None = None
    nullable: bool = True
    default_value: Any = None


class FeatureValidationError(Exception):
    """Exception raised for feature validation errors."""
    pass


class FeatureProcessingError(Exception):
    """Exception raised for feature processing errors."""
    pass


class FeatureOutputHandler:
    """Enhanced handler for processing multiple feature outputs according to specifications."""
    
    def __init__(self, feature_spec: InternalFeatureSpec):
        self.spec = feature_spec
        self.logger = None  # Will be set by FeatureEngine if available
        
    def process_feature_outputs(self, raw_outputs: Any) -> pd.DataFrame:
        """
        Process raw feature computation outputs according to specification.
        Handles multiple output formats and applies validation, type conversion, and naming.
        
        Args:
            raw_outputs: Raw output from feature calculation (Series, DataFrame, dict, list, scalar)
            
        Returns:
            Processed DataFrame with properly named and validated outputs
            
        Raises:
            FeatureProcessingError: If output processing fails
        """
        try:
            # Convert raw outputs to standardized DataFrame format
            standardized_outputs = self._standardize_raw_outputs(raw_outputs)
            
            # Validate output structure matches expectations
            self._validate_output_structure(standardized_outputs)
            
            # Apply output specifications (type conversion, validation)
            processed_outputs = self._apply_output_specifications(standardized_outputs)
            
            # Apply naming pattern
            final_outputs = self._apply_output_naming(processed_outputs)
            
            # Add metadata
            final_outputs = self._add_output_metadata(final_outputs)
            
            if self.logger:
                self.logger.debug(
                    f"Successfully processed {len(final_outputs.columns)} outputs for feature {self.spec.key}"
                )
            
            return final_outputs
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing outputs for feature {self.spec.key}: {e}")
            raise FeatureProcessingError(f"Failed to process feature outputs: {e}")
    
    def _standardize_raw_outputs(self, raw_outputs: Any) -> pd.DataFrame:
        """Convert various output formats to standardized DataFrame."""
        if isinstance(raw_outputs, pd.DataFrame):
            return raw_outputs
        
        elif isinstance(raw_outputs, pd.Series):
            # Single series output
            output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
            return pd.DataFrame({output_name: raw_outputs})
        
        elif isinstance(raw_outputs, np.ndarray):
            # NumPy array - could be 1D or 2D
            if raw_outputs.ndim == 1:
                output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
                return pd.DataFrame({output_name: raw_outputs})
            else:
                # Multi-dimensional array
                columns = [spec.name for spec in self.spec.output_specs[:raw_outputs.shape[1]]]
                if len(columns) < raw_outputs.shape[1]:
                    # Generate default names for additional columns
                    for i in range(len(columns), raw_outputs.shape[1]):
                        columns.append(f"output_{i}")
                return pd.DataFrame(raw_outputs, columns=columns)
        
        elif isinstance(raw_outputs, dict):
            # Dictionary of outputs
            return pd.DataFrame(raw_outputs)
        
        elif isinstance(raw_outputs, list):
            # List of values
            if len(self.spec.output_specs) == 1:
                output_name = self.spec.output_specs[0].name
                return pd.DataFrame({output_name: raw_outputs})
            else:
                # Multiple outputs - map to specs
                output_dict = {}
                for i, spec in enumerate(self.spec.output_specs):
                    if i < len(raw_outputs):
                        output_dict[spec.name] = raw_outputs[i] if isinstance(raw_outputs[i], (list, np.ndarray)) else [raw_outputs[i]]
                return pd.DataFrame(output_dict)
        
        else:
            # Single scalar value
            output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
            return pd.DataFrame({output_name: [raw_outputs]})
    
    def _validate_output_structure(self, outputs: pd.DataFrame) -> None:
        """Validate that outputs match expected structure."""
        if not self.spec.output_specs:
            return  # No validation if no specs defined
        
        expected_count = self.spec.expected_output_count
        actual_count = len(outputs.columns)
        
        if actual_count != expected_count:
            if actual_count < expected_count:
                if self.logger:
                    self.logger.warning(
                        f"Feature {self.spec.key} produced {actual_count} outputs, "
                        f"expected {expected_count}. Adding default values."
                    )
                # Add missing columns with default values
                for i in range(actual_count, expected_count):
                    spec = self.spec.output_specs[i]
                    default_val = spec.default_value if spec.default_value is not None else np.nan
                    outputs[spec.name] = default_val
            
            elif actual_count > expected_count:
                if self.logger:
                    self.logger.warning(
                        f"Feature {self.spec.key} produced {actual_count} outputs, "
                        f"expected {expected_count}. Truncating to expected count."
                    )
                # Keep only expected columns
                expected_columns = [spec.name for spec in self.spec.output_specs]
                outputs = outputs[expected_columns[:expected_count]]
    
    def _apply_output_specifications(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Apply type conversions and validations according to output specs."""
        if not self.spec.output_specs:
            return outputs  # No specs to apply
        
        processed = outputs.copy()
        
        for spec in self.spec.output_specs:
            if spec.name not in processed.columns:
                continue
            
            column = processed[spec.name]
            
            # Apply type conversion
            if spec.output_type == OutputType.NUMERIC:
                processed[spec.name] = pd.to_numeric(column, errors='coerce')
            elif spec.output_type == OutputType.CATEGORICAL:
                processed[spec.name] = column.astype('category')
            elif spec.output_type == OutputType.BOOLEAN:
                processed[spec.name] = column.astype(bool)
            elif spec.output_type == OutputType.TIMESTAMP:
                processed[spec.name] = pd.to_datetime(column, errors='coerce')
            elif spec.output_type == OutputType.STRING:
                processed[spec.name] = column.astype(str)
            
            # Apply validation range
            if spec.validation_range and spec.output_type == OutputType.NUMERIC:
                min_val, max_val = spec.validation_range
                out_of_range = (column < min_val) | (column > max_val)
                if out_of_range.any():
                    if self.logger:
                        self.logger.warning(
                            f"Feature {self.spec.key} output {spec.name} has "
                            f"{out_of_range.sum()} values outside range [{min_val}, {max_val}]"
                        )
                    # Clip values to range
                    processed[spec.name] = column.clip(min_val, max_val)
            
            # Handle nullability
            if not spec.nullable and column.isnull().any():
                if spec.default_value is not None:
                    processed[spec.name] = column.fillna(spec.default_value)
                else:
                    raise FeatureValidationError(
                        f"Feature {self.spec.key} output {spec.name} contains null values "
                        f"but nullability is disabled"
                    )
        
        return processed
    
    def _apply_output_naming(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Apply naming pattern to output columns."""
        if not self.spec.output_naming_pattern:
            return outputs
        
        renamed_outputs = outputs.copy()
        old_to_new_names = {}
        
        for i, spec in enumerate(self.spec.output_specs):
            if spec.name in outputs.columns:
                new_name = self.spec.output_naming_pattern.format(
                    feature_name=self.spec.key,
                    output_name=spec.name,
                    index=i
                )
                old_to_new_names[spec.name] = new_name
        
        return renamed_outputs.rename(columns=old_to_new_names)
    
    def _add_output_metadata(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Add metadata attributes to output DataFrame."""
        # Add feature metadata as DataFrame attributes
        outputs.attrs['feature_name'] = self.spec.key
        outputs.attrs['feature_version'] = self.spec.version
        outputs.attrs['output_count'] = len(outputs.columns)
        outputs.attrs['computation_timestamp'] = pd.Timestamp.now()
        
        if self.spec.tags:
            outputs.attrs['tags'] = self.spec.tags
        
        return outputs


@dataclass
class FeatureExtractionResult:
    """Result of advanced feature extraction."""
    features: pd.DataFrame
    feature_specs: list[InternalFeatureSpec]
    extraction_time: float
    quality_metrics: dict[str, float]
    cache_hits: int = 0
    cache_misses: int = 0


class AdvancedFeatureExtractor:
    """Enterprise-grade advanced feature extraction with technical indicators and market microstructure."""
    
    def __init__(self, config: dict[str, Any], logger_service: Any = None):
        self.config = config
        self.logger = logger_service
        
        # Feature registry and cache
        self.feature_registry: dict[str, InternalFeatureSpec] = {}
        self.feature_cache: dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self.extraction_stats = {
            'features_extracted': 0,
            'extraction_time_total': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Quality thresholds
        self.quality_thresholds = config.get("feature_quality", {
            "min_completeness": 0.8,
            "max_correlation": 0.95,
            "min_variance": 1e-6
        })
        
        self._initialize_advanced_indicators()
    
    def _initialize_advanced_indicators(self) -> None:
        """Initialize advanced technical indicators and market microstructure features."""
        # Enhanced technical indicators
        self.advanced_indicators = {
            # Momentum indicators
            'momentum': self._calculate_momentum,
            'rate_of_change': self._calculate_rate_of_change,
            'williams_percent_r': self._calculate_williams_r,
            'commodity_channel_index': self._calculate_cci,
            
            # Volatility indicators
            'bollinger_width': self._calculate_bollinger_width,
            'true_range': self._calculate_true_range,
            'average_true_range': self._calculate_atr_advanced,
            'volatility_ratio': self._calculate_volatility_ratio,
            
            # Volume indicators
            'on_balance_volume': self._calculate_obv,
            'accumulation_distribution': self._calculate_ad_line,
            'money_flow_index': self._calculate_mfi,
            'volume_oscillator': self._calculate_volume_oscillator,
            
            # Market microstructure
            'effective_spread': self._calculate_effective_spread,
            'quoted_spread': self._calculate_quoted_spread,
            'depth_imbalance': self._calculate_depth_imbalance,
            'order_flow_imbalance': self._calculate_order_flow_imbalance,
            'market_impact': self._calculate_market_impact,
            
            # Statistical features
            'price_momentum_oscillator': self._calculate_pmo,
            'adaptive_moving_average': self._calculate_ama,
            'fractal_dimension': self._calculate_fractal_dimension,
            'hurst_exponent': self._calculate_hurst_exponent,
        }
    
    async def extract_advanced_features(
        self, 
        data: pd.DataFrame, 
        feature_specs: list[InternalFeatureSpec],
        l2_data: dict[str, Any] | None = None,
        trade_data: list[dict[str, Any]] | None = None
    ) -> FeatureExtractionResult:
        """
        Extract advanced features with technical indicators and market microstructure.
        
        Args:
            data: OHLCV DataFrame
            feature_specs: List of feature specifications to compute
            l2_data: Level 2 order book data
            trade_data: Trade-level data
            
        Returns:
            FeatureExtractionResult with computed features and metadata
        """
        import time
        start_time = time.time()
        
        try:
            if self.logger:
                self.logger.info(f"Extracting {len(feature_specs)} advanced features from {len(data)} data points")
            
            # Validate input data
            self._validate_input_data(data)
            
            # Initialize feature DataFrame
            features_df = pd.DataFrame(index=data.index)
            
            # Extract features by category for optimal performance
            for category in [FeatureCategory.TECHNICAL, FeatureCategory.L2_ORDER_BOOK, 
                           FeatureCategory.TRADE_DATA, FeatureCategory.CUSTOM]:
                category_specs = [spec for spec in feature_specs if spec.category == category]
                if category_specs:
                    category_features = await self._extract_category_features(
                        data, category_specs, l2_data, trade_data
                    )
                    features_df = pd.concat([features_df, category_features], axis=1)
            
            # Calculate feature quality metrics
            quality_metrics = self._calculate_feature_quality(features_df)
            
            # Clean and validate features
            features_df = self._clean_features(features_df)
            
            extraction_time = time.time() - start_time
            self.extraction_stats['features_extracted'] += len(feature_specs)
            self.extraction_stats['extraction_time_total'] += extraction_time
            
            result = FeatureExtractionResult(
                features=features_df,
                feature_specs=feature_specs,
                extraction_time=extraction_time,
                quality_metrics=quality_metrics,
                cache_hits=self.extraction_stats['cache_hits'],
                cache_misses=self.extraction_stats['cache_misses']
            )
            
            if self.logger:
                self.logger.info(f"Advanced feature extraction completed in {extraction_time:.2f}s")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Advanced feature extraction failed: {e}")
            raise FeatureProcessingError(f"Advanced feature extraction failed: {e}")
    
    async def _extract_category_features(
        self, 
        data: pd.DataFrame, 
        category_specs: list[InternalFeatureSpec],
        l2_data: dict[str, Any] | None = None,
        trade_data: list[dict[str, Any]] | None = None
    ) -> pd.DataFrame:
        """Extract features for a specific category."""
        category_features = pd.DataFrame(index=data.index)
        
        for spec in category_specs:
            try:
                # Check cache first
                cache_key = self._generate_cache_key(spec, data)
                if spec.cache_enabled and cache_key in self.feature_cache:
                    feature_result = self.feature_cache[cache_key]
                    self.extraction_stats['cache_hits'] += 1
                else:
                    # Compute feature
                    feature_result = await self._compute_advanced_feature(spec, data, l2_data, trade_data)
                    
                    # Cache result if enabled
                    if spec.cache_enabled:
                        self.feature_cache[cache_key] = feature_result
                    
                    self.extraction_stats['cache_misses'] += 1
                
                # Add to category features
                if feature_result is not None and not feature_result.empty:
                    category_features = pd.concat([category_features, feature_result], axis=1)
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to compute feature {spec.key}: {e}")
                continue
        
        return category_features
    
    async def _compute_advanced_feature(
        self, 
        spec: InternalFeatureSpec, 
        data: pd.DataFrame,
        l2_data: dict[str, Any] | None = None,
        trade_data: list[dict[str, Any]] | None = None
    ) -> pd.DataFrame | None:
        """Compute a single advanced feature."""
        if spec.calculator_type in self.advanced_indicators:
            calculator_func = self.advanced_indicators[spec.calculator_type]
            
            # Prepare arguments based on feature type
            if spec.category == FeatureCategory.L2_ORDER_BOOK and l2_data:
                result = calculator_func(data, l2_data, **spec.parameters)
            elif spec.category == FeatureCategory.TRADE_DATA and trade_data:
                result = calculator_func(data, trade_data, **spec.parameters)
            else:
                result = calculator_func(data, **spec.parameters)
            
            # Process through output handler if available
            if hasattr(spec, 'output_specs') and spec.output_specs:
                handler = FeatureOutputHandler(spec)
                handler.logger = self.logger
                return handler.process_feature_outputs(result)
            
            # Convert result to DataFrame if needed
            if isinstance(result, pd.Series):
                return pd.DataFrame({spec.key: result})
            elif isinstance(result, pd.DataFrame):
                return result
            
        return None
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input OHLCV data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Check for sufficient data
        min_periods = max(20, max(spec.required_lookback_periods for spec in self.feature_registry.values() if hasattr(spec, 'required_lookback_periods')))
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: need at least {min_periods} periods, got {len(data)}")
    
    def _calculate_feature_quality(self, features_df: pd.DataFrame) -> dict[str, float]:
        """Calculate quality metrics for extracted features."""
        if features_df.empty:
            return {'completeness': 0.0, 'variance_ratio': 0.0, 'correlation_max': 0.0}
        
        # Completeness (non-null ratio)
        completeness = 1.0 - features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))
        
        # Variance ratio (features with sufficient variance)
        numeric_features = features_df.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            variances = numeric_features.var()
            high_variance_ratio = (variances > self.quality_thresholds['min_variance']).mean()
        else:
            high_variance_ratio = 0.0
        
        # Maximum correlation (feature redundancy check)
        max_correlation = 0.0
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            # Get upper triangle excluding diagonal
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            max_correlation = upper_triangle.max().max() if not upper_triangle.isna().all().all() else 0.0
        
        return {
            'completeness': float(completeness),
            'variance_ratio': float(high_variance_ratio),
            'correlation_max': float(max_correlation)
        }
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        # Remove features with too many NaNs
        threshold = self.quality_thresholds['min_completeness']
        features_df = features_df.dropna(thresh=int(len(features_df) * threshold), axis=1)
        
        # Remove highly correlated features
        numeric_features = features_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find features to drop
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > self.quality_thresholds['max_correlation'])]
            
            features_df = features_df.drop(columns=to_drop)
        
        return features_df
    
    def _generate_cache_key(self, spec: InternalFeatureSpec, data: pd.DataFrame) -> str:
        """Generate cache key for feature computation."""
        data_hash = str(hash(tuple(data.index.tolist() + data.iloc[-1].tolist())))
        params_hash = str(hash(tuple(sorted(spec.parameters.items()))))
        return f"{spec.key}_{data_hash}_{params_hash}"
    
    # Advanced indicator calculations
    def _calculate_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate momentum indicator."""
        return data['close'] - data['close'].shift(period)
    
    def _calculate_rate_of_change(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calculate rate of change."""
        return ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        return -100 * ((high_max - data['close']) / (high_max - low_min))
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def _calculate_bollinger_width(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.Series:
        """Calculate Bollinger Band width."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return (upper_band - lower_band) / sma * 100
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    def _calculate_atr_advanced(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Advanced Average True Range with additional smoothing."""
        tr = self._calculate_true_range(data)
        return tr.ewm(span=period).mean()  # Use exponential moving average
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame, short_period: int = 10, long_period: int = 30) -> pd.Series:
        """Calculate volatility ratio."""
        short_vol = data['close'].pct_change().rolling(window=short_period).std()
        long_vol = data['close'].pct_change().rolling(window=long_period).std()
        return short_vol / long_vol
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        price_change = data['close'].diff()
        obv = np.where(price_change > 0, data['volume'], 
                      np.where(price_change < 0, -data['volume'], 0))
        return pd.Series(obv, index=data.index).cumsum()
    
    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.fillna(0)  # Handle division by zero
        return (mfm * data['volume']).cumsum()
    
    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + money_ratio))
    
    def _calculate_volume_oscillator(self, data: pd.DataFrame, short_period: int = 14, long_period: int = 28) -> pd.Series:
        """Calculate Volume Oscillator."""
        short_vol_avg = data['volume'].rolling(window=short_period).mean()
        long_vol_avg = data['volume'].rolling(window=long_period).mean()
        return ((short_vol_avg - long_vol_avg) / long_vol_avg) * 100
    
    # Market microstructure features
    def _calculate_effective_spread(self, data: pd.DataFrame, l2_data: dict[str, Any]) -> pd.Series:
        """Calculate effective spread from L2 data."""
        # Implementation would use actual trade prices vs. midpoint
        # This is a simplified version
        if 'bids' in l2_data and 'asks' in l2_data and l2_data['bids'] and l2_data['asks']:
            bid_price = float(l2_data['bids'][0][0])
            ask_price = float(l2_data['asks'][0][0])
            midpoint = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            return pd.Series([spread / midpoint * 10000], index=[data.index[-1]])  # in basis points
        return pd.Series([], dtype=float)
    
    def _calculate_quoted_spread(self, data: pd.DataFrame, l2_data: dict[str, Any]) -> pd.Series:
        """Calculate quoted spread."""
        if 'bids' in l2_data and 'asks' in l2_data and l2_data['bids'] and l2_data['asks']:
            bid_price = float(l2_data['bids'][0][0])
            ask_price = float(l2_data['asks'][0][0])
            return pd.Series([ask_price - bid_price], index=[data.index[-1]])
        return pd.Series([], dtype=float)
    
    def _calculate_depth_imbalance(self, data: pd.DataFrame, l2_data: dict[str, Any], levels: int = 5) -> pd.Series:
        """Calculate order book depth imbalance."""
        if 'bids' in l2_data and 'asks' in l2_data:
            bid_depth = sum(float(level[1]) for level in l2_data['bids'][:levels])
            ask_depth = sum(float(level[1]) for level in l2_data['asks'][:levels])
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                imbalance = (bid_depth - ask_depth) / total_depth
                return pd.Series([imbalance], index=[data.index[-1]])
        return pd.Series([], dtype=float)
    
    def _calculate_order_flow_imbalance(self, data: pd.DataFrame, trade_data: list[dict[str, Any]]) -> pd.Series:
        """Calculate order flow imbalance from trade data."""
        if not trade_data:
            return pd.Series([], dtype=float)
        
        buy_volume = sum(float(trade['volume']) for trade in trade_data if trade.get('side') == 'buy')
        sell_volume = sum(float(trade['volume']) for trade in trade_data if trade.get('side') == 'sell')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            imbalance = (buy_volume - sell_volume) / total_volume
            return pd.Series([imbalance], index=[data.index[-1]])
        return pd.Series([], dtype=float)
    
    def _calculate_market_impact(self, data: pd.DataFrame, trade_data: list[dict[str, Any]]) -> pd.Series:
        """Calculate market impact estimate."""
        if not trade_data or len(data) < 2:
            return pd.Series([], dtype=float)
        
        # Simple market impact as price change per unit volume
        recent_volume = sum(float(trade['volume']) for trade in trade_data)
        price_change = float(data['close'].iloc[-1] - data['close'].iloc[-2])
        
        if recent_volume > 0:
            impact = abs(price_change) / recent_volume
            return pd.Series([impact], index=[data.index[-1]])
        return pd.Series([], dtype=float)
    
    # Statistical and advanced features
    def _calculate_pmo(self, data: pd.DataFrame, period1: int = 35, period2: int = 20) -> pd.Series:
        """Calculate Price Momentum Oscillator."""
        roc = data['close'].pct_change(period1) * 100
        pmo = roc.ewm(span=period2).mean()
        return pmo.ewm(span=10).mean()  # Signal line
    
    def _calculate_ama(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Adaptive Moving Average."""
        change = abs(data['close'] - data['close'].shift(period))
        volatility = data['close'].diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / volatility
        
        # Smoothing constants
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
        
        ama = pd.Series(index=data.index, dtype=float)
        ama.iloc[0] = data['close'].iloc[0]
        
        for i in range(1, len(data)):
            ama.iloc[i] = ama.iloc[i-1] + smoothing_constant.iloc[i] * (data['close'].iloc[i] - ama.iloc[i-1])
        
        return ama
    
    def _calculate_fractal_dimension(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate fractal dimension."""
        def fd_single(series):
            if len(series) < 2:
                return np.nan
            
            n = len(series)
            # Calculate relative range
            cumulative_deviations = np.cumsum(series - series.mean())
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            
            # Calculate standard deviation
            S = np.std(series)
            
            if S == 0:
                return np.nan
            
            # Hurst exponent
            rs = R / S
            hurst = np.log(rs) / np.log(n)
            
            # Fractal dimension
            return 2 - hurst
        
        return data['close'].rolling(window=period).apply(fd_single)
    
    def _calculate_hurst_exponent(self, data: pd.DataFrame, period: int = 100) -> pd.Series:
        """Calculate Hurst exponent."""
        def hurst_single(series):
            if len(series) < 10:
                return np.nan
            
            try:
                # Convert to log returns
                log_returns = np.log(series / series.shift(1)).dropna()
                
                # Calculate R/S statistics for different lags
                lags = range(2, min(len(log_returns) // 2, 20))
                rs_values = []
                
                for lag in lags:
                    # Split series into chunks
                    n_chunks = len(log_returns) // lag
                    if n_chunks < 1:
                        continue
                        
                    rs_chunk = []
                    for i in range(n_chunks):
                        chunk = log_returns[i*lag:(i+1)*lag]
                        if len(chunk) == lag:
                            mean_chunk = chunk.mean()
                            cumsum_chunk = (chunk - mean_chunk).cumsum()
                            r_chunk = cumsum_chunk.max() - cumsum_chunk.min()
                            s_chunk = chunk.std()
                            if s_chunk > 0:
                                rs_chunk.append(r_chunk / s_chunk)
                    
                    if rs_chunk:
                        rs_values.append(np.mean(rs_chunk))
                
                if len(rs_values) < 2:
                    return np.nan
                
                # Linear regression to find Hurst exponent
                log_lags = np.log(list(lags[:len(rs_values)]))
                log_rs = np.log(rs_values)
                
                # Simple linear regression
                n = len(log_lags)
                sum_x = np.sum(log_lags)
                sum_y = np.sum(log_rs)
                sum_xy = np.sum(log_lags * log_rs)
                sum_x2 = np.sum(log_lags ** 2)
                
                hurst = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                return hurst
                
            except Exception:
                return np.nan
        
        return data['close'].rolling(window=period).apply(hurst_single)


DEFAULT_FEATURE_REGISTRY_PATH = Path("config/feature_registry.yaml")

class PandasScalerTransformer:
    """A wrapper around sklearn scalers that preserves pandas Series/DataFrame structure.
    
    This transformer wraps any sklearn scaler (StandardScaler, MinMaxScaler, etc.)
    and ensures that the output maintains the same pandas structure as the input,
    including column names and index.
    """

    def __init__(self, scaler: Any) -> None:
        """Initialize with an sklearn scaler instance."""
        self.scaler = scaler
        self._feature_names: Any = None
        self._index: Any = None

    def fit(self, X: Any, y: Any = None) -> PandasScalerTransformer:
        """Fit the scaler and store structure information."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = X.columns
            self._index = X.index
            self.scaler.fit(X.values)
        elif isinstance(X, pd.Series):
            self._feature_names = X.name
            self._index = X.index
            self.scaler.fit(X.values.reshape(-1, 1))
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X: Any) -> Any:
        """Transform the data and restore pandas structure."""
        if isinstance(X, pd.DataFrame):
            transformed = self.scaler.transform(X.values)
            return pd.DataFrame(transformed, columns=self._feature_names or X.columns, index=X.index)
        if isinstance(X, pd.Series):
            transformed = self.scaler.transform(X.values.reshape(-1, 1))
            return pd.Series(transformed.flatten(), name=self._feature_names or X.name, index=X.index)
        return self.scaler.transform(X)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

class FeatureEngine:
    """Orchestrates feature calculation based on market data and configurations.

    The FeatureEngine loads canonical feature definitions from a YAML Feature Registry
    (path defined by `DEFAULT_FEATURE_REGISTRY_PATH`). It then uses the application-level
    configuration (passed during initialization, typically from `config.yaml`) to
    determine which features to activate and what customizations (overrides) to apply
    to their registry definitions. This activation and override mechanism is handled
    by the `_extract_feature_configs` method.

    For each activated feature (now represented as an `InternalFeatureSpec`), the
    `_build_feature_pipelines` method constructs a Scikit-learn `Pipeline`.
    This pipeline typically includes:
    1.  Input data selection and preparation (e.g., input imputation for close price series).
    2.  The core feature calculation logic, delegated to a static `_pipeline_compute_...`
        method corresponding to the feature's `calculator_type`. These methods are
        designed for robust NaN handling, aiming to provide sensible default values
        if a feature cannot be computed (e.g., RSI defaulting to 50.0).
    3.  An optional output imputation step, configured via the `imputation` field in
        the `InternalFeatureSpec`. This serves as a final fallback if a calculator,
        despite its internal NaN handling, still yields NaNs.
    4.  A centralized output scaling step, configured via the `scaling` field in the
        `InternalFeatureSpec`. This ensures that features are scaled (e.g., using
        StandardScaler or MinMaxScaler) before being published, making them ready for
        direct consumption by machine learning models.

    When new market data triggers `_calculate_and_publish_features`, these pipelines are
    executed. The resulting numerical feature values are collected into a dictionary.
    This dictionary is then validated against the `PublishedFeaturesV1` Pydantic model
    to ensure data integrity and adherence to the defined contract. If valid, the
    features are published as `pydantic_model.model_dump()` within a `FEATURES_CALCULATED`
    event on the PubSub system. Downstream services, like predictors, thus receive
    pre-scaled, validated, numerical features.
    """

    _EXPECTED_L2_LEVEL_LENGTH = 2

    def __init__(
        self,
        config: dict[str, Any],
        pubsub_manager: PubSubManager,
        logger_service: LoggerService,
        historical_data_service: HistoricalDataService | None = None,
    ) -> None:
        """Initialize the FeatureEngine with configuration and required services.

        Args:
            config: Overall application configuration dictionary. `FeatureEngine` uses
                `config['feature_engine']['features']` for feature activation and
                overrides, and `config['feature_engine']` for other settings.
                Feature definitions are primarily loaded from the YAML Feature Registry
                invoked by `_extract_feature_configs`.
            pubsub_manager: Instance of PubSubManager for event handling.
            logger_service: Logging service instance.
            historical_data_service: Optional service for fetching historical data.

        Internal State:
            _feature_configs: Stores `InternalFeatureSpec` objects for all activated
                features, populated by `_extract_feature_configs` after processing
                the registry and application-level overrides.
            feature_pipelines: Dictionary of Scikit-learn `Pipeline` objects for
                each feature, built by `_build_feature_pipelines`.
        """
        self.config = config
        self.pubsub_manager = pubsub_manager
        self.logger = logger_service
        self.historical_data_service = historical_data_service
        self._source_module = self.__class__.__name__

        # Feature configuration derived from config
        self._feature_configs: dict[str, InternalFeatureSpec] = {} # Changed type hint
        self._extract_feature_configs()

        # Initialize feature handlers dispatcher
        # RSI and MACD handlers are removed as they will be handled by sklearn pipelines.
        self._feature_handlers: dict[
            str,
            Callable[[dict[str, Any], dict[str, Any]], dict[str, str] | None],
        ] = {
            # "rsi": self._process_rsi_feature, # To be replaced by pipeline
            # "macd": self._process_macd_feature, # To be replaced by pipeline
            # "bbands": self._process_bbands_feature, # To be replaced by pipeline
            # "vwap": self._process_vwap_feature,  # To be replaced by pipeline
            # "roc": self._process_roc_feature, # To be replaced by pipeline
            # "atr": self._process_atr_feature, # To be replaced by pipeline
            # "stdev": self._process_stdev_feature, # To be replaced by pipeline
            # "spread": self._process_l2_spread_feature, # To be replaced by pipeline
            # "imbalance": self._process_l2_imbalance_feature, # To be replaced by pipeline
            # "wap": self._process_l2_wap_feature, # To be replaced by pipeline
            # "depth": self._process_l2_depth_feature, # To be replaced by pipeline
            # "volume_delta": self._process_volume_delta_feature, # To be replaced by pipeline
            # Note: vwap_ohlcv and vwap_trades are handled by _process_vwap_feature
        }

        # Initialize data storage
        # OHLCV data will be stored in a DataFrame per trading pair
        # Columns: timestamp_bar_start (index), open, high, low, close, volume
        self.ohlcv_history: dict[str, pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
            ).astype(
                {
                    "open": "object",  # Store as Decimal initially
                    "high": "object",
                    "low": "object",
                    "close": "object",
                    "volume": "object",
                },
            ),
        )

        # L2 order book data (latest snapshot)
        self.l2_books: dict[str, dict[str, Any]] = defaultdict(dict)

        # Store recent trades for calculating true Volume Delta and trade-based VWAP
        # deque stores: {"ts": datetime, "price": Decimal, "vol": Decimal, "side": "buy"/"sell"}
        trade_history_maxlen = config.get("feature_engine", {}).get("trade_history_maxlen", 2000)
        self.trade_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=trade_history_maxlen),
        )

        # Store L2 book history for better alignment with bar timestamps
        # Each entry: {"timestamp": datetime, "book": {"bids": [...], "asks": [...]}}
        l2_history_maxlen = config.get("feature_engine", {}).get("l2_history_maxlen", 100)
        self.l2_books_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=l2_history_maxlen),
        )

        self.feature_pipelines: dict[str, dict[str, Any]] = {} # Stores {'pipeline': Pipeline, 'spec': InternalFeatureSpec}
        
        # Initialize enhanced components
        self.output_handlers: dict[str, FeatureOutputHandler] = {}
        self.advanced_extractor = AdvancedFeatureExtractor(config.get("feature_engine", {}), logger_service)
        
        # Build pipelines after initializing components
        self._build_feature_pipelines()

        self.logger.info("FeatureEngine with enhanced capabilities initialized.", source_module=self._source_module)

    def _determine_calculator_type_and_input(self, feature_key: str, raw_cfg: dict) -> tuple[str | None, str | None]:
        """Determines calculator type and input type from feature key and raw config.

        This method attempts to infer the type of calculation (e.g., "rsi", "macd")
        and the type of input data required (e.g., "close_series", "ohlcv_df")
        primarily by inspecting the `feature_key` string. It also allows for explicit
        overrides if `calculator_type` (or `type`) and `input_type` are specified
        directly in the raw feature configuration dictionary.

        Args:
            feature_key: The original key for the feature from the configuration (e.g., "rsi_14").
            raw_cfg: The raw dictionary configuration for this specific feature.

        Returns:
            A tuple `(calculator_type, input_type)`, where both can be `None` if
            determination fails.
        """
        # Allow explicit 'type' in config to override key-based inference
        # And 'input_source_column' or 'input_source_type' for explicit input definition

        calc_type = raw_cfg.get("calculator_type", raw_cfg.get("type"))
        input_type = raw_cfg.get("input_type")

        if calc_type and input_type:
            return calc_type, input_type

        # Infer from key if not explicitly provided
        key_lower = feature_key.lower()
        if "rsi" in key_lower: return "rsi", "close_series"
        if "macd" in key_lower: return "macd", "close_series"
        if "bbands" in key_lower: return "bbands", "close_series"
        if "roc" in key_lower: return "roc", "close_series"
        if "atr" in key_lower: return "atr", "ohlcv_df"
        if "stdev" in key_lower: return "stdev", "close_series"
        if "vwap_ohlcv" in key_lower: return "vwap_ohlcv", "ohlcv_df"
        if "l2_spread" in key_lower: return "l2_spread", "l2_book_series"
        if "l2_imbalance" in key_lower: return "l2_imbalance", "l2_book_series"
        if "l2_wap" in key_lower: return "l2_wap", "l2_book_series"
        if "l2_depth" in key_lower: return "l2_depth", "l2_book_series"
        if "vwap_trades" in key_lower: return "vwap_trades", "trades_and_bar_starts"
        if "volume_delta" in key_lower: return "volume_delta", "trades_and_bar_starts"

        self.logger.warning("Could not determine calculator_type or input_type for feature key: %s", feature_key)
        return None, None


    def _extract_feature_configs(self) -> None:
        """Initializes `self._feature_configs` by loading feature definitions.

        This method orchestrates the loading of features by:
        1.  Calling `_load_feature_registry` to fetch all canonical feature definitions
            from the YAML file specified by `DEFAULT_FEATURE_REGISTRY_PATH`.
        2.  Retrieving the application-specific feature configuration from `self.config`
            (usually under the `feature_engine.features` key). This configuration
            determines which features are activated and how their registry definitions
            might be overridden.
        3.  Processing the application configuration:
            *   If it's a list of strings, these are treated as keys to activate
                features directly from the registry using their default settings.
            *   If it's a dictionary, each key-value pair is processed:
                *   The key is the feature name.
                *   If the feature name exists in the loaded registry definitions, the
                  value (a dictionary) is used to override the registry definition.
                  A deep merge (via `_deep_merge_configs`) is applied to handle
                  nested structures like `parameters`, `imputation`, and `scaling`.
                *   If the feature name is *not* in the registry, it's considered an
                  ad-hoc feature definition, defined entirely by the provided dictionary.
                  Ad-hoc definitions must at least specify `calculator_type` and `input_type`.
        4.  For each feature to be activated (whether from registry, overridden, or ad-hoc),
            its final configuration dictionary is passed to `_parse_single_feature_definition`.
        5.  The resulting `InternalFeatureSpec` objects are collected and stored in
            `self._feature_configs`.

        Warnings are logged for issues like missing registry keys, malformed configurations,
        or if no features are ultimately parsed and activated.
        """
        # Initialize variables properly
        raw_features_config = self.config.get("feature_engine", {}).get("features", {})
        parsed_specs: dict[str, InternalFeatureSpec] = {}

        if not isinstance(raw_features_config, dict):
            self.logger.warning(
                "Global 'features' configuration is not a dictionary. No features loaded.",
                source_module=self._source_module,
            )
            self._feature_configs = {}
            return

        for key, raw_cfg_dict in raw_features_config.items():
            if not isinstance(raw_cfg_dict, dict):
                self.logger.warning(
                    "Configuration for feature '%s' is not a dictionary. Skipping.",
                    key, source_module=self._source_module)
                continue

            calculator_type, input_type = self._determine_calculator_type_and_input(key, raw_cfg_dict)
            if not calculator_type or not input_type:
                self.logger.warning("Skipping feature %s due to undetermined type/input.", key)
                continue

            # Parameters for the _pipeline_compute function (e.g. period, length)
            # These are often nested under a 'params' or 'parameters' key in user config,
            # or could be at the top level of raw_cfg_dict.
            parameters = raw_cfg_dict.get("parameters", raw_cfg_dict.get("params", {}))
            # If common params like 'period' or 'length' are top-level, merge them in:
            for common_param_key in ["period", "length", "fast", "slow", "signal", "levels", "std_dev", "length_seconds", "bar_interval_seconds"]:
                if common_param_key in raw_cfg_dict and common_param_key not in parameters:
                    parameters[common_param_key] = raw_cfg_dict[common_param_key]


            imputation_cfg = raw_cfg_dict.get("imputation")
            scaling_cfg = raw_cfg_dict.get("scaling")
            description = raw_cfg_dict.get("description", f"{calculator_type} feature based on {key}")

            category_str = raw_cfg_dict.get("category", "TECHNICAL").upper()
            try:
                category = FeatureCategory[category_str]
            except KeyError:
                self.logger.warning(
                    "Invalid FeatureCategory '%s' for feature '%s'. Defaulting to TECHNICAL.",
                    category_str, key, source_module=self._source_module)
                category = FeatureCategory.TECHNICAL

            spec_val = InternalFeatureSpec(
                key=key,
                calculator_type=calculator_type,
                input_type=input_type,
                category=category,
                parameters=parameters,
                imputation=imputation_cfg,
                scaling=scaling_cfg,
                description=description,
            )
            parsed_specs[key] = spec_val

        # self._feature_configs = parsed_specs # Old logic replaced by new registry-based logic below

        registry_definitions = self._load_feature_registry(DEFAULT_FEATURE_REGISTRY_PATH)
        app_feature_config = self.config.get("features", {}) # This is the app-level config for features

        final_parsed_specs: dict[str, InternalFeatureSpec] = {}

        if isinstance(app_feature_config, list): # Case 1: List of feature keys to activate
            for key in app_feature_config:
                if not isinstance(key, str):
                    self.logger.warning("Feature activation list contains non-string item: %s. Skipping.", key)
                    continue
                if key not in registry_definitions:
                    self.logger.warning("Feature key '%s' from app config not found in registry. Skipping.", key)
                    continue

                feature_def_from_registry = registry_definitions[key]
                if not isinstance(feature_def_from_registry, dict):
                    self.logger.warning("Registry definition for '%s' is not a dictionary. Skipping.", key)
                    continue

                spec_result = self._parse_single_feature_definition(key, feature_def_from_registry.copy())
                if spec_result is not None:
                    final_parsed_specs[key] = spec_result

        elif isinstance(app_feature_config, dict): # Case 2: Dict of feature names with overrides or ad-hoc
            for key, overrides_or_activation in app_feature_config.items():
                if not isinstance(overrides_or_activation, dict):
                    self.logger.warning("Override/activation config for feature '%s' is not a dict. Skipping.", key)
                    continue

                base_config = registry_definitions.get(key)
                final_config_dict: dict[str, Any] = {}

                if base_config: # Key found in registry, apply overrides
                    if not isinstance(base_config, dict):
                        self.logger.warning("Registry definition for '%s' is not a dictionary. Skipping override.", key)
                        continue
                    final_config_dict = self._deep_merge_configs(base_config.copy(), overrides_or_activation)
                else: # Key not in registry - treat as ad-hoc definition
                    self.logger.info("Feature '%s' not found in registry, treating as ad-hoc definition from app config.", key)
                    final_config_dict = overrides_or_activation.copy()
                    # Ad-hoc definitions must provide all necessary fields like calculator_type, input_type
                    if "calculator_type" not in final_config_dict or "input_type" not in final_config_dict:
                        self.logger.warning("Ad-hoc feature '%s' missing 'calculator_type' or 'input_type'. Skipping.", key)
                        continue

                spec_result = self._parse_single_feature_definition(key, final_config_dict)
                if spec_result is not None:
                    final_parsed_specs[key] = spec_result

        else:
            self.logger.warning("App-level 'features' config is neither a list nor a dict. No features will be configured based on it.")

        self._feature_configs = final_parsed_specs
        if not self._feature_configs:
            self.logger.warning("No features were successfully parsed or activated. FeatureEngine might not produce any features.")


    def _deep_merge_configs(self, base: dict, override: dict) -> dict:
        """Deeply merges override dict into base dict.
        Recursively merges the `override` dictionary into the `base` dictionary.

        For keys present in both `base` and `override`:
        - If both values are dictionaries, they are merged recursively.
        - Otherwise, the value from `override` takes precedence.
        Keys present only in `override` are added to the merged dictionary.

        Args:
            base: The base configuration dictionary (e.g., from the feature registry).
            override: The dictionary with override values (e.g., from application config).

        Returns:
            A new dictionary representing the deeply merged configuration.
        """
        merged = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _parse_single_feature_definition(self, feature_key: str, config_dict: dict) -> InternalFeatureSpec | None:
        """Parses a single, consolidated feature configuration dictionary into an
        `InternalFeatureSpec` data object.

        This method is responsible for taking the final configuration dictionary for a
        feature (which may have resulted from merging registry definitions with
        application-level overrides, or could be an ad-hoc definition) and
        translating it into a structured `InternalFeatureSpec`.

        It performs the following steps:
        1.  Determines `calculator_type` and `input_type`:
            - Prefers explicitly defined values in `config_dict`.
            - Falls back to inferring them using `_determine_calculator_type_and_input`
              if not explicit (useful for concise ad-hoc definitions).
        2.  Extracts `parameters`, ensuring it's a dictionary and merging any top-level
            common parameter keys (like 'period', 'length') into it if they are not
            already present under the `parameters` key itself.
        3.  Extracts `imputation`, `scaling`, `description`, `version`, and
            `output_properties` from `config_dict`, applying defaults where necessary.
        4.  Parses the `category` string into a `FeatureCategory` enum member.
        5.  Instantiates and returns the `InternalFeatureSpec`.

        Args:
            feature_key: The unique key for the feature (used for logging and as `spec.key`).
            config_dict: The complete configuration dictionary for this single feature.

        Returns:
            An `InternalFeatureSpec` instance if parsing is successful and essential
            fields like `calculator_type` and `input_type` can be determined.
            Returns `None` if critical information is missing, with errors logged.
        """
        # calculator_type and input_type can be inferred if not explicit (legacy/ad-hoc)
        # For registry-defined features, these should ideally be explicit.
        calculator_type = config_dict.get("calculator_type", config_dict.get("type"))
        input_type = config_dict.get("input_type")

        if not calculator_type or not input_type:
            # Try to infer if they are missing (e.g. for ad-hoc definitions)
            inferred_calc_type, inferred_input_type = self._determine_calculator_type_and_input(feature_key, config_dict)
            if not calculator_type: calculator_type = inferred_calc_type
            if not input_type: input_type = inferred_input_type

            if not calculator_type or not input_type:
                self.logger.warning(
                    "Could not determine calculator_type or input_type for feature '%s' even after inference. Skipping.",
                    feature_key,
                )
                return None

        parameters = config_dict.get("parameters", config_dict.get("params", {}))
        # Ensure parameters is a dict, even if it was null/None in YAML
        if not isinstance(parameters, dict): parameters = {}

        # Merge top-level common param keys if they exist and aren't already in 'parameters'
        for common_param_key in ["period", "length", "fast", "slow", "signal", "levels", "std_dev", "length_seconds", "bar_interval_seconds"]:
            if common_param_key in config_dict and common_param_key not in parameters:
                parameters[common_param_key] = config_dict[common_param_key]

        imputation_cfg = config_dict.get("imputation")
        scaling_cfg = config_dict.get("scaling")
        description = config_dict.get("description", f"{calculator_type} feature based on {feature_key}")
        version = config_dict.get("version")
        output_properties = config_dict.get("output_properties", {})

        category_str = str(config_dict.get("category", "TECHNICAL")).upper() # Ensure string before upper()
        try:
            category = FeatureCategory[category_str]
        except KeyError:
            self.logger.warning(
                "Invalid FeatureCategory '%s' for feature '%s'. Defaulting to TECHNICAL.",
                category_str, feature_key, source_module=self._source_module)
            category = FeatureCategory.TECHNICAL

        spec = InternalFeatureSpec(
            key=feature_key,
            calculator_type=calculator_type,
            input_type=input_type,
            category=category,
            parameters=parameters,
            imputation=imputation_cfg,
            scaling=scaling_cfg,
            description=description,
            version=str(version) if version is not None else None, # Ensure version is string
            output_properties=output_properties if isinstance(output_properties, dict) else {},
        )
        self.logger.info("Successfully parsed feature spec for key: '%s' (Calc: %s, Input: %s)",
                         feature_key, calculator_type, input_type)
        return spec


    def _load_feature_registry(self, registry_path: Path) -> dict[str, Any]:
        """Loads feature definitions from the specified YAML feature registry file.

        Args:
            registry_path: `Path` object pointing to the YAML feature registry file.

        Returns:
            A dictionary where keys are feature names and values are their
            definition dictionaries as loaded from the registry.
            Returns an empty dictionary if the file is not found, cannot be parsed,
            or does not conform to the expected dictionary structure.
        """
        if not registry_path.exists():
            self.logger.error(f"Feature registry file not found: {registry_path}")
            return {}

        try:
            with registry_path.open("r") as f:
                registry_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.logger.exception(f"Error parsing YAML in feature registry {registry_path}: {e}")
            return {}
        except Exception as e:
            self.logger.exception(f"Unexpected error loading feature registry {registry_path}: {e}")
            return {}

        if not isinstance(registry_data, dict):
            self.logger.error(f"Feature registry {registry_path} content is not a dictionary.")
            return {}

        self.logger.info(f"Successfully loaded {len(registry_data)} feature definitions from {registry_path}.")
        return registry_data

    def _build_feature_pipelines(self) -> None:
        """Constructs Scikit-learn pipelines for each feature defined in `self._feature_configs`.

        For each `InternalFeatureSpec`:
        1.  An input imputer might be added (e.g., for 'close_series' inputs).
        2.  A `FunctionTransformer` is created for the core calculation logic,
            mapping to the relevant `_pipeline_compute_{calculator_type}` static method.
            Static parameters from `spec.parameters` are passed as `kw_args`.
        3.  An output imputer step (handling NaNs from the calculator) is added based
            on `spec.imputation` configuration. This is a final fallback, as calculators
            themselves aim to prevent NaNs.
        4.  An output scaler step (e.g., StandardScaler, MinMaxScaler) is added based
            on `spec.scaling` configuration. This ensures features are scaled before publishing.
        5.  The resulting pipeline is set to output pandas objects and stored in
            `self.feature_pipelines`.

        This method centralizes the application of imputation and scaling post-calculation,
        ensuring consistency and adherence to the feature's defined processing steps.
        """
        self.logger.info("Building feature pipelines...", source_module=self._source_module)

        # Helper for output imputation step
        def get_output_imputer_step(imputation_cfg: dict[str, Any] | str | None,
                                    default_fill_value: float = 0.0,
                                    is_dataframe_output: bool = False,
                                    spec_key: str = "") -> tuple[str, FunctionTransformer] | None:
            """Creates a Scikit-learn compatible imputation step based on configuration.
            Handles Series and DataFrame outputs from feature calculators.
            """
            if imputation_cfg == "passthrough":
                self.logger.debug("Imputation set to 'passthrough' for %s.", spec_key)
                return None

            strategy = "default" # Internal default if cfg is None or invalid
            fill_value = default_fill_value

            if isinstance(imputation_cfg, dict):
                strategy = imputation_cfg.get("strategy", "constant")
                if strategy == "constant":
                    fill_value = imputation_cfg.get("fill_value", default_fill_value)
            elif imputation_cfg is None: # Use provided default_fill_value directly
                 pass # strategy remains 'default' -> use default_fill_value
            else: # Invalid config, treat as passthrough or log warning
                self.logger.warning("Unrecognized imputation config for %s: %s. No imputer added.", spec_key, imputation_cfg)
                return None

            step_name_suffix = ""
            transform_func: Callable[[Any], Any] | None = None

            if strategy == "constant":
                step_name_suffix = f"const_{fill_value}"
                transform_func = lambda x: x.fillna(fill_value)
            elif strategy == "mean":
                step_name_suffix = "mean"
                transform_func = lambda x: x.fillna(x.mean())
            elif strategy == "median":
                step_name_suffix = "median"
                transform_func = lambda x: x.fillna(x.median())
            elif strategy == "default": # Use the passed default_fill_value
                step_name_suffix = f"default_fill_{default_fill_value}"
                transform_func = lambda x: x.fillna(default_fill_value)
            else: # Should not be reached if checks are exhaustive
                self.logger.warning("Unknown imputation strategy '%s' for %s. No imputer added.", strategy, spec_key)
                return None

            self.logger.debug("Using output imputer strategy '%s' (fill: %s) for %s", strategy if strategy != "default" else f"default_fill({default_fill_value})", fill_value if strategy == "constant" else "N/A", spec_key)
            return (f"{spec_key}_output_imputer_{step_name_suffix}", FunctionTransformer(transform_func, validate=False))


        # Helper for output scaler step
        def get_output_scaler_step(scaling_cfg: dict[str, Any] | str | None,
                                   spec_key: str = "") -> tuple[str, PandasScalerTransformer] | None:
            """Creates a Scikit-learn compatible scaling step based on the feature's `scaling` configuration.
            This step is managed by the FeatureEngine to make features ready for consumption,
            potentially by models that expect scaled data.
            Uses PandasScalerTransformer to preserve pandas Series/DataFrame structure.
            """
            if scaling_cfg == "passthrough" or scaling_cfg is None:
                if scaling_cfg == "passthrough": self.logger.debug("Scaling set to 'passthrough' for %s.", spec_key)
                else: self.logger.debug("No scaling configured or default 'None' for %s.", spec_key)
                return None

            scaler_instance: StandardScaler | MinMaxScaler | RobustScaler = StandardScaler() # Default scaler
            scaler_name_suffix = "StandardScaler"

            if isinstance(scaling_cfg, dict):
                method = scaling_cfg.get("method", "standard")
                if method == "minmax":
                    scaler_instance = MinMaxScaler(feature_range=scaling_cfg.get("feature_range", (0,1)))
                    scaler_name_suffix = f"MinMaxScaler_{scaler_instance.feature_range}"
                elif method == "robust":
                    scaler_instance = RobustScaler(quantile_range=scaling_cfg.get("quantile_range", (25.0, 75.0)))
                    scaler_name_suffix = f"RobustScaler_{scaler_instance.quantile_range}"
                elif method != "standard":
                    self.logger.warning("Unknown scaling method '%s' for %s. Using StandardScaler.", method, spec_key)
            elif isinstance(scaling_cfg, str) and scaling_cfg not in ["standard", "passthrough"]: # e.g. just "minmax"
                 self.logger.warning("Simple string for scaling method '%s' for %s is ambiguous. Use dict config or 'passthrough'. Defaulting to StandardScaler.", scaling_cfg, spec_key)


            self.logger.debug("Using %s for scaling for %s", type(scaler_instance).__name__, spec_key)
            return (f"{spec_key}_output_scaler_{scaler_name_suffix}", PandasScalerTransformer(scaler_instance))


        for feature_key, spec in self._feature_configs.items(): # Now iterates over InternalFeatureSpec
            pipeline_steps = []
            pipeline_name = f"{spec.key}_pipeline" # Use spec.key for consistency
            
            # Create output handler for this feature
            output_handler = FeatureOutputHandler(spec)
            output_handler.logger = self.logger
            self.output_handlers[spec.key] = output_handler

            # Input imputer for features that take a single series like 'close'
            if spec.input_type == "close_series":
                # TODO: Make input imputer strategy configurable if needed from global or feature spec
                self.logger.debug("Adding standard input imputer (mean) for %s", spec.key)
                pipeline_steps.append((f"{spec.key}_input_imputer", SimpleImputer(strategy="mean")))

            # Calculator step based on spec.calculator_type
            calculator_func = getattr(FeatureEngine, f"_pipeline_compute_{spec.calculator_type}", None)
            if not calculator_func:
                self.logger.error("No _pipeline_compute function found for calculator_type: %s (feature key: %s)", spec.calculator_type, spec.key)
                continue

            # Prepare kw_args for the calculator from spec.parameters
            # Ensure all necessary parameters for the specific calculator are present with defaults
            calc_kw_args = {} # These are static kw_args known at pipeline build time

            if spec.calculator_type in ["rsi", "roc", "stdev"]:
                default_period = 14 if spec.calculator_type == "rsi" else 10 if spec.calculator_type == "roc" else 20
                calc_kw_args["period"] = spec.parameters.get("period", default_period)
                if spec.parameters.get("period") is None: self.logger.debug("Using default period %s for %s ('%s')", calc_kw_args["period"], spec.calculator_type, spec.key)

            elif spec.calculator_type == "macd":
                calc_kw_args["fast"] = spec.parameters.get("fast", 12)
                calc_kw_args["slow"] = spec.parameters.get("slow", 26)
                calc_kw_args["signal"] = spec.parameters.get("signal", 9)
                if any(p not in spec.parameters for p in ["fast", "slow", "signal"]):
                    self.logger.debug("Using default MACD params (f:%s,s:%s,sig:%s) for %s", calc_kw_args["fast"], calc_kw_args["slow"], calc_kw_args["signal"], spec.key)

            elif spec.calculator_type == "bbands":
                calc_kw_args["length"] = spec.parameters.get("length", 20)
                calc_kw_args["std_dev"] = float(spec.parameters.get("std_dev", 2.0))
                if "length" not in spec.parameters or "std_dev" not in spec.parameters:
                     self.logger.debug("Using default BBands params (l:%s,s:%.1f) for %s", calc_kw_args["length"], calc_kw_args["std_dev"], spec.key)

            elif spec.calculator_type == "atr":
                calc_kw_args["length"] = spec.parameters.get("length", 14)
                # high_col, low_col, close_col default in function signature of _pipeline_compute_atr
                if "length" not in spec.parameters: self.logger.debug("Using default ATR length %s for %s", calc_kw_args["length"], spec.key)

            elif spec.calculator_type == "vwap_ohlcv":
                calc_kw_args["length"] = spec.parameters.get("length", 14)
                if "length" not in spec.parameters: self.logger.debug("Using default VWAP_OHLCV length %s for %s", calc_kw_args["length"], spec.key)

            elif spec.calculator_type in ["l2_imbalance", "l2_depth", "l2_wap"]:
                # `ohlcv_close_prices` is NOT included here; it's passed dynamically for l2_wap.
                default_levels = 5
                if spec.calculator_type == "l2_wap": default_levels = 1
                elif spec.calculator_type == "l2_spread": default_levels = 0 # Not applicable for spread

                if default_levels > 0: # Only add 'levels' if applicable
                    calc_kw_args["levels"] = spec.parameters.get("levels", default_levels)
                    if "levels" not in spec.parameters: self.logger.debug("Using default levels %s for %s ('%s')", calc_kw_args["levels"], spec.calculator_type, spec.key)

            elif spec.calculator_type == "vwap_trades" or spec.calculator_type == "volume_delta":
                # `ohlcv_close_prices` is NOT included here; it's passed dynamically.
                # `bar_start_times` is also dynamic, passed at runtime.
                # `trade_history_deque` is the `X` input to fit_transform.
                calc_kw_args["bar_interval_seconds"] = spec.parameters.get("bar_interval_seconds",
                                                                         spec.parameters.get("length_seconds", 60))
                if "bar_interval_seconds" not in spec.parameters and "length_seconds" not in spec.parameters:
                    self.logger.debug("Using default bar_interval_seconds %s for %s ('%s')", calc_kw_args["bar_interval_seconds"], spec.calculator_type, spec.key)
                # `bar_start_times` will be passed dynamically during the call in _calculate_and_publish_features

            # l2_spread currently has no parameters in its _pipeline_compute_l2_spread signature other than X.

            pipeline_steps.append((f"{spec.key}_calculator", FunctionTransformer(calculator_func, kw_args=calc_kw_args, validate=False)))

            # Output Imputation & Scaling using helpers
            is_df_output = spec.calculator_type in ["macd", "bbands", "l2_spread", "l2_depth"]
            # Define default fill values based on feature type characteristics
            default_fill = 0.0 # General default
            if spec.calculator_type == "rsi": default_fill = 50.0
            elif spec.calculator_type in ["atr", "vwap_ohlcv", "l2_wap", "vwap_trades", "stdev"]: default_fill = np.nan # Will be filled by mean then

            imputer_step = get_output_imputer_step(spec.imputation, default_fill_value=default_fill, is_dataframe_output=is_df_output, spec_key=spec.key)
            if imputer_step: pipeline_steps.append(imputer_step)

            scaler_step = get_output_scaler_step(spec.scaling, spec_key=spec.key)
            if scaler_step: pipeline_steps.append(scaler_step)

            if pipeline_steps:
                final_pipeline = Pipeline(steps=pipeline_steps)
                final_pipeline.set_output(transform="pandas") # Ensure pandas output
                self.feature_pipelines[pipeline_name] = {
                    "pipeline": final_pipeline,
                    "input_type": spec.input_type,
                    "params": spec.parameters, # Storing parsed parameters
                    "spec": spec, # Store the full spec for richer context if needed later
                }
                self.logger.info(
                    "Built pipeline: %s with steps: %s, input: %s",
                    pipeline_name, [s[0] for s in pipeline_steps], spec.input_type,
                    source_module=self._source_module,
                )

    def _handle_ohlcv_update(self, trading_pair: str, ohlcv_payload: dict[str, Any]) -> None:
        """Parse and store an OHLCV update."""
        try:
            # Extract and convert data from payload
            # Timestamp parsing (ISO 8601 string to datetime object)
            # According to inter_module_comm.md: payload.timestamp_bar_start (ISO 8601)
            timestamp_str = ohlcv_payload.get("timestamp_bar_start")
            if not timestamp_str:
                self.logger.warning(
                    "Missing 'timestamp_bar_start' in OHLCV payload for %s",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": ohlcv_payload},
                )
                return

            # Convert to datetime. PANDAS will handle timezone if present in string
            # Forcing UTC if not specified, adjust if local timezone is expected/preferred
            bar_timestamp = pd.to_datetime(timestamp_str, utc=True)

            # Price/Volume conversion (string to Decimal for precision)
            open_price = Decimal(ohlcv_payload["open"])
            high_price = Decimal(ohlcv_payload["high"])
            low_price = Decimal(ohlcv_payload["low"])
            close_price = Decimal(ohlcv_payload["close"])
            volume = Decimal(ohlcv_payload["volume"])

            # Prepare new row as a dictionary
            new_bar_data = {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }

            # Get the DataFrame for the trading pair
            df = self.ohlcv_history[trading_pair]

            # Create a new DataFrame for the new row with the correct index
            new_row_df = pd.DataFrame([new_bar_data], index=[bar_timestamp])
            new_row_df.index.name = "timestamp_bar_start"

            # Ensure new_row_df columns match df columns and types are compatible
            # This is important if df was empty or had different types initially
            for col in df.columns:
                if col not in new_row_df:
                    new_row_df[col] = pd.NA  # Or appropriate default
            new_row_df = new_row_df[df.columns]  # Ensure column order

            # If df is empty, new_row_df types might not match self.ohlcv_history default astype.
            # Re-apply astype or ensure compatible types for concat.
            # Assuming concat handles type promotion or columns are compatible.

            # Append new data
            # Check if timestamp already exists to avoid duplicates, update if it does
            if bar_timestamp not in df.index:
                df = pd.concat([df, new_row_df])
            else:
                # Update existing row
                df.loc[bar_timestamp] = new_row_df.iloc[0]
                self.logger.debug(
                    "Updated existing OHLCV bar for %s at %s",
                    trading_pair,
                    bar_timestamp,
                    source_module=self._source_module,
                )

            # Sort by timestamp (index)
            df.sort_index(inplace=True)

            # Prune old data - keep a bit more than strictly required for safety margin
            min_hist = self._get_min_history_required()
            required_length = min_hist + 50  # Keep 50 extra bars as buffer
            if len(df) > required_length:
                df = df.iloc[-required_length:]

            self.ohlcv_history[trading_pair] = df
            self.logger.debug(
                "Processed OHLCV update for %s. History size: %s",
                trading_pair,
                len(df),
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in OHLCV payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in OHLCV payload for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling OHLCV update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload},
            )

    def _handle_l2_update(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Parse and store an L2 order book update."""
        try:
            # Extract bids and asks. Ensure they are lists of lists/tuples as expected.
            # inter_module_comm.md: bids/asks: List of lists [[price_str, volume_str], ...]
            raw_bids = l2_payload.get("bids")
            raw_asks = l2_payload.get("asks")

            if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
                self.logger.warning(
                    "L2 bids/asks are not lists for %s. Payload: %s",
                    trading_pair,
                    l2_payload,
                    source_module=self._source_module,
                )
                # Decide if we should clear the book or keep stale data.
                # For now, we'll just return and not update if format is wrong.
                return

            # Convert price/volume strings to Decimal for precision
            # Bids: List of [Decimal(price), Decimal(volume)]
            # Asks: List of [Decimal(price), Decimal(volume)]
            # We expect bids to be sorted highest first, asks lowest first as per doc.

            processed_bids = []
            for i, bid_level in enumerate(raw_bids):
                if (
                    isinstance(bid_level, list | tuple)
                    and len(bid_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_bids.append([Decimal(bid_level[0]), Decimal(bid_level[1])])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 bid level %s for %s: %s - %s",
                            i,
                            trading_pair,
                            bid_level,
                            e,
                            source_module=self._source_module,
                        )
                        # Optionally skip this level or use NaN/None
                        continue  # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 bid level %s for %s: %s",
                        i,
                        trading_pair,
                        bid_level,
                        source_module=self._source_module,
                    )

            processed_asks = []
            for i, ask_level in enumerate(raw_asks):
                if (
                    isinstance(ask_level, list | tuple)
                    and len(ask_level) == self._EXPECTED_L2_LEVEL_LENGTH
                ):
                    try:
                        processed_asks.append([Decimal(ask_level[0]), Decimal(ask_level[1])])
                    except (ValueError, TypeError) as e:
                        self.logger.warning(
                            "Error converting L2 ask level %s for %s: %s - %s",
                            i,
                            trading_pair,
                            ask_level,
                            e,
                            source_module=self._source_module,
                        )
                        continue  # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 ask level %s for %s: %s",
                        i,
                        trading_pair,
                        ask_level,
                        source_module=self._source_module,
                    )

            # Store the processed L2 book data
            # The L2 book features will expect bids sorted high to low, asks low to high.
            self.l2_books[trading_pair] = {
                "bids": processed_bids,  # Already sorted highest bid first from source
                "asks": processed_asks,  # Already sorted lowest ask first from source
                "timestamp": pd.to_datetime(
                    l2_payload.get("timestamp_exchange") or datetime.utcnow(),
                    utc=True,
                ),
            }
            
            # Store in L2 history with timestamp for better time alignment
            timestamp_str = l2_payload.get("timestamp_exchange") or l2_payload.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = pd.to_datetime(timestamp_str, utc=True)
                    self.l2_books_history[trading_pair].append({
                        "timestamp": timestamp,
                        "book": {"bids": processed_bids, "asks": processed_asks}
                    })
                except Exception as e:
                    self.logger.warning(
                        "Failed to parse L2 timestamp for history: %s",
                        e,
                        source_module=self._source_module,
                    )

            self.logger.debug(
                "Updated L2 book for %s: %s bids, %s asks",
                trading_pair,
                len(processed_bids),
                len(processed_asks),
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in L2 payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling L2 update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload},
            )

    async def _handle_trade_event(self, event_dict: dict[str, Any]) -> None:
        """Handle incoming raw trade events and store them."""
        # This method will be called by pubsub, so it takes the full event_dict
        payload = event_dict.get("payload")
        if not payload:
            self.logger.warning(
                "Trade event missing payload.",
                context=event_dict,
                source_module=self._source_module,
            )
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Trade event payload missing trading_pair.",
                context=payload,
                source_module=self._source_module,
            )
            return

        try:
            trade_timestamp_str = payload.get("timestamp_exchange")
            price_str = payload.get("price")
            volume_str = payload.get("volume")
            side = payload.get("side")  # "buy" or "sell"

            if not all([trade_timestamp_str, price_str, volume_str, side]):
                self.logger.warning(
                    "Trade event for %s is missing required fields "
                    "(timestamp, price, volume, or side).",
                    trading_pair,
                    context=payload,
                    source_module=self._source_module,
                )
                return

            trade_data = {
                "timestamp": pd.to_datetime(trade_timestamp_str, utc=True),
                "price": Decimal(price_str),
                "volume": Decimal(volume_str),
                "side": side.lower(),  # Ensure lowercase for consistency
            }

            if trade_data["side"] not in ["buy", "sell"]:
                self.logger.warning(
                    "Invalid trade side '%s' for %s.",
                    trade_data["side"],
                    trading_pair,
                    context=payload,
                    source_module=self._source_module,
                )
                return

            self.trade_history[trading_pair].append(trade_data)
            self.logger.debug(
                "Stored trade for %s: P=%s, V=%s, Side=%s",
                trading_pair,
                trade_data["price"],
                trade_data["volume"],
                trade_data["side"],
                source_module=self._source_module,
            )

        except KeyError:
            self.logger.exception(
                "Missing key in trade event payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in trade event payload for %s",
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )
        except Exception:
            self.logger.exception(
                "Unexpected error handling trade event for %s",
                trading_pair,
                source_module=self._source_module,
                context=payload,
            )

    def _get_min_history_required(self) -> int:
        """Determine the minimum required history size for TA calculations.
        This function relies on accessing period/length parameters from feature configurations.
        The addition of 'imputation' and 'scaling' keys at the same level in the
        configuration structure does not affect its operation.
        """
        min_size = 1  # Minimum baseline

        # Check various indicator requirements based on the InternalFeatureSpec objects
        periods = []
        for spec in self._feature_configs.values():
            if spec.calculator_type in ["rsi", "roc"]:
                period = spec.parameters.get("period", 14 if spec.calculator_type == "rsi" else 10)
                periods.append(period)
            elif spec.calculator_type in ["bbands", "vwap_ohlcv", "atr", "stdev"]:
                length = spec.parameters.get("length", 20 if spec.calculator_type == "bbands" else 14)
                periods.append(length)

        if periods:
            min_size = max(periods) * 3  # Multiply by 3 for a safe margin

        return max(100, min_size)  # At least 100 bars for good measure

    def _get_period_from_config(
        self,
        feature_name: str,
        field_name: str,
        default_value: int,
    ) -> int:
        """Retrieve the period from config for a specific feature.
        This function relies on accessing specific period/length parameters within a
        feature's configuration dictionary. The addition of 'imputation' and 'scaling'
        keys at the same level does not affect its ability to retrieve these parameters.
        """
        feature_spec = self._feature_configs.get(feature_name)
        if feature_spec and isinstance(feature_spec, InternalFeatureSpec):
            period_value = feature_spec.parameters.get(field_name, default_value)
            return (
                period_value if isinstance(period_value, int) and period_value > 0 else default_value
            )
        # If feature_spec is not found or not the right type
        self.logger.warning(
            "Configuration for feature '%s' not found or invalid when trying to get '%s'. "
            "Returning default value %s.",
            feature_name,
            field_name,
            default_value,
            source_module=self._source_module,
        )
        return default_value

    async def start(self) -> None:
        """Start the feature engine and subscribe to relevant events."""
        try:
            # Subscribe process_market_data to handle both OHLCV and L2 updates
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_OHLCV,
                self.process_market_data,
            )
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_L2,
                self.process_market_data,
            )
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_TRADE,
                self._handle_trade_event,  # New subscription
            )
            self.logger.info(
                "FeatureEngine started and subscribed to MARKET_DATA_OHLCV, "
                "MARKET_DATA_L2, and MARKET_DATA_TRADE events.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine start and subscription",
                source_module=self._source_module,
            )
            # Depending on desired behavior, might re-raise or handle to prevent full stop

    async def stop(self) -> None:
        """Stop the feature engine and clean up resources."""
        try:
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_OHLCV,
                self.process_market_data,
            )
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_L2,
                self.process_market_data,
            )
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_TRADE,
                self._handle_trade_event,  # New unsubscription
            )
            self.logger.info(
                "FeatureEngine stopped and unsubscribed from market data events.",
                source_module=self._source_module,
            )
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine stop and unsubscription",
                source_module=self._source_module,
            )

    async def process_market_data(self, market_data_event_dict: dict[str, Any]) -> None:
        """Process market data to generate features.

        Args:
        ----
            market_data_event_dict: Market data event dictionary
        """
        # Assuming market_data_event_dict is the full event object including
        # event_type, payload, source_module, etc. as per inter_module_comm.md

        event_type = market_data_event_dict.get("event_type")
        payload = market_data_event_dict.get("payload")
        source_module = market_data_event_dict.get("source_module")  # For logging/context

        if not event_type or not payload:
            self.logger.warning(
                "Received market data event with missing event_type or payload.",
                source_module=self._source_module,  # Log from FeatureEngine itself
                context={"original_event": market_data_event_dict},
            )
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Market data event (type: %s) missing trading_pair.",
                event_type,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )
            return

        self.logger.debug(
            "Processing event %s for %s from %s",
            event_type,
            trading_pair,
            source_module,
            source_module=self._source_module,
        )

        if event_type == "MARKET_DATA_OHLCV":
            self._handle_ohlcv_update(trading_pair, payload)
            # OHLCV update is the trigger for calculating all features for that bar's timestamp
            timestamp_bar_start = payload.get("timestamp_bar_start")
            if timestamp_bar_start:
                # We'll define _calculate_and_publish_features as async shortly
                await self._calculate_and_publish_features(trading_pair, timestamp_bar_start)
            else:
                self.logger.warning(
                    "OHLCV event for %s missing 'timestamp_bar_start', "
                    "cannot calculate features.",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": payload},
                )
        elif event_type == "MARKET_DATA_L2":
            self._handle_l2_update(trading_pair, payload)
            # L2 updates typically don't trigger a full feature calculation on their own
            # in this design, as features are aligned with OHLCV bar closures.
            # L2 data is stored and used when an OHLCV bar triggers calculation.
        elif event_type == "MARKET_DATA_TRADE":
            await self._handle_trade_event(market_data_event_dict)
        else:
            self.logger.warning(
                "Received unknown market data event type: %s for %s",
                event_type,
                trading_pair,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict},
            )

    # --- Pipeline-compatible feature calculation methods ---
    # These methods are designed to be used within Scikit-learn FunctionTransformers.
    # They expect float64 inputs and produce float64 outputs (pd.Series or pd.DataFrame).

    @staticmethod
    def _pipeline_compute_rsi(data: pd.Series, period: int) -> pd.Series:
        """Compute RSI using pandas-ta, expecting float64 Series input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            # self.logger.error("_pipeline_compute_rsi expects a pd.Series.") # Logger not available in static method
            return pd.Series(dtype="float64", name=f"rsi_{period}") # Return empty named series on error

        rsi_series = data.ta.rsi(length=period)
        # Fill NaNs (typically at the beginning) with a neutral RSI value.
        rsi_series = rsi_series.fillna(50.0)
        rsi_series.name = f"rsi_{period}"
        return rsi_series.astype("float64")

    @staticmethod
    def _pipeline_compute_macd(
        data: pd.Series,
        fast: int,
        slow: int,
        signal: int,
    ) -> pd.DataFrame:
        """Compute MACD using pandas-ta, expecting float64 Series input.
        Returns a DataFrame with MACD, histogram, and signal lines.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            # self.logger.error("_pipeline_compute_macd expects a pd.Series.")
            return pd.DataFrame(dtype="float64") # Return empty DataFrame on error
        # pandas-ta returns MACD, MACDh (histogram), MACDs (signal)
        macd_df = data.ta.macd(fast=fast, slow=slow, signal=signal)
        # Fill NaNs with 0.0 for all MACD related columns
        if macd_df is not None:
            macd_df = macd_df.fillna(0.0)
        return macd_df.astype("float64") if macd_df is not None else pd.DataFrame(dtype="float64")

    @staticmethod
    def _fillna_bbands(bbands_df: pd.DataFrame, close_prices: pd.Series) -> pd.DataFrame:
        """Helper to fill NaNs in Bollinger Bands results.
        Middle band NaN is filled with close price. Lower/Upper NaNs also with close price.
        """
        if bbands_df is None:
            return pd.DataFrame(dtype="float64")

        # Identify columns by common suffixes, as exact names vary with params
        middle_col = next((col for col in bbands_df.columns if col.startswith("BBM_")), None)
        lower_col = next((col for col in bbands_df.columns if col.startswith("BBL_")), None)
        upper_col = next((col for col in bbands_df.columns if col.startswith("BBU_")), None)

        if middle_col:
            bbands_df[middle_col] = bbands_df[middle_col].fillna(close_prices)
        if lower_col:
            bbands_df[lower_col] = bbands_df[lower_col].fillna(close_prices)
        if upper_col:
            bbands_df[upper_col] = bbands_df[upper_col].fillna(close_prices)

        # Any other columns (like bandwidth or percent) that might be NaN, fill with 0 or another strategy
        for col in bbands_df.columns:
            if col not in [middle_col, lower_col, upper_col] and bbands_df[col].isna().any():
                bbands_df[col] = bbands_df[col].fillna(0.0) # Default for other bbands stats
        return bbands_df

    @staticmethod
    def _pipeline_compute_bbands(data: pd.Series, length: int, std_dev: float) -> pd.DataFrame:
        """Compute Bollinger Bands using pandas-ta, expecting float64 Series input.
        Returns a DataFrame with lower, middle, upper bands.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.DataFrame(dtype="float64")
        bbands_df = data.ta.bbands(length=length, std=std_dev)
        # Custom NaN filling: Middle band with close, Lower/Upper also with close (0 width initially)
        if bbands_df is not None:
            bbands_df = FeatureEngine._fillna_bbands(bbands_df, data)
        return bbands_df.astype("float64") if bbands_df is not None else pd.DataFrame(dtype="float64")

    @staticmethod
    def _pipeline_compute_roc(data: pd.Series, period: int) -> pd.Series:
        """Compute Rate of Change (ROC) using pandas-ta, expecting float64 Series input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype="float64", name=f"roc_{period}")
        roc_series = data.ta.roc(length=period)
        # Fill NaNs (typically at the beginning) with 0.0, representing no change.
        roc_series = roc_series.fillna(0.0)
        roc_series.name = f"roc_{period}"
        return roc_series.astype("float64")

    @staticmethod
    def _pipeline_compute_atr(
        ohlc_data: pd.DataFrame,
        length: int,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.Series:
        """Compute Average True Range (ATR) using pandas-ta.
        Expects a DataFrame with high, low, close columns (float64).
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlc_data, pd.DataFrame):
            return pd.Series(dtype="float64", name=f"atr_{length}")
        # Ensure required columns are present; this check could be more robust
        if not all(col in ohlc_data.columns for col in [high_col, low_col, close_col]):
            # self.logger.error("ATR calculation missing H/L/C columns.") # Logger not available
            return pd.Series(dtype="float64", name=f"atr_{length}")

        atr_series = ta.atr(
            high=ohlc_data[high_col],
            low=ohlc_data[low_col],
            close=ohlc_data[close_col],
            length=length,
        )
        # Fill NaNs with 0.0 as per chosen strategy.
        # This implies zero volatility for initial undefined periods, which is an approximation.
        atr_series = atr_series.fillna(0.0)
        atr_series.name = f"atr_{length}"
        return atr_series.astype("float64")

    @staticmethod
    def _pipeline_compute_stdev(data: pd.Series, length: int) -> pd.Series:
        """Compute Standard Deviation using pandas .rolling().std().
        Expects float64 Series input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype="float64", name=f"stdev_{length}")
        stdev_series = data.rolling(window=length).std()
        # Fill NaNs (typically at the beginning) with 0.0, representing zero volatility.
        stdev_series = stdev_series.fillna(0.0)
        stdev_series.name = f"stdev_{length}"
        return stdev_series.astype("float64")

    @staticmethod
    def _pipeline_compute_vwap_ohlcv(
        ohlcv_df: pd.DataFrame,
        length: int,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.Series:
        """Compute VWAP from OHLCV data using rolling window.
        Expects DataFrame with Decimal objects for price/volume, converts to float64 Series output.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlcv_df, pd.DataFrame):
            return pd.Series(dtype="float64")
        if not all(col in ohlcv_df.columns for col in [high_col, low_col, close_col, volume_col]):
            return pd.Series(dtype="float64") # Or log error

        # Ensure inputs are Decimal for precision in intermediate calculations
        high_d = ohlcv_df[high_col].apply(Decimal)
        low_d = ohlcv_df[low_col].apply(Decimal)
        close_d = ohlcv_df[close_col].apply(Decimal)
        volume_d = ohlcv_df[volume_col].apply(Decimal)

        typical_price = (high_d + low_d + close_d) / Decimal("3.0")
        tp_vol = typical_price * volume_d

        sum_tp_vol = tp_vol.rolling(window=length, min_periods=length).sum()
        sum_vol = volume_d.rolling(window=length, min_periods=length).sum()

        vwap_series_decimal = sum_tp_vol / sum_vol
        # Replace infinities (from division by zero if sum_vol is 0) with NaN before further processing
        vwap_series_decimal = vwap_series_decimal.replace([Decimal("Infinity"), Decimal("-Infinity")], pd.NA)

        # Iterate and fill NaNs or zero-volume results with the typical price of that bar
        for idx in ohlcv_df.index:
            current_sum_vol = sum_vol.get(idx) # Use .get for safety if index alignment isn't perfect
            current_vwap_val = vwap_series_decimal.get(idx)

            if current_sum_vol == Decimal("0") or pd.isna(current_sum_vol) or pd.isna(current_vwap_val):
                # Ensure we access original Decimal values for typical price calculation if ohlcv_df was float
                # However, ohlcv_df input to this function is already converted to Decimal for H,L,C,V
                # So, high_d, low_d, close_d can be used with .loc[idx]
                # Or, re-access from the original ohlcv_df if it was passed with Decimals
                # For simplicity, assume ohlcv_df passed has Decimal type for H,L,C for this fallback
                # If ohlcv_df was passed as float64, this might lose some Decimal precision for typical price.
                # The current `_calculate_and_publish_features` converts ohlcv_df to float64 first,
                # then this function converts selected columns back to Decimal. This is acceptable.
                h = high_d.get(idx, pd.NA)
                l = low_d.get(idx, pd.NA)
                c = close_d.get(idx, pd.NA)
                if pd.notna(h) and pd.notna(l) and pd.notna(c):
                    vwap_series_decimal[idx] = (h + l + c) / Decimal("3.0")
                else:
                    # If any HLC is NA for this specific bar, we can't calculate typical price, so leave/make it NaN
                    vwap_series_decimal[idx] = pd.NA


        # Convert to float64 for pipeline compatibility.
        # NaNs from missing HLC for typical price fallback, or if typical price itself is NaN, will remain.
        vwap_series_float = vwap_series_decimal.astype("float64")
        vwap_series_float.name = f"vwap_ohlcv_{length}"
        # Final fill for any remaining NaNs, e.g., if HLC was missing for a fallback typical price.
        # Using forward fill first, then backfill, is a common strategy for time series.
        # Or fill with a global mean/median if preferred after this point (e.g. in pipeline steps)
        # For now, let's ensure no NaNs by backfilling then forward filling if any remain.
        # This should be handled by subsequent pipeline steps ideally (output_imputer).
        # The goal here is to avoid NaNs from calculation errors like division by zero.
        return vwap_series_float

    @staticmethod
    def _pipeline_compute_vwap_trades(
        trade_history_deque: deque, # Deque of trade dicts {"price": Decimal, "volume": Decimal, "timestamp": datetime}
        bar_start_times: pd.Series, # Series of datetime objects
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series | None = None, # For fallback
    ) -> pd.Series:
        """Compute VWAP from trade data for specified bar start times.
        Returns a float64 Series.
        If no relevant trades or sum_volume is zero, falls back to ohlcv_close_prices.
        If fallback also fails, defaults to 0.0.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        series_name = f"vwap_trades_{bar_interval_seconds}s"
        output_index = bar_start_times.index if isinstance(bar_start_times, pd.Series) else None
        if not isinstance(bar_start_times, pd.Series): # Basic validation for bar_start_times
            # trade_history_deque validation is implicitly handled by checking if trades_df is None/empty
            return pd.Series(dtype="float64", index=output_index, name=series_name)

        vwap_results = []

        trades_df = None
        if trade_history_deque: # Only proceed if deque is not empty
            try:
                # Ensure all elements in deque are dicts before creating DataFrame
                if not all(isinstance(trade, dict) for trade in trade_history_deque):
                    # Log or handle malformed deque elements if necessary
                    trades_df = pd.DataFrame(columns=["price", "volume", "timestamp"]) # Empty DF
                else:
                    trades_df = pd.DataFrame(list(trade_history_deque))

                if not trades_df.empty: # Proceed with type conversion only if DataFrame is not empty
                    trades_df["price"] = trades_df["price"].apply(lambda x: Decimal(str(x)))
                    trades_df["volume"] = trades_df["volume"].apply(lambda x: Decimal(str(x)))
                    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
                else: # trades_df is empty (e.g. deque was empty or contained non-dict items)
                    trades_df = None # Ensure it's None to trigger fallback for all bars
            except (ValueError, TypeError, KeyError, AttributeError):
                # Catch broad errors during DataFrame creation or type conversion
                # self.logger.warning("Error processing trade_history_deque: %s", e) # Logger not available
                trades_df = None # Force fallback for all bars if trade data is corrupt

        for bar_start_dt_idx, bar_start_dt in bar_start_times.items(): # Use .items() for index access
            calculated_vwap = np.nan

            if trades_df is not None and not trades_df.empty:
                bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
                relevant_trades = trades_df[
                    (trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)
                ]

                if not relevant_trades.empty:
                    sum_price_volume = (relevant_trades["price"] * relevant_trades["volume"]).sum()
                    sum_volume = relevant_trades["volume"].sum()

                    if sum_volume > Decimal("0"):
                        vwap_decimal = sum_price_volume / sum_volume
                        calculated_vwap = float(vwap_decimal)

            # Fallback logic
            if pd.isna(calculated_vwap):
                current_bar_ohlcv_close_price = np.nan
                if ohlcv_close_prices is not None:
                    # Try to get by direct index (bar_start_dt may be the index for ohlcv_close_prices)
                    # Or by bar_start_dt_idx if ohlcv_close_prices is aligned with bar_start_times' original index
                    if bar_start_dt in ohlcv_close_prices.index:
                         current_bar_ohlcv_close_price = ohlcv_close_prices.get(bar_start_dt)
                    elif bar_start_dt_idx in ohlcv_close_prices.index: # Fallback to original index if different
                         current_bar_ohlcv_close_price = ohlcv_close_prices.get(bar_start_dt_idx)

                if pd.notna(current_bar_ohlcv_close_price):
                    calculated_vwap = float(current_bar_ohlcv_close_price)
                else: # Close price itself is NaN or not found
                    calculated_vwap = 0.0 # Final fallback

            vwap_results.append(calculated_vwap)

        return pd.Series(vwap_results, index=output_index, dtype="float64", name=series_name)


    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
        # --- Removed _calculate_true_volume_delta_from_trades, _calculate_vwap_from_trades ---


    @staticmethod
    def _pipeline_compute_l2_spread(l2_books_series: pd.Series) -> pd.DataFrame:
        """Computes bid-ask spread from a Series of L2 book snapshots.
        Outputs a DataFrame with 'abs_spread' and 'pct_spread' (float64).
        If L2 book is None, empty, or best bid/ask cannot be determined, outputs 0.0 for spreads.
        Intended for Scikit-learn FunctionTransformer.
        """
        abs_spreads = []
        pct_spreads = []
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None

        for book in l2_books_series:
            current_abs_spread = 0.0
            current_pct_spread = 0.0
            try:
                # Ensure book and its bids/asks are valid and non-empty before attempting access
                if book and \
                   isinstance(book.get("bids"), list) and len(book["bids"]) > 0 and \
                   isinstance(book["bids"][0], (list, tuple)) and len(book["bids"][0]) == 2 and \
                   isinstance(book.get("asks"), list) and len(book["asks"]) > 0 and \
                   isinstance(book["asks"][0], (list, tuple)) and len(book["asks"][0]) == 2:

                    best_bid_price_str = str(book["bids"][0][0])
                    best_ask_price_str = str(book["asks"][0][0])

                    # Check for non-numeric or empty strings before Decimal conversion
                    if not best_bid_price_str or not best_ask_price_str:
                        raise ValueError("Empty price string encountered.")

                    best_bid = Decimal(best_bid_price_str)
                    best_ask = Decimal(best_ask_price_str)

                    if best_ask > best_bid:  # Ensure valid spread
                        abs_spread_val = best_ask - best_bid
                        mid_price = (best_bid + best_ask) / Decimal("2")
                        if mid_price != Decimal("0"):
                            pct_spread_val = (abs_spread_val / mid_price) * Decimal("100")
                        else:
                            pct_spread_val = Decimal("0.0")

                        current_abs_spread = float(abs_spread_val)
                        current_pct_spread = float(pct_spread_val)
                # else: conditions for invalid book structure lead to default 0.0 values
            except (TypeError, IndexError, ValueError, AttributeError):
                # Errors from malformed book data, missing keys, non-Decimal convertible strings, etc.
                # These will result in the default 0.0 values being used.
                pass # current_abs_spread and current_pct_spread remain 0.0

            abs_spreads.append(current_abs_spread)
            pct_spreads.append(current_pct_spread)

        return pd.DataFrame(
            {"abs_spread": abs_spreads, "pct_spread": pct_spreads},
            index=output_index,
            dtype="float64",
        )

    @staticmethod
    def _pipeline_compute_l2_imbalance(l2_books_series: pd.Series, levels: int = 5) -> pd.Series:
        """Computes order book imbalance from a Series of L2 book snapshots.
        Outputs a Series (float64).
        If L2 book is None, empty, levels are malformed, or total volume for imbalance calc is zero, outputs 0.0.
        Intended for Scikit-learn FunctionTransformer.
        """
        imbalances = []
        series_name = f"imbalance_{levels}"
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None

        for book in l2_books_series:
            current_imbalance = 0.0
            try:
                # Validate book structure and content for specified levels
                if book and \
                   isinstance(book.get("bids"), list) and \
                   isinstance(book.get("asks"), list) and \
                   len(book["bids"]) >= levels and \
                   len(book["asks"]) >= levels:

                    # Check integrity of levels up to 'levels'
                    valid_bids = True
                    for i in range(levels):
                        if not (isinstance(book["bids"][i], (list, tuple)) and len(book["bids"][i]) == 2 and book["bids"][i][1] is not None):
                            valid_bids = False; break

                    valid_asks = True
                    for i in range(levels):
                        if not (isinstance(book["asks"][i], (list, tuple)) and len(book["asks"][i]) == 2 and book["asks"][i][1] is not None):
                            valid_asks = False; break

                    if valid_bids and valid_asks:
                        bid_vol_at_levels = sum(Decimal(str(book["bids"][i][1])) for i in range(levels))
                        ask_vol_at_levels = sum(Decimal(str(book["asks"][i][1])) for i in range(levels))

                        total_vol = bid_vol_at_levels + ask_vol_at_levels
                        if total_vol > Decimal("0"):
                            imbalance_val = (bid_vol_at_levels - ask_vol_at_levels) / total_vol
                            current_imbalance = float(imbalance_val)
                # else: conditions for invalid book structure lead to default 0.0
            except (TypeError, IndexError, ValueError, AttributeError):
                # Errors from malformed book data, missing keys, non-Decimal convertible strings etc.
                pass # current_imbalance remains 0.0

            imbalances.append(current_imbalance)

        return pd.Series(imbalances, index=output_index, dtype="float64", name=series_name)

    @staticmethod
    def _pipeline_compute_l2_wap(
        l2_books_series: pd.Series,
        ohlcv_close_prices: pd.Series | None = None, # For fallback
        levels: int = 1, # Typically levels=1 for WAP
    ) -> pd.Series:
        """Computes Weighted Average Price (WAP) from a Series of L2 book snapshots.
        Outputs a Series (float64).
        If WAP cannot be calculated (e.g., invalid book, zero volume for top level),
        it falls back to the corresponding ohlcv_close_prices.loc[index_of_l2_book_entry].
        If fallback also fails or is not available, defaults to 0.0.
        Intended for Scikit-learn FunctionTransformer.
        """
        series_name = f"wap_{levels}"
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None
        if not isinstance(l2_books_series, pd.Series):
            return pd.Series(dtype="float64", name=series_name, index=output_index)

        waps = []
        for book_idx, book in l2_books_series.items():
            calculated_wap = np.nan # Initialize as NaN to indicate not yet calculated/fallback needed

            try:
                # Validate book structure for the specified number of levels (here, only top level for WAP)
                if book and \
                   isinstance(book.get("bids"), list) and len(book["bids"]) >= levels and \
                   isinstance(book["bids"][levels-1], (list, tuple)) and len(book["bids"][levels-1]) == 2 and \
                   isinstance(book.get("asks"), list) and len(book["asks"]) >= levels and \
                   isinstance(book["asks"][levels-1], (list, tuple)) and len(book["asks"][levels-1]) == 2:

                    # For WAP, typically levels=1, so we use index 0
                    best_bid_price_str = str(book["bids"][0][0])
                    best_bid_vol_str = str(book["bids"][0][1])
                    best_ask_price_str = str(book["asks"][0][0])
                    best_ask_vol_str = str(book["asks"][0][1])

                    if not all([best_bid_price_str, best_bid_vol_str, best_ask_price_str, best_ask_vol_str]):
                        raise ValueError("Empty price or volume string encountered.")

                    best_bid_price = Decimal(best_bid_price_str)
                    best_bid_vol = Decimal(best_bid_vol_str)
                    best_ask_price = Decimal(best_ask_price_str)
                    best_ask_vol = Decimal(best_ask_vol_str)

                    total_vol = best_bid_vol + best_ask_vol
                    if total_vol > Decimal("0"):
                        wap_decimal = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / total_vol
                        calculated_wap = float(wap_decimal)
                # else: conditions for invalid book structure, calculated_wap remains np.nan
            except (TypeError, IndexError, ValueError, AttributeError):
                # Errors from malformed book data, missing keys, non-Decimal convertible strings, etc.
                # calculated_wap remains np.nan
                pass

            # Fallback logic
            if pd.isna(calculated_wap):
                if ohlcv_close_prices is not None and book_idx in ohlcv_close_prices.index:
                    fallback_close_price = ohlcv_close_prices.get(book_idx)
                    if pd.notna(fallback_close_price):
                        # Ensure the fallback close price is float
                        calculated_wap = float(fallback_close_price)
                    else: # Close price itself is NaN
                        calculated_wap = 0.0 # Final fallback if close price is NaN
                else: # ohlcv_close_prices not available or index mismatch
                    calculated_wap = 0.0 # Final fallback if close price is not available

            waps.append(calculated_wap)

        return pd.Series(waps, index=output_index, dtype="float64", name=series_name)

    @staticmethod
    def _pipeline_compute_l2_depth(l2_books_series: pd.Series, levels: int = 5) -> pd.DataFrame:
        """Computes bid and ask depth from a Series of L2 book snapshots.
        Outputs a DataFrame with 'bid_depth_{levels}' and 'ask_depth_{levels}' (float64).
        If L2 book is None, empty, or levels are malformed, outputs 0.0 for depths.
        Intended for Scikit-learn FunctionTransformer.
        """
        bid_depths = []
        ask_depths = []
        output_index = l2_books_series.index if isinstance(l2_books_series, pd.Series) else None
        col_name_bid = f"bid_depth_{levels}"
        col_name_ask = f"ask_depth_{levels}"

        for book in l2_books_series:
            current_bid_depth = 0.0
            current_ask_depth = 0.0
            try:
                # Validate book structure and content for specified levels
                if book and \
                   isinstance(book.get("bids"), list) and \
                   isinstance(book.get("asks"), list) and \
                   len(book["bids"]) >= levels and \
                   len(book["asks"]) >= levels:

                    valid_bids = True
                    for i in range(levels):
                        if not (isinstance(book["bids"][i], (list, tuple)) and len(book["bids"][i]) == 2 and book["bids"][i][1] is not None):
                            valid_bids = False; break

                    valid_asks = True
                    for i in range(levels):
                        if not (isinstance(book["asks"][i], (list, tuple)) and len(book["asks"][i]) == 2 and book["asks"][i][1] is not None):
                            valid_asks = False; break

                    if valid_bids and valid_asks:
                        bid_depth_val = sum(Decimal(str(book["bids"][i][1])) for i in range(levels))
                        ask_depth_val = sum(Decimal(str(book["asks"][i][1])) for i in range(levels))
                        current_bid_depth = float(bid_depth_val)
                        current_ask_depth = float(ask_depth_val)
                # else: conditions for invalid book structure lead to default 0.0
            except (TypeError, IndexError, ValueError, AttributeError):
                 # Errors from malformed book data, missing keys, non-Decimal convertible strings etc.
                pass # current_bid_depth and current_ask_depth remain 0.0

            bid_depths.append(current_bid_depth)
            ask_depths.append(current_ask_depth)

        df = pd.DataFrame({
            col_name_bid: bid_depths,
            col_name_ask: ask_depths,
        }, index=output_index, dtype="float64")
        return df

    @staticmethod
    def _pipeline_compute_volume_delta(
        trade_history_deque: deque, # Deque of trade dicts
        bar_start_times: pd.Series, # Series of bar start datetime objects
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series | None = None, # Added for signature consistency, not used by this specific function
    ) -> pd.Series:
        """Computes Volume Delta from trade data for specified bar start times.
        If no trades for a bar, delta is 0.0.
        Outputs a Series (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        deltas = []
        series_name = f"volume_delta_{bar_interval_seconds}s"
        if not isinstance(bar_start_times, pd.Series) or not isinstance(trade_history_deque, deque):
            return pd.Series(dtype="float64", index=bar_start_times.index if isinstance(bar_start_times, pd.Series) else None, name=series_name)

        if not trade_history_deque: # No trades in entire history
            return pd.Series(0.0, index=bar_start_times.index, dtype="float64", name=series_name)

        trades_df = pd.DataFrame(list(trade_history_deque))
        # Ensure 'price' and 'volume' are converted to Decimal, handling potential string inputs
        trades_df["price"] = trades_df["price"].apply(lambda x: Decimal(str(x)))
        trades_df["volume"] = trades_df["volume"].apply(Decimal)
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        # Ensure side is lowercase
        trades_df["side"] = trades_df["side"].str.lower()


        for bar_start_dt in bar_start_times:
            bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
            relevant_trades = trades_df[
                (trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)
            ]

            if relevant_trades.empty:
                deltas.append(0.0) # Or np.nan if preferred for "no trades" vs "zero delta"
                continue

            buy_volume = relevant_trades[relevant_trades["side"] == "buy"]["volume"].sum()
            sell_volume = relevant_trades[relevant_trades["side"] == "sell"]["volume"].sum()
            deltas.append(float(buy_volume - sell_volume))

        return pd.Series(deltas, index=bar_start_times.index, dtype="float64", name=f"volume_delta_{bar_interval_seconds}s")


    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
        # --- Removed _calculate_true_volume_delta_from_trades, _calculate_vwap_from_trades ---


    async def _calculate_and_publish_features(
        self,
        trading_pair: str,
        timestamp_features_for: str,
    ) -> None:
        """Calculate all configured features using pipelines and publish them."""
        ohlcv_df_full_history = self.ohlcv_history.get(trading_pair)
        min_history_req = self._get_min_history_required() # Get actual requirement
        if ohlcv_df_full_history is None or len(ohlcv_df_full_history) < min_history_req: # type: ignore
            self.logger.info(
                "Not enough OHLCV data for %s to calculate features. Need %s, have %s.",
                trading_pair,
                min_history_req, # type: ignore
                len(ohlcv_df_full_history) if ohlcv_df_full_history is not None else 0, # type: ignore
                source_module=self._source_module,
            )
            return

        current_l2_book = self.l2_books.get(trading_pair)
        if current_l2_book and (
            not current_l2_book.get("bids") or not current_l2_book.get("asks")
        ):
            self.logger.debug(
                "L2 book for %s is present but empty or missing bids/asks. "
                "L2 features may be skipped.",
                trading_pair,
                source_module=self._source_module,
            )
            # L2 features might be skipped by their handlers if book is not suitable.

        all_generated_features: dict[str, Any] = {}

        bar_start_datetime = pd.to_datetime(timestamp_features_for, utc=True)

        # Filter OHLCV data up to the current bar's start time for historical context
        # Pipelines will internally select the latest point after calculation over history.
        current_ohlcv_df_decimal = ohlcv_df_full_history[ohlcv_df_full_history.index <= bar_start_datetime]

        if current_ohlcv_df_decimal.empty:
            self.logger.warning("No historical OHLCV data available for %s up to %s.", trading_pair, bar_start_datetime)
            return # Cannot proceed without data for this timestamp

        # Prepare standard input types based on Decimal data
        close_series_for_pipelines = current_ohlcv_df_decimal["close"].astype("float64")
        # Ensure 'open', 'high', 'low', 'close', 'volume' are float for OHLCV df inputs
        ohlcv_df_for_pipelines = current_ohlcv_df_decimal.astype({
            "open": "float64", "high": "float64", "low": "float64",
            "close": "float64", "volume": "float64",
        })

        # L2 book snapshot for the current bar
        # Assuming self.l2_books[trading_pair] holds the latest book, or one aligned by a separate process
        # For pipeline processing, we need a Series (even if single-element)
        latest_l2_book_snapshot = self._get_aligned_l2_book(trading_pair, bar_start_datetime)
        # Use the aligned L2 book for better accuracy
        l2_books_aligned_series = pd.Series([latest_l2_book_snapshot], index=[bar_start_datetime])

        # Trade data for trade-based features
        trades_deque = self.trade_history.get(trading_pair, deque())
        # For single bar calculation, bar_start_times_series is just the current bar
        bar_start_times_series = pd.Series([bar_start_datetime], index=[bar_start_datetime])

        # Prepare the single close price for the current bar, aligned to its timestamp for dynamic injection
        # This Series will have one entry: index=bar_start_datetime, value=close_price_at_bar_start_datetime
        ohlcv_close_for_dynamic_injection = None
        if bar_start_datetime in close_series_for_pipelines.index:
            ohlcv_close_for_dynamic_injection = close_series_for_pipelines.loc[[bar_start_datetime]]
        else:
            self.logger.warning(
                "Could not find close price for current bar %s in historical data. Features needing this fallback may fail or use 0.0.",
                bar_start_datetime,
            )
            # Create an empty series with the right index to prevent downstream errors if it's expected
            ohlcv_close_for_dynamic_injection = pd.Series(dtype="float64", index=[bar_start_datetime])


        for pipeline_name, pipeline_info in self.feature_pipelines.items():
            pipeline_obj: Pipeline = pipeline_info["pipeline"]
            spec: InternalFeatureSpec = pipeline_info["spec"]

            pipeline_input_data: Any = None
            raw_pipeline_output: Any = None # Define here for clarity

            # Determine input data for the pipeline
            if spec.input_type == "close_series":
                pipeline_input_data = close_series_for_pipelines
            elif spec.input_type == "ohlcv_df":
                pipeline_input_data = ohlcv_df_for_pipelines
            elif spec.input_type == "l2_book_series":
                pipeline_input_data = l2_books_aligned_series # Single latest book snapshot in a Series
            elif spec.input_type == "trades_and_bar_starts":
                # For these, X is the trade_history_deque. bar_start_times is injected dynamically.
                pipeline_input_data = trades_deque
            else:
                self.logger.warning("Unknown input_type '%s' for pipeline %s. Skipping.", spec.input_type, pipeline_name)
                continue

            try:
                pipeline_to_run = pipeline_obj # By default, use the original pipeline

                # Dynamic kwarg injection for specific calculators
                if spec.calculator_type in ["l2_wap", "vwap_trades", "volume_delta"]:
                    pipeline_to_run = clone(pipeline_obj) # Clone to modify kw_args safely
                    calculator_step_name = f"{spec.key}_calculator"

                    if calculator_step_name in pipeline_to_run.named_steps:
                        calculator_transformer = pipeline_to_run.named_steps[calculator_step_name]
                        current_kw_args = calculator_transformer.kw_args.copy()

                        # Inject ohlcv_close_prices (aligned to the specific input type's index)
                        if ohlcv_close_for_dynamic_injection is not None and not ohlcv_close_for_dynamic_injection.empty:
                            current_kw_args["ohlcv_close_prices"] = ohlcv_close_for_dynamic_injection
                        else: # Pass None or an empty series if not available, function should handle it
                            current_kw_args["ohlcv_close_prices"] = pd.Series(dtype="float64", index=[bar_start_datetime])


                        # For trade-based features, also inject bar_start_times
                        if spec.input_type == "trades_and_bar_starts":
                            current_kw_args["bar_start_times"] = bar_start_times_series

                        calculator_transformer.kw_args = current_kw_args
                    else:
                        self.logger.error("Calculator step %s not found in cloned pipeline %s. Skipping dynamic args.", calculator_step_name, pipeline_name)

                # Execute the pipeline (original or cloned-and-modified)
                if pipeline_input_data is not None:
                    raw_pipeline_output = pipeline_to_run.fit_transform(pipeline_input_data)
                else:
                    # This case should ideally be caught by input_type checks or earlier validation
                    self.logger.warning("Pipeline input data is None for %s. Skipping execution.", pipeline_name)
                    continue # Skip to next pipeline if input data is None

            except Exception as e:
                self.logger.exception("Error executing pipeline %s: %s", pipeline_name, e)
                continue # Skip this pipeline on error

            # Process pipeline outputs using enhanced output handlers
            try:
                # Check if this feature has enhanced output specs
                output_handler = self.output_handlers.get(spec.key)
                
                if output_handler and spec.output_specs:
                    # Use enhanced output processing
                    processed_outputs = output_handler.process_feature_outputs(raw_pipeline_output)
                    
                    # Extract final values and add to all_generated_features
                    if isinstance(processed_outputs, pd.DataFrame):
                        # Get the latest row if multiple rows
                        if len(processed_outputs) > 1:
                            latest_row = processed_outputs.iloc[-1]
                        else:
                            latest_row = processed_outputs.iloc[0] if not processed_outputs.empty else pd.Series()
                        
                        # Add each column as a feature
                        for col_name, value in latest_row.items():
                            if pd.notna(value):
                                all_generated_features[col_name] = float(value)
                    
                else:
                    # Use traditional output processing for backward compatibility
                    latest_features_values: Any = None
                    if isinstance(raw_pipeline_output, pd.Series):
                        if not raw_pipeline_output.empty:
                            if spec.input_type not in ["l2_book_series", "trades_and_bar_starts"] or len(raw_pipeline_output) > 1:
                                latest_features_values = raw_pipeline_output.iloc[-1]
                            else:
                                latest_features_values = raw_pipeline_output.iloc[0] if len(raw_pipeline_output) == 1 else np.nan
                    elif isinstance(raw_pipeline_output, pd.DataFrame):
                        if not raw_pipeline_output.empty:
                            if spec.input_type not in ["l2_book_series"] or len(raw_pipeline_output) > 1:
                                latest_features_values = raw_pipeline_output.iloc[-1]
                            else:
                                latest_features_values = raw_pipeline_output.iloc[0] if len(raw_pipeline_output) == 1 else pd.Series(dtype="float64")
                    elif isinstance(raw_pipeline_output, np.ndarray):
                        if raw_pipeline_output.ndim == 1 and raw_pipeline_output.size > 0:
                            latest_features_values = raw_pipeline_output[-1]
                        elif raw_pipeline_output.ndim == 2 and raw_pipeline_output.shape[0] > 0:
                            latest_features_values = pd.Series(raw_pipeline_output[-1, :])
                    else:
                        latest_features_values = raw_pipeline_output

                    # Traditional naming and storing for backward compatibility
                    if isinstance(latest_features_values, pd.Series):
                        for idx_name, value in latest_features_values.items():
                            col_name = str(idx_name)
                            base_feature_key = pipeline_name.replace("_pipeline","")
                            feature_output_name = f"{base_feature_key}_{col_name}"
                            all_generated_features[feature_output_name] = value
                    elif pd.notna(latest_features_values):
                        feature_output_name = pipeline_name.replace("_pipeline","")
                        all_generated_features[feature_output_name] = latest_features_values
                        
            except Exception as e:
                self.logger.error(f"Error processing outputs for feature {spec.key}: {e}")
                # Fall back to simple processing
                if pd.notna(raw_pipeline_output):
                    feature_output_name = pipeline_name.replace("_pipeline","")
                    all_generated_features[feature_output_name] = float(raw_pipeline_output) if hasattr(raw_pipeline_output, '__float__') else 0.0

        # Extract advanced features if configured
        try:
            advanced_feature_specs = [spec for spec in self._feature_configs.values() 
                                     if spec.calculator_type in self.advanced_extractor.advanced_indicators]
            
            if advanced_feature_specs:
                # Prepare data for advanced extraction
                l2_data = latest_l2_book_snapshot
                trade_data = list(trades_deque) if trades_deque else None
                
                # Extract advanced features
                advanced_result = await self.advanced_extractor.extract_advanced_features(
                    ohlcv_df_for_pipelines, 
                    advanced_feature_specs,
                    l2_data=l2_data,
                    trade_data=trade_data
                )
                
                # Add advanced features to the main feature set
                if not advanced_result.features.empty:
                    # Get the latest row of advanced features
                    latest_advanced = advanced_result.features.iloc[-1] if len(advanced_result.features) > 1 else advanced_result.features.iloc[0]
                    
                    for feature_name, value in latest_advanced.items():
                        if pd.notna(value):
                            all_generated_features[f"advanced_{feature_name}"] = float(value)
                    
                    self.logger.debug(
                        f"Added {len(latest_advanced)} advanced features for {trading_pair}. "
                        f"Quality metrics: {advanced_result.quality_metrics}"
                    )
                        
        except Exception as e:
            self.logger.warning(f"Advanced feature extraction failed for {trading_pair}: {e}")

        # Validate and structure features using Pydantic model
        try:
            # Ensure all_generated_features contains float values, not Decimals or other types
            # Most pipeline steps should output float64, but final check can be useful.
            float_features = {k: float(v) if pd.notna(v) else np.nan for k, v in all_generated_features.items()}

            # Filter out NaNs before passing to Pydantic if model fields are not Optional
            # Or ensure Pydantic model fields are Optional if NaNs are possible and mean "feature not applicable"
            # For now, assuming Pydantic model fields are non-Optional floats.
            # If a feature is NaN, it means it couldn't be calculated; this should be handled.
            # Pydantic will raise validation error if a required field is missing or not float.
            # Let's filter NaNs for required fields. If a feature *could* be legitimately missing,
            # its corresponding Pydantic field should be Optional.
            # The current PublishedFeaturesV1 expects all defined fields.

            # If a feature calculation results in NaN, it might indicate an issue.
            # For now, we will attempt to pass them and let Pydantic validate.
            # If Pydantic fields are not Optional, NaNs will cause errors if not converted or handled.
            # The current Pydantic model has non-optional float fields.
            # So, if any value in float_features is NaN, Pydantic validation will fail.
            # This is a design choice: either features must always be valid floats, or Pydantic model must use Optional[float].
            # Given "Zero NaN" policy, we expect valid floats.
            # If a feature is missing from all_generated_features, Pydantic will also complain.

            pydantic_features = PublishedFeaturesV1(**float_features)
            # Use model_dump() for the payload if the pubsub system expects a dict.
            features_for_payload = pydantic_features.model_dump()
        except Exception as e: # Catch Pydantic ValidationError or other issues
            self.logger.error(
                "Failed to validate or structure features using Pydantic model for %s at %s: %s. Raw features: %s",
                trading_pair,
                timestamp_features_for,
                e,
                all_generated_features, # Log raw features for debugging
                source_module=self._source_module,
            )
            return # Do not publish if validation fails

        # Fallback for any old handlers if no pipelines were built (mostly for transition)
        if not self.feature_pipelines and not features_for_payload:
            self.logger.debug("No pipelines executed, attempting feature calculation with remaining old handlers.", source_module=self._source_module)
            # ... (old handler logic can be here if needed, but it's mostly empty now) ...


        if not features_for_payload: # Check if any features were produced
            self.logger.info(
                "No features were successfully structured or validated for %s at %s. Not publishing event.",
                trading_pair,
                timestamp_features_for, # This was timestamp_features_for
                source_module=self._source_module,
            )
            return

        # Construct and publish FeatureEvent
        event_payload = {
            "trading_pair": trading_pair,
            "exchange": self.config.get("exchange_name", "kraken"),
            "timestamp_features_for": timestamp_features_for, # Corrected variable name
            "features": features_for_payload, # Use Pydantic model's dict representation
        }

        full_feature_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": EventType.FEATURES_CALCULATED.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source_module": self._source_module,
            "payload": event_payload,
        }

        try:
            # Convert the dictionary to an Event object to match expected type
            from typing import cast

            # Use Any for Event since we can't import the actual type
            await self.pubsub_manager.publish(cast("Any", full_feature_event))
            self.logger.info(
                "Published FEATURES_CALCULATED event for %s at %s",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
                context={
                    "event_id": full_feature_event["event_id"],
                    "num_features": len(features_for_payload), # Use the dict from Pydantic model
                },
            )
        except Exception:
            self.logger.exception(
                "Failed to publish FEATURES_CALCULATED event for %s",
                trading_pair,
                source_module=self._source_module,
            )

    def _format_feature_value(self, value: Decimal | float | object) -> str:
        """Format a feature value to string. Decimal/float to 8 decimal places."""
        if isinstance(value, Decimal | float):
            return f"{value:.8f}"
        return str(value)

    # --- Feature Processing Methods (to be refactored for pipeline usage) ---
    # The _process_rsi_feature and _process_macd_feature methods are removed
    # as their logic will be incorporated into the new pipeline-based feature generation.
    # Other _process_* methods will be updated or replaced in subsequent steps.
    # --- Removed _process_bbands_feature, _process_roc_feature, _process_atr_feature, _process_stdev_feature ---
    # --- Removed _process_vwap_feature ---
    # --- Removed _process_roc_feature, _process_atr_feature, _process_stdev_feature ---
    # --- Removed _process_l2_spread_feature, _process_l2_imbalance_feature, _process_l2_wap_feature ---
    # --- Removed _process_l2_depth_feature, _process_volume_delta_feature ---

    def _get_aligned_l2_book(
        self, 
        trading_pair: str, 
        target_timestamp: datetime,
        max_age_seconds: int = 300  # 5 minutes default
    ) -> dict[str, Any] | None:
        """Get the L2 book snapshot closest to the target timestamp.
        
        Args:
            trading_pair: The trading pair to get L2 book for
            target_timestamp: The timestamp to align with
            max_age_seconds: Maximum age of L2 snapshot to consider valid
            
        Returns:
            L2 book dict or None if no valid book found
        """
        history = self.l2_books_history.get(trading_pair)
        if not history:
            # Fallback to latest book if no history
            return self.l2_books.get(trading_pair)
            
        # Find the book with timestamp closest to but not after target_timestamp
        best_book = None
        best_time_diff = None
        
        for entry in reversed(history):  # Most recent first
            book_timestamp = entry["timestamp"]
            
            # Skip books that are after the target timestamp
            if book_timestamp > target_timestamp:
                continue
                
            time_diff = (target_timestamp - book_timestamp).total_seconds()
            
            # Skip if too old
            if time_diff > max_age_seconds:
                break
                
            if best_time_diff is None or time_diff < best_time_diff:
                best_time_diff = time_diff
                best_book = entry["book"]
                
        if best_book is None:
            self.logger.debug(
                "No valid L2 book found for %s at %s within %s seconds",
                trading_pair,
                target_timestamp,
                max_age_seconds,
                source_module=self._source_module,
            )
            # Fallback to latest book as last resort
            return self.l2_books.get(trading_pair)
            
        return best_book
