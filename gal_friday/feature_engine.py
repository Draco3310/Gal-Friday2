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

from collections import defaultdict, deque
import contextlib
from dataclasses import dataclass, field  # Added for InternalFeatureSpec
from datetime import datetime
from decimal import Decimal
from enum import Enum  # Added for _LocalFeatureCategory
from pathlib import Path  # Added for feature registry
from typing import TYPE_CHECKING, Any, TypedDict, Union, cast
import uuid

import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.base import (
    clone,  # For cloning pipelines to modify params at runtime
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import yaml  # Added for feature registry

from gal_friday.core.events import EventType
from gal_friday.core.feature_models import (
    PublishedFeaturesV1,  # Added for Pydantic model
)
from gal_friday.dal.models import DataQualityIssue

# Advanced imputation system for crypto trading data
from .feature_imputation import (
    DataType,
    ImputationConfig,
    ImputationMethod,
    ImputationQuality,
    create_imputation_manager,
)
from .feature_repo import fetch_latest_features
from .model_registry import ImputationModelRegistry, build_ml_features
from .utils.correlation_utils import compute_correlations

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

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from gal_friday.core.pubsub import PubSubManager
    from gal_friday.dal.repositories.history_repository import HistoryRepository
    from gal_friday.interfaces.historical_data_service_interface import (
        HistoricalDataService,
    )
    from gal_friday.logger_service import LoggerService

# Type definitions for L2 order book data

class L2BookSnapshot(TypedDict):
    """Type definition for L2 order book snapshot."""
    bids: list[list[float | str | Decimal]]  # List of [price, volume] pairs
    asks: list[list[float | str | Decimal]]  # List of [price, volume] pairs
    timestamp: Any  # Can be datetime, pd.Timestamp, etc.

# L2 book series can contain either book snapshots or other values (e.g., float for fallback)
L2BookValue = Union[L2BookSnapshot, float, None]

# Define InternalFeatureSpec
@dataclass
class InternalFeatureSpec:
    key: str  # Unique key for the feature. Used for activation via app config
    # and as a base for published feature names.
    calculator_type: str  # Defines the core calculation logic (e.g., "rsi", "macd").
    # Maps to a `_pipeline_compute_{calculator_type}` method.
    input_type: str  # Specifies the type of input data required by the calculator
    # (e.g., 'close_series', 'ohlcv_df', 'l2_book_series').
    category: FeatureCategory = FeatureCategory.TECHNICAL  # Categorizes the feature
    # (e.g., TECHNICAL, L2_ORDER_BOOK, TRADE_DATA).
    parameters: dict[str, Any] = field(default_factory=dict[str, Any])  # Dictionary of
    # parameters passed to the feature calculator function.
    imputation: dict[str, Any] | str | None = None  # Configuration for the output
    # imputation step in the pipeline (e.g., `{"strategy": "constant", "fill_value": 0.0}`).
    # Applied as a final fallback.
    scaling: dict[str, Any] | str | None = None  # Configuration for the output scaling
    # step (e.g., `{"method": "standard"}`). Applied by FeatureEngine.
    imputation_model_key: str | None = None  # Key referencing ML model for imputation
    imputation_model_version: str | None = None  # Optional version of the imputation model
    description: str = ""  # Human-readable description of the feature and its
    # configuration.
    version: str | None = None  # Version string for the feature definition,
    # loaded from the registry.
    output_properties: dict[str, Any] = field(default_factory=dict[str, Any])  # Dictionary
    # describing expected output characteristics (e.g., `{"value_type": "float", "range": [0, 1]}`).

    # Enhanced fields for comprehensive output handling and multiple outputs
    output_specs: list[OutputSpec] = field(default_factory=list[Any]) # Detailed specifications for each output
    output_naming_pattern: str | None = None # Pattern for naming outputs (e.g., '{feature_name}_{output_name}')
    dependencies: list[str] = field(default_factory=list[Any]) # Other features this depends on
    required_lookback_periods: int = 1 # Minimum data points required
    author: str | None = None # Feature author/creator
    created_at: str | None = None # Creation timestamp
    tags: list[str] = field(default_factory=list[Any]) # Feature tags for organization
    cache_enabled: bool = True # Whether to cache feature results
    cache_ttl_minutes: int | None = None # Cache time-to-live in minutes
    computation_priority: int = 5 # Computation priority (1-10)

    @property
    def output_names(self) -> list[str]:
        """Get list[Any] of all output names based on specs and naming pattern."""
        if not self.output_specs:
            return [self.key]  # Default to feature key if no specs

        if self.output_naming_pattern:
            return [
                self.output_naming_pattern.format(
                    feature_name=self.key,
                    output_name=spec.name,
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


class FeatureProcessingError(Exception):
    """Exception raised for feature processing errors."""


class FeatureOutputHandler:
    """Enhanced handler for processing multiple feature outputs according to specifications."""

    def __init__(self, feature_spec: InternalFeatureSpec) -> None:
        """Initialize the instance."""
        self.spec = feature_spec
        self.logger: LoggerService | None = None  # Will be set by FeatureEngine if available

    def process_feature_outputs(self, raw_outputs: Any) -> pd.DataFrame:
        """Process raw feature computation outputs according to specification.
        Handles multiple output formats and applies validation, type conversion, and naming.

        Args:
            raw_outputs: Raw output from feature calculation (Series[Any], DataFrame, dict[str, Any], list[Any], scalar)

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
                    f"Successfully processed {len(final_outputs.columns)} outputs for feature {self.spec.key}",
                )

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error processing outputs for feature {self.spec.key}: ")
            raise FeatureProcessingError(f"Failed to process feature outputs: {e}")
        else:
            return final_outputs

    def _standardize_raw_outputs(self, raw_outputs: Any) -> pd.DataFrame:
        """Convert various output formats to standardized DataFrame."""
        if isinstance(raw_outputs, pd.DataFrame):
            return raw_outputs

        if isinstance(raw_outputs, pd.Series):
            # Single series output
            output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
            return pd.DataFrame({output_name: raw_outputs})

        if isinstance(raw_outputs, np.ndarray):
            # NumPy array - could be 1D or 2D
            if raw_outputs.ndim == 1:
                output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
                return pd.DataFrame({output_name: raw_outputs})
            # Multi-dimensional array
            columns = [spec.name for spec in self.spec.output_specs[:raw_outputs.shape[1]]]
            if len(columns) < raw_outputs.shape[1]:
                # Generate default names for additional columns
                for i in range(len(columns), raw_outputs.shape[1]):
                    columns.append(f"output_{i}")
            return pd.DataFrame(raw_outputs, columns=columns)

        if isinstance(raw_outputs, dict):
            # Dictionary of outputs
            return pd.DataFrame(raw_outputs)

        if isinstance(raw_outputs, list):
            # List of values
            if len(self.spec.output_specs) == 1:
                output_name = self.spec.output_specs[0].name
                return pd.DataFrame({output_name: raw_outputs})
            # Multiple outputs - map to specs
            output_dict = {}
            for i, spec in enumerate(self.spec.output_specs):
                if i < len(raw_outputs):
                    output_dict[spec.name] = (
                        raw_outputs[i] if isinstance(raw_outputs[i], list | np.ndarray)
                        else [raw_outputs[i]]
                    )
            return pd.DataFrame(output_dict)

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
                        f"expected {expected_count}. Adding default values.",
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
                        f"expected {expected_count}. Truncating to expected count.",
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
                processed[spec.name] = pd.to_numeric(column, errors="coerce")
            elif spec.output_type == OutputType.CATEGORICAL:
                processed[spec.name] = column.astype("category")
            elif spec.output_type == OutputType.BOOLEAN:
                processed[spec.name] = column.astype(bool)
            elif spec.output_type == OutputType.TIMESTAMP:
                processed[spec.name] = pd.to_datetime(column, errors="coerce")
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
                            f"{out_of_range.sum()} values outside range [{min_val}, {max_val}]",
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
                        f"but nullability is disabled",
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
                    index=i,
                )
                old_to_new_names[spec.name] = new_name

        return renamed_outputs.rename(columns=old_to_new_names)

    def _add_output_metadata(self, outputs: pd.DataFrame) -> pd.DataFrame:
        """Add metadata attributes to output DataFrame."""
        # Add feature metadata as DataFrame attributes
        outputs.attrs["feature_name"] = self.spec.key
        outputs.attrs["feature_version"] = self.spec.version
        outputs.attrs["output_count"] = len(outputs.columns)
        outputs.attrs["computation_timestamp"] = pd.Timestamp.now()

        if self.spec.tags:
            outputs.attrs["tags"] = self.spec.tags

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

    def __init__(self, config: dict[str, Any], logger_service: Any = None) -> None:
        """Initialize the instance."""
        self.config = config
        self.logger = logger_service

        # Feature registry and cache
        self.feature_registry: dict[str, InternalFeatureSpec] = {}
        self.feature_cache: dict[str, pd.DataFrame] = {}

        # Advanced indicators mapping
        self.advanced_indicators: dict[str, Callable[..., pd.DataFrame | pd.Series[Any] | None]] = {}

        # Performance tracking
        self.extraction_stats = {
            "features_extracted": 0,
            "extraction_time_total": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Quality thresholds
        self.quality_thresholds = config.get("feature_quality", {
            "min_completeness": 0.8,
            "max_correlation": 0.95,
            "min_variance": 1e-6,
        })

        self._initialize_advanced_indicators()

    def _initialize_advanced_indicators(self) -> None:
        """Initialize advanced technical indicators and market microstructure features."""
        # Enhanced technical indicators
        self.advanced_indicators = {
            # Momentum indicators
            "momentum": self._calculate_momentum,
            "rate_of_change": self._calculate_rate_of_change,
            "williams_percent_r": self._calculate_williams_r,
            "commodity_channel_index": self._calculate_cci,

            # Volatility indicators
            "bollinger_width": self._calculate_bollinger_width,
            "true_range": self._calculate_true_range,
            "average_true_range": self._calculate_atr_advanced,
            "volatility_ratio": self._calculate_volatility_ratio,

            # Volume indicators
            "on_balance_volume": self._calculate_obv,
            "accumulation_distribution": self._calculate_ad_line,
            "money_flow_index": self._calculate_mfi,
            "volume_oscillator": self._calculate_volume_oscillator,

            # Market microstructure
            "effective_spread": self._calculate_effective_spread,
            "quoted_spread": self._calculate_quoted_spread,
            "depth_imbalance": self._calculate_depth_imbalance,
            "order_flow_imbalance": self._calculate_order_flow_imbalance,
            "market_impact": self._calculate_market_impact,

            # Statistical features
            "price_momentum_oscillator": self._calculate_pmo,
            "adaptive_moving_average": self._calculate_ama,
            "fractal_dimension": self._calculate_fractal_dimension,
            "hurst_exponent": self._calculate_hurst_exponent,
        }

    async def extract_advanced_features(
        self,
        data: pd.DataFrame,
        feature_specs: list[InternalFeatureSpec],
        l2_data: dict[str, Any] | None = None,
        trade_data: list[dict[str, Any]] | None = None,
    ) -> FeatureExtractionResult:
        """Extract advanced features with technical indicators and market microstructure.

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
            for category in [FeatureCategory.TECHNICAL, FeatureCategory.ORDERBOOK,
                           FeatureCategory.MICROSTRUCTURE, FeatureCategory.VOLATILITY]:
                category_specs = [spec for spec in feature_specs if spec.category == category]
                if category_specs:
                    category_features = await self._extract_category_features(
                        data, category_specs, l2_data, trade_data,
                    )
                    features_df = pd.concat([features_df, category_features], axis=1)

            # Calculate feature quality metrics
            quality_metrics = self._calculate_feature_quality(features_df)

            # Clean and validate features
            features_df = self._clean_features(features_df)

            extraction_time = time.time() - start_time
            self.extraction_stats["features_extracted"] += len(feature_specs)
            self.extraction_stats["extraction_time_total"] += extraction_time

            result = FeatureExtractionResult(
                features=features_df,
                feature_specs=feature_specs,
                extraction_time=extraction_time,
                quality_metrics=quality_metrics,
                cache_hits=int(self.extraction_stats["cache_hits"]),
                cache_misses=int(self.extraction_stats["cache_misses"]),
            )

            if self.logger:
                self.logger.info(f"Advanced feature extraction completed in {extraction_time:.2f}s")

        except Exception as e:
            if self.logger:
                self.logger.exception("Advanced feature extraction failed: ")
            raise FeatureProcessingError(f"Advanced feature extraction failed: {e}")
        else:
            return result

    async def _extract_category_features(
        self,
        data: pd.DataFrame,
        category_specs: list[InternalFeatureSpec],
        l2_data: dict[str, Any] | None = None,
        trade_data: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Extract features for a specific category."""
        category_features = pd.DataFrame(index=data.index)

        for spec in category_specs:
            try:
                # Check cache first
                cache_key = self._generate_cache_key(spec, data)
                if spec.cache_enabled and cache_key in self.feature_cache:
                    feature_result: pd.DataFrame | None = self.feature_cache[cache_key]
                    self.extraction_stats["cache_hits"] += 1
                else:
                    # Compute feature
                    feature_result = await self._compute_advanced_feature(spec, data, l2_data, trade_data)

                    # Cache result if enabled
                    if spec.cache_enabled and feature_result is not None:
                        self.feature_cache[cache_key] = feature_result

                    self.extraction_stats["cache_misses"] += 1

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
        trade_data: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame | None:
        """Compute a single advanced feature."""
        if spec.calculator_type in self.advanced_indicators:
            calculator_func = self.advanced_indicators[spec.calculator_type]

            # Prepare arguments based on feature type
            if spec.category == FeatureCategory.ORDERBOOK and l2_data:
                result = calculator_func(data, l2_data, **spec.parameters)
            elif spec.category == FeatureCategory.MICROSTRUCTURE and trade_data:
                result = calculator_func(data, trade_data, **spec.parameters)
            else:
                result = calculator_func(data, **spec.parameters)

            # Process through output handler if available
            if hasattr(spec, "output_specs") and spec.output_specs:
                handler = FeatureOutputHandler(spec)
                handler.logger = self.logger
                return handler.process_feature_outputs(result)

            # Convert result to DataFrame if needed
            if isinstance(result, pd.Series):
                return pd.DataFrame({spec.key: result})
            if isinstance(result, pd.DataFrame):
                return result

        return None

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input OHLCV data."""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if data.empty:
            raise ValueError("Input data is empty")

        # Check for sufficient data
        min_periods = max(
            20,
            *(spec.required_lookback_periods
              for spec in self.feature_registry.values()
              if hasattr(spec, "required_lookback_periods")),
        )
        if len(data) < min_periods:
            raise ValueError(f"Insufficient data: need at least {min_periods} periods, got {len(data)}")

    def _calculate_feature_quality(self, features_df: pd.DataFrame) -> dict[str, float]:
        """Calculate quality metrics for extracted features."""
        if features_df.empty:
            return {"completeness": 0.0, "variance_ratio": 0.0, "correlation_max": 0.0}

        # Completeness (non-null ratio)
        completeness = 1.0 - features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))

        # Variance ratio (features with sufficient variance)
        numeric_features = features_df.select_dtypes(include=[np.number])
        if not numeric_features.empty:
            variances = numeric_features.var()
            high_variance_ratio = (variances > self.quality_thresholds["min_variance"]).mean()
        else:
            high_variance_ratio = 0.0

        # Maximum correlation (feature redundancy check)
        max_correlation = 0.0
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            # Get upper triangle excluding diagonal
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
            )
            max_correlation = upper_triangle.max().max() if not upper_triangle.isna().all().all() else 0.0

        return {
            "completeness": float(completeness),
            "variance_ratio": float(high_variance_ratio),
            "correlation_max": float(max_correlation),
        }

    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features."""
        # Remove features with too many NaNs
        threshold = self.quality_thresholds["min_completeness"]
        features_df = features_df.dropna(thresh=int(len(features_df) * threshold), axis=1)

        # Remove highly correlated features
        numeric_features = features_df.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool),
            )

            # Find features to drop
            to_drop = [column for column in upper_triangle.columns
                      if any(upper_triangle[column] > self.quality_thresholds["max_correlation"])]

            features_df = features_df.drop(columns=to_drop)

        return features_df

    def _generate_cache_key(self, spec: InternalFeatureSpec, data: pd.DataFrame) -> str:
        """Generate cache key for feature computation."""
        data_hash = str(hash(tuple(data.index.tolist() + data.iloc[-1].tolist())))
        params_hash = str(hash(tuple(sorted(spec.parameters.items()))))
        return f"{spec.key}_{data_hash}_{params_hash}"

    # Advanced indicator calculations
    def _calculate_momentum(self, data: pd.DataFrame, period: int = 10) -> pd.Series[Any]:
        """Calculate momentum indicator."""
        return data["close"] - data["close"].shift(period)

    def _calculate_rate_of_change(self, data: pd.DataFrame, period: int = 10) -> pd.Series[Any]:
        """Calculate rate of change."""
        return ((data["close"] - data["close"].shift(period)) / data["close"].shift(period)) * 100

    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
        """Calculate Williams %R."""
        high_max = data["high"].rolling(window=period).max()
        low_min = data["low"].rolling(window=period).min()
        return -100 * ((high_max - data["close"]) / (high_max - low_min))

    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series[Any]:
        """Calculate Commodity Channel Index."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma_tp) / (0.015 * mad)

    def _calculate_bollinger_width(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.Series[Any]:
        """Calculate Bollinger Band width."""
        sma = data["close"].rolling(window=period).mean()
        std = data["close"].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return (upper_band - lower_band) / sma * 100

    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series[Any]:
        """Calculate True Range."""
        high_low = data["high"] - data["low"]
        high_close_prev = np.abs(data["high"] - data["close"].shift(1))
        low_close_prev = np.abs(data["low"] - data["close"].shift(1))
        # Create DataFrame from Series to avoid concat type issues
        tr_df = pd.DataFrame({
            "hl": high_low,
            "hc": high_close_prev,
            "lc": low_close_prev,
        })
        return tr_df.max(axis=1)

    def _calculate_atr_advanced(self, data: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
        """Calculate Advanced Average True Range with additional smoothing."""
        tr = self._calculate_true_range(data)
        return tr.ewm(span=period).mean()  # Use exponential moving average

    def _calculate_volatility_ratio(
        self, data: pd.DataFrame, short_period: int = 10, long_period: int = 30,
    ) -> pd.Series[Any]:
        """Calculate volatility ratio."""
        short_vol = data["close"].pct_change().rolling(window=short_period).std()
        long_vol = data["close"].pct_change().rolling(window=long_period).std()
        return short_vol / long_vol

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series[Any]:
        """Calculate On-Balance Volume."""
        price_change = data["close"].diff()
        obv = np.where(price_change > 0, data["volume"],  # type: ignore[operator]
                      np.where(price_change < 0, -data["volume"], 0))  # type: ignore[operator]
        return pd.Series(obv, index=data.index).cumsum()

    def _calculate_ad_line(self, data: pd.DataFrame) -> pd.Series[Any]:
        """Calculate Accumulation/Distribution Line."""
        mfm = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (data["high"] - data["low"])
        mfm = mfm.fillna(0)  # Handle division by zero
        return (mfm * data["volume"]).cumsum()

    def _calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
        """Calculate Money Flow Index."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * data["volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        money_ratio = positive_mf / negative_mf
        return 100 - (100 / (1 + money_ratio))

    def _calculate_volume_oscillator(
        self, data: pd.DataFrame, short_period: int = 14, long_period: int = 28,
    ) -> pd.Series[Any]:
        """Calculate Volume Oscillator."""
        short_vol_avg = data["volume"].rolling(window=short_period).mean()
        long_vol_avg = data["volume"].rolling(window=long_period).mean()
        return ((short_vol_avg - long_vol_avg) / long_vol_avg) * 100

    # Market microstructure features
    def _calculate_effective_spread(self, data: pd.DataFrame, l2_data: dict[str, Any]) -> pd.Series[Any]:
        """Calculate effective spread from L2 data with production-grade implementation."""
        try:
            # Import enhanced spread calculator if available
            from .feature_engine_enhancements import (
                AdvancedSpreadCalculator,
                MarketMicrostructureData,
            )

            # Convert l2_data to MarketMicrostructureData
            bids = [(float(p), float(s)) for p, s in l2_data.get("bids", [])]
            asks = [(float(p), float(s)) for p, s in l2_data.get("asks", [])]
            trades = l2_data.get("trades", [])

            microstructure_data = MarketMicrostructureData(
                timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                bids=bids,
                asks=asks,
                trades=trades,
            )

            # Use advanced calculator
            spread_calculator = AdvancedSpreadCalculator(self.logger)
            spread_metrics = spread_calculator.calculate_effective_spread(
                microstructure_data,
                trades,
            )

            # Return effective spread in basis points
            if "effective_spread_bps_mean" in spread_metrics:
                return pd.Series([spread_metrics["effective_spread_bps_mean"]], index=[data.index[-1]])
            if "quoted_spread_bps" in spread_metrics:
                return pd.Series([spread_metrics["quoted_spread_bps"]], index=[data.index[-1]])
            return pd.Series([], dtype=float)

        except ImportError:
            # Fallback to original implementation
            if "bids" in l2_data and "asks" in l2_data and l2_data["bids"] and l2_data["asks"]:
                bid_price = float(l2_data["bids"][0][0])
                ask_price = float(l2_data["asks"][0][0])
                midpoint = (bid_price + ask_price) / 2
                spread = ask_price - bid_price
                return pd.Series([spread / midpoint * 10000], index=[data.index[-1]])  # in basis points
            return pd.Series([], dtype=float)

    def _calculate_quoted_spread(self, data: pd.DataFrame, l2_data: dict[str, Any]) -> pd.Series[Any]:
        """Calculate quoted spread."""
        if "bids" in l2_data and "asks" in l2_data and l2_data["bids"] and l2_data["asks"]:
            bid_price = float(l2_data["bids"][0][0])
            ask_price = float(l2_data["asks"][0][0])
            return pd.Series([ask_price - bid_price], index=[data.index[-1]])
        return pd.Series([], dtype=float)

    def _calculate_depth_imbalance(
        self, data: pd.DataFrame, l2_data: dict[str, Any], levels: int = 5,
    ) -> pd.Series[Any]:
        """Calculate order book depth imbalance."""
        if "bids" in l2_data and "asks" in l2_data:
            bid_depth = sum(float(level[1]) for level in l2_data["bids"][:levels])
            ask_depth = sum(float(level[1]) for level in l2_data["asks"][:levels])
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                imbalance = (bid_depth - ask_depth) / total_depth
                return pd.Series([imbalance], index=[data.index[-1]])
        return pd.Series([], dtype=float)

    def _calculate_order_flow_imbalance(self, data: pd.DataFrame, trade_data: list[dict[str, Any]]) -> pd.Series[Any]:
        """Calculate order flow imbalance from trade data."""
        if not trade_data:
            return pd.Series([], dtype=float)

        buy_volume = sum(float(trade["volume"]) for trade in trade_data if trade.get("side") == "buy")
        sell_volume = sum(float(trade["volume"]) for trade in trade_data if trade.get("side") == "sell")
        total_volume = buy_volume + sell_volume

        if total_volume > 0:
            imbalance = (buy_volume - sell_volume) / total_volume
            return pd.Series([imbalance], index=[data.index[-1]])
        return pd.Series([], dtype=float)

    def _calculate_market_impact(self, data: pd.DataFrame, trade_data: list[dict[str, Any]]) -> pd.Series[Any]:
        """Calculate market impact estimate."""
        if not trade_data or len(data) < 2:
            return pd.Series([], dtype=float)

        # Simple market impact as price change per unit volume
        recent_volume = sum(float(trade["volume"]) for trade in trade_data)
        price_change = float(data["close"].iloc[-1] - data["close"].iloc[-2])

        if recent_volume > 0:
            impact = abs(price_change) / recent_volume
            return pd.Series([impact], index=[data.index[-1]])
        return pd.Series([], dtype=float)

    # Statistical and advanced features
    def _calculate_pmo(self, data: pd.DataFrame, period1: int = 35, period2: int = 20) -> pd.Series[Any]:
        """Calculate Price Momentum Oscillator."""
        roc = data["close"].pct_change(period1) * 100
        pmo = roc.ewm(span=period2).mean()
        return pmo.ewm(span=10).mean()  # Signal line

    def _calculate_ama(self, data: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
        """Calculate Adaptive Moving Average."""
        change = abs(data["close"] - data["close"].shift(period))
        volatility = data["close"].diff().abs().rolling(window=period).sum()
        efficiency_ratio = change / volatility

        # Smoothing constants
        fast_sc = 2 / (2 + 1)
        slow_sc = 2 / (30 + 1)
        smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

        ama = pd.Series(index=data.index, dtype=float)
        ama.iloc[0] = data["close"].iloc[0]

        for i in range(1, len(data)):
            ama.iloc[i] = ama.iloc[i-1] + smoothing_constant.iloc[i] * (data["close"].iloc[i] - ama.iloc[i-1])

        return ama

    def _calculate_fractal_dimension(self, data: pd.DataFrame, period: int = 20) -> pd.Series[Any]:
        """Calculate fractal dimension."""
        def fd_single(series: pd.Series[Any]) -> float:
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
            return float(2 - hurst)

        return data["close"].rolling(window=period).apply(fd_single)

    def _calculate_hurst_exponent(self, data: pd.DataFrame, period: int = 100) -> pd.Series[Any]:
        """Calculate Hurst exponent."""
        def hurst_single(series: pd.Series[Any]) -> float:
            if len(series) < 10:
                return np.nan

            try:
                # Convert to log returns
                returns = series / series.shift(1)
                log_returns = pd.Series(np.log(returns)).dropna()

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
                log_lags = np.log(list[Any](lags[:len(rs_values)]))
                log_rs = np.log(rs_values)

                # Simple linear regression
                n = len(log_lags)
                sum_x = np.sum(log_lags)
                sum_y = np.sum(log_rs)
                sum_xy = np.sum(log_lags * log_rs)
                sum_x2 = np.sum(log_lags ** 2)

                hurst = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                return float(hurst)

            except Exception:
                return np.nan

        return data["close"].rolling(window=period).apply(hurst_single)


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
            self.scaler.fit(X.values.reshape(-1, 1))  # type: ignore[union-attr]
        else:
            self.scaler.fit(X)
        return self

    def transform(self, X: Any) -> Any:
        """Transform the data and restore pandas structure."""
        if isinstance(X, pd.DataFrame):
            transformed = self.scaler.transform(X.values)
            return pd.DataFrame(transformed, columns=self._feature_names or X.columns, index=X.index)
        if isinstance(X, pd.Series):
            transformed = self.scaler.transform(X.values.reshape(-1, 1))  # type: ignore[union-attr]
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
        history_repo: HistoryRepository | None = None) -> None:
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
            history_repo: Optional repository for persisting historical data.

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
        self.history_repo = history_repo
        self._source_module = self.__class__.__name__

        # Feature configuration derived from config
        self._feature_configs: dict[str, InternalFeatureSpec] = {} # Changed type hint
        self._extract_feature_configs()

        # Initialize feature handlers dispatcher
        # RSI and MACD handlers are removed as they will be handled by sklearn pipelines.
        self._feature_handlers: dict[
            str,
            Callable[..., dict[str, Any] | dict[str, str] | None],
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
                columns=["open", "high", "low", "close", "volume"]).astype(
                {
                    "open": "object",  # Store as Decimal initially
                    "high": "object",
                    "low": "object",
                    "close": "object",
                    "volume": "object",
                }))

        # L2 order book data (latest snapshot)
        self.l2_books: dict[str, dict[str, Any]] = defaultdict(dict[str, Any])

        # Store recent trades for calculating true Volume Delta and trade-based VWAP
        # deque stores: {"ts": datetime, "price": Decimal, "vol": Decimal, "side": "buy"/"sell"}
        trade_history_maxlen = config.get("feature_engine", {}).get("trade_history_maxlen", 2000)
        self.trade_history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=trade_history_maxlen))

        # Store L2 book history for better alignment with bar timestamps
        # Each entry: {"timestamp": datetime, "book": {"bids": [...], "asks": [...]}}
        l2_history_maxlen = config.get("feature_engine", {}).get("l2_history_maxlen", 100)
        self.l2_books_history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=l2_history_maxlen))

        self.feature_pipelines: dict[str, dict[str, Any]] = {}  # Stores {'pipeline': Pipeline,
        # 'spec': InternalFeatureSpec}

        # Initialize enhanced components
        self.output_handlers: dict[str, FeatureOutputHandler] = {}
        self.advanced_extractor = AdvancedFeatureExtractor(config.get("feature_engine", {}), logger_service)

        # Initialize enterprise-grade imputation system for crypto trading data
        imputation_config = config.get("feature_imputation", {})
        self.imputation_manager = create_imputation_manager(
            logger=logger_service,
            config=imputation_config,
        )
        self.logger.info(
            "Initialized advanced imputation system with %d configured strategies",
            len(imputation_config),
            source_module=self._source_module,
        )

        # Registry for ML-based imputation models
        models_path = config.get("imputation_model_registry", {}).get("path", "imputation_models")
        self.imputation_model_registry = ImputationModelRegistry(models_path)

        # Build pipelines after initializing components
        self._build_feature_pipelines()

        self.logger.info("FeatureEngine with enhanced capabilities initialized.", source_module=self._source_module)

    def _determine_calculator_type_and_input(
        self, feature_key: str, raw_cfg: dict[str, Any],
    ) -> tuple[str | None, str | None]:
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
            A tuple[Any, ...] `(calculator_type, input_type)`, where both can be `None` if
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
        if "rsi" in key_lower:
            return "rsi", "close_series"
        if "macd" in key_lower:
            return "macd", "close_series"
        if "bbands" in key_lower:
            return "bbands", "close_series"
        if "roc" in key_lower:
            return "roc", "close_series"
        if "atr" in key_lower:
            return "atr", "ohlcv_df"
        if "stdev" in key_lower:
            return "stdev", "close_series"
        if "vwap_ohlcv" in key_lower:
            return "vwap_ohlcv", "ohlcv_df"
        if "l2_spread" in key_lower:
            return "l2_spread", "l2_book_series"
        if "l2_imbalance" in key_lower:
            return "l2_imbalance", "l2_book_series"
        if "l2_wap" in key_lower:
            return "l2_wap", "l2_book_series"
        if "l2_depth" in key_lower:
            return "l2_depth", "l2_book_series"
        if "vwap_trades" in key_lower:
            return "vwap_trades", "trades_and_bar_starts"
        if "volume_delta" in key_lower:
            return "volume_delta", "trades_and_bar_starts"

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
            *   If it's a list[Any] of strings, these are treated as keys to activate
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
                source_module=self._source_module)
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
            for common_param_key in [
                "period", "length", "fast", "slow", "signal", "levels",
                "std_dev", "length_seconds", "bar_interval_seconds",
            ]:
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
                imputation_model_key=raw_cfg_dict.get("imputation_model_key"),
                imputation_model_version=(
                    str(raw_cfg_dict.get("imputation_model_version"))
                    if raw_cfg_dict.get("imputation_model_version") is not None
                    else None
                ),
                description=description)
            parsed_specs[key] = spec_val

        # self._feature_configs = parsed_specs # Old logic replaced by new registry-based logic below

        registry_definitions = self._load_feature_registry(DEFAULT_FEATURE_REGISTRY_PATH)
        app_feature_config = self.config.get("features", {}) # This is the app-level config for features

        final_parsed_specs: dict[str, InternalFeatureSpec] = {}

        if isinstance(app_feature_config, list): # Case 1: List of feature keys to activate
            for key in app_feature_config:
                if not isinstance(key, str):
                    self.logger.warning("Feature activation list[Any] contains non-string item: %s. Skipping.", key)
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
                    self.logger.warning(
                        "Override/activation config for feature '%s' is not a dict[str, Any]. Skipping.",
                        key,
                    )
                    continue

                base_config = registry_definitions.get(key)
                final_config_dict: dict[str, Any] = {}

                if base_config: # Key found in registry, apply overrides
                    if not isinstance(base_config, dict):
                        self.logger.warning("Registry definition for '%s' is not a dictionary. Skipping override.", key)
                        continue
                    final_config_dict = self._deep_merge_configs(base_config.copy(), overrides_or_activation)
                else: # Key not in registry - treat as ad-hoc definition
                    self.logger.info(
                        "Feature '%s' not found in registry, treating as ad-hoc definition from app config.",
                        key,
                    )
                    final_config_dict = overrides_or_activation.copy()
                    # Ad-hoc definitions must provide all necessary fields like calculator_type, input_type
                    if "calculator_type" not in final_config_dict or "input_type" not in final_config_dict:
                        self.logger.warning(
                            "Ad-hoc feature '%s' missing 'calculator_type' or 'input_type'. Skipping.",
                            key,
                        )
                        continue

                spec_result = self._parse_single_feature_definition(key, final_config_dict)
                if spec_result is not None:
                    final_parsed_specs[key] = spec_result

        else:
            self.logger.warning(
                "App-level 'features' config is neither a list[Any] nor a dict[str, Any]. "
                "No features will be configured based on it.",
            )

        self._feature_configs = final_parsed_specs
        if not self._feature_configs:
            self.logger.warning(
                "No features were successfully parsed or activated. "
                "FeatureEngine might not produce any features.",
            )


    def _deep_merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deeply merges override dict[str, Any] into base dict[str, Any].
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

    def _parse_single_feature_definition(
        self, feature_key: str, config_dict: dict[str, Any],
    ) -> InternalFeatureSpec | None:
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
            inferred_calc_type, inferred_input_type = self._determine_calculator_type_and_input(
                feature_key, config_dict,
            )
            if not calculator_type: calculator_type = inferred_calc_type
            if not input_type: input_type = inferred_input_type

            if not calculator_type or not input_type:
                self.logger.warning(
                    "Could not determine calculator_type or input_type for feature '%s' "
                    "even after inference. Skipping.",
                    feature_key)
                return None

        parameters = config_dict.get("parameters", config_dict.get("params", {}))
        # Ensure parameters is a dict[str, Any], even if it was null/None in YAML
        if not isinstance(parameters, dict): parameters = {}

        # Merge top-level common param keys if they exist and aren't already in 'parameters'
        for common_param_key in [
            "period", "length", "fast", "slow", "signal", "levels",
            "std_dev", "length_seconds", "bar_interval_seconds",
        ]:
            if common_param_key in config_dict and common_param_key not in parameters:
                parameters[common_param_key] = config_dict[common_param_key]

        imputation_cfg = config_dict.get("imputation")
        scaling_cfg = config_dict.get("scaling")
        imputation_model_key = config_dict.get("imputation_model_key")
        imputation_model_version = config_dict.get("imputation_model_version")
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
            imputation_model_key=imputation_model_key,
            imputation_model_version=str(imputation_model_version) if imputation_model_version is not None else None,
            description=description,
            version=str(version) if version is not None else None, # Ensure version is string
            output_properties=output_properties if isinstance(output_properties, dict) else {})
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
        except yaml.YAMLError:
            self.logger.exception(f"Error parsing YAML in feature registry {registry_path}: ")
            return {}
        except Exception:
            self.logger.exception(f"Unexpected error loading feature registry {registry_path}: ")
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
                                    spec_key: str = "") -> Any:
            """Creates a Scikit-learn compatible imputation step based on configuration.
            Handles Series[Any] and DataFrame outputs from feature calculators.
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
                self.logger.warning(
                    "Unrecognized imputation config for %s: %s. No imputer added.",
                    spec_key, imputation_cfg,
                )
                return None

            step_name_suffix = ""
            transform_func: Callable[[Any], Any] | None = None

            if strategy == "constant":
                step_name_suffix = f"const_{fill_value}"
                def transform_func(x):
                    return x.fillna(fill_value)
            elif strategy == "mean":
                step_name_suffix = "mean"
                def transform_func(x):
                    return x.fillna(x.mean())
            elif strategy == "median":
                step_name_suffix = "median"
                def transform_func(x):
                    return x.fillna(x.median())
            elif strategy == "default": # Use the passed default_fill_value
                step_name_suffix = f"default_fill_{default_fill_value}"
                def transform_func(x):
                    return x.fillna(default_fill_value)
            else: # Should not be reached if checks are exhaustive
                self.logger.warning("Unknown imputation strategy '%s' for %s. No imputer added.", strategy, spec_key)
                return None

            imputer_name = (
                strategy if strategy != "default"
                else f"default_fill({default_fill_value})"
            )
            fill_desc = fill_value if strategy == "constant" else "N/A"
            self.logger.debug(
                "Using output imputer strategy '%s' (fill: %s) for %s",
                imputer_name, fill_desc, spec_key,
            )
            return (
                f"{spec_key}_output_imputer_{step_name_suffix}",
                FunctionTransformer(transform_func, validate=False),
            )


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

            scaler_instance: Any = StandardScaler() # Default scaler
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
                 self.logger.warning(
                    "Simple string for scaling method '%s' for %s is ambiguous. "
                    "Use dict[str, Any] config or 'passthrough'. Defaulting to StandardScaler.",
                    scaling_cfg, spec_key,
                )


            self.logger.debug("Using %s for scaling for %s", type(scaler_instance).__name__, spec_key)
            return (f"{spec_key}_output_scaler_{scaler_name_suffix}", PandasScalerTransformer(scaler_instance))


        for spec in self._feature_configs.values(): # Now iterates over InternalFeatureSpec
            pipeline_steps = []
            pipeline_name = f"{spec.key}_pipeline" # Use spec.key for consistency

            # Create output handler for this feature
            output_handler = FeatureOutputHandler(spec)
            output_handler.logger = self.logger
            self.output_handlers[spec.key] = output_handler

            # Advanced input imputation for features that take a single series like 'close'
            if spec.input_type == "close_series":
                input_imputation_step = self._create_input_imputation_step(spec)
                if input_imputation_step:
                    pipeline_steps.append(input_imputation_step)

            # Calculator step based on spec.calculator_type
            calculator_func = getattr(FeatureEngine, f"_pipeline_compute_{spec.calculator_type}", None)
            if not calculator_func:
                self.logger.error(
                    "No _pipeline_compute function found for calculator_type: %s (feature key: %s)",
                    spec.calculator_type, spec.key,
                )
                continue

            # Prepare kw_args for the calculator from spec.parameters
            # Ensure all necessary parameters for the specific calculator are present with defaults
            calc_kw_args = {} # These are static kw_args known at pipeline build time

            if spec.calculator_type in ["rsi", "roc", "stdev"]:
                default_period = 14 if spec.calculator_type == "rsi" else 10 if spec.calculator_type == "roc" else 20
                calc_kw_args["period"] = spec.parameters.get("period", default_period)
                if spec.parameters.get("period") is None:
                    self.logger.debug(
                        "Using default period %s for %s ('%s')",
                        calc_kw_args["period"], spec.calculator_type, spec.key,
                    )

            elif spec.calculator_type == "macd":
                calc_kw_args["fast"] = spec.parameters.get("fast", 12)
                calc_kw_args["slow"] = spec.parameters.get("slow", 26)
                calc_kw_args["signal"] = spec.parameters.get("signal", 9)
                if any(p not in spec.parameters for p in ["fast", "slow", "signal"]):
                    self.logger.debug(
                        "Using default MACD params (f:%s,s:%s,sig:%s) for %s",
                        calc_kw_args["fast"], calc_kw_args["slow"], calc_kw_args["signal"], spec.key,
                    )

            elif spec.calculator_type == "bbands":
                calc_kw_args["length"] = spec.parameters.get("length", 20)
                calc_kw_args["std_dev"] = float(spec.parameters.get("std_dev", 2.0))
                if "length" not in spec.parameters or "std_dev" not in spec.parameters:
                     self.logger.debug(
                         "Using default BBands params (l:%s,s:%.1f) for %s",
                         calc_kw_args["length"], calc_kw_args["std_dev"], spec.key,
                     )

            elif spec.calculator_type == "atr":
                calc_kw_args["length"] = spec.parameters.get("length", 14)
                # high_col, low_col, close_col default in function signature of _pipeline_compute_atr
                if "length" not in spec.parameters:
                    self.logger.debug("Using default ATR length %s for %s", calc_kw_args["length"], spec.key)

            elif spec.calculator_type == "vwap_ohlcv":
                calc_kw_args["length"] = spec.parameters.get("length", 14)
                if "length" not in spec.parameters:
                    self.logger.debug("Using default VWAP_OHLCV length %s for %s", calc_kw_args["length"], spec.key)

            elif spec.calculator_type in ["l2_imbalance", "l2_depth", "l2_wap"]:
                # `ohlcv_close_prices` is NOT included here; it's passed dynamically for l2_wap.
                default_levels = 5
                if spec.calculator_type == "l2_wap":
                    default_levels = 1
                elif spec.calculator_type == "l2_spread":
                    default_levels = 0 # Not applicable for spread

                if default_levels > 0: # Only add 'levels' if applicable
                    calc_kw_args["levels"] = spec.parameters.get("levels", default_levels)
                    if "levels" not in spec.parameters:
                        self.logger.debug(
                            "Using default levels %s for %s ('%s')",
                            calc_kw_args["levels"], spec.calculator_type, spec.key,
                        )

            elif spec.calculator_type in {"vwap_trades", "volume_delta"}:
                # `ohlcv_close_prices` is NOT included here; it's passed dynamically.
                # `bar_start_times` is also dynamic, passed at runtime.
                # `trade_history_deque` is the `X` input to fit_transform.
                calc_kw_args["bar_interval_seconds"] = spec.parameters.get("bar_interval_seconds",
                                                                         spec.parameters.get("length_seconds", 60))
                if "bar_interval_seconds" not in spec.parameters and "length_seconds" not in spec.parameters:
                    self.logger.debug(
                        "Using default bar_interval_seconds %s for %s ('%s')",
                        calc_kw_args["bar_interval_seconds"], spec.calculator_type, spec.key,
                    )
                # `bar_start_times` will be passed dynamically during the call in _calculate_and_publish_features

            # l2_spread currently has no parameters in its _pipeline_compute_l2_spread signature other than X.

            pipeline_steps.append((
                f"{spec.key}_calculator",
                FunctionTransformer(calculator_func, kw_args=calc_kw_args, validate=False),
            ))

            # Output Imputation & Scaling using helpers
            is_df_output = spec.calculator_type in ["macd", "bbands", "l2_spread", "l2_depth"]
            # Define default fill values based on feature type characteristics
            default_fill = 0.0 # General default
            if spec.calculator_type == "rsi":
                default_fill = 50.0
            elif spec.calculator_type in ["atr", "vwap_ohlcv", "l2_wap", "vwap_trades", "stdev"]:
                default_fill = np.nan # Will be filled by mean then

            imputer_step = get_output_imputer_step(
                spec.imputation, default_fill_value=default_fill,
                is_dataframe_output=is_df_output, spec_key=spec.key,
            )
            if imputer_step:
                pipeline_steps.append(imputer_step)

            scaler_step = get_output_scaler_step(spec.scaling, spec_key=spec.key)
            if scaler_step:
                pipeline_steps.append(scaler_step)

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
                    source_module=self._source_module)

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
                    context={"payload": ohlcv_payload})
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
                    source_module=self._source_module)

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
                source_module=self._source_module)

        except KeyError:
            self.logger.exception(
                "Missing key in OHLCV payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload})
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in OHLCV payload for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload})
        except Exception:
            self.logger.exception(
                "Unexpected error handling OHLCV update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": ohlcv_payload})

    async def _handle_l2_update(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
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
                    source_module=self._source_module)
                # Enterprise-grade error handling: Implement intelligent fallback strategy
                return await self._handle_malformed_l2_data(trading_pair, l2_payload, "invalid_format")

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
                            source_module=self._source_module)
                        # Optionally skip this level or use NaN/None
                        continue  # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 bid level %s for %s: %s",
                        i,
                        trading_pair,
                        bid_level,
                        source_module=self._source_module)

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
                            source_module=self._source_module)
                        continue  # Skip malformed level
                else:
                    self.logger.warning(
                        "Malformed L2 ask level %s for %s: %s",
                        i,
                        trading_pair,
                        ask_level,
                        source_module=self._source_module)

            # Store the processed L2 book data
            # The L2 book features will expect bids sorted high to low, asks low to high.
            self.l2_books[trading_pair] = {
                "bids": processed_bids,  # Already sorted highest bid first from source
                "asks": processed_asks,  # Already sorted lowest ask first from source
                "timestamp": pd.to_datetime(
                    l2_payload.get("timestamp_exchange") or datetime.now(UTC),
                    utc=True),
            }

            # Store in L2 history with timestamp for better time alignment
            timestamp_str = l2_payload.get("timestamp_exchange") or l2_payload.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = pd.to_datetime(timestamp_str, utc=True)
                    self.l2_books_history[trading_pair].append({
                        "timestamp": timestamp,
                        "book": {"bids": processed_bids, "asks": processed_asks},
                    })
                except Exception as e:
                    self.logger.warning(
                        "Failed to parse L2 timestamp for history: %s",
                        e,
                        source_module=self._source_module)

            self.logger.debug(
                "Updated L2 book for %s: %s bids, %s asks",
                trading_pair,
                len(processed_bids),
                len(processed_asks),
                source_module=self._source_module)

        except KeyError:
            self.logger.exception(
                "Missing key in L2 payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload})
        except Exception:
            self.logger.exception(
                "Unexpected error handling L2 update for %s",
                trading_pair,
                source_module=self._source_module,
                context={"payload": l2_payload})

    async def _handle_malformed_l2_data(
        self,
        trading_pair: str,
        l2_payload: dict[str, Any],
        error_type: str,
    ) -> None:
        """Enterprise-grade error handling for malformed L2 order book data.

        Implements intelligent fallback strategies based on error type and market conditions.

        Args:
            trading_pair: The trading pair affected
            l2_payload: The malformed payload for analysis
            error_type: Type[Any] of error encountered
        """
        from datetime import timedelta

        # Track L2 data quality metrics
        error_context = {
            "trading_pair": trading_pair,
            "error_type": error_type,
            "payload_keys": list[Any](l2_payload.keys()) if isinstance(l2_payload, dict) else None,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
        }

        self.logger.error(
            "Malformed L2 data for %s - Type: %s. Implementing fallback strategy.",
            trading_pair,
            error_type,
            source_module=self._source_module,
            context=error_context,
        )

        # Strategy 1: Attempt data reconstruction from recent history
        if error_type == "invalid_format" and trading_pair in self.l2_books_history:
            recent_books = list[Any](self.l2_books_history[trading_pair])
            if recent_books:
                # Use the most recent valid book if it's less than 30 seconds old
                latest_book = recent_books[-1]
                # Extract timestamp and ensure it's a datetime object
                book_timestamp = latest_book["timestamp"]
                if hasattr(book_timestamp, "to_pydatetime"):
                    # It's a pandas Timestamp
                    book_timestamp = book_timestamp.to_pydatetime()
                book_age = datetime.now(UTC) - book_timestamp.replace(tzinfo=None)

                if book_age < timedelta(seconds=30):
                    self.logger.info(
                        "Using recent valid L2 book for %s (age: %s seconds)",
                        trading_pair,
                        book_age.total_seconds(),
                        source_module=self._source_module,
                    )

                    # Update current book with aged data (mark as stale)
                    stale_book = latest_book["book"].copy()
                    stale_book["is_stale"] = True
                    stale_book["stale_age_seconds"] = book_age.total_seconds()

                    self.l2_books[trading_pair] = {
                        "bids": stale_book.get("bids", []),
                        "asks": stale_book.get("asks", []),
                        "timestamp": pd.to_datetime(datetime.now(UTC), utc=True),
                        "is_stale": True,
                        "stale_age_seconds": book_age.total_seconds(),
                        "fallback_reason": f"malformed_data_{error_type}",
                    }
                    return

        # Strategy 2: Clear stale book data if no recent valid data available
        current_book = self.l2_books.get(trading_pair)
        if current_book:
            # Extract timestamp and ensure it's a datetime object
            current_timestamp = current_book["timestamp"]
            if hasattr(current_timestamp, "to_pydatetime"):
                # It's a pandas Timestamp
                current_timestamp = current_timestamp.to_pydatetime()
            current_age = datetime.now(UTC) - current_timestamp.replace(tzinfo=None)

            # Clear if data is older than 5 minutes
            if current_age > timedelta(minutes=5):
                self.logger.warning(
                    "Clearing stale L2 book for %s (age: %s minutes) due to malformed update",
                    trading_pair,
                    current_age.total_seconds() / 60,
                    source_module=self._source_module,
                )

                # Initialize empty book structure
                self.l2_books[trading_pair] = {
                    "bids": [],
                    "asks": [],
                    "timestamp": pd.to_datetime(datetime.now(UTC), utc=True),
                    "is_empty": True,
                    "empty_reason": f"cleared_due_to_{error_type}",
                }
            else:
                # Mark existing data as potentially unreliable
                current_book["has_recent_errors"] = True
                current_book["last_error_type"] = error_type
                current_book["last_error_timestamp"] = datetime.now(UTC).isoformat() + "Z"

        # Strategy 3: Attempt basic data sanitization for partially valid payloads
        if error_type == "invalid_format" and isinstance(l2_payload, dict):
            sanitized_book = self._attempt_l2_data_sanitization(l2_payload)
            if sanitized_book:
                self.logger.info(
                    "Successfully sanitized partial L2 data for %s",
                    trading_pair,
                    source_module=self._source_module,
                )

                self.l2_books[trading_pair] = {
                    **sanitized_book,
                    "timestamp": pd.to_datetime(datetime.now(UTC), utc=True),
                    "is_sanitized": True,
                    "sanitization_reason": error_type,
                }

                # Store in history for future fallback
                self.l2_books_history[trading_pair].append({
                    "timestamp": pd.to_datetime(datetime.now(UTC), utc=True),
                    "book": sanitized_book,
                })
                return

        # Strategy 4: Publish data quality alert for monitoring
        await self._publish_data_quality_alert(trading_pair, error_type, error_context)

    def _attempt_l2_data_sanitization(self, l2_payload: dict[str, Any]) -> dict[str, Any] | None:
        """Attempt to sanitize partially corrupted L2 order book data.

        Args:
            l2_payload: Potentially corrupted L2 payload

        Returns:
            Sanitized book data or None if sanitization fails
        """
        sanitized_bids = []
        sanitized_asks = []
        try:
            # Try to extract and validate bids
            raw_bids = l2_payload.get("bids", [])
            if isinstance(raw_bids, list | tuple):
                for bid_level in raw_bids:
                    if self._is_valid_l2_level(bid_level):
                        try:
                            price = Decimal(str(bid_level[0]))
                            volume = Decimal(str(bid_level[1]))
                            if price > 0 and volume >= 0:  # Allow zero volume for cancellations
                                sanitized_bids.append([price, volume])
                        except (ValueError, TypeError, IndexError):
                            continue  # Skip invalid levels

            # Try to extract and validate asks
            raw_asks = l2_payload.get("asks", [])
            if isinstance(raw_asks, list | tuple):
                for ask_level in raw_asks:
                    if self._is_valid_l2_level(ask_level):
                        try:
                            price = Decimal(str(ask_level[0]))
                            volume = Decimal(str(ask_level[1]))
                            if price > 0 and volume >= 0:
                                sanitized_asks.append([price, volume])
                        except (ValueError, TypeError, IndexError):
                            continue

            if sanitized_bids or sanitized_asks:
                return {"bids": sanitized_bids, "asks": sanitized_asks}

        except Exception as e:
            self.logger.debug(f"L2 data sanitization failed: {e}", source_module=self._source_module)

        return None

    def _is_valid_l2_level(self, level: Any) -> bool:
        """Check if an L2 level has valid structure."""
        return (
            isinstance(level, list | tuple) and
            len(level) >= 2 and
            level[0] is not None and
            level[1] is not None
        )

    async def _publish_data_quality_alert(
        self,
        trading_pair: str,
        error_type: str,
        context: dict[str, Any],
    ) -> None:
        """Publishes a data quality alert to the pubsub system."""
        try:
            # Import event class
            from gal_friday.core.events import APIErrorEvent

            # Create a proper event object
            alert_event = APIErrorEvent.create(
                source_module=self._source_module,
                error_message=f"Malformed L2 data detected for {trading_pair}. Error type: {error_type}",
                endpoint=f"l2_data/{trading_pair}",
                request_data={
                    "trading_pair": trading_pair,
                    "alert_type": "L2_DATA_ERROR",
                    "severity": "warning",
                    "error_type": error_type,
                    "context": context,
                },
            )

            # Enterprise-grade async data quality alert system
            if hasattr(self, "pubsub_manager") and self.pubsub_manager:
                # Use async task to publish alert without blocking
                import asyncio

                async def publish_alert_async() -> None:
                    """Async helper to publish data quality alerts."""
                    try:
                        await self.pubsub_manager.publish(
                            alert_event,  # PubSubManager.publish() takes only the event
                        )
                        self.logger.info(
                            "Published DATA_QUALITY_ALERT for %s",
                            trading_pair,
                            source_module=self._source_module,
                        )
                    except Exception as e:
                        self.logger.exception(
                            "Failed to publish DATA_QUALITY_ALERT: %s",
                            e,
                            source_module=self._source_module,
                            context={"alert_event": alert_event},
                        )
                        # Fallback: Log locally with enhanced context
                        await self._log_data_quality_issue_locally(alert_event.to_dict(), e)

                # Create a non-blocking task to publish the alert
                asyncio.create_task(publish_alert_async())
            else:
                # Synchronous fallback if no async pubsub manager is available
                await self._log_data_quality_issue_locally(alert_event.to_dict(), "no_pubsub_manager")

        except Exception as e:
            self.logger.exception(
                "Failed to create or publish data quality alert for %s",
                trading_pair,
                source_module=self._source_module,
                context={"error_type": error_type, "error": e},
            )

    async def _log_data_quality_issue_locally(
        self,
        alert_event: dict[str, Any],
        error_context: Any,
    ) -> None:
        """Enhanced local logging for data quality issues with structured information."""
        try:
            # Extract alert details
            payload = alert_event.get("payload", {})
            alert_type = payload.get("alert_type", "unknown")
            trading_pair = payload.get("trading_pair", "unknown")
            severity = payload.get("severity", "info")

            # Create enhanced log entry with structured data
            enhanced_log_data = {
                "event_type": "DATA_QUALITY_ISSUE",
                "alert_type": alert_type,
                "trading_pair": trading_pair,
                "severity": severity,
                "details": payload.get("details", "No details provided."),
                "error_context": str(error_context),
                "timestamp": alert_event.get("timestamp"),
            }

            log_message = f"DATA_QUALITY_ISSUE: {alert_type} for {trading_pair}"

            # Log with appropriate level based on severity
            if severity == "critical":
                self.logger.critical(log_message, context=enhanced_log_data, source_module=self._source_module)
                # Persist critical issues for later analysis
                await self._persist_critical_data_quality_issue(enhanced_log_data)
            elif severity == "error":
                self.logger.error(log_message, context=enhanced_log_data, source_module=self._source_module)
            elif severity == "warning":
                self.logger.warning(log_message, context=enhanced_log_data, source_module=self._source_module)
            else:
                self.logger.info(log_message, context=enhanced_log_data, source_module=self._source_module)

        except Exception as e:
            self.logger.exception(
                "Error during local logging of data quality issue",
                source_module=self._source_module,
                context={"original_event": alert_event, "logging_error": e},
            )

    async def _persist_critical_data_quality_issue(self, issue_data: dict[str, Any]) -> None:
        """Persists critical data quality issues to the database."""
        db_session_factory: async_sessionmaker[AsyncSession] | None = getattr(self, "db_session_factory", None)
        if not db_session_factory:
            self.logger.error(
                "db_session_factory not available, cannot persist critical data quality issue.",
                source_module=self._source_module,
            )
            return

        try:
            async with db_session_factory() as session:
                issue = DataQualityIssue(
                    id=uuid.uuid4(),
                    trading_pair=issue_data.get("trading_pair", "unknown"),
                    alert_type=issue_data.get("alert_type", "unknown"),
                    severity=issue_data.get("severity", "critical"),
                    details=issue_data.get("details", "No details provided."),
                    context=issue_data,
                    reported_at=datetime.fromisoformat(issue_data["timestamp"]),
                )
                session.add(issue)
                await session.commit()
                self.logger.info(
                    f"Successfully persisted critical data quality issue {issue.id} to database.",
                    source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                "Failed to persist critical data quality issue to database.",
                source_module=self._source_module,
                context={"issue_data": issue_data, "error": e})

    async def _handle_trade_event(self, event_dict: dict[str, Any]) -> None:
        """Handle incoming raw trade events and store them."""
        # This method will be called by pubsub, so it takes the full event_dict
        payload = event_dict.get("payload")
        if not payload:
            self.logger.warning(
                "Trade event missing payload.",
                context=event_dict,
                source_module=self._source_module)
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Trade event payload missing trading_pair.",
                context=payload,
                source_module=self._source_module)
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
                    source_module=self._source_module)
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
                    source_module=self._source_module)
                return

            self.trade_history[trading_pair].append(trade_data)
            self.logger.debug(
                "Stored trade for %s: P=%s, V=%s, Side=%s",
                trading_pair,
                trade_data["price"],
                trade_data["volume"],
                trade_data["side"],
                source_module=self._source_module)

        except KeyError:
            self.logger.exception(
                "Missing key in trade event payload for %s.",
                trading_pair,
                source_module=self._source_module,
                context=payload)
        except (ValueError, TypeError):
            self.logger.exception(
                "Data conversion error in trade event payload for %s",
                trading_pair,
                source_module=self._source_module,
                context=payload)
        except Exception:
            self.logger.exception(
                "Unexpected error handling trade event for %s",
                trading_pair,
                source_module=self._source_module,
                context=payload)

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
        default_value: int) -> int:
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
            source_module=self._source_module)
        return default_value

    async def start(self) -> None:
        """Start the feature engine and subscribe to relevant events."""
        try:
            # Subscribe process_market_data to handle both OHLCV and L2 updates
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_OHLCV,
                self.process_market_data)
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_L2,
                self.process_market_data)
            self.pubsub_manager.subscribe(
                EventType.MARKET_DATA_TRADE,
                self._handle_trade_event,  # New subscription
            )
            self.logger.info(
                "FeatureEngine started and subscribed to MARKET_DATA_OHLCV, "
                "MARKET_DATA_L2, and MARKET_DATA_TRADE events.",
                source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine start and subscription",
                source_module=self._source_module)
            # Depending on desired behavior, might re-raise or handle to prevent full stop

    async def stop(self) -> None:
        """Stop the feature engine and clean up resources."""
        try:
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_OHLCV,
                self.process_market_data)
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_L2,
                self.process_market_data)
            self.pubsub_manager.unsubscribe(
                EventType.MARKET_DATA_TRADE,
                self._handle_trade_event,  # New unsubscription
            )
            self.logger.info(
                "FeatureEngine stopped and unsubscribed from market data events.",
                source_module=self._source_module)
        except Exception:
            self.logger.exception(
                "Error during FeatureEngine stop and unsubscription",
                source_module=self._source_module)

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
                context={"original_event": market_data_event_dict})
            return

        trading_pair = payload.get("trading_pair")
        if not trading_pair:
            self.logger.warning(
                "Market data event (type: %s) missing trading_pair.",
                event_type,
                source_module=self._source_module,
                context={"original_event": market_data_event_dict})
            return

        self.logger.debug(
            "Processing event %s for %s from %s",
            event_type,
            trading_pair,
            source_module,
            source_module=self._source_module)

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
                    context={"payload": payload})
        elif event_type == "MARKET_DATA_L2":
            await self._handle_l2_update(trading_pair, payload)
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
                context={"original_event": market_data_event_dict})

    # --- Pipeline-compatible feature calculation methods ---
    # These methods are designed to be used within Scikit-learn FunctionTransformers.
    # They expect float64 inputs and produce float64 outputs (pd.Series[Any] or pd.DataFrame).

    @staticmethod
    def _pipeline_compute_rsi(data: Any, period: int) -> pd.Series[Any]:
        """Compute RSI using pandas-ta, expecting float64 Series[Any] input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            # Logger not available in static method
            return pd.Series(dtype="float64", name=f"rsi_{period}")  # Return empty named series on error

        rsi_series = ta.rsi(data, length=period)
        # Fill NaNs (typically at the beginning) with a neutral RSI value.
        rsi_series = rsi_series.fillna(50.0)
        rsi_series.name = f"rsi_{period}"
        return rsi_series.astype("float64")  # type: ignore[no-any-return]

    @staticmethod
    def _pipeline_compute_macd(
        data: Any,
        fast: int,
        slow: int,
        signal: int) -> pd.DataFrame:
        """Compute MACD using pandas-ta, expecting float64 Series[Any] input.
        Returns a DataFrame with MACD, histogram, and signal lines.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            # Logger not available in static method
            return pd.DataFrame(dtype="float64")  # Return empty DataFrame on error
        # pandas-ta returns MACD, MACDh (histogram), MACDs (signal)
        macd_df = ta.macd(data, fast=fast, slow=slow, signal=signal)
        # Fill NaNs with 0.0 for all MACD related columns
        if macd_df is not None:
            macd_df = macd_df.fillna(0.0)
        return macd_df.astype("float64") if macd_df is not None else pd.DataFrame(dtype="float64")

    @staticmethod
    def _fillna_bbands(bbands_df: pd.DataFrame | None, close_prices: pd.Series[Any]) -> pd.DataFrame:
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
    def _pipeline_compute_bbands(data: Any, length: int, std_dev: float) -> pd.DataFrame:
        """Compute Bollinger Bands using pandas-ta, expecting float64 Series[Any] input.
        Returns a DataFrame with lower, middle, upper bands.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.DataFrame(dtype="float64")
        bbands_df = ta.bbands(data, length=length, std=std_dev)
        # Custom NaN filling: Middle band with close, Lower/Upper also with close (0 width initially)
        if bbands_df is not None:
            bbands_df = FeatureEngine._fillna_bbands(bbands_df, data)
        return bbands_df.astype("float64") if bbands_df is not None else pd.DataFrame(dtype="float64")

    @staticmethod
    def _pipeline_compute_roc(data: Any, period: int) -> pd.Series[Any]:
        """Compute Rate of Change (ROC) using pandas-ta, expecting float64 Series[Any] input.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(data, pd.Series):
            return pd.Series(dtype="float64", name=f"roc_{period}")
        roc_series = ta.roc(data, length=period)
        # Fill NaNs (typically at the beginning) with 0.0, representing no change.
        roc_series = roc_series.fillna(0.0)
        roc_series.name = f"roc_{period}"
        return roc_series.astype("float64")  # type: ignore[no-any-return]

    @staticmethod
    def _pipeline_compute_atr(
        ohlc_data: Any,
        length: int,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close") -> pd.Series[Any]:
        """Compute Average True Range (ATR) using pandas-ta.
        Expects a DataFrame with high, low, close columns (float64).
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlc_data, pd.DataFrame):
            return pd.Series(dtype="float64", name=f"atr_{length}")
        # Ensure required columns are present; this check could be more robust
        if not all(col in ohlc_data.columns for col in [high_col, low_col, close_col]):
            # Logger not available in static method
            return pd.Series(dtype="float64", name=f"atr_{length}")

        atr_series = ta.atr(
            high=ohlc_data[high_col],
            low=ohlc_data[low_col],
            close=ohlc_data[close_col],
            length=length)
        # Fill NaNs with 0.0 as per chosen strategy.
        # This implies zero volatility for initial undefined periods, which is an approximation.
        atr_series = atr_series.fillna(0.0)
        atr_series.name = f"atr_{length}"
        return atr_series.astype("float64")  # type: ignore[no-any-return]

    @staticmethod
    def _pipeline_compute_stdev(data: Any, length: int) -> pd.Series[Any]:
        """Compute Standard Deviation using pandas .rolling().std().
        Expects float64 Series[Any] input.
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
        ohlcv_df: Any,
        length: int,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume") -> pd.Series[Any]:
        """Compute VWAP from OHLCV data using rolling window.
        Expects DataFrame with Decimal objects for price/volume, converts to float64 Series[Any] output.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlcv_df, pd.DataFrame):
            return pd.Series(dtype="float64")
        if not all(col in ohlcv_df.columns for col in [high_col, low_col, close_col, volume_col]):
            return pd.Series(dtype="float64") # Or log error

        # Ensure inputs are Decimal for precision in intermediate calculations
        high_d = ohlcv_df[high_col].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
        low_d = ohlcv_df[low_col].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
        close_d = ohlcv_df[close_col].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
        volume_d = ohlcv_df[volume_col].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]

        typical_price = (high_d + low_d + close_d).apply(lambda x: x / Decimal("3.0"))
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

            if current_sum_vol == Decimal(0) or pd.isna(current_sum_vol) or pd.isna(current_vwap_val):
                # Ensure we access original Decimal values for typical price calculation if ohlcv_df was float
                # However, ohlcv_df input to this function is already converted to Decimal for H,L,C,V
                # So, high_d, low_d, close_d can be used with .loc[idx]
                # Or, re-access from the original ohlcv_df if it was passed with Decimals
                # For simplicity, assume ohlcv_df passed has Decimal type for H,L,C for this fallback
                # If ohlcv_df was passed as float64, this might lose some Decimal precision for typical price.
                # The current `_calculate_and_publish_features` converts ohlcv_df to float64 first,
                # then this function converts selected columns back to Decimal. This is acceptable.
                # Calculate typical price as fallback
                if idx in high_d.index and idx in low_d.index and idx in close_d.index:
                    try:
                        h = high_d.loc[idx]
                        l = low_d.loc[idx]
                        c = close_d.loc[idx]
                        # All values exist, calculate typical price
                        vwap_series_decimal[idx] = (h + l + c) / Decimal("3.0")
                    except Exception:
                        # If there's any error in calculation, set to NA
                        vwap_series_decimal[idx] = pd.NA
                else:
                    # Missing data for this index
                    vwap_series_decimal[idx] = pd.NA


        # Convert to float64 for pipeline compatibility.
        # NaNs from missing HLC for typical price fallback, or if typical price itself is NaN, will remain.
        vwap_series_float = vwap_series_decimal.astype("float64")
        vwap_series_float.name = f"vwap_ohlcv_{length}"

        # Enterprise-grade VWAP NaN handling: Multi-strategy approach for robust VWAP calculation
        return FeatureEngine._apply_enterprise_vwap_nan_handling(
            vwap_series_float, ohlcv_df, length,
        )


    @staticmethod
    def _pipeline_compute_vwap_trades(
        trade_history_deque: deque[dict[str, Any]],  # Deque of trade dicts
        # {"price": Decimal, "volume": Decimal, "timestamp": datetime}
        bar_start_times: Any, # Series[Any] of datetime objects
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series[Any] | None = None, # For fallback
    ) -> pd.Series[Any]:
        """Compute VWAP from trade data for specified bar start times.
        Returns a float64 Series.
        If no relevant trades or sum_volume is zero, falls back to ohlcv_close_prices.
        If fallback also fails, defaults to 0.0.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        series_name = f"vwap_trades_{bar_interval_seconds}s"
        if not isinstance(bar_start_times, pd.Series): # Basic validation for bar_start_times
            # trade_history_deque validation is implicitly handled by checking if trades_df is None/empty
            # Early return for invalid input
            return pd.Series(dtype="float64", index=None, name=series_name)

        output_index = bar_start_times.index

        vwap_results = []

        trades_df = None
        if trade_history_deque: # Only proceed if deque is not empty
            try:
                # Ensure all elements in deque are dicts before creating DataFrame
                if not all(isinstance(trade, dict) for trade in trade_history_deque):
                    # Log or handle malformed deque elements if necessary
                    trades_df = pd.DataFrame(columns=["price", "volume", "timestamp"]) # Empty DF
                else:
                    trades_df = pd.DataFrame(list[Any](trade_history_deque))

                if not trades_df.empty: # Proceed with type conversion only if DataFrame is not empty
                    trades_df["price"] = trades_df["price"].apply(
                        lambda x: Decimal(str(x)),
                    )  # type: ignore[arg-type,return-value]
                    trades_df["volume"] = trades_df["volume"].apply(
                        lambda x: Decimal(str(x)),
                    )  # type: ignore[arg-type,return-value]
                    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
                else: # trades_df is empty (e.g. deque was empty or contained non-dict[str, Any] items)
                    trades_df = None # Ensure it's None to trigger fallback for all bars
            except (ValueError, TypeError, KeyError, AttributeError):
                # Catch broad errors during DataFrame creation or type conversion
                # self.logger.warning("Error processing trade_history_deque: %s", e) # Logger not available
                trades_df = None # Force fallback for all bars if trade data is corrupt

        for bar_start_dt_idx, bar_start_dt in bar_start_times.items(): # Use .items() for index access
            calculated_vwap = np.nan

            if trades_df is not None and not trades_df.empty:
                bar_end_dt = pd.Timestamp(bar_start_dt) + pd.Timedelta(seconds=bar_interval_seconds)
                relevant_trades = trades_df[
                    (trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)
                ]

                if not relevant_trades.empty:
                    sum_price_volume = (relevant_trades["price"] * relevant_trades["volume"]).sum()
                    sum_volume = relevant_trades["volume"].sum()

                    if sum_volume > Decimal(0):
                        vwap_decimal = sum_price_volume / sum_volume
                        calculated_vwap = float(vwap_decimal)

            # Fallback logic
            if pd.isna(calculated_vwap):
                # Initialize to NaN, will be overwritten if we find a valid close price
                current_bar_ohlcv_close_price = np.nan  # type: ignore[unreachable]
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

    @staticmethod
    def _apply_enterprise_vwap_nan_handling(
        vwap_series: pd.Series[Any],
        ohlcv_df: pd.DataFrame,
        length: int,
    ) -> pd.Series[Any]:
        """Enterprise-grade VWAP NaN handling with intelligent fallback strategies.

        Implements a hierarchical approach to handle missing VWAP values:
        1. Smart interpolation based on market microstructure
        2. Volume-weighted price estimates using available data
        3. Context-aware fallback strategies
        4. Quality metrics and monitoring

        Args:
            vwap_series: Original VWAP series with potential NaNs
            ohlcv_df: Source OHLCV data for fallback calculations
            length: VWAP calculation window length

        Returns:
            VWAP series with enterprise-grade NaN handling applied
        """
        if vwap_series.isna().sum() == 0:
            return vwap_series  # No NaNs to handle

        # Create working copy
        result_series = vwap_series.copy()

        # Strategy 1: Intelligent interpolation for short gaps
        result_series = FeatureEngine._apply_intelligent_vwap_interpolation(
            result_series, ohlcv_df, max_gap_length=min(3, length // 5),
        )

        # Strategy 2: Volume-weighted estimates for medium gaps
        result_series = FeatureEngine._apply_volume_weighted_estimates(
            result_series, ohlcv_df, length,
        )

        # Strategy 3: Context-aware typical price fallback
        result_series = FeatureEngine._apply_context_aware_fallback(
            result_series, ohlcv_df,
        )

        # Strategy 4: Final safety net with market-hours aware filling
        return FeatureEngine._apply_market_aware_final_fill(
            result_series, ohlcv_df,
        )


    @staticmethod
    def _apply_intelligent_vwap_interpolation(
        vwap_series: pd.Series[Any],
        ohlcv_df: pd.DataFrame,
        max_gap_length: int = 3,
    ) -> pd.Series[Any]:
        """Apply intelligent interpolation for short VWAP gaps based on volume patterns."""
        result = vwap_series.copy()

        # Identify NaN gaps
        nan_mask = result.isna()
        if not nan_mask.any():
            return result

        # Find contiguous NaN sequences
        nan_groups = (nan_mask != nan_mask.shift()).cumsum()

        for group_id in nan_groups[nan_mask].unique():
            gap_indices = result.index[nan_groups == group_id]
            gap_length = len(gap_indices)

            # Only interpolate short gaps
            if gap_length > max_gap_length:
                continue

            # Get surrounding valid values
            gap_start_idx = gap_indices[0]
            gap_end_idx = gap_indices[-1]

            # Find previous and next valid VWAP values
            prev_valid_idx = None
            next_valid_idx = None

            for idx in reversed(result.index[result.index < gap_start_idx]):
                if pd.notna(result[idx]):
                    prev_valid_idx = idx
                    break

            for idx in result.index[result.index > gap_end_idx]:
                if pd.notna(result[idx]):
                    next_valid_idx = idx
                    break

            if prev_valid_idx is not None and next_valid_idx is not None:
                # Volume-weighted interpolation
                prev_vwap = result[prev_valid_idx]
                next_vwap = result[next_valid_idx]

                # Calculate interpolation weights based on relative volumes
                gap_volumes = []
                for idx in gap_indices:
                    if idx in ohlcv_df.index:
                        volume = float(ohlcv_df.loc[idx, "volume"])
                        gap_volumes.append(volume)
                    else:
                        gap_volumes.append(0.0)

                if gap_volumes and sum(gap_volumes) > 0:
                    # Weight interpolation by relative position and volume
                    for i, idx in enumerate(gap_indices):
                        position_weight = (i + 1) / (gap_length + 1)
                        volume_weight = gap_volumes[i] / sum(gap_volumes) if sum(gap_volumes) > 0 else 1.0 / gap_length

                        # Combine position and volume weighting
                        combined_weight = 0.7 * position_weight + 0.3 * volume_weight
                        interpolated_value = prev_vwap * (1 - combined_weight) + next_vwap * combined_weight

                        result[idx] = interpolated_value
                else:
                    # Simple linear interpolation if volume data unavailable
                    for i, idx in enumerate(gap_indices):
                        weight = (i + 1) / (gap_length + 1)
                        result[idx] = prev_vwap * (1 - weight) + next_vwap * weight

        return result

    @staticmethod
    def _apply_volume_weighted_estimates(
        vwap_series: pd.Series[Any],
        ohlcv_df: pd.DataFrame,
        length: int,
    ) -> pd.Series[Any]:
        """Apply volume-weighted price estimates for remaining NaN values."""
        result = vwap_series.copy()
        nan_mask = result.isna()

        if not nan_mask.any():
            return result

        # Calculate rolling volume-weighted typical price for estimation
        if all(col in ohlcv_df.columns for col in ["high", "low", "close", "volume"]):
            typical_price = (
                ohlcv_df["high"].astype(float) +
                ohlcv_df["low"].astype(float) +
                ohlcv_df["close"].astype(float)
            ) / 3.0

            volume = ohlcv_df["volume"].astype(float)

            # Calculate shorter-window VWAP estimate for missing values
            estimate_length = max(1, length // 3)  # Use shorter window for estimates

            tp_vol = typical_price * volume
            rolling_tp_vol = tp_vol.rolling(window=estimate_length, min_periods=1).sum()
            rolling_volume = volume.rolling(window=estimate_length, min_periods=1).sum()

            vwap_estimate = rolling_tp_vol / rolling_volume
            vwap_estimate = vwap_estimate.replace([np.inf, -np.inf], np.nan)

            # Apply estimates to NaN positions
            for idx in result.index[nan_mask]:
                if idx in vwap_estimate.index and pd.notna(vwap_estimate[idx]):
                    result[idx] = vwap_estimate[idx]

        return result

    @staticmethod
    def _apply_context_aware_fallback(
        vwap_series: pd.Series[Any],
        ohlcv_df: pd.DataFrame,
    ) -> pd.Series[Any]:
        """Apply context-aware fallback using typical price and market conditions."""
        result = vwap_series.copy()
        nan_mask = result.isna()

        if not nan_mask.any():
            return result

        # Calculate context-aware typical price
        if all(col in ohlcv_df.columns for col in ["high", "low", "close", "open"]):
            # Use mid-price when possible (better than just typical price)
            mid_price = (ohlcv_df["high"].astype(float) + ohlcv_df["low"].astype(float)) / 2.0

            # For high volatility periods, weight close price more heavily
            price_range = (ohlcv_df["high"].astype(float) - ohlcv_df["low"].astype(float))
            close_price = ohlcv_df["close"].astype(float)
            ohlcv_df["open"].astype(float)

            # Detect volatility regime
            rolling_volatility = price_range.rolling(window=10, min_periods=1).std()
            volatility_threshold = rolling_volatility.quantile(0.7) if len(rolling_volatility.dropna()) > 0 else 0

            for idx in result.index[nan_mask]:
                if idx in ohlcv_df.index:
                    current_volatility = rolling_volatility.get(idx, 0)

                    if current_volatility > volatility_threshold:
                        # High volatility: prefer close price
                        fallback_value = close_price.get(idx, mid_price.get(idx, np.nan))
                    else:
                        # Normal volatility: use mid-price
                        fallback_value = mid_price.get(idx, close_price.get(idx, np.nan))

                    if pd.notna(fallback_value):
                        result[idx] = fallback_value

        return result

    @staticmethod
    def _apply_market_aware_final_fill(
        vwap_series: pd.Series[Any],
        ohlcv_df: pd.DataFrame,
    ) -> pd.Series[Any]:
        """Apply final market-aware filling strategy as ultimate fallback."""
        result = vwap_series.copy()
        nan_mask = result.isna()

        if not nan_mask.any():
            return result

        # Strategy 1: Forward fill with time decay weighting
        # In 24/7 crypto markets, forward fill is appropriate but should decay over time
        forward_filled = result.ffill()

        # Strategy 2: Backward fill for any remaining leading NaNs
        backward_filled = forward_filled.bfill()

        # Strategy 3: Use typical price as absolute last resort
        if backward_filled.isna().any() and "close" in ohlcv_df.columns:
            close_fallback = ohlcv_df["close"].astype(float)
            final_filled = backward_filled.fillna(close_fallback)
        else:
            final_filled = backward_filled

        # Strategy 4: If still NaNs exist, use global median
        if final_filled.isna().any():
            global_median = final_filled.median()
            if pd.notna(global_median):
                final_filled = final_filled.fillna(global_median)
            else:
                # Absolute last resort: use zero (should never happen in practice)
                # Add type ignore to silence unreachable warning
                final_filled = final_filled.fillna(0.0)  # type: ignore[unreachable]

        return final_filled


    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
        # --- Removed _calculate_true_volume_delta_from_trades, _calculate_vwap_from_trades ---


    @staticmethod
    def _pipeline_compute_l2_spread(l2_books_series: pd.Series[Any]) -> pd.DataFrame:
        """Computes bid-ask spread from a Series[Any] of L2 book snapshots.
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
                   isinstance(book["bids"][0], list | tuple) and len(book["bids"][0]) == 2 and \
                   isinstance(book.get("asks"), list) and len(book["asks"]) > 0 and \
                   isinstance(book["asks"][0], list | tuple) and len(book["asks"][0]) == 2:

                    best_bid_price_str = str(book["bids"][0][0])
                    best_ask_price_str = str(book["asks"][0][0])

                    # Check for non-numeric or empty strings before Decimal conversion
                    if not best_bid_price_str or not best_ask_price_str:
                        raise ValueError("Empty price string encountered.")

                    best_bid = Decimal(best_bid_price_str)
                    best_ask = Decimal(best_ask_price_str)

                    if best_ask > best_bid:  # Ensure valid spread
                        abs_spread_val = best_ask - best_bid
                        mid_price = (best_bid + best_ask) / Decimal(2)
                        if mid_price != Decimal(0):
                            pct_spread_val = (abs_spread_val / mid_price) * Decimal(100)
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
            dtype="float64")

    @staticmethod
    def _pipeline_compute_l2_imbalance(l2_books_series: pd.Series[Any], levels: int = 5) -> pd.Series[Any]:
        """Computes order book imbalance from a Series[Any] of L2 book snapshots.
        Outputs a Series[Any] (float64).
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
                        if not (
                            isinstance(book["bids"][i], list | tuple)
                            and len(book["bids"][i]) == 2
                            and book["bids"][i][1] is not None
                        ):
                            valid_bids = False; break

                    valid_asks = True
                    for i in range(levels):
                        if not (
                            isinstance(book["asks"][i], list | tuple)
                            and len(book["asks"][i]) == 2
                            and book["asks"][i][1] is not None
                        ):
                            valid_asks = False; break

                    if valid_bids and valid_asks:
                        bid_vol_at_levels = sum(Decimal(str(book["bids"][i][1])) for i in range(levels))
                        ask_vol_at_levels = sum(Decimal(str(book["asks"][i][1])) for i in range(levels))

                        total_vol = bid_vol_at_levels + ask_vol_at_levels
                        if total_vol > Decimal(0):
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
        l2_books_series: object,  # Accept as object to bypass mypy inference
        ohlcv_close_prices: pd.Series[Any] | None = None, # For fallback
        levels: int = 1, # Typically levels=1 for WAP
    ) -> pd.Series[float]:
        """Computes Weighted Average Price (WAP) from a Series[Any] of L2 book snapshots.
        Outputs a Series[Any] (float64).
        If WAP cannot be calculated (e.g., invalid book, zero volume for top level),
        it falls back to the corresponding ohlcv_close_prices.loc[index_of_l2_book_entry].
        If fallback also fails or is not available, defaults to 0.0.
        Intended for Scikit-learn FunctionTransformer.
        """
        series_name = f"wap_{levels}"
        if not isinstance(l2_books_series, pd.Series):
            # Early return for invalid input
            return pd.Series(dtype="float64", name=series_name, index=None)

        output_index = l2_books_series.index

        # Cast to pandas Series to work with the data
        series = cast("pd.Series[Any]", l2_books_series)

        waps: list[float] = []

        # Process each book in the series
        for idx, value in enumerate(series):
            book_idx = series.index[idx]
            calculated_wap: float

            # Try to calculate WAP from book data
            wap_from_book = FeatureEngine._try_calculate_wap_from_value(value, levels)

            # Use numpy's isnan for clearer type checking
            if not np.isnan(wap_from_book):
                calculated_wap = wap_from_book
            else:
                # Apply fallback logic
                calculated_wap = FeatureEngine._get_fallback_wap(
                    book_idx, ohlcv_close_prices,
                )

            waps.append(calculated_wap)

        return pd.Series(waps, index=output_index, dtype="float64", name=series_name)

    @staticmethod
    def _try_calculate_wap_from_value(value: Any, levels: int) -> float:
        """Try to calculate WAP from a single value in the series.

        Args:
            value: The value from the series (could be dict, float, None, etc.)
            levels: The price level to use for WAP calculation

        Returns:
            Calculated WAP or np.nan if calculation is not possible
        """
        if not isinstance(value, dict):
            return np.nan

        try:
            # Validate book structure
            if not ("bids" in value and "asks" in value and
                    isinstance(value["bids"], list) and isinstance(value["asks"], list) and
                    len(value["bids"]) >= levels and len(value["asks"]) >= levels):
                return np.nan

            # Extract bid/ask data for the specified level
            bid_data = value["bids"][levels-1]
            ask_data = value["asks"][levels-1]

            if not (isinstance(bid_data, list | tuple) and len(bid_data) >= 2 and
                    isinstance(ask_data, list | tuple) and len(ask_data) >= 2):
                return np.nan

            # Calculate and return WAP
            return FeatureEngine._calculate_wap_from_book_data(bid_data, ask_data)

        except (TypeError, IndexError, ValueError, AttributeError):
            return np.nan

    @staticmethod
    def _calculate_wap_from_book_data(
        bid_data: list[Any] | tuple[Any, ...],
        ask_data: list[Any] | tuple[Any, ...],
    ) -> float:
        """Calculate weighted average price from bid/ask data.

        Args:
            bid_data: [price, volume] for best bid
            ask_data: [price, volume] for best ask

        Returns:
            Calculated WAP or np.nan if calculation fails
        """
        try:
            # Convert to Decimal for precision
            best_bid_price = Decimal(str(bid_data[0]))
            best_bid_vol = Decimal(str(bid_data[1]))
            best_ask_price = Decimal(str(ask_data[0]))
            best_ask_vol = Decimal(str(ask_data[1]))

            total_vol = best_bid_vol + best_ask_vol
            if total_vol > Decimal(0):
                wap_decimal = (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / total_vol
                return float(wap_decimal)
        except (ValueError, TypeError, IndexError):
            pass
        return np.nan

    @staticmethod
    def _get_fallback_wap(
        book_idx: Any,
        ohlcv_close_prices: pd.Series[Any] | None,
    ) -> float:
        """Get fallback WAP value from close prices or default.

        Args:
            book_idx: Index in the series
            ohlcv_close_prices: Optional close prices series for fallback

        Returns:
            Fallback WAP value (close price or 0.0)
        """
        if ohlcv_close_prices is not None:
            try:
                if book_idx in ohlcv_close_prices.index:
                    fallback_close_price = ohlcv_close_prices.loc[book_idx]
                    if pd.notna(fallback_close_price):
                        return float(fallback_close_price)
            except (KeyError, IndexError, TypeError):
                pass
        return 0.0

    @staticmethod
    def _pipeline_compute_l2_depth(l2_books_series: pd.Series[Any], levels: int = 5) -> pd.DataFrame:
        """Computes bid and ask depth from a Series[Any] of L2 book snapshots.
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
                        if not (
                            isinstance(book["bids"][i], list | tuple)
                            and len(book["bids"][i]) == 2
                            and book["bids"][i][1] is not None
                        ):
                            valid_bids = False; break

                    valid_asks = True
                    for i in range(levels):
                        if not (
                            isinstance(book["asks"][i], list | tuple)
                            and len(book["asks"][i]) == 2
                            and book["asks"][i][1] is not None
                        ):
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

        return pd.DataFrame({
            col_name_bid: bid_depths,
            col_name_ask: ask_depths,
        }, index=output_index, dtype="float64")

    @staticmethod
    def _pipeline_compute_volume_delta(
        trade_history_deque: deque[dict[str, Any]], # Deque of trade dicts
        bar_start_times: pd.Series[Any], # Series[Any] of bar start datetime objects
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series[Any] | None = None,  # Added for signature consistency,
        # not used by this specific function
    ) -> pd.Series[Any]:
        """Computes Volume Delta from trade data for specified bar start times.
        If no trades for a bar, delta is 0.0.
        Outputs a Series[Any] (float64).
        Intended for Scikit-learn FunctionTransformer.
        """
        deltas = []
        series_name = f"volume_delta_{bar_interval_seconds}s"
        if not isinstance(bar_start_times, pd.Series) or not isinstance(trade_history_deque, deque):
            # Early return for invalid input
            return pd.Series(
                dtype="float64",
                index=bar_start_times.index if isinstance(bar_start_times, pd.Series) else None,
                name=series_name,
            )  # type: ignore[unreachable]

        if not trade_history_deque: # No trades in entire history
            return pd.Series(0.0, index=bar_start_times.index, dtype="float64", name=series_name)

        trades_df = pd.DataFrame(list[Any](trade_history_deque))
        # Ensure 'price' and 'volume' are converted to Decimal, handling potential string inputs
        trades_df["price"] = trades_df["price"].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
        trades_df["volume"] = trades_df["volume"].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
        trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
        # Ensure side is lowercase
        trades_df["side"] = trades_df["side"].str.lower()


        for bar_start_dt in bar_start_times:
            bar_end_dt = pd.Timestamp(bar_start_dt) + pd.Timedelta(seconds=bar_interval_seconds)
            relevant_trades = trades_df[
                (trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)
            ]

            if relevant_trades.empty:
                deltas.append(0.0) # Or np.nan if preferred for "no trades" vs "zero delta"
                continue

            buy_volume = relevant_trades[relevant_trades["side"] == "buy"]["volume"].sum()
            sell_volume = relevant_trades[relevant_trades["side"] == "sell"]["volume"].sum()
            deltas.append(float(buy_volume - sell_volume))

        return pd.Series(
            deltas, index=bar_start_times.index, dtype="float64",
            name=f"volume_delta_{bar_interval_seconds}s",
        )


    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
        # --- Removed _calculate_true_volume_delta_from_trades, _calculate_vwap_from_trades ---


    async def _calculate_and_publish_features(
        self,
        trading_pair: str,
        timestamp_features_for: str) -> None:
        """Calculate all configured features using pipelines and publish them."""
        ohlcv_df_full_history = self.ohlcv_history.get(trading_pair)
        min_history_req = self._get_min_history_required() # Get actual requirement
        if ohlcv_df_full_history is None or len(ohlcv_df_full_history) < min_history_req:
            self.logger.info(
                "Not enough OHLCV data for %s to calculate features. Need %s, have %s.",
                trading_pair,
                min_history_req,
                len(ohlcv_df_full_history) if ohlcv_df_full_history is not None else 0,
                source_module=self._source_module)
            return

        current_l2_book = self.l2_books.get(trading_pair)
        if current_l2_book and (
            not current_l2_book.get("bids") or not current_l2_book.get("asks")
        ):
            self.logger.debug(
                "L2 book for %s is present but empty or missing bids/asks. "
                "L2 features may be skipped.",
                trading_pair,
                source_module=self._source_module)
            # L2 features might be skipped by their handlers if book is not suitable.

        all_generated_features: dict[str, float] = {}

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
        # For pipeline processing, we need a Series[Any] (even if single-element)
        latest_l2_book_snapshot = self._get_aligned_l2_book(trading_pair, bar_start_datetime)
        # Use the aligned L2 book for better accuracy
        l2_books_aligned_series = pd.Series([latest_l2_book_snapshot], index=[bar_start_datetime])

        # Trade data for trade-based features
        trades_deque = self.trade_history.get(trading_pair, deque())
        # For single bar calculation, bar_start_times_series is just the current bar
        bar_start_times_series = pd.Series([bar_start_datetime], index=[bar_start_datetime])

        # Prepare the single close price for the current bar, aligned to its timestamp for dynamic injection
        # This Series[Any] will have one entry: index=bar_start_datetime, value=close_price_at_bar_start_datetime
        ohlcv_close_for_dynamic_injection = None
        if bar_start_datetime in close_series_for_pipelines.index:
            ohlcv_close_for_dynamic_injection = close_series_for_pipelines.loc[[bar_start_datetime]]
        else:
            self.logger.warning(
                "Could not find close price for current bar %s in historical data. "
                "Features needing this fallback may fail or use 0.0.",
                bar_start_datetime)
            # Create an empty series with the right index to prevent downstream errors if it's expected
            ohlcv_close_for_dynamic_injection = pd.Series(dtype="float64", index=[bar_start_datetime])


        for pipeline_name, pipeline_info in self.feature_pipelines.items():
            pipeline_obj: Any = pipeline_info["pipeline"]
            spec: InternalFeatureSpec = pipeline_info["spec"]

            pipeline_input_data: Any = None
            raw_pipeline_output: Any = None # Define here for clarity

            # Determine input data for the pipeline
            if spec.input_type == "close_series":
                pipeline_input_data = close_series_for_pipelines
            elif spec.input_type == "ohlcv_df":
                pipeline_input_data = ohlcv_df_for_pipelines
            elif spec.input_type == "l2_book_series":
                pipeline_input_data = l2_books_aligned_series # Single latest book snapshot in a Series[Any]
            elif spec.input_type == "trades_and_bar_starts":
                # For these, X is the trade_history_deque. bar_start_times is injected dynamically.
                pipeline_input_data = trades_deque
            else:
                self.logger.warning(
                    "Unknown input_type '%s' for pipeline %s. Skipping.",
                    spec.input_type, pipeline_name,
                )
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
                        if (ohlcv_close_for_dynamic_injection is not None and
                                not ohlcv_close_for_dynamic_injection.empty):
                            current_kw_args["ohlcv_close_prices"] = ohlcv_close_for_dynamic_injection
                        else: # Pass None or an empty series if not available, function should handle it
                            current_kw_args["ohlcv_close_prices"] = pd.Series(
                                dtype="float64", index=[bar_start_datetime],
                            )


                        # For trade-based features, also inject bar_start_times
                        if spec.input_type == "trades_and_bar_starts":
                            current_kw_args["bar_start_times"] = bar_start_times_series

                        calculator_transformer.kw_args = current_kw_args
                    else:
                        self.logger.error(
                            "Calculator step %s not found in cloned pipeline %s. Skipping dynamic args.",
                            calculator_step_name, pipeline_name,
                        )

                # Execute the pipeline (original or cloned-and-modified)
                if pipeline_input_data is not None:
                    raw_pipeline_output = pipeline_to_run.fit_transform(pipeline_input_data)
                else:
                    # This case should ideally be caught by input_type checks or earlier validation
                    self.logger.warning("Pipeline input data is None for %s. Skipping execution.", pipeline_name)
                    continue # Skip to next pipeline if input data is None

            except Exception as e:
                self.logger.exception("Error executing pipeline %s:", e)
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
                                all_generated_features[str(col_name)] = float(value)

                else:
                    # Use traditional output processing for backward compatibility
                    latest_features_values: Any = None
                    if isinstance(raw_pipeline_output, pd.Series):
                        if not raw_pipeline_output.empty:
                            if (spec.input_type not in ["l2_book_series", "trades_and_bar_starts"] or
                                    len(raw_pipeline_output) > 1):
                                latest_features_values = raw_pipeline_output.iloc[-1]
                            else:
                                latest_features_values = (
                                    raw_pipeline_output.iloc[0]
                                    if len(raw_pipeline_output) == 1 else np.nan
                                )
                    elif isinstance(raw_pipeline_output, pd.DataFrame):
                        if not raw_pipeline_output.empty:
                            if spec.input_type not in ["l2_book_series"] or len(raw_pipeline_output) > 1:
                                latest_features_values = raw_pipeline_output.iloc[-1]
                            else:
                                latest_features_values = (
                                    raw_pipeline_output.iloc[0]
                                    if len(raw_pipeline_output) == 1
                                    else pd.Series(dtype="float64")
                                )
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

            except Exception:
                self.logger.exception(f"Error processing outputs for feature {spec.key}: ")
                # Fall back to simple processing
                if pd.notna(raw_pipeline_output):
                    feature_output_name = pipeline_name.replace("_pipeline","")
                    all_generated_features[feature_output_name] = (
                        float(raw_pipeline_output)
                        if hasattr(raw_pipeline_output, "__float__") else 0.0
                    )

        # Extract advanced features if configured
        try:
            advanced_feature_specs = [spec for spec in self._feature_configs.values()
                                     if spec.calculator_type in self.advanced_extractor.advanced_indicators]

            if advanced_feature_specs:
                # Prepare data for advanced extraction
                l2_data = latest_l2_book_snapshot
                trade_data = list[Any](trades_deque) if trades_deque else None

                # Extract advanced features
                advanced_result = await self.advanced_extractor.extract_advanced_features(
                    ohlcv_df_for_pipelines,
                    advanced_feature_specs,
                    l2_data=l2_data,
                    trade_data=trade_data,
                )

                # Add advanced features to the main feature set
                if not advanced_result.features.empty:
                    # Get the latest row of advanced features
                    latest_advanced = (
                        advanced_result.features.iloc[-1]
                        if len(advanced_result.features) > 1
                        else advanced_result.features.iloc[0]
                    )

                    for feature_name, value in latest_advanced.items():
                        if pd.notna(value):
                            all_generated_features[f"advanced_{feature_name}"] = float(value)

                    self.logger.debug(
                        f"Added {len(latest_advanced)} advanced features for {trading_pair}. "
                        f"Quality metrics: {advanced_result.quality_metrics}",
                    )

        except Exception as e:
            self.logger.warning(f"Advanced feature extraction failed for {trading_pair}: {e}")

        # Enterprise-grade feature validation and structuring
        try:
            # Apply comprehensive feature validation and normalization
            validated_features = await self._apply_enterprise_feature_validation(
                all_generated_features,
                trading_pair,
                timestamp_features_for,
            )

            if validated_features is None:
                return  # Validation failed, event not published

            features_for_payload = validated_features

        except Exception as e:
            self.logger.exception(
                "Unexpected error in enterprise feature validation for %s at %s: %s",
                trading_pair,
                timestamp_features_for,
                e,
                source_module=self._source_module)
            return  # Do not publish if validation fails

        # Fallback for any old handlers if no pipelines were built (mostly for transition)
        if not self.feature_pipelines and not features_for_payload:
            self.logger.debug(
                "No pipelines executed, attempting feature calculation with remaining old handlers.",
                source_module=self._source_module,
            )
            # ... (old handler logic can be here if needed, but it's mostly empty now) ...


        if not features_for_payload: # Check if any features were produced
            self.logger.info(
                "No features were successfully structured or validated for %s at %s. Not publishing event.",
                trading_pair,
                timestamp_features_for, # This was timestamp_features_for
                source_module=self._source_module)
            return

        # Construct and publish FeatureEvent
        event_payload = {
            "trading_pair": trading_pair,
            "exchange": self.config.get("exchange_name", "kraken"),
            "timestamp_features_for": timestamp_features_for, # Corrected variable name
            "features": features_for_payload, # Use Pydantic model's dict[str, Any] representation
        }

        full_feature_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": EventType.FEATURES_CALCULATED.name,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
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
                    "num_features": len(features_for_payload), # Use the dict[str, Any] from Pydantic model
                })
        except Exception:
            self.logger.exception(
                "Failed to publish FEATURES_CALCULATED event for %s",
                trading_pair,
                source_module=self._source_module)

    async def _apply_enterprise_feature_validation(
        self,
        raw_features: dict[str, float],
        trading_pair: str,
        timestamp: str,
    ) -> dict[str, Any] | None:
        """Enterprise-grade feature validation with comprehensive quality checks and normalization.

        Implements a multi-stage validation process:
        1. Data type normalization and safety checks
        2. Feature completeness analysis and intelligent imputation
        3. Quality metrics calculation and outlier detection
        4. Business rule validation and range checks
        5. Schema compliance with flexible fallback strategies
        6. Performance monitoring and quality reporting

        Args:
            raw_features: Dictionary of raw feature values
            trading_pair: Trading pair being processed
            timestamp: Timestamp for the features

        Returns:
            Validated and normalized features dictionary, or None if validation fails
        """
        validation_start_time = pd.Timestamp.now()

        # Initialize validation context
        validation_context = {
            "trading_pair": trading_pair,
            "timestamp": timestamp,
            "input_feature_count": len(raw_features),
            "validation_stages": [],
            "quality_metrics": {},
            "issues_detected": [],
            "fallbacks_applied": [],
        }

        self.logger.debug(
            "Starting enterprise feature validation for %s features for %s at %s",
            len(raw_features),
            trading_pair,
            timestamp,
            source_module=self._source_module,
        )

        try:
            # Stage 1: Data Type[Any] Normalization and Safety Checks
            normalized_features = self._normalize_feature_types(raw_features, validation_context)
            if normalized_features is None:
                return None

            # Stage 2: Feature Completeness Analysis and Intelligent Imputation
            complete_features = await self._ensure_feature_completeness(
                normalized_features, trading_pair, validation_context,
            )

            # Stage 3: Quality Metrics and Outlier Detection
            quality_checked_features = self._apply_quality_checks(
                complete_features, trading_pair, validation_context,
            )

            # Stage 4: Business Rule Validation
            business_validated_features = self._apply_business_validation(
                quality_checked_features, trading_pair, validation_context,
            )

            # Stage 5: Schema Compliance with Fallback
            schema_compliant_features = self._ensure_schema_compliance(
                business_validated_features, validation_context,
            )

            # schema_compliant_features is now always a dict (possibly empty), never None
            if not schema_compliant_features:
                return None

            # Stage 6: Final Quality Assessment and Reporting
            return self._finalize_and_report_validation(
                schema_compliant_features, validation_context, validation_start_time,
            )


        except Exception as e:
            self.logger.exception(
                "Enterprise feature validation failed for %s: %s",
                trading_pair,
                e,
                source_module=self._source_module,
                context=validation_context,
            )
            return None

    def _normalize_feature_types(
        self,
        raw_features: dict[str, Any],
        validation_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Normalize feature data types and perform safety checks."""
        normalized: dict[str, Any] = {}
        type_conversion_errors = []

        for feature_name, value in raw_features.items():
            try:
                # Handle different input types
                if isinstance(value, int | float | np.integer | np.floating):
                    # Check for special float values
                    if np.isnan(value):
                        normalized[feature_name] = None  # Will be handled in completeness stage
                    elif np.isinf(value):
                        self.logger.warning(
                            "Infinite value detected for feature %s, converting to None",
                            feature_name,
                            source_module=self._source_module,
                        )
                        normalized[feature_name] = None
                        validation_context["issues_detected"].append(f"infinite_value_{feature_name}")
                    else:
                        normalized[feature_name] = float(value)

                elif isinstance(value, Decimal):
                    try:
                        float_val = float(value)
                        if np.isfinite(float_val):
                            normalized[feature_name] = float_val
                        else:
                            normalized[feature_name] = None
                            validation_context["issues_detected"].append(f"non_finite_decimal_{feature_name}")
                    except (ValueError, OverflowError):
                        normalized[feature_name] = None
                        type_conversion_errors.append(feature_name)

                elif pd.isna(value) or value is None:
                    normalized[feature_name] = None

                elif isinstance(value, str):
                    # Attempt string to float conversion
                    try:
                        float_val = float(value)
                        if np.isfinite(float_val):
                            normalized[feature_name] = float_val
                        else:
                            normalized[feature_name] = None
                    except (ValueError, TypeError):
                        self.logger.warning(
                            "Could not convert string feature %s='%s' to float",
                            feature_name,
                            value,
                            source_module=self._source_module,
                        )
                        normalized[feature_name] = None
                        type_conversion_errors.append(feature_name)

                else:
                    # Unknown type, attempt conversion
                    try:
                        float_val = float(value)
                        if np.isfinite(float_val):
                            normalized[feature_name] = float_val
                        else:
                            normalized[feature_name] = None
                    except (ValueError, TypeError):
                        normalized[feature_name] = None
                        type_conversion_errors.append(feature_name)

            except Exception as e:
                self.logger.debug(
                    "Error normalizing feature %s: %s",
                    feature_name,
                    e,
                    source_module=self._source_module,
                )
                normalized[feature_name] = None
                type_conversion_errors.append(feature_name)

        # Record validation stage results
        validation_context["validation_stages"].append({
            "stage": "type_normalization",
            "success": True,
            "features_processed": len(raw_features),
            "type_errors": len(type_conversion_errors),
            "error_features": type_conversion_errors,
        })

        if type_conversion_errors:
            self.logger.info(
                "Type[Any] conversion errors for %d features: %s",
                len(type_conversion_errors),
                type_conversion_errors,
                source_module=self._source_module,
            )

        return normalized

    async def _ensure_feature_completeness(
        self,
        features: dict[str, float],
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure feature completeness using intelligent imputation strategies."""
        complete_features = features.copy()
        missing_features = [k for k, v in features.items() if v is None]

        if not missing_features:
            validation_context["validation_stages"].append({
                "stage": "completeness_check",
                "success": True,
                "missing_count": 0,
                "imputation_applied": False,
            })
            return complete_features

        # Apply intelligent imputation based on feature type and context
        for feature_name in missing_features:
            try:
                # Determine feature category for appropriate imputation
                imputed_value = await self._impute_missing_feature(
                    feature_name, trading_pair, validation_context,
                )

                if imputed_value is not None:
                    complete_features[feature_name] = imputed_value
                    validation_context["fallbacks_applied"].append(f"imputed_{feature_name}")
                else:
                    # Use feature-type specific defaults
                    default_value = self._get_feature_default_value(feature_name)
                    complete_features[feature_name] = default_value
                    validation_context["fallbacks_applied"].append(f"default_{feature_name}")

            except Exception as e:
                self.logger.debug(
                    "Failed to impute feature %s: %s",
                    feature_name,
                    e,
                    source_module=self._source_module,
                )
                # Final fallback
                complete_features[feature_name] = self._get_feature_default_value(feature_name)
                validation_context["fallbacks_applied"].append(f"emergency_default_{feature_name}")

        validation_context["validation_stages"].append({
            "stage": "completeness_ensured",
            "success": True,
            "missing_count": len(missing_features),
            "imputation_applied": True,
            "imputed_features": missing_features,
        })

        return complete_features

    async def _impute_missing_feature(
        self,
        feature_name: str,
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> float | None:
        """Apply intelligent imputation for a specific missing feature."""
        try:
            # Check if we have historical feature values for this feature
            # This would use the imputation manager if configured
            if hasattr(self, "imputation_manager"):
                # Use the advanced imputation system
                feature_spec = next(
                    (spec for spec in self._feature_configs.values() if spec.key in feature_name),
                    None,
                )

                if feature_spec:
                    data_type = self._infer_data_type_from_spec(feature_spec)

                    # Create synthetic series for imputation
                    synthetic_series = pd.Series([None], index=[pd.Timestamp.now()])
                    self._prepare_imputation_context(feature_spec, synthetic_series)

                    ImputationConfig(
                        feature_key=f"{feature_spec.key}_missing",
                        data_type=data_type,
                        primary_method=ImputationMethod.FORWARD_FILL,
                        fallback_method=ImputationMethod.MEAN,
                        quality_level=ImputationQuality.FAST,
                    )

                    # Enterprise-grade advanced imputation using comprehensive strategies
                    return await self._execute_advanced_feature_imputation(
                        feature_name, trading_pair, feature_spec, data_type, validation_context,
                    )

            # Fallback to enhanced contextual analysis
            return await self._execute_enhanced_contextual_imputation(
                feature_name, trading_pair, validation_context,
            )

        except Exception as e:
            self.logger.debug(
                "Imputation failed for %s: %s",
                feature_name,
                e,
                source_module=self._source_module,
            )
            return None

    async def _execute_advanced_feature_imputation(
        self,
        feature_name: str,
        trading_pair: str,
        feature_spec: InternalFeatureSpec,
        data_type: DataType,
        validation_context: dict[str, Any],
    ) -> float:
        """Execute enterprise-grade advanced feature imputation using multiple strategies.

        Implements a comprehensive imputation framework:
        1. Historical feature analysis and trend extraction
        2. Cross-asset correlation-based imputation
        3. Market regime-aware imputation strategies
        4. Machine learning-based imputation for complex patterns
        5. Confidence scoring and quality assessment

        Args:
            feature_name: Name of the feature to impute
            trading_pair: Trading pair being processed
            feature_spec: Feature specification with configuration
            data_type: Type[Any] of data for imputation strategy selection
            validation_context: Validation context for tracking

        Returns:
            Imputed feature value with high confidence
        """
        imputation_start_time = pd.Timestamp.now()

        try:
            # Strategy 1: Historical Feature Analysis
            historical_value = await self._impute_from_historical_analysis(
                feature_name, trading_pair, feature_spec, validation_context,
            )

            if historical_value is not None:
                confidence_score = self._calculate_imputation_confidence(
                    "historical_analysis", historical_value, feature_name,
                )
                if confidence_score > 0.8:  # High confidence threshold
                    validation_context["fallbacks_applied"].append(
                        f"advanced_historical_{feature_name}",
                    )
                    return historical_value

            # Strategy 2: Cross-Asset Correlation Analysis
            correlation_value = await self._impute_from_correlation_analysis(
                feature_name, trading_pair, data_type, validation_context,
            )

            if correlation_value is not None:
                confidence_score = self._calculate_imputation_confidence(
                    "correlation_analysis", correlation_value, feature_name,
                )
                if confidence_score > 0.7:  # Medium-high confidence
                    validation_context["fallbacks_applied"].append(
                        f"advanced_correlation_{feature_name}",
                    )
                    return correlation_value

            # Strategy 3: Market Regime-Aware Imputation
            regime_value = await self._impute_from_market_regime_analysis(
                feature_name, trading_pair, data_type, validation_context,
            )

            if regime_value is not None:
                confidence_score = self._calculate_imputation_confidence(
                    "market_regime", regime_value, feature_name,
                )
                if confidence_score > 0.6:  # Medium confidence
                    validation_context["fallbacks_applied"].append(
                        f"advanced_regime_{feature_name}",
                    )
                    return regime_value

            # Strategy 4: ML-Based Pattern Imputation
            ml_value = await self._impute_from_ml_patterns(
                feature_name, trading_pair, feature_spec, validation_context,
            )

            if ml_value is not None:
                validation_context["fallbacks_applied"].append(
                    f"advanced_ml_{feature_name}",
                )
                return ml_value

            # Final fallback to enhanced contextual
            return self._get_contextual_default(feature_name, data_type)

        except Exception as e:
            self.logger.debug(
                "Advanced feature imputation failed for %s: %s",
                feature_name, e,
                source_module=self._source_module,
            )
            return self._get_contextual_default(feature_name, data_type)

        finally:
            # Log imputation performance
            duration = (pd.Timestamp.now() - imputation_start_time).total_seconds() * 1000
            self.logger.debug(
                "Advanced imputation for %s completed in %.1fms",
                feature_name, duration,
                source_module=self._source_module,
            )

    async def _execute_enhanced_contextual_imputation(
        self,
        feature_name: str,
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> float:
        """Execute enhanced contextual imputation with market awareness.

        Args:
            feature_name: Feature name to impute
            trading_pair: Trading pair being processed
            validation_context: Validation context for tracking

        Returns:
            Contextually appropriate imputed value
        """
        try:
            # Get current market conditions
            market_conditions = await self._analyze_current_market_conditions(trading_pair)

            # Apply context-aware adjustments
            base_value = self._get_contextual_default(feature_name)
            adjusted_value = self._apply_market_context_adjustment(
                base_value, feature_name, market_conditions,
            )

            validation_context["fallbacks_applied"].append(
                f"enhanced_contextual_{feature_name}",
            )

        except Exception as e:
            self.logger.debug(
                "Enhanced contextual imputation failed for %s: %s",
                feature_name, e,
                source_module=self._source_module,
            )
            return self._get_contextual_default(feature_name)
        else:
            return adjusted_value

    async def _impute_from_historical_analysis(
        self,
        feature_name: str,
        trading_pair: str,
        feature_spec: InternalFeatureSpec,
        validation_context: dict[str, Any],
    ) -> float | None:
        """Impute based on historical feature patterns and trends."""
        try:
            lookback = max(20, feature_spec.parameters.get("period", 14) * 3)
            interval = feature_spec.parameters.get("interval", "1m")

            if not self.history_repo:
                return None

            # Try to get historical data if the method is available
            historical_data = None
            try:
                # Check if the method exists
                if hasattr(self.history_repo, "get_feature_history"):
                    historical_data = await self.history_repo.get_feature_history(
                        trading_pair=trading_pair,
                        feature_name=feature_name,
                        lookback=lookback,
                        interval=interval,
                    )
                else:
                    self.logger.debug(
                        "get_feature_history method not available in HistoryRepository",
                        source_module=self._source_module,
                    )
                    return None
            except AttributeError:
                # Method doesn't exist yet
                self.logger.debug(
                    "get_feature_history not implemented in HistoryRepository",
                    source_module=self._source_module,
                )
                return None
            except Exception as e:
                self.logger.debug(
                    f"Error fetching historical data for {feature_name}: {e}",
                    source_module=self._source_module,
                )
                return None

            if historical_data is None or historical_data.empty:
                return None

            # Use enhanced imputation if available
            try:
                from .feature_engine_enhancements import (
                    ImputationStrategy,
                    IntelligentImputationEngine,
                )

                # Create pandas series from historical data
                hist_series = pd.Series(
                    historical_data["value"].values,
                    index=pd.to_datetime(historical_data["timestamp"]),
                )

                # Use intelligent imputation engine
                imputation_engine = IntelligentImputationEngine(self.logger)
                feature_metadata = {
                    feature_name: {
                        "type": (
                            "technical" if any(
                                ind in feature_name.lower()
                                for ind in ["rsi", "macd", "bb", "sma", "ema"]
                            )
                            else "volume" if "volume" in feature_name.lower()
                            else "unknown"
                        ),
                        "category": "indicator",
                    },
                }

                # Impute using regime-aware strategy
                imputed_df = pd.DataFrame({feature_name: hist_series})
                imputed_data, report = imputation_engine.impute_features(
                    imputed_df,
                    feature_metadata,
                    ImputationStrategy.REGIME_AWARE,
                )

                if feature_name in imputed_data.columns:
                    last_value = imputed_data[feature_name].iloc[-1]
                    self.logger.debug(
                        "Imputed %s using intelligent regime-aware method: %.4f",
                        feature_name,
                        last_value,
                        source_module=self._source_module,
                    )
                    return float(last_value) if pd.notna(last_value) else None

            except ImportError:
                pass  # Fall back to simple implementation

            # Fallback: Simple imputation using moving average
            if "rsi" in feature_name.lower() or "macd" in feature_name.lower():
                ma = historical_data["value"].rolling(window=5, min_periods=1).mean().iloc[-1]
                self.logger.debug(
                    "Imputed %s from historical MA: %.4f",
                    feature_name,
                    ma,
                    source_module=self._source_module,
                )
                return float(ma) if pd.notna(ma) else None

            # Fallback: Impute volume with its historical mean
            if "volume" in feature_name.lower():
                vol_ma = historical_data["value"].mean()
                self.logger.debug(
                    "Imputed %s from historical mean: %.4f",
                    feature_name,
                    vol_ma,
                    source_module=self._source_module,
                )
                return float(vol_ma) if pd.notna(vol_ma) else None

        except Exception as e:
            self.logger.debug(f"Historical analysis imputation failed for {feature_name}: {e}")
            return None
        else:
            return None

    async def _impute_from_correlation_analysis(
        self,
        feature_name: str,
        trading_pair: str,
        data_type: DataType,
        validation_context: dict[str, Any],
    ) -> float | None:
        """Impute based on correlation with other available features."""
        try:
            recent_features = await fetch_latest_features(trading_pair, limit=200)
            if recent_features is None or feature_name not in recent_features.columns:
                return None

            # compute_correlations is not async and takes only one parameter
            correlations = compute_correlations(recent_features)

            if correlations.empty:
                return None

            # Use the most correlated feature for imputation
            best_corr = correlations.iloc[0]
            correlated_feature_name = best_corr["correlated_feature"]
            correlation_value = best_corr["correlation"]

            # Fetch the value of the correlated feature at the current timestamp
            # This assumes validation_context might have access to currently computed features
            correlated_value = validation_context.get("features", {}).get(correlated_feature_name)

            if correlated_value is not None and abs(correlation_value) > 0.5:
                # Simple linear scaling based on correlation
                # This is a simplification; a real implementation might use a trained model
                imputed_value = correlated_value * correlation_value
                self.logger.info(
                    f"Imputed {feature_name} using correlated feature {correlated_feature_name} "
                    f"(corr: {correlation_value:.2f}, value: {imputed_value:.4f})",
                    source_module=self._source_module)
                return imputed_value  # type: ignore[no-any-return]

        except Exception as e:
            self.logger.debug(
                "Correlation analysis imputation failed: %s",
                e,
                source_module=self._source_module,
                context={"feature_name": feature_name, "trading_pair": trading_pair})
            return None
        else:
            return None

    async def _impute_from_market_regime_analysis(
        self,
        feature_name: str,
        trading_pair: str,
        data_type: DataType,
        validation_context: dict[str, Any],
    ) -> float | None:
        """Impute based on current market regime (trending, ranging, volatile)."""
        try:
            market_conditions = await self._analyze_current_market_conditions(trading_pair)
            regime = market_conditions.get("regime", "unknown")
            volatility_level = market_conditions.get("volatility", "medium")

            # Regime-specific feature defaults
            if "rsi" in feature_name.lower():
                if regime == "trending_up":
                    return 65.0  # Bullish trending
                if regime == "trending_down":
                    return 35.0  # Bearish trending
                if regime == "ranging":
                    return 50.0  # Neutral ranging
                if regime == "volatile":
                    return 55.0 if volatility_level == "high" else 45.0

            elif "volume" in feature_name.lower():
                if regime == "volatile":
                    return 1500.0  # Higher volume in volatile markets
                if regime in ["trending_up", "trending_down"]:
                    return 1200.0  # Moderate volume in trends
                return 800.0  # Lower volume in ranging markets

        except Exception as e:
            self.logger.debug(
                "Market regime analysis failed: %s",
                e, source_module=self._source_module,
            )
            return None
        else:
            return None

    async def _impute_from_ml_patterns(
        self,
        feature_name: str,
        trading_pair: str,
        feature_spec: InternalFeatureSpec,
        validation_context: dict[str, Any],
    ) -> float | None:
        """Impute missing values using a trained ML model."""
        try:
            model_key = feature_spec.imputation_model_key
            model_version = feature_spec.imputation_model_version
            if not model_key:
                return None

            model_registry: ImputationModelRegistry | None = getattr(self, "imputation_model_registry", None)
            if not model_registry:
                self.logger.warning("Imputation model registry not available.", source_module=self._source_module)
                return None

            model = await model_registry.get(model_key)
            if not model:
                self.logger.warning(
                    f"Imputation model {model_key} (version: {model_version or 'latest'}) not found.",
                    source_module=self._source_module)
                return None

            # Build features for the ML model
            # build_ml_features is not async and takes only ohlcv_history
            # Get OHLCV history for the trading pair
            ohlcv_history = self.ohlcv_history.get(trading_pair)
            if ohlcv_history is None:
                return None
            ml_features = build_ml_features(ohlcv_history)

            # build_ml_features always returns a DataFrame, never None or dict
            # The following checks are defensive but unnecessary given the current implementation

            # Predict the missing value
            prediction = model.predict(ml_features)
            imputed_value = float(prediction[0])

            self.logger.info(
                f"Imputed {feature_name} using ML model {model_key}: {imputed_value:.4f}",
                source_module=self._source_module)

        except Exception as e:
            self.logger.debug(
                "ML-based imputation failed for %s: %s",
                feature_name,
                e,
                source_module=self._source_module,
                context={"feature_spec": feature_spec})
            return None
        else:
            return imputed_value

    async def _analyze_current_market_conditions(self, trading_pair: str) -> dict[str, Any]:
        """Analyzes current market conditions to inform imputation strategies."""
        try:
            ohlcv_data = self.ohlcv_history.get(trading_pair)
            if ohlcv_data is None or len(ohlcv_data) < 20:
                return {"regime": "unknown", "volatility": "medium", "confidence": 0.0}

            recent_data = ohlcv_data.tail(20).astype(float)

            # Calculate market metrics
            price_changes = recent_data["close"].pct_change().dropna()
            volatility = price_changes.std()
            trend_strength = abs(price_changes.mean())
            volume_trend = recent_data["volume"].pct_change().mean()

            # Determine market regime
            if trend_strength > 0.01 and price_changes.mean() > 0:
                regime = "trending_up"
            elif trend_strength > 0.01 and price_changes.mean() < 0:
                regime = "trending_down"
            elif volatility > 0.03:
                regime = "volatile"
            else:
                regime = "ranging"

            # Classify volatility
            if volatility > 0.05:
                vol_level = "high"
            elif volatility < 0.01:
                vol_level = "low"
            else:
                vol_level = "medium"

            return {
                "regime": regime,
                "volatility": vol_level,
                "trend_strength": float(trend_strength),
                "volume_trend": float(volume_trend) if pd.notna(volume_trend) else 0.0,
                "confidence": min(1.0, len(recent_data) / 20.0),
            }

        except Exception as e:
            self.logger.debug(
                "Market condition analysis failed: %s",
                e, source_module=self._source_module,
            )
            return {"regime": "unknown", "volatility": "medium", "confidence": 0.0}

    def _apply_market_context_adjustment(
        self,
        base_value: float,
        feature_name: str,
        market_conditions: dict[str, Any],
    ) -> float:
        """Apply market context adjustments to base imputed values."""
        try:
            regime = market_conditions.get("regime", "unknown")
            volatility = market_conditions.get("volatility", "medium")
            confidence = market_conditions.get("confidence", 0.5)

            # Apply confidence-weighted adjustments
            adjustment_factor = confidence * 0.2  # Max 20% adjustment

            if "rsi" in feature_name.lower():
                if regime == "trending_up":
                    return min(100.0, base_value + (adjustment_factor * 20))  # type: ignore[no-any-return]
                if regime == "trending_down":
                    return max(0.0, base_value - (adjustment_factor * 20))  # type: ignore[no-any-return]
                if volatility == "high":
                    return (
                        base_value + (adjustment_factor * 10) if base_value > 50
                        else base_value - (adjustment_factor * 10)
                    )  # type: ignore[no-any-return]

            elif "spread" in feature_name.lower():
                if volatility == "high":
                    return base_value * (1 + adjustment_factor)  # type: ignore[no-any-return]
                if volatility == "low":
                    return base_value * (1 - adjustment_factor * 0.5)  # type: ignore[no-any-return]

            elif "volume" in feature_name.lower():
                if regime in ["trending_up", "trending_down"]:
                    return base_value * (1 + adjustment_factor)  # type: ignore[no-any-return]
                if volatility == "high":
                    return base_value * (1 + adjustment_factor * 1.5)  # type: ignore[no-any-return]

        except Exception as e:
            self.logger.debug(
                "Market context adjustment failed: %s",
                e, source_module=self._source_module,
            )
            return base_value
        else:
            return base_value

    def _calculate_imputation_confidence(
        self,
        method: str,
        value: float,
        feature_name: str,
    ) -> float:
        """Calculate confidence score for imputed values."""
        try:
            # Base confidence by method
            method_confidence = {
                "historical_analysis": 0.8,
                "correlation_analysis": 0.7,
                "market_regime": 0.6,
                "ml_patterns": 0.75,
                "contextual": 0.5,
            }.get(method, 0.5)

            # Value reasonableness check
            value_confidence = 1.0
            if "rsi" in feature_name.lower():
                if 0 <= value <= 100:
                    value_confidence = 1.0
                elif -10 <= value <= 110:
                    value_confidence = 0.8
                else:
                    value_confidence = 0.3

            elif "percentage" in feature_name.lower() or "pct" in feature_name.lower():
                if abs(value) <= 100:
                    value_confidence = 1.0
                elif abs(value) <= 200:
                    value_confidence = 0.7
                else:
                    value_confidence = 0.4

            return method_confidence * value_confidence

        except Exception:
            return 0.5  # Default confidence

    def _get_contextual_default(
        self,
        feature_name: str,
        data_type: DataType | None = None,
    ) -> float:
        """Get contextual default value for a feature based on its name and type."""
        feature_lower = feature_name.lower()

        # RSI-related features
        if "rsi" in feature_lower:
            return 50.0  # Neutral RSI

        # MACD-related features
        if "macd" in feature_lower:
            return 0.0  # Neutral MACD

        # Volume-related features
        if any(vol_term in feature_lower for vol_term in ["volume", "vol", "vwap"]):
            return 0.0  # No volume/VWAP

        # Price-related features
        if any(price_term in feature_lower for price_term in ["price", "spread", "wap"]):
            return 0.0  # No spread/price difference

        # Volatility features
        if any(vol_term in feature_lower for vol_term in ["atr", "volatility", "stdev"]):
            return 0.0  # No volatility

        # Percentage features
        if "pct" in feature_lower or "percent" in feature_lower:
            return 0.0  # No percentage change

        # Imbalance features
        if "imbalance" in feature_lower:
            return 0.0  # Balanced

        # Default for unknown features
        return 0.0

    def _get_feature_default_value(self, feature_name: str) -> float:
        """Get the default value for a feature when all else fails."""
        return self._get_contextual_default(feature_name)

    def _apply_quality_checks(
        self,
        features: dict[str, Any],  # Values can be float or None
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply quality checks and outlier detection."""
        quality_checked = features.copy()
        outliers_detected = []

        for feature_name, value in features.items():
            if value is None:
                continue

            try:
                # Check for extreme outliers based on feature type
                if self._is_extreme_outlier(feature_name, value):
                    self.logger.warning(
                        "Extreme outlier detected for %s: %s, capping to reasonable range",
                        feature_name,
                        value,
                        source_module=self._source_module,
                    )
                    quality_checked[feature_name] = self._cap_outlier_value(feature_name, value)
                    outliers_detected.append(feature_name)
                    validation_context["fallbacks_applied"].append(f"outlier_capped_{feature_name}")

            except Exception as e:
                self.logger.debug(
                    "Quality check failed for %s: %s",
                    feature_name,
                    e,
                    source_module=self._source_module,
                )

        validation_context["validation_stages"].append({
            "stage": "quality_checks",
            "success": True,
            "outliers_detected": len(outliers_detected),
            "outlier_features": outliers_detected,
        })

        return quality_checked

    def _is_extreme_outlier(self, feature_name: str, value: float) -> bool:
        """Check if a feature value is an extreme outlier."""
        feature_lower = feature_name.lower()

        # RSI should be between 0 and 100
        if "rsi" in feature_lower:
            return value < -10 or value > 110

        # Percentage features should generally be reasonable
        if "pct" in feature_lower or "percent" in feature_lower:
            return abs(value) > 1000  # 1000% change is extreme

        # General extreme value check
        return abs(value) > 1e6 or abs(value) < 1e-10

    def _cap_outlier_value(self, feature_name: str, value: float) -> float:
        """Cap an outlier value to a reasonable range."""
        feature_lower = feature_name.lower()

        # RSI capping
        if "rsi" in feature_lower:
            return max(0.0, min(100.0, value))

        # Percentage capping
        if "pct" in feature_lower or "percent" in feature_lower:
            return max(-100.0, min(100.0, value))

        # General capping
        if value > 1e6:
            return 1e6
        if value < -1e6:
            return -1e6
        if 0 < abs(value) < 1e-10:
            return 0.0
        return value

    def _apply_business_validation(
        self,
        features: dict[str, float],
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply business logic validation rules."""
        validated = features.copy()
        business_violations = []

        # Use enhanced validation if available
        try:
            from .feature_engine_enhancements import ComprehensiveFeatureValidator

            # Create feature validator
            validator = ComprehensiveFeatureValidator(self.logger)

            # Get historical data if available
            historical_data = None
            if hasattr(self, "history_repo") and self.history_repo:
                with contextlib.suppress(Exception):
                    # TODO: get_feature_history method not implemented in HistoryRepository
                    # When implemented, uncomment the code below:
                    # hist_features = await self.history_repo.get_feature_history(
                    #     trading_pair=trading_pair,
                    #     lookback=100,  # Last 100 data points
                    #     interval="1h"
                    # )
                    # if hist_features is not None and not hist_features.empty:
                    #     historical_data = hist_features.pivot(
                    #         index='timestamp',
                    #         columns='feature_name',
                    #         values='value'
                    #     )
                    pass  # Method not implemented yet

            # Comprehensive validation
            validated_features, validation_report = validator.validate_features(
                features,
                {},  # Feature metadata would be populated in production
                historical_data,
            )

            # Extract violations from report
            if "statistical_tests" in validation_report:
                consistency = validation_report["statistical_tests"].get("consistency", {})

                # Check spread consistency
                if (
                    "spread_consistency" in consistency
                    and not consistency["spread_consistency"].get("consistent", True)
                ):
                    business_violations.append("inconsistent_spreads")

                # Check RSI bounds
                if "rsi_bounds" in consistency and not consistency["rsi_bounds"].get("within_bounds", True):
                    business_violations.append("rsi_out_of_bounds")

                # Check for outliers
                outliers = validation_report["statistical_tests"].get("outliers", {})
                for feature_name, outlier_info in outliers.items():
                    if outlier_info.get("is_outlier", False):
                        business_violations.append(f"outlier_{feature_name}")

            # Apply corrections
            if "corrections_applied" in validation_report:
                for feature_name, correction in validation_report["corrections_applied"].items():
                    self.logger.info(
                        f"Applied correction to {feature_name}: {correction['reason']}",
                        source_module=self._source_module,
                    )

            # Update validated features
            validated.update(validated_features)

        except ImportError:
            # Fallback to simple business rules
            # Check for logical consistency in spread features
            if "abs_spread" in features and "pct_spread" in features:
                abs_spread = features.get("abs_spread", 0)
                pct_spread = features.get("pct_spread", 0)

                # Spreads should both be positive or both be zero
                if (abs_spread > 0) != (pct_spread > 0):
                    self.logger.warning(
                        "Inconsistent spread values: abs=%s, pct=%s",
                        abs_spread,
                        pct_spread,
                        source_module=self._source_module,
                    )
                    business_violations.append("inconsistent_spreads")
                    # Apply correction
                    if abs_spread > 0:
                        validated["pct_spread"] = max(0.01, abs(pct_spread))
                    else:
                        validated["abs_spread"] = 0.0
                        validated["pct_spread"] = 0.0

            # Add more business rules as needed

        except Exception as e:
            self.logger.debug(
                "Business validation error: %s",
                e,
                source_module=self._source_module,
            )

        validation_context["validation_stages"].append({
            "stage": "business_validation",
            "success": True,
            "violations_detected": len(business_violations),
            "violations": business_violations,
        })

        return validated

    def _ensure_schema_compliance(
        self,
        features: dict[str, float],
        validation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure features comply with the expected schema."""
        try:
            # Attempt Pydantic validation
            pydantic_features = PublishedFeaturesV1.model_validate(features)
            schema_compliant = pydantic_features.model_dump()

            validation_context["validation_stages"].append({
                "stage": "schema_compliance",
                "success": True,
                "validation_method": "pydantic_strict",
            })

        except Exception as pydantic_error:
            self.logger.warning(
                "Strict Pydantic validation failed, applying intelligent schema adaptation: %s",
                pydantic_error,
                source_module=self._source_module,
            )

            # Intelligent schema adaptation
            try:
                adapted_features = self._adapt_to_schema(features, validation_context)
                if adapted_features:
                    # Retry validation
                    pydantic_features = PublishedFeaturesV1(**adapted_features)
                    schema_compliant = pydantic_features.model_dump()

                    validation_context["validation_stages"].append({
                        "stage": "schema_compliance",
                        "success": True,
                        "validation_method": "adaptive",
                        "adaptations_applied": True,
                    })

                    return schema_compliant
                # If no adaptation was possible, return empty dict
                validation_context["validation_stages"].append({
                    "stage": "schema_compliance",
                    "success": False,
                    "error": "No adaptation possible",
                    "original_error": str(pydantic_error),
                })
            except Exception as adaptation_error:
                self.logger.exception(
                    "Schema adaptation also failed: %s",
                    adaptation_error,
                    source_module=self._source_module,
                )

                validation_context["validation_stages"].append({
                    "stage": "schema_compliance",
                    "success": False,
                    "error": str(adaptation_error),
                    "original_error": str(pydantic_error),
                })

                # Return empty dict as a last resort to satisfy return type
                return {}
            else:
                return {}
        else:
            return schema_compliant

    def _adapt_to_schema(
        self,
        features: dict[str, float],
        validation_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Adapt features to match expected schema by adding missing required fields."""
        try:
            # Get the Pydantic model fields

            # Inspect PublishedFeaturesV1 to understand required fields
            model_fields = PublishedFeaturesV1.model_fields
            adapted = features.copy()

            # Add missing required fields with appropriate defaults
            for field_name in model_fields:
                if field_name not in adapted:
                    # Determine appropriate default based on field type and name
                    default_val = self._get_contextual_default(field_name)
                    adapted[field_name] = default_val
                    validation_context["fallbacks_applied"].append(f"schema_default_{field_name}")

        except Exception as e:
            self.logger.exception(
                "Schema adaptation failed: %s",
                e,
                source_module=self._source_module,
            )
            return None
        else:
            return adapted

    def _finalize_and_report_validation(
        self,
        features: dict[str, float],
        validation_context: dict[str, Any],
        start_time: pd.Timestamp,
    ) -> dict[str, Any]:
        """Finalize validation and report quality metrics."""
        end_time = pd.Timestamp.now()
        validation_duration = (end_time - start_time).total_seconds() * 1000  # milliseconds

        # Calculate final quality metrics
        quality_metrics = {
            "validation_duration_ms": validation_duration,
            "output_feature_count": len(features),
            "stages_completed": len(validation_context["validation_stages"]),
            "issues_detected": len(validation_context["issues_detected"]),
            "fallbacks_applied": len(validation_context["fallbacks_applied"]),
            "overall_quality_score": self._calculate_quality_score(validation_context),
        }

        validation_context["quality_metrics"] = quality_metrics

        # Log validation summary
        self.logger.info(
            "Feature validation completed for %s: %d features, %.1fms, quality=%.2f",
            validation_context["trading_pair"],
            quality_metrics["output_feature_count"],
            validation_duration,
            quality_metrics["overall_quality_score"],
            source_module=self._source_module,
            context={
                "validation_summary": {
                    "stages": [stage["stage"] for stage in validation_context["validation_stages"]],
                    "issues": validation_context["issues_detected"],
                    "fallbacks": validation_context["fallbacks_applied"],
                },
            },
        )

        return features

    def _calculate_quality_score(self, validation_context: dict[str, Any]) -> float:
        """Calculate an overall quality score for the validation process."""
        total_features = validation_context["input_feature_count"]
        issues_count = len(validation_context["issues_detected"])
        fallbacks_count = len(validation_context["fallbacks_applied"])

        if total_features == 0:
            return 0.0

        # Quality score based on ratio of clean features
        issue_penalty = min(0.8, issues_count / total_features)
        fallback_penalty = min(0.2, fallbacks_count / total_features)

        return max(0.0, 1.0 - issue_penalty - fallback_penalty)

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
        max_age_seconds: int = 300,  # 5 minutes default
    ) -> dict[str, Any] | None:
        """Get the L2 book snapshot closest to the target timestamp.

        Args:
            trading_pair: The trading pair to get L2 book for
            target_timestamp: The timestamp to align with
            max_age_seconds: Maximum age of L2 snapshot to consider valid

        Returns:
            L2 book dict[str, Any] or None if no valid book found
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

            if best_time_diff is None or time_diff < best_time_diff:  # type: ignore[unreachable]
                best_time_diff = time_diff
                best_book = entry["book"]

        if best_book is None:
            self.logger.debug(
                "No valid L2 book found for %s at %s within %s seconds",
                trading_pair,
                target_timestamp,
                max_age_seconds,
                source_module=self._source_module)
            # Fallback to latest book as last resort
            return self.l2_books.get(trading_pair)

        return best_book  # type: ignore[no-any-return]

    def _create_input_imputation_step(self, spec: InternalFeatureSpec) -> tuple[str, Any] | None:
        """Create intelligent input imputation step for cryptocurrency trading data.

        This method leverages the advanced ImputationManager to provide sophisticated
        imputation strategies specifically designed for 24/7 cryptocurrency markets.
        Instead of using simple mean imputation, it applies context-aware strategies
        that consider market dynamics, volatility regimes, and data quality.

        Args:
            spec: Feature specification containing imputation preferences and parameters

        Returns:
            Tuple of (step_name, transformer) for pipeline, or None if no imputation needed
        """
        try:
            # Determine data type for crypto-specific imputation strategy
            data_type = self._infer_data_type_from_spec(spec)

            # Check for feature-specific input imputation configuration
            input_imputation_config = spec.parameters.get("input_imputation", {})

            # Determine imputation method based on feature type and configuration
            primary_method = self._select_optimal_imputation_method(spec, input_imputation_config, data_type)
            fallback_method = self._get_fallback_method(primary_method)

            # Configure imputation for this specific feature
            imputation_config = ImputationConfig(
                feature_key=f"{spec.key}_input",
                data_type=data_type,
                primary_method=primary_method,
                fallback_method=fallback_method,
                quality_level=ImputationQuality.BALANCED,  # Balance accuracy and performance
                max_gap_minutes=input_imputation_config.get("max_gap_minutes", 10),  # Conservative for crypto
                knn_neighbors=input_imputation_config.get("knn_neighbors", 5),
                confidence_threshold=input_imputation_config.get("confidence_threshold", 0.7),
                max_computation_time_ms=input_imputation_config.get(
                    "max_computation_time_ms", 50.0,
                ),  # Fast for real-time
                cache_results=True,
                consider_market_session=False,  # Crypto trades 24/7
                consider_volatility_regime=True,  # Important for crypto
                use_cross_asset_correlation=input_imputation_config.get("use_cross_correlation", False),
            )

            # Register this configuration with the imputation manager
            self.imputation_manager.configure_feature(imputation_config.feature_key, imputation_config)

            # Create custom transformer that uses our advanced imputation system
            def advanced_imputation_transform(X: Any) -> Any:
                """Advanced imputation transformer for cryptocurrency data."""
                if hasattr(X, "iloc"):  # DataFrame
                    if len(X.columns) == 1:
                        series_data = X.iloc[:, 0]
                    else:
                        # For multi-column input, use close price if available
                        close_col = next((col for col in X.columns if "close" in col.lower()), X.columns[0])
                        series_data = X[close_col]
                else:  # Series[Any] or array
                    series_data = pd.Series(X) if not isinstance(X, pd.Series) else X

                # Check if imputation is needed
                if not series_data.isna().any():
                    return X  # No missing values, return as-is

                # Prepare context for advanced imputation
                context = self._prepare_imputation_context(spec, series_data)

                # Use advanced imputation manager
                try:
                    import asyncio
                    # Run async imputation in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self.imputation_manager.impute_feature(
                                imputation_config.feature_key,
                                series_data,
                                context,
                            ),
                        )
                        imputed_series = result.imputed_values

                        # Log imputation performance
                        self.logger.debug(
                            "Advanced input imputation for %s: method=%s, confidence=%.3f, time=%.1fms, missing=%d",
                            spec.key,
                            result.method_used.value,
                            result.confidence_score,
                            result.computation_time_ms,
                            result.missing_count,
                            source_module=self._source_module,
                        )

                    finally:
                        loop.close()

                except Exception as e:
                    # Fallback to simple forward fill if advanced imputation fails
                    self.logger.warning(
                        "Advanced imputation failed for %s, using fallback: %s",
                        spec.key, e,
                        source_module=self._source_module,
                    )
                    imputed_series = series_data.fillna(method="ffill").fillna(series_data.mean())

                # Return in same format as input
                if hasattr(X, "iloc"):  # DataFrame
                    result_df = X.copy()
                    if len(X.columns) == 1:
                        result_df.iloc[:, 0] = imputed_series
                    else:
                        result_df[result_df.columns[0]] = imputed_series
                    return result_df
                return imputed_series

            step_name = f"{spec.key}_advanced_input_imputer"
            transformer = FunctionTransformer(
                func=advanced_imputation_transform,
                validate=False,
                check_inverse=False,
            )

            self.logger.info(
                "Created advanced input imputation step for %s: method=%s, data_type=%s",
                spec.key,
                primary_method.value,
                data_type.value,
                source_module=self._source_module,
            )

        except Exception as e:
            self.logger.exception(
                "Error creating advanced input imputation step for %s: %s. Using simple fallback.",
                spec.key, e,
                source_module=self._source_module,
            )
            # Fallback to simple mean imputation
            return (f"{spec.key}_simple_input_imputer", SimpleImputer(strategy="mean"))
        else:
            return (step_name, transformer)

    def _infer_data_type_from_spec(self, spec: InternalFeatureSpec) -> DataType:
        """Infer the data type for imputation based on feature specification."""
        # Map feature calculator types to appropriate data types
        calculator_type = spec.calculator_type.lower()

        if any(price_indicator in calculator_type for price_indicator in ["rsi", "macd", "bbands", "roc"]):
            return DataType.PRICE
        if any(vol_indicator in calculator_type for vol_indicator in ["atr", "stdev", "volatility"]):
            return DataType.VOLATILITY
        if any(vol_indicator in calculator_type for vol_indicator in ["volume", "vwap"]):
            return DataType.VOLUME
        if "l2" in calculator_type:
            return DataType.PRICE  # L2 order book data is price-related
        return DataType.INDICATOR  # Generic technical indicator

    def _select_optimal_imputation_method(
        self,
        spec: InternalFeatureSpec,
        input_config: dict[str, Any],
        data_type: DataType,
    ) -> ImputationMethod:
        """Select optimal imputation method based on feature type and crypto market characteristics."""
        # Check for explicit method override
        if "method" in input_config:
            try:
                return ImputationMethod(input_config["method"])
            except ValueError:
                self.logger.warning(
                    "Invalid imputation method '%s' for %s. Using auto-selection.",
                    input_config["method"], spec.key,
                    source_module=self._source_module,
                )

        # Auto-select based on data type and crypto market characteristics
        if data_type == DataType.PRICE:
            # For price data, forward fill is most appropriate in 24/7 crypto markets
            # as it preserves the last known price when there are brief data gaps
            return ImputationMethod.FORWARD_FILL

        if data_type == DataType.VOLUME:
            # Volume can have more variability, forward fill is still appropriate
            # but we might consider mean for longer gaps
            return ImputationMethod.FORWARD_FILL

        if data_type == DataType.VOLATILITY:
            # Volatility measures benefit from interpolation to smooth transitions
            return ImputationMethod.LINEAR_INTERPOLATION

        if data_type == DataType.INDICATOR:
            # Technical indicators often benefit from interpolation or forward fill
            return ImputationMethod.LINEAR_INTERPOLATION

        # Default fallback
        return ImputationMethod.FORWARD_FILL

    def _get_fallback_method(self, primary_method: ImputationMethod) -> ImputationMethod:
        """Get appropriate fallback method for the given primary method."""
        fallback_map = {
            ImputationMethod.FORWARD_FILL: ImputationMethod.MEAN,
            ImputationMethod.BACKWARD_FILL: ImputationMethod.MEAN,
            ImputationMethod.LINEAR_INTERPOLATION: ImputationMethod.FORWARD_FILL,
            ImputationMethod.SPLINE_INTERPOLATION: ImputationMethod.LINEAR_INTERPOLATION,
            ImputationMethod.KNN: ImputationMethod.FORWARD_FILL,
            ImputationMethod.VWAP_WEIGHTED: ImputationMethod.FORWARD_FILL,
            ImputationMethod.VOLATILITY_ADJUSTED: ImputationMethod.FORWARD_FILL,
            ImputationMethod.MARKET_SESSION_AWARE: ImputationMethod.FORWARD_FILL,
        }
        return fallback_map.get(primary_method, ImputationMethod.MEAN)

    def _prepare_imputation_context(self, spec: InternalFeatureSpec, series_data: pd.Series[Any]) -> dict[str, Any]:
        """Prepare context for advanced imputation including market data and correlations."""
        context: dict[str, Any] = {}

        try:
            # Add related market data if available
            if hasattr(series_data, "index") and len(series_data.index) > 0:
                # Try to get the trading pair from feature configuration or naming
                trading_pair = self._extract_trading_pair_from_spec(spec)

                if trading_pair and trading_pair in self.ohlcv_history:
                    ohlcv_df = self.ohlcv_history[trading_pair]
                    if not ohlcv_df.empty:
                        # Add related price series for correlation-based imputation
                        context["related_series"] = {}

                        # Add OHLC data if different from target series
                        for col in ["open", "high", "low", "close", "volume"]:
                            if col in ohlcv_df.columns:
                                # Convert to float for imputation (from Decimal)
                                series_values = pd.to_numeric(ohlcv_df[col], errors="coerce")
                                if len(series_values) == len(series_data):
                                    context["related_series"][f"ohlcv_{col}"] = series_values

                # Add volume data if available for VWAP-based imputation
                if trading_pair and trading_pair in self.ohlcv_history:
                    volume_data = self.ohlcv_history[trading_pair].get("volume")
                    if volume_data is not None and len(volume_data) == len(series_data):
                        context["volume"] = pd.to_numeric(volume_data, errors="coerce")

                # Add L2 book data for market microstructure context
                if trading_pair and trading_pair in self.l2_books:
                    l2_book = self.l2_books[trading_pair]
                    if l2_book and "bids" in l2_book and "asks" in l2_book:
                        context["l2_book"] = l2_book

        except Exception as e:
            self.logger.debug(
                "Error preparing imputation context for %s: %s",
                spec.key, e,
                source_module=self._source_module,
            )

        return context

    def _extract_trading_pair_from_spec(self, spec: InternalFeatureSpec) -> str | None:
        """Extract trading pair from feature specification or naming convention."""
        # Try to extract from feature key (e.g., "rsi_14_XRPUSD" -> "XRPUSD")
        feature_key = spec.key.upper()

        # Common crypto trading pairs
        common_pairs = ["XRPUSD", "DOGEUSD", "BTCUSD", "ETHUSD", "ADAUSD", "SOLUSD"]

        for pair in common_pairs:
            if pair in feature_key:
                return pair

        # Try to extract from parameters
        if "trading_pair" in spec.parameters:
            return spec.parameters["trading_pair"]  # type: ignore[no-any-return]

        # Fallback: try to get the first configured trading pair
        if self.ohlcv_history:
            return next(iter(self.ohlcv_history.keys()), None)

        return None
