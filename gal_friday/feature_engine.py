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
from dataclasses import dataclass, field  # Added for InternalFeatureSpec
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum  # Added for _LocalFeatureCategory
from pathlib import Path  # Added for feature registry
import time
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, cast
import uuid

import asyncio
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import yaml  # Added for feature registry

from gal_friday.core.events import APIErrorEvent, EventType
from gal_friday.core.feature_models import (
    PublishedFeaturesV1,  # Added for Pydantic model
)
from gal_friday.dal.models import DataQualityIssue

# Try to import enhanced features if available
try:
    from .feature_engine_enhancements import (
        AdvancedSpreadCalculator,
        ComprehensiveFeatureValidator,
        ImputationStrategy,
        IntelligentImputationEngine,
        MarketMicrostructureData,
    )

    HAS_ENHANCEMENTS = True
except ImportError:
    # Define dummy classes for type checking
    ComprehensiveFeatureValidator = None  # type: ignore[assignment,misc]
    HAS_ENHANCEMENTS = False

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
    from gal_friday.interfaces import HistoricalDataService
    from gal_friday.logger_service import LoggerService

# Type definitions for L2 order book data


class L2BookSnapshot(TypedDict):
    """Type definition for L2 order book snapshot."""

    bids: list[list[float | str | Decimal]]  # List of [price, volume] pairs
    asks: list[list[float | str | Decimal]]  # List of [price, volume] pairs
    timestamp: Any  # Can be datetime, pd.Timestamp, etc.


# L2 book series can contain either book snapshots or other values (e.g., float for fallback)
L2BookValue = L2BookSnapshot | float | None


# Define InternalFeatureSpec
@dataclass
class InternalFeatureSpec:
    """Internal specification for feature calculation configuration."""

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
    output_specs: list[OutputSpec] = field(default_factory=list[Any])  # Detailed specifications for each output
    output_naming_pattern: str | None = None  # Pattern for naming outputs (e.g., '{feature_name}_{output_name}')
    dependencies: list[str] = field(default_factory=list[Any])  # Other features this depends on
    required_lookback_periods: int = 1  # Minimum data points required
    author: str | None = None  # Feature author/creator
    created_at: str | None = None  # Creation timestamp
    tags: list[str] = field(default_factory=list[Any])  # Feature tags for organization
    cache_enabled: bool = True  # Whether to cache feature results
    cache_ttl_minutes: int | None = None  # Cache time-to-live in minutes
    computation_priority: int = 5  # Computation priority (1-10)

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
            raise FeatureProcessingError(f"Failed to process feature outputs: {e}") from e
        else:
            return final_outputs

    def _standardize_raw_outputs(self, raw_outputs: Any) -> pd.DataFrame:
        """Convert various output formats to standardized DataFrame."""
        # Use type-based dispatch for cleaner code
        converters: dict[type, Callable[[Any], pd.DataFrame]] = {
            pd.DataFrame: self._convert_dataframe,
            pd.Series: self._convert_series,
            np.ndarray: self._convert_numpy_array,
            dict: self._convert_dict,
            list: self._convert_list,
        }

        for output_type, converter in converters.items():
            if isinstance(raw_outputs, output_type):
                return converter(raw_outputs)

        # Default: treat as scalar value
        return self._convert_scalar(raw_outputs)

    def _convert_dataframe(self, raw_outputs: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame output (passthrough)."""
        return raw_outputs

    def _convert_series(self, raw_outputs: pd.Series) -> pd.DataFrame:
        """Convert Series to DataFrame."""
        output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
        return pd.DataFrame({output_name: raw_outputs})

    def _convert_numpy_array(self, raw_outputs: np.ndarray[Any, Any]) -> pd.DataFrame:
        """Convert NumPy array to DataFrame."""
        if raw_outputs.ndim == 1:
            output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
            return pd.DataFrame({output_name: raw_outputs})

        # Multi-dimensional array
        columns = self._generate_column_names(raw_outputs.shape[1])
        return pd.DataFrame(raw_outputs, columns=columns)

    def _convert_dict(self, raw_outputs: dict[str, Any]) -> pd.DataFrame:
        """Convert dictionary to DataFrame."""
        return pd.DataFrame(raw_outputs)

    def _convert_list(self, raw_outputs: list[Any]) -> pd.DataFrame:
        """Convert list to DataFrame."""
        if len(self.spec.output_specs) == 1:
            output_name = self.spec.output_specs[0].name
            return pd.DataFrame({output_name: raw_outputs})

        # Multiple outputs - map to specs
        output_dict = self._map_list_to_specs(raw_outputs)
        return pd.DataFrame(output_dict)

    def _convert_scalar(self, raw_outputs: Any) -> pd.DataFrame:
        """Convert scalar value to DataFrame."""
        output_name = self.spec.output_specs[0].name if self.spec.output_specs else "value"
        return pd.DataFrame({output_name: [raw_outputs]})

    def _generate_column_names(self, num_columns: int) -> list[str]:
        """Generate column names for multi-dimensional output."""
        columns = [spec.name for spec in self.spec.output_specs[:num_columns]]
        # Generate default names for additional columns
        columns.extend([f"output_{i}" for i in range(len(columns), num_columns)])
        return columns

    def _map_list_to_specs(self, raw_outputs: list[Any]) -> dict[str, Any]:
        """Map list values to output specifications."""
        output_dict = {}
        for i, spec in enumerate(self.spec.output_specs):
            if i < len(raw_outputs):
                output_dict[spec.name] = (
                    raw_outputs[i] if isinstance(raw_outputs[i], list | np.ndarray) else [raw_outputs[i]]
                )
        return output_dict

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

            # Process each column according to its specification
            processed[spec.name] = self._process_column_by_spec(
                processed[spec.name],
                spec,
            )

        return processed

    def _process_column_by_spec(self, column: pd.Series, spec: Any) -> pd.Series:
        """Process a single column according to its specification."""
        # Apply type conversion
        column = self._apply_type_conversion(column, spec.output_type)

        # Apply validation range
        column = self._apply_validation_range(column, spec)

        # Handle nullability
        return self._handle_nullability(column, spec)

    def _apply_type_conversion(self, column: pd.Series, output_type: OutputType) -> pd.Series:
        """Apply type conversion based on output type."""
        type_converters = {
            OutputType.NUMERIC: lambda col: pd.to_numeric(col, errors="coerce"),
            OutputType.CATEGORICAL: lambda col: col.astype("category"),
            OutputType.BOOLEAN: lambda col: col.astype(bool),
            OutputType.TIMESTAMP: lambda col: pd.to_datetime(col, errors="coerce"),
            OutputType.STRING: lambda col: col.astype(str),
        }

        converter = type_converters.get(output_type)
        return converter(column) if converter else column

    def _apply_validation_range(self, column: pd.Series, spec: Any) -> pd.Series:
        """Apply validation range to numeric columns."""
        if not (spec.validation_range and spec.output_type == OutputType.NUMERIC):
            return column

        min_val, max_val = spec.validation_range
        out_of_range = (column < min_val) | (column > max_val)

        if out_of_range.any() and self.logger:
            self.logger.warning(
                f"Feature {self.spec.key} output {spec.name} has "
                f"{out_of_range.sum()} values outside range [{min_val}, {max_val}]",
            )

        # Clip values to range
        return column.clip(min_val, max_val)

    def _handle_nullability(self, column: pd.Series, spec: Any) -> pd.Series:
        """Handle null values according to specification."""
        if spec.nullable or not column.isnull().any():
            return column

        if spec.default_value is not None:
            return column.fillna(spec.default_value)

        raise FeatureValidationError(
            f"Feature {self.spec.key} output {spec.name} contains null values but nullability is disabled",
        )

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
        self.quality_thresholds = config.get(
            "feature_quality",
            {
                "min_completeness": 0.8,
                "max_correlation": 0.95,
                "min_variance": 1e-6,
            },
        )

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
        start_time = time.time()

        try:
            if self.logger:
                self.logger.info(f"Extracting {len(feature_specs)} advanced features from {len(data)} data points")

            # Validate input data
            self._validate_input_data(data)

            # Initialize feature DataFrame
            features_df = pd.DataFrame(index=data.index)

            # Extract features by category for optimal performance
            for category in [
                FeatureCategory.TECHNICAL,
                FeatureCategory.ORDERBOOK,
                FeatureCategory.MICROSTRUCTURE,
                FeatureCategory.VOLATILITY,
            ]:
                category_specs = [spec for spec in feature_specs if spec.category == category]
                if category_specs:
                    category_features = await self._extract_category_features(
                        data,
                        category_specs,
                        l2_data,
                        trade_data,
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
            raise FeatureProcessingError(f"Advanced feature extraction failed: {e}") from e
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
            *(
                spec.required_lookback_periods
                for spec in self.feature_registry.values()
                if hasattr(spec, "required_lookback_periods")
            ),
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
            to_drop = [
                column
                for column in upper_triangle.columns
                if any(upper_triangle[column] > self.quality_thresholds["max_correlation"])
            ]

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
        tr_df = pd.DataFrame(
            {
                "hl": high_low,
                "hc": high_close_prev,
                "lc": low_close_prev,
            },
        )
        return tr_df.max(axis=1)

    def _calculate_atr_advanced(self, data: pd.DataFrame, period: int = 14) -> pd.Series[Any]:
        """Calculate Advanced Average True Range with additional smoothing."""
        tr = self._calculate_true_range(data)
        return tr.ewm(span=period).mean()  # Use exponential moving average

    def _calculate_volatility_ratio(
        self,
        data: pd.DataFrame,
        short_period: int = 10,
        long_period: int = 30,
    ) -> pd.Series[Any]:
        """Calculate volatility ratio."""
        short_vol = data["close"].pct_change().rolling(window=short_period).std()
        long_vol = data["close"].pct_change().rolling(window=long_period).std()
        return short_vol / long_vol

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series[Any]:
        """Calculate On-Balance Volume."""
        price_change: pd.Series[Any] = data["close"].diff()
        obv = np.where(
            price_change > 0,
            data["volume"],
            np.where(price_change < 0, -data["volume"], 0),
        )
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
        self,
        data: pd.DataFrame,
        short_period: int = 14,
        long_period: int = 28,
    ) -> pd.Series[Any]:
        """Calculate Volume Oscillator."""
        short_vol_avg = data["volume"].rolling(window=short_period).mean()
        long_vol_avg = data["volume"].rolling(window=long_period).mean()
        return ((short_vol_avg - long_vol_avg) / long_vol_avg) * 100

    # Market microstructure features
    def _calculate_effective_spread(self, data: pd.DataFrame, l2_data: dict[str, Any]) -> pd.Series[Any]:
        """Calculate effective spread from L2 data with production-grade implementation."""
        try:
            # Check if enhanced spread calculator is available
            if not HAS_ENHANCEMENTS:
                self._raise_import_error("Enhanced features not available")

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
        self,
        data: pd.DataFrame,
        l2_data: dict[str, Any],
        levels: int = 5,
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
            ama.iloc[i] = ama.iloc[i - 1] + smoothing_constant.iloc[i] * (data["close"].iloc[i] - ama.iloc[i - 1])

        return ama

    def _calculate_fractal_dimension(self, data: pd.DataFrame, period: int = 20) -> pd.Series[Any]:
        """Calculate fractal dimension."""

        def fd_single(series: pd.Series[Any]) -> float:
            if len(series) < 2:
                return np.nan

            n = len(series)
            # Calculate relative range
            cumulative_deviations = np.cumsum(series - series.mean())
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)  # noqa: N806

            # Calculate standard deviation
            S = np.std(series)  # noqa: N806

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
                # Prepare data
                log_returns = self._calculate_log_returns(series)
                if log_returns is None or len(log_returns) < 4:
                    return np.nan

                # Calculate R/S statistics
                rs_values = self._calculate_rs_statistics(log_returns)
                if len(rs_values) < 2:
                    return np.nan

                # Calculate Hurst exponent via linear regression
                return self._calculate_hurst_from_rs(rs_values)

            except Exception:
                return np.nan

        return data["close"].rolling(window=period).apply(hurst_single)

    def _calculate_log_returns(self, series: pd.Series[Any]) -> pd.Series[Any] | None:
        """Calculate log returns from a price series."""
        try:
            returns = series / series.shift(1)
            log_returns = pd.Series(np.log(returns)).dropna()
            return log_returns if len(log_returns) > 0 else None
        except Exception:
            return None

    def _calculate_rs_statistics(self, log_returns: pd.Series[Any]) -> list[float]:
        """Calculate R/S statistics for different lags."""
        lags = range(2, min(len(log_returns) // 2, 20))
        rs_values = []

        for lag in lags:
            rs_mean = self._calculate_rs_for_lag(log_returns, lag)
            if rs_mean is not None:
                rs_values.append(rs_mean)

        return rs_values

    def _calculate_rs_for_lag(self, log_returns: pd.Series[Any], lag: int) -> float | None:
        """Calculate average R/S statistic for a specific lag."""
        n_chunks = len(log_returns) // lag
        if n_chunks < 1:
            return None

        rs_chunk = []
        for i in range(n_chunks):
            chunk_rs = self._calculate_chunk_rs(log_returns, i, lag)
            if chunk_rs is not None:
                rs_chunk.append(chunk_rs)

        return float(np.mean(rs_chunk)) if rs_chunk else None

    def _calculate_chunk_rs(self, log_returns: pd.Series[Any], chunk_index: int, lag: int) -> float | None:
        """Calculate R/S statistic for a single chunk."""
        chunk = log_returns[chunk_index * lag : (chunk_index + 1) * lag]
        if len(chunk) != lag:
            return None

        mean_chunk = chunk.mean()
        cumsum_chunk = (chunk - mean_chunk).cumsum()
        r_chunk = cumsum_chunk.max() - cumsum_chunk.min()
        s_chunk = chunk.std()

        return r_chunk / s_chunk if s_chunk > 0 else None

    def _calculate_hurst_from_rs(self, rs_values: list[float]) -> float:
        """Calculate Hurst exponent from R/S values using linear regression."""
        n_values = len(rs_values)
        lags = list(range(2, 2 + n_values))

        log_lags = np.log(lags)
        log_rs = np.log(rs_values)

        # Simple linear regression
        n = len(log_lags)
        sum_x = np.sum(log_lags)
        sum_y = np.sum(log_rs)
        sum_xy = np.sum(log_lags * log_rs)
        sum_x2 = np.sum(log_lags**2)

        denominator = n * sum_x2 - sum_x**2
        if denominator == 0:
            return np.nan

        hurst = (n * sum_xy - sum_x * sum_y) / denominator
        return float(hurst)

    def _raise_import_error(self, error_msg: str) -> None:
        """Helper method to raise ImportError, abstracting the raise from try block."""
        raise ImportError(error_msg)


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
        history_repo: HistoryRepository | None = None,
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
        self._background_tasks: set[asyncio.Task[Any]] = set()

        # Feature configuration derived from config
        self._feature_configs: dict[str, InternalFeatureSpec] = {}  # Changed type hint
        self._extract_feature_configs()

        # Initialize feature handlers dispatcher
        # RSI and MACD handlers are removed as they will be handled by sklearn pipelines.
        self._feature_handlers: dict[
            str,
            Callable[..., dict[str, Any] | dict[str, str] | None],
        ] = {
            # Note: vwap_ohlcv and vwap_trades are handled by _process_vwap_feature
        }

        # Initialize data storage
        # OHLCV data will be stored in a DataFrame per trading pair
        # Columns: timestamp_bar_start (index), open, high, low, close, volume
        self.ohlcv_history: dict[str, pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).astype(
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
        self.l2_books: dict[str, dict[str, Any]] = defaultdict(dict[str, Any])

        # Store recent trades for calculating true Volume Delta and trade-based VWAP
        # deque stores: {"ts": datetime, "price": Decimal, "vol": Decimal, "side": "buy"/"sell"}
        trade_history_maxlen = config.get("feature_engine", {}).get("trade_history_maxlen", 2000)
        self.trade_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=trade_history_maxlen))

        # Store L2 book history for better alignment with bar timestamps
        # Each entry: {"timestamp": datetime, "book": {"bids": [...], "asks": [...]}}
        l2_history_maxlen = config.get("feature_engine", {}).get("l2_history_maxlen", 100)
        self.l2_books_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=l2_history_maxlen))

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

    # Feature type dispatch table for determining calculator and input types
    FEATURE_TYPE_DISPATCH: ClassVar[dict[str, tuple[str, str]]] = {
        "rsi": ("rsi", "close_series"),
        "macd": ("macd", "close_series"),
        "bbands": ("bbands", "close_series"),
        "roc": ("roc", "close_series"),
        "atr": ("atr", "ohlcv_df"),
        "stdev": ("stdev", "close_series"),
        "vwap_ohlcv": ("vwap_ohlcv", "ohlcv_df"),
        "l2_spread": ("l2_spread", "l2_book_series"),
        "l2_imbalance": ("l2_imbalance", "l2_book_series"),
        "l2_wap": ("l2_wap", "l2_book_series"),
        "l2_depth": ("l2_depth", "l2_book_series"),
        "vwap_trades": ("vwap_trades", "trades_and_bar_starts"),
        "volume_delta": ("volume_delta", "trades_and_bar_starts"),
    }

    def _determine_calculator_type_and_input(
        self,
        feature_key: str,
        raw_cfg: dict[str, Any],
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
        calc_type = raw_cfg.get("calculator_type", raw_cfg.get("type"))
        input_type = raw_cfg.get("input_type")

        if calc_type and input_type:
            return calc_type, input_type

        # Infer from key if not explicitly provided
        key_lower = feature_key.lower()

        # Use dispatch table to find matching feature type
        for feature_pattern, (calc, input_src) in self.FEATURE_TYPE_DISPATCH.items():
            if feature_pattern in key_lower:
                return calc, input_src

        self.logger.warning("Could not determine calculator_type or input_type for feature key: %s", feature_key)
        return None, None

    def _extract_feature_configs(self) -> None:
        """Orchestrates loading and parsing of feature configurations.

        This simplified method loads feature definitions from the registry and
        application config, then delegates processing based on config type.
        """
        # Load registry definitions
        registry_definitions = self._load_feature_registry(DEFAULT_FEATURE_REGISTRY_PATH)

        # Get app-level feature configuration
        app_feature_config = self.config.get("features", {})

        # Process based on configuration type
        if isinstance(app_feature_config, list):
            parsed_specs = self._process_list_activation(app_feature_config, registry_definitions)
        elif isinstance(app_feature_config, dict):
            parsed_specs = self._process_dict_overrides(app_feature_config, registry_definitions)
        else:
            self.logger.warning(
                "App 'features' config is neither a list nor dict. No features loaded.",
                source_module=self._source_module,
            )
            parsed_specs = {}

        self._feature_configs = parsed_specs
        if not self._feature_configs:
            self.logger.warning(
                "No features were successfully parsed or activated. FeatureEngine might not produce any features.",
                source_module=self._source_module,
            )

    def _process_list_activation(
        self,
        keys: list[Any],
        registry: dict[str, Any],
    ) -> dict[str, InternalFeatureSpec]:
        """Processes features activated from a simple list of keys.

        Args:
            keys: List of feature keys to activate from registry
            registry: Loaded feature registry definitions

        Returns:
            Dictionary of parsed InternalFeatureSpec objects
        """
        specs: dict[str, InternalFeatureSpec] = {}

        for key in keys:
            if not isinstance(key, str):
                self.logger.warning(
                    "Feature activation list contains non-string item: %s. Skipping.",
                    key,
                    source_module=self._source_module,
                )
                continue

            if key not in registry:
                self.logger.warning(
                    "Feature key '%s' not found in registry. Skipping.",
                    key,
                    source_module=self._source_module,
                )
                continue

            feature_def = registry[key]
            if not isinstance(feature_def, dict):
                self.logger.warning(
                    "Registry definition for '%s' is not a dictionary. Skipping.",
                    key,
                    source_module=self._source_module,
                )
                continue

            spec = self._parse_single_feature_definition(key, feature_def.copy())
            if spec:
                specs[key] = spec

        return specs

    def _process_dict_overrides(
        self,
        overrides: dict[str, Any],
        registry: dict[str, Any],
    ) -> dict[str, InternalFeatureSpec]:
        """Processes features from a dictionary of overrides or ad-hoc definitions.

        Args:
            overrides: Dictionary of feature configurations with potential overrides
            registry: Loaded feature registry definitions

        Returns:
            Dictionary of parsed InternalFeatureSpec objects
        """
        specs: dict[str, InternalFeatureSpec] = {}

        for key, config in overrides.items():
            if not isinstance(config, dict):
                self.logger.warning(
                    "Feature config for '%s' is not a dict. Skipping.",
                    key,
                    source_module=self._source_module,
                )
                continue

            # Get final merged configuration
            final_config = self._get_final_feature_config(key, config, registry)
            if not final_config:
                continue

            spec = self._parse_single_feature_definition(key, final_config)
            if spec:
                specs[key] = spec

        return specs

    def _get_final_feature_config(
        self,
        key: str,
        override_config: dict[str, Any],
        registry: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Merges registry and override configs or validates ad-hoc definitions.

        Args:
            key: Feature name
            override_config: Configuration from application
            registry: Loaded feature registry

        Returns:
            Final merged configuration or None if invalid
        """
        base_config = registry.get(key)

        if base_config:
            # Feature exists in registry - merge with overrides
            if not isinstance(base_config, dict):
                self.logger.warning(
                    "Registry definition for '%s' is not a dictionary. Skipping.",
                    key,
                    source_module=self._source_module,
                )
                return None
            return self._deep_merge_configs(base_config.copy(), override_config)
        # Ad-hoc feature definition
        self.logger.info(
            "Feature '%s' not found in registry, treating as ad-hoc definition.",
            key,
            source_module=self._source_module,
        )

        # Validate ad-hoc definition has required fields
        if "calculator_type" not in override_config or "input_type" not in override_config:
            self.logger.warning(
                "Ad-hoc feature '%s' missing 'calculator_type' or 'input_type'. Skipping.",
                key,
                source_module=self._source_module,
            )
            return None

        return override_config.copy()

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
        self,
        feature_key: str,
        config_dict: dict[str, Any],
    ) -> InternalFeatureSpec | None:
        """Parses a single, consolidated feature configuration dictionary into an `InternalFeatureSpec` data object.

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
                feature_key,
                config_dict,
            )
            if not calculator_type:
                calculator_type = inferred_calc_type
            if not input_type:
                input_type = inferred_input_type

            if not calculator_type or not input_type:
                self.logger.warning(
                    "Could not determine calculator_type or input_type for feature '%s' "
                    "even after inference. Skipping.",
                    feature_key,
                )
                return None

        parameters = config_dict.get("parameters", config_dict.get("params", {}))
        # Ensure parameters is a dict[str, Any], even if it was null/None in YAML
        if not isinstance(parameters, dict):
            parameters = {}

        # Merge top-level common param keys if they exist and aren't already in 'parameters'
        for common_param_key in [
            "period",
            "length",
            "fast",
            "slow",
            "signal",
            "levels",
            "std_dev",
            "length_seconds",
            "bar_interval_seconds",
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

        category_str = str(config_dict.get("category", "TECHNICAL")).upper()  # Ensure string before upper()
        try:
            category = FeatureCategory[category_str]
        except KeyError:
            self.logger.warning(
                "Invalid FeatureCategory '%s' for feature '%s'. Defaulting to TECHNICAL.",
                category_str,
                feature_key,
                source_module=self._source_module,
            )
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
            version=str(version) if version is not None else None,  # Ensure version is string
            output_properties=output_properties if isinstance(output_properties, dict) else {},
        )
        self.logger.info(
            "Successfully parsed feature spec for key: '%s' (Calc: %s, Input: %s)",
            feature_key,
            calculator_type,
            input_type,
        )
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
        """Constructs Scikit-learn pipelines for all activated features.

        This simplified method delegates the creation of individual pipeline
        components to specialized methods, making the overall flow clearer.
        """
        self.logger.info("Building feature pipelines...", source_module=self._source_module)

        for spec in self._feature_configs.values():
            # Collect pipeline steps
            steps = [
                step
                for step in [
                    self._create_input_imputation_step(spec) if spec.input_type == "close_series" else None,
                    self._create_calculator_step(spec),
                    self._create_output_imputer_step(spec),
                    self._create_output_scaler_step(spec),
                ]
                if step is not None
            ]

            if not steps:
                self.logger.warning("No pipeline steps for feature '%s'.", spec.key)
                continue

            # Create output handler
            output_handler = FeatureOutputHandler(spec)
            output_handler.logger = self.logger
            self.output_handlers[spec.key] = output_handler

            # Build and store pipeline
            pipeline = Pipeline(steps=steps)
            pipeline.set_output(transform="pandas")
            self.feature_pipelines[f"{spec.key}_pipeline"] = {
                "pipeline": pipeline,
                "input_type": spec.input_type,
                "params": spec.parameters,
                "spec": spec,
            }
            self.logger.info(
                "Built pipeline: %s_pipeline with steps: %s",
                spec.key,
                [s[0] for s in steps],
                source_module=self._source_module,
            )

    def _create_calculator_step(self, spec: InternalFeatureSpec) -> tuple[str, Any] | None:
        """Creates the main feature calculation step for the pipeline.

        Args:
            spec: Feature specification containing calculator type and parameters

        Returns:
            Tuple of (step_name, transformer) or None if invalid
        """
        # Find calculator function
        calculator_func = getattr(FeatureEngine, f"_pipeline_compute_{spec.calculator_type}", None)
        if not calculator_func:
            self.logger.error(
                "No _pipeline_compute function found for calculator_type: %s (feature key: %s)",
                spec.calculator_type,
                spec.key,
                source_module=self._source_module,
            )
            return None

        # Use dispatch dictionary for parameter preparation
        arg_builders = {
            "rsi": self._prepare_basic_period_args,
            "roc": self._prepare_basic_period_args,
            "stdev": self._prepare_basic_period_args,
            "macd": self._prepare_macd_args,
            "bbands": self._prepare_bbands_args,
            "atr": self._prepare_atr_args,
            "vwap_ohlcv": self._prepare_vwap_ohlcv_args,
            "l2_imbalance": self._prepare_l2_args,
            "l2_depth": self._prepare_l2_args,
            "l2_wap": self._prepare_l2_args,
            "l2_spread": lambda spec: {},  # No parameters
            "vwap_trades": self._prepare_trade_based_args,
            "volume_delta": self._prepare_trade_based_args,
        }

        builder = arg_builders.get(spec.calculator_type)
        if not builder:
            self.logger.warning(
                "No parameter builder for calculator: %s. Using empty args.",
                spec.calculator_type,
                source_module=self._source_module,
            )
            kw_args = {}
        else:
            kw_args = builder(spec)

        return (
            f"{spec.key}_calculator",
            FunctionTransformer(calculator_func, kw_args=kw_args, validate=False),
        )

    def _prepare_basic_period_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for basic period-based calculators (RSI, ROC, STDEV)."""
        defaults = {"rsi": 14, "roc": 10, "stdev": 20}
        default_period = defaults.get(spec.calculator_type, 14)
        period = spec.parameters.get("period", default_period)

        if "period" not in spec.parameters:
            self.logger.debug(
                "Using default period %s for %s ('%s')",
                period,
                spec.calculator_type,
                spec.key,
                source_module=self._source_module,
            )

        return {"period": period}

    def _prepare_macd_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for MACD calculator."""
        args = {
            "fast": spec.parameters.get("fast", 12),
            "slow": spec.parameters.get("slow", 26),
            "signal": spec.parameters.get("signal", 9),
        }

        if any(p not in spec.parameters for p in ["fast", "slow", "signal"]):
            self.logger.debug(
                "Using default MACD params (f:%s,s:%s,sig:%s) for %s",
                args["fast"],
                args["slow"],
                args["signal"],
                spec.key,
                source_module=self._source_module,
            )

        return args

    def _prepare_bbands_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for Bollinger Bands calculator."""
        args = {
            "length": spec.parameters.get("length", 20),
            "std_dev": float(spec.parameters.get("std_dev", 2.0)),
        }

        if "length" not in spec.parameters or "std_dev" not in spec.parameters:
            self.logger.debug(
                "Using default BBands params (l:%s,s:%.1f) for %s",
                args["length"],
                args["std_dev"],
                spec.key,
                source_module=self._source_module,
            )

        return args

    def _prepare_atr_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for ATR calculator."""
        length = spec.parameters.get("length", 14)

        if "length" not in spec.parameters:
            self.logger.debug(
                "Using default ATR length %s for %s",
                length,
                spec.key,
                source_module=self._source_module,
            )

        return {"length": length}

    def _prepare_vwap_ohlcv_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for VWAP OHLCV calculator."""
        length = spec.parameters.get("length", 14)

        if "length" not in spec.parameters:
            self.logger.debug(
                "Using default VWAP_OHLCV length %s for %s",
                length,
                spec.key,
                source_module=self._source_module,
            )

        return {"length": length}

    def _prepare_l2_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for L2 book-based calculators."""
        default_levels = {
            "l2_imbalance": 5,
            "l2_depth": 5,
            "l2_wap": 1,
            "l2_spread": 0,
        }

        default = default_levels.get(spec.calculator_type, 5)
        if default == 0:  # l2_spread has no levels parameter
            return {}

        levels = spec.parameters.get("levels", default)

        if "levels" not in spec.parameters:
            self.logger.debug(
                "Using default levels %s for %s ('%s')",
                levels,
                spec.calculator_type,
                spec.key,
                source_module=self._source_module,
            )

        return {"levels": levels}

    def _prepare_trade_based_args(self, spec: InternalFeatureSpec) -> dict[str, Any]:
        """Prepares arguments for trade-based calculators."""
        bar_interval = spec.parameters.get(
            "bar_interval_seconds",
            spec.parameters.get("length_seconds", 60),
        )

        if "bar_interval_seconds" not in spec.parameters and "length_seconds" not in spec.parameters:
            self.logger.debug(
                "Using default bar_interval_seconds %s for %s ('%s')",
                bar_interval,
                spec.calculator_type,
                spec.key,
                source_module=self._source_module,
            )

        return {"bar_interval_seconds": bar_interval}

    def _create_output_imputer_step(self, spec: InternalFeatureSpec) -> tuple[str, Any] | None:
        """Creates output imputation step based on feature configuration.

        Args:
            spec: Feature specification containing imputation config

        Returns:
            Tuple of (step_name, transformer) or None if passthrough
        """
        if spec.imputation == "passthrough":
            self.logger.debug("Imputation set to 'passthrough' for %s.", spec.key)
            return None

        # Determine default fill value based on feature type
        default_fills = {
            "rsi": 50.0,
            "atr": np.nan,
            "vwap_ohlcv": np.nan,
            "l2_wap": np.nan,
            "vwap_trades": np.nan,
            "stdev": np.nan,
        }
        default_fill = default_fills.get(spec.calculator_type, 0.0)

        # Determine strategy and fill value
        strategy = "default"
        fill_value = default_fill

        if isinstance(spec.imputation, dict):
            strategy = spec.imputation.get("strategy", "constant")
            if strategy == "constant":
                fill_value = spec.imputation.get("fill_value", default_fill)
        elif spec.imputation is not None:
            self.logger.warning(
                "Unrecognized imputation config for %s: %s. No imputer added.",
                spec.key,
                spec.imputation,
                source_module=self._source_module,
            )
            return None

        # Create transform function
        transform_funcs = {
            "constant": lambda x, fv=fill_value: x.fillna(fv),
            "mean": lambda x: x.fillna(x.mean()),
            "median": lambda x: x.fillna(x.median()),
            "default": lambda x, fv=default_fill: x.fillna(fv),
        }

        transform_func = transform_funcs.get(strategy)
        if not transform_func:
            self.logger.warning(
                "Unknown imputation strategy '%s' for %s. No imputer added.",
                strategy,
                spec.key,
                source_module=self._source_module,
            )
            return None

        # Create step name
        step_suffixes = {
            "constant": f"const_{fill_value}",
            "mean": "mean",
            "median": "median",
            "default": f"default_fill_{default_fill}",
        }
        step_name_suffix = step_suffixes[strategy]

        self.logger.debug(
            "Using output imputer strategy '%s' for %s",
            strategy,
            spec.key,
            source_module=self._source_module,
        )

        return (
            f"{spec.key}_output_imputer_{step_name_suffix}",
            FunctionTransformer(transform_func, validate=False),
        )

    def _create_output_scaler_step(self, spec: InternalFeatureSpec) -> tuple[str, Any] | None:
        """Creates output scaling step based on feature configuration.

        Args:
            spec: Feature specification containing scaling config

        Returns:
            Tuple of (step_name, transformer) or None if passthrough
        """
        if spec.scaling == "passthrough" or spec.scaling is None:
            if spec.scaling == "passthrough":
                self.logger.debug("Scaling set to 'passthrough' for %s.", spec.key)
            return None

        # Determine scaler type and parameters
        scaler_instance = StandardScaler()  # Default
        scaler_name_suffix = "StandardScaler"

        if isinstance(spec.scaling, dict):
            method = spec.scaling.get("method", "standard")

            if method == "minmax":
                feature_range = spec.scaling.get("feature_range", (0, 1))
                scaler_instance = MinMaxScaler(feature_range=feature_range)
                scaler_name_suffix = f"MinMaxScaler_{feature_range}"

            elif method == "robust":
                quantile_range = spec.scaling.get("quantile_range", (25.0, 75.0))
                scaler_instance = RobustScaler(quantile_range=quantile_range)
                scaler_name_suffix = f"RobustScaler_{quantile_range}"

            elif method != "standard":
                self.logger.warning(
                    "Unknown scaling method '%s' for %s. Using StandardScaler.",
                    method,
                    spec.key,
                    source_module=self._source_module,
                )
        elif isinstance(spec.scaling, str) and spec.scaling not in ["standard", "passthrough"]:
            self.logger.warning(
                "Simple string for scaling '%s' for %s is ambiguous. Using StandardScaler.",
                spec.scaling,
                spec.key,
                source_module=self._source_module,
            )

        self.logger.debug(
            "Using %s for scaling %s",
            type(scaler_instance).__name__,
            spec.key,
            source_module=self._source_module,
        )

        return (
            f"{spec.key}_output_scaler_{scaler_name_suffix}",
            PandasScalerTransformer(scaler_instance),
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

    async def _handle_l2_update(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Parse and store an L2 order book update."""
        try:
            # Validate and extract order book data
            raw_bids, raw_asks = self._extract_l2_data(l2_payload)
            if not self._validate_l2_data(raw_bids, raw_asks):
                return await self._handle_malformed_l2_data(trading_pair, l2_payload, "invalid_format")

            # Process bid and ask levels
            processed_bids = self._process_l2_levels(raw_bids, "bid", trading_pair)
            processed_asks = self._process_l2_levels(raw_asks, "ask", trading_pair)

            # Store processed data
            self._store_l2_book(trading_pair, processed_bids, processed_asks, l2_payload)

            # Update history
            await self._update_l2_history(trading_pair, processed_bids, processed_asks, l2_payload)

            self._log_l2_update(trading_pair, processed_bids, processed_asks)

        except KeyError:
            self._log_l2_key_error(trading_pair, l2_payload)
        except Exception:
            self._log_l2_general_error(trading_pair, l2_payload)

    def _extract_l2_data(self, l2_payload: dict[str, Any]) -> tuple[Any, Any]:
        """Extract bids and asks from L2 payload."""
        return l2_payload.get("bids"), l2_payload.get("asks")

    def _validate_l2_data(self, raw_bids: Any, raw_asks: Any) -> bool:
        """Validate that bids and asks are lists."""
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
            self.logger.warning(
                "L2 bids/asks are not lists",
                source_module=self._source_module,
            )
            return False
        return True

    def _process_l2_levels(
        self,
        raw_levels: list[Any],
        level_type: str,
        trading_pair: str,
    ) -> list[list[Decimal]]:
        """Process bid or ask levels, converting to Decimal."""
        processed_levels = []

        for i, level in enumerate(raw_levels):
            processed_level = self._process_single_l2_level(level, i, level_type, trading_pair)
            if processed_level is not None:
                processed_levels.append(processed_level)

        return processed_levels

    def _process_single_l2_level(
        self,
        level: Any,
        index: int,
        level_type: str,
        trading_pair: str,
    ) -> list[Decimal] | None:
        """Process a single L2 level (bid or ask)."""
        if not isinstance(level, list | tuple) or len(level) != self._EXPECTED_L2_LEVEL_LENGTH:
            self.logger.warning(
                "Malformed L2 %s level %s for %s: %s",
                level_type,
                index,
                trading_pair,
                level,
                source_module=self._source_module,
            )
            return None

        try:
            return [Decimal(level[0]), Decimal(level[1])]
        except (ValueError, TypeError) as e:
            self.logger.warning(
                "Error converting L2 %s level %s for %s: %s - %s",
                level_type,
                index,
                trading_pair,
                level,
                e,
                source_module=self._source_module,
            )
            return None

    def _store_l2_book(
        self,
        trading_pair: str,
        processed_bids: list[list[Decimal]],
        processed_asks: list[list[Decimal]],
        l2_payload: dict[str, Any],
    ) -> None:
        """Store the processed L2 book data."""
        self.l2_books[trading_pair] = {
            "bids": processed_bids,
            "asks": processed_asks,
            "timestamp": pd.to_datetime(l2_payload.get("timestamp_exchange") or datetime.now(UTC), utc=True),
        }

    async def _update_l2_history(
        self,
        trading_pair: str,
        processed_bids: list[list[Decimal]],
        processed_asks: list[list[Decimal]],
        l2_payload: dict[str, Any],
    ) -> None:
        """Update L2 history with timestamp."""
        timestamp_str = l2_payload.get("timestamp_exchange") or l2_payload.get("timestamp")
        if not timestamp_str:
            return

        try:
            timestamp = pd.to_datetime(timestamp_str, utc=True)
            self.l2_books_history[trading_pair].append(
                {
                    "timestamp": timestamp,
                    "book": {"bids": processed_bids, "asks": processed_asks},
                },
            )
        except Exception as e:
            self.logger.warning(
                "Failed to parse L2 timestamp for history: %s",
                e,
                source_module=self._source_module,
            )

    def _log_l2_update(
        self,
        trading_pair: str,
        processed_bids: list[Any],
        processed_asks: list[Any],
    ) -> None:
        """Log successful L2 update."""
        self.logger.debug(
            "Updated L2 book for %s: %s bids, %s asks",
            trading_pair,
            len(processed_bids),
            len(processed_asks),
            source_module=self._source_module,
        )

    def _log_l2_key_error(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Log L2 key error."""
        self.logger.exception(
            "Missing key in L2 payload for %s.",
            trading_pair,
            source_module=self._source_module,
            context={"payload": l2_payload},
        )

    def _log_l2_general_error(self, trading_pair: str, l2_payload: dict[str, Any]) -> None:
        """Log general L2 processing error."""
        self.logger.exception(
            "Unexpected error handling L2 update for %s",
            trading_pair,
            source_module=self._source_module,
            context={"payload": l2_payload},
        )

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
                self.l2_books_history[trading_pair].append(
                    {
                        "timestamp": pd.to_datetime(datetime.now(UTC), utc=True),
                        "book": sanitized_book,
                    },
                )
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
        try:
            sanitized_bids = self._sanitize_order_levels(l2_payload.get("bids", []))
            sanitized_asks = self._sanitize_order_levels(l2_payload.get("asks", []))

            if sanitized_bids or sanitized_asks:
                return {"bids": sanitized_bids, "asks": sanitized_asks}

        except Exception as e:
            self.logger.debug(f"L2 data sanitization failed: {e}", source_module=self._source_module)

        return None

    def _sanitize_order_levels(self, raw_levels: Any) -> list[list[Decimal]]:
        """Sanitize a list of order book levels (bids or asks).

        Args:
            raw_levels: Raw order book levels to sanitize

        Returns:
            List of sanitized [price, volume] pairs
        """
        if not isinstance(raw_levels, list | tuple):
            return []

        sanitized_levels = []
        for level in raw_levels:
            sanitized_level = self._sanitize_single_level(level)
            if sanitized_level:
                sanitized_levels.append(sanitized_level)

        return sanitized_levels

    def _sanitize_single_level(self, level: Any) -> list[Decimal] | None:
        """Sanitize a single order book level.

        Args:
            level: Single level to sanitize

        Returns:
            [price, volume] pair or None if invalid
        """
        if not self._is_valid_l2_level(level):
            return None

        try:
            price = Decimal(str(level[0]))
            volume = Decimal(str(level[1]))

            # Allow zero volume for cancellations
            if price > 0 and volume >= 0:
                return [price, volume]
        except (ValueError, TypeError, IndexError) as e:
            # Log the error for monitoring data quality issues
            self.logger.debug(
                f"Invalid order book level format: {level}, error: {e}",
                source_module=self._source_module,
                context={"error_type": type(e).__name__, "level_data": str(level)[:100]},
            )

        return None

    def _is_valid_l2_level(self, level: Any) -> bool:
        """Check if an L2 level has valid structure."""
        return isinstance(level, list | tuple) and len(level) >= 2 and level[0] is not None and level[1] is not None

    async def _publish_data_quality_alert(
        self,
        trading_pair: str,
        error_type: str,
        context: dict[str, Any],
    ) -> None:
        """Publishes a data quality alert to the pubsub system."""
        try:
            # Import event class

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
                            "Failed to publish DATA_QUALITY_ALERT",
                            source_module=self._source_module,
                            context={"alert_event": alert_event},
                        )
                        # Fallback: Log locally with enhanced context
                        await self._log_data_quality_issue_locally(alert_event.to_dict(), e)

                # Create a non-blocking task to publish the alert
                alert_task = asyncio.create_task(publish_alert_async())
                self._background_tasks.add(alert_task)
                alert_task.add_done_callback(self._background_tasks.discard)
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
                    source_module=self._source_module,
                )
        except Exception as e:
            self.logger.exception(
                "Failed to persist critical data quality issue to database.",
                source_module=self._source_module,
                context={"issue_data": issue_data, "error": e},
            )

    async def _handle_trade_event(self, event_dict: dict[str, Any]) -> None:
        """Handle incoming raw trade events and store them."""
        # This method will be called by pubsub, so it takes the full event_dict
        payload = event_dict.get("payload")
        if not payload:
            self.logger.warning("Trade event missing payload.", context=event_dict, source_module=self._source_module)
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
                    "Trade event for %s is missing required fields (timestamp, price, volume, or side).",
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

    def _get_period_from_config(self, feature_name: str, field_name: str, default_value: int) -> int:
        """Retrieve the period from config for a specific feature.

        This function relies on accessing specific period/length parameters within a
        feature's configuration dictionary. The addition of 'imputation' and 'scaling'
        keys at the same level does not affect its ability to retrieve these parameters.
        """
        feature_spec = self._feature_configs.get(feature_name)
        if feature_spec and isinstance(feature_spec, InternalFeatureSpec):
            period_value = feature_spec.parameters.get(field_name, default_value)
            return period_value if isinstance(period_value, int) and period_value > 0 else default_value
        # If feature_spec is not found or not the right type
        self.logger.warning(
            "Configuration for feature '%s' not found or invalid when trying to get '%s'. Returning default value %s.",
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
            self.pubsub_manager.subscribe(EventType.MARKET_DATA_OHLCV, self.process_market_data)
            self.pubsub_manager.subscribe(EventType.MARKET_DATA_L2, self.process_market_data)
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
            self.pubsub_manager.unsubscribe(EventType.MARKET_DATA_OHLCV, self.process_market_data)
            self.pubsub_manager.unsubscribe(EventType.MARKET_DATA_L2, self.process_market_data)
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
                    "OHLCV event for %s missing 'timestamp_bar_start', cannot calculate features.",
                    trading_pair,
                    source_module=self._source_module,
                    context={"payload": payload},
                )
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
                context={"original_event": market_data_event_dict},
            )

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
    def _pipeline_compute_macd(data: Any, fast: int, slow: int, signal: int) -> pd.DataFrame:
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
                bbands_df[col] = bbands_df[col].fillna(0.0)  # Default for other bbands stats
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
        close_col: str = "close",
    ) -> pd.Series[Any]:
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

        atr_series = ta.atr(high=ohlc_data[high_col], low=ohlc_data[low_col], close=ohlc_data[close_col], length=length)
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
        volume_col: str = "volume",
    ) -> pd.Series[Any]:
        """Compute VWAP from OHLCV data using rolling window.

        Expects DataFrame with Decimal objects for price/volume, converts to float64 Series[Any] output.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        if not isinstance(ohlcv_df, pd.DataFrame):
            return pd.Series(dtype="float64")
        if not all(col in ohlcv_df.columns for col in [high_col, low_col, close_col, volume_col]):
            return pd.Series(dtype="float64")  # Or log error

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
            current_sum_vol = sum_vol.get(idx)  # Use .get for safety if index alignment isn't perfect
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
                        high_val = high_d.loc[idx]
                        low_val = low_d.loc[idx]
                        close_val = close_d.loc[idx]
                        # All values exist, calculate typical price
                        vwap_series_decimal[idx] = (high_val + low_val + close_val) / Decimal("3.0")
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
            vwap_series_float,
            ohlcv_df,
            length,
        )

    @staticmethod
    def _pipeline_compute_vwap_trades(
        trade_history_deque: deque[dict[str, Any]],  # Trade dicts: price, volume, timestamp
        bar_start_times: Any,  # Series[Any] of datetime objects
        bar_interval_seconds: int,
        ohlcv_close_prices: pd.Series[Any] | None = None,  # For fallback
    ) -> pd.Series[Any]:
        """Compute VWAP from trade data for specified bar start times.

        Returns a float64 Series.
        If no relevant trades or sum_volume is zero, falls back to ohlcv_close_prices.
        If fallback also fails, defaults to 0.0.
        Intended for use in Scikit-learn FunctionTransformer.
        """
        series_name = f"vwap_trades_{bar_interval_seconds}s"

        # Validate inputs
        if not isinstance(bar_start_times, pd.Series):
            return pd.Series(dtype="float64", index=None, name=series_name)

        # Prepare trade data
        trades_df = FeatureEngine._prepare_trades_dataframe(trade_history_deque)

        # Calculate VWAP for each bar
        vwap_results = [
            FeatureEngine._calculate_single_bar_vwap(
                pd.Timestamp(bar_start_dt),
                bar_start_dt_idx,
                bar_interval_seconds,
                trades_df,
                ohlcv_close_prices,
            )
            for bar_start_dt_idx, bar_start_dt in bar_start_times.items()
        ]

        return pd.Series(vwap_results, index=bar_start_times.index, dtype="float64", name=series_name)

    @staticmethod
    def _prepare_trades_dataframe(trade_history_deque: deque[dict[str, Any]]) -> pd.DataFrame | None:
        """Prepare and validate trades dataframe from deque."""
        if not trade_history_deque:
            return None

        try:
            # Validate all elements are dicts
            if not all(isinstance(trade, dict) for trade in trade_history_deque):
                return None

            trades_df = pd.DataFrame(list(trade_history_deque))

            if trades_df.empty:
                return None

            # Convert data types
            trades_df["price"] = trades_df["price"].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
            trades_df["volume"] = trades_df["volume"].apply(lambda x: Decimal(str(x)))  # type: ignore[arg-type,return-value]
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        else:
            return trades_df

    @staticmethod
    def _calculate_single_bar_vwap(
        bar_start_dt: pd.Timestamp,
        bar_start_dt_idx: Any,
        bar_interval_seconds: int,
        trades_df: pd.DataFrame | None,
        ohlcv_close_prices: pd.Series[Any] | None,
    ) -> float:
        """Calculate VWAP for a single time bar."""
        # Try to calculate from trades
        vwap = FeatureEngine._calculate_vwap_from_trades(
            bar_start_dt,
            bar_interval_seconds,
            trades_df,
        )

        # If failed, use fallback
        if pd.isna(vwap):
            vwap = FeatureEngine._get_fallback_price(  # type: ignore[unreachable]
                bar_start_dt,
                bar_start_dt_idx,
                ohlcv_close_prices,
            )

        return vwap

    @staticmethod
    def _calculate_vwap_from_trades(
        bar_start_dt: pd.Timestamp,
        bar_interval_seconds: int,
        trades_df: pd.DataFrame | None,
    ) -> float:
        """Calculate VWAP from trade data for a specific time window."""
        if trades_df is None or trades_df.empty:
            return np.nan

        bar_end_dt = bar_start_dt + pd.Timedelta(seconds=bar_interval_seconds)
        relevant_trades = trades_df[(trades_df["timestamp"] >= bar_start_dt) & (trades_df["timestamp"] < bar_end_dt)]

        if relevant_trades.empty:
            return np.nan

        sum_price_volume = (relevant_trades["price"] * relevant_trades["volume"]).sum()
        sum_volume = relevant_trades["volume"].sum()

        if sum_volume > Decimal(0):
            return float(sum_price_volume / sum_volume)

        return np.nan

    @staticmethod
    def _get_fallback_price(
        bar_start_dt: pd.Timestamp,
        bar_start_dt_idx: Any,
        ohlcv_close_prices: pd.Series[Any] | None,
    ) -> float:
        """Get fallback price from OHLCV close prices."""
        if ohlcv_close_prices is None:
            return 0.0

        # Try direct index
        if bar_start_dt in ohlcv_close_prices.index:
            price = ohlcv_close_prices.get(bar_start_dt)
            if pd.notna(price):
                return float(price)  # type: ignore[unreachable]

        # Try original index
        if bar_start_dt_idx in ohlcv_close_prices.index:
            price = ohlcv_close_prices.get(bar_start_dt_idx)
            if pd.notna(price):
                return float(price)  # type: ignore[unreachable]

        return 0.0

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
            result_series,
            ohlcv_df,
            max_gap_length=min(3, length // 5),
        )

        # Strategy 2: Volume-weighted estimates for medium gaps
        result_series = FeatureEngine._apply_volume_weighted_estimates(
            result_series,
            ohlcv_df,
            length,
        )

        # Strategy 3: Context-aware typical price fallback
        result_series = FeatureEngine._apply_context_aware_fallback(
            result_series,
            ohlcv_df,
        )

        # Strategy 4: Final safety net with market-hours aware filling
        return FeatureEngine._apply_market_aware_final_fill(
            result_series,
            ohlcv_df,
        )

    @staticmethod
    def _apply_intelligent_vwap_interpolation(
        vwap_series: pd.Series[Any],
        ohlcv_df: pd.DataFrame,
        max_gap_length: int = 3,
    ) -> pd.Series[Any]:
        """Apply intelligent interpolation for short VWAP gaps based on volume patterns."""
        result = vwap_series.copy()

        # Early return if no NaN values
        nan_mask = result.isna()
        if not nan_mask.any():
            return result

        # Process each gap group
        nan_groups = FeatureEngine._identify_nan_groups(nan_mask)

        for group_id in nan_groups[nan_mask].unique():
            FeatureEngine._interpolate_gap_group(
                result,
                ohlcv_df,
                nan_groups,
                group_id,
                max_gap_length,
            )

        return result

    @staticmethod
    def _identify_nan_groups(nan_mask: pd.Series) -> pd.Series:
        """Identify contiguous NaN sequences."""
        return (nan_mask != nan_mask.shift()).cumsum()

    @staticmethod
    def _interpolate_gap_group(
        result: pd.Series,
        ohlcv_df: pd.DataFrame,
        nan_groups: pd.Series,
        group_id: int,
        max_gap_length: int,
    ) -> None:
        """Interpolate a single gap group."""
        gap_indices = result.index[nan_groups == group_id]
        gap_length = len(gap_indices)

        # Skip long gaps
        if gap_length > max_gap_length:
            return

        # Find boundary values
        boundaries = FeatureEngine._find_gap_boundaries(
            result,
            gap_indices[0],
            gap_indices[-1],
        )

        if boundaries is None:
            return

        prev_vwap, next_vwap = boundaries

        # Perform interpolation
        FeatureEngine._interpolate_gap_values(
            result,
            ohlcv_df,
            gap_indices,
            prev_vwap,
            next_vwap,
        )

    @staticmethod
    def _find_gap_boundaries(
        result: pd.Series,
        gap_start_idx: Any,
        gap_end_idx: Any,
    ) -> tuple[float, float] | None:
        """Find previous and next valid VWAP values around a gap."""
        prev_valid_idx = None
        next_valid_idx = None

        # Find previous valid value
        for idx in reversed(result.index[result.index < gap_start_idx]):
            if pd.notna(result[idx]):
                prev_valid_idx = idx
                break

        # Find next valid value
        for idx in result.index[result.index > gap_end_idx]:
            if pd.notna(result[idx]):
                next_valid_idx = idx
                break

        if prev_valid_idx is None or next_valid_idx is None:
            return None

        return result[prev_valid_idx], result[next_valid_idx]

    @staticmethod
    def _interpolate_gap_values(
        result: pd.Series,
        ohlcv_df: pd.DataFrame,
        gap_indices: pd.Index,
        prev_vwap: float,
        next_vwap: float,
    ) -> None:
        """Interpolate values for gap indices using volume weighting if available."""
        # Extract volumes for gap indices
        gap_volumes = FeatureEngine._get_gap_volumes(ohlcv_df, gap_indices)

        if gap_volumes and sum(gap_volumes) > 0:
            # Volume-weighted interpolation
            FeatureEngine._volume_weighted_interpolation(
                result,
                gap_indices,
                gap_volumes,
                prev_vwap,
                next_vwap,
            )
        else:
            # Simple linear interpolation
            FeatureEngine._linear_interpolation(
                result,
                gap_indices,
                prev_vwap,
                next_vwap,
            )

    @staticmethod
    def _get_gap_volumes(ohlcv_df: pd.DataFrame, gap_indices: pd.Index) -> list[float]:
        """Extract volumes for gap indices."""
        volumes = []
        for idx in gap_indices:
            if idx in ohlcv_df.index:
                volumes.append(float(ohlcv_df.loc[idx, "volume"]))
            else:
                volumes.append(0.0)
        return volumes

    @staticmethod
    def _volume_weighted_interpolation(
        result: pd.Series,
        gap_indices: pd.Index,
        gap_volumes: list[float],
        prev_vwap: float,
        next_vwap: float,
    ) -> None:
        """Apply volume-weighted interpolation."""
        gap_length = len(gap_indices)
        total_volume = sum(gap_volumes)

        for i, idx in enumerate(gap_indices):
            position_weight = (i + 1) / (gap_length + 1)
            volume_weight = gap_volumes[i] / total_volume

            # Combine weights: 70% position, 30% volume
            combined_weight = 0.7 * position_weight + 0.3 * volume_weight
            result[idx] = prev_vwap * (1 - combined_weight) + next_vwap * combined_weight

    @staticmethod
    def _linear_interpolation(
        result: pd.Series,
        gap_indices: pd.Index,
        prev_vwap: float,
        next_vwap: float,
    ) -> None:
        """Apply simple linear interpolation."""
        gap_length = len(gap_indices)

        for i, idx in enumerate(gap_indices):
            weight = (i + 1) / (gap_length + 1)
            result[idx] = prev_vwap * (1 - weight) + next_vwap * weight

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
                ohlcv_df["high"].astype(float) + ohlcv_df["low"].astype(float) + ohlcv_df["close"].astype(float)
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
            price_range = ohlcv_df["high"].astype(float) - ohlcv_df["low"].astype(float)
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
            final_filled = final_filled.fillna(global_median) if pd.notna(global_median) else final_filled.fillna(0.0)

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
                if (
                    book
                    and isinstance(book.get("bids"), list)
                    and len(book["bids"]) > 0
                    and isinstance(book["bids"][0], list | tuple)
                    and len(book["bids"][0]) == 2
                    and isinstance(book.get("asks"), list)
                    and len(book["asks"]) > 0
                    and isinstance(book["asks"][0], list | tuple)
                    and len(book["asks"][0]) == 2
                ):
                    best_bid_price_str = str(book["bids"][0][0])
                    best_ask_price_str = str(book["asks"][0][0])

                    # Check for non-numeric or empty strings before Decimal conversion
                    if not best_bid_price_str or not best_ask_price_str:
                        raise ValueError("Empty price string encountered.")  # noqa: TRY301

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
            except (TypeError, IndexError, ValueError, AttributeError) as e:
                # Log malformed book data for monitoring
                # These errors will result in the default 0.0 values being used
                if hasattr(FeatureEngine, "_instance") and hasattr(FeatureEngine._instance, "logger"):
                    FeatureEngine._instance.logger.debug(
                        f"Malformed L2 book data in spread calculation: {e}",
                        extra={"error_type": type(e).__name__, "book_sample": str(book)[:200] if book else "None"},
                    )

            abs_spreads.append(current_abs_spread)
            pct_spreads.append(current_pct_spread)

        return pd.DataFrame({"abs_spread": abs_spreads, "pct_spread": pct_spreads}, index=output_index, dtype="float64")

    @staticmethod
    def _validate_order_book_levels(book: dict[str, Any], levels: int) -> tuple[bool, bool]:
        """Validate order book has required levels with proper structure."""
        if not book or not isinstance(book.get("bids"), list) or not isinstance(book.get("asks"), list):
            return False, False

        if len(book["bids"]) < levels or len(book["asks"]) < levels:
            return False, False

        # Check bid levels
        valid_bids = all(
            isinstance(book["bids"][i], list | tuple) and len(book["bids"][i]) == 2 and book["bids"][i][1] is not None
            for i in range(levels)
        )

        # Check ask levels
        valid_asks = all(
            isinstance(book["asks"][i], list | tuple) and len(book["asks"][i]) == 2 and book["asks"][i][1] is not None
            for i in range(levels)
        )

        return valid_bids, valid_asks

    @staticmethod
    def _compute_single_book_imbalance(book: Any, levels: int) -> float:
        """Compute imbalance for a single order book."""
        try:
            valid_bids, valid_asks = FeatureEngine._validate_order_book_levels(book, levels)

            if not (valid_bids and valid_asks):
                return 0.0

            bid_vol_at_levels = sum(Decimal(str(book["bids"][i][1])) for i in range(levels))
            ask_vol_at_levels = sum(Decimal(str(book["asks"][i][1])) for i in range(levels))

            total_vol = bid_vol_at_levels + ask_vol_at_levels
            if total_vol > Decimal(0):
                imbalance_val = (bid_vol_at_levels - ask_vol_at_levels) / total_vol
                return float(imbalance_val)

        except (TypeError, IndexError, ValueError, AttributeError) as e:
            # Log malformed book data for monitoring
            if hasattr(FeatureEngine, "_instance") and hasattr(FeatureEngine._instance, "logger"):
                FeatureEngine._instance.logger.debug(
                    f"Malformed L2 book data in imbalance calculation: {e}",
                    extra={"error_type": type(e).__name__, "book_sample": str(book)[:200] if book else "None"},
                )
            return 0.0
        else:
            return 0.0

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
            imbalance = FeatureEngine._compute_single_book_imbalance(book, levels)
            imbalances.append(imbalance)

        return pd.Series(imbalances, index=output_index, dtype="float64", name=series_name)

    @staticmethod
    def _pipeline_compute_l2_wap(
        l2_books_series: object,  # Accept as object to bypass mypy inference
        ohlcv_close_prices: pd.Series[Any] | None = None,  # For fallback
        levels: int = 1,  # Typically levels=1 for WAP
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
                    book_idx,
                    ohlcv_close_prices,
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
            if not (
                "bids" in value
                and "asks" in value
                and isinstance(value["bids"], list)
                and isinstance(value["asks"], list)
                and len(value["bids"]) >= levels
                and len(value["asks"]) >= levels
            ):
                return np.nan

            # Extract bid/ask data for the specified level
            bid_data = value["bids"][levels - 1]
            ask_data = value["asks"][levels - 1]

            if not (
                isinstance(bid_data, list | tuple)
                and len(bid_data) >= 2
                and isinstance(ask_data, list | tuple)
                and len(ask_data) >= 2
            ):
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
        except (ValueError, TypeError, IndexError) as e:
            # Log the error for monitoring WAP calculation issues
            if hasattr(FeatureEngine, "_instance") and hasattr(FeatureEngine._instance, "logger"):
                FeatureEngine._instance.logger.debug(
                    f"Failed to calculate WAP from book data: {e}",
                    extra={
                        "error_type": type(e).__name__,
                        "bid_data": str(bid_data)[:100] if bid_data else "None",
                        "ask_data": str(ask_data)[:100] if ask_data else "None",
                    },
                )
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
            except (KeyError, IndexError, TypeError) as e:
                # Log the error for monitoring fallback WAP issues
                if hasattr(FeatureEngine, "_instance") and hasattr(FeatureEngine._instance, "logger"):
                    FeatureEngine._instance.logger.debug(
                        f"Failed to get fallback WAP from close prices: {e}",
                        extra={"error_type": type(e).__name__, "book_idx": str(book_idx)},
                    )
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
                if (
                    book
                    and isinstance(book.get("bids"), list)
                    and isinstance(book.get("asks"), list)
                    and len(book["bids"]) >= levels
                    and len(book["asks"]) >= levels
                ):
                    valid_bids = True
                    for i in range(levels):
                        if not (
                            isinstance(book["bids"][i], list | tuple)
                            and len(book["bids"][i]) == 2
                            and book["bids"][i][1] is not None
                        ):
                            valid_bids = False
                            break

                    valid_asks = True
                    for i in range(levels):
                        if not (
                            isinstance(book["asks"][i], list | tuple)
                            and len(book["asks"][i]) == 2
                            and book["asks"][i][1] is not None
                        ):
                            valid_asks = False
                            break

                    if valid_bids and valid_asks:
                        bid_depth_val = sum(Decimal(str(book["bids"][i][1])) for i in range(levels))
                        ask_depth_val = sum(Decimal(str(book["asks"][i][1])) for i in range(levels))
                        current_bid_depth = float(bid_depth_val)
                        current_ask_depth = float(ask_depth_val)
                # else: conditions for invalid book structure lead to default 0.0
            except (TypeError, IndexError, ValueError, AttributeError) as e:
                # Log malformed book data for monitoring depth calculation issues
                if hasattr(FeatureEngine, "_instance") and hasattr(FeatureEngine._instance, "logger"):
                    FeatureEngine._instance.logger.debug(
                        f"Malformed L2 book data in depth calculation: {e}",
                        extra={"error_type": type(e).__name__, "book_sample": str(book)[:200] if book else "None"},
                    )

            bid_depths.append(current_bid_depth)
            ask_depths.append(current_ask_depth)

        return pd.DataFrame(
            {
                col_name_bid: bid_depths,
                col_name_ask: ask_depths,
            },
            index=output_index,
            dtype="float64",
        )

    @staticmethod
    def _pipeline_compute_volume_delta(
        trade_history_deque: deque[dict[str, Any]],  # Deque of trade dicts
        bar_start_times: pd.Series[Any],  # Series[Any] of bar start datetime objects
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
            return pd.Series(  # type: ignore[unreachable]
                dtype="float64",
                name=series_name,
            )

        if not trade_history_deque:  # No trades in entire history
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
                deltas.append(0.0)  # Or np.nan if preferred for "no trades" vs "zero delta"
                continue

            buy_volume = relevant_trades[relevant_trades["side"] == "buy"]["volume"].sum()
            sell_volume = relevant_trades[relevant_trades["side"] == "sell"]["volume"].sum()
            deltas.append(float(buy_volume - sell_volume))

        return pd.Series(
            deltas,
            index=bar_start_times.index,
            dtype="float64",
            name=f"volume_delta_{bar_interval_seconds}s",
        )

    # --- Existing feature calculation methods (some may be deprecated/refactored) ---
    # Note: _calculate_bollinger_bands, _calculate_roc, _calculate_atr, _calculate_stdev removed.
    # Note: _calculate_vwap and _calculate_vwap_from_trades removed.
    # --- Removed _calculate_roc, _calculate_atr, _calculate_stdev ---
    # --- Removed _calculate_bid_ask_spread, _calculate_order_book_imbalance, _calculate_wap, _calculate_depth ---
    # --- Removed _calculate_true_volume_delta_from_trades, _calculate_vwap_from_trades ---

    async def _calculate_and_publish_features(self, trading_pair: str, timestamp_features_for: str) -> None:
        """Calculates all features for a given pair and publishes the results.

        This refactored method delegates complex operations to specialized methods,
        improving readability and maintainability.
        """
        # Check data availability
        if not self._has_sufficient_data(trading_pair):
            return

        bar_start_datetime = pd.to_datetime(timestamp_features_for, utc=True)

        # Prepare all input data
        input_data = self._prepare_pipeline_inputs(trading_pair, bar_start_datetime)
        if not input_data:
            self.logger.warning(
                "Could not prepare input data for %s at %s",
                trading_pair,
                bar_start_datetime,
                source_module=self._source_module,
            )
            return

        # Execute all pipelines and collect features
        all_features = {}
        for pipeline_name, pipeline_info in self.feature_pipelines.items():
            features = await self._execute_single_pipeline(
                pipeline_name,
                pipeline_info,
                input_data,
                bar_start_datetime,
            )
            if features:
                all_features.update(features)

        # Extract advanced features if configured
        advanced_features = await self._extract_advanced_features(
            trading_pair,
            input_data,
            all_features,
        )
        all_features.update(advanced_features)

        # Validate and publish features
        await self._validate_and_publish_features(
            all_features,
            trading_pair,
            timestamp_features_for,
        )

    def _has_sufficient_data(self, trading_pair: str) -> bool:
        """Check if there's sufficient historical data for feature calculation."""
        ohlcv_df = self.ohlcv_history.get(trading_pair)
        min_history = self._get_min_history_required()

        if ohlcv_df is None or len(ohlcv_df) < min_history:
            self.logger.info(
                "Not enough OHLCV data for %s. Need %s, have %s.",
                trading_pair,
                min_history,
                len(ohlcv_df) if ohlcv_df is not None else 0,
                source_module=self._source_module,
            )
            return False

        # Check L2 book availability (warning only)
        l2_book = self.l2_books.get(trading_pair)
        if l2_book and (not l2_book.get("bids") or not l2_book.get("asks")):
            self.logger.debug(
                "L2 book for %s is empty or missing bids/asks. L2 features may be skipped.",
                trading_pair,
                source_module=self._source_module,
            )

        return True

    def _prepare_pipeline_inputs(
        self,
        trading_pair: str,
        bar_start_datetime: pd.Timestamp,
    ) -> dict[str, Any] | None:
        """Prepare all input data types needed by pipelines."""
        ohlcv_df_full = self.ohlcv_history[trading_pair]

        # Filter OHLCV data up to current bar
        ohlcv_df_decimal = ohlcv_df_full[ohlcv_df_full.index <= bar_start_datetime]
        if ohlcv_df_decimal.empty:
            self.logger.warning(
                "No historical OHLCV data for %s up to %s.",
                trading_pair,
                bar_start_datetime,
            )
            return None

        # Convert to float64 for pipeline processing
        ohlcv_df = ohlcv_df_decimal.astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            },
        )

        # Prepare close price for dynamic injection
        close_series = ohlcv_df["close"]
        ohlcv_close_injection = (
            close_series.loc[[bar_start_datetime]]
            if bar_start_datetime in close_series.index
            else pd.Series(dtype="float64", index=[bar_start_datetime])
        )

        # L2 book snapshot
        l2_book_snapshot = self._get_aligned_l2_book(trading_pair, bar_start_datetime)
        l2_book_series = pd.Series([l2_book_snapshot], index=[bar_start_datetime])

        # Trade history
        trades_deque = self.trade_history.get(trading_pair, deque())

        # Bar start times for trade-based features
        bar_start_series = pd.Series([bar_start_datetime], index=[bar_start_datetime])

        return {
            "ohlcv_df": ohlcv_df,
            "close_series": close_series,
            "l2_book_series": l2_book_series,
            "trades_deque": trades_deque,
            "bar_start_times": bar_start_series,
            "ohlcv_close_injection": ohlcv_close_injection,
            "l2_book_snapshot": l2_book_snapshot,
        }

    async def _execute_single_pipeline(
        self,
        pipeline_name: str,
        pipeline_info: dict[str, Any],
        input_data: dict[str, Any],
        bar_start_datetime: pd.Timestamp,
    ) -> dict[str, float]:
        """Execute a single feature pipeline and return calculated features."""
        spec = pipeline_info["spec"]
        pipeline = pipeline_info["pipeline"]

        # Get appropriate input data
        pipeline_input = self._get_pipeline_input(spec, input_data)
        if pipeline_input is None:
            self.logger.warning(
                "No input data for pipeline %s with input_type %s",
                pipeline_name,
                spec.input_type,
            )
            return {}

        try:
            # Handle dynamic parameter injection if needed
            if spec.calculator_type in ["l2_wap", "vwap_trades", "volume_delta"]:
                pipeline = self._inject_dynamic_params(
                    pipeline,
                    spec,
                    input_data,
                    bar_start_datetime,
                )

            # Execute pipeline
            raw_output = pipeline.fit_transform(pipeline_input)

            # Process output
            return self._process_pipeline_output(spec, raw_output, pipeline_name)

        except Exception:
            self.logger.exception(
                "Error executing pipeline %s",
                pipeline_name,
                source_module=self._source_module,
            )
            return {}

    def _get_pipeline_input(self, spec: InternalFeatureSpec, input_data: dict[str, Any]) -> Any:
        """Get the appropriate input data for a pipeline based on its input type."""
        input_map = {
            "close_series": input_data["close_series"],
            "ohlcv_df": input_data["ohlcv_df"],
            "l2_book_series": input_data["l2_book_series"],
            "trades_and_bar_starts": input_data["trades_deque"],
        }
        return input_map.get(spec.input_type)

    def _inject_dynamic_params(
        self,
        pipeline: Any,
        spec: InternalFeatureSpec,
        input_data: dict[str, Any],
        bar_start_datetime: pd.Timestamp,
    ) -> Any:
        """Inject dynamic parameters into pipeline for specific calculator types."""
        pipeline_clone = clone(pipeline)
        calculator_step_name = f"{spec.key}_calculator"

        if calculator_step_name in pipeline_clone.named_steps:
            transformer = pipeline_clone.named_steps[calculator_step_name]
            kw_args = transformer.kw_args.copy()

            # Inject close prices
            kw_args["ohlcv_close_prices"] = input_data["ohlcv_close_injection"]

            # Inject bar start times for trade-based features
            if spec.input_type == "trades_and_bar_starts":
                kw_args["bar_start_times"] = input_data["bar_start_times"]

            transformer.kw_args = kw_args
        else:
            self.logger.error(
                "Calculator step %s not found in pipeline",
                calculator_step_name,
            )

        return pipeline_clone

    def _process_pipeline_output(
        self,
        spec: InternalFeatureSpec,
        raw_output: Any,
        pipeline_name: str,
    ) -> dict[str, float]:
        """Process raw pipeline output into feature dictionary."""
        features = {}
        output_handler = self.output_handlers.get(spec.key)

        try:
            if output_handler and spec.output_specs:
                # Enhanced output processing
                processed = output_handler.process_feature_outputs(raw_output)
                if isinstance(processed, pd.DataFrame) and not processed.empty:
                    latest_row = processed.iloc[-1] if len(processed) > 1 else processed.iloc[0]
                    for col_name, value in latest_row.items():
                        if pd.notna(value):
                            features[str(col_name)] = float(value)
            else:
                # Traditional output processing
                features = self._process_traditional_output(spec, raw_output, pipeline_name)

        except Exception:
            self.logger.exception("Error processing output for %s", spec.key)
            # Fallback processing
            if pd.notna(raw_output) and hasattr(raw_output, "__float__"):
                feature_name = pipeline_name.replace("_pipeline", "")
                features[feature_name] = float(raw_output)

        return features

    def _process_traditional_output(
        self,
        spec: InternalFeatureSpec,
        raw_output: Any,
        pipeline_name: str,
    ) -> dict[str, float]:
        """Process output using traditional method for backward compatibility."""
        base_feature_key = pipeline_name.replace("_pipeline", "")

        # Extract latest value
        latest_value = self._extract_latest_value(spec, raw_output)

        # Convert to feature dictionary
        return self._convert_to_feature_dict(latest_value, base_feature_key)

    def _extract_latest_value(self, spec: InternalFeatureSpec, raw_output: Any) -> Any:
        """Extract the latest value from various output types."""
        if isinstance(raw_output, pd.Series):
            return self._extract_from_series(spec, raw_output)
        if isinstance(raw_output, pd.DataFrame):
            return self._extract_from_dataframe(spec, raw_output)
        if isinstance(raw_output, np.ndarray):
            return self._extract_from_array(raw_output)
        return raw_output

    def _extract_from_series(self, spec: InternalFeatureSpec, series: pd.Series) -> Any:
        """Extract value from pandas Series."""
        if series.empty:
            return None

        # For certain input types with single value, take first
        if spec.input_type in ["l2_book_series", "trades_and_bar_starts"] and len(series) == 1:
            return series.iloc[0]
        # Otherwise take last value
        return series.iloc[-1]

    def _extract_from_dataframe(self, spec: InternalFeatureSpec, df: pd.DataFrame) -> Any:
        """Extract value from pandas DataFrame."""
        if df.empty:
            return None

        # For l2_book_series with single row, take first
        if spec.input_type == "l2_book_series" and len(df) == 1:
            return df.iloc[0]
        # Otherwise take last row
        return df.iloc[-1]

    def _extract_from_array(self, array: np.ndarray[Any, Any]) -> Any:
        """Extract value from numpy array."""
        if array.ndim == 1 and array.size > 0:
            return array[-1]
        if array.ndim == 2 and array.shape[0] > 0:
            return pd.Series(array[-1, :])
        return None

    def _convert_to_feature_dict(self, latest_value: Any, base_feature_key: str) -> dict[str, float]:
        """Convert extracted value to feature dictionary."""
        features = {}

        if isinstance(latest_value, pd.Series):
            for idx_name, value in latest_value.items():
                if pd.notna(value):
                    features[f"{base_feature_key}_{idx_name}"] = float(value)
        elif pd.notna(latest_value):
            features[base_feature_key] = float(latest_value)

        return features

    async def _extract_advanced_features(
        self,
        trading_pair: str,
        input_data: dict[str, Any],
        existing_features: dict[str, float],
    ) -> dict[str, float]:
        """Extract advanced features if configured."""
        advanced_features: dict[str, float] = {}

        try:
            advanced_specs = [
                spec
                for spec in self._feature_configs.values()
                if spec.calculator_type in self.advanced_extractor.advanced_indicators
            ]

            if not advanced_specs:
                return advanced_features

            # Prepare data for advanced extraction
            trade_data = list(input_data["trades_deque"]) if input_data["trades_deque"] else None

            # Extract advanced features
            result = await self.advanced_extractor.extract_advanced_features(
                input_data["ohlcv_df"],
                advanced_specs,
                l2_data=input_data["l2_book_snapshot"],
                trade_data=trade_data,
            )

            if not result.features.empty:
                latest = result.features.iloc[-1] if len(result.features) > 1 else result.features.iloc[0]
                for name, value in latest.items():
                    if pd.notna(value):
                        advanced_features[f"advanced_{name}"] = float(value)

                self.logger.debug(
                    "Added %d advanced features for %s. Quality: %s",
                    len(latest),
                    trading_pair,
                    result.quality_metrics,
                )

        except Exception as e:
            self.logger.warning(
                "Advanced feature extraction failed for %s: %s",
                trading_pair,
                e,
            )

        return advanced_features

    async def _validate_and_publish_features(
        self,
        features: dict[str, float],
        trading_pair: str,
        timestamp_features_for: str,
    ) -> None:
        """Validate features and publish event if successful."""
        if not features:
            self.logger.info(
                "No features calculated for %s at %s",
                trading_pair,
                timestamp_features_for,
            )
            return

        # Apply validation
        try:
            validated_features = await self._apply_enterprise_feature_validation(
                features,
                trading_pair,
                timestamp_features_for,
            )

            if not validated_features:
                return

        except Exception:
            self.logger.exception(
                "Feature validation failed for %s at %s",
                trading_pair,
                timestamp_features_for,
            )
            return

        # Publish event
        await self._publish_feature_event(
            validated_features,
            trading_pair,
            timestamp_features_for,
        )

    async def _publish_feature_event(
        self,
        features: dict[str, Any],
        trading_pair: str,
        timestamp_features_for: str,
    ) -> None:
        """Construct and publish feature calculation event."""
        event_payload = {
            "trading_pair": trading_pair,
            "exchange": self.config.get("exchange_name", "kraken"),
            "timestamp_features_for": timestamp_features_for,
            "features": features,
        }

        full_event = {
            "event_id": str(uuid.uuid4()),
            "event_type": EventType.FEATURES_CALCULATED.name,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "source_module": self._source_module,
            "payload": event_payload,
        }

        try:
            await self.pubsub_manager.publish(cast("Any", full_event))

            self.logger.info(
                "Published FEATURES_CALCULATED for %s at %s",
                trading_pair,
                timestamp_features_for,
                source_module=self._source_module,
                context={
                    "event_id": full_event["event_id"],
                    "num_features": len(features),
                },
            )
        except Exception:
            self.logger.exception(
                "Failed to publish feature event for %s",
                trading_pair,
            )

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
                normalized_features,
                trading_pair,
                validation_context,
            )

            # Stage 3: Quality Metrics and Outlier Detection
            quality_checked_features = self._apply_quality_checks(
                complete_features,
                trading_pair,
                validation_context,
            )

            # Stage 4: Business Rule Validation
            business_validated_features = await self._apply_business_validation(
                quality_checked_features,
                trading_pair,
                validation_context,
            )

            # Stage 5: Schema Compliance with Fallback
            schema_compliant_features = self._ensure_schema_compliance(
                business_validated_features,
                validation_context,
            )

            # schema_compliant_features is now always a dict (possibly empty), never None
            if not schema_compliant_features:
                return None

            # Stage 6: Final Quality Assessment and Reporting
            return self._finalize_and_report_validation(
                schema_compliant_features,
                validation_context,
                validation_start_time,
            )

        except Exception:
            self.logger.exception(
                "Enterprise feature validation failed for %s: %s",
                trading_pair,
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
            normalized_value, had_error = self._normalize_single_feature(
                feature_name,
                value,
                validation_context,
            )
            normalized[feature_name] = normalized_value
            if had_error:
                type_conversion_errors.append(feature_name)

        # Record validation stage results
        self._record_normalization_results(
            validation_context,
            raw_features,
            type_conversion_errors,
        )

        return normalized

    def _normalize_single_feature(
        self,
        feature_name: str,
        value: Any,
        validation_context: dict[str, Any],
    ) -> tuple[float | None, bool]:
        """Normalize a single feature value. Returns (normalized_value, had_error)."""
        try:
            # Handle different input types
            if isinstance(value, int | float | np.integer | np.floating):
                return self._normalize_numeric_value(feature_name, value, validation_context)
            if isinstance(value, Decimal):
                return self._normalize_decimal_value(feature_name, value, validation_context)
            if pd.isna(value) or value is None:
                return None, False
            if isinstance(value, str):
                return self._normalize_string_value(feature_name, value)
            return self._normalize_unknown_type(feature_name, value)

        except Exception as e:
            self.logger.debug(
                "Error normalizing feature %s: %s",
                feature_name,
                e,
                source_module=self._source_module,
            )
            return None, True

    def _normalize_numeric_value(
        self,
        feature_name: str,
        value: float | np.integer[Any] | np.floating[Any],
        validation_context: dict[str, Any],
    ) -> tuple[float | None, bool]:
        """Normalize numeric values and check for special float values."""
        if np.isnan(value):
            return None, False  # Will be handled in completeness stage
        if np.isinf(value):
            self.logger.warning(
                "Infinite value detected for feature %s, converting to None",
                feature_name,
                source_module=self._source_module,
            )
            validation_context["issues_detected"].append(f"infinite_value_{feature_name}")
            return None, False
        return float(value), False

    def _normalize_decimal_value(
        self,
        feature_name: str,
        value: Decimal,
        validation_context: dict[str, Any],
    ) -> tuple[float | None, bool]:
        """Normalize Decimal values to float."""
        try:
            float_val = float(value)
            if np.isfinite(float_val):
                return float_val, False
            validation_context["issues_detected"].append(f"non_finite_decimal_{feature_name}")
        except (ValueError, OverflowError):
            return None, True
        else:
            return None, False

    def _normalize_string_value(
        self,
        feature_name: str,
        value: str,
    ) -> tuple[float | None, bool]:
        """Attempt to convert string to float."""
        try:
            float_val = float(value)
            if np.isfinite(float_val):
                return float_val, False
        except (ValueError, TypeError):
            self.logger.warning(
                "Could not convert string feature %s='%s' to float",
                feature_name,
                value,
                source_module=self._source_module,
            )
            return None, True
        else:
            return None, False

    def _normalize_unknown_type(
        self,
        feature_name: str,
        value: Any,
    ) -> tuple[float | None, bool]:
        """Attempt to normalize unknown types."""
        try:
            float_val = float(value)
            if np.isfinite(float_val):
                return float_val, False
        except (ValueError, TypeError):
            return None, True
        else:
            return None, False

    def _record_normalization_results(
        self,
        validation_context: dict[str, Any],
        raw_features: dict[str, Any],
        type_conversion_errors: list[str],
    ) -> None:
        """Record the results of the normalization stage."""
        validation_context["validation_stages"].append(
            {
                "stage": "type_normalization",
                "success": True,
                "features_processed": len(raw_features),
                "type_errors": len(type_conversion_errors),
                "error_features": type_conversion_errors,
            },
        )

        if type_conversion_errors:
            self.logger.info(
                "Type[Any] conversion errors for %d features: %s",
                len(type_conversion_errors),
                type_conversion_errors,
                source_module=self._source_module,
            )

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
            validation_context["validation_stages"].append(
                {
                    "stage": "completeness_check",
                    "success": True,
                    "missing_count": 0,
                    "imputation_applied": False,
                },
            )
            return complete_features

        # Apply intelligent imputation based on feature type and context
        for feature_name in missing_features:
            try:
                # Determine feature category for appropriate imputation
                imputed_value = await self._impute_missing_feature(
                    feature_name,
                    trading_pair,
                    validation_context,
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

        validation_context["validation_stages"].append(
            {
                "stage": "completeness_ensured",
                "success": True,
                "missing_count": len(missing_features),
                "imputation_applied": True,
                "imputed_features": missing_features,
            },
        )

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
                        feature_name,
                        trading_pair,
                        feature_spec,
                        data_type,
                        validation_context,
                    )

            # Fallback to enhanced contextual analysis
            return await self._execute_enhanced_contextual_imputation(
                feature_name,
                trading_pair,
                validation_context,
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
                feature_name,
                trading_pair,
                feature_spec,
                validation_context,
            )

            if historical_value is not None:
                confidence_score = self._calculate_imputation_confidence(
                    "historical_analysis",
                    historical_value,
                    feature_name,
                )
                if confidence_score > 0.8:  # High confidence threshold
                    validation_context["fallbacks_applied"].append(
                        f"advanced_historical_{feature_name}",
                    )
                    return historical_value

            # Strategy 2: Cross-Asset Correlation Analysis
            correlation_value = await self._impute_from_correlation_analysis(
                feature_name,
                trading_pair,
                data_type,
                validation_context,
            )

            if correlation_value is not None:
                confidence_score = self._calculate_imputation_confidence(
                    "correlation_analysis",
                    correlation_value,
                    feature_name,
                )
                if confidence_score > 0.7:  # Medium-high confidence
                    validation_context["fallbacks_applied"].append(
                        f"advanced_correlation_{feature_name}",
                    )
                    return correlation_value

            # Strategy 3: Market Regime-Aware Imputation
            regime_value = await self._impute_from_market_regime_analysis(
                feature_name,
                trading_pair,
                data_type,
                validation_context,
            )

            if regime_value is not None:
                confidence_score = self._calculate_imputation_confidence(
                    "market_regime",
                    regime_value,
                    feature_name,
                )
                if confidence_score > 0.6:  # Medium confidence
                    validation_context["fallbacks_applied"].append(
                        f"advanced_regime_{feature_name}",
                    )
                    return regime_value

            # Strategy 4: ML-Based Pattern Imputation
            ml_value = await self._impute_from_ml_patterns(
                feature_name,
                trading_pair,
                feature_spec,
                validation_context,
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
                feature_name,
                e,
                source_module=self._source_module,
            )
            return self._get_contextual_default(feature_name, data_type)

        finally:
            # Log imputation performance
            duration = (pd.Timestamp.now() - imputation_start_time).total_seconds() * 1000
            self.logger.debug(
                "Advanced imputation for %s completed in %.1fms",
                feature_name,
                duration,
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
                base_value,
                feature_name,
                market_conditions,
            )

            validation_context["fallbacks_applied"].append(
                f"enhanced_contextual_{feature_name}",
            )

        except Exception as e:
            self.logger.debug(
                "Enhanced contextual imputation failed for %s: %s",
                feature_name,
                e,
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
            # Get historical data
            historical_data = await self._fetch_historical_data(
                feature_name,
                trading_pair,
                feature_spec,
            )
            if historical_data is None or historical_data.empty:
                return None

            # Try enhanced imputation first
            enhanced_result = await self._try_enhanced_imputation(
                feature_name,
                historical_data,
            )
            if enhanced_result is not None:
                return enhanced_result

            # Fallback to simple imputation
            return self._simple_historical_imputation(feature_name, historical_data)

        except Exception as e:
            self.logger.debug(f"Historical analysis imputation failed for {feature_name}: {e}")
            return None

    async def _fetch_historical_data(
        self,
        feature_name: str,
        trading_pair: str,
        feature_spec: InternalFeatureSpec,
    ) -> pd.DataFrame | None:
        """Fetch historical data for imputation."""
        if not self.history_repo:
            return None

        lookback = max(20, feature_spec.parameters.get("period", 14) * 3)
        interval = feature_spec.parameters.get("interval", "1m")

        try:
            if hasattr(self.history_repo, "get_feature_history"):
                return await self.history_repo.get_feature_history(
                    trading_pair=trading_pair,
                    lookback=lookback,
                    interval=interval,
                )
            self.logger.debug(
                "get_feature_history method not available in HistoryRepository",
                source_module=self._source_module,
            )
        except (AttributeError, Exception) as e:
            self.logger.debug(
                f"Error fetching historical data for {feature_name}: {e}",
                source_module=self._source_module,
            )

        return None

    async def _try_enhanced_imputation(
        self,
        feature_name: str,
        historical_data: pd.DataFrame,
    ) -> float | None:
        """Try enhanced imputation using intelligent engine."""
        try:
            # Check if enhanced features are available
            if not HAS_ENHANCEMENTS:
                self._raise_import_error("Enhanced features not available")

            # Create pandas series from historical data
            hist_series = pd.Series(
                historical_data["value"].values,
                index=pd.to_datetime(historical_data["timestamp"]),
            )

            # Use intelligent imputation engine
            imputation_engine = IntelligentImputationEngine(self.logger)
            feature_metadata = {
                feature_name: {
                    "type": self._determine_feature_type(feature_name),
                    "category": "indicator",
                },
            }

            # Impute using regime-aware strategy
            imputed_df = pd.DataFrame({feature_name: hist_series})
            imputed_data, _ = imputation_engine.impute_features(
                imputed_df,
                feature_metadata,
                ImputationStrategy.REGIME_AWARE,
            )

            if feature_name in imputed_data.columns:
                last_value = imputed_data[feature_name].iloc[-1]
                if pd.notna(last_value):
                    self.logger.debug(
                        "Imputed %s using intelligent regime-aware method: %.4f",
                        feature_name,
                        last_value,
                        source_module=self._source_module,
                    )
                    return float(last_value)

        except ImportError:
            # Log that enhanced imputation is unavailable
            self.logger.debug(
                "Enhanced imputation engine not available for feature %s, falling back to basic imputation",
                feature_name,
                source_module=self._source_module,
            )

        return None

    def _determine_feature_type(self, feature_name: str) -> str:
        """Determine the type of feature based on its name."""
        name_lower = feature_name.lower()

        if any(ind in name_lower for ind in ["rsi", "macd", "bb", "sma", "ema"]):
            return "technical"
        if "volume" in name_lower:
            return "volume"
        return "unknown"

    def _simple_historical_imputation(
        self,
        feature_name: str,
        historical_data: pd.DataFrame,
    ) -> float | None:
        """Simple fallback imputation using moving averages."""
        name_lower = feature_name.lower()

        # Technical indicators: use moving average
        if "rsi" in name_lower or "macd" in name_lower:
            ma = historical_data["value"].rolling(window=5, min_periods=1).mean().iloc[-1]
            if pd.notna(ma):
                self.logger.debug(
                    "Imputed %s from historical MA: %.4f",
                    feature_name,
                    ma,
                    source_module=self._source_module,
                )
                return float(ma)

        # Volume: use historical mean
        elif "volume" in name_lower:
            vol_ma = historical_data["value"].mean()
            if pd.notna(vol_ma):
                self.logger.debug(
                    "Imputed %s from historical mean: %.4f",
                    feature_name,
                    vol_ma,
                    source_module=self._source_module,
                )
                return float(vol_ma)

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
                    source_module=self._source_module,
                )
                return imputed_value  # type: ignore[no-any-return]

        except Exception as e:
            self.logger.debug(
                "Correlation analysis imputation failed: %s",
                e,
                source_module=self._source_module,
                context={"feature_name": feature_name, "trading_pair": trading_pair},
            )
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

            # Get regime-specific default value
            return self._get_regime_specific_default(
                feature_name,
                market_conditions,
            )

        except Exception as e:
            self.logger.debug(
                "Market regime analysis failed: %s",
                e,
                source_module=self._source_module,
            )
            return None

    def _get_regime_specific_default(
        self,
        feature_name: str,
        market_conditions: dict[str, Any],
    ) -> float | None:
        """Get regime-specific default value for a feature."""
        regime = market_conditions.get("regime", "unknown")
        volatility_level = market_conditions.get("volatility", "medium")
        name_lower = feature_name.lower()

        if "rsi" in name_lower:
            return self._get_rsi_regime_default(regime, volatility_level)
        if "volume" in name_lower:
            return self._get_volume_regime_default(regime)
        return None

    def _get_rsi_regime_default(self, regime: str, volatility_level: str) -> float:
        """Get RSI default value based on market regime."""
        rsi_defaults = {
            "trending_up": 65.0,  # Bullish trending
            "trending_down": 35.0,  # Bearish trending
            "ranging": 50.0,  # Neutral ranging
        }

        if regime == "volatile":
            return 55.0 if volatility_level == "high" else 45.0

        return rsi_defaults.get(regime, 50.0)  # Default to neutral

    def _get_volume_regime_default(self, regime: str) -> float:
        """Get volume default value based on market regime."""
        if regime == "volatile":
            return 1500.0  # Higher volume in volatile markets
        if regime in ["trending_up", "trending_down"]:
            return 1200.0  # Moderate volume in trends
        return 800.0  # Lower volume in ranging markets

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
                    source_module=self._source_module,
                )
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
                source_module=self._source_module,
            )

        except Exception as e:
            self.logger.debug(
                "ML-based imputation failed for %s: %s",
                feature_name,
                e,
                source_module=self._source_module,
                context={"feature_spec": feature_spec},
            )
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
                e,
                source_module=self._source_module,
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
            adjustment_factor = self._calculate_adjustment_factor(market_conditions)

            # Apply feature-specific adjustments
            return self._apply_feature_specific_adjustment(
                base_value,
                feature_name,
                market_conditions,
                adjustment_factor,
            )

        except Exception as e:
            self.logger.debug(
                "Market context adjustment failed: %s",
                e,
                source_module=self._source_module,
            )
            return base_value

    def _calculate_adjustment_factor(self, market_conditions: dict[str, Any]) -> float:
        """Calculate confidence-weighted adjustment factor."""
        confidence = market_conditions.get("confidence", 0.5)
        return float(confidence) * 0.2  # Max 20% adjustment

    def _apply_feature_specific_adjustment(
        self,
        base_value: float,
        feature_name: str,
        market_conditions: dict[str, Any],
        adjustment_factor: float,
    ) -> float:
        """Apply adjustments based on feature type."""
        name_lower = feature_name.lower()

        if "rsi" in name_lower:
            return self._adjust_rsi_value(
                base_value,
                market_conditions,
                adjustment_factor,
            )
        if "spread" in name_lower:
            return self._adjust_spread_value(
                base_value,
                market_conditions,
                adjustment_factor,
            )
        if "volume" in name_lower:
            return self._adjust_volume_value(
                base_value,
                market_conditions,
                adjustment_factor,
            )
        return base_value

    def _adjust_rsi_value(
        self,
        base_value: float,
        market_conditions: dict[str, Any],
        adjustment_factor: float,
    ) -> float:
        """Adjust RSI value based on market conditions."""
        regime = market_conditions.get("regime", "unknown")
        volatility = market_conditions.get("volatility", "medium")

        if regime == "trending_up":
            return min(100.0, base_value + (adjustment_factor * 20))
        if regime == "trending_down":
            return max(0.0, base_value - (adjustment_factor * 20))
        if volatility == "high":
            adjustment = adjustment_factor * 10
            return base_value + adjustment if base_value > 50 else base_value - adjustment
        return base_value

    def _adjust_spread_value(
        self,
        base_value: float,
        market_conditions: dict[str, Any],
        adjustment_factor: float,
    ) -> float:
        """Adjust spread value based on market conditions."""
        volatility = market_conditions.get("volatility", "medium")

        if volatility == "high":
            return base_value * (1 + adjustment_factor)
        if volatility == "low":
            return base_value * (1 - adjustment_factor * 0.5)
        return base_value

    def _adjust_volume_value(
        self,
        base_value: float,
        market_conditions: dict[str, Any],
        adjustment_factor: float,
    ) -> float:
        """Adjust volume value based on market conditions."""
        regime = market_conditions.get("regime", "unknown")
        volatility = market_conditions.get("volatility", "medium")

        if regime in ["trending_up", "trending_down"]:
            return base_value * (1 + adjustment_factor)
        if volatility == "high":
            return base_value * (1 + adjustment_factor * 1.5)
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

        # Define feature patterns and their default values
        feature_patterns = [
            (lambda f: "rsi" in f, 50.0),  # Neutral RSI
            (lambda f: "macd" in f, 0.0),  # Neutral MACD
            (lambda f: any(term in f for term in ["volume", "vol", "vwap"]), 0.0),  # No volume/VWAP
            (lambda f: any(term in f for term in ["price", "spread", "wap"]), 0.0),  # No spread/price difference
            (lambda f: any(term in f for term in ["atr", "volatility", "stdev"]), 0.0),  # No volatility
            (lambda f: "pct" in f or "percent" in f, 0.0),  # No percentage change
            (lambda f: "imbalance" in f, 0.0),  # Balanced
        ]

        # Check each pattern
        for pattern_check, default_value in feature_patterns:
            if pattern_check(feature_lower):
                return default_value

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

        validation_context["validation_stages"].append(
            {
                "stage": "quality_checks",
                "success": True,
                "outliers_detected": len(outliers_detected),
                "outlier_features": outliers_detected,
            },
        )

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

    async def _apply_business_validation(
        self,
        features: dict[str, float],
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply business logic validation rules."""
        validated = features.copy()
        business_violations: list[str] = []

        # Try enhanced validation first
        try:
            return await self._apply_enhanced_validation(
                features,
                trading_pair,
                validation_context,
            )
        except ImportError:
            # Fallback to simple validation
            return self._apply_simple_validation(features)
        except Exception as e:
            self.logger.debug(
                "Business validation error: %s",
                e,
                source_module=self._source_module,
            )
            return {"features": validated, "violations": business_violations}

    async def _apply_enhanced_validation(
        self,
        features: dict[str, float],
        trading_pair: str,
        validation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply comprehensive validation using ComprehensiveFeatureValidator."""
        validated = features.copy()
        business_violations = []

        # Create validator and get historical data
        validator = ComprehensiveFeatureValidator(self.logger)
        historical_data = await self._get_historical_validation_data(trading_pair)

        # Run validation
        validated_features, validation_report = validator.validate_features(
            features,
            {},
            historical_data,
        )

        # Process validation report
        violations = self._extract_violations_from_report(validation_report)
        business_violations.extend(violations)

        # Log corrections
        self._log_validation_corrections(validation_report)

        # Update validated features
        validated.update(validated_features)

        return {"features": validated, "violations": business_violations}

    async def _get_historical_validation_data(
        self,
        trading_pair: str,
    ) -> pd.DataFrame | None:
        """Get historical data for validation."""
        if not hasattr(self, "history_repo") or not self.history_repo:
            return None

        try:
            hist_features = await self.history_repo.get_feature_history(
                trading_pair=trading_pair,
                lookback=100,
                interval="1h",
            )

            if hist_features is not None and not hist_features.empty:
                return hist_features.pivot(
                    index="timestamp",
                    columns="feature_name",
                    values="value",
                )
        except Exception as e:
            # Log the error for monitoring historical data retrieval issues
            self.logger.debug(
                f"Failed to retrieve historical validation data for {trading_pair}: {e}",
                context={"error_type": type(e).__name__, "trading_pair": trading_pair},
                source_module=self._source_module,
            )

        return None

    def _extract_violations_from_report(
        self,
        validation_report: dict[str, Any],
    ) -> list[str]:
        """Extract business violations from validation report."""
        violations: list[str] = []

        if "statistical_tests" not in validation_report:
            return violations

        tests = validation_report["statistical_tests"]
        consistency = tests.get("consistency", {})

        # Check spread consistency
        if not consistency.get("spread_consistency", {}).get("consistent", True):
            violations.append("inconsistent_spreads")

        # Check RSI bounds
        if not consistency.get("rsi_bounds", {}).get("within_bounds", True):
            violations.append("rsi_out_of_bounds")

        # Check for outliers
        outliers = tests.get("outliers", {})
        for feature_name, outlier_info in outliers.items():
            if outlier_info.get("is_outlier", False):
                violations.append(f"outlier_{feature_name}")

        return violations

    def _log_validation_corrections(self, validation_report: dict[str, Any]) -> None:
        """Log any corrections applied during validation."""
        if "corrections_applied" not in validation_report:
            return

        for feature_name, correction in validation_report["corrections_applied"].items():
            self.logger.info(
                f"Applied correction to {feature_name}: {correction['reason']}",
                source_module=self._source_module,
            )

    def _apply_simple_validation(self, features: dict[str, float]) -> dict[str, Any]:
        """Apply simple business rule validation."""
        validated = features.copy()
        business_violations = []

        # Check spread consistency
        if "abs_spread" in features and "pct_spread" in features:
            abs_spread = features.get("abs_spread", 0)
            pct_spread = features.get("pct_spread", 0)

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

        return {"features": validated, "violations": business_violations}

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

            validation_context["validation_stages"].append(
                {
                    "stage": "schema_compliance",
                    "success": True,
                    "validation_method": "pydantic_strict",
                },
            )

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

                    validation_context["validation_stages"].append(
                        {
                            "stage": "schema_compliance",
                            "success": True,
                            "validation_method": "adaptive",
                            "adaptations_applied": True,
                        },
                    )

                    return schema_compliant
                # If no adaptation was possible, return empty dict
                validation_context["validation_stages"].append(
                    {
                        "stage": "schema_compliance",
                        "success": False,
                        "error": "No adaptation possible",
                        "original_error": str(pydantic_error),
                    },
                )
            except Exception as adaptation_error:
                self.logger.exception(
                    "Schema adaptation also failed",
                    source_module=self._source_module,
                )

                validation_context["validation_stages"].append(
                    {
                        "stage": "schema_compliance",
                        "success": False,
                        "error": str(adaptation_error),
                        "original_error": str(pydantic_error),
                    },
                )

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

        except Exception:
            self.logger.exception("Schema adaptation failed", source_module=self._source_module)
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

        return float(max(0.0, 1.0 - issue_penalty - fallback_penalty))

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
                source_module=self._source_module,
            )
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
                    "max_computation_time_ms",
                    50.0,
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
                        spec.key,
                        e,
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

        except Exception:
            self.logger.exception(
                "Error creating advanced input imputation step for %s: %s. Using simple fallback.",
                spec.key,
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
                    input_config["method"],
                    spec.key,
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
            if not (hasattr(series_data, "index") and len(series_data.index) > 0):
                return context

            # Try to get the trading pair from feature configuration or naming
            trading_pair = self._extract_trading_pair_from_spec(spec)
            if not trading_pair:
                return context

            # Add OHLCV related series
            self._add_ohlcv_context(context, trading_pair, series_data)

            # Add volume data for VWAP-based imputation
            self._add_volume_context(context, trading_pair, series_data)

            # Add L2 book data for market microstructure context
            self._add_l2_book_context(context, trading_pair)

        except Exception as e:
            self.logger.debug(
                "Error preparing imputation context for %s: %s",
                spec.key,
                e,
                source_module=self._source_module,
            )

        return context

    def _add_ohlcv_context(self, context: dict[str, Any], trading_pair: str, series_data: pd.Series[Any]) -> None:
        """Add OHLCV data to imputation context."""
        if trading_pair not in self.ohlcv_history:
            return

        ohlcv_df = self.ohlcv_history[trading_pair]
        if ohlcv_df.empty:
            return

        # Add related price series for correlation-based imputation
        context["related_series"] = {}

        # Add OHLC data if different from target series
        for col in ["open", "high", "low", "close", "volume"]:
            if col in ohlcv_df.columns:
                # Convert to float for imputation (from Decimal)
                series_values = pd.to_numeric(ohlcv_df[col], errors="coerce")
                if len(series_values) == len(series_data):
                    context["related_series"][f"ohlcv_{col}"] = series_values

    def _add_volume_context(self, context: dict[str, Any], trading_pair: str, series_data: pd.Series[Any]) -> None:
        """Add volume data to imputation context."""
        if trading_pair not in self.ohlcv_history:
            return

        volume_data = self.ohlcv_history[trading_pair].get("volume")
        if volume_data is not None and len(volume_data) == len(series_data):
            context["volume"] = pd.to_numeric(volume_data, errors="coerce")

    def _add_l2_book_context(self, context: dict[str, Any], trading_pair: str) -> None:
        """Add L2 book data to imputation context."""
        if trading_pair not in self.l2_books:
            return

        l2_book = self.l2_books[trading_pair]
        if l2_book and "bids" in l2_book and "asks" in l2_book:
            context["l2_book"] = l2_book

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

    def _raise_import_error(self, error_msg: str) -> None:
        """Helper method to raise ImportError, abstracting the raise from try block."""
        raise ImportError(error_msg)

    def _raise_value_error(self, error_msg: str) -> None:
        """Helper method to raise ValueError, abstracting the raise from try block."""
        raise ValueError(error_msg)
