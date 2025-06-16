"""Enhanced feature engine interface supporting multimodal inputs and advanced ML features."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..core.asset_registry import AssetSpecification, AssetType


class FeatureCategory(Enum):
    """Categories of features for multimodal ML models."""
    TECHNICAL = auto()        # Traditional technical indicators
    ORDERBOOK = auto()        # L2 order book features
    MICROSTRUCTURE = auto()   # Market microstructure features
    VOLATILITY = auto()       # Volatility-based features
    SENTIMENT = auto()        # News/social sentiment features
    MACRO = auto()           # Macroeconomic indicators
    CROSS_ASSET = auto()     # Cross-asset correlations
    TEMPORAL = auto()        # Time-based features
    VOLUME_PROFILE = auto()  # Volume profile analysis
    FLOWS = auto()           # Order flow analysis


class FeatureFrequency(Enum):
    """Feature calculation frequencies."""
    TICK = auto()           # Every market event
    SECOND = auto()         # Per second aggregation
    MINUTE = auto()         # Per minute aggregation
    FIVE_MINUTE = auto()    # 5-minute aggregation
    HOURLY = auto()         # Hourly aggregation
    DAILY = auto()          # Daily aggregation


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a feature calculation."""
    name: str
    category: FeatureCategory
    frequency: FeatureFrequency
    lookback_periods: int

    # Feature parameters
    parameters: dict[str, Any] = field(default_factory=dict[str, Any])

    # Asset type applicability
    applicable_asset_types: set[AssetType] = field(default_factory=lambda: set[Any](AssetType))

    # Dependencies
    required_data_types: set[str] = field(default_factory=set)  # e.g., {"ohlcv", "l2", "trades"}
    depends_on_features: set[str] = field(default_factory=set)

    # ML model requirements
    output_shape: tuple[Any, ...] | None = None
    normalization_method: str | None = None  # "z_score", "min_max", "robust"

    # Metadata
    description: str = ""
    version: str = "1.0"


@dataclass(frozen=True)
class FeatureVector:
    """Container for calculated features with metadata."""
    symbol: str
    exchange_id: str
    timestamp: datetime
    asset_type: AssetType

    # Feature data organized by category
    technical_features: dict[str, float] = field(default_factory=dict[str, Any])
    orderbook_features: dict[str, float] = field(default_factory=dict[str, Any])
    microstructure_features: dict[str, float] = field(default_factory=dict[str, Any])
    sentiment_features: dict[str, float] = field(default_factory=dict[str, Any])
    macro_features: dict[str, float] = field(default_factory=dict[str, Any])
    cross_asset_features: dict[str, float] = field(default_factory=dict[str, Any])

    # Sequence[Any] features for LSTM/Transformer models
    sequence_features: np.ndarray[Any, Any] | None = None
    sequence_length: int = 0

    # Feature metadata
    feature_names: list[str] = field(default_factory=list[Any])
    feature_importance: dict[str, float] | None = None

    # Quality indicators
    completeness_score: float = 1.0  # Fraction of expected features present
    latency_ms: float | None = None

    def to_array(self, feature_names: list[str] | None = None) -> np.ndarray[Any, Any]:
        """Convert features to numpy array for ML models.

        Args:
            feature_names: Ordered list[Any] of feature names to include

        Returns:
            Feature array in specified order
        """
        if feature_names is None:
            feature_names = self.feature_names

        all_features = {
            **self.technical_features,
            **self.orderbook_features,
            **self.microstructure_features,
            **self.sentiment_features,
            **self.macro_features,
            **self.cross_asset_features,
        }

        return np.array([all_features.get(name, 0.0) for name in feature_names])

    def get_features_by_category(self, category: FeatureCategory) -> dict[str, float]:
        """Get features by category.

        Args:
            category: Feature category to retrieve
        Returns:
            Dictionary of features for the category
        """
        feature_map = {
            FeatureCategory.TECHNICAL: self.technical_features,
            FeatureCategory.ORDERBOOK: self.orderbook_features,
            FeatureCategory.MICROSTRUCTURE: self.microstructure_features,
            FeatureCategory.SENTIMENT: self.sentiment_features,
            FeatureCategory.MACRO: self.macro_features,
            FeatureCategory.CROSS_ASSET: self.cross_asset_features,
        }
        return feature_map.get(category, {})


class FeatureEngineInterface(ABC):
    """Enhanced interface for feature engineering supporting multimodal inputs."""

    def __init__(
        self,
        asset_specifications: list[AssetSpecification],
        **kwargs: dict[str, Any]) -> None:
        """Initialize with supported assets and configuration."""
        self.asset_specifications = {spec.symbol: spec for spec in asset_specifications}
        self.feature_specs: dict[str, FeatureSpec] = {}
        self.feature_cache: dict[str, Any] = {}

    # Feature specification management
    @abstractmethod
    def register_feature(self, feature_spec: FeatureSpec) -> None:
        """Register a feature specification.

        Args:
            feature_spec: Feature specification to register
        """

    @abstractmethod
    def get_feature_specs(self, category: FeatureCategory | None = None,
                         asset_type: AssetType | None = None) -> list[FeatureSpec]:
        """Get registered feature specifications.

        Args:
            category: Optional filter by feature category
            asset_type: Optional filter by applicable asset type
        Returns:
            List of matching feature specifications
        """

    # Core feature calculation methods
    @abstractmethod
    async def calculate_technical_features(self, symbol: str,
                                         ohlcv_data: pd.DataFrame) -> dict[str, float]:
        """Calculate technical analysis features.

        Args:
            symbol: Trading symbol
            ohlcv_data: OHLCV price data
        Returns:
            Dictionary of technical features
        """

    @abstractmethod
    async def calculate_orderbook_features(self, symbol: str,
                                         orderbook_data: dict[str, Any]) -> dict[str, float]:
        """Calculate order book features.

        Args:
            symbol: Trading symbol
            orderbook_data: L2 order book data
        Returns:
            Dictionary of order book features
        """

    @abstractmethod
    async def calculate_microstructure_features(self, symbol: str,
                                              trade_data: pd.DataFrame) -> dict[str, float]:
        """Calculate market microstructure features.

        Args:
            symbol: Trading symbol
            trade_data: Individual trade data
        Returns:
            Dictionary of microstructure features
        """

    @abstractmethod
    async def calculate_sentiment_features(
        self,
        symbol: str,
        news_data: list[dict[str, Any]] | None = None,
        social_data: list[dict[str, Any]] | None = None) -> dict[str, float]:
        """Calculate sentiment features from news and social data.

        Args:
            symbol: Trading symbol
            news_data: News articles data
            social_data: Social media data
        Returns:
            Dictionary of sentiment features
        """

    @abstractmethod
    async def calculate_cross_asset_features(self, primary_symbol: str,
                                           correlated_symbols: list[str]) -> dict[str, float]:
        """Calculate cross-asset correlation and spillover features.

        Args:
            primary_symbol: Primary trading symbol
            correlated_symbols: Related symbols for correlation analysis
        Returns:
            Dictionary of cross-asset features
        """

    # Advanced feature methods for MARL
    @abstractmethod
    async def calculate_marl_state_features(self, symbol: str) -> dict[str, float]:
        """Calculate features specifically for MARL agent state representation.

        Args:
            symbol: Trading symbol
        Returns:
            Dictionary of MARL state features
        """

    @abstractmethod
    async def get_sequence_features(self, symbol: str, sequence_length: int,
                                  feature_names: list[str]) -> np.ndarray[Any, Any]:
        """Get time sequence of features for LSTM/Transformer models.

        Args:
            symbol: Trading symbol
            sequence_length: Number of time steps
            feature_names: Features to include in sequence
        Returns:
            Feature sequence array of shape (sequence_length, num_features)
        """

    # Composite feature generation
    @abstractmethod
    async def generate_feature_vector(
        self,
        symbol: str,
        timestamp: datetime | None = None,
        categories: set[FeatureCategory] | None = None) -> FeatureVector:
        """Generate complete feature vector for a symbol.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp for feature calculation (None for latest)
            categories: Feature categories to include (None for all)

        Returns:
            Complete feature vector
        """

    # Data management and optimization
    @abstractmethod
    async def update_market_data(
        self,
        symbol: str,
        data_type: str,
        data: dict[str, Any]) -> None:
        """Update internal market data for feature calculation.

        Args:
            symbol: Trading symbol
            data_type: Type[Any] of data ("ohlcv", "l2", "trades", "news", etc.)
            data: Market data update
        """

    @abstractmethod
    def get_feature_importance(self, model_type: str) -> dict[str, float]:
        """Get feature importance scores for a model type.

        Args:
            model_type: Type[Any] of model ("xgboost", "lstm", "marl", etc.)

        Returns:
            Dictionary mapping feature names to importance scores
        """

    @abstractmethod
    async def validate_feature_quality(self, feature_vector: FeatureVector) -> dict[str, Any]:
        """Validate feature quality and completeness.

        Args:
            feature_vector: Feature vector to validate
        Returns:
            Dictionary with quality metrics and issues
        """

    # Asset-specific feature support
    def supports_asset_type(self, asset_type: AssetType) -> bool:
        """Check if feature engine supports a specific asset type.

        Args:
            asset_type: Asset type to check
        Returns:
            True if supported, False otherwise
        """
        return any(spec.asset_type == asset_type for spec in self.asset_specifications.values())

    def get_supported_features_for_asset(self, symbol: str) -> list[FeatureSpec]:
        """Get features supported for a specific asset.

        Args:
            symbol: Asset symbol
        Returns:
            List of applicable feature specifications
        """
        asset_spec = self.asset_specifications.get(symbol)
        if not asset_spec:
            return []

        return [
            spec for spec in self.feature_specs.values()
            if asset_spec.asset_type in spec.applicable_asset_types or
               not spec.applicable_asset_types  # Empty set means all types
        ]

    # Performance and monitoring
    @abstractmethod
    async def get_calculation_metrics(self) -> dict[str, Any]:
        """Get feature calculation performance metrics.

        Returns:
            Dictionary with calculation latency, cache hit rates, etc.
        """

    @abstractmethod
    def clear_feature_cache(self, symbol: str | None = None) -> None:
        """Clear feature calculation cache.

        Args:
            symbol: Optional symbol to clear (None clears all)
        """


# Protocol for feature engine factory
class FeatureEngineFactory(Protocol):
    """Protocol for creating feature engines for different asset types."""

    def create_engine(
        self,
        asset_types: set[AssetType],
        **kwargs: dict[str, Any]) -> FeatureEngineInterface:
        """Create a feature engine for specified asset types.

        Args:
            asset_types: Set of asset types to support
            **kwargs: Additional configuration parameters
        Returns:
            Feature engine instance
        """
        ...


# Predefined feature specifications for common features
def get_default_crypto_features() -> list[FeatureSpec]:
    """Get default feature specifications for cryptocurrency trading."""
    return [
        # Technical indicators
        FeatureSpec(
            name="rsi_14",
            category=FeatureCategory.TECHNICAL,
            frequency=FeatureFrequency.MINUTE,
            lookback_periods=14,
            parameters={"period": 14},
            applicable_asset_types={AssetType.CRYPTO},
            required_data_types={"ohlcv"},
            description="14-period Relative Strength Index"),
        FeatureSpec(
            name="macd",
            category=FeatureCategory.TECHNICAL,
            frequency=FeatureFrequency.MINUTE,
            lookback_periods=26,
            parameters={"fast": 12, "slow": 26, "signal": 9},
            applicable_asset_types={AssetType.CRYPTO},
            required_data_types={"ohlcv"},
            description="MACD oscillator"),
        # Order book features
        FeatureSpec(
            name="bid_ask_spread_bps",
            category=FeatureCategory.ORDERBOOK,
            frequency=FeatureFrequency.SECOND,
            lookback_periods=1,
            applicable_asset_types={AssetType.CRYPTO},
            required_data_types={"l2"},
            description="Bid-ask spread in basis points"),
        FeatureSpec(
            name="order_book_imbalance",
            category=FeatureCategory.ORDERBOOK,
            frequency=FeatureFrequency.SECOND,
            lookback_periods=1,
            parameters={"depth_levels": 10},
            applicable_asset_types={AssetType.CRYPTO},
            required_data_types={"l2"},
            description="Order book imbalance ratio"),
        # Microstructure features
        FeatureSpec(
            name="effective_spread",
            category=FeatureCategory.MICROSTRUCTURE,
            frequency=FeatureFrequency.SECOND,
            lookback_periods=10,
            applicable_asset_types={AssetType.CRYPTO},
            required_data_types={"trades", "l2"},
            description="Effective spread based on trade prices"),
        # Volatility features
        FeatureSpec(
            name="realized_volatility_5m",
            category=FeatureCategory.VOLATILITY,
            frequency=FeatureFrequency.FIVE_MINUTE,
            lookback_periods=12,  # 1 hour
            applicable_asset_types={AssetType.CRYPTO},
            required_data_types={"ohlcv"},
            description="5-minute realized volatility"),
    ]