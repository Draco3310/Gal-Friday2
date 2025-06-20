"""Simulate market price data for backtesting trading strategies.

This module provides a service that simulates market price data for backtesting trading strategies.
It uses historical OHLCV data to provide price information,
bid-ask spreads, and simulated order book.
It also supports volatility-adjusted spread calculation and market depth simulation.

"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
import contextlib
import dataclasses
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from enum import Enum
import heapq
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import asyncio
import numpy as np
import pandas as pd

# Import the base class
from .market_price_service import MarketPriceService

# Provider implementations
from .providers import APIDataProvider, DatabaseDataProvider, LocalFileDataProvider

# Import enhanced components
try:
    from .market_price_service_enhancements import (
        AdvancedPriceGenerator,
        EnterpriseConfigurationManager,
        MarketEvent,
        MarketEventType,
        MarketParameters,
        MarketRegime,
        PriceModelType,
        PricePoint,
        RealisticHistoricalDataGenerator,
        create_enterprise_market_service,
    )
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    log_temp = logging.getLogger(__name__)
    log_temp.info(
        "Enhanced market price components not available. "
        "Using basic implementation.",
    )

# Attempt to import the actual ConfigManager
if TYPE_CHECKING:
    from .config_manager import ConfigManager
else:
    try:
        from .config_manager import ConfigManager
    except ImportError:
        # For backward compatibility, define a minimal interface
        class ConfigManager:
            """Minimal placeholder for ConfigManager."""

            def get(self, _key: str, default: object | None = None) -> object | None:
                """Get a value from config."""
                return default

            def get_decimal(self, _key: str, default: Decimal) -> Decimal:
                """Get a decimal value from config."""
                return default

            def get_int(self, _key: str, default: int) -> int:
                """Get an integer value from config."""
                return default

# Attempt to import pandas_ta for ATR calculation
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    log_temp = logging.getLogger(__name__)
    log_temp.warning(
        "pandas_ta library not found. ATR calculation for "
        "volatility-adjusted spread will be disabled.")

_SOURCE_MODULE = "SimulatedMarketPriceService"


# === ENTERPRISE-GRADE HISTORICAL DATA LOADING & CACHING SYSTEM ===

class DataSource(str, Enum):
    """Supported historical data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    BINANCE = "binance"
    KRAKEN = "kraken"
    LOCAL_FILES = "local_files"
    DATABASE = "database"


@dataclass
class DataRequest:
    """Historical data request specification."""
    symbol: str
    start_date: datetime
    end_date: datetime
    frequency: str
    data_source: DataSource | None = None
    include_volume: bool = True
    validate_data: bool = True
    cache_result: bool = True


@dataclass
class HistoricalDataPoint:
    """Single historical data point."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


class HistoricalDataProvider(ABC):
    """Abstract base class for historical data providers."""

    @abstractmethod
    async def fetch_data(self, request: DataRequest) -> list[HistoricalDataPoint]:
        """Fetch historical data for the given request."""

    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by this provider."""


class CacheLayer(ABC):
    """Abstract base class for cache layers."""

    @abstractmethod
    async def get(self, key: str) -> list[HistoricalDataPoint] | None:
        """Get data from cache."""

    @abstractmethod
    async def set(self, key: str, data: list[HistoricalDataPoint], ttl: int = 3600) -> None:
        """Store data in cache with TTL."""


class MemoryCache(CacheLayer):
    """In-memory cache implementation with LRU eviction."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the instance."""
        self.max_size = max_size
        self.cache: dict[str, tuple[list[HistoricalDataPoint], float]] = {}
        self.access_order: dict[str, float] = {}

    async def get(self, key: str) -> list[HistoricalDataPoint] | None:
        if key in self.cache:
            data, timestamp = self.cache[key]
            self.access_order[key] = time.time()
            return data
        return None

    async def set(self, key: str, data: list[HistoricalDataPoint], ttl: int = 3600) -> None:
        # Evict LRU items if cache is full
        while len(self.cache) >= self.max_size:
            lru_key = min(self.access_order.keys(), key=lambda k: self.access_order[k])
            del self.cache[lru_key]
            del self.access_order[lru_key]

        self.cache[key] = (data, time.time() + ttl)
        self.access_order[key] = time.time()


class DiskCache(CacheLayer):
    """Disk-based cache implementation."""

    def __init__(self, cache_path: str = "./cache") -> None:
        """Initialize the instance."""
        self.cache_path = cache_path
        # In a real implementation, this would use file-based storage
        self._disk_cache: dict[str, list[HistoricalDataPoint]] = {}

    async def get(self, key: str) -> list[HistoricalDataPoint] | None:
        return self._disk_cache.get(key)

    async def set(self, key: str, data: list[HistoricalDataPoint], ttl: int = 3600) -> None:
        self._disk_cache[key] = data


class DataLoadingError(Exception):
    """Exception raised for data loading errors."""


class HistoricalDataLoader:
    """Enterprise-grade historical data loading and caching system."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the instance."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize data providers
        self.providers: dict[DataSource, HistoricalDataProvider] = {}
        self._initialize_providers()

        # Initialize cache layers
        self.memory_cache = MemoryCache(config.get("memory_cache_size", 1000))
        self.disk_cache = DiskCache(config.get("disk_cache_path", "./cache"))

        # Performance tracking
        self.cache_stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "provider_requests": 0,
            "total_requests": 0,
        }

    def _initialize_providers(self) -> None:
        """Initialize available data providers based on configuration."""
        provider_names: list[str] = self.config.get("providers", [])

        for name in provider_names:
            try:
                if name == "local_files":
                    self.providers[DataSource.LOCAL_FILES] = LocalFileDataProvider(
                        self.config, self.logger,
                    )
                elif name == "database":
                    self.providers[DataSource.DATABASE] = DatabaseDataProvider(
                        self.config, self.logger,
                    )
                elif name == "api":
                    self.providers[DataSource.YAHOO_FINANCE] = APIDataProvider(
                        self.config, self.logger,
                    )
                else:
                    self.logger.warning("Unknown provider name: %s", name)
                    continue

                self.logger.info("Initialized data provider: %s", name)
            except Exception as exc:  # pragma: no cover - initialization best effort
                self.logger.exception("Failed to initialize provider %s:", exc)

    async def load_historical_data(self, request: DataRequest) -> list[HistoricalDataPoint]:
        """Load historical data with intelligent caching
        Enterprise-grade historical data loading implementation.
        """
        try:
            self.logger.info(
                f"Loading historical data for {request.symbol}: "
                f"{request.start_date} to {request.end_date}",
            )

            self.cache_stats["total_requests"] += 1

            # Generate cache key
            cache_key = self._generate_cache_key(request)

            # Try memory cache first
            data = await self.memory_cache.get(cache_key)
            if data:
                self.cache_stats["memory_hits"] += 1
                self.logger.debug("Data found in memory cache")
                return data

            # Try disk cache
            data = await self.disk_cache.get(cache_key)
            if data:
                self.cache_stats["disk_hits"] += 1
                self.logger.debug("Data found in disk cache")

                # Store in memory cache for faster future access
                if request.cache_result:
                    await self.memory_cache.set(cache_key, data)

                return data

            # Load from data provider (would normally fetch from external source)
            data = await self._load_from_provider(request)

            # Validate data quality if requested
            if request.validate_data:
                quality_score = await self._validate_data_quality(data, request)
                self.logger.info(f"Data quality score: {quality_score:.2f}")

                if quality_score < 0.8:
                    self.logger.warning(f"Low data quality detected for {request.symbol}")

            # Cache the results
            if request.cache_result and data:
                await self.memory_cache.set(cache_key, data)
                await self.disk_cache.set(cache_key, data)

            self.logger.info(f"Loaded {len(data)} data points for {request.symbol}")

        except Exception as e:
            self.logger.exception(f"Error loading historical data for {request.symbol}: ")
            raise DataLoadingError(f"Failed to load historical data: {e}")
        else:
            return data

    async def _load_from_provider(self, request: DataRequest) -> list[HistoricalDataPoint]:
        """Load data from appropriate provider."""
        provider: HistoricalDataProvider | None = None

        if request.data_source is not None:
            provider = self.providers.get(request.data_source)
            if provider is None:
                raise DataLoadingError(
                    f"No provider configured for data source: {request.data_source}",
                )
        else:
            for candidate in self.providers.values():
                try:
                    if await candidate.validate_symbol(request.symbol):
                        provider = candidate
                        break
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.exception("Provider validation failed: ", exc)

        if provider is None:
            raise DataLoadingError(
                f"No provider available for symbol: {request.symbol}",
            )

        self.cache_stats["provider_requests"] += 1

        try:
            return await provider.fetch_data(request)
        except Exception as exc:  # pragma: no cover - wrap and rethrow
            raise DataLoadingError(str(exc)) from exc
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate unique cache key for request."""
        key_parts = [
            request.symbol,
            request.start_date.isoformat(),
            request.end_date.isoformat(),
            request.frequency,
            str(request.data_source.value if request.data_source else "auto"),
            str(request.include_volume),
        ]
        return "_".join(key_parts)

    async def _validate_data_quality(self, data: list[HistoricalDataPoint],
                                   request: DataRequest) -> float:
        """Validate data quality and completeness."""
        if not data:
            return 0.0

        # Simple quality calculation - in real implementation would be more sophisticated
        return 1.0 if len(data) > 0 else 0.0

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["total_requests"]
        cache_hit_rate = 0.0

        if total_requests > 0:
            total_hits = self.cache_stats["memory_hits"] + self.cache_stats["disk_hits"]
            cache_hit_rate = total_hits / total_requests

        return {
            **self.cache_stats,
            "cache_hit_rate": cache_hit_rate,
            "memory_hit_rate": self.cache_stats["memory_hits"] / total_requests if total_requests > 0 else 0,
            "disk_hit_rate": self.cache_stats["disk_hits"] / total_requests if total_requests > 0 else 0,
        }


# === PRICE INTERPOLATION & MISSING DATA HANDLING SYSTEM ===

class InterpolationMethod(str, Enum):
    """Available interpolation methods."""
    LINEAR = "linear"
    SPLINE = "spline"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


@dataclass
class DataGap:
    """Information about a data gap."""
    start_time: datetime
    end_time: datetime
    gap_type: str
    duration_minutes: int
    before_price: float | None = None
    after_price: float | None = None


@dataclass
class InterpolationResult:
    """Result of price interpolation."""
    interpolated_data: list[dict[str, Any]]
    quality_score: float
    method_used: InterpolationMethod
    gaps_filled: list[DataGap]
    warnings: list[str]


class InterpolationError(Exception):
    """Exception raised for interpolation errors."""


class PriceInterpolator:
    """Enterprise-grade price interpolation and missing data handling."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the instance."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._source_module = self.__class__.__name__
        self._use_enhanced_features = False
        self._enterprise_config_manager = None

        # Interpolation methods registry
        self.interpolation_methods = {
            InterpolationMethod.LINEAR: self._linear_interpolation,
            InterpolationMethod.SPLINE: self._spline_interpolation,
            InterpolationMethod.VOLATILITY_ADJUSTED: self._volatility_adjusted_interpolation,
            InterpolationMethod.FORWARD_FILL: self._forward_fill,
            InterpolationMethod.BACKWARD_FILL: self._backward_fill,
        }

    async def interpolate_missing_data(self, data: list[dict[str, Any]], symbol: str,
                                     frequency: str) -> InterpolationResult:
        """Interpolate missing data points using intelligent algorithms
        Enterprise-grade price interpolation implementation.
        """
        try:
            self.logger.info(f"Starting interpolation for {symbol} with {len(data)} data points")

            if not data:
                return InterpolationResult(
                    interpolated_data=[],
                    quality_score=0.0,
                    method_used=InterpolationMethod.LINEAR,
                    gaps_filled=[],
                    warnings=["No data provided for interpolation"],
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
                    warnings=[],
                )

            self.logger.info(f"Detected {len(gaps)} gaps to interpolate")

            # Choose interpolation method
            method = self._select_interpolation_method(gaps, symbol)

            # Perform interpolation
            interpolated_df = await self._perform_interpolation(df, gaps, method)

            # Convert back to list[Any] of dictionaries
            interpolated_data = cast("list[dict[str, Any]]", interpolated_df.to_dict("records"))

            # Calculate quality score
            quality_score = self._calculate_interpolation_quality(df, interpolated_df, gaps)

            result = InterpolationResult(
                interpolated_data=interpolated_data,
                quality_score=quality_score,
                method_used=method,
                gaps_filled=gaps,
                warnings=[],
            )

            self.logger.info(
                f"Interpolation complete: {len(interpolated_data)} points, "
                f"quality={quality_score:.2f}, filled {len(gaps)} gaps",
            )

        except Exception as e:
            self.logger.exception(f"Error during interpolation for {symbol}: ")
            raise InterpolationError(f"Interpolation failed: {e}")
        else:
            return result

    async def _detect_gaps(self, df: pd.DataFrame, symbol: str, frequency: str) -> list[DataGap]:
        """Detect gaps in price data."""
        gaps: list[DataGap] = []

        if len(df) < 2:
            return gaps

        # Calculate expected time interval
        interval_minutes = self._frequency_to_minutes(frequency)
        expected_interval = timedelta(minutes=interval_minutes)

        # Check for gaps between consecutive data points
        for i in range(1, len(df)):
            prev_time = pd.to_datetime(df.iloc[i-1]["timestamp"])
            curr_time = pd.to_datetime(df.iloc[i]["timestamp"])

            time_diff = curr_time - prev_time

            # If gap is larger than expected interval
            if time_diff > expected_interval * 1.5:  # Allow 50% tolerance
                gap = DataGap(
                    start_time=prev_time,
                    end_time=curr_time,
                    gap_type="data_gap",
                    duration_minutes=int(time_diff.total_seconds() / 60),
                    before_price=df.iloc[i-1]["close"],
                    after_price=df.iloc[i]["open"],
                )
                gaps.append(gap)

        return gaps

    def _frequency_to_minutes(self, frequency: str) -> int:
        """Convert frequency string to minutes with comprehensive support.

        Supports various frequency formats:
        - Standard: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w
        - Extended: 2h, 3h, 6h, 8h, 12h, 2d, 3d
        - Custom: Any number followed by m/h/d/w (e.g., 45m, 90m, 36h)
        """
        import re

        # Extended frequency map for common intervals
        standard_freq_map = {
            # Minutes
            "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
            "20m": 20, "30m": 30, "45m": 45,
            # Hours
            "1h": 60, "2h": 120, "3h": 180, "4h": 240, "6h": 360,
            "8h": 480, "12h": 720,
            # Days
            "1d": 1440, "2d": 2880, "3d": 4320, "5d": 7200, "7d": 10080,
            # Weeks
            "1w": 10080, "2w": 20160, "4w": 40320,
            # Special
            "tick": 0,  # Real-time tick data
            "monthly": 43200,  # Approximate 30 days
            "quarterly": 129600,  # Approximate 90 days
        }

        # Check standard map first
        if frequency.lower() in standard_freq_map:
            return standard_freq_map[frequency.lower()]

        # Parse custom format (e.g., "90m", "36h", "14d")
        pattern = r"^(\d+)([mhdw])$"
        match = re.match(pattern, frequency.lower())

        if match:
            value = int(match.group(1))
            unit = match.group(2)

            unit_multipliers = {
                "m": 1,        # minutes
                "h": 60,       # hours to minutes
                "d": 1440,     # days to minutes
                "w": 10080,     # weeks to minutes
            }

            if unit in unit_multipliers:
                return value * unit_multipliers[unit]

        # If using enhanced features, get from config
        if self._use_enhanced_features and self._enterprise_config_manager:
            # This code is actually reachable
            try:  # type: ignore[unreachable]
                custom_frequencies = self.config.get(
                    "market_simulation.custom_frequencies", {},
                )
                if frequency in custom_frequencies:
                    return cast("int", custom_frequencies[frequency])
            except Exception:
                pass

        # Log warning for unrecognized format
        if hasattr(self, "logger"):
            self.logger.warning(
                f"Unrecognized frequency format: {frequency}. Defaulting to 60 minutes.",
                extra={"source_module": self._source_module},
            )

        return 60  # Default to 1 hour

    def _select_interpolation_method(self, gaps: list[DataGap], symbol: str) -> InterpolationMethod:
        """Select best interpolation method based on gap characteristics."""
        if not gaps:
            return InterpolationMethod.LINEAR

        # For small gaps, use linear interpolation
        avg_gap_duration = sum(gap.duration_minutes for gap in gaps) / len(gaps)

        if avg_gap_duration < 60:  # Less than 1 hour
            return InterpolationMethod.LINEAR
        if avg_gap_duration < 240:  # Less than 4 hours
            return InterpolationMethod.VOLATILITY_ADJUSTED
        return InterpolationMethod.FORWARD_FILL

    async def _perform_interpolation(self, df: pd.DataFrame, gaps: list[DataGap],
                                   method: InterpolationMethod) -> pd.DataFrame:
        """Perform actual interpolation to fill gaps."""
        interpolated_df = df.copy()

        for gap in gaps:
            try:
                # Find the indices around the gap
                before_idx = df[pd.to_datetime(df["timestamp"]) <= gap.start_time].index[-1]
                after_idx = df[pd.to_datetime(df["timestamp"]) >= gap.end_time].index[0]

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
        return interpolated_df.sort_values("timestamp").reset_index(drop=True)

    def _generate_gap_timestamps(self, start_time: datetime, end_time: datetime) -> list[datetime]:
        """Generate timestamps for gap filling."""
        timestamps = []
        current = start_time + timedelta(minutes=1)  # Start 1 minute after gap start

        while current < end_time:
            timestamps.append(current)
            current += timedelta(minutes=1)

        return timestamps

    async def _linear_interpolation(self, df: pd.DataFrame, before_idx: int, after_idx: int,
                                  timestamps: list[datetime], gap: DataGap) -> list[dict[str, Any]]:
        """Linear interpolation between two points."""
        before_point = df.iloc[before_idx]
        after_point = df.iloc[after_idx]

        interpolated_points = []

        for _i, timestamp in enumerate(timestamps):
            # Calculate interpolation factor
            total_duration = (gap.end_time - gap.start_time).total_seconds()
            current_duration = (timestamp - gap.start_time).total_seconds()
            factor = current_duration / total_duration if total_duration > 0 else 0

            # Linear interpolation for price
            interpolated_price = before_point["close"] + factor * (after_point["open"] - before_point["close"])

            point = {
                "timestamp": timestamp,
                "open": interpolated_price,
                "high": interpolated_price,
                "low": interpolated_price,
                "close": interpolated_price,
                "volume": self._interpolate_volume(before_point, after_point, factor),
                "interpolated": True,
            }

            interpolated_points.append(point)

        return interpolated_points

    def _interpolate_volume(self, before_point: pd.Series[Any], after_point: pd.Series[Any], factor: float) -> float:
        """Interpolate volume with some randomness."""
        base_volume = before_point["volume"] + factor * (after_point["volume"] - before_point["volume"])

        # Add some randomness (±20%)
        random_factor = np.random.uniform(0.8, 1.2)
        return float(max(0, base_volume * random_factor))

    async def _spline_interpolation(
        self,
        df: pd.DataFrame,
        before_idx: int,
        after_idx: int,
        timestamps: list[datetime],
        gap: DataGap) -> list[dict[str, Any]]:
        """Perform cubic spline interpolation with fallback to linear."""
        try:
            from scipy.interpolate import CubicSpline
        except Exception:
            self.logger.warning(
                "SciPy not available, falling back to linear interpolation")
            return await self._linear_interpolation(df, before_idx, after_idx, timestamps, gap)

        window_start = max(0, before_idx - 2)
        window_end = min(len(df) - 1, after_idx + 2)
        window = df.iloc[window_start : window_end + 1].copy()
        window["timestamp"] = pd.to_datetime(window["timestamp"])
        window = window.set_index("timestamp")

        if len(window) < 4:
            self.logger.warning(
                "Not enough data for spline interpolation, falling back to linear")
            return await self._linear_interpolation(df, before_idx, after_idx, timestamps, gap)

        numeric_index = window.index.astype("int64") // 10**9
        numeric_timestamps = (pd.Series(timestamps).astype("int64") // 10**9).values

        numeric_cols = [
            col
            for col in ["open", "high", "low", "close", "volume"]
            if col in window.columns and pd.api.types.is_numeric_dtype(window[col])
        ]

        splines = {
            col: CubicSpline(numeric_index, window[col].astype(float).values)
            for col in numeric_cols
        }

        before_point = df.iloc[before_idx]
        after_point = df.iloc[after_idx]

        interpolated_points: list[dict[str, Any]] = []
        total_duration = (gap.end_time - gap.start_time).total_seconds()

        for idx, ts in enumerate(timestamps):
            num_ts = numeric_timestamps[idx]
            factor = (ts - gap.start_time).total_seconds() / total_duration if total_duration > 0 else 0
            point: dict[str, Any] = {"timestamp": ts}

            for col in numeric_cols:
                value = float(splines[col](num_ts))
                if col == "volume":
                    linear_base = before_point["volume"] + factor * (
                        after_point["volume"] - before_point["volume"]
                    )
                    random_volume = self._interpolate_volume(before_point, after_point, factor)
                    random_factor = random_volume / linear_base if linear_base else 1.0
                    value = max(0.0, value * random_factor)
                point[col] = value

            point["interpolated"] = True
            interpolated_points.append(point)

        return interpolated_points

    async def _volatility_adjusted_interpolation(self, df: pd.DataFrame, before_idx: int, after_idx: int,
                                               timestamps: list[datetime], gap: DataGap) -> list[dict[str, Any]]:
        """Volatility-adjusted interpolation."""
        # Start with spline interpolation as baseline
        base_points = await self._spline_interpolation(df, before_idx, after_idx, timestamps, gap)

        try:
            import pandas_ta as ta
        except Exception:  # pragma: no cover - dependency missing
            self.logger.warning(
                "pandas_ta not available, skipping volatility adjustment",
            )
            return base_points

        if not {"high", "low", "close"}.issubset(df.columns):
            return base_points

        mask = (
            (pd.to_datetime(df["timestamp"]) >= gap.start_time - timedelta(minutes=10))
            & (pd.to_datetime(df["timestamp"]) <= gap.end_time + timedelta(minutes=10))
        )
        window = df.loc[mask]
        if len(window) < 2:
            return base_points

        length = min(14, len(window))
        atr_series = ta.atr(
            high=window["high"],
            low=window["low"],
            close=window["close"],
            length=length)
        if atr_series.isna().all():
            return base_points

        atr = atr_series.dropna().iloc[-1]

        try:
            close_before = (
                df[pd.to_datetime(df["timestamp"]) < gap.start_time].iloc[-1]["close"]
            )
            close_after = (
                df[pd.to_datetime(df["timestamp"]) > gap.end_time].iloc[0]["close"]
            )
            direction = np.sign(close_after - close_before)
        except Exception:
            direction = 1.0

        adjustments = np.sin(np.linspace(0, np.pi, len(base_points))) * atr * direction * 0.1

        for i, adj in enumerate(adjustments):
            for col in ["open", "high", "low", "close"]:
                if col in base_points[i]:
                    base_points[i][col] += adj

        return base_points

    async def _forward_fill(self, df: pd.DataFrame, before_idx: int, after_idx: int,
                          timestamps: list[datetime], gap: DataGap) -> list[dict[str, Any]]:
        """Forward fill interpolation."""
        before_point = df.iloc[before_idx]
        interpolated_points = []

        for timestamp in timestamps:
            point = {
                "timestamp": timestamp,
                "open": before_point["close"],
                "high": before_point["close"],
                "low": before_point["close"],
                "close": before_point["close"],
                "volume": before_point["volume"] * 0.1,  # Reduced volume
                "interpolated": True,
            }
            interpolated_points.append(point)

        return interpolated_points

    async def _backward_fill(self, df: pd.DataFrame, before_idx: int, after_idx: int,
                           timestamps: list[datetime], gap: DataGap) -> list[dict[str, Any]]:
        """Backward fill interpolation."""
        after_point = df.iloc[after_idx]
        interpolated_points = []

        for timestamp in timestamps:
            point = {
                "timestamp": timestamp,
                "open": after_point["open"],
                "high": after_point["open"],
                "low": after_point["open"],
                "close": after_point["open"],
                "volume": after_point["volume"] * 0.1,  # Reduced volume
                "interpolated": True,
            }
            interpolated_points.append(point)

        return interpolated_points

    def _calculate_interpolation_quality(self, original_df: pd.DataFrame,
                                       interpolated_df: pd.DataFrame,
                                       gaps: list[DataGap]) -> float:
        """Calculate quality score for interpolation."""
        if not gaps:
            return 1.0

        # Simple quality metric based on price continuity
        interpolated_count = len(interpolated_df) - len(original_df)
        gap_coverage = interpolated_count / sum(gap.duration_minutes for gap in gaps) if gaps else 1.0

        return min(1.0, gap_coverage)


# === REAL-TIME SIMULATION ENGINE ===

class SimulationSpeed(str, Enum):
    """Simulation replay speeds."""
    REAL_TIME = "1x"
    FAST_2X = "2x"
    FAST_5X = "5x"
    FAST_10X = "10x"
    FAST_100X = "100x"
    MAX_SPEED = "max"


class SimulationState(str, Enum):
    """Simulation engine states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class SimulationEvent:
    """Simulation event with timing information."""
    timestamp: datetime
    event_type: str
    data: dict[str, Any]
    priority: int = 0

    def __lt__(self, other: "SimulationEvent") -> bool:
        """Make events comparable for priority queue."""
        return self.timestamp < other.timestamp


class SimulationError(Exception):
    """Exception raised for simulation errors."""


class RealTimeSimulationEngine:
    """Enterprise-grade real-time price simulation engine."""

    def __init__(self, config: dict[str, Any], data_loader: "HistoricalDataLoader") -> None:
        """Initialize the instance."""
        self.config = config
        self._data_loader = data_loader
        self.logger = logging.getLogger(__name__)

        # Simulation state
        self.state = SimulationState.STOPPED
        self.current_sim_time: datetime | None = None
        self.start_real_time: float | None = None

        # Event management
        self.event_queue: list[SimulationEvent] = []  # Priority queue for events
        self.event_buffer: deque[Any] = deque(maxlen=config.get("buffer_size", 10000))

        # Event handlers
        self.event_handlers: dict[str, list[Callable[..., Any]]] = {}

        # Speed multiplier calculation
        self.speed_multiplier = self._calculate_speed_multiplier()

        # Simulation task
        self.simulation_task: asyncio.Task[Any] | None = None

    async def start_simulation(self) -> None:
        """Start real-time simulation engine
        Enterprise-grade real-time simulation implementation.
        """
        if self.state != SimulationState.STOPPED:
            raise SimulationError(f"Cannot start simulation in state: {self.state}")

        try:
            self.logger.info("Starting real-time simulation engine")

            # Load initial data
            await self._load_simulation_data()

            # Initialize timing
            self.start_real_time = time.time()
            self.current_sim_time = self.config.get("start_time")

            # Start simulation loop
            self.simulation_task = asyncio.create_task(self._simulation_loop())

            self.state = SimulationState.RUNNING
            self.logger.info("Simulation started successfully")

        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.exception("Failed to start simulation: ")
            raise SimulationError(f"Simulation start failed: {e}")

    async def stop_simulation(self) -> None:
        """Stop simulation gracefully."""
        if self.state not in [SimulationState.RUNNING, SimulationState.PAUSED]:
            self.logger.warning(f"Cannot stop simulation in state: {self.state}")
            return

        self.logger.info("Stopping simulation")

        if self.simulation_task:
            self.simulation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.simulation_task

        self.state = SimulationState.STOPPED
        self.logger.info("Simulation stopped")

    async def set_speed(self, speed: SimulationSpeed) -> None:
        """Change simulation speed during runtime."""
        self.config["speed"] = speed
        self.speed_multiplier = self._calculate_speed_multiplier()
        self.logger.info(f"Simulation speed changed to {speed.value} ({self.speed_multiplier}x)")

    async def _simulation_loop(self) -> None:
        """Main simulation loop."""
        try:
            end_time = self.config.get("end_time")

            while (self.state in [SimulationState.RUNNING, SimulationState.PAUSED] and
                   self.current_sim_time is not None and
                   end_time is not None and
                   self.current_sim_time < end_time):

                # Handle pause state
                while self.state == SimulationState.PAUSED:
                    await asyncio.sleep(0.1)

                # Process events for current time
                await self._process_current_events()

                # Advance simulation time
                await self._advance_simulation_time()

                # Yield control to allow other tasks
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            self.logger.info("Simulation loop cancelled")
            raise
        except Exception:
            self.state = SimulationState.ERROR
            self.logger.exception("Error in simulation loop: ")
            raise

    async def _process_current_events(self) -> None:
        """Process all events scheduled for current simulation time."""
        events_processed = 0

        # Process events from queue
        while (self.event_queue and
               self.current_sim_time is not None and
               self.event_queue[0].timestamp <= self.current_sim_time):

            event = heapq.heappop(self.event_queue)

            try:
                await self._dispatch_event(event)
                events_processed += 1

            except Exception:
                self.logger.exception(f"Error processing event {event.event_type}: ")

        if events_processed > 0:
            self.logger.debug(f"Processed {events_processed} events at {self.current_sim_time}")

    async def _dispatch_event(self, event: SimulationEvent) -> None:
        """Dispatch event to registered handlers."""
        event_type = event.event_type

        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)

                except Exception:
                    self.logger.exception(f"Error in event handler for {event_type}: ")

    async def _advance_simulation_time(self) -> None:
        """Advance simulation time with proper speed control."""
        if self.config.get("speed") == SimulationSpeed.MAX_SPEED:
            # Run as fast as possible
            if self.current_sim_time is not None:
                self.current_sim_time += timedelta(seconds=1)
            return

        # Calculate time advancement
        if self.start_real_time is None or self.current_sim_time is None:
            return

        real_time_elapsed = time.time() - self.start_real_time
        start_time = self.config.get("start_time")
        if start_time is None:
            return
        expected_sim_time = start_time + timedelta(
            seconds=real_time_elapsed * self.speed_multiplier,
        )

        # If we're ahead of schedule, wait
        if self.current_sim_time >= expected_sim_time:
            # Calculate sleep time to maintain proper speed
            ahead_by = (self.current_sim_time - expected_sim_time).total_seconds()
            sleep_time = ahead_by / self.speed_multiplier

            if sleep_time > 0:
                await asyncio.sleep(min(sleep_time, 0.1))  # Cap sleep time

        # Advance simulation time
        time_step = timedelta(seconds=self._get_time_step_seconds())
        self.current_sim_time += time_step

    def _get_time_step_seconds(self) -> float:
        """Get time step in seconds based on configuration."""
        return cast("float", self.config.get("time_step_seconds", 1.0))

    async def _load_simulation_data(self) -> None:
        """Load simulation data and create events."""
        self.logger.info("Loading simulation data")

        symbols = self.config.get("symbols", [])

        for symbol in symbols:
            # Load historical data for symbol
            data_points = await self._load_historical_data(symbol)

            # Create events for each data point
            for data_point in data_points:
                event = SimulationEvent(
                    timestamp=data_point["timestamp"],
                    event_type="price_update",
                    data={
                        "symbol": symbol,
                        "price_data": data_point,
                    },
                )

                heapq.heappush(self.event_queue, event)

        self.logger.info(f"Loaded {len(self.event_queue)} simulation events")

    async def _load_historical_data(self, symbol: str) -> list[dict[str, Any]]:
        """Load historical data for a symbol."""
        try:
            request = DataRequest(
                symbol=symbol,
                start_date=self.config["start_time"],
                end_date=self.config["end_time"],
                frequency=self.config.get("frequency", "1h"))

            points = await self._data_loader.load_historical_data(request)
            return [dataclasses.asdict(p) for p in points]

        except Exception:
            self.logger.exception(f"Failed to load historical data for {symbol}: ")
            return []
    def _calculate_speed_multiplier(self) -> float:
        """Calculate speed multiplier based on configuration."""
        speed_map = {
            SimulationSpeed.REAL_TIME: 1.0,
            SimulationSpeed.FAST_2X: 2.0,
            SimulationSpeed.FAST_5X: 5.0,
            SimulationSpeed.FAST_10X: 10.0,
            SimulationSpeed.FAST_100X: 100.0,
            SimulationSpeed.MAX_SPEED: float("inf"),
        }

        return speed_map.get(self.config.get("speed", SimulationSpeed.REAL_TIME), 1.0)

    def register_event_handler(self, event_type: str, handler: Callable[..., Any]) -> None:
        """Register event handler for specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        self.event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for event type: {event_type}")


@dataclass
class BookLevelConstructionContext:
    """Context for constructing order book levels."""

    best_bid_price_bbo: Decimal
    best_ask_price_bbo: Decimal
    price_format_str: str
    volume_format_str: str
    trading_pair: str


class SimulatedMarketPriceService(MarketPriceService):  # Inherit from MarketPriceService
    """Provide access to the latest market prices.

    based on historical data during a backtest simulation.

    Aligns with the MarketPriceService ABC.
    """

    def __init__(
        self,
        historical_data: dict[str, pd.DataFrame],
        config_manager: ConfigManager | None = None,
        logger: logging.Logger | None = None) -> None:
        """Initialize the service with historical market data.

        Args:
        ----
            historical_data: A dictionary where keys are trading pairs (e.g., "XRP/USD")
                             and values are pandas DataFrames containing OHLCV data
                             indexed by timestamp (UTC).
            config_manager: An instance of ConfigManager for accessing configuration.
            logger: An instance of logging.Logger. If None, a default logger is used.
        """
        self.historical_data = historical_data
        self._current_timestamp: datetime | None = None

        # Properly handle logger and config initialization
        # Check if MarketPriceService ABC has set these attributes
        # Initialize logger
        if logger is not None:
            self.logger = logger
        else:
            # Create default logger
            self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize config
        if config_manager is not None:
            # Use provided config
            self.config = config_manager
        else:
            # Create config manager for defaults
            self.config = ConfigManager()

        self._source_module = _SOURCE_MODULE

        self._load_simulation_config()

        # === ENHANCED COMPONENTS INITIALIZATION ===
        if False:  # Disabled due to logger type mismatch
            # This code is unreachable due to False condition
            try:  # type: ignore[unreachable]
                # Initialize enterprise components
                (
                    self._enterprise_config_manager,
                    self._advanced_price_generator,
                    self._historical_data_generator,
                ) = asyncio.run(create_enterprise_market_service(
                    self.config,
                    self.logger,
                ))

                self.logger.info(
                    "Enterprise market price components initialized successfully",
                    extra={"source_module": self._source_module},
                )
                self._use_enhanced_features = True
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize enterprise components: {e}. "
                    "Falling back to basic implementation.",
                    extra={"source_module": self._source_module},
                )
                self._use_enhanced_features = False
                # These are Optional types
                self._enterprise_config_manager = None
                self._advanced_price_generator = None
                self._historical_data_generator = None
        else:
            self._use_enhanced_features = False  # type: ignore[has-type]
            self._enterprise_config_manager = None  # type: ignore
            self._advanced_price_generator = None  # type: ignore
            self._historical_data_generator = None  # type: ignore

        # === ENTERPRISE-GRADE ENHANCEMENTS ===

        # Initialize Historical Data Loader
        data_loader_config = {
            "memory_cache_size": 1000,
            "disk_cache_path": "./cache",
            "providers": ["local_files"],  # Start with local file support
        }
        self._data_loader = HistoricalDataLoader(data_loader_config)

        # Initialize Price Interpolator
        interpolation_config = {
            "default_method": InterpolationMethod.LINEAR,
            "quality_threshold": 0.8,
            "max_gap_duration_hours": 24,
        }
        self._price_interpolator = PriceInterpolator(interpolation_config)

        # Initialize Real-time Simulation Engine
        simulation_config = {
            "speed": SimulationSpeed.REAL_TIME,
            "buffer_size": 10000,
            "time_step_seconds": 1.0,
            "symbols": list[Any](historical_data.keys()) if historical_data else [],
            "start_time": datetime.now(UTC),
            "end_time": datetime.now(UTC) + timedelta(days=1),
        }
        self._simulation_engine = RealTimeSimulationEngine(
            simulation_config,
            self._data_loader)
        # Register price update handler for simulation engine
        self._simulation_engine.register_event_handler("price_update", self._handle_price_update_event)

        # === END ENTERPRISE-GRADE ENHANCEMENTS ===

        # Validate data format, including HLC columns if volatility is enabled
        self._validate_historical_data(historical_data)

        self.logger.info(
            "SimulatedMarketPriceService initialized with enterprise-grade enhancements.",
            extra={"source_module": self._source_module})

    def _validate_historical_data(self, historical_data: dict[str, pd.DataFrame]) -> None:
        """Validate the format and content of historical data."""
        for pair, df in historical_data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                self.logger.warning(
                    "Historical data for %s does not have a DatetimeIndex. "
                    "This may cause issues with time-based lookups.",
                    pair,
                    extra={"source_module": self._source_module})
                # Attempt to convert index to DatetimeIndex if possible
                try:
                    df.index = pd.to_datetime(df.index, utc=True)
                    historical_data[pair] = df
                    self.logger.info(
                        "Successfully converted index to DatetimeIndex for %s",
                        pair,
                        extra={"source_module": self._source_module})
                except Exception as e:
                    self.logger.exception(
                        "Failed to convert index to DatetimeIndex for %s: %s",
                        pair,
                        str(e),
                        extra={"source_module": self._source_module})

            required_cols = {self._price_column}
            if self._volatility_enabled and ta is not None:  # Check ta availability too
                required_cols.update({self._atr_high_col, self._atr_low_col, self._atr_close_col})

            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                self.logger.warning(
                    "Historical data for %s is missing required columns: %s. "
                    "Available columns: %s",
                    pair,
                    missing_cols,
                    list[Any](df.columns),
                    extra={"source_module": self._source_module})
                # Disable features that require missing columns
                if self._price_column in missing_cols:
                    self.logger.exception(
                        "Critical: Price column '%s' missing for %s. "
                        "This pair may not function properly.",
                        self._price_column,
                        pair,
                        extra={"source_module": self._source_module})
                if any(col in missing_cols for col in [self._atr_high_col, self._atr_low_col, self._atr_close_col]):
                    self.logger.warning(
                        "Disabling volatility features for %s due to missing HLC columns",
                        pair,
                        extra={"source_module": self._source_module})

    def _apply_config_values_from_manager(self) -> None:
        """Apply configuration values from the ConfigManager."""
        if self.config is None:
            # Use default values if config is not available
            self._apply_default_config_values()  # type: ignore[unreachable]
            return

        sim_config = self.config.get("simulation", {})
        self._price_column = sim_config.get("price_column", "close")

        spread_config = sim_config.get("spread", {})
        self._default_spread_pct = self.config.get_decimal(
            "simulation.spread.default_pct",
            Decimal("0.1"))
        self._pair_specific_spread_config = spread_config.get("pairs", {})
        self._volatility_multiplier = self.config.get_decimal(
            "simulation.spread.volatility_multiplier",
            Decimal("1.5"))

        vol_config = spread_config.get("volatility", {})
        self._volatility_enabled = vol_config.get(
            "enabled",
            ta is not None)  # Default true only if ta available
        self._volatility_lookback_period = vol_config.get("lookback_period", 14)
        self._min_volatility_data_points = vol_config.get(
            "min_data_points",
            self._volatility_lookback_period + 5)
        self._atr_high_col = vol_config.get("atr_high_col", "high")
        self._atr_low_col = vol_config.get("atr_low_col", "low")
        self._atr_close_col = vol_config.get("atr_close_col", "close")
        self._max_volatility_adjustment_factor = self.config.get_decimal(
            "simulation.spread.volatility.max_adjustment_factor",
            Decimal("2.0"))

        depth_config = sim_config.get("depth", {})
        self._depth_simulation_enabled = depth_config.get("enabled", True)
        self._depth_num_levels = depth_config.get("num_levels", 5)
        self._depth_price_step_pct = self.config.get_decimal(
            "simulation.depth.price_step_pct",
            Decimal("0.001"))
        self._depth_base_volume = self.config.get_decimal(
            "simulation.depth.base_volume",
            Decimal("10.0"))
        self._depth_volume_decay_factor = self.config.get_decimal(
            "simulation.depth.volume_decay_factor",
            Decimal("0.8"))
        self._depth_price_precision = self.config.get_int("simulation.depth.price_precision", 8)
        self._depth_volume_precision = self.config.get_int("simulation.depth.volume_precision", 4)

        conv_config = sim_config.get("conversion", {})
        self._intermediary_conversion_currency = conv_config.get("intermediary_currency", "USD")

        self.logger.info(
            "Loaded simulation config via ConfigManager: price_column='%s', default_spread_pct=%s",
            self._price_column,
            self._default_spread_pct,
            extra={"source_module": self._source_module})
        self.logger.info(
            "Volatility params: enabled=%s, lookback=%s, multiplier=%s, max_factor=%s",
            self._volatility_enabled,
            self._volatility_lookback_period,
            self._volatility_multiplier,
            self._max_volatility_adjustment_factor,
            extra={"source_module": self._source_module})
        self.logger.info(
            "Loaded depth params via ConfigManager: enabled=%s, levels=%s, price_step_pct=%s",
            self._depth_simulation_enabled,
            self._depth_num_levels,
            self._depth_price_step_pct,
            extra={"source_module": self._source_module})
        self.logger.info(
            "Intermediary currency for conversion: '%s'",
            self._intermediary_conversion_currency,
            extra={"source_module": self._source_module})

    def _apply_default_config_values(self) -> None:
        """Apply default simulation configuration values."""
        self.logger.warning(
            "ConfigManager not provided. Using default simulation "
            "parameters for SimulatedMarketPriceService.",
            extra={"source_module": self._source_module})
        self._price_column = "close"
        self._default_spread_pct = Decimal("0.1")
        self._pair_specific_spread_config = {}
        self._volatility_multiplier = Decimal("1.5")

        self._volatility_enabled = ta is not None
        self._volatility_lookback_period = 14
        self._min_volatility_data_points = self._volatility_lookback_period + 5
        self._atr_high_col = "high"
        self._atr_low_col = "low"
        self._atr_close_col = "close"
        self._max_volatility_adjustment_factor = Decimal("2.0")

        self._depth_simulation_enabled = True
        self._depth_num_levels = 5
        self._depth_price_step_pct = Decimal("0.001")
        self._depth_base_volume = Decimal("10.0")
        self._depth_volume_decay_factor = Decimal("0.8")
        self._depth_price_precision = 8
        self._depth_volume_precision = 4
        self._intermediary_conversion_currency = "USD"

        self.logger.info(
            "Using default simulation config: price_column='%s', default_spread_pct=%s",
            self._price_column,
            self._default_spread_pct,
            extra={"source_module": self._source_module})
        self.logger.info(
            "Using default volatility params: enabled=%s, "
            "lookback=%s, "
            "multiplier=%s, "
            "max_factor=%s",
            self._volatility_enabled,
            self._volatility_lookback_period,
            self._volatility_multiplier,
            self._max_volatility_adjustment_factor,
            extra={"source_module": self._source_module})
        self.logger.info(
            "Using default depth params: enabled=%s, levels=%s, price_step_pct=%s",
            self._depth_simulation_enabled,
            self._depth_num_levels,
            self._depth_price_step_pct,
            extra={"source_module": self._source_module})
        self.logger.info(
            "Using default intermediary currency for conversion: '%s'",
            self._intermediary_conversion_currency,
            extra={"source_module": self._source_module})

    def _load_simulation_config(self) -> None:
        """Load simulation-specific configurations.

        Uses defaults if config_manager is not available.
        """
        if self.config:  # ConfigManager is provided
            self._apply_config_values_from_manager()
        else:  # ConfigManager is NOT provided, use defaults
            self._apply_default_config_values()

    def _get_atr_dataframe_slice(
        self,
        trading_pair: str,
        pair_data_full: pd.DataFrame) -> pd.DataFrame | None:
        """Get the relevant slice of data for ATR calculation."""
        if not isinstance(self._current_timestamp, datetime):
            self.logger.error(
                "Internal error: _current_timestamp is not a datetime object for %s "
                "at %s before slicing for ATR.",
                trading_pair,
                self._current_timestamp,
                extra={"source_module": self._source_module})
            return None  # Should be unreachable

        try:
            timestamp_for_slice = pd.Timestamp(self._current_timestamp)
        except Exception:  # Handle potential errors during Timestamp conversion
            self.logger.exception(
                "Error converting _current_timestamp to pd.Timestamp for %s at %s",
                trading_pair,
                self._current_timestamp,
                extra={"source_module": self._source_module})
            return None

        df_slice = pair_data_full.loc[:timestamp_for_slice]

        if len(df_slice) < self._min_volatility_data_points:
            self.logger.debug(
                "Not enough data points (%s < %s) for %s at %s to calculate ATR.",
                len(df_slice),
                self._min_volatility_data_points,
                trading_pair,
                self._current_timestamp,
                extra={"source_module": self._source_module})
            return None
        return df_slice

    def _calculate_atr_from_slice(
        self,
        df_slice: pd.DataFrame,
        trading_pair: str) -> Decimal | None:
        """Calculate ATR from a given data slice."""
        required_atr_cols = {self._atr_high_col, self._atr_low_col, self._atr_close_col}
        missing_cols = required_atr_cols - set(df_slice.columns)
        if missing_cols:
            self.logger.warning(
                "Missing columns %s required for ATR calculation for %s.",
                missing_cols,
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        try:
            high_series = pd.to_numeric(df_slice[self._atr_high_col], errors="coerce")
            low_series = pd.to_numeric(df_slice[self._atr_low_col], errors="coerce")
            close_series = pd.to_numeric(df_slice[self._atr_close_col], errors="coerce")

            if (
                high_series.isnull().any()
                or low_series.isnull().any()
                or close_series.isnull().any()
            ):
                self.logger.warning(
                    "NaN values found in HLC columns after coercion for %s, cannot calculate ATR.",
                    trading_pair,
                    extra={"source_module": self._source_module})
                return None

            atr_series = ta.atr(
                high=high_series,
                low=low_series,
                close=close_series,
                length=self._volatility_lookback_period)

            if (
                atr_series is None
                or atr_series.empty
                or atr_series.iloc[-1] is None
                or pd.isna(atr_series.iloc[-1])
            ):
                self.logger.debug(
                    "ATR calculation returned None or NaN for %s at %s.",
                    trading_pair,
                    self._current_timestamp,
                    extra={"source_module": self._source_module})
                return None
            return Decimal(str(atr_series.iloc[-1]))
        except Exception:
            self.logger.exception(
                "Error during ATR calculation for %s",
                trading_pair,
                extra={"source_module": self._source_module})
            return None

    def _get_raw_atr_for_pair(self, trading_pair: str) -> Decimal | None:
        """Calculate the raw ATR for a given trading pair at the current time."""
        atr_to_return: Decimal | None = None

        if not self._volatility_enabled or ta is None:
            self.logger.debug(
                "Volatility adjustment or pandas_ta is disabled, cannot calculate raw ATR.",
                extra={"source_module": self._source_module})
        elif self._current_timestamp is None:
            self.logger.warning(
                "Cannot calculate raw ATR for %s: current_timestamp is not set.",
                trading_pair,
                extra={"source_module": self._source_module})
        else:
            pair_data_full = self.historical_data.get(trading_pair)
            if pair_data_full is None or pair_data_full.empty:
                self.logger.debug(
                    "No historical data for %s to calculate raw ATR.",
                    trading_pair,
                    extra={"source_module": self._source_module})
            elif not isinstance(pair_data_full.index, pd.DatetimeIndex):
                self.logger.warning(
                    "Cannot calculate raw ATR for %s at %s: DataFrame index is type %s, "
                    "not DatetimeIndex.",
                    trading_pair,
                    self._current_timestamp,
                    type(pair_data_full.index).__name__,
                    extra={"source_module": self._source_module})
            else:
                if not pair_data_full.index.is_monotonic_increasing:
                    self.logger.debug(
                        "Data for %s is not sorted by index. Sorting now for ATR calculation.",
                        trading_pair,
                        extra={"source_module": self._source_module})
                    pair_data_full = pair_data_full.sort_index()
                    self.historical_data[trading_pair] = pair_data_full

                df_slice = self._get_atr_dataframe_slice(trading_pair, pair_data_full)
                if df_slice is not None:
                    raw_atr_calculated = self._calculate_atr_from_slice(df_slice, trading_pair)
                    if raw_atr_calculated is not None and raw_atr_calculated > Decimal(0):
                        atr_to_return = raw_atr_calculated
                    else:
                        # Logging for zero/negative/None ATR is done in _calculate_atr_from_slice
                        # or the conditions leading to it (e.g. not enough data).
                        # This debug log might be redundant or could be made more specific.
                        self.logger.debug(
                            "Calculated raw ATR is zero, negative, or None for %s. "
                            "Final value: %s",
                            trading_pair,
                            raw_atr_calculated,  # Log the value for clarity
                            extra={"source_module": self._source_module})
                # If df_slice is None, logging is done in _get_atr_dataframe_slice

        if atr_to_return is not None:
            self.logger.debug(
                "Successfully calculated raw ATR for %s: %s",
                trading_pair,
                atr_to_return,
                extra={"source_module": self._source_module})

        return atr_to_return

    def _calculate_normalized_atr(self, trading_pair: str) -> Decimal | None:
        """Calculate ATR and normalize it by the current close price."""
        raw_atr = self._get_raw_atr_for_pair(trading_pair)

        if raw_atr is None:
            # If raw_atr is None, logging about why would have occurred in _get_raw_atr_for_pair.
            # It also implies raw_atr was not positive if it was calculable.
            return None

        current_close_price = self._get_latest_price_at_current_time(trading_pair)
        if current_close_price is None or current_close_price <= Decimal(0):
            self.logger.debug(
                "Could not get a positive current close price for %s to normalize ATR. "
                "Raw ATR was %s.",
                trading_pair,
                raw_atr,
                extra={"source_module": self._source_module})
            return None

        normalized_atr = raw_atr / current_close_price
        self.logger.debug(
            "Calculated normalized ATR for %s: %s (Raw ATR: %s, Close: %s)",
            trading_pair,
            normalized_atr,
            raw_atr,
            current_close_price,
            extra={"source_module": self._source_module})
        return normalized_atr

    def update_time(self, timestamp: datetime) -> None:
        """Update the current simulation time."""
        self.logger.debug(
            "Updating simulated time to: %s",
            timestamp,
            extra={"source_module": self._source_module})
        self._current_timestamp = timestamp

    def _get_price_from_dataframe_asof(
        self,
        df: pd.DataFrame,
        trading_pair: str,  # Needed for updating self.historical_data if sorted
        timestamp_to_lookup: datetime) -> Decimal | None:
        """Extract price from a DataFrame using asof, handling column checks and sorting."""
        price_col = self._price_column
        if price_col not in df.columns:
            self.logger.error(
                "Configured price column '%s' not found in data for %s.",
                price_col,
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        if not df.index.is_monotonic_increasing:
            self.logger.debug(
                "Data for %s is not sorted by index. Sorting now for price lookup.",
                trading_pair,
                extra={"source_module": self._source_module})
            df = df.sort_index()
            self.historical_data[trading_pair] = df  # Update the stored DataFrame

        price_series = df[price_col]
        price_at_timestamp = price_series.asof(timestamp_to_lookup)

        if pd.isna(price_at_timestamp):
            self.logger.warning(
                "Could not find price for %s using column '%s' at or before %s "
                "(asof returned NaN).",
                trading_pair,
                price_col,
                timestamp_to_lookup,
                extra={"source_module": self._source_module})
            return None

        return Decimal(str(price_at_timestamp))

    def _get_latest_price_at_current_time(self, trading_pair: str) -> Decimal | None:
        """Get the latest known price for a trading pair at the current simulation time."""
        current_ts = self._current_timestamp

        if current_ts is None:
            self.logger.error(
                "Cannot get latest price: Simulation time not set.",
                extra={"source_module": self._source_module})
            return None

        # Handle self-referential pairs like "USD/USD"
        if trading_pair.count("/") == 1:
            base, quote = trading_pair.split("/")
            if base == quote:
                return Decimal("1.0")

        # General lookup for all other cases
        pair_data_df = self.historical_data.get(trading_pair)

        if pair_data_df is None:
            self.logger.warning(
                "No historical data found for trading pair: %s",
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        # At this point, data exists; attempt to process it using the helper
        try:
            return self._get_price_from_dataframe_asof(pair_data_df, trading_pair, current_ts)
        except Exception:
            # Use self._price_column for logging in except block
            self.logger.exception(
                "Error retrieving latest price for %s at %s using column '%s'",
                trading_pair,
                current_ts,
                self._price_column,
                extra={"source_module": self._source_module})
            return None

    # --- Interface Alignment Methods (as per MarketPriceService ABC) ---

    async def start(self) -> None:
        """Initialize the simulated service (no-op for simulation)."""
        self.logger.info(
            "SimulatedMarketPriceService started.",
            extra={"source_module": self._source_module})
        # No external connections needed for simulation

    async def stop(self) -> None:
        """Stop the simulated service (no-op for simulation)."""
        self.logger.info(
            "SimulatedMarketPriceService stopped.",
            extra={"source_module": self._source_module})
        # No external connections

    async def get_latest_price(
        self,
        trading_pair: str) -> Decimal | None:  # Changed return type
        """Get the latest known price at the current simulation time.

        Returns the price as a Decimal or None.
        """
        return self._get_latest_price_at_current_time(trading_pair)

    async def get_bid_ask_spread(
        self,
        trading_pair: str) -> tuple[Decimal, Decimal] | None:  # Changed return type
        """Get the simulated bid and ask prices at the current simulation time.

        Returns a tuple[Any, ...] (bid, ask) or None.
        """
        close_price = self._get_latest_price_at_current_time(trading_pair)
        if close_price is None:
            return None

        try:
            pair_specific_cfg = self._pair_specific_spread_config.get(trading_pair, {})
            base_spread_pct_str = pair_specific_cfg.get("base_pct", str(self._default_spread_pct))
            base_spread_pct = Decimal(base_spread_pct_str)

            final_spread_pct = base_spread_pct

            if self._volatility_enabled:
                normalized_atr = self._calculate_normalized_atr(trading_pair)
                if normalized_atr is not None and normalized_atr > Decimal(0):
                    volatility_impact = normalized_atr * self._volatility_multiplier
                    spread_adjustment_factor = Decimal(1) + volatility_impact

                    # Cap the adjustment factor
                    if spread_adjustment_factor > self._max_volatility_adjustment_factor:
                        spread_adjustment_factor = self._max_volatility_adjustment_factor
                        self.logger.debug(
                            "Spread adjustment factor for %s capped at %s.",
                            trading_pair,
                            self._max_volatility_adjustment_factor,
                            extra={"source_module": self._source_module})

                    final_spread_pct = base_spread_pct * spread_adjustment_factor
                    self.logger.debug(
                        "Adjusted spread for %s due to volatility: "
                        "final_spread_pct=%.6f (base=%.6f, norm_atr=%.6f, factor=%.4f)",
                        trading_pair,
                        final_spread_pct,
                        base_spread_pct,
                        normalized_atr,
                        spread_adjustment_factor,
                        extra={"source_module": self._source_module})
                else:
                    self.logger.debug(
                        "Could not calculate normalized ATR or it was non-positive for %s. "
                        "Using base spread.",
                        trading_pair,
                        extra={"source_module": self._source_module})

            if final_spread_pct < Decimal(0):
                self.logger.warning(
                    "Final spread_pct (%s) is negative for %s after adjustments. Clamping to 0.",
                    final_spread_pct,
                    trading_pair,
                    extra={"source_module": self._source_module})
                final_spread_pct = Decimal(0)

            # Call the helper method for the final bid/ask calculation and comparison
            return self._calculate_bid_ask_tuple(close_price, final_spread_pct, trading_pair)

        except Exception:  # pylint: disable=broad-except
            self.logger.exception(
                "Error calculating simulated spread for %s",
                trading_pair,
                extra={"source_module": self._source_module})
            return None

    async def get_price_timestamp(self, trading_pair: str) -> datetime | None:
        """Get the simulation timestamp for which the current price is valid."""
        price = self._get_latest_price_at_current_time(trading_pair)
        if price is not None and self._current_timestamp is not None:
            # Ensure timestamp is timezone-aware if it originated with timezone
            if (
                self._current_timestamp.tzinfo is None
                and self._current_timestamp.utcoffset() is None
            ):
                # If a naive datetime is somehow set, log warning or make UTC
                # For simulation, _current_timestamp should consistently be UTC.
                # Example usage in file implies it is timezone-aware (pytz.UTC).
                pass  # Assuming _current_timestamp is already correctly UTC and aware
            return self._current_timestamp
        return None

    async def get_raw_atr(self, trading_pair: str) -> Decimal | None:
        """Get the latest calculated raw Average True Range (ATR) for the trading pair.

        This value can be used by other services (e.g., SimulatedExecutionHandler)
        to implement volatility-based slippage models.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").

        Returns:
        -------
            The raw ATR as a Decimal, or None if it cannot be calculated.
        """
        # This method is async to align with the MarketPriceService interface,
        # but the underlying calls are currently synchronous.
        return self._get_raw_atr_for_pair(trading_pair)

    async def is_price_fresh(self, trading_pair: str, _max_age_seconds: float = 60.0) -> bool:
        """Check if price data is available at the current simulation time."""
        price_ts = await self.get_price_timestamp(trading_pair)
        return price_ts is not None

    def _calculate_bid_ask_tuple(
        self,
        close_price: Decimal,
        final_spread_pct: Decimal,
        trading_pair: str) -> tuple[Decimal, Decimal] | None:
        """Calculate bid/ask tuple[Any, ...] from close price and final spread percentage."""
        half_spread_amount = close_price * (final_spread_pct / Decimal(200))
        bid = close_price - half_spread_amount
        ask = close_price + half_spread_amount

        if bid > ask:
            self.logger.warning(
                "Simulated spread: bid (%s) > ask (%s) for %s with final_spread_pct %s. "
                "Returning None.",
                bid,
                ask,
                trading_pair,
                final_spread_pct,
                extra={"source_module": self._source_module})
            return None
        if bid == ask:
            self.logger.debug(
                "Calculated zero spread for %s at %s with final_spread_pct %s. Returning as is.",
                trading_pair,
                close_price,
                final_spread_pct,
                extra={"source_module": self._source_module})
            return (bid, ask)
        # bid < ask
        return (bid, ask)

    def _create_book_level_entries(
        self,
        current_bid_for_level: Decimal,
        current_ask_for_level: Decimal,
        current_volume_for_level: Decimal,
        level_index: int,
        context: BookLevelConstructionContext,
    ) -> tuple[list[str] | None, list[str] | None, bool]:  # (bid_entry, ask_entry, stop_gen)
        """Create bid/ask entries for a single order book level and check for termination."""
        bid_entry: list[str] | None = None
        ask_entry: list[str] | None = None
        stop_generation = False

        if level_index == 0:  # BBO level
            bid_entry = [
                context.price_format_str.format(current_bid_for_level),
                context.volume_format_str.format(current_volume_for_level),
            ]
            ask_entry = [
                context.price_format_str.format(current_ask_for_level),
                context.volume_format_str.format(current_volume_for_level),
            ]
        else:
            # Check for crossed/invalid book before creating entries for non-BBO levels
            if current_bid_for_level >= current_ask_for_level and not (
                current_bid_for_level == context.best_bid_price_bbo
                and current_ask_for_level == context.best_ask_price_bbo
                and context.best_bid_price_bbo == context.best_ask_price_bbo
            ):
                self.logger.debug(
                    "At level %s, calculated bid %s crossed/met ask %s for %s. "
                    "Stopping depth here.",
                    level_index + 1,
                    current_bid_for_level,
                    current_ask_for_level,
                    context.trading_pair,
                    extra={"source_module": self._source_module})
                stop_generation = True
                return bid_entry, ask_entry, stop_generation

            if current_bid_for_level <= Decimal(0):
                self.logger.debug(
                    "At level %s, calculated bid %s is zero/negative for %s. "
                    "Stopping bid depth here.",
                    level_index + 1,
                    current_bid_for_level,
                    context.trading_pair,
                    extra={"source_module": self._source_module})
                # Bid entry remains None, but asks can continue if valid
            else:
                bid_entry = [
                    context.price_format_str.format(current_bid_for_level),
                    context.volume_format_str.format(current_volume_for_level),
                ]

            # Ask entry is always attempted if not stopped by crossed book,
            # as a zero bid doesn't preclude valid asks.
            ask_entry = [
                context.price_format_str.format(current_ask_for_level),
                context.volume_format_str.format(current_volume_for_level),
            ]

        return bid_entry, ask_entry, stop_generation

    async def get_order_book_snapshot(
        self,
        trading_pair: str) -> dict[str, list[list[str]]] | None:
        """Generate a simulated order book snapshot with market depth.

        Args:
        ----
            trading_pair: The trading_pair symbol (e.g., "XRP/USD").

        Returns:
        -------
            A dictionary with 'bids' and 'asks' lists, or None if depth cannot be generated.
            Each inner list[Any] is [price_str, volume_str].
        """
        if not self._depth_simulation_enabled:
            self.logger.debug(
                "Market depth simulation is disabled. Skipping for %s.",
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        bbo = await self.get_bid_ask_spread(trading_pair)
        if bbo is None:
            self.logger.warning(
                "Could not retrieve BBO for %s. Cannot generate order book.",
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        best_bid_price_bbo = bbo[0]  # Store initial BBO for reference
        best_ask_price_bbo = bbo[1]

        if best_bid_price_bbo <= Decimal(0) or best_ask_price_bbo <= Decimal(0):
            self.logger.warning(
                "BBO for %s has non-positive price (%s, %s). Cannot generate order book.",
                trading_pair,
                best_bid_price_bbo,
                best_ask_price_bbo,
                extra={"source_module": self._source_module})
            return None

        if best_bid_price_bbo > best_ask_price_bbo:
            self.logger.error(
                "Best bid %s is greater than best ask %s for %s. Order book invalid.",
                best_bid_price_bbo,
                best_ask_price_bbo,
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        bids_levels: list[list[str]] = []
        asks_levels: list[list[str]] = []

        current_level_bid_price = best_bid_price_bbo
        current_level_ask_price = best_ask_price_bbo
        current_level_volume = self._depth_base_volume

        price_format_str = f"{{:.{self._depth_price_precision}f}}"
        volume_format_str = f"{{:.{self._depth_volume_precision}f}}"
        quantizer_price = Decimal("1e-" + str(self._depth_price_precision))
        quantizer_volume = Decimal("1e-" + str(self._depth_volume_precision))

        # Create context object for _create_book_level_entries
        construction_context = BookLevelConstructionContext(
            best_bid_price_bbo=best_bid_price_bbo,
            best_ask_price_bbo=best_ask_price_bbo,
            price_format_str=price_format_str,
            volume_format_str=volume_format_str,
            trading_pair=trading_pair)

        for i in range(self._depth_num_levels):
            quantized_bid = current_level_bid_price.quantize(quantizer_price, rounding=ROUND_DOWN)
            quantized_ask = current_level_ask_price.quantize(quantizer_price, rounding=ROUND_UP)
            quantized_volume = current_level_volume.quantize(quantizer_volume, rounding=ROUND_DOWN)

            if quantized_volume <= Decimal(0):
                self.logger.debug(
                    "Volume for level %s for %s became zero or negative. "
                    "Stopping depth generation.",
                    i + 1,
                    trading_pair,
                    extra={"source_module": self._source_module})
                break

            bid_entry, ask_entry, stop_generation = self._create_book_level_entries(
                quantized_bid,
                quantized_ask,
                quantized_volume,
                i,
                construction_context)

            if bid_entry:
                bids_levels.append(bid_entry)
            if ask_entry:
                asks_levels.append(ask_entry)

            if stop_generation:
                break

            # Calculate prices for the *next* level (i+1)
            if current_level_bid_price > Decimal(0):
                step_bid = current_level_bid_price * self._depth_price_step_pct
                current_level_bid_price -= step_bid

            step_ask = current_level_ask_price * self._depth_price_step_pct
            current_level_ask_price += step_ask
            current_level_volume *= self._depth_volume_decay_factor

        # This is implicitly handled now if bid_entry from helper is None for those cases.
        # Re-affirm by explicit filtering if needed, but helper logic should suffice.

        if not bids_levels and not asks_levels:
            self.logger.info(
                "Generated an empty order book for %s (e.g. BBO was zero spread and "
                "only one level requested, or other edge case).",
                trading_pair,
                extra={"source_module": self._source_module})

        return {"bids": bids_levels, "asks": asks_levels}

    # --- End Interface Alignment Methods ---

    async def _get_direct_or_reverse_price(
        self,
        from_currency: str,
        to_currency: str) -> tuple[Decimal, bool] | None:  # Returns (price, is_direct_rate)
        """Get direct or reverse conversion rate."""
        # Direct conversion: from_currency/to_currency
        pair1 = f"{from_currency}/{to_currency}"
        price1 = await self.get_latest_price(pair1)  # Returns Optional[Decimal]
        if price1 is not None and price1 > 0:
            return price1, True

        # Reverse conversion: to_currency/from_currency
        pair2 = f"{to_currency}/{from_currency}"
        price2 = await self.get_latest_price(pair2)  # Returns Optional[Decimal]
        if price2 is not None and price2 > 0:
            return price2, False
        return None

    async def _get_cross_conversion_price(
        self,
        from_amount: Decimal,
        from_currency: str,
        to_currency: str,
        intermediary: str) -> Decimal | None:
        """Get cross-conversion via an intermediary currency."""
        # Path: from_currency -> intermediary -> to_currency
        from_to_intermediary_rate_info = await self._get_direct_or_reverse_price(
            from_currency,
            intermediary)

        if from_to_intermediary_rate_info:
            rate1, is_direct1 = from_to_intermediary_rate_info
            amount_in_intermediary = from_amount * rate1 if is_direct1 else from_amount / rate1

            intermediary_to_target_rate_info = await self._get_direct_or_reverse_price(
                intermediary,
                to_currency)
            if intermediary_to_target_rate_info:
                rate2, is_direct2 = intermediary_to_target_rate_info
                if is_direct2:
                    return amount_in_intermediary * rate2
                return amount_in_intermediary / rate2
        return None

    async def convert_amount(
        self,
        from_amount: Decimal,
        from_currency: str,
        to_currency: str) -> Decimal | None:
        """Convert an amount from one currency to another using available market data."""
        # 1. Ensure from_amount is Decimal
        if not isinstance(from_amount, Decimal):
            # This code is reachable when from_amount is not Decimal
            self.logger.warning(  # type: ignore[unreachable]
                "convert_amount received non-Decimal from_amount: %s. Attempting conversion.",
                type(from_amount),
                extra={"source_module": self._source_module})
            try:
                from_amount = Decimal(str(from_amount))
            except Exception:
                self.logger.exception(
                    "Could not convert from_amount to Decimal in convert_amount.",
                    extra={"source_module": self._source_module})
                return None  # Early exit if input is not convertible

        # 2. Handle same currency
        if from_currency == to_currency:
            return from_amount

        # 3. Try direct or reverse conversion
        direct_or_reverse_info = await self._get_direct_or_reverse_price(
            from_currency,
            to_currency)
        if direct_or_reverse_info:
            rate, is_direct = direct_or_reverse_info
            try:
                return from_amount * rate if is_direct else from_amount / rate
            except ZeroDivisionError:
                self.logger.exception(
                    "Zero price for direct/reverse conversion (%s/%s or %s/%s).",
                    from_currency,  # Argument for first %s
                    to_currency,  # Argument for second %s
                    to_currency,  # Argument for third %s
                    from_currency,  # Argument for fourth %s
                    extra={"source_module": self._source_module})
                # Fall through to try cross-conversion or fail if ZeroDivisionError occurs

        # 4. Try cross-conversion
        intermediary_currency = self._intermediary_conversion_currency
        # Ensure cross-conversion is applicable (not converting to/from intermediary
        # if direct failed, and ensure currencies are different from intermediary
        # to avoid redundant steps)
        if all(c != intermediary_currency for c in (from_currency, to_currency)):
            try:
                converted_amount = await self._get_cross_conversion_price(
                    from_amount,
                    from_currency,
                    to_currency,
                    intermediary_currency)
                if converted_amount is not None:
                    return converted_amount  # Successful cross-conversion
            except ZeroDivisionError:
                self.logger.exception(
                    "Zero price encountered during %s-mediated cross-conversion from %s to %s.",
                    intermediary_currency,
                    from_currency,
                    to_currency,
                    extra={"source_module": self._source_module})
                # Fall through to final failure if ZeroDivisionError occurs here

        # 5. Log failure if no path found or previous attempts failed through to here
        self.logger.warning(
            "Could not convert %s %s to %s. No direct, reverse, or %s-mediated path found.",
            from_amount,
            from_currency,
            to_currency,
            intermediary_currency,
            extra={"source_module": self._source_module})
        return None

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        timeframe: str,  # - Required for API compatibility
        since: datetime,
        limit: int | None = None) -> list[dict[str, Any]] | None:
        """Fetch historical OHLCV data for a trading pair from the stored historical data.

        Args:
            trading_pair: The trading pair symbol (e.g., "XRP/USD").
            timeframe: The timeframe for the candles (e.g., "1m", "1h", "1d").
            since: Python datetime object indicating the start time for fetching data (UTC).
            limit: The maximum number of candles to return.

        Returns:
        -------
            A list[Any] of dictionaries, where each dictionary represents an OHLCV candle:
            {'timestamp': datetime_obj, 'open': Decimal, 'high': Decimal,
             'low': Decimal, 'close': Decimal, 'volume': Decimal},
            or None if data is unavailable or an error occurs. Timestamps are UTC.
        """
        # Check if we have data for this trading pair
        if trading_pair not in self.historical_data:
            self.logger.warning(
                "No historical data available for trading pair %s",
                trading_pair,
                extra={"source_module": self._source_module})
            return None

        df = self.historical_data[trading_pair]

        # Filter data from the provided start time
        if since is not None:
            df = df[df.index >= since]

        # Apply limit if specified
        if limit is not None and limit > 0:
            df = df.head(limit)

        # If no data available after filtering
        if df.empty:
            return None

        # Convert DataFrame to list[Any] of dictionaries
        result = []
        for timestamp, row in df.iterrows():
            candle = {
                "timestamp": timestamp,
                "open": Decimal(str(row.get("open", 0))),
                "high": Decimal(str(row.get("high", 0))),
                "low": Decimal(str(row.get("low", 0))),
                "close": Decimal(str(row.get("close", 0))),
                "volume": Decimal(str(row.get("volume", 0))),
            }
            result.append(candle)

        return result

    async def get_volatility(
        self,
        trading_pair: str,
        lookback_hours: int = 24) -> float | None:
        """Calculate the price volatility for a trading pair.

        For the simulated service, this returns the normalized ATR if available,
        which represents volatility as a percentage of the current price.
        """
        # Try to calculate volatility using ATR
        try:
            normalized_atr = self._calculate_normalized_atr(trading_pair)
            if normalized_atr is not None:
                # Convert to percentage
                volatility_pct = float(normalized_atr * Decimal(100))
                self.logger.debug(
                    "Calculated volatility for %s: %.2f%% (lookback: %d hours)",
                    trading_pair,
                    volatility_pct,
                    lookback_hours,
                    extra={"source_module": self._source_module})
                return volatility_pct
        except Exception as e:
            self.logger.error(
                "Error calculating volatility for %s: %s",
                trading_pair,
                str(e),
                extra={"source_module": self._source_module})

        # Fallback: calculate simple standard deviation of returns
        try:
            if self._current_timestamp is None:
                self.logger.warning(
                    "Cannot calculate volatility: current timestamp not set",
                    extra={"source_module": self._source_module})
                return None

            pair_data = self.historical_data.get(trading_pair)
            if pair_data is None or pair_data.empty:
                self.logger.warning(
                    "No historical data available for %s to calculate volatility",
                    trading_pair,
                    extra={"source_module": self._source_module})
                return None

            # Get data for lookback period
            lookback_start = self._current_timestamp - timedelta(hours=lookback_hours)
            recent_data = pair_data.loc[lookback_start:self._current_timestamp]  # type: ignore[misc]

            if len(recent_data) < 2:
                self.logger.warning(
                    "Insufficient data points for volatility calculation for %s",
                    trading_pair,
                    extra={"source_module": self._source_module})
                return None

            # Calculate returns
            prices = recent_data[self._price_column]
            returns = prices.pct_change().dropna()

            if len(returns) == 0:
                return None

            # Calculate annualized volatility (assuming 365 days for crypto)
            # Hourly returns -> annualize by sqrt(24 * 365)
            hourly_volatility = returns.std()
            annualized_volatility = hourly_volatility * np.sqrt(24 * 365)

            return float(annualized_volatility * 100)  # Return as percentage

        except Exception as e:
            self.logger.exception(
                "Error in fallback volatility calculation for %s: %s",
                trading_pair,
                str(e),
                extra={"source_module": self._source_module})
            return None


# === ENTERPRISE-GRADE MARKET PRICE SERVICE METHODS ===

    async def load_historical_data_advanced(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "1h",
        data_source: DataSource | None = None,
        force_refresh: bool = False,
    ) -> list[HistoricalDataPoint] | None:
        """Load historical data using the enterprise-grade data loading system.

        This method leverages the advanced caching and multi-source data loading
        capabilities to efficiently retrieve historical market data.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            frequency: Data frequency (1m, 5m, 1h, 1d, etc.)
            data_source: Specific data source to use
            force_refresh: Whether to bypass cache and fetch fresh data

        Returns:
            List of historical data points or None if unavailable
        """
        try:
            request = DataRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                data_source=data_source,
                include_volume=True,
                validate_data=True,
                cache_result=not force_refresh,
            )

            self.logger.info(
                f"Loading advanced historical data for {symbol} from {start_date} to {end_date}",
                extra={"source_module": self._source_module})

            data_points = await self._data_loader.load_historical_data(request)

            if data_points:
                self.logger.info(
                    f"Successfully loaded {len(data_points)} data points for {symbol}",
                    extra={"source_module": self._source_module})

        except DataLoadingError:
            self.logger.exception(
                f"Failed to load historical data for {symbol}: ",
                extra={"source_module": self._source_module})
            return None
        except Exception:
            self.logger.exception(
                f"Unexpected error loading historical data for {symbol}: ",
                extra={"source_module": self._source_module})
            return None
        else:
            return data_points

    async def interpolate_missing_prices(
        self,
        trading_pair: str,
        data: list[dict[str, Any]] | None = None,
        frequency: str = "1h",
        method: InterpolationMethod | None = None,
    ) -> InterpolationResult | None:
        """Interpolate missing price data using advanced algorithms.

        This method uses sophisticated interpolation techniques to fill gaps
        in historical price data, maintaining realistic market characteristics.

        Args:
            trading_pair: Trading pair symbol
            data: Price data to interpolate (if None, uses current historical data)
            frequency: Expected data frequency
            method: Specific interpolation method to use

        Returns:
            InterpolationResult with filled data and quality metrics
        """
        try:
            # Use provided data or extract from historical data
            if data is None:
                pair_data = self.historical_data.get(trading_pair)
                if pair_data is None:
                    self.logger.warning(
                        f"No historical data available for {trading_pair} to interpolate",
                        extra={"source_module": self._source_module})
                    return None

                # Convert DataFrame to list[Any] of dictionaries
                data = cast("list[dict[str, Any]]", pair_data.reset_index().to_dict("records"))
                # Ensure timestamp column is properly named
                if data and "index" in data[0] and "timestamp" not in data[0]:
                    for row in data:
                        row["timestamp"] = row.pop("index")

            self.logger.info(
                f"Starting price interpolation for {trading_pair} with {len(data) if data else 0} data points",
                extra={"source_module": self._source_module})

            result = await self._price_interpolator.interpolate_missing_data(
                data if data else [], trading_pair, frequency,
            )

            if result.gaps_filled:
                self.logger.info(
                    f"Interpolation completed for {trading_pair}: "
                    f"filled {len(result.gaps_filled)} gaps, "
                    f"quality score: {result.quality_score:.2f}",
                    extra={"source_module": self._source_module})

                # Update historical data with interpolated results if applicable
                if trading_pair in self.historical_data:
                    interpolated_df = pd.DataFrame(result.interpolated_data)
                    interpolated_df.set_index("timestamp", inplace=True)
                    self.historical_data[trading_pair] = interpolated_df

        except InterpolationError:
            self.logger.exception(
                f"Failed to interpolate prices for {trading_pair}: ",
                extra={"source_module": self._source_module})
            return None
        except Exception:
            self.logger.exception(
                f"Unexpected error during price interpolation for {trading_pair}: ",
                extra={"source_module": self._source_module})
            return None
        else:
            return result

    async def start_real_time_simulation(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        speed: SimulationSpeed = SimulationSpeed.REAL_TIME,
        symbols: list[str] | None = None,
    ) -> bool:
        """Start the real-time simulation engine.

        This method initiates advanced market simulation with configurable
        replay speeds and comprehensive event handling.

        Args:
            start_time: Simulation start time (defaults to earliest data)
            end_time: Simulation end time (defaults to latest data)
            speed: Simulation replay speed
            symbols: Symbols to include in simulation

        Returns:
            True if simulation started successfully, False otherwise
        """
        try:
            # Update simulation configuration
            if start_time:
                self._simulation_engine.config["start_time"] = start_time
            if end_time:
                self._simulation_engine.config["end_time"] = end_time
            if symbols:
                self._simulation_engine.config["symbols"] = symbols

            # Set simulation speed
            await self._simulation_engine.set_speed(speed)

            self.logger.info(
                f"Starting real-time simulation: speed={speed.value}, "
                f"symbols={len(symbols) if symbols else 'all'}",
                extra={"source_module": self._source_module})

            await self._simulation_engine.start_simulation()

            self.logger.info(
                "Real-time simulation started successfully",
                extra={"source_module": self._source_module})

        except SimulationError:
            self.logger.exception(
                "Failed to start simulation: ",
                extra={"source_module": self._source_module})
            return False
        except Exception:
            self.logger.exception(
                "Unexpected error starting simulation: ",
                extra={"source_module": self._source_module})
            return False
        else:
            return True

    async def stop_real_time_simulation(self) -> bool:
        """Stop the real-time simulation engine.

        Returns:
            True if simulation stopped successfully, False otherwise
        """
        try:
            await self._simulation_engine.stop_simulation()

            self.logger.info(
                "Real-time simulation stopped successfully",
                extra={"source_module": self._source_module})

        except Exception:
            self.logger.exception(
                "Error stopping simulation: ",
                extra={"source_module": self._source_module})
            return False
        else:
            return True

    async def set_simulation_speed(self, speed: SimulationSpeed) -> bool:
        """Change the simulation replay speed during runtime.

        Args:
            speed: New simulation speed

        Returns:
            True if speed changed successfully, False otherwise
        """
        try:
            await self._simulation_engine.set_speed(speed)

            self.logger.info(
                f"Simulation speed changed to {speed.value}",
                extra={"source_module": self._source_module})

        except Exception:
            self.logger.exception(
                "Error changing simulation speed: ",
                extra={"source_module": self._source_module})
            return False
        else:
            return True

    def get_simulation_state(self) -> SimulationState:
        """Get the current state of the simulation engine.

        Returns:
            Current simulation state
        """
        return self._simulation_engine.state

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get performance statistics for the data loading cache.

        Returns:
            Dictionary containing cache hit rates and performance metrics
        """
        return self._data_loader.get_cache_statistics()

    async def _handle_price_update_event(self, event: SimulationEvent) -> None:
        """Handle price update events from the simulation engine.

        This method is called when the simulation engine generates price update events
        and integrates them with the market price service.

        Args:
            event: Price update event containing market data
        """
        try:
            symbol = event.data.get("symbol")
            price_data = event.data.get("price_data")

            if symbol and price_data:
                # Update current timestamp to match simulation time
                self.update_time(event.timestamp)

                self.logger.debug(
                    f"Processed price update event for {symbol} at {event.timestamp}",
                    extra={"source_module": self._source_module})

        except Exception:
            self.logger.exception(
                "Error handling price update event: ",
                extra={"source_module": self._source_module})

    async def validate_data_quality(
        self,
        trading_pair: str,
        quality_threshold: float = 0.8,
    ) -> dict[str, Any] | None:
        """Validate the quality of historical data for a trading pair.

        This method performs comprehensive data quality analysis including
        completeness, consistency, and anomaly detection.

        Args:
            trading_pair: Trading pair to validate
            quality_threshold: Minimum acceptable quality score (0.0 to 1.0)

        Returns:
            Dictionary containing quality metrics and recommendations
        """
        try:
            pair_data = self.historical_data.get(trading_pair)
            if pair_data is None:
                return {
                    "quality_score": 0.0,
                    "issues": ["No data available"],
                    "recommendations": ["Load historical data for this trading pair"],
                }

            # Basic quality checks
            total_points = len(pair_data)
            missing_values = pair_data.isnull().sum().sum()
            duplicate_timestamps = pair_data.index.duplicated().sum()

            # Calculate quality score
            completeness = 1.0 - (missing_values / (total_points * len(pair_data.columns)))
            uniqueness = 1.0 - (duplicate_timestamps / total_points)
            quality_score = (completeness + uniqueness) / 2.0

            issues = []
            recommendations = []

            if missing_values > 0:
                issues.append(f"Found {missing_values} missing values")
                recommendations.append("Consider using price interpolation to fill gaps")

            if duplicate_timestamps > 0:
                issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
                recommendations.append("Remove or consolidate duplicate entries")

            if quality_score < quality_threshold:
                recommendations.append("Data quality below threshold - review data sources")

            self.logger.info(
                f"Data quality validation for {trading_pair}: "
                f"score={quality_score:.2f}, issues={len(issues)}",
                extra={"source_module": self._source_module})

        except Exception:
            self.logger.exception(
                f"Error validating data quality for {trading_pair}: ",
                extra={"source_module": self._source_module})
            return None
        else:
            return {
                "quality_score": quality_score,
                "completeness": completeness,
                "uniqueness": uniqueness,
                "total_points": total_points,
                "missing_values": missing_values,
                "duplicate_timestamps": duplicate_timestamps,
                "issues": issues,
                "recommendations": recommendations,
                "meets_threshold": quality_score >= quality_threshold,
            }

# === END ENTERPRISE-GRADE METHODS ===
