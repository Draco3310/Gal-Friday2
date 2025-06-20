"""Gap detection for time series data."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, NamedTuple, cast as typing_cast

import numpy as np
import pandas as pd

from gal_friday.logger_service import LoggerService


class DataGap(NamedTuple):
    """Represents a gap in time series data."""
    start: datetime
    end: datetime
    duration: timedelta
    expected_points: int
    actual_points: int
    severity: str  # 'minor', 'major', 'critical'


class InterpolationMethod(str, Enum):
    """Available interpolation methods for gap filling."""
    LINEAR = "linear"
    SPLINE = "spline"
    CUBIC_SPLINE = "cubic_spline"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    TIME_WEIGHTED = "time_weighted"
    POLYNOMIAL = "polynomial"
    SEASONAL = "seasonal"
    NONE = "none"  # Skip interpolation


@dataclass
class GapInfo:
    """Information about a detected data gap."""
    start_time: datetime
    end_time: datetime
    duration: timedelta
    symbol: str
    data_type: str  # 'ohlcv', 'trades', 'orderbook'
    gap_size: int  # number of missing data points
    preceding_data_quality: float  # quality score of data before gap
    following_data_quality: float  # quality score of data after gap
    severity: str = "minor"  # 'minor', 'major', 'critical'


@dataclass
class InterpolationConfig:
    """Configuration for interpolation methods."""
    default_method: InterpolationMethod = InterpolationMethod.LINEAR
    max_gap_duration: timedelta = timedelta(minutes=30)
    min_surrounding_data_points: int = 5
    quality_threshold: float = 0.8
    method_overrides: dict[str, InterpolationMethod] | None = None
    validation_enabled: bool = True

    def __post_init__(self) -> None:
        """Initialize method_overrides if None."""
        if self.method_overrides is None:
            self.method_overrides = {}


class DataInterpolator:
    """Production-grade data interpolation system."""

    def __init__(self, config: InterpolationConfig, logger: LoggerService) -> None:
        """Initialize the data interpolator.

        Args:
            config: Interpolation configuration
            logger: Logger service instance
        """
        self.config = config
        self.logger = logger
        self._source_module = "DataInterpolator"
        self.interpolation_stats: dict[str, Any] = {
            "total_gaps_processed": 0,
            "successful_interpolations": 0,
            "failed_interpolations": 0,
            "methods_used": {},
            "average_gap_duration": timedelta(0),
        }

    def interpolate_missing_data(self, gap_info: GapInfo,
                               preceding_data: pd.DataFrame,
                               following_data: pd.DataFrame) -> pd.DataFrame | None:
        """Interpolate missing data points for a detected gap.

        Args:
            gap_info: Information about the gap to fill
            preceding_data: Data points before the gap
            following_data: Data points after the gap

        Returns:
            DataFrame with interpolated data points, or None if interpolation failed
        """
        try:
            self.logger.info(
                f"Interpolating {gap_info.gap_size} missing data points for "
                f"{gap_info.symbol} ({gap_info.data_type}) gap from "
                f"{gap_info.start_time} to {gap_info.end_time}",
                source_module=self._source_module)

            # Validate interpolation feasibility
            if not self._validate_interpolation_feasibility(gap_info, preceding_data, following_data):
                self.logger.warning(
                    f"Gap interpolation not feasible for {gap_info.symbol}",
                    source_module=self._source_module)
                return None

            # Select interpolation method
            method = self._select_interpolation_method(gap_info, preceding_data, following_data)

            # Perform interpolation
            interpolated_data = self._perform_interpolation(
                method, gap_info, preceding_data, following_data)

            # Validate interpolated results
            if self.config.validation_enabled:
                if not self._validate_interpolated_data(interpolated_data, preceding_data, following_data):
                    self.logger.exception(
                        f"Interpolated data failed validation for {gap_info.symbol}",
                        source_module=self._source_module)
                    return None

            # Update statistics
            self._update_interpolation_stats(gap_info, method, success=True)

            # Add metadata to interpolated data
            interpolated_data = self._add_interpolation_metadata(interpolated_data, method, gap_info)

            self.logger.info(
                f"Successfully interpolated {len(interpolated_data)} data points "
                f"using {method.value} method for {gap_info.symbol}",
                source_module=self._source_module)

        except Exception as e:
            self.logger.error(
                f"Error interpolating data for {gap_info.symbol}: {e}",
                source_module=self._source_module)
            self._update_interpolation_stats(gap_info, None, success=False)
            return None
        else:
            return interpolated_data

    def _validate_interpolation_feasibility(self, gap_info: GapInfo,
                                          preceding_data: pd.DataFrame,
                                          following_data: pd.DataFrame) -> bool:
        """Validate that interpolation is feasible and advisable."""
        # Check gap duration limits
        if gap_info.duration > self.config.max_gap_duration:
            self.logger.warning(
                f"Gap duration {gap_info.duration} exceeds maximum "
                f"{self.config.max_gap_duration} for {gap_info.symbol}",
                source_module=self._source_module)
            return False

        # Check surrounding data availability
        if (len(preceding_data) < self.config.min_surrounding_data_points or
            len(following_data) < self.config.min_surrounding_data_points):
            self.logger.warning(
                f"Insufficient surrounding data for interpolation: "
                f"preceding={len(preceding_data)}, following={len(following_data)}",
                source_module=self._source_module)
            return False

        # Check data quality
        if (gap_info.preceding_data_quality < self.config.quality_threshold or
            gap_info.following_data_quality < self.config.quality_threshold):
            self.logger.warning(
                f"Surrounding data quality too low for reliable interpolation: "
                f"preceding={gap_info.preceding_data_quality}, "
                f"following={gap_info.following_data_quality}",
                source_module=self._source_module)
            return False

        return True

    def _select_interpolation_method(self, gap_info: GapInfo,
                                   preceding_data: pd.DataFrame,
                                   following_data: pd.DataFrame) -> InterpolationMethod:
        """Select optimal interpolation method based on gap characteristics."""
        # Check for method overrides
        override_key = f"{gap_info.symbol}_{gap_info.data_type}"
        if self.config.method_overrides and override_key in self.config.method_overrides:
            return self.config.method_overrides[override_key]

        # Select method based on gap characteristics
        if gap_info.duration <= timedelta(minutes=5):
            # Short gaps - use linear interpolation
            return InterpolationMethod.LINEAR

        if gap_info.duration <= timedelta(minutes=15):
            # Medium gaps - use cubic spline for smoothness
            return InterpolationMethod.CUBIC_SPLINE

        if gap_info.data_type == "ohlcv":
            # OHLCV data - use time-weighted method
            return InterpolationMethod.TIME_WEIGHTED

        # Default to forward fill for other cases
        return InterpolationMethod.FORWARD_FILL

    def _perform_interpolation(self, method: InterpolationMethod,
                             gap_info: GapInfo,
                             preceding_data: pd.DataFrame,
                             following_data: pd.DataFrame) -> pd.DataFrame:
        """Execute the selected interpolation method."""
        # Create time index for missing data points
        time_index = self._create_interpolation_time_index(gap_info)

        # Combine surrounding data for interpolation
        combined_data = pd.concat([preceding_data, following_data]).sort_index()

        if method == InterpolationMethod.LINEAR:
            return self._linear_interpolation(combined_data, time_index)

        if method == InterpolationMethod.SPLINE:
            return self._spline_interpolation(combined_data, time_index)

        if method == InterpolationMethod.CUBIC_SPLINE:
            return self._cubic_spline_interpolation(combined_data, time_index)

        if method == InterpolationMethod.FORWARD_FILL:
            return self._forward_fill_interpolation(preceding_data, time_index)

        if method == InterpolationMethod.BACKWARD_FILL:
            return self._backward_fill_interpolation(following_data, time_index)

        if method == InterpolationMethod.TIME_WEIGHTED:
            return self._time_weighted_interpolation(combined_data, time_index, gap_info)

        if method == InterpolationMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_interpolation(combined_data, time_index, gap_info)

        raise ValueError(f"Unsupported interpolation method: {method}")

    def _create_interpolation_time_index(self, gap_info: GapInfo) -> pd.DatetimeIndex:
        """Create time index for interpolation points."""
        # Estimate frequency based on gap size and duration
        if gap_info.gap_size > 1:
            freq_seconds = gap_info.duration.total_seconds() / (gap_info.gap_size - 1)
            freq = pd.Timedelta(seconds=freq_seconds)
        else:
            # Default to 1-minute intervals
            freq = pd.Timedelta(minutes=1)

        # Generate time range
        return pd.date_range(
            start=gap_info.start_time,
            end=gap_info.end_time,
            freq=freq,
            inclusive="neither",  # Exclude start and end points
        )

    def _linear_interpolation(self, data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform linear interpolation."""
        # Reindex data to include missing time points
        extended_data = data.reindex(data.index.union(time_index))

        # Interpolate missing values
        interpolated = extended_data.interpolate(method="linear")

        # Return only the interpolated points
        return interpolated.loc[time_index]

    def _cubic_spline_interpolation(self, data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform cubic spline interpolation for smoother results."""
        try:
            from scipy.interpolate import CubicSpline
        except ImportError:
            self.logger.warning(
                "SciPy not available, falling back to linear interpolation",
                source_module=self._source_module)
            return self._linear_interpolation(data, time_index)

        # Convert timestamps to numeric for spline fitting
        data_numeric = data.copy()
        data_numeric.index = data_numeric.index.astype(np.int64) // 10**9  # Convert to seconds

        interpolated_data = []

        for column in data.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue

            # Fit cubic spline
            cs = CubicSpline(data_numeric.index.values, data_numeric[column].values)

            # Interpolate at new time points
            interpolated_values = cs(time_index.astype(np.int64) // 10**9)
            interpolated_data.append(interpolated_values)

        # Create result DataFrame
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        return pd.DataFrame(
            np.column_stack(interpolated_data),
            index=time_index,
            columns=numeric_columns)


    def _spline_interpolation(
        self, data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform spline interpolation with fallback to linear."""
        if len(data) < 4:
            # Not enough data points for spline of order 3
            return self._linear_interpolation(data, time_index)

        try:
            import scipy  # noqa: F401
        except Exception:
            self.logger.warning(
                "SciPy not available, falling back to linear interpolation",
                source_module=self._source_module)
            return self._linear_interpolation(data, time_index)

        # Reindex to include interpolation points
        extended_data = data.reindex(data.index.union(time_index))

        # Convert index to numeric for spline
        numeric_index = extended_data.index.astype("int64") // 10**9
        numeric_df = extended_data.copy()
        numeric_df.index = numeric_index

        interpolated = numeric_df.interpolate(method="spline", order=3)
        interpolated.index = extended_data.index

        return interpolated.loc[time_index]

    def _volatility_adjusted_interpolation(
        self,
        data: pd.DataFrame,
        time_index: pd.DatetimeIndex,
        gap_info: GapInfo) -> pd.DataFrame:
        """Interpolate using spline and adjust results based on volatility."""
        base = self._spline_interpolation(data, time_index)

        try:
            import pandas_ta as ta
        except Exception:
            self.logger.warning(
                "pandas_ta not available, skipping volatility adjustment",
                source_module=self._source_module)
            return base

        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(data.columns):
            return base

        window = data.loc[
            (data.index >= gap_info.start_time - timedelta(minutes=10))
            & (data.index <= gap_info.end_time + timedelta(minutes=10))
        ]
        if len(window) < 2:
            return base

        length = min(14, len(window))
        atr_series = ta.atr(
            high=window["high"],
            low=window["low"],
            close=window["close"],
            length=length)
        if atr_series.isna().all():
            return base

        atr = atr_series.dropna().iloc[-1]

        try:
            close_before = data[data.index < gap_info.start_time].iloc[-1]["close"]
            close_after = data[data.index > gap_info.end_time].iloc[0]["close"]
            direction = np.sign(close_after - close_before)
        except Exception:
            direction = 1.0

        adjustments = np.sin(np.linspace(0, np.pi, len(base))) * atr * direction * 0.1

        for col in ["open", "high", "low", "close"]:
            if col in base.columns:
                base[col] = base[col] + adjustments

        return base

    def _forward_fill_interpolation(self, preceding_data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform forward fill interpolation."""
        if preceding_data.empty:
            raise ValueError("No preceding data available for forward fill")

        # Use last known values
        last_values = preceding_data.iloc[-1]

        # Create DataFrame with repeated values
        return pd.DataFrame(
            [last_values] * len(time_index),
            index=time_index)


    def _backward_fill_interpolation(self, following_data: pd.DataFrame, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Perform backward fill interpolation."""
        if following_data.empty:
            raise ValueError("No following data available for backward fill")

        # Use first known values
        first_values = following_data.iloc[0]

        # Create DataFrame with repeated values
        return pd.DataFrame(
            [first_values] * len(time_index),
            index=time_index)


    def _time_weighted_interpolation(self, data: pd.DataFrame,
                                   time_index: pd.DatetimeIndex,
                                   gap_info: GapInfo) -> pd.DataFrame:
        """Perform time-weighted interpolation considering temporal distance."""
        # Get boundary values
        before_gap = data[data.index < gap_info.start_time]
        after_gap = data[data.index > gap_info.end_time]

        if before_gap.empty or after_gap.empty:
            raise ValueError("Insufficient boundary data for time-weighted interpolation")

        last_before = before_gap.iloc[-1]
        first_after = after_gap.iloc[0]

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
        """Validate interpolated data for reasonableness."""
        try:
            # Check for NaN values
            if interpolated_data.isnull().any().any():
                self.logger.exception("Interpolated data contains NaN values", source_module=self._source_module)
                return False

            # Check for reasonable value ranges
            combined_data = pd.concat([preceding_data, following_data])

            for column in interpolated_data.columns:
                if not pd.api.types.is_numeric_dtype(interpolated_data[column]):
                    continue

                # Get bounds with buffer
                col_min = combined_data[column].min()
                col_max = combined_data[column].max()
                buffer = (col_max - col_min) * 0.1
                lower_bound = col_min - buffer
                upper_bound = col_max + buffer

                if (interpolated_data[column] < lower_bound).any():
                    self.logger.warning(
                        f"Interpolated {column} values below reasonable range",
                        source_module=self._source_module)
                    return False
                if (interpolated_data[column] > upper_bound).any():
                    self.logger.warning(
                        f"Interpolated {column} values above reasonable range",
                        source_module=self._source_module)
                    return False

            # Check for smoothness (no sudden jumps)
            for column in interpolated_data.columns:
                if not pd.api.types.is_numeric_dtype(interpolated_data[column]):
                    continue

                if len(interpolated_data) > 1:
                    col_series: pd.Series[Any] = interpolated_data[column]
                    diffs: pd.Series[Any] = col_series.diff().abs()
                    max_diff_val: Any = diffs.max()
                    max_diff = float(max_diff_val) if pd.notna(max_diff_val) else 0.0

                    combined_col: pd.Series[Any] = combined_data[column]
                    quantile_val: Any = combined_col.diff().abs().quantile(0.95)
                    typical_diff = float(quantile_val) if pd.notna(quantile_val) else 0.0

                    if (pd.notna(typical_diff) and pd.notna(max_diff)
                            and max_diff > typical_diff * 3):  # Allow 3x typical movement
                        self.logger.warning(
                            f"Interpolated {column} shows unusual jumps",
                            source_module=self._source_module)
                        return False

        except Exception:
            self.logger.exception("Error validating interpolated data: ", source_module=self._source_module)
            return False
        else:
            return True

    def _add_interpolation_metadata(
        self, data: pd.DataFrame, method: InterpolationMethod, gap_info: GapInfo,
    ) -> pd.DataFrame:
        """Add metadata columns to mark interpolated data."""
        result = data.copy()
        result["_interpolated"] = True
        result["_interpolation_method"] = method.value
        result["_gap_duration"] = gap_info.duration.total_seconds()
        result["_gap_severity"] = gap_info.severity

        return result

    def _update_interpolation_stats(self, gap_info: GapInfo, method: InterpolationMethod | None, success: bool) -> None:
        """Update interpolation statistics."""
        self.interpolation_stats["total_gaps_processed"] += 1

        if success:
            self.interpolation_stats["successful_interpolations"] += 1
            if method:
                method_name = method.value
                if method_name not in self.interpolation_stats["methods_used"]:
                    self.interpolation_stats["methods_used"][method_name] = 0
                self.interpolation_stats["methods_used"][method_name] += 1
        else:
            self.interpolation_stats["failed_interpolations"] += 1

    def get_interpolation_stats(self) -> dict[str, Any]:
        """Get current interpolation statistics."""
        return self.interpolation_stats.copy()


class GapDetector:
    """Detects and analyzes gaps in time series data.

    Features:
    - Configurable gap thresholds
    - Multiple severity levels
    - Gap statistics and reporting
    - Enterprise-grade interpolation strategies
    """

    def __init__(self, logger: LoggerService, interpolation_config: InterpolationConfig | None = None) -> None:
        """Initialize gap detector.

        Args:
            logger: Logger service
            interpolation_config: Configuration for interpolation system
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Gap thresholds (in expected intervals)
        self.minor_threshold = 2  # 2x expected interval
        self.major_threshold = 5  # 5x expected interval
        self.critical_threshold = 10  # 10x expected interval

        # Initialize interpolation system
        if interpolation_config is None:
            interpolation_config = InterpolationConfig()

        self.interpolator = DataInterpolator(interpolation_config, logger)

    def detect_gaps(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        expected_interval: timedelta | None = None) -> list[DataGap]:
        """Detect gaps in time series data.

        Args:
            data: DataFrame with time series data
            timestamp_col: Name of timestamp column
            expected_interval: Expected time between data points

        Returns:
            List of detected gaps
        """
        if data.empty:
            return []

        # Sort by timestamp
        data = data.sort_values(timestamp_col)
        timestamps = pd.to_datetime(data[timestamp_col])

        # Auto-detect interval if not provided
        if expected_interval is None:
            expected_interval = self._detect_interval(timestamps)

        gaps = []

        for i in range(1, len(timestamps)):
            actual_interval = timestamps.iloc[i] - timestamps.iloc[i-1]

            # Check if gap exists
            if actual_interval > expected_interval * 1.5:
                gap_start = timestamps.iloc[i-1]
                gap_end = timestamps.iloc[i]

                # Calculate gap properties
                expected_points = int(actual_interval / expected_interval)
                actual_points = 1  # Only one transition

                # Determine severity
                severity = self._classify_severity(
                    actual_interval,
                    expected_interval)

                gap = DataGap(
                    start=gap_start,
                    end=gap_end,
                    duration=actual_interval,
                    expected_points=expected_points,
                    actual_points=actual_points,
                    severity=severity)

                gaps.append(gap)

        if gaps:
            self.logger.warning(
                f"Detected {len(gaps)} gaps in time series data",
                source_module=self._source_module,
                context={
                    "minor": sum(1 for g in gaps if g.severity == "minor"),
                    "major": sum(1 for g in gaps if g.severity == "major"),
                    "critical": sum(1 for g in gaps if g.severity == "critical"),
                })

        return gaps

    def analyze_gap_patterns(
        self,
        gaps: list[DataGap]) -> dict[str, Any]:
        """Analyze patterns in detected gaps.

        Args:
            gaps: List of detected gaps

        Returns:
            Gap analysis statistics
        """
        if not gaps:
            return {
                "total_gaps": 0,
                "total_duration": timedelta(0),
                "patterns": [],
            }

        # Basic statistics
        total_duration = sum((g.duration for g in gaps), timedelta())
        [g.duration.total_seconds() for g in gaps]

        durations_seconds = [g.duration.total_seconds() for g in gaps] # ensure list[Any] is not empty before np.mean
        avg_duration_seconds = np.mean(durations_seconds) if durations_seconds else 0.0

        stats = {
            "total_gaps": len(gaps),
            "total_duration": total_duration,
            "average_duration": timedelta(seconds=float(avg_duration_seconds)), # Explicit float cast
            "max_duration": max(g.duration for g in gaps),
            "min_duration": min(g.duration for g in gaps),
            "severity_distribution": {
                "minor": sum(1 for g in gaps if g.severity == "minor"),
                "major": sum(1 for g in gaps if g.severity == "major"),
                "critical": sum(1 for g in gaps if g.severity == "critical"),
            },
        }

        # Detect patterns (e.g., regular gaps at specific times)
        patterns = self._detect_patterns(gaps)
        stats["patterns"] = patterns

        return stats

    def fill_gaps(
        self,
        data: pd.DataFrame,
        gaps: list[DataGap],
        timestamp_col: str = "timestamp",
        method: str = "interpolate",
        symbol: str = "unknown",
        data_type: str = "ohlcv") -> pd.DataFrame:
        """Fill detected gaps in data using enterprise-grade interpolation.

        Args:
            data: Original DataFrame
            gaps: List of gaps to fill
            timestamp_col: Name of timestamp column
            method: Gap filling method ('interpolate', 'enterprise', 'forward', 'zero')
            symbol: Symbol identifier for the data
            data_type: Type[Any] of data ('ohlcv', 'trades', 'orderbook')

        Returns:
            DataFrame with filled gaps
        """
        if not gaps:
            return data

        filled_data = data.copy()

        # Sort data by timestamp for proper interpolation
        filled_data = filled_data.sort_values(timestamp_col).reset_index(drop=True)
        filled_data[timestamp_col] = pd.to_datetime(filled_data[timestamp_col])
        filled_data.set_index(timestamp_col, inplace=True)

        for gap in gaps:
            if gap.severity == "critical":
                # Don't fill critical gaps
                self.logger.warning(
                    f"Skipping critical gap from {gap.start} to {gap.end}",
                    source_module=self._source_module)
                continue

            if method == "enterprise":
                # Use the new enterprise-grade interpolation system
                interpolated_data = self._fill_gap_with_interpolation(
                    gap, filled_data, symbol, data_type)
                if interpolated_data is not None:
                    # Insert interpolated data into the main DataFrame
                    filled_data = pd.concat([filled_data, interpolated_data]).sort_index()
                    filled_data = filled_data[~filled_data.index.duplicated(keep="first")]
                continue

            # Legacy gap filling methods
            freq = self._detect_frequency(filled_data.index.to_series())
            missing_times = pd.date_range(
                start=gap.start + pd.Timedelta(freq),
                end=gap.end - pd.Timedelta(freq),
                freq=freq)

            if len(missing_times) == 0:
                continue

            # Create rows for missing data
            missing_rows = pd.DataFrame(index=missing_times)

            # Set other columns based on method
            numeric_cols = filled_data.select_dtypes(include=[np.number]).columns

            if method == "interpolate":
                # Basic interpolation after merge
                pass
            elif method == "forward":
                # Forward fill from last known value
                before_gap = filled_data[filled_data.index <= gap.start]
                if not before_gap.empty:
                    last_value = before_gap.iloc[-1]
                    for col in numeric_cols:
                        missing_rows[col] = last_value[col]
            elif method == "zero":
                # Fill with zeros
                for col in numeric_cols:
                    missing_rows[col] = 0

            # Merge missing rows
            filled_data = pd.concat([filled_data, missing_rows]).sort_index()

        # Apply basic interpolation if requested
        if method == "interpolate":
            numeric_cols = filled_data.select_dtypes(include=[np.number]).columns
            filled_data[numeric_cols] = filled_data[numeric_cols].interpolate(method="time")

        self.logger.info(
            f"Filled {len(gaps)} gaps using {method} method",
            source_module=self._source_module)

        # Reset index to return DataFrame in original format
        filled_data.reset_index(inplace=True)
        return filled_data

    def _fill_gap_with_interpolation(
        self,
        gap: DataGap,
        data: pd.DataFrame,
        symbol: str,
        data_type: str) -> pd.DataFrame | None:
        """Fill a single gap using the enterprise interpolation system.

        Args:
            gap: DataGap object describing the gap
            data: DataFrame with timestamp index
            symbol: Symbol identifier
            data_type: Type[Any] of data

        Returns:
            Interpolated data for the gap, or None if interpolation failed
        """
        try:
            # Get surrounding data
            before_gap = data[data.index < gap.start]
            after_gap = data[data.index > gap.end]

            # Take recent surrounding data points for interpolation
            context_points = self.interpolator.config.min_surrounding_data_points * 2
            preceding_data = before_gap.tail(context_points) if not before_gap.empty else pd.DataFrame()
            following_data = after_gap.head(context_points) if not after_gap.empty else pd.DataFrame()

            # Calculate data quality scores (simplified)
            preceding_quality = self._calculate_data_quality(preceding_data)
            following_quality = self._calculate_data_quality(following_data)

            # Create GapInfo object
            gap_info = GapInfo(
                start_time=gap.start,
                end_time=gap.end,
                duration=gap.duration,
                symbol=symbol,
                data_type=data_type,
                gap_size=gap.expected_points,
                preceding_data_quality=preceding_quality,
                following_data_quality=following_quality,
                severity=gap.severity)

            # Use the enterprise interpolation system
            return self.interpolator.interpolate_missing_data(
                gap_info, preceding_data, following_data)

        except Exception as e:
            self.logger.error(
                f"Error in enterprise gap filling: {e}",
                source_module=self._source_module)
            return None

    def _calculate_data_quality(self, data: pd.DataFrame) -> float:
        """Calculate a simple data quality score.

        Args:
            data: DataFrame to assess

        Returns:
            Quality score between 0.0 and 1.0
        """
        if data.empty:
            return 0.0

        # Simple quality metrics
        total_cells = data.size
        null_cells = data.isnull().sum().sum()
        null_ratio = null_cells / total_cells if total_cells > 0 else 1.0

        # Check for anomalies in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        anomaly_ratio = 0.0

        if len(numeric_cols) > 0:
            for col in numeric_cols:
                if len(data[col].dropna()) > 3:  # Need at least 3 points for statistics
                    q75, q25 = np.percentile(data[col].dropna(), [75, 25])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr

                    outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                    anomaly_ratio += outliers / len(data[col])

            anomaly_ratio /= len(numeric_cols)

        # Combine metrics (higher is better)
        completeness_score = 1.0 - null_ratio
        normality_score = 1.0 - min(anomaly_ratio, 1.0)

        return (completeness_score + normality_score) / 2.0

    def get_interpolation_stats(self) -> dict[str, Any]:
        """Get current interpolation statistics.

        Returns:
            Dictionary containing interpolation statistics
        """
        return self.interpolator.get_interpolation_stats()

    def configure_interpolation(
        self,
        method_overrides: dict[str, InterpolationMethod] | None = None,
        max_gap_duration: timedelta | None = None,
        quality_threshold: float | None = None) -> None:
        """Configure interpolation settings.

        Args:
            method_overrides: Symbol/data type specific method overrides
            max_gap_duration: Maximum gap duration to interpolate
            quality_threshold: Minimum data quality threshold for interpolation
        """
        if method_overrides is not None and self.interpolator.config.method_overrides is not None:
            self.interpolator.config.method_overrides.update(method_overrides)

        if max_gap_duration is not None:
            self.interpolator.config.max_gap_duration = max_gap_duration

        if quality_threshold is not None:
            self.interpolator.config.quality_threshold = quality_threshold

    def _detect_interval(self, timestamps: pd.Series[Any]) -> timedelta:
        """Auto-detect the expected interval between timestamps."""
        if len(timestamps) < 2:
            return timedelta(minutes=1)  # Default for insufficient data

        intervals = timestamps.diff().dropna()

        if intervals.empty:
            self.logger.warning(
                "Could not determine interval from timestamps (empty after diff/dropna), defaulting to 1 minute.",
                source_module=self._source_module)
            return timedelta(minutes=1)

        # Use mode (most common interval)
        mode_interval_series = intervals.mode()
        if not mode_interval_series.empty:
            # Convert pandas.Timedelta to datetime.timedelta
            # Access first item and convert to timedelta
            first_mode = typing_cast("Any", mode_interval_series.iloc[0])
            if hasattr(first_mode, "to_pytimedelta"):
                py_delta = first_mode.to_pytimedelta()
            # If already a timedelta or similar
            elif pd.notna(first_mode):
                py_delta = pd.Timedelta(first_mode).to_pytimedelta()
            else:
                py_delta = timedelta(minutes=1)  # Default fallback
            return typing_cast("timedelta", py_delta) # Cast to timedelta

        # Fallback to median if mode is empty (e.g., all intervals are unique)
        # Since intervals is guaranteed to be non-empty at this point,
        # median() will always return a valid value
        median_val = intervals.median()
        if hasattr(median_val, "to_pytimedelta"):
            py_delta = median_val.to_pytimedelta()
        else:
            py_delta = pd.Timedelta(median_val).to_pytimedelta()
        return typing_cast("timedelta", py_delta) # Cast to timedelta

    def _detect_frequency(self, timestamps: pd.Series[Any]) -> str:
        """Detect pandas frequency string."""
        intervals = timestamps.diff().dropna()
        median_interval = intervals.median()
        if hasattr(median_interval, "total_seconds"):
            median_seconds = median_interval.total_seconds()
        else:
            # Convert to Timedelta if needed
            median_seconds = pd.Timedelta(median_interval).total_seconds()

        # Map to pandas frequency
        if median_seconds < 60:
            return f"{int(median_seconds)}S"
        if median_seconds < 3600:
            return f"{int(median_seconds/60)}T"
        if median_seconds < 86400:
            return f"{int(median_seconds/3600)}H"
        return f"{int(median_seconds/86400)}D"

    def _classify_severity(
        self,
        actual_interval: timedelta,
        expected_interval: timedelta) -> str:
        """Classify gap severity."""
        ratio = actual_interval / expected_interval

        if ratio < self.minor_threshold:
            return "minor"
        if ratio < self.major_threshold:
            return "major"
        return "critical"

    def _detect_patterns(self, gaps: list[DataGap]) -> list[dict[str, Any]]:
        """Detect patterns in gaps (e.g., regular occurrences)."""
        patterns = []

        # Check for regular time-of-day patterns
        gap_hours = [g.start.hour for g in gaps]
        hour_counts = pd.Series(gap_hours).value_counts()

        # If certain hours have multiple gaps
        for hour, count in hour_counts.items():
            if count >= 3:  # At least 3 occurrences
                patterns.append({
                    "type": "time_of_day",
                    "hour": hour,
                    "occurrences": count,
                    "description": f"Frequent gaps at {hour:02d}:00",
                })

        # Check for day-of-week patterns
        gap_days = [g.start.weekday() for g in gaps]
        day_counts = pd.Series(gap_days).value_counts()

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]

        for day, count in day_counts.items():
            if count >= 2:  # At least 2 occurrences
                patterns.append({
                    "type": "day_of_week",
                    "day": day,
                    "day_name": day_names[int(day)] if isinstance(day, int | np.integer) else str(day),
                    "occurrences": count,
                    "description": (
                        f"Frequent gaps on "
                        f"{day_names[int(day)] if isinstance(day, int | np.integer) else str(day)}"
                    ),
                })

        return patterns
