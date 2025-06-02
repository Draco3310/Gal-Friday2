"""Gap detection for time series data."""

from datetime import datetime, timedelta
from typing import Any, NamedTuple
from typing import cast as typing_cast

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


class GapDetector:
    """Detects and analyzes gaps in time series data.

    Features:
    - Configurable gap thresholds
    - Multiple severity levels
    - Gap statistics and reporting
    - Automatic gap filling strategies
    """

    def __init__(self, logger: LoggerService) -> None:
        """Initialize gap detector.

        Args:
            logger: Logger service
        """
        self.logger = logger
        self._source_module = self.__class__.__name__

        # Gap thresholds (in expected intervals)
        self.minor_threshold = 2  # 2x expected interval
        self.major_threshold = 5  # 5x expected interval
        self.critical_threshold = 10  # 10x expected interval

    def detect_gaps(
        self,
        data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        expected_interval: timedelta | None = None,
    ) -> list[DataGap]:
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
                    expected_interval,
                )

                gap = DataGap(
                    start=gap_start,
                    end=gap_end,
                    duration=actual_interval,
                    expected_points=expected_points,
                    actual_points=actual_points,
                    severity=severity,
                )

                gaps.append(gap)

        if gaps:
            self.logger.warning(
                f"Detected {len(gaps)} gaps in time series data",
                source_module=self._source_module,
                context={
                    "minor": sum(1 for g in gaps if g.severity == "minor"),
                    "major": sum(1 for g in gaps if g.severity == "major"),
                    "critical": sum(1 for g in gaps if g.severity == "critical"),
                },
            )

        return gaps

    def analyze_gap_patterns(
        self,
        gaps: list[DataGap],
    ) -> dict[str, Any]:
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
        # F841: durations = [g.duration.total_seconds() for g in gaps] # Unused

        durations_seconds = [
            g.duration.total_seconds() for g in gaps
        ]  # ensure list is not empty before np.mean
        avg_duration_seconds = np.mean(durations_seconds) if durations_seconds else 0.0
        median_duration_seconds = np.median(durations_seconds) if durations_seconds else 0.0


        stats = {
            "total_gaps": len(gaps),
            "total_duration": total_duration,
            "average_duration": timedelta(
                seconds=float(avg_duration_seconds),
            ),  # Explicit float cast
            "max_duration": max((g.duration for g in gaps), default=timedelta()),
            "min_duration": min((g.duration for g in gaps), default=timedelta()),
            "median_duration": timedelta(seconds=float(median_duration_seconds)),
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
    ) -> pd.DataFrame:
        """Fill detected gaps in data.

        Args:
            data: Original DataFrame
            gaps: List of gaps to fill
            timestamp_col: Name of timestamp column
            method: Gap filling method ('interpolate', 'forward', 'zero')

        Returns:
            DataFrame with filled gaps
        """
        if not gaps:
            return data

        filled_data = data.copy()

        for gap in gaps:
            if gap.severity == "critical":
                # Don't fill critical gaps
                self.logger.warning(
                    f"Skipping critical gap from {gap.start} to {gap.end}",
                    source_module=self._source_module,
                )
                continue

            # Generate missing timestamps
            freq = self._detect_frequency(data[timestamp_col])
            missing_times = pd.date_range(
                start=gap.start + pd.Timedelta(freq),
                end=gap.end - pd.Timedelta(freq),
                freq=freq,
            )

            if len(missing_times) == 0:
                continue

            # Create rows for missing data
            missing_rows = pd.DataFrame({
                timestamp_col: missing_times,
            })

            # Set other columns based on method
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            if method == "interpolate":
                # Will interpolate after merge
                pass
            elif method == "forward":
                # Forward fill from last known value
                last_value = filled_data[
                    filled_data[timestamp_col] <= gap.start
                ].iloc[-1]
                for col in numeric_cols:
                    if col != timestamp_col:
                        missing_rows[col] = last_value[col]
            elif method == "zero":
                # Fill with zeros
                for col in numeric_cols:
                    if col != timestamp_col:
                        missing_rows[col] = 0

            # Merge missing rows
            filled_data = pd.concat([filled_data, missing_rows], ignore_index=True)

        # Sort and interpolate if needed
        filled_data = filled_data.sort_values(timestamp_col)

        if method == "interpolate":
            filled_data[numeric_cols] = filled_data[numeric_cols].interpolate(
                method="time",
            )

        self.logger.info(
            f"Filled {len(gaps)} gaps using {method} method",
            source_module=self._source_module,
        )

        return filled_data

    def _detect_interval(self, timestamps: pd.Series) -> timedelta:
        """Auto-detect the expected interval between timestamps."""
        if len(timestamps) < 2:
            return timedelta(minutes=1)  # Default for insufficient data

        intervals = timestamps.diff().dropna()

        if intervals.empty:
            self.logger.warning(
                (
                    "Could not determine interval from timestamps (empty after diff/dropna), "
                    "defaulting to 1 minute."
                ),
                source_module=self._source_module,
            )
            return timedelta(minutes=1)

        # Use mode (most common interval)
        mode_interval_series = intervals.mode()
        if not mode_interval_series.empty:
            # Convert pandas.Timedelta to datetime.timedelta
            py_delta = mode_interval_series.iloc[0].to_pytimedelta()
            return typing_cast("timedelta", py_delta) # Cast to timedelta

        # Fallback to median if mode is empty (e.g., all intervals are unique)
        median_val = intervals.median()
        if pd.isna(median_val):  # Check if median itself is NaT
            self.logger.warning(
                "Median interval calculation resulted in NaT, defaulting to 1 minute.",
                source_module=self._source_module,
            )
            return timedelta(minutes=1)

        py_delta = median_val.to_pytimedelta()
        return typing_cast("timedelta", py_delta) # Cast to timedelta

    def _detect_frequency(self, timestamps: pd.Series) -> str:
        """Detect pandas frequency string."""
        intervals = timestamps.diff().dropna()
        median_seconds = intervals.median().total_seconds()

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
        expected_interval: timedelta,
    ) -> str:
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
                    "day_name": day_names[day],
                    "occurrences": count,
                    "description": f"Frequent gaps on {day_names[day]}",
                })

        return patterns
