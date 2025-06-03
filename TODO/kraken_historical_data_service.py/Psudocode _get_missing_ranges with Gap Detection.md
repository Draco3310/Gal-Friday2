# File: gal_friday/kraken_historical_data_service.py
# Method: _get_missing_ranges
# TODO: Line 818 - Check for gaps within the data range

# --- Dependencies/Collaborators ---
# - pandas as pd
# - datetime, timedelta from datetime
# - GapDetector from .data_ingestion.gap_detector (adjust import path as necessary)
# - LoggerService
# - ConfigManager (to get expected interval if not passed directly)

# ... (preceding code in KrakenHistoricalDataService class) ...

    def __init__(self, config: dict[str, Any], logger_service: LoggerService) -> None:
        """Initialize the Kraken historical data service."""
        self.config = config
        self.logger = logger_service
        self._source_module = self.__class__.__name__
        # ... (other initializations) ...
        
        # Initialize GapDetector (assuming it's part of data_ingestion or a utility)
        # The GapDetector itself might need a logger.
        # We might instantiate it here or pass it in if it's a shared service.
        # For this pseudocode, let's assume we instantiate it when needed or have it as a member.
        if GapDetector is not None: # Check if GapDetector class is available
            self.gap_detector = GapDetector(logger=self.logger) 
        else:
            self.gap_detector = None
            self.logger.warning("GapDetector class not available. Intra-range gap detection will be skipped.")

        # ... (rest of __init__) ...

    def _interval_str_to_timedelta(self, interval_str: str) -> timedelta | None:
        """Converts an interval string (e.g., "1m", "1h", "1d") to a timedelta object."""
        # This is a helper that might already exist or need to be created.
        # It's similar to _map_interval_to_kraken_code but returns timedelta.
        # Example:
        unit = interval_str[-1].lower()
        value = 0
        try:
            value = int(interval_str[:-1])
        except ValueError:
            self.logger.error(f"Invalid interval format: {interval_str}", source_module=self._source_module)
            return None

        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        # Add other units like 's' for seconds, 'w' for weeks if needed
        else:
            self.logger.error(f"Unsupported interval unit: {unit} in {interval_str}", source_module=self._source_module)
            return None

    def _get_missing_ranges(
        self,
        df: pd.DataFrame | None,
        start_time: datetime, # Expected to be timezone-aware (UTC)
        end_time: datetime,   # Expected to be timezone-aware (UTC)
        expected_interval_str: str | None = None, # e.g., "1m", "5m", "1h". Needed for gap detection.
    ) -> list[tuple[datetime, datetime]]:
        """Determine what date ranges are missing from the data, including intra-range gaps."""
        self.logger.debug(
            "Getting missing ranges for data between %s and %s, expected interval: %s",
            start_time, end_time, expected_interval_str,
            source_module=self._source_module
        )

        # Ensure start_time and end_time are timezone-aware (UTC)
        start_time = start_time.replace(tzinfo=UTC) if start_time.tzinfo is None else start_time
        end_time = end_time.replace(tzinfo=UTC) if end_time.tzinfo is None else end_time

        if df is None or df.empty:
            self.logger.info("DataFrame is empty or None. Entire range %s to %s is missing.", start_time, end_time, source_module=self._source_module)
            return [(start_time, end_time)]

        # Ensure DataFrame index is a DatetimeIndex and timezone-aware (UTC) for reliable comparison
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception as e:
                self.logger.error(f"Failed to convert DataFrame index to DatetimeIndex: {e}", exc_info=True, source_module=self._source_module)
                return [(start_time, end_time)] # Fallback: assume entire range is missing if index is problematic
        elif df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        elif df.index.tz.utcoffset(None) != UTC.utcoffset(None): # Check if it's UTC
             df.index = df.index.tz_convert(UTC)


        # Sort DataFrame by timestamp index to ensure correct gap detection
        df = df.sort_index()

        missing_ranges: list[tuple[datetime, datetime]] = []
        current_check_start = start_time

        # 1. Check if data starts after the requested overall start_time
        if not df.empty and df.index.min() > current_check_start:
            gap_end = df.index.min()
            # Ensure we don't create a negative or zero duration range
            if gap_end > current_check_start:
                 missing_ranges.append((current_check_start, gap_end))
            current_check_start = gap_end # Move the start of our check to the beginning of actual data
        elif df.empty : # If df became empty after some processing or was initially empty
             missing_ranges.append((start_time, end_time))
             return missing_ranges


        # 2. Check for intra-range gaps using GapDetector (if available and interval provided)
        if self.gap_detector and expected_interval_str:
            expected_interval_td = self._interval_str_to_timedelta(expected_interval_str)
            if expected_interval_td:
                # GapDetector expects a 'timestamp' column. If df.index is the timestamp:
                df_for_gap_detection = df.reset_index() # Creates 'timestamp' column from index
                
                # Ensure the column name matches what GapDetector expects, or pass it.
                # Assuming GapDetector uses 'timestamp' by default for its timestamp_col.
                # The df_for_gap_detection needs to be sliced to the relevant overall range for accurate gap detection within bounds.
                df_in_range = df_for_gap_detection[
                    (df_for_gap_detection['timestamp'] >= start_time) & 
                    (df_for_gap_detection['timestamp'] <= end_time)
                ]

                if not df_in_range.empty:
                    detected_gaps = self.gap_detector.detect_gaps(
                        data=df_in_range, # Pass the DataFrame itself
                        timestamp_col='timestamp', # Name of the timestamp column
                        expected_interval=expected_interval_td
                    )
                    
                    # The GapDetector returns DataGap(start, end, duration, expected_points, actual_points, severity)
                    # We need to convert these into (start_of_gap, end_of_gap) tuples for missing_ranges.
                    # The `DataGap.start` is the last known data point BEFORE the gap.
                    # The `DataGap.end` is the first known data point AFTER the gap.
                    # So, the actual missing range is (DataGap.start + expected_interval, DataGap.end)
                    for gap_info in detected_gaps:
                        # The gap_info.start is the timestamp of the last data point *before* the gap.
                        # The gap_info.end is the timestamp of the first data point *after* the gap.
                        # The actual missing data is between these two points.
                        # A common convention for missing_ranges is (last_good_ts + interval, first_good_ts_after_gap)
                        # Or, more simply, (gap_info.start, gap_info.end) represents the bounds *around* the missing data.
                        # Let's use (gap_info.start + expected_interval_td, gap_info.end) to define the actual missing candles.
                        
                        actual_gap_start = gap_info.start + expected_interval_td 
                        actual_gap_end = gap_info.end 

                        if actual_gap_end > actual_gap_start: # Ensure valid range
                             missing_ranges.append((actual_gap_start.to_pydatetime(), actual_gap_end.to_pydatetime()))
                        else:
                            self.logger.debug(f"Skipping zero or negative duration detected gap: {gap_info}", source_module=self._source_module)
                else:
                    self.logger.debug("No data within the specified range for intra-range gap detection.", source_module=self._source_module)
            else:
                self.logger.warning(
                    f"Could not determine timedelta for expected_interval_str: {expected_interval_str}. Skipping intra-range gap detection.",
                    source_module=self._source_module
                )
        elif not self.gap_detector:
             self.logger.info("GapDetector not initialized, skipping detailed intra-range gap check.", source_module=self._source_module)
        elif not expected_interval_str:
             self.logger.info("Expected interval not provided, skipping detailed intra-range gap check.", source_module=self._source_module)


        # 3. Check if data ends before the requested overall end_time
        # current_check_start for this part should be the end of the last known data point
        last_data_point_ts = df.index.max() if not df.empty else start_time # Fallback if df is empty

        if last_data_point_ts < end_time:
            # Ensure we don't create a negative or zero duration range
            if end_time > last_data_point_ts:
                 missing_ranges.append((last_data_point_ts, end_time)) # This might need +interval logic too
                                                                    # depending on how ranges are consumed.
                                                                    # For fetching, (last_data_point_ts, end_time) is okay.
                                                                    # If last_data_point_ts is the timestamp of the last candle,
                                                                    # the next data needed starts from last_data_point_ts + interval.
                                                                    # Let's adjust:
                 actual_missing_start_from_tail = last_data_point_ts
                 if expected_interval_str and self._interval_str_to_timedelta(expected_interval_str):
                     actual_missing_start_from_tail = last_data_point_ts + self._interval_str_to_timedelta(expected_interval_str)
                 
                 if end_time > actual_missing_start_from_tail:
                      missing_ranges.append((actual_missing_start_from_tail, end_time))


        # 4. Post-process missing_ranges: Sort, merge overlapping/contiguous ranges
        if missing_ranges:
            # Sort by start time
            missing_ranges.sort(key=lambda x: x[0])
            
            merged_ranges = []
            if not missing_ranges: return [] # Should not happen if we got here, but defensive

            current_start, current_end = missing_ranges[0]

            for i in range(1, len(missing_ranges)):
                next_start, next_end = missing_ranges[i]
                # If the next range starts before or exactly at the current_end (allowing for contiguous ranges)
                # and assuming expected_interval_td is available for precise contiguity check.
                # For simplicity here, if next_start <= current_end, they are overlapping or contiguous.
                if next_start <= current_end:
                    current_end = max(current_end, next_end) # Extend the current range
                else:
                    merged_ranges.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            
            merged_ranges.append((current_start, current_end)) # Add the last processed range
            missing_ranges = merged_ranges

            self.logger.info("Detected missing ranges: %s", missing_ranges, source_module=self._source_module)
        else:
            self.logger.debug("No missing ranges detected for %s to %s.", start_time, end_time, source_module=self._source_module)

        return missing_ranges

# ... (rest of KrakenHistoricalDataService class) ...
