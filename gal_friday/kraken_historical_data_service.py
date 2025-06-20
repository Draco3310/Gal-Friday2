"""Kraken historical data service implementation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import logging
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import aiohttp
import asyncio
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import pandas_ta as ta

from gal_friday.data_ingestion.gap_detector import GapDetector
from gal_friday.interfaces.historical_data_service_interface import (
    HistoricalDataService,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from gal_friday.logger_service import LoggerService


class CircuitBreakerError(Exception):
    """Custom exception for CircuitBreaker errors."""


class RateLimitTracker:
    """Tracks and manages API rate limits."""

    def __init__(self, tier: str = "default", logger: LoggerService | None = None) -> None:
        """Initialize rate limit tracker with specified tier settings.

        Args:
        ----
            tier: API tier level determining rate limits (default, intermediate, pro)
            logger: Logger service for logging rate limit events
        """
        self.tier = tier
        self.logger = logger or logging.getLogger(__name__)

        # Initialize limit configuration based on tier (values from Kraken API docs)
        self.limits = {
            "default": {"rate": 1, "per_second": 1},  # 1 request per second
            "intermediate": {"rate": 2, "per_second": 1},  # 2 requests per second
            "pro": {"rate": 5, "per_second": 1},  # 5 requests per second
        }

        self.current_limit = self.limits.get(tier, self.limits["default"])
        self.last_request_time = datetime.now() - timedelta(seconds=10)  # Initial offset

    async def wait_if_needed(self) -> None:
        """Wait if needed to comply with rate limits."""
        now = datetime.now()
        min_interval = timedelta(seconds=1 / self.current_limit["rate"])
        time_since_last = now - self.last_request_time

        if time_since_last < min_interval:
            wait_time = (min_interval - time_since_last).total_seconds()
            if wait_time > 0:
                self.logger.debug("Rate limit: Waiting %.3fs before next request", wait_time)
                await asyncio.sleep(wait_time)

        self.last_request_time = datetime.now()


class CircuitBreaker:
    """Implements circuit breaker pattern for API calls."""

    OPEN_MESSAGE = "Circuit breaker is OPEN - API calls blocked"
    _RT = TypeVar("_RT")  # For the return type of the function being executed
    _P = ParamSpec("_P")  # For the parameters of the function being executed

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: int = 60,
        logger: LoggerService | None = None) -> None:
        """Initialize circuit breaker with specified thresholds.

        Args:
        ----
            failure_threshold: Number of consecutive failures before opening circuit
            reset_timeout: Time in seconds before attempting to close circuit after failure
            logger: Logger service for logging circuit state changes
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout  # seconds
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time: datetime | None = None
        self.logger = logger or logging.getLogger(__name__)

    async def execute(
        self,
        func: Callable[_P, Coroutine[Any, Any, _RT]],
        *args: _P.args,
        **kwargs: _P.kwargs) -> _RT:
        """Execute function with circuit breaker pattern."""
        if self.state == "OPEN":
            # Check if timeout has elapsed to transition to HALF-OPEN
            if (
                self.last_failure_time
                and (datetime.now() - self.last_failure_time).total_seconds() > self.reset_timeout
            ):
                self.state = "HALF-OPEN"
                self.logger.info("Circuit breaker state changed to HALF-OPEN")
            else:
                raise CircuitBreakerError(self.OPEN_MESSAGE)

        try:
            result = await func(*args, **kwargs)

            # Success - reset circuit breaker if in HALF-OPEN
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.logger.info("Circuit breaker state changed to CLOSED")

        except Exception:
            # Failure - increment counter and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == "HALF-OPEN" or self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(
                    "Circuit breaker state changed to OPEN after %s failures",
                    self.failure_count)

            raise  # TRY201: Re-raise the original exception to preserve its type and traceback
        else:
            return result  # TRY300: Moved return to else block


class KrakenHistoricalDataService(HistoricalDataService):
    """Kraken implementation of the HistoricalDataService interface."""

    def __init__(self, config: dict[str, Any], logger_service: LoggerService) -> None:
        """Initialize the Kraken historical data service."""
        self.config = config
        self.logger = logger_service
        self._source_module = self.__class__.__name__

        # InfluxDB configuration
        influxdb_config = self.config.get("influxdb", {})
        self.influxdb_client = InfluxDBClient(
            url=influxdb_config.get("url", "http://localhost:8086"),
            token=influxdb_config.get("token"),
            org=influxdb_config.get("org", "gal_friday"))
        self.write_api = self.influxdb_client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.influxdb_client.query_api()

        # API configuration
        self.api_base_url = self.config.get("kraken", {}).get("api_url", "https://api.kraken.com")

        # Measurement names for InfluxDB
        self.ohlcv_measurement = "market_data_ohlcv"
        self.trades_measurement = "market_data_trades"

        # Initialize circuit breaker and rate limiter
        kraken_config = self.config.get("kraken_api", {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=kraken_config.get("circuit_breaker_threshold", 3),
            reset_timeout=kraken_config.get("circuit_breaker_timeout", 60),
            logger=self.logger)

        tier = kraken_config.get("tier", "default")
        self.rate_limiter = RateLimitTracker(tier=tier, logger=self.logger)

        # API configuration
        self.api_key = self.config.get("kraken", {}).get("api_key")
        self.api_secret = self.config.get("kraken", {}).get("api_secret")

        # Initialize GapDetector for enhanced gap detection
        self.gap_detector = GapDetector(logger=self.logger)
        self.logger.info(
            "GapDetector initialized for enhanced data gap detection",
            source_module=self._source_module)

        # Cache for frequently accessed data
        self._ohlcv_cache: dict[Any, pd.DataFrame] = {}  # Simple cache for OHLCV data

        self.logger.info(
            "KrakenHistoricalDataService initialized",
            source_module=self._source_module)

    async def get_historical_ohlcv(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,  # e.g., "1m", "5m", "1h"
    ) -> pd.DataFrame | None:
        """Get historical OHLCV data for a given pair, time range, and interval.

        This method first checks InfluxDB for stored data, then fetches any missing
        data from the Kraken API.
        """
        self.logger.info(
            "Getting historical OHLCV for %s from %s to %s (interval: %s)",
            trading_pair,
            start_time,
            end_time,
            interval,
            source_module=self._source_module)

        # Check if data already exists in InfluxDB
        df = await self._query_ohlcv_data_from_influxdb(
            trading_pair,
            start_time,
            end_time,
            interval)

        # If complete data is available in InfluxDB, return it
        if df is not None and not df.empty:
            available_start = df.index.min()
            available_end = df.index.max()

            # If we have all the data needed
            if available_start <= start_time and available_end >= end_time:
                self.logger.info(
                    "Complete OHLCV data found in InfluxDB for %s",
                    trading_pair,
                    source_module=self._source_module)
                # Slice to exact range and return
                return df[(df.index >= start_time) & (df.index <= end_time)]

        # Determine what data is missing
        missing_ranges = self._get_missing_ranges(df, start_time, end_time, interval)
        if not missing_ranges:
            # Should have data but is empty
            self.logger.warning(
                "No missing ranges but data is empty for %s",
                trading_pair,
                source_module=self._source_module)
            return None

        # Fetch missing data from Kraken API
        complete_df = df if df is not None and not df.empty else None

        for range_start, range_end in missing_ranges:
            self.logger.info(
                "Fetching missing data for %s from %s to %s",
                trading_pair,
                range_start,
                range_end,
                source_module=self._source_module)

            # Fetch data from Kraken API
            new_data = await self._fetch_ohlcv_data(trading_pair, range_start, range_end, interval)

            if new_data is not None and not new_data.empty:
                # Validate the data
                if self._validate_ohlcv_data(new_data):
                    # Store data in InfluxDB
                    await self._store_ohlcv_data_in_influxdb(new_data, trading_pair, interval)

                    # Merge with existing data if needed
                    if complete_df is None:
                        complete_df = new_data
                    else:
                        complete_df = pd.concat([complete_df, new_data])
                        complete_df = complete_df[~complete_df.index.duplicated(keep="last")]
                        complete_df = complete_df.sort_index()
                else:
                    self.logger.warning(
                        "Validation failed for %s OHLCV data",
                        trading_pair,
                        source_module=self._source_module)
            else:
                self.logger.warning(
                    "No data returned from API for %s between %s and %s",
                    trading_pair,
                    range_start,
                    range_end,
                    source_module=self._source_module)

        # Final results
        if complete_df is not None and not complete_df.empty:
            # Slice to exact requested range
            result_df = complete_df[
                (complete_df.index >= start_time) & (complete_df.index <= end_time)
            ]
            self.logger.info(
                "Returning %s OHLCV rows for %s",
                len(result_df),
                trading_pair,
                source_module=self._source_module)
            return result_df
        self.logger.warning(
            "No OHLCV data available for %s",
            trading_pair,
            source_module=self._source_module)
        return None

    async def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime) -> pd.DataFrame | None:
        """Get historical trade data for a given pair and time range.

        This method first checks InfluxDB for stored data, then fetches any missing
        data from the Kraken API using the self.fetch_trades() method.
        """
        self.logger.info(
            "Getting historical trades for %s from %s to %s",
            trading_pair,
            start_time,
            end_time,
            source_module=self._source_module)

        # 1. Ensure times are timezone-aware (UTC)
        start_time_utc = start_time.replace(tzinfo=UTC) if start_time.tzinfo is None else start_time
        end_time_utc = end_time.replace(tzinfo=UTC) if end_time.tzinfo is None else end_time

        # 2. Attempt to fetch from InfluxDB first
        db_df = await self._query_trades_data_from_influxdb(trading_pair, start_time_utc, end_time_utc)

        # 3. Check if data from DB is complete for the requested range
        if db_df is not None and not db_df.empty:
            # Ensure DataFrame index is timezone-aware (UTC) for comparison
            if isinstance(db_df.index, pd.DatetimeIndex) and db_df.index.tz is None:
                db_df.index = db_df.index.tz_localize(UTC)

            available_start = db_df.index.min()
            available_end = db_df.index.max()

            if available_start <= start_time_utc and available_end >= end_time_utc:
                self.logger.info(
                    "Complete trade data found in InfluxDB for %s in range %s to %s.",
                    trading_pair, start_time_utc, end_time_utc,
                    source_module=self._source_module)
                # Slice to exact requested range and return
                return db_df[(db_df.index >= start_time_utc) & (db_df.index <= end_time_utc)]

        self.logger.info(
            "Trade data in InfluxDB for %s is incomplete or missing for range %s to %s. Will attempt API fetch.",
            trading_pair, start_time_utc, end_time_utc,
            source_module=self._source_module)

        # 4. Fetch missing data from Kraken API
        self.logger.info(
            "Fetching historical trades from Kraken API for %s from %s to %s",
            trading_pair, start_time_utc, end_time_utc,
            source_module=self._source_module)

        # Call the existing self.fetch_trades method to get data from Kraken API
        api_trades_list = await self.fetch_trades(
            trading_pair=trading_pair,
            since=start_time_utc,
            until=end_time_utc,
            limit=None,  # Fetch all available in the range
        )

        api_df = None
        if api_trades_list:
            try:
                api_df = pd.DataFrame(api_trades_list)
                if not api_df.empty:
                    api_df["timestamp"] = pd.to_datetime(api_df["timestamp"], utc=True)
                    api_df = api_df.set_index("timestamp")
                    # Sort by timestamp as API might not guarantee order with pagination
                    api_df = api_df.sort_index()

                    self.logger.info(
                        "Fetched %s trades from API for %s.", len(api_df), trading_pair,
                        source_module=self._source_module)

                    # 5. Store fetched data into InfluxDB
                    await self._store_trades_data_in_influxdb(api_df, trading_pair)
                else:
                    api_df = None
            except Exception as e:
                self.logger.error(
                    "Failed to process trades from API into DataFrame for %s: %s",
                    trading_pair, e, exc_info=True, source_module=self._source_module)
                api_df = None
        else:
            self.logger.info(
                "No new trades fetched from API for %s in range %s to %s.",
                trading_pair, start_time_utc, end_time_utc,
                source_module=self._source_module)

        # 6. Combine with existing InfluxDB data (if any)
        combined_df = None
        if db_df is not None and not db_df.empty and api_df is not None and not api_df.empty:
            combined_df = pd.concat([db_df, api_df])
            # Remove duplicates, keeping the first occurrence
            combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
            combined_df = combined_df.sort_index()
        elif api_df is not None and not api_df.empty:
            combined_df = api_df
        elif db_df is not None and not db_df.empty:
            combined_df = db_df

        if combined_df is not None and not combined_df.empty:
            # Ensure index is timezone-aware UTC before slicing
            if isinstance(combined_df.index, pd.DatetimeIndex) and combined_df.index.tz is None:
                combined_df.index = combined_df.index.tz_localize(UTC)
            # Slice to the exact requested range
            final_df = combined_df[(combined_df.index >= start_time_utc) & (combined_df.index <= end_time_utc)]
            if not final_df.empty:
                self.logger.info(
                    "Returning %s combined trades for %s.", len(final_df), trading_pair,
                    source_module=self._source_module)
                return final_df
            self.logger.info(
                "Combined trades for %s resulted in an empty DataFrame for the requested range.",
                trading_pair, source_module=self._source_module)
            return None
        self.logger.warning(
            "No trade data available for %s after DB query and API fetch for range %s to %s.",
            trading_pair, start_time_utc, end_time_utc,
            source_module=self._source_module)
        return None

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> pd.Series[Any] | None:
        """Get the next available OHLCV bar after the given timestamp."""
        # Convert to synchronous query for this method
        query = f"""
        from(bucket: "{self.config.get("influxdb", {}).get("bucket", "gal_friday")}")
          |> range(start: {timestamp.isoformat()}, stop: {
            (timestamp + timedelta(days=7)).isoformat()
        })
          |> filter(fn: (r) => r["_measurement"] == "{self.ohlcv_measurement}")
          |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc: false)
          |> limit(n:1)
        """

        try:
            tables = self.query_api.query(query)
            if not tables:
                self.logger.debug(
                    "No next bar found for %s after %s",
                    trading_pair,
                    timestamp,
                    source_module=self._source_module)
                return None

            # Convert to pandas Series[Any]
            for table in tables:
                for record in table.records:
                    return pd.Series(
                        {
                            "open": float(record.values.get("open", 0)),
                            "high": float(record.values.get("high", 0)),
                            "low": float(record.values.get("low", 0)),
                            "close": float(record.values.get("close", 0)),
                            "volume": float(record.values.get("volume", 0)),
                        },
                        name=record.values.get("_time"))

        except Exception:
            self.logger.exception(
                "Error getting next bar for %s:",
                trading_pair,
                source_module=self._source_module)
            return None
        else:
            return None

    def get_atr(
        self,
        trading_pair: str,
        timestamp: datetime,
        period: int = 14) -> Decimal | None:
        """Get the Average True Range indicator value at the given timestamp."""
        # Need to get enough bars before the timestamp for ATR calculation
        query = f"""
        from(bucket: "{self.config.get("influxdb", {}).get("bucket", "gal_friday")}")
          |> range(start: {(timestamp - timedelta(days=30)).isoformat()}, stop: {
            timestamp.isoformat()
        })
          |> filter(fn: (r) => r["_measurement"] == "{self.ohlcv_measurement}")
          |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> sort(columns: ["_time"], desc: false)
        """

        try:
            tables = self.query_api.query(query)
            if not tables:
                self.logger.debug(
                    "No data found for ATR calculation for %s",
                    trading_pair,
                    source_module=self._source_module)
                return None

            # Convert to pandas DataFrame for ATR calculation
            records = []
            for table in tables:
                for record in table.records:
                    records.append(
                        {
                            "timestamp": record.values.get("_time"),
                            "high": float(record.values.get("high", 0)),
                            "low": float(record.values.get("low", 0)),
                            "close": float(record.values.get("close", 0)),
                        })

            if not records:
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

            # Calculate ATR
            if len(df) >= period:
                atr = ta.atr(df["high"], df["low"], df["close"], length=period)
                if atr is not None and not atr.empty:
                    # Get ATR value at or closest before timestamp
                    atr = atr[atr.index <= timestamp]
                    if not atr.empty:
                        return Decimal(str(atr.iloc[-1]))

        except Exception:
            self.logger.exception(
                "Error calculating ATR for %s:",
                trading_pair,
                source_module=self._source_module)
            return None
        else:
            return None

    async def _fetch_ohlcv_data(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str) -> pd.DataFrame | None:
        """Fetch OHLCV data from Kraken API."""
        try:
            # Wait for rate limit if needed
            await self.rate_limiter.wait_if_needed()

            # Execute API call with circuit breaker and return the result
            return await self.circuit_breaker.execute(
                self._fetch_ohlcv_data_from_api,
                trading_pair,
                start_time,
                end_time,
                interval)

        except Exception:
            self.logger.exception(
                "Error fetching OHLCV data:",
                source_module=self._source_module)
            return None

    async def _fetch_ohlcv_data_from_api(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str) -> pd.DataFrame | None:
        """Fetch OHLCV data directly from Kraken API for the given range and interval."""
        self.logger.info(
            "Fetching OHLCV from API for %s: %s to %s, interval %s",
            trading_pair, start_time, end_time, interval,
            source_module=self._source_module)

        kraken_pair_name = self._get_kraken_pair_name(trading_pair)
        if not kraken_pair_name:
            self.logger.error(
                "Could not get Kraken pair name for %s", trading_pair,
                source_module=self._source_module)
            return None

        kraken_interval_code = self._map_interval_to_kraken_code(interval)
        if kraken_interval_code is None:
            self.logger.error(
                "Unsupported interval string for Kraken API: %s", interval,
                source_module=self._source_module)
            return None

        all_ohlcv_data = []
        current_since_timestamp = int(start_time.timestamp()) # Kraken API uses Unix timestamp for 'since'

        # Kraken returns a max of 720 data points per call for OHLC.
        # We need to paginate if the requested range is larger.
        MAX_API_CALLS = self.config.get("kraken_api", {}).get("ohlcv_max_pagination_calls", 20) # Safety break
        api_calls_count = 0

        while api_calls_count < MAX_API_CALLS:
            api_calls_count += 1
            params = {
                "pair": kraken_pair_name,
                "interval": kraken_interval_code,
                "since": current_since_timestamp,
            }

            self.logger.debug(
                "Kraken API OHLC request for %s: params=%s", trading_pair, params,
                source_module=self._source_module)

            response_data = await self._make_public_request("/0/public/OHLC", params)

            if not response_data or response_data.get("error"):
                error_messages = (
                    response_data.get("error", ["Unknown API error"]) if response_data
                    else ["No response from API"]
                )
                self.logger.error(
                    "Kraken API error fetching OHLCV for %s (call %s): %s. Params: %s",
                    trading_pair, api_calls_count, error_messages, params,
                    source_module=self._source_module)
                break # Exit pagination loop on error

            result = response_data.get("result", {})
            pair_data = result.get(kraken_pair_name) # Kraken nests data under the pair key

            if not pair_data:
                self.logger.info(
                    "No more OHLCV data returned from Kraken API for %s (call %s, since: %s).",
                    trading_pair, api_calls_count, current_since_timestamp,
                    source_module=self._source_module)
                break # No more data for this pair

            # Process the candle data
            # Each item in pair_data is: [<time>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <count>]
            data_beyond_end = False
            for candle in pair_data:
                try:
                    timestamp_unix = int(candle[0])
                    dt_object = datetime.fromtimestamp(timestamp_unix, tz=UTC)

                    # Stop if we've fetched data beyond the requested end_time
                    if dt_object > end_time:
                        self.logger.debug(
                            "Fetched OHLCV data beyond requested end_time for %s. Stopping pagination.",
                            trading_pair,
                        )
                        data_beyond_end = True
                        break

                    # Only add data within the original [start_time, end_time] inclusive range
                    if dt_object >= start_time:
                         all_ohlcv_data.append({
                            "timestamp": dt_object,
                            "open": Decimal(str(candle[1])),
                            "high": Decimal(str(candle[2])),
                            "low": Decimal(str(candle[3])),
                            "close": Decimal(str(candle[4])),
                            "volume": Decimal(str(candle[6])), # candle[5] is vwap, candle[7] is count
                        })
                except (IndexError, ValueError, TypeError) as e:
                    self.logger.warning(
                        "Error processing individual OHLCV candle data for %s: %s. Candle: %s",
                        trading_pair, e, candle, source_module=self._source_module)
                    continue # Skip this malformed candle

            if data_beyond_end:
                break

            # Update 'since' for the next iteration using the 'last' timestamp from the response
            last_timestamp_in_response = result.get("last")
            if last_timestamp_in_response is None:
                self.logger.warning(
                    "Kraken API OHLCV response for %s missing 'last' key for pagination. Stopping.",
                    trading_pair, source_module=self._source_module)
                break

            # If the 'last' timestamp is not advancing, it means no more new data or stuck.
            if int(last_timestamp_in_response) <= current_since_timestamp and len(pair_data) < 720:
                self.logger.info(
                    "Kraken API 'last' timestamp (%s) did not advance from 'since' (%s) for %s and "
                    "not a full page. Assuming end of data.",
                    last_timestamp_in_response, current_since_timestamp, trading_pair,
                    source_module=self._source_module)
                break

            current_since_timestamp = int(last_timestamp_in_response)

            # Small delay to respect potential implicit rate limits
            await asyncio.sleep(self.config.get("kraken_api", {}).get("ohlcv_pagination_delay_s", 0.2))

        if api_calls_count >= MAX_API_CALLS:
            self.logger.warning(
                "Reached max API calls (%s) for OHLCV pagination for %s. Data might be incomplete.",
                MAX_API_CALLS, trading_pair, source_module=self._source_module)

        if not all_ohlcv_data:
            self.logger.info(
                "No OHLCV data points collected from API for %s in the specified range.",
                trading_pair, source_module=self._source_module)
            return None

        # Create DataFrame
        try:
            df = pd.DataFrame(all_ohlcv_data)
            if df.empty:
                return None
            df = df.set_index("timestamp")
            df = df.sort_index() # Ensure chronological order
            # Remove potential duplicates that might arise from pagination logic
            df = df[~df.index.duplicated(keep="first")]

            # Convert Decimal columns to float for pandas compatibility
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            self.logger.info(
                "Successfully fetched and processed %s OHLCV data points from API for %s.",
                len(df), trading_pair, source_module=self._source_module)
        except Exception as e:
            self.logger.error(
                "Failed to create DataFrame from fetched OHLCV data for %s: %s",
                trading_pair, e, exc_info=True, source_module=self._source_module)
            return None
        else:
            return df

    def _map_interval_to_kraken_code(self, interval_str: str) -> int | None:
        """Maps human-readable interval string to Kraken API integer code."""
        # Kraken intervals: 1, 5, 15, 30, 60 (1h), 240 (4h), 1440 (1d), 10080 (1w), 21600 (15d)
        mapping = {
            "1m": 1, "1min": 1,
            "5m": 5, "5min": 5,
            "15m": 15, "15min": 15,
            "30m": 30, "30min": 30,
            "1h": 60, "60m": 60, "60min": 60,
            "4h": 240, "240m": 240, "240min": 240,
            "1d": 1440, "1day": 1440,
            "1w": 10080, "1week": 10080, "7d": 10080,
            "15d": 21600, "15day": 21600,
        }
        return mapping.get(interval_str.lower())

    def _interval_str_to_timedelta(self, interval_str: str) -> timedelta | None:
        """Convert an interval string (e.g., "1m", "1h", "1d") to a timedelta object.

        Args:
            interval_str: Human-readable interval string

        Returns:
            timedelta object or None if invalid format
        """
        if not interval_str or len(interval_str) < 2:
            self.logger.error(
                f"Invalid interval format: {interval_str}",
                source_module=self._source_module)
            return None

        unit = interval_str[-1].lower()
        try:
            value = int(interval_str[:-1])
        except ValueError:
            self.logger.exception(
                f"Invalid interval value in: {interval_str}",
                source_module=self._source_module)
            return None

        if unit == "s":
            return timedelta(seconds=value)
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)
        if unit == "w":
            return timedelta(weeks=value)
        self.logger.error(
            f"Unsupported interval unit: {unit} in {interval_str}",
            source_module=self._source_module)
        return None

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data for correctness and completeness."""
        if df is None or df.empty:
            self.logger.warning(
                "Empty DataFrame provided for validation",
                source_module=self._source_module)
            return False

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column not in df.columns:
                self.logger.warning(
                    "Missing required column: %s",
                    column,
                    source_module=self._source_module)
                return False

        # Check for NaN values
        if df[required_columns].isna().any().any():
            self.logger.warning(
                "NaN values found in OHLCV data",
                source_module=self._source_module)
            return False

        # Check for negative values
        if (df[required_columns] < 0).any().any():
            self.logger.warning(
                "Negative values found in OHLCV data",
                source_module=self._source_module)
            return False

        # Check OHLC relationship (high >= open,close >= low)
        valid_ohlc = (
            (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["open"] >= df["low"])
            & (df["close"] >= df["low"])
        )

        if not valid_ohlc.all():
            self.logger.warning(
                "Invalid OHLC relationship found",
                source_module=self._source_module)
            return False

        return True

    async def _store_ohlcv_data_in_influxdb(
        self,
        df: pd.DataFrame,
        trading_pair: str,
        interval: str) -> bool:
        """Store OHLCV data in InfluxDB."""
        if df is None or df.empty:
            return False

        try:
            bucket = self.config.get("influxdb", {}).get("bucket", "gal_friday")
            points = []

            # Convert DataFrame to InfluxDB points
            for timestamp, row in df.iterrows():
                point = (
                    Point(self.ohlcv_measurement)
                    .tag("trading_pair", trading_pair)
                    .tag("exchange", "kraken")
                    .tag("interval", interval)
                    .field("open", float(row["open"]))
                    .field("high", float(row["high"]))
                    .field("low", float(row["low"]))
                    .field("close", float(row["close"]))
                    .field("volume", float(row["volume"]))
                    .time(timestamp)
                )

                points.append(point)

            # Batch write points to InfluxDB
            self.write_api.write(bucket=bucket, record=points)

            self.logger.info(
                "Stored %s OHLCV points in InfluxDB for %s",
                len(points),
                trading_pair,
                source_module=self._source_module)

        except Exception:
            self.logger.exception(
                "Error storing OHLCV data in InfluxDB:",
                source_module=self._source_module)
            return False
        else:
            return True

    async def _store_trades_data_in_influxdb(self, df: pd.DataFrame, trading_pair: str) -> bool:
        """Store trade data in InfluxDB.

        Args:
            df: DataFrame with trades data containing price, volume, and side columns
            trading_pair: Trading pair identifier

        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty:
            return False

        # Ensure required columns exist
        required_cols = ["price", "volume", "side"]
        if not all(col in df.columns for col in required_cols):
            self.logger.error(
                f"DataFrame for InfluxDB trade storage for {trading_pair} is missing one of required columns: "
                f"{required_cols}. Columns present: {df.columns.tolist()}",
                source_module=self._source_module)
            return False

        try:
            bucket = self.config.get("influxdb", {}).get("bucket", "gal_friday")
            points = []

            for timestamp, row in df.iterrows():
                # Ensure timestamp is timezone-aware (UTC) before writing
                ts_to_write = timestamp
                if isinstance(timestamp, pd.Timestamp) and timestamp.tzinfo is None:
                    ts_to_write = timestamp.tz_localize(UTC)

                point = (
                    Point(self.trades_measurement)
                    .tag("trading_pair", trading_pair)
                    .tag("exchange", "kraken")
                    .field("price", float(row["price"]))
                    .field("volume", float(row["volume"]))
                    .field("side", str(row["side"]))
                    .time(ts_to_write)
                )

                # Add optional fields if present
                if "order_type" in row:
                    point = point.field("order_type", str(row.get("order_type", "")))
                if "misc" in row:
                    point = point.field("misc", str(row.get("misc", "")))

                points.append(point)

            if points:
                self.write_api.write(bucket=bucket, record=points)
                self.logger.info(
                    "Stored %s trade points in InfluxDB for %s",
                    len(points), trading_pair,
                    source_module=self._source_module)
        except Exception as e:
            self.logger.exception(
                "Error storing trade data in InfluxDB for %s: %s",
                trading_pair, e,
                source_module=self._source_module)
            return False
        else:
            if points:
                return True
            return False

    async def _query_ohlcv_data_from_influxdb(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str) -> pd.DataFrame | None:
        """Query OHLCV data from InfluxDB."""
        try:
            bucket = self.config.get("influxdb", {}).get("bucket", "gal_friday")

            # Build Flux query
            query = f"""
            from(bucket: "{bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r["_measurement"] == "{self.ohlcv_measurement}")
              |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
              |> filter(fn: (r) => r["interval"] == "{interval}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> sort(columns: ["_time"], desc: false)
            """

            tables = self.query_api.query(query)

            # Convert result to DataFrame
            records = []
            for table in tables:
                for record in table.records:
                    records.append(
                        {
                            "timestamp": record.values.get("_time"),
                            "open": float(record.values.get("open", 0)),
                            "high": float(record.values.get("high", 0)),
                            "low": float(record.values.get("low", 0)),
                            "close": float(record.values.get("close", 0)),
                            "volume": float(record.values.get("volume", 0)),
                        })

            if not records:
                self.logger.debug(
                    "No OHLCV data found in InfluxDB for %s",
                    trading_pair,
                    source_module=self._source_module)
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

        except Exception:
            self.logger.exception(
                "Error querying OHLCV data from InfluxDB:",
                source_module=self._source_module)
            return None
        else:
            return df

    async def _query_trades_data_from_influxdb(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime) -> pd.DataFrame | None:
        """Query trade data from InfluxDB."""
        try:
            bucket = self.config.get("influxdb", {}).get("bucket", "gal_friday")

            # Build Flux query
            query = f"""
            from(bucket: "{bucket}")
              |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
              |> filter(fn: (r) => r["_measurement"] == "{self.trades_measurement}")
              |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
              |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
              |> sort(columns: ["_time"], desc: false)
            """

            tables = self.query_api.query(query)

            # Convert result to DataFrame
            records = []
            for table in tables:
                for record in table.records:
                    records.append(
                        {
                            "timestamp": record.values.get("_time"),
                            "price": float(record.values.get("price", 0)),
                            "volume": float(record.values.get("volume", 0)),
                            "side": record.values.get("side", ""),
                        })

            if not records:
                self.logger.debug(
                    "No trade data found in InfluxDB for %s",
                    trading_pair,
                    source_module=self._source_module)
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

        except Exception:
            self.logger.exception(
                "Error querying trade data from InfluxDB:",
                source_module=self._source_module)
            return None
        else:
            return df

    def _get_missing_ranges(
        self,
        df: pd.DataFrame | None,
        start_time: datetime,
        end_time: datetime,
        expected_interval_str: str | None = None) -> list[tuple[datetime, datetime]]:
        """Determine what date ranges are missing from the data, including intra-range gaps.

        Args:
            df: DataFrame with time series data (index should be datetime)
            start_time: Start of the requested range
            end_time: End of the requested range
            expected_interval_str: Expected interval between data points (e.g., "1m", "5m")

        Returns:
            List of tuples (start, end) representing missing data ranges
        """
        self.logger.debug(
            "Getting missing ranges for data between %s and %s, expected interval: %s",
            start_time, end_time, expected_interval_str,
            source_module=self._source_module)

        # Ensure start_time and end_time are timezone-aware (UTC)
        start_time = start_time.replace(tzinfo=UTC) if start_time.tzinfo is None else start_time
        end_time = end_time.replace(tzinfo=UTC) if end_time.tzinfo is None else end_time

        if df is None or df.empty:
            self.logger.info(
                "DataFrame is empty or None. Entire range %s to %s is missing.",
                start_time, end_time,
                source_module=self._source_module)
            return [(start_time, end_time)]

        # Ensure DataFrame index is a DatetimeIndex and timezone-aware (UTC)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception as e:
                self.logger.error(
                    f"Failed to convert DataFrame index to DatetimeIndex: {e}",
                    exc_info=True,
                    source_module=self._source_module)
                return [(start_time, end_time)]
        # At this point we know df.index is a DatetimeIndex
        elif df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        elif df.index.tz != UTC:
            df.index = df.index.tz_convert(UTC)

        # Sort DataFrame by timestamp index
        df = df.sort_index()

        missing_ranges: list[tuple[datetime, datetime]] = []
        current_check_start = start_time

        # 1. Check if data starts after the requested start_time
        if not df.empty and df.index.min() > current_check_start:
            gap_end = df.index.min()
            if gap_end > current_check_start:
                missing_ranges.append((current_check_start, gap_end))
            current_check_start = gap_end

        # 2. Check for intra-range gaps using GapDetector (if available and interval provided)
        if self.gap_detector and expected_interval_str:
            expected_interval_td = self._interval_str_to_timedelta(expected_interval_str)
            if expected_interval_td:
                # GapDetector expects a 'timestamp' column
                df_for_gap_detection = df.reset_index()
                df_for_gap_detection.rename(columns={df_for_gap_detection.columns[0]: "timestamp"}, inplace=True)

                # Filter to the relevant range for accurate gap detection
                df_in_range = df_for_gap_detection[
                    (df_for_gap_detection["timestamp"] >= start_time) &
                    (df_for_gap_detection["timestamp"] <= end_time)
                ]

                if not df_in_range.empty:
                    detected_gaps = self.gap_detector.detect_gaps(
                        data=df_in_range,
                        timestamp_col="timestamp",
                        expected_interval=expected_interval_td)

                    # Convert DataGap objects to missing ranges
                    for gap_info in detected_gaps:
                        # The gap_info.start is the last data point before the gap
                        # The gap_info.end is the first data point after the gap
                        # The actual missing data is between these points
                        actual_gap_start = gap_info.start + expected_interval_td
                        actual_gap_end = gap_info.end

                        if actual_gap_end > actual_gap_start:
                            missing_ranges.append((
                                (
                                    actual_gap_start.to_pydatetime()
                                    if hasattr(actual_gap_start, "to_pydatetime")
                                    else actual_gap_start
                                ),
                                (
                                    actual_gap_end.to_pydatetime()
                                    if hasattr(actual_gap_end, "to_pydatetime")
                                    else actual_gap_end
                                ),
                            ))
                        else:
                            self.logger.debug(
                                f"Skipping zero or negative duration detected gap: {gap_info}",
                                source_module=self._source_module)
                else:
                    self.logger.debug(
                        "No data within the specified range for intra-range gap detection.",
                        source_module=self._source_module)
            else:
                self.logger.warning(
                    f"Could not determine timedelta for expected_interval_str: {expected_interval_str}. "
                    "Skipping intra-range gap detection.",
                    source_module=self._source_module)
        elif not self.gap_detector:
            self.logger.info(
                "GapDetector not initialized, skipping detailed intra-range gap check.",
                source_module=self._source_module)
        elif not expected_interval_str:
            self.logger.info(
                "Expected interval not provided, skipping detailed intra-range gap check.",
                source_module=self._source_module)

        # 3. Check if data ends before the requested end_time
        last_data_point_ts = df.index.max() if not df.empty else start_time

        if last_data_point_ts < end_time:
            # Calculate the actual missing start considering the expected interval
            actual_missing_start = last_data_point_ts
            if expected_interval_str:
                interval_delta = self._interval_str_to_timedelta(expected_interval_str)
                if interval_delta:
                    actual_missing_start = last_data_point_ts + interval_delta

            if end_time > actual_missing_start:
                missing_ranges.append((actual_missing_start, end_time))

        # 4. Post-process missing_ranges: Sort and merge overlapping/contiguous ranges
        if missing_ranges:
            # Sort by start time
            missing_ranges.sort(key=lambda x: x[0])

            # Merge overlapping or contiguous ranges
            merged_ranges = []
            current_start, current_end = missing_ranges[0]

            for i in range(1, len(missing_ranges)):
                next_start, next_end = missing_ranges[i]
                # If ranges overlap or are contiguous
                if next_start <= current_end:
                    current_end = max(current_end, next_end)
                else:
                    merged_ranges.append((current_start, current_end))
                    current_start, current_end = next_start, next_end

            merged_ranges.append((current_start, current_end))
            missing_ranges = merged_ranges

            self.logger.info(
                "Detected %s missing ranges: %s",
                len(missing_ranges),
                [(str(s), str(e)) for s, e in missing_ranges],
                source_module=self._source_module)
        else:
            self.logger.debug(
                "No missing ranges detected for %s to %s.",
                start_time, end_time,
                source_module=self._source_module)

        return missing_ranges

    async def _get_latest_timestamp_from_influxdb(
        self,
        trading_pair: str,
        interval: str) -> datetime | None:
        """Get the latest timestamp for a trading pair/interval in InfluxDB."""
        try:
            bucket = self.config.get("influxdb", {}).get("bucket", "gal_friday")

            # Build Flux query to get the latest timestamp
            query = f"""
            from(bucket: "{bucket}")
              |> range(start: -30d)
              |> filter(fn: (r) => r["_measurement"] == "{self.ohlcv_measurement}")
              |> filter(fn: (r) => r["trading_pair"] == "{trading_pair}")
              |> filter(fn: (r) => r["interval"] == "{interval}")
              |> group()
              |> sort(columns: ["_time"], desc: true)
              |> limit(n:1)
            """

            tables = self.query_api.query(query)

            # Extract timestamp from result
            for table in tables:
                for record in table.records:
                    # Ensure that the record time is a datetime object
                    record_time = record.get_time()
                    if isinstance(record_time, datetime):
                        return record_time
                    # Handle cases where _time might not be what's expected, or log an error
                    self.logger.warning(
                        "Unexpected type for record time: %s",
                        type(record_time),
                        source_module=self._source_module)
                    return None

        except Exception:
            self.logger.exception(
                "Error getting latest timestamp from InfluxDB:",
                source_module=self._source_module)
            return None
        else:
            return None

    async def fetch_trades(
        self,
        trading_pair: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None) -> list[dict[str, Any]] | None:
        """Fetch historical trade data from Kraken.

        Args:
            trading_pair: Trading pair (e.g., "XRP/USD")
            since: Start time (inclusive)
            until: End time (exclusive)
            limit: Maximum number of trades to fetch

        Returns:
            List of trade dictionaries or None on error
        """
        try:
            kraken_pair = self._get_kraken_pair_name(trading_pair)
            if not kraken_pair:
                return None

            all_trades: list[dict[str, Any]] = [] # Added type hint
            last_id: str | None = None

            # Kraken returns max 1000 trades per request
            batch_size = min(limit or 1000, 1000)

            while True:
                # Prepare request parameters
                params = {
                    "pair": kraken_pair,
                    "count": batch_size,
                }

                if since and not last_id:
                    # Use timestamp for first request
                    params["since"] = str(int(since.timestamp() * 1_000_000_000))
                elif last_id:
                    # Use last trade ID for pagination
                    params["since"] = str(last_id)

                # Make API request
                endpoint = "/0/public/Trades"
                result = await self._make_public_request(endpoint, params)

                if not result or result.get("error"):
                    self.logger.error(
                        f"Error fetching trades: {result.get('error') if result else 'No response'}",
                        source_module=self._source_module)
                    break

                trades_data = result.get("result", {}).get(kraken_pair, [])
                if not trades_data:
                    break

                # Process trades
                for trade in trades_data:
                    trade_time = datetime.fromtimestamp(float(trade[2]), tz=UTC)

                    # Check time bounds
                    if until and trade_time >= until:
                        return all_trades

                    trade_dict = {
                        "timestamp": trade_time,
                        "price": Decimal(trade[0]),
                        "volume": Decimal(trade[1]),
                        "side": "buy" if trade[3] == "b" else "sell",
                        "order_type": "market" if trade[4] == "m" else "limit",
                        "misc": trade[5] if len(trade) > 5 else "",
                    }

                    all_trades.append(trade_dict)

                # Update last ID for pagination
                last_id = result.get("result", {}).get("last")

                # Check if we've fetched enough
                if limit and len(all_trades) >= limit:
                    return all_trades[:limit]

                # Check if there are more trades
                if len(trades_data) < batch_size:
                    break

                # Rate limiting
                await asyncio.sleep(0.5)

            self.logger.info(
                f"Fetched {len(all_trades)} trades for {trading_pair}",
                source_module=self._source_module,
                context={
                    "since": since.isoformat() if since else None,
                    "until": until.isoformat() if until else None,
                })

        except Exception:
            self.logger.exception(
                f"Failed to fetch trades for {trading_pair}",
                source_module=self._source_module)
            return None
        else:
            return all_trades

    async def _make_public_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Make a public API request to Kraken.

        Args:
            endpoint: API endpoint
            params: Request parameters

        Returns:
            API response or None on error
        """
        try:
            url = f"{self.api_base_url}{endpoint}"

            async with aiohttp.ClientSession() as session, session.get(url, params=params) as response:
                if response.status != 200:
                    self.logger.error(
                        f"API request failed with status {response.status}",
                        source_module=self._source_module,
                        context={"endpoint": endpoint})
                    return None

                return cast("dict[str, Any]", await response.json())

        except Exception:
            self.logger.exception(
                f"Error making API request to {endpoint}",
                source_module=self._source_module)
            return None

    def _get_kraken_pair_name(self, trading_pair: str) -> str | None:
        """Convert internal pair format to Kraken format.

        Args:
            trading_pair: Internal format (e.g., "XRP/USD")

        Returns:
            Kraken format (e.g., "XXRPZUSD") or None if not found
        """
        # Common mappings
        mappings = {
            "BTC": "XBT",
            "XRP": "XXRP",
            "DOGE": "XDOGE",
            "USD": "ZUSD",
        }

        # Split pair
        if "/" in trading_pair:
            base, quote = trading_pair.split("/")

            # Apply mappings
            kraken_base = mappings.get(base, base)
            kraken_quote = mappings.get(quote, quote)

            # Try different formats
            formats = [
                f"{kraken_base}{kraken_quote}",
                f"{base}{quote}",
                f"{kraken_base}/{kraken_quote}",
                f"{base}/{quote}",
            ]

            # Return the first valid format
            # In a real implementation, we'd check against known pairs
            return formats[0]

        return None
