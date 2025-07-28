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


from dataclasses import dataclass


@dataclass
class TradesFetchState:
    """State for tracking trades fetch operation."""

    kraken_pair: str
    batch_size: int
    limit: int | None
    since: datetime | None
    until: datetime | None
    last_id: str | None = None


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
        self.last_request_time = datetime.now(UTC) - timedelta(seconds=10)  # Initial offset

    async def wait_if_needed(self) -> None:
        """Wait if needed to comply with rate limits."""
        now = datetime.now(UTC)
        min_interval = timedelta(seconds=1 / self.current_limit["rate"])
        time_since_last = now - self.last_request_time

        if time_since_last < min_interval:
            wait_time = (min_interval - time_since_last).total_seconds()
            if wait_time > 0:
                self.logger.debug("Rate limit: Waiting %.3fs before next request", wait_time)
                await asyncio.sleep(wait_time)

        self.last_request_time = datetime.now(UTC)


class CircuitBreaker:
    """Implements circuit breaker pattern for API calls."""

    OPEN_MESSAGE = "Circuit breaker is OPEN - API calls blocked"
    _RT = TypeVar("_RT")  # For the return type of the function being executed
    _P = ParamSpec("_P")  # For the parameters of the function being executed

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: int = 60,
        logger: LoggerService | None = None,
    ) -> None:
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

    async def execute(self, func: Callable[_P, Coroutine[Any, Any, _RT]], *args: _P.args, **kwargs: _P.kwargs) -> _RT:
        """Execute function with circuit breaker pattern."""
        if self.state == "OPEN":
            # Check if timeout has elapsed to transition to HALF-OPEN
            if (
                self.last_failure_time
                and (datetime.now(UTC) - self.last_failure_time).total_seconds() > self.reset_timeout
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
            self.last_failure_time = datetime.now(UTC)

            if self.state == "HALF-OPEN" or self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning("Circuit breaker state changed to OPEN after %s failures", self.failure_count)

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
            org=influxdb_config.get("org", "gal_friday"),
        )
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
            logger=self.logger,
        )

        tier = kraken_config.get("tier", "default")
        self.rate_limiter = RateLimitTracker(tier=tier, logger=self.logger)

        # API configuration
        self.api_key = self.config.get("kraken", {}).get("api_key")
        self.api_secret = self.config.get("kraken", {}).get("api_secret")

        # Initialize GapDetector for enhanced gap detection
        self.gap_detector = GapDetector(logger=self.logger)
        self.logger.info("GapDetector initialized for enhanced data gap detection", source_module=self._source_module)

        # Cache for frequently accessed data
        self._ohlcv_cache: dict[Any, pd.DataFrame] = {}  # Simple cache for OHLCV data

        self.logger.info("KrakenHistoricalDataService initialized", source_module=self._source_module)

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
            source_module=self._source_module,
        )

        # Check if data already exists in InfluxDB
        df = await self._query_ohlcv_data_from_influxdb(trading_pair, start_time, end_time, interval)

        # If complete data is available in InfluxDB, return it
        if df is not None and not df.empty:
            available_start = df.index.min()
            available_end = df.index.max()

            # If we have all the data needed
            if available_start <= start_time and available_end >= end_time:
                self.logger.info(
                    "Complete OHLCV data found in InfluxDB for %s",
                    trading_pair,
                    source_module=self._source_module,
                )
                # Slice to exact range and return
                return df[(df.index >= start_time) & (df.index <= end_time)]

        # Determine what data is missing
        missing_ranges = self._get_missing_ranges(df, start_time, end_time, interval)
        if not missing_ranges:
            # Should have data but is empty
            self.logger.warning(
                "No missing ranges but data is empty for %s",
                trading_pair,
                source_module=self._source_module,
            )
            return None

        # Fetch missing data from Kraken API
        complete_df = df if df is not None and not df.empty else None

        for range_start, range_end in missing_ranges:
            self.logger.info(
                "Fetching missing data for %s from %s to %s",
                trading_pair,
                range_start,
                range_end,
                source_module=self._source_module,
            )

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
                        source_module=self._source_module,
                    )
            else:
                self.logger.warning(
                    "No data returned from API for %s between %s and %s",
                    trading_pair,
                    range_start,
                    range_end,
                    source_module=self._source_module,
                )

        # Final results
        if complete_df is not None and not complete_df.empty:
            # Slice to exact requested range
            result_df = complete_df[(complete_df.index >= start_time) & (complete_df.index <= end_time)]
            self.logger.info(
                "Returning %s OHLCV rows for %s",
                len(result_df),
                trading_pair,
                source_module=self._source_module,
            )
            return result_df
        self.logger.warning("No OHLCV data available for %s", trading_pair, source_module=self._source_module)
        return None

    async def get_historical_trades(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Get historical trade data for a given pair and time range.

        This method first checks InfluxDB for stored data, then fetches any missing
        data from the Kraken API using the self.fetch_trades() method.
        """
        self.logger.info(
            "Getting historical trades for %s from %s to %s",
            trading_pair,
            start_time,
            end_time,
            source_module=self._source_module,
        )

        # Ensure times are timezone-aware (UTC)
        start_time_utc = self._ensure_utc_timezone(start_time)
        end_time_utc = self._ensure_utc_timezone(end_time)

        # Check database for existing data
        db_df = await self._get_database_trades(trading_pair, start_time_utc, end_time_utc)
        if db_df is not None:
            return db_df  # Complete data found

        # Fetch from API if needed
        api_df = await self._fetch_api_trades(trading_pair, start_time_utc, end_time_utc)

        # Combine and return data
        return await self._combine_and_filter_trades(
            db_df,
            api_df,
            trading_pair,
            start_time_utc,
            end_time_utc,
        )

    def _ensure_utc_timezone(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware (UTC)."""
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt

    async def _get_database_trades(
        self,
        trading_pair: str,
        start_time_utc: datetime,
        end_time_utc: datetime,
    ) -> pd.DataFrame | None:
        """Get trades from database and check if data is complete."""
        db_df = await self._query_trades_data_from_influxdb(trading_pair, start_time_utc, end_time_utc)

        if db_df is None or db_df.empty:
            return None

        # Ensure DataFrame index is timezone-aware
        db_df = self._ensure_dataframe_utc_index(db_df)

        # Check if data covers the requested range
        if self._is_data_complete(db_df, start_time_utc, end_time_utc):
            self.logger.info(
                "Complete trade data found in InfluxDB for %s in range %s to %s.",
                trading_pair,
                start_time_utc,
                end_time_utc,
                source_module=self._source_module,
            )
            # Return sliced data
            return db_df[(db_df.index >= start_time_utc) & (db_df.index <= end_time_utc)]

        self.logger.info(
            "Trade data in InfluxDB for %s is incomplete or missing for range %s to %s. Will attempt API fetch.",
            trading_pair,
            start_time_utc,
            end_time_utc,
            source_module=self._source_module,
        )
        return None

    def _ensure_dataframe_utc_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame index is timezone-aware (UTC)."""
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        return df

    def _is_data_complete(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
    ) -> bool:
        """Check if DataFrame covers the requested time range."""
        available_start = df.index.min()
        available_end = df.index.max()
        return bool(available_start <= start_time and available_end >= end_time)

    async def _fetch_api_trades(
        self,
        trading_pair: str,
        start_time_utc: datetime,
        end_time_utc: datetime,
    ) -> pd.DataFrame | None:
        """Fetch trades from Kraken API."""
        self.logger.info(
            "Fetching historical trades from Kraken API for %s from %s to %s",
            trading_pair,
            start_time_utc,
            end_time_utc,
            source_module=self._source_module,
        )

        api_trades_list = await self.fetch_trades(
            trading_pair=trading_pair,
            since=start_time_utc,
            until=end_time_utc,
            limit=None,
        )

        if not api_trades_list:
            self.logger.info(
                "No new trades fetched from API for %s in range %s to %s.",
                trading_pair,
                start_time_utc,
                end_time_utc,
                source_module=self._source_module,
            )
            return None

        return await self._process_api_trades(api_trades_list, trading_pair)

    async def _process_api_trades(
        self,
        api_trades_list: list[dict[str, Any]],
        trading_pair: str,
    ) -> pd.DataFrame | None:
        """Process API trades into DataFrame and store in database."""
        try:
            api_df = pd.DataFrame(api_trades_list)
            if api_df.empty:
                return None

            api_df["timestamp"] = pd.to_datetime(api_df["timestamp"], utc=True)
            api_df = api_df.set_index("timestamp")
            api_df = api_df.sort_index()

            self.logger.info(
                "Fetched %s trades from API for %s.",
                len(api_df),
                trading_pair,
                source_module=self._source_module,
            )

            # Store fetched data into InfluxDB
            await self._store_trades_data_in_influxdb(api_df, trading_pair)

        except Exception:
            self.logger.exception(
                "Failed to process trades from API into DataFrame for %s: %s",
                trading_pair,
                source_module=self._source_module,
            )
            return None
        else:
            return api_df

    async def _combine_and_filter_trades(
        self,
        db_df: pd.DataFrame | None,
        api_df: pd.DataFrame | None,
        trading_pair: str,
        start_time_utc: datetime,
        end_time_utc: datetime,
    ) -> pd.DataFrame | None:
        """Combine database and API data, then filter to requested range."""
        # Combine dataframes
        combined_df = self._combine_dataframes(db_df, api_df)

        if combined_df is None or combined_df.empty:
            self.logger.warning(
                "No trade data available for %s after DB query and API fetch for range %s to %s.",
                trading_pair,
                start_time_utc,
                end_time_utc,
                source_module=self._source_module,
            )
            return None

        # Ensure timezone and filter to range
        combined_df = self._ensure_dataframe_utc_index(combined_df)
        final_df = combined_df[(combined_df.index >= start_time_utc) & (combined_df.index <= end_time_utc)]

        if final_df.empty:
            self.logger.info(
                "Combined trades for %s resulted in an empty DataFrame for the requested range.",
                trading_pair,
                source_module=self._source_module,
            )
            return None

        self.logger.info(
            "Returning %s combined trades for %s.",
            len(final_df),
            trading_pair,
            source_module=self._source_module,
        )
        return final_df

    def _combine_dataframes(
        self,
        db_df: pd.DataFrame | None,
        api_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Combine database and API dataframes, removing duplicates."""
        if db_df is not None and not db_df.empty and api_df is not None and not api_df.empty:
            combined_df = pd.concat([db_df, api_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
            return combined_df.sort_index()
        if api_df is not None and not api_df.empty:
            return api_df
        if db_df is not None and not db_df.empty:
            return db_df
        return None

    def get_next_bar(self, trading_pair: str, timestamp: datetime) -> pd.Series[Any] | None:
        """Get the next available OHLCV bar after the given timestamp."""
        # Convert to synchronous query for this method
        query = f"""
        from(bucket: "{self.config.get("influxdb", {}).get("bucket", "gal_friday")}")
          |> range(start: {timestamp.isoformat()}, stop: {(timestamp + timedelta(days=7)).isoformat()})
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
                    source_module=self._source_module,
                )
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
                        name=record.values.get("_time"),
                    )

        except Exception:
            self.logger.exception("Error getting next bar for %s:", trading_pair, source_module=self._source_module)
            return None
        else:
            return None

    def get_atr(self, trading_pair: str, timestamp: datetime, period: int = 14) -> Decimal | None:
        """Get the Average True Range indicator value at the given timestamp."""
        # Need to get enough bars before the timestamp for ATR calculation
        query = f"""
        from(bucket: "{self.config.get("influxdb", {}).get("bucket", "gal_friday")}")
          |> range(start: {(timestamp - timedelta(days=30)).isoformat()}, stop: {timestamp.isoformat()})
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
                    source_module=self._source_module,
                )
                return None

            # Convert to pandas DataFrame for ATR calculation
            records = []
            for table in tables:
                records.extend(
                    [
                        {
                            "timestamp": record.values.get("_time"),
                            "high": float(record.values.get("high", 0)),
                            "low": float(record.values.get("low", 0)),
                            "close": float(record.values.get("close", 0)),
                        }
                        for record in table.records
                    ],
                )

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
            self.logger.exception("Error calculating ATR for %s:", trading_pair, source_module=self._source_module)
            return None
        else:
            return None

    async def _fetch_ohlcv_data(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
    ) -> pd.DataFrame | None:
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
                interval,
            )

        except Exception:
            self.logger.exception("Error fetching OHLCV data:", source_module=self._source_module)
            return None

    async def _fetch_ohlcv_data_from_api(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV data directly from Kraken API for the given range and interval."""
        self.logger.info(
            "Fetching OHLCV from API for %s: %s to %s, interval %s",
            trading_pair,
            start_time,
            end_time,
            interval,
            source_module=self._source_module,
        )

        # Validate and prepare API parameters
        kraken_pair_name = self._validate_kraken_pair(trading_pair)
        if not kraken_pair_name:
            return None

        kraken_interval_code = self._validate_kraken_interval(interval)
        if kraken_interval_code is None:
            return None

        # Fetch paginated data from API
        all_ohlcv_data = await self._fetch_paginated_ohlcv(
            trading_pair,
            kraken_pair_name,
            kraken_interval_code,
            start_time,
            end_time,
        )

        if not all_ohlcv_data:
            self.logger.info(
                "No OHLCV data points collected from API for %s in the specified range.",
                trading_pair,
                source_module=self._source_module,
            )
            return None

        # Convert collected data to DataFrame
        return self._create_ohlcv_dataframe(all_ohlcv_data, trading_pair)

    def _validate_kraken_pair(self, trading_pair: str) -> str | None:
        """Validate and get Kraken pair name."""
        kraken_pair_name = self._get_kraken_pair_name(trading_pair)
        if not kraken_pair_name:
            self.logger.error("Could not get Kraken pair name for %s", trading_pair, source_module=self._source_module)
        return kraken_pair_name

    def _validate_kraken_interval(self, interval: str) -> int | None:
        """Validate and map interval to Kraken code."""
        kraken_interval_code = self._map_interval_to_kraken_code(interval)
        if kraken_interval_code is None:
            self.logger.error(
                "Unsupported interval string for Kraken API: %s",
                interval,
                source_module=self._source_module,
            )
        return kraken_interval_code

    async def _fetch_paginated_ohlcv(
        self,
        trading_pair: str,
        kraken_pair_name: str,
        kraken_interval_code: int,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        """Fetch OHLCV data with pagination support."""
        all_ohlcv_data: list[dict[str, Any]] = []
        current_since_timestamp = int(start_time.timestamp())
        max_api_calls = self.config.get("kraken_api", {}).get("ohlcv_max_pagination_calls", 20)
        api_calls_count = 0

        while api_calls_count < max_api_calls:
            api_calls_count += 1

            # Make API request
            response_data = await self._make_ohlcv_api_request(
                trading_pair,
                kraken_pair_name,
                kraken_interval_code,
                current_since_timestamp,
                api_calls_count,
            )

            if not response_data:
                break

            # Process response
            result = response_data.get("result", {})
            pair_data = result.get(kraken_pair_name)

            if not pair_data:
                self._log_no_more_data(trading_pair, api_calls_count, current_since_timestamp)
                break

            # Process candles and check if we should continue
            data_beyond_end, candles_processed = self._process_ohlcv_candles(
                pair_data,
                start_time,
                end_time,
                trading_pair,
                all_ohlcv_data,
            )

            if data_beyond_end:
                break

            # Update pagination timestamp
            should_continue = await self._update_pagination_timestamp(
                result,
                current_since_timestamp,
                pair_data,
                trading_pair,
            )

            if not should_continue:
                break

            current_since_timestamp = int(result.get("last"))

            # Rate limit delay
            await asyncio.sleep(self.config.get("kraken_api", {}).get("ohlcv_pagination_delay_s", 0.2))

        if api_calls_count >= max_api_calls:
            self._log_max_api_calls_reached(max_api_calls, trading_pair)

        return all_ohlcv_data

    async def _make_ohlcv_api_request(
        self,
        trading_pair: str,
        kraken_pair_name: str,
        kraken_interval_code: int,
        current_since_timestamp: int,
        api_calls_count: int,
    ) -> dict[str, Any] | None:
        """Make a single OHLCV API request."""
        params = {
            "pair": kraken_pair_name,
            "interval": kraken_interval_code,
            "since": current_since_timestamp,
        }

        self.logger.debug(
            "Kraken API OHLC request for %s: params=%s",
            trading_pair,
            params,
            source_module=self._source_module,
        )

        response_data = await self._make_public_request("/0/public/OHLC", params)

        if not response_data or response_data.get("error"):
            self._log_api_error(response_data, trading_pair, api_calls_count, params)
            return None

        return response_data

    def _log_api_error(
        self,
        response_data: dict[str, Any] | None,
        trading_pair: str,
        api_calls_count: int,
        params: dict[str, Any],
    ) -> None:
        """Log API error details."""
        error_messages = (
            response_data.get("error", ["Unknown API error"]) if response_data else ["No response from API"]
        )
        self.logger.error(
            "Kraken API error fetching OHLCV for %s (call %s): %s. Params: %s",
            trading_pair,
            api_calls_count,
            error_messages,
            params,
            source_module=self._source_module,
        )

    def _log_no_more_data(self, trading_pair: str, api_calls_count: int, current_since_timestamp: int) -> None:
        """Log when no more data is available."""
        self.logger.info(
            "No more OHLCV data returned from Kraken API for %s (call %s, since: %s).",
            trading_pair,
            api_calls_count,
            current_since_timestamp,
            source_module=self._source_module,
        )

    def _process_ohlcv_candles(
        self,
        pair_data: list[Any],
        start_time: datetime,
        end_time: datetime,
        trading_pair: str,
        all_ohlcv_data: list[dict[str, Any]],
    ) -> tuple[bool, int]:
        """Process individual candles from API response."""
        data_beyond_end = False
        candles_processed = 0

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
                    all_ohlcv_data.append(self._parse_candle_data(candle, dt_object))
                    candles_processed += 1

            except (IndexError, ValueError, TypeError) as e:
                self._log_candle_parse_error(trading_pair, e, candle)
                continue

        return data_beyond_end, candles_processed

    def _parse_candle_data(self, candle: list[Any], dt_object: datetime) -> dict[str, Any]:
        """Parse a single candle into structured data."""
        # Each item in pair_data is: [<time>, <open>, <high>, <low>, <close>, <vwap>, <volume>, <count>]
        return {
            "timestamp": dt_object,
            "open": Decimal(str(candle[1])),
            "high": Decimal(str(candle[2])),
            "low": Decimal(str(candle[3])),
            "close": Decimal(str(candle[4])),
            "volume": Decimal(str(candle[6])),  # candle[5] is vwap, candle[7] is count
        }

    def _log_candle_parse_error(self, trading_pair: str, error: Exception, candle: Any) -> None:
        """Log error when parsing candle data."""
        self.logger.warning(
            "Error processing individual OHLCV candle data for %s: %s. Candle: %s",
            trading_pair,
            error,
            candle,
            source_module=self._source_module,
        )

    async def _update_pagination_timestamp(
        self,
        result: dict[str, Any],
        current_since_timestamp: int,
        pair_data: list[Any],
        trading_pair: str,
    ) -> bool:
        """Update pagination timestamp and determine if we should continue."""
        last_timestamp_in_response = result.get("last")
        if last_timestamp_in_response is None:
            self.logger.warning(
                "Kraken API OHLCV response for %s missing 'last' key for pagination. Stopping.",
                trading_pair,
                source_module=self._source_module,
            )
            return False

        # If the 'last' timestamp is not advancing, it means no more new data or stuck.
        if int(last_timestamp_in_response) <= current_since_timestamp and len(pair_data) < 720:
            self.logger.info(
                "Kraken API 'last' timestamp (%s) did not advance from 'since' (%s) for %s and "
                "not a full page. Assuming end of data.",
                last_timestamp_in_response,
                current_since_timestamp,
                trading_pair,
                source_module=self._source_module,
            )
            return False

        return True

    def _log_max_api_calls_reached(self, max_api_calls: int, trading_pair: str) -> None:
        """Log when max API calls limit is reached."""
        self.logger.warning(
            "Reached max API calls (%s) for OHLCV pagination for %s. Data might be incomplete.",
            max_api_calls,
            trading_pair,
            source_module=self._source_module,
        )

    def _create_ohlcv_dataframe(self, all_ohlcv_data: list[dict[str, Any]], trading_pair: str) -> pd.DataFrame | None:
        """Create DataFrame from collected OHLCV data."""
        try:
            df = pd.DataFrame(all_ohlcv_data)
            if df.empty:
                return None

            df = df.set_index("timestamp")
            df = df.sort_index()  # Ensure chronological order
            # Remove potential duplicates that might arise from pagination logic
            df = df[~df.index.duplicated(keep="first")]

            # Convert Decimal columns to float for pandas compatibility
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            self.logger.info(
                "Successfully fetched and processed %s OHLCV data points from API for %s.",
                len(df),
                trading_pair,
                source_module=self._source_module,
            )

        except Exception:
            self.logger.exception(
                "Failed to create DataFrame from fetched OHLCV data for %s: %s",
                trading_pair,
                source_module=self._source_module,
            )
            return None
        else:
            return df

    def _map_interval_to_kraken_code(self, interval_str: str) -> int | None:
        """Maps human-readable interval string to Kraken API integer code."""
        # Kraken intervals: 1, 5, 15, 30, 60 (1h), 240 (4h), 1440 (1d), 10080 (1w), 21600 (15d)
        mapping = {
            "1m": 1,
            "1min": 1,
            "5m": 5,
            "5min": 5,
            "15m": 15,
            "15min": 15,
            "30m": 30,
            "30min": 30,
            "1h": 60,
            "60m": 60,
            "60min": 60,
            "4h": 240,
            "240m": 240,
            "240min": 240,
            "1d": 1440,
            "1day": 1440,
            "1w": 10080,
            "1week": 10080,
            "7d": 10080,
            "15d": 21600,
            "15day": 21600,
        }
        return mapping.get(interval_str.lower())

    def _interval_str_to_timedelta(self, interval_str: str) -> timedelta | None:
        """Convert an interval string (e.g., "1m", "1h", "1d") to a timedelta object.

        Args:
            interval_str: Human-readable interval string

        Returns:
            timedelta object or None if invalid format
        """
        if not self._is_valid_interval_format(interval_str):
            return None

        value = self._extract_interval_value(interval_str)
        if value is None:
            return None

        unit = interval_str[-1].lower()
        return self._create_timedelta_for_unit(unit, value, interval_str)

    def _is_valid_interval_format(self, interval_str: str) -> bool:
        """Check if interval string has valid format."""
        if not interval_str or len(interval_str) < 2:
            self.logger.error(f"Invalid interval format: {interval_str}", source_module=self._source_module)
            return False
        return True

    def _extract_interval_value(self, interval_str: str) -> int | None:
        """Extract numeric value from interval string."""
        try:
            return int(interval_str[:-1])
        except ValueError:
            self.logger.exception(f"Invalid interval value in: {interval_str}", source_module=self._source_module)
            return None

    def _create_timedelta_for_unit(self, unit: str, value: int, interval_str: str) -> timedelta | None:
        """Create timedelta based on unit and value."""
        unit_mapping = {
            "s": lambda v: timedelta(seconds=v),
            "m": lambda v: timedelta(minutes=v),
            "h": lambda v: timedelta(hours=v),
            "d": lambda v: timedelta(days=v),
            "w": lambda v: timedelta(weeks=v),
        }

        if unit in unit_mapping:
            return unit_mapping[unit](value)

        self.logger.error(f"Unsupported interval unit: {unit} in {interval_str}", source_module=self._source_module)
        return None

    def _validate_ohlcv_data(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data for correctness and completeness."""
        if df is None or df.empty:
            self.logger.warning("Empty DataFrame provided for validation", source_module=self._source_module)
            return False

        # Check required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            if column not in df.columns:
                self.logger.warning("Missing required column: %s", column, source_module=self._source_module)
                return False

        # Check for NaN values
        if df[required_columns].isna().any().any():
            self.logger.warning("NaN values found in OHLCV data", source_module=self._source_module)
            return False

        # Check for negative values
        if (df[required_columns] < 0).any().any():
            self.logger.warning("Negative values found in OHLCV data", source_module=self._source_module)
            return False

        # Check OHLC relationship (high >= open,close >= low)
        valid_ohlc = (
            (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["open"] >= df["low"])
            & (df["close"] >= df["low"])
        )

        if not valid_ohlc.all():
            self.logger.warning("Invalid OHLC relationship found", source_module=self._source_module)
            return False

        return True

    async def _store_ohlcv_data_in_influxdb(self, df: pd.DataFrame, trading_pair: str, interval: str) -> bool:
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
                source_module=self._source_module,
            )

        except Exception:
            self.logger.exception("Error storing OHLCV data in InfluxDB:", source_module=self._source_module)
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
                source_module=self._source_module,
            )
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
                    len(points),
                    trading_pair,
                    source_module=self._source_module,
                )
        except Exception:
            self.logger.exception(
                "Error storing trade data in InfluxDB for %s: %s",
                trading_pair,
                source_module=self._source_module,
            )
            return False
        else:
            return bool(points)

    async def _query_ohlcv_data_from_influxdb(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval: str,
    ) -> pd.DataFrame | None:
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
                records.extend(
                    [
                        {
                            "timestamp": record.values.get("_time"),
                            "open": float(record.values.get("open", 0)),
                            "high": float(record.values.get("high", 0)),
                            "low": float(record.values.get("low", 0)),
                            "close": float(record.values.get("close", 0)),
                            "volume": float(record.values.get("volume", 0)),
                        }
                        for record in table.records
                    ],
                )

            if not records:
                self.logger.debug(
                    "No OHLCV data found in InfluxDB for %s",
                    trading_pair,
                    source_module=self._source_module,
                )
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

        except Exception:
            self.logger.exception("Error querying OHLCV data from InfluxDB:", source_module=self._source_module)
            return None
        else:
            return df

    async def _query_trades_data_from_influxdb(
        self,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
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
                records.extend(
                    [
                        {
                            "timestamp": record.values.get("_time"),
                            "price": float(record.values.get("price", 0)),
                            "volume": float(record.values.get("volume", 0)),
                            "side": record.values.get("side", ""),
                        }
                        for record in table.records
                    ],
                )

            if not records:
                self.logger.debug(
                    "No trade data found in InfluxDB for %s",
                    trading_pair,
                    source_module=self._source_module,
                )
                return None

            df = pd.DataFrame(records)
            df.set_index("timestamp", inplace=True)

        except Exception:
            self.logger.exception("Error querying trade data from InfluxDB:", source_module=self._source_module)
            return None
        else:
            return df

    def _get_missing_ranges(
        self,
        df: pd.DataFrame | None,
        start_time: datetime,
        end_time: datetime,
        expected_interval_str: str | None = None,
    ) -> list[tuple[datetime, datetime]]:
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
            start_time,
            end_time,
            expected_interval_str,
            source_module=self._source_module,
        )

        # Ensure start_time and end_time are timezone-aware (UTC)
        start_time = self._ensure_utc_timezone(start_time)
        end_time = self._ensure_utc_timezone(end_time)

        # Handle empty DataFrame case
        if df is None or df.empty:
            return self._handle_empty_dataframe(start_time, end_time)

        # Prepare DataFrame for analysis
        df = self._prepare_dataframe_for_gap_analysis(df, start_time, end_time)
        if df is None:
            return [(start_time, end_time)]

        # Find all missing ranges
        missing_ranges = self._find_all_missing_ranges(df, start_time, end_time, expected_interval_str)

        # Merge overlapping ranges and log results
        return self._merge_and_log_missing_ranges(missing_ranges, start_time, end_time)

    def _handle_empty_dataframe(self, start_time: datetime, end_time: datetime) -> list[tuple[datetime, datetime]]:
        """Handle case when DataFrame is empty or None."""
        self.logger.info(
            "DataFrame is empty or None. Entire range %s to %s is missing.",
            start_time,
            end_time,
            source_module=self._source_module,
        )
        return [(start_time, end_time)]

    def _prepare_dataframe_for_gap_analysis(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Prepare DataFrame by ensuring proper datetime index and timezone."""
        # Ensure DataFrame index is a DatetimeIndex and timezone-aware (UTC)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                self.logger.exception(
                    "Failed to convert DataFrame index to DatetimeIndex",
                    source_module=self._source_module,
                )
                return None
        # At this point we know df.index is a DatetimeIndex
        elif df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        elif df.index.tz != UTC:
            df.index = df.index.tz_convert(UTC)

        # Sort DataFrame by timestamp index
        return df.sort_index()

    def _find_all_missing_ranges(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        expected_interval_str: str | None,
    ) -> list[tuple[datetime, datetime]]:
        """Find all missing ranges including start gaps, intra-range gaps, and end gaps."""
        missing_ranges: list[tuple[datetime, datetime]] = []

        # Check for gap at start
        start_gap = self._check_start_gap(df, start_time)
        if start_gap:
            missing_ranges.append(start_gap)

        # Check for intra-range gaps
        intra_gaps = self._check_intra_range_gaps(df, start_time, end_time, expected_interval_str)
        missing_ranges.extend(intra_gaps)

        # Check for gap at end
        end_gap = self._check_end_gap(df, end_time, expected_interval_str)
        if end_gap:
            missing_ranges.append(end_gap)

        return missing_ranges

    def _check_start_gap(self, df: pd.DataFrame, start_time: datetime) -> tuple[datetime, datetime] | None:
        """Check if data starts after the requested start_time."""
        if not df.empty and df.index.min() > start_time:
            gap_end = df.index.min()
            if gap_end > start_time:
                return (start_time, gap_end)
        return None

    def _check_intra_range_gaps(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        expected_interval_str: str | None,
    ) -> list[tuple[datetime, datetime]]:
        """Check for gaps within the data range using GapDetector."""
        if not self.gap_detector or not expected_interval_str:
            self._log_gap_detection_skip_reason(expected_interval_str)
            return []

        expected_interval_td = self._interval_str_to_timedelta(expected_interval_str)
        if not expected_interval_td:
            self.logger.warning(
                f"Could not determine timedelta for expected_interval_str: {expected_interval_str}. "
                "Skipping intra-range gap detection.",
                source_module=self._source_module,
            )
            return []

        return self._detect_gaps_with_gap_detector(df, start_time, end_time, expected_interval_td)

    def _log_gap_detection_skip_reason(self, expected_interval_str: str | None) -> None:
        """Log reason for skipping gap detection."""
        if not self.gap_detector:
            self.logger.info(
                "GapDetector not initialized, skipping detailed intra-range gap check.",
                source_module=self._source_module,
            )
        elif not expected_interval_str:
            self.logger.info(
                "Expected interval not provided, skipping detailed intra-range gap check.",
                source_module=self._source_module,
            )

    def _detect_gaps_with_gap_detector(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        expected_interval_td: timedelta,
    ) -> list[tuple[datetime, datetime]]:
        """Use GapDetector to find intra-range gaps."""
        # GapDetector expects a 'timestamp' column
        df_for_gap_detection = df.reset_index()
        df_for_gap_detection.rename(columns={df_for_gap_detection.columns[0]: "timestamp"}, inplace=True)

        # Filter to the relevant range for accurate gap detection
        df_in_range = df_for_gap_detection[
            (df_for_gap_detection["timestamp"] >= start_time) & (df_for_gap_detection["timestamp"] <= end_time)
        ]

        if df_in_range.empty:
            self.logger.debug(
                "No data within the specified range for intra-range gap detection.",
                source_module=self._source_module,
            )
            return []

        detected_gaps = self.gap_detector.detect_gaps(
            data=df_in_range,
            timestamp_col="timestamp",
            expected_interval=expected_interval_td,
        )

        # Convert DataGap objects to missing ranges
        return self._convert_detected_gaps_to_ranges(detected_gaps, expected_interval_td)

    def _convert_detected_gaps_to_ranges(
        self,
        detected_gaps: list[Any],
        expected_interval_td: timedelta,
    ) -> list[tuple[datetime, datetime]]:
        """Convert DataGap objects to missing ranges."""
        missing_ranges = []

        for gap_info in detected_gaps:
            # The gap_info.start is the last data point before the gap
            # The gap_info.end is the first data point after the gap
            actual_gap_start = gap_info.start + expected_interval_td
            actual_gap_end = gap_info.end

            if actual_gap_end > actual_gap_start:
                missing_ranges.append(
                    (
                        self._to_datetime(actual_gap_start),
                        self._to_datetime(actual_gap_end),
                    ),
                )
            else:
                self.logger.debug(
                    f"Skipping zero or negative duration detected gap: {gap_info}",
                    source_module=self._source_module,
                )

        return missing_ranges

    def _to_datetime(self, dt_like: Any) -> datetime:
        """Convert datetime-like object to datetime."""
        result = dt_like.to_pydatetime() if hasattr(dt_like, "to_pydatetime") else dt_like
        return cast("datetime", result)

    def _check_end_gap(
        self,
        df: pd.DataFrame,
        end_time: datetime,
        expected_interval_str: str | None,
    ) -> tuple[datetime, datetime] | None:
        """Check if data ends before the requested end_time."""
        last_data_point_ts = df.index.max() if not df.empty else end_time

        if last_data_point_ts < end_time:
            # Calculate the actual missing start considering the expected interval
            actual_missing_start = last_data_point_ts
            if expected_interval_str:
                interval_delta = self._interval_str_to_timedelta(expected_interval_str)
                if interval_delta:
                    actual_missing_start = last_data_point_ts + interval_delta

            if end_time > actual_missing_start:
                return (actual_missing_start, end_time)

        return None

    def _merge_and_log_missing_ranges(
        self,
        missing_ranges: list[tuple[datetime, datetime]],
        start_time: datetime,
        end_time: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Sort, merge overlapping ranges, and log results."""
        if not missing_ranges:
            self.logger.debug(
                "No missing ranges detected for %s to %s.",
                start_time,
                end_time,
                source_module=self._source_module,
            )
            return missing_ranges

        # Sort by start time
        missing_ranges.sort(key=lambda x: x[0])

        # Merge overlapping or contiguous ranges
        merged_ranges = self._merge_overlapping_ranges(missing_ranges)

        self.logger.info(
            "Detected %s missing ranges: %s",
            len(merged_ranges),
            [(str(s), str(e)) for s, e in merged_ranges],
            source_module=self._source_module,
        )

        return merged_ranges

    def _merge_overlapping_ranges(
        self,
        ranges: list[tuple[datetime, datetime]],
    ) -> list[tuple[datetime, datetime]]:
        """Merge overlapping or contiguous time ranges."""
        if not ranges:
            return []

        merged_ranges = []
        current_start, current_end = ranges[0]

        for i in range(1, len(ranges)):
            next_start, next_end = ranges[i]
            # If ranges overlap or are contiguous
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = next_start, next_end

        merged_ranges.append((current_start, current_end))
        return merged_ranges

    async def _get_latest_timestamp_from_influxdb(self, trading_pair: str, interval: str) -> datetime | None:
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
                        source_module=self._source_module,
                    )
                    return None

        except Exception:
            self.logger.exception("Error getting latest timestamp from InfluxDB:", source_module=self._source_module)
            return None
        else:
            return None

    async def fetch_trades(
        self,
        trading_pair: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | None:
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

            fetch_state = TradesFetchState(
                kraken_pair=kraken_pair,
                batch_size=min(limit or 1000, 1000),
                limit=limit,
                since=since,
                until=until,
            )

            all_trades = await self._fetch_all_trades(fetch_state)

            self._log_fetch_completion(all_trades, trading_pair, since, until)

        except Exception:
            self.logger.exception(f"Failed to fetch trades for {trading_pair}", source_module=self._source_module)
            return None
        else:
            return all_trades

    async def _fetch_all_trades(self, fetch_state: TradesFetchState) -> list[dict[str, Any]]:
        """Fetch all trades using pagination."""
        all_trades: list[dict[str, Any]] = []

        while True:
            # Fetch batch of trades
            batch_result = await self._fetch_trades_batch(fetch_state)
            if batch_result is None:
                break

            trades_data, last_id = batch_result
            if not trades_data:
                break

            # Process batch
            processed = self._process_trades_batch(trades_data, fetch_state.until)
            if processed is None:  # Hit until boundary
                return all_trades

            all_trades.extend(processed)
            fetch_state.last_id = last_id

            # Check completion conditions
            if self._should_stop_fetching(all_trades, trades_data, fetch_state):
                return all_trades[: fetch_state.limit] if fetch_state.limit else all_trades

            # Rate limiting
            await asyncio.sleep(0.5)

        return all_trades

    async def _fetch_trades_batch(
        self,
        fetch_state: TradesFetchState,
    ) -> tuple[list[Any], str | None] | None:
        """Fetch a single batch of trades from API."""
        params = self._build_trades_request_params(fetch_state)

        endpoint = "/0/public/Trades"
        result = await self._make_public_request(endpoint, params)

        if not result or result.get("error"):
            self.logger.error(
                f"Error fetching trades: {result.get('error') if result else 'No response'}",
                source_module=self._source_module,
            )
            return None

        trades_data = result.get("result", {}).get(fetch_state.kraken_pair, [])
        last_id = result.get("result", {}).get("last")

        return trades_data, last_id

    def _build_trades_request_params(self, fetch_state: TradesFetchState) -> dict[str, Any]:
        """Build request parameters for trades API call."""
        params = {
            "pair": fetch_state.kraken_pair,
            "count": fetch_state.batch_size,
        }

        if fetch_state.since and not fetch_state.last_id:
            # Use timestamp for first request
            params["since"] = str(int(fetch_state.since.timestamp() * 1_000_000_000))
        elif fetch_state.last_id:
            # Use last trade ID for pagination
            params["since"] = str(fetch_state.last_id)

        return params

    def _process_trades_batch(
        self,
        trades_data: list[Any],
        until: datetime | None,
    ) -> list[dict[str, Any]] | None:
        """Process a batch of raw trades into structured format."""
        processed_trades = []

        for trade in trades_data:
            trade_time = datetime.fromtimestamp(float(trade[2]), tz=UTC)

            # Check time bounds
            if until and trade_time >= until:
                return None  # Signal to stop fetching

            trade_dict = {
                "timestamp": trade_time,
                "price": Decimal(trade[0]),
                "volume": Decimal(trade[1]),
                "side": "buy" if trade[3] == "b" else "sell",
                "order_type": "market" if trade[4] == "m" else "limit",
                "misc": trade[5] if len(trade) > 5 else "",
            }
            processed_trades.append(trade_dict)

        return processed_trades

    def _should_stop_fetching(
        self,
        all_trades: list[dict[str, Any]],
        trades_data: list[Any],
        fetch_state: TradesFetchState,
    ) -> bool:
        """Check if we should stop fetching more trades."""
        # Check if we've fetched enough
        if fetch_state.limit and len(all_trades) >= fetch_state.limit:
            return True

        # Check if there are more trades
        return len(trades_data) < fetch_state.batch_size

    def _log_fetch_completion(
        self,
        all_trades: list[dict[str, Any]],
        trading_pair: str,
        since: datetime | None,
        until: datetime | None,
    ) -> None:
        """Log completion of trades fetch."""
        self.logger.info(
            f"Fetched {len(all_trades)} trades for {trading_pair}",
            source_module=self._source_module,
            context={
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
            },
        )

    async def _make_public_request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
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
                        context={"endpoint": endpoint},
                    )
                    return None

                return cast("dict[str, Any]", await response.json())

        except Exception:
            self.logger.exception(f"Error making API request to {endpoint}", source_module=self._source_module)
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
